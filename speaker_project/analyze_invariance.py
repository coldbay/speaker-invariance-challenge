"""Offline accuracy + invariance analysis of MOSS, embedding-verified MOSS and WeSpeaker.

Reads existing MOSS raw outputs (pilot, boundary control, diagnostics) and,
when present, the invariance sweep. Labels are read only for scoring, after
all predictions are produced. No GPU is used.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import zipfile

from diarization_projection import project_predictions
from hybrid_verify import TAU, Embedder, clip_context, hybrid_predictions, verify
from moss_diagnostic_protocol import selected_segments
from score_predictions import evaluate, read_jsonl

TAU_ABLATIONS = (0., .05, .2)


def load_raw(args):
    records = []
    for path in sorted(args.pilot.glob('case_*/raw_output.json')):
        raw = json.loads(path.read_text(encoding='utf-8'))
        records.append({'source': 'pilot', 'condition': 'pad0000ms_fwd', 'clip_id': raw['clip_id'],
                        'segments': raw['segments'], 'clip_offset': raw['clip_offset'], 'mapping': raw['speaker_mapping']})
    extra = [(args.boundary, {'pad_02': 'pad2000ms_fwd', 'pad_30': 'pad30000ms_fwd', 'pad_32': 'pad32000ms_fwd'}),
             (args.diagnostics, {'full_reverse': 'pad0000ms_rev'})]
    for run, names in extra:
        for path in sorted(run.glob('jobs/*/case_*/full/raw_output.json')):
            raw = json.loads(path.read_text(encoding='utf-8'))
            if raw['job']['condition'] in names:
                records.append({'source': run.name, 'condition': names[raw['job']['condition']], 'clip_id': raw['job']['clip_id'],
                                'segments': raw['segments'], 'clip_offset': raw['audio']['clip_offset'], 'mapping': raw['speaker_mapping']})
    sweep = []
    archive = args.sweep/'results_only.zip'
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            if z.testzip():
                raise ValueError('Sweep ZIP CRC failure.')
            for name in sorted(n for n in z.namelist() if n.startswith('jobs/') and n.endswith('raw_output.json')):
                raw = json.loads(z.read(name).decode('utf-8'))
                if not raw['completed']:
                    raise ValueError(f'Incomplete sweep job: {name}')
                sweep.append({'source': 'sweep', 'condition': raw['job']['condition'], 'clip_id': raw['job']['clip_id'],
                              'segments': raw['segments'], 'clip_offset': raw['audio']['clip_offset'], 'mapping': raw['speaker_mapping']})
    return records, sweep


def predict_all(records, rows, contexts, embed):
    """{method: {(condition, clip_id): predictions}} with labels never touched."""
    out = defaultdict(dict)
    for r in records:
        targets = selected_segments(rows[r['clip_id']])
        key = (r['condition'], r['clip_id'])
        context = contexts[r['clip_id']]
        out['moss'][key] = project_predictions(r['segments'], r['mapping'], targets, r['clip_offset'])
        relinked, hybrid, _, _ = hybrid_predictions(r['segments'], r['clip_offset'], targets, context, embed)
        out['moss_relink'][key] = relinked
        out['moss_relink_verify'][key] = hybrid
        out['moss_verify_only'][key] = verify(out['moss'][key], context)
        for tau in TAU_ABLATIONS:
            out[f'ablation_tau_{tau:.2f}'][key] = verify(relinked, context, tau)
        out['wespeaker'][key] = verify([dict(p, speaker=None) for p in relinked], context)  # embedding-only choice
    return out


def summarize(methods, truth, conditions, clips):
    truth = [t for t in truth if t['clip_id'] in clips]
    labels = {(t['clip_id'], u['segment_id']): u for t in truth for u in t['utterances'] if u['score_eligible']}
    result = {}
    for method, table in methods.items():
        per_condition, by_utt, pairs = {}, defaultdict(list), []
        for condition in conditions:
            rows = [{'clip_id': cid, 'predictions': [{'segment_id': p['segment_id'], 'speaker': p['speaker']} for p in table[(condition, cid)]]}
                    for cid in clips]
            scores = evaluate(truth, rows)
            per_condition[condition] = {'correct': sum(d['correct'] for d in scores['details']), 'scored': len(scores['details']),
                                        'pairs_correct': sum(p['both_correct'] for p in scores['same_person_details']),
                                        'pairs': len(scores['same_person_details'])}
            for d in scores['details']:
                by_utt[(d['clip_id'], d['segment_id'])].append((d['predicted_speaker'], d['correct']))
            pairs.append(per_condition[condition]['pairs_correct'])
        # Short/overlapping utterances outside the scored set: reported to expose hidden damage.
        everything = {(t['clip_id'], u['segment_id']): u for t in truth for u in t['utterances']}
        unscored = [p['speaker'] == everything[(cid, p['segment_id'])]['speaker']
                    for condition in conditions for cid in clips for p in table[(condition, cid)]
                    if not everything[(cid, p['segment_id'])]['score_eligible']]
        n = len(by_utt)
        accuracies = [v['correct']/v['scored'] for v in per_condition.values()]
        result[method] = {
            'conditions': len(conditions), 'utterances': n,
            'mean_accuracy': sum(accuracies)/len(accuracies), 'worst_accuracy': min(accuracies),
            'consistent_correct': sum(all(c for _, c in v) for v in by_utt.values()),
            'flipped_utterances': sum(len({p for p, _ in v}) > 1 for v in by_utt.values()),
            'pairs_total': per_condition[conditions[0]]['pairs'], 'pairs_correct_by_condition': pairs,
            'unscored_correct': sum(unscored), 'unscored_total': len(unscored),
            'per_condition': per_condition}
        assert set(by_utt) == set(labels)
    return result


def table(summary, title):
    lines = [f'### {title}', '', '| 방법 | 평균 정답률 | 최저 조건 | 모든 조건 일관 정답 | 조건 따라 바뀐 발화 | 쌍 동시 정답(조건별) | 채점 제외 발화 정답(전 조건 합) |',
             '|---|---:|---:|---:|---:|---|---:|']
    for method, s in summary.items():
        lines.append(f'| {method} | {s["mean_accuracy"]:.1%} | {s["worst_accuracy"]:.1%} | {s["consistent_correct"]}/{s["utterances"]} | '
                     f'{s["flipped_utterances"]}/{s["utterances"]} | {",".join(map(str, s["pairs_correct_by_condition"]))} / {s["pairs_total"]} | '
                     f'{s["unscored_correct"]}/{s["unscored_total"]} |')
    return lines + ['']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/ami/prepared'))
    parser.add_argument('--pilot', type=Path, default=Path('runs/speaker_moss_pilot_01'))
    parser.add_argument('--boundary', type=Path, default=Path('runs/speaker_boundary_02'))
    parser.add_argument('--diagnostics', type=Path, default=Path('runs/speaker_moss_diagnostics_01'))
    parser.add_argument('--sweep', type=Path, default=Path('runs/speaker_invariance_01'))
    parser.add_argument('--output', type=Path, default=Path('runs/speaker_invariance_analysis'))
    args = parser.parse_args()
    rows = {r['clip_id']: r for r in read_jsonl(args.data/'inputs.jsonl')}
    embed = Embedder()
    contexts = {cid: clip_context(row, args.data, embed) for cid, row in rows.items()}
    records, sweep = load_raw(args)
    # All predictions are made before labels are opened.
    existing = predict_all(records, rows, contexts, embed)
    swept = predict_all(sweep, rows, contexts, embed) if sweep else None
    truth = read_jsonl(args.data/'labels.jsonl')
    split = {t['clip_id']: t['split'] for t in truth}
    report = {'tau': TAU, 'tau_ablations': list(TAU_ABLATIONS), 'sets': {}}
    lines = ['# 변형 불변성과 임베딩 검증 보정', '',
             f'τ={TAU}는 채점 전에 고정했다. 다른 τ는 민감도 확인용이다. MOSS 전사·시각은 바꾸지 않고 화자 라벨만 바꾼다. '
             'wespeaker는 같은 등록 프로토타입으로 발화 임베딩만 쓴 선택이며 입력 변형과 무관하므로 불변성이 자명하다.', '']
    pilot_clips = sorted({r['clip_id'] for r in records if r['source'] == 'pilot'})
    for name in ('development', 'evaluation'):
        clips = [c for c in pilot_clips if split[c] == name]
        s = summarize(existing, truth, ['pad0000ms_fwd'], clips)
        report['sets'][f'pilot_{name}'] = s
        lines += table(s, f'기존 pilot 8대화 중 {name} ({len(clips)}대화, 1조건)')
    multi = sorted({r['clip_id'] for r in records if r['source'] != 'pilot'})
    conds = ['pad0000ms_fwd', 'pad2000ms_fwd', 'pad30000ms_fwd', 'pad32000ms_fwd', 'pad0000ms_rev']
    s = summarize(existing, truth, conds, multi)
    report['sets']['existing_multi_condition'] = s
    lines += table(s, f'기존 다조건 2대화 ({", ".join(c[:7] for c in multi)}; 무음 0/2/30/32초 + 등록 역순)')
    if swept:
        sweep_clips = sorted({r['clip_id'] for r in sweep})
        sweep_conds = sorted({r['condition'] for r in sweep})
        for name in ('development', 'evaluation'):
            clips = [c for c in sweep_clips if split[c] == name]
            s = summarize(swept, truth, sweep_conds, clips)
            report['sets'][f'sweep_{name}'] = s
            lines += table(s, f'sweep {name} ({len(clips)}대화 × {len(sweep_conds)}조건)')
    else:
        lines += ['sweep 결과(runs/speaker_invariance_01/results_only.zip)가 아직 없다.', '']
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    decisions = {m: {f'{c}|{k}': [{x: p.get(x) for x in ('segment_id', 'speaker', 'moss_speaker', 'decision')} for p in v]
                     for (c, k), v in t.items()} for m, t in existing.items() if m in ('moss', 'moss_relink', 'moss_relink_verify')}
    (args.output/'existing_predictions.json').write_text(json.dumps(decisions, ensure_ascii=False)+'\n', encoding='utf-8')
    (args.output/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
