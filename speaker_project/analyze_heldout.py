"""Held-out evaluation: main + hard track, accuracy and invariance (CPU).

Uses the frozen methods (MOSS, embedding relink+verify with TAU=0.10,
WeSpeaker) unchanged from the starter-set analysis; nothing is tuned here.
All predictions are made before labels.jsonl is opened.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import zipfile

from hard_tiers import TIERS, tier
from analyze_invariance import summarize
from diarization_projection import project_predictions
from hybrid_verify import TAU, Embedder, clip_context, hybrid_predictions, verify
from moss_diagnostic_protocol import selected_segments
from score_predictions import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/ami/heldout'))
    parser.add_argument('--run', type=Path, default=Path('runs/speaker_heldout_01'))
    args = parser.parse_args()
    rows = {r['clip_id']: r for r in read_jsonl(args.data/'inputs.jsonl')}
    with zipfile.ZipFile(args.run/'results_only.zip') as z:
        if z.testzip():
            raise ValueError('ZIP CRC failure.')
        plan = json.loads(z.read('plan.json'))
        raws = [json.loads(z.read('jobs/'+j['job_id']+'/raw_output.json')) for j in plan['jobs']]
    if any(not r['completed'] or r['job'] not in plan['jobs'] for r in raws):
        raise ValueError('Incomplete or foreign job.')
    embed = Embedder()
    contexts = {cid: clip_context(row, args.data, embed) for cid, row in rows.items()}
    main_preds, all_preds = defaultdict(dict), defaultdict(dict)
    for r in raws:
        cid, cond = r['job']['clip_id'], r['job']['condition']
        offset, segments, ctx = r['audio']['clip_offset'], r['segments'], contexts[cid]
        for table, targets in ((main_preds, selected_segments(rows[cid])), (all_preds, rows[cid]['segments'])):
            _, hybrid, _, _ = hybrid_predictions(segments, offset, targets, ctx, embed)
            table['moss'][(cond, cid)] = project_predictions(segments, r['speaker_mapping'], targets, offset)
            table['moss_relink_verify'][(cond, cid)] = hybrid
            table['wespeaker'][(cond, cid)] = verify([dict(p, speaker=None) for p in hybrid], ctx)
    # Baseline submissions in the public format, scored by evaluate_submission.py.
    for method, table in all_preds.items():
        folder = args.run/'submissions'/method; folder.mkdir(parents=True, exist_ok=True)
        for condition in sorted({c for c, _ in table}):
            rows = [{'clip_id': cid, 'predictions': [{'segment_id': p['segment_id'], 'speaker': p['speaker']} for p in table[(condition, cid)]]}
                    for cid in sorted(rows_ for (c, rows_) in table if c == condition)]
            (folder/f'{condition}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
    truth = read_jsonl(args.data/'labels.jsonl')  # Labels opened only now.
    conditions = sorted({r['job']['condition'] for r in raws})
    clips = sorted(rows)
    main_summary = summarize(main_preds, truth, conditions, clips)
    clip_scores = {m: {cid: summarize({m: main_preds[m]}, truth, conditions, [cid])[m] for cid in clips} for m in main_preds}
    hard = {}
    items = [(t['clip_id'], u) for t in truth for u in t['utterances'] if tier(u)]
    for m, table in all_preds.items():
        look = {(c, cid, p['segment_id']): p['speaker'] for (c, cid), ps in table.items() for p in ps}
        stats = defaultdict(lambda: defaultdict(int))
        for cid, u in items:
            guesses = [look[(c, cid, u['segment_id'])] for c in conditions]
            right = [g == u['speaker'] for g in guesses]
            for g in (tier(u), 'all'):
                s = stats[g]; s['utterances'] += 1; s['correct_runs'] += sum(right); s['runs'] += len(right)
                s['consistent_correct'] += all(right); s['flipped'] += len(set(guesses)) > 1
        hard[m] = {g: dict(v, accuracy=v['correct_runs']/v['runs']) for g, v in stats.items()}
    out = args.run/'analysis'; out.mkdir(exist_ok=True)
    result = {'tau': TAU, 'conditions': conditions, 'main': main_summary, 'main_per_clip': clip_scores, 'hard': hard,
              'total_inference_seconds': sum(r['inference_seconds'] for r in raws),
              'peak_cuda_gib': max(r['peak_cuda_allocated_bytes'] for r in raws)/2**30,
              'token_limit_hits': sum(r['reached_token_limit'] for r in raws),
              'registration': {f'{r["job"]["condition"]}|{r["job"]["clip_id"]}': f'{len(set(r["speaker_mapping"].values()))}/{len(r["job"]["reference_order"])}' for r in raws}}
    (out/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('MAIN', {m: (round(s['mean_accuracy'], 3), round(s['worst_accuracy'], 3), f'{s["consistent_correct"]}/{s["utterances"]}',
                       s['flipped_utterances'], f'{s["unscored_correct"]}/{s["unscored_total"]}') for m, s in main_summary.items()})
    for m in clip_scores:
        print(m, {cid[:7]+cid[7:9]: f'{s["consistent_correct"]}/{s["utterances"]} flip{s["flipped_utterances"]}' for cid, s in clip_scores[m].items()})
    for m, h in hard.items():
        print('HARD', m, {g: (h[g]['utterances'], round(h[g]['accuracy'], 3), h[g]['consistent_correct'], h[g]['flipped']) for g in (*TIERS, 'all') if g in h})
    bad = {k: v for k, v in result['registration'].items() if v.split('/')[0] != v.split('/')[1]}
    print('registration failures', len(bad), bad)


if __name__ == '__main__':
    main()
