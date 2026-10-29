"""Hard track v1: overlapped and short utterances excluded from the main score.

Tiers are fixed from annotation properties only (duration, word-overlap
fraction, truncation), never from model outputs:
  overlap_light  0.8-15 s, other-speaker overlap (5%, 30%]
  overlap_heavy  0.8-15 s, overlap > 30%
  short_clean    0.3-0.8 s, overlap <= 5%
  short_overlap  0.3-0.8 s, overlap > 5%
Utterances truncated by the clip edge are excluded. Predictions for every
public segment are produced from the existing MOSS sweep outputs before labels
are read; the target is the annotated dialogue-act owner.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from analyze_invariance import load_raw
from diarization_projection import project_predictions
from hard_tiers import TIERS, tier
from hybrid_verify import Embedder, clip_context, hybrid_predictions, verify
from score_predictions import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/ami/prepared'))
    parser.add_argument('--sweep', type=Path, default=Path('runs/speaker_invariance_01'))
    parser.add_argument('--output', type=Path, default=Path('runs/speaker_hard_track_01'))
    args = parser.parse_args()
    args.pilot = args.boundary = args.diagnostics = Path('__unused__')
    rows = {r['clip_id']: r for r in read_jsonl(args.data/'inputs.jsonl')}
    embed = Embedder()
    contexts = {cid: clip_context(row, args.data, embed) for cid, row in rows.items()}
    _, sweep = load_raw(args)
    if len(sweep) != 80:
        raise ValueError('Expected the 80-run invariance sweep.')
    predictions = defaultdict(dict)
    for r in sweep:  # All public segments; labels are not open yet.
        targets = rows[r['clip_id']]['segments']
        key = (r['condition'], r['clip_id'])
        _, hybrid, _, _ = hybrid_predictions(r['segments'], r['clip_offset'], targets, contexts[r['clip_id']], embed)
        predictions['moss'][key] = project_predictions(r['segments'], r['mapping'], targets, r['clip_offset'])
        predictions['moss_relink_verify'][key] = hybrid
        predictions['wespeaker'][key] = verify([dict(p, speaker=None) for p in hybrid], contexts[r['clip_id']])
    labels = read_jsonl(args.data/'labels.jsonl')
    track = [{'clip_id': t['clip_id'], 'split': t['split'], 'segment_id': u['segment_id'], 'speaker': u['speaker'],
              'tier': tier(u), 'duration': u['end']-u['start'], 'overlap': u['other_speaker_word_overlap_fraction'],
              'speakers_in_clip': len(rows[t['clip_id']]['registered_speakers'])}
             for t in labels for u in t['utterances'] if tier(u)]
    conditions = sorted({r['condition'] for r in sweep})
    summary, per_utterance = {}, []
    for method, table in predictions.items():
        lookup = {(c, cid, p['segment_id']): p['speaker'] for (c, cid), ps in table.items() for p in ps}
        stats = defaultdict(lambda: defaultdict(int))
        for item in track:
            guesses = [lookup[(c, item['clip_id'], item['segment_id'])] for c in conditions]
            right = [g == item['speaker'] for g in guesses]
            for group in (f'{item["split"]}/{item["tier"]}', f'{item["split"]}/all', 'all/all'):
                s = stats[group]
                s['utterances'] += 1; s['correct_runs'] += sum(right); s['runs'] += len(right)
                s['consistent_correct'] += all(right); s['flipped'] += len(set(guesses)) > 1
                s['missing_runs'] += sum(g is None for g in guesses)
            per_utterance.append(dict(item, method=method, correct_runs=sum(right), runs=len(right),
                                      predictions=dict(zip(conditions, guesses))))
        summary[method] = {g: dict(v, accuracy=v['correct_runs']/v['runs']) for g, v in sorted(stats.items())}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'hard_track_v1.jsonl').write_text(''.join(json.dumps(t, ensure_ascii=False)+'\n' for t in track), encoding='utf-8')
    (args.output/'summary.json').write_text(json.dumps({'tiers': TIERS, 'conditions': conditions, 'summary': summary},
                                                       ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (args.output/'per_utterance.jsonl').write_text(''.join(json.dumps(p, ensure_ascii=False)+'\n' for p in per_utterance), encoding='utf-8')
    groups = [f'{s}/{t}' for s in ('development', 'evaluation') for t in (*TIERS, 'all')] + ['all/all']
    lines = ['| 그룹 | 발화 | ' + ' | '.join(f'{m} 정답률 / 일관 정답 / 바뀜' for m in summary) + ' |',
             '|---|---:|' + '---|' * len(summary)]
    for g in groups:
        first = summary['moss'].get(g)
        if not first:
            continue
        cells = [f'{summary[m][g]["accuracy"]:.1%} / {summary[m][g]["consistent_correct"]} / {summary[m][g]["flipped"]}' for m in summary]
        lines.append(f'| {g} | {first["utterances"]} | ' + ' | '.join(cells) + ' |')
    print('\n'.join(lines))
    (args.output/'table.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
