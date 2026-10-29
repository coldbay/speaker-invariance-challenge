"""Score any model on the invariance benchmark (CPU, no model code needed).

A submission is a folder with one JSONL file per condition, named
<condition>.jsonl for all 10 conditions (pad0000ms_fwd ... pad4000ms_fwd,
pad0000ms_rev). Each line:
  {"clip_id": "...", "predictions": [{"segment_id": "segment_000", "speaker": "A"}, ...]}
Speakers are the registered IDs in inputs.jsonl; null or a missing segment
counts as wrong. Predict every public segment to be scored on both tracks.

Reports, per track, invariant accuracy (correct under all conditions), mean and
worst per-condition accuracy, flips (prediction changes across conditions) and,
on Main, same-person pairs.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from hard_tiers import TIERS, tier
from score_predictions import evaluate, read_jsonl

PADS = (0., .5, 1., 1.5, 2., 2.5, 3., 3.5, 4.)
CONDITIONS = [f'pad{round(p*1000):04d}ms_fwd' for p in PADS] + ['pad0000ms_rev']


def load(folder, clips):
    table = {}
    for condition in CONDITIONS:
        path = folder/f'{condition}.jsonl'
        if not path.exists():
            raise FileNotFoundError(f'Missing condition file: {path}')
        rows = {r['clip_id']: r for r in read_jsonl(path)}
        if set(rows) - set(clips):
            raise ValueError(f'Unknown clip IDs in {path.name}')
        for cid in clips:
            table[(condition, cid)] = {p['segment_id']: p.get('speaker') for p in rows.get(cid, {'predictions': []})['predictions']}
    return table


def score(table, truth):
    clips = [t['clip_id'] for t in truth]
    per_condition, by_utt = {}, defaultdict(list)
    for condition in CONDITIONS:
        rows = [{'clip_id': cid, 'predictions': [{'segment_id': s, 'speaker': g} for s, g in table[(condition, cid)].items()]}
                for cid in clips]
        result = evaluate(truth, rows)
        per_condition[condition] = {'correct': sum(d['correct'] for d in result['details']), 'scored': len(result['details']),
                                    'pairs_correct': sum(p['both_correct'] for p in result['same_person_details']),
                                    'pairs': len(result['same_person_details'])}
        for d in result['details']:
            by_utt[(d['clip_id'], d['segment_id'])].append((d['predicted_speaker'], d['correct']))
    accuracies = [v['correct']/v['scored'] for v in per_condition.values()]
    main = {'utterances': len(by_utt), 'invariant_correct': sum(all(c for _, c in v) for v in by_utt.values()),
            'mean_accuracy': sum(accuracies)/len(accuracies), 'worst_accuracy': min(accuracies),
            'flipped': sum(len({p for p, _ in v}) > 1 for v in by_utt.values()),
            'pairs': per_condition[CONDITIONS[0]]['pairs'],
            'pairs_correct_min': min(v['pairs_correct'] for v in per_condition.values()),
            'pairs_correct_max': max(v['pairs_correct'] for v in per_condition.values()),
            'per_condition': per_condition}
    main['invariant_accuracy'] = main['invariant_correct']/main['utterances']
    stats = defaultdict(lambda: defaultdict(int))
    for t in truth:
        for u in t['utterances']:
            group = tier(u)
            if not group:
                continue
            guesses = [table[(c, t['clip_id'])].get(u['segment_id']) for c in CONDITIONS]
            right = [g == u['speaker'] for g in guesses]
            for g in (group, 'all'):
                s = stats[g]
                s['utterances'] += 1; s['correct_runs'] += sum(right); s['runs'] += len(right)
                s['invariant_correct'] += all(right); s['flipped'] += len(set(guesses)) > 1
    hard = {g: dict(v, mean_accuracy=v['correct_runs']/v['runs'], invariant_accuracy=v['invariant_correct']/v['utterances'])
            for g, v in stats.items()}
    return {'conditions': CONDITIONS, 'main': main, 'hard': {g: hard[g] for g in (*TIERS, 'all') if g in hard}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--predictions', type=Path, required=True, help='Folder with the 10 <condition>.jsonl files.')
    parser.add_argument('--data', type=Path, default=Path('data/ami/heldout'))
    parser.add_argument('--output', type=Path, help='Optional JSON report path.')
    args = parser.parse_args()
    truth = read_jsonl(args.data/'labels.jsonl')
    report = score(load(args.predictions, [t['clip_id'] for t in truth]), truth)
    m, h = report['main'], report['hard']['all']
    print(f'Main: invariant {m["invariant_correct"]}/{m["utterances"]} ({m["invariant_accuracy"]:.1%}), '
          f'mean {m["mean_accuracy"]:.1%}, worst {m["worst_accuracy"]:.1%}, flips {m["flipped"]}, '
          f'pairs {m["pairs_correct_min"]}-{m["pairs_correct_max"]}/{m["pairs"]}')
    print(f'Hard: invariant {h["invariant_correct"]}/{h["utterances"]} ({h["invariant_accuracy"]:.1%}), '
          f'mean {h["mean_accuracy"]:.1%}, flips {h["flipped"]}')
    for g in TIERS:
        if g in report['hard']:
            s = report['hard'][g]
            print(f'  {g}: invariant {s["invariant_correct"]}/{s["utterances"]}, mean {s["mean_accuracy"]:.1%}, flips {s["flipped"]}')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
