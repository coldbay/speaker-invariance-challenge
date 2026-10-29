"""Upgrade existing labels without downloading audio or resetting review.csv."""
import argparse
import hashlib
import json
from pathlib import Path
from focus_protocol import add_focus_pairs
from score_predictions import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/ami/prepared'))
    args = parser.parse_args()
    path = args.data / 'labels.jsonl'
    before = path.read_bytes()
    rows = read_jsonl(path)
    inputs = {r['clip_id']: r for r in read_jsonl(args.data / 'inputs.jsonl')}
    for row in rows:
        add_focus_pairs(row, inputs[row['clip_id']]['segments'])
    backup = args.data / 'labels.v1.jsonl.bak'
    if not backup.exists():
        backup.write_bytes(before)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    report = {'protocol_version': 'natural-qa-v2-same-person',
              'previous_labels_sha256': hashlib.sha256(before).hexdigest(),
              'labels_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'same_person_pairs': sum(len(r['same_speaker_pairs']) for r in rows),
              'eligible_pairs': sum(p['score_eligible'] for r in rows for p in r['same_speaker_pairs']),
              'audio_unchanged': True, 'review_csv_unchanged': True}
    (args.data / 'focus_protocol_manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
