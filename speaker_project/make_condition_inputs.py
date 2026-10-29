"""Write the 10 perturbed model inputs per clip for participants (no labels).

For joint models that take one audio stream: each condition's WAV is
[enrollment refs + 0.4 s gaps in the given order] + conversation, with the
stated leading silence. condition_inputs.jsonl gives, per WAV, the enrollment
intervals (speaker ID + seconds) and every public target segment shifted into
that WAV's timeline. Embedding-only models can ignore this and read
inputs.jsonl directly; their predictions are then the same in all conditions.
"""
import argparse
import json
from pathlib import Path

from boundary_control import pack_phase
from run_invariance_sweep import sweep_jobs
from score_predictions import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/ami/heldout'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.data/'inputs.jsonl')
    by_id = {r['clip_id']: r for r in rows}
    _, jobs = sweep_jobs(rows)
    index = []
    for job in jobs:
        wav = args.output/job['condition']/f'{job["clip_id"]}.wav'
        audio = pack_phase(job, by_id[job['clip_id']], args.data, wav, .4)
        index.append({'condition': job['condition'], 'clip_id': job['clip_id'], 'audio': wav.relative_to(args.output).as_posix(),
                      'enrollment_intervals': audio['enrollment_intervals'],
                      'targets': [dict(t, start=t['start']+audio['clip_offset'], end=t['end']+audio['clip_offset'])
                                  for t in by_id[job['clip_id']]['segments']],
                      'input_wav_sha256': audio['input_wav_sha256']})
    (args.output/'condition_inputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in index), encoding='utf-8')
    print(f'{len(index)} inputs written to {args.output}')


if __name__ == '__main__':
    main()
