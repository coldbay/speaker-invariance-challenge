"""Audit synchronized predictions without model execution or modifying raw results."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import wave

from diarization_projection import overlap
from score_predictions import evaluate, read_jsonl


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pcm(path, start_frame=0):
    with wave.open(str(path)) as audio:
        audio.setpos(start_frame)
        return audio.readframes(audio.getnframes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/ami/prepared'))
    parser.add_argument('--joint', type=Path, default=Path('runs/speaker_moss_pilot_01'))
    parser.add_argument('--acoustic', type=Path, default=Path('runs/speaker_wespeaker_01'))
    parser.add_argument('--output', type=Path, default=Path('runs/speaker_comparison/analysis'))
    args = parser.parse_args()
    labels = read_jsonl(args.data / 'labels.jsonl')
    inputs = {r['clip_id']: r for r in read_jsonl(args.data / 'inputs.jsonl')}
    predictions = read_jsonl(args.joint / 'predictions.jsonl')
    acoustic = {r['clip_id']: {p['segment_id']: p for p in r['predictions']}
                for r in read_jsonl(args.acoustic / 'predictions.jsonl')}
    pred = {r['clip_id']: {p['segment_id']: p for p in r['predictions']} for r in predictions}
    if set(pred) != set(inputs):
        raise ValueError('A full run with all input clip IDs is required.')
    manifest = read(args.joint / 'run_manifest.json')
    if manifest['inputs_sha256'] != sha(args.data / 'inputs.jsonl'):
        raise ValueError('Inputs changed since the model run.')
    raw = {r['clip_id']: (path, r) for path in sorted(args.joint.glob('case_*/raw_output.json'))
           for r in [read(path)]}
    scores = evaluate(labels, predictions)
    if scores != read(args.joint / 'scores.json'):
        raise ValueError('Recomputed scores differ from synchronized scores.json.')
    audit = {'recomputed_scores_match': True, 'input_sha256_matches': True,
             'clips': [], 'errors': [], 'same_person_state_counts': {},
             'runtime': {'gpu': manifest['gpu'], 'packages': manifest['packages'],
                         'torch_cuda': manifest['torch_cuda'], 'elapsed_seconds': manifest['elapsed_seconds']}}
    for row in labels:
        cid = row['clip_id']
        file, output = raw[cid]
        detail = [d for d in scores['details'] if d['clip_id'] == cid]
        audio = file.parent / 'model_input.wav'
        exact_pcm = None
        if audio.exists():
            exact_pcm = pcm(audio, round(output['clip_offset'] * 16000)) == pcm(args.data / inputs[cid]['audio'])
            if not exact_pcm:
                raise ValueError(f'Evaluation audio differs from original: {cid}')
        missing_agents = set(inputs[cid]['registered_speakers']) - set(output['speaker_mapping'].values())
        merges = {key: [sp for sp, seconds in weights.items() if seconds > 0]
                  for key, weights in output['enrollment_overlap_weights'].items()
                  if sum(seconds > 0 for seconds in weights.values()) > 1}
        audit['clips'].append({'clip_id': cid, 'split': row['split'], 'event': row['event'],
            'correct': sum(d['correct'] for d in detail), 'scored': len(detail),
            'registered_speakers': sorted(inputs[cid]['registered_speakers']),
            'scored_speaker_counts': dict(Counter(d['speaker'] for d in detail)),
            'unmapped_registered_speakers': sorted(missing_agents),
            'anonymous_ids_overlapping_multiple_enrollment_speakers': merges,
            'evaluation_pcm_exact': exact_pcm,
            'token_limit_reached': output['reached_token_limit'],
            'inference_seconds': output['inference_seconds'],
            'peak_cuda_allocated_gib': output['peak_cuda_allocated_bytes'] / 2**30})
        for u in row['utterances']:
            prediction = pred[cid][u['segment_id']]
            if not u['score_eligible'] or prediction['speaker'] == u['speaker']:
                continue
            start = u['start'] + output['clip_offset']
            end = u['end'] + output['clip_offset']
            matched = [dict(s, overlap_seconds=overlap(start, end, s['start'], s['end']))
                       for s in output['segments'] if overlap(start, end, s['start'], s['end']) > 0]
            audit['errors'].append({'clip_id': cid, 'utterance': u, 'joint_prediction': prediction,
                                    'overlapping_raw_segments': matched,
                                    'acoustic_prediction': acoustic[cid][u['segment_id']]})
    states = Counter(('correct' if p['before_correct'] else 'wrong') + '_to_' +
                     ('correct' if p['after_correct'] else 'wrong') for p in scores['same_person_details'])
    audit['same_person_state_counts'] = dict(states)
    audit['same_person_summary'] = scores['same_person_summary']
    audit['scored_segments'] = scores['scored_segments']
    audit['correct_segments'] = sum(d['correct'] for d in scores['details'])
    audit['missing_predictions'] = sum(d['missing'] for d in scores['details'])
    audit['runtime']['summed_inference_seconds'] = sum(c['inference_seconds'] for c in audit['clips'])
    audit['runtime']['peak_cuda_allocated_gib'] = max(c['peak_cuda_allocated_gib'] for c in audit['clips'])
    # Descriptive RMS contrast only; different phonetic content is not controlled.
    for error in audit['errors']:
        row = next(r for r in labels if r['clip_id'] == error['clip_id'])
        later = next(u for u in row['utterances'] if u.get('focus_slot') == 'after_answer')
        same_person = later['speaker'] == error['utterance']['speaker']
        if same_person:
            ref = acoustic[row['clip_id']][later['segment_id']]
            error['later_answer_rms_difference_db'] = 20 * math.log10(
                ref['rms_pcm'] / error['acoustic_prediction']['rms_pcm'])
    smoke_file = args.joint.parent / 'speaker_moss_smoke_01/case_000/raw_output.json'
    if smoke_file.exists():
        smoke = read(smoke_file)
        pilot = raw[smoke['clip_id']][1]
        audit['smoke_pilot_first_clip_equal_text_and_segments'] = (
            smoke['text'] == pilot['text'] and smoke['segments'] == pilot['segments'])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in audit.items() if key not in ('clips', 'errors', 'same_person_summary')}, indent=2))


if __name__ == '__main__':
    main()
