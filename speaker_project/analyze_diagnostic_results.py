"""Audit the synchronized ZIP, re-build all input hashes, and classify failures.

Offline only: no model weights, inference, or benchmark prediction changes.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import tempfile
import zipfile

from diarization_projection import map_enrollment, overlap, project_predictions
from moss_diagnostic_protocol import CONDITIONS, pack_job, sha
from report_moss_diagnostics import summarize
from score_predictions import read_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('runs/speaker_moss_diagnostics_01'))
    parser.add_argument('--data', type=Path, default=Path('data/ami/prepared'))
    args = parser.parse_args()
    archive_path = args.run / 'results_only.zip'
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names) or archive.testzip():
            raise ValueError('Duplicate ZIP entries or CRC failure.')
        files = {name: archive.read(name) for name in names}
    parsed = {name: json.loads(data.decode('utf-8-sig')) for name, data in files.items() if name.endswith('.json')}
    plan, stored = parsed['plan.json'], parsed['diagnostic_summary.json']
    plan_hash = sha(json.dumps(plan, sort_keys=True).encode())
    if sha((args.data / 'inputs.jsonl').read_bytes()) != plan['inputs_sha256']:
        raise ValueError('Inputs differ from the experiment.')
    if sha((args.data / 'labels.jsonl').read_bytes()) != stored['labels_sha256']:
        raise ValueError('Labels differ from the experiment report.')
    inputs = {r['clip_id']: r for r in read_jsonl(args.data / 'inputs.jsonl')}
    truth = [r for r in read_jsonl(args.data / 'labels.jsonl') if r['clip_id'] in plan['clip_ids']]
    utterances = {r['clip_id']: {u['segment_id']: u for u in r['utterances']} for r in truth}
    audit = {'archive_sha256': sha(archive_path.read_bytes()), 'archive_bytes': archive_path.stat().st_size,
             'zip_crc_passed': True, 'planned_jobs': len(plan['jobs']), 'input_audio_metadata_rebuilt': 0,
             'predictions_reprojected': 0, 'conditions': {}, 'errors': [], 'baseline_repeat_checks': stored['baseline_repeat_checks'],
             'audited_frontend_chunk_seconds': 30, 'scored_target_frontend_positions': []}
    all_raw = defaultdict(list)
    with tempfile.TemporaryDirectory(prefix='moss-input-audit-') as temporary:
        for job in plan['jobs']:
            raw = parsed['jobs/' + job['job_id'] + '/raw_output.json']
            if raw['job'] != job or raw['plan_sha256'] != plan_hash or not raw['completed']:
                raise ValueError('A raw result does not match the frozen plan.')
            if (raw['runtime']['profile'] != plan['profile'] or raw['runtime']['packages'] != plan['expected_packages']
                    or raw['runtime']['torch_cuda'] != plan['expected_torch_cuda']):
                raise ValueError('Runtime changed across conditions.')
            rebuilt = pack_job(job, inputs[job['clip_id']], args.data, Path(temporary) / 'input.wav', plan['profile']['gap_seconds'])
            if rebuilt != raw['audio']:
                raise ValueError(f'Input metadata/PCM hash differs: {job["job_id"]}')
            audit['input_audio_metadata_rebuilt'] += 1
            mapping, weights = map_enrollment(raw['segments'], rebuilt['enrollment_intervals'])
            predictions = project_predictions(raw['segments'], mapping, rebuilt['public_targets'], rebuilt['clip_offset'])
            if mapping != raw['speaker_mapping'] or weights != raw['enrollment_overlap_weights'] or predictions != raw['predictions']:
                raise ValueError(f'Projection does not reproduce result: {job["job_id"]}')
            audit['predictions_reprojected'] += 1
            all_raw[job['condition']].append(raw)
    reference_scores = None
    for condition in CONDITIONS:
        raw_rows = all_raw[condition]
        combined = {cid: {} for cid in plan['clip_ids']}
        for raw in raw_rows:
            for p in raw['predictions']:
                target = combined[raw['job']['clip_id']]
                if p['segment_id'] in target:
                    raise ValueError('Duplicate target across jobs.')
                target[p['segment_id']] = p
        predictions = [{'clip_id': cid, 'predictions': list(items.values())} for cid, items in combined.items()]
        result, scores = summarize(truth, predictions, raw_rows, reference_scores)
        if condition == 'full_forward':
            reference_scores = scores
            result, scores = summarize(truth, predictions, raw_rows, reference_scores)
        if result != stored['summary'][condition] or scores != parsed[f'conditions/{condition}/scores.json']:
            raise ValueError('Recomputed metrics differ from synchronized report.')
        types = Counter()
        registration = defaultdict(Counter)
        focus = defaultdict(lambda: {'correct': 0, 'scored': 0})
        for raw in raw_rows:
            cid = raw['job']['clip_id']
            missing = set(raw['job']['reference_order']) - set(raw['speaker_mapping'].values())
            registration[cid]['calls'] += 1
            registration[cid]['incomplete'] += bool(missing)
            for agent in missing:
                registration[cid]['missing_' + agent] += 1
            for prediction in raw['predictions']:
                u = utterances[cid][prediction['segment_id']]
                if not u['score_eligible']:
                    continue
                correct = u['speaker'] == prediction['speaker']
                slot = u.get('focus_slot')
                if slot:
                    focus[slot]['correct'] += correct
                    focus[slot]['scored'] += 1
                t = next(s for s in raw['audio']['public_targets'] if s['segment_id'] == prediction['segment_id'])
                start, end = t['start'] + raw['audio']['clip_offset'], t['end'] + raw['audio']['clip_offset']
                audit['scored_target_frontend_positions'].append({'condition': condition, 'clip_id': cid,
                    'segment_id': prediction['segment_id'], 'start': start, 'end': end, 'correct': correct,
                    'crosses_30s_boundary': math.floor(start / 30) != math.floor((end - 1e-9) / 30)})
                if correct:
                    continue
                overlaps = [dict(s, target_overlap_seconds=overlap(start, end, s['start'], s['end']))
                            for s in raw['segments'] if overlap(start, end, s['start'], s['end']) > 0]
                kind = ('wrong_mapped_speaker' if prediction['speaker'] is not None else
                        'unmapped_anonymous_id' if overlaps else 'missing_target_or_timestamp')
                types[kind] += 1
                audit['errors'].append({'condition': condition, 'clip_id': cid, 'job_id': raw['job']['job_id'],
                    'utterance': u, 'prediction': prediction, 'error_type': kind, 'overlapping_raw_segments': overlaps,
                    'speaker_mapping': raw['speaker_mapping'], 'enrollment_overlap_weights': raw['enrollment_overlap_weights']})
        audit['conditions'][condition] = {'recomputed_summary_matches': True, 'summary': result,
                                          'error_types': dict(types), 'focus_slot_scores': dict(focus),
                                          'registration_by_clip': {cid: dict(v) for cid, v in registration.items()}}
    audit['total_model_inference_seconds'] = sum(r['inference_seconds'] for rows in all_raw.values() for r in rows)
    audit['max_cuda_allocated_gib'] = max(r['peak_cuda_allocated_bytes'] for rows in all_raw.values() for r in rows) / 2**30
    audit['matching_local_files'] = sum((args.run / name).is_file() and (args.run / name).read_bytes() == data for name, data in files.items())
    audit['archive_file_count'] = len(files)
    output = args.run / 'analysis'
    output.mkdir(exist_ok=True)
    (output / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in audit.items() if key not in ('conditions', 'errors', 'scored_target_frontend_positions')}, indent=2))


if __name__ == '__main__':
    main()
