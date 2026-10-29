"""Content-preserving perturbation sweep for MOSS speaker attribution.

All 8 prepared conversations. Perturbations change only leading silence
(0-4 s, 0.5 s steps) or enrollment order (reversed, no padding); speech PCM,
order, gaps and enrollment content are unchanged. Labels are never read here:
the server writes raw outputs and a results ZIP, and scoring happens offline
(analyze_invariance.py).
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time
import zipfile

from boundary_control import pack_phase
from diarization_projection import map_enrollment, project_predictions, validate_segments
from moss_diagnostic_protocol import make_jobs, sha
from run_moss_diagnostics import write_json
from score_predictions import read_jsonl

PADS = (0., .5, 1., 1.5, 2., 2.5, 3., 3.5, 4.)
ADAPTER_FILES = ('run_invariance_sweep.py', 'boundary_control.py', 'moss_diagnostic_protocol.py', 'diarization_projection.py')


def condition_name(pad, reverse):
    return f'pad{round(pad*1000):04d}ms_{"rev" if reverse else "fwd"}'


def sweep_jobs(rows):
    clip_ids = [r['clip_id'] for r in rows]
    base = {(j['condition'], j['clip_id']): j for j in make_jobs(rows, clip_ids)
            if j['condition'] in ('full_forward', 'full_reverse')}
    settings = [(pad, False) for pad in PADS] + [(0., True)]
    jobs = []
    for pad, reverse in settings:
        name = condition_name(pad, reverse)
        for index, cid in enumerate(clip_ids):
            j = base[('full_reverse' if reverse else 'full_forward', cid)]
            jobs.append(dict(j, condition=name, job_id=f'{name}/case_{index:03d}/full',
                             leading_silence_seconds=pad, reversed_enrollment=reverse))
    return clip_ids, jobs


def archive(run):
    with zipfile.ZipFile(run/'results_only.zip', 'w', zipfile.ZIP_DEFLATED) as out:
        for path in sorted(run.rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.md') and '.scratch' not in path.parts:
                out.write(path, path.relative_to(run).as_posix())
    with zipfile.ZipFile(run/'results_only.zip') as check:
        if check.testzip():
            raise ValueError('Result archive CRC failure.')
    print('Download:', run/'results_only.zip', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/ami/prepared'))
    parser.add_argument('--baseline', type=Path, default=Path('runs/speaker_moss_pilot_01'))
    parser.add_argument('--cuda-id')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    base = json.loads((args.baseline/'run_manifest.json').read_text())
    rows = read_jsonl(args.data/'inputs.jsonl')
    if sha((args.data/'inputs.jsonl').read_bytes()) != base['inputs_sha256']:
        raise ValueError('Inputs differ from pilot.')
    clip_ids, jobs = sweep_jobs(rows)
    plan = {'protocol': 'invariance-sweep-v1', 'inputs_sha256': base['inputs_sha256'],
            'baseline_manifest_sha256': sha((args.baseline/'run_manifest.json').read_bytes()),
            'sources': base['sources'], 'expected_packages': base['packages'], 'torch_cuda': base['torch_cuda'],
            'clip_ids': clip_ids, 'pads_seconds': list(PADS), 'reversed_enrollment_pads_seconds': [0.],
            'profile': {'dtype': base['dtype'], 'max_new_tokens': base['max_new_tokens'], 'gap_seconds': base['gap_seconds'],
                        'do_sample': False, 'seed': 0, 'attention': 'sdpa', 'cpu_threads': 2},
            'adapter_sha256_lf': {n: sha(Path(__file__).with_name(n).read_bytes().replace(b'\r\n', b'\n')) for n in ADAPTER_FILES},
            'jobs': jobs}
    args.output.mkdir(parents=True, exist_ok=True)
    plan_path = args.output/'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding='utf-8')) != plan:
            raise ValueError('Plan/code changed. Use a new output folder.')
    else:
        if any(args.output.iterdir()):
            raise ValueError('Output folder is not empty; use a new folder.')
        write_json(plan_path, plan)
    print(json.dumps({'model_calls': len(jobs), 'clips': len(clip_ids), 'pads_seconds': list(PADS),
                      'output': str(args.output)}, indent=2), flush=True)
    if args.prepare_only:
        by_id = {r['clip_id']: r for r in rows}
        for job in jobs:  # Packing check only; no model, no labels.
            pack_phase(job, by_id[job['clip_id']], args.data, args.output/'.scratch/check.wav', base['gap_seconds'])
        print('Prepared only; all inputs packed. No GPU or model download.', flush=True)
        return
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    os.environ.setdefault('HF_HOME', str(Path('models/huggingface').resolve()))
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    os.environ.setdefault('NUMBA_NUM_THREADS', '2')
    if args.cuda_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import build_transcription_messages, generate_transcription
    torch.set_num_threads(2)
    packages = {name: importlib.metadata.version(name) for name in base['packages']}
    if packages != base['packages'] or torch.version.cuda != base['torch_cuda']:
        raise ValueError('Use the same successful MOSS pilot environment.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable.')
    runtime = {'packages': packages, 'profile': plan['profile'], 'gpu': torch.cuda.get_device_name(0),
               'torch_cuda': torch.version.cuda, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    plan_hash = sha(json.dumps(plan, sort_keys=True).encode())
    pending = []
    for job in jobs:
        path = args.output/'jobs'/job['job_id']/'raw_output.json'
        if path.exists():
            old = json.loads(path.read_text(encoding='utf-8'))
            if old.get('plan_sha256') != plan_hash or old.get('runtime') != runtime or old.get('job') != job or not old.get('completed'):
                raise ValueError('Resume result differs from current plan/runtime.')
        else:
            pending.append(job)
    write_json(args.output/'runtime.json', runtime)
    if pending:
        device = torch.device('cuda:0'); dtype = getattr(torch, base['dtype'])
        model_id = 'OpenMOSS-Team/MOSS-Transcribe-Diarize'; revision = base['sources']['model']['revision']
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, trust_remote_code=True,
            dtype=dtype, device_map={'': str(device)}, attn_implementation='sdpa').eval()
        processor = AutoProcessor.from_pretrained(model_id, revision=revision, trust_remote_code=True)
        by_id = {r['clip_id']: r for r in rows}
        scratch = args.output/'.scratch/model_input.wav'
        for index, job in enumerate(pending):
            audio = pack_phase(job, by_id[job['clip_id']], args.data, scratch, base['gap_seconds'])
            print(f'{len(jobs)-len(pending)+index+1}/{len(jobs)} {job["job_id"]}', flush=True)
            torch.manual_seed(0); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); started = time.perf_counter()
            result = generate_transcription(model, processor, build_transcription_messages(str(scratch.resolve())),
                max_new_tokens=base['max_new_tokens'], do_sample=False, device=device, dtype=dtype)
            torch.cuda.synchronize(); elapsed = time.perf_counter()-started
            segments = validate_segments([{'start': float(s.start), 'end': float(s.end), 'speaker': str(s.speaker), 'text': s.text}
                                          for s in parse_transcript(result['text'])], audio['input_audio_seconds'])
            mapping, weights = map_enrollment(segments, audio['enrollment_intervals'])
            predictions = project_predictions(segments, mapping, audio['public_targets'], audio['clip_offset'])
            raw = dict(result, job=job, completed=True, plan_sha256=plan_hash, runtime=runtime, audio=audio,
                       segments=segments, speaker_mapping=mapping, enrollment_overlap_weights=weights,
                       predictions=predictions, inference_seconds=elapsed,
                       peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                       reached_token_limit=result.get('generated_tokens', 0) >= base['max_new_tokens'])
            path = args.output/'jobs'/job['job_id']/'raw_output.json'; path.parent.mkdir(parents=True, exist_ok=True)
            write_json(path, raw)
    archive(args.output)


if __name__ == '__main__':
    main()
