"""Run a fixed 2x2 MOSS diagnostic, with safe resume and one model load.

Model inference reads inputs.jsonl and a baseline environment manifest only.
Labels/transcripts are used exclusively by report_moss_diagnostics.py.
"""
import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

from diarization_projection import map_enrollment, project_predictions, validate_segments
from moss_diagnostic_protocol import DEFAULT_CLIPS, make_jobs, pack_job, sha

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/ami/prepared'))
    parser.add_argument('--baseline', type=Path, default=Path('runs/speaker_moss_pilot_01'))
    parser.add_argument('--clips', nargs='+', default=list(DEFAULT_CLIPS))
    parser.add_argument('--cuda-id')
    parser.add_argument('--prepare-only', action='store_true', help='Freeze/inspect the plan without importing torch or running models.')
    parser.add_argument('--model-dir', type=Path)
    args = parser.parse_args()
    data = args.data.resolve()
    baseline = json.loads((args.baseline / 'run_manifest.json').read_text(encoding='utf-8'))
    input_bytes = (data / 'inputs.jsonl').read_bytes()
    if sha(input_bytes) != baseline['inputs_sha256']:
        raise ValueError('Inputs differ from baseline. Use the exact pilot inputs.')
    sources = json.loads(Path(__file__).with_name('moss_sources.json').read_text())
    if sources != baseline['sources']:
        raise ValueError('Pinned model/code sources differ from baseline.')
    rows = [json.loads(line) for line in input_bytes.decode('utf-8-sig').splitlines() if line.strip()]
    jobs = make_jobs(rows, args.clips)
    profile = {'dtype': baseline['dtype'], 'max_new_tokens': baseline['max_new_tokens'],
               'gap_seconds': baseline['gap_seconds'], 'do_sample': False, 'seed': 0,
               'attention': 'sdpa', 'cpu_threads': 2}
    code_hashes = {name: sha(Path(__file__).with_name(name).read_bytes().replace(b'\r\n', b'\n'))
                   for name in ('run_moss_diagnostics.py', 'moss_diagnostic_protocol.py', 'diarization_projection.py')}
    plan = {'protocol': 'moss-context-enrollment-2x2-v1', 'sources': sources,
            'inputs_sha256': sha(input_bytes), 'clip_ids': args.clips, 'profile': profile,
            'expected_packages': baseline['packages'], 'expected_torch_cuda': baseline['torch_cuda'],
            'baseline_manifest_sha256': sha((args.baseline / 'run_manifest.json').read_bytes()),
            'adapter_source_sha256_lf': code_hashes,
            'selection': 'All public segments of selected clips with duration 0.8–15s; no eligibility labels read.',
            'scope': 'Post-hoc exploratory diagnostics, not a new held-out leaderboard.', 'jobs': jobs}
    args.output.mkdir(parents=True, exist_ok=True)
    plan_file = args.output / 'plan.json'
    if plan_file.exists():
        if json.loads(plan_file.read_text(encoding='utf-8')) != plan:
            raise ValueError('Existing plan differs. Use a new output folder; previous results are preserved.')
    else:
        if any(args.output.iterdir()):
            raise ValueError('Nonempty output has no recognized plan. Use a new folder.')
        write_json(plan_file, plan)
    counts = Counter(j['condition'] for j in jobs)
    print(json.dumps({'model_calls': len(jobs), 'conditions': dict(counts), 'clip_ids': args.clips,
                      'prepare_only': args.prepare_only, 'output': str(args.output)}, indent=2), flush=True)
    if args.prepare_only:
        print('Plan ready. No model inference or model download was started.', flush=True)
        return

    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    os.environ.setdefault('HF_HOME', str((ROOT / 'models/huggingface').resolve()))
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    os.environ.setdefault('NUMBA_NUM_THREADS', '2')
    if args.cuda_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import build_transcription_messages, generate_transcription

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; check assigned GPU and environment.')
    packages = {name: importlib.metadata.version(name) for name in baseline['packages']}
    if packages != baseline['packages'] or torch.version.cuda != baseline['torch_cuda']:
        raise ValueError(f'Runtime differs from baseline: {packages}, CUDA {torch.version.cuda}. Keep the verified pilot environment.')
    torch.set_num_threads(2)
    runtime = {'packages': packages, 'torch_cuda': torch.version.cuda, 'python': platform.python_version(),
               'gpu': torch.cuda.get_device_name(0), 'cuda_device_order': os.environ.get('CUDA_DEVICE_ORDER'),
               'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'profile': profile}
    plan_hash = sha(json.dumps(plan, sort_keys=True).encode())
    pending = []
    for job in jobs:
        path = args.output / 'jobs' / job['job_id'] / 'raw_output.json'
        if path.exists():
            previous = json.loads(path.read_text(encoding='utf-8'))
            if previous.get('plan_sha256') != plan_hash or previous.get('job') != job or not previous.get('completed'):
                raise ValueError(f'Unrecognized/stale result: {path}')
            if previous.get('runtime') != runtime:
                raise ValueError('Resume environment differs. Use the same GPU/runtime or a new output folder.')
        else:
            pending.append(job)
    write_json(args.output / 'runtime.json', runtime)
    print(f'Completed {len(jobs)-len(pending)}/{len(jobs)}; remaining {len(pending)}', flush=True)
    if not pending:
        print('All jobs already completed. Run report_moss_diagnostics.py.', flush=True)
        return
    device, dtype = torch.device('cuda:0'), getattr(torch, profile['dtype'])
    model_id = 'OpenMOSS-Team/MOSS-Transcribe-Diarize'
    revision = sources['model']['revision']
    model_source = str(args.model_dir.resolve()) if args.model_dir else model_id
    if args.model_dir:
        downloaded = json.loads((args.model_dir / 'download_manifest.json').read_text())
        if downloaded['revision'] != revision:
            raise ValueError('Local model snapshot revision differs from the plan.')
    model = AutoModelForCausalLM.from_pretrained(model_source, revision=revision, trust_remote_code=True,
                dtype=dtype, device_map={'': str(device)}, attn_implementation='sdpa').eval()
    processor = AutoProcessor.from_pretrained(model_source, revision=revision, trust_remote_code=True)
    by_id = {r['clip_id']: r for r in rows}
    scratch = args.output / '.scratch/model_input.wav'
    for index, job in enumerate(pending):
        audio = pack_job(job, by_id[job['clip_id']], data, scratch, profile['gap_seconds'])
        print(f'{len(jobs)-len(pending)+index+1}/{len(jobs)} {job["job_id"]} | {audio["input_audio_seconds"]:.2f}s', flush=True)
        torch.manual_seed(profile['seed'])
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = generate_transcription(model, processor, build_transcription_messages(str(scratch.resolve())),
                    max_new_tokens=profile['max_new_tokens'], do_sample=False, device=device, dtype=dtype)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        segments = [{'start': float(s.start), 'end': float(s.end), 'speaker': str(s.speaker), 'text': s.text}
                    for s in parse_transcript(result['text'])]
        segments = validate_segments(segments, audio['input_audio_seconds'])
        mapping, weights = map_enrollment(segments, audio['enrollment_intervals'])
        predicted = project_predictions(segments, mapping, audio['public_targets'], audio['clip_offset'])
        raw = dict(result, completed=True, job=job, plan_sha256=plan_hash, runtime=runtime,
                   audio=audio, segments=segments, speaker_mapping=mapping,
                   enrollment_overlap_weights=weights, predictions=predicted,
                   reached_token_limit=result.get('generated_tokens', 0) >= profile['max_new_tokens'],
                   inference_seconds=elapsed, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        path = args.output / 'jobs' / job['job_id'] / 'raw_output.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, raw)
        print(f'  saved | {elapsed:.2f}s | mapped {len(mapping)}/{len(job["reference_order"])} | token_limit={raw["reached_token_limit"]}', flush=True)
    print('Finished:', args.output, flush=True)
    print('Next: python speaker_project/report_moss_diagnostics.py --run', args.output, flush=True)


if __name__ == '__main__':
    main()
