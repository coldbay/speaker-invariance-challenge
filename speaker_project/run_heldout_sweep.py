"""Invariance sweep on the held-out 4-speaker set, shardable over GPUs.

Same 10 content-preserving conditions as run_invariance_sweep.py (leading
silence 0-4 s in 0.5 s steps, plus reversed enrollment). Run one process per
GPU with --shard K/N; every shard writes into the same output folder and the
last one to finish packs results_only.zip. Labels are never read here.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import time

from boundary_control import pack_phase
from diarization_projection import map_enrollment, project_predictions, validate_segments
from moss_diagnostic_protocol import sha
from run_invariance_sweep import PADS, archive, sweep_jobs
from run_moss_diagnostics import write_json
from score_predictions import read_jsonl

ADAPTER_FILES = ('run_heldout_sweep.py', 'run_invariance_sweep.py', 'boundary_control.py',
                 'moss_diagnostic_protocol.py', 'diarization_projection.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/ami/heldout'))
    parser.add_argument('--baseline', type=Path, default=Path('runs/speaker_moss_pilot_01'),
                        help='Source of the frozen model revision, packages and decoding profile only.')
    parser.add_argument('--cuda-id')
    parser.add_argument('--shard', default='0/1', help='K/N: run every N-th job starting at K.')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    shard, shards = map(int, args.shard.split('/'))
    if not 0 <= shard < shards:
        raise ValueError('--shard must be K/N with 0 <= K < N.')
    base = json.loads((args.baseline/'run_manifest.json').read_text())
    rows = read_jsonl(args.data/'inputs.jsonl')
    clip_ids, jobs = sweep_jobs(rows)
    plan = {'protocol': 'heldout-invariance-sweep-v1', 'inputs_sha256': sha((args.data/'inputs.jsonl').read_bytes()),
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
            raise ValueError('Plan/code/data changed. Use a new output folder.')
    else:
        write_json(plan_path, plan)
    mine = jobs[shard::shards]
    print(json.dumps({'model_calls': len(jobs), 'this_shard': len(mine), 'shard': args.shard, 'clips': len(clip_ids),
                      'output': str(args.output)}, indent=2), flush=True)
    by_id = {r['clip_id']: r for r in rows}
    if args.prepare_only:
        for job in mine:
            pack_phase(job, by_id[job['clip_id']], args.data, args.output/f'.scratch/check_{shard}.wav', base['gap_seconds'])
        print('Prepared only; inputs packed. No GPU or model download.', flush=True)
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
               'torch_cuda': torch.version.cuda, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
               'shard': args.shard}
    plan_hash = sha(json.dumps(plan, sort_keys=True).encode())
    pending = []
    for job in mine:
        path = args.output/'jobs'/job['job_id']/'raw_output.json'
        if path.exists():
            old = json.loads(path.read_text(encoding='utf-8'))
            if old.get('plan_sha256') != plan_hash or old.get('job') != job or not old.get('completed'):
                raise ValueError('Resume result differs from current plan.')
        else:
            pending.append(job)
    write_json(args.output/f'runtime_shard{shard}.json', runtime)
    if pending:
        device = torch.device('cuda:0'); dtype = getattr(torch, base['dtype'])
        model_id = 'OpenMOSS-Team/MOSS-Transcribe-Diarize'; revision = base['sources']['model']['revision']
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, trust_remote_code=True,
            dtype=dtype, device_map={'': str(device)}, attn_implementation='sdpa').eval()
        processor = AutoProcessor.from_pretrained(model_id, revision=revision, trust_remote_code=True)
        scratch = args.output/f'.scratch/model_input_{shard}.wav'
        for index, job in enumerate(pending):
            audio = pack_phase(job, by_id[job['clip_id']], args.data, scratch, base['gap_seconds'])
            print(f'shard {args.shard}: {len(mine)-len(pending)+index+1}/{len(mine)} {job["job_id"]}', flush=True)
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
    done = sum((args.output/'jobs'/j['job_id']/'raw_output.json').exists() for j in jobs)
    if done == len(jobs):
        archive(args.output)
    else:
        print(f'Shard finished; {done}/{len(jobs)} jobs complete. The last shard packs the ZIP.', flush=True)


if __name__ == '__main__':
    main()
