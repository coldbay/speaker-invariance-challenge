"""GPU adapter for official MOSS; model execution is pending server validation.

Prepends the SAME held-out enrollment WAVs supplied to the acoustic baseline.
Original conversation order is preserved. Anonymous output IDs are mapped only
from enrollment-time overlap, then projected onto public act boundaries.
No evaluation labels, transcripts, roles, or event names are given to the model.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import wave

from diarization_projection import map_enrollment, project_predictions, validate_segments


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pcm(path, expected):
    if digest(path) != expected:
        raise ValueError(f'Audio checksum mismatch: {path}')
    with wave.open(str(path), 'rb') as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, 16000):
            raise ValueError(f'Expected mono PCM16 16kHz: {path}')
        return source.readframes(source.getnframes())


def pack_audio(row, data, target, gap_seconds):
    chunks, intervals = [], []
    frames = 0
    gap = bytes(round(gap_seconds * 16000) * 2)
    for speaker, references in sorted(row['registered_speakers'].items()):
        for ref in references:
            samples = pcm(data / ref['path'], ref['sha256'])
            length = len(samples) // 2
            intervals.append({'speaker': speaker, 'start': frames / 16000, 'end': (frames + length) / 16000})
            chunks.extend((samples, gap))
            frames += length + len(gap) // 2
    clip_offset = frames / 16000
    chunks.append(pcm(data / row['audio'], row['audio_sha256']))
    with wave.open(str(target), 'wb') as output:
        output.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        for chunk in chunks:
            output.writeframes(chunk)
    return intervals, clip_offset, sum(len(c) for c in chunks) / 32000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/ami/prepared'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, help='Optional pinned local snapshot from download_moss.py.')
    parser.add_argument('--limit', type=int, default=0, help='First N input rows; use 1 for setup smoke check.')
    parser.add_argument('--cuda-id', help='PCI_BUS_ID index inside this container, set before torch import.')
    parser.add_argument('--max-new-tokens', type=int, default=4096)
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--gap-seconds', type=float, default=.4)
    parser.add_argument('--dtype', choices=('bfloat16', 'float16', 'float32'), default='bfloat16')
    args = parser.parse_args()
    if args.gap_seconds < 0:
        raise ValueError('Gap must be nonnegative.')
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    os.environ.setdefault('HF_HOME', str(Path('models/huggingface').resolve()))
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    os.environ.setdefault('NUMBA_NUM_THREADS', str(args.cpu_threads))
    if args.cuda_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id
    import torch
    torch.set_num_threads(args.cpu_threads)
    from transformers import AutoModelForCausalLM, AutoProcessor
    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import build_transcription_messages, generate_transcription

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Check assigned GPU and the isolated MOSS environment.')
    if args.output.exists():
        raise FileExistsError(f'Use a new run folder: {args.output}')
    args.output.mkdir(parents=True)
    source = json.loads(Path(__file__).with_name('moss_sources.json').read_text())
    revision = source['model']['revision']
    model_id = 'OpenMOSS-Team/MOSS-Transcribe-Diarize'
    model_source = str(args.model_dir.resolve()) if args.model_dir else model_id
    if args.model_dir:
        downloaded = json.loads((args.model_dir / 'download_manifest.json').read_text())
        if downloaded['revision'] != revision:
            raise ValueError('Local snapshot revision differs from moss_sources.json')
    dtype = getattr(torch, args.dtype)
    device = torch.device('cuda:0')
    model = AutoModelForCausalLM.from_pretrained(model_source, revision=revision,
                trust_remote_code=True, dtype=dtype, device_map={'': str(device)},
                attn_implementation='sdpa').eval()
    processor = AutoProcessor.from_pretrained(model_source, revision=revision, trust_remote_code=True)
    rows = [json.loads(line) for line in (args.data / 'inputs.jsonl').read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    torch.manual_seed(0)
    started = time.perf_counter()
    manifest = {'method': 'MOSS 0.9B / reference-prefix enrollment / time-overlap projection',
                'model': model_id, 'sources': source, 'adapter_status': 'first_server_validation_required',
                'inputs_sha256': digest(args.data / 'inputs.jsonl'), 'labels_or_transcripts_read': False,
                'gpu': torch.cuda.get_device_name(0), 'cuda_device_order': os.environ.get('CUDA_DEVICE_ORDER'),
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'torch_cuda': torch.version.cuda, 'python': platform.python_version(),
                'dtype': args.dtype, 'max_new_tokens': args.max_new_tokens,
                'do_sample': False, 'gap_seconds': args.gap_seconds,
                'packages': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'moss-transcribe-diarize')},
                'limitations': ['Mapping depends on successful enrollment diarization, and failures count as missing.',
                                'Generation timestamp errors and boundary omissions can affect projected accuracy.',
                                'Comparison with per-act acoustic inference does not isolate a causal text/context effect.',
                                'Reference prefix is concatenated setup audio; evaluation conversation remains contiguous.']}
    try:
        manifest['pip_freeze'] = subprocess.check_output([os.sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines()
    except subprocess.SubprocessError:
        pass
    (args.output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with (args.output / 'predictions.jsonl').open('w', encoding='utf-8') as predictions:
        for index, row in enumerate(rows):
            case_dir = args.output / f'case_{index:03d}'
            case_dir.mkdir()
            audio = case_dir / 'model_input.wav'
            intervals, offset, seconds = pack_audio(row, args.data, audio, args.gap_seconds)
            print(f'{index+1}/{len(rows)} | {seconds:.1f}s input | GPU {manifest["gpu"]}', flush=True)
            messages = build_transcription_messages(str(audio.resolve()))
            torch.cuda.reset_peak_memory_stats()
            case_started = time.perf_counter()
            result = generate_transcription(model, processor, messages,
                        max_new_tokens=args.max_new_tokens, do_sample=False, device=device, dtype=dtype)
            segments = [{'start': float(s.start), 'end': float(s.end), 'speaker': str(s.speaker), 'text': s.text}
                        for s in parse_transcript(result['text'])]
            segments = validate_segments(segments, seconds)
            mapping, weights = map_enrollment(segments, intervals)
            predicted = project_predictions(segments, mapping, row['segments'], offset)
            raw = dict(result, clip_id=row['clip_id'], segments=segments, enrollment_intervals=intervals,
                       clip_offset=offset, speaker_mapping=mapping, enrollment_overlap_weights=weights,
                       inference_seconds=time.perf_counter()-case_started,
                       peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                       reached_token_limit=result.get('generated_tokens', 0) >= args.max_new_tokens)
            (case_dir / 'raw_output.json').write_text(json.dumps(raw, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            predictions.write(json.dumps({'clip_id': row['clip_id'], 'predictions': predicted}) + '\n')
            predictions.flush()
    manifest.update(adapter_status='completed_in_this_environment', elapsed_seconds=time.perf_counter()-started,
                    completed_clips=len(rows), limited_smoke_run=bool(args.limit))
    (args.output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Finished:', args.output, flush=True)


if __name__ == '__main__':
    main()
