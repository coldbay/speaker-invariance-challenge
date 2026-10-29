"""Audio-only registered-speaker baseline. Never reads labels or transcripts.

ONNX checkpoint: official WeSpeaker ResNet34-LM. Kaldi-native fbank follows
the official infer_onnx.py settings (80 bins, 25/10ms, hamming, CMN).
This is a small reference adapter, not a new speaker model or official score.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time
import wave

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_pcm(path, expected_sha=None):
    if expected_sha and sha256(path) != expected_sha:
        raise ValueError(f'Audio checksum mismatch: {path}')
    with wave.open(str(path), 'rb') as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
            raise ValueError(f'Expected mono 16kHz PCM16: {path}')
        return np.frombuffer(audio.readframes(audio.getnframes()), dtype='<i2').astype(np.float32)


def fbank(samples):
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.frame_length_ms = 25
    opts.frame_opts.frame_shift_ms = 10
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = True
    opts.frame_opts.window_type = 'hamming'
    opts.mel_opts.num_bins = 80
    opts.mel_opts.low_freq = 20
    opts.mel_opts.high_freq = 0
    opts.use_energy = False
    bank = knf.OnlineFbank(opts)
    # The official frontend scales float [-1, 1] by 32768; these are PCM units.
    bank.accept_waveform(16000, samples.tolist())
    bank.input_finished()
    values = np.stack([bank.get_frame(i) for i in range(bank.num_frames_ready)])
    return (values - values.mean(axis=0, keepdims=True)).astype(np.float32)


def unit(vector):
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError('Non-finite or zero speaker embedding')
    return vector / norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/ami/prepared'))
    parser.add_argument('--model-dir', type=Path, default=Path('models/wespeaker-resnet34'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--min-seconds', type=float, default=.3,
                        help='Shorter segments get missing predictions; all official scored pilot segments are >=0.8s.')
    args = parser.parse_args()
    started = time.perf_counter()
    model_path = args.model_dir / 'voxceleb_resnet34_LM.onnx'
    manifest = json.loads((args.model_dir / 'manifest.json').read_text())
    if sha256(model_path) != manifest['files'][model_path.name]['sha256']:
        raise ValueError('Model checksum mismatch')
    rows = [json.loads(line) for line in (args.data / 'inputs.jsonl').read_text().splitlines() if line.strip()]
    if args.output.exists():
        raise FileExistsError(f'Use a new run folder; existing output is preserved: {args.output}')
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), options, providers=['CPUExecutionProvider'])

    def embed(samples):
        features = fbank(samples)
        vector = session.run(['embs'], {'feats': features[None]})[0].reshape(-1)
        return unit(vector)

    args.output.mkdir(parents=True)
    cache, results, archive = {}, [], {}
    for row in rows:
        audio = read_pcm(args.data / row['audio'], row['audio_sha256'])
        centroids = {}
        for speaker, references in row['registered_speakers'].items():
            vectors = []
            for reference in references:
                key = reference['path']
                if key not in cache:
                    cache[key] = embed(read_pcm(args.data / key, reference['sha256']))
                vectors.append(cache[key])
            centroids[speaker] = unit(np.mean(vectors, axis=0))
        predictions = []
        for segment in row['segments']:
            start = max(0, math.floor(segment['start'] * 16000))
            end = min(len(audio), math.ceil(segment['end'] * 16000))
            samples = audio[start:end]
            result = {'segment_id': segment['segment_id'], 'duration': len(samples) / 16000}
            if len(samples) < args.min_seconds * 16000:
                result.update(speaker=None, reason='below_min_duration')
            else:
                vector = embed(samples)
                scores = {speaker: float(np.dot(vector, centroid)) for speaker, centroid in centroids.items()}
                ranked = sorted(scores, key=scores.get, reverse=True)
                result.update(speaker=ranked[0], cosine_scores=scores,
                              margin=scores[ranked[0]] - scores[ranked[1]],
                              rms_pcm=float(np.sqrt(np.mean(samples ** 2))))
                archive[row['clip_id'] + '/' + segment['segment_id']] = vector
            predictions.append(result)
        results.append({'clip_id': row['clip_id'], 'predictions': predictions})
        print(f'{len(results)}/{len(rows)} {row["clip_id"]}: {len(predictions)} segments', flush=True)
    for key, value in cache.items():
        archive['reference/' + key] = value
    np.savez_compressed(args.output / 'embeddings.npz', **archive)
    (args.output / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in results), encoding='utf-8')
    report = {'method': 'WeSpeaker-ResNet34-LM / per-act cosine / mean normalized enrollment',
              'audio_only': True, 'transcripts_or_labels_read': False,
              'model_manifest': manifest, 'inputs_sha256': sha256(args.data / 'inputs.jsonl'),
              'minimum_segment_seconds': args.min_seconds, 'threads': args.threads,
              'elapsed_seconds': time.perf_counter() - started,
              'python': platform.python_version(), 'platform': platform.platform(),
              'packages': {p: importlib.metadata.version(p) for p in ('numpy', 'onnxruntime', 'kaldi-native-fbank')},
              'frontend': 'PCM16 units; 80-bin Kaldi fbank, 25ms/10ms, hamming, dither0, snip_edges=True, utterance CMN',
              'limitations': ['Kaldi-native-fbank frontend; numeric parity with torchaudio has not been checked.',
                              'Oracle act boundaries and registered speakers; not end-to-end diarization.',
                              'Model short-utterance/domain sensitivity may affect results; no context is used.']}
    (args.output / 'run_manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f'Finished in {report["elapsed_seconds"]:.1f}s. {args.output}', flush=True)


if __name__ == '__main__':
    main()
