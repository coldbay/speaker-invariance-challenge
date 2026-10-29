"""Pure-stdlib 2x2 diagnostic protocol; no labels or transcripts are read."""
import hashlib
import math
from pathlib import Path
import wave

DEFAULT_CLIPS = ('IS1009d_question_answer_reversal_01', 'IS1000d_question_answer_reversal_01')
CONDITIONS = ('full_forward', 'full_reverse', 'isolated_forward', 'isolated_reverse')
RATE = 16000


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_pcm(path, expected=None):
    if expected and sha(path.read_bytes()) != expected:
        raise ValueError(f'Checksum mismatch: {path}')
    with wave.open(str(path), 'rb') as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, RATE):
            raise ValueError(f'Expected mono PCM16 16kHz: {path}')
        return audio.readframes(audio.getnframes())


def selected_segments(row):
    # Public boundaries only. Overlap/score eligibility stays in the scorer.
    return [s for s in row['segments'] if .8 <= s['end'] - s['start'] <= 15]


def make_jobs(rows, clip_ids):
    by_id = {r['clip_id']: r for r in rows}
    if len(set(clip_ids)) != len(clip_ids) or set(clip_ids) - set(by_id):
        raise ValueError('Clip IDs must be unique and present in inputs.jsonl.')
    jobs = []
    for condition in CONDITIONS:
        for index, cid in enumerate(clip_ids):
            row = by_id[cid]
            targets = selected_segments(row)
            if not targets:
                raise ValueError(f'No public segments in the fixed 0.8–15s range: {cid}')
            agents = sorted(row['registered_speakers'], reverse=condition.endswith('_reverse'))
            groups = [[s] for s in targets] if condition.startswith('isolated_') else [targets]
            for group in groups:
                name = group[0]['segment_id'] if condition.startswith('isolated_') else 'full'
                jobs.append({'job_id': f'{condition}/case_{index:03d}/{name}',
                             'condition': condition, 'clip_id': cid,
                             'context': 'isolated' if condition.startswith('isolated_') else 'full',
                             'reference_order': agents,
                             'target_segment_ids': [s['segment_id'] for s in group]})
    return jobs


def pack_job(job, row, data, output, gap_seconds=.4):
    if sorted(job['reference_order']) != sorted(row['registered_speakers']):
        raise ValueError('Reference order must be a permutation of all registered speakers.')
    if len(set(job['reference_order'])) != len(job['reference_order']):
        raise ValueError('Duplicate registered speaker in reference order.')
    original = read_pcm(data / row['audio'], row['audio_sha256'])
    source_frames = len(original) // 2
    targets = {s['segment_id']: s for s in row['segments']}
    selected = [targets[sid] for sid in job['target_segment_ids']]
    if job['context'] == 'isolated':
        if len(selected) != 1:
            raise ValueError('Each isolated job contains exactly one target.')
        left = max(0, math.floor(selected[0]['start'] * RATE))
        right = min(source_frames, math.ceil(selected[0]['end'] * RATE))
    else:
        left, right = 0, source_frames
    evaluation = original[left * 2:right * 2]
    if not evaluation:
        raise ValueError('Empty evaluation range.')
    chunks, intervals, reference_hashes = [], [], []
    frames = 0
    gap = bytes(round(gap_seconds * RATE) * 2)
    for speaker in job['reference_order']:
        for ref in row['registered_speakers'][speaker]:
            samples = read_pcm(data / ref['path'], ref['sha256'])
            length = len(samples) // 2
            intervals.append({'speaker': speaker, 'start': frames / RATE, 'end': (frames + length) / RATE})
            reference_hashes.append(ref['sha256'])
            chunks.extend((samples, gap))
            frames += length + len(gap) // 2
    offset = frames / RATE
    chunks.append(evaluation)
    public_targets = [dict(s, start=s['start'] - left / RATE, end=s['end'] - left / RATE) for s in selected]
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), 'wb') as audio:
        audio.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))
        for chunk in chunks:
            audio.writeframes(chunk)
    return {'enrollment_intervals': intervals, 'clip_offset': offset,
            'public_targets': public_targets,
            'input_audio_seconds': (frames + len(evaluation) // 2) / RATE,
            'evaluation_source_frame_start': left, 'evaluation_source_frame_end': right,
            'evaluation_pcm_sha256': sha(evaluation), 'input_pcm_sha256': sha(b''.join(chunks)),
            'input_wav_sha256': sha(output.read_bytes()),
            'reference_sha256_multiset': sorted(reference_hashes)}
