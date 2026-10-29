"""Embedding-verified attribution on top of MOSS outputs (CPU, no labels).

Two frozen steps, applied to any MOSS raw output:
1. relink: map each anonymous MOSS ID to the enrolled speaker whose WeSpeaker
   prototype is closest to that ID's conversation audio (many-to-one, so a
   split speaker stays linked). Replaces the fragile transcript-overlap link.
2. verify: keep the relinked MOSS speaker for a target utterance unless the
   utterance embedding prefers another enrolled speaker by more than TAU
   cosine; a missing MOSS speaker is filled by the embedding choice.
MOSS transcripts and timestamps are untouched; only speaker labels change.
"""
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort

from diarization_projection import project_predictions
from run_wespeaker import fbank, read_pcm, sha256, unit

RATE = 16000
TAU = .10                 # Fixed before scoring; other values are ablations only.
MIN_CLUSTER_SEGMENT = .5  # Seconds; shorter MOSS segments are ignored for relinking.


class Embedder:
    def __init__(self, model_dir=Path('models/wespeaker-resnet34'), threads=2):
        model = model_dir/'voxceleb_resnet34_LM.onnx'
        manifest = json.loads((model_dir/'manifest.json').read_text())
        if sha256(model) != manifest['files'][model.name]['sha256']:
            raise ValueError('Model checksum mismatch')
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), options, providers=['CPUExecutionProvider'])
        self.cache = {}

    def __call__(self, key, samples):
        if key not in self.cache:
            vector = self.session.run(['embs'], {'feats': fbank(samples)[None]})[0].reshape(-1)
            self.cache[key] = unit(vector)
        return self.cache[key]


def clip_context(row, data, embed):
    """Conversation PCM, enrolled prototypes and target-utterance embeddings."""
    audio = read_pcm(data/row['audio'], row['audio_sha256'])
    prototypes = {}
    for speaker, refs in row['registered_speakers'].items():
        vectors = [embed(('ref', r['path']), read_pcm(data/r['path'], r['sha256'])) for r in refs]
        prototypes[speaker] = unit(np.mean(vectors, axis=0))
    utterances = {}
    for s in row['segments']:
        left, right = max(0, math.floor(s['start']*RATE)), min(len(audio), math.ceil(s['end']*RATE))
        if right - left >= .3*RATE:
            utterances[s['segment_id']] = embed((row['clip_id'], left, right), audio[left:right])
    return {'row': row, 'audio': audio, 'prototypes': prototypes, 'utterances': utterances}


def relink(segments, clip_offset, context, embed):
    audio, row = context['audio'], context['row']
    total = len(audio)/RATE
    sums, mapping, scores = {}, {}, {}
    for s in segments:
        start, end = max(0., s['start']-clip_offset), min(total, s['end']-clip_offset)
        if end - start < MIN_CLUSTER_SEGMENT:
            continue
        left, right = math.floor(start*RATE), math.ceil(end*RATE)
        vector = embed((row['clip_id'], left, right), audio[left:right])
        sums[str(s['speaker'])] = sums.get(str(s['speaker']), 0) + (end-start)*vector
    for anonymous, vector in sums.items():
        centroid = unit(vector)
        cos = {spk: float(centroid @ p) for spk, p in context['prototypes'].items()}
        mapping[anonymous] = max(cos, key=cos.get)
        scores[anonymous] = cos
    return mapping, scores


def verify(predictions, context, tau=TAU):
    result = []
    for p in predictions:
        vector = context['utterances'].get(p['segment_id'])
        item = dict(p, moss_speaker=p['speaker'])
        if vector is not None:
            cos = {spk: float(vector @ proto) for spk, proto in context['prototypes'].items()}
            best = max(cos, key=cos.get)
            item['embedding_cosine'] = cos
            if p['speaker'] is None:
                item.update(speaker=best, decision='filled_missing')
            elif best != p['speaker'] and cos[best] - cos[p['speaker']] > tau:
                item.update(speaker=best, decision='embedding_override')
            else:
                item['decision'] = 'kept_moss'
        else:
            item['decision'] = 'no_embedding'
        result.append(item)
    return result


def hybrid_predictions(segments, clip_offset, targets, context, embed, tau=TAU):
    """Returns (relinked-only, relinked+verified) prediction lists."""
    mapping, scores = relink(segments, clip_offset, context, embed)
    relinked = project_predictions(segments, mapping, targets, clip_offset)
    return relinked, verify(relinked, context, tau), mapping, scores
