"""Fixed padding controls and conservative voting; no labels are read here."""
from collections import Counter
import math
import wave

from moss_diagnostic_protocol import RATE, make_jobs, pack_job, read_pcm, sha

PADS = (0, 2, 30, 32)
GUARD_SECONDS = .25


def phase_jobs(rows, clips):
    originals = [j for j in make_jobs(rows, clips) if j['condition'] == 'full_forward']
    return [dict(j, condition=f'pad_{pad:02d}', job_id=f'pad_{pad:02d}/case_{index:03d}/full',
                 leading_silence_seconds=pad) for pad in PADS for index, j in enumerate(originals)]


def pack_phase(job, row, data, target, gap_seconds=.4):
    info = pack_job(job, row, data, target, gap_seconds)
    original = read_pcm(target)
    pad = job['leading_silence_seconds']
    samples = bytes(round(pad * RATE) * 2) + original
    with wave.open(str(target), 'wb') as audio:
        audio.setparams((1, 2, RATE, 0, 'NONE', 'not compressed'))
        audio.writeframes(samples)
    info['enrollment_intervals'] = [dict(r, start=r['start']+pad, end=r['end']+pad)
                                    for r in info['enrollment_intervals']]
    info['clip_offset'] += pad
    info['input_audio_seconds'] = len(samples) / (RATE * 2)
    info.update(leading_silence_seconds=pad, unpadded_input_pcm_sha256=sha(original),
                input_pcm_sha256=sha(samples), input_wav_sha256=sha(target.read_bytes()),
                frontend_chunk_count=math.ceil(len(samples)/(RATE*2*30)))
    return info


def placement(start, end):
    crossing = math.floor(start/30) != math.floor((end-1e-9)/30)
    clearance = 0. if crossing else min(start % 30, 30-(start % 30), end % 30, 30-(end % 30))
    return {'start': start, 'end': end, 'crosses_30s_boundary': crossing,
            'boundary_clearance_seconds': clearance,
            'safe': not crossing and clearance >= GUARD_SECONDS}


def vote(raw_by_pad, segment_id, guarded):
    base = next(p for p in raw_by_pad[0]['predictions'] if p['segment_id'] == segment_id)
    items = []
    for pad in PADS:
        raw = raw_by_pad[pad]
        p = next(p for p in raw['predictions'] if p['segment_id'] == segment_id)
        t = next(t for t in raw['audio']['public_targets'] if t['segment_id'] == segment_id)
        at = placement(t['start']+raw['audio']['clip_offset'], t['end']+raw['audio']['clip_offset'])
        speaker = p['speaker']
        covered = p.get('coverage_seconds_by_registered_speaker', {}).get(speaker, 0.)
        evidence = max((raw['enrollment_overlap_weights'].get(anonymous, {}).get(speaker, 0.)
                        for anonymous, agent in raw['speaker_mapping'].items() if agent == speaker), default=0.)
        eligible = speaker is not None
        if guarded:
            eligible = (eligible and at['safe'] and covered >= .5*(t['end']-t['start']) and evidence >= 1.)
        items.append({'pad_seconds': pad, 'speaker': speaker, 'eligible': eligible,
                      'registration_evidence_seconds': evidence, 'target_coverage_seconds': covered, **at})
    eligible = [i for i in items if i['eligible']]
    chosen, reason = base['speaker'], 'preserve_base'
    if guarded:
        if (len(eligible) >= 2 and len({i['pad_seconds']//30 for i in eligible}) >= 2
                and len({i['speaker'] for i in eligible}) == 1):
            chosen, reason = eligible[0]['speaker'], 'unanimous_boundary_safe_votes'
    else:
        counts = Counter(i['speaker'] for i in eligible)
        ranked = counts.most_common()
        if ranked and ranked[0][1] >= 2 and ranked[0][1]/len(eligible) >= .75:
            chosen, reason = ranked[0][0], 'three_quarter_majority'
    return {'segment_id': segment_id, 'speaker': chosen, 'base_speaker': base['speaker'],
            'decision': reason, 'votes': items}
