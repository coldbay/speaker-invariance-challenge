"""Held-out 4-speaker set for the invariance benchmark (data/ami/heldout/).

Meeting groups never used before (not ES2003/TS3006/IS1000/IS1009), so all
speakers are unseen. Two kinds of contiguous windows, selected from manual
annotations only (never model outputs):
  question_answer_reversal  every 4-speaker reversal candidate from the full
                            annotation scan (same rule as the starter set)
  overlap_dense             per extra group, the 60 s window with the most
                            overlapped 0.8-15 s acts while all 4 speakers talk
Audio is a byte-range copy of the official Mix-Headset WAV. Enrollment uses
two clean 3 s references per speaker outside every selected window.
The existing data/ami/prepared/ files are not touched.
"""
import argparse
import json
from pathlib import Path

from build_protocol import write_jsonl
from focus_protocol import add_focus_pairs
from prepare_ami import BASE, download_clip, overlap_fraction, parse_meeting, wav_header, write_json

USED_GROUPS = {'ES2003', 'TS3006', 'IS1000', 'IS1009'}
DENSE_MEETINGS = ('ES2008a', 'IS1006b', 'TS3004c', 'IS1008c')
WINDOW, STEP = 60., 5.


def acts_in(data, start, end):
    return sorted((a for a in data['acts'].values() if a['end'] > start and a['start'] < end), key=lambda a: (a['start'], a['speaker']))


def dense_window(data):
    best = None
    last = max(a['end'] for a in data['acts'].values())
    t = 0.
    while t + WINDOW <= last:
        inside = [a for a in acts_in(data, t, t+WINDOW) if a['start'] >= t and a['end'] <= t+WINDOW]
        if len({a['speaker'] for a in inside}) == 4:
            sized = [a for a in inside if .8 <= a['end']-a['start'] <= 15]
            fractions = [overlap_fraction(a['start'], a['end'], a['speaker'], data['words']) for a in sized]
            score = (sum(f > .05 for f in fractions), sum(f <= .05 for f in fractions))
            if score[1] >= 5 and (best is None or score > best[0]):
                best = (score, t)
        t += STEP
    if best is None:
        raise ValueError(f'No dense 4-speaker window: {data["meeting"]}')
    return best[1], best[1]+WINDOW


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/ami'))
    parser.add_argument('--reference-seconds', type=float, default=3)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root/'heldout'
    if (out/'inputs.jsonl').exists():
        raise FileExistsError('Held-out set already built; keep it frozen.')
    scan = json.loads((root/'scan/all_candidates.json').read_text(encoding='utf-8'))
    cases = [dict(c, clip_id=f'{c["meeting"]}_question_answer_reversal_01') for c in scan
             if c['event'] == 'question_answer_reversal' and len(c['speakers_present']) == 4
             and c['meeting'][:6] not in USED_GROUPS and c['duration'] <= 80]
    if len({c['meeting'][:6] for c in cases}) != len(cases):
        raise ValueError('Expected one reversal per meeting group.')
    parsed = {m: parse_meeting(root/'annotations', m) for m in sorted({c['meeting'] for c in cases} | set(DENSE_MEETINGS))}
    for meeting in DENSE_MEETINGS:
        if meeting[:6] in USED_GROUPS | {c['meeting'][:6] for c in cases}:
            raise ValueError('Dense meeting group must be new.')
        start, end = dense_window(parsed[meeting])
        cases.append({'meeting': meeting, 'meeting_type': parsed[meeting]['meeting_type'], 'event': 'overlap_dense',
                      'start': start, 'end': end, 'duration': end-start, 'speakers_present': sorted(parsed[meeting]['speakers']),
                      'review_status': 'automatic_candidate_not_human_verified', 'clip_id': f'{meeting}_overlap_dense_01'})
    known = json.loads((root/'scan/speaker_groups.json').read_text(encoding='utf-8'))
    old_people = {p for g, s in known.items() if g[:6] in USED_GROUPS for p in s.values()}
    new_people = {p for m in parsed for p in parsed[m]['speakers'].values()}
    if old_people & new_people:
        raise ValueError('Held-out speakers overlap the starter set.')
    headers = {}
    for case in cases:
        meeting = case['meeting']
        url = f'{BASE}/{meeting}/audio/{meeting}.Mix-Headset.wav'
        headers.setdefault(meeting, wav_header(url))
        print(f'clip {case["clip_id"]}: {case["duration"]:.1f}s', flush=True)
        case['audio'] = download_clip(url, headers[meeting], case['start'], case['end'], out/'audio'/(case['clip_id']+'.wav'))
    enrollment = {}
    for meeting in sorted({c['meeting'] for c in cases}):
        data = parsed[meeting]
        excluded = [(c['start']-2, c['end']+2) for c in cases if c['meeting'] == meeting]
        enrollment[meeting] = {}
        for agent in sorted({s for c in cases if c['meeting'] == meeting for s in c['speakers_present']}):
            refs = []
            for act in sorted(data['acts'].values(), key=lambda a: a['start']):
                if act['speaker'] != agent or act['end']-act['start'] < args.reference_seconds+.2:
                    continue
                start = act['start']+.1; end = start+args.reference_seconds
                if any(end > l and start < r for l, r in excluded) or overlap_fraction(start, end, agent, data['words']) > .005:
                    continue
                url = f'{BASE}/{meeting}/audio/{meeting}.Mix-Headset.wav'
                asset = download_clip(url, headers[meeting], start, end, out/'references'/f'{meeting}_{agent}_ref{len(refs)+1:02d}.wav')
                asset.update(start=start, end=end, source_dialogue_act=act['id'], global_speaker=data['speakers'][agent])
                refs.append(asset)
                if len(refs) == 2:
                    break
            if not refs:
                raise ValueError(f'No clean enrollment: {meeting}/{agent}')
            enrollment[meeting][agent] = refs
    inputs, labels = [], []
    for case in cases:
        meeting, data = case['meeting'], parsed[case['meeting']]
        rate = headers[meeting]['sample_rate']
        source_start = case['audio']['source_frame_start']/rate
        source_end = case['audio']['source_frame_end_exclusive']/rate
        public, truth = [], []
        targets = {case[k]['answer']['id']: k.split('_')[1] for k in ('qa_before', 'qa_after')} if 'qa_before' in case else {}
        for i, act in enumerate(acts_in(data, source_start, source_end)):
            sid = f'segment_{i:03d}'
            left, right = max(act['start'], source_start), min(act['end'], source_end)
            fraction = overlap_fraction(left, right, act['speaker'], data['words'])
            truncated = act['start'] < source_start or act['end'] > source_end
            public.append({'segment_id': sid, 'start': left-source_start, 'end': right-source_start})
            truth.append({'segment_id': sid, 'speaker': act['speaker'], 'global_speaker': act['global_speaker'],
                          'source_dialogue_act': act['id'], 'dialogue_act': act['dialogue_act'], 'text': act['text'],
                          'score_eligible': not truncated and .8 <= right-left <= 15 and fraction <= .05,
                          'other_speaker_word_overlap_fraction': fraction, 'truncated': truncated,
                          'position': 'answer_'+targets[act['id']] if act['id'] in targets else 'context'})
        refs = {a: [{'path': str(Path(r['path']).relative_to(out)).replace('\\', '/'), 'sha256': r['sha256']}
                    for r in enrollment[meeting][a]] for a in case['speakers_present']}
        inputs.append({'clip_id': case['clip_id'], 'split': 'heldout',
                       'audio': str(Path(case['audio']['path']).relative_to(out)).replace('\\', '/'),
                       'audio_sha256': case['audio']['sha256'], 'registered_speakers': refs, 'segments': public})
        row = {'clip_id': case['clip_id'], 'split': 'heldout', 'meeting': meeting, 'meeting_type': case['meeting_type'],
               'event': case['event'], 'review_status': case['review_status'], 'source_start': source_start,
               'source_end': source_end, 'utterances': truth}
        if 'qa_before' in case:
            row.update(qa_before=case['qa_before'], qa_after=case['qa_after'])
            row = add_focus_pairs(row, public)
        else:
            for u, s in zip(truth, public):
                u.update(start=s['start'], end=s['end'], duration=s['end']-s['start'])
            row.update(same_speaker_pairs=[], protocol_version='natural-qa-v2-same-person')
        labels.append(row)
    write_jsonl(out/'inputs.jsonl', inputs)
    write_jsonl(out/'labels.jsonl', labels)
    write_json(out/'enrollment_manifest.json', enrollment)
    write_json(out/'source_audio_headers.json', headers)
    write_json(out/'selected_candidates.json', cases)
    summary = {'status': 'heldout_candidates_pending_human_review', 'clips': len(cases),
               'events': {e: sum(c['event'] == e for c in cases) for e in sorted({c['event'] for c in cases})},
               'audio_seconds': sum(c['duration'] for c in cases), 'meetings': [c['meeting'] for c in cases],
               'speakers_disjoint_from_starter_set': True,
               'scored_utterances': sum(u['score_eligible'] for r in labels for u in r['utterances']),
               'utterances': sum(len(r['utterances']) for r in labels),
               'provenance': 'Contiguous PCM ranges of official AMI Mix-Headset WAV; CC BY 4.0 (see ../prepared/LICENCE_AMI.txt).'}
    write_json(out/'dataset_manifest.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
