"""Package AMI candidate clips, held-out enrollment audio, and isolated labels."""

import argparse
import json
from pathlib import Path

from prepare_ami import BASE, download_clip, overlap_fraction, parse_meeting, write_json
from focus_protocol import add_focus_pairs


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('data/ami'))
    parser.add_argument('--reference-seconds',type=float,default=3)
    args=parser.parse_args()
    root=args.root.resolve()
    prepared=root/'prepared'
    cases=json.loads((prepared/'selected_candidates.json').read_text(encoding='utf-8'))
    headers=json.loads((prepared/'source_audio_headers.json').read_text(encoding='utf-8'))
    meetings=sorted({case['meeting'] for case in cases})
    parsed={m:parse_meeting(root/'annotations',m) for m in meetings}
    group_splits={'ES2003':'development','TS3006':'development','IS1000':'evaluation','IS1009':'evaluation'}
    splits={meeting:group_splits.get(meeting[:6]) for meeting in meetings}
    if any(splits[m] is None for m in meetings):
        raise ValueError('Explicitly define splits for the selected meetings before packaging.')
    people={name:{p for m in meetings if splits[m]==name for p in parsed[m]['speakers'].values()}
            for name in ('development','evaluation')}
    if people['development'] & people['evaluation']:
        raise ValueError('Development/evaluation speakers overlap.')
    enrollment={}
    for meeting in meetings:
        data=parsed[meeting]
        selected=[c for c in cases if c['meeting']==meeting]
        excluded=[(c['start']-2,c['end']+2) for c in selected]
        agents=sorted({s for c in selected for s in c['speakers_present']})
        enrollment[meeting]={}
        for agent in agents:
            references=[]
            for act in sorted(data['acts'].values(),key=lambda a:a['start']):
                if act['speaker']!=agent or act['end']-act['start']<args.reference_seconds+.2:
                    continue
                start=act['start']+.1
                end=start+args.reference_seconds
                if any(end>left and start<right for left,right in excluded):
                    continue
                if overlap_fraction(start,end,agent,data['words'])>.005:
                    continue
                source_url=f'{BASE}/{meeting}/audio/{meeting}.Mix-Headset.wav'
                path=prepared/'references'/f'{meeting}_{agent}_ref{len(references)+1:02d}.wav'
                print(f'enrollment {meeting}/{agent}: {start:.2f}-{end:.2f}s',flush=True)
                asset=download_clip(source_url,headers[meeting],start,end,path)
                asset.update(start=start,end=end,source_dialogue_act=act['id'],global_speaker=data['speakers'][agent])
                references.append(asset)
                if len(references)==2:
                    break
            if not references:
                raise ValueError(f'No clean held-out enrollment found for {meeting}/{agent}')
            enrollment[meeting][agent]=references
            write_json(prepared/'enrollment_manifest.json',enrollment)

    inputs,labels=[],[]
    for case in cases:
        if 'audio' not in case or not Path(case['audio']['path']).is_file():
            raise ValueError(f'Missing downloaded case audio: {case["clip_id"]}')
        meeting=case['meeting']
        data=parsed[meeting]
        frame_start=case['audio']['source_frame_start']
        source_start=frame_start/headers[meeting]['sample_rate']
        source_end=case['audio']['source_frame_end_exclusive']/headers[meeting]['sample_rate']
        acts=sorted((a for a in data['acts'].values() if a['end']>source_start and a['start']<source_end),key=lambda a:(a['start'],a['speaker']))
        public_segments,truth=[],[]
        target_before=case['qa_before']['answer']['id']
        target_after=case['qa_after']['answer']['id']
        for i,act in enumerate(acts):
            identifier=f'segment_{i:03d}'
            left,right=max(act['start'],source_start),min(act['end'],source_end)
            fraction=overlap_fraction(left,right,act['speaker'],data['words'])
            truncated=act['start']<source_start or act['end']>source_end
            eligible=not truncated and .8<=right-left<=15 and fraction<=.05
            public_segments.append({'segment_id':identifier,'start':left-source_start,'end':right-source_start})
            truth.append({'segment_id':identifier,'speaker':act['speaker'],'global_speaker':act['global_speaker'],
                'source_dialogue_act':act['id'],'dialogue_act':act['dialogue_act'],'text':act['text'],
                'score_eligible':eligible,'other_speaker_word_overlap_fraction':fraction,'truncated':truncated,
                'position':'answer_after' if act['id']==target_after else 'answer_before' if act['id']==target_before else 'context'})
        active_refs={agent:[{'path':str(Path(ref['path']).relative_to(prepared)).replace('\\','/'),
                            'sha256':ref['sha256']} for ref in enrollment[meeting][agent]]
                     for agent in case['speakers_present']}
        inputs.append({'clip_id':case['clip_id'],'split':splits[meeting],
            'audio':str(Path(case['audio']['path']).relative_to(prepared)).replace('\\','/'),
            'audio_sha256':case['audio']['sha256'],'registered_speakers':active_refs,'segments':public_segments})
        label_row={'clip_id':case['clip_id'],'split':splits[meeting],'meeting':meeting,'meeting_type':case['meeting_type'],
            'event':case['event'],'review_status':case['review_status'],'source_start':source_start,'source_end':source_end,
            'utterances':truth,'qa_before':case['qa_before'],'qa_after':case['qa_after']}
        labels.append(add_focus_pairs(label_row,public_segments))
    write_jsonl(prepared/'inputs.jsonl',inputs)
    write_jsonl(prepared/'labels.jsonl',labels)
    summary={'status':'pilot_candidates_pending_human_review','clip_count':len(cases),
        'event_counts':{event:sum(c['event']==event for c in cases) for event in sorted({c['event'] for c in cases})},
        'audio_seconds':sum((c['audio']['source_frame_end_exclusive']-c['audio']['source_frame_start'])/headers[c['meeting']]['sample_rate'] for c in cases),
        'reference_count':sum(len(refs) for meeting in enrollment.values() for refs in meeting.values()),
        'reference_seconds':args.reference_seconds,'speaker_sets_disjoint':True,
        'split_speaker_ids':{key:sorted(value) for key,value in people.items()},
        'clip_speaker_counts':{c['clip_id']:len(c['speakers_present']) for c in cases},
        'eligible_segment_count':sum(u['score_eligible'] for item in labels for u in item['utterances']),
        'scored_answer_after_count':sum(u['score_eligible'] and u['position']=='answer_after' for item in labels for u in item['utterances']),
        'protocol_version':'natural-qa-v2-same-person',
        'same_person_pair_count':sum(len(item['same_speaker_pairs']) for item in labels),
        'eligible_same_person_pair_count':sum(p['score_eligible'] for item in labels for p in item['same_speaker_pairs']),
        'provenance':'Contiguous PCM ranges of official AMI Mix-Headset WAV. No resampling, reordering, or speaker removal.',
        'limitations':['Small starter set with balanced selected event counts, not a representative final benchmark.',
                      'Selected meetings are scenario-elicited spontaneous meetings, not scripts read by speakers.',
                      'Event interpretations still require transcript/audio review.',
                      'References come from outside all selected clips and can be recorded after the evaluation interval.',
                      'At most 5% other-speaker aligned-word overlap is allowed for scored segments; this is not an exact overlap-free annotation.',
                      'Model outputs are stored separately in runs/; dataset labels remain pending listening review.']}
    write_json(prepared/'dataset_manifest.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
