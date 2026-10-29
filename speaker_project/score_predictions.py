"""Score registered-speaker segment predictions on the prepared AMI pilot."""

import argparse
from collections import defaultdict
import json
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def evaluate(truth_rows, prediction_rows):
    predictions={}
    for row in prediction_rows:
        clip=row['clip_id']
        if clip in predictions:
            raise ValueError(f'Duplicate prediction clip: {clip}')
        mapping={}
        for item in row['predictions']:
            if item['segment_id'] in mapping:
                raise ValueError(f'Duplicate segment prediction: {clip}/{item["segment_id"]}')
            mapping[item['segment_id']]=item.get('speaker')
        predictions[clip]=mapping
    known={row['clip_id'] for row in truth_rows}
    if set(predictions)-known:
        raise ValueError('Predictions contain unknown clip IDs.')
    details=[]
    pair_details=[]
    for row in truth_rows:
        pred=predictions.get(row['clip_id'],{})
        if set(pred)-{u['segment_id'] for u in row['utterances']}:
            raise ValueError(f'Unknown predicted segment in {row["clip_id"]}')
        for utterance in row['utterances']:
            if not utterance['score_eligible']:
                continue
            guessed=pred.get(utterance['segment_id'])
            details.append({'clip_id':row['clip_id'],'split':row['split'],'meeting':row['meeting'],
                'event':row['event'],'position':utterance['position'],
                'segment_id':utterance['segment_id'],'correct':guessed==utterance['speaker'],
                'missing':guessed is None,'review_status':row['review_status'],
                'duration':utterance.get('duration'),'speaker':utterance['speaker'],'predicted_speaker':guessed})
        for pair in row.get('same_speaker_pairs',[]):
            if not pair['score_eligible']:
                continue
            first=pred.get(pair['before_segment_id'])
            later=pred.get(pair['after_segment_id'])
            pair_details.append(dict(pair,clip_id=row['clip_id'],split=row['split'],event=row['event'],
                before_correct=first==pair['speaker'],after_correct=later==pair['speaker'],
                both_correct=first==later==pair['speaker'],
                missing=first is None or later is None,
                inconsistent=first!=later if first is not None and later is not None else None))
    groups=defaultdict(list)
    for d in details:
        groups[(d['split'],d['event'],'all')].append(d)
        if d['position'] in ('answer_before','answer_after'):
            groups[(d['split'],d['event'],d['position'])].append(d)
    summary=[]
    for (split,event,position),items in sorted(groups.items()):
        clips=defaultdict(list)
        for item in items:
            clips[item['clip_id']].append(item['correct'])
        summary.append({'split':split,'event':event,'position':position,'clips':len(clips),
            'segments':len(items),'micro_accuracy':sum(i['correct'] for i in items)/len(items),
            'macro_clip_accuracy':sum(sum(values)/len(values) for values in clips.values())/len(clips),
            'missing_predictions':sum(i['missing'] for i in items)})
    pair_groups=defaultdict(list)
    for pair in pair_details:
        pair_groups[(pair['split'],pair['event'])].append(pair)
    pair_summary=[]
    for (split,event),items in sorted(pair_groups.items()):
        initially_correct=[p for p in items if p['before_correct']]
        complete=[p for p in items if not p['missing']]
        pair_summary.append({'split':split,'event':event,'pairs':len(items),
            'both_correct_count':sum(p['both_correct'] for p in items),
            'both_correct_rate':sum(p['both_correct'] for p in items)/len(items),
            'after_accuracy':sum(p['after_correct'] for p in items)/len(items),
            'initially_correct_pairs':len(initially_correct),
            'correct_retention':sum(p['after_correct'] for p in initially_correct)/len(initially_correct) if initially_correct else None,
            'complete_pairs':len(complete),
            'inconsistency_rate':sum(p['inconsistent'] for p in complete)/len(complete) if complete else None,
            'missing_pairs':sum(p['missing'] for p in items)})
    return {'status':'pilot_diagnostic_not_final_leaderboard',
            'pending_review_clips':sum(row['review_status']=='automatic_candidate_not_human_verified' for row in truth_rows),
            'scored_segments':len(details),'summary':summary,'details':details,
            'same_person_summary':pair_summary,'same_person_details':pair_details,
            'interpretation':'Descriptive paired diagnostics; different utterances and unmatched controls do not establish a causal role-transition effect.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels',required=True,type=Path)
    parser.add_argument('--predictions',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    result=evaluate(read_jsonl(args.labels),read_jsonl(args.predictions))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('details','same_person_details')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
