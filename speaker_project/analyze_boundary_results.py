"""Offline audit of a boundary result ZIP; preserve all synchronized artifacts."""
import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile
import zipfile

from boundary_control import PADS, pack_phase, placement, vote
from diarization_projection import map_enrollment, overlap, project_predictions
from moss_diagnostic_protocol import sha
from score_predictions import evaluate, read_jsonl


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('runs/speaker_boundary_02'))
    parser.add_argument('--data',type=Path,default=Path('data/ami/prepared'))
    args=parser.parse_args()
    archive_path=args.run/'results_only.zip'
    with zipfile.ZipFile(archive_path) as archive:
        names=archive.namelist()
        if len(set(names))!=len(names) or archive.testzip():
            raise ValueError('Invalid ZIP.')
        files={name:archive.read(name) for name in names}
    parsed={name:json.loads(content.decode('utf-8-sig')) for name,content in files.items() if name.endswith('.json')}
    plan,summary=parsed['plan.json'],parsed['summary.json']
    plan_hash=sha(json.dumps(plan,sort_keys=True).encode())
    if sha((args.data/'inputs.jsonl').read_bytes())!=plan['inputs_sha256'] or sha((args.data/'labels.jsonl').read_bytes())!=summary['labels_sha256']:
        raise ValueError('Dataset changed.')
    source_checks={name:sha(Path(__file__).with_name(name).read_bytes().replace(b'\r\n',b'\n'))==expected
                   for name,expected in plan['adapter_sha256_lf'].items()}
    if not all(source_checks.values()):
        raise ValueError('Local code differs from the frozen experiment; use that source version.')
    inputs={r['clip_id']:r for r in read_jsonl(args.data/'inputs.jsonl')}
    truth=[r for r in read_jsonl(args.data/'labels.jsonl') if r['clip_id'] in plan['clip_ids']]
    truths={r['clip_id']:{u['segment_id']:u for u in r['utterances']} for r in truth}
    by_clip={cid:{} for cid in plan['clip_ids']}
    audit={'archive_bytes':archive_path.stat().st_size,'archive_sha256':sha(archive_path.read_bytes()),
           'archive_file_count':len(files),'zip_crc_passed':True,'source_checks':source_checks,
           'input_rebuilds':0,'reprojected_jobs':0,'methods':{},'errors':[],'voting_changes_all_targets':{}}
    with tempfile.TemporaryDirectory(prefix='boundary-audit-') as folder:
        for job in plan['jobs']:
            raw=parsed['jobs/'+job['job_id']+'/raw_output.json']
            if raw['job']!=job or raw['plan_sha256']!=plan_hash or not raw['completed']:
                raise ValueError('Raw result differs from plan.')
            if (raw['runtime']['packages']!=plan['expected_packages'] or raw['runtime']['profile']!=plan['profile']
                    or raw['runtime']['torch_cuda']!=plan['torch_cuda']):
                raise ValueError('Runtime changed.')
            rebuilt=pack_phase(job,inputs[job['clip_id']],args.data,Path(folder)/'input.wav',plan['profile']['gap_seconds'])
            if rebuilt!=raw['audio']:
                raise ValueError('Rebuilt input differs from saved audio hashes/timing.')
            mapping,weights=map_enrollment(raw['segments'],rebuilt['enrollment_intervals'])
            predicted=project_predictions(raw['segments'],mapping,rebuilt['public_targets'],rebuilt['clip_offset'])
            if mapping!=raw['speaker_mapping'] or weights!=raw['enrollment_overlap_weights'] or predicted!=raw['predictions']:
                raise ValueError('Mapping/projection differs from saved predictions.')
            audit['input_rebuilds']+=1;audit['reprojected_jobs']+=1
            by_clip[job['clip_id']][job['leading_silence_seconds']]=raw
    methods={f'pad_{pad:02d}':[{'clip_id':cid,'predictions':by_clip[cid][pad]['predictions']} for cid in plan['clip_ids']] for pad in PADS}
    for name,guarded in (('majority_vote',False),('boundary_guard',True)):
        methods[name]=[{'clip_id':cid,'predictions':[vote(by_clip[cid],t['segment_id'],guarded)
                      for t in by_clip[cid][0]['audio']['public_targets']]} for cid in plan['clip_ids']]
    original={(r['clip_id'],p['segment_id']):p['speaker'] for r in methods['pad_00'] for p in r['predictions']}
    baseline=evaluate(truth,methods['pad_00']);baseline_by_id={(d['clip_id'],d['segment_id']):d for d in baseline['details']}
    for name,rows in methods.items():
        stored_rows=[json.loads(line) for line in files[f'methods/{name}/predictions.jsonl'].decode().splitlines() if line.strip()]
        if rows!=stored_rows:
            raise ValueError('Recomputed method differs from saved method.')
        scores=evaluate(truth,rows)
        if scores!=parsed[f'methods/{name}/scores.json']:
            raise ValueError('Recomputed scores differ from stored scores.')
        fixed=[d for d in scores['details'] if d['correct'] and not baseline_by_id[(d['clip_id'],d['segment_id'])]['correct']]
        broken=[d for d in scores['details'] if not d['correct'] and baseline_by_id[(d['clip_id'],d['segment_id'])]['correct']]
        fresh={'correct':sum(d['correct'] for d in scores['details']),'scored':len(scores['details']),
               'pairs_correct':sum(p['both_correct'] for p in scores['same_person_details']),
               'pairs':len(scores['same_person_details']),'fixed':len(fixed),'new_errors':len(broken),
               'missing':sum(d['missing'] for d in scores['details']),'changed_segments':{'fixed':fixed,'new_errors':broken}}
        if fresh!=summary['summary'][name]:
            raise ValueError('Summary differs.')
        audit['methods'][name]=dict(fresh,per_clip={cid:{'correct':sum(d['correct'] for d in scores['details'] if d['clip_id']==cid),
                                             'scored':sum(d['clip_id']==cid for d in scores['details'])} for cid in plan['clip_ids']})
        if name in ('majority_vote','boundary_guard'):
            audit['voting_changes_all_targets'][name]=[{'clip_id':r['clip_id'],'segment_id':p['segment_id'],
                'before':original[(r['clip_id'],p['segment_id'])],'after':p['speaker'],'decision':p['decision']}
                for r in rows for p in r['predictions'] if p['speaker']!=original[(r['clip_id'],p['segment_id'])]]
    for cid,conditions in by_clip.items():
        for pad,raw in conditions.items():
            for p in raw['predictions']:
                u=truths[cid][p['segment_id']]
                if not u['score_eligible'] or p['speaker']==u['speaker']:
                    continue
                t=next(t for t in raw['audio']['public_targets'] if t['segment_id']==p['segment_id'])
                left,right=t['start']+raw['audio']['clip_offset'],t['end']+raw['audio']['clip_offset']
                overlaps=[dict(s,target_overlap_seconds=overlap(left,right,s['start'],s['end']))
                          for s in raw['segments'] if overlap(left,right,s['start'],s['end'])>0]
                kind='wrong_mapped_speaker' if p['speaker'] is not None else 'unmapped_anonymous_id' if overlaps else 'missing_target_or_timestamp'
                audit['errors'].append({'clip_id':cid,'pad':pad,'utterance':u,'prediction':p,'kind':kind,
                    'overlapping_raw_segments':overlaps,'placement':placement(left,right),
                    'mapping':raw['speaker_mapping'],'enrollment_overlap_weights':raw['enrollment_overlap_weights']})
    audit['error_type_counts']={str(pad):dict(Counter(e['kind'] for e in audit['errors'] if e['pad']==pad)) for pad in PADS}
    audit['baseline_repeat_checks']=summary['baseline_repeat_checks']
    audit['total_inference_seconds']=sum(r['inference_seconds'] for v in by_clip.values() for r in v.values())
    audit['peak_cuda_allocated_gib']=max(r['peak_cuda_allocated_bytes'] for v in by_clip.values() for r in v.values())/2**30
    audit['all_local_files_match']=all((args.run/name).is_file() and (args.run/name).read_bytes()==content for name,content in files.items())
    output=args.run/'analysis';output.mkdir(exist_ok=True)
    (output/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:value for key,value in audit.items() if key not in ('methods','errors')},indent=2))


if __name__=='__main__':
    main()
