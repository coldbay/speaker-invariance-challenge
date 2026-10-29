"""Boundary-alignment diagnostic and frozen conservative correction candidate.

Prepare/inference read public inputs only. Gold labels are opened exclusively
after all model calls, by report(), or explicitly via --report-only.
"""
import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import time
import zipfile

from boundary_control import GUARD_SECONDS, PADS, pack_phase, phase_jobs, placement, vote
from diarization_projection import map_enrollment, project_predictions, validate_segments
from moss_diagnostic_protocol import DEFAULT_CLIPS, sha
from run_moss_diagnostics import write_json
from score_predictions import evaluate, read_jsonl


def report(run, data, plan, baseline=None):
    truth = [r for r in read_jsonl(data/'labels.jsonl') if r['clip_id'] in plan['clip_ids']]
    plan_hash = sha(json.dumps(plan, sort_keys=True).encode())
    raw_by_clip = {cid: {} for cid in plan['clip_ids']}
    raw_rows = []
    for job in plan['jobs']:
        path = run/'jobs'/job['job_id']/'raw_output.json'
        if not path.exists():
            raise ValueError('Incomplete experiment; run the same inference command to resume.')
        raw = json.loads(path.read_text(encoding='utf-8'))
        if raw['plan_sha256'] != plan_hash or raw['job'] != job or not raw['completed']:
            raise ValueError('Result differs from plan.')
        if raw['runtime']['packages'] != plan['expected_packages'] or raw['runtime']['profile'] != plan['profile']:
            raise ValueError('Result runtime differs from plan.')
        raw_by_clip[job['clip_id']][job['leading_silence_seconds']] = raw
        raw_rows.append(raw)
    for cid, conditions in raw_by_clip.items():
        if set(conditions) != set(PADS):
            raise ValueError('Missing a padding condition.')
        if len({r['audio']['unpadded_input_pcm_sha256'] for r in conditions.values()}) != 1:
            raise ValueError('Speech or internal timing changed across padding conditions.')
        expected = {s['segment_id'] for s in conditions[0]['audio']['public_targets']}
        if any({p['segment_id'] for p in r['predictions']} != expected for r in conditions.values()):
            raise ValueError('Target sets differ across conditions.')
    methods = {}
    for pad in PADS:
        methods[f'pad_{pad:02d}'] = [{'clip_id': cid, 'predictions': raw_by_clip[cid][pad]['predictions']} for cid in plan['clip_ids']]
    for name, guarded in (('majority_vote', False), ('boundary_guard', True)):
        methods[name] = [{'clip_id': cid, 'predictions': [vote(raw_by_clip[cid], t['segment_id'], guarded)
                         for t in raw_by_clip[cid][0]['audio']['public_targets']]} for cid in plan['clip_ids']]
    baseline_scores = evaluate(truth, methods['pad_00'])
    base_lookup = {(d['clip_id'],d['segment_id']): d for d in baseline_scores['details']}
    summary = {}
    for name, predictions in methods.items():
        present = {r['clip_id']: {p['segment_id'] for p in r['predictions']} for r in predictions}
        if any(u['score_eligible'] and u['segment_id'] not in present[r['clip_id']] for r in truth for u in r['utterances']):
            raise ValueError('Scored target missing.')
        scores = evaluate(truth, predictions)
        fixed = [d for d in scores['details'] if d['correct'] and not base_lookup[(d['clip_id'],d['segment_id'])]['correct']]
        broken = [d for d in scores['details'] if not d['correct'] and base_lookup[(d['clip_id'],d['segment_id'])]['correct']]
        summary[name] = {'correct': sum(d['correct'] for d in scores['details']), 'scored': len(scores['details']),
                         'pairs_correct': sum(p['both_correct'] for p in scores['same_person_details']),
                         'pairs': len(scores['same_person_details']), 'fixed': len(fixed), 'new_errors': len(broken),
                         'missing': sum(d['missing'] for d in scores['details']),
                         'changed_segments': {'fixed': fixed, 'new_errors': broken}}
        folder = run/'methods'/name
        folder.mkdir(parents=True, exist_ok=True)
        (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions), encoding='utf-8')
        write_json(folder/'scores.json', scores)
    positions = []
    for raw in raw_rows:
        cid = raw['job']['clip_id']
        for t in raw['audio']['public_targets']:
            p = next(p for p in raw['predictions'] if p['segment_id'] == t['segment_id'])
            positions.append({'clip_id': cid, 'pad': raw['job']['leading_silence_seconds'], 'segment_id': t['segment_id'],
                              'speaker': p['speaker'], **placement(t['start']+raw['audio']['clip_offset'], t['end']+raw['audio']['clip_offset'])})
    repeat_checks = []
    if baseline is not None:
        if sha((baseline/'run_manifest.json').read_bytes()) != plan['baseline_manifest_sha256']:
            raise ValueError('Pilot manifest changed.')
        previous = {json.loads(p.read_text(encoding='utf-8'))['clip_id']: p for p in baseline.glob('case_*/raw_output.json')}
        for cid in plan['clip_ids']:
            path=previous[cid];old=json.loads(path.read_text(encoding='utf-8'));zero=raw_by_clip[cid][0]
            wav=path.parent/'model_input.wav'
            repeat_checks.append({'clip_id':cid,'same_text':old['text']==zero['text'],
                                  'same_segments':old['segments']==zero['segments'],
                                  'same_input_wav':sha(wav.read_bytes())==zero['audio']['input_wav_sha256'] if wav.exists() else None})
    audit = {'status': 'exploratory_phase_control_and_correction_candidate', 'summary': summary,
             'labels_sha256': sha((data/'labels.jsonl').read_bytes()), 'positions': positions,
             'baseline_repeat_checks':repeat_checks,
             'total_inference_seconds': sum(r['inference_seconds'] for r in raw_rows),
             'registration': [{'clip_id': r['job']['clip_id'], 'pad': r['job']['leading_silence_seconds'],
                               'mapped': len(r['speaker_mapping']), 'registered': len(r['job']['reference_order']),
                               'chunks': r['audio']['frontend_chunk_count'],
                               'token_limit':r['reached_token_limit']} for r in raw_rows]}
    write_json(run/'summary.json', audit)
    lines = ['# 30초 경계 정렬 진단과 보수적 합의 보정', '',
             '학습 모델의 30초 인코더 길이는 유지한다. 등록·대화 전체 앞 무음만 0/2/30/32초로 바꿨다. '
             '0과30, 2와32는 각각 30초 경계 정렬이 같고 절대 시각·길이는 다르다. 기존 오류를 본 뒤 설계한 탐색 실험이다.', '',
             '| 방법 | 발화 정답 | 동일 인물 쌍 | 수정 / 새 오류 | 누락 |', '|---|---:|---:|---:|---:|']
    for name, s in summary.items():
        lines.append(f'| {name} | {s["correct"]}/{s["scored"]} | {s["pairs_correct"]}/{s["pairs"]} | {s["fixed"]} / {s["new_errors"]} | {s["missing"]} |')
    lines += ['', '## 0초 조건 재현 검사', '']
    for r in repeat_checks:
        lines.append(f'- {r["clip_id"]}: 입력 WAV 동일={r["same_input_wav"]}, 전사 동일={r["same_text"]}, 구간 동일={r["same_segments"]}.')
    lines += ['', 'majority_vote는 유효한 화자 예측 중 75% 이상·최소2개의 지지를 받을 때 수정한다. '
              'boundary_guard는 경계를 가로지르지 않고 경계에서0.25초 이상 떨어진 입력만 사용한다. '
              '대상 구간50% 이상 출력, 예측 화자의 등록 근거1초 이상, 30초 단위 이동 그룹0/1 모두에서 총2개 이상 만장일치일 때만 수정한다. '
              '조건이 부족하면 원래 pad_00 예측을 유지한다. 이 규칙은 실행 전에 plan.json으로 고정한다.', '',
              '## 등록 연결과 인코더 청크 수', '', '| 대화 | 앞 무음 초 | 등록 연결 | 청크 수 | 토큰 상한 도달 |', '|---|---:|---:|---:|---|']
    for r in audit['registration']:
        lines.append(f'| {r["clip_id"]} | {r["pad"]} | {r["mapped"]}/{r["registered"]} | {r["chunks"]} | {r["token_limit"]} |')
    truth_map = {(r['clip_id'],u['segment_id']):u for r in truth for u in r['utterances']}
    lookups = {name: {(r['clip_id'],p['segment_id']):p['speaker'] for r in rows for p in r['predictions']} for name,rows in methods.items()}
    lines += ['', '## 발화별 화자', '', '| 대화 / 발화 | 정답 | 0초 | 2초 | 30초 | 32초 | 단순 합의 | 경계 보정 |', '|---|---|---|---|---|---|---|---|']
    for (cid,sid),u in truth_map.items():
        if u['score_eligible']:
            lines.append('| '+cid[:7]+' / '+sid+' | '+u['speaker']+' | '+' | '.join(str(lookups[name].get((cid,sid))) for name in methods)+' |')
    lines += ['', '## 해석', '',
              '- 0/30에서 반복되는 오류가 2/32에서 함께 사라지면 경계 정렬 민감성의 근거가 강해진다. 절대 시각·무음·청크 수 영향까지 완전히 제거한 증명은 아니다.',
              '- 같은 정렬인 0과30 또는2와32가 다르면 절대 위치·입력 길이·추가 빈 청크·등록 출력 변화도 확인한다.',
              '- 원래 전체 대화·등록 내용이 동일하므로 발화별 분할보다 자연 대화 문맥을 더 잘 통제한다. 그러나 의미 편향만을 증명하지 않는다.',
              '- 보정은 화자 귀속 결과만 생성한다. 원래 MOSS 전사·시각은 보존하며 ASR 누락이 고쳐졌다고 주장하지 않는다.',
              '- 최고 결과를 고른 뒤 새로운 대화에서 재검증해야 한다. 현재 2개 대화의 점수를 최종 독립 평가로 보고하지 않는다.', '']
    (run/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    with zipfile.ZipFile(run/'results_only.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run.rglob('*')):
            if path.is_file() and path.suffix in ('.json','.jsonl','.md') and '.scratch' not in path.parts:
                archive.write(path,path.relative_to(run).as_posix())
    with zipfile.ZipFile(run/'results_only.zip') as archive:
        if archive.testzip():
            raise ValueError('Result archive CRC failure.')
    print('Report:',run/'REPORT.md',flush=True)
    print('Download:',run/'results_only.zip',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=Path('data/ami/prepared'))
    parser.add_argument('--baseline',type=Path,default=Path('runs/speaker_moss_pilot_01'))
    parser.add_argument('--cuda-id')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only',action='store_true')
    mode.add_argument('--report-only',action='store_true')
    args = parser.parse_args()
    base = json.loads((args.baseline/'run_manifest.json').read_text())
    rows = read_jsonl(args.data/'inputs.jsonl')
    if sha((args.data/'inputs.jsonl').read_bytes()) != base['inputs_sha256']:
        raise ValueError('Inputs differ from pilot.')
    plan = {'protocol':'boundary-phase-control-v1','inputs_sha256':base['inputs_sha256'],
            'baseline_manifest_sha256':sha((args.baseline/'run_manifest.json').read_bytes()),
            'sources':base['sources'],'expected_packages':base['packages'],'torch_cuda':base['torch_cuda'],
            'clip_ids':list(DEFAULT_CLIPS),'pads_seconds':list(PADS),
            'profile':{'dtype':base['dtype'],'max_new_tokens':base['max_new_tokens'],'gap_seconds':base['gap_seconds'],
                       'do_sample':False,'seed':0,'attention':'sdpa','cpu_threads':2},
            'correction_rule':{'guard_seconds':GUARD_SECONDS,'minimum_votes':2,'require_both_30s_groups':True,
                               'minimum_coverage_fraction':.5,'minimum_registration_seconds':1.,'require_unanimity':True},
            'adapter_sha256_lf':{n:sha(Path(__file__).with_name(n).read_bytes().replace(b'\r\n',b'\n'))
                                 for n in ('run_boundary_control.py','boundary_control.py','moss_diagnostic_protocol.py','diarization_projection.py')},
            'jobs':phase_jobs(rows,list(DEFAULT_CLIPS))}
    args.output.mkdir(parents=True,exist_ok=True)
    plan_path = args.output/'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding='utf-8')) != plan:
            raise ValueError('Plan/code changed. Use a new output folder.')
    else:
        if any(args.output.iterdir()):
            raise ValueError('Output folder is not empty; use a new folder.')
        write_json(plan_path,plan)
    print(json.dumps({'model_calls':len(plan['jobs']),'pads_seconds':list(PADS),'output':str(args.output)},indent=2),flush=True)
    if args.prepare_only:
        print('Prepared only. No GPU or model download.',flush=True)
        return
    if args.report_only:
        report(args.output,args.data,plan,args.baseline)
        return
    os.environ.setdefault('CUDA_DEVICE_ORDER','PCI_BUS_ID')
    os.environ.setdefault('HF_HOME',str(Path('models/huggingface').resolve()))
    os.environ.setdefault('HF_HUB_DISABLE_XET','1')
    os.environ.setdefault('NUMBA_NUM_THREADS','2')
    if args.cuda_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES']=args.cuda_id
    import torch
    from transformers import AutoModelForCausalLM,AutoProcessor
    from moss_transcribe_diarize import parse_transcript
    from moss_transcribe_diarize.inference_utils import build_transcription_messages,generate_transcription
    torch.set_num_threads(2)
    packages={name:importlib.metadata.version(name) for name in base['packages']}
    if packages != base['packages'] or torch.version.cuda != base['torch_cuda']:
        raise ValueError('Use the same successful MOSS pilot environment.')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable.')
    runtime={'packages':packages,'profile':plan['profile'],'gpu':torch.cuda.get_device_name(0),
             'torch_cuda':torch.version.cuda,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES')}
    plan_hash=sha(json.dumps(plan,sort_keys=True).encode())
    pending=[]
    for job in plan['jobs']:
        path=args.output/'jobs'/job['job_id']/'raw_output.json'
        if path.exists():
            old=json.loads(path.read_text(encoding='utf-8'))
            if old.get('plan_sha256')!=plan_hash or old.get('runtime')!=runtime or old.get('job')!=job or not old.get('completed'):
                raise ValueError('Resume result differs from current plan/runtime.')
        else:
            pending.append(job)
    write_json(args.output/'runtime.json',runtime)
    if pending:
        device=torch.device('cuda:0');dtype=getattr(torch,base['dtype'])
        model_id='OpenMOSS-Team/MOSS-Transcribe-Diarize';revision=base['sources']['model']['revision']
        model=AutoModelForCausalLM.from_pretrained(model_id,revision=revision,trust_remote_code=True,
            dtype=dtype,device_map={'':str(device)},attn_implementation='sdpa').eval()
        processor=AutoProcessor.from_pretrained(model_id,revision=revision,trust_remote_code=True)
        by_id={r['clip_id']:r for r in rows}
        scratch=args.output/'.scratch/model_input.wav'
        for index,job in enumerate(pending):
            audio=pack_phase(job,by_id[job['clip_id']],args.data,scratch,base['gap_seconds'])
            print(f'{len(plan["jobs"])-len(pending)+index+1}/{len(plan["jobs"])} {job["job_id"]}',flush=True)
            torch.manual_seed(0);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
            result=generate_transcription(model,processor,build_transcription_messages(str(scratch.resolve())),
                max_new_tokens=base['max_new_tokens'],do_sample=False,device=device,dtype=dtype)
            torch.cuda.synchronize();elapsed=time.perf_counter()-started
            segments=validate_segments([{'start':float(s.start),'end':float(s.end),'speaker':str(s.speaker),'text':s.text}
                                      for s in parse_transcript(result['text'])],audio['input_audio_seconds'])
            mapping,weights=map_enrollment(segments,audio['enrollment_intervals'])
            predictions=project_predictions(segments,mapping,audio['public_targets'],audio['clip_offset'])
            raw=dict(result,job=job,completed=True,plan_sha256=plan_hash,runtime=runtime,audio=audio,
                     segments=segments,speaker_mapping=mapping,enrollment_overlap_weights=weights,
                     predictions=predictions,inference_seconds=elapsed,
                     peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                     reached_token_limit=result.get('generated_tokens',0)>=base['max_new_tokens'])
            path=args.output/'jobs'/job['job_id']/'raw_output.json';path.parent.mkdir(parents=True,exist_ok=True)
            write_json(path,raw)
    report(args.output,args.data,plan,args.baseline)


if __name__=='__main__':
    main()
