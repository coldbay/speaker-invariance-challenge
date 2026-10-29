"""Score completed 2x2 diagnostics; labels are read only in this reporting step."""
import argparse
from collections import Counter
import json
from pathlib import Path
import zipfile

from moss_diagnostic_protocol import CONDITIONS, sha
from score_predictions import evaluate, read_jsonl


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def combine_condition(plan, run, condition):
    plan_hash = sha(json.dumps(plan, sort_keys=True).encode())
    combined = {cid: {} for cid in plan['clip_ids']}
    raw_rows = []
    for job in (j for j in plan['jobs'] if j['condition'] == condition):
        path = run / 'jobs' / job['job_id'] / 'raw_output.json'
        if not path.exists():
            raise ValueError(f'Incomplete run: {job["job_id"]}. Re-run the same inference command to resume.')
        raw = json.loads(path.read_text(encoding='utf-8'))
        if not raw.get('completed') or raw.get('plan_sha256') != plan_hash or raw.get('job') != job:
            raise ValueError(f'Result differs from frozen plan: {job["job_id"]}')
        runtime = raw.get('runtime', {})
        if (runtime.get('profile') != plan['profile'] or runtime.get('packages') != plan['expected_packages']
                or runtime.get('torch_cuda') != plan['expected_torch_cuda']):
            raise ValueError('Result runtime differs from the frozen plan.')
        items = raw['predictions']
        identifiers = [p['segment_id'] for p in items]
        if len(identifiers) != len(set(identifiers)) or set(identifiers) != set(job['target_segment_ids']):
            raise ValueError('Missing/duplicate/extra target IDs in result.')
        target = combined[job['clip_id']]
        for p in items:
            if p['segment_id'] in target:
                raise ValueError('Duplicate target across isolated jobs.')
            target[p['segment_id']] = p
        raw_rows.append(raw)
    return [{'clip_id': cid, 'predictions': list(combined[cid].values())} for cid in plan['clip_ids']], raw_rows


def summarize(truth, predictions, raw_rows, reference_scores=None):
    present = {r['clip_id']: {p['segment_id'] for p in r['predictions']} for r in predictions}
    for row in truth:
        missing = {u['segment_id'] for u in row['utterances'] if u['score_eligible']} - present[row['clip_id']]
        if missing:
            raise ValueError(f'Scored targets excluded from plan: {missing}')
    scores = evaluate(truth, predictions)
    states = Counter(('correct' if p['before_correct'] else 'wrong') + '_to_' +
                     ('correct' if p['after_correct'] else 'wrong') for p in scores['same_person_details'])
    merged_jobs = 0
    for raw in raw_rows:
        # One second of registration overlap avoids counting tiny boundary spill.
        merged_jobs += any(sum(seconds >= 1 for seconds in w.values()) >= 2
                           for w in raw['enrollment_overlap_weights'].values())
    result = {'correct': sum(d['correct'] for d in scores['details']), 'scored': len(scores['details']),
              'missing': sum(d['missing'] for d in scores['details']),
              'pairs_both_correct': sum(p['both_correct'] for p in scores['same_person_details']),
              'pairs': len(scores['same_person_details']), 'same_person_states': dict(states),
              'model_calls': len(raw_rows),
              'calls_with_incomplete_registration': sum(len(set(r['speaker_mapping'].values())) < len(r['job']['reference_order']) for r in raw_rows),
              'calls_with_substantial_registration_merge': merged_jobs,
              'calls_reaching_token_limit': sum(r['reached_token_limit'] for r in raw_rows),
              'inference_seconds': sum(r['inference_seconds'] for r in raw_rows),
              'per_clip': {}, 'changes_vs_full_forward': {}}
    for row in truth:
        details = [d for d in scores['details'] if d['clip_id'] == row['clip_id']]
        result['per_clip'][row['clip_id']] = {'correct': sum(d['correct'] for d in details), 'scored': len(details)}
    if reference_scores:
        original = {(d['clip_id'], d['segment_id']): d for d in reference_scores['details']}
        fixed, broken = [], []
        for d in scores['details']:
            old = original[(d['clip_id'], d['segment_id'])]
            if not old['correct'] and d['correct']:
                fixed.append([d['clip_id'], d['segment_id']])
            if old['correct'] and not d['correct']:
                broken.append([d['clip_id'], d['segment_id']])
        result['changes_vs_full_forward'] = {'fixed_count': len(fixed), 'new_error_count': len(broken),
                                            'fixed_segments': fixed, 'new_error_segments': broken}
    return result, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/ami/prepared'))
    parser.add_argument('--baseline', type=Path, default=Path('runs/speaker_moss_pilot_01'))
    args = parser.parse_args()
    plan = json.loads((args.run / 'plan.json').read_text(encoding='utf-8'))
    if sha((args.data / 'inputs.jsonl').read_bytes()) != plan['inputs_sha256']:
        raise ValueError('Inputs changed since plan creation.')
    if sha((args.baseline / 'run_manifest.json').read_bytes()) != plan['baseline_manifest_sha256']:
        raise ValueError('Original baseline manifest changed.')
    truth = [r for r in read_jsonl(args.data / 'labels.jsonl') if r['clip_id'] in plan['clip_ids']]
    if {r['clip_id'] for r in truth} != set(plan['clip_ids']):
        raise ValueError('Truth clips do not match the plan.')
    previous_rows = [r for r in read_jsonl(args.baseline / 'predictions.jsonl') if r['clip_id'] in plan['clip_ids']]
    previous_scores = evaluate(truth, previous_rows)
    condition_rows, summary, all_raw, scored = {}, {}, {}, {}
    reference_scores = None
    for condition in CONDITIONS:
        predictions, raw = combine_condition(plan, args.run, condition)
        result, scores = summarize(truth, predictions, raw, reference_scores)
        if condition == 'full_forward':
            reference_scores = scores
            result, scores = summarize(truth, predictions, raw, reference_scores)
        condition_rows[condition], all_raw[condition], summary[condition], scored[condition] = predictions, raw, result, scores
        folder = args.run / 'conditions' / condition
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in predictions), encoding='utf-8')
        write_json(folder / 'scores.json', scores)
    by_clip = {json.loads(p.read_text(encoding='utf-8'))['clip_id']: p for p in args.baseline.glob('case_*/raw_output.json')}
    repeat_checks = []
    for raw in all_raw['full_forward']:
        path = by_clip[raw['job']['clip_id']]
        old = json.loads(path.read_text(encoding='utf-8'))
        wav = path.parent / 'model_input.wav'
        repeat_checks.append({'clip_id': raw['job']['clip_id'],
                              'same_input_wav': sha(wav.read_bytes()) == raw['audio']['input_wav_sha256'] if wav.exists() else None,
                              'same_generated_text': old['text'] == raw['text'],
                              'same_segments': old['segments'] == raw['segments']})
    # Check intended controls across all conditions, independently of scores.
    for cid in plan['clip_ids']:
        reference_sets = {tuple(r['audio']['reference_sha256_multiset']) for rows in all_raw.values() for r in rows if r['job']['clip_id'] == cid}
        if len(reference_sets) != 1:
            raise ValueError('Registration audio contents/counts differ across conditions.')
        full = [r for c in ('full_forward', 'full_reverse') for r in all_raw[c] if r['job']['clip_id'] == cid]
        if len({r['audio']['evaluation_pcm_sha256'] for r in full}) != 1:
            raise ValueError('Full conversation changed between reference orders.')
        first = {r['job']['target_segment_ids'][0]: r for r in all_raw['isolated_forward'] if r['job']['clip_id'] == cid}
        later = {r['job']['target_segment_ids'][0]: r for r in all_raw['isolated_reverse'] if r['job']['clip_id'] == cid}
        if set(first) != set(later) or any(first[s]['audio']['evaluation_pcm_sha256'] != later[s]['audio']['evaluation_pcm_sha256'] for s in first):
            raise ValueError('Isolated target audio differs between registration orders.')
    result = {'status': 'posthoc_exploratory_diagnostic_not_final_leaderboard',
              'clip_ids': plan['clip_ids'], 'labels_sha256': sha((args.data / 'labels.jsonl').read_bytes()),
              'baseline_repeat_checks': repeat_checks, 'summary': summary,
              'previous_pilot_correct': sum(d['correct'] for d in previous_scores['details']),
              'previous_pilot_scored': previous_scores['scored_segments'],
              'controls_verified': ['Same reference content/counts', 'Full dialogue PCM/order preserved',
                                    'Same isolated target PCM across registration orders', 'No target selection from labels']}
    write_json(args.run / 'diagnostic_summary.json', result)
    lines = ['# MOSS 입력 구조 2×2 진단 결과', '',
             '**기존 오류를 본 뒤 선정한 2개 대화의 탐색 실험이다. 새 평가 성능이나 행동 전환의 인과 증명으로 보고하지 않는다.**', '',
             '등록 정순은 A→B→C→D, 역순은 D→C→B→A다. 각 사람의 등록 파일 순서·내용·횟수는 동일하다. 전체 대화 순서는 유지한다.', '',
             '| 조건 | 발화 정답 | 동일 인물 쌍 두 발화 정답 | 전후 정답→오답 쌍 | 기존 오류 수정 / 새 오류 | 등록 미완료 호출 |',
             '|---|---:|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        s = summary[condition]; changes = s['changes_vs_full_forward']
        lines.append(f'| {condition} | {s["correct"]}/{s["scored"]} | {s["pairs_both_correct"]}/{s["pairs"]} | '
                     f'{s["same_person_states"].get("correct_to_wrong",0)} | {changes["fixed_count"]} / {changes["new_error_count"]} | '
                     f'{s["calls_with_incomplete_registration"]}/{s["model_calls"]} |')
    lines += ['', '수정·새 오류는 이번 full_forward 재실행을 기준으로 계산한다. 등록 미완료는 모델 호출 단위이며 전체/발화별 호출 수가 다르다.', '',
              '## 원래 전체 입력의 재실행 확인', '']
    for check in repeat_checks:
        lines.append(f'- {check["clip_id"]}: 입력 WAV 동일={check["same_input_wav"]}, 출력 전사 동일={check["same_generated_text"]}, 구간 목록 동일={check["same_segments"]}.')
    lines += ['', '## 대화별 점수', '', '| 대화 | full_forward | full_reverse | isolated_forward | isolated_reverse |', '|---|---:|---:|---:|---:|']
    for cid in plan['clip_ids']:
        values = [f'{summary[c]["per_clip"][cid]["correct"]}/{summary[c]["per_clip"][cid]["scored"]}' for c in CONDITIONS]
        lines.append('| ' + cid + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '## 발화별 비교', '', '| 대화 / 발화 | 정답 | 원래 pilot | full_forward | full_reverse | isolated_forward | isolated_reverse |', '|---|---|---|---|---|---|---|']
    old_lookup = {(r['clip_id'],p['segment_id']):p.get('speaker') for r in previous_rows for p in r['predictions']}
    lookups = {c: {(r['clip_id'],p['segment_id']):p.get('speaker') for r in condition_rows[c] for p in r['predictions']} for c in CONDITIONS}
    for row in truth:
        for u in row['utterances']:
            if not u['score_eligible']:
                continue
            key = (row['clip_id'],u['segment_id'])
            values = [str(old_lookup.get(key))] + [str(lookups[c].get(key)) for c in CONDITIONS]
            lines.append('| ' + row['meeting'] + ' / ' + u['segment_id'] + ' | ' + u['speaker'] + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '## 해석 기준', '',
              '- 전체 입력에서만 틀리고 발화별 입력에서 두 등록 순서 모두 맞히면, 전체 문맥·입력 길이·분할 방식에 따른 민감성 후보가 된다.',
              '- 등록 순서만 바꿔 정답이나 등록 병합이 달라지면 등록 순서에 대한 민감성이 관측된 것이다.',
              '- 한 조합에서만 좋아지면 문맥과 등록 순서의 상호작용을 먼저 본다. 두 요인을 하나의 원인으로 합치지 않는다.',
              '- 전후 모두 같은 오답이면 일관성이 높더라도 성공으로 세지 않는다. 수정한 오류와 함께 새로 망가뜨린 정답을 확인한다.',
              '- 발화별 입력은 음향 문맥과 언어 문맥을 함께 줄이고 생성 길이·경계도 바꾼다. 점수 차이만으로 의미 편향이나 질문→답변 전환의 인과 효과가 증명되지는 않는다.',
              '- 원래 전체 입력 재실행이 다르면 반복성·환경 차이부터 확인한다. 이 결과로 설정을 선택했다면 새 대화에서 별도로 검증해야 한다.', '',
              '상세 등록 겹침·원본 출력·처리 시간·VRAM은 jobs/**/raw_output.json, 조건별 채점은 conditions/*/scores.json에 있다.', '']
    (args.run / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    bundle = args.run / 'results_only.zip'
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(args.run.rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.md') and '.scratch' not in path.parts:
                archive.write(path, path.relative_to(args.run).as_posix())
    with zipfile.ZipFile(bundle) as archive:
        if archive.testzip():
            raise ValueError('Result archive CRC failure.')
    print('Report:', args.run / 'REPORT.md')
    print('Download:', bundle, f'({bundle.stat().st_size/2**20:.2f} MiB, excludes WAV/checkpoints)')


if __name__ == '__main__':
    main()
