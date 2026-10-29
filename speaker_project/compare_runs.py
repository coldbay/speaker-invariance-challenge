"""Report actual acoustic/joint model runs, separating mapping and timing issues."""
import argparse
import json
from pathlib import Path
from score_predictions import evaluate, read_jsonl


def percent(n, d):
    return f'{n}/{d} ({n/d:.1%})' if d else '—'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/ami/prepared'))
    parser.add_argument('--acoustic', type=Path, default=Path('runs/speaker_wespeaker_01'))
    parser.add_argument('--joint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    labels = read_jsonl(args.data / 'labels.jsonl')
    inputs = {r['clip_id']: r for r in read_jsonl(args.data / 'inputs.jsonl')}
    outputs = []
    for name, path in (('WeSpeaker ResNet34-LM', args.acoustic), ('MOSS 0.9B', args.joint)):
        rows = read_jsonl(path / 'predictions.jsonl')
        if {r['clip_id'] for r in rows} != set(inputs):
            raise ValueError(f'Incomplete or extra run: {path}; do not compare smoke output as full benchmark.')
        scores = evaluate(labels, rows)
        (path / 'scores.json').write_text(json.dumps(scores, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        outputs.append((name, path, scores))
    lines = ['# 음성 전용 / 통합 모델 실제 비교', '',
             '모든 후보는 아직 청취 검토 대기다. 작은 시작용 표본의 관측값이며 역할 전환의 인과 효과나 최종 리더보드로 해석하지 않는다.', '',
             '| 방법 | 평가 전체 발화 ↑ | 평가 전환 쌍 두 발화 정답 ↑ | 평가 대조 쌍 두 발화 정답 ↑ |',
             '|---|---:|---:|---:|']
    for name, path, scores in outputs:
        details = [d for d in scores['details'] if d['split'] == 'evaluation']
        groups = {g['event']: g for g in scores['same_person_summary'] if g['split'] == 'evaluation'}
        pair_values = []
        for event in ('question_answer_reversal', 'same_direction_control'):
            group = groups.get(event, {})
            pair_values.append(percent(group.get('both_correct_count', 0), group.get('pairs', 0)))
        lines.append(f'| {name} | {percent(sum(d["correct"] for d in details), len(details))} | {pair_values[0]} | {pair_values[1]} |')
    lines.extend(['', '## 통합 모델 실행·등록 매핑 진단', '',
                  '| 클립 | 등록 ID 연결 | 토큰 상한 도달 | 추론 초 | GPU 최대 할당 GiB |',
                  '|---|---:|---|---:|---:|'])
    raw_rows = []
    for file in sorted(args.joint.glob('case_*/raw_output.json')):
        raw = json.loads(file.read_text(encoding='utf-8'))
        count = len(inputs[raw['clip_id']]['registered_speakers'])
        mapped = len(set(raw['speaker_mapping'].values()))
        lines.append(f'| {raw["clip_id"]} | {mapped}/{count} | {raw["reached_token_limit"]} | {raw.get("inference_seconds", 0):.2f} | {raw.get("peak_cuda_allocated_bytes", 0)/2**30:.2f} |')
        raw_rows.append(raw)
    lines.extend(['', '## 통합 모델의 오답·누락', ''])
    joint = outputs[1][2]
    truth = {(r['clip_id'], u['segment_id']): u for r in labels for u in r['utterances']}
    mistakes = [d for d in joint['details'] if not d['correct']]
    for d in mistakes:
        u = truth[(d['clip_id'], d['segment_id'])]
        lines.append(f'- {d["clip_id"]}/{d["segment_id"]}: 정답 {u["speaker"]}, 예측 {d["predicted_speaker"]}, 길이 {u.get("duration", 0):.2f}s, 겹침 {u["other_speaker_word_overlap_fraction"]:.1%}. 전사: {u["text"]}')
    if not mistakes:
        lines.append('점수 대상 오답·누락 0개. 현재 표본에서 통합 모델 오류는 재현되지 않았다.')
    lines.extend(['', '통합 모델은 등록 음성을 앞에 연결한 전체 대화를 전사·분할하고, 음성 전용 방법은 정답 경계의 각 발화를 따로 분류한다. '
                  '매핑·시각·발화 단위·모델 구조가 달라, 점수 차이만으로 내용 편향을 입증할 수 없다. 원본 출력과 청취를 함께 확인해야 한다.', ''])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines), encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
