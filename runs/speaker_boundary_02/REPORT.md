# 30초 경계 정렬 진단과 보수적 합의 보정

학습 모델의 30초 인코더 길이는 유지한다. 등록·대화 전체 앞 무음만 0/2/30/32초로 바꿨다. 0과30, 2와32는 각각 30초 경계 정렬이 같고 절대 시각·길이는 다르다. 기존 오류를 본 뒤 설계한 탐색 실험이다.

| 방법 | 발화 정답 | 동일 인물 쌍 | 수정 / 새 오류 | 누락 |
|---|---:|---:|---:|---:|
| pad_00 | 22/24 | 2/3 | 0 / 0 | 0 |
| pad_02 | 22/24 | 2/3 | 0 / 0 | 0 |
| pad_30 | 23/24 | 2/3 | 2 / 1 | 1 |
| pad_32 | 15/24 | 1/3 | 0 / 7 | 2 |
| majority_vote | 22/24 | 2/3 | 0 / 0 | 0 |
| boundary_guard | 22/24 | 2/3 | 0 / 0 | 0 |

## 0초 조건 재현 검사

- IS1009d_question_answer_reversal_01: 입력 WAV 동일=True, 전사 동일=True, 구간 동일=True.
- IS1000d_question_answer_reversal_01: 입력 WAV 동일=True, 전사 동일=True, 구간 동일=True.

majority_vote는 유효한 화자 예측 중 75% 이상·최소2개의 지지를 받을 때 수정한다. boundary_guard는 경계를 가로지르지 않고 경계에서0.25초 이상 떨어진 입력만 사용한다. 대상 구간50% 이상 출력, 예측 화자의 등록 근거1초 이상, 30초 단위 이동 그룹0/1 모두에서 총2개 이상 만장일치일 때만 수정한다. 조건이 부족하면 원래 pad_00 예측을 유지한다. 이 규칙은 실행 전에 plan.json으로 고정한다.

## 등록 연결과 인코더 청크 수

| 대화 | 앞 무음 초 | 등록 연결 | 청크 수 | 토큰 상한 도달 |
|---|---:|---:|---:|---|
| IS1009d_question_answer_reversal_01 | 0 | 3/4 | 3 | False |
| IS1000d_question_answer_reversal_01 | 0 | 4/4 | 3 | False |
| IS1009d_question_answer_reversal_01 | 2 | 4/4 | 3 | False |
| IS1000d_question_answer_reversal_01 | 2 | 4/4 | 3 | False |
| IS1009d_question_answer_reversal_01 | 30 | 3/4 | 4 | False |
| IS1000d_question_answer_reversal_01 | 30 | 4/4 | 4 | False |
| IS1009d_question_answer_reversal_01 | 32 | 4/4 | 4 | False |
| IS1000d_question_answer_reversal_01 | 32 | 3/4 | 4 | False |

## 발화별 화자

| 대화 / 발화 | 정답 | 0초 | 2초 | 30초 | 32초 | 단순 합의 | 경계 보정 |
|---|---|---|---|---|---|---|---|
| IS1000d / segment_000 | B | B | B | B | D | B | B |
| IS1000d / segment_002 | B | B | B | B | D | B | B |
| IS1000d / segment_003 | C | C | C | C | C | C | C |
| IS1000d / segment_004 | D | D | D | D | D | D | D |
| IS1000d / segment_005 | D | D | D | D | None | D | D |
| IS1000d / segment_011 | A | A | A | A | A | A | A |
| IS1000d / segment_021 | B | B | B | B | D | B | B |
| IS1000d / segment_026 | B | B | B | B | D | B | B |
| IS1000d / segment_032 | C | C | C | C | C | C | C |
| IS1000d / segment_033 | D | D | D | D | None | D | D |
| IS1000d / segment_034 | B | B | B | B | D | B | B |
| IS1009d / segment_001 | A | B | B | A | B | B | B |
| IS1009d / segment_003 | D | D | D | D | D | D | D |
| IS1009d / segment_005 | D | D | D | D | D | D | D |
| IS1009d / segment_006 | A | B | B | A | B | B | B |
| IS1009d / segment_007 | D | D | D | D | D | D | D |
| IS1009d / segment_008 | D | D | D | D | D | D | D |
| IS1009d / segment_009 | D | D | D | D | D | D | D |
| IS1009d / segment_010 | D | D | D | D | D | D | D |
| IS1009d / segment_011 | D | D | D | D | D | D | D |
| IS1009d / segment_012 | D | D | D | D | D | D | D |
| IS1009d / segment_017 | D | D | D | D | D | D | D |
| IS1009d / segment_018 | D | D | D | D | D | D | D |
| IS1009d / segment_019 | A | A | A | None | A | A | A |

## 해석

- 0/30에서 반복되는 오류가 2/32에서 함께 사라지면 경계 정렬 민감성의 근거가 강해진다. 절대 시각·무음·청크 수 영향까지 완전히 제거한 증명은 아니다.
- 같은 정렬인 0과30 또는2와32가 다르면 절대 위치·입력 길이·추가 빈 청크·등록 출력 변화도 확인한다.
- 원래 전체 대화·등록 내용이 동일하므로 발화별 분할보다 자연 대화 문맥을 더 잘 통제한다. 그러나 의미 편향만을 증명하지 않는다.
- 보정은 화자 귀속 결과만 생성한다. 원래 MOSS 전사·시각은 보존하며 ASR 누락이 고쳐졌다고 주장하지 않는다.
- 최고 결과를 고른 뒤 새로운 대화에서 재검증해야 한다. 현재 2개 대화의 점수를 최종 독립 평가로 보고하지 않는다.
