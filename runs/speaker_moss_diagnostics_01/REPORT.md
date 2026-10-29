# MOSS 입력 구조 2×2 진단 결과

**기존 오류를 본 뒤 선정한 2개 대화의 탐색 실험이다. 새 평가 성능이나 행동 전환의 인과 증명으로 보고하지 않는다.**

등록 정순은 A→B→C→D, 역순은 D→C→B→A다. 각 사람의 등록 파일 순서·내용·횟수는 동일하다. 전체 대화 순서는 유지한다.

| 조건 | 발화 정답 | 동일 인물 쌍 두 발화 정답 | 전후 정답→오답 쌍 | 기존 오류 수정 / 새 오류 | 등록 미완료 호출 |
|---|---:|---:|---:|---:|---:|
| full_forward | 22/24 | 2/3 | 0 | 0 / 0 | 1/2 |
| full_reverse | 20/24 | 2/3 | 0 | 0 / 2 | 1/2 |
| isolated_forward | 21/24 | 3/3 | 0 | 2 / 3 | 0/37 |
| isolated_reverse | 23/24 | 3/3 | 0 | 2 / 1 | 13/37 |

수정·새 오류는 이번 full_forward 재실행을 기준으로 계산한다. 등록 미완료는 모델 호출 단위이며 전체/발화별 호출 수가 다르다.

## 원래 전체 입력의 재실행 확인

- IS1009d_question_answer_reversal_01: 입력 WAV 동일=True, 출력 전사 동일=True, 구간 목록 동일=True.
- IS1000d_question_answer_reversal_01: 입력 WAV 동일=True, 출력 전사 동일=True, 구간 목록 동일=True.

## 대화별 점수

| 대화 | full_forward | full_reverse | isolated_forward | isolated_reverse |
|---|---:|---:|---:|---:|
| IS1009d_question_answer_reversal_01 | 11/13 | 11/13 | 12/13 | 13/13 |
| IS1000d_question_answer_reversal_01 | 11/11 | 9/11 | 9/11 | 10/11 |

## 발화별 비교

| 대화 / 발화 | 정답 | 원래 pilot | full_forward | full_reverse | isolated_forward | isolated_reverse |
|---|---|---|---|---|---|---|
| IS1000d / segment_000 | B | B | B | B | B | B |
| IS1000d / segment_002 | B | B | B | B | B | B |
| IS1000d / segment_003 | C | C | C | None | C | C |
| IS1000d / segment_004 | D | D | D | D | D | D |
| IS1000d / segment_005 | D | D | D | D | None | D |
| IS1000d / segment_011 | A | A | A | A | A | A |
| IS1000d / segment_021 | B | B | B | B | B | B |
| IS1000d / segment_026 | B | B | B | B | B | B |
| IS1000d / segment_032 | C | C | C | None | C | None |
| IS1000d / segment_033 | D | D | D | D | D | D |
| IS1000d / segment_034 | B | B | B | B | D | B |
| IS1009d / segment_001 | A | B | B | B | A | A |
| IS1009d / segment_003 | D | D | D | D | D | D |
| IS1009d / segment_005 | D | D | D | D | D | D |
| IS1009d / segment_006 | A | B | B | B | A | A |
| IS1009d / segment_007 | D | D | D | D | A | D |
| IS1009d / segment_008 | D | D | D | D | D | D |
| IS1009d / segment_009 | D | D | D | D | D | D |
| IS1009d / segment_010 | D | D | D | D | D | D |
| IS1009d / segment_011 | D | D | D | D | D | D |
| IS1009d / segment_012 | D | D | D | D | D | D |
| IS1009d / segment_017 | D | D | D | D | D | D |
| IS1009d / segment_018 | D | D | D | D | D | D |
| IS1009d / segment_019 | A | A | A | A | A | A |

## 해석 기준

- 전체 입력에서만 틀리고 발화별 입력에서 두 등록 순서 모두 맞히면, 전체 문맥·입력 길이·분할 방식에 따른 민감성 후보가 된다.
- 등록 순서만 바꿔 정답이나 등록 병합이 달라지면 등록 순서에 대한 민감성이 관측된 것이다.
- 한 조합에서만 좋아지면 문맥과 등록 순서의 상호작용을 먼저 본다. 두 요인을 하나의 원인으로 합치지 않는다.
- 전후 모두 같은 오답이면 일관성이 높더라도 성공으로 세지 않는다. 수정한 오류와 함께 새로 망가뜨린 정답을 확인한다.
- 발화별 입력은 음향 문맥과 언어 문맥을 함께 줄이고 생성 길이·경계도 바꾼다. 점수 차이만으로 의미 편향이나 질문→답변 전환의 인과 효과가 증명되지는 않는다.
- 원래 전체 입력 재실행이 다르면 반복성·환경 차이부터 확인한다. 이 결과로 설정을 선택했다면 새 대화에서 별도로 검증해야 한다.

상세 등록 겹침·원본 출력·처리 시간·VRAM은 jobs/**/raw_output.json, 조건별 채점은 conditions/*/scores.json에 있다.
