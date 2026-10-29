# 경계 영향 진단 후 보수적 화자 보정

## 30초를 먼저 늘리지 않는 이유

현재 MOSS 전처리는 WhisperFeatureExtractor의 n_samples=480000, sampling_rate=16000, chunk_length=30을 사용한다. Whisper 인코더의 max_source_positions=1500과 convolution stride가 요구하는 멜 입력 길이는3000프레임이다. 전처리 숫자만60초로 바꾸면6000프레임 입력이 되어 이 인코더와 맞지 않는다. [고정 Transformers5.6 인코더 구현](https://github.com/huggingface/transformers/blob/v5.6.0/src/transformers/models/whisper/modeling_whisper.py)에서 expected_seq_length 검사를 확인했다.

따라서 먼저 학습 모델과 30초 처리 구조를 유지하고, 오류가 경계 정렬에 민감한지 확인한다. 영향이 확인되면 같은 모델의 경계 회피 입력·보수적 합의로 보정할 수 있다. 의미 편향과 경계 효과를 섞어 해석하지 않는다.

## 1단계: 내용은 그대로, 전체 입력 앞 무음만 변경

등록 A/B/C/D 음성·내용·횟수·순서, 등록 사이 간격, 원래 연속 대화의 PCM·순서·음량·상대 시간은 유지한다. 등록과 대화를 합친 전체 입력의 맨 앞에만 무음을 추가한다. 등록 시각과 평가 시작 시각을 모두 같은 값만큼 이동해 채점한다.

| 앞 무음 | 첫 오류 발화의 입력 시각 | 30초 경계 통과 | 역할 |
|---:|---|---|---|
| 0초 | 29.20–30.93초 | 통과 | 원래 조건 재현 |
| 2초 | 31.20–32.93초 | 통과하지 않음 | 경계 정렬 변경 |
| 30초 | 59.20–60.93초 | 통과 | 같은 정렬·다른 절대 위치/길이 |
| 32초 | 61.20–62.93초 | 통과하지 않음 | 이동한 정렬·다른 절대 위치/길이 |

0/30초와2/32초는 각각 경계 정렬이 같다. 원래 오류 사건IS1009d와 원래 성공 사건IS1000d 두 개를 사용해 **4조건×2대화=8회 추론**한다. 30/32초 조건은 빈 인코더 청크 하나와 입력 길이도 추가하므로, 완전한 단일 원인 증명은 아니다. 이것도 출력에 영향을 주는지 확인하는 대조다.

0/30에서 같은 오류가 반복되고2/32에서 모두 고쳐지면 정렬 민감성의 근거가 강해진다. 같은 정렬에서도 결과가 다르면 절대 위치·길이·빈 청크·등록 누락을 함께 확인한다. 첫 오류가 경계에 걸친다는 사실만으로 두 번째 오류까지 경계 때문이라고 주장하지 않는다.

## 2단계: 같은 출력으로 개선 후보 두 개를 비교

모든 규칙은 실행 전 plan.json에 고정한다. 정답·전사는 추론과 보정에 제공하지 않는다. 보고서 생성 단계에서만 정답을 읽는다.

| 방법 | 수정 조건 |
|---|---|
| 원래 MOSS | 0초 조건의 예측 유지 |
| 단순 합의 | 화자 예측이 있는 입력 중 최소2개·75% 이상이 같은 화자를 지지 |
| 경계를 피한 합의 보정 | 아래 조건을 전부 만족할 때만 수정; 부족하면 원래0초 예측 유지 |

경계를 피한 합의 보정의 조건:

- 해당 발화가30초 경계를 가로지르지 않고, 양 끝이 경계에서0.25초 이상 떨어져 있다.
- 모델이 대상 발화 길이의50% 이상을 해당 등록 화자로 출력했다.
- 예측 화자가 등록 음성 구간에서1초 이상의 연결 근거를 갖는다.
- 유효한 예측이 최소2개이고, 0/2초 그룹과30/32초 그룹에 모두 존재한다.
- 유효한 예측들이 실제 등록 화자A/B/C/D 기준으로 모두 일치한다. 익명S01/S02 번호는 매번 다시 연결한다.

이것은 확률적 신뢰도 추정이 아니라 보수적인 휴리스틱이다. 위 규칙이 실제로 도움이 되는지 새 오류까지 함께 측정한다. 네 조건 모두 같은 화자로 틀리면 보정하지 못한다. 원래 전사·시각은 보존하고 별도의 화자 귀속 결과만 생성하므로 ASR 누락을 해결한 결과로 해석하지 않는다.

## 3단계: 등록·음성 근거 보완은 별도 평가

앞선 실험의 C등록미연결과 A→B실제오배정은 다른 문제다. 경계 대조 후에도 남는다면 등록 시각의 전사 출력만으로ID를 연결하는 의존성을 음성 임베딩으로 보완하는 것이 다음 후보가 된다. 등록 프로토타입은 고정하고 평가 예측으로 자동 갱신하지 않는다. 보완 방식은 별도 방법으로 비교하며, 현재 WeSpeaker가 모두 맞힌 표본에서100%가 나왔다고 새로운 우수한 화자 모델로 주장하지 않는다.

지금 구현한 후보는 경계 정렬의 합의 보정까지다. 별도 임베딩 기반 보정은 이번 코드에 구현하지 않았으며, 경계 대조 결과와 새로운 대화에서의 오류를 보고 필요성을 판단한다.

## 서버 실행

speaker_project/를 동기화한다. 새 파일은 boundary_control.py, run_boundary_control.py, test_boundary_control.py다. 기존 moss_diagnostic_protocol.py, diarization_projection.py, run_moss_diagnostics.py, score_predictions.py도 사용하므로 폴더 단위로 동기화하는 편이 간단하다. 원본8개실험과78개진단 파일은 덮어쓰지 않는다.

```bash
cd /tmp/pycharm_project_e52e89ff
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4

.venv-moss-reuse/bin/python speaker_project/run_boundary_control.py \
  --prepare-only \
  --output runs/speaker_boundary_01
```

현재 배정 GPU에 맞춰4를 바꾼다. model_calls=8, pads_seconds=[0,2,30,32]를 확인한다. 이 단계는 모델을 로딩하거나 추론하지 않는다.

실제 실행:

```bash
.venv-moss-reuse/bin/python speaker_project/run_boundary_control.py \
  --output runs/speaker_boundary_01
```

모델은 한 번만 로딩하고8개를 순차 실행한다. 완료 후 **보고서·보정 후보 채점·결과 ZIP까지 자동 생성**한다. 중단되면 같은 명령으로 재개한다. 설정이나 코드를 바꿨다면 새 폴더를 사용한다. 실제8회추론시간은 아직 측정하지 않았다.

보고서만 재생성하려면:

```bash
.venv-moss-reuse/bin/python speaker_project/run_boundary_control.py \
  --report-only \
  --output runs/speaker_boundary_01
```

결과:

```text
runs/speaker_boundary_01/
  plan.json
  runtime.json
  REPORT.md
  summary.json
  methods/pad_00|pad_02|pad_30|pad_32/
  methods/majority_vote/
  methods/boundary_guard/
  jobs/<무음 조건>/<사건>/full/raw_output.json
  results_only.zip
```

로컬에는 **runs/speaker_boundary_01/results_only.zip**을 가져오면 된다. 기존 78회 진단의 ZIP과 파일명이 같으므로 폴더를 구분한다. 음성·가중치는 포함하지 않는다.

## 판단과 검증 상태

보고서에서 먼저 0초 입력이 원래 결과를 재현했는지 확인하고, 네 조건의 등록 성공·토큰 상한·경계 위치와 정답 변화를 본다. 보정 후에는 수정한 오류와 새로 망가뜨린 정답을 함께 비교한다. 경계 대조 결과가 다르다고 곧바로 의미 편향의 증명이나 최종 개선 성능으로 보고하지 않는다. 현재 두 대화는 이미 분석한 표본이므로 설정을 선택했다면 미관측 대화에서 다시 검증한다.

무음 추가 외 PCM 보존·시각 이동·경계 판정·보수적 합의·보고서 생성을 포함해 로컬 19개 검사가 통과했다. 합성 테스트 예측은 임시 폴더에서만 사용했으며 실제 모델 결과가 아니다. 실제 경계 대조 GPU 실험은 서버 실행 대기다.
