# 다음 실험: 같은 MOSS의 입력 문맥 × 등록 순서

실행 업데이트: 실제 78회 추론과 보고서 생성을 완료했다. 결과 ZIP을 로컬에서 재검증했고, 상세 해석은 ../runs/speaker_moss_diagnostics_01/analysis/REPORT.md에 있다. 아래 명령은 재현 절차다.

목적은 MOSS의 두 A→B 오답이 전체 대화 입력이나 등록 초기화에 민감한지 분리하는 것이다. 모델·가중치·정밀도·프롬프트·등록 음성 내용과 횟수는 고정한다. 원래 평가 대화의 발화 순서를 바꾸지 않는다. 기존 오류를 본 뒤 설계한 탐색 진단이며, 새 독립 평가 성능이나 질문·답변 전환의 인과 증명을 위한 실험은 아니다.

## 대상과 네 조건

- 오류 사건: IS1009d_question_answer_reversal_01. 기존 MOSS 11/13 정답, A의 앞선 두 발화가 B로 배정됐다.
- 기존 정답 사건: IS1000d_question_answer_reversal_01. 기존 MOSS 11/11 정답, 등록 4명 모두 연결됐다. 이것도 질문·답변 전환 대화이며 역할 유지 대조가 아니다.

| 조건 | 평가 음성 | 등록 순서 | 확인하는 점 |
|---|---|---|---|
| full_forward | 원래 연속 대화 | A→B→C→D | 기존 실험 재현 여부 |
| full_reverse | 같은 연속 대화 | D→C→B→A | 등록 순서만 바꾼 영향 |
| isolated_forward | 발화 한 개씩 독립 입력 | A→B→C→D | 전체 대화 문맥·길이·분할 방식의 영향 |
| isolated_reverse | 같은 발화 한 개씩 | D→C→B→A | 등록 순서와 입력 단위의 상호작용 |

각 사람의 등록 파일 2개의 내부 순서는 고정한다. 익명 S01/S02 번호가 바뀌는 것은 오류가 아니며, 매번 등록 시각을 이용해 실제 A/B/C/D로 다시 연결한다. 평가 정답으로 순열을 맞추지 않는다.

발화 선택은 inputs.jsonl의 공개 경계에서 길이 0.8–15초인 모든 구간으로 고정한다. 대상은 37개이며 추론기는 정답·전사·겹침 라벨을 읽지 않는다. 최종 점수 대상은 원래 기준을 만족하는 24개 발화·3개 동일 인물 쌍이다. 겹침 때문에 점수에서 빠지는 구간도 일부 추론되며, 점수 분모에는 넣지 않는다.

전체 조건은 각 2회, 발화별 조건은 각 37회, 합계 **78회 추론**이다. 현재 모델을 한 번만 로딩하고 순차 처리한다. 신규 모델·패키지 설치는 필요 없다. 기존 8개 실행보다 호출 수가 많으므로 더 오래 걸리며, 전체 소요 시간은 실제 실행 후 기록한다.

## 동기화할 코드

speaker_project/의 아래 파일을 함께 동기화한다. data/와 기존 runs/speaker_moss_pilot_01/는 서버에 있는 것을 사용한다.

- run_moss_diagnostics.py
- moss_diagnostic_protocol.py
- report_moss_diagnostics.py
- diarization_projection.py, score_predictions.py
- moss_sources.json
- test_moss_diagnostics.py, test_protocol.py (검사용)

로컬의 .venv나 models를 서버에 전송하지 않는다. 기존 서버 환경과 모델 캐시를 사용한다. 결과 ZIP의 본문에는 평가 정답·전사가 포함되므로, 모델 입력으로 쓰지 않는다.

## 서버 명령

현재 사용한 GPU는 4였다. 배정이 바뀌었다면 CUDA_VISIBLE_DEVICES를 바꾼다. 기존 성공한 .venv-moss-reuse/bin/python을 사용한다.

```bash
cd /tmp/pycharm_project_e52e89ff
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4
```

먼저 모델을 실행하지 않고 계획·입력 해시·기존 환경 설정을 확인한다.

```bash
.venv-moss-reuse/bin/python speaker_project/run_moss_diagnostics.py \
  --prepare-only \
  --output runs/speaker_moss_diagnostics_01
```

model_calls=78, full_forward=2, full_reverse=2, isolated_forward=37, isolated_reverse=37을 확인한다. 기존 plan과 다르면 덮어쓰지 않고 새 폴더를 요구한다. 설정을 수정했다면 `_02`처럼 이름을 바꾼다.

검사(선택):

```bash
.venv-moss-reuse/bin/python -m unittest discover -s speaker_project -p 'test*.py' -v
```

실제 실행:

```bash
.venv-moss-reuse/bin/python speaker_project/run_moss_diagnostics.py \
  --output runs/speaker_moss_diagnostics_01
```

`1/78`부터 진행한다. 중단되어도 **같은 명령을 재실행하면 완료된 작업을 건너뛴다.** 재개 시 코드·계획·패키지·GPU 설정이 다르면 혼합 결과를 만들지 않고 중단한다. 실험 설정을 바꾸려면 새 출력 폴더를 사용한다. 실험이 완전히 끝난 폴더를 다시 실행해도 모델을 다시 추론하지 않는다.

실제 음성 입력은 .scratch/model_input.wav 하나를 재사용해 대용량 파일이 쌓이지 않게 했다. 각 작업의 PCM·WAV 해시, 프레임 범위, 등록 구간, 생성 텍스트·시각·화자, 등록 매핑, 처리 시간·VRAM을 jobs/에 기록한다. 원본 data/는 수정하지 않는다.

모든 작업 완료 후 보고서 생성:

```bash
.venv-moss-reuse/bin/python speaker_project/report_moss_diagnostics.py \
  --run runs/speaker_moss_diagnostics_01
```

주요 결과:

```text
runs/speaker_moss_diagnostics_01/
  plan.json
  runtime.json
  REPORT.md
  diagnostic_summary.json
  conditions/<조건>/predictions.jsonl
  conditions/<조건>/scores.json
  jobs/<조건>/<사건>/<발화>/raw_output.json
  results_only.zip
```

**로컬 분석에는 results_only.zip만 가져오면 된다.** ZIP에는 계획·메타데이터·실제 원본 출력·보고서·예측·점수가 포함되며, WAV와 모델 체크포인트는 제외한다. 서버에서 모든 작업이 끝나기 전에는 보고서가 완성된 것처럼 처리하지 않는다.

## 결과별 판단

- 전체 입력에서만 오류가 나고 발화별 입력에서 두 등록 순서 모두 맞히면 전체 문맥·입력 길이·생성 분할에 대한 민감성 후보가 된다. 음향 문맥과 언어 문맥을 함께 줄였으므로 내용 편향만의 증명은 아니다.
- 등록 순서만 바꿔 오류나 B/C 병합이 달라지면 등록 순서에 따른 민감성을 관측한 것이다.
- 한 조합에서만 좋아지면 두 요인의 상호작용을 먼저 본다.
- 기존 성공 사건에서 새 오류가 늘면 수정 개수만으로 좋은 설정을 고르지 않는다.
- full_forward 재실행이 기존 결과와 다르면 입력·출력 반복성부터 확인한다.
- 여기서 설정을 선택했다면 최종 리더보드는 별도 고정한 새로운 대화에서 검증한다.

## 현재 검증 상태

로컬에서 13개 검사가 통과했다. 전체 입력이 기존 서버 WAV와 일치하는지, 순서 변경 후 평가 PCM·등록 내용·횟수 유지, 발화 절취 범위, 익명 번호 재배정, 누락·중복 결과 거부, 수정·새 오류 집계와 보고서 ZIP 생성을 검사했다. 보고서 단위 검사는 임시 폴더의 명시적인 합성 fixture로 수행했다. 그 후 서버에서 실제 새 MOSS 진단 78회를 완료했고, 동기화된 실제 입력 메타데이터·투영·점수도 오프라인 재검증했다.
