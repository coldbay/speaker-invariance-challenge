# 자연 대화의 질문·답변 전환에서 화자 귀속 평가

상태: 2026-10-29 **WeSpeaker 70/70·동일 인물 12/12쌍, MOSS 68/70·11/12쌍 실제 실행 완료.** MOSS 오류 2개는 지정 전환 전 A 발화이며 후반 A 답변은 정답이다. 전환 후 악화는 입증하지 않았다. 후보 청취 검토는 대기 중이다. 전체 인수인계는 ../WORK_STATUS.md를 먼저 읽는다.

바로 볼 파일: [실제 결과 보고서](../runs/speaker_wespeaker_01/REPORT.md), [청취 검토 페이지](../data/ami/prepared/review.html), [A5000 실행 안내](SERVER_RUN.md).

최신 분석: [MOSS 비교 심층 보고서](../runs/speaker_comparison/analysis/REPORT.md), [MOSS 실제 예측 청취 검토](../runs/speaker_comparison/analysis/listening_review.html). analyze_pilot.py로 입력 해시·재채점·원본 PCM·오류 방향을 검증한다.

추가 2×2 실험도 실제 78회 완료했다. 선정 24개 발화 기준 전체·정순22/24, 전체·역순20/24, 발화별·정순21/24, 발화별·역순23/24다. 기존 두 A 오류는 발화별 입력에서 두 등록 순서 모두 수정됐지만 새 오류도 생겼다. [심층 진단 분석](../runs/speaker_moss_diagnostics_01/analysis/REPORT.md), [실험 설계·재현](NEXT_EXPERIMENT.md). 입력 구조 민감성은 관측했으며 의미 편향·대화행동의 인과 효과는 미입증이다.

다음 개선 설계·코드: [30초 경계 정렬 대조와 보수적 합의 보정](BOUNDARY_IMPROVEMENT.md). 모델의 30초 길이를 늘리지 않고 전체 입력 앞 무음 0/2/30/32초×2대화, 총 8회로 정렬·절대 위치를 점검한다. 경계에서 떨어진 조건들의 만장일치일 때만 수정하는 후보를 함께 채점한다. 로컬 19개 검사 통과, 실제 새 GPU 실험 대기다.

로컬 MOSS 환경 설치는 공식 PyTorch 대용량 다운로드가 느려 중단하고 서버 실행으로 전환했다. **서버에서 실제 8개를 완료하고 결과를 로컬에 동기화했다.** 새 서버 실행 시에는 `python speaker_project/check_environment.py --cuda-id 4`로 현재 버전·배정 GPU를 확인한다. 4는 실행 당시 값이므로 현재 할당에 맞춘다.

서버 PyTorch2.4.0/cu118, Transformers5.6.0, A5000에서 해당 pilot 추론이 실제로 성공했다. `bash speaker_project/setup_moss_reuse_torch.sh`가 기존 torch를 공유·버전 고정한 별도 환경을 구성한다. ensurepip 누락은 공식 standalone virtualenv로 우회했다.

## 현재 방향과 범위

원래 순서를 유지한 연속 대화에서, 질문하던 사람이 답변하고 답변하던 사람이 질문할 때 실제 화자를 유지하는지 평가한다. 이력 재배열은 보조 진단으로만 남긴다. 독창성이나 개선 효과는 아직 입증하지 않았다.

첫 데이터에는 2~4명의 화자가 등장한다. 기준 음성·발화 경계를 제공하는 화자 귀속 과제이며 사전 정보 없는 일반 diarization이나 VAD/DER 비교와 구분한다. 다른 참여자의 말을 삭제해 2인 대화처럼 만들지 않는다.

## 받은 데이터

- AMI 공식 수동 주석 ZIP, 대화행동 가이드 PDF, 원본 LICENCE.txt.
- 질문/응답 교대 후보 4개와 같은 방향 질문/응답 대조 후보 4개: 총 8개, 약 265.970초.
- 평가 클립 밖에서 얻은 3초 등록 음성 40개. 각 필요한 화자에 2개씩.
- 70개 평가 가능 대화행동 구간, 그중 지정된 후반 응답 8개.
- 개발/평가의 AMI global speaker ID가 겹치지 않는 것을 확인하는 구조.

선정된 AMI 회의는 scenario 회의다. 과제 상황이 주어졌지만 발화를 대본으로 번갈아 읽은 것은 아니다. 완전히 자유로운 현장 회의라고 표현하지 않는다. 교대·대조 이벤트는 수동 AMI 주석을 이용해 자동 선정했으며 전사·청취 검토는 아직 완료하지 않았다.

## 실제 구조와 데이터 흐름

```text
data/ami/
  raw/                    공식 수동 주석 ZIP, 대화행동 가이드 PDF
  annotations/            원본 XML·LICENCE.txt·화자 자원
  download_manifest.json  출처 URL·크기·로컬 SHA256
  scan/                   전체 주석 후보 탐색 기록
  prepared/
    audio/                원래 순서를 유지한 연속 대화 8개
    references/           평가 범위와 떨어진 등록 음성 40개
    inputs.jsonl          모델용 오디오·등록 음성·발화 경계
    labels.jsonl          평가용 화자 정답·대화행동·전사·이벤트
    review.csv            자동 후보의 전사·청취 검토표
    dataset_manifest.json 실제 개수·분할·제한
    selected_candidates.json
    enrollment_manifest.json
    source_audio_headers.json
    LICENCE_AMI.txt
    ATTRIBUTION.md
```

흐름: 원본 주석 → 질문/응답 연결 탐색 → 원본 WAV의 연속 범위 다운로드 → 별도 등록 음성 → 모델 예측 → 조건별 채점.

inputs.jsonl의 오디오·등록 음성 경로는 prepared/ 기준의 상대 경로다. 서버 추론에는 speaker_project/와 data/ami/prepared/를 전송한다. 원본 주석을 다시 탐색할 때에만 raw/와 annotations/가 추가로 필요하다. 모델에는 labels.jsonl이나 파일명·clip_id의 이벤트 이름을 프롬프트로 제공하지 않는다.

## 코드와 재현

- download_ami.py: 공식 주석과 가이드 다운로드, ZIP CRC 및 안전한 추출.
- prepare_ami.py: NXT 단어 범위·대화행동·질문/응답 연결 파싱, 후보 탐색, HTTP 범위 다운로드.
- build_protocol.py: 등록 음성, 입력/정답 분리, 개발/평가 화자 중복 검사.
- focus_protocol.py / enrich_protocol.py: 같은 사람의 질문→답변·답변→질문 쌍 표시. 기존 labels를 백업하고 음성·review.csv는 보존한다.
- score_predictions.py: 전체 발화 정답률과 같은 사람의 전후 두 발화 동시 정답률. 누락 예측은 오답 처리한다.
- download_wespeaker.py / run_wespeaker.py: 공식 모델 커밋·해시를 고정한 음성 전용 CPU 기준 방법. 실제 실행 완료.
- diarization_projection.py / run_moss.py: 통합 모델 어댑터. 등록 음성 시각만으로 익명 화자를 매핑한다. 단위 검사와 서버 pilot 8개 실행을 완료했다.
- make_review_page.py: 연속 음성과 등록 음성, 지정 발화 재생, 전사·실제 예측을 함께 보는 청취 검토 페이지.

```bash
python speaker_project/download_ami.py --annotations-only
python speaker_project/prepare_ami.py --meetings ES2003a ES2003d TS3006a TS3006b IS1000b IS1000d IS1009d --max-window 90 --per-type 1 --download-clips
python speaker_project/build_protocol.py
```

전체 주석 탐색은 prepare_ami.py --scan-all --max-window 90. 위 prepare 명령은 파생 후보와 review.csv를 재생성하므로 사람이 검토표를 수정한 뒤에는 백업 없이 재실행하지 않는다. 이미 완성된 prepared/를 사용하면 재다운로드할 필요가 없다.

예측 형식은 JSONL이며 각 행은 clip_id와 predictions 목록을 가진다. 각 예측은 segment_id, speaker(등록된 ID)를 가진다. 실제 기준 출력은 runs/speaker_wespeaker_01/predictions.jsonl에 있다.

```bash
python speaker_project/score_predictions.py --labels data/ami/prepared/labels.jsonl --predictions PREDICTIONS.jsonl --output runs/speaker_pilot/scores.json
```

PREDICTIONS.jsonl은 채점할 실제 출력 경로로 바꾼다. WeSpeaker 결과를 재현하려면 새 CPU venv에 requirements-cpu.txt를 설치한 뒤 다음을 실행한다. 모델은 약 26.5MB이며 첫 명령에서 다운로드한다.

```bash
python speaker_project/download_wespeaker.py
python speaker_project/run_wespeaker.py --output runs/speaker_wespeaker_02
python speaker_project/score_predictions.py --labels data/ami/prepared/labels.jsonl --predictions runs/speaker_wespeaker_02/predictions.jsonl --output runs/speaker_wespeaker_02/scores.json
python -m unittest discover -s speaker_project -p test_protocol.py -v
```

로컬 생성 환경은 `.venv-speaker`다. Windows에서는 `.venv-speaker/Scripts/python.exe`를 사용할 수 있다. 기존 폴더에 추론을 덮어쓰지 않으므로 새 실행 이름을 쓴다. MOSS 설치·GPU 지정은 SERVER_RUN.md 참조.

## 제한과 재개 시 확인

- 점수 대상은 완전한 대화행동 구간, 길이 0.8~15초, 다른 화자와 정렬 단어 기준 겹침 5% 이하로 선정했다. 이것이 정밀한 무겹침 정답을 의미하지는 않는다.
- 등록 음성은 선택된 모든 평가 클립과 떨어져 있지만 평가 구간 이후에서 얻은 경우도 있다. 과거만 허용한 온라인 평가가 아니다.
- 대조 조건은 선택한 두 질문/응답의 방향이 같다는 뜻이다. 전체 구간의 모든 사람이 전혀 대화행동을 바꾸지 않는다는 보장은 아니다.
- 개발 그룹 ES2003*, TS3006*와 평가 그룹 IS1000*, IS1009*를 분리했다. AMI 공식 train/test split과 동일하지 않다.
- 다음에는 review.csv와 실제 오디오를 확인하고 후보를 확정한다.
- 같은 사람의 전후 발화 쌍은 16개 중 12개가 길이·겹침·잘림 조건을 만족한다. 그중 전환 6쌍, 대조 6쌍이다. 기존 answer_before/after 점수 차이는 서로 다른 사람을 비교할 수 있어 역할 효과로 해석하지 않는다.
- 현재 음성 전용 모델은 모두 맞혔고 통합 모델은 2개를 틀렸다. 두 오류는 지정 전환 전 발화다. 실제 화자 ID 불일치와 전환 후 악화를 구분하고, 청취·등록 초기화·입력 단위 진단 뒤 개선을 설계한다.

공식 출처: https://groups.inf.ed.ac.uk/ami/corpus/ 및 https://groups.inf.ed.ac.uk/ami/download/ . 라이선스·출처 표시는 prepared/LICENCE_AMI.txt와 ATTRIBUTION.md 참조.
