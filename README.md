# Invariant Speaker Attribution Challenge

회의 음성의 화자 귀속을 **한 번 맞히는가**가 아니라 **흔들림 없이 맞히는가**로 평가하는 벤치마크입니다.
통합 전사·화자분리 모델(MOSS-Transcribe-Diarize)은 한 입력에서 91.4%를 맞히지만, 대화 내용은 그대로 둔 10가지 변형(앞 무음 0–4초, 등록 순서 역순) 모두에서 맞힌 발화는 67.1%입니다.

- 챌린지 페이지: `submission/index.html`
- 데이터 출처: AMI Meeting Corpus (CC BY 4.0, `data/ami/prepared/LICENCE_AMI.txt`, `ATTRIBUTION.md`)

## 구성

| 경로 | 내용 |
|---|---|
| `data/ami/heldout/` | **최종 평가 데이터**: 미관측 4인 회의 8개, Main 70발화, Hard 175발화 |
| `data/ami/prepared/` | Starter 데이터: 개발·진단용 8구간 |
| `speaker_project/` | Baseline, 제안 방법, 데이터 구성, 평가 코드 |
| `runs/speaker_heldout_01/` | Held-out 서버 원본 결과(`results_only.zip`), 분석, baseline 제출 파일(`submissions/`) |
| `runs/…` | Starter 실험 결과와 분석 보고서 |
| `models/wespeaker-resnet34/manifest.json` | WeSpeaker 모델 출처와 SHA256 (모델 파일은 스크립트로 내려받음) |

## 빠른 시작: 자기 모델 채점

채점은 Python 표준 라이브러리만 사용합니다.

```bash
python speaker_project/make_condition_inputs.py --output condition_inputs        # 조건별 입력 80개 (통합 모델용)
python speaker_project/evaluate_submission.py --predictions my_predictions/      # 예측 폴더 채점
```

예측 폴더에는 조건마다 `<조건>.jsonl` 파일 하나씩, 총 10개를 둡니다.
조건 이름: `pad0000ms_fwd`, `pad0500ms_fwd`, …, `pad4000ms_fwd`, `pad0000ms_rev`.

```json
{"clip_id": "TS3010b_question_answer_reversal_01", "predictions": [{"segment_id": "segment_000", "speaker": "A"}]}
```

baseline 결과 확인:

```bash
python speaker_project/evaluate_submission.py --predictions runs/speaker_heldout_01/submissions/moss
```

## Baseline 재현

```bash
pip install -r speaker_project/requirements-cpu.txt
python speaker_project/download_wespeaker.py
bash speaker_project/setup_moss_reuse_torch.sh                                   # MOSS 전용 환경 (CUDA PyTorch 필요)
python speaker_project/run_heldout_sweep.py --shard 0/1 --output runs/heldout    # MOSS 80회 (GPU)
python speaker_project/analyze_heldout.py --run runs/heldout                     # WeSpeaker·제안 방법 + 채점
```

MOSS 실행은 `runs/speaker_moss_pilot_01/run_manifest.json`에 고정된 모델 revision, 패키지 버전, 디코딩 설정을 사용합니다.
실측 환경: RTX A5000, PyTorch 2.4.0+cu118, Transformers 5.6.0, BF16, greedy.

## 리더보드 (Held-out)

| 방법 | Main 불변 정답률 | Main flip | Hard 불변 정답률 | Hard flip |
|---|---:|---:|---:|---:|
| MOSS + 임베딩 재연결·확인 (제안) | 98.6% | 0 | 62.9% | 14 |
| WeSpeaker ResNet34-LM | 97.1% | 0 | 64.6% | 0 |
| MOSS-Transcribe-Diarize | 67.1% | 23 | 36.0% | 81 |

자세한 지표 정의, 관련 연구, 한계는 챌린지 페이지를 참고하세요.
