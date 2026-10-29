# 로컬 GPU 실험

**현재 이 절차는 중단했다.** 사용자가 서버 실행을 제안하여 진행 중인 로컬 대용량 설치·wheel 다운로드를 멈췄다. MOSS 모델 파일은 받았지만 `.venv-moss` 설치와 GPU 추론은 완료하지 않았다. 재현용 명령을 아래에 보존하며, 현재 실행 안내는 SERVER_RUN.md다.

2026-10-29 확인한 장비: Windows, RTX 4060 Laptop GPU 8GB, 드라이버 591.66. 당시 여유 VRAM 6544MiB, 시스템 RAM 총 19.69GiB/여유 2.70GiB. A5000 서버에 접속하지 않고 이 장비에서 먼저 소형 실행한다.

기존 음성 기준 `.venv-speaker`와 별도로 `.venv-moss`를 사용한다. 전처리·모델·가중치 커밋은 `moss_sources.json`에 고정한다. 시스템 RAM 여유가 작아 `device_map={'': 'cuda:0'}`로 모델을 바로 GPU에 적재한다. 양자화나 음성 다운샘플링은 하지 않는다.

PowerShell에서 프로젝트 폴더 기준:

```powershell
python -m venv .venv-moss
.\.venv-moss\Scripts\python.exe -m pip install --no-cache-dir torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv-moss\Scripts\python.exe -m pip install --no-cache-dir 'moss-transcribe-diarize @ git+https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git@79c675f0670cc2f824bcbcdd98263704f40f1f7e' 'transformers==5.6.0' 'numpy==2.2.6' 'accelerate>=1.2,<2'
.\.venv-moss\Scripts\python.exe speaker_project/download_moss.py
.\.venv-moss\Scripts\python.exe speaker_project/run_moss.py --model-dir models/moss-transcribe-diarize --cuda-id 0 --limit 1 --output runs/speaker_moss_local_smoke_01
```

단일 로컬 GPU이므로 `--cuda-id 0`을 사용한다. 이 옵션은 torch import 전에 PCI_BUS_ID 순서와 CUDA_VISIBLE_DEVICES를 지정한다. 첫 다운로드는 PyTorch 약 2.9GB와 모델 약 1.8GB를 포함한다. 실제 추론 시간과 구분한다. MOSS 다운로드 체크섬은 스트리밍으로 계산하여 가중치 전체를 RAM에 추가로 올리지 않는다.

smoke에서 메모리와 출력 형식을 확인한 뒤 전체 실행:

```powershell
.\.venv-moss\Scripts\python.exe speaker_project/run_moss.py --model-dir models/moss-transcribe-diarize --cuda-id 0 --output runs/speaker_moss_local_01
.\.venv-speaker\Scripts\python.exe speaker_project/compare_runs.py --joint runs/speaker_moss_local_01 --output runs/speaker_comparison_local/REPORT.md
```

기존 출력 폴더를 덮어쓰지 않으므로 재실행은 `_02`처럼 새 이름을 사용한다. 실제 실행 결과가 나오기 전에는 위 명령의 성공을 가정하지 않는다. 비교 해석·익명 화자 연결 원칙은 SERVER_RUN.md를 따른다.
