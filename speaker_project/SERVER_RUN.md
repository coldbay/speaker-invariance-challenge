# A5000에서 통합 모델 비교

2026-10-29 기준, WeSpeaker 로컬 실험과 **MOSS 서버 8개 pilot 실행을 완료했다.** PyTorch2.4.0+cu118·Transformers5.6.0·A5000에서 성공했다. 아래는 재현 절차이며, 새 환경에서는 첫 1개 확인부터 진행한다. 실제 결과 분석은 ../runs/speaker_comparison/analysis/REPORT.md 참조.

## 전송할 파일

프로젝트 기준 `speaker_project/`와 `data/ami/prepared/`만 있으면 된다. AMI 원본 회의 전체, annotations, seed-vc, 보컬 변환 runs는 필요 없다. 준비된 `speaker_pilot_transfer.zip`은 이 두 폴더와 WeSpeaker 결과를 포함한다. 서버 프로젝트 폴더에서 압축을 풀면 된다.

현재 과거 서버 경로는 `/tmp/pycharm_project_e52e89ff`다. 실제 프로젝트를 옮겼다면 그 위치에서 실행한다. 아래 명령은 Bash/SSH 터미널용이다.

우선 현재 서버 환경을 확인한다. 아래 4는 이전 할당 값으로 현재 배정에 맞춰 바꾼다. 이 검사는 설치·업그레이드를 하지 않는다.

```bash
cd /tmp/pycharm_project_e52e89ff
python speaker_project/check_environment.py --cuda-id 4
```

PyTorch가 이미 2.8 이상이고 CUDA 동작이 확인되면 그 환경에서 MOSS 관련 의존성만 준비할 수 있다. 사용자는 현재 2.4/cu118임을 확인했다. **2.4가 반드시 불가능하다는 뜻은 아니다.** MOSS의 선택 torch-runtime extra는 >=2.8을 요구하지만, 고정 Transformers 5.6.0의 PyTorch 활성화 기준은 >=2.4다. 대용량 PyTorch 재설치 전에 아래 호환성 시험을 먼저 할 수 있다. 로컬에서 MOSS 모델 파일은 확보했으나 PyTorch 설치 지연 때문에 로컬 추론은 시작하지 않고 중단했다. 작은 전송 ZIP에는 가중치를 포함하지 않는다.

## 기존 PyTorch 2.4를 재사용하는 우선 시험

이 경로는 공식 검증 환경이 아니라 실제 호환 여부를 확인하는 smoke 실험이다. 원래 PyTorch를 공유하는 별도 venv를 만들고, torch/torchaudio/torchvision 버전을 constraints로 고정해 자동 업그레이드를 막는다. MOSS의 torch-runtime extra는 설치하지 않는다. NumPy는 1.26.4로 지정한다. 원래 전역 환경을 수정하지 않는다.

서버에서 ensurepip 누락으로 처음 환경 생성이 실패했다. 수정한 설치 스크립트는 이 경우 공식 PyPA virtualenv zipapp으로 자동 전환한다. apt/sudo 설치는 필요 없다. 실패한 `.venv-moss-reuse`는 `.incomplete.<시각>` 이름으로 보존하고 다시 생성하며, 정상 환경은 재사용해 중간 설치 실패 후에도 같은 명령을 재실행할 수 있다. `create_moss_environment.py`도 함께 동기화한 뒤 아래 명령을 실행한다. [PyPA 공식 zipapp 안내](https://virtualenv.pypa.io/en/latest/how-to/install.html).

```bash
bash speaker_project/setup_moss_reuse_torch.sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4
.venv-moss-reuse/bin/python speaker_project/run_moss.py --limit 1 --output runs/speaker_moss_smoke_01
```

4는 현재 배정 GPU에 맞춘다. import 또는 추론 단계에서 버전 관련 오류가 나면 해당 traceback을 기준으로 호환 API를 판단한다. 고정 Transformers의 기준은 [공식 v5.6.0 소스](https://github.com/huggingface/transformers/blob/v5.6.0/src/transformers/utils/import_utils.py)의 is_torch_available 참조. 이 호환성 경로로 실제 8개 pilot을 완료했다.

성공하면 전체 8개를 실행한다.

```bash
.venv-moss-reuse/bin/python speaker_project/run_moss.py --output runs/speaker_moss_pilot_01
.venv-moss-reuse/bin/python speaker_project/compare_runs.py --joint runs/speaker_moss_pilot_01 --output runs/speaker_comparison/REPORT.md
```

## 격리 환경

공식 MOSS 현재 코드는 Python >=3.10, Transformers >=5.6, 선택 torch-runtime은 PyTorch >=2.8을 요구한다. 공식 테스트 환경은 Python 3.12다. 기존 SVC의 PyTorch 2.4 환경과 분리하여 새 venv에 설치한다. Python 3.12가 있으면 첫 줄을 `python3.12 -m venv .venv-moss`로 바꾼다.

```bash
cd /tmp/pycharm_project_e52e89ff
nvidia-smi
python3 -m venv .venv-moss
source .venv-moss/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install 'moss-transcribe-diarize @ git+https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git@79c675f0670cc2f824bcbcdd98263704f40f1f7e'
python -m pip install 'accelerate>=1.2,<2'
```

이 CUDA 12.6 PyTorch wheel이 서버 드라이버에서 동작하는지는 아래 사전 확인으로 판단한다. 기존 컨테이너의 `nvcc`가 11.8이라는 정보만으로 드라이버 호환 여부를 판정하지 않는다. 설치가 실패하면 해당 오류와 nvidia-smi 출력이 다음 진단 자료다. flash-attn/vLLM 설치는 이 실험에 필요하지 않다.

공식 근거: [MOSS 환경·사용법](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize), [PyTorch 2.8 wheel](https://pytorch.org/get-started/previous-versions/#v280).

## GPU 지정과 1개 확인

아래 `4`는 이전 실험에서 사용한 값이다. 현재 배정받은 컨테이너 내부 GPU 번호로 바꾼다. Docker가 GPU를 재매핑했다면 호스트 번호와 다를 수 있다. 환경 변수를 설정하고 한 GPU만 노출한 프로세스 내부에서는 `cuda:0`을 쓴다.

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4
python -c "import torch; print(torch.__version__, torch.version.cuda); print('available:',torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.zeros(1,device='cuda:0'))"
python -m unittest discover -s speaker_project -p test_protocol.py -v
python speaker_project/run_moss.py --limit 1 --output runs/speaker_moss_smoke_01
```

첫 실행에는 공개 모델 다운로드 시간이 포함된다. 실제 pilot의 모델 로딩 후 8개 처리 시간은 86.46초, 최대 PyTorch 할당 VRAM은 약 1.90GiB였다. 다운로드·모델 로딩·드라이버 전체 메모리 수치는 제외한다. `case_000/raw_output.json`의 전사, 시각, `speaker_mapping`, `reached_token_limit`을 확인한다. smoke 결과 한 개만 전체 정답 파일에 채점하면 나머지가 누락 오답이 되므로 전체 성능으로 보고하지 않는다.

## 전체 8개와 채점

```bash
python speaker_project/run_moss.py --output runs/speaker_moss_pilot_01
python speaker_project/score_predictions.py --labels data/ami/prepared/labels.jsonl --predictions runs/speaker_moss_pilot_01/predictions.jsonl --output runs/speaker_moss_pilot_01/scores.json
```

폴더가 이미 있으면 `_02`처럼 새 이름을 쓴다. 기존 결과는 보존한다. 결과 분석에는 `scores.json`, `predictions.jsonl`, `run_manifest.json`, 각 `case_*/raw_output.json`을 가져오면 된다. 입력 WAV를 다시 가져올 필요는 없다.

## 비교 해석

- MOSS 입력 앞에는 WeSpeaker와 동일한 등록 음성들을 붙인다. 그 뒤 평가 대화는 원래 PCM·시간 순서를 유지한다. 이 등록용 접두부는 자연 대화 자체로 주장하지 않는다.
- 익명 S01/S02를 실제 A/B로 연결할 때 **등록 음성 시각만 사용**한다. 평가 정답으로 최적 화자 순열을 찾지 않는다. 등록 화자를 못 찾으면 관련 평가 예측이 누락될 수 있다.
- 통합 모델의 시각 구간을 공개된 대화행동 구간에 최대 겹침으로 투영한다. 전사나 시각 누락도 점수에 영향을 줄 수 있어, 점수 하락을 모두 내용 편향으로 해석하지 않는다.
- 주 지표는 같은 사람의 전후 두 발화를 **모두 정확히 맞힌 비율**이다. 계속 같은 오답을 내는 경우를 성공으로 치지 않는다. 전체 발화 정답률·누락·전환 전 정답이던 쌍의 유지율도 함께 보고한다.
- 현재 모든 8개 이벤트는 청취 검토 대기이고 표본이 작다. 차이가 나도 역할 변화의 인과 효과나 최종 독창성 입증으로 표현하지 않는다.
