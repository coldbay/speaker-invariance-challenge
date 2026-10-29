#!/usr/bin/env bash
# Compatibility smoke route, NOT an upstream-validated torch2.4 environment.
# Creates a venv that inherits the existing CUDA torch, without torch upgrades.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_root"
python speaker_project/create_moss_environment.py
source .venv-moss-reuse/bin/activate
mkdir -p runs/moss_setup
python - <<'PY'
import importlib.metadata
from pathlib import Path
import torch
versions=[]
for name in ('torch','torchaudio','torchvision'):
    try:
        versions.append(name+'=='+importlib.metadata.version(name))
    except importlib.metadata.PackageNotFoundError:
        pass
Path('runs/moss_setup/preserve_torch.txt').write_text('\n'.join(versions)+'\n')
assert tuple(map(int,torch.__version__.split('+')[0].split('.')[:2])) >= (2,4), 'Need torch >=2.4 for pinned Transformers.'
print('Preserving:',versions)
PY
python -m pip install --upgrade pip
python -m pip install -c runs/moss_setup/preserve_torch.txt \
  'moss-transcribe-diarize @ git+https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git@79c675f0670cc2f824bcbcdd98263704f40f1f7e' \
  'transformers==5.6.0' 'numpy==1.26.4' 'accelerate>=1.2,<2'
python - <<'PY'
import torch, transformers
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.utils import torch_compilable_check
from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import generate_transcription
print('Imports passed:',torch.__version__,torch.version.cuda,transformers.__version__)
print('This is a compatibility smoke route; real MOSS inference remains to be tested.')
PY
echo 'Next: export CUDA_DEVICE_ORDER=PCI_BUS_ID; export CUDA_VISIBLE_DEVICES=<assigned index>'
echo '.venv-moss-reuse/bin/python speaker_project/run_moss.py --limit 1 --output runs/speaker_moss_smoke_01'
