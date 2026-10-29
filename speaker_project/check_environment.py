"""Read-only preflight: report GPU/runtime compatibility before installing MOSS."""
import argparse
import importlib.metadata
import json
import os
import platform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda-id', help='Assigned GPU index inside the container, PCI_BUS_ID order.')
    args = parser.parse_args()
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    if args.cuda_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id
    result = {'python': platform.python_version(), 'platform': platform.platform(),
              'CUDA_DEVICE_ORDER': os.environ.get('CUDA_DEVICE_ORDER'),
              'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES')}
    for name in ('torch', 'torchaudio', 'transformers', 'accelerate', 'moss-transcribe-diarize'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    try:
        import torch
        result['torch_cuda_runtime'] = torch.version.cuda
        result['cuda_available'] = torch.cuda.is_available()
        result['torch_at_least_2_8'] = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]) >= (2, 8)
        if torch.cuda.is_available():
            result['gpu'] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info(0)
            result['vram_free_gib'] = free / 2**30
            result['vram_total_gib'] = total / 2**30
            result['cuda_tensor_check'] = float(torch.zeros(1, device='cuda:0').cpu()[0]) == 0
            result['bf16_supported'] = torch.cuda.is_bf16_supported()
    except Exception as exc:
        result['runtime_error'] = f'{type(exc).__name__}: {exc}'
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
