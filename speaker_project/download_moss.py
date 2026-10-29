"""Download a pinned official MOSS snapshot, verify Hub LFS SHA256 where supplied."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('models/moss-transcribe-diarize'))
    args = parser.parse_args()
    revision = json.loads(Path(__file__).with_name('moss_sources.json').read_text())['model']['revision']
    repo = 'OpenMOSS-Team/MOSS-Transcribe-Diarize'
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true', timeout=60) as response:
        metadata = json.load(response)
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for item in metadata['siblings']:
        name = item['rfilename']
        if '/' in name or Path(name).suffix not in ('.json', '.py', '.jinja', '.txt', '.safetensors', '.md'):
            continue
        target = args.output / name
        url = f'https://huggingface.co/{repo}/resolve/{revision}/{name}'
        expected = item.get('lfs', {}).get('sha256')
        if not target.exists():
            part = target.with_suffix(target.suffix + '.part')
            print('Downloading', name, flush=True)
            started = time.monotonic()
            progress = started
            with urllib.request.urlopen(url, timeout=120) as response, part.open('wb') as stream:
                total = 0
                while block := response.read(1024 * 1024):
                    stream.write(block)
                    total += len(block)
                    if time.monotonic() - progress >= 20:
                        print(f'  {total/2**20:.0f} MiB downloaded', flush=True)
                        progress = time.monotonic()
            part.replace(target)
        hasher = hashlib.sha256()
        with target.open('rb') as stream:
            while block := stream.read(4 * 1024 * 1024):
                hasher.update(block)
        digest = hasher.hexdigest()
        if expected and digest != expected:
            raise ValueError(f'Official LFS checksum mismatch: {name}')
        records.append({'file': name, 'url': url, 'bytes': target.stat().st_size,
                        'sha256': digest, 'hub_lfs_sha256': expected})
    (args.output / 'download_manifest.json').write_text(json.dumps({'repo': repo, 'revision': revision, 'files': records}, indent=2) + '\n')
    print('Finished:', args.output, flush=True)


if __name__ == '__main__':
    main()
