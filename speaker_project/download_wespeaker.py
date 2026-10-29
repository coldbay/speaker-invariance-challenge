"""Fetch the official, immutable-revision WeSpeaker ONNX checkpoint and metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REPO = 'Wespeaker/wespeaker-voxceleb-resnet34-LM'
DEFAULT_REVISION = 'f0c48c298fd835726c27956a5d617bad7115627e'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('models/wespeaker-resnet34'))
    parser.add_argument('--revision', help='Full HF commit; defaults to the recorded or pinned pilot revision.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'manifest.json'
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    revision = args.revision or old.get('revision') or DEFAULT_REVISION
    if not revision:
        with urllib.request.urlopen(f'https://huggingface.co/api/models/{REPO}', timeout=60) as response:
            revision = json.load(response)['sha']
    files = {}
    for name in ('voxceleb_resnet34_LM.onnx', 'config.yaml', 'README.md'):
        path = args.output / name
        url = f'https://huggingface.co/{REPO}/resolve/{revision}/{name}'
        if not path.exists() or old.get('revision') != revision:
            temp = path.with_suffix(path.suffix + '.part')
            print('Downloading', name, flush=True)
            with urllib.request.urlopen(url, timeout=90) as response, temp.open('wb') as target:
                while block := response.read(1024 * 1024):
                    target.write(block)
            temp.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if old.get('revision') == revision and name in old.get('files', {}):
            if digest != old['files'][name]['sha256']:
                raise ValueError(f'Existing checkpoint changed: {path}')
        files[name] = {'url': url, 'bytes': path.stat().st_size, 'sha256': digest}
    result = {'repository': REPO, 'revision': revision, 'license': 'CC-BY-4.0', 'files': files}
    manifest_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
