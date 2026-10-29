"""Package only the pilot data, code, and actual baseline outputs for SSH transfer."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('speaker_pilot_transfer.zip'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    files = []
    for relative in ('speaker_project', 'data/ami/prepared', 'runs/speaker_wespeaker_01'):
        for path in (root / relative).rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix in ('.bak', '.pyc'):
                continue
            files.append(path)
    with zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(root).as_posix())
    with zipfile.ZipFile(args.output) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError('Archive CRC failed: ' + bad)
    report = {'archive': str(args.output), 'files': len(files), 'bytes': args.output.stat().st_size,
              'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
              'includes': ['speaker_project/', 'data/ami/prepared/', 'runs/speaker_wespeaker_01/'],
              'excludes': ['virtual environments', 'checkpoints', 'full AMI source', 'seed-vc', 'old SVC runs', 'v1 label backup']}
    args.output.with_suffix('.manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
