"""Faster first setup on this Windows laptop: official wheel, ranges, SHA256."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import html
import json
from pathlib import Path
import re
import tempfile
import time
import urllib.parse
import urllib.request

NAME = 'torch-2.8.0+cu126-cp311-cp311-win_amd64.whl'
URL = 'https://download-r2.pytorch.org/whl/cu126/' + urllib.parse.quote(NAME)
SIZE = 2915474854
CHUNK = 16 * 1024 * 1024


def main():
    target = Path('models/installers') / NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen('https://download.pytorch.org/whl/cu126/torch/', timeout=60) as response:
        index = response.read().decode()
    matches = [html.unescape(m) for m in re.findall(r'href="([^"]+)"', index)
               if urllib.parse.quote(NAME) in m or NAME in m]
    if len(matches) != 1:
        raise ValueError('Cannot uniquely find official wheel checksum.')
    expected = urllib.parse.parse_qs(urllib.parse.urlsplit(matches[0]).fragment)['sha256'][0]
    done = set()
    state_path = target.with_suffix('.ranges.json')
    if target.exists() and state_path.exists():
        done = set(json.loads(state_path.read_text())['completed_chunks'])
    else:
        # The pip process can keep appending; copy only a stable complete prefix.
        candidates = list(Path(tempfile.gettempdir()).glob('pip-*/' + NAME))
        source = max(candidates, key=lambda p: p.stat().st_size) if candidates else None
        prefix = source.stat().st_size // CHUNK if source else 0
        with target.open('wb') as stream:
            if source:
                with source.open('rb') as prior:
                    for number in range(prefix):
                        block = prior.read(CHUNK)
                        if len(block) != CHUNK:
                            break
                        stream.write(block)
                        done.add(number)
            stream.truncate(SIZE)
    print(f'Reused {len(done)*CHUNK/2**20:.0f} MiB; downloading remaining ranges', flush=True)

    def fetch(number):
        start, end = number * CHUNK, min(SIZE, (number + 1) * CHUNK) - 1
        for attempt in range(4):
            try:
                request = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(request, timeout=90) as response:
                    expected_range = f'bytes {start}-{end}/{SIZE}'
                    if response.status != 206 or response.headers.get('Content-Range') != expected_range:
                        raise ValueError('Range response mismatch')
                    payload = response.read()
                if len(payload) != end - start + 1:
                    raise ValueError('Incomplete wheel range')
                with target.open('r+b', buffering=0) as stream:
                    stream.seek(start)
                    stream.write(payload)
                return number
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1 + attempt)

    numbers = [n for n in range((SIZE + CHUNK - 1) // CHUNK) if n not in done]
    last = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for future in as_completed([pool.submit(fetch, n) for n in numbers]):
            done.add(future.result())
            state_path.write_text(json.dumps({'completed_chunks': sorted(done)}))
            if time.monotonic() - last > 15:
                print(f'{min(SIZE,len(done)*CHUNK)/2**20:.0f}/{SIZE/2**20:.0f} MiB', flush=True)
                last = time.monotonic()
    hasher = hashlib.sha256()
    with target.open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            hasher.update(block)
    if hasher.hexdigest() != expected:
        raise ValueError('Wheel SHA256 differs from official PyTorch index.')
    target.with_suffix('.verified.json').write_text(json.dumps({'url': URL, 'sha256': expected, 'bytes': SIZE}, indent=2))
    print('Verified official wheel:', target, flush=True)


if __name__ == '__main__':
    main()
