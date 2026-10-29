"""Download a small AMI starter set from the official Edinburgh mirror."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen
import wave
import zipfile

ANNOTATIONS = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
GUIDELINES = "https://groups.inf.ed.ac.uk/ami/corpus/Guidelines/dialogue_acts_manual_1.0.pdf"
AUDIO_BASE = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"reuse {target} ({target.stat().st_size / 2**20:.1f} MiB)", flush=True)
    else:
        partial = target.with_name(target.name + ".part")
        request = Request(url, headers={"User-Agent": "AudioCourseResearch/1.0"})
        print(f"download {url}", flush=True)
        with urlopen(request, timeout=60) as response, partial.open("wb") as output:
            expected = response.headers.get("Content-Length")
            total = 0
            last_report = time.monotonic()
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                total += len(chunk)
                if time.monotonic() - last_report >= 8:
                    print(f"  {target.name}: {total / 2**20:.1f} MiB", flush=True)
                    last_report = time.monotonic()
            if expected and total != int(expected):
                raise IOError(f"Incomplete download: {target}: {total}/{expected}")
        partial.replace(target)
    return {"url": url, "path": str(target), "bytes": target.stat().st_size,
            "sha256": sha256(target), "checksum_origin": "computed_locally"}


def extract_annotations(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad:
            raise ValueError(f"ZIP CRC failed: {bad}")
        for info in z.infolist():
            # Reject absolute paths, drive letters and traversal before extraction.
            name = info.filename.replace("\\", "/")
            if name.startswith("/") or ":" in name:
                raise ValueError(f"Unsafe ZIP path: {name}")
            path = (root / name).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f"Unsafe ZIP path: {name}")
            if info.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            data = z.read(info)
            if path.exists():
                if path.read_bytes() != data:
                    raise FileExistsError(f"Existing annotation differs; preserving it: {path}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
    print(f"annotations extracted to {root}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/ami"))
    parser.add_argument("--annotations-only", action="store_true")
    parser.add_argument("--meetings", nargs="+", default=["IB4003", "ES2002a", "IS1000a"])
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for meeting in args.meetings:
        if not re.fullmatch(r"[A-Z]{2}\d{4}[a-z]?", meeting):
            parser.error(f"Invalid AMI meeting ID: {meeting}")
    manifest_path = root / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"dataset": "AMI Meeting Corpus", "assets": {}}

    def record(key, asset):
        manifest["assets"][key] = asset
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    annotation_path = root / "raw" / "ami_public_manual_1.6.2.zip"
    record("annotations", fetch(ANNOTATIONS, annotation_path))
    extract_annotations(annotation_path, root / "annotations")
    record("dialogue_act_guidelines", fetch(GUIDELINES, root / "raw" / "dialogue_acts_manual_1.0.pdf"))
    if not args.annotations_only:
        for meeting in args.meetings:
            filename = f"{meeting}.Mix-Headset.wav"
            path = root / "audio" / filename
            asset = fetch(f"{AUDIO_BASE}/{meeting}/audio/{filename}", path)
            with wave.open(str(path), "rb") as audio:
                asset["audio"] = {"sample_rate": audio.getframerate(), "channels": audio.getnchannels(),
                                  "sample_width": audio.getsampwidth(), "frames": audio.getnframes(),
                                  "duration_seconds": audio.getnframes() / audio.getframerate()}
            record(meeting, asset)
            print(f"verified {meeting}: {asset['audio']}", flush=True)
    print(f"Finished. Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
