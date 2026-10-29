"""Index original AMI dialogue acts and prepare contiguous role-change clips.

Automatic event labels are candidates for human review, not verified role-change
ground truth. No audio reordering, concatenation, or speaker removal is used.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import struct
from urllib.request import Request, urlopen
import wave
import xml.etree.ElementTree as ET

NITE = "{http://nite.sourceforge.net/}"
BASE = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def span(href, elements):
    ids = re.findall(r"id\(([^)]+)\)", href)
    if not ids:
        return []
    positions = {element.get(NITE + "id"): i for i, element in enumerate(elements)}
    if any(identifier not in positions for identifier in ids):
        raise ValueError(f"Unresolved NXT span: {href}")
    left, right = positions[ids[0]], positions[ids[-1]]
    if right < left:
        raise ValueError(f"Reversed NXT span: {href}")
    return elements[left:right + 1]


def parse_meeting(annotation_root, meeting):
    ontology = ET.parse(annotation_root / "ontologies/da-types.xml").getroot()
    labels = {e.get(NITE + "id"): e.get("gloss") for e in ontology.iter() if e.get(NITE + "id")}
    resource = next(e for e in ET.parse(annotation_root / "corpusResources/meetings.xml").getroot()
                    if e.get("observation") == meeting)
    speakers = {e.get("nxt_agent"): e.get("global_name") for e in resource if e.tag == "speaker"}
    acts, words_all = {}, []
    for speaker in speakers:
        words = list(ET.parse(annotation_root / "words" / f"{meeting}.{speaker}.words.xml").getroot())
        for element in words:
            if element.tag != "w" or element.get("punc") == "true":
                continue
            try:
                start, end = float(element.get("starttime")), float(element.get("endtime"))
            except (ValueError, TypeError):
                continue
            if math.isfinite(start) and math.isfinite(end) and end > start:
                words_all.append({"speaker": speaker, "start": start, "end": end})
        filename = annotation_root / "dialogueActs" / f"{meeting}.{speaker}.dialog-act.xml"
        if not filename.exists():
            continue
        for element in ET.parse(filename).getroot():
            selected, type_id = [], None
            for child in element:
                href = child.get("href", "")
                if child.tag == NITE + "child":
                    selected.extend(span(href, words))
                elif child.tag == NITE + "pointer" and child.get("role") == "da-aspect":
                    ids = re.findall(r"id\(([^)]+)\)", href)
                    type_id = ids[0] if ids else None
            timed = []
            for word in selected:
                try:
                    start, end = float(word.get("starttime")), float(word.get("endtime"))
                except (ValueError, TypeError):
                    continue
                if math.isfinite(start) and math.isfinite(end) and end > start and word.tag == "w":
                    timed.append((start, end))
            if not timed:
                continue
            acts[element.get(NITE + "id")] = {"id": element.get(NITE + "id"), "speaker": speaker,
                "global_speaker": speakers[speaker], "start": min(t[0] for t in timed),
                "end": max(t[1] for t in timed), "dialogue_act": labels.get(type_id, "Unknown"),
                "type_id": type_id, "text": " ".join(w.text or "" for w in selected if w.tag == "w")}
    qa, unresolved = [], 0
    filename = annotation_root / "dialogueActs" / f"{meeting}.adjacency-pairs.xml"
    for element in ET.parse(filename).getroot():
        pointers = {}
        for child in element:
            ids = re.findall(r"id\(([^)]+)\)", child.get("href", ""))
            if ids:
                pointers[child.get("role")] = ids[0]
        source, target = acts.get(pointers.get("source")), acts.get(pointers.get("target"))
        if source is None or target is None:
            unresolved += 1
            continue
        expected_response = {"ami_da_5": "ami_da_4", "ami_da_11": "ami_da_9"}
        if (source["type_id"] in expected_response and target["type_id"] == expected_response[source["type_id"]]
                and source["speaker"] != target["speaker"] and pointers.get("type") == "apt_1"
                and 0 <= target["start"] - source["start"] <= 30):
            qa.append({"id": element.get(NITE + "id"), "question": source, "answer": target})
    qa.sort(key=lambda row: row["question"]["start"])
    return {"meeting": meeting, "meeting_type": resource.get("type"), "speakers": speakers,
            "acts": acts, "words": words_all, "qa_pairs": qa, "unresolved_adjacency_pairs": unresolved}


def overlap_fraction(start, end, speaker, words):
    intervals = sorted((max(start, w["start"]), min(end, w["end"])) for w in words
                       if w["speaker"] != speaker and w["end"] > start and w["start"] < end)
    total, rightmost = 0.0, start
    for left, right in intervals:
        if right > rightmost:
            total += right - max(left, rightmost)
            rightmost = right
    return total / (end - start)


def find_candidates(data, max_window):
    candidates = []
    for i, first in enumerate(data["qa_pairs"]):
        for second in data["qa_pairs"][i + 1:]:
            if second["question"]["start"] < first["answer"]["end"]:
                continue
            duration = second["answer"]["end"] - first["question"]["start"]
            if duration > max_window - 4:
                break
            q1, a1 = first["question"], first["answer"]
            q2, a2 = second["question"], second["answer"]
            reverse = q1["speaker"] == a2["speaker"] and a1["speaker"] == q2["speaker"]
            same = q1["speaker"] == q2["speaker"] and a1["speaker"] == a2["speaker"]
            if not (reverse or same):
                continue
            targets = [a1, a2]
            if any(not .8 <= a["end"] - a["start"] <= 15 for a in targets):
                continue
            overlaps = [overlap_fraction(a["start"], a["end"], a["speaker"], data["words"]) for a in targets]
            if max(overlaps) > .05:
                continue
            start, end = max(0, q1["start"] - 2), a2["end"] + 2
            present = sorted({w["speaker"] for w in data["words"] if w["end"] > start and w["start"] < end})
            candidates.append({"meeting": data["meeting"], "meeting_type": data["meeting_type"],
                "event": "question_answer_reversal" if reverse else "same_direction_control",
                "start": start, "end": end, "duration": end - start, "speakers_present": present,
                "qa_before": first, "qa_after": second, "target_overlap_fractions": overlaps,
                "review_status": "automatic_candidate_not_human_verified"})
    return candidates


def read_range(url, first, last):
    request = Request(url, headers={"Range": f"bytes={first}-{last}", "User-Agent": "AudioCourseResearch/1.0"})
    with urlopen(request, timeout=60) as response:
        expected = f"bytes {first}-{last}/"
        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected):
            raise IOError("Server did not honor byte range; refusing an unintended full-meeting download.")
        raw = response.read(last - first + 2)
    if len(raw) != last - first + 1:
        raise IOError("Incomplete byte range.")
    return raw


def wav_header(url):
    header = read_range(url, 0, 65535)
    if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise ValueError("Expected RIFF/WAVE.")
    offset, fmt = 12, None
    while offset + 8 <= len(header):
        tag, size = header[offset:offset+4], struct.unpack_from("<I", header, offset + 4)[0]
        if tag == b"fmt ":
            fmt = struct.unpack_from("<HHIIHH", header, offset + 8)
        if tag == b"data":
            if fmt is None or fmt[0] != 1 or fmt[5] != 16:
                raise ValueError("Only PCM16 WAV is supported for ranged extraction.")
            return {"channels": fmt[1], "sample_rate": fmt[2], "block_align": fmt[4],
                    "sample_width": 2, "data_offset": offset + 8, "data_bytes": size,
                    "duration": size / fmt[4] / fmt[2]}
        offset += 8 + size + size % 2
    raise ValueError("WAV header exceeds the supported probe size.")


def download_clip(url, header, start, end, path):
    first_frame = max(0, math.floor(start * header["sample_rate"]))
    last_frame = min(math.ceil(end * header["sample_rate"]), header["data_bytes"] // header["block_align"])
    if last_frame <= first_frame:
        raise ValueError("Empty WAV selection.")
    if not path.exists():
        first_byte = header["data_offset"] + first_frame * header["block_align"]
        last_byte = header["data_offset"] + last_frame * header["block_align"] - 1
        pcm = read_range(url, first_byte, last_byte)
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as output:
            output.setnchannels(header["channels"])
            output.setsampwidth(header["sample_width"])
            output.setframerate(header["sample_rate"])
            output.writeframes(pcm)
    with wave.open(str(path), "rb") as audio:
        if audio.getnframes() != last_frame-first_frame or audio.getframerate() != header["sample_rate"]:
            raise ValueError(f"Existing clip differs from selection: {path}")
        if len(audio.readframes(audio.getnframes())) != (last_frame-first_frame)*header['block_align']:
            raise ValueError(f"Truncated PCM data: {path}")
    return {"path": str(path), "source_url": url, "source_frame_start": first_frame,
            "source_frame_end_exclusive": last_frame,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/ami"))
    parser.add_argument("--meetings", nargs="+", default=["IB4003", "ES2002a", "IS1000a"])
    parser.add_argument("--max-window", type=float, default=75)
    parser.add_argument("--per-type", type=int, default=2)
    parser.add_argument("--download-clips", action="store_true")
    parser.add_argument("--scan-all", action="store_true", help="Index all meetings with manual adjacency links; do not download audio.")
    args = parser.parse_args()
    if args.scan_all and args.download_clips:
        parser.error("Scan annotations first and explicitly select a small set for audio download.")
    root = args.root.resolve()
    derived = root / "prepared"
    derived.mkdir(parents=True, exist_ok=True)
    if args.scan_all:
        args.meetings = sorted(p.name.removesuffix('.adjacency-pairs.xml')
                               for p in (root / 'annotations/dialogueActs').glob('*.adjacency-pairs.xml'))
        derived = root / 'scan'
        derived.mkdir(exist_ok=True)
    all_candidates, selected, statistics, speaker_groups = [], [], {}, {}
    for meeting in args.meetings:
        data = parse_meeting(root / "annotations", meeting)
        candidates = find_candidates(data, args.max_window)
        all_candidates.extend(candidates)
        speaker_groups[meeting] = data["speakers"]
        statistics[meeting] = {"meeting_type": data["meeting_type"], "speakers": data["speakers"],
            "dialogue_act_count": len(data["acts"]), "linked_information_qa_count": len(data["qa_pairs"]),
            "unresolved_adjacency_pairs": data["unresolved_adjacency_pairs"],
            "candidate_reversals": sum(c["event"] == "question_answer_reversal" for c in candidates),
            "candidate_controls": sum(c["event"] == "same_direction_control" for c in candidates)}
        used = []
        for event in ["question_answer_reversal", "same_direction_control"]:
            count = 0
            for candidate in sorted((c for c in candidates if c["event"] == event), key=lambda c: (c["duration"],c["start"])):
                if any(candidate["end"] > left and candidate["start"] < right for left,right in used):
                    continue
                entry = dict(candidate, clip_id=f"{meeting}_{event}_{count+1:02d}")
                selected.append(entry)
                used.append((entry["start"],entry["end"]))
                count += 1
                if count >= args.per_type:
                    break
        print(meeting, statistics[meeting], flush=True)
        write_json(derived / f"{meeting}.dialogue_acts.json", sorted(data["acts"].values(),key=lambda a:a["start"]))
    write_json(derived / "annotation_statistics.json", statistics)
    write_json(derived / "all_candidates.json", all_candidates)
    write_json(derived / "selected_candidates.json", selected)
    write_json(derived / "speaker_groups.json", speaker_groups)
    with (derived / "review.csv").open("w",encoding="utf-8-sig",newline="") as output:
        fields=["clip_id","meeting","event","start","end","speakers_present","question_before","answer_before","question_after","answer_after","review_status","review_notes"]
        writer=csv.DictWriter(output,fieldnames=fields)
        writer.writeheader()
        for entry in selected:
            writer.writerow({"clip_id":entry["clip_id"],"meeting":entry["meeting"],"event":entry["event"],
                "start":entry["start"],"end":entry["end"],"speakers_present":','.join(entry["speakers_present"]),
                "question_before":entry["qa_before"]["question"]["text"],"answer_before":entry["qa_before"]["answer"]["text"],
                "question_after":entry["qa_after"]["question"]["text"],"answer_after":entry["qa_after"]["answer"]["text"],
                "review_status":entry["review_status"],"review_notes":""})
    if args.download_clips:
        headers = {}
        for entry in selected:
            meeting = entry["meeting"]
            url=f"{BASE}/{meeting}/audio/{meeting}.Mix-Headset.wav"
            if meeting not in headers:
                headers[meeting]=wav_header(url)
            print(f"download contiguous clip {entry['clip_id']}: {entry['duration']:.1f}s",flush=True)
            asset=download_clip(url,headers[meeting],entry["start"],entry["end"],derived/"audio"/(entry["clip_id"]+".wav"))
            entry["audio"]=asset
            write_json(derived/"selected_candidates.json",selected)
        write_json(derived/"source_audio_headers.json",headers)
    print(f"Prepared {len(selected)} automatic candidates. Human review required: {derived/'review.csv'}",flush=True)


if __name__ == "__main__":
    main()
