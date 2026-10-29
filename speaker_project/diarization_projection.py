"""Map anonymous diarization IDs using enrollment time only, never test labels."""
from collections import defaultdict
import math


def overlap(a, b, c, d):
    return max(0., min(b, d) - max(a, c))


def validate_segments(segments, total_seconds):
    result = []
    for s in segments:
        start, end = float(s['start']), float(s['end'])
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        if end <= 0 or start >= total_seconds:
            continue
        result.append(dict(s, start=max(0., start), end=min(total_seconds, end)))
    return result


def map_enrollment(segments, enrollment_intervals):
    agents = sorted({r['speaker'] for r in enrollment_intervals})
    weights = defaultdict(lambda: defaultdict(float))
    for s in segments:
        for r in enrollment_intervals:
            weights[str(s['speaker'])][r['speaker']] += overlap(s['start'], s['end'], r['start'], r['end'])
    # Maximum positive-overlap one-to-one assignment. Only enrollment intervals
    # contribute. Missing/fragmented enrollment identities remain unsupported.
    states = {0: (0., {})}
    for anonymous in sorted(weights):
        following = dict(states)
        for mask, (value, mapping) in states.items():
            for index, agent in enumerate(agents):
                weight = weights[anonymous][agent]
                if mask & (1 << index) or weight <= 0:
                    continue
                new_mask = mask | (1 << index)
                candidate = value + weight
                if new_mask not in following or candidate > following[new_mask][0]:
                    following[new_mask] = (candidate, dict(mapping, **{anonymous: agent}))
        states = following
    best = max(states.values(), key=lambda item: item[0])
    return best[1], {key: dict(value) for key, value in weights.items()}


def project_predictions(segments, mapping, public_segments, clip_offset):
    predictions = []
    for target in public_segments:
        left, right = target['start'] + clip_offset, target['end'] + clip_offset
        coverage = defaultdict(float)
        for s in segments:
            if str(s['speaker']) in mapping:
                coverage[mapping[str(s['speaker'])]] += overlap(left, right, s['start'], s['end'])
        positive = {key: value for key, value in coverage.items() if value > 0}
        ranked = sorted(positive, key=lambda key: (-positive[key], key))
        tied = len(ranked) > 1 and abs(positive[ranked[0]] - positive[ranked[1]]) < 1e-9
        guessed = ranked[0] if ranked and not tied else None
        predictions.append({'segment_id': target['segment_id'], 'speaker': guessed,
                            'coverage_seconds_by_registered_speaker': positive,
                            'reason': 'equal_overlap_tie' if tied else 'no_mapped_coverage' if not ranked else 'max_time_overlap'})
    return predictions
