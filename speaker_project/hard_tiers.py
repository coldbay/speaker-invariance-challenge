"""Hard-track tiers from annotation properties only (stdlib, no model code).

  overlap_light  0.8-15 s, other-speaker overlap (5%, 30%]
  overlap_heavy  0.8-15 s, overlap > 30%
  short_clean    0.3-0.8 s, overlap <= 5%
  short_overlap  0.3-0.8 s, overlap > 5%
Main-track (score_eligible) and clip-edge truncated utterances are excluded.
"""

TIERS = ('overlap_light', 'overlap_heavy', 'short_clean', 'short_overlap')


def tier(u):
    duration, share = u['end']-u['start'], u['other_speaker_word_overlap_fraction']
    if u['truncated'] or u['score_eligible']:
        return None
    if .8 <= duration <= 15 and share > .05:
        return 'overlap_light' if share <= .3 else 'overlap_heavy'
    if .3 <= duration < .8:
        return 'short_clean' if share <= .05 else 'short_overlap'
    return None
