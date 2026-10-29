"""Mark paired acts by the same physical speaker, without using predictions."""


def add_focus_pairs(row, public_segments):
    times = {s['segment_id']: s for s in public_segments}
    acts = {u['source_dialogue_act']: u for u in row['utterances']}
    slots = {}
    for phase in ('before', 'after'):
        for role in ('question', 'answer'):
            act = row['qa_' + phase][role]
            utterance = acts[act['id']]
            utterance['focus_slot'] = phase + '_' + role
            slots[(phase, role)] = utterance
    for utterance in row['utterances']:
        interval = times[utterance['segment_id']]
        utterance.update(start=interval['start'], end=interval['end'],
                         duration=interval['end'] - interval['start'])
    reversal = row['event'] == 'question_answer_reversal'
    pairs = []
    for role in ('question', 'answer'):
        later_role = ('answer' if role == 'question' else 'question') if reversal else role
        first, later = slots[('before', role)], slots[('after', later_role)]
        if first['speaker'] != later['speaker']:
            raise ValueError(f'Same-person pair mismatched: {row["clip_id"]}/{role}')
        pairs.append({'pair_id': role + '_to_' + later_role,
                      'transition': role + '_to_' + later_role,
                      'speaker': first['speaker'],
                      'before_segment_id': first['segment_id'],
                      'after_segment_id': later['segment_id'],
                      'score_eligible': first['score_eligible'] and later['score_eligible']})
    row['same_speaker_pairs'] = pairs
    row['protocol_version'] = 'natural-qa-v2-same-person'
    return row
