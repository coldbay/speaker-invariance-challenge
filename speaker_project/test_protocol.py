"""Guard against misleading same-person metrics and malformed predictions."""
import copy
import unittest
from pathlib import Path
from focus_protocol import add_focus_pairs
from score_predictions import evaluate, read_jsonl
from diarization_projection import map_enrollment, project_predictions
from run_moss import pack_audio
import hashlib
import tempfile
import wave


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1] / 'data/ami/prepared'
        cls.labels = read_jsonl(root / 'labels.jsonl')
        cls.inputs = {r['clip_id']: r for r in read_jsonl(root / 'inputs.jsonl')}

    def perfect(self):
        return [{'clip_id': r['clip_id'], 'predictions': [
            {'segment_id': u['segment_id'], 'speaker': u['speaker']} for u in r['utterances']]} for r in self.labels]

    def test_pair_identity_and_public_times(self):
        for row in self.labels:
            fresh = add_focus_pairs(copy.deepcopy(row), self.inputs[row['clip_id']]['segments'])
            self.assertEqual(row['same_speaker_pairs'], fresh['same_speaker_pairs'])
            by_id = {u['segment_id']: u for u in row['utterances']}
            for pair in row['same_speaker_pairs']:
                first, later = by_id[pair['before_segment_id']], by_id[pair['after_segment_id']]
                self.assertEqual(first['speaker'], later['speaker'])
                self.assertLess(first['start'], later['start'])
                self.assertEqual(pair['score_eligible'], first['score_eligible'] and later['score_eligible'])

    def test_perfect_and_missing(self):
        perfect = evaluate(self.labels, self.perfect())
        self.assertEqual(perfect['scored_segments'], 70)
        self.assertTrue(all(x['both_correct_rate'] == 1 for x in perfect['same_person_summary']))
        empty = evaluate(self.labels, [])
        self.assertTrue(all(x['both_correct_rate'] == 0 for x in empty['same_person_summary']))
        self.assertTrue(all(x['correct_retention'] is None for x in empty['same_person_summary']))

    def test_stable_wrong_is_not_success(self):
        predictions = self.perfect()
        for row in predictions:
            for item in row['predictions']:
                item['speaker'] = 'INVALID_SPEAKER'
        result = evaluate(self.labels, predictions)
        self.assertTrue(all(x['both_correct_rate'] == 0 for x in result['same_person_summary']))
        self.assertTrue(all(x['inconsistency_rate'] == 0 for x in result['same_person_summary']))

    def test_bad_identifiers_rejected(self):
        bad = self.perfect()
        bad[0]['predictions'].append({'segment_id': 'typo', 'speaker': 'B'})
        with self.assertRaises(ValueError):
            evaluate(self.labels, bad)
        bad = self.perfect()
        bad[0]['predictions'].append(bad[0]['predictions'][0])
        with self.assertRaises(ValueError):
            evaluate(self.labels, bad)

    def test_anonymous_permutation_uses_enrollment_only(self):
        references = [{'speaker': 'A', 'start': 0, 'end': 3},
                      {'speaker': 'B', 'start': 4, 'end': 7}]
        generated = [{'speaker': 'S99', 'start': 0, 'end': 3},
                     {'speaker': 'S12', 'start': 4, 'end': 7},
                     {'speaker': 'S99', 'start': 10, 'end': 12}]
        mapping, _ = map_enrollment(generated, references)
        self.assertEqual(mapping, {'S99': 'A', 'S12': 'B'})
        public = [{'segment_id': 'x', 'start': 0, 'end': 2}]
        self.assertEqual(project_predictions(generated, mapping, public, 10)[0]['speaker'], 'A')
        # A new, unsupported anonymous identity must not borrow a test gold label.
        generated[-1]['speaker'] = 'S03'
        self.assertIsNone(project_predictions(generated, mapping, public, 10)[0]['speaker'])

    def test_contiguous_conversation_is_preserved_in_prefix_adapter(self):
        root = Path(__file__).resolve().parents[1] / 'data/ami/prepared'
        row = next(iter(self.inputs.values()))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'input.wav'
            intervals, offset, seconds = pack_audio(row, root, path, .4)
            with wave.open(str(path)) as audio:
                audio.setpos(round(offset * 16000))
                suffix = audio.readframes(audio.getnframes())
            with wave.open(str(root / row['audio'])) as original:
                expected = original.readframes(original.getnframes())
            self.assertEqual(hashlib.sha256(suffix).digest(), hashlib.sha256(expected).digest())
            self.assertEqual(len(intervals), 4)
            self.assertGreater(seconds, offset)


if __name__ == '__main__':
    unittest.main()
