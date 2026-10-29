"""Diagnostic protocol tests; no models are loaded and no GPU is used."""
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from diarization_projection import map_enrollment, project_predictions
from moss_diagnostic_protocol import CONDITIONS, DEFAULT_CLIPS, RATE, make_jobs, pack_job, read_pcm, sha
from report_moss_diagnostics import combine_condition
from report_moss_diagnostics import main as report_main
from score_predictions import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = ROOT / 'data/ami/prepared'
        cls.rows = read_jsonl(cls.data / 'inputs.jsonl')
        cls.by_id = {r['clip_id']: r for r in cls.rows}
        cls.jobs = make_jobs(cls.rows, list(DEFAULT_CLIPS))

    def job(self, condition, target=None):
        return next(j for j in self.jobs if j['clip_id'] == DEFAULT_CLIPS[0]
                    and j['condition'] == condition and (target is None or j['target_segment_ids'] == [target]))

    def test_fixed_plan_and_scored_target_coverage(self):
        self.assertEqual(len(self.jobs), 78)
        self.assertEqual(sum(j['condition'] == 'isolated_forward' for j in self.jobs), 37)
        truth = read_jsonl(self.data / 'labels.jsonl')
        eligible = {(r['clip_id'], u['segment_id']) for r in truth if r['clip_id'] in DEFAULT_CLIPS
                    for u in r['utterances'] if u['score_eligible']}
        self.assertEqual(len(eligible), 24)
        for condition in CONDITIONS:
            targets = [(j['clip_id'], sid) for j in self.jobs if j['condition'] == condition for sid in j['target_segment_ids']]
            self.assertEqual(len(targets), len(set(targets)))
            self.assertTrue(eligible.issubset(targets))

    def test_full_forward_reproduces_actual_server_input_bytes(self):
        baseline = ROOT / 'runs/speaker_moss_pilot_01'
        cases = {json.loads(p.read_text())['clip_id']: p.parent / 'model_input.wav'
                 for p in baseline.glob('case_*/raw_output.json')}
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'input.wav'
            for job in (j for j in self.jobs if j['condition'] == 'full_forward'):
                info = pack_job(job, self.by_id[job['clip_id']], self.data, target)
                self.assertEqual(info['input_wav_sha256'], sha(cases[job['clip_id']].read_bytes()))

    def test_reverse_preserves_reference_multiset_and_conversation(self):
        row = self.by_id[DEFAULT_CLIPS[0]]
        with tempfile.TemporaryDirectory() as folder:
            first = pack_job(self.job('full_forward'), row, self.data, Path(folder) / 'a.wav')
            later = pack_job(self.job('full_reverse'), row, self.data, Path(folder) / 'b.wav')
        self.assertEqual(first['reference_sha256_multiset'], later['reference_sha256_multiset'])
        self.assertEqual(first['clip_offset'], later['clip_offset'])
        self.assertEqual(first['evaluation_pcm_sha256'], later['evaluation_pcm_sha256'])
        self.assertNotEqual(first['input_pcm_sha256'], later['input_pcm_sha256'])

    def test_isolated_target_is_exact_original_pcm(self):
        row = self.by_id[DEFAULT_CLIPS[0]]
        target = next(s for s in row['segments'] if s['segment_id'] == 'segment_006')
        source = read_pcm(self.data / row['audio'])
        start, end = math.floor(target['start'] * RATE), math.ceil(target['end'] * RATE)
        with tempfile.TemporaryDirectory() as folder:
            a = pack_job(self.job('isolated_forward', 'segment_006'), row, self.data, Path(folder) / 'a.wav')
            b = pack_job(self.job('isolated_reverse', 'segment_006'), row, self.data, Path(folder) / 'b.wav')
        self.assertEqual(a['evaluation_pcm_sha256'], sha(source[start*2:end*2]))
        self.assertEqual(a['evaluation_pcm_sha256'], b['evaluation_pcm_sha256'])
        self.assertAlmostEqual(a['public_targets'][0]['start'], target['start'] - start / RATE)

    def test_anonymous_label_renumbering_is_not_identity_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            for condition in ('isolated_forward', 'isolated_reverse'):
                row = self.by_id[DEFAULT_CLIPS[0]]
                job = self.job(condition, 'segment_006')
                audio = pack_job(job, row, self.data, Path(folder) / 'input.wav')
                ids = {agent: f'S{i+1:02d}' for i, agent in enumerate(job['reference_order'])}
                segments = [dict(r, speaker=ids[r['speaker']]) for r in audio['enrollment_intervals']]
                t = audio['public_targets'][0]
                segments.append({'start': audio['clip_offset'] + t['start'],
                                 'end': audio['clip_offset'] + t['end'], 'speaker': ids['A']})
                mapping, _ = map_enrollment(segments, audio['enrollment_intervals'])
                prediction = project_predictions(segments, mapping, [t], audio['clip_offset'])[0]
                self.assertEqual(prediction['speaker'], 'A')

    def test_reporting_rejects_incomplete_or_duplicate_results(self):
        job = self.job('full_forward')
        plan = {'clip_ids': [job['clip_id']], 'jobs': [job], 'profile': {'seed': 0},
                'expected_packages': {'torch': 'test'}, 'expected_torch_cuda': 'test'}
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            with self.assertRaises(ValueError):
                combine_condition(plan, run, 'full_forward')
            path = run / 'jobs' / job['job_id'] / 'raw_output.json'
            path.parent.mkdir(parents=True)
            raw = {'completed': True, 'job': job,
                   'plan_sha256': sha(json.dumps(plan, sort_keys=True).encode()),
                   'runtime': {'profile': plan['profile'], 'packages': plan['expected_packages'], 'torch_cuda': 'test'},
                   'predictions': [{'segment_id': sid, 'speaker': 'A'} for sid in job['target_segment_ids']]}
            path.write_text(json.dumps(raw))
            combined, _ = combine_condition(plan, run, 'full_forward')
            self.assertEqual(len(combined[0]['predictions']), len(job['target_segment_ids']))
            raw['predictions'].append(raw['predictions'][0])
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                combine_condition(plan, run, 'full_forward')

    def test_end_to_end_report_counts_new_errors_and_excludes_audio(self):
        # Synthetic fixtures exist only in a TemporaryDirectory. These are
        # scorer tests, never model predictions or reported benchmark scores.
        baseline = ROOT / 'runs/speaker_moss_pilot_01'
        manifest_file = baseline / 'run_manifest.json'
        manifest = json.loads(manifest_file.read_text())
        plan = {'clip_ids': list(DEFAULT_CLIPS), 'jobs': self.jobs,
                'inputs_sha256': sha((self.data / 'inputs.jsonl').read_bytes()),
                'baseline_manifest_sha256': sha(manifest_file.read_bytes()),
                'profile': {'seed': 0}, 'expected_packages': manifest['packages'],
                'expected_torch_cuda': manifest['torch_cuda']}
        plan_hash = sha(json.dumps(plan, sort_keys=True).encode())
        truths = {r['clip_id']: {u['segment_id']: u['speaker'] for u in r['utterances']}
                  for r in read_jsonl(self.data / 'labels.jsonl')}
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / 'plan.json').write_text(json.dumps(plan))
            for job in self.jobs:
                audio = pack_job(job, self.by_id[job['clip_id']], self.data, run / '.scratch/input.wav')
                predictions = [{'segment_id': sid, 'speaker': truths[job['clip_id']][sid]} for sid in job['target_segment_ids']]
                if (job['condition'] == 'isolated_forward' and job['clip_id'] == DEFAULT_CLIPS[0]
                        and job['target_segment_ids'] == ['segment_001']):
                    predictions[0]['speaker'] = 'B'
                raw = {'completed': True, 'job': job, 'plan_sha256': plan_hash,
                       'runtime': {'profile': plan['profile'], 'packages': plan['expected_packages'], 'torch_cuda': manifest['torch_cuda']},
                       'audio': audio, 'text': 'SYNTHETIC UNIT TEST ONLY', 'segments': [],
                       'speaker_mapping': {f'S{i}': agent for i, agent in enumerate(job['reference_order'])},
                       'enrollment_overlap_weights': {}, 'predictions': predictions,
                       'reached_token_limit': False, 'inference_seconds': 0}
                path = run / 'jobs' / job['job_id'] / 'raw_output.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(raw))
            argv = ['report', '--run', str(run), '--data', str(self.data), '--baseline', str(baseline)]
            with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()):
                report_main()
            summary = json.loads((run / 'diagnostic_summary.json').read_text())['summary']
            self.assertEqual(summary['full_forward']['correct'], 24)
            self.assertEqual(summary['isolated_forward']['correct'], 23)
            self.assertEqual(summary['isolated_forward']['changes_vs_full_forward']['new_error_count'], 1)
            self.assertTrue((run / 'REPORT.md').exists())
            import zipfile
            with zipfile.ZipFile(run / 'results_only.zip') as archive:
                self.assertTrue(all(not name.endswith('.wav') for name in archive.namelist()))


if __name__ == '__main__':
    unittest.main()
