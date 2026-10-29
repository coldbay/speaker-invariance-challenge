"""Boundary control and candidate-correction tests; never model inference."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from boundary_control import PADS, pack_phase, phase_jobs, placement, vote
from moss_diagnostic_protocol import DEFAULT_CLIPS, read_pcm, sha
from run_boundary_control import report
from score_predictions import read_jsonl

ROOT=Path(__file__).resolve().parents[1]


def fake_votes(labels):
    # Synthetic inputs to the voting function only, no benchmark artifacts.
    result={}
    for pad in PADS:
        speaker=labels[pad]
        result[pad]={'audio':{'public_targets':[{'segment_id':'x','start':0,'end':1.73}],
                             'clip_offset':29.2+pad},
                     'predictions':[{'segment_id':'x','speaker':speaker,
                                      'coverage_seconds_by_registered_speaker':{speaker:1.73}}],
                     'speaker_mapping':{'S':speaker},'enrollment_overlap_weights':{'S':{speaker:6}}}
    return result


class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=ROOT/'data/ami/prepared'
        cls.rows=read_jsonl(cls.data/'inputs.jsonl')
        cls.by_id={r['clip_id']:r for r in cls.rows}
        cls.jobs=phase_jobs(cls.rows,list(DEFAULT_CLIPS))

    def test_eight_calls_and_padding_factorial(self):
        self.assertEqual(len(self.jobs),8)
        self.assertEqual({j['leading_silence_seconds'] for j in self.jobs},set(PADS))

    def test_only_leading_silence_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            base=None
            for job in [j for j in self.jobs if j['clip_id']==DEFAULT_CLIPS[0]]:
                path=Path(folder)/'input.wav'
                info=pack_phase(job,self.by_id[job['clip_id']],self.data,path)
                samples=read_pcm(path);pad=job['leading_silence_seconds'];n=pad*16000*2
                self.assertEqual(samples[:n],bytes(n))
                if base is None:
                    base=samples;base_info=info
                self.assertEqual(samples[n:],base)
                self.assertEqual(info['evaluation_pcm_sha256'],base_info['evaluation_pcm_sha256'])
                self.assertEqual(info['reference_sha256_multiset'],base_info['reference_sha256_multiset'])
                self.assertAlmostEqual(info['clip_offset']-base_info['clip_offset'],pad)
                for a,b in zip(info['enrollment_intervals'],base_info['enrollment_intervals']):
                    self.assertAlmostEqual(a['start']-b['start'],pad)
                self.assertEqual(info['frontend_chunk_count'],3 if pad<30 else 4)

    def test_boundary_geometry_and_exact_edges(self):
        self.assertTrue(placement(29.2,30.93)['crosses_30s_boundary'])
        self.assertFalse(placement(31.2,32.93)['crosses_30s_boundary'])
        self.assertTrue(placement(59.2,60.93)['crosses_30s_boundary'])
        self.assertFalse(placement(61.2,62.93)['crosses_30s_boundary'])
        self.assertFalse(placement(29,30)['safe'])
        self.assertFalse(placement(30,31)['safe'])

    def test_guard_uses_safe_agreement_in_both_time_groups(self):
        raw=fake_votes({0:'B',2:'A',30:'B',32:'A'})
        self.assertEqual(vote(raw,'x',False)['speaker'],'B')
        corrected=vote(raw,'x',True)
        self.assertEqual(corrected['speaker'],'A')
        self.assertEqual(sum(i['eligible'] for i in corrected['votes']),2)

    def test_guard_preserves_on_conflict_or_weak_registration(self):
        raw=fake_votes({0:'B',2:'A',30:'B',32:'B'})
        self.assertEqual(vote(raw,'x',True)['speaker'],'B')
        raw=fake_votes({0:'B',2:'A',30:'B',32:'A'})
        raw[32]['enrollment_overlap_weights']['S']['A']=.5
        self.assertEqual(vote(raw,'x',True)['speaker'],'B')

    def test_report_pipeline_with_synthetic_temporary_results(self):
        # All synthesized predictions are explicitly marked and deleted with
        # the temporary directory. They are not real experimental results.
        truth={r['clip_id']:{u['segment_id']:u['speaker'] for u in r['utterances']}
               for r in read_jsonl(self.data/'labels.jsonl')}
        plan={'clip_ids':list(DEFAULT_CLIPS),'jobs':self.jobs,'expected_packages':{'torch':'unit-test'},
              'profile':{'unit_test':True}}
        plan_hash=sha(json.dumps(plan,sort_keys=True).encode())
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)
            for job in self.jobs:
                pad=job['leading_silence_seconds'];cid=job['clip_id']
                audio=pack_phase(job,self.by_id[cid],self.data,run/'.scratch/input.wav')
                predicted=[]
                for t in audio['public_targets']:
                    speaker=truth[cid][t['segment_id']]
                    if cid==DEFAULT_CLIPS[0] and t['segment_id']=='segment_001' and pad in (0,30):
                        speaker='B'
                    predicted.append({'segment_id':t['segment_id'],'speaker':speaker,
                                      'coverage_seconds_by_registered_speaker':{speaker:t['end']-t['start']}})
                mapping={s:s for s in job['reference_order']}
                raw={'completed':True,'job':job,'plan_sha256':plan_hash,
                     'runtime':{'packages':plan['expected_packages'],'profile':plan['profile']},
                     'text':'SYNTHETIC UNIT TEST ONLY','audio':audio,'predictions':predicted,
                     'speaker_mapping':mapping,'enrollment_overlap_weights':{s:{s:6} for s in mapping},
                     'inference_seconds':0,'reached_token_limit':False}
                path=run/'jobs'/job['job_id']/'raw_output.json';path.parent.mkdir(parents=True)
                path.write_text(json.dumps(raw))
            with contextlib.redirect_stdout(io.StringIO()):
                report(run,self.data,plan)
            result=json.loads((run/'summary.json').read_text())['summary']
            self.assertEqual(result['pad_00']['correct'],23)
            self.assertEqual(result['boundary_guard']['correct'],24)
            self.assertEqual(result['boundary_guard']['new_errors'],0)
            with zipfile.ZipFile(run/'results_only.zip') as archive:
                self.assertIsNone(archive.testzip())
                self.assertTrue(all(not n.endswith('.wav') for n in archive.namelist()))


if __name__=='__main__':
    unittest.main()
