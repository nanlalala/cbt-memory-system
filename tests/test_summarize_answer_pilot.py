"""Tiny fake-pilot fixtures only. No real corpus, provider, Secret or model."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tests'))
import test_audit_context_revision as fixtures
SPEC=importlib.util.spec_from_file_location('summarize_answer_pilot',ROOT/'evaluation/cbt-memory-effect/summarize_answer_pilot.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)
D=M.D;S=M.S

class SummaryTests(unittest.TestCase):
    def fixture(self,root):
        old,pilot,proposal,targets=fixtures.AssessorTests().fixture(root)
        requests=D.read_json(pilot/'requests.json');journal=S.Journal(pilot,proposal)
        for request in requests:
            target=next(t for t in targets if t['case_id']==request['case_id'])
            body=json.dumps({'quote':target['target_quote'],'source_id':'s:'+target['target_message_id'],'unknown':False})
            if journal.intent(request) is None:
                journal.outcome(request,status='published',response=body,usage={'prompt_tokens':9,'completion_tokens':3,'total_tokens':12},
                    latency=0.3,observed=0.6,provider_model='fixture-provider-only',finish_reason='stop')
            S.append(pilot/'answers_unscored.jsonl',S.signed({'proposal_sha256':proposal['proposal_sha256'],'case_id':request['case_id'],
                'arm':request['arm'],'request_sha256':request['request_sha256'],'response_text':body}))
        M.assessor.assess(pilot,root/'assessed',fixture_only=True)
        return old,pilot,proposal

    def test_shared_receipts_count_once_and_allocations_reconcile(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);old,pilot,proposal=self.fixture(root)
            result=M.summarize(pilot,root/'assessed',root/'summary',fixture_only=True)
            pairs=result['paired_raw_vs_managed']['counts']
            self.assertEqual(pairs,{'tie_both_correct':3})
            cost=result['new_answer_cost']
            self.assertEqual(cost['unique_provider_attempts'],3)
            self.assertEqual(cost['shared_attempts'],3)
            self.assertEqual(cost['observed_tokens']['total_tokens'],36)
            self.assertAlmostEqual(sum(a['allocated_observed_tokens']['total_tokens'] for a in cost['arm_allocations'].values()),36)
            self.assertTrue(all(cost['allocation_reconciles'].values()))
            self.assertIsNone(cost['monetary_cost']);self.assertEqual(result['new_provider_calls_this_summary'],0)

    def test_unknown_usage_is_null_and_not_zero(self):
        requests=[{'arm':arm,'request_sha256':'shared'} for arm in M.ARMS]
        cost=M.aggregate_cost(requests,{'shared':{}},{})
        self.assertIsNone(cost['observed_tokens']['total_tokens'])
        self.assertEqual(cost['missing_usage_attempts']['total_tokens'],1)
        self.assertTrue(all(cost['allocation_reconciles'].values()))
        self.assertTrue(all(a['allocated_observed_tokens']['total_tokens'] is None for a in cost['arm_allocations'].values()))

    def test_tampered_assessment_request_binding_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);old,pilot,proposal=self.fixture(root)
            path=root/'assessed/audit.json';audit=D.read_json(path)
            audit['rows'][0]['request_sha256']='wrong-request';path.write_text(json.dumps(audit))
            with self.assertRaises(ValueError):M.summarize(pilot,root/'assessed',root/'summary',fixture_only=True)
            self.assertFalse((root/'summary').exists())

if __name__=='__main__':unittest.main()
