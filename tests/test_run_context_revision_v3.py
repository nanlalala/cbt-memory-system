"""Tiny invented checkpoints and fake transport only; never provider/Secret access."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
import test_prepare_context_revision_v3 as fixtures
D=fixtures.M
SPEC=importlib.util.spec_from_file_location('run_context_revision_v3',ROOT/'evaluation/cbt-memory-effect/run_context_revision_v3.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)

class AnswerPilotTests(unittest.TestCase):
    def setup_proposal(self,root):
        old=fixtures.CheckpointDryRunTests().fixture(root)
        old_protocol=D.read_json(old/'protocol.json')
        old_protocol['settings']['accepted_provider_model']='fixture-provider-only'
        old_protocol['protocol_sha256']=D.digest({k:v for k,v in old_protocol.items() if k!='protocol_sha256'})
        (old/'protocol.json').write_text(json.dumps(old_protocol))
        anchor=old/'cases/probe-0/prepared_contexts.json'
        envelope=D.read_json(anchor);envelope['protocol_sha256']=old_protocol['protocol_sha256'];anchor.write_text(json.dumps(envelope))
        diag=root/'diagnostic';D.prepare(old,diag,product_repo=ROOT,fixture_only=True)
        (old/'summary.json').write_text(json.dumps({'cost':{'actual_provider_attempts':0},'fixture_only':True}))
        p=M.prepare_proposal(diag,old,root/'pilot',product_repo=ROOT,fixture_only=True)
        auth=root/'auth.json'
        auth.write_text(json.dumps({'user_authorized':True,'proposal_sha256':p['proposal_sha256'],
            'maximum_new_provider_calls':9,'authorization_source':'unit fixture only'}))
        return old,diag,root/'pilot',p,auth

    def test_proposal_freezes_nine_payloads_and_does_not_read_gold(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);old,diag,pilot,p,auth=self.setup_proposal(root)
            self.assertEqual(p['maximum_new_provider_calls'],9)
            self.assertEqual(len(D.read_json(pilot/'requests.json')),9)
            self.assertFalse(p['model_generation_uses_gold'])
            self.assertFalse((pilot/'call_intents.jsonl').exists())

    def test_absent_or_wrong_authorization_blocks_before_transport(self):
        with tempfile.TemporaryDirectory() as folder:
            old,diag,pilot,p,auth=self.setup_proposal(Path(folder))
            auth.write_text(json.dumps({'user_authorized':True,'proposal_sha256':'other','maximum_new_provider_calls':9}))
            with self.assertRaises(PermissionError):M.run(pilot,auth,fixture_only=True,transport=lambda _:self.fail('must not send'))
            self.assertFalse((pilot/'call_intents.jsonl').exists())

    def test_complete_fake_pilot_is_finite_and_preserves_original_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);old,diag,pilot,p,auth=self.setup_proposal(root)
            before={str(path):D.file_hash(path) for path in old.rglob('*') if path.is_file()}
            calls=[]
            def fake(payload):
                calls.append(payload)
                return {'hypothesis':'Fixture answer only.','usage':{'prompt_tokens':10,'completion_tokens':2,'total_tokens':12},'latency_seconds':0.01,'provider_model':'fixture-provider-only','finish_reason':'stop'}
            result=M.run(pilot,auth,fixture_only=True,transport=fake)
            self.assertLessEqual(len(calls),9)
            self.assertEqual(len(D.read_rows(pilot/'answers_unscored.jsonl')),9)
            self.assertEqual(result['total_provider_attempts'],len(calls))
            answers=D.read_rows(pilot/'answers_unscored.jsonl')
            self.assertTrue(any(row['canonical_response_shared'] for row in answers))
            self.assertTrue(all(row['provider_model']=='fixture-provider-only' and row['finish_reason']=='stop'
                                for row in D.read_rows(pilot/'call_outcomes.jsonl')))
            self.assertEqual(before,{str(path):D.file_hash(path) for path in old.rglob('*') if path.is_file()})

    def test_wrong_provider_identity_is_rejected_in_fixture_too(self):
        with tempfile.TemporaryDirectory() as folder:
            old,diag,pilot,p,auth=self.setup_proposal(Path(folder))
            def wrong(payload):
                return {'hypothesis':'Wrong provider body.','usage':{'total_tokens':12},
                        'provider_model':'wrong-fixture-provider','finish_reason':'stop'}
            with self.assertRaises(ValueError):M.run(pilot,auth,fixture_only=True,transport=wrong)
            self.assertFalse((pilot/'answers_unscored.jsonl').exists())
            outcome=D.read_rows(pilot/'call_outcomes.jsonl')[0]
            self.assertEqual(outcome['status'],'response_rejected')
            self.assertEqual(outcome['usage']['total_tokens'],12)
            self.assertIsNone(outcome['response_text'])

    def test_existing_new_store_is_rechecked_against_frozen_source_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            old,diag,pilot,p,auth=self.setup_proposal(Path(folder))
            data=M._load_inputs(diag,old,ROOT)
            request=D.read_json(pilot/'requests.json')[0]
            case=data[4][request['case_id']]
            sources=M._fence(data,p,case,request['context'])
            store,user,run_id,session_id,generation=M._new_store(pilot,p,case,sources,request['context'],data[-1])
            with store.store.transaction(immediate=True):
                store.connection.execute('UPDATE dm_sources SET text=? WHERE user_id=? AND source_id=?',
                    ('Changed content after preparation.',user,sources[0]['source_id']))
            store.close()
            with self.assertRaises(PermissionError):
                M.run(pilot,auth,fixture_only=True,transport=lambda _:self.fail('changed store must not send'))
            self.assertFalse((pilot/'call_intents.jsonl').exists())

    def test_unknown_send_is_journalled_and_not_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            old,diag,pilot,p,auth=self.setup_proposal(Path(folder))
            calls=[]
            def fail(payload): calls.append(payload);raise TimeoutError('fixture transport unknown')
            with self.assertRaises(TimeoutError):M.run(pilot,auth,fixture_only=True,transport=fail)
            self.assertEqual(len(calls),1)
            with self.assertRaises(M.UnknownAttempt):M.run(pilot,auth,fixture_only=True,transport=fail)
            self.assertEqual(len(calls),1)
            outcome=D.read_rows(pilot/'call_outcomes.jsonl')[0]
            self.assertEqual(outcome['status'],'attempt_unknown')
            self.assertIsNone(outcome['response_text'])

    def test_old_source_control_change_after_provider_prevents_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);old,diag,pilot,p,auth=self.setup_proposal(root)
            def mutate(payload):
                # Stand in for an independently committed withdrawal by changing
                # one frozen dependency: the publication fence must reject it.
                (old/'generation/cases.jsonl').write_text('{}\n')
                return {'hypothesis':'Must not be published.','usage':{'total_tokens':12},'latency_seconds':0.25,'provider_model':'fixture-provider-only','finish_reason':'stop'}
            with self.assertRaises(ValueError):M.run(pilot,auth,fixture_only=True,transport=mutate)
            self.assertFalse((pilot/'answers_unscored.jsonl').exists())
            outcome=D.read_rows(pilot/'call_outcomes.jsonl')[0]
            self.assertIsNone(outcome['response_text'])
            self.assertEqual(outcome['status'],'stale_discarded')
            self.assertEqual(outcome['usage']['total_tokens'],12)
            self.assertEqual(outcome['api_latency_seconds'],0.25)
            self.assertEqual(outcome['provider_model'],'fixture-provider-only')

if __name__=='__main__':unittest.main()
