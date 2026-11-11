"""Invented tiny replay fixtures only; no corpus, network, Secret or model."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
import test_audit_context_revision as fixtures
D=fixtures.D;S=fixtures.S
SPEC=importlib.util.spec_from_file_location('replay_quote_binding',ROOT/'evaluation/cbt-memory-effect/replay_quote_binding.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


class ReplayTests(unittest.TestCase):
    def test_binder_loads_only_from_new_replay_checkout(self):
        module,path,digest=M._binding_module()
        self.assertEqual(path,(ROOT/'modules/agent-integration/source_quote_binding.py').resolve())
        self.assertEqual(Path(module.__file__).resolve(),path)
        self.assertEqual(digest,D.file_hash(path))

    def fixture(self,root,*,mode='wrong_id',receipts=True):
        old,pilot,proposal,targets=fixtures.AssessorTests().fixture(root)
        requests=D.read_json(pilot/'requests.json');journal=S.Journal(pilot,proposal)
        bodies={};target_map={r['case_id']:r for r in targets}
        for request in requests:
            key=request['request_sha256']
            target=target_map[request['case_id']]
            catalog=M.wire_catalog(request['context'])
            # Fixture authors select visible text; replay itself is never given this target map.
            self.assertTrue(any(row['text']==target['target_quote'] for row in catalog))
            body=json.dumps({'quote':target['target_quote'],'source_id':'fixture-wrong-id','unknown':False})
            if mode=='valid':body=json.dumps({'quote':target['target_quote'],'source_id':'s:'+target['target_message_id'],'unknown':False})
            if mode=='unknown':body=' {"quote":"","source_id":"","unknown":true} '
            if mode=='not_visible':body=json.dumps({'quote':'A private report absent from every model input.','source_id':'s:turn-0','unknown':False})
            if key not in bodies:
                if receipts:
                    journal.intent(request);journal.outcome(request,status='published',response=body,
                        usage={'total_tokens':1},provider_model='fixture-provider-only',finish_reason='stop')
                bodies[key]=body
            else:self.assertEqual(bodies[key],body)
            S.append(pilot/'answers_unscored.jsonl',S.signed({'proposal_sha256':proposal['proposal_sha256'],
                'case_id':request['case_id'],'arm':request['arm'],'request_sha256':key,'response_text':body}))
        return old,pilot,proposal

    def test_same_rule_all_nine_freezes_before_gold_and_preserves_originals(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);old,pilot,proposal=self.fixture(root)
            output=root/'replay';before=M._snapshot((old,pilot,root/'diagnostic'))
            original_read=M.diagnostic.read_rows
            observations=[]
            def checked_read(path):
                if Path(path)==old/'assessor_only/targets.jsonl':
                    self.assertTrue((output/'transformation_manifest.json').is_file())
                    frozen=D.read_json(output/'transformation_manifest.json');S.checked(frozen)
                    self.assertEqual(D.file_hash(output/'transformed_answers.jsonl'),frozen['transformed_rows_sha256'])
                    self.assertEqual(len(original_read(output/'transformed_answers.jsonl')),9)
                    observations.append('gold_after_freeze')
                return original_read(path)
            with patch.object(M.diagnostic,'read_rows',side_effect=checked_read),patch('socket.create_connection',side_effect=AssertionError('no network')):
                result=M.replay(pilot,output,fixture_only=True)
            self.assertEqual(observations,['gold_after_freeze'])
            self.assertEqual(before,M._snapshot((old,pilot,root/'diagnostic')))
            self.assertEqual(result['new_provider_calls'],0)
            self.assertTrue(result['same_rule_all_arms']);self.assertFalse(result['independent_holdout'])
            for value in result['by_arm'].values():
                self.assertEqual(value['original_correct'],0);self.assertEqual(value['replay_correct'],3)
                self.assertEqual(value['source_id_repairs'],3)
                self.assertEqual(value['original_source_binding_errors'],3)
                self.assertEqual(value['replay_source_binding_errors'],0)

    def test_valid_and_unknown_answers_are_retained_exactly(self):
        for mode in ('valid','unknown'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);old,pilot,proposal=self.fixture(root,mode=mode)
                M.replay(pilot,root/'replay',fixture_only=True)
                original=D.read_rows(pilot/'answers_unscored.jsonl')
                transformed=D.read_rows(root/'replay/transformed_answers.jsonl')
                self.assertEqual([r['response_text'] for r in original],[r['output_text'] for r in transformed])
                self.assertTrue(all(not r['changed'] for r in transformed))

    def test_unavailable_quote_cannot_be_recovered_from_database_or_target(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);old,pilot,proposal=self.fixture(root,mode='not_visible')
            report=M.replay(pilot,root/'replay',fixture_only=True)
            transformed=D.read_rows(root/'replay/transformed_answers.jsonl')
            self.assertTrue(all(r['reason']=='full_quote_not_in_visible_catalog' for r in transformed))
            self.assertTrue(all(json.loads(r['output_text'])['unknown'] for r in transformed))
            self.assertTrue(all(not r['replay_score']['literal_success'] for r in report['rows']))

    def test_signed_but_unpublished_answers_fail_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);old,pilot,proposal=self.fixture(root,receipts=False)
            with self.assertRaisesRegex(ValueError,'matching published'):M.replay(pilot,root/'replay',fixture_only=True)
            self.assertFalse((root/'replay').exists())

    def test_catalog_excludes_current_input_and_assistant_and_detects_unmapped_recent(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);old,pilot,proposal=self.fixture(root)
            context=D.read_json(pilot/'requests.json')[0]['context']
            catalog=M.wire_catalog(context)
            self.assertTrue(all(r['role']=='user' for r in catalog))
            self.assertNotIn(context['current_input_source_id'],[r['source_id'] for r in catalog])
            self.assertNotIn(context['messages'][-1]['content'],[r['text'] for r in catalog])
            modified=json.loads(json.dumps(context))
            modified['messages']=[r for r in modified['messages'] if not r['content'].startswith(M.assembly.RECENT_MAP_HEADER)]
            with self.assertRaisesRegex(ValueError,'completely and uniquely'):M.wire_catalog(modified)

    def test_source_control_change_after_transform_prevents_artifact_freeze(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);old,pilot,proposal=self.fixture(root)
            original_fence=M.sender._fence
            calls=[]
            def revoke_after_initial_fences(*args,**kwargs):
                calls.append(1)
                if len(calls)==10:raise PermissionError('fixture current source controls revoked')
                return original_fence(*args,**kwargs)
            with patch.object(M.sender,'_fence',side_effect=revoke_after_initial_fences):
                with self.assertRaisesRegex(PermissionError,'controls revoked'):
                    M.replay(pilot,root/'replay',fixture_only=True)
            self.assertEqual(len(calls),10)
            self.assertFalse((root/'replay').exists())

    def test_live_entry_requires_hosted_colab_before_reading_paths(self):
        with patch.object(M.diagnostic,'require_colab',side_effect=RuntimeError('hosted required')):
            with self.assertRaisesRegex(RuntimeError,'hosted required'):M.replay('/nonexistent','/nonexistent-out')

if __name__=='__main__':unittest.main()
