"""Tiny invented assessor fixtures: no real data, network, Secret or provider."""
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
SPEC=importlib.util.spec_from_file_location('audit_context_revision',ROOT/'evaluation/cbt-memory-effect/audit_context_revision.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)
S=M.sender

class AssessorTests(unittest.TestCase):
    def fixture(self,root):
        old=fixtures.CheckpointDryRunTests().fixture(root)
        cases=D.read_rows(old/'generation/cases.jsonl')
        targets=[]
        for index,case in enumerate(cases):
            turn=case['history'][0 if index==0 else 2]
            targets.append({'case_id':case['case_id'],'case_sha256':case['case_sha256'],
                'source_family_id':case['source_family_id'],'target_message_id':turn['message_id'],
                'target_quote':turn['content'],'position':['early','middle','recent'][index]})
        assessor=old/'assessor_only';assessor.mkdir()
        (assessor/'targets.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in targets))
        protocol=D.read_json(old/'protocol.json')
        protocol['settings']['accepted_provider_model']='fixture-provider-only'
        protocol['files_sha256']['assessor_only/targets.jsonl']=D.file_hash(assessor/'targets.jsonl')
        relative='evaluation/cbt-memory-effect/run_cactus_effect.py'
        protocol['code_sha256'][relative]=D.file_hash(ROOT/relative)
        protocol['protocol_sha256']=D.digest({k:v for k,v in protocol.items() if k!='protocol_sha256'})
        (old/'protocol.json').write_text(json.dumps(protocol))
        anchor=old/'cases/probe-0/prepared_contexts.json';prepared=D.read_json(anchor)
        prepared['protocol_sha256']=protocol['protocol_sha256'];anchor.write_text(json.dumps(prepared))
        (old/'summary.json').write_text(json.dumps({'fixture_only':True,'cost':{'actual_provider_attempts':0}}))
        diag=root/'diagnostic';D.prepare(old,diag,product_repo=ROOT,fixture_only=True)
        proposal=S.prepare_proposal(diag,old,root/'pilot',product_repo=ROOT,fixture_only=True)
        return old,root/'pilot',proposal,targets

    def test_no_answers_are_not_generated_and_never_accuracy(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);old,pilot,proposal,targets=self.fixture(root)
            before={str(p):D.file_hash(p) for p in old.rglob('*') if p.is_file()}
            with patch('socket.create_connection',side_effect=AssertionError('no network')):
                report=M.assess(pilot,root/'audit',fixture_only=True)
            self.assertEqual(report['planned_rows'],9)
            self.assertEqual(report['new_provider_calls'],0)
            self.assertTrue(all(row['answer_status']=='not_generated' and row['answer_accuracy'] is None for row in report['cases']))
            self.assertTrue(all(value['answer_accuracy'] is None for value in report['by_arm'].values()))
            full=(root/'audit/audit.json').read_text()
            self.assertTrue(all(target['target_quote'] not in full for target in targets))
            self.assertEqual(before,{str(p):D.file_hash(p) for p in old.rglob('*') if p.is_file()})

    def test_gold_checksum_mismatch_blocks_assessment(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);old,pilot,proposal,targets=self.fixture(root)
            (old/'assessor_only/targets.jsonl').write_text('{}\n')
            with self.assertRaises(ValueError):M.assess(pilot,root/'audit',fixture_only=True)
            self.assertFalse((root/'audit').exists())

    def test_frozen_scorer_and_published_answer_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);old,pilot,proposal,targets=self.fixture(root)
            request=D.read_json(pilot/'requests.json')[0]
            target=targets[0];sid='s:'+target['target_message_id']
            body=json.dumps({'quote':target['target_quote'],'source_id':sid,'unknown':False})
            journal=S.Journal(pilot,proposal);journal.intent(request)
            journal.outcome(request,status='published',response=body,usage={'total_tokens':5},
                provider_model='fixture-provider-only',finish_reason='stop')
            S.append(pilot/'answers_unscored.jsonl',S.signed({'proposal_sha256':proposal['proposal_sha256'],
                'case_id':request['case_id'],'arm':request['arm'],'request_sha256':request['request_sha256'],
                'response_text':body}))
            report=M.assess(pilot,root/'audit',fixture_only=True)
            row=report['cases'][0]
            self.assertEqual(row['answer_status'],'correct');self.assertTrue(row['answer_accuracy'])
            self.assertTrue(all(r['answer_status']=='not_generated' for r in report['cases'][1:]))
            scorer,_,_=M.frozen_scorer(ROOT,D.read_json(old/'protocol.json'))
            malformed=json.dumps({'quote':target['target_quote'],'source_id':sid,'unknown':False,'extra':'not allowed'})
            self.assertEqual(scorer(malformed,target,sid)['status'],'technical_failure')

    def test_unknown_provider_receipt_does_not_become_an_answer_or_accuracy(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);old,pilot,proposal,targets=self.fixture(root)
            request=D.read_json(pilot/'requests.json')[0]
            journal=S.Journal(pilot,proposal);journal.intent(request)
            journal.outcome(request,status='attempt_unknown',usage=None,failure='fixture timeout')
            report=M.assess(pilot,root/'audit',fixture_only=True)
            self.assertEqual(report['provider_attempts'],1)
            self.assertEqual(report['provider_status_counts']['attempt_unknown'],1)
            self.assertTrue(all(row['answer_status']=='not_generated' and row['answer_accuracy'] is None for row in report['cases']))
            self.assertEqual(report['cases'][0]['provider_status'],'attempt_unknown')

    def test_unpublished_answer_cannot_be_scored(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);old,pilot,proposal,targets=self.fixture(root)
            request=D.read_json(pilot/'requests.json')[0]
            S.append(pilot/'answers_unscored.jsonl',S.signed({'proposal_sha256':proposal['proposal_sha256'],
                'case_id':request['case_id'],'arm':request['arm'],'request_sha256':request['request_sha256'],
                'response_text':'fabricated fixture answer without provider receipt'}))
            with self.assertRaises(ValueError):M.assess(pilot,root/'audit',fixture_only=True)

if __name__=='__main__':unittest.main()
