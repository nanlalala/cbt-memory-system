"""Tiny locator fixtures only; no real corpus, Secret, model or API."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('audit_locator_shortcut',ROOT/'evaluation/cbt-memory-effect/audit_locator_shortcut.py')
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)
P=M.common.product_modules(ROOT)

class LocatorAuditTests(unittest.TestCase):
    def fixture(self,root):
        old=root/'old';(old/'generation').mkdir(parents=True);(old/'assessor_only').mkdir()
        texts=['I felt nervous today about my interview.','The thought became less convincing after reflection.','I will try tomorrow after discussing the plan.']
        history=[{'message_id':'t'+str(i),'source_order':i,'role':'user','content':text} for i,text in enumerate(texts)]
        cases=[];targets=[]
        for i,text in enumerate(texts):
            cue=' '.join(text.split()[:4])
            case={'case_id':'fixture-probe-'+str(i),'source_family_id':'fixture-family','history':history,
                  'history_cutoff_order_exclusive':3,'input':'A source '+M.MARKER+json.dumps(cue)+'. Return the complete message.'}
            case['case_sha256']=M.common.digest(case);cases.append(case)
            targets.append({'case_id':case['case_id'],'case_sha256':case['case_sha256'],'source_family_id':'fixture-family',
                'target_message_id':history[i]['message_id'],'target_quote':text,'position':['early','middle','recent'][i]})
        (old/'generation/cases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cases))
        (old/'assessor_only/targets.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in targets))
        code='modules/agent-integration/controlled_dialogue.py'
        protocol={'version':'cactus-source-quotation-recall-v2','clock_mode':'ordinal-only-v1','ready_probes':3,
            'code_sha256':{code:M.common.file_hash(ROOT/code)},
            'files_sha256':{name:M.common.file_hash(old/name) for name in ['generation/cases.jsonl','assessor_only/targets.jsonl']}}
        protocol['protocol_sha256']=M.common.digest(protocol);(old/'protocol.json').write_text(json.dumps(protocol))
        return old,cases,targets

    def test_three_unique_matches_are_not_answer_accuracy_and_register_exposure(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);old,cases,targets=self.fixture(root)
            before={str(p):M.common.file_hash(p) for p in old.rglob('*') if p.is_file()}
            with patch('socket.create_connection',side_effect=AssertionError('no network')):
                result=M.audit(old,root/'audit',product_repo=ROOT,fixture_only=True)
            self.assertEqual(result['all']['unique_prefix_matches'],3)
            self.assertEqual(result['all']['unique_matches_verifying_original_target'],3)
            self.assertIsNone(result['answer_accuracy']);self.assertIsNone(result['source_id_joint_score'])
            self.assertFalse(result['match_phase_read_gold']);self.assertEqual(result['actual_api_calls'],0)
            exposure=M.common.read_json(root/'audit/assessed_family_ids.json')
            self.assertEqual(exposure['family_ids'],['fixture-family'])
            self.assertFalse(exposure['may_be_called_unseen_holdout_after_this_audit'])
            self.assertTrue(all(t['target_quote'] not in (root/'audit/results.jsonl').read_text() for t in targets))
            self.assertEqual(before,{str(p):M.common.file_hash(p) for p in old.rglob('*') if p.is_file()})

    def test_matching_excludes_assistant_future_and_permission_withheld_messages(self):
        case={'case_id':'fixture','source_family_id':'f','case_sha256':'fixture','history_cutoff_order_exclusive':3,
            'input':M.MARKER+json.dumps('I felt nervous today')+'.',
            'history':[{'message_id':'assistant','source_order':0,'role':'assistant','content':'I felt nervous today said by assistant.'},
                       {'message_id':'old','source_order':1,'role':'user','content':'I felt nervous today as a user.'},
                       {'message_id':'future','source_order':3,'role':'user','content':'I felt nervous today in the future.'}]}
        result=M.match_without_gold(case,P)
        self.assertEqual(result['match_count'],1);self.assertEqual(result['message_id'],'old')
        class Withheld:
            @staticmethod
            def permissions_for_input(text,defaults):return {**defaults,'response_external':False}
        self.assertEqual(M.match_without_gold(case,Withheld)['match_count'],0)
        case['history'].append({'message_id':'also-old','source_order':2,'role':'user','content':'I felt nervous today as another user message.'})
        result=M.match_without_gold(case,P)
        self.assertEqual(result['match_count'],2);self.assertIsNone(result['message_id']);self.assertIsNone(result['quote_sha256'])

    def test_matches_are_written_before_any_gold_file_read(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);old,cases,targets=self.fixture(root)
            original=M.common.file_hash
            def watch(path):
                if Path(path)==old/'assessor_only/targets.jsonl':
                    self.assertTrue((root/'audit/matches_before_gold.jsonl').is_file())
                return original(path)
            with patch.object(M.common,'file_hash',side_effect=watch):
                M.audit(old,root/'audit',product_repo=ROOT,fixture_only=True)
            (old/'generation/cases.jsonl').write_text('{}\n')
            with self.assertRaises(ValueError):M.audit(old,root/'other',product_repo=ROOT,fixture_only=True)

    def test_json_locator_is_decoded_exactly(self):
        cue='I said "no" yesterday'
        self.assertEqual(M.parse_cue(M.MARKER+json.dumps(cue)+'. Return message.'),cue)
        with self.assertRaises(ValueError):M.parse_cue(M.MARKER+'{"not":"string"}')
        with self.assertRaises(ValueError):M.parse_cue('No explicit locator field.')

if __name__=='__main__':unittest.main()
