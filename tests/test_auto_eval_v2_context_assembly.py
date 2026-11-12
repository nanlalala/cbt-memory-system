"""Invented renderer contracts; not model-effect evidence."""
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evaluation/cbt-memory-effect'))
from auto_eval_v2.context_assembly import assemble, ARMS


def example():
    texts=['I agreed to practise a thought record seven times.',
           'Please change that target to two times. I have not started yet.']
    sources=[dict(source_id=f's{i}',source_order=i,role='user',content=t,
        subject_id='invented:one',retained=True,use_allowed=True,external_allowed=True)
        for i,t in enumerate(texts)]
    memory=dict(memory_id='target',version=2,subject_id='invented:one',active=True,
        deleted=False,valid_at_cutoff=True,superseded_by=None,support='verified_projection',
        verifier_revision='invented-rule-v1',epistemic_status='supported_state',
        summary='The user revised the thought-record target from seven to two practices and reported not starting.',
        source_ids=['s0','s1'],evidence=[dict(source_id=s['source_id'],start=0,
        end=len(s['content']),text=s['content']) for s in sources])
    return dict(sources=sources,selected_ids=['s0','s1'],memories=[memory],recent=[],
        current_input='What did I agree to, and what have I completed?',subject_id='invented:one',
        cutoff_order=1,max_request_chars=6000,request_options=dict(model='example-model',max_tokens=256),
        verifier_revision='invented-rule-v1')

class AssemblyTests(unittest.TestCase):
    def test_three_actual_requests(self):
        runs=[assemble(**example(),arm=a) for a in ARMS]
        self.assertEqual(len({r['trace']['request_sha256'] for r in runs}),3)
        for r in runs:
            self.assertEqual([m['role'] for m in r['request']['messages']],['system','user','user'])
            self.assertNotIn('source_sha256',r['serialized_request'])
            self.assertIsNone(r['trace']['request_tokens'])
        self.assertIn('revised',runs[1]['request']['messages'][-2]['content'])
        self.assertNotIn('"text":',runs[1]['request']['messages'][-2]['content'])
        self.assertIn('"text":',runs[2]['request']['messages'][-2]['content'])
    def test_permission_deletion_future_and_subject(self):
        for key,val in [('retained',False),('external_allowed',False),('subject_id','other'),('source_order',9)]:
            x=example();x['sources'][0][key]=val
            with self.assertRaises(ValueError):assemble(**x,arm='processed')
    def test_superseded_and_inference_not_promoted(self):
        for key,val in [('superseded_by','new'),('deleted',True),('support','unverified'),('epistemic_status','inference')]:
            x=example();x['memories'][0][key]=val
            r=assemble(**x,arm='processed')
            self.assertIn('target',r['trace']['rejected_memories'])
            self.assertNotIn('revised',r['serialized_request'])
    def test_binding_and_dependency(self):
        x=example();x['memories'][0]['evidence'][0]['text']='fabricated'
        self.assertIn('target',assemble(**x,arm='processed')['trace']['rejected_memories'])
        x=example();x['selected_ids']=['s0']
        self.assertIn('target',assemble(**x,arm='processed')['trace']['rejected_memories'])
    def test_budget_and_recent_integrity(self):
        x=example();x['max_request_chars']=10
        with self.assertRaises(ValueError):assemble(**x,arm='raw')
        x=example();x['recent']=[dict(source_id='s0',role='user',content='changed')]
        with self.assertRaises(ValueError):assemble(**x,arm='raw')

if __name__=='__main__':unittest.main()
