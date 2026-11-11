"""Invented text and authenticated fake receipts only; no real model/data."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest
import tests.test_controlled_open_memory as fixtures
M=fixtures.M

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('format_diagnostic_fixture',ROOT/'evaluation/cbt-memory-effect/audit_open_reader_response_format.py')
D=importlib.util.module_from_spec(spec);sys.modules[spec.name]=D;spec.loader.exec_module(D)

class FormatParserTests(unittest.TestCase):
 def test_exact_json_is_not_a_recovery(self):
  r=D.recover_response(' {"answer":2}\n')
  self.assertEqual((r['status'],r['value']),('strict_json',2))
 def test_fence_or_benign_prose_can_expose_one_object(self):
  for text in ('```json\n{"answer":2}\n```','Here is the requested object: {"answer":2}'):
   self.assertEqual(D.recover_response(text)['value'],2)
   self.assertEqual(D.recover_response(text)['status'],'recovered')
 def test_only_declared_scalar_string_normalizations(self):
  for text,value in [('{"answer":"2"}',2),('{"answer":"UNKNOWN"}',None),('{"answer":"helpful"}','helpful')]:
   r=D.recover_response(text);self.assertEqual(r['value'],value)
  self.assertEqual(D.recover_response('{"answer":"02"}')['value'],'02')
  self.assertEqual(D.recover_response('{"answer":"HELPFUL"}')['value'],'HELPFUL')
 def test_multiple_or_duplicate_objects_do_not_choose_convenient_value(self):
  for text in ('{"answer":2} {"answer":3}','{"answer":2} {"answer":2}','{"answer":2,"answer":3}'):
   self.assertEqual(D.recover_response(text)['status'],'unresolved')
 def test_no_free_text_guess_or_nested_schema_upgrade(self):
  for text in ('The answer is 2.','{"wrapper":{"answer":2}}','{"note": {"answer":2}','{"note":2,"answer":2,"answer":2} {"answer":2}','{"foo":"a","foo":{"answer":2}}','{"answer":true}','{"answer":2.0}','{"answer":2,"why":"text"}'):
   self.assertEqual(D.recover_response(text)['status'],'unresolved')
 def test_conflicting_prose_is_unresolved(self):
  for text in ('The answer is 3. {"answer":2}','The answer is not 2. {"answer":2}','Not the answer: {"answer":2}','NOT helpful. {"answer":"helpful"}','The outcome is helpful. {"answer":"unhelpful"}','An example is {"answer":2}'):
   self.assertEqual(D.recover_response(text)['status'],'unresolved')
 def test_consistent_exterior_indicator_does_not_supply_missing_json(self):
  self.assertEqual(D.recover_response('The answer is 2. {"answer":2}')['value'],2)
  self.assertEqual(D.recover_response('The answer is 2.')['status'],'unresolved')
 def test_failure_classification_keeps_assertion_and_abstention_separate_from_format(self):
  examples=[('{"answer":7}',None,False,'unsupported_assertion'),
   ('{"answer":"7"}',None,False,'unsupported_assertion'),
   ('{"answer":null}',2,False,'incorrect_abstention'),
   ('{"answer":"unknown"}',2,False,'incorrect_abstention'),
   ('{"answer":4}',2,False,'wrong_value'),
   ('{"answer":"2"}',2,False,'recoverable_format'),
   ('{"answer":"unknown"}',None,False,'recoverable_format'),
   ('no object',2,False,'unresolved_format'),('{"answer":2}',2,True,'correct')]
  for text,target,strict,wanted in examples:
   self.assertEqual(D.classify_recovery(D.recover_response(text),target,strict),wanted)

class FormatAuditTests(unittest.TestCase):
 def setUp(self):
  self.helper=fixtures.ControlledOpenTests();self.helper.setUp();self.addCleanup(self.helper.doCleanups)
  self.rows=self.helper.prepared(1);values=[]
  for i,row in enumerate(self.rows):
   text='Here is the requested object: {"answer":"2"}' if i==0 else '{"answer":2}'
   values.append({'case_id':row['case_id'],'arm':row['arm'],'status':'completed',
    'request_sha256':row['request_sha256'],'source_checks_passed':True,
    'response_text':text,'response_sha256':hashlib.sha256(text.encode()).hexdigest()})
  self.answers,self.modelroot=self.helper.authentic_fake_archive(self.rows,values)
  self.score=Path(self.helper.tmp.name)/'primary-score.json'
  M.score(self.helper.out,self.answers,self.score,model_root=self.modelroot)
  self.expected=D.sha(self.score);self.output=Path(self.helper.tmp.name)/'secondary.json'
 def call(self):
  return D.audit(self.helper.out,self.answers,self.score,self.modelroot,self.output,
   expected_score_sha256=self.expected,fixture_only=True)
 def test_primary_unchanged_and_all_rows_retained(self):
  original=self.score.read_bytes();r=self.call()
  self.assertEqual(self.score.read_bytes(),original)
  self.assertEqual(r['by_arm']['raw_history_rag']['strict_correct'],0)
  self.assertEqual(r['by_arm']['raw_history_rag']['secondary_typed_value_correct'],1)
  self.assertEqual(r['by_arm']['managed_memory_rag']['strict_correct'],1)
  data=json.loads(self.output.read_text());self.assertEqual(len(data['per_question']),2)
  self.assertTrue(all('original_response_text' in x for x in data['per_question']))
  prior=json.loads(self.score.read_text())['per_question']
  self.assertEqual([r['primary_reason'] for r in data['per_question']],[r['reason'] for r in prior])
  self.assertEqual(data['per_question'][0]['diagnostic_failure_class'],'recoverable_format')
 def test_expected_primary_hash_is_mandatory(self):
  self.expected='0'*64
  with self.assertRaisesRegex(ValueError,'score_file_changed'):self.call()
  self.assertFalse(self.output.exists())
 def test_planner_path_is_explicit_and_must_match_frozen_code_identity(self):
  planner=ROOT/'evaluation/cbt-memory-effect/controlled_open_memory.py'
  r=D.audit(self.helper.out,self.answers,self.score,self.modelroot,self.output,
   expected_score_sha256=self.expected,planner_path=planner,fixture_only=True)
  self.assertEqual(r['bindings']['planner_code_sha256'],D.sha(planner))
  protocol=self.helper.out/'protocol.json';v=json.loads(protocol.read_text());v['code_identity']={};protocol.write_text(json.dumps(v))
  # Identity is rejected before authenticating altered protocol or opening gold.
  with self.assertRaisesRegex(ValueError,'planner_identity'):
   D.audit(self.helper.out,self.answers,self.score,self.modelroot,Path(self.helper.tmp.name)/'other.json',
    expected_score_sha256=self.expected,planner_path=planner,fixture_only=True)
 def test_omitting_unfavorable_row_is_rejected(self):
  data=json.loads(self.score.read_text());data['per_question']=data['per_question'][1:]
  self.score.write_text(json.dumps(data));self.expected=D.sha(self.score)
  with self.assertRaisesRegex(ValueError,'all_planned_unique_rows'):self.call()
 def test_changed_primary_label_cannot_redefine_expected_answer(self):
  data=json.loads(self.score.read_text());data['per_question'][0]['expected_answer']=99
  self.score.write_text(json.dumps(data));self.expected=D.sha(self.score)
  with self.assertRaisesRegex(ValueError,'score_row_binding_changed'):self.call()
 def test_rewritten_answer_cannot_replace_canonical_receipt(self):
  data=[json.loads(x) for x in self.answers.read_text().splitlines()]
  data[0]['response_text']='{"answer":2}';data[0]['response_sha256']=hashlib.sha256(data[0]['response_text'].encode()).hexdigest()
  data[0]['record_sha256']=M.stable({k:v for k,v in data[0].items() if k!='record_sha256'})
  self.answers.write_text(''.join(json.dumps(x)+'\n' for x in data))
  complete=self.answers.parent/'completion_summary.json';v=json.loads(complete.read_text());v['answers_sha256']=D.sha(self.answers);complete.write_text(json.dumps(v))
  score=json.loads(self.score.read_text());score['answers_file_sha256']=D.sha(self.answers);self.score.write_text(json.dumps(score));self.expected=D.sha(self.score)
  with self.assertRaisesRegex(ValueError,'canonical model output'):self.call()

if __name__=='__main__':unittest.main()
