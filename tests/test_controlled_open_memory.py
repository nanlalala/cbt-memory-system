"""Tiny invented fixtures; no dataset, model, credential or network calls."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('controlled_open_memory',ROOT/'evaluation/cbt-memory-effect/controlled_open_memory.py')
M=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=M;SPEC.loader.exec_module(M)

class ControlledOpenTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.out=Path(self.tmp.name)/'prepared'
 def prepared(self,n=2):
  M.prepare(self.out,fixture_only=True,case_limit=n)
  return [json.loads(l) for l in (self.out/'requests.jsonl').read_text().splitlines()]
 def authentic_fake_archive(self,rows,answers):
  spec=importlib.util.spec_from_file_location('fixture_receipts',ROOT/'evaluation/cbt-memory-effect/open_reader_colab.py')
  reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
  modelroot=Path(self.tmp.name)/'model';modelroot.mkdir()
  out=Path(self.tmp.name)/'reader';out.mkdir()
  model={'fixture_model':'invented journal only'}
  run={'prepared_protocol_sha256':M.filehash(self.out/'protocol.json'),
   'requests_file_sha256':M.filehash(self.out/'requests.jsonl'),
   'reader_code_sha256':M.filehash(ROOT/'evaluation/cbt-memory-effect/open_reader_colab.py'),
   'model':model,'fixture_only':True}
  reader._write_new(out/'run_protocol.json',run);runhash=reader.digest(run)
  for row,a in zip(rows,answers):
   key=reader.digest({'model':model,'payload':row['request_payload'],'max_new_tokens':64,'do_sample':False})
   cache={'canonical_key':key,'response_text':a['response_text'],'response_sha256':a['response_sha256'],
    'finish_reason':'eos','usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2}}
   cachepath=out/'canonical'/(key+'.json');reader._write_new(cachepath,cache)
   reader._append(modelroot/'generation_ledger.jsonl',{'event':'generation_intent','canonical_key':key})
   reader._append(modelroot/'generation_ledger.jsonl',{'event':'generation_outcome','canonical_key':key,
    'status':'published','cache_path':str(cachepath),'cache_sha256':reader.file_hash(cachepath),
    'response_sha256':a['response_sha256'],'finish_reason':'eos','usage':cache['usage']})
   reader._append(out/'answers_unscored.jsonl',{**a,'canonical_key':key,'run_protocol_sha256':runhash,'finish_reason':'eos'})
  path=out/'answers_unscored.jsonl'
  reader._write_new(out/'completion_summary.json',{'complete':True,'run_protocol_sha256':runhash,
   'answers_sha256':reader.file_hash(path)})
  return path,modelroot
 def test_fixed_materials_are_bilingual_four_related_template_families(self):
  cases,targets=M.fixed_materials();self.assertEqual((len(cases),len(targets)),(24,24))
  self.assertEqual(len({c['source_family_id'] for c in cases}),4)
  self.assertEqual({c['language'] for c in cases},{'zh','en'})
  self.assertTrue(all('answer' not in c for c in cases))
  self.assertEqual([t['answer'] for t in targets[:6]],[2,2,3,3,4,4])
 def test_same_recent_and_question_source_cap_and_budget(self):
  rows=self.prepared(1);self.assertEqual(len(rows),2)
  direct=[]
  for r in rows:
   p=json.loads(Path(r['prepared_case_path']).read_text());c=p['contexts'][r['arm']]
   self.assertLessEqual(c['outbound_chars'],6000);self.assertLessEqual(len(c['source_ids']),9)
   direct.append([x for x in r['messages'] if x['role']!='system'])
   self.assertEqual(len(direct[-1]),7)
  self.assertEqual(direct[0],direct[1])
  self.assertEqual(rows[0]['messages'][0],rows[1]['messages'][0])
 def test_managed_chinese_correction_is_product_projected_not_gold_injected(self):
  rows=self.prepared(1);managed=next(r for r in rows if r['arm']=='managed_memory_rag')
  claims=[]
  for msg in managed['messages']:
   if msg['content'].startswith('SOURCED MEMORY'):
    claims.extend(json.loads(msg['content'].split('\n',1)[1])['claims'])
  field=next(c for c in claims if c['field']=='target_count')
  self.assertEqual(field['value'],2);self.assertEqual(field['support_status'],'rule_verified')
  self.assertTrue(any('turn-002' in e['source_id'] for e in field['evidence']))
 def test_raw_retrieval_does_not_spend_slots_on_direct_recent_sources(self):
  rows=self.prepared(1);raw=next(r for r in rows if r['arm']=='raw_history_rag')
  p=json.loads(Path(raw['prepared_case_path']).read_text());c=p['contexts'][raw['arm']]
  recent={x['source_id'] for x in c['binding']['recent_sources']}
  self.assertFalse(recent.intersection(c['source_ids']))
 def test_foreign_cached_product_module_is_rejected(self):
  M.product();module=sys.modules['task_projector']
  with patch.object(module,'__file__','/unrelated/old-checkout/task_projector.py'):
   with self.assertRaisesRegex(RuntimeError,'foreign cached product'):M.product()
 def test_fresh_databases_have_identical_wire_but_keep_distinct_actual_task_ids(self):
  rows=self.prepared(1);other=Path(self.tmp.name)/'second'
  M.prepare(other,fixture_only=True,case_limit=1)
  second=[json.loads(l) for l in (other/'requests.jsonl').read_text().splitlines()]
  self.assertEqual([r['request_sha256'] for r in rows],[r['request_sha256'] for r in second])
  a=json.loads(Path(rows[0]['prepared_case_path']).read_text())
  b=json.loads(Path(second[0]['prepared_case_path']).read_text())
  self.assertNotEqual(a['wire_task_aliases'][0]['actual_task_instance_id'],b['wire_task_aliases'][0]['actual_task_instance_id'])
  self.assertEqual(a['wire_task_aliases'][0]['wire_alias'],b['wire_task_aliases'][0]['wire_alias'])
  for r in rows:
   self.assertNotIn(a['wire_task_aliases'][0]['actual_task_instance_id'],json.dumps(r['messages']))
   self.assertTrue(M.make_gates(self.out).preflight(r)['expected_task_versions'])
 def test_alias_rendering_does_not_change_user_text_values_or_evidence_offsets(self):
  product=M.product();actual='task-deadbeef';alias='task-view-fixed'
  text='My literal diary happens to mention task-deadbeef.'
  span=product.EvidenceSpan('source:original','span:'+actual,text,0,len(text),'2026-11-03T08:00:00Z','user')
  claim=product.EvidenceClaim(actual+':target_count','target_count',2,(span,),support_status='rule_verified')
  bundle=product.EvidenceBundle('task:'+actual,(claim,),actual)
  result=M._alias_bundle_ids([bundle],{actual:alias})[0];changed=result.claims[0]
  self.assertEqual(result.task_instance_id,alias);self.assertEqual(changed.value,2)
  witness=changed.spans[0]
  self.assertEqual((witness.source_id,witness.text,witness.start,witness.end,witness.role,witness.source_time),
   (span.source_id,span.text,span.start,span.end,span.role,span.source_time))
  self.assertIn(actual,witness.text)
 def test_preflight_does_not_read_assessor_target(self):
  row=self.prepared(1)[0];g=M.make_gates(self.out);target=self.out/'assessor_only'/'targets.json'
  original=Path.read_text
  def guarded(p,*a,**k):
   if p==target:raise AssertionError('target read by gate')
   return original(p,*a,**k)
  with patch.object(Path,'read_text',guarded):self.assertEqual(g.preflight(row)['version'],M.VERSION)
 def test_mutated_request_row_and_frozen_wire_fail(self):
  row=self.prepared(1)[0];g=M.make_gates(self.out)
  changed=json.loads(json.dumps(row));changed['messages'][-1]['content']='Changed question'
  with self.assertRaisesRegex(ValueError,'row changed'):g.preflight(changed)
  with (self.out/'requests.jsonl').open('a') as f:f.write('\n')
  with self.assertRaisesRegex(ValueError,'request file changed'):g.preflight(row)
 def test_source_withdrawal_blocks_and_late_finish_discards(self):
  row=self.prepared(1)[0];g=M.make_gates(self.out);a=g.begin(row)
  p,c=g._prepared(row);d=g._store(p)
  try:d.revoke(c['binding']['user_id'],'source',c['binding']['current_source_id'])
  finally:d.close()
  with self.assertRaisesRegex(ValueError,'generation changed'):g.preflight(row)
  result=g.finish(row,a,outcome='success',response={'text':'must not publish'},usage=None)
  self.assertFalse(result['publishable']);self.assertIsNone(result['response'])
 def test_unknown_admission_is_not_automatically_retried(self):
  row=self.prepared(1)[0];g=M.make_gates(self.out);g.begin(row)
  with self.assertRaises(M.loadlegacy()._modules().AttemptUnknown):g.begin(row)
 def test_successful_unchanged_admission_is_replayed(self):
  row=self.prepared(1)[0];g=M.make_gates(self.out);a=g.begin(row)
  self.assertFalse(a['replayed']);self.assertTrue(g.finish(row,a,outcome='success',response=None,usage=None)['publishable'])
  self.assertTrue(g.begin(row)['replayed'])
 def test_scorer_refuses_incomplete_answers_before_targets(self):
  self.prepared(1);a=Path(self.tmp.name)/'answers.jsonl';a.write_text('')
  with self.assertRaisesRegex(ValueError,'all planned answers'):M.score(self.out,a,Path(self.tmp.name)/'scores.json',model_root=Path(self.tmp.name)/'model')
 def test_strict_scorer_keeps_malformed_and_wrong_outputs(self):
  rows=self.prepared(1);answers=[]
  for i,row in enumerate(rows):
   text='{"answer":2}' if i==1 else '{"answer":"2"}'
   answers.append({'case_id':row['case_id'],'arm':row['arm'],'status':'completed',
    'request_sha256':row['request_sha256'],'source_checks_passed':True,
    'response_text':text,'response_sha256':hashlib.sha256(text.encode()).hexdigest()})
  a,modelroot=self.authentic_fake_archive(rows,answers)
  r=M.score(self.out,a,Path(self.tmp.name)/'scores.json',model_root=Path(self.tmp.name)/'model')
  self.assertEqual(r['arms']['raw_history_rag']['correct'],0)
  self.assertEqual(r['arms']['raw_history_rag']['errors'],{'output_type_error':1})
  self.assertEqual(r['arms']['managed_memory_rag']['correct'],1)
  self.assertEqual(r['paired_outcomes'],{'managed_win':1})
  self.assertFalse(r['clinical_quality_claim'])
  # Rewriting an answer and its local checksums cannot replace the model's
  # canonical receipt, even when a fabricated completion checksum is updated.
  data=[json.loads(l) for l in a.read_text().splitlines()]
  data[0]['response_text']='{"answer":2}'
  data[0]['response_sha256']=hashlib.sha256(data[0]['response_text'].encode()).hexdigest()
  data[0]['record_sha256']=M.stable({k:v for k,v in data[0].items() if k!='record_sha256'})
  a.write_text(''.join(M.canonical(x)+'\n' for x in data))
  complete=a.parent/'completion_summary.json';value=json.loads(complete.read_text());value['answers_sha256']=M.filehash(a);complete.write_text(M.canonical(value)+'\n')
  with self.assertRaisesRegex(ValueError,'canonical model output'):
   M.score(self.out,a,Path(self.tmp.name)/'tampered.json',model_root=modelroot)
  for text in ('{"answer":true}','{"answer":2,"answer":3}','{"answer":2,"extra":1}','answer:2'):
   with self.assertRaises(ValueError):M._answer(text)

if __name__=='__main__':unittest.main()
