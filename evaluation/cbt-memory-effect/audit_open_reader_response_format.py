"""Secondary, goal-independent format diagnostic; strict primary scores stand.

Declared during generation, before any response text or gold score inspection.
Only unique single-answer JSON objects are recovered. Canonical unsigned integer
strings and the exact case-insensitive word 'unknown' may normalize; other text
is never guessed into an answer. Conflicting exterior numeric/status indications
are unresolved. This is descriptive, not a replacement primary/clinical score.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent
VERSION='open-reader-secondary-format-diagnostic-v1'
RULES={'unique_json_object':True,'multiple_objects':'reject','duplicate_keys':'reject',
 'numeric_string':r'0|[1-9][0-9]*','unknown_string':'case-insensitive unknown only',
 'other_string_normalization':False,'exterior_conflicting_value':'unresolved',
 'unparseable_brace_object':'reject','exterior_negation_or_example':'unresolved',
 'free_text_number_guessing':False,'primary_scores_unchanged':True}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _unique(pairs):
 out={}
 for k,v in pairs:
  if k in out:raise ValueError('duplicate_json_key')
  out[k]=v
 return out
def _constant(value):raise ValueError('nonfinite_json')
def _normalize(value):
 if type(value) is str and re.fullmatch(RULES['numeric_string'],value):return int(value),'numeric_string'
 if type(value) is str and value.strip().casefold()=='unknown':return None,'unknown_string'
 return value,None
def recover_response(text):
 """No question, reference answer, case identity or score enters this parser."""
 reject=lambda reason:{'status':'unresolved','reason':reason,'value':None,'normalization':None}
 if type(text) is not str or len(text)>20000:return reject('invalid_response_text')
 decoder=json.JSONDecoder(object_pairs_hook=_unique,parse_constant=_constant)
 objects=[];cursor=0
 while True:
  start=text.find('{',cursor)
  if start<0:break
  try:obj,length=decoder.raw_decode(text[start:])
  except (ValueError,json.JSONDecodeError):return reject('invalid_json_object')
  objects.append((obj,start,start+length));cursor=start+length
 if len(objects)!=1:return reject('missing_or_multiple_json_objects')
 obj,start,end=objects[0]
 if type(obj) is not dict or set(obj)!={'answer'}:return reject('not_single_answer_object')
 value,normalization=_normalize(obj['answer'])
 if value is not None and type(value) not in (int,str):return reject('unsupported_answer_type')
 exterior=text[:start]+text[end:]
 if re.search(r'\b(?:example|e\.g|such\s+as)\b|例如|示例',exterior,re.I):return reject('exterior_example_not_answer')
 indicators=re.findall(r'(?<![\w.])-?[0-9]+(?:\.[0-9]+)?(?![\w]|\.[0-9])|\b(?:true|false|null|unknown|helpful|unhelpful|mixed)\b',exterior,re.I)
 if re.search(r"\b(?:not|never|isn't|isn’t|cannot|incorrect|wrong)\b|不是|并非|不能|不对",exterior,re.I):return reject('negated_or_contradictory_exterior')
 for token in indicators:
  if re.fullmatch(RULES['numeric_string'],token):other=int(token)
  elif token.casefold() in {'unknown','null'}:other=None
  else:other=token.casefold()
  if type(other) is not type(value) or other!=value:return reject('exterior_conflicting_or_unresolved_value')
 return {'status':'recovered' if normalization or text.strip()!=text[start:end] else 'strict_json',
  'reason':'unique_goal_independent_json','value':value,'normalization':normalization}
def _primary(planner_path=None):
 name='secondary_format_primary'
 spec=importlib.util.spec_from_file_location(name,Path(planner_path or HERE/'controlled_open_memory.py').resolve(strict=True))
 module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
 return module
def classify_recovery(parsed,target,strict_correct):
 """Scoring only; expected value never enters recover_response()."""
 if strict_correct:return 'correct'
 if parsed['status']=='unresolved':return 'unresolved_format'
 value=parsed['value']
 if target is None and value is not None:return 'unsupported_assertion'
 if target is not None and value is None:return 'incorrect_abstention'
 if type(value) is type(target) and value==target:return 'recoverable_format'
 return 'wrong_value'
def audit(prepared_dir,answers_file,score_file,model_root,output,*,expected_score_sha256,planner_path=None,fixture_only=False):
 if not fixture_only:
  sys.path.insert(0,str(HERE.parents[1]/'app'));from colab_provider import require_colab
  require_colab()
 root=Path(prepared_dir);answers_file=Path(answers_file);score_file=Path(score_file)
 if sha(score_file)!=expected_score_sha256:raise ValueError('primary_score_file_changed')
 p=_primary(planner_path);protocol=json.loads((root/'protocol.json').read_text());report=json.loads(score_file.read_text())
 if p.identity()!=protocol.get('code_identity'):raise ValueError('planner_identity_does_not_match_prepared')
 if (report.get('answers_file_sha256')!=sha(answers_file) or report.get('protocol_file_sha256')!=sha(root/'protocol.json')
  or report.get('version')!=p.VERSION+'-strict-assessor-v1' or report.get('generation_receipts_independently_reconciled') is not True):raise ValueError('primary_score_binding_invalid')
 requests=[json.loads(l) for l in (root/'requests.jsonl').read_text().splitlines() if l.strip()]
 if sha(root/'requests.jsonl')!=protocol['requests_file_sha256']:raise ValueError('requests_changed')
 answers,run=p._authenticate_answers(root,answers_file,model_root)
 if run.get('fixture_only') is not fixture_only:raise ValueError('fixture_and_actual_run_mixed')
 key=lambda r:(r['case_id'],r['arm'])
 grid={key(r):r for r in requests};observed={key(r):r for r in answers};scored={key(r):r for r in report['per_question']}
 if any(len(d)!=n for d,n in ((grid,len(requests)),(observed,len(answers)),(scored,len(report['per_question'])))) or set(grid)!=set(observed) or set(grid)!=set(scored):raise ValueError('all_planned_unique_rows_required')
 gates=p.make_gates(root)
 for req in requests:gates.preflight(req)
 if sha(root/'assessor_only/targets.json')!=protocol['targets_file_sha256']:raise ValueError('frozen_targets_changed')
 targets={r['case_id']:r for r in json.loads((root/'assessor_only/targets.json').read_text())}
 rows=[]
 for identity,req in grid.items():
  a=observed[identity];s=scored[identity];target=targets[identity[0]]['answer']
  canonical_key=p.stable({'model':run['model'],'payload':req['request_payload'],'max_new_tokens':req['request_payload']['max_tokens'],'do_sample':False})
  if a['canonical_key']!=canonical_key or a['request_sha256']!=req['request_sha256'] or a.get('source_checks_passed') is not True or a.get('status')!='completed':raise ValueError('unbound_answer')
  if s.get('request_sha256')!=req['request_sha256'] or s.get('response_sha256')!=a['response_sha256'] or type(s.get('expected_answer')) is not type(target) or s.get('expected_answer')!=target:raise ValueError('score_row_binding_changed')
  try:v=p._answer(a['response_text']);strict=type(v) is type(target) and v==target
  except (ValueError,TypeError):strict=False
  if type(s.get('correct')) is not bool or s['correct']!=strict:raise ValueError('strict_score_disagrees')
  parsed=recover_response(a['response_text']);value=parsed['value']
  recovered=parsed['status']!='unresolved' and type(value) is type(target) and value==target
  rows.append({'case_id':identity[0],'arm':identity[1],'source_family_id':s['source_family_id'],
   'strict_correct':strict,'secondary_typed_value_correct':recovered,'format_diagnostic':parsed,
   'primary_reason':s['reason'],'diagnostic_failure_class':classify_recovery(parsed,target,strict),
   'request_sha256':req['request_sha256'],'response_sha256':a['response_sha256'],'original_response_text':a['response_text']})
 for req in requests:gates.preflight(req)
 if sha(score_file)!=expected_score_sha256 or sha(answers_file)!=report['answers_file_sha256'] or sha(root/'protocol.json')!=report['protocol_file_sha256']:raise ValueError('input_changed_during_diagnostic')
 summary={arm:{'planned_rows':sum(r['arm']==arm for r in rows),
  'strict_correct':sum(r['strict_correct'] for r in rows if r['arm']==arm),
  'secondary_typed_value_correct':sum(r['secondary_typed_value_correct'] for r in rows if r['arm']==arm),
  'diagnostic_failure_class':dict(Counter(r['diagnostic_failure_class'] for r in rows if r['arm']==arm)),
  'recovered_unknown_count':sum(r['format_diagnostic']['status']!='unresolved' and r['format_diagnostic']['value'] is None for r in rows if r['arm']==arm),
  'diagnostic_status':dict(Counter(r['format_diagnostic']['status'] for r in rows if r['arm']==arm))} for arm in p.ARMS}
 result={'version':VERSION,'rules':RULES,'by_arm':summary,'per_question':rows,'primary_scores_unchanged':True,
  'secondary_descriptive_only':True,'clinical_quality_claim':False,'new_model_calls':0,
  'declaration_timing':'added_during_generation_before_response_text_or_gold_score_inspection',
  'bindings':{'score_sha256':sha(score_file),'answers_sha256':sha(answers_file),'protocol_sha256':sha(root/'protocol.json'),'diagnostic_code_sha256':sha(__file__),
   'declaration_sha256':sha(HERE/'SECONDARY_FORMAT_DIAGNOSTIC_DECLARATION.md'),
   'planner_path':str(Path(p.__file__).resolve()),'planner_code_sha256':sha(p.__file__),'planner_identity_sha256':p.stable(p.identity())}}
 p.writejson(output,result)
 return {k:v for k,v in result.items() if k!='per_question'}
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for field in ('prepared-dir','answers','primary-score','model-root','output','expected-score-sha256'):parser.add_argument('--'+field,required=True)
 parser.add_argument('--planner-path',default=str(HERE/'controlled_open_memory.py'))
 a=parser.parse_args();print(json.dumps(audit(a.prepared_dir,a.answers,a.primary_score,a.model_root,a.output,expected_score_sha256=a.expected_score_sha256,planner_path=a.planner_path),sort_keys=True))
if __name__=='__main__':main()
