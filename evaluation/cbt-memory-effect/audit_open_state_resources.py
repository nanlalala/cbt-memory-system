"""Read-only hosted-Colab resource reconciliation; never generate or open gold.

Only tiny invented metadata fixtures exercise pure accounting helpers outside
Colab. Untimed preparation stages stay null; referenced cache costs are not
added again to physical generation costs.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import importlib.util
import json
import math
from pathlib import Path
import sqlite3

HERE=Path(__file__).resolve().parent
VERSION='open-state-resources-readonly-v1'
TOKENS=('prompt_tokens','completion_tokens','total_tokens')

def _reader():
 spec=importlib.util.spec_from_file_location('open_resource_reader',HERE/'open_reader_colab.py')
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def costs(keys,intents,outcomes):
 keys=set(keys);statuses=Counter(outcomes.get(k,{}).get('status','intent_without_outcome') for k in keys)
 observed={};missing={}
 for field in TOKENS:
  values=[(outcomes.get(k,{}).get('usage') or {}).get(field) for k in keys]
  valid=[v for v in values if type(v) is int and v>=0]
  observed[field]=sum(valid) if valid or not keys else None;missing[field]=len(keys)-len(valid)
 times=[]
 for k in keys:
  row=outcomes.get(k,{})
  value=row.get('generation_wall_seconds') if row.get('status')=='published' else row.get('observed_wall_seconds')
  if type(value) in (int,float) and math.isfinite(value) and value>=0:times.append(value)
 return {'unique_admitted_attempts':len(keys),'status_counts':dict(statuses),
  'observed_tokens':observed,'usage_missing_attempts':missing,
  'answer_path_wall_seconds_sum':sum(times) if times or not keys else None,
  'missing_wall_time_attempts':len(keys)-len(times),
  'wall_time_scope':'canonical answer path including generation and publication checks/cache work; not pure model forward latency',
  'external_inference_calls':0}

def arm_allocations(rows,owner_keys,outcomes):
 owners=defaultdict(set)
 for row in rows:
  if row['canonical_key'] in owner_keys:owners[row['canonical_key']].add(row['arm'])
 result={arm:{'allocated_attempt_equivalents':0.,'allocated_observed_tokens':{f:0. for f in TOKENS},
  'missing_usage_attempt_equivalents':{f:0. for f in TOKENS},'allocated_observed_answer_path_wall_seconds':0.,
  'missing_wall_attempt_equivalents':0.} for arm in sorted({r['arm'] for r in rows})}
 for key,arms in owners.items():
  share=1/len(arms);outcome=outcomes[key]
  for arm in arms:
   value=result[arm];value['allocated_attempt_equivalents']+=share
   for field in TOKENS:
    number=(outcome.get('usage') or {}).get(field)
    if type(number) is int and number>=0:value['allocated_observed_tokens'][field]+=number*share
    else:value['missing_usage_attempt_equivalents'][field]+=share
   seconds=outcome.get('generation_wall_seconds')
   if type(seconds) in (int,float) and math.isfinite(seconds) and seconds>=0:value['allocated_observed_answer_path_wall_seconds']+=seconds*share
   else:value['missing_wall_attempt_equivalents']+=share
 for value in result.values():
  count=value['allocated_attempt_equivalents']
  if count:
   for field in TOKENS:
    if value['missing_usage_attempt_equivalents'][field]==count:value['allocated_observed_tokens'][field]=None
   if value['missing_wall_attempt_equivalents']==count:value['allocated_observed_answer_path_wall_seconds']=None
 return {'rule':'equal allocation across distinct arms owning the same canonical receipt; accounting, not independently measured per-arm wall time','arms':result}

def _database_counts(path):
 # Plain SQLite read-only avoids DurableMemoryStore.open side effects/migrations.
 connection=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)
 try:
  connection.execute('PRAGMA query_only=ON')
  tables={r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
  counts={t:connection.execute('SELECT COUNT(*) FROM '+t).fetchone()[0]
   for t in ('dm_sources','dm_jobs','dm_objects','dm_effects','dm_dependencies','memories') if t in tables}
  attempts=[{'purpose':p,'status':s,'count':n} for p,s,n in connection.execute(
   'SELECT purpose,status,COUNT(*) FROM dm_attempts GROUP BY purpose,status')]
  jobs=dict(connection.execute('SELECT status,COUNT(*) FROM dm_jobs GROUP BY status'))
  kinds=dict(connection.execute("SELECT json_extract(payload_json,'$.kind'),COUNT(*) FROM dm_objects GROUP BY json_extract(payload_json,'$.kind')"))
  task_statuses={field:dict(connection.execute("SELECT json_extract(payload_json,?),COUNT(*) FROM dm_objects WHERE json_extract(payload_json,'$.kind')='task' GROUP BY json_extract(payload_json,?)",(path,path)))
   for field,path in ((name,'$.projection.'+name) for name in ('acceptance','participation','outcome','eligibility'))}
  candidate_statuses=[{'confirmation_state':c,'lifecycle_status':s,'count':n} for c,s,n in connection.execute("SELECT json_extract(payload_json,'$.record.confirmation_state'),json_extract(payload_json,'$.record.lifecycle_status'),COUNT(*) FROM dm_objects WHERE json_extract(payload_json,'$.kind')='candidate' GROUP BY 1,2")]
  return {'table_counts':counts,'attempt_counts':attempts,'job_status_counts':jobs,
   'object_kind_counts':{str(k):v for k,v in kinds.items()},'task_projection_status_counts':task_statuses,
   'candidate_status_counts':candidate_statuses,'source_stage_wall_seconds':None,
   'source_stage_timing_status':'not_measured_in_original_preparation'}
 finally:connection.close()

def collect(model_root,prepared_dirs,output_dirs):
 r=_reader();r.require_colab()  # Before reading actual model/data artifacts.
 model_root=Path(model_root).resolve();prepared_dirs=[Path(p).resolve() for p in prepared_dirs]
 output_dirs=[Path(p).resolve() for p in output_dirs];hashes={}
 def tracked(path):
  path=Path(path).resolve();before=r.file_hash(path);data=r.read_json(path)
  if r.file_hash(path)!=before:raise ValueError('resource JSON changed during read')
  hashes[str(path)]=before;return data
 def tracked_rows(path):
  path=Path(path).resolve()
  if not path.exists():return []
  before=r.file_hash(path);rows=r.read_rows(path)
  if r.file_hash(path)!=before:raise ValueError('resource journal changed during read')
  hashes[str(path)]=before;return rows
 ledger=[r._checked(x) for x in tracked_rows(model_root/'generation_ledger.jsonl')]
 intents={};outcomes={}
 for row in ledger:
  kind=row.get('event');target=intents if kind=='generation_intent' else outcomes if kind=='generation_outcome' else None
  if target is None or row['canonical_key'] in target:raise ValueError('duplicate or unknown global event')
  target[row['canonical_key']]=row
 if set(outcomes)-set(intents):raise ValueError('unbound global outcome')
 for row in outcomes.values():
  if row.get('status')=='published':
   r._cache(row);hashes[str(Path(row['cache_path']).resolve())]=r.file_hash(row['cache_path'])
 model_files=tracked(model_root/'model_files.json');initial=tracked(model_root/'model_load_receipt.json')
 tracked(model_root/'load_protocol.json')
 setups=[r._checked(x) for x in tracked_rows(model_root/'model_load_events.jsonl')]
 setup_counts=Counter(x.get('event','unknown') for x in setups)
 setup_rows=[{k:x[k] for k in ('event','setup_id','at_utc','setup_seconds','observed_setup_seconds','failure_stage','failure_type','answer_generations_during_setup','local_files_only') if k in x} for x in setups]
 runs=[];referenced=set()
 for folder in output_dirs:
  run=tracked(folder/'run_protocol.json');completion=tracked(folder/'completion_summary.json')
  if completion.get('complete') is not True or completion.get('answers_sha256')!=r.file_hash(folder/'answers_unscored.jsonl') or completion.get('run_protocol_sha256')!=r.digest(run):raise ValueError('completed reader archive not bound')
  for filename,key in (('model_files.json','model_files_sha256'),('load_protocol.json','load_protocol_sha256'),('model_load_receipt.json','load_receipt_sha256')):
   if r.file_hash(model_root/filename)!=run['model'][key]:raise ValueError('model setup identity changed')
  if run['model'].get('model_id')!=r.MODEL_ID or run['model'].get('model_revision')!=r.MODEL_REVISION or run.get('external_inference_calls')!=0:raise ValueError('unexpected reader model or external-call policy')
  prepared=Path(run['prepared_dir']).resolve()
  if prepared not in prepared_dirs:raise ValueError('reader preparation not supplied')
  protocol=tracked(prepared/'protocol.json')
  if run['prepared_protocol_sha256']!=r.file_hash(prepared/'protocol.json') or run['requests_file_sha256']!=r.file_hash(prepared/'requests.jsonl'):raise ValueError('prepared identity differs')
  requests=tracked_rows(prepared/'requests.jsonl');bykey={(x['case_id'],x['arm']):x for x in requests}
  if len(bykey)!=len(requests):raise ValueError('duplicate planned logical row')
  answers=[r._checked(x) for x in tracked_rows(folder/'answers_unscored.jsonl')];seen=set()
  for a in answers:
   logical=(a['case_id'],a['arm']);req=bykey.get(logical)
   if logical in seen or req is None or a.get('source_checks_passed') is not True or a.get('run_protocol_sha256')!=r.digest(run):raise ValueError('unbound logical answer')
   key=r.digest({'model':run['model'],'payload':req['request_payload'],'max_new_tokens':req['request_payload']['max_tokens'],'do_sample':False})
   if a['canonical_key']!=key or a['request_sha256']!=req['request_sha256'] or outcomes.get(key,{}).get('status')!='published':raise ValueError('canonical answer binding differs')
   cache=r._cache(outcomes[key])
   if any(cache.get(k)!=a.get(k) for k in ('response_text','response_sha256','finish_reason')):raise ValueError('answer differs from model cache')
   seen.add(logical)
  if seen!=set(bykey) or completion.get('logical_completed')!=len(answers):raise ValueError('incomplete logical grid')
  keys={a['canonical_key'] for a in answers};referenced|=keys
  owned={k for k,x in intents.items() if Path(x.get('output','/invalid')).resolve()==folder}
  invocation_rows=[r._checked(x) for x in tracked_rows(folder/'invocations.jsonl')]
  failure_rows=[r._checked(x) for x in tracked_rows(folder/'failures.jsonl')]
  runs.append({'output_dir':str(folder),'prepared_dir':str(prepared),'purpose':protocol.get('purpose',protocol.get('material','not_recorded')),'logical_rows':len(answers),
   'unique_referenced_canonical_receipts':len(keys),'owned_generation_cost':costs(owned,intents,outcomes),
   'referenced_cost_nonadditive':costs(keys,intents,outcomes),
   'reused_canonical_receipts':len(keys-owned),'per_arm_new_owned_allocation':arm_allocations(answers,owned,outcomes),
   'reader_invocations':len(invocation_rows),'failure_rows':len(failure_rows)})
 preparations=[]
 for folder in prepared_dirs:
  protocol=tracked(folder/'protocol.json');requests=tracked_rows(folder/'requests.jsonl')
  if protocol['requests_file_sha256']!=r.file_hash(folder/'requests.jsonl'):raise ValueError('preparation requests changed')
  case_file=folder/'generation_cases.json'
  if r.file_hash(case_file)!=protocol['case_file_sha256']:raise ValueError('source case file changed')
  hashes[str(case_file)]=r.file_hash(case_file)
  paths=sorted({x['prepared_case_path'] for x in requests});worker=Counter();source=Counter();totals=Counter();attempts=Counter();jobs=Counter();kinds=Counter();candidates=Counter();task_statuses=defaultdict(Counter);source_only=True
  for name in paths:
   path=Path(name).resolve();expected=protocol['prepared_files_sha256'][str(path.relative_to(folder))]
   if r.file_hash(path)!=expected:raise ValueError('prepared case changed')
   case=tracked(path);trace=case['parent_trace']
   database=Path(case['database']).resolve()
   for artifact in (database,Path(str(database)+'-wal')):
    if artifact.exists():hashes[str(artifact)]=r.file_hash(artifact)
   db=_database_counts(database)
   source_only &= case.get('no_llm_extraction') is True and trace.get('semantic_extraction_mode')=='source_only_no_model'
   for s in trace['source_receipts']:source[s['receipt'].get('source_status','unknown')]+=1
   source[trace['current_source_receipt'].get('source_status','unknown')]+=1
   for x in trace['worker_results']:worker[(x.get('status','unknown'),x.get('semantic_status','missing'))]+=1
   totals.update(db['table_counts'])
   jobs.update(db['job_status_counts']);kinds.update(db['object_kind_counts'])
   for x in db['candidate_status_counts']:candidates[(x['confirmation_state'],x['lifecycle_status'])]+=x['count']
   for field,counts in db['task_projection_status_counts'].items():task_statuses[field].update({str(k):v for k,v in counts.items()})
   for x in db['attempt_counts']:attempts[(x['purpose'],x['status'])]+=x['count']
  extraction_attempts=sum(n for (purpose,_),n in attempts.items() if purpose=='extraction')
  no_extraction=source_only and extraction_attempts==0
  preparations.append({'prepared_dir':str(folder),'prepared_cases':len(paths),'source_receipt_statuses':dict(source),
   'worker_statuses':[{'status':k[0],'semantic_status':k[1],'count':v} for k,v in worker.items()],
   'database_table_counts':dict(totals),'durable_attempt_counts':[{'purpose':k[0],'status':k[1],'count':v} for k,v in attempts.items()],
   'job_status_counts':dict(jobs),'object_kind_counts':dict(kinds),
   'candidate_status_counts':[{'confirmation_state':k[0],'lifecycle_status':k[1],'count':v} for k,v in candidates.items()],
   'task_projection_status_counts':{k:dict(v) for k,v in task_statuses.items()},
   'extraction_model_attempts':extraction_attempts,'extraction_tokens':{k:0 if no_extraction else None for k in TOKENS},
   'source_only_no_model_proved':no_extraction,
   'source_ingestion_wall_seconds':None,'source_only_worker_wall_seconds':None,'indexing_organization_wall_seconds':None,
   'state_projection_wall_seconds':None,'retrieval_packing_wall_seconds':None,
   'preparation_latency_status':'not_measured_in_original_run; never infer from created_at timestamps'})
 for name,expected in hashes.items():
  if r.file_hash(name)!=expected:raise ValueError('resource input changed during read-only audit')
 purpose_keys=defaultdict(set)
 for key,intent in intents.items():purpose_keys[intent.get('purpose','not_recorded_in_global_intent')].add(key)
 return {'version':VERSION,'global_unique_generation_cost':costs(set(intents),intents,outcomes),
  'global_cost_by_explicit_intent_purpose':{p:costs(keys,intents,outcomes) for p,keys in purpose_keys.items()},
  'supplied_runs_unique_generation_cost':costs(referenced,intents,outcomes),
  'other_global_attempt_cost':costs(set(intents)-referenced,intents,outcomes),
  'runs':runs,'preparations':preparations,
  'model_setup':{'initial_successful_load_receipt_present':True,'unrecorded_initial_setup_attempts':None,'initial_receipt_observed_fields':{k:v for k,v in initial.items() if ('seconds' in k or 'bytes' in k or 'rss' in k) and type(v) in (int,float)},
   'model_file_bytes':sum(x.get('bytes',0) for x in model_files.get('files',[])),
   'reload_event_counts':dict(setup_counts),'reload_events':setup_rows,
   'setup_cost_is_separate_from_answer_cost':True},
  'model_and_preparation_costs_are_shared_not_independently_attributed_to_arms':True,
  'unlisted_preparation_costs':'not included; list every preparation directory to extend scope',
  'monetary_cost':None,'human_reviews':0,'new_model_calls_by_audit':0,'new_provider_calls':0,
  'scoring_targets_read':False,'all_read_artifacts_unchanged':True,'input_sha256':hashes,
  'auditor_code_sha256':r.file_hash(__file__),
  'attempt_counter_semantics':'admitted generation intents; a failed/unknown intent is not necessarily a completed model.generate call'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--model-root',required=True)
 p.add_argument('--prepared-dir',action='append',required=True);p.add_argument('--reader-output',action='append',required=True)
 p.add_argument('--output',required=True);a=p.parse_args();result=collect(a.model_root,a.prepared_dir,a.reader_output)
 out=Path(a.output)
 with out.open('x',encoding='utf-8') as f:f.write(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:result[k] for k in ('version','global_unique_generation_cost','supplied_runs_unique_generation_cost','other_global_attempt_cost','new_model_calls_by_audit','all_read_artifacts_unchanged')},sort_keys=True),flush=True)
if __name__=='__main__':main()
