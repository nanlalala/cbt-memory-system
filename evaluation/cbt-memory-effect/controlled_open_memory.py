"""New bounded synthetic task-state comparison for a real local open reader.

No paid provider, LLM extraction, gold injection or clinical assessment. The
actual product rules/projector manage explicit dated reports. Both conditions
may reason from the same allowed originals, recent-six, source cap and budget.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
VERSION='controlled-open-reader-state-v2-stable-wire-task-aliases'
ARMS=('raw_history_rag','managed_memory_rag')
SOURCE_CAP=9
RECENT=6
MAX_CHARS=6000
MAX_OUTPUT=64
PERMISSIONS={k:True for k in ('save','long_term_use','response_external','extraction_external')}
INSTRUCTION=('Answer the current factual question from the explicit dated user reports. '
 'You may reconstruct current reported state from original quotations. '
 'A later explicit correction replaces the matching earlier value. Cumulative totals are not '
 'separate attempts and repeated totals must not be added. A plan/proposal is not agreement. '
 'Reported progress does not establish benefit. Return exactly one JSON object with the single '
 'key answer: an integer for counts, a lowercase status word for outcome, or null if unknown. '
 'Do not include explanations, markdown, source IDs or additional keys.')

def stable(x):
 return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def filehash(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def writejson(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f: f.write(canonical(x)+'\n');f.flush()
def loadlegacy():
 name='controlled_open_parent_adapter'
 if name not in sys.modules:
  spec=importlib.util.spec_from_file_location(name,HERE/'controlled_replay_adapter.py')
  m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
 return sys.modules[name]
def product():
 m=loadlegacy()._modules()
 for name in ('dialogue_service','durable_memory','evidence_bundles','extraction_contract',
  'memory_contract','memory_manager','memory_retriever','natural_memory_policy',
  'sqlite_store','stable_retrieval','task_projector','task_input_rules','knowledge_adapter_v2',
  'source_candidate_selection'):
  module=sys.modules.get(name)
  try:Path(module.__file__).resolve().relative_to((ROOT/'modules').resolve())
  except (AttributeError,ValueError):raise RuntimeError('foreign cached product module: '+name+'; use fresh product imports') from None
 return m

def identity():
 paths=[Path(__file__),HERE/'controlled_replay_adapter.py']+sorted((ROOT/'modules').rglob('*.py'))
 return {'version':VERSION,'source_files':{str(p.relative_to(ROOT)):filehash(p) for p in paths}}

def fixed_materials():
 """Predeclared invented source templates and separate literal reference labels.

 Twenty-four bilingual cases have FOUR related template families, not patients. The
 expected answers below are independent fixed construction values, never read
 from product projections. Do not change after viewing model outputs.
 """
 cases=[];targets=[]
 for kind in ('target_correction','cumulative_repeat','proposal_not_agreement','outcome_not_progress'):
  for variant in range(1,4):
   n=variant+1
   if kind=='target_correction':
    texts=[('user','我同意每周做7次想法记录。'),
     ('assistant','We will keep the earlier plan available as a dated report.'),
     ('user','想法记录每周改成'+str(n)+'次。'),
     ('assistant','The correction is recorded as your current explicit report.')]
    query='What is my currently agreed target count for 想法记录 this week?';answer=n
    evidence=['turn-000','turn-002']
   elif kind=='cumulative_repeat':
    texts=[('user','我同意每周做7次想法记录。'),
     ('assistant','We can keep progress separate from the intended target.'),
     ('user','想法记录累计做了1次。'),
     ('assistant','That was a cumulative progress report.'),
     ('user','想法记录累计做了1次。'),
     ('assistant','This repeated total does not describe a new attempt.'),
     ('user','想法记录累计做了'+str(n)+'次。'),
     ('assistant','We can follow the latest explicit total.')]
    query='How many 想法记录 practices have I completed so far this week?';answer=n
    evidence=['turn-006']
   elif kind=='proposal_not_agreement':
    texts=[('assistant','A thought record practice of '+str(n)+' times is one possible suggestion.'),
     ('user','我计划做'+str(n)+'次想法记录。'),
     ('assistant','That describes a proposal, not proof of an agreed plan or completion.'),
     ('user','I am still considering the details.')]
    query='What is my explicitly agreed target count for 想法记录 this week?';answer=None
    evidence=['turn-000','turn-001']
   else:
    outcome=('helpful','unhelpful','mixed')[variant-1]
    texts=[('user','我同意每周做6次想法记录。'),
     ('assistant','The intended count and the observed outcome are different.'),
     ('user','想法记录累计做了'+str(n)+'次。'),
     ('assistant','The practice count is a progress report.'),
     ('user',{'helpful':'想法记录对我有帮助。','unhelpful':'想法记录对我没有帮助。','mixed':'想法记录有好有坏。'}[outcome]),
     ('assistant','That is a separate report of your experience.')]
    query='What outcome have I explicitly reported for 想法记录 practice?';answer=outcome
    evidence=['turn-004']
   # A shared direct recent buffer contains ordinary unrelated reflection, so
   # the test needs earlier evidence without manufacturing a missing history.
   texts += [('user','The weather was calm while I wrote in my journal.'),
    ('assistant','Thank you for sharing the daily context.'),
    ('user','I had tea and read a short article in the afternoon.'),
    ('assistant','We can keep that everyday context in mind.'),
    ('user','I am ready for a brief factual recap of the earlier discussion.'),
    ('assistant','Please ask the factual question you want to revisit.')]
   start=datetime(2026,11,3,8,tzinfo=timezone.utc)
   history=[{'message_id':f'turn-{i:03d}','role':role,'content':text,
             'date':(start+timedelta(hours=i)).isoformat(),'source_order':i}
             for i,(role,text) in enumerate(texts)]
   translations={'我同意每周做7次想法记录。':'I agreed to do thought record 7 times this week.',
    '我同意每周做6次想法记录。':'I agreed to do thought record 6 times this week.',
    '想法记录每周改成'+str(n)+'次。':'The thought record target is corrected to '+str(n)+' times this week.',
    '想法记录累计做了1次。':'I completed thought record, cumulative total 1 times.',
    '想法记录累计做了'+str(n)+'次。':'I completed thought record, cumulative total '+str(n)+' times.',
    '我计划做'+str(n)+'次想法记录。':'I plan to try thought record '+str(n)+' times this week.',
    '想法记录对我有帮助。':'The thought record outcome is helpful.',
    '想法记录对我没有帮助。':'The thought record outcome is unhelpful.',
    '想法记录有好有坏。':'The thought record outcome is mixed.'}
   for language in ('zh','en'):
    cid=f'controlled-{kind}-{language}-{variant}'
    rendered=[{**h,'content':translations.get(h['content'],h['content'])} for h in history] if language=='en' else history
    question=query.replace('想法记录','thought record') if language=='en' else query
    cases.append({'case_id':cid,'source_family_id':'template-'+kind,'calendar_time_available':True,
     'language':language,'history':rendered,'input':question,'input_message_id':'current-question',
     'input_date':(start+timedelta(hours=len(history))).isoformat()})
    targets.append({'case_id':cid,'source_family_id':'template-'+kind,'answer':answer,'language':language,
     'required_message_ids':evidence,'material':'invented_controlled_mechanism_not_natural_CBT',
     'template_variant':variant})
 return cases,targets

def _task_wire_aliases(session):
 """Deterministic view names; actual durable IDs remain authoritative off-wire.

 The stable key includes the first explicit source event, so distinct new task
 instances for the same user, activity and period do not collapse together.
 A task without such a source witness cannot acquire a guessed stable identity.
 """
 aliases={};audit=[]
 for obj in session._objects('task'):
  payload=obj['payload'];tid=payload['task_instance_id'];events=payload.get('events',[])
  if not events:continue
  first=min(events,key=lambda e:(e['reported_at'],e['event_id']))
  key={'user_id':session.user_id,'goal_id':payload['goal_id'],
   'object_label':payload['object_label'],'period_id':payload['period_id'],
   'initial_source_id':first['source_id'],'initial_event_id':first['event_id']}
  alias='task-view-'+stable(key)[:24]
  if alias in aliases.values():raise ValueError('stable task view key is ambiguous')
  aliases[tid]=alias
  audit.append({'actual_task_instance_id':tid,'wire_alias':alias,'stable_key_sha256':stable(key)})
 return aliases,audit

def _alias_bundle_ids(bundles,aliases):
 """Translate identifier metadata only; literal text/value/source IDs stay exact."""
 def translated(identifier):
  if identifier is None:return None
  for actual,alias in sorted(aliases.items()):identifier=identifier.replace(actual,alias)
  return identifier
 def span(value):return replace(value,span_id=translated(value.span_id))
 result=[]
 for b in bundles:
  claims=[]
  for c in b.claims:
   claims.append(replace(c,claim_id=translated(c.claim_id),
    spans=tuple(span(v) for v in c.spans),
    evidence_options=tuple(tuple(span(v) for v in option) for option in c.evidence_options)))
  result.append(replace(b,bundle_id=translated(b.bundle_id),
   task_instance_id=translated(b.task_instance_id),claims=tuple(claims)))
 return result

def _cap_bundles(bundles):
 # Whole product bundles, including all field witnesses, are admitted atomically.
 out=[];used=set();dropped=[]
 for b in sorted(bundles,key=lambda b:-max((c.priority for c in b.claims),default=0)):
  ids={s.source_id for c in b.claims for s in c.spans}
  if len(used|ids)>SOURCE_CAP:
   dropped.append({'bundle_id':b.bundle_id,'reason':'common_distinct_source_cap'});continue
  used|=ids;out.append(b)
 return out,dropped

def prepare_one(case,directory,*,fixture_only=False):
 legacy=loadlegacy();m=product()
 database=Path(directory)/'state.sqlite'
 # Reuse source ingestion, source-only bounded worker and actual dated rules.
 parent=legacy.prepare_contexts(case,database=database,approved_root=directory,
  permissions=PERMISSIONS,config=legacy.ReplayConfig(max_outbound_chars=18000,
   max_response_tokens=MAX_OUTPUT,maximum_response_calls=3),fixture_only=fixture_only)
 trace=parent['trace'];uid=trace['user_id'];stamp=case['input_date']
 session=m.ControlledDialogueSession.open(database,approved_root=directory,user_id=uid,
  run_id=trace['run_id'],session_id=trace['session_id'],provider=legacy._ContextOnlyProvider(),
  background=False,permissions=PERMISSIONS,
  limits=m.PrototypeLimits(**trace['product_protocol']['limits']),
  durable_limits=m.DurableLimits(**trace['product_protocol']['durable_limits']))
 try:
  pool=session.durable.list_sources(uid,as_of=stamp)
  current=next(s for s in pool if s['source_id']==trace['current_source_id'])
  paused,_=session._paused_source_ids(stamp)
  history=[s for s in pool if s['source_id']!=current['source_id'] and s['source_id'] not in paused]
  recent=sorted(history,key=lambda s:(s['source_time'],s['source_id']))[-RECENT:]
  recent_ids={s['source_id'] for s in recent}
  older=[s for s in history if s['source_id'] not in recent_ids]
  rawsearch=m.StableSourceRetriever(history).search(case['input'],user_id=uid,source_cutoff=stamp,
   as_of=stamp,for_external_response=True,allowed_source_ids={s['source_id'] for s in older},top_k=SOURCE_CAP)
  raw=legacy._quote_bundles(rawsearch.hits)
  task=session._task_for_input(case['input'],stamp)
  managed,managedtrace,state=session._bundles(case['input'],as_of=stamp,current_source=current,task=task)
  # Give both arms the same capable original ranking without wasting slots
  # repeating the direct recent buffer. Product state bundles are the treatment.
  state_bundles=[b for b in managed if any(c.field!='history_quote' for c in b.claims)]
  raw_keys={(x.source_id,x.start,x.end) for x in rawsearch.hits}
  extras=[b for b in managed if b.bundle_id.startswith('semantic:') and
   all(s.source_id not in recent_ids and (s.source_id,s.start,s.end) not in raw_keys for c in b.claims for s in c.spans)]
  managed=legacy._preserve_managed_quote_rank(state_bundles+raw+extras,rawsearch.hits)
  # Exactly the same direct original recent messages and source/date map.
  recent_meta=[{'source_id':s['source_id'],'source_time':s['source_time'],'role':s['role']} for s in recent]
  base=[{'role':'system','content':m.SYSTEM+'\n'+INSTRUCTION+'\nAs of: '+stamp+
    '\nRecent original source map: '+canonical(recent_meta)}]
  base += [{'role':s['role'],'content':s['text']} for s in recent]
  base += [{'role':'user','content':case['input']}]
  contexts={}
  aliases,alias_audit=_task_wire_aliases(session)
  for arm,bundles in zip(ARMS,(raw,managed)):
   capped,capdrops=_cap_bundles(_alias_bundle_ids(bundles,aliases))
   package=m.build_outbound_messages(base,capped,max_outbound_chars=MAX_CHARS,
    max_response_tokens=MAX_OUTPUT,request_metadata={'model':'Qwen2.5-1.5B-Instruct','temperature':0},usage=None)
   legacy.assert_evidence_integrity(package,pool)
   context={'messages':list(package.messages),'request_payload':package.request_payload,
    'serialized_request':package.serialized_request,'request_sha256':stable(package.request_payload),
    'outbound_chars':package.char_count,'source_ids':list(package.source_ids),
    'included_claim_ids':list(package.included_claim_ids),'source_cap_dropped':capdrops,
    'packing_dropped':package.dropped_claims}
   deps=sorted(set(context['source_ids']+[current['source_id']]+[s['source_id'] for s in recent]))
   context['binding']={'version':VERSION,'user_id':uid,'run_id':VERSION,'session_id':trace['session_id'],
    'generation':session.durable.current_generation(uid),'source_ids':deps,
    'current_source_id':current['source_id'],'recent_sources':recent_meta,
    'expected_task_versions':{task['object_id']:task['version']} if task and state else {},
    'request_sha256':context['request_sha256'],'maximum_chars':MAX_CHARS,'maximum_output_tokens':MAX_OUTPUT,
    'maximum_response_calls':2}
   contexts[arm]=context
  with session.store.transaction(immediate=True):
   session.store.connection.execute('CREATE TABLE open_context_snapshots '
    '(user_id TEXT, request_sha256 TEXT, binding_json TEXT NOT NULL, PRIMARY KEY(user_id,request_sha256))')
   for c in contexts.values():
    session.store.connection.execute('INSERT OR IGNORE INTO open_context_snapshots VALUES (?,?,?)',
     (uid,c['request_sha256'],canonical(c['binding'])))
  result={'version':VERSION,'case_id':case['case_id'],'source_family_id':case['source_family_id'],
   'contexts':contexts,'parent_trace':trace,'code_identity':identity(),
   'recent_messages':base[1:-1],'database':str(database.resolve()),'approved_root':str(Path(directory).resolve()),
   'actual_state_projection_present':any(b.bundle_id.startswith('task:') for b in managed),
   'wire_task_aliases':alias_audit,
   'no_llm_extraction':True,'actual_new_model_calls':0}
  writejson(Path(directory)/'prepared_case.json',result)
  return result
 finally: session.close()

def prepare(output,*,fixture_only=False,case_limit=None):
 legacy=loadlegacy()
 if not fixture_only:
  sys.path.insert(0,str(ROOT/'app'));from colab_provider import require_colab
  require_colab()
 out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
 cases,targets=fixed_materials()
 if case_limit is not None: cases,targets=cases[:case_limit],targets[:case_limit]
 # Freeze all source cases and independent literal targets before any preparation.
 writejson(out/'generation_cases.json',cases)
 writejson(out/'assessor_only'/'targets.json',targets)
 rows=[]; diagnostics=[]
 for case in cases:
  case_dir=out/'cases'/case['case_id'];case_dir.mkdir(parents=True)
  prepared=prepare_one(case,case_dir,fixture_only=fixture_only)
  for arm in ARMS:
   c=prepared['contexts'][arm]
   rows.append({'case_id':case['case_id'],'source_family_id':case['source_family_id'],'arm':arm,
    'messages':c['messages'],'request_payload':c['request_payload'],'request_sha256':c['request_sha256'],
    'source_dependencies':c['binding']['source_ids'],'prepared_case_path':str((case_dir/'prepared_case.json').resolve()),
    'database_path':prepared['database']})
   diagnostics.append({'case_id':case['case_id'],'arm':arm,'outbound_chars':c['outbound_chars'],
    'retrieved_source_count':len(c['source_ids']),'state_projection_present':prepared['actual_state_projection_present']})
 with (out/'requests.jsonl').open('x') as f:
  for row in rows:f.write(canonical(row)+'\n')
 protocol={'version':VERSION,'material':'invented_controlled_mechanism_not_natural_CBT',
  'independent_human_review':False,'clinical_quality_claim':False,'case_count':len(cases),
  'template_family_count':len({c['source_family_id'] for c in cases}),'arms':list(ARMS),
  'recent_messages':RECENT,'retrieved_source_cap_each_arm':SOURCE_CAP,'max_outbound_chars':MAX_CHARS,
  'recent_sources_do_not_consume_raw_retrieval_slots':True,
  'raw_and_managed_share_original_source_ranking':True,
  'task_wire_alias_policy':'stable_user_goal_activity_period_initial_explicit_event_v1',
  'actual_durable_task_ids_retained_in_private_audit':True,
  'max_new_tokens':MAX_OUTPUT,'no_llm_extraction':True,'new_provider_calls':0,
  'case_file_sha256':filehash(out/'generation_cases.json'),
  'targets_file_sha256':filehash(out/'assessor_only'/'targets.json'),
  'requests_file_sha256':filehash(out/'requests.jsonl'),
  'prepared_files_sha256':{str(Path(r['prepared_case_path']).relative_to(out)):filehash(r['prepared_case_path']) for r in rows},
  'code_identity':identity(),'gold_read_by_preparer_after_freeze':False,
  'expected_answer_source':'fixed_materials literals, not product projection',
  'diagnostics':diagnostics}
 writejson(out/'protocol.json',protocol)
 return {'version':VERSION,'case_count':len(cases),'request_rows':len(rows),
  'template_families':protocol['template_family_count'],'new_model_calls':0,
  'requests_file_sha256':protocol['requests_file_sha256'],'source_scope':protocol['material']}

class Gates:
 def __init__(self,prepared_dir):
  self.root=Path(prepared_dir).resolve();self.protocol=json.loads((self.root/'protocol.json').read_text())
  self.rows=[json.loads(l) for l in (self.root/'requests.jsonl').read_text().splitlines() if l]
 def identity(self):return identity()
 def _prepared(self,row):
  if self.protocol['code_identity']!=identity():raise ValueError('gate code changed')
  if filehash(self.root/'requests.jsonl')!=self.protocol['requests_file_sha256']:raise ValueError('request file changed')
  expected=next((r for r in self.rows if r['case_id']==row['case_id'] and r['arm']==row['arm']),None)
  if expected!=row:raise ValueError('request row changed')
  path=Path(row['prepared_case_path']).resolve();rel=str(path.relative_to(self.root))
  if filehash(path)!=self.protocol['prepared_files_sha256'][rel]:raise ValueError('prepared case changed')
  p=json.loads(path.read_text());c=p['contexts'][row['arm']]
  if c['request_payload']!=row['request_payload'] or c['messages']!=row['messages'] or stable(c['request_payload'])!=row['request_sha256']:raise ValueError('wire changed')
  return p,c
 def _store(self,p):
  legacy=loadlegacy();m=product()
  return m.DurableMemoryStore.open(p['database'],approved_root=p['approved_root'],
   run_limits=m.DurableLimits(**p['parent_trace']['product_protocol']['durable_limits']))
 def _check(self,p,c,d):
  b=c['binding'];m=product()
  frozen=d.store.connection.execute('SELECT binding_json FROM open_context_snapshots WHERE user_id=? AND request_sha256=?',(b['user_id'],b['request_sha256'])).fetchone()
  if not frozen or json.loads(frozen[0])!=b:raise ValueError('authoritative binding changed')
  wire=json.dumps(c['request_payload'],ensure_ascii=True,sort_keys=True,allow_nan=False)
  if wire!=c['serialized_request'] or len(wire)>b['maximum_chars'] or c['request_payload']['max_tokens']!=MAX_OUTPUT:raise ValueError('complete request budget changed')
  if d.current_generation(b['user_id'])!=b['generation']:raise ValueError('withdrawal generation changed')
  sources=[]
  for sid in b['source_ids']:
   s=d.get_source(b['user_id'],sid)
   if s is None or any(s['permissions'].get(k) is not True for k in ('save','long_term_use','response_external')):raise PermissionError('source no longer permitted')
   sources.append(s)
  byid={s['source_id']:s for s in sources};current=byid[b['current_source_id']]
  if c['messages'][-1]!={'role':'user','content':current['text']}:raise ValueError('question changed')
  expected_recent=[{'role':byid[x['source_id']]['role'],'content':byid[x['source_id']]['text']} for x in b['recent_sources']]
  if [x for x in c['messages'] if x['role']!='system'][:-1]!=expected_recent:raise ValueError('direct recent source changed')
  loadlegacy().assert_evidence_integrity(SimpleNamespace(messages=c['messages']),sources)
  for oid,v in b['expected_task_versions'].items():
   state=d.get_object_state(b['user_id'],oid)
   if state['status']!='available' or state['version']!=v:raise ValueError('task state version changed')
  return b
 def preflight(self,row):
  p,c=self._prepared(row);d=self._store(p)
  try:
   with d.store.transaction(immediate=True):return self._check(p,c,d)
  finally:d.close()
 def begin(self,row):
  p,c=self._prepared(row);d=self._store(p)
  try:
   with d.store.transaction(immediate=True):
    b=self._check(p,c,d);key='open-reader:'+b['request_sha256']
    count=d.store.connection.execute("SELECT COUNT(*) FROM dm_attempts WHERE user_id=? AND run_id=? AND purpose='response'",(b['user_id'],VERSION)).fetchone()[0]
    if d.get_attempt(b['user_id'],key,run_id=VERSION,session_id=b['session_id']) is None and count>=2:raise RuntimeError('case response cap exhausted')
    return d.begin_attempt(b['user_id'],key,'response',b['source_ids'],run_id=VERSION,
     session_id=b['session_id'],expected_generation=b['generation'],expected_object_versions=b['expected_task_versions'])
  finally:d.close()
 def finish(self,row,admission,*,outcome,response=None,usage=None):
  p,c=self._prepared(row);d=self._store(p)
  try:
   b=c['binding'];known=d.get_attempt(b['user_id'],'open-reader:'+b['request_sha256'],run_id=VERSION,session_id=b['session_id'])
   if not known or known['attempt_id']!=admission['attempt_id']:raise ValueError('foreign admission')
   return d.finish_attempt(admission['attempt_id'],outcome=outcome,response=response,usage=usage,retain_response=False)
  finally:d.close()
def make_gates(prepared_dir):return Gates(prepared_dir)

def _answer(text):
 def unique(pairs):
  out={}
  for k,v in pairs:
   if k in out:raise ValueError('duplicate answer field')
   out[k]=v
  return out
 obj=json.loads(text,object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
 if type(obj) is not dict or set(obj)!={'answer'}:raise ValueError('single answer key required')
 val=obj['answer']
 if val is not None and type(val) not in (int,str):raise ValueError('invalid answer type')
 return val

def _authenticate_answers(root,answers_file,model_root):
 name='controlled_open_reader_receipt_audit'
 spec=importlib.util.spec_from_file_location(name,HERE/'open_reader_colab.py')
 reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
 result_dir=Path(answers_file).resolve().parent
 run=reader.read_json(result_dir/'run_protocol.json')
 completion=reader.read_json(result_dir/'completion_summary.json')
 if run.get('prepared_protocol_sha256')!=filehash(root/'protocol.json') or run.get('requests_file_sha256')!=filehash(root/'requests.jsonl'):raise ValueError('reader protocol belongs to another prepared input')
 if run.get('reader_code_sha256')!=filehash(HERE/'open_reader_colab.py'):raise ValueError('reader code identity changed')
 run_hash=reader.digest(run)
 if completion.get('complete') is not True or completion.get('run_protocol_sha256')!=run_hash or completion.get('answers_sha256')!=filehash(answers_file):raise ValueError('completed answer archive changed or incomplete')
 intents,outcomes=reader._global_state(Path(model_root))
 authenticated=[]
 for raw in reader.read_rows(Path(answers_file)):
  a=reader._checked(raw)
  if a.get('run_protocol_sha256')!=run_hash:raise ValueError('answer reader protocol differs')
  receipt=outcomes.get(a.get('canonical_key'))
  if receipt is None:raise ValueError('answer lacks canonical generation receipt')
  cached=reader._cache(receipt)
  if any(cached.get(k)!=a.get(k) for k in ('response_text','response_sha256','finish_reason')):raise ValueError('answer differs from canonical model output')
  authenticated.append(a)
 return authenticated,run

def score(prepared_dir,answers_file,output,*,model_root):
 root=Path(prepared_dir);protocol=json.loads((root/'protocol.json').read_text())
 if filehash(root/'requests.jsonl')!=protocol['requests_file_sha256']:raise ValueError('requests changed')
 requests=[json.loads(l) for l in (root/'requests.jsonl').read_text().splitlines() if l]
 bykey={(r['case_id'],r['arm']):r for r in requests};answers={}
 preliminary=[json.loads(l) for l in Path(answers_file).read_text().splitlines() if l.strip()]
 if {(a.get('case_id'),a.get('arm')) for a in preliminary}!=set(bykey):raise ValueError('freeze all planned answers before opening targets')
 authenticated,reader_run=_authenticate_answers(root,answers_file,model_root)
 for a in authenticated:
  key=(a['case_id'],a['arm'])
  if key in answers or key not in bykey:raise ValueError('duplicate or unplanned answer')
  if a.get('status')!='completed' or a['request_sha256']!=bykey[key]['request_sha256'] or a.get('source_checks_passed') is not True:raise ValueError('unpublished or unbound answer')
  if hashlib.sha256(a['response_text'].encode()).hexdigest()!=a['response_sha256']:raise ValueError('answer bytes changed')
  expected_key=stable({'model':reader_run['model'],'payload':bykey[key]['request_payload'],'max_new_tokens':bykey[key]['request_payload']['max_tokens'],'do_sample':False})
  if a.get('canonical_key')!=expected_key:raise ValueError('canonical output belongs to another request')
  answers[key]=a
 if set(answers)!=set(bykey):raise ValueError('freeze all planned answers before opening targets')
 if filehash(root/'assessor_only'/'targets.json')!=protocol['targets_file_sha256']:raise ValueError('targets changed')
 targets={x['case_id']:x for x in json.loads((root/'assessor_only'/'targets.json').read_text())}
 rows=[]
 for key,req in bykey.items():
  target=targets[key[0]];a=answers[key]
  try:
   val=_answer(a['response_text']);correct=type(val) is type(target['answer']) and val==target['answer']
   reason=('correct' if correct else 'output_type_error' if type(val) is not type(target['answer'])
    else 'unsupported_assertion' if target['answer'] is None else 'incorrect_abstention' if val is None else 'wrong_value')
  except (ValueError,TypeError,json.JSONDecodeError):val=None;correct=False;reason='malformed_output'
  rows.append({'case_id':key[0],'arm':key[1],'source_family_id':target['source_family_id'],
   'language':target['language'],'correct':correct,'reason':reason,'expected_answer':target['answer'],
   'observed_answer':val,'request_sha256':req['request_sha256'],'response_sha256':a['response_sha256']})
 counts={arm:{'correct':sum(r['correct'] for r in rows if r['arm']==arm),
  'planned':sum(r['arm']==arm for r in rows),
  'errors':dict(Counter(r['reason'] for r in rows if r['arm']==arm and not r['correct']))} for arm in ARMS}
 pairs=[]
 for cid in targets:
  grid={r['arm']:r for r in rows if r['case_id']==cid};raw=grid[ARMS[0]]['correct'];managed=grid[ARMS[1]]['correct']
  pairs.append({'case_id':cid,'source_family_id':targets[cid]['source_family_id'],
   'language':targets[cid]['language'],'raw_correct':raw,'managed_correct':managed,
   'outcome':'managed_win' if managed and not raw else 'raw_win' if raw and not managed else 'tie'})
 report={'version':VERSION+'-strict-assessor-v1','material':protocol['material'],
  'case_count':len(targets),'template_family_count':len({r['source_family_id'] for r in rows}),
  'arms':counts,'paired_outcomes':dict(Counter(p['outcome'] for p in pairs)),
  'per_question':rows,'pairs':pairs,'new_model_calls':0,'human_reviews':0,
  'clinical_quality_claim':False,'population_effect_interval':None,
  'interpretation':'Actual reader answers on invented bilingual controlled task-state cases; four related templates, not independent clinical evidence.',
  'answers_file_sha256':filehash(answers_file),'protocol_file_sha256':filehash(root/'protocol.json'),
  'reader_run_fixture_only':reader_run.get('fixture_only'),
  'generation_receipts_independently_reconciled':True}
 writejson(output,report)
 return {k:v for k,v in report.items() if k not in {'per_question','pairs'}}

def main():
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
 a=sub.add_parser('prepare');a.add_argument('--output',required=True)
 a=sub.add_parser('score');a.add_argument('--prepared-dir',required=True);a.add_argument('--answers',required=True);a.add_argument('--output',required=True);a.add_argument('--model-root',required=True)
 args=p.parse_args()
 result=prepare(args.output) if args.command=='prepare' else score(args.prepared_dir,args.answers,args.output,model_root=args.model_root)
 print(canonical(result),flush=True)
if __name__=='__main__':main()
