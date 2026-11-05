#!/usr/bin/env python3
"""New, bounded two-arm Persona FULL-manager development experiment in Colab."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile

from persona_management_journal import Journal, BudgetExhausted, request_body, rows, stable

ROOT=Path(__file__).resolve().parents[2]
PLAN='940a2b3d8fa02aa683f7855f94931ff1d9710c7835b598c70fc201582ae8d568'
PLAN_FILE='dbd41e4ea0fdf0fff3d357ad05ef9d72e546a5594481db518a31e93a13ea2f2e'
METADATA_FILE='9837a6825337898a441002abccde583417188f4d7edf4317869c6fe7c0962e36'
OLD_EXECUTABLE='1511706b3a4b0ac71416998b5b4d752a13c4e313'
ARMS=('raw_history_rag','managed_memory_rag')
STATIC_FILES=('cases.jsonl','scope_plan.json','extraction_requests.jsonl','quote_control_gate.json')
INPUT_FILES=('protocol.json',*STATIC_FILES,'prepared_manifest.json','prepared_rows.jsonl','generation_rows.jsonl','requests.jsonl','attempts.jsonl','outcomes.jsonl','journal_cost.json','phase_status.json')

def require_colab():
    try: present=importlib.util.find_spec('google.colab') is not None
    except (ValueError,ModuleNotFoundError): present=False
    if not (Path('/content').is_dir() and os.environ.get('COLAB_RELEASE_TAG') and present):
        raise RuntimeError('Genuine hosted Colab required before experiment access')

def data(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file(): raise ValueError('Regular frozen artifact required')
    return path.read_bytes()

def digest(path):return hashlib.sha256(data(path)).hexdigest()
def read_json(path):return json.loads(data(path))
def legacy_stable(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()).hexdigest()

def atomic_json(path,value):
    body=(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode();atomic_bytes(path,body)

def atomic_rows(path,values):
    atomic_bytes(path,b''.join((json.dumps(v,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode() for v in values))

def atomic_bytes(path,body):
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:stream.write(body);stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)

def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()

def code_snapshot():
    names=git('ls-files','--','*.py','evaluation/objective-memory/dataset_registry.json').splitlines()
    return {name:digest(ROOT/name) for name in names}

@contextmanager
def lock(path,*,create=False):
    if path.is_symlink() or (path/'.execution.lock').is_symlink():raise ValueError('Symlinked run lock')
    stream=(path/'.execution.lock').open('a+b' if create else 'rb')
    try:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB if create else fcntl.LOCK_SH|fcntl.LOCK_NB);yield
    finally:stream.close()

def controlled(path):
    value=Path(path).resolve()
    if not value.is_relative_to(Path('/content')):raise ValueError('Controlled Colab paths required')
    return value

def retain_original_locks(protocol,stack):
    suite=Path(protocol['original_suite']);stack.enter_context(lock(suite))
    for directory in sorted({(suite/r['relative_path']).parent for r in protocol['reference_files']}):stack.enter_context(lock(directory))

def load_original(suite):
    """Read only generation artifacts; references are only named and hashed here."""
    suite=controlled(suite);watched={}
    with ExitStack() as stack:
        stack.enter_context(lock(suite))
        meta_path=suite.parent/'derived-final-20261105/public_metadata.json'
        if digest(suite/'suite_plan.json')!=PLAN_FILE or digest(meta_path)!=METADATA_FILE:raise ValueError('Original frozen identity differs')
        plan=read_json(suite/'suite_plan.json');meta=read_json(meta_path)
        if plan['suite_plan_sha256']!=PLAN or legacy_stable({k:v for k,v in plan.items() if k!='suite_plan_sha256'})!=PLAN or meta['execution_commit']!=OLD_EXECUTABLE or meta['suite_plan_sha256']!=PLAN:raise ValueError('Original executable/plan differs')
        watched[str(suite/'suite_plan.json')]=PLAN_FILE;watched[str(meta_path)]=METADATA_FILE
        cases=[];references=[];raw_prompts={};shards=[s for s in plan['shards'] if s['dataset_key']=='personamem_v1']
        if len(shards)!=4:raise ValueError('Expected four original Persona shards')
        for shard in shards:
            directory=suite/'shards'/shard['shard_id'];stack.enter_context(lock(directory))
            for name in ('protocol.json','cases.jsonl','inventory.jsonl','prompts.jsonl'):
                path=directory/name;expected=meta['source_artifacts_sha256'][str(path.relative_to(suite.parent))]
                if digest(path)!=expected:raise ValueError('Original source artifact differs')
                watched[str(path)]=expected
            protocol=read_json(directory/'protocol.json');manifest=protocol['dataset_manifest']
            if protocol['protocol_sha256']!=legacy_stable({k:v for k,v in protocol.items() if k!='protocol_sha256'}) or protocol['case_ids']!=shard['case_ids'] or manifest.get('suite_shard_id')!=shard['shard_id'] or manifest.get('suite_plan_sha256')!=PLAN:raise ValueError('Original shard identity differs')
            if any(manifest.get(k)!=v for k,v in plan['source_identities']['personamem_v1'].items()) or protocol['settings']!=plan['datasets']['personamem_v1']['settings']:raise ValueError('Original source/settings differ')
            for name in ('cases.jsonl','inventory.jsonl','prompts.jsonl'):
                if watched[str(directory/name)]!=protocol['file_sha256'].get(name):raise ValueError('Original source file identity differs')
            by_context={}
            for item in rows(directory/'inventory.jsonl'):
                source=item['source'];context=source['turn_id'].rsplit(':turn:',1)[0]
                key=(item['subject_id'],context);bucket=by_context.setdefault(key,{})
                if source['order'] in bucket and bucket[source['order']]!=source:raise ValueError('Contradictory source event')
                bucket[source['order']]=source
            compact=rows(directory/'cases.jsonl')
            if [c['case_id'] for c in compact]!=shard['case_ids']:raise ValueError('Original case grid differs')
            for case in compact:
                cutoff=case['visible_context_end_exclusive'];source_map=by_context[(case['subject_id'],f"persona32k:{case['source_context_id']}")]
                history=[source_map[i] for i in sorted(source_map) if i<cutoff]
                if [r['order'] for r in history]!=list(range(cutoff)) or legacy_stable([case['subject_id'],history])!=case['history_inventory_sha256']:raise ValueError('Original history boundary/hash differs')
                # Compact transport identity is verified above; it is not a
                # generation CASE_FIELDS member and must not reach raw adapter.
                cases.append({**{k:v for k,v in case.items() if k!='history_inventory_sha256'},'history':history})
            for prompt in rows(directory/'prompts.jsonl'):
                if prompt['arm']!='raw_history_rag':continue
                cid=prompt['case_id']
                if cid in raw_prompts or cid not in shard['case_ids'] or prompt['prompt_sha256']!=legacy_stable(prompt['messages']):raise ValueError('Original raw prompt identity differs')
                raw_prompts[cid]=prompt['messages']
            ref_relative=str((directory/'references.jsonl').relative_to(suite))
            expected=meta['source_artifacts_sha256'][str((directory/'references.jsonl').relative_to(suite.parent))]
            if expected!=protocol['file_sha256'].get('references.jsonl'):raise ValueError('Reference declared identity differs')
            references.append({'relative_path':ref_relative,'sha256':expected,'case_ids':shard['case_ids']})
        if len(cases)!=589 or len({c['case_id'] for c in cases})!=589 or len({c['group_id'] for c in cases})!=20 or len({str(c['source_context_id']) for c in cases})!=37:raise ValueError('Original Persona census differs')
        if any(digest(Path(p))!=h for p,h in watched.items()):raise ValueError('Source changed during read')
        if set(raw_prompts)!={c['case_id'] for c in cases}:raise ValueError('Original raw prompt census differs')
        return cases,plan['datasets']['personamem_v1']['settings'],watched,references,meta['code_sha256'],raw_prompts

def verify_protocol(run):
    protocol=read_json(run/'protocol.json')
    if protocol['protocol_sha256']!=stable({k:v for k,v in protocol.items() if k!='protocol_sha256'}):raise ValueError('New protocol seal differs')
    if git('rev-parse','HEAD')!=protocol['executable_commit'] or code_snapshot()!=protocol['code_sha256']:raise ValueError('Frozen executable/code changed')
    if any(digest(Path(p))!=h for p,h in protocol['original_source_sha256'].items()):raise ValueError('Original generation source changed')
    for name,h in protocol['prepared_file_sha256'].items():
        if digest(run/name)!=h:raise ValueError('Prepared generation artifact changed')
    prepared=read_json(run/'prepared_manifest.json')
    if prepared!={'protocol_sha256':protocol['protocol_sha256'],'prepared_file_sha256':protocol['prepared_file_sha256']}:raise ValueError('Prepared manifest differs')
    return protocol

def prepare(args):
    require_colab();suite=controlled(args.original_suite);run=controlled(args.run)
    if run.exists() or run.is_relative_to(suite) or run.is_relative_to(ROOT):raise ValueError('Fresh separate experiment directory required')
    if git('rev-parse','HEAD')!=args.executable_commit or git('status','--porcelain','--untracked-files=no'):raise ValueError('Committed clean executable checkout required')
    cases,settings,source_hashes,references,old_code,old_raw=load_original(suite)
    if settings['max_response_tokens']!=64 or settings['backend']!='lexical':raise ValueError('Original fixed Persona answer policy differs')
    from persona_management_adapter import prepare_scope_plan,extraction_requests,prepare_arms
    plan=prepare_scope_plan(cases,window_chars=8000,prior_context_chars=4000,chunk_chars=1200);extractions=extraction_requests(plan)
    ids=[r['window_id'] for r in extractions]
    if len(ids)!=len(set(ids)) or any(r['request_sha256']!=stable(r['messages']) for r in extractions):raise ValueError('Extraction window/prompt identity differs')
    for item in extractions:request_body(item['messages'],'extraction',4096)
    if type(args.maximum_provider_attempts) is not int or not 1178<=args.maximum_provider_attempts<=2500 or len(extractions)+1178>args.maximum_provider_attempts:raise ValueError('All-window plus 1178 fresh-answer maximum exceeds finite authorized cap')
    # Only modules actually recorded by the old run can be byte-compared.
    # The reviewed follow-up transport/guard are new code, separately frozen.
    current=code_snapshot()
    for name in ('evaluation/end-to-end-dialogue/run_three_arm.py','evaluation/objective-memory/score_objective_comparison.py'):
        if current.get(name)!=old_code.get(name):raise ValueError('Shared provider/official scorer changed from original')
    empty_outputs={r['window_id']:'{"candidates":[]}' for r in extractions}
    gate_rows=[]
    for case in cases:
        prepared=prepare_arms(case,plan=plan,extraction_outputs=empty_outputs,settings=settings)['prepared_by_arm']
        raw=prepared['raw_history_rag']['messages'];managed=prepared['managed_memory_rag']['messages']
        if raw!=old_raw[case['case_id']] or raw!=managed:raise ValueError('Zero-call quote-control parity failed; no model requests authorized')
        gate_rows.append({'case_id':case['case_id'],'prompt_sha256':stable(raw),'new_raw_equals_frozen_old_raw':True,'empty_semantic_manager_equals_raw':True})
    gate={'schema':'persona-full-manager-quote-control-gate-v1','case_count':len(gate_rows),'new_raw_equals_frozen_old_raw_cases':len(gate_rows),
          'empty_semantic_manager_equals_raw_cases':len(gate_rows),'case_rows':gate_rows,'new_provider_calls':0,
          'interpretation':'Source/context/retriever parity check only; empty synthetic extraction is not actual model extraction or management efficacy.'}
    if any(digest(Path(p))!=h for p,h in source_hashes.items()):raise ValueError('Original source changed during quote-only gate')
    run.mkdir(parents=True)
    with lock(run,create=True):
        atomic_rows(run/'cases.jsonl',cases);atomic_json(run/'scope_plan.json',plan);atomic_rows(run/'extraction_requests.jsonl',extractions);atomic_json(run/'quote_control_gate.json',gate)
        protocol={'schema':'persona-full-memory-management-development-v1','executable_commit':args.executable_commit,'code_sha256':current,
                  'original_suite':str(suite),'original_executable_commit':OLD_EXECUTABLE,'original_suite_plan_sha256':PLAN,'original_source_sha256':source_hashes,
                  'reference_files':references,'case_ids':[c['case_id'] for c in cases],'planned_cases':589,'planned_arm_rows':1178,'arms':list(ARMS),
                  'settings':settings,'source_window_policy':{'window_chars':8000,'prior_context_chars':4000,'chunk_chars':1200,'maximum_candidates_per_window':12,'question_prefix_boundaries':'all exclusive cutoffs; no extraction query/options/gold'},
                  'scope_plan_sha256':stable(plan),'extraction_windows':len(extractions),'maximum_provider_attempts':args.maximum_provider_attempts,
                  'output_limits':{'extraction':4096,'response':64},'workers':4,'maximum_attempts_per_canonical_request':1,
                  'deduplication_policy':'Exact purpose, messages and fixed model/settings share one NEW canonical response across logical rows; no original-run responses are reused.',
                  'dispatch_policy':'Frozen case order, raw then managed interleaved; four workers; selection never uses answers or scoring references.',
                  'generation_identity':{'model':'deepseek-v4.1-flash','accepted_provider_models':['deepseek-v4-1-flash-260910'],'api_base':'https://tokendance.space/gateway/v1','temperature':0,'thinking_mode':'disabled'},
                  'transport_policy':'New reviewed followup_transport/Colab guard frozen by current code hashes; original model/API/settings retained. Only old recorded shared run_three_arm and official scorer bytes are asserted identical.',
                  'prepared_file_sha256':{name:digest(run/name) for name in STATIC_FILES},'all_answer_arms_fresh':True,'old_response_reuse':False,
                  'fixture_only':False,'evaluation_policy':{'bootstrap_seed':42,'bootstrap_samples':10000,'primary_weighting':'equal connected source groups'},
                  'gold_policy':'References never read by preparation/extraction/answering; official scoring only after all 1178 terminal answers.',
                  'research_boundary':'Fully exposed developmental benchmark; 20 connected clusters, no independent holdout or clinical/human efficacy claim.',
                  'provider_weight_identity_boundary':'Exact returned alias is checked; matching names do not independently attest immutable provider weights. Both arms are fresh and interleaved within this run.',
                  'clock_policy':'Released source order only where calendar is missing; no synthetic consent or user confirmation.',
                  'cost_policy':'Count every actual attempt and failure, extraction and answers separately; observed tokens, character input budgets, no invented currency prices.'}
        protocol['protocol_sha256']=stable(protocol);atomic_json(run/'protocol.json',protocol)
        atomic_json(run/'prepared_manifest.json',{'protocol_sha256':protocol['protocol_sha256'],'prepared_file_sha256':protocol['prepared_file_sha256']})
        for name in ('requests.jsonl','attempts.jsonl','outcomes.jsonl'):atomic_rows(run/name,[])
        atomic_json(run/'phase_status.json',{'phase':'prepared','protocol_sha256':protocol['protocol_sha256'],'new_provider_calls':0})
    return {'status':'prepared','protocol_sha256':protocol['protocol_sha256'],'extraction_windows':len(extractions),'planned_arm_rows':1178,'maximum_provider_attempts':args.maximum_provider_attempts,'new_provider_calls':0}

def dispatch(journal,jobs,purpose):
    canonical={}
    for item in jobs:
        key=stable(request_body(item['messages'],purpose,journal.protocol['output_limits'][purpose]));canonical.setdefault(key,item['messages'])
    stopped=False;errors=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(journal.call,messages,purpose):key for key,messages in canonical.items() if key not in journal.outcomes}
        for future in as_completed(futures):
            try:future.result()
            except BudgetExhausted:stopped=True
            except Exception as error:
                if journal.aborted:stopped=True
                else:errors.append(error)
    if errors:raise errors[0]
    return stopped

def extraction_outputs(run,protocol,journal):
    outputs={};complete=True
    for row in rows(run/'extraction_requests.jsonl'):
        key=stable(request_body(row['messages'],'extraction',protocol['output_limits']['extraction']));outcome=journal.outcomes.get(key)
        if outcome is None:complete=False
        outputs[row['window_id']]=outcome['response']['hypothesis'] if outcome and outcome['status']=='success' else None
    return outputs,complete

def materialize_prepared(run,protocol,journal):
    outputs,complete=extraction_outputs(run,protocol,journal)
    if not complete:raise RuntimeError('Extraction phase incomplete; no answer preparation')
    from persona_management_adapter import prepare_arms
    plan=read_json(run/'scope_plan.json');prepared=[]
    for case in rows(run/'cases.jsonl'):
        value=prepare_arms(case,plan=plan,extraction_outputs=outputs,settings=protocol['settings'])
        if value['case_id']!=case['case_id'] or set(value['prepared_by_arm'])!=set(ARMS):raise ValueError('Adapter case/arm identity differs')
        for arm in ARMS:
            item=value['prepared_by_arm'][arm]
            if item['prompt_sha256']!=stable(item['messages']):raise ValueError('Adapter prompt identity differs')
            request_body(item['messages'],'response',64)
        prepared.append({**value,'group_id':case['group_id']})
    if (run/'prepared_rows.jsonl').exists():
        if rows(run/'prepared_rows.jsonl')!=prepared:raise ValueError('Prepared response prompts changed on replay')
    else:atomic_rows(run/'prepared_rows.jsonl',prepared)
    return prepared

def generation_rows(prepared,protocol,journal):
    output=[]
    for row in prepared:
        for arm in ARMS:
            item=row['prepared_by_arm'][arm];key=stable(request_body(item['messages'],'response',protocol['output_limits']['response']));outcome=journal.outcomes.get(key)
            response=outcome['response'] if outcome and outcome['status']=='success' else None
            raw=response['hypothesis'] if response else None
            output.append({'case_id':row['case_id'],'group_id':row['group_id'],'arm':arm,'request_key':key,'completion_status':outcome['status'] if outcome else 'pending',
                           'raw_response':raw,'response_sha256':stable(raw) if raw is not None else None,'prompt_sha256':item['prompt_sha256'],
                           'trace':item['trace'],'requested_model':protocol['generation_identity']['model'],'provider_model':response['provider_model'] if response else None})
    return output

def run_phase(args):
    require_colab();run=controlled(args.run)
    with ExitStack() as stack:
        stack.enter_context(lock(run,create=True));protocol=verify_protocol(run);retain_original_locks(protocol,stack)
        journal=Journal(run,protocol,max_new_calls=args.max_new_calls,verify_identity=lambda:verify_protocol(run))
        state={'phase':'halted_internal_validation'}
        try:
            if args.phase=='extract':
                dispatch(journal,rows(run/'extraction_requests.jsonl'),'extraction');_,complete=extraction_outputs(run,protocol,journal)
                state={'phase':'extraction_complete' if complete else 'extraction_partial'}
            elif args.phase=='answer':
                prepared=materialize_prepared(run,protocol,journal)
                dispatch(journal,[r['prepared_by_arm'][arm] for r in prepared for arm in ARMS],'response')
                generated=generation_rows(prepared,protocol,journal);atomic_rows(run/'generation_rows.jsonl',generated)
                state={'phase':'answers_complete' if all(r['completion_status']!='pending' for r in generated) else 'answers_partial',
                       'terminal_answer_rows':sum(r['completion_status']!='pending' for r in generated),'planned_arm_rows':1178}
            else:raise ValueError('Use bounded extract or answer phase')
            verify_protocol(run)
        finally:
            if journal.aborted:state.update({'phase':'halted_identity_validation','identity_halt_kind':journal.abort_kind})
            atomic_json(run/'journal_cost.json',journal.cost());state.update({'protocol_sha256':protocol['protocol_sha256'],'new_provider_calls':journal.new_calls,'cost':journal.cost()});atomic_json(run/'phase_status.json',state)
        return state

def readonly_state(run_dir,include_references=False):
    require_colab();run=controlled(run_dir)
    if include_references:raise ValueError('Only scorer may open references after terminal generation')
    with ExitStack() as stack:
        stack.enter_context(lock(run));protocol=verify_protocol(run);retain_original_locks(protocol,stack)
        journal=Journal(run,protocol);cases=rows(run/'cases.jsonl')
        prepared=rows(run/'prepared_rows.jsonl')
        if prepared:
            replayed=materialize_prepared(run,protocol,journal)
            if prepared!=replayed:raise ValueError('Prepared replay differs')
        generated=generation_rows(prepared,protocol,journal) if prepared else []
        if (run/'generation_rows.jsonl').exists() and rows(run/'generation_rows.jsonl')!=generated:raise ValueError('Published generation rows differ from journal')
        if prepared and [r['case_id'] for r in prepared]!=protocol['case_ids']:raise ValueError('Prepared case grid differs')
        complete=len(generated)==1178 and all(r['completion_status']!='pending' for r in generated)
        cost=journal.cost()
        if (run/'journal_cost.json').exists() and read_json(run/'journal_cost.json')!=cost:raise ValueError('Published cost differs from journal')
        hashes={name:digest(run/name) for name in INPUT_FILES if (run/name).exists()}
        return {'protocol':protocol,'cases':cases,'prepared_rows':prepared,'generation_rows':generated,'journal_cost':cost,'terminal_complete':complete,'input_sha256':hashes}

def cli():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=('prepare','extract','answer','status','score'))
    parser.add_argument('--run',required=True);parser.add_argument('--original-suite');parser.add_argument('--executable-commit');parser.add_argument('--maximum-provider-attempts',type=int,default=2500);parser.add_argument('--max-new-calls',type=int,default=0)
    args=parser.parse_args()
    if args.phase=='prepare':result=prepare(args)
    elif args.phase in {'extract','answer'}:result=run_phase(args)
    elif args.phase=='score':
        require_colab();from persona_management_scoring import score_run
        result=score_run(Path(args.run),Path(args.original_suite) if args.original_suite else None)
    else:
        state=readonly_state(args.run);result={'status':'terminal_complete' if state['terminal_complete'] else 'partial','protocol_sha256':state['protocol']['protocol_sha256'],'cost':state['journal_cost']}
    # Never print source text, model output, question/options/gold or provider errors.
    print(json.dumps(result,sort_keys=True,allow_nan=False))

if __name__=='__main__':
    try:cli()
    except Exception as error:
        print(json.dumps({'status':'safe_error','error_type':type(error).__name__,'source_model_or_gold_text_displayed':False}));raise SystemExit(1)
