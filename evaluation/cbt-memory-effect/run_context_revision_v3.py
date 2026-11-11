"""Finite three-probe development answer pilot; never an automatic v2 resume.

prepare-proposal freezes all nine logical payloads without Secrets or scoring
answers. run requires a matching, explicit new authorization. It copies only
currently permitted original sources into new durable stores, fences every send
and publication against the old source controls, and never modifies old runs.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import timedelta, datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prepare_context_revision_v3 as diagnostic
import context_assembly_v3 as assembly

VERSION = 'context-revision-three-probe-answer-pilot-v1'
MAX_CALLS = 9
SYSTEM = ('Answer from the supplied original conversation. Preserve speaker roles and '
          'source identity. Source order is not a calendar date. Do not invent missing information.')
TOKEN_FIELDS = ('prompt_tokens', 'completion_tokens', 'total_tokens')


def stamp(): return datetime.now(timezone.utc).isoformat()
def sha(value): return hashlib.sha256(assembly.canonical(value).encode()).hexdigest()
def signed(value): return {**value, 'record_sha256': sha(value)}
def checked(value):
    body = {k:v for k,v in value.items() if k != 'record_sha256'}
    if value.get('record_sha256') != sha(body): raise ValueError('journal checksum changed')
    return value


def write_new(path, value):
    path = Path(path)
    with path.open('x') as stream:
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n')
        stream.flush(); os.fsync(stream.fileno())


def rows(path):
    return diagnostic.read_rows(path) if Path(path).exists() else []


def _verify_hashes(values):
    for name, expected in values.items():
        if diagnostic.file_hash(name) != expected: raise ValueError('frozen dependency changed: '+name)


def _source_snapshot(source, order):
    return {key: value for key,value in {**source, 'source_order':order}.items()
            if key in {'source_id','role','text','source_order'}}


def validate_context(context, *, case, sources, settings, maximum_chars):
    """Reconstruct every model-visible byte from permitted original sources."""
    pool = {s['source_id']:s for s in sources}
    recent, selected = context['recent_source_ids'], context['retrieved_source_ids']
    if len(set(recent+selected)) != len(recent+selected): raise ValueError('duplicate history source')
    deps = list(dict.fromkeys(recent+selected))
    if deps != context['source_dependencies'] or any(sid not in pool for sid in deps):
        raise ValueError('unbound history dependency')
    if any(pool[a]['source_order'] >= pool[b]['source_order'] for a,b in zip(recent,recent[1:])):
        raise ValueError('recent speaker order changed')
    if recent and pool[recent[0]]['role'] == 'assistant': raise ValueError('orphan recent assistant')
    for sid in deps:
        snapshot = _source_snapshot(pool[sid],pool[sid]['source_order'])
        if sha(snapshot) != context['source_snapshot_sha256'][sid]: raise ValueError('source snapshot changed')
    messages = [{'role':'system','content':SYSTEM}]
    if selected:
        packet = {'sources':[_source_snapshot(pool[sid],pool[sid]['source_order']) for sid in selected]}
        messages.append({'role':'system','content':assembly.PACKET_HEADER+json.dumps(packet,ensure_ascii=False,sort_keys=True,allow_nan=False)})
    if recent:
        mapping={'recent_sources':[{'recent_index':index,'source_id':sid,
            'source_order':pool[sid]['source_order'],'role':pool[sid]['role']} for index,sid in enumerate(recent)]}
        messages.append({'role':'system','content':assembly.RECENT_MAP_HEADER+
                         json.dumps(mapping,ensure_ascii=False,sort_keys=True,allow_nan=False)})
    messages.extend({'role':pool[sid]['role'],'content':pool[sid]['text']} for sid in recent)
    messages.append({'role':'user','content':case['input']})
    payload = {'model':settings['model'],'temperature':0,'thinking':{'type':'disabled'},
               'max_tokens':settings['output_tokens']['response'],'messages':messages}
    wire = assembly.canonical(payload)
    if (context['messages'] != messages or context['request_payload'] != payload or
        context['serialized_request'] != wire or context['request_sha256'] != sha(payload) or
        context['outbound_chars'] != len(wire) or len(wire) > maximum_chars):
        raise ValueError('complete outbound request changed')
    if context['current_input_source_id'] != 'v3-query:'+case['case_id']:
        raise ValueError('current input source identity changed')
    return payload


def _load_inputs(diagnostic_dir, old_run, product_repo):
    diag=Path(diagnostic_dir).resolve(strict=True); old=Path(old_run).resolve(strict=True)
    protocol=diagnostic.read_json(diag/'protocol.json')
    if protocol.get('protocol_sha256') != diagnostic.digest({k:v for k,v in protocol.items() if k!='protocol_sha256'}):
        raise ValueError('diagnostic protocol changed')
    _verify_hashes(protocol['source_checkpoint_hashes']); _verify_hashes(protocol['code_sha256'])
    product=diagnostic.product_modules(product_repo)
    for name,h in protocol['product_code_sha256'].items():
        if diagnostic.file_hash(Path(product_repo)/name)!=h: raise ValueError('frozen product code changed')
    old_protocol=diagnostic.read_json(old/'protocol.json')
    if old_protocol['protocol_sha256'] != protocol['old_protocol_sha256']: raise ValueError('foreign v2 run')
    if diagnostic.file_hash(old/'generation/cases.jsonl') != old_protocol['files_sha256']['generation/cases.jsonl']:
        raise ValueError('generation changed')
    cases={r['case_id']:r for r in diagnostic.read_rows(old/'generation/cases.jsonl')}
    traces=diagnostic.read_rows(diag/'context_rank_pack_traces.jsonl')
    if len(traces)!=3 or [r['case_id'] for r in traces]!=protocol['selected_cases']:
        raise ValueError('exactly three frozen probes required')
    anchors=[Path(p) for p in protocol['source_checkpoint_hashes'] if p.endswith('/prepared_contexts.json')]
    if len(anchors)!=1 or not anchors[0].is_relative_to(old): raise ValueError('one original source anchor required')
    prepared=diagnostic.read_json(anchors[0])['prepared']
    return diag,old,protocol,old_protocol,cases,traces,anchors[0].parent/'memory.sqlite',prepared,product


def prepare_proposal(diagnostic_dir, old_run, output, *, product_repo, fixture_only=False):
    if not fixture_only: diagnostic.require_colab()
    output=Path(output).resolve()
    data=_load_inputs(diagnostic_dir,old_run,product_repo)
    diag,old,protocol,old_protocol,cases,traces,database,prepared,product=data
    if output.exists() or output.is_relative_to(old) or output.is_relative_to(diag):
        raise ValueError('new separate proposal directory required')
    requests=[]
    with diagnostic.readonly_durable(database,product) as durable:
        generation=durable.current_generation(prepared['trace']['user_id'])
        for row in traces:
            case=cases[row['case_id']]
            if case['case_sha256']!=row['case_sha256']: raise ValueError('foreign case')
            sources,_=diagnostic.checkpoint_sources(durable,prepared,case)
            expected_recent=None
            for arm in assembly.ARMS:
                context=row['assembly']['contexts'][arm]
                payload=validate_context(context,case=case,sources=sources,settings=old_protocol['settings'],maximum_chars=protocol['max_outbound_chars'])
                if expected_recent is None: expected_recent=context['recent_source_ids']
                if context['recent_source_ids']!=expected_recent: raise ValueError('arms have different recent history')
                requests.append({'case_id':case['case_id'],'arm':arm,'context':context,'request_sha256':sha(payload)})
    _verify_hashes(protocol['source_checkpoint_hashes'])
    proposal={'version':VERSION,'fixture_only':fixture_only,'maximum_new_provider_calls':MAX_CALLS,
        'maximum_logical_rows':9,'selected_cases':protocol['selected_cases'],'diagnostic_dir':str(diag),
        'old_run':str(old),'product_repo':str(Path(product_repo).resolve()),'old_database':str(database),
        'diagnostic_protocol_sha256':protocol['protocol_sha256'],'old_generation':generation,
        'requests_sha256':sha(requests),'settings':old_protocol['settings'],
        'reuse_cost_note':'Existing extraction and embedding costs are reused development inputs, not free new extraction.',
        'old_cost_artifact':{'path':str(old/'summary.json'),'sha256':diagnostic.file_hash(old/'summary.json')},
        'code_sha256':{str(Path(__file__).resolve()):diagnostic.file_hash(__file__),
                       str(Path(diagnostic.__file__).resolve()):diagnostic.file_hash(diagnostic.__file__),
                       str(Path(assembly.__file__).resolve()):diagnostic.file_hash(assembly.__file__)},
        'model_generation_uses_gold':False,'automatic_retry':False,'prepared_at_utc':stamp()}
    proposal['proposal_sha256']=sha(proposal)
    output.mkdir(parents=True,mode=0o700)
    write_new(output/'proposal.json',proposal); write_new(output/'requests.json',requests)
    return proposal


def authorization(proposal,path):
    value=diagnostic.read_json(path)
    if (value.get('proposal_sha256')!=proposal['proposal_sha256'] or value.get('user_authorized') is not True or
        type(value.get('maximum_new_provider_calls')) is not int or value['maximum_new_provider_calls']!=MAX_CALLS or
        not isinstance(value.get('authorization_source'),str) or not value['authorization_source'].strip()):
        raise PermissionError('explicit new user authorization matching this nine-call proposal is required')
    return value


def _numeric_usage(value):
    value=value if isinstance(value,dict) else {}
    return {k:value[k] if type(value.get(k)) is int and value[k]>=0 else None for k in TOKEN_FIELDS}


class UnknownAttempt(RuntimeError): pass


class Journal:
    def __init__(self,folder,proposal):
        self.folder=Path(folder);self.proposal=proposal
        self.intents={};self.outcomes={}
        for name,target in [('call_intents.jsonl',self.intents),('call_outcomes.jsonl',self.outcomes)]:
            for r in rows(self.folder/name):
                checked(r); key=r['request_sha256']
                if key in target or r['proposal_sha256']!=proposal['proposal_sha256']: raise ValueError('duplicate or foreign journal')
                if name=='call_intents.jsonl' and sha(r['request_payload'])!=key:
                    raise ValueError('journal request payload changed')
                target[key]=r
        for outcome in self.outcomes.values():
            if outcome['status']=='published' and (outcome.get('provider_model')!=proposal['settings']['accepted_provider_model'] or
                                                   outcome.get('finish_reason')!='stop'):
                raise ValueError('cached provider receipt identity changed')
        if set(self.outcomes)-set(self.intents) or len(self.intents)>MAX_CALLS: raise ValueError('unbound call outcome or budget')
        if set(self.intents)-set(self.outcomes) or any(r['status']!='published' for r in self.outcomes.values()):
            raise UnknownAttempt('preserved interrupted, failed or stale call; automatic retry blocked')

    def intent(self,request):
        key=request['request_sha256']
        if key in self.outcomes:return self.outcomes[key]
        if len(self.intents)>=MAX_CALLS: raise ValueError('finite provider ceiling reached')
        row=signed({'proposal_sha256':self.proposal['proposal_sha256'],'request_sha256':key,
                    'request_payload':request['context']['request_payload'],'intent_at_utc':stamp()})
        append(self.folder/'call_intents.jsonl',row);self.intents[key]=row
        return None

    def outcome(self,request,*,status,response=None,usage=None,latency=None,observed=None,failure=None,provider_model=None,finish_reason=None):
        row=signed({'proposal_sha256':self.proposal['proposal_sha256'],'request_sha256':request['request_sha256'],
            'status':status,'response_text':response if status=='published' else None,'usage':_numeric_usage(usage),
            'api_latency_seconds':latency,'observed_invocation_seconds':observed,'failure_kind':failure,
            'provider_model':provider_model,'finish_reason':finish_reason,'recorded_at_utc':stamp()})
        append(self.folder/'call_outcomes.jsonl',row);self.outcomes[request['request_sha256']]=row
        return row


def _fence(data,proposal,case,context):
    _,_,protocol,_,_,_,database,prepared,product=data
    _verify_hashes(protocol['source_checkpoint_hashes'])
    with diagnostic.readonly_durable(database,product) as old:
        if old.current_generation(prepared['trace']['user_id'])!=proposal['old_generation']:
            raise PermissionError('original source controls changed')
        sources,_=diagnostic.checkpoint_sources(old,prepared,case)
        validate_context(context,case=case,sources=sources,settings=proposal['settings'],maximum_chars=protocol['max_outbound_chars'])
    return sources


def _new_store(root,proposal,case,sources,context,product):
    case_root=root/'cases'/case['case_id'];case_root.mkdir(parents=True,exist_ok=True)
    database=case_root/'memory.sqlite';new=not database.exists()
    limits=product.DurableLimits(max_run_calls=MAX_CALLS,max_session_calls=MAX_CALLS,max_queue_length=1,max_batch_size=1)
    store=product.DurableMemoryStore.open(database,approved_root=case_root,run_limits=limits)
    run_id='context-v3:'+proposal['proposal_sha256'];session_id=case['case_id'];user=sources[0]['user_id']
    try:
        store.configure_run(run_id,limits)
        if new:
            for source in sources:
                record={key:source[key] for key in ('user_id','source_id','input_id','text','role','source_time','permissions')}
                store.submit_source(record,enqueue=False,run_id=run_id,session_id=session_id)
            current={'user_id':user,'source_id':context['current_input_source_id'],'input_id':'v3-input:'+case['case_id'],
                'text':case['input'],'role':'user','source_time':(diagnostic.EPOCH+timedelta(seconds=case['history_cutoff_order_exclusive'])).isoformat(),
                'permissions':{k:True for k in ('save','long_term_use','response_external','extraction_external')}}
            store.submit_source(current,enqueue=False,run_id=run_id,session_id=session_id)
        # Existing stores are not trusted merely because their controls are
        # valid. Bind this exact old context to the source contents we checked,
        # and capture its generation in the same locked read transaction.
        current={'user_id':user,'source_id':context['current_input_source_id'],'input_id':'v3-input:'+case['case_id'],
            'text':case['input'],'role':'user','source_time':(diagnostic.EPOCH+timedelta(seconds=case['history_cutoff_order_exclusive'])).isoformat(),
            'permissions':{k:True for k in ('save','long_term_use','response_external','extraction_external')}}
        with store.store.transaction(immediate=True):
            generation=store.current_generation(user)
            for source in sources:
                expected={key:source[key] for key in ('user_id','source_id','input_id','text','role','source_time','permissions')}
                if store.get_source(user,source['source_id'])!=expected:
                    raise PermissionError('new pilot history differs from this frozen source snapshot')
            if store.get_source(user,current['source_id'])!=current:
                raise PermissionError('new pilot current input differs from the frozen question')
            receipt=store.get_input(user,current['input_id'])
            if receipt is None or any(receipt.get(key)!=value for key,value in
                    {'source_id':current['source_id'],'run_id':run_id,'session_id':session_id,'source_status':'committed'}.items()):
                raise PermissionError('new pilot current input binding differs')
        return store,user,run_id,session_id,generation
    except BaseException:
        store.close();raise


def run(proposal_dir, authorization_json, *, fixture_only=False, transport=None):
    if not fixture_only: diagnostic.require_colab()
    if fixture_only and transport is None: raise ValueError('fixture requires explicit local fake transport')
    if not fixture_only and transport is not None: raise ValueError('real execution forbids injected transports')
    root=Path(proposal_dir).resolve(strict=True);proposal=diagnostic.read_json(root/'proposal.json')
    if proposal.get('proposal_sha256')!=sha({k:v for k,v in proposal.items() if k!='proposal_sha256'}): raise ValueError('proposal changed')
    authorization(proposal,authorization_json)
    if proposal['fixture_only']!=fixture_only: raise ValueError('fixture/live identity changed')
    _verify_hashes(proposal['code_sha256'])
    requests=diagnostic.read_json(root/'requests.json')
    if sha(requests)!=proposal['requests_sha256'] or len(requests)!=9: raise ValueError('frozen requests changed')
    data=_load_inputs(proposal['diagnostic_dir'],proposal['old_run'],proposal['product_repo'])
    if data[2]['protocol_sha256']!=proposal['diagnostic_protocol_sha256']: raise ValueError('foreign diagnostic')
    cases=data[4];product=data[-1]
    lock=(root/'.execution.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        journal=Journal(root,proposal)
        # Do all request/source validation before credential access.
        for request in requests:_fence(data,proposal,cases[request['case_id']],request['context'])
        prior=rows(root/'answers_unscored.jsonl')
        for row in prior: checked(row)
        prior_keys={(r['case_id'],r['arm']) for r in prior}
        if len(prior_keys)!=len(prior) or any(r['proposal_sha256']!=proposal['proposal_sha256'] for r in prior):
            raise ValueError('duplicate or foreign logical answer')
        expected_keys={(r['case_id'],r['arm']) for r in requests}
        if prior_keys - expected_keys: raise ValueError('answer not in frozen proposal')
        if prior_keys==expected_keys:
            if not (root/'completion_summary.json').exists():
                raise UnknownAttempt('logical completion lacks its final receipt; inspect before continuing')
            return diagnostic.read_json(root/'completion_summary.json')
        if transport is None:
            key=importlib.import_module('google.colab.userdata').get('Capstone')
            directory=Path(proposal['product_repo'])/'evaluation/dialogue-benefit'
            sys.path.insert(0,str(directory));module=importlib.import_module('followup_transport')
            if Path(module.__file__).resolve()!=directory/'followup_transport.py': raise RuntimeError('foreign transport')
            settings=proposal['settings']
            def transport(payload):
                return module.generate_response(payload['messages'],model=settings['model'],allowed_model=settings['accepted_provider_model'],
                    api_base=settings['api_base'],key=key,max_response_tokens=settings['output_tokens']['response'])
        logical=[]
        for request in requests:
            if (request['case_id'],request['arm']) in prior_keys: continue

            case=cases[request['case_id']];context=request['context']
            sources=_fence(data,proposal,case,context)
            store,user,run_id,session_id,context_generation=_new_store(root,proposal,case,sources,context,product)
            admission=None;finalized=False;received_reply=None;provider_validated=False;started=time.monotonic()
            try:
                ids=list(context['source_dependencies'])+[context['current_input_source_id']]
                admission=store.begin_attempt(user,'response:v3-input:'+case['case_id']+':'+request['request_sha256'],
                    'response',ids,run_id=run_id,session_id=session_id,expected_generation=context_generation)
                cached=journal.intent(request)
                if cached is not None:
                    # Every logical arm has its own durable binding; the wire may share one provider receipt.
                    reply={'hypothesis':cached['response_text'],'usage':cached['usage'],'latency_seconds':None,
                           'provider_model':cached['provider_model'],'finish_reason':cached['finish_reason']}
                else:
                    _fence(data,proposal,case,context)
                    reply=transport(context['request_payload'])
                    received_reply=reply if isinstance(reply,dict) else None
                if not isinstance(reply,dict) or not isinstance(reply.get('hypothesis'),str) or not reply['hypothesis'].strip():
                    raise ValueError('invalid provider response')
                if reply.get('provider_model')!=proposal['settings']['accepted_provider_model'] or reply.get('finish_reason')!='stop':
                    raise ValueError('provider identity or completion changed')
                provider_validated=True
                _fence(data,proposal,case,context)
                if admission.get('replayed'):
                    if admission.get('status')!='succeeded' or cached is None:
                        raise UnknownAttempt('durable canonical result is not safely reusable')
                else:
                    finished=store.finish_attempt(admission['attempt_id'],outcome='success',response=None,usage=reply.get('usage'),retain_response=False)
                    finalized=True
                    if not finished['publishable']: raise PermissionError('source changed while generating response')
                outcome=cached or journal.outcome(request,status='published',response=reply['hypothesis'],usage=reply.get('usage'),
                    latency=reply.get('latency_seconds'),observed=time.monotonic()-started,
                    provider_model=reply.get('provider_model'),finish_reason=reply.get('finish_reason'))
                logical_row=signed({'proposal_sha256':proposal['proposal_sha256'],'case_id':case['case_id'],'arm':request['arm'],
                    'request_sha256':request['request_sha256'],'canonical_response_shared':cached is not None,'response_text':outcome['response_text'],
                    'scored':False,'source_control_checks_passed':True})
                append(root/'answers_unscored.jsonl',logical_row);logical.append(logical_row)
            except BaseException as error:
                # A post-generation source fence may reject publication after a
                # fully received, billed response. Retain that known usage and
                # latency even though its body must never enter the answer log.
                known_usage=(received_reply.get('usage') if received_reply is not None else None)
                if known_usage is None: known_usage=getattr(error,'reported_usage',None)
                known_latency=(received_reply.get('latency_seconds') if received_reply is not None else None)
                if known_latency is None: known_latency=getattr(error,'latency_seconds',None)
                if admission and not admission.get('replayed') and not finalized:
                    store.finish_attempt(admission['attempt_id'],outcome='failed' if received_reply is not None else 'unknown',usage=known_usage,retain_response=False)
                if request['request_sha256'] in journal.intents and request['request_sha256'] not in journal.outcomes:
                    journal.outcome(request,status=('stale_discarded' if provider_validated else 'response_rejected' if received_reply is not None else 'attempt_unknown'),usage=known_usage,
                        latency=known_latency,observed=time.monotonic()-started,failure=getattr(error,'failure_kind',type(error).__name__),
                        provider_model=received_reply.get('provider_model') if received_reply is not None else None,
                        finish_reason=received_reply.get('finish_reason') if received_reply is not None else None)
                raise
            finally:store.close()
        usage={key:(sum(observed) if observed else None) for key in TOKEN_FIELDS
               for observed in [[row['usage'][key] for row in journal.outcomes.values() if row['usage'][key] is not None]]}
        missing={key:sum(row['usage'][key] is None for row in journal.outcomes.values()) for key in TOKEN_FIELDS}
        summary={'version':VERSION,'proposal_sha256':proposal['proposal_sha256'],'logical_rows_this_invocation':len(logical),
            'total_provider_attempts':len(journal.intents),'maximum_provider_attempts':MAX_CALLS,'tokens_observed':usage,'usage_missing_attempts':missing,
            'human_reviews':0,'effect_scores':None,'old_checkpoint_unchanged':True,'automatic_retry':False}
        if (root/'completion_summary.json').exists():
            return diagnostic.read_json(root/'completion_summary.json')
        write_new(root/'completion_summary.json',summary)
        return summary
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);commands=p.add_subparsers(dest='command',required=True)
    proposal=commands.add_parser('prepare-proposal')
    proposal.add_argument('--diagnostic-dir',required=True,type=Path);proposal.add_argument('--old-run',required=True,type=Path)
    proposal.add_argument('--product-repo',required=True,type=Path);proposal.add_argument('--output',required=True,type=Path)
    execute=commands.add_parser('run');execute.add_argument('--proposal-dir',required=True,type=Path);execute.add_argument('--authorization-json',required=True,type=Path)
    args=p.parse_args()
    result=(prepare_proposal(args.diagnostic_dir,args.old_run,args.output,product_repo=args.product_repo) if args.command=='prepare-proposal'
            else run(args.proposal_dir,args.authorization_json))
    print(json.dumps(result,sort_keys=True))

if __name__=='__main__':main()
