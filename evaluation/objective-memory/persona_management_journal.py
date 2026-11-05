"""Single-attempt durable provider journal for the new Persona management study."""
from __future__ import annotations
from collections import Counter
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
from threading import Lock
import time
import statistics

MODEL = 'deepseek-v4.1-flash'
ALIAS = 'deepseek-v4-1-flash-260910'
API = 'https://tokendance.space/gateway/v1'
TOKENS = ('prompt_tokens','completion_tokens','total_tokens')

def stable(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def rows(path):
    if not path.exists(): return []
    if path.is_symlink() or not path.is_file(): raise ValueError('Regular journal required')
    body=path.read_bytes()
    if body and not body.endswith(b'\n'): raise ValueError('Incomplete journal tail; no repair/retry')
    return [json.loads(line) for line in body.splitlines() if line.strip()]

def append(path,value):
    with path.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())

def numeric_usage(value):
    if not isinstance(value,dict): return {}
    return {key:value[key] for key in TOKENS if type(value.get(key)) is int and value[key]>=0}

class BudgetExhausted(RuntimeError): pass

def request_body(messages,purpose,output_limit):
    if purpose not in {'extraction','response'} or type(output_limit) is not int or not 1<=output_limit<=8192: raise ValueError('Unknown finite request policy')
    if not isinstance(messages,list) or not messages or any(set(m)!={'role','content'} or m['role'] not in {'system','user','assistant'} or not isinstance(m['content'],str) for m in messages): raise ValueError('Invalid role/content request')
    if sum(len(m['content']) for m in messages)>24000: raise ValueError('Fixed request character ceiling exceeded')
    return {'purpose':purpose,'messages':messages,'model':MODEL,'accepted_provider_models':[ALIAS],'api_base':API,
            'max_response_tokens':output_limit,'temperature':0,'thinking_mode':'disabled'}

def validate_response(response):
    def fail(kind,fatal=False):
        error=RuntimeError(kind);error.fatal=fatal
        error.reported_usage=numeric_usage(response.get('usage')) if isinstance(response,dict) else {}
        error.latency_seconds=response.get('latency_seconds') if isinstance(response,dict) else None
        raise error
    if not isinstance(response,dict) or response.get('provider_model')!=ALIAS:
        fail('provider_model_identity_failure',True)
    if response.get('finish_reason')!='stop' or not isinstance(response.get('hypothesis'),str) or not response['hypothesis'].strip(): fail('incomplete_provider_output')
    latency=response.get('latency_seconds')
    if not isinstance(latency,(int,float)) or isinstance(latency,bool) or not math.isfinite(latency) or latency<0: fail('invalid_numeric_latency',True)
    if not isinstance(response.get('usage'),dict): fail('missing_numeric_usage',True)

class Journal:
    def __init__(self,run,protocol,*,max_new_calls=0,verify_identity=None,transport=None):
        self.run=Path(run);self.protocol=protocol;self.verify_identity=verify_identity;self.transport=transport
        if type(max_new_calls) is not int or max_new_calls<0 or max_new_calls>protocol['maximum_provider_attempts']: raise ValueError('Finite invocation budget required')
        self.max_new_calls=max_new_calls;self.new_calls=0;self.mutex=Lock();self.aborted=False
        self.requests={};self.attempts={};self.outcomes={}
        self.reload()

    def reload(self):
        self.requests={};self.attempts={};self.outcomes={};self.aborted=False;self.abort_kind=None
        for row in rows(self.run/'requests.jsonl'):
            key=row['request_key'];request=row['request']
            if key in self.requests or row['protocol_sha256']!=self.protocol['protocol_sha256'] or stable(request)!=key: raise ValueError('Request journal identity differs')
            if request!=request_body(request['messages'],request['purpose'],self.protocol['output_limits'][request['purpose']]): raise ValueError('Request policy differs')
            self.requests[key]=request
        for name,collection in (('attempts.jsonl',self.attempts),('outcomes.jsonl',self.outcomes)):
            for row in rows(self.run/name):
                key=row['request_key']
                if key not in self.requests or key in collection or row['protocol_sha256']!=self.protocol['protocol_sha256'] or row.get('record_sha256')!=stable({k:v for k,v in row.items() if k!='record_sha256'}): raise ValueError('Journal seal differs')
                collection[key]=row
        if set(self.attempts)!=set(self.outcomes): raise RuntimeError('Unresolved in-flight request; manual reconciliation required, no retry')
        if any(r.get('attempt_index')!=1 or r.get('purpose')!=self.requests[k]['purpose'] for k,r in self.attempts.items()): raise ValueError('Attempt identity/policy differs')
        if len(self.attempts)>self.protocol['maximum_provider_attempts']: raise ValueError('Historical budget exceeded')
        for key,row in self.outcomes.items():
            if row.get('status')=='success': validate_response(row['response'])
            elif row.get('status') not in {'provider_failure','protocol_validation_failure'}: raise ValueError('Unknown terminal outcome')
            if row.get('fatal_identity_failure'):self.aborted=True;self.abort_kind='provider_identity'

    def _provider(self,request):
        from persona_management_runner import require_colab
        require_colab()
        key=os.environ.get('CBT_EVAL_API_KEY')
        if not key: raise RuntimeError('Colab environment credential unavailable')
        root=Path(__file__).resolve().parents[2];directory=root/'evaluation/dialogue-benefit'
        existing=sys.modules.get('followup_transport')
        if existing is not None and Path(existing.__file__).resolve()!=directory/'followup_transport.py': raise RuntimeError('Foreign cached provider module')
        sys.path.insert(0,str(directory));transport=importlib.import_module('followup_transport')
        return transport.generate_response(request['messages'],model=MODEL,allowed_model=ALIAS,api_base=API,key=key,max_response_tokens=request['max_response_tokens'])

    def call(self,messages,purpose):
        request=request_body(messages,purpose,self.protocol['output_limits'][purpose]);key=stable(request)
        with self.mutex:
            if self.aborted: raise RuntimeError('Fatal identity validation blocks new calls')
            if key in self.outcomes: return key,self.outcomes[key]
            if key in self.attempts: raise RuntimeError('Request already in flight; never duplicate')
            if self.new_calls>=self.max_new_calls or len(self.attempts)>=self.protocol['maximum_provider_attempts']: raise BudgetExhausted('Finite invocation/run budget exhausted')
            if self.verify_identity:
                try:self.verify_identity()
                except Exception:
                    self.aborted=True;self.abort_kind='frozen_input_identity';raise
            # Credential/guard failure is before intent and is not a provider attempt.
            if self.transport is None:
                from persona_management_runner import require_colab
                require_colab()
                if not os.environ.get('CBT_EVAL_API_KEY'): raise RuntimeError('Colab environment credential unavailable')
            if key not in self.requests:
                append(self.run/'requests.jsonl',{'protocol_sha256':self.protocol['protocol_sha256'],'request_key':key,'request':request});self.requests[key]=request
            intent={'protocol_sha256':self.protocol['protocol_sha256'],'request_key':key,'attempt_index':1,'purpose':purpose,'started_at_unix':time.time()}
            intent['record_sha256']=stable(intent);append(self.run/'attempts.jsonl',intent);self.attempts[key]=intent;self.new_calls+=1
        try:
            response=self.transport(messages,purpose) if self.transport is not None else self._provider(request)
            validate_response(response)
            usage=numeric_usage(response['usage'])
            cleaned={k:response[k] for k in ('hypothesis','provider_model','finish_reason','latency_seconds')};cleaned['usage']=usage
            outcome={'protocol_sha256':self.protocol['protocol_sha256'],'request_key':key,'status':'success','response':cleaned,
                     'reported_usage_missing_or_partial':len(usage)!=3,'billable_usage_unknown':len(usage)!=3}
        except Exception as error:
            fatal=bool(getattr(error,'fatal',False));usage=numeric_usage(getattr(error,'reported_usage',None))
            outcome={'protocol_sha256':self.protocol['protocol_sha256'],'request_key':key,'status':'protocol_validation_failure' if fatal else 'provider_failure',
                     'error_type':type(error).__name__,'reported_usage':usage,'reported_usage_missing_or_partial':len(usage)!=3,
                     'billable_usage_unknown':len(usage)!=3,'retry_sent':False,'fatal_identity_failure':fatal}
            latency=getattr(error,'latency_seconds',None)
            if isinstance(latency,(int,float)) and not isinstance(latency,bool) and math.isfinite(latency) and latency>=0:outcome['reported_latency_seconds']=latency
            if fatal:
                with self.mutex:self.aborted=True;self.abort_kind='provider_identity'
        outcome['record_sha256']=stable(outcome)
        with self.mutex:
            append(self.run/'outcomes.jsonl',outcome);self.outcomes[key]=outcome
        return key,outcome

    def cost(self):
        def one(keys):
            attempts=[self.attempts[k] for k in keys];outcomes=[self.outcomes[k] for k in keys if k in self.outcomes]
            usage=Counter()
            for row in outcomes: usage.update(numeric_usage(row.get('response',{}).get('usage') if row['status']=='success' else row.get('reported_usage')))
            latencies=[r['response']['latency_seconds'] if r['status']=='success' else r['reported_latency_seconds'] for r in outcomes if r['status']=='success' or 'reported_latency_seconds' in r]
            return {'provider_attempts':len(attempts),'successful_calls':sum(r['status']=='success' for r in outcomes),
                    'failed_calls':sum(r['status']!='success' for r in outcomes),'unresolved_inflight_calls':len(attempts)-len(outcomes),
                    'reported_usage':{key:usage[key] for key in TOKENS},'usage_missing_or_partial_calls':sum(r.get('reported_usage_missing_or_partial',True) for r in outcomes),
                    'billable_usage_unknown_calls':sum(r.get('billable_usage_unknown',True) for r in outcomes),
                    'reported_latency_seconds_sum':sum(latencies),'reported_latency_seconds_median':statistics.median(latencies) if latencies else None,
                    'latency_missing_calls':len(outcomes)-len(latencies)}
        result=one(list(self.attempts));result['by_purpose']={purpose:one([k for k in self.attempts if self.requests[k]['purpose']==purpose]) for purpose in ('extraction','response')}
        result['currency_cost_available']=False;result['automatic_retries']=0
        return result
