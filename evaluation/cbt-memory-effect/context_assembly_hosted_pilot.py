"""Hosted-only invented development pilot; no paid API or Secret access."""
import os,sys,json,hashlib,time,importlib.util
from pathlib import Path
assert os.environ.get('COLAB_RELEASE_TAG') and Path('/content').is_dir() and importlib.util.find_spec('google.colab')
from auto_eval_v2.context_assembly import assemble,ARMS
from auto_eval_v2.retrieval import canonical
R=Path('/content/context-assembly-three-arm-pilot-20261113')
assert not R.exists(),'Existing run: reconcile its journals; no automatic retry'
R.mkdir()
fixtures=[('correction',['I agreed to practise a thought record seven times.','Please change that target to two times. I have not started yet.'],'The user revised the thought-record target from seven to two practices and reported not starting.','What is the latest agreed practice target? Reply only with JSON {"answer":number}.',2),('pause',['I agreed to do two thought records.','I have completed one thought record.','I want to pause further thought records for now.'],'The user reported completing one thought record, then requested a pause in further practice.','Is further thought-record practice currently paused? Reply only with JSON {"answer":true_or_false}.',True),('proposal',['Perhaps you could practise breathing three times.','I am not agreeing to that plan yet.'],'The assistant proposed three breathing practices; the user explicitly did not agree to the plan.','Has the user agreed to the breathing plan? Reply only with JSON {"answer":true_or_false}.',False)]
requests=[];targets={}
for cid,texts,summary,question,target in fixtures:
    subject='invented:'+cid
    sources=[dict(source_id=cid+':s'+str(i),source_order=i,role=('assistant' if cid=='proposal' and i==0 else 'user'),content=t,subject_id=subject,retained=True,use_allowed=True,external_allowed=True) for i,t in enumerate(texts)]
    memory=dict(memory_id=cid,version=1,subject_id=subject,active=True,deleted=False,valid_at_cutoff=True,superseded_by=None,support='verified_projection',verifier_revision='invented-fixture-reviewed-v1',epistemic_status='supported_state',summary=summary,source_ids=[s['source_id'] for s in sources],evidence=[dict(source_id=s['source_id'],start=0,end=len(s['content']),text=s['content']) for s in sources])
    targets[cid]=target
    for arm in ARMS:
        package=assemble(sources=sources,selected_ids=[s['source_id'] for s in sources],memories=[memory],recent=[],current_input=question,subject_id=subject,cutoff_order=len(sources)-1,max_request_chars=6000,request_options={'model':'Qwen/Qwen2.5-1.5B-Instruct','temperature':0,'max_tokens':64},verifier_revision='invented-fixture-reviewed-v1',arm=arm)
        requests.append({'case_id':cid,'arm':arm,**package})
protocol={'study':'representation-mechanism-pilot-v1','scope':'three invented development cases; manually prepared supported representations; not automatic extraction or natural CBT efficacy','code':'c21d17b86e9fb32eee6efa079452646ee2ce518f','model':'Qwen/Qwen2.5-1.5B-Instruct','revision':'989aa7980e4cf806f80c7fef2b1adb7bc71aa306','max_new_generations':9,'max_new_tokens':64,'paid_api_calls':0,'secret_access':False,'requests_sha256':hashlib.sha256(canonical(requests).encode()).hexdigest(),'targets_sha256':hashlib.sha256(canonical(targets).encode()).hexdigest(),'stop':'runtime error or unresolved intent; no automatic retry','scoring':'strict parsed JSON with exact value and type; no substring match; failures stay in denominator'}
for name,obj in [('protocol',protocol),('requests',requests),('targets',targets)]: (R/(name+'.json')).write_text(canonical(obj))
print(canonical({'stage':'frozen',**protocol}),flush=True)
import torch,transformers,importlib.metadata
from transformers import AutoTokenizer,AutoModelForCausalLM
start=time.monotonic()
torch.set_num_threads(2)
tok=AutoTokenizer.from_pretrained(protocol['model'],revision=protocol['revision'],trust_remote_code=False)
model=AutoModelForCausalLM.from_pretrained(protocol['model'],revision=protocol['revision'],trust_remote_code=False,torch_dtype=torch.float32).eval()
receipt={'stage':'model_loaded','seconds':time.monotonic()-start,'device':'cpu','dtype':'float32','packages':{n:importlib.metadata.version(n) for n in ['torch','transformers','huggingface-hub']},'chat_template_sha256':hashlib.sha256(tok.chat_template.encode()).hexdigest()}
(R/'model_receipt.json').write_text(canonical(receipt));print(canonical(receipt),flush=True)
answers=[]
def journal(row):
    with (R/'generation_ledger.jsonl').open('a') as stream:
        stream.write(canonical(row)+'\n');stream.flush();os.fsync(stream.fileno())
for i,row in enumerate(requests):
    assert i<9
    text=tok.apply_chat_template(row['request']['messages'],tokenize=False,add_generation_prompt=True)
    encoded=tok(text,return_tensors='pt',add_special_tokens=False)
    n=encoded['input_ids'].shape[-1]
    journal({'event':'intent','index':i,'request_sha256':row['trace']['request_sha256'],'input_tokens':n,'rendered_prompt_sha256':hashlib.sha256(text.encode()).hexdigest()})
    at=time.monotonic()
    with torch.inference_mode():output=model.generate(**encoded,max_new_tokens=64,do_sample=False,pad_token_id=tok.eos_token_id)
    answer=tok.decode(output[0,n:],skip_special_tokens=True)
    result={'case_id':row['case_id'],'arm':row['arm'],'text':answer,'input_tokens':n,'output_tokens':output.shape[-1]-n,'generation_seconds':time.monotonic()-at,'request_sha256':row['trace']['request_sha256']}
    answers.append(result)
    journal({'event':'outcome','index':i,**result})
    with (R/'answers.jsonl').open('a') as stream:stream.write(canonical(result)+'\n');stream.flush();os.fsync(stream.fileno())
    print(canonical({'stage':'answer_recorded','index':i,**result}),flush=True)
for r in answers:
    try:
        parsed=json.loads(r['text']);value=parsed['answer'];gold=targets[r['case_id']]
        r['correct']=set(parsed)=={'answer'} and type(value) is type(gold) and value==gold
        r['parse_ok']=True
    except (ValueError,TypeError,KeyError):r['correct']=False;r['parse_ok']=False
summary={'scope':protocol['scope'],'all_planned_completed':len(answers)==9,'by_arm':{arm:{'correct':sum(r['correct'] for r in answers if r['arm']==arm),'planned':3,'valid_json':sum(r['parse_ok'] for r in answers if r['arm']==arm)} for arm in ARMS},'new_model_generations':9,'paid_api_calls':0,'secret_access':False,'total_input_tokens':sum(r['input_tokens'] for r in answers),'total_output_tokens':sum(r['output_tokens'] for r in answers),'rows':answers,'interpretation':'development mechanism pilot only; fixed selected sources, not retrieval or natural CBT efficacy'}
(R/'summary.json').write_text(canonical(summary));print(canonical({'stage':'COMPLETE',**summary}),flush=True)
