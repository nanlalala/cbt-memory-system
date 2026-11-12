from pathlib import Path
import os,sys,json,time,hashlib,subprocess,importlib.util
assert os.environ.get('COLAB_RELEASE_TAG') and importlib.util.find_spec('google.colab') is not None
root=Path('/content/cbt-qwen7b-gpu-state-study-20261112-v9');stage=Path('/content/cbt-qwen7b-gpu-v7-code/evaluation/cbt-memory-effect/gguf_state_comparison_v9_colab.py');claim=Path('/content/cbt-gpu-v9-main-execution-20261112-v1')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(stage)=='b4dc236135c421e2451afd2b264adcd24c52fd16a2a305e458e3ade9107d12ed'
assert sha(root/'main_freeze.json')=='c3638f7f9492e57d2e5179bfaa4ebd57db57152242a07315abf179b3ee54edb5'
assert sha(root/'study_protocol.json')=='be618cc1492351111f76d6c97dd3dd6c26a138cb3b8212a994272bd778eeac94'
plan=json.loads((root/'main_freeze.json').read_text());assert plan['main_unique_payloads']==51;assert not (root/'generation_ledger.jsonl').exists();claim.mkdir(mode=0o700,exist_ok=False)
runtime=json.loads((root/'runtime_manifest.json').read_text());env=dict(os.environ);env['LD_LIBRARY_PATH']=runtime['child_ld_library_path'];env['PYTHONPATH']=runtime['site_packages']+':'+str(stage.parent);base=[sys.executable,'-u',str(stage),'--study-root',str(root)]
(claim/'execution_intent.json').open('x').write(json.dumps({'source_sha256':sha(__file__),'freeze_sha256':sha(root/'main_freeze.json'),'actual_frozen_unique':51,'independent_attempt_cap':57,'automatic_retry':False,'start_unix':time.time()}))
for name,argv in [('before',['run-side','--side','before']),('after',['run-side','--side','after']),('score',['score'])]:
 log=Path('/content/cbt-gpu-v9-main-'+name+'-20261112.log');t=time.monotonic();d={'phase':name,'at_unix':time.time(),'argv':base+argv+['--constructor-log-path',str(log)],'automatic_retry':False};(claim/(name+'-intent.json')).open('x').write(json.dumps(d));print('Starting actual phase:',name,flush=True)
 with log.open('xb') as f:
  try:
   p=subprocess.run(d['argv'],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1800);d['returncode']=p.returncode
  except BaseException as e:d.update({'failure_type':type(e).__name__,'status':'failed_or_unknown_no_retry'})
 d['whole_process_wall_seconds']=time.monotonic()-t;(claim/(name+'-completion.json')).open('x').write(json.dumps(d));print(json.dumps({'phase':name,'returncode':d.get('returncode'),'whole_process_wall_seconds':d['whole_process_wall_seconds']}),flush=True)
 if d.get('returncode')!=0:raise RuntimeError('phase_stopped_inspect_actual_journals_no_automatic_retry:'+name)
 if name in ('before','after'):
  done=json.loads((root/('answers_'+name)/'completion_summary.json').read_text());assert done['complete'] is True and done['logical_completed']==48;print(json.dumps({'phase':name,'logical_completed':done['logical_completed'],'global_unique_attempts':done['global_new_study_attempts']}),flush=True)
 else:
  report=root/'reports/summary.json';assert report.is_file();print('ALL_96_COMPLETED_AND_SCORED',str(report),'SHA256',sha(report),flush=True)
(claim/'all-phases-completed.json').open('x').write(json.dumps({'before_after_logical_rows':96,'actual_frozen_unique':51,'completed_unix':time.time(),'new_paid_provider_calls':0,'secret_access':False}))
