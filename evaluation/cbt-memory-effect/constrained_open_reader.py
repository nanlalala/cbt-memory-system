"""Two NEW invented schema/readability pilots, using remaining shared 62→64 slots.

No previous reader/preparer/scorer is modified or monkeypatched. No old cache
key is reused. This is not memory effectiveness, CBT quality or human review.
"""
import argparse
from contextlib import ExitStack
import hashlib
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import open_reader_colab as reader
import constrained_json_grammar as grammar

VERSION='independent-common-json-constraint-two-pilot-v1'
PURPOSE='new_invented_schema_readability_pilot_not_memory_effect'
MODEL=Path('/content/cbt-open-reader-20261112-v1')
SYSTEM=('Answer the factual question using only the explicit supplied report. Return exactly one compact JSON '
    'object with the sole key answer. Use an integer for a count, the stated lowercase outcome word '
    'for an outcome, or null when unknown. Do not include explanations or markdown.')
FIXTURES=(('format-pilot-library-books','A fictional library clerk reports: "I counted 11 returned books." '
    'What count did the clerk explicitly report?',11),
    ('format-pilot-machine-outcome','A fictional technician reports: "After the routine adjustment, '
    'the machine outcome is unchanged." What outcome did the technician explicitly report?','unchanged'))


def guard(*paths):
    reader.require_colab()
    if any(not Path(p).resolve().is_relative_to(Path('/content')) for p in paths):
        raise ValueError('hosted_content_paths_required')


def requests():
    rows=[]
    for case_id,text,_ in FIXTURES:
        payload={'model':reader.MODEL_ID,'temperature':0,'max_tokens':64,
            'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':text}]}
        rows.append({'case_id':case_id,'request_payload':payload,'request_sha256':reader.digest(payload)})
    return rows


def code():
    return {str(p):reader.file_hash(p) for p in (Path(__file__),Path(grammar.__file__),Path(reader.__file__))}


def prepare(output):
    guard(output);output=Path(output)
    if output.exists(): raise FileExistsError('new_preparation_required_no_overwrite')
    output.mkdir(parents=True,mode=0o700);rows=requests()
    reader._write_new(output/'requests.json',rows)
    assessor=output/'assessor_only';assessor.mkdir(mode=0o700)
    reader._write_new(assessor/'targets.json',{cid:answer for cid,_,answer in FIXTURES})
    reader._write_new(output/'protocol.json',{'version':VERSION,'purpose':PURPOSE,'requests_sha256':reader.digest(rows),
        'targets_sha256':reader.file_hash(assessor/'targets.json'),
        'code_sha256':code(),'decoding_policy':grammar.policy(),'model_root':str(MODEL),
        'expected_initial_global_attempts':62,'maximum_global_attempts':64,'maximum_new_generations':2,
        'new_human_reviews':0,'new_provider_calls':0,'secret_access':False,
        'material':'two independent invented non-CBT schema/readability fixtures; not memory-effect cases',
        'expected_answers_not_used_for_decoding':True})
    return {'stage':'two_new_pilots_frozen_not_run','protocol_sha256':reader.file_hash(output/'protocol.json')}


def prepared(root):
    protocol=reader.read_json(root/'protocol.json');rows=reader.read_json(root/'requests.json')
    if (protocol.get('version')!=VERSION or protocol.get('purpose')!=PURPOSE or protocol.get('code_sha256')!=code()
        or protocol.get('decoding_policy')!=grammar.policy() or rows!=requests()
        or reader.digest(rows)!=protocol.get('requests_sha256') or protocol.get('model_root')!=str(MODEL)):
        raise ValueError('frozen_preparation_changed')
    return protocol,rows


def clinical_finished():
    root=Path('/content/therapyjudge-five-calibration-results-v1')
    summary=reader.read_json(root/'completion_summary.json')
    if (summary.get('complete') is not True or summary.get('logical_completed')!=5
        or summary.get('answers_sha256')!=reader.file_hash(root/'answers_unscored.jsonl')
        or reader.read_rows(root/'failures.jsonl')): raise RuntimeError('clinical_five_not_settled_successfully')
    events=[reader._checked(r) for r in reader.read_rows(root/'execution_events.jsonl')]
    if not any(r.get('event')=='calibration_fresh_process_finished' and r.get('complete') is True for r in events):
        raise RuntimeError('clinical_process_not_finished')


def run(root,output,model_root=MODEL):
    guard(root,output,model_root);root=Path(root).resolve();output=Path(output).resolve()
    if Path(model_root).resolve()!=MODEL: raise ValueError('original_shared_64_root_required')
    if output.exists(): raise FileExistsError('existing_pilot_attempt_inspect_no_resend')
    protocol,rows=prepared(root)
    with ExitStack() as locks:
        for name in ('.model_setup.lock','.generation.lock'):locks.enter_context(reader._locked(MODEL/name))
        intents,outcomes=reader._global_state(MODEL);clinical_finished()
        if len(intents)!=62 or any(r.get('purpose')==PURPOSE for r in intents.values()):
            raise RuntimeError('exact_62_settled_and_no_prior_pilot_attempts_required')
        output.mkdir(parents=True,mode=0o700)
        reader._write_new(output/'start_intent.json',{'version':VERSION,'prepared_protocol_sha256':reader.file_hash(root/'protocol.json'),
            'initial_global_attempts':62,'automatic_retry':False,'at_utc':reader.stamp()})
        model,tokenizer,setup=reader._load_cached_model(MODEL)
        identity=reader._model_identity(MODEL,model,tokenizer,fixture_only=False)
        trie=grammar.AnswerTrie(tokenizer,model.generation_config.eos_token_id)
        if trie.maximum_candidate_tokens>=64: raise ValueError('generic_language_does_not_fit_frozen_output_budget')
        token_preflight=[]
        for row in rows:
            tensors=tokenizer.apply_chat_template(row['request_payload']['messages'],add_generation_prompt=True,
                tokenize=True,return_dict=True,return_tensors='pt')
            count=int(tensors['input_ids'].shape[-1])
            if count>4096: raise ValueError('pilot_input_limit_before_generation_admission')
            token_preflight.append({'case_id':row['case_id'],'input_tokens':count,
                'input_tensor_sha256':grammar.digest(tensors['input_ids'].tolist())})
        run_protocol={'version':VERSION,'purpose':PURPOSE,'model':identity,'decoding':trie.identity,
            'actual_tensor_preflight':token_preflight,
            'prepared_protocol_sha256':reader.file_hash(root/'protocol.json'),'code_sha256':code(),
            'model_setup_receipt':setup,'maximum_global_attempts':64,'maximum_new_generations':2,
            'automatic_retry':False,'old_cache_reused':False,'clinical_or_memory_effect_claim':False}
        reader._write_new(output/'run_protocol.json',run_protocol);run_hash=reader.digest(run_protocol)
        answers=[]
        for row in rows:
            prepared(root)
            key=reader.digest({'model':identity,'decoding':trie.identity,'payload':row['request_payload'],
                'max_new_tokens':64,'do_sample':False})
            if key in intents or key in outcomes or len(intents)>=64: raise RuntimeError('new_budget_or_canonical_identity_required')
            start=time.monotonic();result=None
            intent=reader._append(MODEL/'generation_ledger.jsonl',{'event':'generation_intent','canonical_key':key,
                'purpose':PURPOSE,'request_sha256':row['request_sha256'],'run_protocol_sha256':run_hash,
                'output':str(output),'model_revision':reader.MODEL_REVISION,'started_at_utc':reader.stamp(),
                'decoding_identity_sha256':grammar.digest(trie.identity),'attempt_number':len(intents)+1})
            intents[key]=intent
            try:
                result=grammar.generate(model,tokenizer,row['request_payload']['messages'],trie,max_new_tokens=64)
                prepared(root)
                cache=output/'canonical'/(key+'.json')
                reader._write_new(cache,{**result,'canonical_key':key})
                outcome=reader._append(MODEL/'generation_ledger.jsonl',{'event':'generation_outcome','canonical_key':key,
                    'status':'published','cache_path':str(cache),'cache_sha256':reader.file_hash(cache),
                    'response_sha256':result['response_sha256'],'usage':result['usage'],'finish_reason':result['finish_reason'],
                    'generation_wall_seconds':time.monotonic()-start,'finished_at_utc':reader.stamp()})
                outcomes[key]=outcome
                answers.append(reader._append(output/'answers_unscored.jsonl',{'case_id':row['case_id'],
                    'canonical_key':key,'request_sha256':row['request_sha256'],'run_protocol_sha256':run_hash,
                    'status':'completed','scored':False,**result}))
            except BaseException as error:
                if key not in outcomes:
                    reader._append(MODEL/'generation_ledger.jsonl',{'event':'generation_outcome','canonical_key':key,
                        'status':'constrained_pilot_failed_or_unknown','response_text':None,
                        'usage':result.get('usage') if result else None,'failure_type':type(error).__name__,
                        'observed_wall_seconds':time.monotonic()-start,'finished_at_utc':reader.stamp()})
                reader._append(output/'failures.jsonl',{'case_id':row['case_id'],'failure_type':type(error).__name__,'automatic_retry':False})
                raise
        reader._global_state(MODEL)
        summary={'version':VERSION,'complete':len(answers)==2,'logical_completed':len(answers),
            'global_attempts':len(intents),'answers_sha256':reader.file_hash(output/'answers_unscored.jsonl'),
            'run_protocol_sha256':run_hash,'new_provider_calls':0,'new_human_reviews':0,
            'scope':'format/readability only, not CBT quality or memory effectiveness'}
        reader._write_new(output/'completion_summary.json',summary)
        return summary


def score(root,output):
    """Read fixed pilot labels only after both original outputs are frozen."""
    guard(root,output);root=Path(root);output=Path(output)
    protocol,rows=prepared(root);run_protocol=reader.read_json(output/'run_protocol.json')
    summary=reader.read_json(output/'completion_summary.json')
    if (summary.get('complete') is not True or summary.get('logical_completed')!=2
        or summary.get('answers_sha256')!=reader.file_hash(output/'answers_unscored.jsonl')
        or summary.get('run_protocol_sha256')!=reader.digest(run_protocol)
        or run_protocol.get('prepared_protocol_sha256')!=reader.file_hash(root/'protocol.json')
        or run_protocol.get('code_sha256')!=code() or run_protocol.get('decoding',{}).get('policy')!=grammar.policy()):
        raise ValueError('complete_bound_two_pilot_outputs_required')
    _,outcomes=reader._global_state(MODEL)
    answers=[reader._checked(r) for r in reader.read_rows(output/'answers_unscored.jsonl')]
    if len(answers)!=2 or [a['case_id'] for a in answers]!=[r['case_id'] for r in rows]:
        raise ValueError('exact_two_pilot_answer_order_required')
    for row,answer in zip(rows,answers):
        key=reader.digest({'model':run_protocol['model'],'decoding':run_protocol['decoding'],
            'payload':row['request_payload'],'max_new_tokens':64,'do_sample':False})
        if (answer['canonical_key']!=key or answer['request_sha256']!=row['request_sha256']
            or answer['run_protocol_sha256']!=reader.digest(run_protocol) or key not in outcomes):
            raise ValueError('unbound_pilot_answer')
        cached=reader._cache(outcomes[key])
        for field in ('response_text','response_sha256','chosen_answer','raw_generated_token_ids','decoding_identity_sha256'):
            if answer.get(field)!=cached.get(field): raise ValueError('pilot_cache_receipt_disagreement')
    target_path=root/'assessor_only/targets.json'
    if reader.file_hash(target_path)!=protocol['targets_sha256']: raise ValueError('fixed_pilot_labels_changed')
    targets=reader.read_json(target_path)
    if targets!={cid:value for cid,_,value in FIXTURES}: raise ValueError('fixture_label_identity_changed')
    checked=[]
    import json
    for answer in answers:
        value=json.loads(answer['response_text'])
        valid=(type(value) is dict and set(value)=={'answer'} and
            (value['answer'] is None or type(value['answer']) is int and 0<=value['answer']<=9999
             or type(value['answer']) is str and value['answer'] in grammar.STATUSES))
        expected=targets[answer['case_id']]
        checked.append({'case_id':answer['case_id'],'schema_valid':valid,'chosen_answer':answer['chosen_answer'],
            'basic_readability_correct':valid and type(value['answer']) is type(expected) and value['answer']==expected,
            'response_sha256':answer['response_sha256']})
    report={'version':VERSION,'pilot_rows':checked,'schema_valid':sum(r['schema_valid'] for r in checked),
        'basic_readability_correct':sum(r['basic_readability_correct'] for r in checked),
        'denominator':2,'new_model_calls':0,'new_human_reviews':0,'memory_effect_evidence':False,
        'CBT_quality_evidence':False,'answers_sha256':summary['answers_sha256']}
    reader._write_new(output/'pilot_score.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    x=sub.add_parser('prepare');x.add_argument('--output',required=True,type=Path)
    x=sub.add_parser('run');x.add_argument('--prepared-dir',required=True,type=Path);x.add_argument('--output',required=True,type=Path)
    x=sub.add_parser('score');x.add_argument('--prepared-dir',required=True,type=Path);x.add_argument('--output',required=True,type=Path)
    a=p.parse_args();result=prepare(a.output) if a.command=='prepare' else run(a.prepared_dir,a.output) if a.command=='run' else score(a.prepared_dir,a.output)
    print(reader.canonical(result),flush=True)


if __name__=='__main__':main()
