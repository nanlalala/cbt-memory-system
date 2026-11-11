"""Hosted-only, gold-blind source-binding development replay; no provider calls.

All nine published answers are transformed and durably frozen before the assessor
opens reference answers. Only the original, currently allowed wire quotations can
repair a source ID. This is a common postprocessor for every experimental arm.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import audit_context_revision as audit
import context_assembly_v3 as assembly
import prepare_context_revision_v3 as diagnostic
import run_context_revision_v3 as sender

VERSION='source-quote-binding-retrospective-replay-v1'


def _binding_module():
    # The original product checkout remains frozen. New postprocessing code is
    # loaded only from this replay's new checkout, never added to the old one.
    path=(HERE.parents[1]/'modules/agent-integration/source_quote_binding.py').resolve(strict=True)
    expected_hash=diagnostic.file_hash(path)
    spec=importlib.util.spec_from_file_location('source_quote_binding_replay',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    if Path(module.__file__).resolve(strict=True)!=path or diagnostic.file_hash(path)!=expected_hash:
        raise ValueError('actual source binding module identity changed during import')
    return module,path,expected_hash


def wire_catalog(context):
    """Extract only packet/recent user evidence, after caller's source fence.

    Current user input is deliberately excluded. This function never has a DB,
    target, retrieval cue, or unavailable historical quotation as an argument.
    """
    messages=context['messages'];direct=[m for m in messages[:-1] if m['role']!='system']
    catalog=[];seen={};packet_count=map_count=0;covered=set()
    def add(sid,role,text):
        if not isinstance(sid,str) or not sid or role not in {'user','assistant'} or not isinstance(text,str):
            raise ValueError('invalid wire source')
        if sid==context['current_input_source_id']:raise ValueError('current input is not historical evidence')
        if sid in seen:raise ValueError('duplicate wire source identity')
        seen[sid]=(role,text)
        if role=='user':catalog.append({'source_id':sid,'role':role,'text':text})
    for message in messages:
        if message['role']!='system':continue
        content=message['content']
        if content.startswith(assembly.PACKET_HEADER):
            packet_count+=1
            for row in json.loads(content[len(assembly.PACKET_HEADER):])['sources']:
                add(row['source_id'],row['role'],row['text'])
        elif content.startswith(assembly.RECENT_MAP_HEADER):
            map_count+=1
            for row in json.loads(content[len(assembly.RECENT_MAP_HEADER):])['recent_sources']:
                index=row['recent_index']
                if type(index) is not int or not 0<=index<len(direct) or index in covered:
                    raise ValueError('invalid recent wire mapping')
                covered.add(index);message=direct[index]
                if row['role']!=message['role']:raise ValueError('recent wire speaker changed')
                add(row['source_id'],row['role'],message['content'])
    if packet_count>1 or map_count>1 or covered!=set(range(len(direct))):
        raise ValueError('wire history is not completely and uniquely mapped')
    if set(seen)!=set(context['source_dependencies']):raise ValueError('wire dependencies differ')
    return catalog


def _snapshot(roots):
    return {str(path):diagnostic.file_hash(path) for root in roots
            for path in sorted(root.rglob('*')) if path.is_file()}


def _fsync_directory(path):
    descriptor=os.open(path,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def _freeze(output,rows,manifest):
    output.mkdir(parents=True,mode=0o700)
    path=output/'transformed_answers.jsonl'
    with path.open('x') as stream:
        for row in rows:stream.write(json.dumps(row,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
    manifest={**manifest,'transformed_rows_sha256':diagnostic.file_hash(path),'logical_rows':len(rows)}
    sender.write_new(output/'transformation_manifest.json',sender.signed(manifest))
    _fsync_directory(output)
    return manifest


def _target_identity(target,case,receipts):
    if (target is None or target.get('case_sha256')!=case['case_sha256'] or
        target.get('source_family_id')!=case['source_family_id'] or
        case['case_sha256']!=diagnostic.digest({k:v for k,v in case.items() if k!='case_sha256'})):
        raise ValueError('target/case/family identity mismatch')
    matches=[turn for turn in case['history'] if turn['message_id']==target['target_message_id']]
    receipt=receipts.get(target['target_message_id'])
    if (len(matches)!=1 or receipt is None or receipt['role']!='user' or matches[0]['role']!='user' or
        matches[0]['content']!=target['target_quote'] or
        receipt['text_sha256']!=hashlib.sha256(target['target_quote'].encode()).hexdigest() or
        not 0<=matches[0]['source_order']<case['history_cutoff_order_exclusive']):
        raise ValueError('target is not a complete in-cutoff released Client source')
    return receipt['source_id']


def replay(proposal_dir,output,*,fixture_only=False):
    if not fixture_only:diagnostic.require_colab()
    proposal_dir=Path(proposal_dir).resolve(strict=True);output=Path(output).resolve()
    proposal=diagnostic.read_json(proposal_dir/'proposal.json')
    if proposal.get('proposal_sha256')!=sender.sha({k:v for k,v in proposal.items() if k!='proposal_sha256'}):
        raise ValueError('proposal checksum mismatch')
    if proposal.get('fixture_only')!=fixture_only:raise ValueError('fixture/live replay identity changed')
    sender._verify_hashes(proposal['code_sha256'])
    requests=diagnostic.read_json(proposal_dir/'requests.json')
    if sender.sha(requests)!=proposal['requests_sha256'] or len(requests)!=9:
        raise ValueError('all nine frozen logical requests required')
    data=sender._load_inputs(proposal['diagnostic_dir'],proposal['old_run'],proposal['product_repo'])
    diag,old,protocol,old_protocol,cases,traces,database,prepared,product=data
    if protocol['protocol_sha256']!=proposal['diagnostic_protocol_sha256']:
        raise ValueError('diagnostic differs from proposal')
    roots=(proposal_dir,diag,old)
    if output.exists() or any(output.is_relative_to(root) for root in roots):
        raise ValueError('new separate replay output directory required')
    before=_snapshot(roots)
    trace_by_id={row['case_id']:row for row in traces}
    keys={(r['case_id'],r['arm']) for r in requests}
    if len(trace_by_id)!=3 or len(keys)!=9 or keys!={(cid,arm) for cid in trace_by_id for arm in assembly.ARMS}:
        raise ValueError('exactly three cases and every arm once required')
    catalogs={}
    for request in requests:
        key=(request['case_id'],request['arm']);context=request['context']
        if context!=trace_by_id[key[0]]['assembly']['contexts'][key[1]]:
            raise ValueError('proposal request differs from diagnostic')
        sender._fence(data,proposal,cases[key[0]],context)
        if request['request_sha256']!=context['request_sha256']:raise ValueError('request identity changed')
        catalogs[key]=wire_catalog(context)
    intents,outcomes,provider_counts=audit.audit_journal(proposal_dir,proposal,requests)
    answers={}
    for answer in sender.rows(proposal_dir/'answers_unscored.jsonl'):
        sender.checked(answer);key=(answer['case_id'],answer['arm'])
        if key not in keys or key in answers or answer.get('proposal_sha256')!=proposal['proposal_sha256']:
            raise ValueError('duplicate or foreign answer')
        request=next(r for r in requests if (r['case_id'],r['arm'])==key)
        receipt=outcomes.get(answer.get('request_sha256'))
        if (answer.get('request_sha256')!=request['request_sha256'] or receipt is None or
            receipt.get('status')!='published' or receipt.get('response_text')!=answer.get('response_text')):
            raise ValueError('answer has no matching published provider receipt')
        answers[key]=answer
    if set(answers)!=keys:raise ValueError('nine published answer rows required for complete replay')
    binding,binding_path,binding_hash=_binding_module()
    transformed=[]
    for request in requests:
        key=(request['case_id'],request['arm']);answer=answers[key]
        result=binding.repair_source_quote_binding(answer['response_text'],catalogs[key])
        transformed.append(sender.signed({'case_id':key[0],'arm':key[1],
            'proposal_sha256':proposal['proposal_sha256'],'request_sha256':request['request_sha256'],
            'original_response_sha256':hashlib.sha256(answer['response_text'].encode()).hexdigest(),
            'wire_catalog_sha256':sender.sha(catalogs[key]),**result}))
    for request in requests:sender._fence(data,proposal,cases[request['case_id']],request['context'])
    if before!=_snapshot(roots):raise ValueError('original artifacts changed before transformation freeze')
    if diagnostic.file_hash(binding_path)!=binding_hash:raise ValueError('source binding module changed during transformation')
    manifest=_freeze(output,transformed,{'version':VERSION,'proposal_sha256':proposal['proposal_sha256'],
        'fixture_only':fixture_only,'retrospective_development_replay':True,'new_provider_calls':0,
        'secret_access':False,'gold_parsed_or_used_before_transform_freeze':False,
        'pre_freeze_gold_access':'integrity SHA-256 bytes only; no parsing or repair use',
        'original_artifact_hashes':before,'code_sha256':{str(Path(__file__).resolve()):diagnostic.file_hash(__file__),
            str(binding_path):binding_hash,str(Path(audit.__file__).resolve()):diagnostic.file_hash(audit.__file__)},
        'catalog_policy':'current allowed user quotations from exact frozen wire only; no question, gold or DB supplementation',
        'same_rule_all_arms':True,'independent_holdout':False,
        'frozen_scorer_expected_sha256':old_protocol.get('code_sha256',{}).get('evaluation/cbt-memory-effect/run_cactus_effect.py')})
    # PHASE B: no target/answer scoring is allowed above this durable boundary.
    frozen=diagnostic.read_json(output/'transformation_manifest.json');sender.checked(frozen)
    if (diagnostic.file_hash(output/'transformed_answers.jsonl')!=manifest['transformed_rows_sha256'] or
        frozen.get('transformed_rows_sha256')!=manifest['transformed_rows_sha256']):
        raise ValueError('transformation artifact changed before scoring')
    replay_rows=diagnostic.read_rows(output/'transformed_answers.jsonl')
    if replay_rows!=transformed:raise ValueError('frozen transformations differ')
    for row in replay_rows:sender.checked(row)
    for request in requests:sender._fence(data,proposal,cases[request['case_id']],request['context'])
    targets_path=old/'assessor_only/targets.jsonl'
    expected=old_protocol.get('files_sha256',{}).get('assessor_only/targets.jsonl')
    if expected is None or diagnostic.file_hash(targets_path)!=expected:raise ValueError('original targets checksum mismatch')
    targets={}
    for target in diagnostic.read_rows(targets_path):
        if target['case_id'] in targets:raise ValueError('duplicate assessor target')
        targets[target['case_id']]=target
    score_quote,_,scorer_hash=audit.frozen_scorer(proposal['product_repo'],old_protocol)
    receipts={r['message_id']:r for r in prepared['trace']['source_receipts']}
    scored=[]
    for row in replay_rows:
        key=(row['case_id'],row['arm']);target=targets.get(key[0]);case=cases[key[0]]
        source_id=_target_identity(target,case,receipts)
        original=score_quote(answers[key]['response_text'],target,source_id)
        revised=score_quote(row['output_text'],target,source_id)
        scored.append({'case_id':key[0],'arm':key[1],'position':target.get('position'),
            'request_sha256':row['request_sha256'],'original_response_sha256':row['original_response_sha256'],
            'replay_response_sha256':hashlib.sha256(row['output_text'].encode()).hexdigest(),
            'changed':row['changed'],'action':row['action'],'reason':row['reason'],
            'original_score':original,'replay_score':revised,
            'original_source_binding_error':original.get('source_binding_correct') is False,
            'replay_source_binding_error':revised.get('source_binding_correct') is False})
    for request in requests:sender._fence(data,proposal,cases[request['case_id']],request['context'])
    if before!=_snapshot(roots):raise ValueError('original artifacts changed during replay')
    if diagnostic.file_hash(output/'transformed_answers.jsonl')!=manifest['transformed_rows_sha256']:
        raise ValueError('transformation artifact changed during scoring')
    by_arm={}
    for arm in assembly.ARMS:
        subset=[row for row in scored if row['arm']==arm]
        by_arm[arm]={'logical_rows':len(subset),'original_correct':sum(r['original_score']['literal_success'] for r in subset),
            'replay_correct':sum(r['replay_score']['literal_success'] for r in subset),
            'changed':sum(r['changed'] for r in subset),'source_id_repairs':sum(r['action']=='repaired_source_id' for r in subset),
            'original_source_binding_errors':sum(r['original_source_binding_error'] for r in subset),
            'replay_source_binding_errors':sum(r['replay_source_binding_error'] for r in subset)}
    report={'version':VERSION,'fixture_only':fixture_only,'retrospective_development_replay':True,
        'new_provider_calls':0,'secret_access':False,'independent_holdout':False,'same_rule_all_arms':True,
        'original_artifacts_unchanged':True,'transformed_rows_sha256':manifest['transformed_rows_sha256'],
        'original_targets_sha256':expected,'original_scorer_sha256':scorer_hash,
        'provider_attempts_reused':len(intents),'provider_status_counts':provider_counts,'by_arm':by_arm,'rows':scored,
        'interpretation':'Retrospective development replay of existing answers with a common source-binding validator. No new model calls, independent holdout effect or managed-only benefit.'}
    sender.write_new(output/'replay_report.json',report)
    _fsync_directory(output)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal-dir',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    report=replay(args.proposal_dir,args.output)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},sort_keys=True))

if __name__=='__main__':main()
