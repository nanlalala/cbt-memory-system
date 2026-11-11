"""Hosted-only read-only assessor for frozen context-revision proposals.

Gold is opened only here, after frozen input/request verification. Presence in a
request is a retrieval/packing observation, never an answer or therapy outcome.
Outputs contain identifiers, hashes and booleans rather than dialogue text.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import prepare_context_revision_v3 as diagnostic
import context_assembly_v3 as assembly
import run_context_revision_v3 as sender

VERSION='context-revision-readonly-assessor-v1'


def _norm(text): return ' '.join(text.split())
def _unique(values): return list(dict.fromkeys(values))
def _rank(values, target): return values.index(target)+1 if target in values else None


def frozen_scorer(product_repo, old_protocol):
    """Compile only three pure scoring definitions from the hash-verified v2 file.

    No module top-level code, model, transport or generation runner is executed.
    """
    relative='evaluation/cbt-memory-effect/run_cactus_effect.py'
    path=Path(product_repo)/relative
    expected=old_protocol.get('code_sha256',{}).get(relative)
    if expected is None or diagnostic.file_hash(path)!=expected:
        raise ValueError('hash-verified original quotation scorer required')
    tree=ast.parse(path.read_text())
    names={'strict_json','norm','score_quote'}
    selected=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    if {node.name for node in selected}!=names or len(selected)!=3:
        raise ValueError('frozen scorer definitions changed')
    namespace={'json':json}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['score_quote'],namespace['strict_json'],expected


def _packet_presence(context, target, source_id):
    packed=False; recent=False
    direct=[message for message in context['messages'][:-1] if message['role']!='system']
    for message in context['messages']:
        if message['role']!='system':continue
        if message['content'].startswith(assembly.PACKET_HEADER):
            packet=json.loads(message['content'][len(assembly.PACKET_HEADER):])
            packed=any(row['source_id']==source_id and row['role']=='user' and
                _norm(row['text'])==_norm(target['target_quote']) for row in packet['sources'])
        if message['content'].startswith(assembly.RECENT_MAP_HEADER):
            mapping=json.loads(message['content'][len(assembly.RECENT_MAP_HEADER):])['recent_sources']
            for row in mapping:
                index=row['recent_index']
                if not 0<=index<len(direct):raise ValueError('recent source map index exceeds direct exchange')
                if row['source_id']==source_id and row['role']=='user':
                    recent=(direct[index]['role']=='user' and
                            _norm(direct[index]['content'])==_norm(target['target_quote']))
    return packed,recent


def _answer_components(body,target,source_id,strict_json):
    try:
        value=strict_json(body)
    except (ValueError,TypeError,json.JSONDecodeError):
        return {'extra_text_or_fields':True,'extra_fields':None,'missing_fields':None}
    if not isinstance(value,dict):
        return {'extra_text_or_fields':True,'extra_fields':None,'missing_fields':None}
    expected={'quote','source_id','unknown'}
    extra=sorted(set(value)-expected);missing=sorted(expected-set(value))
    return {'extra_text_or_fields':bool(extra or missing),'extra_fields':extra,'missing_fields':missing}


def audit_journal(proposal_dir,proposal,requests):
    """Read-only receipt reconciliation, retaining unknown states without retry."""
    expected={request['request_sha256']:request['context']['request_payload'] for request in requests}
    intents={};outcomes={}
    for row in sender.rows(Path(proposal_dir)/'call_intents.jsonl'):
        sender.checked(row);key=row['request_sha256']
        if (key in intents or key not in expected or row.get('proposal_sha256')!=proposal['proposal_sha256'] or
            sender.sha(row.get('request_payload'))!=key or row['request_payload']!=expected[key]):
            raise ValueError('intent differs from frozen canonical request')
        intents[key]=row
    if len(intents)>proposal['maximum_new_provider_calls']:raise ValueError('proposal attempt ceiling exceeded')
    for row in sender.rows(Path(proposal_dir)/'call_outcomes.jsonl'):
        sender.checked(row);key=row['request_sha256']
        if key in outcomes or key not in intents or row.get('proposal_sha256')!=proposal['proposal_sha256']:
            raise ValueError('duplicate or unbound provider outcome')
        if row.get('status') not in {'published','attempt_unknown','stale_discarded','response_rejected'}:
            raise ValueError('unknown provider receipt state')
        if row['status']=='published':
            if (row.get('provider_model')!=proposal['settings']['accepted_provider_model'] or
                row.get('finish_reason')!='stop' or type(row.get('response_text')) is not str or not row['response_text'].strip()):
                raise ValueError('published provider identity/completion/body invalid')
        elif row.get('response_text') is not None:
            raise ValueError('nonpublishable provider outcome retains answer body')
        outcomes[key]=row
    counts={state:sum(row['status']==state for row in outcomes.values()) for state in sorted({r['status'] for r in outcomes.values()})}
    counts['intent_without_outcome']=len(set(intents)-set(outcomes))
    return intents,outcomes,counts


def assess(proposal_dir,output,*,fixture_only=False):
    if not fixture_only:diagnostic.require_colab()
    proposal_dir=Path(proposal_dir).resolve(strict=True);output=Path(output).resolve()
    proposal=diagnostic.read_json(proposal_dir/'proposal.json')
    if proposal.get('proposal_sha256')!=sender.sha({k:v for k,v in proposal.items() if k!='proposal_sha256'}):
        raise ValueError('proposal checksum mismatch')
    if proposal.get('fixture_only')!=fixture_only:raise ValueError('fixture/live assessor identity changed')
    sender._verify_hashes(proposal['code_sha256'])
    requests=diagnostic.read_json(proposal_dir/'requests.json')
    if sender.sha(requests)!=proposal['requests_sha256'] or len(requests)!=9:
        raise ValueError('all nine frozen requests required')
    data=sender._load_inputs(proposal['diagnostic_dir'],proposal['old_run'],proposal['product_repo'])
    diag,old,protocol,old_protocol,cases,traces,database,prepared,product=data
    if protocol['protocol_sha256']!=proposal['diagnostic_protocol_sha256']:
        raise ValueError('diagnostic differs from proposal')
    if (output.exists() or any(output.is_relative_to(root) for root in (proposal_dir,diag,old))):
        raise ValueError('new independent assessment output directory required')
    before={str(path):diagnostic.file_hash(path) for root in (proposal_dir,diag,old)
            for path in root.rglob('*') if path.is_file()}
    trace_by_id={row['case_id']:row for row in traces}
    if len(trace_by_id)!=3:raise ValueError('exactly three diagnostic cases required')
    # Validate every model-visible source/input byte before opening assessor gold.
    for request in requests:
        if request['case_id'] not in trace_by_id or request['arm'] not in assembly.ARMS:
            raise ValueError('request outside selected case/arm scope')
        if request['context']!=trace_by_id[request['case_id']]['assembly']['contexts'][request['arm']]:
            raise ValueError('proposal request differs from frozen diagnostic')
        sender._fence(data,proposal,cases[request['case_id']],request['context'])
        if request['request_sha256']!=request['context']['request_sha256']:
            raise ValueError('proposal request hash differs from context')
    keys={(r['case_id'],r['arm']) for r in requests}
    if len(keys)!=9:raise ValueError('duplicate logical request')
    targets_path=old/'assessor_only/targets.jsonl'
    expected=old_protocol.get('files_sha256',{}).get('assessor_only/targets.jsonl')
    if expected is None or diagnostic.file_hash(targets_path)!=expected:
        raise ValueError('original assessor targets checksum mismatch')
    targets={row['case_id']:row for row in diagnostic.read_rows(targets_path)}
    score_quote,strict_json,scorer_hash=frozen_scorer(proposal['product_repo'],old_protocol)
    answers={}
    answers_path=proposal_dir/'answers_unscored.jsonl'
    intents,outcomes,provider_counts=audit_journal(proposal_dir,proposal,requests)
    for answer in sender.rows(answers_path):
        sender.checked(answer);key=(answer['case_id'],answer['arm'])
        if key in answers or key not in keys or answer.get('proposal_sha256')!=proposal['proposal_sha256']:
            raise ValueError('duplicate or foreign answer row')
        request=next(r for r in requests if (r['case_id'],r['arm'])==key)
        if answer.get('request_sha256')!=request['request_sha256']:
            raise ValueError('answer bound to another request')
        receipt=outcomes.get(answer['request_sha256'])
        if receipt is None or receipt.get('status')!='published' or receipt.get('response_text')!=answer.get('response_text'):
            raise ValueError('answer has no matching published provider receipt')
        answers[key]=answer
    output_rows=[]
    receipts={r['message_id']:r for r in prepared['trace']['source_receipts']}
    for request in requests:
        case=cases[request['case_id']];trace=trace_by_id[case['case_id']]
        target=targets.get(case['case_id'])
        if target is None or target.get('case_sha256')!=case['case_sha256'] or target.get('source_family_id')!=case['source_family_id']:
            raise ValueError('target/case/family identity mismatch')
        if case['case_sha256']!=diagnostic.digest({k:v for k,v in case.items() if k!='case_sha256'}):
            raise ValueError('generation case digest mismatch')
        receipt=receipts.get(target['target_message_id'])
        turn=next((t for t in case['history'] if t['message_id']==target['target_message_id']),None)
        if receipt is None or turn is None or receipt['role']!='user' or turn['role']!='user' or turn['content']!=target['target_quote']:
            raise ValueError('target does not identify the complete released Client source')
        if not 0<=turn['source_order']<case['history_cutoff_order_exclusive']:
            raise ValueError('target exceeds frozen history cutoff')
        source_id=receipt['source_id'];context=request['context'];arm=request['arm']
        raw_ids=_unique(row['source_id'] for row in trace['raw_retrieval_trace']['candidates'] if row['selected'])
        candidate_ids=assembly.project_candidate_sources(trace['candidate_index_trace']['ranked_index_hits'])
        fused_ids=[row['source_id'] for row in trace['assembly']['fusion']['source_ranks']]
        packed,recent=_packet_presence(context,target,source_id)
        available=source_id in (raw_ids if arm=='raw_history_rag' else fused_ids if arm=='managed_memory_rag' else [])
        stage=('present_in_recent' if recent else 'present_in_retrieved_packet' if packed else
               'packing_omission' if available else 'target_not_retrieved')
        answer=answers.get((case['case_id'],arm));body=answer.get('response_text') if answer else None
        provider_status=(outcomes[request['request_sha256']]['status'] if request['request_sha256'] in outcomes else
                         'intent_without_outcome' if request['request_sha256'] in intents else 'not_attempted')
        score=score_quote(body,target,source_id) if isinstance(body,str) else None
        row={'case_id':case['case_id'],'position':target.get('position'),'arm':arm,
            'target_source_id':source_id,'target_quote_sha256':hashlib.sha256(target['target_quote'].encode()).hexdigest(),
            'request_sha256':request['request_sha256'],'outbound_chars':context['outbound_chars'],
            'raw_source_rank':_rank(raw_ids,source_id),'candidate_source_rank':_rank(candidate_ids,source_id),
            'managed_fused_rank':_rank(fused_ids,source_id),'target_in_raw_ranking':source_id in raw_ids,
            'target_in_candidate_projection':source_id in candidate_ids,'target_in_recent':recent,
            'target_in_retrieved_packet':packed,'complete_quote_and_source_present':packed or recent,
            'memory_input_stage':stage,'packing_drop_recorded':source_id in context['dropped_sources'],
            'provider_status':provider_status,
            'answer_status':score['status'] if score else ('not_generated' if answer is None else 'not_published'),
            'answer_accuracy':score['literal_success'] if score else None,
            'quote_identity_correct':score.get('quote_identity_correct') if score else None,
            'source_binding_correct':score.get('source_binding_correct') if score else None,
            'response_sha256':hashlib.sha256(body.encode()).hexdigest() if isinstance(body,str) else None,
            **(_answer_components(body,target,source_id,strict_json) if isinstance(body,str) else
               {'extra_text_or_fields':None,'extra_fields':None,'missing_fields':None})}
        output_rows.append(row)
    if before!={str(path):diagnostic.file_hash(path) for root in (proposal_dir,diag,old)
                for path in root.rglob('*') if path.is_file()}:
        raise ValueError('frozen inputs changed during read-only assessment')
    by_arm={}
    for arm in assembly.ARMS:
        subset=[row for row in output_rows if row['arm']==arm]
        scored=[row for row in subset if row['answer_accuracy'] is not None]
        by_arm[arm]={'planned_rows':len(subset),'generated_answer_rows':len(scored),
            'targets_present_in_input':sum(row['complete_quote_and_source_present'] for row in subset),
            'input_stage_counts':{stage:sum(row['memory_input_stage']==stage for row in subset) for stage in sorted({row['memory_input_stage'] for row in subset})},
            'answer_accuracy':sum(row['answer_accuracy'] for row in scored)/len(subset) if len(scored)==len(subset) else None,
            'accuracy_among_generated':sum(row['answer_accuracy'] for row in scored)/len(scored) if scored else None}
    report={'version':VERSION,'fixture_only':fixture_only,'role':'assessor_only','proposal_sha256':proposal['proposal_sha256'],
        'diagnostic_protocol_sha256':protocol['protocol_sha256'],'original_target_file_sha256':expected,
        'original_scorer_sha256':scorer_hash,'provider_attempts':len(intents),'provider_status_counts':provider_counts,'planned_rows':len(output_rows),'by_arm':by_arm,'rows':output_rows,
        'new_provider_calls':0,'secret_access':False,'frozen_inputs_unchanged':True,
        'interpretation':'Input target availability is a retrieval/packing observation. Answer accuracy requires generated answers. Three related development probes are not a held-out dataset result.'}
    output.mkdir(parents=True,mode=0o700)
    (output/'audit.json').write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
    fields=[key for key in output_rows[0] if key not in {'extra_fields','missing_fields'}]
    with (output/'per_question.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(output_rows)
    compact={'version':VERSION,'planned_rows':9,'by_arm':by_arm,'provider_attempts':len(intents),'provider_status_counts':provider_counts,
        'cases':[{'position':row['position'],'arm':row['arm'],'raw_rank':row['raw_source_rank'],
            'candidate_rank':row['candidate_source_rank'],'input_stage':row['memory_input_stage'],
            'provider_status':row['provider_status'],'answer_status':row['answer_status'],'answer_accuracy':row['answer_accuracy']} for row in output_rows],
        'new_provider_calls':0,'secret_access':False,'input_presence_is_answer_accuracy':False}
    (output/'compact_report.json').write_text(json.dumps(compact,sort_keys=True,allow_nan=False)+'\n')
    return compact


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal-dir',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(assess(args.proposal_dir,args.output),sort_keys=True))

if __name__=='__main__':main()
