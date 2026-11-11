"""Hosted-only read-only aggregation of an already-scored finite answer pilot.

No generation, rescoring, Secrets or models. Canonical provider receipts are
counted once; arm allocations are explicit accounting, not observed billing.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import audit_context_revision as assessor
D=assessor.diagnostic;S=assessor.sender;ARMS=assessor.assembly.ARMS
VERSION='answer-pilot-readonly-summary-v1'
FIELDS=('prompt_tokens','completion_tokens','total_tokens')


def observable_error(row):
    if row['provider_status']!='published':return 'provider_or_control_'+row['provider_status']
    status=row['answer_status']
    if status=='correct':return 'correct'
    if status=='explicit_unknown':return 'evidence_present_abstention' if row['complete_quote_and_source_present'] else 'evidence_missing_abstention'
    if status=='technical_failure':return 'output_format_or_schema_error'
    if status=='wrong_quote_or_source_binding':
        return ('source_binding_error' if row['quote_identity_correct'] is True else
                'quote_identity_error' if row['source_binding_correct'] is True else 'quote_and_source_binding_error')
    return 'answer_not_completed'


def _sum_known(values):return sum(v for v in values if v is not None) if any(v is not None for v in values) else None


def aggregate_cost(requests,intents,outcomes):
    owners={key:sorted({r['arm'] for r in requests if r['request_sha256']==key}) for key in intents}
    token_values={key:[outcomes.get(h,{}).get('usage',{}).get(key) for h in intents] for key in FIELDS}
    for values in token_values.values():
        if any(v is not None and (type(v) is not int or v<0) for v in values):raise ValueError('invalid observed token usage')
    latencies={key:[outcomes.get(h,{}).get(key) for h in intents] for key in ('api_latency_seconds','observed_invocation_seconds')}
    for values in latencies.values():
        if any(v is not None and (type(v) not in (int,float) or not math.isfinite(v) or v<0) for v in values):raise ValueError('invalid observed latency')
    tokens={k:_sum_known(v) for k,v in token_values.items()};latency={k:_sum_known(v) for k,v in latencies.items()}
    allocations={}
    for arm in ARMS:
        own=[h for h in intents if arm in owners[h]]
        def allocated(metric):
            values=[(outcomes[h].get('usage',{}).get(metric) if metric in FIELDS else outcomes[h].get(metric))
                    if h in outcomes else None for h in own]
            return (_sum_known([value/len(owners[h]) if value is not None else None for h,value in zip(own,values)]),
                    sum(1/len(owners[h]) for h,value in zip(own,values) if value is None))
        allocations[arm]={'canonical_attempts_referenced':len(own),
            'exclusive_attempts':sum(len(owners[h])==1 for h in own),
            'shared_attempts_referenced_nonadditive':sum(len(owners[h])>1 for h in own),
            'allocated_attempt_equivalents':sum(1/len(owners[h]) for h in own),
            'allocated_observed_tokens':{key:allocated(key)[0] for key in FIELDS},
            'allocated_missing_usage_attempt_equivalents':{key:allocated(key)[1] for key in FIELDS},
            'allocated_observed_latency_seconds':{key:allocated(key)[0] for key in latencies},
            'allocated_missing_latency_attempt_equivalents':{key:allocated(key)[1] for key in latencies}}
    reconciles={'attempts':math.isclose(sum(v['allocated_attempt_equivalents'] for v in allocations.values()),len(intents),abs_tol=1e-9)}
    for key,total in tokens.items():
        observed=[v['allocated_observed_tokens'][key] for v in allocations.values()]
        reconciles[key]=(_sum_known(observed) is None if total is None else math.isclose(_sum_known(observed),total,abs_tol=1e-9))
    for key,total in latency.items():
        observed=[v['allocated_observed_latency_seconds'][key] for v in allocations.values()]
        reconciles[key]=(_sum_known(observed) is None if total is None else math.isclose(_sum_known(observed),total,abs_tol=1e-9))
    if not all(reconciles.values()):raise ValueError('canonical cost allocation did not reconcile')
    return {'unique_provider_attempts':len(intents),'outcome_receipts':len(outcomes),
        'observed_tokens':tokens,'missing_usage_attempts':{k:sum(v is None for v in vals) for k,vals in token_values.items()},
        'observed_latency_seconds':latency,'missing_latency_attempts':{k:sum(v is None for v in vals) for k,vals in latencies.items()},
        'exclusive_attempts':sum(len(v)==1 for v in owners.values()),'shared_attempts':sum(len(v)>1 for v in owners.values()),
        'canonical_owners':owners,'arm_allocation_rule':'equal division across distinct arms owning each canonical request',
        'arm_allocations':allocations,'allocation_reconciles':reconciles,'monetary_cost':None,
        'allocation_is_measured_per_arm_wall_time':False}


def summarize(proposal_dir,assessment_dir,output,*,fixture_only=False):
    if not fixture_only:D.require_colab()
    proposal_dir=Path(proposal_dir).resolve(strict=True);assessment_dir=Path(assessment_dir).resolve(strict=True);output=Path(output).resolve()
    proposal=D.read_json(proposal_dir/'proposal.json');audit=D.read_json(assessment_dir/'audit.json')
    if proposal.get('proposal_sha256')!=S.sha({k:v for k,v in proposal.items() if k!='proposal_sha256'}):raise ValueError('proposal checksum mismatch')
    if proposal.get('fixture_only')!=fixture_only or audit.get('fixture_only')!=fixture_only:raise ValueError('fixture/live identity mismatch')
    S._verify_hashes(proposal['code_sha256']);requests=D.read_json(proposal_dir/'requests.json')
    if S.sha(requests)!=proposal['requests_sha256'] or len(requests)!=9:raise ValueError('requests checksum mismatch')
    if audit.get('proposal_sha256')!=proposal['proposal_sha256'] or audit.get('version')!=assessor.VERSION or audit.get('role')!='assessor_only' or audit.get('frozen_inputs_unchanged') is not True:raise ValueError('foreign or incomplete assessment')
    data=S._load_inputs(proposal['diagnostic_dir'],proposal['old_run'],proposal['product_repo'])
    if audit['diagnostic_protocol_sha256']!=data[2]['protocol_sha256'] or audit['original_target_file_sha256']!=data[3]['files_sha256']['assessor_only/targets.jsonl']:raise ValueError('assessment protocol/target identity mismatch')
    if output.exists() or any(output.is_relative_to(root) for root in (proposal_dir,assessment_dir,data[0],data[1])):raise ValueError('new independent summary directory required')
    S._verify_hashes({proposal['old_cost_artifact']['path']:proposal['old_cost_artifact']['sha256']})
    intents,outcomes,status_counts=assessor.audit_journal(proposal_dir,proposal,requests)
    request_by_key={(r['case_id'],r['arm']):r for r in requests};rows=audit['rows']
    if len(rows)!=9 or {(r['case_id'],r['arm']) for r in rows}!=set(request_by_key):raise ValueError('assessment row identities mismatch')
    for row in rows:
        request=request_by_key[(row['case_id'],row['arm'])];key=request['request_sha256'];receipt=outcomes.get(key)
        if row['request_sha256']!=key:raise ValueError('assessment refers to another request')
        expected_status=receipt['status'] if receipt else 'intent_without_outcome' if key in intents else 'not_attempted'
        if row['provider_status']!=expected_status:raise ValueError('assessment/provider status mismatch')
        if row['response_sha256'] is not None:
            if receipt is None or receipt['status']!='published' or row['response_sha256']!=hashlib.sha256(receipt['response_text'].encode()).hexdigest():raise ValueError('assessment/provider body mismatch')
        elif row['answer_accuracy'] is not None:raise ValueError('answer score has no published response')
        if row['answer_status']=='correct' and (row['answer_accuracy'] is not True or row['quote_identity_correct'] is not True or row['source_binding_correct'] is not True):raise ValueError('inconsistent strict correct flags')
    grouped={case:{row['arm']:row for row in rows if row['case_id']==case} for case in proposal['selected_cases']}
    pairs=[]
    for case,arms in grouped.items():
        raw=arms['raw_history_rag']['answer_accuracy'];managed=arms['managed_memory_rag']['answer_accuracy']
        outcome='unresolved' if raw is None or managed is None else 'managed_win' if managed and not raw else 'raw_win' if raw and not managed else 'tie_both_correct' if raw else 'tie_both_incorrect'
        pairs.append({'case_id':case,'position':arms['raw_history_rag']['position'],'raw_correct':raw,'managed_correct':managed,'paired_outcome':outcome})
    annotated=[{k:row[k] for k in ('case_id','position','arm','request_sha256','memory_input_stage','answer_status','answer_accuracy')}|{'observable_error_stage':observable_error(row)} for row in rows]
    summary={'version':VERSION,'proposal_sha256':proposal['proposal_sha256'],'assessment_sha256':D.file_hash(assessment_dir/'audit.json'),
        'summary_code_sha256':D.file_hash(__file__),
        'source_journal_sha256':{name:D.file_hash(proposal_dir/name) if (proposal_dir/name).exists() else None for name in ('call_intents.jsonl','call_outcomes.jsonl')},
        'paired_raw_vs_managed':{'planned_pairs':3,'counts':dict(Counter(p['paired_outcome'] for p in pairs)),'pairs':pairs},
        'by_arm':{arm:{'correct_completed_answers':sum(row['answer_accuracy'] is True for row in rows if row['arm']==arm),'planned_rows':3,
            'error_stage_counts':dict(Counter(row['observable_error_stage'] for row in annotated if row['arm']==arm))} for arm in ARMS},
        'provider_status_counts':status_counts,'new_answer_cost':aggregate_cost(requests,intents,outcomes),
        'local_embedding_cost':{'embedding_load_seconds':data[2].get('embedding_load_seconds'),'local_embedding_work':data[2].get('local_embedding_work'),
            'per_arm_embedding_seconds':None,'scope':'final frozen diagnostic only; shared model, combined raw and candidate encoding time'},
        'reused_memory_cost_reference':proposal['old_cost_artifact'],'human_reviews':0,'new_provider_calls_this_summary':0,'secret_access':False,
        'interpretation':'Strict quotation/source correctness and observable pipeline errors on three related development probes. Cost allocation is accounting; encoder work is separate from answer correctness.'}
    output.mkdir(parents=True,mode=0o700);(output/'summary.json').write_text(json.dumps(summary,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (output/'observable_errors.jsonl').write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in annotated))
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--proposal-dir',required=True,type=Path)
    p.add_argument('--assessment-dir',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();print(json.dumps(summarize(args.proposal_dir,args.assessment_dir,args.output),sort_keys=True))

if __name__=='__main__':main()
