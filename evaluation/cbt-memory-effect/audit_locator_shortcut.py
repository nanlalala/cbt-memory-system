"""Hosted-only assessor of an explicit prefix-locator shortcut, with zero API use.

Matching reads frozen generation inputs only. Gold is opened only after those
matches are written. No LLM answers, source-ID joint scores or management gains
are produced. Every family whose targets are opened is registered as assessed.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import prepare_context_revision_v3 as common
VERSION='frozen-prefix-locator-shortcut-audit-v1'
MARKER='began with this locator cue: '
PERMISSIONS={key:True for key in ('save','long_term_use','response_external','extraction_external')}

def text_sha(value):return hashlib.sha256(value.encode()).hexdigest()
def write_new(path,value):
    with Path(path).open('x') as stream:stream.write(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n')

def parse_cue(text):
    if type(text) is not str or text.count(MARKER)!=1:raise ValueError('one explicit JSON locator cue required')
    cue,offset=json.JSONDecoder().raw_decode(text.split(MARKER,1)[1])
    if type(cue) is not str or not cue or len(cue.split())!=4:raise ValueError('four-word JSON string locator required')
    return cue

def legal_clients(case,product):
    cutoff=case['history_cutoff_order_exclusive'];seen=set();result=[]
    if type(cutoff) is not int or cutoff<0:raise ValueError('explicit ordinal cutoff required')
    for turn in case['history']:
        if turn['message_id'] in seen:raise ValueError('duplicate source message ID')
        seen.add(turn['message_id'])
        order=turn['source_order']
        if type(order) is not int or order<0:raise ValueError('invalid original source order')
        if turn['role']!='user' or order>=cutoff:continue
        permissions=product.permissions_for_input(turn['content'],PERMISSIONS)
        if all(permissions[key] is True for key in ('save','long_term_use','response_external')):result.append(turn)
    return result

def match_without_gold(case,product):
    cue=parse_cue(case['input'])
    matches=[turn for turn in legal_clients(case,product) if turn['content'].startswith(cue)]
    return {'case_id':case['case_id'],'source_family_id':case['source_family_id'],'case_sha256':case['case_sha256'],
        'locator_sha256':text_sha(cue),'match_count':len(matches),
        'message_id':matches[0]['message_id'] if len(matches)==1 else None,
        'quote_sha256':text_sha(matches[0]['content']) if len(matches)==1 else None}

def audit(old_run,output,*,product_repo,fixture_only=False):
    if not fixture_only:common.require_colab()
    old=Path(old_run).resolve(strict=True);output=Path(output).resolve()
    if output.exists() or output.is_relative_to(old):raise ValueError('new independent output directory required')
    protocol=common.read_json(old/'protocol.json')
    if protocol.get('protocol_sha256')!=common.digest({k:v for k,v in protocol.items() if k!='protocol_sha256'}):raise ValueError('protocol digest mismatch')
    if protocol.get('version')!='cactus-source-quotation-recall-v2' or protocol.get('clock_mode')!='ordinal-only-v1':raise ValueError('frozen ordinal v2 required')
    product=common.product_modules(product_repo)
    for relative,expected in protocol['code_sha256'].items():
        rel=Path(relative)
        if rel.is_absolute() or '..' in rel.parts or common.file_hash(Path(product_repo)/rel)!=expected:raise ValueError('frozen product code mismatch')
    gen=old/'generation/cases.jsonl';gen_sha=common.file_hash(gen)
    if gen_sha!=protocol['files_sha256']['generation/cases.jsonl']:raise ValueError('generation checksum mismatch')
    cases=common.read_rows(gen);by_id={case['case_id']:case for case in cases}
    if len(by_id)!=len(cases) or len(cases)!=protocol['ready_probes']:raise ValueError('duplicate or incomplete frozen probe set')
    for case in cases:
        if case['case_sha256']!=common.digest({k:v for k,v in case.items() if k!='case_sha256'}):raise ValueError('case digest mismatch')
    # Phase A: matches are finalized on disk before any assessor target read.
    matches=[match_without_gold(case,product) for case in cases]
    output.mkdir(parents=True,mode=0o700)
    phase_body=''.join(json.dumps(row,sort_keys=True,ensure_ascii=False)+'\n' for row in matches)
    (output/'matches_before_gold.jsonl').write_text(phase_body)
    phase_sha=common.file_hash(output/'matches_before_gold.jsonl')
    target_path=old/'assessor_only/targets.jsonl';target_sha=common.file_hash(target_path)
    if target_sha!=protocol['files_sha256']['assessor_only/targets.jsonl']:raise ValueError('target checksum mismatch')
    families=sorted({case['source_family_id'] for case in cases})
    # Register exposure before parsing gold, including a subsequently failing audit.
    write_new(output/'assessed_family_ids.json',{'scope':'all frozen families opened in this locator-shortcut assessor','family_ids':families,'may_be_called_unseen_holdout_after_this_audit':False})
    target_rows=common.read_rows(target_path);targets={row['case_id']:row for row in target_rows}
    if len(targets)!=len(target_rows) or set(by_id)!=set(targets):raise ValueError('duplicate, missing or foreign assessor target')
    results=[]
    for match in matches:
        case=by_id[match['case_id']];target=targets[case['case_id']]
        if target['case_sha256']!=case['case_sha256'] or target['source_family_id']!=case['source_family_id']:raise ValueError('target/case identity mismatch')
        turn=next((row for row in legal_clients(case,product) if row['message_id']==target['target_message_id']),None)
        if turn is None or turn['content']!=target['target_quote']:raise ValueError('target is not a legal complete original Client message')
        results.append({**match,'position':target['position'],
            'matched_target_message':match['message_id']==target['target_message_id'],
            'matched_target_quote':match['quote_sha256']==text_sha(target['target_quote']),
            'answer_status':'not_generated','answer_accuracy':None})
    if common.file_hash(gen)!=gen_sha or common.file_hash(target_path)!=target_sha or common.file_hash(output/'matches_before_gold.jsonl')!=phase_sha:raise ValueError('input or phase-A matches changed')
    def summary(rows):
        return {'probes':len(rows),'families':len({r['source_family_id'] for r in rows}),
            'match_count_histogram':dict(sorted(Counter(str(r['match_count']) for r in rows).items())),
            'unique_prefix_matches':sum(r['match_count']==1 for r in rows),
            'unique_matches_verifying_original_target':sum(r['match_count']==1 and r['matched_target_message'] and r['matched_target_quote'] for r in rows),
            'answer_accuracy':None}
    report={'version':VERSION,'all':summary(results),'by_position':{position:summary([r for r in results if r['position']==position]) for position in ('early','middle','recent')},
        'assessed_family_count':len(families),'assessed_family_ids_sha256':common.file_hash(output/'assessed_family_ids.json'),
        'all_target_families_registered_as_assessed':True,'match_phase_read_gold':False,'assessor_phase_read_gold':True,
        'actual_api_calls':0,'secret_access':False,'embedding_calls':0,'answer_status':'not_generated','answer_accuracy':None,
        'source_id_joint_score':None,'management_gain':None,
        'interpretation':'Unique prefix lookup is a deterministic locator shortcut, not an LLM answer result or memory-management improvement.'}
    assessment={'version':VERSION,'fixture_only':fixture_only,'old_protocol_sha256':protocol['protocol_sha256'],
        'generation_sha256':gen_sha,'targets_sha256':target_sha,'matches_before_gold_sha256':phase_sha,
        'code_sha256':common.file_hash(__file__),'frozen_inputs_unchanged':True,'actual_api_calls':0,'secret_access':False}
    assessment['protocol_sha256']=common.digest(assessment)
    write_new(output/'protocol.json',assessment);write_new(output/'summary.json',report)
    (output/'results.jsonl').write_text(''.join(json.dumps(row,sort_keys=True,ensure_ascii=False)+'\n' for row in results))
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--old-run',required=True,type=Path)
    p.add_argument('--product-repo',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();print(json.dumps(audit(args.old_run,args.output,product_repo=args.product_repo),sort_keys=True))

if __name__=='__main__':main()
