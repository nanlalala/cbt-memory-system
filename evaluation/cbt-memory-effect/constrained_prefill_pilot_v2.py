"""Four prospectively frozen readability diagnostics; no loader or model ledger.

Original two questions, unchanged: two unconstrained references, then two common
whitespace-prefill conditions. Only authenticated prefill outputs determine the
prospective main-run gate. This is not CBT quality or memory-effect evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import constrained_open_reader as v1
import constrained_json_prefill_v2 as grammar
import open_reader_colab as reader

VERSION='four-fixed-reference-whitespace-prefill-pilots-v2'
MODES=('unconstrained','prefill_v2')
FROZEN_CASES=v1.FIXTURES


def requests():
    return [{**row,'arm':mode,'decode_mode':mode,'logical_id':mode+':'+row['case_id']}
            for mode in MODES for row in v1.requests()]


def code_identity():
    return {str(p.resolve()):reader.file_hash(p) for p in (Path(__file__),Path(grammar.__file__),
        Path(grammar.previous.__file__),Path(v1.__file__),Path(reader.__file__))}


def prepare(output):
    reader.require_colab();output=Path(output).resolve()
    if not output.is_relative_to(Path('/content')) or output==Path('/content'):
        raise ValueError('named_hosted_preparation_required')
    if output.exists():raise FileExistsError('new_pilot_preparation_required')
    output.mkdir(parents=True,mode=0o700);rows=requests()
    reader._write_new(output/'requests.json',rows)
    assessor=output/'assessor_only';assessor.mkdir(mode=0o700)
    reader._write_new(assessor/'targets.json',{cid:value for cid,_,value in FROZEN_CASES})
    protocol={'version':VERSION,'code_sha256':code_identity(),'requests_sha256':reader.digest(rows),
        'targets_file_sha256':reader.file_hash(assessor/'targets.json'),'logical_rows':4,
        'fixed_order':['unconstrained_count','unconstrained_outcome','prefill_count','prefill_outcome'],
        'same_original_v1_questions':True,'v1_results_preserved_not_retried':True,
        'reference_evaluation':'retain exact unconstrained raw text; no invented format/quality score',
        'main_gate':'both prefill_v2 outputs strict schema valid AND typed-value correct',
        'v2_policy':grammar.policy(),'post_hoc_selection_or_retries':False,
        'material':'two invented non-CBT questions, each under two prospective decoding conditions',
        'new_human_reviews':0,'CBT_quality_evidence':False,'memory_effect_evidence':False}
    reader._write_new(output/'protocol.json',protocol)
    return {'prepared':True,'logical_rows':4,'protocol_sha256':reader.file_hash(output/'protocol.json')}


def validate_prepared(root):
    root=Path(root);protocol=reader.read_json(root/'protocol.json');rows=reader.read_json(root/'requests.json')
    if (protocol.get('version')!=VERSION or protocol.get('code_sha256')!=code_identity()
        or protocol.get('v2_policy')!=grammar.policy() or protocol.get('logical_rows')!=4
        or rows!=requests() or protocol.get('requests_sha256')!=reader.digest(rows)):
        raise ValueError('prospectively_frozen_pilots_changed')
    return protocol,rows


def strict_answer(text):
    def unique(pairs):
        obj={}
        for key,value in pairs:
            if key in obj:raise ValueError('duplicate_field')
            obj[key]=value
        return obj
    value=json.loads(text,object_pairs_hook=unique,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite')))
    if type(value) is not dict or set(value)!={'answer'}:raise ValueError('single_answer_object_required')
    answer=value['answer']
    if not (answer is None or type(answer) is int and 0<=answer<=9999
        or type(answer) is str and answer in grammar.STATUSES):raise ValueError('outside_fixed_domain')
    return answer


def score_authenticated(answers,*,prepared_dir):
    """Caller MUST first authenticate all four against new ledger/cache/input.

    This function opens labels only after complete frozen logical coverage.
    Unconstrained text is reference material, not post-hoc semantic gold scoring.
    """
    protocol,planned=validate_prepared(prepared_dir)
    key=lambda row:(row['case_id'],row.get('decode_mode',row.get('arm')))
    if (len(answers)!=4 or len({key(a) for a in answers})!=4
        or [key(a) for a in answers]!=[key(r) for r in planned]
        or any(a.get('status')!='completed' or type(a.get('response_text')) is not str for a in answers)):
        raise ValueError('all_four_authenticated_outputs_required_before_labels')
    target_path=Path(prepared_dir)/'assessor_only/targets.json'
    if reader.file_hash(target_path)!=protocol['targets_file_sha256']:raise ValueError('frozen_targets_changed')
    targets=reader.read_json(target_path)
    if targets!={cid:value for cid,_,value in FROZEN_CASES}:raise ValueError('same_fixed_v1_question_labels_required')
    rows=[];schema=correct=0
    for answer in answers:
        mode=key(answer)[1]
        row={'case_id':answer['case_id'],'decode_mode':mode,
            'response_sha256':hashlib.sha256(answer['response_text'].encode()).hexdigest()}
        if mode=='unconstrained':
            row.update({'reference_raw_text':answer['response_text'],'schema_valid':None,
                'basic_readability_correct':None,'reference_only_not_scored':True})
        else:
            try:value=strict_answer(answer['response_text']);valid=True
            except (ValueError,TypeError,json.JSONDecodeError):value=None;valid=False
            expected=targets[answer['case_id']]
            is_correct=valid and type(value) is type(expected) and value==expected
            schema+=valid;correct+=is_correct
            row.update({'schema_valid':valid,'chosen_answer':value,
                'basic_readability_correct':is_correct,'reference_only_not_scored':False})
        rows.append(row)
    return {'version':VERSION,'logical_rows':4,'schema_valid_by_mode':{'unconstrained':None,'prefill_v2':schema},
        'basic_readability_by_mode':{'unconstrained':None,'prefill_v2':correct},
        'prefill_schema_valid':schema,'prefill_basic_readability_correct':correct,'prefill_denominator':2,
        'ready_for_main':schema==2 and correct==2,'rows':rows,
        'decision_policy':'if prefill gate fails stop this whole study; no alternative cases or automatic retries',
        'new_model_calls_by_scoring':0,'new_human_reviews':0,'CBT_quality_evidence':False,
        'memory_effect_evidence':False,'independent_holdout':False,
        'prepared_protocol_sha256':reader.file_hash(Path(prepared_dir)/'protocol.json')}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    print(reader.canonical(prepare(a.output)),flush=True)
