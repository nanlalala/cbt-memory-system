"""Post-hoc lossless field-name correction of an already completed hosted run.

The original archive, parser, judgments and paid receipts remain unchanged.
No model, credential or network access is used by this rescore.
"""
from collections import Counter
import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'evaluation/cbt-memory-effect'))
from auto_eval_v2 import calibration, calibration_prompt
from architecture_retest.journal import digest

spec = importlib.util.spec_from_file_location('calibration_adapter', ROOT/'evaluation/architecture-calibration/run_calibration.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)

arguments = argparse.ArgumentParser(description=__doc__)
arguments.add_argument('--archive', type=Path, required=True)
arguments.add_argument('--output', type=Path, required=True)
args = arguments.parse_args()
SOURCE = args.archive
EXPECTED = '96ed73f205caf61323a526cdc97cdb202ae3c5669d67d31538b3770314f075d0'
assert sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED
with ZipFile(SOURCE) as archive:
    protocol = json.loads(archive.read('cbt-v4-calibration/protocol.json'))
    plan = json.loads(archive.read('cbt-v4-calibration/plan.json'))
    cases = json.loads(archive.read('cbt-v4-calibration/cases.json'))
    intents = [json.loads(line) for line in archive.read('cbt-v4-calibration/intents.jsonl').splitlines()]
    outcomes = [json.loads(line) for line in archive.read('cbt-v4-calibration/outcomes.jsonl').splitlines()]
    published_bytes = archive.read('cbt-v4-calibration-bank/assessor_only/therapyjudgebench.jsonl')
    old_summary = json.loads(archive.read('cbt-v4-calibration/summary.json'))
assert sha256(published_bytes).hexdigest() == calibration.SOURCE_SHA256
for module in (calibration, calibration_prompt):
    module_path = Path(module.__file__).resolve()
    assert sha256(module_path.read_bytes()).hexdigest() == protocol['code_sha256'][str(module_path.relative_to(ROOT))]
assert digest(cases) == protocol['cases_sha256'] and digest(plan) == protocol['plan_sha256']
assert len(cases) == 106 and len(plan) == len(intents) == len(outcomes) == 117
assert len({row['key'] for row in outcomes}) == len(outcomes)
assert all(row['status']=='received' and row['provider_model']=='qwen3.5-plus-2026-04-20' for row in outcomes)
by_key = {row['key']:row for row in outcomes}
for intent in intents:
    assert sha256(json.dumps(intent['payload'],allow_nan=False).encode()).hexdigest() == intent['wire_sha256']
    assert intent['key'] in by_key and intent['protocol_sha256']==protocol['protocol_sha256']

packets = {row['case_id']:row for row in cases}
reviews, corrections = [], []
before_statuses, after_statuses = Counter(), Counter()
for item in plan:
    key=digest({'purpose':item['purpose'],'payload':item['payload']})
    receipt=by_key[key]
    raw=receipt['text']
    before=calibration_prompt.validate_review(raw,packets[item['case_id']])
    value=calibration_prompt._read_json(raw)
    changed_aliases=[]
    for section, dimensions in (('ctrs',calibration.CTRS),('safety',calibration.SAFETY)):
        for canonical_name in dimensions:
            alias=canonical_name.replace(' ','_')
            if alias!=canonical_name and alias in value.get(section,{}) and canonical_name not in value[section]:
                value[section][canonical_name]=value[section].pop(alias)
                changed_aliases.append(section+'.'+alias)
    normalized=json.dumps(value,ensure_ascii=False)
    after=calibration_prompt.validate_review(normalized,packets[item['case_id']])
    before_statuses[before['status']]+=1
    after_statuses[after['status']]+=1
    reviews.append({'item_id':item['item_id'],'case_id':item['case_id'],'repeat_of':item['repeat_of'],
                    'delivery':'received','review':after})
    corrections.append({'item_id':item['item_id'],'case_id':item['case_id'],'repeat_of':item['repeat_of'],
        'request_key':key,'native_response_sha256':sha256(raw.encode()).hexdigest(),
        'field_aliases_corrected':changed_aliases,'original_parser_status':before['status'],'rescored_parser_status':after['status']})
    assert after['accepted'] and after['status']=='completed'
    original=json.loads(raw)
    assert original['reason']==value['reason']
    for section in ('ctrs','safety'):
        assert {k.replace('_',' '):v for k,v in original[section].items()}==value[section]

published={row['id']:row for row in map(json.loads,published_bytes.splitlines())}
gold=[]
for packet in cases:
    record=published[packet['case_id']]
    assert calibration.project_dialogue(record)['messages']==packet['messages']
    labels=record['expert_rating']
    gold.append({'case_id':packet['case_id'],'ctrs':labels['ctrs_ratings'],'safety':labels['safety_ethics']})
predictions=[{'case_id':row['case_id'],'status':'completed','ctrs':row['review']['assessment']['ctrs'],
              'safety':row['review']['assessment']['safety']} for row in reviews if row['repeat_of'] is None]
results=calibration.calibrate(gold,predictions)
safety_values={d:dict(Counter(str(row['safety'][d]) for row in predictions)) for d in calibration.SAFETY}
report={
 'record_type':'Lossless post-hoc key-alias rescore of completed hosted model receipts',
 'version':'cbt-v4-published-calibration-mechanical-analysis-v2',
 'original_archive_sha256':EXPECTED,'original_protocol_sha256':protocol['protocol_sha256'],
 'published_source_sha256':calibration.SOURCE_SHA256,
 'transformation':{'aliases':{'ctrs':{'INTERPERSONAL_EFFECTIVENESS':'INTERPERSONAL EFFECTIVENESS'},
                  'safety':{d.replace(' ','_'):d for d in calibration.SAFETY}},
    'scope':'only declared space-underscore aliases when canonical key absent; every rating value and reason unchanged',
    'native_responses_or_original_summary_modified':False,'new_model_calls':0},
 'planned_primary':106,'planned_repeats':11,'received_provider_calls':117,
 'original_all_review_statuses':dict(before_statuses),'rescored_all_review_statuses':dict(after_statuses),
 'alias_changed_primary':sum(bool(r['field_aliases_corrected']) and r['repeat_of'] is None for r in corrections),
 'alias_changed_repeats':sum(bool(r['field_aliases_corrected']) and r['repeat_of'] is not None for r in corrections),
 'original_primary_calibration':old_summary['calibration'],
 'rescored_primary_calibration':results,
 'rescored_repeat_consistency':adapter.repeat_consistency(reviews),
 'primary_safety_prediction_counts':safety_values,'tokens':old_summary['tokens'],
 'conclusions':[
  'The strict parser rejected 40 primary and 3 repeat responses solely because five field names used underscores instead of spaces.',
  'The separate lossless alias rescore retains all original numeric and safety judgments.',
  'The judge provides imperfect ordinal CTRS agreement and systematic dimension-specific bias.',
  'Safety flags show failed sensitivity to released positive labels and cannot certify safety.',
  'This source does not calibrate custom R1 factual continuity or R3 next-step memory judgments.'
 ],'per_item_alias_audit':corrections}
output=args.output
assert output.resolve()!=SOURCE.resolve()
output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'output':str(output),'primary':results['prediction_status_counts'],
 'ctrs':{k:{a:v[a] for a in ['n_compared','spearman','mae','mean_bias_prediction_minus_expert']} for k,v in results['ctrs'].items()},
 'safety':results['safety'],'repeat':report['rescored_repeat_consistency']},indent=2))
