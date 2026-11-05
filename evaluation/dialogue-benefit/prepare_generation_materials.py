"""Compile fictional CBT follow-up materials without sending assessor gold to a model.

The task snapshots are the only answer/extraction context input. replay_events is
an ENGINE-ONLY chronological local event stream, never a model payload: it exists
so historically admitted sources can actually be ingested and subsequently
removed. Scoring references are written separately and are not read by the public
package validator. These researcher-authored materials are development cases,
not independently validated references, people, consent or clinical evidence.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

from study_capture import _verify_task_ledger, canonical_hash

SCHEMA = 'cbt-source-generation-v1'
SOURCE_FIELDS = {'source_id', 'source_order', 'visit', 'date', 'role', 'text', 'text_sha256', 'storage_policy'}
CONTROL_FIELDS = {'event_id', 'action', 'target_source_ids', 'as_of', 'source_order', 'before_task_id'}
TASK_FIELDS = {'case_id', 'variant', 'task_id', 'visit', 'as_of', 'query', 'query_sha256',
               'history_prefix_source_order', 'start_new_session', 'visible_source_ids',
               'current_session_source_ids', 'history', 'controls', 'exclusions'}
ROOT_FIELDS = {'schema_version', 'materials_version', 'materials_sha256', 'purpose', 'origin_policy',
               'independent_human_reference_validator_count', 'clinical_claim', 'variants', 'tasks',
               'model_input_policy', 'generation_package_sha256'}
ID_RE = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}\Z')
HASH_RE = re.compile(r'[0-9a-f]{64}\Z')


def sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def time(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError('Source timestamp must be an explicit timezone-aware string')
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Source timestamp requires a timezone')
    return result.astimezone(timezone.utc)


def source_projection(source: dict) -> dict:
    result = {'source_id': source['source_id'], 'source_order': source['source_order'],
              'visit': source['visit'], 'date': source['date'], 'role': source['role'],
              'text': source['source_text'], 'text_sha256': source['source_text_sha256'],
              'storage_policy': source['storage_policy']}
    if 'valid_to' in source:
        result['valid_to'] = source['valid_to']
    return result


def _verify_sources(sources: list[dict]) -> dict[str, list[dict]]:
    grouped = {variant: [] for variant in 'ABC'}
    ids = set()
    for source in sources:
        if set(source) not in (SOURCE_FIELDS, SOURCE_FIELDS | {'valid_to'}):
            raise ValueError('Source contains non-whitelisted fields')
        sid = source['source_id']
        if not isinstance(sid, str) or not ID_RE.fullmatch(sid) or ':' in sid or sid in ids or sid[:1] not in grouped:
            raise ValueError('Source identity is duplicate or outside the fixed variants')
        ids.add(sid)
        if type(source['source_order']) is not int or source['source_order'] < 1 or type(source['visit']) is not int or not 1 <= source['visit'] <= 4:
            raise ValueError('Source order/visit is invalid')
        if source['role'] not in {'user', 'assistant', 'system_observation'} or source['storage_policy'] not in {'allowed', 'session_only'}:
            raise ValueError('Unsupported source role/storage policy')
        if not isinstance(source['text'], str) or not source['text'].strip() or sha(source['text'].encode()) != source['text_sha256']:
            raise ValueError('Exact source text hash differs')
        at = time(source['date'])
        if 'valid_to' in source and time(source['valid_to']) <= at:
            raise ValueError('Source expiry must follow its original date')
        grouped[sid[:1]].append(source)
    for variant, rows in grouped.items():
        rows.sort(key=lambda row: row['source_order'])
        if len(rows) != 16 or [r['source_order'] for r in rows] != list(range(1, 17)):
            raise ValueError('Each fixed variant must contain its original 16 ordered sources')
        if any(time(a['date']) > time(b['date']) or a['visit'] > b['visit'] for a, b in zip(rows, rows[1:])):
            raise ValueError('Source dates/visits disagree with chronological source order')
    return grouped


def _verify_controls(events: list[dict], sources: dict[str, list[dict]]) -> dict[str, list[dict]]:
    grouped = {variant: [] for variant in 'ABC'}
    event_ids = set()
    for event in events:
        if set(event) != {'variant', 'kind', 'control'} or event['kind'] != 'control' or event['variant'] not in grouped:
            raise ValueError('Non-whitelisted control event')
        control = event['control']
        if not isinstance(control, dict) or set(control) != CONTROL_FIELDS:
            raise ValueError('Non-whitelisted control payload')
        variant = event['variant']; inventory = {s['source_id']: s for s in sources[variant]}
        if not isinstance(control['event_id'], str) or not ID_RE.fullmatch(control['event_id']) or control['event_id'] in event_ids:
            raise ValueError('Duplicate control event')
        event_ids.add(control['event_id'])
        order = control['source_order']
        if type(order) is not int or order not in range(1, 17) or time(control['as_of']) != time(sources[variant][order - 1]['date']):
            raise ValueError('Control date/order is not bound to its source boundary')
        targets = control['target_source_ids']
        if not isinstance(targets, list) or len(targets) != len(set(targets)) or any(t not in inventory for t in targets):
            raise ValueError('Control targets an unknown or foreign source')
        if control['action'] in {'delete', 'do_not_store'}:
            if control['before_task_id'] is not None or not targets:
                raise ValueError('Source controls require target IDs and no task selector')
            if any(inventory[t]['source_order'] > order for t in targets):
                raise ValueError('Control cannot target a future source')
            if control['action'] == 'do_not_store' and any(inventory[t]['storage_policy'] != 'session_only' for t in targets):
                raise ValueError('Do-not-store control must match session-only policy')
        elif control['action'] == 'new_session':
            if targets or control['before_task_id'] != 'T12' or order != 16:
                raise ValueError('New-session control must match the fixed task boundary')
        else:
            raise ValueError('Unsupported source-control action')
        grouped[variant].append(control)
    for controls in grouped.values():
        if len(controls) != 3 or sorted(c['action'] for c in controls) != ['delete', 'do_not_store', 'new_session']:
            raise ValueError('Each fictional variant needs its three original privacy/session controls')
        controls.sort(key=lambda c: (c['source_order'], c['before_task_id'] or ''))
    return grouped


def snapshot_task(task: dict, sources: list[dict], controls: list[dict]) -> dict:
    """Build a per-task model-eligible prefix; never accept scorer fields here."""
    variant, prefix, visit = task['variant'], task['history_prefix_source_order'], task['visit']
    if type(prefix) is not int or not 1 <= prefix <= len(sources) or visit != sources[prefix - 1]['visit']:
        raise ValueError('Task history boundary/visit differs from the source prefix')
    as_of = sources[prefix - 1]['date']
    effective = [deepcopy(c) for c in controls if c['source_order'] <= prefix and
                 (c['before_task_id'] is None or c['before_task_id'] == task['task_id'])]
    new_session = any(c['action'] == 'new_session' for c in effective)
    if type(task['start_new_session']) is not bool or task['start_new_session'] != new_session:
        raise ValueError('Task reset flag differs from the original explicit control')
    deleted = {sid for c in effective if c['action'] == 'delete' for sid in c['target_source_ids']}
    eligible, excluded = [], []
    for source in sources[:prefix]:
        if time(source['date']) > time(as_of):
            raise ValueError('Future source in task prefix')
        reason = ('deleted' if source['source_id'] in deleted else
                  'expired' if source.get('valid_to') and time(source['valid_to']) <= time(as_of) else
                  'dont_save' if source['storage_policy'] == 'session_only' and (source['visit'] != visit or new_session) else None)
        if reason:
            excluded.append({'source_id': source['source_id'], 'reason': reason})
        else:
            eligible.append(deepcopy(source))
    visible = [s['source_id'] for s in eligible]
    if task['visible_source_ids'] != visible:
        raise ValueError('Declared task history differs from privacy-filtered exact prefix')
    query = task.get('query', task.get('task_query'))
    query_hash = task.get('query_sha256', task.get('task_query_sha256'))
    if not isinstance(query, str) or not query.strip() or sha(query.encode()) != query_hash:
        raise ValueError('Task query identity differs')
    return {'case_id': f"{variant}:{task['task_id']}", 'variant': variant, 'task_id': task['task_id'],
            'visit': visit, 'as_of': as_of, 'query': query, 'query_sha256': query_hash,
            'history_prefix_source_order': prefix, 'start_new_session': new_session,
            'visible_source_ids': visible,
            'current_session_source_ids': [] if new_session else [s['source_id'] for s in eligible if s['visit'] == visit],
            'history': eligible, 'controls': effective, 'exclusions': excluded}


def validate_generation_package(generation: dict, replay_events: list[dict]) -> None:
    """Validate source-only objects without opening any scoring-reference file."""
    if not isinstance(generation, dict) or set(generation) != ROOT_FIELDS or generation['schema_version'] != SCHEMA:
        raise ValueError('Unexpected source-only package schema/fields')
    if generation['generation_package_sha256'] != canonical_hash({k: v for k, v in generation.items() if k != 'generation_package_sha256'}):
        raise ValueError('Generation-package identity differs')
    if (generation['materials_version'] != 'four-visit-cbt-fictional-task-draft-v1' or
            not isinstance(generation['materials_sha256'], str) or not HASH_RE.fullmatch(generation['materials_sha256'])):
        raise ValueError('Unknown fictional materials identity')
    if (generation['purpose'] != 'fictional_development_not_human_trial' or generation['clinical_claim'] is not False or
            type(generation['independent_human_reference_validator_count']) is not int or
            generation['independent_human_reference_validator_count'] != 0 or generation['variants'] != list('ABC') or
            generation['origin_policy'] != 'fictional_ledger_observations_not_authenticated_consent' or
            generation['model_input_policy'] != 'only_validated_per_task_context_never_full_package_or_engine_event_file'):
        raise ValueError('Development/consent/model-payload boundary differs')
    source_records, control_records = [], []
    for event in replay_events:
        if not isinstance(event, dict) or event.get('variant') not in tuple('ABC'):
            raise ValueError('Invalid replay event variant')
        if event.get('kind') == 'source':
            if set(event) != {'variant', 'kind', 'source'} or not isinstance(event['source'], dict) or event['source'].get('source_id', '')[:1] != event['variant']:
                raise ValueError('Non-whitelisted or cross-variant source replay')
            source_records.append(event['source'])
        elif event.get('kind') == 'control':
            control_records.append(event)
        else:
            raise ValueError('Unsupported replay event kind')
    grouped = _verify_sources(source_records)
    controls = _verify_controls(control_records, grouped)
    canonical_events = []
    for variant in 'ABC':
        for source in grouped[variant]:
            canonical_events.append({'variant': variant, 'kind': 'source', 'source': source})
            canonical_events.extend({'variant': variant, 'kind': 'control', 'control': control} for control in controls[variant] if control['source_order'] == source['source_order'])
    if replay_events != canonical_events:
        raise ValueError('Replay events must follow the original source/control chronology')
    tasks = generation['tasks']
    if not isinstance(tasks, list) or len(tasks) != 36:
        raise ValueError('Generation requires all 36 fixed development cases')
    seen = set()
    for task in tasks:
        if not isinstance(task, dict) or set(task) != TASK_FIELDS or task['variant'] not in tuple('ABC') or task['task_id'] not in {f'T{i:02d}' for i in range(1, 13)}:
            raise ValueError('Unexpected task field or identity')
        pair = (task['variant'], task['task_id'])
        if pair in seen:
            raise ValueError('Duplicate task identity')
        seen.add(pair)
        if task != snapshot_task(task, grouped[task['variant']], controls[task['variant']]):
            raise ValueError('Task context/reset/controls differ from authoritative replay prefix')
    if seen != {(v, f'T{i:02d}') for v in 'ABC' for i in range(1, 13)}:
        raise ValueError('Generation task census differs')


def compile_materials(ledger: dict) -> tuple[dict, list[dict], list[dict]]:
    _verify_task_ledger(ledger)
    sources = _verify_sources([source_projection(s) for s in ledger['sources']])
    source_lookup = {s['source_id']: s for rows in sources.values() for s in rows}
    events = []
    for item in ledger.get('source_controls', []):
        variant = item['variant']; before = item.get('effective_before_task_id')
        trigger = source_lookup[item['effective_after_source_id']] if not before else sources[variant][-1]
        if item.get('scope') not in {'all_future_retrieval_and_prompt_use', 'session_3_only_never_long_term', 'clear_short_term_ephemeral_state_preserve_permitted_long_term_only'}:
            raise ValueError('Unsupported original source-control scope')
        events.append({'variant': variant, 'kind': 'control', 'control': {
            'event_id': item['event_id'], 'action': item['action'], 'target_source_ids': list(item.get('target_source_ids', [])),
            'as_of': trigger['date'], 'source_order': trigger['source_order'], 'before_task_id': before}})
    controls = _verify_controls(events, sources)
    replay_events = []
    for variant in 'ABC':
        for source in sources[variant]:
            replay_events.append({'variant': variant, 'kind': 'source', 'source': deepcopy(source)})
            replay_events.extend({'variant': variant, 'kind': 'control', 'control': deepcopy(c)} for c in controls[variant] if c['source_order'] == source['source_order'])
    tasks, refs = [], []
    for task in ledger['tasks']:
        variant = task['variant']
        snapshot = snapshot_task(task, sources[variant], controls[variant])
        absent = {row['source_id']: row['reason'] for row in snapshot['exclusions']}
        for reference in task['source_reference']:
            original = source_lookup[reference['source_id']]
            if reference.get('source_text') != original['text']:
                raise ValueError('Scoring reference source text differs from the original source')
            if original['source_order'] > task['history_prefix_source_order']:
                raise ValueError('Scoring reference uses a future source')
            if reference.get('permitted_absence_reason') != absent.get(reference['source_id']):
                raise ValueError('Scoring reference absence differs from control/expiry policy')
        tasks.append(snapshot)
        refs.append({'case_id': snapshot['case_id'], 'variant': variant, 'task_id': task['task_id'], 'visit': task['visit'],
                     'query_sha256': snapshot['query_sha256'], 'completion_criterion': task['completion_criterion'],
                     'criterion_sha256': task['criterion_sha256'], 'source_reference': deepcopy(task['source_reference']),
                     'process_focus': task['process_focus'], 'reference_status': task['reference_status'],
                     'materials_sha256': canonical_hash(ledger)})
    result = {'schema_version': SCHEMA, 'materials_version': ledger['materials_version'], 'materials_sha256': canonical_hash(ledger),
              'purpose': 'fictional_development_not_human_trial', 'origin_policy': 'fictional_ledger_observations_not_authenticated_consent',
              'independent_human_reference_validator_count': 0, 'clinical_claim': False, 'variants': list('ABC'), 'tasks': tasks,
              'model_input_policy': 'only_validated_per_task_context_never_full_package_or_engine_event_file'}
    result['generation_package_sha256'] = canonical_hash(result)
    validate_generation_package(result, replay_events)
    return result, replay_events, refs


def _bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def _rows(value) -> bytes:
    return ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n' for row in value).encode()


def prepare_materials(ledger_path: Path, output_dir: Path) -> dict:
    ledger_path, output_dir = Path(ledger_path), Path(output_dir)
    if ledger_path.is_symlink() or not ledger_path.is_file() or output_dir.exists() or output_dir.is_symlink():
        raise ValueError('Use an existing regular ledger and a fresh output directory')
    generation, replay_events, refs = compile_materials(json.loads(ledger_path.read_text()))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output_dir.name + '.tmp-', dir=output_dir.parent))
    try:
        artifacts = {'generation_materials.json': _bytes(generation), 'replay_events.jsonl': _rows(replay_events), 'scoring_references.jsonl': _rows(refs)}
        for name, body in artifacts.items():
            (staging / name).write_bytes(body)
        manifest = {'schema_version': SCHEMA, 'materials_sha256': generation['materials_sha256'],
                    'original_ledger_file_sha256': sha(ledger_path.read_bytes()), 'compiler_sha256': sha(Path(__file__).read_bytes()),
                    'artifact_sha256': {name: sha(body) for name, body in artifacts.items()},
                    'cases': 36, 'variant_count': 3, 'independent_people': 0, 'source_turns': 48, 'replay_events': 57,
                    'human_source_validators': 0, 'clinical_claim': False, 'engine_replay_must_never_be_model_payload': True,
                    'scoring_references_must_never_be_generation_input': True,
                    'expiry_cases_present': any('valid_to' in event.get('source', {}) for event in replay_events),
                    'status': 'fictional_development_materials_prepared_not_human_study_launched'}
        (staging / 'manifest.json').write_bytes(_bytes(manifest))
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def read_prepared_generation(root: Path) -> dict:
    """Load only generation/controller files; never open scoring_references."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Expected a regular prepared source directory')
    manifest_path = root / 'manifest.json'
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError('Expected a regular preparation manifest')
    manifest = json.loads(manifest_path.read_text())
    data = {}
    for name in ('generation_materials.json', 'replay_events.jsonl'):
        path = root / name
        if path.is_symlink() or not path.is_file() or sha(path.read_bytes()) != manifest['artifact_sha256'][name]:
            raise ValueError('Prepared generation/controller file identity differs')
        data[name] = json.loads(path.read_text()) if name.endswith('.json') else [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    validate_generation_package(data['generation_materials.json'], data['replay_events.jsonl'])
    return {'generation': data['generation_materials.json'], 'replay_events': data['replay_events.jsonl'], 'manifest': manifest}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, default=Path(__file__).with_name('task_materials_draft.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_materials(args.ledger, args.output_dir), ensure_ascii=False, indent=2))
