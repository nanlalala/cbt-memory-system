"""Eight-attempt hosted native-GGUF readability gate; never starts a main run.

Reuses the exact already exposed v4 invented materials and target bytes for a
distinct fixed public reader. Old v4 6/8 remains unchanged. Successful syntax
alone cannot pass: all eight original whole-JSON typed answers must be correct.
There is one exclusive run claim, durable admission before every generation,
and no automatic retry, continuation, package install or model download.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gguf_schema_reader_v5_colab as native

ROOT = Path('/content/cbt-qwen7b-reader-gate-20261112-v5')
OLD_GATE = Path('/content/cbt-question-schema-reader-gate-20261112-v4')
MODEL_SETUP = Path('/content/cbt-qwen7b-reader-setup-20261112-v1')
PYTHON_SETUP = Path('/content/cbt-qwen7b-python-20261112-v1')
MODEL_METADATA = Path('/content/cbt-source-and-reader-metadata-20261112-v1/qwen7b_model.json')
OLD_PROTOCOL_SHA = '276ea9a36fab20fbf8ea6f051b42136df06beb37a6928cc437ed78948fb07a67'
MODEL_METADATA_SHA = '4beda630a9a3b5c4f3bf0a0c4fdda517639d4dc359ed1c6c947165802290d8aa'
MODEL_REVISION = 'bb5d59e06d9551d752d08b292a50eb208b07ab1f'
MODEL_FILES = (
    {'name': 'qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf', 'bytes': 3993201344,
     'sha256': 'dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db'},
    {'name': 'qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf', 'bytes': 689872288,
     'sha256': '539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a'},
)
VERSION = 'official-qwen7b-gguf-eight-case-readability-gate-v5'
CAP = 8
BASE_DEPENDENCIES = {'numpy': '2.4.6', 'jinja2': '3.1.6', 'typing-extensions': '4.16.0'}
HISTORICAL_LEDGERS = (
    (Path('/content/cbt-open-reader-20261112-v1/generation_ledger.jsonl'), 64),
    (Path('/content/cbt-constrained-state-study-20261112-v2/generation_ledger.jsonl'), 4),
    (Path('/content/cbt-unconstrained-state-study-20261112-v3/generation_ledger.jsonl'), 57),
    (OLD_GATE / 'generation_ledger.jsonl', 8),
)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    return native.file_sha(path)


def hosted():
    native.schema.require_hosted_colab()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(canonical(value) + '\n')
        stream.flush(); os.fsync(stream.fileno())


def copy_bytes(source, target):
    with Path(target).open('xb') as stream:
        stream.write(Path(source).read_bytes())
        stream.flush(); os.fsync(stream.fileno())


def journal_rows(path):
    if not Path(path).exists():
        return []
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    previous = None
    for index, row in enumerate(rows, 1):
        value = dict(row)
        expected = value.pop('record_sha256', None)
        if (value.get('sequence') != index or value.get('previous_record_sha256') != previous
                or digest(value) != expected):
            raise ValueError('chained_journal_integrity_changed')
        previous = expected
    return rows


def append(path, value):
    prior = journal_rows(path)
    bound = {**value, 'sequence': len(prior) + 1,
             'previous_record_sha256': prior[-1]['record_sha256'] if prior else None}
    record = {**bound, 'record_sha256': digest(bound)}
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(canonical(record) + '\n')
        stream.flush(); os.fsync(stream.fileno())


def identities():
    paths = [Path(__file__), HERE / 'gguf_schema_reader_v5_colab.py', HERE / 'schema_reader_v4.py',
             HERE.parents[1] / 'modules/agent-integration/question_field_selection_v4.py',
             HERE.parents[1] / 'modules/agent-integration/evidence_bundles.py']
    return {str(path.resolve()): sha(path) for path in paths}


def _check_expected_sha(value):
    if type(value) is not str or not re.fullmatch(r'[0-9a-f]{64}', value):
        raise ValueError('externally_verified_complete_receipt_sha256_required')


def _historical_references():
    references = {}
    if sha(OLD_GATE / 'gate_protocol.json') != OLD_PROTOCOL_SHA:
        raise ValueError('exact_old_v4_gate_protocol_required')
    old = read(OLD_GATE / 'gate_protocol.json')
    # Verify the identities already frozen by v4 before taking any new hashes.
    # A later continuation cannot re-legitimize changed older bytes merely by
    # counting their rows and recording the changed file's current checksum.
    for mapping_name in ('historical_reference_sha256', 'code_sha256'):
        expected_files = old.get(mapping_name)
        if type(expected_files) is not dict or not expected_files:
            raise ValueError('old_v4_frozen_reference_catalog_required')
        for name, expected in expected_files.items():
            if sha(Path(name)) != expected:
                raise ValueError('old_v4_recorded_reference_or_code_identity_changed')
            references[str(Path(name))] = expected
    inherited = old.get('inherited_identity_sha256')
    if (type(inherited) is not dict or set(inherited) != {
            'load_protocol.json', 'model_files.json', 'model_load_receipt.json'}):
        raise ValueError('old_v4_frozen_model_identity_catalog_required')
    for name, expected in inherited.items():
        path = OLD_GATE / name
        if sha(path) != expected:
            raise ValueError('old_v4_inherited_model_identity_changed')
        references[str(path)] = expected
    for filename, key in [('generation_cases.json', 'generation_cases_sha256'),
                          ('assessor_only/targets.json', 'targets_sha256')]:
        path = OLD_GATE / filename
        if sha(path) != old[key]:
            raise ValueError('exact_exposed_v4_material_bytes_required')
        references[str(path)] = sha(path)
    for path, expected_count in HISTORICAL_LEDGERS:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        intents = [row for row in rows if row.get('event') == 'generation_intent']
        outcomes = [row for row in rows if row.get('event') == 'generation_outcome']
        if (len(intents) != expected_count or len(outcomes) != expected_count
                or any(row.get('status') != 'published' for row in outcomes)):
            raise RuntimeError('all_referenced_older_attempts_must_be_settled')
        references[str(path)] = sha(path)
    for name in ('gate_protocol.json', 'gate_results.json', 'actual_tensor_preflight.json', 'run_claim.json'):
        references[str(OLD_GATE / name)] = sha(OLD_GATE / name)
    # Old 6/8 is retained as immutable bytes. It is neither reclassified nor used
    # as new-answer gold. The new target values stay unread until all eight bind.
    return references


def _setup_material(expected_download_receipt_sha, expected_python_receipt_sha):
    _check_expected_sha(expected_download_receipt_sha)
    _check_expected_sha(expected_python_receipt_sha)
    download_receipt = MODEL_SETUP / 'download_complete_receipt.json'
    python_receipt = PYTHON_SETUP / 'setup_receipt.json'
    if sha(download_receipt) != expected_download_receipt_sha or sha(python_receipt) != expected_python_receipt_sha:
        raise ValueError('externally_verified_ready_receipt_bytes_changed')
    d, dp = read(download_receipt), read(MODEL_SETUP / 'download_protocol.json')
    p, pp = read(python_receipt), read(PYTHON_SETUP / 'setup_protocol.json')
    if (d.get('stage') != 'both_pinned_shards_downloaded_and_verified'
            or d.get('repo') != native.MODEL_ID or d.get('revision') != MODEL_REVISION
            or d.get('protocol_sha256') != sha(MODEL_SETUP / 'download_protocol.json')
            or dp.get('model') != {'repo': native.MODEL_ID, 'revision': MODEL_REVISION, 'files': list(MODEL_FILES)}
            or d.get('model_loaded') is not False or d.get('new_model_inference_attempts') != 0
            or len(d.get('files', [])) != 2):
        raise ValueError('pinned_official_model_setup_not_ready')
    for actual, expected in zip(d['files'], MODEL_FILES):
        if any(actual.get(key) != expected[key] for key in ('name', 'bytes', 'sha256')):
            raise ValueError('downloaded_split_identity_changed')
        if actual.get('verified_bytes') != expected['bytes'] or actual.get('verified_sha256') != expected['sha256']:
            raise ValueError('both_split_full_verification_receipts_required')
    if (sha(MODEL_METADATA) != MODEL_METADATA_SHA
            or sha(MODEL_SETUP / 'official_model_metadata.json') != MODEL_METADATA_SHA):
        raise ValueError('official_model_metadata_identity_changed')
    if (p.get('stage') != 'isolated_cpu_python_ready_no_model_loaded'
            or p.get('protocol_sha256') != sha(PYTHON_SETUP / 'setup_protocol.json')
            or p.get('llama_cpp_version') != native.PACKAGE_VERSION
            or p.get('diskcache_version') != pp.get('diskcache_version')
            or p.get('model_generations') != 0 or p.get('base_environment_modified') is not False
            or pp.get('target') != str(PYTHON_SETUP / 'site-packages')):
        raise ValueError('isolated_python_setup_not_ready')
    site = PYTHON_SETUP / 'site-packages'
    if not Path(p['llama_cpp_path']).resolve().is_relative_to(site.resolve()):
        raise ValueError('setup_import_must_be_isolated_target')
    sys.path.insert(0, str(site))
    versions = {**BASE_DEPENDENCIES, 'diskcache': p['diskcache_version']}
    for name, version in versions.items():
        if importlib.metadata.version(name) != version:
            raise ValueError('actual_frozen_dependency_version_changed')
    wheels = pp.get('wheels')
    if type(wheels) is not list or len(wheels) != 2:
        raise ValueError('exact_two_isolated_wheels_required')
    cpp = next(item for item in wheels if item['filename'].startswith('llama_cpp_python-'))
    if cpp['size'] != native.WHEEL_BYTES or cpp['sha256'] != native.WHEEL_SHA256:
        raise ValueError('pinned_official_cpu_wheel_identity_changed')
    runtime = {'package_name': 'llama-cpp-python', 'package_version': native.PACKAGE_VERSION,
               'wheel_path': str(PYTHON_SETUP / cpp['filename']), 'wheel_bytes': native.WHEEL_BYTES,
               'wheel_sha256': native.WHEEL_SHA256, 'site_packages': str(site),
               'dependencies': [{'name': name, 'version': version} for name, version in sorted(versions.items())]}
    model = {'model_id': native.MODEL_ID, 'model_revision': MODEL_REVISION,
             'quantization': native.QUANTIZATION, 'license': 'apache-2.0',
             'files': [{'path': item['name'], 'bytes': item['bytes'], 'sha256': item['sha256']} for item in MODEL_FILES]}
    paths = [download_receipt, MODEL_SETUP / 'download_protocol.json', MODEL_SETUP / 'download_journal.jsonl',
             MODEL_SETUP / 'official_model_metadata.json', MODEL_METADATA,
             python_receipt, PYTHON_SETUP / 'setup_protocol.json', PYTHON_SETUP / 'setup_journal.jsonl',
             PYTHON_SETUP / 'diskcache_pypi_metadata.json']
    paths += [PYTHON_SETUP / item['filename'] for item in wheels]
    # Both wheels are frozen, including the isolated diskcache dependency.
    for wheel in wheels:
        path = PYTHON_SETUP / wheel['filename']
        if path.stat().st_size != wheel['size'] or sha(path) != wheel['sha256']:
            raise ValueError('isolated_wheel_bytes_changed')
    return model, runtime, {str(path): sha(path) for path in paths}


def initialize(*, expected_download_receipt_sha, expected_python_receipt_sha):
    hosted()
    started = time.monotonic()
    if ROOT.exists():
        raise FileExistsError('fresh_v5_gate_root_required_no_retry')
    references = _historical_references()
    model, runtime, setup_refs = _setup_material(expected_download_receipt_sha, expected_python_receipt_sha)
    references.update(setup_refs)
    generation = read(OLD_GATE / 'generation_cases.json')
    check_generation_cases(generation)
    # Pre-generation frozen identity checks only; no model is imported or loaded.
    verified_model = native.verify_model_artifacts(MODEL_SETUP, model)
    verified_runtime = native._verify_runtime(runtime)
    ROOT.mkdir(mode=0o700)
    (ROOT / 'assessor_only').mkdir()
    copy_bytes(OLD_GATE / 'generation_cases.json', ROOT / 'generation_cases.json')
    copy_bytes(OLD_GATE / 'assessor_only/targets.json', ROOT / 'assessor_only/targets.json')
    write(ROOT / 'model_manifest.json', model)
    write(ROOT / 'runtime_manifest.json', runtime)
    write(ROOT / 'pre_generation_artifact_verification.json', {
        'model': verified_model, 'runtime': verified_runtime,
        'freeze_preparation_seconds': time.monotonic() - started,
        'model_loads': 0, 'model_generations': 0})
    protocol = {'version': VERSION, 'maximum_new_attempts': CAP, 'planned_generations': CAP,
        'main_generations_authorized_by_this_entry': 0, 'automatic_main_run': False,
        'acceptance': 'all eight EOS-complete whole-JSON schema-valid and exact typed correct; labels after authentication of all eight',
        'material': 'same exposed four invented nonclinical reading fixtures and two evidence views as settled v4; not heldout or memory-effect evidence',
        'new_model_condition': 'official Qwen2.5-7B-Instruct GGUF Q4_K_M; distinct from prior float32 1.5B condition',
        'backend_policy': native.policy(), 'new_paid_provider_calls': 0, 'secret_access': False,
        'automatic_retry': False, 'new_human_reviews': 0,
        'code_sha256': identities(), 'historical_and_setup_reference_sha256': references,
        'generation_cases_sha256': sha(ROOT / 'generation_cases.json'),
        'targets_sha256': sha(ROOT / 'assessor_only/targets.json'),
        'model_manifest_sha256': sha(ROOT / 'model_manifest.json'),
        'runtime_manifest_sha256': sha(ROOT / 'runtime_manifest.json'),
        'artifact_verification_sha256': sha(ROOT / 'pre_generation_artifact_verification.json'),
        'old_v4_gate_protocol_sha256': OLD_PROTOCOL_SHA,
        'no_target_values_in_backend_or_generation_api': True}
    write(ROOT / 'gate_protocol.json', protocol)
    print(canonical({'stage': 'v5_gate_frozen_no_model_load_or_generation',
                     'protocol_sha256': sha(ROOT / 'gate_protocol.json'), 'cap': CAP, 'main_calls': 0}), flush=True)


def check_generation_cases(rows):
    if (type(rows) is not list or len(rows) != CAP or len({row.get('case_id') for row in rows}) != CAP
            or {row.get('family') for row in rows} != {'correction', 'repeat_total', 'outcome', 'unagreed'}):
        raise ValueError('exact_eight_distinct_invented_gate_rows_required')
    for row in rows:
        if (set(row) != {'case_id', 'family', 'view', 'question', 'messages', 'request_sha256', 'max_new_tokens'}
                or row['view'] not in {'source_only', 'source_and_state'}
                or row['case_id'] != row['family'] + '__' + row['view']
                or row['max_new_tokens'] != 64 or row['messages'][-1] != {'role': 'user', 'content': row['question']}
                or digest(row['messages']) != row['request_sha256']
                or len(canonical(row['messages'])) > 6000):
            raise ValueError('exact_frozen_generation_case_request_required')
    return rows


def frozen():
    protocol = read(ROOT / 'gate_protocol.json')
    if (protocol.get('version') != VERSION or protocol.get('maximum_new_attempts') != CAP
            or protocol.get('planned_generations') != CAP or protocol.get('code_sha256') != identities()
            or protocol.get('backend_policy') != native.policy()
            or protocol.get('main_generations_authorized_by_this_entry') != 0
            or protocol.get('automatic_main_run') is not False):
        raise ValueError('frozen_v5_gate_protocol_or_code_changed')
    for path, expected in protocol['historical_and_setup_reference_sha256'].items():
        if sha(path) != expected:
            raise ValueError('frozen_historical_or_setup_reference_changed')
    for name, key in [('generation_cases.json', 'generation_cases_sha256'),
                      ('assessor_only/targets.json', 'targets_sha256'),
                      ('model_manifest.json', 'model_manifest_sha256'),
                      ('runtime_manifest.json', 'runtime_manifest_sha256'),
                      ('pre_generation_artifact_verification.json', 'artifact_verification_sha256')]:
        if sha(ROOT / name) != protocol[key]:
            raise ValueError('frozen_v5_gate_material_changed')
    return protocol, check_generation_cases(read(ROOT / 'generation_cases.json'))


def authenticate(protocol, rows, backend, plans):
    """Authenticate all eight originals before target values are opened."""
    persisted_plans = read(ROOT / 'actual_native_input_preflight.json')
    if (len(plans) != CAP or len(persisted_plans) != CAP
            or persisted_plans != [{'case_id': row['case_id'], 'request_sha256': row['request_sha256'], **plan}
                                  for row, plan in zip(rows, plans)]):
        raise ValueError('all_eight_native_preflight_vectors_must_be_persisted_exactly')
    setup_rows = journal_rows(ROOT / 'model_setup_journal.jsonl')
    setup = read(ROOT / 'model_setup_receipt.json')
    if (len(setup_rows) != 2 or setup_rows[0].get('event') != 'native_model_setup_intent'
            or setup_rows[1].get('event') != 'native_model_setup_completed'
            or setup_rows[1].get('receipt_sha256') != sha(ROOT / 'model_setup_receipt.json')
            or setup.get('model_identity') != backend.identity
            or setup.get('answer_generations_during_setup') != 0):
        raise ValueError('actual_native_setup_identity_must_bind_all_answers')
    records = journal_rows(ROOT / 'generation_ledger.jsonl')
    if len(records) != 2 * CAP:
        raise ValueError('all_eight_published_originals_required_before_labels')
    saved_answers = []
    for index, (row, preflight) in enumerate(zip(rows, plans), 1):
        intent, outcome = records[2 * (index - 1):2 * index]
        expected_name = f'response-{index:02}.json'
        if (intent.get('event') != 'generation_intent' or outcome.get('event') != 'generation_outcome'
                or outcome.get('status') != 'published'
                or intent.get('attempt') != index or outcome.get('attempt') != index
                or intent.get('case_id') != row['case_id'] or outcome.get('case_id') != row['case_id']
                or intent.get('request_sha256') != row['request_sha256']
                or outcome.get('request_sha256') != row['request_sha256']
                or intent.get('protocol_sha256') != sha(ROOT / 'gate_protocol.json')
                or intent.get('input_token_ids_sha256') != preflight['input_token_ids_sha256']
                or intent.get('question_schema_sha256') != preflight['question_schema_sha256']
                or intent.get('preflight_file_sha256') != sha(ROOT / 'actual_native_input_preflight.json')
                or outcome.get('artifact') != expected_name):
            raise ValueError('eight_exact_ordered_request_admissions_required')
        path = ROOT / expected_name
        answer = read(path)
        if sha(path) != outcome['artifact_sha256']:
            raise ValueError('published_native_answer_bytes_changed')
        ids = answer['raw_generated_token_ids']
        if not ids or any(type(token) is not int or token < 0 for token in ids) or len(ids) > 64:
            raise ValueError('actual_bounded_native_output_token_ids_required')
        eos = ids[-1] in backend.eog_ids
        if eos != (answer['finish_reason'] == 'eos') or any(token in backend.eog_ids for token in ids[:-1]):
            raise ValueError('observed_native_eog_binding_changed')
        body = backend.engine.detokenize(ids[:-1] if eos else ids,
            prev_tokens=preflight['input_token_ids'], special=False)
        raw = backend.engine.detokenize(ids, prev_tokens=preflight['input_token_ids'], special=True)
        text, raw_text = body.decode('utf-8', errors='replace'), raw.decode('utf-8', errors='replace')
        utf8_valid = True
        try:
            body.decode('utf-8', errors='strict'); raw.decode('utf-8', errors='strict')
        except UnicodeDecodeError:
            utf8_valid = False
        checked = native.schema.validate_response(text, preflight['question_schema'])
        if not eos or not utf8_valid:
            checked = {'accepted': False,
                'reason': answer['finish_reason'] if utf8_valid else 'invalid_utf8', 'answer': None}
        usage = {'prompt_tokens': preflight['input_tokens'], 'completion_tokens': len(ids),
                 'total_tokens': preflight['input_tokens'] + len(ids)}
        expected_decoding = {'backend_policy_sha256': digest(native.policy()),
            'model_identity_sha256': digest(backend.identity),
            'question_schema_sha256': digest(preflight['question_schema']),
            'gbnf_sha256': hashlib.sha256(native.grammar_text(preflight['question_schema']).encode()).hexdigest(),
            'max_new_tokens': 64, 'native_eog_token_ids': backend.eog_ids}
        if (answer.get('case_id') != row['case_id'] or answer.get('request_sha256') != row['request_sha256']
                or answer['response_text'] != text or answer['raw_generated_text'] != raw_text
                or base64.b64decode(answer['response_bytes_base64'], validate=True) != body
                or base64.b64decode(answer['raw_generated_bytes_base64'], validate=True) != raw
                or answer['response_sha256'] != hashlib.sha256(text.encode()).hexdigest()
                or answer['response_bytes_sha256'] != hashlib.sha256(body).hexdigest()
                or outcome['response_sha256'] != answer['response_sha256']
                or answer['input_token_ids_sha256'] != preflight['input_token_ids_sha256']
                or answer['rendered_prompt_sha256'] != preflight['rendered_prompt_sha256']
                or answer['question_schema_sha256'] != preflight['question_schema_sha256']
                or answer['decoding_identity'] != expected_decoding
                or answer['decoding_identity_sha256'] != digest(expected_decoding)
                or answer['schema_validation'] != checked
                or answer['chosen_answer'] != checked['answer']
                or type(answer['chosen_answer']) is not type(checked['answer'])
                or answer['utf8_valid'] is not utf8_valid
                or answer['usage'] != usage or outcome['usage'] != usage
                or not math.isfinite(answer['answer_path_seconds']) or answer['answer_path_seconds'] < 0
                or outcome['answer_path_seconds'] != answer['answer_path_seconds']):
            raise ValueError('complete_native_answer_authentication_failed')
        saved_answers.append(answer)
    return saved_answers


def score_after_authentication(protocol, rows, answers, setup):
    if len(rows) != CAP or len(answers) != CAP:
        raise ValueError('all_eight_authenticated_answers_required_before_labels')
    authentication = read(ROOT / 'all_eight_authentication_receipt.json')
    if (authentication.get('authenticated_original_answers') != CAP
            or authentication.get('target_values_opened_during_authentication') is not False
            or authentication.get('answers_sha256') != digest(answers)
            or authentication.get('ledger_sha256') != sha(ROOT / 'generation_ledger.jsonl')
            or authentication.get('preflight_sha256') != sha(ROOT / 'actual_native_input_preflight.json')):
        raise ValueError('complete_actual_original_authentication_required_before_labels')
    if sha(ROOT / 'assessor_only/targets.json') != protocol['targets_sha256']:
        raise ValueError('frozen_assessor_bytes_changed')
    target_rows = read(ROOT / 'assessor_only/targets.json')
    targets = {row['case_id']: row['answer'] for row in target_rows}
    if len(target_rows) != CAP or set(targets) != {row['case_id'] for row in rows}:
        raise ValueError('complete_exact_gate_reference_coverage_required')
    details = []
    for row, answer in zip(rows, answers):
        valid = (answer['finish_reason'] == 'eos' and answer['schema_validation']['accepted'])
        observed = answer['schema_validation']['answer']
        target = targets[row['case_id']]
        correct = valid and type(observed) is type(target) and observed == target
        details.append({'case_id': row['case_id'], 'family': row['family'], 'view': row['view'],
            'schema_valid': valid, 'typed_correct': correct, 'expected_answer': target,
            'observed_answer': observed, 'request_sha256': row['request_sha256'],
            'response_sha256': answer['response_sha256']})
    resources = {'unique_admitted_attempts': CAP, 'published_outcomes': CAP,
        'observed_tokens': {key: sum(answer['usage'][key] for answer in answers)
            for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
        'answer_path_seconds_sum': sum(answer['answer_path_seconds'] for answer in answers),
        'native_generation_seconds_sum': sum(answer['native_generation_wall_seconds'] for answer in answers),
        'new_model_setup_seconds': setup['setup_seconds'],
        'freeze_preparation_seconds': read(ROOT / 'pre_generation_artifact_verification.json')['freeze_preparation_seconds'],
        'all_eight_input_preflight_seconds': read(ROOT / 'preflight_timing_receipt.json')['input_preflight_seconds'],
        'model_artifact_verification_seconds': setup.get('model_artifact_verification_seconds'),
        'runtime_artifact_verification_seconds': setup.get('runtime_artifact_verification_seconds'),
        'engine_load_and_identity_seconds': setup.get('engine_load_and_identity_seconds'),
        'setup_timing_components_are_nonadditive_with_total': True,
        'new_paid_provider_calls': 0, 'preparation_indexing_seconds': None,
        'monetary_cost': None,
        'old_study_or_transport_package_setup_cost_not_added': True}
    return {'version': VERSION, 'gate_passed': all(row['typed_correct'] for row in details),
        'schema_valid': sum(row['schema_valid'] for row in details),
        'typed_correct': sum(row['typed_correct'] for row in details), 'planned': CAP,
        'per_question': details, 'resources': resources,
        'material': 'exposed invented model readability development gate, four fixtures in two evidence views',
        'clinical_or_memory_effect_claim': False, 'automatic_main_run': False,
        'main_generations_authorized_by_this_entry': 0,
        'protocol_sha256': sha(ROOT / 'gate_protocol.json')}


def run():
    hosted()
    protocol, rows = frozen()
    write(ROOT / 'run_claim.json', {'pid': os.getpid(), 'started_unix': time.time(),
                                   'automatic_retry': False, 'maximum_new_attempts': CAP})
    runtime = read(ROOT / 'runtime_manifest.json')
    sys.path.insert(0, runtime['site_packages'])
    setup_journal = ROOT / 'model_setup_journal.jsonl'
    append(setup_journal, {'event': 'native_model_setup_intent', 'at_unix': time.time(),
                          'answer_generations_during_setup': 0, 'secret_access': False})
    setup_started = time.monotonic()
    try:
        backend, setup = native.load_backend(MODEL_SETUP, read(ROOT / 'model_manifest.json'), runtime_manifest=runtime)
        write(ROOT / 'model_setup_receipt.json', setup)
        append(setup_journal, {'event': 'native_model_setup_completed',
            'receipt_sha256': sha(ROOT / 'model_setup_receipt.json'),
            'observed_wall_seconds': time.monotonic() - setup_started, 'at_unix': time.time()})
    except BaseException as error:
        append(setup_journal, {'event': 'native_model_setup_failed_or_unknown', 'failure_type': type(error).__name__,
            'observed_wall_seconds': time.monotonic() - setup_started, 'automatic_retry': False, 'at_unix': time.time()})
        raise
    preflight_started = time.monotonic()
    plans = [backend.input_preflight(row['messages']) for row in rows]
    preflight_seconds = time.monotonic() - preflight_started
    write(ROOT / 'actual_native_input_preflight.json', [
        {'case_id': row['case_id'], 'request_sha256': row['request_sha256'], **preflight}
        for row, preflight in zip(rows, plans)])
    write(ROOT / 'preflight_timing_receipt.json', {'input_preflight_seconds': preflight_seconds,
        'logical_rows': CAP, 'scope': 'all eight input vectors, rendering checks and question contracts; no generation'})
    ledger = ROOT / 'generation_ledger.jsonl'
    for index, (row, preflight) in enumerate(zip(rows, plans), 1):
        frozen()  # Checks target bytes but does not open their values.
        if len([entry for entry in journal_rows(ledger) if entry['event'] == 'generation_intent']) != index - 1:
            raise RuntimeError('exact_once_gate_admission_required_no_retry')
        intent = {'event': 'generation_intent', 'attempt': index, 'case_id': row['case_id'],
            'request_sha256': row['request_sha256'], 'protocol_sha256': sha(ROOT / 'gate_protocol.json'),
            'input_token_ids_sha256': preflight['input_token_ids_sha256'],
            'preflight_file_sha256': sha(ROOT / 'actual_native_input_preflight.json'),
            'question_schema_sha256': preflight['question_schema_sha256'], 'at_unix': time.time()}
        append(ledger, intent)
        started = time.monotonic()
        try:
            generated = backend.generate(row['messages'], preflight['question_schema'], max_new_tokens=64)
            answer = {'case_id': row['case_id'], 'request_sha256': row['request_sha256'],
                      'answer_path_seconds': time.monotonic() - started, **generated}
            path = ROOT / f'response-{index:02}.json'
            write(path, answer)
            append(ledger, {'event': 'generation_outcome', 'status': 'published', 'attempt': index,
                'case_id': row['case_id'], 'request_sha256': row['request_sha256'],
                'artifact': path.name, 'artifact_sha256': sha(path),
                'response_sha256': answer['response_sha256'], 'usage': answer['usage'],
                'answer_path_seconds': answer['answer_path_seconds'], 'at_unix': time.time()})
            print(canonical({'stage': 'v5_gate_generation_published', 'published': index, 'cap': CAP}), flush=True)
        except BaseException as error:
            partial = getattr(error, 'generated_record', None)
            partial_ref = None
            if partial is not None:
                partial_path = ROOT / f'failed-partial-{index:02}.json'
                write(partial_path, partial)
                partial_ref = {'artifact': partial_path.name, 'artifact_sha256': sha(partial_path),
                               'usage': partial.get('usage')}
            append(ledger, {'event': 'generation_outcome', 'status': 'failed_or_unknown', 'attempt': index,
                'case_id': row['case_id'], 'request_sha256': row['request_sha256'],
                'failure_type': type(error).__name__, 'observed_partial': partial_ref,
                'observed_wall_seconds': time.monotonic() - started, 'may_have_generated': True,
                'automatic_retry': False, 'at_unix': time.time()})
            raise
    protocol, rows = frozen()
    answers = authenticate(protocol, rows, backend, plans)
    write(ROOT / 'all_eight_authentication_receipt.json', {
        'authenticated_original_answers': len(answers), 'target_values_opened_during_authentication': False,
        'answers_sha256': digest(answers),
        'ledger_sha256': sha(ledger), 'preflight_sha256': sha(ROOT / 'actual_native_input_preflight.json'),
        'at_unix': time.time()})
    report = score_after_authentication(protocol, rows, answers, setup)
    write(ROOT / 'gate_results.json', report)
    print(canonical({'stage': 'v5_readability_gate_complete', **{key: report[key] for key in
        ('gate_passed', 'schema_valid', 'typed_correct', 'planned')}, 'resources': report['resources'],
        'automatic_main_run': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('initialize', 'run'))
    parser.add_argument('--expected-download-receipt-sha')
    parser.add_argument('--expected-python-receipt-sha')
    arguments = parser.parse_args()
    try:
        if arguments.command == 'initialize':
            initialize(expected_download_receipt_sha=arguments.expected_download_receipt_sha,
                       expected_python_receipt_sha=arguments.expected_python_receipt_sha)
        else:
            run()
    except BaseException as error:
        print(canonical({'stage': 'v5_gate_stopped_no_retry', 'failure_type': type(error).__name__,
                         'automatic_retry': False, 'automatic_main_run': False}), flush=True)
        sys.exit(1)
