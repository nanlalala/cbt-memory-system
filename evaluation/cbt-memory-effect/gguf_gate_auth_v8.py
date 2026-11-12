"""Lightweight byte/metadata gate check before creating a new main study.

No model, tokenizer, native package or gate runner is loaded here. Token-to-text
reproduction is explicitly deferred to the frozen runner's authenticate method
after a fresh actual backend load. Three externally observed hashes are required.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gguf_schema_reader_v7_colab as native

VERSION = 'lightweight-completed-t4-eight-gate-binding-v7'
GATE_VERSION = 'official-qwen7b-t4-eight-case-readability-gate-v7'
MODEL_DIR = Path('/content/cbt-qwen7b-gpu-reader-setup-20261112-v7')
MODEL_REVISION = 'bb5d59e06d9551d752d08b292a50eb208b07ab1f'
FILES = [
    {'path': 'qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf', 'bytes': 3993201344,
     'sha256': 'dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db'},
    {'path': 'qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf', 'bytes': 689872288,
     'sha256': '539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a'}]


def digest(value):
    return native.digest(value)


def sha(path):
    return native.file_sha(path)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate_json_key')
        result[key] = value
    return result


def read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('regular_original_gate_file_required')
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite_json')))


def _expected(value):
    if type(value) is not str or not re.fullmatch(r'[0-9a-f]{64}', value):
        raise ValueError('externally_observed_complete_sha256_required')


def journal(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('regular_original_journal_required')
    records = [json.loads(line, object_pairs_hook=_unique) for line in path.read_text().splitlines()]
    previous = None
    for index, record in enumerate(records, 1):
        payload = dict(record)
        expected = payload.pop('record_sha256', None)
        if (payload.get('sequence') != index or payload.get('previous_record_sha256') != previous
                or digest(payload) != expected):
            raise ValueError('native_gate_chained_journal_changed')
        previous = expected
    return records


def _duration(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _identity(model, runtime, identity):
    if (model != {'model_id': native.MODEL_ID, 'model_revision': MODEL_REVISION,
            'quantization': native.QUANTIZATION, 'license': 'apache-2.0', 'files': FILES}
            or runtime.get('package_name') != 'llama-cpp-python'
            or runtime.get('package_version') != native.PACKAGE_VERSION
            or runtime.get('wheel_bytes') != native.WHEEL_BYTES
            or runtime.get('wheel_sha256') != native.WHEEL_SHA256):
        raise ValueError('fixed_official_7b_q4km_model_and_runtime_required')
    eog = identity.get('native_eog_token_ids')
    if (identity.get('model_id') != native.MODEL_ID or identity.get('model_revision') != MODEL_REVISION
            or identity.get('quantization') != native.QUANTIZATION
            or identity.get('license') != 'apache-2.0' or identity.get('files') != FILES
            or identity.get('files_verified') != 2
            or identity.get('bytes_verified') != sum(f['bytes'] for f in FILES)
            or identity.get('manifest_sha256') != digest(model)
            or identity.get('backend_policy_sha256') != digest(native.policy())
            or identity.get('control_token_ids') != native.CONTROL_IDS
            or type(eog) is not list or len(eog) != len(set(eog))
            or any(type(i) is not int or i < 0 for i in eog)
            or 151645 not in eog or not set(eog).issubset({151643, 151645})):
        raise ValueError('actual_gate_setup_model_identity_changed')
    actual_runtime = identity.get('runtime_identity', {})
    if (actual_runtime.get('runtime_manifest_sha256') != digest(runtime)
            or actual_runtime.get('package_version') != native.PACKAGE_VERSION
            or actual_runtime.get('site_packages') != runtime.get('site_packages')
            or actual_runtime.get('wheel_bytes') != native.WHEEL_BYTES
            or actual_runtime.get('wheel_sha256') != native.WHEEL_SHA256
            or actual_runtime.get('dependencies') != runtime.get('dependencies')):
        raise ValueError('actual_gate_setup_runtime_identity_changed')
    return eog


def verify_gate(gate_root, *, expected_protocol_sha256, expected_report_sha256,
                expected_authentication_sha256, fixture_only=False):
    if not fixture_only:
        native.schema.require_hosted_colab()
    root = Path(gate_root).resolve()
    if not fixture_only and (not root.is_relative_to(Path('/content')) or root == Path('/content')):
        raise ValueError('owned_hosted_gate_root_required')
    for expected in (expected_protocol_sha256, expected_report_sha256, expected_authentication_sha256):
        _expected(expected)
    external = {'gate_protocol.json': expected_protocol_sha256,
                'gate_results.json': expected_report_sha256,
                'all_eight_authentication_receipt.json': expected_authentication_sha256}
    for name, expected in external.items():
        if sha(root / name) != expected:
            raise ValueError('externally_observed_gate_artifact_changed')
    protocol = read(root / 'gate_protocol.json')
    if (protocol.get('version') != GATE_VERSION or protocol.get('maximum_new_attempts') != 8
            or protocol.get('planned_generations') != 8
            or protocol.get('main_generations_authorized_by_this_entry') != 0
            or protocol.get('automatic_main_run') is not False
            or protocol.get('backend_policy') != native.policy()
            or protocol.get('new_paid_provider_calls') != 0 or protocol.get('secret_access') is not False):
        raise ValueError('frozen_native_eight_gate_contract_required')
    refs = {}
    for mapping in ('code_sha256', 'historical_and_setup_reference_sha256'):
        values = protocol.get(mapping)
        if type(values) is not dict or not values:
            raise ValueError('nonempty_frozen_gate_reference_catalog_required')
        for name, expected in values.items():
            _expected(expected)
            if sha(name) != expected:
                raise ValueError('frozen_gate_code_or_history_changed')
            refs[str(Path(name).resolve())] = expected
    code = protocol['code_sha256']
    runners = [p for p in code if Path(p).name == 'run_gguf_reader_gate_v7_colab.py']
    if len(runners) != 1:
        raise ValueError('one_frozen_native_gate_runner_required')
    for actual in (native.__file__, native.schema.__file__):
        matching = [v for p, v in code.items() if Path(p).name == Path(actual).name]
        if matching != [sha(actual)]:
            raise ValueError('current_shared_native_or_schema_code_differs_from_gate')
    for name, key in [('generation_cases.json', 'generation_cases_sha256'),
                      ('assessor_only/targets.json', 'targets_sha256'),
                      ('model_manifest.json', 'model_manifest_sha256'),
                      ('runtime_manifest.json', 'runtime_manifest_sha256'),
                      ('pre_generation_artifact_verification.json', 'artifact_verification_sha256')]:
        if sha(root / name) != protocol[key]:
            raise ValueError('frozen_gate_material_changed')
    rows = read(root / 'generation_cases.json')
    if (type(rows) is not list or len(rows) != 8
            or {r['family'] for r in rows} != {'correction', 'repeat_total', 'outcome', 'unagreed'}
            or len({r['case_id'] for r in rows}) != 8):
        raise ValueError('full_eight_distinct_invented_gate_cases_required')
    for row in rows:
        if (row['view'] not in ('source_only', 'source_and_state')
                or row['case_id'] != row['family'] + '__' + row['view']
                or row['messages'][-1] != {'role': 'user', 'content': row['question']}
                or digest(row['messages']) != row['request_sha256'] or row['max_new_tokens'] != 64):
            raise ValueError('frozen_public_question_request_binding_changed')
    setup = read(root / 'model_setup_receipt.json')
    model, runtime = read(root / 'model_manifest.json'), read(root / 'runtime_manifest.json')
    identity = setup.get('model_identity', {})
    eog = _identity(model, runtime, identity)
    proof = setup.get('gpu_execution_proof')
    stable_gpu = identity.get('gpu_execution_condition', {})
    if type(stable_gpu) is not dict or not stable_gpu:
        raise ValueError('stable_actual_t4_condition_and_offload_proof_disagree')
    native.verify_gpu_execution_proof(proof, expected_condition=stable_gpu)
    constructor_log_path = Path(proof['constructor_log']['path']).resolve()
    refs[str(constructor_log_path)] = sha(constructor_log_path)
    setup_records = journal(root / 'model_setup_journal.jsonl')
    if (len(setup_records) != 2 or setup_records[0].get('event') != 'native_model_setup_intent'
            or setup_records[1].get('event') != 'native_model_setup_completed'
            or setup_records[1].get('receipt_sha256') != sha(root / 'model_setup_receipt.json')
            or setup.get('answer_generations_during_setup') != 0 or setup.get('secret_access') is not False
            or setup.get('new_provider_calls') != 0 or not _duration(setup.get('setup_seconds'))):
        raise ValueError('complete_native_model_setup_journal_required')
    plans = read(root / 'actual_native_input_preflight.json')
    records = journal(root / 'generation_ledger.jsonl')
    if len(plans) != 8 or len(records) != 16:
        raise ValueError('eight_complete_native_preflights_and_published_attempts_required')
    answers = []
    for index, (row, pre) in enumerate(zip(rows, plans), 1):
        contract = native.schema.schema_for_question(row['question'])
        vector = pre['input_token_ids']
        rendered = ''.join(text for text, _ in native.chatml_segments(row['messages'])).encode()
        if (pre.get('case_id') != row['case_id'] or pre.get('request_sha256') != row['request_sha256']
                or type(vector) is not list or not vector or len(vector) > 4096
                or any(type(t) is not int or t < 0 for t in vector)
                or pre['input_tokens'] != len(vector) or pre['input_token_ids_sha256'] != digest(vector)
                or pre['rendered_prompt_sha256'] != hashlib.sha256(rendered).hexdigest()
                or pre['rendered_prompt_utf8_bytes'] != len(rendered)
                or pre['rendering_policy_sha256'] != digest(native.policy())
                or pre['question_schema'] != contract or pre['question_schema_sha256'] != digest(contract)
                or pre['native_context_capacity'] != native.CONTEXT_TOKENS
                or pre['prefilled_token_count'] != 0):
            raise ValueError('saved_native_input_vector_or_question_binding_changed')
        intent, outcome = records[2 * (index - 1):2 * index]
        name = f'response-{index:02}.json'
        if (intent.get('event') != 'generation_intent' or outcome.get('event') != 'generation_outcome'
                or outcome.get('status') != 'published' or intent.get('attempt') != index
                or outcome.get('attempt') != index
                or any(x.get('case_id') != row['case_id'] or x.get('request_sha256') != row['request_sha256']
                       for x in (intent, outcome))
                or intent.get('protocol_sha256') != expected_protocol_sha256
                or intent.get('input_token_ids_sha256') != pre['input_token_ids_sha256']
                or intent.get('question_schema_sha256') != pre['question_schema_sha256']
                or intent.get('preflight_file_sha256') != sha(root / 'actual_native_input_preflight.json')
                or outcome.get('artifact') != name or sha(root / name) != outcome.get('artifact_sha256')):
            raise ValueError('ordered_eight_native_admissions_or_published_bytes_changed')
        answer = read(root / name)
        ids = answer['raw_generated_token_ids']
        if (type(ids) is not list or not ids or len(ids) > 64
                or any(type(t) is not int or t < 0 for t in ids)
                or ids[-1] not in eog or any(t in eog for t in ids[:-1])
                or answer.get('finish_reason') != 'eos' or answer.get('utf8_valid') is not True):
            raise ValueError('complete_observed_native_eog_required')
        body = base64.b64decode(answer['response_bytes_base64'], validate=True)
        raw = base64.b64decode(answer['raw_generated_bytes_base64'], validate=True)
        text, raw_text = body.decode('utf-8'), raw.decode('utf-8')
        checked = native.schema.validate_response(text, contract)
        usage = {'prompt_tokens': len(vector), 'completion_tokens': len(ids), 'total_tokens': len(vector) + len(ids)}
        decoding = {'backend_policy_sha256': digest(native.policy()), 'model_identity_sha256': digest(identity),
            'question_schema_sha256': digest(contract),
            'gbnf_sha256': hashlib.sha256(native.grammar_text(contract).encode()).hexdigest(),
            'max_new_tokens': 64, 'native_eog_token_ids': eog}
        if (answer.get('case_id') != row['case_id'] or answer.get('request_sha256') != row['request_sha256']
                or answer['response_text'] != text or answer['raw_generated_text'] != raw_text
                or answer['response_sha256'] != hashlib.sha256(body).hexdigest()
                or answer['response_bytes_sha256'] != hashlib.sha256(body).hexdigest()
                or outcome['response_sha256'] != answer['response_sha256']
                or answer['input_token_ids_sha256'] != pre['input_token_ids_sha256']
                or answer['rendered_prompt_sha256'] != pre['rendered_prompt_sha256']
                or answer['question_schema_sha256'] != digest(contract)
                or answer['decoding_identity'] != decoding or answer['decoding_identity_sha256'] != digest(decoding)
                or answer['schema_validation'] != checked
                or answer['chosen_answer'] != checked['answer']
                or type(answer['chosen_answer']) is not type(checked['answer'])
                or answer['usage'] != usage or outcome['usage'] != usage
                or not _duration(answer['answer_path_seconds'])
                or outcome['answer_path_seconds'] != answer['answer_path_seconds']):
            raise ValueError('native_answer_bytes_usage_schema_or_identity_changed')
        answers.append(answer)
    # The externally hashed receipt records a prior actual native reproduction.
    authentication = read(root / 'all_eight_authentication_receipt.json')
    if (authentication.get('authenticated_original_answers') != 8
            or authentication.get('target_values_opened_during_authentication') is not False
            or authentication.get('answers_sha256') != digest(answers)
            or authentication.get('ledger_sha256') != sha(root / 'generation_ledger.jsonl')
            or authentication.get('preflight_sha256') != sha(root / 'actual_native_input_preflight.json')):
        raise ValueError('externally_observed_full_native_authentication_receipt_changed')
    # Target values are parsed only after every published answer authenticates.
    targets_list = read(root / 'assessor_only/targets.json')
    targets = {r['case_id']: r['answer'] for r in targets_list}
    if len(targets_list) != 8 or len(targets) != 8 or set(targets) != {r['case_id'] for r in rows}:
        raise ValueError('complete_eight_target_coverage_required')
    details = []
    for row, answer in zip(rows, answers):
        observed, target = answer['schema_validation']['answer'], targets[row['case_id']]
        valid = answer['schema_validation']['accepted'] and answer['finish_reason'] == 'eos'
        correct = valid and type(observed) is type(target) and observed == target
        details.append({'case_id': row['case_id'], 'family': row['family'], 'view': row['view'],
            'schema_valid': valid, 'typed_correct': correct, 'expected_answer': target,
            'observed_answer': observed, 'request_sha256': row['request_sha256'],
            'response_sha256': answer['response_sha256']})
    report = read(root / 'gate_results.json')
    if (not all(r['typed_correct'] for r in details) or report.get('gate_passed') is not True
            or report.get('schema_valid') != 8 or report.get('typed_correct') != 8 or report.get('planned') != 8
            or report.get('per_question') != details or report.get('protocol_sha256') != expected_protocol_sha256
            or report.get('automatic_main_run') is not False
            or report.get('main_generations_authorized_by_this_entry') != 0):
        raise RuntimeError('eight_of_eight_exact_typed_native_gate_required_before_main')
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('gate_checkpoint_symlinks_prohibited')
        if path.is_file():
            refs[str(path.resolve())] = sha(path)
    return {'version': VERSION, 'gate_root': str(root), 'typed_correct': 8, 'schema_valid': 8,
        'gate_protocol_sha256': expected_protocol_sha256, 'gate_result_sha256': expected_report_sha256,
        'gate_authentication_sha256': expected_authentication_sha256,
        'gate_artifact_sha256': refs, 'reference_sha256': refs,
        'model_manifest': model, 'runtime_manifest': runtime,
        'expected_native_model_identity': identity, 'model_dir': str(MODEL_DIR),
        'frozen_gate_runner_path': runners[0], 'gate_runner_path': runners[0],
        'native_authentication_method': 'authenticate',
        'scope': 'byte/request/vector/schema bindings and prior externally hashed native-auth receipt; token-to-text not rerun here',
        'fresh_actual_backend_must_repeat_frozen_runner_authenticate_before_main_admission': True,
        'new_model_loads': 0, 'new_model_calls': 0, 'new_target_or_clinical_gold': False}
