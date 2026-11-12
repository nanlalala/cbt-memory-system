"""Metadata-and-byte closure of a completed V6 CPU gate, never a model runner.

Retains original /content-relative paths and every transitive checksum-map
reference. Excluded model/runtime binaries are explicitly described, not
silently dropped or treated as a complete runnable restore. Standard library
only: no imports of experimental code, packages, secrets or native backends.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import time
import zipfile

VERSION = 'completed-cpu-v6-byte-reference-closure-v1'
CONTENT = Path('/content')
GATE = CONTENT / 'cbt-qwen7b-reader-gate-20261112-v6'
V5 = CONTENT / 'cbt-qwen7b-reader-gate-20261112-v5'
V4 = CONTENT / 'cbt-question-schema-reader-gate-20261112-v4'
DIAGNOSTIC = CONTENT / 'cbt-qwen7b-load-diagnostic-20261112-v1'
MODEL_SETUP = CONTENT / 'cbt-qwen7b-reader-setup-20261112-v1'
PYTHON_SETUP = CONTENT / 'cbt-qwen7b-python-20261112-v1'
PUBLIC_METADATA = CONTENT / 'cbt-source-and-reader-metadata-20261112-v1'
ARCHIVE = CONTENT / 'CBT_CPU_V6_Closure_20261112_PRIVATE.zip'
RECEIPT = CONTENT / 'CBT_CPU_V6_Closure_20261112_PRIVATE.receipt.json'
WHEEL_NAME = 'llama_cpp_python-0.3.36-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl'
WHEEL_PATH = PYTHON_SETUP / WHEEL_NAME
WHEEL_BYTES = 27077498
WHEEL_SHA = 'c8a3e98093e950d7a770f794842e36f2db82b2933e8762ed7b5ffd435011e8e4'
WHEEL_URL = 'https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.36/' + WHEEL_NAME
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 100 * 1024 * 1024
MAX_FILES = 4000
ALLOWED_SUFFIXES = frozenset({'.json', '.jsonl', '.py', '.log', '.txt', '.md', '.csv',
                              '.html', '.sqlite', '.db', '.gz', '.whl'})
BANNED_PARTS = frozenset({'site-packages', '.git', '.ssh', '.config', '.cache', '__pycache__',
                         'credentials', 'secrets'})
BANNED_NAMES = frozenset({'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt',
                         'special_tokens_map.json', 'credentials.json', '.env'})
CODE_NAMES = frozenset({
    'gguf_schema_reader_v5_colab.py', 'gguf_schema_reader_v6_colab.py',
    'run_gguf_reader_gate_v5_colab.py', 'run_gguf_reader_gate_v6_colab.py',
    'gguf_gate_auth_v5.py', 'gguf_gate_auth_v6.py',
    'gguf_state_runtime_v5.py', 'gguf_state_runtime_v6.py',
    'gguf_state_comparison_v5_colab.py', 'gguf_state_comparison_v6_colab.py',
    'schema_reader_v4.py', 'question_retrieval_policy_v4.py',
    'schema_state_runtime_v4.py', 'schema_state_comparison_v4_colab.py',
    'run_schema_reader_gate_v4_colab.py', 'setup_gguf_cpu_python_v5_colab.py',
    'download_qwen7b_gguf_colab.py', 'diagnose_gguf_load_v5_colab.py'})


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def hosted():
    try:
        okay = (bool(os.environ.get('COLAB_RELEASE_TAG')) and CONTENT.is_dir()
                and importlib.util.find_spec('google.colab') is not None)
    except (ImportError, ValueError, AttributeError):
        okay = False
    if not okay:
        raise RuntimeError('genuine_hosted_colab_required_before_artifact_io')


def owned(path):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts or path == CONTENT or not path.is_relative_to(CONTENT):
        raise ValueError('only_original_absolute_content_paths_allowed')
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError('symlinked_checkpoint_path_prohibited')
        if parent == CONTENT:
            break
    return path


def excluded_kind(path):
    path = Path(path)
    if path == WHEEL_PATH:
        return 'required_official_runtime_wheel_omitted_reacquire_before_restore'
    if any(part in BANNED_PARTS for part in path.parts) or path.name in BANNED_NAMES:
        return 'excluded_runtime_installation_credentials_or_tokenizer'
    if path.suffix.lower() in {'.gguf', '.partial', '.bin', '.safetensors', '.pt', '.pth'}:
        return 'excluded_model_weights_or_transport_probe'
    return None


def safe_member(path):
    path = owned(path)
    if excluded_kind(path) or path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError('file_outside_metadata_code_and_checkpoint_whitelist')
    return 'content/' + path.relative_to(CONTENT).as_posix()


def read(path):
    path = owned(path)
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError('bounded_json_metadata_required')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate_json_key')
            result[key] = value
        return result
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite_json')))


def journal(path):
    rows = [json.loads(line) for line in owned(path).read_text().splitlines()]
    previous = None
    for index, row in enumerate(rows, 1):
        payload = dict(row)
        expected = payload.pop('record_sha256', None)
        if (payload.get('sequence') != index or payload.get('previous_record_sha256') != previous
                or digest(payload) != expected):
            raise ValueError('completed_gate_chained_journal_changed')
        previous = expected
    return rows


def require_hash(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('externally_observed_sha256_required')


def validate_gate(expected_protocol, expected_report, expected_auth):
    for name, expected in [('gate_protocol.json', expected_protocol),
                           ('gate_results.json', expected_report),
                           ('all_eight_authentication_receipt.json', expected_auth)]:
        require_hash(expected)
        if sha(owned(GATE / name)) != expected:
            raise ValueError('externally_observed_completed_gate_bytes_changed')
    protocol = read(GATE / 'gate_protocol.json')
    if (protocol.get('version') != 'official-qwen7b-gguf-eight-case-readability-gate-v6-context4352'
            or protocol.get('maximum_new_attempts') != 8 or protocol.get('planned_generations') != 8
            or protocol.get('automatic_main_run') is not False
            or protocol.get('main_generations_authorized_by_this_entry') != 0):
        raise ValueError('exact_eight_case_v6_gate_required')
    rows = read(GATE / 'generation_cases.json')
    plans = read(GATE / 'actual_native_input_preflight.json')
    ledger = journal(GATE / 'generation_ledger.jsonl')
    if (len(rows) != 8 or len(plans) != 8 or len(ledger) != 16
            or len({row['case_id'] for row in rows}) != 8):
        raise ValueError('full_eight_original_answers_and_preflights_required')
    for name, key in [('generation_cases.json', 'generation_cases_sha256'),
                      ('assessor_only/targets.json', 'targets_sha256'),
                      ('model_manifest.json', 'model_manifest_sha256'),
                      ('runtime_manifest.json', 'runtime_manifest_sha256'),
                      ('pre_generation_artifact_verification.json', 'artifact_verification_sha256')]:
        if sha(GATE / name) != protocol[key]:
            raise ValueError('frozen_gate_material_bytes_changed')
    answers = []
    for index, (row, pre) in enumerate(zip(rows, plans), 1):
        intent, outcome = ledger[2 * index - 2:2 * index]
        name = f'response-{index:02}.json'
        if (intent.get('event') != 'generation_intent' or outcome.get('event') != 'generation_outcome'
                or outcome.get('status') != 'published' or intent.get('attempt') != index
                or outcome.get('attempt') != index or outcome.get('artifact') != name
                or intent.get('case_id') != row['case_id'] or outcome.get('case_id') != row['case_id']
                or intent.get('request_sha256') != row['request_sha256']
                or outcome.get('request_sha256') != row['request_sha256']
                or digest(row['messages']) != row['request_sha256']
                or intent.get('protocol_sha256') != expected_protocol
                or sha(GATE / name) != outcome.get('artifact_sha256')):
            raise ValueError('all_eight_published_request_and_artifact_bindings_required')
        answer = read(GATE / name)
        body = base64.b64decode(answer['response_bytes_base64'], validate=True)
        raw = base64.b64decode(answer['raw_generated_bytes_base64'], validate=True)
        ids = answer['raw_generated_token_ids']
        vector = pre['input_token_ids']
        usage = {'prompt_tokens': len(vector), 'completion_tokens': len(ids),
                 'total_tokens': len(vector) + len(ids)}
        if (answer.get('case_id') != row['case_id'] or answer.get('request_sha256') != row['request_sha256']
                or answer.get('finish_reason') != 'eos' or answer.get('utf8_valid') is not True
                or not ids or len(ids) > 64 or any(type(x) is not int or x < 0 for x in ids)
                or not vector or len(vector) > 4096
                or pre['input_token_ids_sha256'] != digest(vector)
                or answer['input_token_ids_sha256'] != digest(vector)
                or intent['input_token_ids_sha256'] != digest(vector)
                or intent['preflight_file_sha256'] != sha(GATE / 'actual_native_input_preflight.json')
                or answer['response_text'] != body.decode('utf-8')
                or answer['raw_generated_text'] != raw.decode('utf-8')
                or answer['response_sha256'] != hashlib.sha256(body).hexdigest()
                or answer['response_bytes_sha256'] != hashlib.sha256(body).hexdigest()
                or outcome['response_sha256'] != answer['response_sha256']
                or answer['usage'] != usage or outcome['usage'] != usage
                or answer['schema_validation'].get('accepted') is not True):
            raise ValueError('all_eight_original_native_bytes_and_usage_required')
        obj = json.loads(body)
        if type(obj) is not dict or set(obj) != {'answer'} or type(obj['answer']) not in (int, str, type(None)):
            raise ValueError('whole_single_answer_object_required')
        if (obj['answer'] != answer['chosen_answer']
                or type(obj['answer']) is not type(answer['chosen_answer'])):
            raise ValueError('original_typed_answer_binding_changed')
        answers.append(answer)
    authentication = read(GATE / 'all_eight_authentication_receipt.json')
    if (authentication.get('authenticated_original_answers') != 8
            or authentication.get('target_values_opened_during_authentication') is not False
            or authentication.get('answers_sha256') != digest(answers)
            or authentication.get('ledger_sha256') != sha(GATE / 'generation_ledger.jsonl')
            or authentication.get('preflight_sha256') != sha(GATE / 'actual_native_input_preflight.json')):
        raise ValueError('prior_complete_native_authentication_receipt_required')
    targets = read(GATE / 'assessor_only/targets.json')
    if len(targets) != 8 or len({r['case_id'] for r in targets}) != 8:
        raise ValueError('eight_frozen_target_values_required')
    gold = {row['case_id']: row['answer'] for row in targets}
    if any(a['chosen_answer'] != gold[a['case_id']]
           or type(a['chosen_answer']) is not type(gold[a['case_id']]) for a in answers):
        raise ValueError('all_eight_exact_typed_answers_must_pass')
    report = read(GATE / 'gate_results.json')
    if (report.get('gate_passed') is not True or report.get('typed_correct') != 8
            or report.get('schema_valid') != 8 or report.get('planned') != 8
            or report.get('protocol_sha256') != expected_protocol
            or report.get('automatic_main_run') is not False):
        raise ValueError('completed_eight_of_eight_v6_report_required')
    if (V5 / 'generation_ledger.jsonl').exists():
        raise ValueError('failed_v5_zero_admissions_must_remain_unchanged')
    return protocol, {'published_attempts': 8, 'typed_correct': 8, 'schema_valid': 8,
        'gate_protocol_sha256': expected_protocol, 'gate_report_sha256': expected_report,
        'prior_native_authentication_sha256': expected_auth,
        'native_detokenization_repeated_by_exporter': False,
        'scope': 'byte/journal/request/token-count/typed-answer checks plus retained prior native authentication receipt'}


def _product_root_from_origin(origin):
    text = str(origin)
    sides = [side for side in ('before', 'after')
             if ('-' + side + '-') in text or ('/' + side + '/') in text]
    if len(sides) == 1:
        return CONTENT / ('cbt-open-state-' + sides[0] + '-20261112-v2')
    return None


def hash_references(value, origin, *, product_root=None, source_prepared=None):
    """Yield only explicit path-to-SHA maps, never arbitrary model manifest paths."""
    if type(value) is dict:
        product_root = product_root or _product_root_from_origin(origin)
        if type(value.get('checkout')) is str:
            product_root = owned(value['checkout'])
        if type(value.get('source_prepared')) is str:
            source_prepared = owned(value['source_prepared'])
        for key, child in value.items():
            if type(child) is dict and (key.endswith('_sha256') or key in {'code_hashes', 'reference_hashes', 'source_files'}):
                for filename, expected in child.items():
                    if type(expected) is str and re.fullmatch('[0-9a-f]{64}', expected):
                        if type(filename) is not str:
                            raise ValueError('reference_path_must_be_text')
                        candidate = Path(filename)
                        if not candidate.is_absolute():
                            if key == 'source_files':
                                if product_root is None:
                                    raise ValueError('relative_product_source_map_requires_exact_checkout_identity')
                                base = product_root
                            elif key == 'source_files_sha256':
                                if source_prepared is None:
                                    raise ValueError('relative_source_snapshot_requires_exact_preparation_root')
                                base = source_prepared
                            else:
                                base = origin.parent
                            candidate = base / candidate
                        yield owned(candidate), expected, key
            yield from hash_references(child, origin, product_root=product_root,
                                       source_prepared=source_prepared)
    elif type(value) is list:
        for child in value:
            yield from hash_references(child, origin, product_root=product_root,
                                       source_prepared=source_prepared)


def collect(protocol):
    included, excluded, queue = {}, {}, []

    def add(path, *, expected=None, origin='explicit_checkpoint_seed'):
        path = owned(path)
        kind = excluded_kind(path)
        if kind:
            if expected is not None and path != WHEEL_PATH:
                raise ValueError('referenced_artifact_has_unhandled_exclusion_' + path.name)
            if path == WHEEL_PATH:
                actual = sha(path)
                if path.stat().st_size != WHEEL_BYTES or actual != WHEEL_SHA or expected not in (None, actual):
                    raise ValueError('excluded_required_cpu_wheel_identity_changed')
                excluded[str(path)] = {'source_path': str(path), 'bytes': WHEEL_BYTES, 'sha256': actual,
                    'reason': kind, 'required_for_runtime_restore': True,
                    'reacquire_original_url': WHEEL_URL, 'download_or_install_performed_by_exporter': False}
            return
        member = safe_member(path)
        if not path.is_file():
            raise FileNotFoundError('required_checkpoint_reference_missing_' + str(path))
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            raise ValueError('unexpected_large_nonexcluded_artifact_' + path.name)
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError('transitive_frozen_reference_changed_' + path.name)
        if str(path) in included:
            if included[str(path)]['sha256'] != actual:
                raise ValueError('artifact_changed_during_closure_walk')
            included[str(path)]['origins'].add(origin)
            return
        included[str(path)] = {'source_path': str(path), 'archive_path': member,
            'bytes': size, 'sha256': actual, 'mtime_ns': path.stat().st_mtime_ns, 'origins': {origin}}
        if len(included) > MAX_FILES or sum(v['bytes'] for v in included.values()) > MAX_TOTAL_BYTES:
            raise ValueError('bounded_metadata_closure_limit_exceeded')
        if path.suffix == '.json':
            queue.append(path)

    def seed_tree(root):
        owned(root)
        if not root.is_dir():
            raise FileNotFoundError('required_owned_checkpoint_directory_missing_' + root.name)
        for path in sorted(root.rglob('*')):
            owned(path)
            if path.is_dir():
                continue
            if excluded_kind(path):
                if path == WHEEL_PATH:
                    add(path)
                continue
            add(path)

    for root in (GATE, V5, V4, DIAGNOSTIC, MODEL_SETUP, PYTHON_SETUP, PUBLIC_METADATA):
        # Do not recursively traverse an installed runtime tree, even to list it.
        if root == PYTHON_SETUP:
            for path in sorted(root.iterdir()):
                owned(path)
                if path.is_file():
                    add(path)
                elif path.name != 'site-packages':
                    raise ValueError('unexpected_python_setup_subdirectory')
        else:
            seed_tree(root)
    for side in ('before', 'after'):
        root = CONTENT / ('cbt-open-state-prepared-' + side + '-20261112-v2')
        for name in ('protocol.json', 'requests.jsonl'):
            add(root / name, origin='required_legacy_source_pair')
    # Preserve relevant code adjacent to the exact frozen runner, plus known
    # standalone setup/diagnostic files. This never searches user directories.
    runner_paths = [Path(p) for p in protocol['code_sha256']
                    if Path(p).name == 'run_gguf_reader_gate_v6_colab.py']
    if len(runner_paths) != 1:
        raise ValueError('one_frozen_v6_runner_path_required')
    for directory in (runner_paths[0].parent, CONTENT):
        for name in CODE_NAMES:
            path = directory / name
            if path.is_file():
                add(path, origin='known_selected_reader_or_setup_code')
    for path in sorted(CONTENT.glob('cbt-qwen7b-*20261112*.log')):
        add(path, origin='known_qwen7b_owned_top_level_log')
    add(Path(__file__).resolve(), origin='this_read_only_exporter')
    while queue:
        origin = queue.pop(0)
        for path, expected, key in hash_references(read(origin), origin):
            add(path, expected=expected, origin=str(origin) + '#' + key)
    if str(WHEEL_PATH) not in excluded:
        raise ValueError('required_omitted_cpu_wheel_descriptor_missing')
    return included, list(excluded.values())


def export(expected_protocol, expected_report, expected_auth):
    hosted()
    protocol, validation = validate_gate(expected_protocol, expected_report, expected_auth)
    if ARCHIVE.exists() or RECEIPT.exists() or ARCHIVE.is_symlink() or RECEIPT.is_symlink():
        raise FileExistsError('fresh_cpu_closure_archive_and_receipt_required')
    included, excluded = collect(protocol)
    snapshots = {name: (item['bytes'], item['mtime_ns'], item['sha256']) for name, item in included.items()}
    manifest = {'version': VERSION, 'created_unix': time.time(), 'validation': validation,
        'files': [{**item, 'origins': sorted(item['origins'])} for _, item in sorted(included.items())],
        'excluded_required_artifacts': excluded,
        'excluded_model_storage': {'original_root': str(MODEL_SETUP),
            'not_archived': ['*.gguf', '*.partial', '*.bin'],
            'model_download_receipts_and_original_frozen_digests_preserved': True,
            'old_1_5b_weights_and_tokenizer_files_archived': False},
        'excluded_runtime_installation': str(PYTHON_SETUP / 'site-packages'),
        'complete_direct_runtime_restore': False,
        'restore_note': 'restore included original absolute paths; reacquire and verify declared runtime/model dependencies before any load; never reset control identities',
        'new_model_loads': 0, 'new_model_calls': 0, 'secret_access': False,
        'frozen_sources_or_protocols_modified': False}
    temporary = ARCHIVE.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, item in sorted(included.items()):
            archive.write(name, item['archive_path'])
        archive.writestr('MANIFEST.json', canonical(manifest) + '\n')
    # All terminal gate bytes and every historical reference must stay stable
    # throughout capture. An active writer prevents publication, never a retry.
    for name, expected in snapshots.items():
        path = Path(name)
        if (path.stat().st_size, path.stat().st_mtime_ns, sha(path)) != expected:
            raise ValueError('source_artifact_changed_during_checkpoint_capture')
    validate_gate(expected_protocol, expected_report, expected_auth)
    with zipfile.ZipFile(temporary) as archive:
        for item in manifest['files']:
            with archive.open(item['archive_path']) as stream:
                h, count = hashlib.sha256(), 0
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block); count += len(block)
            if count != item['bytes'] or h.hexdigest() != item['sha256']:
                raise ValueError('archived_member_bytes_failed_full_verification')
    os.link(temporary, ARCHIVE, follow_symlinks=False)
    temporary.unlink()
    receipt = {'version': VERSION, 'archive': str(ARCHIVE), 'bytes': ARCHIVE.stat().st_size,
        'sha256': sha(ARCHIVE), 'manifest_sha256': hashlib.sha256((canonical(manifest) + '\n').encode()).hexdigest(),
        'included_files': len(included), 'included_uncompressed_bytes': sum(x['bytes'] for x in included.values()),
        'excluded_required_artifacts': excluded, 'validation': validation,
        'new_model_calls': 0, 'new_model_loads': 0, 'secret_access': False,
        'complete_direct_runtime_restore': False, 'automatic_retry_or_resume': False}
    with RECEIPT.open('x', encoding='utf-8') as stream:
        stream.write(canonical(receipt) + '\n'); stream.flush(); os.fsync(stream.fileno())
    print(canonical(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-protocol-sha', required=True)
    parser.add_argument('--expected-report-sha', required=True)
    parser.add_argument('--expected-authentication-sha', required=True)
    args = parser.parse_args()
    export(args.expected_protocol_sha, args.expected_report_sha, args.expected_authentication_sha)
