"""Verify a fixed CPU closure locally; restore only in genuine hosted Colab.

No model, tokenizer, package, dataset parser, network request or archived code
is executed. All archive bytes and all existing destinations are checked before
any restore write. Byte-identical existing files are retained; conflicts stop.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile
import time
import zipfile

VERSION = 'safe-original-path-cpu-v6-closure-restore-v1'
EXPECTED_ARCHIVE_BYTES = 1374854
EXPECTED_ARCHIVE_SHA = '5356f82a0330af8c467bbc2de546b27d889858f1a7a4abc4931b8a8c66e8b9dc'
EXPECTED_PAYLOAD_FILES = 225
CONTENT = Path('/content')
REPOSITORY = 'https://github.com/nanlalala/cbt-memory-system'
CHECKOUTS = (
    ('/content/cbt-open-state-before-20261112-v2', 'e2899b10f7404847defb246ed7769aeb46cf228f'),
    ('/content/cbt-open-state-after-20261112-v2', '06223cffc09a6ab7972793cef78faa7364c4229d'),
    ('/content/cbt-prefill-runtime-code-20261112-v2', '8d1b0f7e6688dc480ad4ef282a80b8c1fd4916fb'),
)
DEFAULT_RECEIPT = CONTENT / 'cbt-cpu-v6-closure-restore-receipt-20261112-v1.json'
BANNED_PARTS = frozenset({'.git', '.ssh', '.config', '.cache', 'site-packages', 'secrets', 'credentials'})
BANNED_NAMES = frozenset({'.env', 'credentials.json', 'tokenizer.json', 'tokenizer_config.json',
                          'vocab.json', 'merges.txt', 'special_tokens_map.json'})
BANNED_SUFFIXES = frozenset({'.gguf', '.safetensors', '.bin', '.partial', '.pt', '.pth'})
ALLOWED_SUFFIXES = frozenset({'.json', '.jsonl', '.py', '.log', '.txt', '.md', '.csv',
                              '.html', '.sqlite', '.db', '.gz', '.whl'})
MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 100 * 1024 * 1024


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def strict_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate_json_key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite_json')))


def is_hosted():
    try:
        return (bool(os.environ.get('COLAB_RELEASE_TAG')) and CONTENT.is_dir()
                and importlib.util.find_spec('google.colab') is not None)
    except (ImportError, ValueError, AttributeError):
        return False


def require_hosted():
    if not is_hosted() or CONTENT.resolve() != CONTENT:
        raise RuntimeError('genuine_hosted_colab_required_before_destination_io')


def member_target(member):
    """Pure archive path validation, without trusting extraction defaults."""
    if (type(member) is not str or not member or '\\' in member or '\x00' in member
            or ':' in member or member.startswith('/') or member.endswith('/')):
        raise ValueError('unsafe_archive_member_name')
    parts = PurePosixPath(member).parts
    if (not parts or parts[0] != 'content' or len(parts) < 2
            or any(part in ('', '.', '..') for part in parts)
            or PurePosixPath(member).as_posix() != member):
        raise ValueError('archive_member_must_map_exactly_under_content')
    target = Path('/').joinpath(*parts)
    if (any(part in BANNED_PARTS for part in parts) or target.name in BANNED_NAMES
            or target.suffix.lower() in BANNED_SUFFIXES
            or target.suffix.lower() not in ALLOWED_SUFFIXES):
        raise ValueError('binary_runtime_model_or_private_path_not_restorable')
    return target


def destination_state(path, size, expected_sha):
    """Read-only target check; never traverse a symlink or regular-file parent."""
    path = Path(path)
    if not path.is_absolute() or not path.is_relative_to(CONTENT) or '..' in path.parts or path == CONTENT:
        raise ValueError('original_owned_content_path_required')
    for parent in path.parents:
        if parent.is_symlink():
            return 'conflict_symlink_parent'
        if parent.exists() and not parent.is_dir():
            return 'conflict_nondirectory_parent'
        if parent == CONTENT:
            break
    if path.is_symlink():
        return 'conflict_symlink'
    if not path.exists():
        return 'missing'
    if not path.is_file():
        return 'conflict_nonregular'
    return ('identical_existing' if path.stat().st_size == size and sha(path) == expected_sha
            else 'conflict_different_bytes')


def audit_archive(archive_path):
    archive_path = Path(archive_path)
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ValueError('regular_known_closure_archive_required')
    if archive_path.stat().st_size != EXPECTED_ARCHIVE_BYTES or sha(archive_path) != EXPECTED_ARCHIVE_SHA:
        raise ValueError('exact_verified_cpu_closure_bytes_required')
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.infolist()
        names = [item.filename for item in info]
        if len(names) != len(set(names)) or names.count('MANIFEST.json') != 1:
            raise ValueError('duplicate_or_missing_manifest_archive_members')
        if len(info) != EXPECTED_PAYLOAD_FILES + 1:
            raise ValueError('exact_225_payloads_plus_manifest_required')
        manifest_info = archive.getinfo('MANIFEST.json')
        if manifest_info.file_size > 2 * 1024 * 1024:
            raise ValueError('bounded_manifest_required')
        manifest_bytes = archive.read('MANIFEST.json')
        manifest = strict_json(manifest_bytes)
        files = manifest.get('files')
        if type(files) is not list or len(files) != EXPECTED_PAYLOAD_FILES:
            raise ValueError('exact_original_payload_catalog_required')
        expected = {}
        destinations = set()
        total = 0
        for row in files:
            member = row['archive_path']
            target = member_target(member)
            size, checksum = row['bytes'], row['sha256']
            if (row['source_path'] != str(target) or member in expected or str(target) in destinations
                    or type(size) is not int or not 0 <= size <= MAX_MEMBER_BYTES
                    or type(checksum) is not str or re.fullmatch('[0-9a-f]{64}', checksum) is None):
                raise ValueError('invalid_or_conflicting_original_source_mapping')
            expected[member] = row
            destinations.add(str(target)); total += size
        if total > MAX_TOTAL_BYTES or set(names) != {'MANIFEST.json', *expected}:
            raise ValueError('unlisted_or_oversized_archive_content')
        verified = []
        for item in info:
            mode = item.external_attr >> 16
            if item.is_dir() or (stat.S_IFMT(mode) not in (0, stat.S_IFREG)) or item.flag_bits & 1:
                raise ValueError('only_unencrypted_regular_archive_files_allowed')
            if item.filename == 'MANIFEST.json':
                continue
            row = expected[item.filename]
            if item.file_size != row['bytes']:
                raise ValueError('zip_header_and_manifest_size_disagree')
            h, count = hashlib.sha256(), 0
            with archive.open(item) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    count += len(block); h.update(block)
                    if count > row['bytes']:
                        raise ValueError('archive_member_exceeds_frozen_size')
            if count != row['bytes'] or h.hexdigest() != row['sha256']:
                raise ValueError('original_archive_member_byte_identity_failed')
            verified.append({key: row[key] for key in ('source_path', 'archive_path', 'bytes', 'sha256')})
        validation = manifest.get('validation', {})
        if (validation.get('published_attempts') != 8 or validation.get('typed_correct') != 8
                or validation.get('schema_valid') != 8
                or manifest.get('new_model_calls') != 0 or manifest.get('secret_access') is not False):
            raise ValueError('terminal_cpu_gate_manifest_identity_required')
        exclusions = manifest.get('excluded_required_artifacts')
        if type(exclusions) is not list or not exclusions:
            raise ValueError('explicit_missing_restore_dependencies_required')
        for row in exclusions:
            if (type(row.get('source_path')) is not str or not Path(row['source_path']).is_relative_to(CONTENT)
                    or type(row.get('bytes')) is not int or row['bytes'] < 0
                    or re.fullmatch('[0-9a-f]{64}', row.get('sha256', '')) is None):
                raise ValueError('invalid_required_dependency_descriptor')
        # These are public setup manifests, not semantic dialogue data.
        model = strict_json(archive.read('content/cbt-qwen7b-reader-gate-20261112-v6/model_manifest.json'))
        runtime = strict_json(archive.read('content/cbt-qwen7b-reader-gate-20261112-v6/runtime_manifest.json'))
    audit = {'version': VERSION, 'archive_path': str(archive_path.resolve()),
        'archive_bytes': EXPECTED_ARCHIVE_BYTES, 'archive_sha256': EXPECTED_ARCHIVE_SHA,
        'zip_members': len(info), 'payload_files_verified': len(verified),
        'manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
        'all_payload_sizes_and_sha256_verified': True,
        'payload_uncompressed_bytes': total, 'validation': validation,
        'files': sorted(verified, key=lambda row: row['source_path']),
        'required_excluded_runtime_artifacts': exclusions,
        'required_model_manifest': model, 'required_runtime_manifest': runtime,
        'checkout_requirements': [{'path': path, 'head': head, 'repository': REPOSITORY,
            'identity_basis': 'previously observed frozen hosted checkout; .git is not included in closure',
            'clone_argv_if_missing': ['git', 'clone', '--no-checkout', REPOSITORY, path],
            'checkout_argv_after_clone': ['git', '-C', path, 'checkout', '--detach', head]}
            for path, head in CHECKOUTS],
        'semantic_response_or_dialogue_review': False, 'native_detokenization_repeated': False,
        'models_executed': 0, 'network_calls': 0, 'source_archive_modified': False}
    return audit


def preflight_destinations(audit):
    require_hosted()
    clone_states = []
    for directory, expected in CHECKOUTS:
        path = Path(directory)
        if path.is_symlink():
            clone_states.append({'path': directory, 'expected_head': expected, 'status': 'conflict_symlink'})
            continue
        if not (path / '.git').exists():
            clone_states.append({'path': directory, 'expected_head': expected, 'status': 'missing_git_checkout'})
            continue
        actual = subprocess.run(['git', '-C', directory, 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=10, check=False)
        observed = actual.stdout.strip() if actual.returncode == 0 else None
        clone_states.append({'path': directory, 'expected_head': expected, 'observed_head': observed,
            'status': 'exact_head' if observed == expected else 'conflict_git_head'})
    targets = [{**row, 'destination_state': destination_state(row['source_path'], row['bytes'], row['sha256'])}
               for row in audit['files']]
    conflicts = [row for row in targets if row['destination_state'].startswith('conflict_')]
    return {'checkout_states': clone_states, 'destination_files': targets,
        'metadata_restore_ready': not conflicts and all(row['status'] == 'exact_head' for row in clone_states),
        'conflicts': conflicts, 'missing_files': sum(row['destination_state'] == 'missing' for row in targets),
        'identical_existing_files': sum(row['destination_state'] == 'identical_existing' for row in targets),
        'runtime_ready': False,
        'runtime_note': 'CPU wheel, installed packages and full model shards must be separately reacquired/verified; this helper performs none of those actions'}


def save_new_json(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(canonical(value) + '\n'); stream.flush(); os.fsync(stream.fileno())


def restore(archive_path, audit, *, receipt):
    require_hosted()
    receipt = Path(receipt)
    if (not receipt.is_absolute() or not receipt.is_relative_to(CONTENT) or '..' in receipt.parts
            or receipt.is_symlink() or receipt.exists()
            or str(receipt) in {row['source_path'] for row in audit['files']}
            or destination_state(receipt, 0, '0' * 64) != 'missing'):
        raise FileExistsError('fresh_owned_restore_receipt_required')
    plan = preflight_destinations(audit)
    if not plan['metadata_restore_ready']:
        raise RuntimeError('all_git_heads_and_destination_conflicts_must_pass_before_any_restore_write')
    written, skipped = [], []
    started = time.monotonic()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            for row in plan['destination_files']:
                target = Path(row['source_path'])
                current = destination_state(target, row['bytes'], row['sha256'])
                if current == 'identical_existing':
                    skipped.append(str(target)); continue
                if current != 'missing':
                    raise ValueError('destination_changed_after_preflight_no_overwrite')
                target.parent.mkdir(parents=True, exist_ok=True)
                if destination_state(target, row['bytes'], row['sha256']) != 'missing':
                    raise ValueError('destination_race_before_exclusive_install')
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(mode='wb', prefix='.cpu-v6-restore-',
                            dir=target.parent, delete=False) as output:
                        temporary = Path(output.name)
                        h, count = hashlib.sha256(), 0
                        with archive.open(row['archive_path']) as source:
                            for block in iter(lambda: source.read(1024 * 1024), b''):
                                output.write(block); h.update(block); count += len(block)
                        output.flush(); os.fsync(output.fileno())
                    if count != row['bytes'] or h.hexdigest() != row['sha256']:
                        raise ValueError('staged_restore_bytes_changed')
                    os.link(temporary, target, follow_symlinks=False)  # no replacement
                    temporary.unlink(); temporary = None
                    directory_fd = os.open(target.parent, os.O_RDONLY)
                    try:
                        os.fsync(directory_fd)
                    finally:
                        os.close(directory_fd)
                    if sha(target) != row['sha256']:
                        raise ValueError('installed_restore_bytes_failed_verification')
                    written.append(str(target))
                finally:
                    if temporary is not None and temporary.exists():
                        temporary.unlink()
        if sha(archive_path) != EXPECTED_ARCHIVE_SHA:
            raise ValueError('input_archive_changed_during_restore')
        for row in audit['files']:
            if destination_state(row['source_path'], row['bytes'], row['sha256']) != 'identical_existing':
                raise ValueError('final_original_path_restore_verification_failed')
        status, failure = 'all_225_original_paths_restored_or_retained_exactly', None
    except BaseException as error:
        status, failure = 'restore_stopped_partial_files_retained_no_overwrite', type(error).__name__
    result = {'version': VERSION, 'stage': status, 'archive_sha256': EXPECTED_ARCHIVE_SHA,
        'archive_bytes': EXPECTED_ARCHIVE_BYTES, 'restored_files': written, 'identical_files_retained': skipped,
        'failure_type': failure, 'seconds': time.monotonic() - started,
        'protocol_bytes_modified': False, 'database_controls_reset': False,
        'required_runtime_artifacts_not_restored': audit['required_excluded_runtime_artifacts'],
        'full_runtime_ready': False, 'model_loads': 0, 'model_calls': 0, 'network_calls': 0,
        'automatic_retry': False}
    save_new_json(receipt, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--audit-output', type=Path)
    parser.add_argument('--receipt', type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    audit = audit_archive(args.archive)
    if args.verify_only:
        if is_hosted():
            audit['target_preflight'] = preflight_destinations(audit)
        else:
            audit['target_preflight'] = {'performed': False, 'reason': 'outside_genuine_hosted_colab'}
        if args.audit_output:
            save_new_json(args.audit_output, audit)
        print(canonical({key: audit[key] for key in ('version', 'archive_bytes', 'archive_sha256',
            'zip_members', 'payload_files_verified', 'all_payload_sizes_and_sha256_verified', 'target_preflight')}))
    else:
        print(canonical(restore(args.archive, audit, receipt=args.receipt)))
