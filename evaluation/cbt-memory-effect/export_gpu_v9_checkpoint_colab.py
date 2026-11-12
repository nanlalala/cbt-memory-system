"""Read-only GPU checkpoint with registered study identities and preserved failures.

Standard library only. Does not import a reader, tokenizer, model or scorer;
does not read credentials, notebook profiles, Secrets or environment values.
Native token-to-text reproduction is NOT re-executed by this byte exporter.
SQLite online backups are labelled snapshots, never original live DB bytes.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import ExitStack
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import time
import zipfile

VERSION = 'private-gpu-study-byte-checkpoint-v3-explicit-v8-v9-registry'
STUDY_VERSION = 'native-t4-gguf-question-schema-state-development-study-v9-identical-binding-registration'
STUDY_IDENTITIES = {
    'cbt-qwen7b-gpu-state-study-20261112-v8': 'native-t4-gguf-question-schema-state-development-study-v8-import-isolation',
    'cbt-qwen7b-gpu-state-study-20261112-v9': 'native-t4-gguf-question-schema-state-development-study-v9-identical-binding-registration',
}
CONTENT = Path('/content')
GATE = CONTENT / 'cbt-qwen7b-gpu-reader-gate-20261112-v7'
STUDY = CONTENT / 'cbt-qwen7b-gpu-state-study-20261112-v9'
FAILED_V8 = CONTENT / 'cbt-qwen7b-gpu-state-study-20261112-v8'
FAILED_V8_PROTOCOL_SHA = 'e2c14988f5f4ddca7472e7ef861a9b73ae630d0254b08a7ade357a49a1223eb1'
FAILED_V7 = CONTENT / 'cbt-qwen7b-gpu-state-study-20261112-v7'
FAILED_V7_PROTOCOL_SHA = '73f992b4fc6885d1ed005228e6266b0b8726969585f8382ea8c09cecebab745b'
CPU_GATE = CONTENT / 'cbt-qwen7b-reader-gate-20261112-v6'
PREVIOUS_V7_ARCHIVE_SHA = '3d48a1d26ecbed6e604dce8a49b05b621c4c7b312f9acea7864dc9ef12dd8d0d'
PREVIOUS_V7_ARCHIVE_BYTES = 3753249
CODE = CONTENT / 'cbt-qwen7b-gpu-v7-code'
MODEL = CONTENT / 'cbt-qwen7b-gpu-reader-setup-20261112-v7'
PYTHON = CONTENT / 'cbt-qwen7b-gpu-python-20261112-v7'
RECOVERY8 = CONTENT / 'cbt-qwen7b-gpu-python-recovery-20261112-v8'
RECOVERY9 = CONTENT / 'cbt-qwen7b-gpu-python-recovery-20261112-v9'
READINESS = CONTENT / 'cbt-gpu-readiness-20261112-v1'
NVIDIA_META = CONTENT / 'cbt-cuda12-dependency-metadata-20261112-v8'
PUBLIC_META = CONTENT / 'cbt-source-and-reader-metadata-20261112-v1'
PREPARATION = CONTENT / 'cbt-gpu-v7-main-preparation-20261112-v1'
CPU_CLOSURE = CONTENT / 'CBT_CPU_V6_Closure_20261112_PRIVATE.zip'
CPU_CLOSURE_SHA = '5356f82a0330af8c467bbc2de546b27d889858f1a7a4abc4931b8a8c66e8b9dc'
ARCHIVE = CONTENT / 'CBT_GPU_Study_Checkpoint_20261112_PRIVATE.zip'
RECEIPT = CONTENT / 'CBT_GPU_Study_Checkpoint_20261112_PRIVATE.receipt.json'
MAX_FILE = 16 * 1024 * 1024
MAX_TOTAL = 100 * 1024 * 1024
MAX_ZIP = 10 * 1024 * 1024
MAX_FILES = 6000
SUFFIXES = {'.json', '.jsonl', '.py', '.md', '.txt', '.log', '.csv', '.html', '.db', '.sqlite', '.sqlite3', '.gz', '.lock'}
SKIP_DIRS = {'site-packages', '__pycache__', '.git', '.ssh', '.config', '.cache', 'secrets', 'credentials', 'node_modules'}
SKIP_NAMES = {'.env', '.netrc', 'credentials.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'special_tokens_map.json'}
BINARY_SUFFIXES = {'.gguf', '.safetensors', '.pt', '.pth', '.whl', '.so', '.bin', '.partial'}
META_NAMES = {'study_protocol.json', 'main_freeze.json', 'gate_protocol.json', 'model_manifest.json', 'runtime_manifest.json',
    'protocol.json', 'setup_protocol.json', 'setup_receipt.json', 'recovery_protocol.json', 'native_dependency_manifest.json',
    'model_setup_receipt.json', 'download_protocol.json', 'download_complete_receipt.json'}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def hosted():
    try:
        okay = (bool(os.environ.get('COLAB_RELEASE_TAG')) and CONTENT.is_dir()
                and importlib.util.find_spec('google.colab') is not None)
    except (ImportError, ValueError, AttributeError): okay = False
    if not okay: raise RuntimeError('genuine_hosted_colab_required_before_artifact_io')


def owned(path):
    p = Path(path)
    if not p.is_absolute() or '..' in p.parts or p == CONTENT or not p.is_relative_to(CONTENT):
        raise ValueError('only_explicit_original_content_paths_allowed')
    for parent in (p, *p.parents):
        if parent.is_symlink(): raise ValueError('symlinked_checkpoint_path_prohibited')
        if parent == CONTENT: break
    return p


def require_sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('externally_observed_exporter_sha256_required')


def read(path):
    p = owned(path)
    if p.stat().st_size > MAX_FILE: raise ValueError('bounded_metadata_required')
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result: raise ValueError('duplicate_json_key')
            result[k] = v
        return result
    return json.loads(p.read_text(encoding='utf-8'), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite_json')))


def stat_info(path):
    s = Path(path).stat()
    return {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns, 'device': s.st_dev, 'inode': s.st_ino}


def excluded(path):
    p = Path(path)
    if any(part in SKIP_DIRS for part in p.parts) or p.name in SKIP_NAMES: return 'installation_credentials_or_tokenizer_excluded'
    if p.suffix.lower() in BINARY_SUFFIXES or '.so.' in p.name: return 'model_or_runtime_binary_excluded'
    if p.name.endswith(('-wal', '-shm', '-journal')): return 'sqlite_sidecar_represented_by_online_backup'
    return None


def rows(path, *, require_record_sha=False):
    """Return only complete original records; a torn final line is explicit."""
    p = owned(path)
    if not p.exists(): return [], {'exists': False, 'torn_final_line': False, 'record_errors': []}
    if p.stat().st_size > MAX_FILE: raise ValueError('bounded_journal_required')
    b = p.read_bytes()
    complete = b if b.endswith(b'\n') else b[:b.rfind(b'\n') + 1]
    parsed, errors = [], []
    previous = None
    for n, line in enumerate(complete.splitlines(), 1):
        try:
            r = json.loads(line)
            if type(r) is not dict: raise ValueError('journal_object_required')
            expected = r.get('record_sha256')
            if require_record_sha and expected is None: raise ValueError('record_sha256_required')
            if expected is not None:
                value = dict(r); value.pop('record_sha256')
                if digest(value) != expected: raise ValueError('record_sha256_changed')
            if 'sequence' in r:
                if r['sequence'] != n or r.get('previous_record_sha256') != previous:
                    raise ValueError('record_chain_changed')
            parsed.append(r); previous = expected
        except (ValueError, TypeError) as e:
            errors.append({'line': n, 'failure_type': type(e).__name__})
    return parsed, {'exists': True, 'torn_final_line': complete != b, 'record_errors': errors}


def infer_product(origin, checkouts):
    text = str(origin)
    found = [side for side in ('before', 'after') if ('/' + side + '/') in text or ('-' + side + '-') in text]
    return checkouts.get(found[0]) if len(found) == 1 else None


def hash_references(value, origin, checkouts, *, product=None, source=None):
    if type(value) is dict:
        product = product or infer_product(origin, checkouts)
        if type(value.get('checkout')) is str: product = owned(value['checkout'])
        if type(value.get('source_prepared')) is str: source = owned(value['source_prepared'])
        for key, child in value.items():
            if type(child) is dict and (key.endswith('_sha256') or key in {'source_files', 'code_hashes', 'reference_hashes'}):
                for name, checksum in child.items():
                    if type(name) is str and type(checksum) is str and re.fullmatch('[0-9a-f]{64}', checksum):
                        if key == 'cpu_gate_reference_sha256':
                            # These three names are role aliases in the actual V7
                            # GPU protocol, not filenames relative to that protocol.
                            catalog = {'protocol': CPU_GATE / 'gate_protocol.json',
                                'result': CPU_GATE / 'gate_results.json',
                                'ledger': CPU_GATE / 'generation_ledger.jsonl'}
                            if origin != GATE / 'gate_protocol.json' or name not in catalog:
                                yield None, checksum, key + ':unknown_or_foreign_alias:' + name
                            else: yield catalog[name], checksum, key + ':explicit_cpu_gate_alias:' + name
                            continue
                        p = Path(name)
                        if not p.is_absolute():
                            if p.parts[0] in {'app', 'modules', 'evaluation', 'tests', 'tools'} or key == 'source_files':
                                if product is None: yield None, checksum, key + ':' + name; continue
                                base = product
                            elif key == 'source_files_sha256':
                                if source is None: yield None, checksum, key + ':' + name; continue
                                base = source
                            else: base = origin.parent
                            p = base / p
                        yield p, checksum, key
            yield from hash_references(child, origin, checkouts, product=product, source=source)
    elif type(value) is list:
        for child in value: yield from hash_references(child, origin, checkouts, product=product, source=source)


class Collector:
    def __init__(self, stage, issues):
        self.stage, self.issues = stage, issues
        self.entries, self.queue, self.descriptors = {}, [], {}
        self.log_ranges = []
        self.pinned_extra_archives = {}
        self.checkouts = {}
        if (STUDY / 'main_freeze.json').exists():
            plan = read(STUDY / 'main_freeze.json')
            self.checkouts = {side: owned(s['checkout']) for side, s in plan.get('sections', {}).items() if type(s.get('checkout')) is str}
        for side in ('before', 'after'):
            fallback = CONTENT / ('cbt-open-state-' + side + '-20261112-v2')
            if side not in self.checkouts and fallback.exists(): self.checkouts[side] = fallback

    def binary_descriptor(self, path, *, expected=None, item=None, origin='excluded_binary'):
        p = Path(path)
        old = self.descriptors.get(str(p), {})
        item = item or {}
        checksum = expected or item.get('sha256') or item.get('verified_sha256') or old.get('sha256')
        size = item.get('bytes') or item.get('size') or old.get('bytes')
        observed = stat_info(p) if p.is_file() else None
        if size is None and observed is not None: size = observed['bytes']
        url = item.get('url') or item.get('reacquire_original_url') or old.get('reacquire_original_url')
        self.descriptors[str(p)] = {**old, 'source_path': str(p), 'bytes': size,
            'sha256': checksum, 'observed_stat': observed, 'reason': excluded(p) or 'host_native_library_excluded',
            'required_for_runtime_restore': old.get('required_for_runtime_restore', p.suffix not in {'.bin', '.partial'}),
            'reacquire_original_url': url,
            'sha256_scope': 'declared frozen/publisher identity; excluded binary is not read or rehashed by exporter',
            'download_or_install_performed_by_exporter': False, 'origin': origin}
        if observed is not None and size is not None and observed['bytes'] != size:
            self.issues.append({'kind': 'excluded_binary_size_differs_from_declared', 'path': str(p)})

    def add(self, path, expected=None, origin='explicit_owned_seed'):
        if path is None:
            self.issues.append({'kind': 'unresolved_relative_required_reference', 'origin': origin, 'expected_sha256': expected}); return
        path = Path(path)
        if not path.is_relative_to(CONTENT):
            if expected and (path.suffix in {'.so', '.1'} or '.so.' in path.name):
                self.binary_descriptor(path, expected=expected, origin=origin); return
            self.issues.append({'kind': 'required_reference_outside_explicit_content_scope', 'path': str(path), 'origin': origin}); return
        try: path = owned(path)
        except ValueError:
            self.issues.append({'kind': 'unsafe_required_path', 'path': str(path), 'origin': origin}); return
        reason = excluded(path)
        if reason:
            if expected is not None or path.suffix in BINARY_SUFFIXES:
                self.binary_descriptor(path, expected=expected, origin=origin)
            return
        if not path.is_file():
            self.issues.append({'kind': 'required_reference_missing', 'path': str(path), 'origin': origin}); return
        if (path.suffix not in SUFFIXES and path != CPU_CLOSURE and str(path) not in self.pinned_extra_archives
                and '.cbt-memory-control' not in path.parts):
            self.issues.append({'kind': 'required_file_outside_whitelist', 'path': str(path), 'origin': origin}); return
        before = stat_info(path)
        if before['bytes'] > MAX_FILE: raise ValueError('nonbinary_checkpoint_file_too_large')
        checksum = sha(path)
        if expected is not None and checksum != expected:
            self.issues.append({'kind': 'required_reference_sha256_changed', 'path': str(path), 'expected': expected, 'actual': checksum})
        if str(path) in self.entries:
            if self.entries[str(path)]['source_sha256'] != checksum:
                self.issues.append({'kind': 'source_changed_during_collection', 'path': str(path)})
            return
        representation = 'original_bytes'
        archive_source = self.stage / ('capture-' + hashlib.sha256(str(path).encode()).hexdigest())
        if path.suffix in {'.db', '.sqlite', '.sqlite3'}:
            target = self.stage / ('sqlite-' + hashlib.sha256(str(path).encode()).hexdigest() + '.db')
            with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=5) as src, sqlite3.connect(target) as dest:
                deadline = time.monotonic() + 15
                def progress(*_):
                    if time.monotonic() > deadline: raise TimeoutError('sqlite_snapshot_deadline')
                src.execute('PRAGMA query_only=ON'); src.backup(dest, pages=64, progress=progress, sleep=0.02)
            representation, archive_source = 'sqlite_online_backup_not_original_live_database_bytes', target
        else:
            with path.open('rb') as source, archive_source.open('xb') as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
        after = stat_info(path)
        if before != after: self.issues.append({'kind': 'source_stat_changed_during_capture', 'path': str(path)})
        if representation == 'original_bytes' and sha(archive_source) != checksum:
            self.issues.append({'kind': 'source_bytes_changed_during_capture', 'path': str(path)})
        self.entries[str(path)] = {'source_path': str(path), 'archive_path': 'content/' + path.relative_to(CONTENT).as_posix(),
            'bytes': archive_source.stat().st_size, 'sha256': sha(archive_source), 'source_sha256': checksum,
            'source_stat_before': before, 'source_stat_after': after, 'representation': representation,
            'capture_source': str(archive_source), 'origin': origin}
        if len(self.entries) > MAX_FILES or sum(x['bytes'] for x in self.entries.values()) > MAX_TOTAL:
            raise ValueError('bounded_checkpoint_scope_exceeded')
        if path.suffix == '.json' and (path.name in META_NAMES or 'manifest' in path.name
                or 'protocol' in path.name or 'receipt' in path.name or path.name.startswith(('native_setup_', 'setup_claim_'))):
            self.queue.append(path)

    def tree(self, root):
        root = owned(root)
        if not root.exists(): return
        for base, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
            for name in sorted(files):
                p = Path(base) / name
                if p.name in SKIP_NAMES: continue
                if p.suffix == '.whl' or p.suffix == '.gguf' or p.suffix in {'.bin', '.partial'}:
                    self.binary_descriptor(p); continue
                if p.name.endswith(('-wal', '-shm', '-journal')): continue
                self.add(p)

    def catalogs(self, value, origin):
        """Describe binaries using preserved public publisher metadata, no binary reads."""
        if type(value) is dict:
            if origin.name == 'range_preflight_receipt.json' and value.get('passed') is True:
                p = origin.parent / 'first_shard_exact_1MiB_probe.bin'
                self.binary_descriptor(p, item=value, origin=str(origin))
                self.descriptors[str(p)]['required_for_runtime_restore'] = False
                self.descriptors[str(p)]['restore_requirement'] = 'diagnostic transport probe; not needed to restore inference runtime'
            if type(value.get('installed_package_files')) is list and type(value.get('site_packages')) is str:
                for entry in value['installed_package_files']:
                    if type(entry) is dict and type(entry.get('path')) is str:
                        self.binary_descriptor(Path(value['site_packages']) / entry['path'], item=entry, origin=str(origin))
            if type(value.get('constructor_log')) is dict:
                log = value['constructor_log']
                try:
                    p = owned(log['path']); self.add(p, origin=str(origin) + '#constructor_log')
                    start, end = log['start_byte'], log['end_byte']
                    if type(start) is not int or type(end) is not int or not 0 <= start < end or end - start > MAX_FILE:
                        raise ValueError('invalid_log_range')
                    with p.open('rb') as f:
                        f.seek(start); body = f.read(end - start)
                    if len(body) != end - start or log.get('bytes') != len(body) or hashlib.sha256(body).hexdigest() != log['sha256']:
                        raise ValueError('log_range_bytes_changed')
                    self.log_ranges.append({**log, 'origin': str(origin), 'byte_range_sha256_verified': True})
                except (KeyError, ValueError, OSError, TypeError) as e:
                    self.issues.append({'kind': 'gpu_constructor_log_range_not_verified', 'origin': str(origin), 'failure_type': type(e).__name__})
            if type(value.get('path')) is str and Path(value['path']).is_absolute() and excluded(Path(value['path'])):
                self.binary_descriptor(Path(value['path']), item=value, origin=str(origin))
            if type(value.get('model')) is dict:
                model = value['model']
                if type(model.get('repo')) is str and type(model.get('revision')) is str:
                    for f in model.get('files', []):
                        if type(f) is dict and type(f.get('name')) is str:
                            self.binary_descriptor(origin.parent / f['name'], item={**f,
                                'url': 'https://huggingface.co/' + model['repo'] + '/resolve/' + model['revision'] + '/' + f['name']}, origin=str(origin))
            if type(value.get('wheels')) is list:
                directory = Path(value.get('wheel_directory', origin.parent))
                for f in value['wheels']:
                    if type(f) is dict and type(f.get('filename')) is str:
                        self.binary_descriptor(directory / f['filename'], item=f, origin=str(origin))
            if type(value.get('reused_llama_cpp_wheel')) is dict:
                f = value['reused_llama_cpp_wheel']
                if type(f.get('path')) is str:
                    self.binary_descriptor(Path(f['path']), item=f, origin=str(origin))
            if type(value.get('driver_reference')) is dict:
                f = value['driver_reference']
                if type(f.get('resolved_path')) is str:
                    self.binary_descriptor(Path(f['resolved_path']), item=f, origin=str(origin))
            for k, child in value.items():
                if type(child) is str and (k.endswith('_path') or k.endswith('_file') or k == 'path') and Path(child).is_absolute():
                    p = Path(child)
                    if p.is_relative_to(CONTENT) and p.is_file() and not excluded(p): self.add(p, origin=str(origin) + '#' + k)
                elif type(child) in (dict, list): self.catalogs(child, origin)
        elif type(value) is list:
            for child in value: self.catalogs(child, origin)

    def close_references(self):
        while self.queue:
            origin = self.queue.pop(0)
            try:
                value = read(origin)
                self.catalogs(value, origin)
                for path, expected, key in hash_references(value, origin, self.checkouts):
                    self.add(path, expected, str(origin) + '#' + key)
            except (ValueError, KeyError, TypeError) as e:
                self.issues.append({'kind': 'metadata_reference_parse_or_resolution_failed', 'path': str(origin), 'failure_type': type(e).__name__})

    def cpu_restore_descriptors(self):
        if not CPU_CLOSURE.is_file(): return
        try:
            with zipfile.ZipFile(CPU_CLOSURE) as z:
                info = z.getinfo('MANIFEST.json')
                if info.file_size > MAX_FILE: raise ValueError('bounded_nested_manifest_required')
                value = json.loads(z.read(info))
            for entry in value.get('excluded_required_artifacts', []):
                if type(entry) is not dict or type(entry.get('source_path')) is not str:
                    raise ValueError('nested_cpu_exclusion_descriptor_required')
                self.binary_descriptor(Path(entry['source_path']), item=entry, origin='preserved_nested_cpu_closure_manifest')
        except (KeyError, ValueError, OSError, zipfile.BadZipFile) as e:
            self.issues.append({'kind': 'nested_cpu_restore_descriptors_unavailable', 'failure_type': type(e).__name__})

    def finalize_restore_descriptors(self):
        wheels = [d for d in self.descriptors.values() if d['source_path'].endswith('.whl') and d.get('reacquire_original_url')]
        for d in self.descriptors.values():
            p = Path(d['source_path']); text = str(p)
            if d.get('reacquire_original_url'): continue
            prefix = None
            if '/nvidia/cuda_runtime/' in text: prefix = 'nvidia_cuda_runtime_cu12-'
            elif '/nvidia/cublas/' in text: prefix = 'nvidia_cublas_cu12-'
            elif '/site-packages/llama_cpp/' in text: prefix = 'llama_cpp_python-'
            elif '/site-packages/diskcache/' in text: prefix = 'diskcache-'
            candidates = [w for w in wheels if prefix and Path(w['source_path']).name.startswith(prefix)
                and (not text.startswith(str(PYTHON)) or w['source_path'].startswith(str(PYTHON)))]
            if len(candidates) == 1:
                w = candidates[0]
                d['reacquire_original_url'] = w['reacquire_original_url']
                d['restore_from_original_wheel'] = {k: w[k] for k in ('source_path', 'bytes', 'sha256', 'reacquire_original_url')}
                d['reacquire_url_is_container_wheel_not_individual_library'] = True
            elif not p.is_relative_to(CONTENT):
                d['restore_requirement'] = 'same hosted NVIDIA driver library identity; not a redistributable project artifact'
            if d['required_for_runtime_restore'] and not d.get('sha256'):
                self.issues.append({'kind': 'excluded_restore_artifact_has_no_frozen_sha256', 'path': str(p)})


def ledger_state(root):
    rs, integrity = rows(root / 'generation_ledger.jsonl', require_record_sha=True)
    intents = [r for r in rs if r.get('event') == 'generation_intent']
    outcomes = [r for r in rs if r.get('event') == 'generation_outcome']
    ik, ok = [r.get('canonical_key', r.get('attempt')) for r in intents], [r.get('canonical_key', r.get('attempt')) for r in outcomes]
    binding = {'duplicate_intents': len(ik) != len(set(ik)), 'duplicate_outcomes': len(ok) != len(set(ok)),
        'outcome_without_intent': sorted(set(ok) - set(ik)),
        'unknown_events': [r.get('event') for r in rs if r.get('event') not in {'generation_intent', 'generation_outcome'}]}
    return {'admitted': len(intents), 'published': sum(r.get('status') == 'published' for r in outcomes),
        'failed_or_unknown': sum(r.get('status') != 'published' for r in outcomes),
        'unsettled_intents': len(set(ik) - set(ok)), 'integrity': integrity, 'event_binding': binding,
        'records': rs}


def validate_answer_bytes(answer):
    ids = answer.get('raw_generated_token_ids')
    if type(ids) is not list or not 1 <= len(ids) <= 64 or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError('observed_native_output_ids_required')
    if any(type(answer['usage'].get(k)) is not int or answer['usage'][k] < 0
           for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')):
        raise ValueError('observed_nonnegative_integer_native_token_usage_required')
    body = base64.b64decode(answer['response_bytes_base64'], validate=True)
    raw = base64.b64decode(answer['raw_generated_bytes_base64'], validate=True)
    if (hashlib.sha256(body).hexdigest() != answer['response_bytes_sha256']
            or body.decode('utf-8', errors='replace') != answer['response_text']
            or raw.decode('utf-8', errors='replace') != answer['raw_generated_text']
            or hashlib.sha256(answer['response_text'].encode()).hexdigest() != answer['response_sha256']
            or answer['usage']['completion_tokens'] != len(ids)
            or answer['usage']['total_tokens'] != answer['usage']['prompt_tokens'] + len(ids)):
        raise ValueError('raw_native_bytes_or_usage_binding_changed')
    return True


def validate_side_bytes(side, section, ledger):
    """Check original recorded bindings only; never retokenize, rescore or load."""
    output = STUDY / ('answers_' + side)
    required = {output / 'answers_unscored.jsonl', output / 'run_protocol.json', output / 'completion_summary.json',
        owned(section['fresh_prepared']) / 'requests.jsonl', STUDY / 'study_protocol.json'}
    answers, integrity = rows(output / 'answers_unscored.jsonl', require_record_sha=True)
    requests, request_integrity = rows(owned(section['fresh_prepared']) / 'requests.jsonl')
    run = read(output / 'run_protocol.json'); run_hash = digest(run)
    summary = read(output / 'completion_summary.json')
    study_sha = sha(STUDY / 'study_protocol.json')
    if (integrity['torn_final_line'] or integrity['record_errors'] or request_integrity['torn_final_line']
            or request_integrity['record_errors'] or len(answers) != len(requests)
            or len(answers) != 48 or run.get('rows_sha256') != digest(requests)
            or run.get('study_protocol_sha256') != study_sha or summary.get('complete') is not True
            or summary.get('logical_completed') != len(answers)
            or summary.get('answers_sha256') != sha(output / 'answers_unscored.jsonl')
            or summary.get('run_protocol_sha256') != run_hash):
        raise ValueError('side_run_summary_or_ordered_request_bytes_changed')
    preflights = run.get('actual_native_input_preflight', [])
    if len(preflights) != len(answers): raise ValueError('complete_ordered_native_input_vectors_required')
    intents = {r['canonical_key']: r for r in ledger['records'] if r.get('event') == 'generation_intent'}
    outcomes = {r['canonical_key']: r for r in ledger['records'] if r.get('event') == 'generation_outcome'}
    receipt_path = STUDY / ('native_setup_' + side + '.json')
    setup = read(receipt_path); proof = setup['gpu_execution_proof']
    revalidation_path = STUDY / ('gate_revalidation_' + side + '.json')
    required.update({receipt_path, revalidation_path})
    revalidation = read(revalidation_path)
    if (setup.get('model_identity') != run.get('model') or proof.get('proof_passed') is not True
            or proof.get('all_layers_offloaded') is not True
            or revalidation.get('native_originals_reproduced') != 8 or revalidation.get('typed_correct') != 8
            or run.get('actual_gpu_phase_execution_binding') != {'phase': side,
                'setup_receipt_sha256': sha(receipt_path), 'gpu_execution_proof_sha256': digest(proof),
                'gate_revalidation_sha256': sha(revalidation_path)}):
        raise ValueError('recorded_gpu_phase_binding_changed')
    for row, answer, pre in zip(requests, answers, preflights):
        validate_answer_bytes(answer)
        elapsed = answer.get('logical_processing_wall_seconds')
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('measured_logical_processing_time_required')
        ids = pre['input_token_ids']
        key = digest({'new_study_protocol_sha256': study_sha, 'model': run['model'],
            'decoding': run['decoding_identity'], 'payload': row['request_payload'], 'max_new_tokens': 64, 'do_sample': False})
        if (digest(row['request_payload']) != row['request_sha256']
                or any(answer.get(k) != row[k] for k in ('case_id', 'arm', 'request_sha256'))
                or answer.get('canonical_key') != key or answer.get('run_protocol_sha256') != run_hash
                or answer.get('source_checks_passed') is not True or key not in outcomes or key not in intents
                or type(ids) is not list or not ids or any(type(x) is not int or x < 0 for x in ids)
                or digest(ids) != pre['input_token_ids_sha256'] or len(ids) != pre['input_tokens']
                or digest(pre['question_schema']) != pre['question_schema_sha256']):
            raise ValueError('logical_request_or_recorded_input_vector_binding_changed')
        outcome, intent = outcomes[key], intents[key]
        cache_path = owned(outcome['cache_path']); owner = owned(intent['output'])
        if not cache_path.is_relative_to(STUDY) or not owner.is_relative_to(STUDY):
            raise ValueError('cache_or_owner_outside_current_study')
        value, owner_run = read(cache_path), read(owner / 'run_protocol.json')
        required.update({cache_path, owner / 'run_protocol.json'})
        if (sha(cache_path) != outcome['cache_sha256'] or any(answer.get(k) != v for k, v in value.items())
                or intent.get('run_protocol_sha256') != digest(owner_run)
                or intent.get('request_sha256') != row['request_sha256']
                or owner_run.get('study_protocol_sha256') != study_sha
                or outcome.get('usage') != value['usage'] or outcome.get('response_sha256') != value['response_sha256']
                or value['usage']['prompt_tokens'] != len(ids)
                or any(value.get(k) != pre[k] for k in ('input_token_ids_sha256', 'rendered_prompt_sha256', 'question_schema_sha256'))
                or any(intent.get(k) != pre[k] for k in ('input_token_ids_sha256', 'question_schema_sha256'))):
            raise ValueError('canonical_original_cache_journal_or_vector_binding_changed')
    return {'logical_rows': len(answers), 'ordered_request_cache_vector_phase_byte_bindings_verified': True,
        'native_token_detokenization_reexecuted': False, 'source_authority_reopened': False,
        'validated_original_file_sha256': {str(p): sha(p) for p in sorted(required)}}


def completion_snapshot_closure(collector, state, issues):
    """Live completion evidence must also be present as identical archived bytes."""
    required = {}
    for value in state.get('side_original_byte_bindings', {}).values():
        required.update(value.get('validated_original_file_sha256', {}))
    if state.get('main_completion_candidate'):
        paths = [STUDY / 'main_freeze.json', STUDY / 'generation_ledger.jsonl',
            *[STUDY / 'reports' / n for n in ('summary.json', 'resources.json', 'before_metrics.json', 'after_metrics.json')],
            *[STUDY / n for n in ('answer_phase_timing_before.json', 'answer_phase_timing_after.json', 'score_phase_timing.json')],
            *[GATE / n for n in ('gate_protocol.json', 'gate_results.json', 'generation_ledger.jsonl',
                'all_eight_authentication_receipt.json', 'actual_native_input_preflight.json')]]
        for p in paths:
            if p.is_file(): required[str(p)] = sha(p)
            else: required[str(p)] = None
    failures = []
    for name, expected in required.items():
        e = collector.entries.get(name)
        if expected is None or e is None or e['sha256'] != expected or e['representation'] != 'original_bytes':
            failures.append(name)
    state['completion_required_snapshot_count'] = len(required)
    state['completion_required_snapshot_failures'] = failures
    if failures:
        issues.append({'kind': 'live_completion_evidence_not_identical_in_captured_archive', 'paths': failures})
        state['main_completion_candidate'] = False
        state['state'] = 'main_partial_completion_evidence_not_in_captured_inventory'


def observed_state(issues):
    result = {'gate_root_exists': GATE.is_dir(), 'main_root_exists': STUDY.is_dir(),
        'native_token_to_text_reexecuted_by_exporter': False,
        'automatic_resume_or_resend_authorized': False, 'clinical_or_memory_effect_inferred': False}
    claims = []
    for p in sorted(STUDY.glob('setup_claim_*.json')) if STUDY.is_dir() else []:
        d = read(p); pid = d.get('pid')
        exists = None
        if type(pid) is int and pid > 0:
            try: os.kill(pid, 0); exists = True
            except ProcessLookupError: exists = False
            except PermissionError: exists = None
        claims.append({'path': str(p), 'declared_pid': pid, 'phase': d.get('phase'),
            'process_exists_signal0': exists, 'liveness_is_not_phase_completion': True})
    result['observed_setup_process_claims'] = claims
    for key, root in [('gate', GATE), ('main', STUDY)]:
        state = ledger_state(root)
        result[key + '_ledger'] = {k: v for k, v in state.items() if k != 'records'}
        if state['integrity']['torn_final_line'] or state['integrity']['record_errors']:
            issues.append({'kind': key + '_journal_partial_or_changed'})
        if any(state['event_binding'].values()):
            issues.append({'kind': key + '_journal_event_binding_errors'})
    gate_report = GATE / 'gate_results.json'
    if gate_report.exists():
        d = read(gate_report)
        result['gate_reported_result'] = {k: d.get(k) for k in ('gate_passed', 'schema_valid', 'typed_correct', 'planned')}
    main_report = STUDY / 'reports/summary.json'
    if main_report.exists():
        d = read(main_report)
        result['main_reported_summary'] = d
    frozen = STUDY / 'main_freeze.json'
    if frozen.exists():
        f = read(frozen); result['planned_logical_rows'] = f.get('main_logical_rows')
        result['planned_unique_payloads'] = f.get('main_unique_payloads')
    logical, canonical, answer_errors = 0, set(), []
    for side in ('before', 'after'):
        path = STUDY / ('answers_' + side) / 'answers_unscored.jsonl'
        answers, integrity = rows(path, require_record_sha=True)
        result[side + '_logical_rows'] = len(answers)
        if integrity['torn_final_line'] or integrity['record_errors']:
            answer_errors.append(side + '_logical_rows_integrity')
        for answer in answers:
            try:
                validate_answer_bytes(answer)
                canonical.add(answer['canonical_key'])
                if answer.get('source_checks_passed') is not True: raise ValueError('source_checks_not_passed')
            except (KeyError, ValueError, TypeError) as e:
                answer_errors.append({'side': side, 'case_id': answer.get('case_id'), 'failure_type': type(e).__name__})
        logical += len(answers)
    result.update(logical_answers=logical, canonical_answer_keys=len(canonical), answer_byte_validation_errors=answer_errors)
    if answer_errors: issues.append({'kind': 'logical_answer_byte_binding_errors', 'count': len(answer_errors)})
    gate_pass = result.get('gate_reported_result', {}).get('gate_passed') is True
    ml = result['main_ledger']; target = result.get('planned_unique_payloads')
    bindings = {}
    if frozen.exists():
        plan = read(frozen)
        for side in ('before', 'after'):
            if not (STUDY / ('answers_' + side) / 'completion_summary.json').exists(): continue
            try:
                bindings[side] = validate_side_bytes(side, plan['sections'][side], ledger_state(STUDY))
            except (KeyError, ValueError, TypeError, OSError) as e:
                bindings[side] = {'ordered_request_cache_vector_phase_byte_bindings_verified': False,
                    'failure_type': type(e).__name__}
                issues.append({'kind': 'main_side_original_byte_binding_failed', 'side': side, 'failure_type': type(e).__name__})
    result['side_original_byte_bindings'] = bindings
    closures = {}
    for phase, name in [('before', 'answer_phase_timing_before.json'), ('after', 'answer_phase_timing_after.json'),
                        ('score', 'score_phase_timing.json')]:
        p = STUDY / name
        closures[phase] = read(p).get('complete') is True if p.is_file() else False
    result['reported_phase_closures'] = closures
    result['main_completion_candidate'] = bool(gate_pass and main_report.exists()
        and result.get('planned_logical_rows') == 96 and logical == 96
        and type(target) is int and target <= 57 and ml['admitted'] == ml['published'] == target
        and ml['failed_or_unknown'] == ml['unsettled_intents'] == 0 and not answer_errors
        and not any(ml['event_binding'].values()) and all(closures.values())
        and len(bindings) == 2 and all(x['ordered_request_cache_vector_phase_byte_bindings_verified'] for x in bindings.values()))
    result['state'] = ('main_completion_candidate_requires_original_native_authentication_evidence'
        if result['main_completion_candidate'] else 'main_partial_or_not_started')
    if not GATE.exists(): result['state'] = 'gpu_gate_not_initialized'
    elif not gate_report.exists(): result['state'] = 'gpu_gate_partial_or_setup_failed'
    elif not gate_pass: result['state'] = 'gpu_readability_gate_not_passed_main_not_established'
    return result


def verify_explicit_study_identity(study_root, expected_study_version):
    if study_root is None: raise ValueError('explicit_registered_study_root_and_version_required')
    selected = owned(study_root)
    if selected.parent != CONTENT or STUDY_IDENTITIES.get(selected.name) != expected_study_version:
        raise ValueError('explicit_registered_study_root_and_version_required')
    selected_protocol = selected / 'study_protocol.json'
    if selected_protocol.is_file() and (read(selected_protocol).get('version') != expected_study_version
            or read(selected_protocol).get('study_root') != str(selected)):
        raise ValueError('actual_study_protocol_does_not_match_explicit_registered_identity')
    return selected, expected_study_version


def export(archive, receipt, expected_exporter_sha256, extras=(), *, study_root=None,
           expected_study_version=None, previous_v7_archive=None, previous_v7_receipt=None, extra_roots=()):
    global STUDY, STUDY_VERSION
    hosted(); require_sha(expected_exporter_sha256)
    if sha(__file__) != expected_exporter_sha256: raise ValueError('pre_frozen_exporter_code_changed')
    STUDY, STUDY_VERSION = verify_explicit_study_identity(study_root, expected_study_version)
    if previous_v7_archive is None:
        raise ValueError('explicit_preserved_actual_v7_archive_required')
    previous_v7_archive = owned(previous_v7_archive)
    if (previous_v7_archive.stat().st_size != PREVIOUS_V7_ARCHIVE_BYTES
            or sha(previous_v7_archive) != PREVIOUS_V7_ARCHIVE_SHA):
        raise ValueError('actual_previous_v7_archive_identity_changed')
    archive, receipt = owned(archive), owned(receipt)
    if archive.exists() or receipt.exists() or archive == receipt:
        raise FileExistsError('fresh_archive_and_receipt_paths_required_no_overwrite')
    started, wall_started = time.monotonic(), time.time()
    issues, busy = [], []
    with ExitStack() as stack, tempfile.TemporaryDirectory(prefix='cbt_gpu_registered_checkpoint_') as directory:
        for root in (GATE, STUDY):
            for name in ('.model_setup.lock', '.generation.lock'):
                lock = root / name
                if lock.is_file():
                    f = stack.enter_context(lock.open('rb'))
                    try: fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    except BlockingIOError: busy.append(str(lock))
        collector = Collector(Path(directory), issues)
        collector.pinned_extra_archives[str(previous_v7_archive)] = PREVIOUS_V7_ARCHIVE_SHA
        for root in (GATE, STUDY, CODE, MODEL, PYTHON, RECOVERY8, RECOVERY9, READINESS, NVIDIA_META, PUBLIC_META, PREPARATION):
            collector.tree(root)
        collector.tree(FAILED_V7)
        collector.add(FAILED_V7 / 'study_protocol.json', FAILED_V7_PROTOCOL_SHA, 'preserved_actual_zero_generation_v7_failure')
        collector.tree(FAILED_V8)
        collector.add(FAILED_V8 / 'study_protocol.json', FAILED_V8_PROTOCOL_SHA, 'preserved_actual_zero_generation_v8_failure')
        for name in ('cbt-gpu-v8-main-preparation-20261112-v1', 'cbt-gpu-v9-main-preparation-20261112-v1'):
            collector.tree(CONTENT / name)
        diagnostic = CONTENT / 'cbt-gpu-v8-binding-diagnostic-20261112-v1.json'
        if diagnostic.is_file(): collector.add(diagnostic, origin='preserved_actual_v8_binding_diagnostic')
        for root in extra_roots:
            p = owned(root)
            if not p.name.startswith(('cbt-gpu-v7-', 'cbt-gpu-v8-', 'cbt-qwen7b-gpu-')):
                raise ValueError('only_explicit_owned_gpu_project_extra_roots_allowed')
            collector.tree(p)
        collector.add(previous_v7_archive, PREVIOUS_V7_ARCHIVE_SHA, 'preserved_actual_v7_checkpoint_with_original_false_alias_issues')
        if previous_v7_receipt is not None:
            collector.add(owned(previous_v7_receipt), origin='preserved_actual_v7_checkpoint_receipt_unchanged')
        if CPU_CLOSURE.is_file(): collector.add(CPU_CLOSURE, CPU_CLOSURE_SHA, 'required_completed_cpu_closure_archive')
        else: issues.append({'kind': 'historical_cpu_closure_archive_missing', 'path': str(CPU_CLOSURE)})
        for side in ('before', 'after'):
            r = CONTENT / ('cbt-open-state-prepared-' + side + '-20261112-v2')
            for n in ('protocol.json', 'requests.jsonl'):
                collector.add(r / n, origin='original_legacy_source_reference_pair')
        for p in extras: collector.add(owned(p), origin='caller_explicit_owned_extra_file')
        for name in ('cbt-gpu-v7-main-preparation-source-20261112.py', 'cbt-gpu-v7-main-preparation-launch-20261112.log'):
            p = CONTENT / name
            if p.is_file(): collector.add(p, origin='observed_main_preparation_launcher')
        for name in ('download_qwen7b_gguf_gpu_v7_colab.py', 'setup_gguf_cuda_python_v7_colab.py',
                     'recover_gguf_cuda_dependencies_v8_colab.py', 'validate_gguf_cuda_driver_path_v9_colab.py'):
            p = CONTENT / name
            if p.is_file(): collector.add(p, origin='standalone_frozen_setup_source_basename')
        for pattern in ('cbt-gpu-v7-*.log', 'cbt-gpu-v8-*.log', 'cbt-gpu-v9-*.log',
                        'cbt-gpu-v7-*.py', 'cbt-gpu-v8-*.py', 'cbt-gpu-v9-*.py',
                        'cbt-*v7*launch*.json', 'cbt-*v8*launch*.json', 'cbt-*git*recovery*.json', 'cbt-cpu-v6-closure-restore-receipt-*.json'):
            for p in CONTENT.glob(pattern):
                if p.is_file(): collector.add(p, origin='explicit_project_log_or_recovery_pattern')
        collector.add(Path(__file__).resolve(), expected_exporter_sha256, 'this_pre_frozen_read_only_exporter')
        collector.close_references()
        collector.cpu_restore_descriptors()
        collector.finalize_restore_descriptors()
        state = observed_state(issues)
        completion_snapshot_closure(collector, state, issues)
        if busy: issues.append({'kind': 'active_writer_lock_checkpoint_is_partial', 'paths': busy})
        partial = Path(str(archive) + '.partial')
        if partial.exists(): raise FileExistsError('fresh_partial_archive_required')
        with zipfile.ZipFile(partial, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for e in sorted(collector.entries.values(), key=lambda x: x['archive_path']):
                path = Path(e['source_path'])
                if stat_info(path) != e['source_stat_after'] or sha(path) != e['source_sha256']:
                    issues.append({'kind': 'source_changed_after_collection_checkpoint_partial', 'path': str(path)})
                z.write(e['capture_source'], e['archive_path'])
            if busy or any(i['kind'].startswith('source_') for i in issues):
                state['main_completion_candidate'] = False
                state['archive_source_consistency'] = 'partial_active_or_changed_source_detected'
            else: state['archive_source_consistency'] = 'captured_source_byte_snapshots_rechecked'
            entries = [{k: v for k, v in e.items() if k != 'capture_source'} for e in collector.entries.values()]
            manifest = {'version': VERSION, 'exporter_code_sha256': expected_exporter_sha256,
                'explicit_study_root': str(STUDY), 'expected_study_version': STUDY_VERSION,
                'preserved_failed_v7_root': str(FAILED_V7),
                'preserved_failed_v8_root': str(FAILED_V8),
                'preserved_previous_v7_archive': {'path': str(previous_v7_archive), 'bytes': PREVIOUS_V7_ARCHIVE_BYTES,
                    'sha256': PREVIOUS_V7_ARCHIVE_SHA, 'original_manifest_and_false_alias_issues_modified': False},
                'capture_started_unix': wall_started, 'capture_ended_unix': time.time(),
                'source_clock_scope': 'actual hosted wall clock and source stat mtime/device/inode; not fabricated experiment timestamps',
                'files': sorted(entries, key=lambda x: x['archive_path']),
                'excluded_required_restore_artifacts': list(collector.descriptors.values()),
                'gpu_constructor_log_ranges': collector.log_ranges,
                'issues': issues, 'observed_state': state,
                'reference_closure_complete': not issues,
                'observed_state_scope': 'live reads during capture; each payload has a separate immutable byte snapshot and before/after source identity',
                'scope': 'private original metadata/code/journals/answers plus labelled SQLite snapshots; no native decoding or scoring rerun',
                'binary_restore_requires_verified_reacquisition': True,
                'new_model_generations': 0, 'new_paid_provider_calls': 0, 'secret_access': False,
                'automatic_resume_or_resend_authorized': False}
            b = (canonical(manifest) + '\n').encode(); z.writestr('MANIFEST.json', b)
        if partial.stat().st_size > MAX_ZIP:
            raise ValueError('private_checkpoint_exceeds_10MiB_limit_preserve_partial_no_overwrite')
        # Verify every captured payload, including SQLite backup bytes, before publish.
        with zipfile.ZipFile(partial) as z:
            if set(z.namelist()) != {e['archive_path'] for e in entries} | {'MANIFEST.json'}:
                raise ValueError('archive_member_inventory_changed')
            for e in entries:
                h, total = hashlib.sha256(), 0
                with z.open(e['archive_path']) as f:
                    for block in iter(lambda: f.read(1024 * 1024), b''): total += len(block); h.update(block)
                if total != e['bytes'] or h.hexdigest() != e['sha256']:
                    raise ValueError('archive_payload_identity_changed')
        os.link(partial, archive, follow_symlinks=False); partial.unlink()
        result = {'version': VERSION, 'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': sha(archive),
            'explicit_study_root': str(STUDY), 'expected_study_version': STUDY_VERSION,
            'preserved_failed_v7_root': str(FAILED_V7), 'preserved_previous_v7_archive_sha256': PREVIOUS_V7_ARCHIVE_SHA,
            'preserved_failed_v8_root': str(FAILED_V8),
            'manifest_sha256': hashlib.sha256(b).hexdigest(), 'artifact_files': len(entries),
            'excluded_restore_artifacts': len(collector.descriptors), 'issues': issues,
            'observed_state': state, 'wall_seconds': time.monotonic() - started,
            'reference_closure_complete': not issues,
            'new_model_generations': 0, 'new_paid_provider_calls': 0, 'secret_access': False,
            'automatic_resume_or_resend_authorized': False}
        with receipt.open('x', encoding='utf-8') as f:
            f.write(canonical(result) + '\n'); f.flush(); os.fsync(f.fileno())
        print(canonical(result), flush=True)
        return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, default=ARCHIVE)
    p.add_argument('--receipt', type=Path, default=RECEIPT)
    p.add_argument('--expected-exporter-sha256', required=True)
    p.add_argument('--study-root', type=Path, required=True)
    p.add_argument('--expected-study-version', required=True)
    p.add_argument('--previous-v7-archive', type=Path, required=True)
    p.add_argument('--previous-v7-receipt', type=Path)
    p.add_argument('--extra-root', type=Path, action='append', default=[])
    p.add_argument('--extra-file', type=Path, action='append', default=[])
    a = p.parse_args()
    export(a.archive, a.receipt, a.expected_exporter_sha256, a.extra_file,
        study_root=a.study_root, expected_study_version=a.expected_study_version,
        previous_v7_archive=a.previous_v7_archive, previous_v7_receipt=a.previous_v7_receipt, extra_roots=a.extra_root)
