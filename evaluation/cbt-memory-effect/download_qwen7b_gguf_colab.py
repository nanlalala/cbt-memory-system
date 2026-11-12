"""One bounded, no-retry download of two pinned official Qwen GGUF shards.

Actual transport is restricted to genuine hosted Colab. This entry performs no
package installation, inference, credential lookup, or data-corpus operation.
The inherited user authorization is for a stronger public reader in a separate
cloud setup; neither old model/run identities nor generation caps are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import sys
import time
import urllib.request

VERSION = 'official-qwen7b-q4-k-m-bounded-transport-v1'
ROOT = Path('/content/cbt-qwen7b-reader-setup-20261112-v1')
METADATA = Path('/content/cbt-source-and-reader-metadata-20261112-v1/qwen7b_model.json')
METADATA_SHA256 = '4beda630a9a3b5c4f3bf0a0c4fdda517639d4dc359ed1c6c947165802290d8aa'
REPO = 'Qwen/Qwen2.5-7B-Instruct-GGUF'
REVISION = 'bb5d59e06d9551d752d08b292a50eb208b07ab1f'
BLOCK_BYTES = 16 * 1024 * 1024
PROBE_BYTES = 1024 * 1024
DEADLINE_SECONDS = 900
REQUEST_TIMEOUT_SECONDS = 30
FILES = (
    {'name': 'qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf',
     'bytes': 3993201344,
     'sha256': 'dfce12e3862a5283ccfb88221b48480e58745165de856439950d0f22590580db'},
    {'name': 'qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf',
     'bytes': 689872288,
     'sha256': '539cf93f78e887edea1c04e2d7d8cdaca9d01dae9c9025bcb8accbe29df3d72a'},
)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path, *, deadline=None):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(PROBE_BYTES), b''):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('transport_deadline_exhausted_during_full_integrity_check')
            checksum.update(block)
    return checksum.hexdigest()


def require_colab():
    # Check before any metadata or model-file I/O. Tests exercise pure helpers.
    try:
        okay = (bool(os.environ.get('COLAB_RELEASE_TAG')) and Path('/content').is_dir()
                and importlib.util.find_spec('google.colab') is not None)
    except (ImportError, ValueError, AttributeError):
        okay = False
    if not okay:
        raise RuntimeError('genuine_hosted_colab_required_before_download_artifact_io')
    if Path('/content').resolve() != Path('/content'):
        raise RuntimeError('ordinary_hosted_content_directory_required')


def fixed_url(item):
    if item not in FILES or Path(item['name']).name != item['name']:
        raise ValueError('only_two_frozen_official_shards_allowed')
    return f'https://huggingface.co/{REPO}/resolve/{REVISION}/{item["name"]}'


def ranges(size):
    if type(size) is not int or size < 1:
        raise ValueError('positive_integer_size_required')
    return tuple((start, min(start + BLOCK_BYTES, size) - 1)
                 for start in range(0, size, BLOCK_BYTES))


def validate_metadata(raw):
    if hashlib.sha256(raw).hexdigest() != METADATA_SHA256:
        raise ValueError('frozen_official_model_metadata_sha256_changed')
    return validate_metadata_value(json.loads(raw))


def validate_metadata_value(value):
    """Pure official API metadata validation; no network or model operation."""
    if (type(value) is not dict or value.get('id', value.get('modelId')) != REPO
            or value.get('sha') != REVISION):
        raise ValueError('official_repo_revision_identity_changed')
    siblings = value.get('siblings')
    if type(siblings) is not list:
        raise ValueError('official_expanded_file_metadata_required')
    catalog = {}
    for row in siblings:
        if type(row) is not dict or type(row.get('rfilename')) is not str:
            raise ValueError('invalid_official_file_entry')
        if row['rfilename'] in catalog:
            raise ValueError('duplicate_official_filename')
        catalog[row['rfilename']] = row
    for item in FILES:
        row = catalog.get(item['name'])
        if not row:
            raise ValueError('both_pinned_shards_required_in_official_metadata')
        lfs = row.get('lfs')
        if type(lfs) is not dict:
            raise ValueError('official_shard_lfs_digest_required')
        checksum = lfs.get('sha256', lfs.get('oid'))
        if type(checksum) is str and checksum.startswith('sha256:'):
            checksum = checksum.removeprefix('sha256:')
        if (row.get('size', lfs.get('size')) != item['bytes']
                or lfs.get('size') != item['bytes'] or checksum != item['sha256']):
            raise ValueError('official_shard_size_or_sha256_changed')
    return {'repo': REPO, 'revision': REVISION, 'files': [dict(f) for f in FILES]}


def validate_headers(status, headers, *, start, end, total):
    """Accept only an exact, unencoded partial response for this fixed file."""
    if status != 206:
        raise ValueError('exact_range_requires_http_206')
    normalized = {str(k).lower(): str(v).strip() for k, v in headers.items()}
    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', normalized.get('content-range', ''))
    if not match or tuple(map(int, match.groups())) != (start, end, total):
        raise ValueError('content_range_does_not_match_exact_request_and_frozen_total')
    if normalized.get('content-length') != str(end - start + 1):
        raise ValueError('content_length_does_not_match_exact_range')
    if normalized.get('content-encoding', 'identity').lower() != 'identity':
        raise ValueError('encoded_range_transport_prohibited')
    return {'status': status, 'content_range': normalized['content-range'],
            'content_length': int(normalized['content-length']), 'content_encoding': 'identity'}


def new_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        f.write(canonical(value) + '\n')
        f.flush(); os.fsync(f.fileno())


class Journal:
    def __init__(self, path):
        self.path = Path(path)
        self.sequence = 0
        self.previous = None
        # Exclusive claim: a previous partial transport is never resumed.
        with self.path.open('x', encoding='utf-8') as f:
            f.flush(); os.fsync(f.fileno())

    def append(self, value):
        item = {**value, 'sequence': self.sequence + 1, 'previous_record_sha256': self.previous}
        checksum = digest(item)
        with self.path.open('a', encoding='utf-8') as f:
            f.write(canonical({**item, 'record_sha256': checksum}) + '\n')
            f.flush(); os.fsync(f.fileno())
        self.sequence += 1
        self.previous = checksum
        return checksum


class HttpsRedirectOnly(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Signed CDN URLs may be followed, but never printed or exported.
        from urllib.parse import urlsplit
        parsed = urlsplit(newurl)
        if parsed.scheme != 'https' or parsed.username is not None or parsed.password is not None:
            raise ValueError('only_credential_free_https_vendor_redirects_allowed')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def initialize():
    require_colab()
    if ROOT.exists() or ROOT.is_symlink():
        raise FileExistsError('fresh_setup_root_required_no_resume')
    raw = METADATA.read_bytes()
    catalog = validate_metadata(raw)
    ROOT.mkdir(mode=0o700)
    with (ROOT / 'official_model_metadata.json').open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    protocol = {'version': VERSION, 'root': str(ROOT), 'model': catalog,
        'original_metadata_path': str(METADATA), 'original_metadata_sha256': METADATA_SHA256,
        'downloader_code_sha256': file_sha(__file__),
        'sources': [fixed_url(f) for f in FILES],
        'maximum_logical_get_attempts': 1 + sum(len(ranges(f['bytes'])) for f in FILES),
        'logical_attempt_definition': 'one urllib opener call; ordinary vendor redirects are inside that call',
        'block_bytes': BLOCK_BYTES, 'probe_bytes': PROBE_BYTES,
        'preflight': 'one exact first-shard 1MiB range; full transfer starts from byte zero only after probe success',
        'preflight_is_separate_diagnostic_not_a_retry_or_partial_resume': True,
        'transport_deadline_seconds_including_probe': DEADLINE_SECONDS,
        'per_request_socket_timeout_seconds': REQUEST_TIMEOUT_SECONDS,
        'automatic_retry': False, 'automatic_resume': False,
        'overwrite_existing_partial_or_final': False,
        'publish_policy': 'both complete shard sizes and full SHA256 verify before either final name; no-overwrite link then unlink',
        'new_model_inference_attempts': 0, 'package_installs': 0,
        'new_paid_provider_calls': 0, 'secret_access': False,
        'license': 'Apache-2.0 reported by official model metadata; copied metadata is authoritative evidence',
        'authorization_basis': 'user-authorized bounded stronger public-reader setup on genuine hosted Colab',
        'created_unix': time.time()}
    new_json(ROOT / 'download_protocol.json', protocol)
    print(canonical({'stage': 'download_protocol_frozen_no_network',
        'protocol_sha256': file_sha(ROOT / 'download_protocol.json'),
        'maximum_logical_get_attempts': protocol['maximum_logical_get_attempts'],
        'total_full_shard_bytes': sum(f['bytes'] for f in FILES), 'inference_attempts': 0}), flush=True)


def check_protocol(protocol, expected_protocol_sha):
    if (file_sha(ROOT / 'download_protocol.json') != expected_protocol_sha
            or protocol.get('version') != VERSION
            or protocol.get('root') != str(ROOT)
            or protocol.get('downloader_code_sha256') != file_sha(__file__)
            or file_sha(ROOT / 'official_model_metadata.json') != METADATA_SHA256
            or file_sha(METADATA) != METADATA_SHA256
            or protocol.get('model') != {'repo': REPO, 'revision': REVISION, 'files': list(FILES)}
            or protocol.get('sources') != [fixed_url(f) for f in FILES]
            or protocol.get('maximum_logical_get_attempts') != 1 + sum(len(ranges(f['bytes'])) for f in FILES)
            or protocol.get('block_bytes') != BLOCK_BYTES
            or protocol.get('probe_bytes') != PROBE_BYTES
            or protocol.get('transport_deadline_seconds_including_probe') != DEADLINE_SECONDS
            or protocol.get('per_request_socket_timeout_seconds') != REQUEST_TIMEOUT_SECONDS):
        raise ValueError('frozen_transport_protocol_or_source_changed')


def download():
    require_colab()
    protocol_path = ROOT / 'download_protocol.json'
    protocol_sha = file_sha(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    check_protocol(protocol, protocol_sha)
    new_json(ROOT / 'download_run_claim.json', {'pid': os.getpid(), 'started_unix': time.time(),
        'protocol_sha256': protocol_sha, 'automatic_retry': False, 'automatic_resume': False})
    journal = Journal(ROOT / 'download_journal.jsonl')
    started = time.monotonic()
    deadline = started + DEADLINE_SECONDS
    def deadline_alarm(signum, frame):
        raise TimeoutError('absolute_transport_deadline_exhausted')
    previous_alarm_handler = signal.signal(signal.SIGALRM, deadline_alarm)
    signal.setitimer(signal.ITIMER_REAL, DEADLINE_SECONDS)
    opener = urllib.request.build_opener(HttpsRedirectOnly())
    attempts = 0
    received_total = 0
    probe_seconds = None
    partial_paths = []

    def get_exact(item, start, end, *, purpose, output):
        nonlocal attempts, received_total
        check_protocol(protocol, protocol_sha)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('transport_deadline_exhausted_no_new_request')
        if attempts >= protocol['maximum_logical_get_attempts']:
            raise RuntimeError('fixed_logical_transport_attempt_cap_reached')
        attempts += 1
        attempt = attempts
        url = fixed_url(item)
        intent = {'event': 'range_intent', 'attempt': attempt, 'purpose': purpose,
            'file': item['name'], 'original_fixed_url': url, 'start': start, 'end': end,
            'total': item['bytes'], 'requested_bytes': end - start + 1,
            'protocol_sha256': protocol_sha, 'at_unix': time.time()}
        journal.append(intent)  # Durable admission always precedes the request.
        count = 0
        checksum = hashlib.sha256()
        request_started = time.monotonic()
        checked_headers = None
        try:
            request = urllib.request.Request(url, headers={
                'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity',
                'User-Agent': 'Capstone-pinned-public-model-transport/1.0'})
            with opener.open(request, timeout=min(REQUEST_TIMEOUT_SECONDS, remaining)) as response:
                checked_headers = validate_headers(response.status, response.headers,
                    start=start, end=end, total=item['bytes'])
                expected = end - start + 1
                while count < expected:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('total_transport_deadline_exceeded')
                    chunk = response.read(min(PROBE_BYTES, expected - count))
                    if not chunk:
                        raise EOFError('short_range_body_no_retry')
                    count += len(chunk); received_total += len(chunk)
                    checksum.update(chunk)
                    output.write(chunk)
                if time.monotonic() >= deadline:
                    raise TimeoutError('total_transport_deadline_exceeded_before_eof_check')
                extra = response.read(1)
                if extra:
                    count += len(extra); received_total += len(extra)
                    raise ValueError('oversized_range_body_no_retry')
            output.flush(); os.fsync(output.fileno())
            journal.append({'event': 'range_outcome', 'attempt': attempt, 'status': 'success',
                'purpose': purpose, 'file': item['name'], 'start': start, 'end': end,
                'bytes_received': count, 'body_sha256': checksum.hexdigest(),
                'headers': checked_headers, 'elapsed_seconds': time.monotonic() - request_started,
                'at_unix': time.time()})
        except BaseException as error:
            output.flush(); os.fsync(output.fileno())
            # Exception text may contain signed redirect URLs; preserve only type.
            journal.append({'event': 'range_outcome', 'attempt': attempt,
                'status': 'failed_or_unknown', 'purpose': purpose, 'file': item['name'],
                'start': start, 'end': end, 'bytes_received': count,
                'failure_type': type(error).__name__, 'headers': checked_headers,
                'elapsed_seconds': time.monotonic() - request_started,
                'automatic_retry': False, 'automatic_resume': False, 'at_unix': time.time()})
            raise

    try:
        # Diagnostic bytes are kept separately; no unverified fragment is reused.
        probe_started = time.monotonic()
        with (ROOT / 'first_shard_exact_1MiB_probe.bin').open('xb') as output:
            get_exact(FILES[0], 0, PROBE_BYTES - 1, purpose='exact_range_preflight', output=output)
        probe_seconds = time.monotonic() - probe_started
        new_json(ROOT / 'range_preflight_receipt.json', {'passed': True,
            'bytes': PROBE_BYTES, 'sha256': file_sha(ROOT / 'first_shard_exact_1MiB_probe.bin'),
            'elapsed_seconds': probe_seconds, 'inference_attempts': 0})
        verified = []
        full_download_started = time.monotonic()
        for item in FILES:
            partial = ROOT / (item['name'] + '.partial')
            final = ROOT / item['name']
            if final.exists() or final.is_symlink():
                raise FileExistsError('final_shard_already_exists_no_overwrite')
            partial_paths.append(partial)
            with partial.open('xb') as output:
                for start, end in ranges(item['bytes']):
                    if output.tell() != start:
                        raise ValueError('partial_file_position_must_equal_contiguous_range_start')
                    get_exact(item, start, end, purpose='full_shard_transfer', output=output)
                output.flush(); os.fsync(output.fileno())
            verification_started = time.monotonic()
            actual_size = partial.stat().st_size
            actual_sha = file_sha(partial, deadline=deadline)
            if actual_size != item['bytes'] or actual_sha != item['sha256']:
                raise ValueError('full_shard_size_or_sha256_verification_failed')
            verified.append({**item, 'verified_bytes': actual_size, 'verified_sha256': actual_sha,
                'verification_seconds': time.monotonic() - verification_started})
            journal.append({'event': 'full_shard_verified', 'file': item['name'],
                'bytes': actual_size, 'sha256': actual_sha, 'at_unix': time.time()})
        if len(verified) != 2:
            raise RuntimeError('both_complete_shards_required_before_publish')
        if time.monotonic() >= deadline:
            raise TimeoutError('transport_deadline_exhausted_before_publish')
        # No overwrite: hard-link each verified owned partial then unlink it.
        # The ready receipt is published only after both final names exist.
        for item, partial in zip(FILES, partial_paths):
            final = ROOT / item['name']
            os.link(partial, final, follow_symlinks=False)
            partial.unlink()
        directory_fd = os.open(ROOT, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        receipt = {'version': VERSION, 'stage': 'both_pinned_shards_downloaded_and_verified',
            'protocol_sha256': protocol_sha, 'repo': REPO, 'revision': REVISION,
            'files': verified, 'logical_get_attempts': attempts,
            'expected_logical_get_attempts': protocol['maximum_logical_get_attempts'],
            'full_shard_bytes': sum(f['bytes'] for f in FILES),
            'diagnostic_bytes': PROBE_BYTES, 'observed_body_bytes_received': received_total,
            'exact_range_preflight_seconds': probe_seconds,
            'full_download_and_verification_seconds': time.monotonic() - full_download_started,
            'total_transport_seconds_including_probe': time.monotonic() - started,
            'new_paid_provider_calls': 0, 'secret_access': False, 'package_installs': 0,
            'new_model_inference_attempts': 0, 'model_loaded': False,
            'automatic_retry': False, 'automatic_resume': False,
            'journal_last_record_sha256': journal.previous, 'at_unix': time.time()}
        new_json(ROOT / 'download_complete_receipt.json', receipt)
        print(canonical(receipt), flush=True)
    except BaseException as error:
        receipt = {'version': VERSION, 'stage': 'transport_stopped_failed_or_unknown',
            'failure_type': type(error).__name__, 'logical_get_attempts': attempts,
            'observed_body_bytes_received': received_total,
            'elapsed_seconds_including_probe': time.monotonic() - started,
            'exact_range_preflight_seconds': probe_seconds,
            'partials': [{'name': p.name, 'observed_bytes': p.stat().st_size}
                         for p in partial_paths if p.is_file()],
            'automatic_retry': False, 'automatic_resume': False,
            'new_paid_provider_calls': 0, 'secret_access': False, 'new_model_inference_attempts': 0,
            'model_loaded': False, 'journal_last_record_sha256': journal.previous,
            'at_unix': time.time()}
        new_json(ROOT / 'download_stopped_receipt.json', receipt)
        print(canonical(receipt), flush=True)
        # Do not emit signed URL-bearing exception messages or a traceback.
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm_handler)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('initialize', 'download'))
    command = parser.parse_args().command
    try:
        result = initialize() if command == 'initialize' else download()
    except BaseException as error:
        print(canonical({'stage': 'entry_stopped_before_transport_or_receipt',
                         'failure_type': type(error).__name__, 'automatic_retry': False}), flush=True)
        result = 1
    sys.exit(result or 0)
