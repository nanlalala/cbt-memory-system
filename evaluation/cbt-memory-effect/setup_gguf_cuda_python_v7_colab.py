"""One bounded isolated official CUDA-wheel setup on genuine hosted Colab.

No model files, model execution, Secret access, base-package upgrade, purchase,
automatic retry or resume. Publisher metadata is fixed before downloading.
"""
from __future__ import annotations

import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path('/content/cbt-qwen7b-gpu-python-20261112-v7')
META = Path('/content/cbt-gpu-readiness-20261112-v1')
RELEASE_SHA = 'e30f7cf9b0fa6d50736abea343a4a53d481e97bb873fce0d6f858ea56d402a42'
RELEASE_TAG = 'v0.3.36-cu124'
WHEEL = 'llama_cpp_python-0.3.36-py3-none-manylinux_2_35_x86_64.whl'
WHEEL_SHA = '3d53f94464d8d69028d993033d6120439c994108c04375a46205bd1935bc30ad'
WHEEL_BYTES = 1776573061
DISKCACHE_VERSION = '5.6.3'
DISKCACHE_URL = 'https://pypi.org/pypi/diskcache/5.6.3/json'
DEADLINE_SECONDS = 600
REQUEST_TIMEOUT_SECONDS = 40
BASE_DEPENDENCIES = {'numpy': '2.4.6', 'jinja2': '3.1.6', 'typing-extensions': '4.16.0'}


def sha(path, *, deadline=None):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            if deadline is not None:
                remaining(deadline)
            h.update(part)
    return h.hexdigest()


def remaining(deadline):
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError('absolute_gpu_package_setup_deadline_exhausted')
    return value


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def event(value):
    with (ROOT / 'setup_journal.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps({**value, 'at_unix': time.time()}, sort_keys=True, allow_nan=False) + '\n')
        f.flush(); os.fsync(f.fileno())


def require_colab():
    try:
        okay = (bool(os.environ.get('COLAB_RELEASE_TAG')) and Path('/content').is_dir()
                and importlib.util.find_spec('google.colab') is not None)
    except (ImportError, ValueError, AttributeError):
        okay = False
    if not okay:
        raise RuntimeError('genuine_hosted_colab_required_before_io')


def validate_release(raw):
    if hashlib.sha256(raw).hexdigest() != RELEASE_SHA:
        raise ValueError('official_cuda_release_metadata_changed')
    return validate_release_value(json.loads(raw))


def validate_release_value(data):
    selected = [x for x in data['assets'] if x.get('name') == WHEEL]
    if len(selected) != 1:
        raise ValueError('one_pinned_official_cuda_wheel_required')
    asset = selected[0]
    if (data.get('tag_name') != RELEASE_TAG or asset.get('size') != WHEEL_BYTES
            or asset.get('digest') != 'sha256:' + WHEEL_SHA
            or asset.get('browser_download_url') !=
               'https://github.com/abetlen/llama-cpp-python/releases/download/' + RELEASE_TAG + '/' + WHEEL):
        raise ValueError('pinned_official_cuda_wheel_identity_changed')
    return asset


def diskcache_wheel(body):
    package = json.loads(body)
    if package.get('info', {}).get('name', '').casefold() != 'diskcache' or package['info']['version'] != DISKCACHE_VERSION:
        raise ValueError('exact_diskcache_release_required')
    candidates = [x for x in package['urls'] if x.get('packagetype') == 'bdist_wheel'
                  and x.get('filename', '').endswith('py3-none-any.whl') and not x.get('yanked')]
    if len(candidates) != 1:
        raise ValueError('one_universal_diskcache_wheel_required')
    x = candidates[0]
    if (not x['url'].startswith('https://files.pythonhosted.org/packages/')
            or Path(x['filename']).name != x['filename']
            or type(x['size']) is not int or not 0 < x['size'] < 5_000_000
            or not re.fullmatch(r'[0-9a-f]{64}', x.get('digests', {}).get('sha256', ''))):
        raise ValueError('bounded_official_pypi_wheel_required')
    return {'filename': x['filename'], 'size': x['size'],
            'sha256': x['digests']['sha256'], 'url': x['url']}


def download(item, deadline):
    """Exactly one streamed GET, with no partial reuse or URL-bearing errors."""
    path = ROOT / item['filename']
    if Path(item['filename']).name != item['filename'] or path.exists() or path.is_symlink():
        raise FileExistsError('fresh_owned_wheel_path_required_before_network')
    started, size = time.monotonic(), 0
    event({'event': 'wheel_download_intent', 'filename': item['filename'],
           'bytes': item['size'], 'sha256': item['sha256'], 'source_url': item['url']})
    try:
        h = hashlib.sha256()
        with urllib.request.urlopen(item['url'], timeout=min(REQUEST_TIMEOUT_SECONDS, remaining(deadline))) as response, path.open('xb') as f:
            if response.status != 200:
                raise ValueError('wheel_http_200_required')
            while True:
                remaining(deadline)
                part = response.read(1024 * 1024)
                if not part:
                    break
                size += len(part)
                if size > item['size']:
                    raise ValueError('publisher_wheel_size_exceeded')
                f.write(part); h.update(part)
            f.flush(); os.fsync(f.fileno())
        if (size != item['size'] or h.hexdigest() != item['sha256']
                or sha(path, deadline=deadline) != item['sha256']):
            raise ValueError('publisher_wheel_hash_or_size_mismatch')
        event({'event': 'wheel_download_verified', 'filename': item['filename'],
               'bytes_received': size, 'sha256': h.hexdigest(), 'seconds': time.monotonic() - started})
        return path
    except BaseException as e:
        event({'event': 'wheel_download_failed_or_unknown', 'filename': item['filename'],
               'bytes_received': size, 'error_type': type(e).__name__,
               'seconds': time.monotonic() - started, 'automatic_retry': False})
        raise


def gpu_query(deadline):
    command = ['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used', '--format=csv,noheader']
    p = subprocess.run(command, capture_output=True, text=True, timeout=min(15, remaining(deadline)), check=False)
    if p.returncode or 'T4' not in p.stdout:
        raise RuntimeError('actual_t4_driver_query_required')
    return p.stdout.strip()


def main():
    require_colab()
    if ROOT.exists() or ROOT.is_symlink():
        raise FileExistsError('new_gpu_python_target_required_no_retry')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('official_linux_x86_64_cuda_wheel_required')
    libc_name, libc_version = platform.libc_ver()
    if libc_name != 'glibc' or tuple(map(int, libc_version.split('.')[:2])) < (2, 35):
        raise RuntimeError('official_cuda_wheel_requires_glibc_2_35_or_newer')
    raw = (META / 'official_cuda_release.json').read_bytes()
    asset = validate_release(raw)
    ROOT.mkdir(mode=0o700)
    started = time.monotonic()
    deadline = started + DEADLINE_SECONDS
    phase, installed = 'hardware_query', None
    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('absolute_gpu_package_setup_deadline_exhausted')))
    signal.setitimer(signal.ITIMER_REAL, DEADLINE_SECONDS)
    try:
        hardware = gpu_query(deadline)
        event({'event': 'gpu_hardware_observed', 'gpu_query': hardware})
        for name, version in BASE_DEPENDENCIES.items():
            if importlib.metadata.version(name) != version:
                raise ValueError('verified_gpu_base_dependency_version_changed')
        phase = 'diskcache_metadata'
        event({'event': 'diskcache_metadata_intent', 'url': DISKCACHE_URL})
        with urllib.request.urlopen(DISKCACHE_URL, timeout=min(20, remaining(deadline))) as response:
            body = response.read(1048577)
            if response.status != 200 or len(body) > 1048576:
                raise ValueError('bounded_pypi_metadata_required')
        with (ROOT / 'diskcache_pypi_metadata.json').open('xb') as f:
            f.write(body); f.flush(); os.fsync(f.fileno())
        dependency = diskcache_wheel(body)
        event({'event': 'diskcache_metadata_frozen', 'bytes': len(body),
               'sha256': hashlib.sha256(body).hexdigest()})
        wheels = [{'filename': WHEEL, 'size': WHEEL_BYTES, 'sha256': WHEEL_SHA,
                   'url': asset['browser_download_url']}, dependency]
        target = ROOT / 'site-packages'
        protocol = {'version': 'isolated-official-cuda-python-setup-v7',
            'code_sha256': sha(__file__), 'official_release_metadata_sha256': RELEASE_SHA,
            'official_release_tag': RELEASE_TAG,
            'diskcache_metadata_sha256': hashlib.sha256(body).hexdigest(),
            'diskcache_version': DISKCACHE_VERSION, 'wheels': wheels,
            'target': str(target), 'install_options': ['--no-deps', '--no-index'],
            'absolute_deadline_seconds': DEADLINE_SECONDS, 'request_timeout_seconds': REQUEST_TIMEOUT_SECONDS,
            'actual_gpu_query_before_setup': hardware, 'platform': platform.platform(),
            'base_dependency_versions': BASE_DEPENDENCIES,
            'base_environment_modified': False, 'model_generations': 0,
            'paid_provider_calls': 0, 'secret_access': False, 'automatic_retry': False}
        write(ROOT / 'setup_protocol.json', protocol)
        phase = 'publisher_wheel_downloads'
        paths = [download(item, deadline) for item in wheels]
        command = [sys.executable, '-m', 'pip', '--disable-pip-version-check', 'install',
                   '--no-deps', '--no-index', '--target', str(target), *map(str, paths)]
        phase = 'isolated_pip_install'
        event({'event': 'isolated_pip_install_intent', 'command': command})
        with (ROOT / 'install.log').open('xb') as log:
            child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=min(180, remaining(deadline)), check=False)
        event({'event': 'isolated_pip_install_returned', 'exit_code': child.returncode})
        if child.returncode:
            raise RuntimeError('isolated_cuda_package_install_failed_no_retry')
        phase = 'isolated_import_and_compiled_cuda_probe'
        probe = ('import importlib.metadata,json,sys;sys.path.insert(0,' + repr(str(target)) + ');'
                 'import llama_cpp,diskcache;'
                 'print(json.dumps({"llama_cpp_version":importlib.metadata.version("llama-cpp-python"),'
                 '"diskcache_version":importlib.metadata.version("diskcache"),'
                 '"llama_cpp_path":llama_cpp.__file__,"diskcache_path":diskcache.__file__,'
                 '"compiled_gpu_offload_support":bool(llama_cpp.llama_supports_gpu_offload()),'
                 '"dependencies":[{"name":n,"version":importlib.metadata.version(n)} '
                 'for n in ("numpy","jinja2","typing-extensions","diskcache")]}))')
        event({'event': 'isolated_import_probe_intent', 'model_generations': 0})
        result = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True,
                                timeout=min(30, remaining(deadline)), check=False)
        # Import stderr may contain diagnostics; preserve its bytes locally, do
        # not copy it into the externally printed failure message.
        with (ROOT / 'import_probe_stderr.log').open('xb') as f:
            f.write(result.stderr.encode())
        if result.returncode:
            raise RuntimeError('isolated_cuda_import_failed_no_retry')
        installed = json.loads(result.stdout)
        if (installed['llama_cpp_version'] != '0.3.36'
                or installed['diskcache_version'] != DISKCACHE_VERSION
                or not Path(installed['llama_cpp_path']).is_relative_to(target)
                or not Path(installed['diskcache_path']).is_relative_to(target)
                or installed.get('compiled_gpu_offload_support') is not True):
            raise ValueError('isolated_cuda_distribution_identity_or_compiled_support_changed')
        if {x['name']: x['version'] for x in installed['dependencies']} != {
                **BASE_DEPENDENCIES, 'diskcache': DISKCACHE_VERSION}:
            raise ValueError('base_dependencies_changed_during_isolated_setup')
        post_hardware = gpu_query(deadline)
        receipt = {'stage': 'isolated_cuda_python_ready_no_model_loaded',
            'protocol_sha256': sha(ROOT / 'setup_protocol.json'), **installed,
            'setup_seconds': time.monotonic() - started,
            'gpu_query_before_setup': hardware, 'gpu_query_after_import': post_hardware,
            'compiled_support_is_not_model_offload_evidence': True,
            'official_cuda_wheel_sha256': WHEEL_SHA, 'official_cuda_wheel_bytes': WHEEL_BYTES,
            'model_generations': 0, 'paid_provider_calls': 0,
            'secret_access': False, 'base_environment_modified': False, 'automatic_retry': False}
        write(ROOT / 'setup_receipt.json', receipt)
        event({'event': 'isolated_cuda_setup_completed', 'receipt_sha256': sha(ROOT / 'setup_receipt.json')})
        print(json.dumps(receipt, sort_keys=True), flush=True)
    except BaseException as error:
        event({'event': 'isolated_cuda_setup_failed_or_unknown', 'phase': phase,
               'failure_type': type(error).__name__, 'elapsed_seconds': time.monotonic() - started,
               'automatic_retry': False})
        observed = [{'filename': item.name, 'bytes': item.stat().st_size}
                    for item in ROOT.glob('*.whl') if item.is_file()]
        receipt = {'stage': 'isolated_cuda_setup_stopped_failed_or_unknown', 'phase': phase,
            'failure_type': type(error).__name__, 'elapsed_seconds': time.monotonic() - started,
            'observed_wheel_files': observed, 'automatic_retry': False,
            'model_generations': 0, 'paid_provider_calls': 0, 'secret_access': False,
            'base_environment_modified': False}
        write(ROOT / 'setup_stopped_receipt.json', receipt)
        print(json.dumps(receipt, sort_keys=True), flush=True)
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    return 0


if __name__ == '__main__':
    try:
        result = main()
    except BaseException as error:
        print(json.dumps({'stage': 'cuda_setup_entry_stopped_before_artifact_or_receipt',
                          'failure_type': type(error).__name__, 'automatic_retry': False}), flush=True)
        result = 1
    sys.exit(result or 0)
