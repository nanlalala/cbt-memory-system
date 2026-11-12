"""Install two publisher-hashed CPU wheels into a new hosted-only target.

No base-environment upgrade, model download, model execution or credential use.
One bounded attempt per download/install; existing directories block resends.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path('/content/cbt-qwen7b-python-20261112-v1')
META = Path('/content/cbt-source-and-reader-metadata-20261112-v1')
RELEASE_SHA = 'b8aeb5f5a949326afb3f9e48a9fa877505b338b5b8648acf7c1f124b923a760b'
WHEEL = 'llama_cpp_python-0.3.36-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl'
WHEEL_SHA = 'c8a3e98093e950d7a770f794842e36f2db82b2933e8762ed7b5ffd435011e8e4'
WHEEL_BYTES = 27077498


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def event(value):
    with (ROOT / 'setup_journal.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps({**value, 'at_unix': time.time()}, sort_keys=True) + '\n')
        f.flush(); os.fsync(f.fileno())


def download(item):
    path = ROOT / item['filename']
    started = time.monotonic()
    event({'event': 'wheel_download_intent', 'filename': item['filename'],
           'bytes': item['size'], 'sha256': item['sha256'], 'source_url': item['url']})
    try:
        h, size = hashlib.sha256(), 0
        with urllib.request.urlopen(item['url'], timeout=40) as response, path.open('xb') as f:
            if response.status != 200:
                raise ValueError('wheel_http_200_required')
            while True:
                part = response.read(1024 * 1024)
                if not part:
                    break
                size += len(part)
                if size > item['size']:
                    raise ValueError('publisher_wheel_size_exceeded')
                f.write(part); h.update(part)
            f.flush(); os.fsync(f.fileno())
        if size != item['size'] or h.hexdigest() != item['sha256'] or sha(path) != item['sha256']:
            raise ValueError('publisher_wheel_hash_or_size_mismatch')
        event({'event': 'wheel_download_verified', 'filename': item['filename'],
               'bytes': size, 'sha256': h.hexdigest(), 'seconds': time.monotonic() - started})
        return path
    except BaseException as e:
        event({'event': 'wheel_download_failed_no_retry', 'filename': item['filename'],
               'error_type': type(e).__name__, 'seconds': time.monotonic() - started})
        raise


def main():
    if (not os.environ.get('COLAB_RELEASE_TAG') or not Path('/content').is_dir()
            or importlib.util.find_spec('google.colab') is None):
        raise RuntimeError('genuine_hosted_colab_required_before_io')
    if ROOT.exists():
        raise FileExistsError('new_python_target_required_no_automatic_retry')
    release = META / 'llama_cpp_python_v0.3.36_release.json'
    if sha(release) != RELEASE_SHA:
        raise ValueError('official_release_metadata_changed')
    data = json.loads(release.read_text())
    asset = next(x for x in data['assets'] if x['name'] == WHEEL)
    if (data['tag_name'] != 'v0.3.36' or asset['size'] != WHEEL_BYTES
            or asset['digest'] != 'sha256:' + WHEEL_SHA
            or asset['browser_download_url'] != 'https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.36/' + WHEEL):
        raise ValueError('pinned_official_cpu_wheel_identity_changed')
    ROOT.mkdir(mode=0o700)
    started = time.monotonic()
    event({'event': 'diskcache_metadata_intent', 'url': 'https://pypi.org/pypi/diskcache/json'})
    with urllib.request.urlopen('https://pypi.org/pypi/diskcache/json', timeout=20) as response:
        body = response.read(1048577)
        if response.status != 200 or len(body) > 1048576:
            raise ValueError('bounded_pypi_metadata_required')
    with (ROOT / 'diskcache_pypi_metadata.json').open('xb') as f:
        f.write(body)
    package = json.loads(body)
    candidates = [x for x in package['urls'] if x['packagetype'] == 'bdist_wheel'
                  and x['filename'].endswith('py3-none-any.whl') and not x.get('yanked')]
    if len(candidates) != 1:
        raise ValueError('one_universal_diskcache_wheel_required')
    dependency = candidates[0]
    if (not dependency['url'].startswith('https://files.pythonhosted.org/packages/')
            or Path(dependency['filename']).name != dependency['filename']
            or not 0 < dependency['size'] < 5_000_000):
        raise ValueError('bounded_official_pypi_wheel_required')
    wheels = [
        {'filename': WHEEL, 'size': WHEEL_BYTES, 'sha256': WHEEL_SHA,
         'url': asset['browser_download_url']},
        {'filename': dependency['filename'], 'size': dependency['size'],
         'sha256': dependency['digests']['sha256'], 'url': dependency['url']}]
    target = ROOT / 'site-packages'
    protocol = {'version': 'isolated-official-cpu-python-setup-v5',
        'code_sha256': sha(__file__), 'official_release_metadata_sha256': RELEASE_SHA,
        'diskcache_metadata_sha256': hashlib.sha256(body).hexdigest(),
        'diskcache_version': package['info']['version'], 'wheels': wheels,
        'target': str(target), 'install_options': ['--no-deps', '--no-index'],
        'base_environment_modified': False, 'model_generations': 0,
        'paid_provider_calls': 0, 'secret_access': False, 'automatic_retry': False}
    write(ROOT / 'setup_protocol.json', protocol)
    paths = [download(item) for item in wheels]
    command = [sys.executable, '-m', 'pip', '--disable-pip-version-check', 'install',
               '--no-deps', '--no-index', '--target', str(target), *map(str, paths)]
    event({'event': 'isolated_pip_install_intent', 'command': command})
    with (ROOT / 'install.log').open('xb') as log:
        child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=120, check=False)
    event({'event': 'isolated_pip_install_returned', 'exit_code': child.returncode})
    if child.returncode:
        raise RuntimeError('isolated_package_install_failed_no_retry')
    probe = ('import importlib.metadata,json,sys;sys.path.insert(0,' + repr(str(target)) + ');'
             'import llama_cpp,diskcache;'
             'print(json.dumps({"llama_cpp_version":importlib.metadata.version("llama-cpp-python"),'
             '"diskcache_version":importlib.metadata.version("diskcache"),'
             '"llama_cpp_path":llama_cpp.__file__,"diskcache_path":diskcache.__file__}))')
    result = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, timeout=30, check=False)
    if result.returncode:
        write(ROOT / 'import_failure.json', {'exit_code': result.returncode,
             'stderr': result.stderr[-4000:], 'automatic_retry': False})
        raise RuntimeError('isolated_package_import_failed_no_retry')
    installed = json.loads(result.stdout)
    if (installed['llama_cpp_version'] != '0.3.36'
            or installed['diskcache_version'] != package['info']['version']
            or not Path(installed['llama_cpp_path']).is_relative_to(target)
            or not Path(installed['diskcache_path']).is_relative_to(target)):
        raise ValueError('isolated_distribution_identity_changed')
    receipt = {'stage': 'isolated_cpu_python_ready_no_model_loaded',
        'protocol_sha256': sha(ROOT / 'setup_protocol.json'), **installed,
        'setup_seconds': time.monotonic() - started, 'model_generations': 0,
        'paid_provider_calls': 0, 'secret_access': False, 'base_environment_modified': False}
    write(ROOT / 'setup_receipt.json', receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
