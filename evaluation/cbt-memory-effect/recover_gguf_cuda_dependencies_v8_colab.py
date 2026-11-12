"""One isolated CUDA12 dependency recovery; no model load or generation.

Keeps the failed v7 setup and verified cu124 wheel/installed package unchanged.
Only two publisher-pinned NVIDIA wheels are installed into a fresh v8 target.
The library path applies to new child processes, never the base environment.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path('/content/cbt-qwen7b-gpu-python-recovery-20261112-v8')
OLD = Path('/content/cbt-qwen7b-gpu-python-20261112-v7')
META = Path('/content/cbt-cuda12-dependency-metadata-20261112-v8')
CPP_WHEEL = 'llama_cpp_python-0.3.36-py3-none-manylinux_2_35_x86_64.whl'
CPP_SHA = '3d53f94464d8d69028d993033d6120439c994108c04375a46205bd1935bc30ad'
CPP_BYTES = 1776573061
DEADLINE = 600
PACKAGES = (
    {'name': 'nvidia-cuda-runtime-cu12', 'version': '12.4.127',
     'metadata_sha256': '654aff4bc6655e035d973cc320d484e085cd7af8205ac5016a91932652e3a740',
     'bytes': 883737, 'sha256': '64403288fa2136ee8e467cdc9c9427e0434110899d07c779f25b5c068934faa5',
     'library_subdir': 'nvidia/cuda_runtime/lib'},
    {'name': 'nvidia-cublas-cu12', 'version': '12.4.5.8',
     'metadata_sha256': 'f6ede528be91742346bb448a8dbb8d78a2ba657eaa3580eaf2680b1a32cdd346',
     'bytes': 363438805, 'sha256': '2fc8da60df463fdefa81e323eef2e36489e1c94335b5358bcb38360adf75ac9b',
     'library_subdir': 'nvidia/cublas/lib'},
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def event(value):
    with (ROOT / 'recovery_journal.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps({**value, 'at_unix': time.time()}, sort_keys=True) + '\n')
        f.flush(); os.fsync(f.fileno())


def left(deadline):
    value = deadline - time.monotonic()
    if value <= 0: raise TimeoutError('cuda12_recovery_absolute_deadline_exhausted')
    return value


def wheel_entry(package, raw):
    if hashlib.sha256(raw).hexdigest() != package['metadata_sha256']:
        raise ValueError('externally_verified_nvidia_pypi_metadata_changed')
    return wheel_entry_value(package, json.loads(raw))


def wheel_entry_value(package, data):
    if data['info']['name'].replace('_', '-').casefold() != package['name'] or data['info']['version'] != package['version']:
        raise ValueError('exact_nvidia_release_required')
    selected = [x for x in data['urls'] if x.get('packagetype') == 'bdist_wheel'
        and x.get('filename', '').endswith('x86_64.whl') and 'manylinux' in x['filename'] and not x.get('yanked')]
    if len(selected) != 1: raise ValueError('one_official_nvidia_linux_wheel_required')
    x = selected[0]
    if (x['size'] != package['bytes'] or x['digests']['sha256'] != package['sha256']
            or Path(x['filename']).name != x['filename']
            or not x['url'].startswith('https://files.pythonhosted.org/packages/')):
        raise ValueError('publisher_nvidia_wheel_identity_changed')
    return {**package, 'filename': x['filename'], 'url': x['url']}


def download(item, deadline):
    p = ROOT / item['filename']
    if p.exists(): raise FileExistsError('new_wheel_path_required_no_retry')
    received, started = 0, time.monotonic()
    event({'event': 'dependency_download_intent', 'name': item['name'], 'filename': p.name,
           'bytes': item['bytes'], 'sha256': item['sha256'], 'source_url': item['url']})
    try:
        h = hashlib.sha256()
        with urllib.request.urlopen(item['url'], timeout=min(40, left(deadline))) as q, p.open('xb') as f:
            if q.status != 200: raise ValueError('nvidia_wheel_http_200_required')
            while True:
                left(deadline); b = q.read(1024 * 1024)
                if not b: break
                received += len(b)
                if received > item['bytes']: raise ValueError('publisher_nvidia_wheel_size_exceeded')
                f.write(b); h.update(b)
            f.flush(); os.fsync(f.fileno())
        if received != item['bytes'] or h.hexdigest() != item['sha256'] or sha(p) != item['sha256']:
            raise ValueError('publisher_nvidia_wheel_size_or_digest_mismatch')
        event({'event': 'dependency_download_verified', 'name': item['name'],
               'bytes_received': received, 'sha256': item['sha256'], 'seconds': time.monotonic() - started})
        return p
    except BaseException as e:
        event({'event': 'dependency_download_failed_or_unknown', 'name': item['name'],
               'bytes_received': received, 'seconds': time.monotonic() - started,
               'failure_type': type(e).__name__, 'automatic_retry': False})
        raise


def main():
    try:
        hosted = (bool(os.environ.get('COLAB_RELEASE_TAG')) and Path('/content').is_dir()
                  and importlib.util.find_spec('google.colab') is not None)
    except (ImportError, ValueError, AttributeError): hosted = False
    if not hosted: raise RuntimeError('genuine_hosted_colab_required_before_io')
    if ROOT.exists(): raise FileExistsError('fresh_dependency_recovery_root_required_no_retry')
    if (OLD / 'setup_receipt.json').exists(): raise ValueError('old_v7_must_remain_failed')
    failed = json.loads((OLD / 'setup_stopped_receipt.json').read_text())
    old_protocol = json.loads((OLD / 'setup_protocol.json').read_text())
    if failed.get('phase') != 'isolated_import_and_compiled_cuda_probe' or failed.get('model_generations') != 0:
        raise ValueError('specific_zero_generation_old_setup_failure_required')
    cpp = OLD / CPP_WHEEL
    if cpp.stat().st_size != CPP_BYTES or sha(cpp) != CPP_SHA:
        raise ValueError('reuse_only_publisher_verified_original_cuda_wheel')
    root_refs = {str(OLD / n): sha(OLD / n) for n in ('setup_protocol.json', 'setup_stopped_receipt.json', 'setup_journal.jsonl', 'import_probe_stderr.log')}
    wheels = [wheel_entry(p, (META / (p['name'] + '.json')).read_bytes()) for p in PACKAGES]
    target = ROOT / 'site-packages'
    libdirs = [target / p['library_subdir'] for p in PACKAGES]
    library_path = ':'.join(map(str, [*libdirs, OLD / 'site-packages/llama_cpp/lib']))
    ROOT.mkdir(mode=0o700)
    started, deadline = time.monotonic(), time.monotonic() + DEADLINE
    phase, probe = 'dependency_downloads', None
    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('cuda12_recovery_absolute_deadline_exhausted')))
    signal.setitimer(signal.ITIMER_REAL, DEADLINE)
    try:
        protocol = {'version': 'isolated-cu124-runtime-dependencies-recovery-v8', 'code_sha256': sha(__file__),
            'old_failed_setup_references': root_refs, 'reused_llama_cpp_wheel': {'path': str(cpp), 'bytes': CPP_BYTES, 'sha256': CPP_SHA},
            'nvidia_pypi_metadata': [{'path': str(META / (p['name'] + '.json')), 'sha256': p['metadata_sha256']} for p in PACKAGES],
            'wheels': wheels, 'dependency_target': str(target), 'llama_cpp_site_packages': str(OLD / 'site-packages'),
            'child_ld_library_path': library_path, 'absolute_deadline_seconds': DEADLINE,
            'base_environment_modified': False, 'old_setup_modified': False, 'automatic_retry': False,
            'model_generations': 0, 'paid_provider_calls': 0, 'secret_access': False}
        write(ROOT / 'recovery_protocol.json', protocol)
        paths = [download(w, deadline) for w in wheels]
        phase = 'isolated_dependency_install'
        command = [sys.executable, '-m', 'pip', '--disable-pip-version-check', 'install', '--no-index', '--no-deps', '--target', str(target), *map(str, paths)]
        event({'event': 'dependency_install_intent', 'command': command})
        with (ROOT / 'install.log').open('xb') as log:
            child = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=min(180, left(deadline)), check=False)
        event({'event': 'dependency_install_returned', 'exit_code': child.returncode})
        if child.returncode: raise RuntimeError('isolated_nvidia_install_failed_no_retry')
        phase = 'native_linker_diagnosis'
        env = dict(os.environ); env['LD_LIBRARY_PATH'] = library_path
        reports = []
        for name in ('libllama.so', 'libggml-cuda.so', 'libggml.so', 'libggml-base.so'):
            path = OLD / 'site-packages/llama_cpp/lib' / name
            q = subprocess.run(['ldd', str(path)], env=env, capture_output=True, text=True, timeout=min(15, left(deadline)), check=False)
            reports.append({'path': str(path), 'exit_code': q.returncode, 'stdout': q.stdout, 'stderr': q.stderr})
        write(ROOT / 'resolved_native_linker_report.json', reports)
        if any(q['exit_code'] or 'not found' in q['stdout'] for q in reports):
            raise RuntimeError('native_dependency_still_missing_no_model_attempted')
        phase = 'isolated_cuda_import_no_model'
        code = ('import sys,json,importlib.metadata;sys.path[:0]=' + repr([str(OLD / 'site-packages'), str(target)]) + ';'
                'import llama_cpp;import llama_cpp.llama_cpp as low;'
                'print(json.dumps({"llama_cpp_version":llama_cpp.__version__,"llama_cpp_path":llama_cpp.__file__,'
                '"native_library_path":str(low._lib._name),"compiled_gpu_offload_support":bool(llama_cpp.llama_supports_gpu_offload()),'
                '"nvidia_dependencies":[{"name":n,"version":importlib.metadata.version(n)} for n in '
                '("nvidia-cuda-runtime-cu12","nvidia-cublas-cu12")]}))')
        q = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=min(30, left(deadline)), check=False)
        with (ROOT / 'import_probe_stderr.log').open('xb') as f: f.write(q.stderr.encode())
        if q.returncode: raise RuntimeError('recovered_cuda_import_failed_no_retry')
        probe = json.loads(q.stdout)
        if (probe.get('llama_cpp_version') != '0.3.36' or probe.get('compiled_gpu_offload_support') is not True
                or Path(probe['llama_cpp_path']).resolve() != OLD / 'site-packages/llama_cpp/__init__.py'
                or {x['name']: x['version'] for x in probe['nvidia_dependencies']} != {p['name']: p['version'] for p in PACKAGES}):
            raise ValueError('recovered_runtime_identity_changed')
        phase = 'installed_dependency_manifest'
        files = []
        for directory in libdirs:
            if not directory.is_dir(): raise ValueError('exact_isolated_nvidia_lib_directory_required')
            for path in sorted(directory.glob('*.so*')):
                if path.is_symlink() or not path.is_file(): raise ValueError('regular_nvidia_shared_library_required')
                files.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)})
        if not {'libcudart.so.12', 'libcublas.so.12', 'libcublasLt.so.12'}.issubset({Path(x['path']).name for x in files}):
            raise ValueError('all_three_required_cuda12_shared_libraries_required')
        write(ROOT / 'native_dependency_manifest.json', {'version': protocol['version'], 'wheels': wheels,
            'installed_library_files': files, 'child_ld_library_path': library_path,
            'dependency_site_packages': str(target), 'llama_cpp_site_packages': str(OLD / 'site-packages')})
        if any(sha(p) != expected for p, expected in root_refs.items()) or (OLD / 'setup_receipt.json').exists():
            raise ValueError('old_failed_setup_bytes_must_remain_unchanged')
        receipt = {'stage': 'isolated_cuda12_dependencies_ready_no_model_loaded', 'protocol_sha256': sha(ROOT / 'recovery_protocol.json'),
            'native_dependency_manifest_path': str(ROOT / 'native_dependency_manifest.json'),
            'native_dependency_manifest_sha256': sha(ROOT / 'native_dependency_manifest.json'),
            'runtime_manifest_extension': {'native_dependency_manifest_path': str(ROOT / 'native_dependency_manifest.json'),
                'native_dependency_manifest_sha256': sha(ROOT / 'native_dependency_manifest.json'),
                'child_ld_library_path': library_path, 'dependency_site_packages': str(target)},
            **probe, 'setup_seconds': time.monotonic() - started, 'model_generations': 0,
            'base_environment_modified': False, 'old_failed_setup_preserved': True, 'automatic_retry': False,
            'compiled_support_is_not_model_offload_evidence': True, 'paid_provider_calls': 0, 'secret_access': False}
        write(ROOT / 'setup_receipt.json', receipt)
        print(json.dumps(receipt, sort_keys=True), flush=True)
    except BaseException as e:
        event({'event': 'recovery_failed_or_unknown', 'phase': phase, 'failure_type': type(e).__name__,
               'elapsed_seconds': time.monotonic() - started, 'automatic_retry': False})
        receipt = {'stage': 'isolated_cuda12_recovery_stopped_failed_or_unknown', 'phase': phase,
            'failure_type': type(e).__name__, 'elapsed_seconds': time.monotonic() - started,
            'model_generations': 0, 'automatic_retry': False, 'paid_provider_calls': 0, 'secret_access': False}
        write(ROOT / 'recovery_stopped_receipt.json', receipt); print(json.dumps(receipt, sort_keys=True), flush=True)
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)
    return 0


if __name__ == '__main__':
    try: result = main()
    except BaseException as e:
        print(json.dumps({'stage': 'cuda12_recovery_entry_stopped_before_root', 'failure_type': type(e).__name__, 'automatic_retry': False}), flush=True); result = 1
    sys.exit(result or 0)
