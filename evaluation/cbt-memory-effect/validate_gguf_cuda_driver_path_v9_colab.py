"""One no-network, no-install, no-model CUDA driver-path validation.

Preserves both stopped v7/v8 setups and reuses their publisher-verified bytes.
Only a child-process library path gains the already observed Colab driver dir.
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
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recover_gguf_cuda_dependencies_v8_colab as prior

VERSION = 'isolated-cu124-host-driver-path-validation-v9'
ROOT = Path('/content/cbt-qwen7b-gpu-python-recovery-20261112-v9')
OLD7 = Path('/content/cbt-qwen7b-gpu-python-20261112-v7')
OLD8 = Path('/content/cbt-qwen7b-gpu-python-recovery-20261112-v8')
DRIVER_DIR = Path('/usr/lib64-nvidia')
DRIVER = DRIVER_DIR / 'libcuda.so.1'
PRIOR_SHA = '9c17df0533a5c77bf33005eed5e980c4fbf4703a5aadc58d17efe3c1da930480'
DEADLINE = 600
sha = prior.sha


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def event(value):
    with (ROOT / 'validation_journal.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps({**value, 'at_unix': time.time()}, sort_keys=True) + '\n')
        f.flush(); os.fsync(f.fileno())


def seconds(deadline):
    left = deadline - time.monotonic()
    if left <= 0: raise TimeoutError('driver_validation_absolute_deadline_exhausted')
    return left


def stream_member_sha(archive, member):
    h = hashlib.sha256()
    with archive.open(member) as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def library_prefix():
    target = OLD8 / 'site-packages'
    return ':'.join(map(str, [target / p['library_subdir'] for p in prior.PACKAGES] +
                        [OLD7 / 'site-packages/llama_cpp/lib', DRIVER_DIR]))


def main():
    try:
        hosted = (bool(os.environ.get('COLAB_RELEASE_TAG')) and Path('/content').is_dir()
                  and importlib.util.find_spec('google.colab') is not None)
    except (ImportError, ValueError, AttributeError): hosted = False
    if not hosted: raise RuntimeError('genuine_hosted_colab_required_before_io')
    if ROOT.exists(): raise FileExistsError('new_driver_validation_root_required_no_retry')
    if sha(prior.__file__) != PRIOR_SHA: raise ValueError('frozen_prior_v8_source_changed')
    if os.environ.get('LD_LIBRARY_PATH') != str(DRIVER_DIR):
        raise ValueError('exact_observed_colab_host_driver_path_required')
    if (OLD7 / 'setup_receipt.json').exists() or (OLD8 / 'setup_receipt.json').exists():
        raise ValueError('both_older_setups_must_remain_failed')
    failed8 = json.loads((OLD8 / 'recovery_stopped_receipt.json').read_text())
    if failed8.get('phase') != 'native_linker_diagnosis' or failed8.get('model_generations') != 0:
        raise ValueError('specific_zero_generation_v8_linker_failure_required')
    ROOT.mkdir(mode=0o700)
    started, deadline = time.monotonic(), time.monotonic() + DEADLINE
    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('driver_validation_absolute_deadline_exhausted')))
    signal.setitimer(signal.ITIMER_REAL, DEADLINE)
    phase = 'preserved_input_verification'
    try:
        refs = {str(Path(prior.__file__)): PRIOR_SHA}
        for r, names in [(OLD7, ('setup_protocol.json', 'setup_stopped_receipt.json', 'setup_journal.jsonl', 'import_probe_stderr.log')),
                         (OLD8, ('recovery_protocol.json', 'recovery_stopped_receipt.json', 'recovery_journal.jsonl', 'resolved_native_linker_report.json', 'install.log'))]:
            for name in names: refs[str(r / name)] = sha(r / name)
        p8 = json.loads((OLD8 / 'recovery_protocol.json').read_text())
        if p8.get('code_sha256') != PRIOR_SHA or p8.get('model_generations') != 0:
            raise ValueError('old_v8_frozen_recovery_identity_changed')
        wheels = [prior.wheel_entry(p, (prior.META / (p['name'] + '.json')).read_bytes()) for p in prior.PACKAGES]
        if wheels != p8.get('wheels'): raise ValueError('old_v8_nvidia_wheel_catalog_changed')
        files = []
        for package in wheels:
            wheel = OLD8 / package['filename']
            if wheel.stat().st_size != package['bytes'] or sha(wheel) != package['sha256']:
                raise ValueError('reused_publisher_nvidia_wheel_bytes_changed')
            with zipfile.ZipFile(wheel) as z:
                for member in z.infolist():
                    if member.is_dir() or not member.filename.startswith(package['library_subdir'] + '/') or '.so' not in Path(member.filename).name:
                        continue
                    relative = Path(member.filename)
                    if relative.is_absolute() or '..' in relative.parts: raise ValueError('unsafe_dependency_member')
                    path = OLD8 / 'site-packages' / relative
                    expected = stream_member_sha(z, member)
                    if path.is_symlink() or not path.is_file() or path.stat().st_size != member.file_size or sha(path) != expected:
                        raise ValueError('installed_nvidia_library_differs_from_publisher_wheel')
                    files.append({'path': str(path), 'bytes': member.file_size, 'sha256': expected})
        if not {'libcudart.so.12', 'libcublas.so.12', 'libcublasLt.so.12'}.issubset({Path(x['path']).name for x in files}):
            raise ValueError('three_verified_cuda12_shared_libraries_required')
        cpp = OLD7 / prior.CPP_WHEEL
        if cpp.stat().st_size != prior.CPP_BYTES or sha(cpp) != prior.CPP_SHA:
            raise ValueError('original_publisher_cuda_wheel_changed')
        resolved = DRIVER.resolve(strict=True)
        if not resolved.is_file() or 'stubs' in resolved.parts:
            raise ValueError('actual_colab_driver_library_required')
        q = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=min(15, seconds(deadline)), check=False)
        if q.returncode or 'T4' not in q.stdout or '580.82.07' not in q.stdout:
            raise ValueError('same_actual_t4_driver_required')
        driver = {'library_path': str(DRIVER), 'resolved_path': str(resolved), 'bytes': resolved.stat().st_size,
            'sha256': sha(resolved), 'library_directory': str(DRIVER_DIR), 'actual_gpu_query': q.stdout.strip()}
        prefix = library_prefix()
        protocol = {'version': VERSION, 'code_sha256': sha(__file__), 'prior_v8_source_sha256': PRIOR_SHA,
            'old_failed_setup_references': refs, 'wheels': wheels, 'wheel_directory': str(OLD8),
            'nvidia_pypi_metadata': p8['nvidia_pypi_metadata'],
            'reused_llama_cpp_wheel': {'path': str(cpp), 'bytes': prior.CPP_BYTES, 'sha256': prior.CPP_SHA},
            'installed_library_files': files, 'driver_reference': driver, 'child_ld_library_path': prefix,
            'dependency_site_packages': str(OLD8 / 'site-packages'), 'llama_cpp_site_packages': str(OLD7 / 'site-packages'),
            'dependency_target': str(OLD8 / 'site-packages'),
            'maximum_network_requests': 0, 'package_installs': 0, 'model_generations': 0,
            'absolute_deadline_seconds': DEADLINE, 'automatic_retry': False, 'base_environment_modified': False}
        write(ROOT / 'recovery_protocol.json', protocol)
        phase = 'driver_path_linker_verification'
        env = dict(os.environ); env['LD_LIBRARY_PATH'] = prefix
        reports = []
        for name in ('libllama.so', 'libggml-cuda.so', 'libggml.so', 'libggml-base.so'):
            path = OLD7 / 'site-packages/llama_cpp/lib' / name
            q = subprocess.run(['ldd', str(path)], env=env, capture_output=True, text=True,
                               timeout=min(15, seconds(deadline)), check=False)
            reports.append({'path': str(path), 'exit_code': q.returncode, 'stdout': q.stdout, 'stderr': q.stderr})
        write(ROOT / 'resolved_native_linker_report.json', reports)
        if any(x['exit_code'] or 'not found' in x['stdout'] for x in reports):
            raise RuntimeError('linker_still_missing_no_model_attempted')
        phase = 'isolated_import_no_model'
        code = ('import sys,json,importlib.metadata;sys.path[:0]=' + repr([str(OLD7 / 'site-packages'), str(OLD8 / 'site-packages')]) + ';'
                'import llama_cpp;import llama_cpp.llama_cpp as low;'
                'print(json.dumps({"llama_cpp_version":llama_cpp.__version__,"llama_cpp_path":llama_cpp.__file__,'
                '"native_library_path":str(low._lib._name),"compiled_gpu_offload_support":bool(llama_cpp.llama_supports_gpu_offload()),'
                '"nvidia_dependencies":[{"name":n,"version":importlib.metadata.version(n)} for n in '
                '("nvidia-cuda-runtime-cu12","nvidia-cublas-cu12")]}))')
        q = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True,
                           timeout=min(30, seconds(deadline)), check=False)
        with (ROOT / 'import_probe_stderr.log').open('xb') as f: f.write(q.stderr.encode())
        if q.returncode: raise RuntimeError('driver_path_cuda_import_failed_no_retry')
        probe = json.loads(q.stdout)
        if (probe.get('llama_cpp_version') != '0.3.36' or probe.get('compiled_gpu_offload_support') is not True
                or Path(probe['llama_cpp_path']).resolve() != OLD7 / 'site-packages/llama_cpp/__init__.py'
                or {x['name']: x['version'] for x in probe['nvidia_dependencies']} != {x['name']: x['version'] for x in wheels}):
            raise ValueError('actual_recovered_runtime_identity_changed')
        manifest = {'version': VERSION, 'wheels': wheels, 'wheel_directory': str(OLD8), 'installed_library_files': files,
            'driver_reference': driver, 'child_ld_library_path': prefix, 'dependency_site_packages': str(OLD8 / 'site-packages'),
            'llama_cpp_site_packages': str(OLD7 / 'site-packages')}
        write(ROOT / 'native_dependency_manifest.json', manifest)
        if any(sha(p) != value for p, value in refs.items()) or (OLD7 / 'setup_receipt.json').exists() or (OLD8 / 'setup_receipt.json').exists():
            raise ValueError('older_failed_setup_bytes_must_remain_unchanged')
        extension = {k: manifest[k] for k in ('child_ld_library_path', 'dependency_site_packages', 'driver_reference', 'wheel_directory')}
        extension.update(native_dependency_manifest_path=str(ROOT / 'native_dependency_manifest.json'),
                         native_dependency_manifest_sha256=sha(ROOT / 'native_dependency_manifest.json'))
        receipt = {'stage': 'isolated_cuda12_driver_path_ready_no_model_loaded', 'protocol_sha256': sha(ROOT / 'recovery_protocol.json'),
            'native_dependency_manifest_path': str(ROOT / 'native_dependency_manifest.json'),
            'native_dependency_manifest_sha256': sha(ROOT / 'native_dependency_manifest.json'), 'runtime_manifest_extension': extension,
            **probe, 'setup_seconds': time.monotonic() - started, 'model_generations': 0, 'network_requests': 0,
            'package_installs': 0, 'base_environment_modified': False, 'older_failed_setups_preserved': True,
            'old_failed_setup_preserved': True,
            'compiled_support_is_not_model_offload_evidence': True, 'automatic_retry': False, 'paid_provider_calls': 0, 'secret_access': False}
        write(ROOT / 'setup_receipt.json', receipt); print(json.dumps(receipt, sort_keys=True), flush=True)
    except BaseException as e:
        receipt = {'stage': 'isolated_cuda12_driver_validation_stopped_failed_or_unknown', 'phase': phase,
            'failure_type': type(e).__name__, 'elapsed_seconds': time.monotonic() - started,
            'automatic_retry': False, 'model_generations': 0, 'network_requests': 0, 'package_installs': 0,
            'paid_provider_calls': 0, 'secret_access': False}
        event(receipt); write(ROOT / 'validation_stopped_receipt.json', receipt); print(json.dumps(receipt, sort_keys=True), flush=True)
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, previous)
    return 0


if __name__ == '__main__':
    try: result = main()
    except BaseException as e:
        print(json.dumps({'stage': 'driver_validation_entry_stopped_before_root', 'failure_type': type(e).__name__, 'automatic_retry': False}), flush=True); result = 1
    sys.exit(result or 0)
