"""One explicit, separately recorded hosted load diagnostic; no generation.

The failed v5 gate is never resumed or modified. This records actual native
context/token metadata and constructor errors to diagnose its pre-admission
failure. It does not call generate/eval or return answers.
"""
import importlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gguf_schema_reader_v5_colab as native

ROOT = Path('/content/cbt-qwen7b-load-diagnostic-20261112-v1')
GATE = Path('/content/cbt-qwen7b-reader-gate-20261112-v5')
MODEL = Path('/content/cbt-qwen7b-reader-setup-20261112-v1')
GATE_SHA = 'df5d2f401af1fc1ae7fc4e900dcb6ac3c99c8a58556c56a5f3fb28990a471413'


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def main():
    native.schema.require_hosted_colab()
    if ROOT.exists():
        raise FileExistsError('one_fresh_diagnostic_only_no_retry')
    if native.file_sha(GATE / 'gate_protocol.json') != GATE_SHA:
        raise ValueError('failed_v5_gate_identity_changed')
    if native.file_sha(native.__file__) != '7968f8a327a2a66ca350c83d4e584d0ee688ec69bbe75016801054a53d6ff864':
        raise ValueError('frozen_native_backend_changed')
    if (GATE / 'generation_ledger.jsonl').exists():
        raise RuntimeError('diagnostic_requires_zero_admitted_old_generation_attempts')
    gate_protocol = json.loads((GATE / 'gate_protocol.json').read_text())
    for name, key in [('model_manifest.json', 'model_manifest_sha256'),
                      ('runtime_manifest.json', 'runtime_manifest_sha256')]:
        if native.file_sha(GATE / name) != gate_protocol[key]:
            raise ValueError('frozen_setup_manifest_changed')
    runtime = json.loads((GATE / 'runtime_manifest.json').read_text())
    manifest = json.loads((GATE / 'model_manifest.json').read_text())
    ROOT.mkdir(mode=0o700)
    kwargs = dict(n_ctx=4160, n_threads=2, n_threads_batch=2, n_batch=128,
        n_ubatch=128, n_gpu_layers=0, use_mmap=True, use_mlock=False,
        logits_all=False, embedding=False, seed=0, verbose=False)
    protocol = {'version': 'single-hosted-native-load-diagnostic-v1',
        'maximum_load_attempts': 1, 'maximum_answer_generations': 0,
        'automatic_retry': False, 'old_gate_protocol_sha256': GATE_SHA,
        'native_backend_sha256': native.file_sha(native.__file__),
        'diagnostic_code_sha256': native.file_sha(__file__),
        'model_manifest_sha256': native.file_sha(GATE / 'model_manifest.json'),
        'runtime_manifest_sha256': native.file_sha(GATE / 'runtime_manifest.json'),
        'constructor_kwargs': kwargs, 'secret_access': False, 'paid_provider_calls': 0}
    write(ROOT / 'diagnostic_protocol.json', protocol)
    started = time.monotonic()
    phases, engine = [], None
    status, failure = 'diagnostic_complete_no_answers', None
    details = {}
    try:
        t = time.monotonic()
        native.verify_model_artifacts(MODEL, manifest)
        phases.append({'phase': 'complete_model_bytes_verified', 'seconds': time.monotonic() - t})
        sys.path.insert(0, runtime['site_packages'])
        t = time.monotonic()
        verified_runtime = native._verify_runtime(runtime)
        module = importlib.import_module('llama_cpp')
        low = importlib.import_module('llama_cpp.llama_cpp')
        native._verify_loaded_native(verified_runtime, low)
        phases.append({'phase': 'runtime_and_actual_dll_verified', 'seconds': time.monotonic() - t})
        write(ROOT / 'load_intent.json', {'at_unix': time.time(), 'maximum_answer_generations': 0})
        t = time.monotonic()
        engine = module.Llama(model_path=str(MODEL / native.MODEL_FILES[0]), **kwargs)
        phases.append({'phase': 'native_constructor_returned', 'seconds': time.monotonic() - t})
        details['actual_n_ctx'] = engine.n_ctx()
        details['requested_n_ctx'] = kwargs['n_ctx']
        details['actual_control_token_ids'] = {s: engine.tokenize(s.encode(), add_bos=False, special=True)
            for s in native.CONTROL_IDS}
        vocab = module.llama_model_get_vocab(engine.model)
        details['actual_token_eos'] = engine.token_eos()
        details['native_eog_flags'] = {str(i): bool(module.llama_vocab_is_eog(vocab, i))
            for i in sorted(set([engine.token_eos(), *native.CONTROL_IDS.values()]))}
        details['relevant_model_metadata'] = {k: v for k, v in engine.metadata.items()
            if k in {'general.architecture', 'general.name', 'tokenizer.ggml.bos_token_id',
                     'tokenizer.ggml.eos_token_id', 'qwen2.context_length'}}
        native._verify_loaded_native(verified_runtime, low)
        details['post_load_dll_verification_passed'] = True
    except BaseException as e:
        status = 'diagnostic_stopped_no_answers'
        # There is no network in this diagnostic. Constructor/own guard error
        # text can be recorded without capturing signed vendor redirect URLs.
        failure = {'type': type(e).__name__, 'message': str(e)[:1000]}
    finally:
        if engine is not None:
            try:
                engine.close()
            except BaseException as e:
                details['close_error_type'] = type(e).__name__
        receipt = {'stage': status, 'failure': failure, 'phases': phases,
            'actual_details': details, 'elapsed_seconds': time.monotonic() - started,
            'answer_generations': 0, 'paid_provider_calls': 0, 'secret_access': False,
            'failed_v5_gate_unchanged': native.file_sha(GATE / 'gate_protocol.json') == GATE_SHA
                and not (GATE / 'generation_ledger.jsonl').exists(),
            'automatic_retry': False}
        write(ROOT / 'diagnostic_receipt.json', receipt)
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
