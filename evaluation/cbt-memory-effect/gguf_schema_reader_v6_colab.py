"""Prospective local-only GGUF reader; actual model work is hosted Colab only.

The separately preserved v5 attempt stopped before any generation because the
native library allocated 4352 tokens for requested 4160. This v6 requests that
observed native capacity explicitly; logical input4096/output64 caps do not change.

Official Qwen 7B Q4_K_M is a new model/quantization condition, not a continuation
of the earlier float32 1.5B scores. This adapter neither downloads artifacts nor
admits attempts: a journalled caller must freeze/admit each request first.

The native llama.cpp token vector and sampled token IDs are recorded directly.
Content uses special=False independently of trusted ChatML boundaries, so an
original quotation containing a role marker cannot introduce a new role.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import schema_reader_v4 as schema

VERSION = "hosted-official-qwen7b-q4km-question-schema-native-reader-v6-context4352"
PACKAGE_VERSION = "0.3.36"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-GGUF"
QUANTIZATION = "Q4_K_M"
MODEL_FILES = (
    "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
    "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf",
)
CONTEXT_TOKENS = 4352
MAX_INPUT_TOKENS = 4096
MAX_NEW_TOKENS = 64
N_THREADS = 2
WHEEL_BYTES = 27077498
WHEEL_SHA256 = "c8a3e98093e950d7a770f794842e36f2db82b2933e8762ed7b5ffd435011e8e4"
EXPECTED_SCHEMA_SHA = "3f78c83975dba44c93851b136be10e239c4316eea40d5998c2f6e2c9d52cdfb6"
CONTROL_IDS = {"<|im_start|>": 151644, "<|im_end|>": 151645,
               "<|endoftext|>": 151643}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def policy():
    return {
        "version": VERSION, "model_id": MODEL_ID, "quantization": QUANTIZATION,
        "llama_cpp_python_version": PACKAGE_VERSION,
        "actual_execution": "genuine hosted Colab CPU only; local artifact paths only",
        "n_ctx": CONTEXT_TOKENS, "maximum_input_tokens": MAX_INPUT_TOKENS,
        "context_allocation_reason": "explicit observed native padding capacity; logical input/output budgets unchanged",
        "maximum_new_tokens": MAX_NEW_TOKENS, "n_threads": N_THREADS,
        "n_gpu_layers": 0, "use_mmap": True,
        "rendering": "explicit Qwen2.5 ChatML, standard text roles, no tools or assistant prefill",
        "tokenization": "trusted boundaries special=True; exact content special=False; add_bos=False",
        "sampling": {"temp": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0,
                     "typical_p": 1.0, "repeat_penalty": 1.0,
                     "frequency_penalty": 0.0, "presence_penalty": 0.0,
                     "reset": True, "seed": 0},
        "eos": "only actual native EOG tokens observed from the iterator; never appended",
        "schema": "public question only; same v4 type domain in every arm",
        "truncation_or_cutoff_repair": False,
        "automatic_retry_or_download": False,
        "native_token_vector_not_pytorch_tensor": True,
    }


def _owned_schema():
    if (Path(schema.__file__).resolve() != HERE / "schema_reader_v4.py"
            or file_sha(schema.__file__) != EXPECTED_SCHEMA_SHA):
        raise RuntimeError("frozen_question_schema_reader_identity_changed")


def verify_model_artifacts(model_dir, manifest, *, fixture_only=False):
    if not fixture_only:
        schema.require_hosted_colab()
    root = Path(model_dir).resolve()
    if not fixture_only and (not root.is_relative_to(Path("/content")) or root == Path("/content")):
        raise ValueError("owned_hosted_model_directory_required")
    if (type(manifest) is not dict or manifest.get("model_id") != MODEL_ID
            or manifest.get("quantization") != QUANTIZATION
            or manifest.get("license") != "apache-2.0"
            or not re.fullmatch(r"[0-9a-f]{40}", manifest.get("model_revision", ""))):
        raise ValueError("pinned_official_model_identity_required")
    entries = manifest.get("files")
    if (type(entries) is not list or len(entries) != 2
            or {entry.get("path") for entry in entries if type(entry) is dict} != set(MODEL_FILES)):
        raise ValueError("exact_two_official_split_files_required")
    verified = []
    for entry in sorted(entries, key=lambda value: value["path"]):
        name = entry["path"]
        if (Path(name).name != name or type(entry.get("bytes")) is not int
                or entry["bytes"] < 1 or not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", ""))):
            raise ValueError("exact_split_bytes_and_sha256_required")
        target = root / name
        if target.is_symlink() or not target.is_file() or target.resolve().parent != root:
            raise ValueError("regular_owned_split_file_required")
        if target.stat().st_size != entry["bytes"] or file_sha(target) != entry["sha256"]:
            raise ValueError("official_split_size_or_checksum_mismatch")
        verified.append({"path": name, "bytes": entry["bytes"], "sha256": entry["sha256"]})
    return {"model_id": MODEL_ID, "model_revision": manifest["model_revision"],
            "quantization": QUANTIZATION, "license": "apache-2.0", "files": verified,
            "files_verified": 2, "bytes_verified": sum(item["bytes"] for item in verified),
            "manifest_sha256": digest(manifest)}


def _messages(messages):
    if type(messages) is not list or not messages:
        raise ValueError("nonempty_chat_message_list_required")
    if any(type(message) is not dict or set(message) != {"role", "content"}
           or message["role"] not in {"system", "user", "assistant"}
           or type(message["content"]) is not str for message in messages):
        raise ValueError("text_only_standard_chat_messages_required")
    if messages[0]["role"] != "system" or messages[-1]["role"] != "user":
        raise ValueError("explicit_system_and_final_user_question_required")
    return messages


def chatml_segments(messages):
    """Trusted-boundary flags are independent of exact original message text."""
    result = []
    for message in _messages(messages):
        result.extend((("<|im_start|>" + message["role"] + "\n", True),
                       (message["content"], False), ("<|im_end|>\n", True)))
    result.append(("<|im_start|>assistant\n", True))
    return result


def grammar_text(question_schema):
    contract = schema._checked_contract(question_schema)
    choices = ['"null"']
    lines = ["root ::= " + json.dumps('{"answer":') + " value " + json.dumps("}")]
    if contract["integer_allowed"]:
        choices.append("integer")
        lines.append('integer ::= "0" | [1-9] [0-9]{0,3}')
    if contract["status_allowed"]:
        choices.append("status")
        lines.append("status ::= " + " | ".join(json.dumps(json.dumps(value)) for value in schema.STATUSES))
    lines.insert(1, "value ::= " + " | ".join(choices))
    return "\n".join(lines) + "\n"


@dataclass
class GenerationFailure(Exception):
    failure_type: str
    generated_record: dict

    def __post_init__(self):
        Exception.__init__(self, self.failure_type)


class Backend:
    def __init__(self, engine, llama_module, identity, *, fixture_only=False):
        if not fixture_only:
            schema.require_hosted_colab()
        _owned_schema()
        self.engine = engine
        self.llama_module = llama_module
        self.identity = identity
        controls = {}
        for text, expected_id in CONTROL_IDS.items():
            ids = engine.tokenize(text.encode(), add_bos=False, special=True)
            if ids != [expected_id]:
                raise ValueError("pinned_qwen_special_token_identity_mismatch")
            controls[text] = ids[0]
        if engine.n_ctx() != CONTEXT_TOKENS:
            raise ValueError("actual_native_context_capacity_changed")
        vocab = llama_module.llama_model_get_vocab(engine.model)
        possible = sorted(set([engine.token_eos(), controls["<|im_end|>"], controls["<|endoftext|>"]]))
        self.eog_ids = [token for token in possible if llama_module.llama_vocab_is_eog(vocab, token)]
        if engine.token_eos() not in self.eog_ids or controls["<|im_end|>"] not in self.eog_ids:
            raise ValueError("native_qwen_eog_identity_required")
        self.identity = {**identity, "control_token_ids": controls, "native_eog_token_ids": self.eog_ids}

    def input_preflight(self, messages, *, fixture_only=False):
        if not fixture_only:
            schema.require_hosted_colab()
        _owned_schema()
        segments = chatml_segments(messages)
        ids = []
        for text, special in segments:
            segment_ids = self.engine.tokenize(text.encode(), add_bos=False, special=special)
            if not special and any(token in CONTROL_IDS.values() for token in segment_ids):
                raise ValueError("untrusted_content_produced_native_control_token")
            ids.extend(segment_ids)
        if not ids or len(ids) > MAX_INPUT_TOKENS:
            raise ValueError("actual_native_input_limit_no_truncation")
        if len(ids) + MAX_NEW_TOKENS > self.engine.n_ctx():
            raise ValueError("native_prompt_plus_output_does_not_fit_context")
        rendered = "".join(text for text, _ in segments).encode()
        if self.engine.detokenize(ids, special=True) != rendered:
            raise ValueError("segmented_chatml_tokenization_did_not_preserve_exact_text")
        question_schema = schema.schema_for_question(messages[-1]["content"])
        return {"input_tokens": len(ids), "input_token_ids": ids,
                "input_token_ids_sha256": digest(ids),
                "rendered_prompt_sha256": hashlib.sha256(rendered).hexdigest(),
                "rendered_prompt_utf8_bytes": len(rendered),
                "rendering_policy_sha256": digest(policy()),
                "question_schema": question_schema,
                "question_schema_sha256": digest(question_schema),
                "native_context_capacity": self.engine.n_ctx(), "prefilled_token_count": 0}

    def generate(self, messages, question_schema=None, *, max_new_tokens=MAX_NEW_TOKENS, fixture_only=False):
        if not fixture_only:
            schema.require_hosted_colab()
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
            raise ValueError("prospective_output_cap_one_to_64_required")
        pre = self.input_preflight(messages, fixture_only=fixture_only)
        contract = pre["question_schema"]
        if question_schema is not None and question_schema != contract:
            raise ValueError("provided_schema_must_equal_public_question_contract")
        gbnf = grammar_text(contract)
        grammar = self.llama_module.LlamaGrammar.from_string(gbnf, verbose=False)
        decoding = {"backend_policy_sha256": digest(policy()),
                    "model_identity_sha256": digest(self.identity),
                    "question_schema_sha256": digest(contract),
                    "gbnf_sha256": hashlib.sha256(gbnf.encode()).hexdigest(),
                    "max_new_tokens": max_new_tokens, "native_eog_token_ids": self.eog_ids}
        started = time.monotonic()
        sampled = []
        reason, failure = "length", None
        self.engine.reset()
        iterator = self.engine.generate(pre["input_token_ids"], temp=0.0, top_k=1, top_p=1.0,
            min_p=0.0, typical_p=1.0, repeat_penalty=1.0, reset=True,
            frequency_penalty=0.0, presence_penalty=0.0, grammar=grammar)
        try:
            for token in iterator:
                if type(token) is not int or token < 0:
                    raise ValueError("native_generator_returned_noninteger_token")
                sampled.append(token)
                if token in self.eog_ids:
                    reason = "eos"
                    break
                if len(sampled) == max_new_tokens:
                    break
            else:
                reason = "iterator_ended_without_eog"
        except Exception as exc:
            reason = "native_generation_failed_or_unknown"
            failure = type(exc).__name__
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                try:
                    close()
                except Exception as exc:
                    reason = "native_generation_failed_or_unknown"
                    failure = failure or type(exc).__name__
        elapsed = time.monotonic() - started
        observed = {"raw_generated_token_ids": list(sampled),
            "finish_reason": reason,
            "input_token_ids_sha256": pre["input_token_ids_sha256"],
            "rendered_prompt_sha256": pre["rendered_prompt_sha256"],
            "question_schema_sha256": pre["question_schema_sha256"],
            "decoding_identity_sha256": digest(decoding), "decoding_identity": decoding,
            "native_generation_wall_seconds": elapsed,
            "usage": {"prompt_tokens": pre["input_tokens"], "completion_tokens": len(sampled),
                      "total_tokens": pre["input_tokens"] + len(sampled)},
            "usage_scope": "native prompt vector length and observed yielded output tokens, including observed EOG; failed evaluation consumption may be unknown",
            "prompt_evaluation_completed": True if sampled else None,
            "prefilled_token_count": 0}
        body_ids = sampled[:-1] if reason == "eos" else sampled
        try:
            response_bytes = self.engine.detokenize(body_ids, prev_tokens=pre["input_token_ids"], special=False)
            raw_bytes = self.engine.detokenize(sampled, prev_tokens=pre["input_token_ids"], special=True)
            utf8_valid = True
            try:
                response_text = response_bytes.decode("utf-8", errors="strict")
                raw_text = raw_bytes.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                utf8_valid = False
                response_text = response_bytes.decode("utf-8", errors="replace")
                raw_text = raw_bytes.decode("utf-8", errors="replace")
            checked = schema.validate_response(response_text, contract)
            if reason != "eos" or not utf8_valid:
                checked = {"accepted": False, "reason": reason if utf8_valid else "invalid_utf8",
                           "answer": None}
            result = {**observed, "response_text": response_text,
                "response_sha256": hashlib.sha256(response_text.encode()).hexdigest(),
                "response_bytes_sha256": hashlib.sha256(response_bytes).hexdigest(),
                "response_bytes_base64": base64.b64encode(response_bytes).decode(),
                "raw_generated_text": raw_text,
                "raw_generated_bytes_base64": base64.b64encode(raw_bytes).decode(),
                "chosen_answer": checked["answer"], "schema_validation": checked,
                "utf8_valid": utf8_valid}
        except Exception as exc:
            minimum = {**observed, "native_finish_reason_before_postprocessing": reason,
                "finish_reason": "postprocessing_failed_after_native_generation",
                "postprocessing_failure_type": type(exc).__name__, "response_text": None,
                "response_sha256": None, "response_bytes_sha256": None,
                "response_bytes_base64": None, "raw_generated_text": None,
                "raw_generated_bytes_base64": None, "chosen_answer": None, "utf8_valid": None,
                "schema_validation": {"accepted": False, "reason": "postprocessing_failed", "answer": None}}
            raise GenerationFailure(failure or type(exc).__name__, minimum) from exc
        if failure:
            raise GenerationFailure(failure, result)
        return result


def _verify_runtime(runtime_manifest):
    """Bind installed executable package files to the pinned official wheel."""
    if (type(runtime_manifest) is not dict
            or runtime_manifest.get("package_name") != "llama-cpp-python"
            or runtime_manifest.get("package_version") != PACKAGE_VERSION
            or runtime_manifest.get("wheel_bytes") != WHEEL_BYTES
            or runtime_manifest.get("wheel_sha256") != WHEEL_SHA256):
        raise ValueError("pinned_official_runtime_wheel_manifest_required")
    root = Path(runtime_manifest["site_packages"]).resolve()
    wheel = Path(runtime_manifest["wheel_path"])
    if (wheel.is_symlink() or not wheel.is_file()
            or not wheel.resolve().is_relative_to(Path("/content"))
            or not root.is_relative_to(Path("/content"))):
        raise ValueError("owned_hosted_runtime_wheel_and_target_required")
    if wheel.stat().st_size != WHEEL_BYTES or file_sha(wheel) != WHEEL_SHA256:
        raise ValueError("official_runtime_wheel_size_or_sha256_mismatch")
    distribution = importlib.metadata.distribution("llama-cpp-python")
    if (distribution.version != PACKAGE_VERSION
            or Path(distribution.locate_file("")).resolve() != root):
        raise ValueError("actual_installed_runtime_distribution_location_or_version_changed")
    checked = []
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            if not member.filename.startswith("llama_cpp/") or member.is_dir():
                continue
            relative = Path(member.filename)
            if ".." in relative.parts or relative.is_absolute():
                raise ValueError("runtime_wheel_unsafe_member_path")
            installed = root / relative
            if installed.is_symlink() or not installed.is_file() or installed.resolve().is_relative_to(root) is not True:
                raise ValueError("regular_owned_installed_runtime_file_required")
            reference = hashlib.sha256(archive.read(member)).hexdigest()
            if installed.stat().st_size != member.file_size or file_sha(installed) != reference:
                raise ValueError("installed_runtime_differs_from_pinned_official_wheel")
            checked.append({"path": member.filename, "bytes": member.file_size, "sha256": reference})
    if not checked or not any(item["path"].endswith(".so") for item in checked):
        raise ValueError("official_runtime_native_library_files_required")
    dependencies = runtime_manifest.get("dependencies")
    if type(dependencies) is not list or {item.get("name") for item in dependencies} != {
            "numpy", "jinja2", "typing-extensions", "diskcache"}:
        raise ValueError("explicit_four_runtime_dependencies_required")
    versions = []
    for dependency in dependencies:
        version = importlib.metadata.version(dependency["name"])
        if version != dependency["version"]:
            raise ValueError("actual_runtime_dependency_version_changed")
        versions.append({"name": dependency["name"], "version": version})
    return {"package_version": distribution.version, "site_packages": str(root),
            "wheel_sha256": WHEEL_SHA256, "wheel_bytes": WHEEL_BYTES,
            "installed_package_files": checked, "dependencies": versions,
            "runtime_manifest_sha256": digest(runtime_manifest)}


def _verify_loaded_native(runtime, low_module):
    """Bind actual ctypes DLL and mapped GGML libraries to verified wheel files."""
    root = Path(runtime["site_packages"]).resolve()
    if Path(low_module.__file__).resolve() != root / "llama_cpp/llama_cpp.py":
        raise RuntimeError("actual_native_binding_module_outside_pinned_target")
    libraries = {str((root / entry["path"]).resolve()): entry for entry in runtime["installed_package_files"]
                 if ".so" in Path(entry["path"]).name}
    actual_library = Path(low_module._lib._name).resolve()
    if str(actual_library) not in libraries:
        raise RuntimeError("actual_ctypes_library_outside_pinned_official_wheel")
    basenames = {Path(path).name for path in libraries}
    mapped = {}
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        pieces = line.split(maxsplit=5)
        if len(pieces) != 6:
            continue
        name = pieces[5]
        deleted = name.endswith(" (deleted)")
        clean = name.removesuffix(" (deleted)")
        if Path(clean).name not in basenames:
            continue
        path = Path(clean).resolve()
        if deleted or str(path) not in libraries:
            raise RuntimeError("mapped_native_library_outside_verified_wheel_or_deleted")
        if str(path) in mapped:
            continue
        expected = libraries[str(path)]
        if path.stat().st_size != expected["bytes"] or file_sha(path) != expected["sha256"]:
            raise RuntimeError("mapped_native_library_bytes_changed_after_wheel_verification")
        mapped[str(path)] = expected
    if str(actual_library) not in mapped:
        raise RuntimeError("actual_ctypes_library_missing_from_native_process_maps")
    return {"ctypes_library_path": str(actual_library),
            "mapped_official_native_libraries": [mapped[path] for path in sorted(mapped)],
            "native_mapping_scope": "actual ctypes main DLL plus mapped files sharing any verified wheel native-library basename"}


def load_backend(model_dir, manifest, *, runtime_manifest):
    schema.require_hosted_colab()
    _owned_schema()
    all_setup_started = time.monotonic()
    phase_started = time.monotonic()
    verified = verify_model_artifacts(model_dir, manifest)
    model_verification_seconds = time.monotonic() - phase_started
    phase_started = time.monotonic()
    runtime = _verify_runtime(runtime_manifest)
    runtime_verification_seconds = time.monotonic() - phase_started
    import llama_cpp
    if (Path(llama_cpp.__file__).resolve() != Path(runtime["site_packages"]) / "llama_cpp/__init__.py"
            or llama_cpp.__version__ != PACKAGE_VERSION):
        raise RuntimeError("actual_import_does_not_match_pinned_runtime_target")
    low_module = importlib.import_module("llama_cpp.llama_cpp")
    _verify_loaded_native(runtime, low_module)
    started = time.monotonic()
    engine = llama_cpp.Llama(model_path=str(Path(model_dir).resolve() / MODEL_FILES[0]),
        n_ctx=CONTEXT_TOKENS, n_threads=N_THREADS, n_threads_batch=N_THREADS,
        n_batch=128, n_ubatch=128, n_gpu_layers=0, use_mmap=True, use_mlock=False,
        logits_all=False, embedding=False, seed=0, verbose=False)
    metadata = dict(engine.metadata)
    loaded_native = _verify_loaded_native(runtime, low_module)
    template = metadata.get("tokenizer.chat_template")
    identity = {**verified, "version": VERSION, "llama_cpp_python_version": PACKAGE_VERSION,
                "runtime_identity": runtime,
                "actual_loaded_native_identity": loaded_native,
                "backend_code_sha256": file_sha(__file__),
                "schema_reader_code_sha256": EXPECTED_SCHEMA_SHA,
                "backend_policy_sha256": digest(policy()),
                "native_metadata_sha256": digest(metadata),
                "native_chat_template_sha256": hashlib.sha256(template.encode()).hexdigest()
                    if type(template) is str else None,
                "rendering_uses_explicit_segmented_qwen2_5_chatml": True,
                "n_ctx": CONTEXT_TOKENS, "n_threads": N_THREADS, "n_gpu_layers": 0}
    backend = Backend(engine, llama_cpp, identity)
    receipt = {"version": VERSION, "model_identity": backend.identity,
               "setup_seconds": time.monotonic() - all_setup_started,
               "engine_load_and_identity_seconds": time.monotonic() - started,
               "model_artifact_verification_seconds": model_verification_seconds,
               "runtime_artifact_verification_seconds": runtime_verification_seconds,
               "setup_timing_scope": "model and runtime verification plus import, native model load and identity checks",
               "answer_generations_during_setup": 0, "new_provider_calls": 0,
               "secret_access": False, "local_model_files_only": True}
    return backend, receipt
