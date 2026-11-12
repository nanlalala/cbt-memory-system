"""Prospective local-only GGUF reader; actual model work is hosted Colab only.

The separately preserved CPU v6 gate passed all eight reading fixtures. This
v7 is a distinct CUDA execution condition on one observed hosted Tesla T4, with
the same pinned Q4_K_M artifacts and input4096/output64/context4352 budgets.
No CPU answer is treated as a GPU generation or reused through a GPU cache key.

Official Qwen 7B Q4_K_M is a new model/quantization condition, not a continuation
of the earlier float32 1.5B scores. This adapter neither downloads artifacts nor
admits attempts: a journalled caller must freeze/admit each request first.

The native llama.cpp token vector and sampled token IDs are recorded directly.
Content uses special=False independently of trusted ChatML boundaries, so an
original quotation containing a role marker cannot introduce a new role.
"""
from __future__ import annotations

import base64
import csv
from dataclasses import dataclass
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import schema_reader_v4 as schema

VERSION = "hosted-official-qwen7b-q4km-t4-question-schema-native-reader-v7"
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
WHEEL_BYTES = 1776573061
WHEEL_SHA256 = "3d53f94464d8d69028d993033d6120439c994108c04375a46205bd1935bc30ad"
CUDA_RELEASE_TAG = "v0.3.36-cu124"
CUDA_RELEASE_METADATA_SHA = "e30f7cf9b0fa6d50736abea343a4a53d481e97bb873fce0d6f858ea56d402a42"
EXPECTED_GPU_NAME = "Tesla T4"
EXPECTED_DRIVER = "580.82.07"
EXPECTED_COMPUTE_CAPABILITY = "7.5"
EXPECTED_GPU_MEMORY_MIB = 15360
MINIMUM_OFFLOAD_MEMORY_DELTA_MIB = 2048
CUDA_DEPENDENCY_VERSION = "isolated-cu124-host-driver-path-validation-v9"
CUDA_DEPENDENCY_ROOT = Path("/content/cbt-qwen7b-gpu-python-recovery-20261112-v9")
CUDA_DEPENDENCY_ARTIFACT_ROOT = Path("/content/cbt-qwen7b-gpu-python-recovery-20261112-v8")
CUDA_DEPENDENCY_PACKAGES = (
    {"name": "nvidia-cuda-runtime-cu12", "version": "12.4.127",
     "metadata_sha256": "654aff4bc6655e035d973cc320d484e085cd7af8205ac5016a91932652e3a740",
     "bytes": 883737, "sha256": "64403288fa2136ee8e467cdc9c9427e0434110899d07c779f25b5c068934faa5",
     "library_subdir": "nvidia/cuda_runtime/lib"},
    {"name": "nvidia-cublas-cu12", "version": "12.4.5.8",
     "metadata_sha256": "f6ede528be91742346bb448a8dbb8d78a2ba657eaa3580eaf2680b1a32cdd346",
     "bytes": 363438805, "sha256": "2fc8da60df463fdefa81e323eef2e36489e1c94335b5358bcb38360adf75ac9b",
     "library_subdir": "nvidia/cublas/lib"},
)
REQUIRED_CUDA_LIBRARIES = {"libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12"}
REQUIRED_MAPPED_CUDA_LIBRARIES = {"libcudart.so.12", "libcublas.so.12"}
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
        "cuda_release_tag": CUDA_RELEASE_TAG,
        "official_cuda_release_metadata_sha256": CUDA_RELEASE_METADATA_SHA,
        "isolated_cuda_dependency_versions": [{"name": item["name"], "version": item["version"]}
            for item in CUDA_DEPENDENCY_PACKAGES],
        "actual_execution": "genuine hosted Colab Tesla T4 CUDA only; local artifact paths only",
        "n_ctx": CONTEXT_TOKENS, "maximum_input_tokens": MAX_INPUT_TOKENS,
        "context_allocation_reason": "explicit observed native padding capacity; logical input/output budgets unchanged",
        "maximum_new_tokens": MAX_NEW_TOKENS, "n_threads": N_THREADS,
        "n_gpu_layers": -1, "main_gpu": 0, "offload_kqv": True, "use_mmap": True,
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
        "gpu_proof": "compiled offload support plus observed full-layer CUDA0 constructor log and same-device memory increase",
        "dynamic_gpu_proof_is_outside_stable_cache_identity": True,
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
            verify_gpu_execution_proof(self.gpu_execution_proof,
                expected_condition=self.identity["gpu_execution_condition"])
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


def _stream_member_sha(archive, member):
    result = hashlib.sha256()
    with archive.open(member) as member_stream:
        while block := member_stream.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def _verify_platform_driver(reference):
    if type(reference) is not dict:
        raise ValueError("frozen_colab_platform_driver_reference_required")
    alias = Path(reference["library_path"])
    directory = Path(reference["library_directory"])
    resolved = Path(reference["resolved_path"])
    if (not alias.is_absolute() or alias.name != "libcuda.so.1" or alias.parent != directory
            or not alias.is_file() or alias.resolve() != resolved or not resolved.is_file()
            or resolved.stat().st_size != reference.get("bytes")
            or not re.fullmatch(r"[0-9a-f]{64}", reference.get("sha256", ""))
            or file_sha(resolved) != reference["sha256"]):
        raise ValueError("actual_platform_driver_alias_or_bytes_changed")
    return dict(reference)


def _verify_cuda_dependencies(runtime_manifest):
    """Verify the separate recovery, publisher wheels, and installed CUDA12 SOs."""
    receipt_path = Path(runtime_manifest["cuda_recovery_receipt_path"])
    protocol_path = Path(runtime_manifest["cuda_recovery_protocol_path"])
    manifest_path = Path(runtime_manifest["native_dependency_manifest_path"])
    target = Path(runtime_manifest["dependency_site_packages"]).resolve()
    if (receipt_path != CUDA_DEPENDENCY_ROOT / "setup_receipt.json"
            or protocol_path != CUDA_DEPENDENCY_ROOT / "recovery_protocol.json"
            or manifest_path != CUDA_DEPENDENCY_ROOT / "native_dependency_manifest.json"
            or target != CUDA_DEPENDENCY_ARTIFACT_ROOT / "site-packages"):
        raise ValueError("exact_isolated_cuda_dependency_recovery_paths_required")
    for path, expected in ((receipt_path, runtime_manifest["cuda_recovery_receipt_sha256"]),
                           (protocol_path, runtime_manifest["cuda_recovery_protocol_sha256"]),
                           (manifest_path, runtime_manifest["native_dependency_manifest_sha256"])):
        if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(Path("/content"))
                or not re.fullmatch(r"[0-9a-f]{64}", expected) or file_sha(path) != expected):
            raise ValueError("pinned_cuda_dependency_recovery_bytes_changed")
    receipt = json.loads(receipt_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if (receipt.get("stage") != "isolated_cuda12_driver_path_ready_no_model_loaded"
            or receipt.get("protocol_sha256") != runtime_manifest["cuda_recovery_protocol_sha256"]
            or receipt.get("native_dependency_manifest_sha256") != runtime_manifest["native_dependency_manifest_sha256"]
            or receipt.get("compiled_gpu_offload_support") is not True
            or receipt.get("model_generations") != 0 or receipt.get("old_failed_setup_preserved") is not True
            or protocol.get("version") != CUDA_DEPENDENCY_VERSION
            or manifest.get("version") != CUDA_DEPENDENCY_VERSION
            or protocol.get("base_environment_modified") is not False
            or manifest.get("dependency_site_packages") != str(target)
            or manifest.get("llama_cpp_site_packages") != runtime_manifest["site_packages"]):
        raise ValueError("isolated_cuda_dependency_recovery_contract_changed")
    driver = _verify_platform_driver(runtime_manifest.get("driver_reference"))
    if driver != manifest.get("driver_reference") or driver != receipt.get("runtime_manifest_extension", {}).get("driver_reference"):
        raise ValueError("same_frozen_platform_driver_reference_required")
    wheel_directory = Path(runtime_manifest["wheel_directory"])
    if wheel_directory != CUDA_DEPENDENCY_ARTIFACT_ROOT or manifest.get("wheel_directory") != str(wheel_directory):
        raise ValueError("verified_original_nvidia_wheel_directory_required")
    expected_ld = ":".join([str(target / item["library_subdir"]) for item in CUDA_DEPENDENCY_PACKAGES] +
                           [str(Path(runtime_manifest["site_packages"]) / "llama_cpp/lib"), driver["library_directory"]])
    if (runtime_manifest.get("child_ld_library_path") != expected_ld
            or manifest.get("child_ld_library_path") != expected_ld
            or protocol.get("child_ld_library_path") != expected_ld
            or os.environ.get("LD_LIBRARY_PATH") != expected_ld):
        raise ValueError("new_child_process_requires_exact_isolated_cuda_ld_library_path")
    wheels = manifest.get("wheels")
    if wheels != protocol.get("wheels") or type(wheels) is not list or len(wheels) != 2:
        raise ValueError("two_publisher_pinned_nvidia_dependency_wheels_required")
    expected_by_name = {item["name"]: item for item in CUDA_DEPENDENCY_PACKAGES}
    installed = {}
    wheel_files = []
    for entry in wheels:
        spec = expected_by_name.get(entry.get("name"))
        if spec is None or any(entry.get(key) != value for key, value in spec.items()):
            raise ValueError("fixed_nvidia_dependency_version_or_digest_changed")
        filename = entry.get("filename")
        if type(filename) is not str or Path(filename).name != filename:
            raise ValueError("owned_publisher_nvidia_wheel_filename_required")
        wheel = wheel_directory / filename
        if wheel.is_symlink() or not wheel.is_file() or wheel.stat().st_size != spec["bytes"] or file_sha(wheel) != spec["sha256"]:
            raise ValueError("publisher_nvidia_dependency_wheel_bytes_changed")
        wheel_files.append({"path": str(wheel), "bytes": spec["bytes"], "sha256": spec["sha256"]})
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.infolist():
                relative = Path(member.filename)
                if member.is_dir() or relative.parent.as_posix() != spec["library_subdir"] or ".so" not in relative.name:
                    continue
                path = target / relative
                reference = _stream_member_sha(archive, member)
                if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(target)
                        or path.stat().st_size != member.file_size or file_sha(path) != reference):
                    raise ValueError("installed_cuda_shared_library_differs_from_publisher_wheel")
                installed[str(path)] = {"path": str(path), "bytes": member.file_size, "sha256": reference}
    recorded = manifest.get("installed_library_files")
    if (type(recorded) is not list or {item.get("path"): item for item in recorded} != installed
            or not REQUIRED_CUDA_LIBRARIES.issubset({Path(path).name for path in installed})):
        raise ValueError("all_three_installed_cuda12_library_identities_required")
    metadata = protocol.get("nvidia_pypi_metadata")
    if (type(metadata) is not list or len(metadata) != 2
            or {item.get("sha256") for item in metadata} != {item["metadata_sha256"] for item in CUDA_DEPENDENCY_PACKAGES}):
        raise ValueError("both_publisher_nvidia_metadata_files_required")
    for item in metadata:
        if file_sha(item["path"]) != item["sha256"] or item["sha256"] not in {s["metadata_sha256"] for s in CUDA_DEPENDENCY_PACKAGES}:
            raise ValueError("publisher_nvidia_dependency_metadata_changed")
    return {"recovery_receipt_sha256": runtime_manifest["cuda_recovery_receipt_sha256"],
            "recovery_protocol_sha256": runtime_manifest["cuda_recovery_protocol_sha256"],
            "native_dependency_manifest_sha256": runtime_manifest["native_dependency_manifest_sha256"],
            "dependency_site_packages": str(target), "child_ld_library_path": expected_ld,
            "driver_reference": driver,
            "wheels": wheel_files, "installed_library_files": [installed[path] for path in sorted(installed)]}


def _verify_runtime(runtime_manifest):
    """Bind installed executable package files to the pinned official wheel."""
    if (type(runtime_manifest) is not dict
            or runtime_manifest.get("package_name") != "llama-cpp-python"
            or runtime_manifest.get("package_version") != PACKAGE_VERSION
            or runtime_manifest.get("backend") != "cuda"
            or runtime_manifest.get("cuda_release_tag") != CUDA_RELEASE_TAG
            or runtime_manifest.get("official_release_metadata_sha256") != CUDA_RELEASE_METADATA_SHA
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
    cuda_dependencies = _verify_cuda_dependencies(runtime_manifest)
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
            reference = _stream_member_sha(archive, member)
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
            "backend": "cuda", "cuda_release_tag": CUDA_RELEASE_TAG,
            "wheel_sha256": WHEEL_SHA256, "wheel_bytes": WHEEL_BYTES,
            "installed_package_files": checked, "dependencies": versions,
            "native_dependency_identity": cuda_dependencies,
            "runtime_manifest_sha256": digest(runtime_manifest)}


def _verify_loaded_native(runtime, low_module, *, require_cuda=False):
    """Bind actual ctypes DLL and mapped GGML libraries to verified wheel files."""
    root = Path(runtime["site_packages"]).resolve()
    if Path(low_module.__file__).resolve() != root / "llama_cpp/llama_cpp.py":
        raise RuntimeError("actual_native_binding_module_outside_pinned_target")
    libraries = {str((root / entry["path"]).resolve()): entry for entry in runtime["installed_package_files"]
                 if ".so" in Path(entry["path"]).name}
    for entry in runtime.get("native_dependency_identity", {}).get("installed_library_files", []):
        libraries[str(Path(entry["path"]).resolve())] = entry
    driver = runtime.get("native_dependency_identity", {}).get("driver_reference")
    if driver is not None:
        driver = _verify_platform_driver(driver)
        libraries[str(Path(driver["resolved_path"]).resolve())] = {
            "path": driver["resolved_path"], "bytes": driver["bytes"], "sha256": driver["sha256"]}
    actual_library = Path(low_module._lib._name).resolve()
    if str(actual_library) not in libraries:
        raise RuntimeError("actual_ctypes_library_outside_pinned_official_wheel")
    basenames = {Path(path).name for path in libraries}
    if driver is not None:
        basenames.add(Path(driver["library_path"]).name)
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
    if require_cuda and not REQUIRED_MAPPED_CUDA_LIBRARIES.issubset({Path(path).name for path in mapped}):
        raise RuntimeError("actual_pinned_cudart_and_cublas_must_be_mapped")
    if require_cuda and (driver is None or driver["resolved_path"] not in mapped):
        raise RuntimeError("actual_frozen_colab_gpu_driver_must_be_mapped")
    return {"ctypes_library_path": str(actual_library),
            "mapped_official_native_libraries": [mapped[path] for path in sorted(mapped)
                if driver is None or path != driver["resolved_path"]],
            "mapped_platform_gpu_driver": driver if driver is not None and driver["resolved_path"] in mapped else None,
            "native_mapping_scope": "actual ctypes main DLL plus mapped files sharing any verified wheel native-library basename"}


def query_gpu():
    """One bounded metadata query; this never evaluates a model or samples."""
    schema.require_hosted_colab()
    fields = "index,name,uuid,driver_version,memory.total,memory.used,compute_cap"
    completed = subprocess.run(["nvidia-smi", "--query-gpu=" + fields,
        "--format=csv,noheader,nounits"], capture_output=True, text=True,
        timeout=10, check=False)
    if completed.returncode != 0:
        raise RuntimeError("bounded_gpu_metadata_query_failed_no_retry")
    rows = list(csv.reader(completed.stdout.strip().splitlines()))
    if len(rows) != 1 or len(rows[0]) != 7:
        raise ValueError("exact_one_observed_t4_gpu_required")
    values = [value.strip() for value in rows[0]]
    try:
        gpu = {"index": int(values[0]), "name": values[1], "uuid": values[2],
               "driver_version": values[3], "memory_total_mib": int(values[4]),
               "memory_used_mib": int(values[5]), "compute_capability": values[6]}
    except ValueError as exc:
        raise ValueError("gpu_metadata_must_be_exact_native_numbers") from exc
    gpu_condition(gpu)
    return gpu


def gpu_condition(gpu):
    """Only stable execution facts enter the model/cache identity."""
    if (type(gpu) is not dict or gpu.get("index") != 0
            or type(gpu.get("index")) is not int
            or gpu.get("name") != EXPECTED_GPU_NAME
            or gpu.get("driver_version") != EXPECTED_DRIVER
            or gpu.get("compute_capability") != EXPECTED_COMPUTE_CAPABILITY
            or gpu.get("memory_total_mib") != EXPECTED_GPU_MEMORY_MIB
            or type(gpu.get("memory_total_mib")) is not int
            or type(gpu.get("memory_used_mib")) is not int
            or not 0 <= gpu["memory_used_mib"] <= gpu["memory_total_mib"]
            or type(gpu.get("uuid")) is not str or not gpu["uuid"].startswith("GPU-")):
        raise ValueError("exact_observed_t4_hardware_condition_required")
    return {key: gpu[key] for key in ("index", "name", "uuid", "driver_version",
                                     "memory_total_mib", "compute_capability")} | {
        "n_gpu_layers": -1, "main_gpu": 0, "offload_kqv": True,
        "compiled_gpu_offload_supported": True, "all_layers_offloaded": True}


def _constructor_log(path):
    path = Path(path)
    if (not path.is_absolute() or not path.resolve().is_relative_to(Path("/content"))
            or path.is_symlink() or not path.is_file()
            or not stat.S_ISREG(os.fstat(2).st_mode)
            or Path(os.readlink("/proc/self/fd/2")).resolve() != path.resolve()
            or os.fstat(2).st_ino != path.stat().st_ino
            or os.fstat(2).st_dev != path.stat().st_dev):
        raise ValueError("actual_stderr_must_be_directly_redirected_to_owned_regular_constructor_log")
    sys.stderr.flush()
    return path.resolve(), path.stat().st_size


def _offload_counts(data):
    text = data.decode("utf-8", errors="replace")
    counts = re.findall(r"\boffloaded\s+(\d+)\s*/\s*(\d+)\s+layers\s+to\s+GPU\b", text, re.I)
    if len(counts) != 1 or re.search(r"\bCUDA0\b", text, re.I) is None:
        raise ValueError("one_actual_full_layer_cuda0_offload_log_required")
    return tuple(int(value) for value in counts[0])


def _read_log_range(log, start, end):
    if (type(start) is not int or type(end) is not int or start < 0
            or not start < end or end - start > 16 * 1024 * 1024):
        raise ValueError("bounded_complete_constructor_log_range_required")
    path = Path(log)
    if path.is_symlink() or not path.is_file() or path.stat().st_size < end:
        raise ValueError("original_regular_constructor_log_bytes_required")
    with path.open("rb") as stream:
        stream.seek(start)
        data = stream.read(end - start)
    if len(data) != end - start:
        raise ValueError("complete_constructor_log_range_required")
    return data


def verify_gpu_execution_proof(proof, *, expected_condition=None, constructor_log_path=None):
    """Metadata/log audit only; no native load, GPU query, tokenizer or model."""
    if (type(proof) is not dict or proof.get("proof_passed") is not True
            or proof.get("compiled_gpu_offload_supported") is not True
            or proof.get("all_layers_offloaded") is not True):
        raise ValueError("actual_full_gpu_execution_proof_required")
    before, after = proof["before_load_gpu"], proof["after_load_gpu"]
    condition = gpu_condition(before)
    if condition != gpu_condition(after) or proof.get("gpu_condition") != condition:
        raise ValueError("gpu_device_condition_changed_during_setup")
    if expected_condition is not None and condition != expected_condition:
        raise ValueError("gpu_proof_does_not_match_frozen_execution_condition")
    delta = after["memory_used_mib"] - before["memory_used_mib"]
    if (type(proof.get("memory_delta_mib")) is not int or proof["memory_delta_mib"] != delta
            or delta < MINIMUM_OFFLOAD_MEMORY_DELTA_MIB):
        raise ValueError("actual_model_gpu_memory_increase_required")
    blocks = proof.get("model_block_count")
    if (type(blocks) is not int or blocks < 1
            or proof.get("offloaded_layers") != blocks + 1
            or proof.get("total_layers") != blocks + 1):
        raise ValueError("all_model_blocks_and_output_layer_must_be_offloaded")
    log = proof["constructor_log"]
    if log.get("stderr_fd_verified") is not True:
        raise ValueError("actual_stderr_capture_evidence_required")
    data = _read_log_range(constructor_log_path or log["path"], log["start_byte"], log["end_byte"])
    if (log.get("bytes") != len(data)
            or log.get("sha256") != hashlib.sha256(data).hexdigest()
            or _offload_counts(data) != (proof["offloaded_layers"], proof["total_layers"])):
        raise ValueError("gpu_offload_proof_log_bytes_changed")
    return True


def load_backend(model_dir, manifest, *, runtime_manifest, constructor_log_path):
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
    supported = bool(llama_cpp.llama_supports_gpu_offload())
    if not supported:
        raise RuntimeError("actual_cuda_backend_gpu_offload_support_required")
    before_gpu = query_gpu()
    if before_gpu["memory_total_mib"] - before_gpu["memory_used_mib"] < 6144:
        raise RuntimeError("observed_free_t4_memory_at_least_6GiB_required")
    log_path, log_start = _constructor_log(constructor_log_path)
    started = time.monotonic()
    engine = llama_cpp.Llama(model_path=str(Path(model_dir).resolve() / MODEL_FILES[0]),
        n_ctx=CONTEXT_TOKENS, n_threads=N_THREADS, n_threads_batch=N_THREADS,
        n_batch=128, n_ubatch=128, n_gpu_layers=-1, main_gpu=0, offload_kqv=True,
        use_mmap=True, use_mlock=False, logits_all=False, embedding=False, seed=0, verbose=True)
    sys.stderr.flush()
    os.fsync(2)
    log_end = log_path.stat().st_size
    after_gpu = query_gpu()
    metadata = dict(engine.metadata)
    loaded_native = _verify_loaded_native(runtime, low_module, require_cuda=True)
    template = metadata.get("tokenizer.chat_template")
    raw_blocks = metadata.get("qwen2.block_count")
    if type(raw_blocks) not in (str, int) or not str(raw_blocks).isdigit():
        raise ValueError("verified_qwen_metadata_block_count_required")
    log_bytes = _read_log_range(log_path, log_start, log_end)
    offloaded, total = _offload_counts(log_bytes)
    condition = gpu_condition(before_gpu)
    proof = {"proof_passed": True, "compiled_gpu_offload_supported": supported,
        "all_layers_offloaded": offloaded == total == int(raw_blocks) + 1,
        "model_block_count": int(raw_blocks), "offloaded_layers": offloaded, "total_layers": total,
        "before_load_gpu": before_gpu, "after_load_gpu": after_gpu,
        "memory_delta_mib": after_gpu["memory_used_mib"] - before_gpu["memory_used_mib"],
        "gpu_condition": condition,
        "constructor_log": {"path": str(log_path), "start_byte": log_start, "end_byte": log_end,
            "bytes": len(log_bytes), "sha256": hashlib.sha256(log_bytes).hexdigest(),
            "stderr_fd_verified": True}}
    verify_gpu_execution_proof(proof, expected_condition=condition)
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
                "n_ctx": CONTEXT_TOKENS, "n_threads": N_THREADS,
                "gpu_execution_condition": condition, "n_gpu_layers": -1}
    backend = Backend(engine, llama_cpp, identity)
    backend.gpu_execution_proof = proof
    receipt = {"version": VERSION, "model_identity": backend.identity,
               "gpu_execution_proof": proof,
               "setup_seconds": time.monotonic() - all_setup_started,
               "engine_load_and_identity_seconds": time.monotonic() - started,
               "model_artifact_verification_seconds": model_verification_seconds,
               "runtime_artifact_verification_seconds": runtime_verification_seconds,
               "setup_timing_scope": "model and runtime verification plus import, native model load and identity checks",
               "answer_generations_during_setup": 0, "new_provider_calls": 0,
               "secret_access": False, "local_model_files_only": True}
    return backend, receipt
