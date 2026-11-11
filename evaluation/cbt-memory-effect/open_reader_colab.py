"""Bounded CPU open-model reader with source gates and a global attempt journal.

The caller supplies already loaded model/tokenizer objects and an explicit gate
adapter. No credential, model download, assessor target, score or remote model
API is accessed here. Real execution requires genuine hosted Colab; local unit
tests use only small invented requests and fake model/tokenizer objects.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

VERSION = "controlled-open-reader-colab-cpu-v1"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
ARMS = {"raw_history_rag", "managed_memory_rag", "short_term"}
MAX_GENERATIONS = 64
MAX_INPUT_TOKENS = 4096
MAX_NEW_TOKENS = 64


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require_colab():
    try:
        valid = (bool(os.environ.get("COLAB_RELEASE_TAG")) and Path("/content").is_dir()
                 and importlib.util.find_spec("google.colab") is not None)
    except (ImportError, ValueError, AttributeError):
        valid = False
    if not valid:
        raise RuntimeError("genuine_hosted_colab_required")


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate_json_key")
        value[key] = item
    return value


def read_json(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular_json_file_required")
    return json.loads(path.read_text(), object_pairs_hook=_unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular_jsonl_file_required")
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise RuntimeError("incomplete_journal_tail_no_automatic_retry")
    return [json.loads(line, object_pairs_hook=_unique) for line in data.splitlines() if line.strip()]


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _append(path, value):
    record = {**value, "record_sha256": digest(value)}
    with Path(path).open("a") as stream:
        stream.write(canonical(record) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return record


def _checked(record):
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    if record.get("record_sha256") != digest(body):
        raise ValueError("journal_checksum_changed")
    return record


def stamp():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _locked(path):
    with Path(path).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _requests(prepared_dir):
    protocol_path = prepared_dir / "protocol.json"
    requests_path = prepared_dir / "requests.jsonl"
    protocol = read_json(protocol_path)
    requests = read_rows(requests_path)
    if not requests or len(requests) > 10000:
        raise ValueError("bounded_frozen_requests_required")
    actual_hash = file_hash(requests_path)
    declared = protocol.get("requests_file_sha256")
    if declared is not None and declared != actual_hash:
        raise ValueError("requests_file_changed")
    if protocol.get("requests_sha256") not in {None, actual_hash, digest(requests)}:
        raise ValueError("requests_changed")
    if declared is None and protocol.get("requests_sha256") is None:
        raise ValueError("prepared_protocol_must_bind_requests")
    seen = set()
    for row in requests:
        if not isinstance(row, dict) or row.get("arm") not in ARMS:
            raise ValueError("invalid_request_row")
        identity = (row.get("case_id"), row["arm"])
        if not isinstance(identity[0], str) or not identity[0] or identity in seen:
            raise ValueError("duplicate_or_missing_logical_request")
        seen.add(identity)
        messages, payload = row.get("messages"), row.get("request_payload")
        forbidden = {"gold", "answer", "expected", "expected_state", "targets",
                     "reference_response", "correct_answer", "score"}
        if forbidden.intersection(row) or (isinstance(payload, dict) and forbidden.intersection(payload)):
            raise ValueError("assessor_fields_prohibited_in_generation_request")
        if (not isinstance(messages, list) or not messages or not isinstance(payload, dict)
                or messages != payload.get("messages") or digest(payload) != row.get("request_sha256")):
            raise ValueError("request_payload_binding_changed")
        if type(payload.get("max_tokens")) is not int or not 1 <= payload["max_tokens"] <= MAX_NEW_TOKENS:
            raise ValueError("request_output_limit_changed")
        for message in messages:
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message["role"] not in {"system", "user", "assistant"}
                    or type(message["content"]) is not str):
                raise ValueError("invalid_message_schema")
    return protocol, requests, actual_hash


def _model_identity(model_root, model, tokenizer, *, fixture_only):
    protocol = read_json(model_root / "load_protocol.json")
    receipt = read_json(model_root / "model_load_receipt.json")
    files = read_json(model_root / "model_files.json")
    if (protocol.get("model_id") != MODEL_ID or protocol.get("model_revision") != MODEL_REVISION
            or protocol.get("maximum_generated_answers") != MAX_GENERATIONS
            or protocol.get("max_new_tokens") != MAX_NEW_TOKENS
            or protocol.get("maximum_input_tokens") != MAX_INPUT_TOKENS
            or protocol.get("sampling") is not False or protocol.get("external_inference_calls") != 0
            or receipt.get("stage") != "open_reader_model_loaded"
            or receipt.get("model_revision") != MODEL_REVISION
            or receipt.get("device") != "cpu" or receipt.get("dtype") != "torch.float32"
            or files.get("revision") != MODEL_REVISION):
        raise ValueError("fixed_cpu_model_identity_changed")
    if (str(model.device) != "cpu" or str(next(model.parameters()).dtype) != "torch.float32"
            or model.training is not False):
        raise ValueError("cpu_float32_model_required")
    if not fixture_only:
        # The loader already hashes every exact source file before loading. Bind
        # both objects to that immutable HF snapshot and recheck its config now.
        model_path = Path(model.config._name_or_path).resolve(strict=True)
        tokenizer_path = Path(tokenizer.name_or_path).resolve(strict=True)
        if model_path != tokenizer_path or model_path.name != MODEL_REVISION:
            raise ValueError("objects_do_not_bind_pinned_snapshot")
        catalog = {item["name"]: item for item in files["files"]}
        if file_hash(model_path / "config.json") != catalog["config.json"]["sha256"]:
            raise ValueError("model_config_changed")
    return {"model_id": MODEL_ID, "model_revision": MODEL_REVISION,
            "generation_config_sha256": digest(model.generation_config.to_dict()),
            "chat_template_sha256": digest(tokenizer.chat_template),
            "load_protocol_sha256": file_hash(model_root / "load_protocol.json"),
            "model_files_sha256": file_hash(model_root / "model_files.json"),
            "load_receipt_sha256": file_hash(model_root / "model_load_receipt.json")}


def _generate(model, tokenizer, messages, *, max_new_tokens):
    import torch
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt")
    if any(getattr(value, "device", "cpu") != model.device for value in inputs.values()):
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
    input_count = int(inputs["input_ids"].shape[-1])
    if input_count > MAX_INPUT_TOKENS:
        raise ValueError("input_token_limit_exceeded_no_truncation")
    input_digest = digest(inputs["input_ids"].tolist())
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens,
            do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    generated = output[0][input_count:]
    ids = generated.tolist()
    text = tokenizer.decode(generated, skip_special_tokens=True)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
    return {"response_text": text, "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "usage": {"prompt_tokens": input_count, "completion_tokens": len(ids),
                      "total_tokens": input_count + len(ids)},
            "input_tensor_sha256": input_digest,
            "finish_reason": "eos" if ids and ids[-1] in eos else "length"}


def _global_state(model_root):
    intents, outcomes = {}, {}
    for raw in read_rows(model_root / "generation_ledger.jsonl"):
        row = _checked(raw)
        key = row["canonical_key"]
        target = intents if row["event"] == "generation_intent" else outcomes if row["event"] == "generation_outcome" else None
        if target is None or key in target:
            raise ValueError("duplicate_or_unknown_global_generation_event")
        target[key] = row
    if set(outcomes) - set(intents) or len(intents) > MAX_GENERATIONS:
        raise ValueError("unbound_outcome_or_global_budget_changed")
    if set(intents) != set(outcomes) or any(row["status"] != "published" for row in outcomes.values()):
        raise RuntimeError("interrupted_failed_or_stale_generation_no_automatic_retry")
    return intents, outcomes


def _cache(outcome):
    path = Path(outcome["cache_path"])
    if path.name != outcome["canonical_key"] + ".json" or path.parent.name != "canonical":
        raise ValueError("unexpected_canonical_cache_path")
    value = read_json(path)
    if file_hash(path) != outcome["cache_sha256"] or value.get("canonical_key") != outcome["canonical_key"]:
        raise ValueError("canonical_cache_changed")
    text = value.get("response_text")
    if type(text) is not str or hashlib.sha256(text.encode()).hexdigest() != value.get("response_sha256"):
        raise ValueError("response_cache_binding_changed")
    if any(value.get(key) != outcome.get(key) for key in ("response_sha256", "usage", "finish_reason")):
        raise ValueError("canonical_cache_receipt_disagreement")
    return value


def _cost(outcomes, keys):
    selected = [outcomes[key] for key in keys]
    observed, missing = {}, {}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [(row.get("usage") or {}).get(field) for row in selected]
        good = [value for value in values if type(value) is int and value >= 0]
        observed[field] = sum(good) if good or not selected else None
        missing[field] = len(values) - len(good)
    timings = [row.get("generation_wall_seconds") for row in selected]
    valid_times = [value for value in timings if type(value) in (int, float) and value >= 0]
    return {"canonical_attempts": len(selected), "observed_tokens": observed,
            "usage_missing_attempts": missing,
            "generation_wall_seconds_sum": sum(valid_times) if valid_times or not selected else None,
            "missing_wall_time_attempts": len(timings) - len(valid_times),
            "external_inference_calls": 0}


def run_open_reader(prepared_dir, model, tokenizer, model_root, output, *,
                    max_new_generations=MAX_GENERATIONS, gates=None, fixture_only=False):
    """Generate frozen logical rows; source checks are mandatory even on reuse.

    Gate interface: identity(), preflight(row), begin(row), and
    finish(row, admission, outcome, response, usage). The gate owns authoritative
    source/task/withdrawal checks and durable admission. A replayed successful
    admission is reusable only with our existing exact canonical receipt.
    """
    if not fixture_only:
        require_colab()
    if gates is None or any(not callable(getattr(gates, name, None))
                            for name in ("identity", "preflight", "begin", "finish")):
        raise ValueError("explicit_source_gate_adapter_required")
    if type(max_new_generations) is not int or not 0 <= max_new_generations <= MAX_GENERATIONS:
        raise ValueError("finite_invocation_generation_cap_required")
    prepared_dir, model_root, output = (Path(p).resolve() for p in (prepared_dir, model_root, output))
    if not fixture_only and any(not p.is_relative_to(Path("/content")) for p in (prepared_dir, model_root, output)):
        raise ValueError("hosted_content_paths_required")
    protocol, requests, requests_hash = _requests(prepared_dir)
    if protocol.get("fixture_only") is not None and protocol["fixture_only"] is not fixture_only:
        raise ValueError("fixture_and_real_preparation_identity_changed")
    model_identity = _model_identity(model_root, model, tokenizer, fixture_only=fixture_only)
    gate_identity = gates.identity()
    if type(gate_identity) is not dict or not gate_identity:
        raise ValueError("frozen_gate_identity_required")
    run_protocol = {"version": VERSION, "fixture_only": fixture_only,
        "prepared_dir": str(prepared_dir), "prepared_protocol_sha256": file_hash(prepared_dir / "protocol.json"),
        "requests_file_sha256": requests_hash, "model": model_identity,
        "gate_identity": gate_identity, "maximum_global_generations": MAX_GENERATIONS,
        "maximum_input_tokens": MAX_INPUT_TOKENS, "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False, "use_cache": True, "external_inference_calls": 0, "secret_access": False,
        "new_human_reviews": 0, "scoring_targets_read": False,
        "reader_code_sha256": file_hash(__file__)}
    if output.exists():
        if read_json(output / "run_protocol.json") != run_protocol:
            raise ValueError("new_run_directory_required_after_change")
    else:
        output.mkdir(parents=True, mode=0o700)
        _write_new(output / "run_protocol.json", run_protocol)
    run_hash = digest(run_protocol)
    with _locked(model_root / ".generation.lock"):
        intents, outcomes = _global_state(model_root)
        prior = [_checked(row) for row in read_rows(output / "answers_unscored.jsonl")]
        seen = {(row["case_id"], row["arm"]) for row in prior}
        if len(seen) != len(prior) or any(row["run_protocol_sha256"] != run_hash for row in prior):
            raise ValueError("duplicate_or_foreign_logical_answer")
        expected = {(row["case_id"], row["arm"]) for row in requests}
        if seen - expected:
            raise ValueError("logical_answer_not_in_frozen_grid")
        generated_now = 0
        invocation_keys = set()
        for row in requests:
            logical_id = (row["case_id"], row["arm"])
            if logical_id in seen:
                continue
            key = digest({"model": model_identity, "payload": row["request_payload"],
                          "max_new_tokens": row["request_payload"]["max_tokens"], "do_sample": False})
            cached = outcomes.get(key)
            if cached is None and (generated_now >= max_new_generations or len(intents) >= MAX_GENERATIONS):
                break
            gates.preflight(row)
            admission = gates.begin(row)
            replayed = bool(admission.get("replayed"))
            if replayed and (admission.get("status") != "succeeded" or cached is None):
                raise RuntimeError("durable_attempt_not_safely_reusable")
            started = time.monotonic()
            result = None
            finalized = replayed
            global_intent = None
            try:
                if cached is not None:
                    result = _cache(cached)
                else:
                    global_intent = _append(model_root / "generation_ledger.jsonl", {
                        "event": "generation_intent", "canonical_key": key,
                        "request_sha256": row["request_sha256"], "run_protocol_sha256": run_hash,
                        "output": str(output), "model_revision": MODEL_REVISION,
                        "started_at_utc": stamp(), "attempt_number": len(intents) + 1})
                    intents[key] = global_intent
                    generated_now += 1
                    invocation_keys.add(key)
                    gates.preflight(row)
                    result = _generate(model, tokenizer, row["messages"],
                                       max_new_tokens=row["request_payload"]["max_tokens"])
                gates.preflight(row)
                if not replayed:
                    finished = gates.finish(row, admission, outcome="success", response=None,
                                            usage=result["usage"])
                    finalized = True
                    if not isinstance(finished, dict) or finished.get("publishable") is not True:
                        raise PermissionError("source_changed_before_publication")
                gates.preflight(row)
                if cached is None:
                    result = {**result, "canonical_key": key}
                    cache_path = output / "canonical" / (key + ".json")
                    _write_new(cache_path, result)
                    cached = _append(model_root / "generation_ledger.jsonl", {
                        "event": "generation_outcome", "canonical_key": key,
                        "status": "published", "cache_path": str(cache_path),
                        "cache_sha256": file_hash(cache_path), "response_sha256": result["response_sha256"],
                        "usage": result["usage"], "finish_reason": result["finish_reason"],
                        "generation_wall_seconds": time.monotonic() - started,
                        "finished_at_utc": stamp()})
                    outcomes[key] = cached
                answer = _append(output / "answers_unscored.jsonl", {
                    "run_protocol_sha256": run_hash, "case_id": row["case_id"], "arm": row["arm"],
                    "request_sha256": row["request_sha256"], "canonical_key": key,
                    "canonical_generation_shared": global_intent is None,
                    "response_text": result["response_text"], "response_sha256": result["response_sha256"],
                    "finish_reason": result["finish_reason"], "source_checks_passed": True,
                    "scored": False, "status": "completed"})
                seen.add(logical_id)
                print(canonical({"stage": "open_reader_answer_recorded", "case_id": row["case_id"],
                    "arm": row["arm"], "logical_completed": len(seen), "logical_planned": len(requests),
                    "global_generated_attempts": len(intents), "shared": answer["canonical_generation_shared"],
                    "new_provider_calls": 0}), flush=True)
            except BaseException as error:
                usage = result.get("usage") if isinstance(result, dict) else None
                if not finalized:
                    try:
                        gates.finish(row, admission, outcome="failed" if result else "unknown",
                                     response=None, usage=usage)
                    except BaseException:
                        pass
                if global_intent is not None and key not in outcomes:
                    _append(model_root / "generation_ledger.jsonl", {
                        "event": "generation_outcome", "canonical_key": key,
                        "status": "stale_discarded" if result is not None else "generation_failed_or_unknown",
                        "response_text": None, "usage": usage,
                        "observed_wall_seconds": time.monotonic() - started,
                        "failure_type": type(error).__name__, "finished_at_utc": stamp()})
                _append(output / "failures.jsonl", {"case_id": row["case_id"], "arm": row["arm"],
                    "request_sha256": row["request_sha256"], "failure_type": type(error).__name__,
                    "automatic_retry": False, "response_text": None})
                raise
        summary = {"version": VERSION, "run_protocol_sha256": run_hash,
            "logical_planned": len(requests), "logical_completed": len(seen),
            "generated_attempts_this_invocation": generated_now,
            "global_generated_attempts": len(intents), "maximum_global_generations": MAX_GENERATIONS,
            "complete": seen == expected, "new_provider_calls": 0, "new_human_reviews": 0,
            "secret_access": False, "effect_scores": None, "fixture_only": fixture_only,
            "answers_sha256": file_hash(output / "answers_unscored.jsonl") if seen else None}
        completed_rows = read_rows(output / "answers_unscored.jsonl")
        referenced_keys = {row["canonical_key"] for row in completed_rows}
        owning_keys = {key for key, row in intents.items() if row.get("output") == str(output)}
        summary.update(generation_cost_this_invocation=_cost(outcomes, invocation_keys),
            generation_cost_owning_run=_cost(outcomes, owning_keys),
            referenced_canonical_cost_nonadditive=_cost(outcomes, referenced_keys),
            reused_canonical_receipts_from_other_runs=len(referenced_keys - owning_keys),
            attempt_counter_semantics="admitted intents; failed tokenization is not a confirmed model.generate call")
        _append(output / "invocations.jsonl", summary)
        if seen == expected:
            target = output / "completion_summary.json"
            if target.exists():
                if read_json(target)["answers_sha256"] != summary["answers_sha256"]:
                    raise ValueError("completed_answers_changed")
                return read_json(target)
            _write_new(target, summary)
        return summary


def _gate_module():
    """Resolve the planner from this checkout, never a notebook's cached import."""
    folder = Path(__file__).resolve().parent
    sys.path.insert(0, str(folder))
    module = importlib.import_module("controlled_open_memory")
    if Path(module.__file__).resolve() != folder / "controlled_open_memory.py":
        raise RuntimeError("foreign_controlled_open_memory_module")
    return module


def _load_cached_model(model_root):
    """Offline CPU reload with separate setup receipts, never a new download."""
    require_colab()
    model_root = Path(model_root).resolve(strict=True)
    protocol = read_json(model_root / "load_protocol.json")
    catalog = read_json(model_root / "model_files.json")
    if (protocol.get("model_id") != MODEL_ID or protocol.get("model_revision") != MODEL_REVISION
            or catalog.get("revision") != MODEL_REVISION):
        raise ValueError("pinned_model_setup_identity_changed")
    started = time.monotonic()
    setup_id = digest([str(model_root), stamp(), os.getpid()])
    events = model_root / "model_load_events.jsonl"
    _append(events, {"event": "model_reload_intent", "setup_id": setup_id,
        "at_utc": stamp(), "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "reader_code_sha256": file_hash(__file__), "process_id": os.getpid(),
        "local_files_only": True, "new_provider_calls": 0, "secret_access": False})
    print(canonical({"stage": "offline_cpu_model_reload_started", "setup_id": setup_id,
                     "new_provider_calls": 0, "weights_loaded": False}), flush=True)
    stage = "offline_cache_resolution"
    try:
        # These switches apply only to this fresh child. The explicit token=False
        # and local_files_only flags remain required at the library boundary.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        from huggingface_hub import snapshot_download
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION,
            local_files_only=True, token=False,
            allow_patterns=["*.json", "*.safetensors", "tokenizer*", "vocab.json",
                            "merges.txt", "LICENSE", "README.md"]))
        if snapshot.name != MODEL_REVISION or not snapshot.is_dir():
            raise ValueError("cached_snapshot_revision_changed")
        stage = "streaming_source_integrity"
        expected = catalog.get("files")
        if not isinstance(expected, list) or not expected:
            raise ValueError("pinned_model_file_catalog_required")
        names = set()
        total = 0
        for item in expected:
            name = item.get("name")
            if (type(name) is not str or Path(name).name != name or name in names
                    or type(item.get("bytes")) is not int or item["bytes"] < 0):
                raise ValueError("invalid_model_file_catalog_entry")
            names.add(name)
            file = snapshot / name
            if not file.is_file() or file.stat().st_size != item["bytes"] or file_hash(file) != item.get("sha256"):
                raise ValueError("cached_model_file_integrity_changed")
            total += item["bytes"]
        if "config.json" not in names or not any(name.endswith(".safetensors") for name in names) or not 0 < total < 4_000_000_000:
            raise ValueError("bounded_complete_model_catalog_required")
        print(canonical({"stage": "cached_model_bytes_verified", "setup_id": setup_id,
                         "files": len(expected), "bytes": total, "new_provider_calls": 0}), flush=True)
        stage = "cpu_float32_load"
        torch.set_num_threads(min(2, os.cpu_count() or 1))
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True,
                                                  trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True,
            trust_remote_code=False, dtype=torch.float32, device_map="cpu",
            low_cpu_mem_usage=True).eval()
        identity = _model_identity(model_root, model, tokenizer, fixture_only=False)
        packages = {}
        for name in ("torch", "transformers", "huggingface-hub", "accelerate"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        receipt = {"event": "model_reload_completed", "setup_id": setup_id,
            "at_utc": stamp(), "process_id": os.getpid(), "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION, "snapshot_path": str(snapshot),
            "files_verified": len(expected), "bytes_verified": total,
            "setup_seconds": time.monotonic() - started, "packages": packages,
            "device": str(model.device), "dtype": str(next(model.parameters()).dtype),
            "model_identity": identity, "new_provider_calls": 0,
            "answer_generations_during_setup": 0, "local_files_only": True}
        _append(events, receipt)
        print(canonical({"stage": "offline_cpu_model_reload_completed", **receipt}), flush=True)
        return model, tokenizer, receipt
    except BaseException as error:
        _append(events, {"event": "model_reload_failed", "setup_id": setup_id,
            "at_utc": stamp(), "failure_stage": stage, "failure_type": type(error).__name__,
            "observed_setup_seconds": time.monotonic() - started,
            "new_provider_calls": 0, "automatic_retry": False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--prepared-dir", required=True, type=Path)
    run.add_argument("--model-root", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--max-new-generations", type=int, default=MAX_GENERATIONS)
    args = parser.parse_args(argv)
    require_colab()  # Before gates, files, locks, libraries or model objects.
    if any(not p.resolve().is_relative_to(Path("/content")) for p in
           (args.prepared_dir, args.model_root, args.output)):
        raise ValueError("hosted_content_paths_required")
    if not 0 <= args.max_new_generations <= MAX_GENERATIONS:
        raise ValueError("finite_invocation_generation_cap_required")
    planner = _gate_module()
    protocol, rows, _ = _requests(args.prepared_dir.resolve())
    # The official controlled planner predates an explicit fixture_only field.
    # Match run_open_reader: absence is valid, while an explicit fixture marker
    # cannot enter the real CLI. Hosted guard and authoritative gates still run.
    if protocol.get("fixture_only") is not None and protocol["fixture_only"] is not False:
        raise ValueError("real_cli_requires_nonfixture_preparation")
    gates = planner.make_gates(args.prepared_dir)
    for row in rows:
        gates.preflight(row)
    # The root CLI is intentionally a fresh process. Holding this lock across
    # setup and inference prevents two children from loading duplicate weights
    # into the same small hosted runtime concurrently.
    with _locked(args.model_root / ".model_setup.lock"):
        _global_state(args.model_root)  # Unknown work blocks before costly reload.
        model, tokenizer, setup = _load_cached_model(args.model_root)
        result = run_open_reader(args.prepared_dir, model, tokenizer, args.model_root,
            args.output, max_new_generations=args.max_new_generations, gates=gates)
        _append(args.output / "execution_events.jsonl", {
            "event": "fresh_process_reader_finished", "setup_id": setup["setup_id"],
            "process_id": os.getpid(), "model_setup_seconds_this_process": setup["setup_seconds"],
            "run_protocol_sha256": result["run_protocol_sha256"],
            "logical_completed": result["logical_completed"], "complete": result["complete"],
            "new_provider_calls": 0})
        print(canonical({"stage": "fresh_process_open_reader_finished", **result,
                         "model_setup_seconds_this_process": setup["setup_seconds"]}), flush=True)
        return result


if __name__ == "__main__":
    main()
