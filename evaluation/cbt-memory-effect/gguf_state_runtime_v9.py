"""New hosted-only question-schema regression, maximum 57 unique attempts.

The eight-question reader gate must fully pass before initialization. Full JSON
is generated directly; no output adapter or response repair is used. Older
studies and their budgets remain independent immutable historical references.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import open_reader_colab as reader
import gguf_schema_reader_v7_colab as native
import gguf_gate_auth_v9 as gate_audit

VERSION = "native-t4-gguf-question-schema-state-development-study-v9-identical-binding-registration"
STUDY_ROOT = Path("/content/cbt-qwen7b-gpu-state-study-20261112-v9")
CAP = 57
PILOT_COUNT = 0
MAIN_UNIQUE = 57
SIDES = ("before", "after")
ARMS = ("raw_history_rag", "managed_memory_rag")
PURPOSE = "exposed_controlled_state_development_regression_not_clinical"
PLANNER = Path("evaluation/cbt-memory-effect/controlled_open_memory.py")


def schema_module():
    return native.schema


def guard(*paths):
    reader.require_colab()
    if any(not Path(p).resolve().is_relative_to(Path("/content")) for p in paths):
        raise ValueError("genuine_hosted_content_paths_required")


def code_identity():
    field_path = HERE.parents[1] / "modules/agent-integration/question_field_selection_v4.py"
    return {str(Path(p).resolve()): reader.file_hash(p) for p in
        (__file__, reader.__file__, native.__file__, native.schema.__file__, gate_audit.__file__,
         field_path, field_path.parent / "evidence_bundles.py",
         field_path.parent / "question_field_selection_v8.py",
         HERE / "gguf_state_comparison_v9_colab.py", HERE / "question_retrieval_policy_v9.py")}


def study(root):
    root = Path(root).resolve()
    guard(root)
    started, valid = time.monotonic(), False
    try:
        value = reader.read_json(root / "study_protocol.json")
        if (value.get("version") != VERSION or value.get("maximum_new_study_attempts") != CAP
                or value.get("code_sha256") != code_identity()
                or value.get("study_root") != str(root)):
            raise ValueError("new_study_identity_changed")
        for name, expected in value["historical_reference_sha256"].items():
            if reader.file_hash(name) != expected:
                raise ValueError("old_run_reference_changed_stop")
        for name, expected in value["inherited_identity_sha256"].items():
            if reader.file_hash(root / name) != expected:
                raise ValueError("inherited_model_identity_changed")
        valid = True
        return value
    finally:
        if root.is_dir():
            reader._append(root / "frozen_material_validation_timings.jsonl", {
                "event": "frozen_material_validation", "complete": valid,
                "wall_seconds": time.monotonic() - started,
                "scope": "complete code, protocol, historical/setup file SHA256 and inherited manifest validation; excludes writing this timing record",
                "new_model_generations": 0})


GATE_ROOT = Path("/content/cbt-qwen7b-gpu-reader-gate-20261112-v7")
MODEL_DIR = Path("/content/cbt-qwen7b-gpu-reader-setup-20261112-v7")
SETUP_PHASES = ("before", "after", "score-authentication")
FAILED_MAIN_ROOT = Path("/content/cbt-qwen7b-gpu-state-study-20261112-v8")
FAILED_MAIN_PROTOCOL_SHA256 = "e2c14988f5f4ddca7472e7ef861a9b73ae630d0254b08a7ade357a49a1223eb1"


def failed_main_references(root=FAILED_MAIN_ROOT, *, expected_protocol_sha256=FAILED_MAIN_PROTOCOL_SHA256):
    """Preserve the zero-generation V8 failure and its complete V7 history."""
    root = Path(root).resolve()
    guard(root)
    protocol_path = root / "study_protocol.json"
    if (expected_protocol_sha256 != FAILED_MAIN_PROTOCOL_SHA256
            or reader.file_hash(protocol_path) != expected_protocol_sha256):
        raise ValueError("exact_failed_v8_main_protocol_required")
    protocol = reader.read_json(protocol_path)
    if (protocol.get("version") != "native-t4-gguf-question-schema-state-development-study-v8-import-isolation"
            or protocol.get("study_root") != str(root)
            or reader.read_rows(root / "generation_ledger.jsonl")
            or (root / "main_freeze.json").exists()):
        raise RuntimeError("unchanged_v8_preparation_failure_with_zero_admissions_required")
    references = {str(protocol_path): expected_protocol_sha256}
    for path, expected in {**protocol["historical_reference_sha256"], **protocol["code_sha256"]}.items():
        if reader.file_hash(path) != expected:
            raise ValueError("failed_v8_frozen_source_or_history_changed")
        references[path] = expected
    paths = [root / "prepare_before.log", root / "render_before.log",
        root / "preparation_timings/before_source_prepare.json",
        root / "preparation_timings/before_render_and_register.json",
        root / "frozen_material_validation_timings.jsonl",
        Path("/content/cbt-gpu-v8-binding-diagnostic-20261112-v1.json")]
    if "effective_binding_already_exists_no_overwrite" not in (root / "render_before.log").read_text():
        raise ValueError("recorded_v8_identical_binding_registration_failure_required")
    references.update({str(path): reader.file_hash(path) for path in paths})
    return references


def initialize(root=STUDY_ROOT, *, gate_root=GATE_ROOT, expected_gate_protocol_sha256,
               expected_gate_report_sha256, expected_gate_authentication_sha256,
               failed_main_root=FAILED_MAIN_ROOT,
               expected_failed_main_protocol_sha256=FAILED_MAIN_PROTOCOL_SHA256):
    root, gate_root = Path(root).resolve(), Path(gate_root).resolve()
    guard(root, gate_root)
    if root.exists() or root.is_relative_to(gate_root) or gate_root.is_relative_to(root):
        raise FileExistsError("fresh_independent_v9_main_root_required")
    started = time.monotonic()
    gate = gate_audit.verify_gate(gate_root,
        expected_protocol_sha256=expected_gate_protocol_sha256,
        expected_report_sha256=expected_gate_report_sha256,
        expected_authentication_sha256=expected_gate_authentication_sha256)
    # Gate rejection happens before any main directory or load/admission exists.
    if gate.get("typed_correct") != 8 or gate.get("schema_valid") != 8:
        raise RuntimeError("all_eight_native_gate_answers_must_pass_before_main")
    historical = dict(gate["reference_sha256"])
    historical.update(failed_main_references(failed_main_root,
        expected_protocol_sha256=expected_failed_main_protocol_sha256))
    root.mkdir(parents=True, mode=0o700)
    for name in ("model_manifest.json", "runtime_manifest.json"):
        with (root / name).open("xb") as stream:
            stream.write((gate_root / name).read_bytes())
    inherited = {name: reader.file_hash(root / name) for name in ("model_manifest.json", "runtime_manifest.json")}
    protocol = {"version": VERSION, "study_root": str(root), "maximum_new_study_attempts": CAP,
        "planned_pilot_attempts_in_this_ledger": 0, "maximum_main_unique_attempts": CAP,
        "planned_main_logical_rows": 96, "actual_unique_grid_freezes_before_inference": True,
        "new_authorization_basis": "Continued bounded hosted-cloud optimization, separately frozen ceiling57; no gate or historical budget transfers.",
        "historical_reference_sha256": historical,
        "failed_v8_main_root": str(Path(failed_main_root).resolve()),
        "failed_v8_main_protocol_sha256": expected_failed_main_protocol_sha256,
        "failed_v8_main_generation_admissions": 0,
        "failed_v8_preparation_costs_nonadditive_history": True,
        "registration_fix_only_model_question_policy_and_answer_contracts_unchanged": True,
        "inherited_identity_sha256": inherited, "reader_gate": gate,
        "model_dir": str(MODEL_DIR), "native_backend_policy": native.policy(),
        "model_manifest_sha256": inherited["model_manifest.json"],
        "runtime_manifest_sha256": inherited["runtime_manifest.json"],
        "reader_gate_generations_are_separate_nonadditive_history": 8,
        "new_actual_setup_cost_source": "only this main root's phase setup journals and receipts",
        "maximum_setup_attempts": 3, "setup_phases_once_each": list(SETUP_PHASES),
        "gate_full_native_revalidation_before_any_main_admission": True,
        "code_sha256": code_identity(), "decoding_policy": decoding_identity(),
        "stop_policy": "full frozen grid; no intermediate scoring, automatic retry, outcome stopping or cap increases",
        "all_cases_exposed_development_only": True, "new_paid_provider_calls": 0,
        "secret_access": False, "new_human_reviews": 0, "clinical_or_natural_CBT_effect_claim": False,
        "metric": "EOS-complete sole-answer whole JSON exact typed value; no response repair",
        "main_shared_budget": {"complete_serialized_request_characters": 6000,
            "maximum_native_input_tokens": 4096, "maximum_new_output_tokens": 64,
            "assistant_prefix": None}, "created_at_utc": reader.stamp(),
        "initialization_and_lightweight_gate_verification_wall_seconds": time.monotonic() - started}
    reader._write_new(root / "study_protocol.json", protocol)
    return {"stage": "v9_main_initialized_after_eight_gate_no_model_load_or_generation",
            "study_protocol_sha256": reader.file_hash(root / "study_protocol.json"),
            "maximum_new_study_attempts": CAP, "new_model_calls": 0}


def _gate_runner(protocol):
    path = Path(protocol["reader_gate"]["gate_runner_path"])
    expected = protocol["historical_reference_sha256"].get(str(path))
    if expected is None or reader.file_hash(path) != expected:
        raise ValueError("frozen_native_gate_runner_source_required")
    name = "v9_main_original_native_gate_auditor"
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path.resolve():
            raise RuntimeError("foreign_gate_runner_module")
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("original_gate_runner_module_required")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(module.ROOT).resolve() != Path(protocol["reader_gate"]["gate_root"]).resolve():
        raise ValueError("original_gate_root_must_remain_at_original_path")
    return module


def revalidate_gate_with_backend(root, backend, *, phase):
    started = time.monotonic()
    protocol = study(root)
    module = _gate_runner(protocol)
    gate_protocol, rows = module.frozen()
    plans = [backend.input_preflight(row["messages"]) for row in rows]
    answers = module.authenticate(gate_protocol, rows, backend, plans, original_load=False)
    report = module.score_after_authentication(gate_protocol, rows, answers,
        module.read(module.ROOT / "model_setup_receipt.json"))
    if report != module.read(module.ROOT / "gate_results.json") or report.get("gate_passed") is not True:
        raise RuntimeError("unchanged_native_eight_gate_must_reproduce_before_main")
    receipt = {"phase": phase, "gate_protocol_sha256": reader.file_hash(module.ROOT / "gate_protocol.json"),
        "original_gate_report_sha256": reader.file_hash(module.ROOT / "gate_results.json"),
        "native_originals_reproduced": len(answers), "typed_correct": report["typed_correct"],
        "wall_seconds": time.monotonic() - started, "new_model_generations": 0,
        "scope": "complete gate revalidation including immutable full-file SHA256 checks, native original token detokenization, vectors, usage and exact gate answers; no new inference"}
    reader._write_new(Path(root) / ("gate_revalidation_" + phase + ".json"), receipt)
    return receipt


def load_backend_for_phase(root, phase, *, constructor_log_path):
    root = Path(root).resolve()
    protocol = study(root)
    constructor_log_path = Path(constructor_log_path).resolve()
    guard(constructor_log_path)
    if phase not in SETUP_PHASES:
        raise ValueError("fixed_native_setup_phase_required")
    claim = root / ("setup_claim_" + phase + ".json")
    reader._write_new(claim, {"phase": phase, "pid": os.getpid(), "at_utc": reader.stamp(),
        "automatic_retry": False, "new_generation_admissions": 0,
        "constructor_log_path": str(constructor_log_path)})
    runtime = reader.read_json(root / "runtime_manifest.json")
    sys.path.insert(0, runtime["site_packages"])
    journal = root / "native_setup_events.jsonl"
    reader._append(journal, {"event": "native_model_setup_intent", "phase": phase,
        "at_utc": reader.stamp(), "secret_access": False, "answer_generations_during_setup": 0})
    started = time.monotonic()
    try:
        backend, receipt = native.load_backend(protocol["model_dir"], reader.read_json(root / "model_manifest.json"),
            runtime_manifest=runtime, constructor_log_path=constructor_log_path)
        if receipt.get("gpu_execution_proof") != backend.gpu_execution_proof:
            raise ValueError("actual_gpu_execution_proof_receipt_disagreement")
        verify_gpu_proof = getattr(native, "verify_gpu_execution_proof", None)
        if verify_gpu_proof is None:
            raise RuntimeError("prospectively_frozen_gpu_proof_verifier_required")
        verify_gpu_proof(receipt["gpu_execution_proof"],
            expected_condition=backend.identity["gpu_execution_condition"])
        path = root / ("native_setup_" + phase + ".json")
        reader._write_new(path, receipt)
        reader._append(journal, {"event": "native_model_setup_completed", "phase": phase,
            "receipt_path": str(path), "receipt_sha256": reader.file_hash(path),
            "observed_wall_seconds": time.monotonic() - started, "at_utc": reader.stamp()})
        revalidate_gate_with_backend(root, backend, phase=phase)
        return backend, receipt
    except BaseException as error:
        reader._append(journal, {"event": "native_model_setup_or_gate_revalidation_failed", "phase": phase,
            "failure_type": type(error).__name__, "observed_wall_seconds": time.monotonic() - started,
            "automatic_retry": False, "at_utc": reader.stamp()})
        raise


def phase_execution_binding(root, backend, phase):
    """Bind per-load hardware evidence without contaminating cache identity."""
    if phase not in ("before", "after"):
        raise ValueError("native_answer_side_required_for_gpu_proof_binding")
    path = Path(root) / ("native_setup_" + phase + ".json")
    receipt = reader.read_json(path)
    proof = receipt.get("gpu_execution_proof")
    if (receipt.get("model_identity") != backend.identity
            or proof != backend.gpu_execution_proof):
        raise ValueError("owned_phase_gpu_execution_proof_required")
    native.verify_gpu_execution_proof(proof,
        expected_condition=backend.identity["gpu_execution_condition"])
    revalidation_path = Path(root) / ("gate_revalidation_" + phase + ".json")
    revalidation = reader.read_json(revalidation_path)
    if revalidation.get("native_originals_reproduced") != 8 or revalidation.get("typed_correct") != 8:
        raise RuntimeError("full_native_gate_revalidation_required_before_gpu_admission")
    return {"phase": phase, "setup_receipt_sha256": reader.file_hash(path),
        "gpu_execution_proof_sha256": native.digest(proof),
        "gate_revalidation_sha256": reader.file_hash(revalidation_path)}


def ledger(root):
    """The new 57 counter; failures/unknown outcomes prohibit automatic resend."""
    intents, outcomes = {}, {}
    for raw in reader.read_rows(Path(root) / "generation_ledger.jsonl"):
        row = reader._checked(raw)
        target = intents if row.get("event") == "generation_intent" else outcomes if row.get("event") == "generation_outcome" else None
        key = row.get("canonical_key")
        if target is None or key in target:
            raise ValueError("duplicate_or_unknown_new_study_event")
        target[key] = row
    if set(outcomes) - set(intents) or len(intents) > CAP:
        raise ValueError("new_57_budget_or_outcome_binding_changed")
    if set(intents) != set(outcomes) or any(x.get("status") != "published" for x in outcomes.values()):
        raise RuntimeError("failed_unknown_or_interrupted_new_study_no_automatic_retry")
    return intents, outcomes


def generation_key(root, model_identity, decoding, payload):
    return reader.digest({"new_study_protocol_sha256": reader.file_hash(Path(root) / "study_protocol.json"),
        "model": model_identity, "decoding": decoding, "payload": payload,
        "max_new_tokens": 64, "do_sample": False})


def cost(outcomes, keys):
    selected = [outcomes[k] for k in set(keys)]
    result = {"unique_admitted_attempts": len(selected), "status_counts": dict(Counter(x["status"] for x in selected)),
              "observed_tokens": {}, "usage_missing_attempts": {}, "new_provider_calls": 0}
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [(x.get("usage") or {}).get(field) for x in selected]
        good = [x for x in values if type(x) is int and x >= 0]
        result["observed_tokens"][field] = sum(good) if good or not selected else None
        result["usage_missing_attempts"][field] = len(values) - len(good)
    times = [x.get("generation_wall_seconds") for x in selected]
    good = [x for x in times if type(x) in (int, float) and math.isfinite(x) and x >= 0]
    result.update(answer_path_wall_seconds_sum=sum(good) if good or not selected else None,
                  missing_wall_time_attempts=len(times) - len(good),
                  wall_time_scope="generation plus source fences and cache/publication work; not pure model-forward latency")
    native_times = [x.get("native_generation_wall_seconds") for x in selected]
    observed = [x for x in native_times if type(x) in (int, float) and math.isfinite(x) and x >= 0]
    result.update(native_generation_wall_seconds_sum=sum(observed) if observed or not selected else None,
        native_generation_wall_seconds_missing_attempts=len(native_times) - len(observed),
        native_generation_timing_is_nonadditive_with_answer_path=True)
    result.update(native_token_count_scope="submitted native input vector length and observed yielded output tokens; failed prompt evaluation consumption may be unknown",
        prompt_evaluation_completion_unknown_attempts=sum(x.get("prompt_evaluation_completed") is not True for x in selected))
    return result


def cached(root, outcome):
    if not Path(outcome["cache_path"]).resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("old_or_foreign_study_cache_prohibited")
    return reader._cache(outcome)


def decoding_identity():
    return {"version": native.VERSION, "backend_policy": native.policy(),
            "assistant_prefix": None, "max_new_tokens": 64, "do_sample": False,
            "question_type_only_not_gold_answer": True, "native_token_vectors": True}


def input_preflight(backend, messages):
    return backend.input_preflight(messages)


def authenticate_native_result(backend, messages, preflight, value):
    """Reproduce native vectors, EOG, bytes, schema and usage; no targets."""
    import base64
    import hashlib
    actual = backend.input_preflight(messages)
    if actual != preflight:
        raise ValueError("native_actual_input_vector_changed")
    ids = value.get("raw_generated_token_ids")
    if (not isinstance(ids, list) or not ids or len(ids) > 64
            or any(type(token) is not int or token < 0 for token in ids)):
        raise ValueError("actual_bounded_native_output_tokens_required")
    eos = ids[-1] in backend.eog_ids
    if eos != (value.get("finish_reason") == "eos") or any(t in backend.eog_ids for t in ids[:-1]):
        raise ValueError("actual_observed_native_eog_required")
    body = backend.engine.detokenize(ids[:-1] if eos else ids,
        prev_tokens=preflight["input_token_ids"], special=False)
    raw = backend.engine.detokenize(ids, prev_tokens=preflight["input_token_ids"], special=True)
    text, raw_text = body.decode("utf-8", errors="replace"), raw.decode("utf-8", errors="replace")
    utf8 = True
    try:
        body.decode("utf-8", errors="strict"); raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        utf8 = False
    checked = native.schema.validate_response(text, preflight["question_schema"])
    if not eos or not utf8:
        checked = {"accepted": False, "reason": value["finish_reason"] if utf8 else "invalid_utf8", "answer": None}
    usage = {"prompt_tokens": preflight["input_tokens"], "completion_tokens": len(ids),
        "total_tokens": preflight["input_tokens"] + len(ids)}
    decoding = {"backend_policy_sha256": native.digest(native.policy()),
        "model_identity_sha256": native.digest(backend.identity),
        "question_schema_sha256": native.digest(preflight["question_schema"]),
        "gbnf_sha256": hashlib.sha256(native.grammar_text(preflight["question_schema"]).encode()).hexdigest(),
        "max_new_tokens": 64, "native_eog_token_ids": backend.eog_ids}
    elapsed = value.get("native_generation_wall_seconds")
    if (value.get("response_text") != text or value.get("raw_generated_text") != raw_text
            or base64.b64decode(value["response_bytes_base64"], validate=True) != body
            or base64.b64decode(value["raw_generated_bytes_base64"], validate=True) != raw
            or value.get("response_sha256") != hashlib.sha256(text.encode()).hexdigest()
            or value.get("response_bytes_sha256") != hashlib.sha256(body).hexdigest()
            or value.get("input_token_ids_sha256") != preflight["input_token_ids_sha256"]
            or value.get("rendered_prompt_sha256") != preflight["rendered_prompt_sha256"]
            or value.get("question_schema_sha256") != preflight["question_schema_sha256"]
            or value.get("decoding_identity") != decoding
            or value.get("decoding_identity_sha256") != native.digest(decoding)
            or value.get("schema_validation") != checked
            or value.get("chosen_answer") != checked["answer"]
            or type(value.get("chosen_answer")) is not type(checked["answer"])
            or value.get("utf8_valid") is not utf8 or value.get("usage") != usage
            or value.get("prefilled_token_count") != 0
            or type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0):
        raise ValueError("native_original_tokens_bytes_schema_or_usage_changed")
    return value


def generate_rows(root, output, rows, backend, *, purpose, gates, metadata=None):
    """Each logical answer passes actual source fences, including shared caches."""
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError("existing_attempt_archive_inspect_no_automatic_resume")
    protocol = study(root)
    if gates is None:
        raise ValueError("actual_source_gates_required")
    phase_binding = phase_execution_binding(root, backend, (metadata or {}).get("side"))
    if (backend.identity.get("model_id") != native.MODEL_ID
            or backend.identity.get("quantization") != native.QUANTIZATION
            or backend.identity.get("backend_policy_sha256") != native.digest(native.policy())):
        raise ValueError("actual_fixed_native_backend_identity_required")
    preflight_started = time.monotonic()
    preflights = []
    for row in rows:
        if (reader.digest(row["request_payload"]) != row["request_sha256"]
                or row["request_payload"].get("max_tokens") != 64
                or len(json.dumps(row["request_payload"], ensure_ascii=True, sort_keys=True, allow_nan=False)) > 6000):
            raise ValueError("request_binding_or_shared_complete_budget_changed")
        gates.preflight(row)
        preflights.append(backend.input_preflight(row["request_payload"]["messages"]))
    preflight_seconds = time.monotonic() - preflight_started
    run = {"version": VERSION, "purpose": purpose, "model": backend.identity,
        "study_protocol_sha256": reader.file_hash(root / "study_protocol.json"),
        "code_sha256": code_identity(), "maximum_new_study_attempts": CAP,
        "rows_sha256": reader.digest(rows), "actual_native_input_preflight": preflights,
        "native_input_preflight_wall_seconds": preflight_seconds,
        "native_input_preflight_timing_scope": "all side source preflights plus native input vectors and question schemas; no generation",
        "decoding_identity": decoding_identity(), "gate_identity": gates.identity(),
        "actual_gpu_phase_execution_binding": phase_binding,
        "source_gate_role": "actual fresh product source, permission, cutoff and version fences",
        "metadata": metadata or {}, "runner_code_sha256": reader.file_hash(__file__),
        "automatic_retry": False, "paid_provider_calls": 0, "secret_access": False,
        "clinical_or_natural_CBT_effect_claim": False}
    output.mkdir(parents=True, mode=0o700)
    reader._write_new(output / "run_protocol.json", run)
    run_hash = reader.digest(run)
    intents, outcomes = ledger(root)
    for row, expected_input in zip(rows, preflights):
        started = time.monotonic()
        study(root)
        if phase_execution_binding(root, backend, (metadata or {}).get("side")) != phase_binding:
            raise ValueError("gpu_execution_proof_changed_before_admission")
        key = generation_key(root, backend.identity, decoding_identity(), row["request_payload"])
        previous = outcomes.get(key)
        admission, intent, result = None, None, None
        finalized = False
        try:
            if previous is None and len(intents) >= CAP:
                raise RuntimeError("new_57_cap_exhausted_no_automatic_increase")
            gates.preflight(row)
            admission = gates.begin(row)
            if admission.get("replayed") and (admission.get("status") != "succeeded" or previous is None):
                raise RuntimeError("new_source_admission_without_owned_cache_cannot_reuse_old_request")
            finalized = bool(admission.get("replayed"))
            if previous is not None:
                result = cached(root, previous)
            else:
                intent = reader._append(root / "generation_ledger.jsonl", {
                    "event": "generation_intent", "canonical_key": key, "purpose": purpose,
                    "case_id": row["case_id"], "arm": row["arm"],
                    "request_sha256": row["request_sha256"], "run_protocol_sha256": run_hash,
                    "output": str(output), "model_revision": backend.identity["model_revision"],
                    "decoding_identity_sha256": reader.digest(decoding_identity()),
                    "input_token_ids_sha256": expected_input["input_token_ids_sha256"],
                    "question_schema_sha256": expected_input["question_schema_sha256"],
                    "started_at_utc": reader.stamp(), "attempt_number": len(intents) + 1})
                intents[key] = intent
                gates.preflight(row)
                result = backend.generate(row["request_payload"]["messages"],
                    expected_input["question_schema"], max_new_tokens=64)
            authenticate_native_result(backend, row["request_payload"]["messages"], expected_input, result)
            gates.preflight(row)
            if not finalized:
                receipt = gates.finish(row, admission, outcome="success", response=None, usage=result["usage"])
                finalized = True
                if receipt.get("publishable") is not True:
                    raise PermissionError("source_changed_before_publication")
            gates.preflight(row)
            if previous is None:
                result = {**result, "canonical_key": key}
                cache_path = output / "canonical" / (key + ".json")
                reader._write_new(cache_path, result)
                outcomes[key] = reader._append(root / "generation_ledger.jsonl", {
                    "event": "generation_outcome", "canonical_key": key, "status": "published",
                    "cache_path": str(cache_path), "cache_sha256": reader.file_hash(cache_path),
                    "response_sha256": result["response_sha256"], "usage": result["usage"],
                    "prompt_evaluation_completed": result.get("prompt_evaluation_completed"),
                    "usage_scope": result.get("usage_scope"),
                    "finish_reason": result["finish_reason"],
                    "native_generation_wall_seconds": result["native_generation_wall_seconds"],
                    "generation_wall_seconds": time.monotonic() - started, "finished_at_utc": reader.stamp()})
            reader._append(output / "answers_unscored.jsonl", {
                **result, "case_id": row["case_id"], "arm": row["arm"],
                "canonical_key": key, "request_sha256": row["request_sha256"],
                "run_protocol_sha256": run_hash, "canonical_generation_shared": intent is None,
                "logical_processing_wall_seconds": time.monotonic() - started,
                "logical_processing_timing_scope": "source admission/fences, generation or cache validation, and canonical publication before logical-row serialization",
                "source_checks_passed": True, "status": "completed", "scored": False})
            print(reader.canonical({"stage": "v9_main_answer_recorded", "logical_case": row["case_id"],
                "arm": row["arm"], "global_attempts": len(intents), "shared": intent is None,
                "scores_inspected": False}), flush=True)
        except BaseException as error:
            observed = result if result is not None else getattr(error, "generated_record", None)
            if admission is not None and not finalized:
                try:
                    gates.finish(row, admission, outcome="failed" if observed else "unknown", response=None,
                        usage=observed.get("usage") if observed else None)
                except BaseException:
                    pass
            partial_ref = None
            if intent is not None and key not in outcomes:
                # Preserve observed native failures only while the original source
                # authority still permits them; revoked output remains discarded.
                if observed is not None:
                    try:
                        gates.preflight(row)
                    except BaseException:
                        pass
                    else:
                        partial_path = output / "failed_partials" / (key + ".json")
                        reader._write_new(partial_path, observed)
                        partial_ref = {"path": str(partial_path), "sha256": reader.file_hash(partial_path)}
                reader._append(root / "generation_ledger.jsonl", {
                    "event": "generation_outcome", "canonical_key": key,
                    "status": "discarded_or_failed_or_unknown", "response_text": None,
                    "observed_partial": partial_ref, "usage": observed.get("usage") if observed else None,
                    "prompt_evaluation_completed": observed.get("prompt_evaluation_completed") if observed else None,
                    "usage_scope": observed.get("usage_scope") if observed else None,
                    "native_generation_wall_seconds": observed.get("native_generation_wall_seconds") if observed else None,
                    "generation_wall_seconds": time.monotonic() - started,
                    "failure_type": type(error).__name__, "automatic_retry": False,
                    "finished_at_utc": reader.stamp()})
            reader._append(output / "failures.jsonl", {"case_id": row["case_id"],
                "failure_type": type(error).__name__, "automatic_retry": False,
                "global_attempt_may_have_generated": intent is not None})
            raise
    intents, outcomes = ledger(root)
    answers = reader.read_rows(output / "answers_unscored.jsonl")
    keys = {answer["canonical_key"] for answer in answers}
    owned = {key for key, intent in intents.items() if intent["output"] == str(output)}
    summary = {"version": VERSION, "complete": len(answers) == len(rows),
        "logical_completed": len(answers), "logical_planned": len(rows),
        "answers_sha256": reader.file_hash(output / "answers_unscored.jsonl"),
        "run_protocol_sha256": run_hash, "global_new_study_attempts": len(intents),
        "owned_generation_cost": cost(outcomes, owned), "referenced_cost_nonadditive": cost(outcomes, keys),
        "reused_canonical_receipts": len(keys - owned), "maximum_new_study_attempts": CAP,
        "cache_reuse_processing_wall_seconds_sum": sum(a["logical_processing_wall_seconds"]
            for a in answers if a["canonical_generation_shared"]),
        "new_provider_calls": 0, "new_human_reviews": 0}
    reader._write_new(output / "completion_summary.json", summary)
    return summary


def authenticate(root, output, rows, backend):
    study(root)
    output = Path(output)
    run = reader.read_json(output / "run_protocol.json")
    summary = reader.read_json(output / "completion_summary.json")
    if (summary.get("complete") is not True or summary.get("logical_completed") != len(rows)
            or summary.get("answers_sha256") != reader.file_hash(output / "answers_unscored.jsonl")
            or summary.get("run_protocol_sha256") != reader.digest(run)
            or run.get("study_protocol_sha256") != reader.file_hash(Path(root) / "study_protocol.json")
            or run.get("rows_sha256") != reader.digest(rows) or run.get("code_sha256") != code_identity()
            or run.get("decoding_identity") != decoding_identity() or run.get("model") != backend.identity):
        raise ValueError("complete_bound_native_side_archive_required")
    phase = run.get("metadata", {}).get("side")
    proof_receipt_path = Path(root) / ("native_setup_" + str(phase) + ".json")
    proof_receipt = reader.read_json(proof_receipt_path)
    native.verify_gpu_execution_proof(proof_receipt["gpu_execution_proof"],
        expected_condition=backend.identity["gpu_execution_condition"])
    if (proof_receipt.get("model_identity") != backend.identity
            or run.get("actual_gpu_phase_execution_binding") != {
                "phase": phase, "setup_receipt_sha256": reader.file_hash(proof_receipt_path),
                "gpu_execution_proof_sha256": native.digest(proof_receipt["gpu_execution_proof"]),
                "gate_revalidation_sha256": reader.file_hash(Path(root) / ("gate_revalidation_" + str(phase) + ".json"))}):
        raise ValueError("authenticated_gpu_phase_proof_binding_changed")
    intents, outcomes = ledger(root)
    answers = [reader._checked(row) for row in reader.read_rows(output / "answers_unscored.jsonl")]
    preflights = run.get("actual_native_input_preflight", [])
    if len(answers) != len(rows) or len(preflights) != len(rows):
        raise ValueError("complete_ordered_native_preflights_and_answers_required")
    for row, answer, preflight in zip(rows, answers, preflights):
        key = generation_key(root, backend.identity, run["decoding_identity"], row["request_payload"])
        if (answer.get("case_id") != row["case_id"] or answer.get("arm") != row["arm"]
                or answer.get("request_sha256") != row["request_sha256"] or answer.get("canonical_key") != key
                or answer.get("run_protocol_sha256") != reader.digest(run) or key not in outcomes
                or answer.get("source_checks_passed") is not True):
            raise ValueError("native_logical_answer_request_or_source_binding_changed")
        value = cached(root, outcomes[key])
        owner = Path(intents[key]["output"]).resolve()
        if not owner.is_relative_to(Path(root).resolve()):
            raise ValueError("native_cache_owner_outside_new_study")
        owner_run = reader.read_json(owner / "run_protocol.json")
        if (intents[key].get("request_sha256") != row["request_sha256"]
                or intents[key].get("run_protocol_sha256") != reader.digest(owner_run)
                or owner_run.get("study_protocol_sha256") != run["study_protocol_sha256"]
                or owner_run.get("code_sha256") != code_identity()):
            raise ValueError("native_cache_intent_owner_binding_changed")
        for field, expected in value.items():
            if answer.get(field) != expected:
                raise ValueError("native_original_cache_and_logical_row_disagree")
        authenticate_native_result(backend, row["request_payload"]["messages"], preflight, value)
        elapsed = answer.get("logical_processing_wall_seconds")
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("measured_logical_processing_time_required")
        if (outcomes[key].get("usage") != value["usage"]
                or outcomes[key].get("response_sha256") != value["response_sha256"]
                or intents[key].get("input_token_ids_sha256") != preflight["input_token_ids_sha256"]
                or intents[key].get("question_schema_sha256") != preflight["question_schema_sha256"]):
            raise ValueError("native_cache_journal_and_actual_vector_disagree")
    return answers, run


if __name__ == "__main__":
    raise SystemExit("Use gguf_state_comparison_v9_colab.py explicit subcommands")
