"""Independent 57-intent hosted-only unconstrained development regression.

Raw model output is immutable. A common output-only adapter is separately
frozen, stored and authenticated. The failed v2 pilot gate and its four settled
attempts remain unchanged; this new study does not spend its unused allowance.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import open_reader_colab as reader

VERSION = "unconstrained-state-development-study-v3"
STUDY_ROOT = Path("/content/cbt-unconstrained-state-study-20261112-v3")
CAP = 57
PILOT_COUNT = 0
MAIN_UNIQUE = 57
SIDES = ("before", "after")
ARMS = ("raw_history_rag", "managed_memory_rag")
PURPOSE = "exposed_controlled_state_development_regression_not_clinical"
PLANNER = Path("evaluation/cbt-memory-effect/controlled_open_memory.py")


def adapter_module():
    module = importlib.import_module("common_answer_adapter_v3")
    if Path(module.__file__).resolve() != HERE / "common_answer_adapter_v3.py":
        raise RuntimeError("foreign_cached_answer_adapter")
    return module



def guard(*paths):
    reader.require_colab()
    if any(not Path(p).resolve().is_relative_to(Path("/content")) for p in paths):
        raise ValueError("genuine_hosted_content_paths_required")


def code_identity():
    adapter = adapter_module()
    return {str(Path(p).resolve()): reader.file_hash(p) for p in
        (__file__, reader.__file__, adapter.__file__,
         HERE / "unconstrained_state_comparison_v3_colab.py",
         HERE / "common_retrieval_policy_v1.py")}



def study(root):
    root = Path(root).resolve()
    guard(root)
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
    return value


def initialize(root=STUDY_ROOT, *, old_root, failed_v2_root):
    root, old_root, failed_v2_root = (Path(p).resolve() for p in (root, old_root, failed_v2_root))
    guard(root, old_root, failed_v2_root)
    if (any(root.is_relative_to(prior) for prior in (old_root, failed_v2_root))
            or root.exists()):
        raise FileExistsError("new_independent_study_root_required")
    old_ledger = old_root / "generation_ledger.jsonl"
    old_sha = reader.file_hash(old_ledger)
    old_intents, _ = reader._global_state(old_root)
    if len(old_intents) != 64:
        raise RuntimeError("old_64_run_must_be_settled_before_new_study")
    # Use the unchanged v2 implementation to authenticate its real rejected gate.
    v2 = importlib.import_module("run_prefill_pilot_v2_colab")
    if Path(v2.__file__).resolve() != HERE / "run_prefill_pilot_v2_colab.py":
        raise RuntimeError("foreign_historical_v2_runtime")
    v2.study(failed_v2_root)
    v2_intents, _ = v2.ledger(failed_v2_root)
    _, pilot = v2.helpers()
    answers, _ = v2.authenticate(failed_v2_root, failed_v2_root / "pilot_answers", pilot.requests())
    saved_score = reader.read_json(failed_v2_root / "pilot_answers/pilot_score.json")
    if (len(v2_intents) != 4 or (failed_v2_root / "main_freeze.json").exists()
            or saved_score != pilot.score_authenticated(answers, prepared_dir=failed_v2_root / "pilot_prepared")
            or saved_score.get("ready_for_main") is not False):
        raise RuntimeError("unchanged_settled_rejected_v2_pilot_required")
    old_model = reader.read_json(old_root / "load_protocol.json")
    if (old_model.get("model_id") != reader.MODEL_ID or old_model.get("model_revision") != reader.MODEL_REVISION
            or reader.read_json(old_root / "model_files.json").get("revision") != reader.MODEL_REVISION):
        raise ValueError("pinned_source_model_identity_required")
    historical = {str(old_ledger): old_sha}
    for name in ("study_protocol.json", "generation_ledger.jsonl", "pilot_prepared/protocol.json",
                 "pilot_prepared/requests.json", "pilot_prepared/assessor_only/targets.json",
                 "pilot_answers/run_protocol.json", "pilot_answers/answers_unscored.jsonl",
                 "pilot_answers/completion_summary.json", "pilot_answers/pilot_score.json"):
        historical[str(failed_v2_root / name)] = reader.file_hash(failed_v2_root / name)
    root.mkdir(parents=True, mode=0o700)
    inherited = {}
    for name in ("load_protocol.json", "model_files.json", "model_load_receipt.json"):
        source = old_root / name
        with (root / name).open("xb") as stream:
            stream.write(source.read_bytes())
        inherited[name] = reader.file_hash(root / name)
        historical[str(source)] = reader.file_hash(source)
    if any(reader.file_hash(name) != expected for name, expected in historical.items()):
        raise ValueError("historical_identity_changed_during_reference_capture")
    adapter = adapter_module()
    protocol = {
        "version": VERSION, "study_root": str(root), "maximum_new_study_attempts": CAP,
        "planned_pilot_attempts": 0, "planned_main_unique_attempts": MAIN_UNIQUE,
        "planned_main_logical_rows": 96, "new_authorization_basis":
        "User authorized continued cloud optimization and this separately versioned bounded 57-attempt protocol; not unused allowance from old studies.",
        "old_study_root": str(old_root), "old_study_attempts": 64,
        "failed_v2_study_root": str(failed_v2_root), "failed_v2_study_attempts": 4,
        "failed_v2_gate_unchanged_and_closed": True,
        "historical_reference_sha256": historical, "inherited_identity_sha256": inherited,
        "inherited_setup_is_nonadditive_historical_reference": True,
        "inherited_load_protocol_64_is_model_identity_metadata_not_new_study_authorization": True,
        "new_actual_setup_cost_source": "only this root's newly written model_load_events.jsonl",
        "code_sha256": code_identity(), "decoding_policy": decoding_identity(),
        "output_adapter_identity": adapter.identity(),
        "adapter_development_material": "the already exposed v2 unconstrained diagnostics; no new independent pilot or gate is claimed",
        "stop_policy": "complete the frozen grid; no intermediate scoring, outcome-dependent stopping, automatic retries or cap increases",
        "old_24_cases_are_exposed_development_regression_only": True,
        "paid_provider_calls": 0, "secret_access": False, "new_human_reviews": 0,
        "clinical_quality_claim": False, "natural_CBT_memory_effect_claim": False,
        "metrics": ["original raw single-answer JSON/type/value accuracy", "prospectively declared common-adapter canonical typed-value accuracy"],
        "raw_model_outputs_immutable": True, "adapter_failure_is_not_null": True,
        "common_adapter_never_reads_question_arm_sources_or_gold": True,
        "main_stage_code_identity": "bound in code_sha256 and additionally frozen with complete request/source audit grid",
        "main_shared_budget": {"complete_serialized_request_characters": 6000,
            "assistant_prefix": None, "maximum_actual_input_tokens": 4096,
            "maximum_new_output_tokens": 64},
        "created_at_utc": reader.stamp(),
    }
    reader._write_new(root / "study_protocol.json", protocol)
    return {"version": VERSION, "stage": "new_57_cap_study_initialized_no_generation",
            "study_protocol_sha256": reader.file_hash(root / "study_protocol.json"), "new_model_calls": 0}



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
    return result


def cached(root, outcome):
    if not Path(outcome["cache_path"]).resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("old_or_foreign_study_cache_prohibited")
    return reader._cache(outcome)


def decoding_identity():
    return {"version": "unconstrained-greedy-common-adapter-reader-v3", "constraint": None,
            "assistant_prefix": None, "max_new_tokens": 64, "do_sample": False}


def input_preflight(tokenizer, messages):
    values = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt")
    count = int(values["input_ids"].shape[-1])
    if count > 4096:
        raise ValueError("input_token_cap_no_truncation")
    return {"input_tokens": count, "input_tensor_sha256": reader.digest(values["input_ids"].tolist())}



def generate_rows(root, output, rows, model, tokenizer, *, purpose, gates, metadata=None):
    """Caller holds both new-root locks; every logical reuse gets fresh fences."""
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError("existing_attempt_archive_inspect_no_automatic_resume")
    model_id = reader._model_identity(root, model, tokenizer, fixture_only=False)
    adapter = adapter_module()
    if gates is None:
        raise ValueError("actual_source_gates_required")
    preflights = []
    for row in rows:
        if reader.digest(row["request_payload"]) != row["request_sha256"]:
            raise ValueError("request_payload_binding_changed")
        if gates is not None:
            gates.preflight(row)
        if row["request_payload"].get("max_tokens") != 64 or len(json.dumps(row["request_payload"], ensure_ascii=True, sort_keys=True, allow_nan=False)) > 6000:
            raise ValueError("common_complete_request_or_output_budget_changed")
        preflights.append(input_preflight(tokenizer, row["request_payload"]["messages"]))
    run = {"version": VERSION, "purpose": purpose, "model": model_id,
        "study_protocol_sha256": reader.file_hash(root / "study_protocol.json"),
        "code_sha256": code_identity(), "maximum_new_study_attempts": CAP,
        "rows_sha256": reader.digest(rows), "actual_tensor_preflight": preflights,
        "decoding_identity": decoding_identity(), "output_adapter_identity": adapter.identity(),
        "gate_identity": gates.identity() if gates is not None else None,
        "source_gate_role": "actual fresh product source/permission/version fences" if gates is not None else "new invented nonpersonal format fixtures",
        "metadata": metadata or {}, "runner_code_sha256": reader.file_hash(__file__),
        "borrowed_loader_code_sha256": reader.file_hash(reader.__file__),
        "automatic_retry": False, "paid_provider_calls": 0, "secret_access": False,
        "clinical_or_natural_CBT_effect_claim": False}
    output.mkdir(parents=True, mode=0o700)
    reader._write_new(output / "run_protocol.json", run)
    run_hash = reader.digest(run)
    intents, outcomes = ledger(root)
    for row, expected_input in zip(rows, preflights):
        study(root)
        key = generation_key(root, model_id, decoding_identity(), row["request_payload"])
        previous = outcomes.get(key)
        admission = None
        finalized = False
        intent = None
        result = None
        started = time.monotonic()
        try:
            if previous is None and len(intents) >= CAP:
                raise RuntimeError("new_57_cap_exhausted_no_automatic_increase")
            if gates is not None:
                gates.preflight(row)
                admission = gates.begin(row)
                if admission.get("replayed") and (admission.get("status") != "succeeded" or previous is None):
                    raise RuntimeError("fresh_gate_without_new_study_cache_cannot_reuse_old_admission")
                finalized = bool(admission.get("replayed"))
            if previous is not None:
                result = cached(root, previous)
            else:
                if len(intents) >= CAP:
                    raise RuntimeError("new_57_cap_exhausted_no_automatic_increase")
                intent = reader._append(root / "generation_ledger.jsonl", {
                    "event": "generation_intent", "canonical_key": key, "purpose": purpose,
                    "case_id": row["case_id"], "arm": row["arm"],
                    "request_sha256": row["request_sha256"], "run_protocol_sha256": run_hash,
                    "output": str(output), "model_revision": reader.MODEL_REVISION,
                    "decoding_identity_sha256": reader.digest(decoding_identity()),
                    "started_at_utc": reader.stamp(), "attempt_number": len(intents) + 1})
                intents[key] = intent
                if gates is not None:
                    gates.preflight(row)
                result = reader._generate(model, tokenizer, row["request_payload"]["messages"], max_new_tokens=64)
            if result["input_tensor_sha256"] != expected_input["input_tensor_sha256"]:
                raise ValueError("actual_input_tensor_changed_after_preflight")
            if result["usage"]["prompt_tokens"] != expected_input["input_tokens"]:
                raise ValueError("usage_must_count_actual_unconstrained_input_tokens")
            if gates is not None:
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
                    "finish_reason": result["finish_reason"], "generation_wall_seconds": time.monotonic() - started,
                    "finished_at_utc": reader.stamp()})
            adaptation_started = time.monotonic()
            adapted = adapter.adapt(result["response_text"])
            adaptation_seconds = time.monotonic() - adaptation_started
            reader._append(output / "answers_unscored.jsonl", {
                **result, "case_id": row["case_id"], "arm": row["arm"],
                "answer_adapter": adapted, "output_adaptation_wall_seconds": adaptation_seconds,
                "canonical_key": key, "request_sha256": row["request_sha256"],
                "run_protocol_sha256": run_hash, "canonical_generation_shared": intent is None,
                "source_checks_passed": gates is not None, "status": "completed", "scored": False})
            print(reader.canonical({"stage": "new_study_answer_recorded", "purpose": purpose,
                "logical_case": row["case_id"], "arm": row["arm"], "global_attempts": len(intents),
                "shared": intent is None, "scores_inspected": False}), flush=True)
        except BaseException as error:
            if gates is not None and admission is not None and not finalized:
                try:
                    gates.finish(row, admission, outcome="failed" if result else "unknown", response=None,
                                 usage=result.get("usage") if result else None)
                except BaseException:
                    pass
            if intent is not None and key not in outcomes:
                reader._append(root / "generation_ledger.jsonl", {
                    "event": "generation_outcome", "canonical_key": key,
                    "status": "stale_discarded" if result is not None else "generation_failed_or_unknown",
                    "response_text": None, "usage": result.get("usage") if result else None,
                    "failure_type": type(error).__name__, "observed_wall_seconds": time.monotonic() - started,
                    "finished_at_utc": reader.stamp()})
            reader._append(output / "failures.jsonl", {"case_id": row["case_id"], "failure_type": type(error).__name__,
                "automatic_retry": False, "global_attempt_may_have_generated": intent is not None})
            raise
    intents, outcomes = ledger(root)
    answers = reader.read_rows(output / "answers_unscored.jsonl")
    keys = {a["canonical_key"] for a in answers}
    owned = {k for k, i in intents.items() if i["output"] == str(output)}
    summary = {"version": VERSION, "complete": len(answers) == len(rows), "logical_completed": len(answers),
        "logical_planned": len(rows), "answers_sha256": reader.file_hash(output / "answers_unscored.jsonl"),
        "run_protocol_sha256": run_hash, "global_new_study_attempts": len(intents),
        "owned_generation_cost": cost(outcomes, owned), "referenced_cost_nonadditive": cost(outcomes, keys),
        "reused_canonical_receipts": len(keys - owned), "maximum_new_study_attempts": CAP,
        "new_provider_calls": 0, "new_human_reviews": 0}
    reader._write_new(output / "completion_summary.json", summary)
    return summary


def authenticate(root, output, rows):
    study(root)
    output = Path(output)
    run = reader.read_json(output / "run_protocol.json")
    summary = reader.read_json(output / "completion_summary.json")
    if (summary.get("complete") is not True or summary.get("logical_completed") != len(rows)
            or summary.get("answers_sha256") != reader.file_hash(output / "answers_unscored.jsonl")
            or summary.get("run_protocol_sha256") != reader.digest(run)
            or run.get("study_protocol_sha256") != reader.file_hash(Path(root) / "study_protocol.json")
            or run.get("rows_sha256") != reader.digest(rows) or run.get("code_sha256") != code_identity()
            or run.get("decoding_identity") != decoding_identity()
            or run.get("output_adapter_identity") != adapter_module().identity()):
        raise ValueError("complete_new_study_archive_binding_required")
    intents, outcomes = ledger(root)
    answers = [reader._checked(a) for a in reader.read_rows(output / "answers_unscored.jsonl")]
    if len(answers) != len(rows) or len(run.get("actual_tensor_preflight", [])) != len(rows):
        raise ValueError("duplicate_or_incomplete_logical_answers")
    for row, answer, expected_input in zip(rows, answers, run["actual_tensor_preflight"]):
        key = generation_key(root, run["model"], run["decoding_identity"], row["request_payload"])
        if (answer.get("case_id") != row["case_id"] or answer.get("arm") != row["arm"]
                or answer.get("request_sha256") != row["request_sha256"] or answer.get("canonical_key") != key
                or answer.get("run_protocol_sha256") != reader.digest(run) or key not in outcomes):
            raise ValueError("unbound_new_study_logical_answer")
        value = cached(root, outcomes[key])
        intent = intents[key]
        owner = Path(intent["output"]).resolve()
        if not owner.is_relative_to(Path(root).resolve()):
            raise ValueError("canonical_generation_owner_outside_new_study")
        owner_run = reader.read_json(owner / "run_protocol.json")
        if (intent.get("request_sha256") != row["request_sha256"]
                or intent.get("run_protocol_sha256") != reader.digest(owner_run)
                or owner_run.get("study_protocol_sha256") != run["study_protocol_sha256"]
                or owner_run.get("code_sha256") != code_identity()):
            raise ValueError("canonical_intent_owner_run_disagreement")
        for field in ("response_text", "response_sha256", "finish_reason", "input_tensor_sha256", "usage",
                "raw_generated_token_ids", "raw_generated_text", "decoding_identity_sha256", "chosen_answer",
                "fixed_assistant_prefix", "prefilled_token_ids", "prefilled_token_count",
                "base_input_token_count", "actual_input_token_count", "base_input_tensor_sha256"):
            if answer.get(field) != value.get(field):
                raise ValueError("new_study_answer_cache_disagreement")
        if answer.get("answer_adapter") != adapter_module().adapt(value["response_text"]):
            raise ValueError("answer_adapter_differs_from_frozen_output_only_rule")
        adapter_seconds = answer.get("output_adaptation_wall_seconds")
        if type(adapter_seconds) not in (int, float) or not math.isfinite(adapter_seconds) or adapter_seconds < 0:
            raise ValueError("measured_nonnegative_output_adaptation_time_required")
        if (answer["input_tensor_sha256"] != expected_input["input_tensor_sha256"]
                or answer["usage"]["prompt_tokens"] != expected_input["input_tokens"]):
            raise ValueError("authenticated_answer_differs_from_actual_tensor_preflight")
        if "arm" in row and row["arm"] in ARMS and answer.get("source_checks_passed") is not True:
            raise ValueError("source_fences_not_attested")
    return answers, run



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=STUDY_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("initialize")
    init.add_argument("--old-root", required=True, type=Path)
    init.add_argument("--failed-v2-root", required=True, type=Path)
    args = parser.parse_args()
    print(reader.canonical(initialize(args.study_root, old_root=args.old_root,
        failed_v2_root=args.failed_v2_root)), flush=True)


if __name__ == "__main__":
    main()
