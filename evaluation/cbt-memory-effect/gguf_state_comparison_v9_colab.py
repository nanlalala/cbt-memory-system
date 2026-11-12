"""Prospective v9 idempotent-registration native-T4-GGUF 96-row source-gated development regression.

Maximum 57 unique attempts; actual deduplicated grid is frozen before inference.
An independently authenticated eight-answer reading gate is a hard prerequisite.
EOS-complete typed answers only, with actual native vectors/tokens and no repair.
The 7B Q4_K_M reader is a distinct condition from earlier 1.5B float32 results.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gguf_state_runtime_v9 import (reader, VERSION, STUDY_ROOT, CAP, PILOT_COUNT,
    MAIN_UNIQUE, SIDES, ARMS, PURPOSE, PLANNER, guard, study, initialize,
    ledger, generate_rows, authenticate, cost, schema_module, decoding_identity, GATE_ROOT, native, load_backend_for_phase,
    FAILED_MAIN_ROOT, FAILED_MAIN_PROTOCOL_SHA256)
import question_retrieval_policy_v9 as common

EXPECTED_CHECKOUTS = {
    "before": "e2899b10f7404847defb246ed7769aeb46cf228f",
    "after": "06223cffc09a6ab7972793cef78faa7364c4229d",
}


def common_identity():
    if Path(common.__file__).resolve() != HERE / "question_retrieval_policy_v9.py":
        raise RuntimeError("foreign_cached_common_retrieval_policy")
    return common.code_identity()


def require_stage_code(expected):
    if reader.file_hash(__file__) != expected:
        raise ValueError("frozen_main_stage_code_changed")


def require_main_identity(plan):
    require_stage_code(plan.get("main_stage_code_sha256"))
    if plan.get("common_retrieval_policy") != common_identity():
        raise ValueError("frozen_common_retrieval_policy_changed")


def verify_checkout(checkout, reference, *, side):
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if actual != EXPECTED_CHECKOUTS[side]:
        raise ValueError("frozen_product_git_commit_changed")
    protocol = reader.read_json(Path(reference) / "protocol.json")
    if reader.file_hash(Path(reference) / "requests.jsonl") != protocol["requests_file_sha256"]:
        raise ValueError("original_reference_requests_changed")
    for name, expected in protocol["code_identity"]["source_files"].items():
        if reader.file_hash(Path(checkout) / name) != expected:
            raise ValueError("frozen_source_checkout_differs_from_reference")
    return protocol


def fresh_database(path):
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        rows = connection.execute("SELECT purpose,status,COUNT(*) FROM dm_attempts GROUP BY purpose,status").fetchall()
    if rows:
        raise RuntimeError("fresh_source_gate_database_must_have_no_prior_model_admissions")


def render_side(root, side, checkout, expected_stage_sha256, expected_renderer_sha256):
    """Bind neutral requests using this side's real store at its original path."""
    root, checkout = Path(root).resolve(), Path(checkout).resolve()
    guard(root, checkout)
    require_stage_code(expected_stage_sha256)
    if common.renderer_sha256() != expected_renderer_sha256:
        raise ValueError("frozen_common_retrieval_renderer_changed")
    study(root)
    source, effective = root / "source_prepared" / side, root / "prepared" / side
    protocol = reader.read_json(source / "protocol.json")
    gates = source_gates({"checkout": str(checkout), "fresh_prepared": str(source),
        "source_code_identity": protocol["code_identity"]}, expected_stage_sha256=expected_stage_sha256)
    result = common.render_prepared(source, effective, source_gates=gates)
    require_stage_code(expected_stage_sha256)
    if common.renderer_sha256() != expected_renderer_sha256:
        raise ValueError("common_retrieval_renderer_changed_during_registration")
    return result


def audit_side(root, side, checkout, prepared, output, expected_stage_sha256, expected_renderer_sha256):
    """Fresh child only: validate source fences without admitting a generation."""
    root, checkout, prepared, output = (Path(p).resolve() for p in (root, checkout, prepared, output))
    guard(root, checkout, prepared, output)
    require_stage_code(expected_stage_sha256)
    if common.renderer_sha256() != expected_renderer_sha256:
        raise ValueError("frozen_common_retrieval_renderer_changed")
    study(root)
    if (side not in SIDES or prepared != root / "prepared" / side
            or output != root / "source_gate_audits" / (side + ".json")):
        raise ValueError("owned_side_preparation_and_audit_receipt_required")
    if output.exists():
        raise FileExistsError("existing_source_gate_audit_inspect_no_automatic_retry")
    protocol_hash = reader.file_hash(prepared / "protocol.json")
    requests_hash = reader.file_hash(prepared / "requests.jsonl")
    source_prepared = root / "source_prepared" / side
    protocol, rows = common.validate_prepared(prepared, source_prepared=source_prepared)
    if (len(rows) != 48 or len({(r["case_id"], r["arm"]) for r in rows}) != 48
            or requests_hash != protocol["requests_file_sha256"]):
        raise ValueError("complete_bound_side_required_for_source_gate_audit")
    databases = {Path(r["database_path"]).resolve() for r in rows}
    if len(databases) != 24 or any(not db.is_relative_to(source_prepared) for db in databases):
        raise ValueError("fresh_owned_side_databases_required")
    for db in databases:
        fresh_database(db)
    section = {"checkout": str(checkout), "fresh_prepared": str(prepared),
        "source_code_identity": protocol["code_identity"]}
    gates = source_gates(section, expected_stage_sha256=expected_stage_sha256)
    checks = []
    for row in rows:
        binding = gates.preflight(row)
        checks.append({"case_id": row["case_id"], "arm": row["arm"],
            "request_sha256": row["request_sha256"], "source_binding_sha256": reader.digest(binding)})
    for db in databases:
        fresh_database(db)
    require_stage_code(expected_stage_sha256)
    if (reader.file_hash(prepared / "protocol.json") != protocol_hash
            or reader.file_hash(prepared / "requests.jsonl") != requests_hash
            or gates.identity() != protocol["code_identity"]
            or common.renderer_sha256() != expected_renderer_sha256):
        raise ValueError("source_gate_audit_inputs_changed")
    receipt = {"version": VERSION + "-source-gate-audit", "side": side,
        "main_stage_code_sha256": expected_stage_sha256,
        "study_protocol_sha256": reader.file_hash(root / "study_protocol.json"),
        "checkout": str(checkout), "fresh_prepared": str(prepared),
        "fresh_protocol_sha256": protocol_hash, "requests_sha256": requests_hash,
        "source_code_identity": protocol["code_identity"], "new_database_count": len(databases),
        "source_prepared": str(source_prepared),
        "source_protocol_sha256": protocol["source_protocol_sha256"],
        "source_requests_sha256": protocol["source_requests_file_sha256"],
        "renderer_code_sha256": protocol["renderer_code_sha256"],
        "common_policy_sha256": protocol["common_policy_sha256"],
        "request_mapping_sha256": protocol["request_mapping_sha256"],
        "all_preflight_passed": True, "checked_row_count": len(checks), "ordered_checks": checks,
        "process_id": os.getpid(), "parent_process_id": os.getppid(),
        "new_model_calls": 0, "generation_admissions": 0, "automatic_retry": False,
        "completed_at_utc": reader.stamp()}
    reader._write_new(output, receipt)
    return receipt


def verify_audit_receipt(root, side, section, stage_sha256):
    path = Path(section["source_gate_audit_path"]).resolve()
    if (path != Path(root).resolve() / "source_gate_audits" / (side + ".json")
            or reader.file_hash(path) != section["source_gate_audit_sha256"]):
        raise ValueError("frozen_source_gate_audit_changed")
    receipt = reader.read_json(path)
    expected = {"version": VERSION + "-source-gate-audit", "side": side,
        "main_stage_code_sha256": stage_sha256,
        "study_protocol_sha256": reader.file_hash(Path(root) / "study_protocol.json"),
        **{key: section[key] for key in ("checkout", "fresh_prepared", "fresh_protocol_sha256",
            "requests_sha256", "source_code_identity", "new_database_count", "source_prepared",
            "source_protocol_sha256", "source_requests_sha256", "renderer_code_sha256",
            "common_policy_sha256", "request_mapping_sha256")},
        "all_preflight_passed": True, "checked_row_count": 48,
        "new_model_calls": 0, "generation_admissions": 0, "automatic_retry": False}
    rows = reader.read_rows(Path(section["fresh_prepared"]) / "requests.jsonl")
    checks = receipt.get("ordered_checks", [])
    if (any(receipt.get(key) != value for key, value in expected.items()) or len(checks) != len(rows)
            or [{key: check.get(key) for key in ("case_id", "arm", "request_sha256")} for check in checks]
                != [{key: row[key] for key in ("case_id", "arm", "request_sha256")} for row in rows]
            or any(type(check.get("source_binding_sha256")) is not str
                or len(check["source_binding_sha256"]) != 64 for check in checks)):
        raise ValueError("frozen_source_gate_audit_binding_changed")
    return receipt


def preparation_child(root, side, phase, command):
    started = time.monotonic()
    status, process = "failed_or_unknown", None
    try:
        process = subprocess.run(command, check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        status = "completed" if process.returncode == 0 else "failed"
        return process
    finally:
        elapsed = time.monotonic() - started
        reader._write_new(Path(root) / "preparation_timings" / (side + "_" + phase + ".json"),
            {"side": side, "phase": phase, "wall_seconds": elapsed, "status": status,
             "return_code": process.returncode if process is not None else None,
             "scope": "complete child process; source substages are not separately measured",
             "new_model_generations": 0, "automatic_retry": False})


def freeze_main(root, checkouts, references):
    root = Path(root).resolve()
    guard(root, *checkouts.values(), *references.values())
    stage_sha256 = reader.file_hash(__file__)
    renderer_identity = common_identity()
    study(root)
    intents, _ = ledger(root)
    if intents or (root / "main_freeze.json").exists():
        raise RuntimeError("freeze_once_before_any_new_v4_generation")
    freeze_started = time.monotonic()
    sections = {}
    all_payloads = set()
    raw_requests = {}
    databases = set()
    for side in SIDES:
        side_started = time.monotonic()
        checkout, reference = Path(checkouts[side]).resolve(), Path(references[side]).resolve()
        old = verify_checkout(checkout, reference, side=side)
        source = root / "source_prepared" / side
        fresh = root / "prepared" / side
        source.parent.mkdir(parents=True, exist_ok=True)
        if source.exists() or fresh.exists():
            raise FileExistsError("fresh_preparation_required_no_old_database_reuse")
        process = preparation_child(root, side, "source_prepare",
            [sys.executable, str(checkout / PLANNER), "prepare", "--output", str(source)])
        with (root / ("prepare_" + side + ".log")).open("x") as stream:
            stream.write(process.stdout)
        if process.returncode:
            raise RuntimeError("fresh_source_preparation_failed_no_model_calls")
        source_protocol = reader.read_json(source / "protocol.json")
        for name in ("case_file_sha256", "targets_file_sha256"):
            if source_protocol[name] != old[name]:
                raise ValueError("exposed_case_or_literal_target_changed")
        if (source_protocol["case_count"] != 24 or source_protocol["template_family_count"] != 4
                or source_protocol["code_identity"] != old["code_identity"]):
            raise ValueError("frozen_development_material_or_source_code_changed")
        source_rows, old_rows = reader.read_rows(source / "requests.jsonl"), reader.read_rows(reference / "requests.jsonl")
        wanted = {(r["case_id"], r["arm"]): r for r in old_rows}
        if len(source_rows) != 48 or len(wanted) != 48:
            raise ValueError("full_48_row_version_required")
        for row in source_rows:
            previous = wanted[(row["case_id"], row["arm"])]
            if row["request_payload"] != previous["request_payload"] or row["request_sha256"] != previous["request_sha256"]:
                raise ValueError("request_changed_from_exposed_development_reference")
            db = Path(row["database_path"]).resolve()
            if not db.is_relative_to(source) or db == Path(previous["database_path"]).resolve():
                raise ValueError("fresh_owned_database_required")
        source_dbs = {Path(r["database_path"]).resolve() for r in source_rows}
        if len(source_dbs) != 24:
            raise ValueError("independent_source_snapshot_databases_required")
        for db in source_dbs:
            fresh_database(db)
        process = preparation_child(root, side, "render_and_register", [sys.executable, str(Path(__file__).resolve()), "--study-root", str(root),
            "render-side", "--side", side, "--checkout", str(checkout),
            "--expected-stage-sha256", stage_sha256,
            "--expected-renderer-sha256", renderer_identity["renderer_code_sha256"]])
        with (root / ("render_" + side + ".log")).open("x") as stream:
            stream.write(process.stdout)
        if process.returncode:
            raise RuntimeError("authenticated_neutral_registration_failed_no_main_generation")
        new, rows = common.validate_prepared(fresh, source_prepared=source)
        if (new["case_file_sha256"] != old["case_file_sha256"]
                or new["targets_file_sha256"] != old["targets_file_sha256"]
                or new["code_identity"] != old["code_identity"] or len(rows) != 48):
            raise ValueError("neutral_render_changed_cases_labels_or_product_identity")
        source_by_key = {(r["case_id"], r["arm"]): r for r in source_rows}
        raw_requests[side] = {}
        for row in rows:
            original = source_by_key[(row["case_id"], row["arm"])]
            common.assert_payload_mapping(original["request_payload"], row["request_payload"])
            all_payloads.add(row["request_sha256"])
            if row["arm"] == "raw_history_rag":
                raw_requests[side][row["case_id"]] = row["request_payload"]
        side_dbs = {Path(r["database_path"]).resolve() for r in rows}
        if (len(side_dbs) != 24 or databases & side_dbs or side_dbs != source_dbs
                or any(not db.is_relative_to(source) for db in side_dbs)):
            raise ValueError("independent_fresh_version_databases_required")
        for db in side_dbs:
            fresh_database(db)
        databases |= side_dbs
        sections[side] = {"checkout": str(checkout), "reference_prepared": str(reference),
            "reference_protocol_sha256": reader.file_hash(reference / "protocol.json"),
            "fresh_prepared": str(fresh), "fresh_protocol_sha256": reader.file_hash(fresh / "protocol.json"),
            "requests_sha256": reader.file_hash(fresh / "requests.jsonl"), "case_file_sha256": new["case_file_sha256"],
            "targets_file_sha256": new["targets_file_sha256"], "source_code_identity": new["code_identity"],
            "new_database_count": len(side_dbs), "preparation_wall_seconds": None,
            "source_prepared": str(source), "source_protocol_sha256": reader.file_hash(source / "protocol.json"),
            "source_requests_sha256": reader.file_hash(source / "requests.jsonl"),
            "renderer_code_sha256": new["renderer_code_sha256"],
            "common_policy_sha256": new["common_policy_sha256"],
            "request_mapping_sha256": new["request_mapping_sha256"]}
        sections[side]["unique_request_count"] = len({r["request_sha256"] for r in rows})
        audit_path = root / "source_gate_audits" / (side + ".json")
        process = preparation_child(root, side, "source_audit", [sys.executable, str(Path(__file__).resolve()), "--study-root", str(root),
            "audit-side", "--side", side, "--checkout", str(checkout), "--prepared", str(fresh),
            "--output", str(audit_path), "--expected-stage-sha256", stage_sha256,
            "--expected-renderer-sha256", renderer_identity["renderer_code_sha256"]])
        with (root / ("source_gate_audit_" + side + ".log")).open("x") as stream:
            stream.write(process.stdout)
        if process.returncode:
            raise RuntimeError("fresh_source_gate_audit_failed_no_main_freeze_or_model_calls")
        sections[side].update(source_gate_audit_path=str(audit_path),
            source_gate_audit_sha256=reader.file_hash(audit_path))
        receipt = verify_audit_receipt(root, side, sections[side], stage_sha256)
        if receipt.get("parent_process_id") != os.getpid() or receipt.get("process_id") == os.getpid():
            raise ValueError("separate_fresh_source_gate_audit_process_required")
        sections[side]["preparation_wall_seconds"] = time.monotonic() - side_started
    if not 1 <= len(all_payloads) <= CAP:
        raise ValueError("actual_unique_grid_must_fit_predeclared_57_ceiling")
    if len(raw_requests["before"]) != 24 or raw_requests["before"] != raw_requests["after"]:
        raise ValueError("common_raw_baseline_must_match_across_product_versions")
    value = {"version": VERSION, "study_protocol_sha256": reader.file_hash(root / "study_protocol.json"),
        "main_stage_code_sha256": stage_sha256,
        "main_rendering_version": common.VERSION, "common_retrieval_policy": renderer_identity,
        "original_source_static_preparations_preserved": True,
        "source_databases_are_live_mutable_authorities": True, "live_database_relocation": False,
        "only_common_headers_and_provenance_redundant_fields_may_change": True,
        "no_response_adapter_or_repair": True,
        "decoding_identity": decoding_identity(), "independent_eight_gate_precondition": True,
        "sections": sections, "main_logical_rows": 96,
        "main_unique_payloads": len(all_payloads), "total_study_cap": CAP,
        "all_cases_exposed_development_only": True, "new_gold_read_for_freeze": False,
        "complete_before_scoring": True, "freeze_wall_seconds": time.monotonic() - freeze_started,
        "preparation_timing_scope": "whole side preparation and three child processes measured prospectively without product-code changes",
        "frozen_at_utc": reader.stamp()}
    require_main_identity(value)
    reader._write_new(root / "main_freeze.json", value)
    return {"stage": "full_development_grid_frozen_no_main_generation", "main_unique": len(all_payloads),
            "main_logical": 96, "maximum_new_study_attempts": CAP}


def frozen(root):
    study(root)
    value = reader.read_json(Path(root) / "main_freeze.json")
    require_main_identity(value)
    if (value["study_protocol_sha256"] != reader.file_hash(Path(root) / "study_protocol.json")
            or value.get("decoding_identity") != decoding_identity()):
        raise ValueError("main_freeze_decoder_binding_changed")
    if (set(value.get("sections", {})) != set(SIDES)
            or value.get("main_logical_rows") != 96 or value.get("total_study_cap") != CAP
            or type(value.get("main_unique_payloads")) is not int
            or not 1 <= value["main_unique_payloads"] <= CAP):
        raise ValueError("complete_bounded_main_grid_required")
    all_payloads, raw_controls = set(), {}
    for side, section in value["sections"].items():
        fresh = Path(section["fresh_prepared"])
        verify_checkout(section["checkout"], section["reference_prepared"], side=side)
        if (reader.file_hash(Path(section["reference_prepared"]) / "protocol.json") != section["reference_protocol_sha256"]
                or reader.file_hash(Path(section["source_prepared"]) / "protocol.json") != section["source_protocol_sha256"]
                or reader.file_hash(Path(section["source_prepared"]) / "requests.jsonl") != section["source_requests_sha256"]
                or reader.file_hash(fresh / "protocol.json") != section["fresh_protocol_sha256"]
                or reader.file_hash(fresh / "requests.jsonl") != section["requests_sha256"]):
            raise ValueError("frozen_fresh_preparation_changed")
        protocol, rows = common.validate_prepared(fresh, source_prepared=section["source_prepared"])
        if len(rows) != 48 or len({(r["case_id"], r["arm"]) for r in rows}) != 48:
            raise ValueError("complete_bound_48_row_side_required")
        if section.get("unique_request_count") != len({r["request_sha256"] for r in rows}):
            raise ValueError("frozen_per_side_unique_count_changed")
        all_payloads.update(r["request_sha256"] for r in rows)
        raw_controls[side] = {r["case_id"]: r["request_payload"] for r in rows if r["arm"] == ARMS[0]}
        for key in ("renderer_code_sha256", "common_policy_sha256", "request_mapping_sha256"):
            if protocol[key] != section[key]:
                raise ValueError("frozen_neutral_request_mapping_changed")
        verify_audit_receipt(root, side, section, value["main_stage_code_sha256"])
    if (len(all_payloads) != value["main_unique_payloads"]
            or len(raw_controls["before"]) != 24 or raw_controls["before"] != raw_controls["after"]):
        raise ValueError("frozen_actual_unique_count_or_common_raw_control_changed")
    return value


def source_gates(section, *, expected_stage_sha256):
    """Only call from a fresh CLI process, once for one frozen source version."""
    require_stage_code(expected_stage_sha256)
    if "controlled_open_parent_adapter" in sys.modules or "task_projector" in sys.modules:
        raise RuntimeError("fresh_process_required_for_version_specific_source_gates")
    path = Path(section["checkout"]) / PLANNER
    spec = importlib.util.spec_from_file_location("new_study_frozen_source_planner", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    gates = module.make_gates(section["fresh_prepared"])
    if gates.identity() != section["source_code_identity"]:
        raise ValueError("actual_source_gate_runtime_differs_from_frozen_version")
    return gates


def _run_side(root, side, *, constructor_log_path):
    root = Path(root).resolve()
    plan = frozen(root)
    require_main_identity(plan)
    section = plan["sections"][side]
    rows = reader.read_rows(Path(section["fresh_prepared"]) / "requests.jsonl")
    gates = source_gates(section, expected_stage_sha256=plan["main_stage_code_sha256"])
    for row in rows:
        gates.preflight(row)
    with ExitStack() as locks:
        for name in (".model_setup.lock", ".generation.lock"):
            locks.enter_context(reader._locked(root / name))
        intents, _ = ledger(root)
        expected = PILOT_COUNT if side == "before" else PILOT_COUNT + plan["sections"]["before"]["unique_request_count"]
        if len(intents) != expected:
            raise RuntimeError("fixed_order_complete_versions_required_no_resume")
        require_main_identity(plan)
        backend, _ = load_backend_for_phase(root, side, constructor_log_path=constructor_log_path)
        if side == "after":
            authenticate_main(root, "before", plan["sections"]["before"], backend)
        return generate_rows(root, root / ("answers_" + side), rows, backend,
            purpose=PURPOSE, gates=gates, metadata={"side": side, "main_freeze_sha256": reader.file_hash(root / "main_freeze.json"),
                "fresh_prepared_protocol_sha256": section["fresh_protocol_sha256"],
                "renderer_code_sha256": section["renderer_code_sha256"],
                "common_policy_sha256": section["common_policy_sha256"],
                "request_mapping_sha256": section["request_mapping_sha256"],
                "all_cases_exposed_development_only": True})


def run_side(root, side, *, constructor_log_path):
    root = Path(root).resolve()
    guard(root)
    started, complete = time.monotonic(), False
    try:
        result = _run_side(root, side, constructor_log_path=constructor_log_path)
        complete = True
        return result
    finally:
        if root.is_dir():
            reader._write_new(root / ("answer_phase_timing_" + side + ".json"), {
                "side": side, "complete": complete, "wall_seconds": time.monotonic() - started,
                "scope": "entire side CLI work through archive publication, including frozen validation, setup, gate revalidation, preflight, source fences and native/cache answers",
                "substage_timings_are_nonadditive_with_this_total": True})


def authenticate_main(root, side, section, backend):
    rows = reader.read_rows(Path(section["fresh_prepared"]) / "requests.jsonl")
    answers, run = authenticate(root, Path(root) / ("answers_" + side), rows, backend)
    expected = {"side": side, "main_freeze_sha256": reader.file_hash(Path(root) / "main_freeze.json"),
        "fresh_prepared_protocol_sha256": section["fresh_protocol_sha256"],
        **{key: section[key] for key in ("renderer_code_sha256", "common_policy_sha256", "request_mapping_sha256")},
        "all_cases_exposed_development_only": True}
    if run.get("metadata") != expected or run.get("gate_identity") != section["source_code_identity"]:
        raise ValueError("main_answer_renderer_or_actual_gate_identity_changed")
    return answers, run


def strict_answer(text):
    """Same single-key/type parser as the frozen original primary metric."""
    value = json.loads(text, object_pairs_hook=reader._unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_answer")))
    if type(value) is not dict or set(value) != {"answer"}:
        raise ValueError("single_answer_key_required")
    answer = value["answer"]
    if answer is not None and type(answer) not in (int, str):
        raise ValueError("invalid_answer_type")
    return answer


def strict_result(text, expected):
    try:
        value = strict_answer(text)
    except (ValueError, TypeError):
        return {"correct": False, "reason": "malformed_output", "observed_answer": None}
    correct = type(value) is type(expected) and value == expected
    reason = ("correct" if correct else "unsupported_assertion" if expected is None
        else "incorrect_abstention" if value is None else "output_type_error" if type(value) is not type(expected) else "wrong_value")
    return {"correct": correct, "reason": reason, "observed_answer": value}


def _score_main(root, *, constructor_log_path):
    root = Path(root).resolve()
    plan = frozen(root)
    intents, outcomes = ledger(root)
    if len(intents) != plan["main_unique_payloads"]:
        raise RuntimeError("all_frozen_unique_admissions_and_96_main_answers_must_finish_before_scoring")
    if (root / "reports").exists():
        raise FileExistsError("existing_scores_inspect_no_automatic_scoring_retry")
    backend, _ = load_backend_for_phase(root, "score-authentication", constructor_log_path=constructor_log_path)
    all_answers = {}
    for side in SIDES:
        section = plan["sections"][side]
        all_answers[side], _ = authenticate_main(root, side, section, backend)
    # No main target is decoded until both complete original answer archives authenticate.
    require_main_identity(plan)
    reports = {}
    for side in SIDES:
        section = plan["sections"][side]
        target_path = Path(section["fresh_prepared"]) / "assessor_only/targets.json"
        if reader.file_hash(target_path) != section["targets_file_sha256"]:
            raise ValueError("frozen_literal_labels_changed")
        targets = {x["case_id"]: x for x in reader.read_json(target_path)}
        scored = []
        for answer in all_answers[side]:
            target = targets[answer["case_id"]]
            complete = answer.get("finish_reason") == "eos" and answer.get("schema_validation", {}).get("accepted") is True
            strict = (strict_result(answer["response_text"], target["answer"]) if complete else
                {"correct": False, "reason": "unfinished_or_invalid_native_output", "observed_answer": None})
            question = next(r["request_payload"]["messages"][-1]["content"] for r in
                reader.read_rows(Path(section["fresh_prepared"]) / "requests.jsonl")
                if r["case_id"] == answer["case_id"] and r["arm"] == answer["arm"])
            schema_checked = schema_module().validate_response(answer["response_text"], schema_module().schema_for_question(question))
            scored.append({"case_id": answer["case_id"], "arm": answer["arm"],
                "source_family_id": target["source_family_id"], "language": target["language"],
                "expected_answer": target["answer"], "raw_strict": strict,
                "question_schema_valid": complete and schema_checked["accepted"],
                "finish_reason": answer["finish_reason"],
                "request_sha256": answer["request_sha256"], "response_sha256": answer["response_sha256"]})
        arms = {}
        for arm in ARMS:
            selected = [r for r in scored if r["arm"] == arm]
            arms[arm] = {"planned": 24, "question_schema_valid": sum(r["question_schema_valid"] for r in selected)}
            for metric in ("raw_strict",):
                arms[arm][metric] = {"correct": sum(r[metric]["correct"] for r in selected),
                    "errors": dict(Counter(r[metric]["reason"] for r in selected if not r[metric]["correct"]))}
        paired = {}
        for metric in ("raw_strict",):
            pairs = {case: {r["arm"]: r[metric]["correct"] for r in scored if r["case_id"] == case}
                for case in {r["case_id"] for r in scored}}
            paired[metric] = {"managed_wins": sum(p[ARMS[1]] and not p[ARMS[0]] for p in pairs.values()),
                "raw_wins": sum(p[ARMS[0]] and not p[ARMS[1]] for p in pairs.values()),
                "both_correct": sum(all(p.values()) for p in pairs.values()),
                "both_wrong": sum(not any(p.values()) for p in pairs.values()),
                "case_count": len(pairs), "independent_clinical_observations": False}
        reports[side] = {"version": VERSION + "-strict-direct-json-metric", "side": side,
            "all_material_exposed_development_regression_only": True, "clinical_quality_claim": False,
            "arms": arms, "paired_by_metric": paired,
            "raw_metric_is_unchanged_original_parser": True,
            "direct_generation_has_no_adapter_or_response_repair": True,
            "question_schema_policy": schema_module().policy(),
            "per_question": scored, "new_model_calls_by_scoring": 0, "human_reviews": 0,
            "answers_sha256": reader.file_hash(root / ("answers_" + side) / "answers_unscored.jsonl"),
            "main_freeze_sha256": reader.file_hash(root / "main_freeze.json"), "scorer_code_sha256": reader.file_hash(__file__)}
    events = [reader._checked(x) for x in reader.read_rows(root / "native_setup_events.jsonl")]
    completed_setups = [r for r in events if r["event"] == "native_model_setup_completed"]
    setup_receipts = []
    for event in completed_setups:
        path = Path(event["receipt_path"])
        if not path.resolve().is_relative_to(root) or reader.file_hash(path) != event["receipt_sha256"]:
            raise ValueError("owned_actual_native_setup_receipt_changed")
        setup_receipts.append({"phase": event["phase"], **reader.read_json(path)})
    gate_revalidations = [reader.read_json(root / ("gate_revalidation_" + phase + ".json"))
        for phase in ("before", "after", "score-authentication")]
    native_runs = [reader.read_json(root / ("answers_" + side) / "run_protocol.json") for side in SIDES]
    preparation_timing_paths = sorted((root / "preparation_timings").glob("*.json"))
    preparation_timings = [reader.read_json(path) for path in preparation_timing_paths]
    current_study = study(root)
    material_validation_timings = [reader._checked(row) for row in reader.read_rows(root / "frozen_material_validation_timings.jsonl")]
    resources = {"new_study_unique_generation_cost": cost(outcomes, intents),
        "by_purpose": {p: cost(outcomes, [k for k, i in intents.items() if i["purpose"] == p]) for p in sorted({i["purpose"] for i in intents.values()})},
        "actual_new_model_setup_events": events, "actual_new_model_setup_receipts": setup_receipts,
        "actual_new_model_setup_seconds_sum": sum(r["setup_seconds"] for r in setup_receipts),
        "native_setup_substage_timings_nonadditive_with_setup_total": True,
        "historical_setup_receipts_are_nonadditive": True,
        "all_historical_studies_gates_and_failed_preparations_costs_are_nonadditive": True,
        "preserved_failed_v8_main": {
            "protocol_sha256": current_study["failed_v8_main_protocol_sha256"],
            "generation_admissions": current_study["failed_v8_main_generation_admissions"],
            "binding_diagnostic_path": "/content/cbt-gpu-v8-binding-diagnostic-20261112-v1.json",
            "cost_scope": "original V8 preparation receipts and transitive V7 history; excluded from new V9 generation and setup totals"},
        "response_adapter_used": False,
        "frozen_material_validation_wall_seconds_sum": sum(row["wall_seconds"] for row in material_validation_timings),
        "frozen_material_validation_calls": len(material_validation_timings),
        "validation_timing_scope": "full immutable code/protocol/historical/setup SHA256 checks; includes the complete CUDA wheel each validation",
        "frozen_material_validation_timings_nonadditive_with_other_paths": True,
        "whole_answer_phase_timings": {side: reader.read_json(root / ("answer_phase_timing_" + side + ".json")) for side in SIDES},
        "native_input_preflight_wall_seconds_sum": sum(r["native_input_preflight_wall_seconds"] for r in native_runs),
        "reader_gate_revalidations": gate_revalidations,
        "reader_gate_revalidation_wall_seconds_sum": sum(r["wall_seconds"] for r in gate_revalidations),
        "preparation_child_timings": preparation_timings,
        "whole_side_preparation_wall_seconds": {side: plan["sections"][side]["preparation_wall_seconds"] for side in SIDES},
        "whole_freeze_wall_seconds": plan["freeze_wall_seconds"],
        "initialization_and_gate_byte_verification_seconds": current_study["initialization_and_lightweight_gate_verification_wall_seconds"],
        "preparation_child_timings_nonadditive_with_whole_side_total": True,
        "logical_cache_reuse_source_validation_and_publication_wall_seconds": sum(
            answer["logical_processing_wall_seconds"] for answers in all_answers.values()
            for answer in answers if answer["canonical_generation_shared"]),
        "cache_reuse_timing_scope": "source admission/fences, native cache reproduction and checks before logical-row serialization; excludes canonical generation",
        "source_ingestion_wall_seconds": None, "extraction_wall_seconds": None,
        "indexing_organization_wall_seconds": None, "state_projection_wall_seconds": None,
        "retrieval_packing_wall_seconds": None, "preparation_timing_status": "whole_side_and_subprocess_wall_measured; ingestion/projection/packing not separately instrumented",
        "tokenizer_verification_wall_seconds": None, "schema_candidate_compilation_wall_seconds": None,
        "monetary_cost": None, "new_paid_provider_calls": 0}
    for side, value in reports.items():
        reader._write_new(root / "reports" / (side + "_metrics.json"), value)
    reader._write_new(root / "reports/resources.json", resources)
    summary = {"version": VERSION, "main_logical_rows": 96, "main_unique_generations": plan["main_unique_payloads"],
        "pilot_generations": PILOT_COUNT, "new_study_unique_generations": len(intents), "study_cap": CAP,
        "arms_by_version": {side: value["arms"] for side, value in reports.items()},
        "all_cases_exposed_development_only": True, "clinical_quality_claim": False,
        "new_provider_calls": 0, "new_human_reviews": 0}
    reader._write_new(root / "reports/summary.json", summary)
    return summary


def score_main(root, *, constructor_log_path):
    root = Path(root).resolve()
    guard(root)
    started, complete = time.monotonic(), False
    try:
        result = _score_main(root, constructor_log_path=constructor_log_path)
        complete = True
        return result
    finally:
        if root.is_dir():
            reader._write_new(root / "score_phase_timing.json", {
                "complete": complete, "wall_seconds": time.monotonic() - started,
                "scope": "whole score CLI through report publication, including validation, native load and all96 authentication before labels",
                "substage_timings_are_nonadditive_with_this_total": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=STUDY_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("initialize")
    init.add_argument("--gate-root", type=Path, default=GATE_ROOT)
    init.add_argument("--expected-gate-protocol-sha256", required=True)
    init.add_argument("--expected-gate-report-sha256", required=True)
    init.add_argument("--expected-gate-authentication-sha256", required=True)
    init.add_argument("--failed-main-root", type=Path, default=FAILED_MAIN_ROOT)
    init.add_argument("--expected-failed-main-protocol-sha256", default=FAILED_MAIN_PROTOCOL_SHA256)
    render = sub.add_parser("render-side")
    render.add_argument("--side", choices=SIDES, required=True)
    render.add_argument("--checkout", required=True, type=Path)
    render.add_argument("--expected-stage-sha256", required=True)
    render.add_argument("--expected-renderer-sha256", required=True)
    audit = sub.add_parser("audit-side")
    audit.add_argument("--side", choices=SIDES, required=True)
    for name in ("checkout", "prepared", "output"):
        audit.add_argument("--" + name, required=True, type=Path)
    audit.add_argument("--expected-stage-sha256", required=True)
    audit.add_argument("--expected-renderer-sha256", required=True)
    freeze = sub.add_parser("freeze-main")
    for side in SIDES:
        freeze.add_argument("--" + side + "-checkout", required=True, type=Path)
        freeze.add_argument("--" + side + "-reference", required=True, type=Path)
    run = sub.add_parser("run-side"); run.add_argument("--side", choices=SIDES, required=True)
    run.add_argument("--constructor-log-path", required=True, type=Path)
    score = sub.add_parser("score")
    score.add_argument("--constructor-log-path", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "initialize": result = initialize(args.study_root, gate_root=args.gate_root,
        expected_gate_protocol_sha256=args.expected_gate_protocol_sha256,
        expected_gate_report_sha256=args.expected_gate_report_sha256,
        expected_gate_authentication_sha256=args.expected_gate_authentication_sha256,
        failed_main_root=args.failed_main_root,
        expected_failed_main_protocol_sha256=args.expected_failed_main_protocol_sha256)
    elif args.command == "render-side": result = render_side(args.study_root, args.side, args.checkout,
        args.expected_stage_sha256, args.expected_renderer_sha256)
    elif args.command == "audit-side": result = audit_side(args.study_root, args.side, args.checkout,
        args.prepared, args.output, args.expected_stage_sha256, args.expected_renderer_sha256)
    elif args.command == "freeze-main":
        result = freeze_main(args.study_root, {s: getattr(args, s + "_checkout") for s in SIDES},
            {s: getattr(args, s + "_reference") for s in SIDES})
    elif args.command == "run-side": result = run_side(args.study_root, args.side, constructor_log_path=args.constructor_log_path)
    else: result = score_main(args.study_root, constructor_log_path=args.constructor_log_path)
    print(reader.canonical(result), flush=True)


if __name__ == "__main__":
    main()
