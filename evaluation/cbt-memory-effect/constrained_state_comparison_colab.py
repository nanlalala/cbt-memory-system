"""Separate full stage: freeze only after immutable four-pilot acceptance.

The pilot/runtime module has its own frozen code identity. This file is frozen
independently for the exposed 96-row/57-generation source-gated main stage.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_prefill_pilot_v2_colab import (reader, VERSION, STUDY_ROOT, CAP, PILOT_COUNT,
    MAIN_UNIQUE, SIDES, ARMS, PURPOSE, PLANNER, guard, study, helpers, initialize,
    pilot_run, pilot_passed, ledger, generate_rows, authenticate, cost)
import common_retrieval_policy_v1 as common


def common_identity():
    if Path(common.__file__).resolve() != HERE / "common_retrieval_policy_v1.py":
        raise RuntimeError("foreign_cached_common_retrieval_policy")
    return common.code_identity()


def require_stage_code(expected):
    if reader.file_hash(__file__) != expected:
        raise ValueError("frozen_main_stage_code_changed")


def require_main_identity(plan):
    require_stage_code(plan.get("main_stage_code_sha256"))
    if plan.get("common_retrieval_policy") != common_identity():
        raise ValueError("frozen_common_retrieval_policy_changed")


def verify_checkout(checkout, reference):
    protocol = reader.read_json(Path(reference) / "protocol.json")
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


def freeze_main(root, checkouts, references):
    root = Path(root).resolve()
    guard(root, *checkouts.values(), *references.values())
    stage_sha256 = reader.file_hash(__file__)
    renderer_identity = common_identity()
    study(root)
    acceptance = pilot_passed(root)
    intents, _ = ledger(root)
    if len(intents) != PILOT_COUNT or (root / "main_freeze.json").exists():
        raise RuntimeError("freeze_once_after_only_four_successful_pilots")
    sections = {}
    all_payloads = set()
    raw_requests = {}
    databases = set()
    for side in SIDES:
        checkout, reference = Path(checkouts[side]).resolve(), Path(references[side]).resolve()
        old = verify_checkout(checkout, reference)
        source = root / "source_prepared" / side
        fresh = root / "prepared" / side
        source.parent.mkdir(parents=True, exist_ok=True)
        if source.exists() or fresh.exists():
            raise FileExistsError("fresh_preparation_required_no_old_database_reuse")
        process = subprocess.run([sys.executable, str(checkout / PLANNER), "prepare", "--output", str(source)],
            check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--study-root", str(root),
            "render-side", "--side", side, "--checkout", str(checkout),
            "--expected-stage-sha256", stage_sha256,
            "--expected-renderer-sha256", renderer_identity["renderer_code_sha256"]],
            check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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
        if side == "before" and len({r["request_sha256"] for r in rows}) != 48:
            raise ValueError("fixed_before_stage_requires_48_unique_requests")
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
        audit_path = root / "source_gate_audits" / (side + ".json")
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--study-root", str(root),
            "audit-side", "--side", side, "--checkout", str(checkout), "--prepared", str(fresh),
            "--output", str(audit_path), "--expected-stage-sha256", stage_sha256,
            "--expected-renderer-sha256", renderer_identity["renderer_code_sha256"]],
            check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        with (root / ("source_gate_audit_" + side + ".log")).open("x") as stream:
            stream.write(process.stdout)
        if process.returncode:
            raise RuntimeError("fresh_source_gate_audit_failed_no_main_freeze_or_model_calls")
        sections[side].update(source_gate_audit_path=str(audit_path),
            source_gate_audit_sha256=reader.file_hash(audit_path))
        receipt = verify_audit_receipt(root, side, sections[side], stage_sha256)
        if receipt.get("parent_process_id") != os.getpid() or receipt.get("process_id") == os.getpid():
            raise ValueError("separate_fresh_source_gate_audit_process_required")
    if len(all_payloads) != MAIN_UNIQUE:
        raise ValueError("predeclared_57_unique_payload_grid_changed_no_generation")
    if len(raw_requests["before"]) != 24 or raw_requests["before"] != raw_requests["after"]:
        raise ValueError("common_raw_baseline_must_match_across_product_versions")
    value = {"version": VERSION, "study_protocol_sha256": reader.file_hash(root / "study_protocol.json"),
        "main_stage_code_sha256": stage_sha256,
        "main_rendering_version": common.VERSION, "common_retrieval_policy": renderer_identity,
        "original_source_static_preparations_preserved": True,
        "source_databases_are_live_mutable_authorities": True, "live_database_relocation": False,
        "only_common_instruction_headers_change_from_source_payloads": True,
        "pilot_acceptance": acceptance, "sections": sections, "main_logical_rows": 96,
        "main_unique_payloads": MAIN_UNIQUE, "total_study_cap": CAP,
        "all_cases_exposed_development_only": True, "new_gold_read_for_freeze": False,
        "complete_before_scoring": True, "frozen_at_utc": reader.stamp()}
    require_main_identity(value)
    reader._write_new(root / "main_freeze.json", value)
    return {"stage": "full_development_grid_frozen_no_main_generation", "main_unique": MAIN_UNIQUE,
            "main_logical": 96, "maximum_new_study_attempts": CAP}


def frozen(root):
    study(root)
    value = reader.read_json(Path(root) / "main_freeze.json")
    require_main_identity(value)
    if value["study_protocol_sha256"] != reader.file_hash(Path(root) / "study_protocol.json") or value["pilot_acceptance"] != pilot_passed(root):
        raise ValueError("main_freeze_or_pilot_binding_changed")
    for side, section in value["sections"].items():
        fresh = Path(section["fresh_prepared"])
        verify_checkout(section["checkout"], section["reference_prepared"])
        if (reader.file_hash(Path(section["reference_prepared"]) / "protocol.json") != section["reference_protocol_sha256"]
                or reader.file_hash(Path(section["source_prepared"]) / "protocol.json") != section["source_protocol_sha256"]
                or reader.file_hash(Path(section["source_prepared"]) / "requests.jsonl") != section["source_requests_sha256"]
                or reader.file_hash(fresh / "protocol.json") != section["fresh_protocol_sha256"]
                or reader.file_hash(fresh / "requests.jsonl") != section["requests_sha256"]):
            raise ValueError("frozen_fresh_preparation_changed")
        protocol, _ = common.validate_prepared(fresh, source_prepared=section["source_prepared"])
        for key in ("renderer_code_sha256", "common_policy_sha256", "request_mapping_sha256"):
            if protocol[key] != section[key]:
                raise ValueError("frozen_neutral_request_mapping_changed")
        verify_audit_receipt(root, side, section, value["main_stage_code_sha256"])
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


def run_side(root, side):
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
        expected = PILOT_COUNT if side == "before" else PILOT_COUNT + 48
        if len(intents) != expected:
            raise RuntimeError("fixed_order_complete_versions_required_no_resume")
        if side == "after":
            authenticate_main(root, "before", plan["sections"]["before"])
        require_main_identity(plan)
        model, tokenizer, _ = reader._load_cached_model(root)
        grammar, _ = helpers()
        trie = grammar.AnswerTrie(tokenizer, model.generation_config.eos_token_id)
        if trie.identity != plan["pilot_acceptance"]["decoding_identity"]:
            raise ValueError("main_decoder_differs_from_successful_pilot")
        return generate_rows(root, root / ("answers_" + side), rows, model, tokenizer, trie,
            purpose=PURPOSE, gates=gates, metadata={"side": side, "main_freeze_sha256": reader.file_hash(root / "main_freeze.json"),
                "fresh_prepared_protocol_sha256": section["fresh_protocol_sha256"],
                "renderer_code_sha256": section["renderer_code_sha256"],
                "common_policy_sha256": section["common_policy_sha256"],
                "request_mapping_sha256": section["request_mapping_sha256"],
                "all_cases_exposed_development_only": True})


def authenticate_main(root, side, section):
    rows = reader.read_rows(Path(section["fresh_prepared"]) / "requests.jsonl")
    answers, run = authenticate(root, Path(root) / ("answers_" + side), rows)
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


def score_main(root):
    root = Path(root).resolve()
    plan = frozen(root)
    intents, outcomes = ledger(root)
    if len(intents) != CAP:
        raise RuntimeError("all_61_admissions_and_96_main_answers_must_finish_before_scoring")
    all_answers = {}
    for side in SIDES:
        section = plan["sections"][side]
        all_answers[side], _ = authenticate_main(root, side, section)
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
            scored.append({"case_id": answer["case_id"], "arm": answer["arm"],
                "source_family_id": target["source_family_id"], "language": target["language"],
                "expected_answer": target["answer"], **strict_result(answer["response_text"], target["answer"]),
                "request_sha256": answer["request_sha256"], "response_sha256": answer["response_sha256"]})
        reports[side] = {"version": VERSION + "-original-strict-metric", "side": side,
            "all_material_exposed_development_regression_only": True, "clinical_quality_claim": False,
            "arms": {arm: {"correct": sum(r["correct"] for r in scored if r["arm"] == arm), "planned": 24,
                "errors": dict(Counter(r["reason"] for r in scored if r["arm"] == arm and not r["correct"]))} for arm in ARMS},
            "per_question": scored, "new_model_calls_by_scoring": 0, "human_reviews": 0,
            "answers_sha256": reader.file_hash(root / ("answers_" + side) / "answers_unscored.jsonl"),
            "main_freeze_sha256": reader.file_hash(root / "main_freeze.json"), "scorer_code_sha256": reader.file_hash(__file__)}
    events = [reader._checked(x) for x in reader.read_rows(root / "model_load_events.jsonl")]
    resources = {"new_study_unique_generation_cost": cost(outcomes, intents),
        "by_purpose": {p: cost(outcomes, [k for k, i in intents.items() if i["purpose"] == p]) for p in sorted({i["purpose"] for i in intents.values()})},
        "actual_new_model_setup_events": events, "historical_setup_receipts_are_nonadditive": True,
        "old_64_generation_cost_not_added_to_new_study": True,
        "source_ingestion_wall_seconds": None, "extraction_wall_seconds": None,
        "indexing_organization_wall_seconds": None, "state_projection_wall_seconds": None,
        "retrieval_packing_wall_seconds": None, "preparation_timing_status": "not_measured",
        "monetary_cost": None, "new_paid_provider_calls": 0}
    for side, value in reports.items():
        reader._write_new(root / "reports" / (side + "_strict.json"), value)
    reader._write_new(root / "reports/resources.json", resources)
    summary = {"version": VERSION, "main_logical_rows": 96, "main_unique_generations": MAIN_UNIQUE,
        "pilot_generations": PILOT_COUNT, "new_study_unique_generations": len(intents), "study_cap": CAP,
        "arms_by_version": {side: value["arms"] for side, value in reports.items()},
        "all_cases_exposed_development_only": True, "clinical_quality_claim": False,
        "new_provider_calls": 0, "new_human_reviews": 0}
    reader._write_new(root / "reports/summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=STUDY_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("initialize"); init.add_argument("--old-root", required=True, type=Path)
    sub.add_parser("pilot")
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
    sub.add_parser("score")
    args = parser.parse_args()
    if args.command == "initialize": result = initialize(args.study_root, old_root=args.old_root)
    elif args.command == "pilot": result = pilot_run(args.study_root)
    elif args.command == "render-side": result = render_side(args.study_root, args.side, args.checkout,
        args.expected_stage_sha256, args.expected_renderer_sha256)
    elif args.command == "audit-side": result = audit_side(args.study_root, args.side, args.checkout,
        args.prepared, args.output, args.expected_stage_sha256, args.expected_renderer_sha256)
    elif args.command == "freeze-main":
        result = freeze_main(args.study_root, {s: getattr(args, s + "_checkout") for s in SIDES},
            {s: getattr(args, s + "_reference") for s in SIDES})
    elif args.command == "run-side": result = run_side(args.study_root, args.side)
    else: result = score_main(args.study_root)
    print(reader.canonical(result), flush=True)


if __name__ == "__main__":
    main()
