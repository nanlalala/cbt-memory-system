#!/usr/bin/env python3
"""Read-only, Colab-only derived export for the frozen 151170 full suite.

Run after the finite launcher has stopped. This helper never calls a provider,
loads an embedding, repairs a journal, scores into the run, or reads a Secret.
The controlled answer export contains original model output, not benchmark
questions, histories, options, or gold references. Do not publish it by default.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile

EXECUTION_COMMIT = "1511706b3a4b0ac71416998b5b4d752a13c4e313"
EXPECTED_PLAN = "940a2b3d8fa02aa683f7855f94931ff1d9710c7835b598c70fc201582ae8d568"
SCHEMA = "capstone-derived-readonly-v1"
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
JOURNALS = ("response_attempts.jsonl", "response_outcomes.jsonl", "responses.jsonl", "failures.jsonl")
FORBIDDEN_METADATA_KEYS = {"history", "query", "question", "questions", "messages", "accepted_answers", "correct_option", "gold", "api_key", "secret", "authorization", "credentials", "access_token"}


def digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def stable(value) -> str:
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True).encode())


def safe_file(path: Path, *, optional=False) -> bytes | None:
    if path.is_symlink():
        raise ValueError("symlinked input rejected")
    if not path.exists() and optional:
        return None
    if not path.is_file():
        raise ValueError("expected regular artifact file")
    return path.read_bytes()


def read_json(path: Path, *, optional=False):
    data = safe_file(path, optional=optional)
    return None if data is None else json.loads(data)


def read_rows(path: Path):
    body = safe_file(path, optional=True)
    return [] if body is None else [json.loads(line) for line in body.splitlines() if line.strip()]


def metadata_safe(value):
    """Fail closed rather than silently exporting a new raw-data metadata field."""
    if isinstance(value, dict):
        for key, item in value.items():
            count_only = str(key).lower() in {"questions", "question"} and isinstance(item, int) and not isinstance(item, bool)
            file_identity_only = str(key).lower() == "questions" and isinstance(item, dict) and set(item) == {"name", "bytes", "sha256"}
            if str(key).lower() in FORBIDDEN_METADATA_KEYS and not (count_only or file_identity_only):
                raise ValueError("unexpected restricted field in metadata")
            metadata_safe(item)
    elif isinstance(value, list):
        for item in value:
            metadata_safe(item)


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def usage_safe(usage):
    """Only numeric usage counters, never arbitrary provider response metadata."""
    if not isinstance(usage, dict):
        return {}
    result = {k: usage[k] for k in (*TOKEN_FIELDS, "prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "reasoning_tokens") if number(usage.get(k))}
    for key, fields in (("prompt_tokens_details", ("cached_tokens",)), ("completion_tokens_details", ("reasoning_tokens",))):
        details = usage.get(key)
        if isinstance(details, dict):
            chosen = {k: details[k] for k in fields if number(details.get(k))}
            if chosen:
                result[key] = chosen
    return result


def source_observations(trace):
    """IDs/counts only; linked entity names and source text are not exported."""
    result = {}
    for key in ("encoding", "selection_representation", "retrieval_version", "source_unit_count", "source_chunk_count", "stored_quote_count", "storage_inventory_sha256", "current_use", "interactive_clarification"):
        if key in trace:
            result[key] = trace[key]
    for key in ("retrieved_ids", "included_short_ids", "included_long_ids", "budget_dropped_short_ids", "budget_dropped_long_ids"):
        values = trace.get(key, [])
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise ValueError("unexpected source ID structure")
        result[key] = values
        result[key + "_count"] = len(values)
    for key in ("storage", "excluded_source_units", "included_source_spans"):
        values = trace.get(key, [])
        if isinstance(values, list):
            result[key + "_count"] = len(values)
    result["sufficiency_annotation_available"] = False
    result["causal_stage_attribution"] = "unresolved_without_independent_evidence_annotations"
    return result


def diagnostics_safe(supplemental):
    diagnostic = supplemental.get("format", {})
    fields = ("unavailable", "within_short_answer_limits", "abstention_cue_detected", "abstention_pattern_indices", "disjunction_cue_detected", "enumeration_cue_detected", "bare_option_label", "short_single_line_without_abstention")
    return {key: diagnostic[key] for key in fields if key in diagnostic}


def empty_supplemental():
    return {"official_score": 0.0, "exact_alias_or_option_score": 0.0, "strict_committed_score": 0.0, "official_positive_with_abstention_cue": False, "official_positive_not_strict": False, "format": {"unavailable": True}}


def new_cost():
    return {"recorded_provider_attempt_intents": 0, "intents_without_outcome": 0,
            "successful_unique_provider_calls": 0, "failed_provider_attempts": 0,
            "failed_attempts_without_token_usage": 0, "successful_calls_without_token_usage": 0,
            "reported_tokens_including_failed_attempts_when_supplied": {k: 0 for k in TOKEN_FIELDS},
            "usage_field_missing_counts": {"successful_calls": {k: 0 for k in TOKEN_FIELDS}, "failed_attempts": {k: 0 for k in TOKEN_FIELDS}},
            "all_recorded_valid_response_count": 0, "saved_responses_without_success_outcome": 0,
            "sum_successful_api_latency_seconds": 0.0, "median_successful_api_latency_seconds": None,
            "latency_observation_count": 0, "successful_calls_without_latency": 0,
            "monetary_cost": None, "cross_shard_deduplication": False}


def audit_shard_cost(shard_id, attempts, outcomes, raw_responses):
    cost = new_cost()
    ids = lambda r: (r["canonical_request_key"], r["attempt_index"])
    imap, omap, rmap = ({ids(r): r for r in rows} for rows in (attempts, outcomes, raw_responses))
    if len(imap) != len(attempts) or len(omap) != len(outcomes) or len(rmap) != len(raw_responses):
        raise ValueError("duplicate attempt identity in shard journals")
    if set(omap) - set(imap) or set(rmap) - set(imap):
        raise ValueError("response/outcome without recorded attempt")
    for row in raw_responses:
        if row.get("response_sha256") != stable({k: v for k, v in row.items() if k != "response_sha256"}):
            raise ValueError("response integrity mismatch")
    cost["recorded_provider_attempt_intents"] = len(imap)
    cost["intents_without_outcome"] = len(set(imap) - set(omap))
    cost["all_recorded_valid_response_count"] = len(rmap)
    exported, latencies, models = [], [], Counter()
    for key, intent in sorted(imap.items()):
        outcome, response = omap.get(key), rmap.get(key)
        status = outcome["status"] if outcome else "inflight"
        if status == "identity_or_integrity_failure":
            raise ValueError("identity/integrity failure prohibits formal export")
        if status == "success" and response is None:
            raise ValueError("successful outcome without validated response")
        if response and status != "success":
            cost["saved_responses_without_success_outcome"] += 1
        usage = usage_safe(response.get("usage")) if status == "success" else usage_safe((outcome or {}).get("usage"))
        measured = any(k in usage for k in TOKEN_FIELDS)
        if status == "success":
            cost["successful_unique_provider_calls"] += 1
            cost["successful_calls_without_token_usage"] += not measured
            kind = "successful_calls"
            latency = response.get("latency_seconds")
            if number(latency):
                latencies.append(latency)
            else:
                cost["successful_calls_without_latency"] += 1
            models[response["provider_model"]] += 1
        elif outcome:
            cost["failed_provider_attempts"] += 1
            cost["failed_attempts_without_token_usage"] += not measured
            kind = "failed_attempts"
        else:
            kind = None
        if kind:
            for token in TOKEN_FIELDS:
                cost["usage_field_missing_counts"][kind][token] += token not in usage
                cost["reported_tokens_including_failed_attempts_when_supplied"][token] += usage.get(token, 0)
        exported.append({"shard_id": shard_id, "canonical_request_key": key[0], "attempt_index": key[1],
                         "protocol_sha256": intent["protocol_sha256"], "prompt_sha256": intent["prompt_sha256"],
                         "started_at_unix": intent["started_at_unix"], "status": status,
                         "retryable": outcome.get("retryable") if outcome else None,
                         "error_type": outcome.get("error_type") if outcome else None,
                         "provider_model": response.get("provider_model") if response else None,
                         "latency_seconds": response.get("latency_seconds") if response else None,
                         "reported_usage": usage, "response_sha256": response.get("response_sha256") if response else None,
                         "provider_receipt_or_billing_verified": False})
    cost["sum_successful_api_latency_seconds"] = sum(latencies)
    cost["median_successful_api_latency_seconds"] = statistics.median(latencies) if latencies else None
    cost["latency_observation_count"] = len(latencies)
    cost["reported_provider_models"] = dict(sorted(models.items()))
    return cost, exported, latencies


def sum_costs(costs, latencies):
    result = new_cost()
    nested = {"reported_tokens_including_failed_attempts_when_supplied", "usage_field_missing_counts"}
    calculated = {"median_successful_api_latency_seconds", "monetary_cost", "cross_shard_deduplication"}
    for cost in costs:
        for key in result:
            if key not in nested | calculated:
                result[key] += cost[key]
        for token in TOKEN_FIELDS:
            result["reported_tokens_including_failed_attempts_when_supplied"][token] += cost["reported_tokens_including_failed_attempts_when_supplied"][token]
            for kind in ("successful_calls", "failed_attempts"):
                result["usage_field_missing_counts"][kind][token] += cost["usage_field_missing_counts"][kind][token]
    result["median_successful_api_latency_seconds"] = statistics.median(latencies) if latencies else None
    result["latency_observation_count"] = len(latencies)
    models = Counter()
    for cost in costs:
        models.update(cost["reported_provider_models"])
    result["reported_provider_models"] = dict(sorted(models.items()))
    return result


def compare_cost(original, actual):
    for key, value in original.items():
        if key not in actual or key in {"boundary"}:
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isclose(value, actual[key], rel_tol=1e-12, abs_tol=1e-8):
                raise ValueError("original cost differs from independent journal audit")
        elif value != actual[key]:
            raise ValueError("original cost differs from independent journal audit")


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows))


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Do not mutate the original output to prevent spreadsheet formulas.
            writer.writerow({key: ("'" + value if isinstance(value, str) and value.startswith(("=", "+", "-", "@")) else value) for key, value in row.items() if key in fields})


def export(args):
    # Do this before touching any experimental file or importing its modules.
    if not (Path("/content").is_dir() and os.environ.get("COLAB_RELEASE_TAG") and importlib.util.find_spec("google.colab") is not None):
        raise RuntimeError("This reporting helper requires genuine hosted Google Colab; no experiment data was read")
    repo, work, out = (Path(p).expanduser().resolve() for p in (args.repo, args.work, args.out))
    suite = work / "full-800-589"
    if any(not p.is_relative_to(Path("/content")) for p in (repo, work, out)) or out.exists() or out.is_relative_to(repo) or out.is_relative_to(suite):
        raise ValueError("use fixed /content checkout/work and a fresh output outside repo/suite")
    for path in (Path(args.repo), Path(args.work), Path(args.out), suite):
        if path.is_symlink():
            raise ValueError("symlinked root rejected")
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != EXECUTION_COMMIT or subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD"], check=False).returncode:
        raise ValueError("reporting requires clean frozen executable commit 151170")
    module_dir = repo / "evaluation/objective-memory"
    sys.path.insert(0, str(module_dir))
    runner = importlib.import_module("v3_run_objective_comparison")
    driver = importlib.import_module("v3_suite_driver")
    scorer = importlib.import_module("score_objective_comparison")
    diagnostic = importlib.import_module("v3_response_diagnostics")
    for module in (runner, driver, scorer, diagnostic):
        if not Path(module.__file__).resolve().is_relative_to(module_dir):
            raise ValueError("reporting imported an unexpected module")
    runner.require_colab()
    source_hashes, code_hashes, costs, attempts_export, rows_export, shard_summaries = {}, {}, [], [], [], []
    all_latencies, dataset_costs, dataset_latencies = [], defaultdict(list), defaultdict(list)
    all_counts = Counter()
    original_scores, summary = None, None
    with ExitStack() as locks:
        locks.enter_context(runner._readonly_existing_lock(suite))
        plan = driver._load_plan(suite)
        if plan.get("suite_plan_sha256") != EXPECTED_PLAN or plan.get("all_cases") != 1389 or plan.get("all_arm_rows") != 4167 or len(plan.get("shards", [])) != 12:
            raise ValueError("unexpected full-suite frozen identity/grid")
        launcher = read_json(work / "launcher_manifest.json")
        if launcher.get("executable") != EXECUTION_COMMIT:
            raise ValueError("launcher executable differs")
        metadata_safe(plan)
        metadata_safe(launcher)
        for path in (suite / "suite_plan.json", work / "launcher_manifest.json"):
            source_hashes[str(path.relative_to(work))] = digest(safe_file(path))
        summary = read_json(suite / "suite_summary.json", optional=True)
        if summary is not None:
            metadata_safe(summary)
            source_hashes["full-800-589/suite_summary.json"] = digest(safe_file(suite / "suite_summary.json"))
            original_scores = read_rows(suite / "suite_scores.jsonl")
            source_hashes["full-800-589/suite_scores.jsonl"] = digest(safe_file(suite / "suite_scores.jsonl"))
        for shard in plan["shards"]:
            shard_id, dataset = shard["shard_id"], shard["dataset_key"]
            path = suite / "shards" / shard_id
            if path.is_symlink():
                raise ValueError("symlinked shard rejected")
            locks.enter_context(runner._readonly_existing_lock(path))
            state = runner.read_run_state_readonly(path)
            protocol, prompts, references = state["protocol"], state["prompts"], state["references"]
            manifest = protocol["dataset_manifest"]
            if protocol["case_ids"] != shard["case_ids"] or len(prompts) != shard["case_count"] * 3:
                raise ValueError("shard grid differs from plan")
            if manifest.get("suite_plan_sha256") != EXPECTED_PLAN or manifest.get("suite_shard_id") != shard_id or any(manifest.get(k) != v for k, v in plan["source_identities"][dataset].items()):
                raise ValueError("shard source identity differs from plan")
            if manifest.get("development_partition") != plan["datasets"][dataset]["development_partition"] or protocol["settings"] != plan["datasets"][dataset]["settings"] or protocol["api_base"] != plan["api_base"]:
                raise ValueError("shard settings/exposure differs from plan")
            code_hashes.update(protocol["code_sha256"])
            for name in ("protocol.json", *runner.FROZEN_ROWS, *JOURNALS, "objective_summary.json", "objective_progress.json", "objective_scores.jsonl"):
                body = safe_file(path / name, optional=True)
                if body is not None:
                    source_hashes[str((path / name).relative_to(work))] = digest(body)
            attempts, outcomes, responses = (read_rows(path / name) for name in JOURNALS[:3])
            cost, attempt_rows, latencies = audit_shard_cost(shard_id, attempts, outcomes, responses)
            costs.append(cost)
            attempts_export.extend(attempt_rows)
            all_latencies.extend(latencies)
            dataset_costs[dataset].append(cost)
            dataset_latencies[dataset].extend(latencies)
            counts = Counter(o["status"] for o in state["outcomes"].values())
            all_counts.update(counts)
            shard_summaries.append({"shard_id": shard_id, "dataset_key": dataset, "planned_cases": shard["case_count"], "planned_arm_rows": len(prompts), "completion_counts": dict(counts), "complete": not (counts["pending"] or counts["inflight"]), "cost": cost, "protocol_sha256": protocol["protocol_sha256"]})
            refs = {r["case_id"]: r for r in references}
            exposure = plan["datasets"][dataset]["development_partition"]
            development = set(exposure["development_case_ids"])
            for prompt in prompts:
                case_id, arm = prompt["case_id"], prompt["arm"]
                key = (case_id, arm)
                ref, response, outcome = refs[case_id], state["completed"].get(key), state["outcomes"][key]
                for field in ("group_id", "dataset_key", "task"):
                    if prompt[field] != ref[field]:
                        raise ValueError("frozen reference/prompt identity mismatch")
                available, terminal = response is not None, outcome["status"] in {"success", "exhausted_failure"}
                supplemental = diagnostic.score_supplemental(response["hypothesis"], ref) if available else empty_supplemental()
                score = scorer.deterministic_score(response["hypothesis"], ref)[0] if available else (0.0 if terminal else None)
                hypothesis = response["hypothesis"] if available else None
                row = {"shard_id": shard_id, "case_id": case_id, "dataset_key": dataset, "task": ref["task"], "group_id": ref["group_id"], "arm": arm, "metric": ref["metric"], "score": score, "available": available, "completion_status": outcome["status"], "canonical_request_key": prompt["canonical_request_key"], "charged_call": response.get("charged_call", True) if available else False, "deduplicated_call": response.get("deduplicated_call", False) if available else None,
                       "strict_committed_score": supplemental["strict_committed_score"] if terminal else None, "exact_alias_or_option_score": supplemental["exact_alias_or_option_score"] if terminal else None, "official_positive_with_abstention_cue": supplemental["official_positive_with_abstention_cue"], "official_positive_not_strict": supplemental["official_positive_not_strict"],
                       "pipeline_cause": "provider_technical_failure" if terminal and not available else "pending_or_inflight" if not terminal else "none" if score else "answer_incorrect_stage_unresolved",
                       "pipeline_stage_attribution_status": "not_independently_annotated", "source_path_observations": source_observations(prompt.get("trace", {})), "format_diagnostics": diagnostics_safe(supplemental), "development_question_exposed": case_id in development,
                       "original_hypothesis": hypothesis, "response_text_sha256": digest(hypothesis.encode()) if available else None, "source_response_sha256": response.get("source_response_sha256") if available else None,
                       "provider_model": response.get("provider_model") if available else None, "api_latency_seconds": response.get("latency_seconds") if available else None, "reported_usage": usage_safe(response.get("usage")) if available else {}}
                rows_export.append(row)
        code_hashes.update(plan["code_sha256"])
        complete = not (all_counts["pending"] or all_counts["inflight"])
        expected = {(cid, arm) for metadata in plan["datasets"].values() for cid in metadata["case_ids"] for arm in runner.ARMS}
        actual = {(r["case_id"], r["arm"]) for r in rows_export}
        if len(rows_export) != 4167 or actual != expected or len(actual) != 4167:
            raise ValueError("derived full task grid differs")
        cost_audit = sum_costs(costs, all_latencies)
        cost_audit["per_dataset"] = {dataset: sum_costs(value, dataset_latencies[dataset]) for dataset, value in dataset_costs.items()}
        checks = {"frozen_execution_commit": True, "suite_plan_identity": True, "all_4167_planned_rows_retained": True, "all_shard_journals_validated_readonly": True, "attempt_identity_qualified_by_shard": True, "no_cross_shard_deduplication": True, "canonical_cost_not_sum_of_logical_arm_rows": True, "original_final_score_grid_and_values": False, "original_final_costs": False}
        if summary is not None:
            if not complete or summary.get("status") != "complete" or summary.get("final_effectiveness_result") is not True or summary.get("planned_cases") != 1389 or summary.get("planned_arm_rows") != 4167 or summary.get("suite_plan_sha256") != EXPECTED_PLAN:
                raise ValueError("original final summary does not match terminal suite")
            source_map = {(r["case_id"], r["arm"]): r for r in original_scores}
            if len(original_scores) != 4167 or set(source_map) != expected:
                raise ValueError("original final score grid differs")
            for row in rows_export:
                original = source_map[(row["case_id"], row["arm"])]
                for field in ("case_id", "dataset_key", "task", "group_id", "arm", "metric", "score", "available", "completion_status", "canonical_request_key", "charged_call", "deduplicated_call"):
                    if original.get(field) != row[field]:
                        raise ValueError("original per-question score differs from read-only recomputation")
                for field in ("strict_committed_score", "exact_alias_or_option_score", "official_positive_with_abstention_cue", "official_positive_not_strict"):
                    if original["supplemental"].get(field) != row[field]:
                        raise ValueError("original supplemental score differs")
            if summary.get("completion_counts") != dict(all_counts):
                raise ValueError("original completion counts differ")
            compare_cost(summary["cost"], cost_audit)
            checks["original_final_score_grid_and_values"] = True
            checks["original_final_costs"] = True
        if args.final and (not complete or summary is None or cost_audit["intents_without_outcome"] or cost_audit["saved_responses_without_success_outcome"]):
            raise ValueError("--final requires original complete 4167-row scoring and no unresolved attempts")
        cost_audit["independent_audit_checks"] = checks
        cost_audit["cost_boundary"] = "Recorded intents are not verified provider receipts. Missing failure usage is unknown, not free. Successful API latency excludes failures, preparation/retrieval and parallel wall time. No verified currency price."
        # Copy only known frozen repository code, never benchmark input files.
        code_bytes = {}
        for relative, expected_hash in sorted(code_hashes.items()):
            code_path = repo / relative
            if not code_path.resolve().is_relative_to(repo) or code_path.suffix not in {".py", ".json"}:
                raise ValueError("unexpected frozen source-code path")
            body = safe_file(code_path)
            if digest(body) != expected_hash:
                raise ValueError("frozen code differs while exporting")
            code_bytes[relative] = body
    # All full experimental objects stay in Colab. The fresh destination contains
    # only whitelisted derived fields, original answer text and frozen code.
    helper_bytes = Path(__file__).read_bytes()
    public = {"schema": SCHEMA, "exported_at_utc": datetime.now(timezone.utc).isoformat(), "execution_commit": EXECUTION_COMMIT, "suite_plan_sha256": EXPECTED_PLAN, "status": "complete" if complete else "partial", "complete": complete, "final_scores_available": summary is not None, "planned_cases": 1389, "planned_arm_rows": 4167, "source_artifacts_sha256": source_hashes, "reporting_helper_sha256": digest(helper_bytes), "code_sha256": code_hashes, "dataset_metadata": {key: {field: value[field] for field in ("case_count", "independent_groups", "development_partition", "settings", "capacity_census", "group_shard_plan") if field in value} for key, value in plan["datasets"].items()}, "counts": dict(all_counts), "provenance": "Read-only existing-lock snapshots; deterministic per-row recomputation inside genuine hosted Colab; original full dataset and gold never exported by this helper", "limitations": ["Exploratory adapted benchmark; not human CBT process or clinical effectiveness", "MemoryAgentBench one conservative family; PersonaMem20 connected groups", "PersonaMem raw/managed lossless-source control may share canonical outputs", "No independent evidence annotations; incorrect answers retain unresolved causal stage", "Literal strict/exact diagnostics are not independent semantic correctness", "Question-held-out histories remain shared/development-exposed", "Original model answer text is controlled output, not a public release by default"]}
    metadata_safe(public)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=out.name + ".tmp-", dir=out.parent))
    try:
        write_json(staging / "public_metadata.json", public)
        write_json(staging / "suite_plan.json", plan)
        write_json(staging / "launcher_manifest.json", launcher)
        write_json(staging / "cost_audit.json", cost_audit)
        write_json(staging / "per_shard_summary.json", shard_summaries)
        if summary is not None:
            write_json(staging / "suite_summary.json", summary)
        else:
            write_json(staging / "suite_progress.json", {"status": "partial" if not complete else "terminal_unscored", "final_effectiveness_result": False, "planned_cases": 1389, "planned_arm_rows": 4167, "completion_counts": dict(all_counts), "cost": cost_audit})
        write_rows(staging / "per_question_rows.jsonl", rows_export)
        write_rows(staging / "per_question_answers.jsonl", [{key: row[key] for key in ("shard_id", "case_id", "dataset_key", "task", "group_id", "arm", "canonical_request_key", "completion_status", "original_hypothesis", "response_text_sha256", "source_response_sha256", "provider_model")} for row in rows_export])
        # Slim original scores use explicit fields; full source traces are omitted.
        slim = [{k: v for k, v in row.items() if k not in {"original_hypothesis", "source_path_observations"}} for row in rows_export]
        write_rows(staging / "suite_scores.jsonl", slim)
        write_rows(staging / "per_attempt_costs.jsonl", attempts_export)
        csv_fields = ("shard_id", "case_id", "dataset_key", "task", "group_id", "arm", "metric", "score", "available", "completion_status", "charged_call", "deduplicated_call", "strict_committed_score", "exact_alias_or_option_score", "official_positive_with_abstention_cue", "official_positive_not_strict", "pipeline_cause", "pipeline_stage_attribution_status", "development_question_exposed", "canonical_request_key", "provider_model", "api_latency_seconds", "response_text_sha256")
        write_csv(staging / "per_question_metadata.csv", rows_export, csv_fields)
        write_csv(staging / "per_question_original_answers.csv", rows_export, (*csv_fields, "original_hypothesis"))
        (staging / "reporting_helper.py").write_bytes(helper_bytes)
        for relative, body in code_bytes.items():
            target = staging / "frozen_code" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
        checksums = {str(path.relative_to(staging)): digest(path.read_bytes()) for path in sorted(staging.rglob("*")) if path.is_file()}
        write_json(staging / "SHA256SUMS.json", checksums)
        staging.rename(out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"schema": SCHEMA, "output": str(out), "status": public["status"], "planned_arm_rows": len(rows_export), "recorded_provider_attempt_intents": cost_audit["recorded_provider_attempt_intents"], "final_scores_available": summary is not None, "reporting_helper_sha256": digest(helper_bytes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
