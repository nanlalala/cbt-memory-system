"""New two-arm PersonaMem scoring and controlled export; no provider calls.

The primary rule is the unchanged released Persona option parser. Full data is
processed only in hosted Colab; explicit tiny fictional fixtures exercise pure
aggregation. This module never changes or resumes an older three-arm run.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import statistics
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VERSION = "persona-full-management-two-arm-scoring-v1"
ARMS = ("raw_history_rag", "managed_memory_rag")
TERMINAL = {"success", "provider_failure", "protocol_validation_failure"}
OFFICIAL_SCORER_SHA256 = "262243267a1b2f73bf1838d0c88f50afc235402c9d934252982a38a0e3ffb719"
SUPPLEMENTAL_SHA256 = "3a61b6c2059e9647340b86797476b1f32267f099dd25e0d46f0cbe3225b046f0"
RUN_EXPORT_FILES = frozenset(("protocol.json", "cases.jsonl", "scope_plan.json",
    "extraction_requests.jsonl", "quote_control_gate.json", "prepared_rows.jsonl", "generation_rows.jsonl",
    "requests.jsonl", "attempts.jsonl", "outcomes.jsonl", "journal_cost.json",
    "phase_status.json", "prepared_manifest.json"))
SOURCE_EXPORT_NAMES = frozenset(("plan.json", "suite_plan.json", "metadata.json",
    "protocol.json", "cases.jsonl", "inventory.jsonl", "references.jsonl",
    "prompts.jsonl", "preparation_summary.json", "dataset_manifest.json", "public_metadata.json"))
SCORE_FILES = ("persona_management_summary.json", "persona_management_scores.jsonl")
SAFE_TRACE_SCALARS = ("pipeline_valid", "full_pipeline_valid", "extraction_available",
    "semantic_pipeline_status", "pipeline_status", "source_unit_count", "source_chunk_count",
    "semantic_candidate_count", "valid_candidate_count", "invalid_candidate_count",
    "stored_quote_count", "context_chars", "context_characters", "max_context_chars",
    "visible_context_end_exclusive", "source_quote_only_generation", "model_summary_promoted_to_fact",
    "preparation_seconds", "retrieval_seconds", "clarification_count", "user_confirmation_count",
    "store_state_sha256")
SAFE_TRACE_IDS = ("included_short_ids", "included_long_ids", "included_memory_ids", "retrieved_ids",
    "included_source_ids", "retrieved_source_ids", "retrieved_memory_ids", "context_memory_ids",
    "budget_dropped_short_ids", "budget_dropped_long_ids", "budget_dropped_memory_ids",
    "dropped_source_ids", "invalid_source_ids", "invalid_window_ids", "extraction_request_keys", "operation_ids")


def stable(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def require_colab():
    try:
        available = importlib.util.find_spec("google.colab") is not None
    except (ValueError, ModuleNotFoundError):
        available = False
    if not (Path("/content").is_dir() and os.environ.get("COLAB_RELEASE_TAG") and available):
        raise RuntimeError("Actual Persona processing requires genuine hosted Colab before controlled reads")


def _regular(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Regular controlled file required")
    return path


def _module(name):
    expected = HERE / (name + ".py")
    if name in sys.modules and Path(sys.modules[name].__file__).resolve() != expected.resolve():
        raise ValueError("Foreign cached scoring/runner module")
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    module = importlib.import_module(name)
    if Path(module.__file__).resolve() != expected.resolve():
        raise ValueError("Scoring/runner module origin differs")
    return module


def _official():
    if sha_bytes(_regular(HERE / "score_objective_comparison.py").read_bytes()) != OFFICIAL_SCORER_SHA256:
        raise ValueError("The frozen official-parser module changed")
    return _module("score_objective_comparison")


def _supplemental():
    if sha_bytes(_regular(HERE / "v3_response_diagnostics.py").read_bytes()) != SUPPLEMENTAL_SHA256:
        raise ValueError("The frozen secondary-format diagnostic module changed")
    return _module("v3_response_diagnostics")


def project_cost(cost):
    """Numeric allowlist, never requests, response text, environment or secrets."""
    integer_keys = ("provider_attempts", "successful_calls", "failed_calls", "unresolved_calls",
        "unresolved_inflight_calls", "usage_missing_or_partial_calls", "billable_usage_unknown_calls",
        "latency_missing_calls")
    token_keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    def one(row):
        result = {}
        for key in integer_keys:
            if key in row:
                value = row[key]
                if type(value) is not int or value < 0:
                    raise ValueError("Cost counts must remain nonnegative integers")
                result[key] = value
        for key in ("provider_attempts", "successful_calls", "failed_calls"):
            if key not in result:
                raise ValueError("Required actual-call count missing")
        for key in ("reported_latency_seconds_sum", "reported_latency_seconds_median"):
            if key in row:
                value = row[key]
                if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                    raise ValueError("Invalid observed provider latency")
                result[key] = value
        usage = row.get("reported_usage")
        if not isinstance(usage, dict):
            raise ValueError("Reported numeric usage required; unknown fields need explicit counts")
        result["reported_usage"] = {}
        for key in token_keys:
            value = usage.get(key)
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError("Invalid reported token usage")
            result["reported_usage"][key] = value
        if result["successful_calls"] + result["failed_calls"] > result["provider_attempts"]:
            raise ValueError("Actual provider outcomes exceed attempts")
        return result
    result = one(cost)
    purposes = cost.get("by_purpose")
    if not isinstance(purposes, dict) or set(purposes) - {"extraction", "response"}:
        raise ValueError("Only extraction and response calls belong to this protocol")
    result["by_purpose"] = {purpose: one(row) for purpose, row in purposes.items()}
    for key in ("provider_attempts", "successful_calls", "failed_calls"):
        if sum(row[key] for row in result["by_purpose"].values()) != result[key]:
            raise ValueError("Per-purpose actual cost differs from total")
    for key in token_keys:
        values = [row["reported_usage"][key] for row in result["by_purpose"].values()]
        if all(value is not None for value in values) and result["reported_usage"][key] is not None:
            if not math.isclose(sum(values), result["reported_usage"][key], rel_tol=0, abs_tol=1e-8):
                raise ValueError("Per-purpose token usage differs from total")
    result.update(monetary_cost=None, total_compute_matched=False,
        boundary="Actual canonical extraction and response attempts, including reported failed usage; missing billable usage remains unknown. Equal response budgets do not mean equal total computation. No independently verified price schedule.")
    return result


def safe_trace(trace):
    """Safe scalar/identity projection; not semantic evidence or causal blame."""
    if not isinstance(trace, dict):
        raise ValueError("Prepared source trace must be an object")
    out = {}
    def safe_value(value):
        if isinstance(value, str) and not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,256}", value):
            raise ValueError("Only enum/identity strings may enter public source diagnostics")
        if value is not None and type(value) not in (str, int, float, bool):
            raise ValueError("Only scalar source diagnostics may enter the public projection")
        return value
    for key in SAFE_TRACE_SCALARS:
        if key in trace and (trace[key] is None or type(trace[key]) in (str, int, float, bool)):
            out[key] = safe_value(trace[key])
    for key in SAFE_TRACE_IDS:
        if key in trace:
            values = trace[key]
            if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
                raise ValueError("Source trace identities must be lists of IDs")
            out[key] = [safe_value(value) for value in values]
            out[key + "_count"] = len(values)
    # Source span text is kept only in the controlled original prepared file.
    spans = trace.get("included_source_spans", trace.get("rendered_source_spans"))
    if spans is not None:
        if not isinstance(spans, list):
            raise ValueError("Source spans must be a list")
        fields = ("source_id", "source_document_id", "chunk_id", "memory_id", "start", "end",
            "start_character", "end_character", "role", "channel", "quoted_text_sha256",
            "source_text_sha256", "complete_source_rendered", "ambiguous_span_position", "complete", "offset_unit")
        fields += ("normalization", "original_start", "original_end", "original_source_text_sha256")
        out["included_source_spans"] = [{k: safe_value(span[k]) for k in fields if k in span} for span in spans]
    for key, fields in (("source_catalog", ("source_id", "source_text_sha256", "role", "source_order")),
            ("semantic_window_statuses", ("window_id", "status", "candidate_count", "accepted_count", "reason_code")),
            ("source_event_operations", ("operation_id", "window_id", "memory_id", "action", "confirmation_state"))):
        if key in trace:
            if not isinstance(trace[key], list):
                raise ValueError("Source-stage trace must remain a list")
            out[key] = [{field: safe_value(row[field]) for field in fields if field in row} for row in trace[key]]
    out["trace_sha256"] = stable(trace)
    out["answer_error_causal_stage_established"] = False
    return out


def verify_source_bindings(case, trace, *, fixture=False):
    """Check source identity/role and exact offsets; never infer sufficiency."""
    catalog = trace.get("source_catalog")
    spans = trace.get("included_source_spans")
    if not fixture and (not isinstance(catalog, list) or not isinstance(spans, list)):
        raise ValueError("Actual source catalog and included spans are required")
    history, _ = _module("run_objective_comparison")._eligible(case)
    cutoff = case.get("visible_context_end_exclusive")
    if cutoff is not None and any(source["order"] >= cutoff for source in history):
        raise ValueError("Future source exceeds released exclusive prefix")
    documents = {}
    for source in history:
        did = _official().stable_digest([case["subject_id"], source["session_id"], source["turn_id"]])
        text = source["text"].strip()
        documents[did] = {"text": text, "sha256": sha_bytes(text.encode()),
            "original_sha256": sha_bytes(source["text"].encode()),
            "leading_trim": len(source["text"]) - len(source["text"].lstrip()),
            "role": source["role"], "order": source["order"], "session_id": source["session_id"],
            "date": source.get("observed_at") or f"source_order:{source['order']}"}
    for entry in catalog or []:
        document = documents.get(entry.get("source_id"))
        if document is None or entry.get("source_text_sha256") != document["sha256"] or entry.get("role") != document["role"] or entry.get("source_order") != document["order"]:
            raise ValueError("Source catalog differs from the visible source history")
    if catalog is not None and len({row["source_id"] for row in catalog}) != len(catalog):
        raise ValueError("Duplicate source identity in trace catalog")
    if not fixture and {row["source_id"] for row in catalog} != set(documents):
        raise ValueError("Source catalog must describe the complete eligible prefix")
    for span in spans or []:
        sid = span.get("source_id", span.get("source_document_id"))
        document = documents.get(sid)
        start, end = span.get("start"), span.get("end")
        if (document is None or type(start) is not int or type(end) is not int or
            not 0 <= start < end <= len(document["text"]) or span.get("role") != document["role"] or
            span.get("source_text_sha256") != document["sha256"] or
            span.get("quoted_text_sha256") != sha_bytes(document["text"][start:end].encode())):
            raise ValueError("Included quote does not bind exact visible source characters and role")
        if "original_source_text_sha256" in span and (
            span["original_source_text_sha256"] != document["original_sha256"] or
            span.get("original_start") != start + document["leading_trim"] or
            span.get("original_end") != end + document["leading_trim"] or
            span.get("normalization") != "legacy_document_strip"):
            raise ValueError("Original-view quote provenance differs from released source bytes")
        if "chunk_id" in span:
            chunk_id = _official().stable_digest([case["subject_id"], sid, document["session_id"],
                document["date"], document["role"], start, end, document["text"][start:end]])
            if span["chunk_id"] != chunk_id:
                raise ValueError("Included span chunk identity differs from exact source provenance")
    return {"included_span_bindings_verified": len(spans or []),
            "source_catalog_bindings_verified": len(catalog or []), "semantic_sufficiency_established": False}


def _pipeline_valid(trace, arm):
    if arm != ARMS[1]:
        return None
    flags = [trace[key] for key in ("pipeline_valid", "full_pipeline_valid", "extraction_available")
             if key in trace and trace[key] is not None]
    if any(type(value) is not bool for value in flags):
        raise ValueError("Pipeline availability is a boolean observation, not a model grade")
    if len(set(flags)) > 1:
        # Extraction availability can differ from an end-to-end pipeline flag;
        # preserve the explicit full flag and the separate extraction trace.
        return trace.get("full_pipeline_valid", trace.get("pipeline_valid"))
    return flags[0] if flags else None


def _summarize(rows, *, seed, resamples):
    result = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        groups = defaultdict(list)
        for row in selected:
            groups[row["group_id"]].append(row["official_score"])
        result[arm] = {"planned_questions": len(selected), "connected_source_groups": len(groups),
            "correct": sum(row["official_score"] for row in selected),
            "accuracy_micro": statistics.mean(row["official_score"] for row in selected),
            "accuracy_group_equal": statistics.mean(statistics.mean(v) for v in groups.values()),
            "available_answers": sum(row["available"] for row in selected),
            "technical_failures": sum(not row["available"] for row in selected),
            "pipeline_valid_questions": sum(row["pipeline_valid"] is True for row in selected) if arm == ARMS[1] else None,
            "pipeline_invalid_questions": sum(row["pipeline_valid"] is False for row in selected) if arm == ARMS[1] else None,
            "pipeline_validity_unknown_questions": sum(row["pipeline_valid"] is None for row in selected) if arm == ARMS[1] else None,
            "secondary_exact_option_accuracy": statistics.mean(row["exact_option_score"] for row in selected),
            "secondary_strict_committed_accuracy": statistics.mean(row["strict_committed_score"] for row in selected),
            "official_positive_not_strict": sum(row["official_positive_not_strict"] for row in selected)}
    mapped = {(row["case_id"], row["arm"]): row for row in rows}
    groups = defaultdict(list)
    identical, shared, wins, ties, losses = 0, 0, 0, 0, 0
    for cid in sorted({row["case_id"] for row in rows}):
        raw, managed = (mapped[cid, arm] for arm in ARMS)
        delta = managed["official_score"] - raw["official_score"]
        groups[raw["group_id"]].append(delta)
        wins += delta > 0; ties += delta == 0; losses += delta < 0
        identical += raw["prompt_sha256"] == managed["prompt_sha256"]
        same_call = raw["canonical_request_key"] == managed["canonical_request_key"]
        shared += same_call
        if same_call and (raw["response_sha256"] != managed["response_sha256"] or raw["completion_status"] != managed["completion_status"]):
            raise ValueError("A canonical call cannot have different logical outcomes")
    deltas = [value for values in groups.values() for value in values]
    all_shared = shared == len(deltas)
    intervals = ({"group_equal_95_percent_interval": None, "micro_95_percent_interval": None,
        "interval_status": "blocked_by_identical_inputs_and_shared_responses"} if all_shared else
        _official().paired_cluster_intervals(groups, seed=seed, resamples=resamples))
    contrast = {"contrast": "managed_memory_rag_minus_raw_history_rag", "primary": True,
        "planned_pairs": len(deltas), "connected_source_groups": len(groups),
        "mean_accuracy_delta_group_equal": statistics.mean(statistics.mean(v) for v in groups.values()),
        "mean_accuracy_delta_micro": statistics.mean(deltas), "wins": wins, "ties": ties, "losses": losses,
        "identical_prompt_cases": identical, "shared_canonical_response_cases": shared,
        "management_input_contrast_present": identical != len(deltas),
        "identical_input_boundary": "Identical prompts/shared responses are construction controls, not empirical efficacy or statistical equivalence; all planned pairs remain primary.", **intervals}
    return {"conditions": result, "primary_paired_comparison": contrast}


def score_loaded_run(protocol, cases, references, generation_rows, journal_cost, *,
                     prepared_rows=None, bootstrap_samples=10000, fixture=False):
    """Pure tiny-fixture API; actual full-cohort calls enforce hosted Colab."""
    if not fixture:
        require_colab()
    if protocol.get("fixture_only") is not fixture or protocol.get("arms") != list(ARMS):
        raise ValueError("This is a new, explicitly two-arm protocol")
    if not fixture and (protocol.get("all_answer_arms_fresh") is not True or protocol.get("old_response_reuse") is not False):
        raise ValueError("Both answer arms must be freshly generated under this protocol")
    ids = [case["case_id"] for case in cases]
    if not ids or len(set(ids)) != len(ids) or set(ids) != set(protocol["case_ids"]):
        raise ValueError("Cases differ from the planned unique cohort")
    if protocol.get("planned_cases", len(ids)) != len(ids) or protocol.get("planned_arm_rows", 2 * len(ids)) != 2 * len(ids):
        raise ValueError("Frozen planned denominator differs from the two-arm cohort")
    if fixture and (len(cases) > 8 or sum(len(source.get("text", "")) for case in cases
                                         for source in case.get("history", [])) > 12000):
        raise ValueError("Fixture mode permits only tiny invented material")
    if not fixture and (len(cases) != 589 or len({c["group_id"] for c in cases}) != 20):
        raise ValueError("Expected the full 589-question, 20-connected-group Persona cohort")
    if any(case.get("dataset_key") != "personamem_v1" for case in cases):
        raise ValueError("Persona-only protocol required")
    expected = {(cid, arm) for cid in ids for arm in ARMS}
    mapped = {(r["case_id"], r["arm"]): r for r in generation_rows}
    if len(mapped) != len(generation_rows) or not set(mapped).issubset(expected):
        raise ValueError("Duplicate/unplanned logical response row")
    if any(row.get("completion_status") not in TERMINAL | {"pending", "inflight"} for row in generation_rows):
        raise ValueError("Unknown response state")
    cost = project_cost(journal_cost)
    common = {"scoring_version": VERSION, "protocol_sha256": protocol["protocol_sha256"],
        "fixture_only": fixture, "planned_cases": len(cases), "planned_arm_rows": len(expected),
        "connected_source_groups": len({c["group_id"] for c in cases}),
        "primary_contrast": "managed_memory_rag_minus_raw_history_rag",
        "primary_weighting": "equal connected persona/shared-context source groups",
        "completion_counts": dict(Counter(r["completion_status"] for r in generation_rows)), "cost": cost,
        "human_participants": 0, "independent_cbt_ratings": 0, "clinical_outcomes": 0,
        "clinical_or_user_benefit_claim": False, "confirmatory_claim": False,
        "answers_freshly_generated": True, "old_responses_reused": False,
        "requested_generation_model": protocol.get("generation_identity", {}).get("model"),
        "provider_identity_boundary": "Frozen requested/reported model identifiers are checked by the runner; underlying provider weights are not independently proven.",
        "official_metric_source": _official().METRIC_SOURCES["personamem_option_exact_match"],
        "official_scorer_module_sha256": OFFICIAL_SCORER_SHA256,
        "secondary_format_diagnostic_sha256": SUPPLEMENTAL_SHA256,
        "denominator": "all planned questions; terminal technical failures score zero; invalid extraction stays in the assigned arm",
        "interpretation": "Public synthetic evolving-persona option benchmark, not 589 independent people, CBT professional evaluation or clinical effectiveness. Twenty connected groups do not remove development exposure or benchmark dependence."}
    if set(mapped) != expected or any(row["completion_status"] not in TERMINAL for row in generation_rows):
        return {**common, "status": "partial", "final_benchmark_result": False,
            "remaining_arm_rows": len(expected) - sum(r["completion_status"] in TERMINAL for r in generation_rows),
            "comparative_scores_available": False}, []
    refs = {ref["case_id"]: ref for ref in references}
    if len(refs) != len(references) or set(refs) != set(ids):
        raise ValueError("Exactly one postgeneration scoring reference per planned case required")
    pmap = {row["case_id"]: row for row in prepared_rows} if prepared_rows is not None else None
    if pmap is not None and (len(pmap) != len(prepared_rows) or set(pmap) != set(ids)):
        raise ValueError("Prepared case grid differs")
    by_case = {case["case_id"]: case for case in cases}
    output = []
    official, diagnostic = _official(), _supplemental()
    for cid in ids:
        case, ref = by_case[cid], refs[cid]
        if ref.get("metric") != "personamem_option_exact_match":
            raise ValueError("Only the released Persona metric is permitted")
        official.deterministic_score("", ref)
        for key in ("group_id", "dataset_key", "task"):
            if ref.get(key) != case.get(key) or not ref.get(key):
                raise ValueError("Scoring reference metadata differs from gold-free case")
        for arm in ARMS:
            row = mapped[cid, arm]
            if row.get("group_id") != case["group_id"]:
                raise ValueError("Response group differs")
            if any(not isinstance(row.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", row[key])
                   for key in ("request_key", "prompt_sha256")):
                raise ValueError("Response and prompt require exact canonical SHA identities")
            raw = row.get("raw_response")
            available = row["completion_status"] == "success"
            if available:
                if not isinstance(raw, str) or row.get("response_sha256") != stable(raw):
                    raise ValueError("Answer differs from its canonical JSON string SHA")
                score, parsed = official.deterministic_score(raw, ref)
                extra = diagnostic.score_supplemental(raw, ref)
                if extra["official_score"] != score:
                    raise ValueError("Secondary diagnostics altered the primary official rule")
            else:
                if raw is not None or row.get("response_sha256") is not None:
                    raise ValueError("Technical failure cannot contain a fabricated answer")
                score, parsed = 0.0, None
                extra = {"exact_alias_or_option_score": 0.0, "strict_committed_score": 0.0,
                         "official_positive_not_strict": False}
            trace = row.get("trace", {})
            if pmap is not None:
                prep = pmap[cid]["prepared_by_arm"][arm]
                if trace != prep["trace"] or row["prompt_sha256"] != prep["prompt_sha256"]:
                    raise ValueError("Score trace/prompt differs from validated preparation")
            bindings = verify_source_bindings(case, trace, fixture=fixture)
            output.append({"case_id": cid, "group_id": case["group_id"], "task": case["task"],
                "dataset_key": "personamem_v1", "arm": arm, "metric": ref["metric"],
                "official_score": score, "available": available, "completion_status": row["completion_status"],
                "parsed_prediction": parsed, "response_sha256": row.get("response_sha256"),
                "prompt_sha256": row["prompt_sha256"], "canonical_request_key": row["request_key"],
                "pipeline_valid": _pipeline_valid(trace, arm), "trace": {**safe_trace(trace), **bindings},
                "exact_option_score": extra["exact_alias_or_option_score"],
                "strict_committed_score": extra["strict_committed_score"],
                "official_positive_not_strict": extra["official_positive_not_strict"],
                "failure_stage": "provider_or_protocol_technical_failure" if not available else
                    "none" if score else "answer_incorrect_stage_unresolved"})
    policy = protocol.get("evaluation_policy", {})
    if "bootstrap_samples" in policy and policy["bootstrap_samples"] != bootstrap_samples:
        raise ValueError("Bootstrap count differs from the frozen evaluation policy")
    seed = policy.get("bootstrap_seed", 42)
    summary = _summarize(output, seed=seed, resamples=bootstrap_samples)
    summary["by_task"] = {task: _summarize([r for r in output if r["task"] == task], seed=seed,
        resamples=bootstrap_samples) for task in sorted({r["task"] for r in output})}
    return {**common, **summary, "status": "complete", "final_benchmark_result": not fixture,
        "comparative_scores_available": True, "bootstrap_seed": seed, "bootstrap_samples": bootstrap_samples,
        "pipeline_and_attribution_boundary": "Candidate contract/storage/retrieval/context observations are not independent semantic validation or proof of why an answer was wrong. Valid-only subsets never replace the full planned denominator.",
        "secondary_boundary": "Exact/strict option metrics are pre-existing format diagnostics only; the unchanged released parser remains primary."}, output


def _relative(root, relative):
    if not isinstance(relative, str):
        raise ValueError("Safe relative artifact path required")
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts or "\\" in relative:
        raise ValueError("Safe relative artifact path required")
    path = Path(root) / relative
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Artifact escapes frozen root")
    return _regular(path)


def _load_references(protocol, original_suite):
    frozen = Path(protocol["original_suite"]).resolve()
    if original_suite is not None and Path(original_suite).resolve() != frozen:
        raise ValueError("Cannot substitute a different scoring source suite")
    rows = []
    for descriptor in protocol["reference_files"]:
        path = _relative(frozen, descriptor["relative_path"])
        data = path.read_bytes()
        if sha_bytes(data) != descriptor["sha256"]:
            raise ValueError("Original reference file identity changed")
        part = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
        if {r["case_id"] for r in part} != set(descriptor["case_ids"]):
            raise ValueError("Original reference shard case identities changed")
        rows.extend(part)
    return rows


def _atomic_json(path, value):
    _atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def _atomic_bytes(path, body):
    fd, name = tempfile.mkstemp(prefix="." + path.name + "-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(body); stream.flush(); os.fsync(stream.fileno())
        Path(name).replace(path)
    finally:
        Path(name).unlink(missing_ok=True)


def _load_and_score(run_dir, original_suite, bootstrap_samples):
    runner = _module("persona_management_runner")
    state = runner.readonly_state(Path(run_dir), include_references=False)
    references = _load_references(state["protocol"], original_suite) if state["terminal_complete"] else []
    summary, scores = score_loaded_run(state["protocol"], state["cases"], references,
        state["generation_rows"], state["journal_cost"], prepared_rows=state["prepared_rows"],
        bootstrap_samples=bootstrap_samples)
    summary["input_artifacts_sha256"] = state["input_sha256"]
    summary["scoring_code_sha256"] = sha_bytes(_regular(Path(__file__)).read_bytes())
    gate_path = _relative(run_dir, "quote_control_gate.json")
    gate_bytes = gate_path.read_bytes()
    if sha_bytes(gate_bytes) != state["input_sha256"].get("quote_control_gate.json"):
        raise ValueError("Raw-baseline control gate differs from frozen input")
    gate = json.loads(gate_bytes)
    summary["raw_baseline_source_prompt_parity"] = verify_quote_control_gate(gate,
        state["protocol"]["case_ids"], state["prepared_rows"])
    return state, summary, scores


def verify_quote_control_gate(gate, case_ids, prepared_rows):
    count = len(case_ids)
    rows = gate.get("case_rows", [])
    mapped = {row["case_id"]: row for row in rows}
    if (gate.get("schema") != "persona-full-manager-quote-control-gate-v1" or
        gate.get("new_provider_calls") != 0 or gate.get("case_count") != count or
        gate.get("new_raw_equals_frozen_old_raw_cases") != count or
        gate.get("empty_semantic_manager_equals_raw_cases") != count or
        len(mapped) != len(rows) or set(mapped) != set(case_ids) or
        any(row.get("new_raw_equals_frozen_old_raw") is not True or
            row.get("empty_semantic_manager_equals_raw") is not True for row in rows)):
        raise ValueError("Whole-cohort raw/quote-only prompt parity gate differs")
    for prepared in prepared_rows:
        if prepared["prepared_by_arm"][ARMS[0]]["prompt_sha256"] != mapped[prepared["case_id"]]["prompt_sha256"]:
            raise ValueError("Actual raw prompt differs from pre-extraction baseline gate")
    return {"case_count": count, "new_raw_equals_frozen_old_raw_cases": count,
        "empty_semantic_manager_equals_raw_cases": count, "gate_payload_sha256": stable(gate),
        "new_provider_calls_by_gate": 0,
        "boundary": "Exact source/context/prompt parity control only; synthetic empty candidates are not actual extraction quality or management efficacy."}


def score_run(run_dir, original_suite=None, *, write_outputs=True, bootstrap_samples=10000):
    require_colab()  # prior to opening a run, suite, source or scoring reference
    run = Path(run_dir)
    if run.is_symlink():
        raise ValueError("Symlinked run root rejected")
    runner = _module("persona_management_runner")
    with runner.lock(run):
        _, summary, scores = _load_and_score(run, original_suite, bootstrap_samples)
        if write_outputs:
            if summary["status"] == "complete":
                _atomic_bytes(run / SCORE_FILES[1], "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in scores).encode())
                _atomic_json(run / SCORE_FILES[0], summary)
                (run / "persona_management_progress.json").unlink(missing_ok=True)
            else:
                for name in SCORE_FILES:
                    (run / name).unlink(missing_ok=True)
                _atomic_json(run / "persona_management_progress.json", summary)
    return summary


def public_rows(scores):
    fields = ("case_id", "group_id", "task", "dataset_key", "arm", "metric", "official_score", "available",
        "completion_status", "response_sha256", "prompt_sha256", "canonical_request_key", "pipeline_valid",
        "exact_option_score", "strict_committed_score", "official_positive_not_strict", "failure_stage")
    return [{**{key: row[key] for key in fields}, "trace": row["trace"]} for row in scores]


def _zip_publish(path, members):
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError("Choose a fresh archive path")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".persona-audit-", suffix=".zip.part", dir=path.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in sorted(members.items()):
                archive.writestr(name, data)
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def replay_lock_members(run, protocol):
    """Copy empty lock files required by the read-only replay entry points.

    Copying bytes does not transfer a live flock or modify/unlink its inode.
    Restored files supply new local lock inodes in the copied directories.
    """
    suite = Path(protocol["original_suite"])
    paths = {"run/.execution.lock": _regular(Path(run) / ".execution.lock"),
             "source_suite/.execution.lock": _regular(suite / ".execution.lock")}
    restore = {"source_suite/.execution.lock": str((suite / ".execution.lock").resolve())}
    for descriptor in protocol["reference_files"]:
        relative = descriptor["relative_path"]
        rel = PurePosixPath(relative)
        if (rel.is_absolute() or len(rel.parts) != 3 or rel.parts[0] != "shards" or
            rel.name != "references.jsonl" or ".." in rel.parts or "\\" in relative):
            raise ValueError("Expected an explicitly frozen Persona shard reference path")
        lock_relative = str(rel.parent / ".execution.lock")
        member = "source_suite/" + lock_relative
        paths[member] = _relative(suite, lock_relative)
        restore[member] = str(paths[member].resolve())
    members = {}
    for name, path in paths.items():
        value = path.read_bytes()
        if value != b"":
            raise ValueError("Only actual empty execution-lock bytes may enter replay export")
        members[name] = value
    return members, restore


def _export_run_locked(run, out, original_suite, bootstrap_samples):
    state, summary, scores = _load_and_score(run, original_suite, bootstrap_samples)
    protocol = state["protocol"]
    # Only validated, explicit artifact names. Reproducibility uses full replay;
    # the manager's transient SQLite database is not claimed as a saved snapshot.
    controlled = {}
    for name, expected in state["input_sha256"].items():
        if name not in RUN_EXPORT_FILES:
            raise ValueError("Unrecognized run artifact cannot enter controlled export")
        data = _relative(run, name).read_bytes()
        if sha_bytes(data) != expected:
            raise ValueError("Run changed after read-only scoring")
        controlled["run/" + name] = data
    for relative, expected in protocol["code_sha256"].items():
        data = _relative(ROOT, relative).read_bytes()
        if sha_bytes(data) != expected:
            raise ValueError("Frozen executable dependency differs")
        controlled["code/" + relative] = data
    source_root = Path(protocol["original_suite"])
    source_hashes = dict(protocol.get("original_source_sha256", {}))
    restore_paths = {}
    if state["terminal_complete"]:
        source_hashes.update({d["relative_path"]: d["sha256"] for d in protocol["reference_files"]})
    for source_name, expected in source_hashes.items():
        source_path = Path(source_name)
        if source_path.is_absolute():
            if source_path.is_relative_to(source_root):
                relative = str(source_path.relative_to(source_root))
                base = source_root
                prefix = "source_suite/"
            elif source_path == source_root.parent / "derived-final-20261105/public_metadata.json":
                relative = "derived-final-20261105/public_metadata.json"
                base = source_root.parent
                prefix = "source_suite_parent/"
            else:
                raise ValueError("Original source path is outside the frozen suite boundary")
        else:
            relative, base, prefix = source_name, source_root, "source_suite/"
        if PurePosixPath(relative).name not in SOURCE_EXPORT_NAMES:
            raise ValueError("Unrecognized original-suite artifact cannot enter export")
        data = _relative(base, relative).read_bytes()
        if sha_bytes(data) != expected:
            raise ValueError("Frozen source dependency differs")
        controlled[prefix + relative] = data
        restore_paths[prefix + relative] = str((base / relative).resolve())
    locks, lock_restore_paths = replay_lock_members(run, protocol)
    controlled.update(locks)
    restore_paths.update(lock_restore_paths)
    encode = lambda value: (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    safe = {"summary.json": encode(summary), "per_question_safe.jsonl": "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in public_rows(scores)).encode()}
    if scores:
        controlled["derived/" + SCORE_FILES[1]] = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in scores).encode()
        # Controlled tabular view preserves the actual answer; no source/gold
        # text is copied into the public metadata or a public spreadsheet.
        import io
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        fields = ("case_id", "group_id", "task", "arm", "official_score", "completion_status", "pipeline_valid",
                  "response_sha256", "prompt_sha256", "failure_stage")
        writer.writerow((*fields, "question", "original_response"))
        cases_by_id = {case["case_id"]: case for case in state["cases"]}
        original_by_pair = {(r["case_id"], r["arm"]): r for r in state["generation_rows"]}
        def csv_safe(value):
            return "'" + value if isinstance(value, str) and value.lstrip(" \t\r\n\ufeff").startswith(("=", "+", "-", "@")) else value
        for row in scores:
            writer.writerow([*(csv_safe(row[field]) for field in fields),
                csv_safe(cases_by_id[row["case_id"]].get("query", "")),
                csv_safe(original_by_pair[row["case_id"], row["arm"]].get("raw_response") or "")])
        controlled["derived/per_question_original_answers.csv"] = ("\ufeff" + stream.getvalue()).encode()
    controlled["derived/" + SCORE_FILES[0]] = encode(summary)
    declaration = {"schema": "persona-full-management-audit-v1", "status": summary["status"],
        "protocol_sha256": protocol["protocol_sha256"], "code_revision": protocol.get("executable_commit"),
        "model_calls_by_exporter": 0, "sqlite_snapshot_saved": False,
        "state_reproduction": "Frozen source histories, extraction requests/responses, operation traces and executable code permit manager replay; transient SQLite is not exported.",
        "public_contains_source_query_answer_gold_or_extraction_text": False,
        "controlled_material": "Controlled benchmark text and model responses; do not automatically publish or upload as public results.",
        "clinical_or_user_benefit_claim": False}
    for members, private in ((controlled, True), (safe, False)):
        members["AUDIT_DECLARATION.json"] = encode({**declaration, "contains_benchmark_derived_text": private,
            **({"frozen_source_restore_paths": restore_paths,
                "restored_empty_execution_lock_members": sorted(locks),
                "execution_lock_restore_boundary": "Empty copied files provide fresh local lock inodes; no live process lock is copied and original locks are never modified or unlinked.",
                "recompute_instruction": "In genuine hosted Colab, restore source files to their frozen original paths, restore run/ to a fresh directory, use the pinned clean executable checkout and run persona_management_scoring.py score --run-dir <restored-run>. No provider call or Secret is needed."} if private else {})})
        members["SHA256SUMS.json"] = encode({name: sha_bytes(data) for name, data in sorted(members.items())})
    if state["input_sha256"] != {name: sha_bytes(_relative(run, name).read_bytes()) for name in state["input_sha256"]}:
        raise ValueError("Run changed during controlled export")
    out.mkdir(parents=True)
    _zip_publish(out / "controlled_reproducibility.zip", controlled)
    _zip_publish(out / "public_safe_results.zip", safe)
    return {"status": summary["status"], "out": str(out), "new_provider_calls": 0,
        "controlled_archive_sha256": sha_bytes((out / "controlled_reproducibility.zip").read_bytes()),
        "public_archive_sha256": sha_bytes((out / "public_safe_results.zip").read_bytes()),
        "planned_arm_rows": summary["planned_arm_rows"], "clinical_or_user_benefit_claim": False}


def export_run(run_dir, out_dir, original_suite=None, *, bootstrap_samples=10000):
    require_colab()
    run, out = Path(run_dir), Path(out_dir)
    if run.is_symlink() or out.is_symlink() or out.exists() or out.resolve().is_relative_to(run.resolve()):
        raise ValueError("Use a fresh output directory outside the original run")
    runner = _module("persona_management_runner")
    with runner.lock(run):
        return _export_run_locked(run, out, original_suite, bootstrap_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("score", "export"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--original-suite", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "export" and args.out is None:
        parser.error("export requires --out")
    result = (score_run(args.run_dir, args.original_suite) if args.command == "score" else
        export_run(args.run_dir, args.out, args.original_suite))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
