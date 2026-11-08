"""Prospective evaluation contracts for the controlled memory prototype.

This module freezes identities and scoring semantics. It does not collect data,
call a model, set product memory state, supply assessor labels to generation, or
change the existing frozen v1 protocols. All numerical adoption thresholds are
provided by the caller before generation; none are invented here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
from typing import Callable


MATERIALS_SCHEMA = "cbt-followup-grouping-v2"
PROTOCOL_SCHEMA = "cbt-followup-evaluation-contract-v2"
SPLITS = frozenset({"development", "validation", "blind_test"})
ARMS = frozenset({"short_term", "raw_history_rag", "managed_memory_rag"})
PRIMARY_ARMS = frozenset({"raw_history_rag", "managed_memory_rag"})
RESOURCE_STAGES = frozenset({
    "source_commit", "extraction", "updates", "indexing", "retrieval",
    "context", "generation",
})
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")


def canonical_hash(value: object) -> str:
    """Hash JSON values while rejecting NaN and other non-JSON numbers."""
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _keys(value: object, required: set[str], optional: set[str] | None = None) -> dict:
    if not isinstance(value, dict) or set(value) - required - (optional or set()):
        raise ValueError("Non-whitelisted object fields")
    if required - set(value):
        raise ValueError("Required contract fields are missing")
    return value


def _identifier(value: object) -> str:
    if not isinstance(value, str) or not ID_RE.fullmatch(value):
        raise ValueError("Identity must be a nonempty explicit string")
    return value


def _text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("An explicit nonempty description is required")
    return value


def _ids(value: object, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError("Identity list is missing")
    result = [_identifier(item) for item in value]
    if len(set(result)) != len(result):
        raise ValueError("Duplicate identities are not allowed")
    return sorted(result)


def _number(value: object, *, minimum: float = 0, strict: bool = False) -> float:
    if type(value) not in {int, float} or not math.isfinite(value):
        raise ValueError("A finite JSON number is required")
    if value < minimum or (strict and value == minimum):
        raise ValueError("Number is outside the declared contract")
    return float(value)


def _integer(value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError("An explicit bounded integer is required")
    return value


def _hash(value: object) -> str:
    if not isinstance(value, str) or not HASH_RE.fullmatch(value):
        raise ValueError("A complete SHA256 identity is required")
    return value


def policy_identity(policy: Callable) -> str:
    """Fingerprint the actual shared callable and its containing source file.

    The run's complete code manifest must also freeze imported dependencies.
    A coincidentally equal count of displayed questions is not a policy identity.
    """
    if not callable(policy):
        raise ValueError("Clarification policy must be an actual callable")
    try:
        source = inspect.getsource(policy)
        location = inspect.getsourcefile(policy)
        if location is None:
            raise ValueError("Callable has no inspectable source")
        module_bytes = Path(location).read_bytes()
    except (OSError, TypeError) as exc:
        raise ValueError("Policy source must be frozen before generation") from exc
    return canonical_hash({
        "callable": f"{policy.__module__}.{policy.__qualname__}",
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "module_sha256": hashlib.sha256(module_bytes).hexdigest(),
    })


def freeze_materials(cases: list[dict], *, exposure_ledger: list[dict] | None = None,
                     exposure_ledger_complete: bool = False) -> dict:
    """Connect shared sources, mothers, and derivative chains before splitting.

    Every parent case must be represented in this inventory. Historical exposure
    entries can connect previously separate-looking mothers/sources, and their
    exposure propagates to the entire connected group. The output contains
    identities only, never questions, gold, or source text.
    """
    if not isinstance(cases, list) or not cases:
        raise ValueError("A nonempty complete case inventory is required")
    if type(exposure_ledger_complete) is not bool:
        raise ValueError("Exposure-ledger completeness must be explicit")
    if exposure_ledger_complete and exposure_ledger is None:
        raise ValueError("A complete ledger must be supplied, even when empty")
    normalized: dict[str, dict] = {}
    required = {"case_id", "split", "source_group_ids", "mother_ids",
                "parent_case_ids", "exposed"}
    for item in cases:
        row = _keys(item, required, {"exposure_reasons"})
        cid = _identifier(row["case_id"])
        if (cid in normalized or not isinstance(row["split"], str)
                or row["split"] not in SPLITS or type(row["exposed"]) is not bool):
            raise ValueError("Duplicate case, invalid split, or nonboolean exposure")
        reasons = row.get("exposure_reasons", [])
        if not isinstance(reasons, list) or any(not isinstance(x, str) or not x.strip() for x in reasons):
            raise ValueError("Exposure reasons must be explicit descriptions")
        if reasons and not row["exposed"]:
            raise ValueError("Known exposure reasons cannot be marked unexposed")
        normalized[cid] = {
            "case_id": cid, "split": row["split"],
            "source_group_ids": _ids(row["source_group_ids"], nonempty=True),
            "mother_ids": _ids(row["mother_ids"], nonempty=True),
            "parent_case_ids": _ids(row["parent_case_ids"]),
            "exposed": row["exposed"], "exposure_reasons": sorted(set(reasons)),
        }
    if any(parent not in normalized for row in normalized.values() for parent in row["parent_case_ids"]):
        raise ValueError("Uninventoried parent prevents lineage/exposure verification")
    visited: set[str] = set()
    active: set[str] = set()
    def visit(cid: str) -> None:
        if cid in active:
            raise ValueError("Derivative lineage cannot contain a cycle")
        if cid not in visited:
            active.add(cid)
            for prior in normalized[cid]["parent_case_ids"]:
                visit(prior)
            active.remove(cid)
            visited.add(cid)
    for cid in normalized:
        visit(cid)

    parent = {cid: cid for cid in normalized}
    def find(cid: str) -> str:
        while parent[cid] != cid:
            parent[cid] = parent[parent[cid]]
            cid = parent[cid]
        return cid
    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    token_cases: dict[tuple[str, str], list[str]] = {}
    for cid, row in normalized.items():
        for field in ("source_group_ids", "mother_ids"):
            for token in row[field]:
                token_cases.setdefault((field, token), []).append(cid)
        for prior in row["parent_case_ids"]:
            union(cid, prior)
    for connected in token_cases.values():
        for other in connected[1:]:
            union(connected[0], other)

    if exposure_ledger is None:
        exposure_ledger = []
    if not isinstance(exposure_ledger, list):
        raise ValueError("Exposure ledger must be an explicit list")
    ledger = []
    for entry in exposure_ledger:
        row = _keys(entry, {"case_ids", "source_group_ids", "mother_ids", "reason"})
        clean = {key: _ids(row[key]) for key in ("case_ids", "source_group_ids", "mother_ids")}
        clean["reason"] = _text(row["reason"])
        if not any(clean[key] for key in ("case_ids", "source_group_ids", "mother_ids")):
            raise ValueError("Exposure record has no attributable identity")
        members = {cid for cid in clean["case_ids"] if cid in normalized}
        for field in ("source_group_ids", "mother_ids"):
            for token in clean[field]:
                members.update(token_cases.get((field, token), []))
        ordered = sorted(members)
        for other in ordered[1:]:
            union(ordered[0], other)
        ledger.append((clean, members))

    groups: dict[str, list[str]] = {}
    for cid in sorted(normalized):
        groups.setdefault(find(cid), []).append(cid)
    output = []
    for members in groups.values():
        splits = {normalized[cid]["split"] for cid in members}
        if len(splits) != 1:
            raise ValueError("A connected mother/source/derivative group crosses splits")
        reasons = set()
        exposed = False
        for cid in members:
            row = normalized[cid]
            if row["exposed"]:
                exposed = True
                reasons.update(row["exposure_reasons"] or [f"declared_exposure:{cid}"])
        for entry, attributed in ledger:
            if attributed.intersection(members):
                exposed = True
                reasons.add(entry["reason"])
        split = next(iter(splits))
        if split == "blind_test" and (exposed or not exposure_ledger_complete):
            raise ValueError("Blind test requires unexposed lineage and a complete exposure ledger")
        output.append({
            "group_id": "group-" + canonical_hash(members)[:20],
            "case_ids": members, "split": split, "exposed": exposed,
            "exposure_reasons": sorted(reasons),
            "source_group_ids": sorted({x for cid in members for x in normalized[cid]["source_group_ids"]}),
            "mother_ids": sorted({x for cid in members for x in normalized[cid]["mother_ids"]}),
        })
    result = {
        "schema": MATERIALS_SCHEMA, "cases": [normalized[cid] for cid in sorted(normalized)],
        "groups": sorted(output, key=lambda item: item["group_id"]),
        "exposure_ledger": sorted((entry for entry, _ in ledger), key=canonical_hash),
        "exposure_ledger_complete": exposure_ledger_complete,
        "independent_unit": "connected_source_group",
        "contains_generation_or_scoring_text": False,
    }
    result["materials_sha256"] = canonical_hash(result)
    return result


def _adoption_gates(gates: object, *, unit: str) -> dict:
    value = _keys(gates, {"meaningful_improvement", "cluster_interval", "critical_errors", "resources"})
    if unit == "participant":
        improvements = _keys(value["meaningful_improvement"], {
            "task_success_fraction", "restricted_elapsed_seconds",
        })
        for endpoint, maximum in [("task_success_fraction", 1), ("restricted_elapsed_seconds", 3600)]:
            improvement = _keys(improvements[endpoint], {"minimum_absolute_difference", "justification"})
            if _number(improvement["minimum_absolute_difference"], strict=True) > maximum:
                raise ValueError("Co-primary improvement exceeds the endpoint range")
            _text(improvement["justification"])
    else:
        improvement = _keys(value["meaningful_improvement"], {"minimum_absolute_difference", "justification"})
        difference = _number(improvement["minimum_absolute_difference"], strict=True)
        if difference > 1:
            raise ValueError("Success-fraction improvement cannot exceed one")
        _text(improvement["justification"])
    interval_fields = {"method", "unit", "level", "replicates", "seed"}
    interval_fields.add("favorable_bound_rules" if unit == "participant" else "favorable_bound_rule")
    interval = _keys(value["cluster_interval"], interval_fields)
    if interval["method"] != "cluster_bootstrap" or interval["unit"] != unit:
        raise ValueError("Uncertainty must use the independently assigned analysis unit")
    if not 0 < _number(interval["level"]) < 1:
        raise ValueError("Interval level must be between zero and one")
    _integer(interval["replicates"], minimum=1)
    _integer(interval["seed"])
    if unit == "participant":
        if interval["favorable_bound_rules"] != {
            "task_success_fraction": "lower_bound_above_zero",
            "restricted_elapsed_seconds": "upper_bound_below_zero",
        }:
            raise ValueError("Human co-primary directions are higher success and lower time")
    elif interval["favorable_bound_rule"] != "lower_bound_above_zero":
        raise ValueError("Freeze the favorable interval-bound rule explicitly")
    errors = value["critical_errors"]
    if not isinstance(errors, dict) or not errors:
        raise ValueError("Critical-error categories and limits must be prespecified")
    for category, maximum in errors.items():
        _identifier(category)
        _integer(maximum)
    resources = _keys(value["resources"], {
        "maximum_provider_attempts", "maximum_total_tokens", "maximum_wall_seconds",
        "include_failed_attempts", "include_preparation", "stage_coverage",
    })
    _integer(resources["maximum_provider_attempts"], minimum=1)
    _integer(resources["maximum_total_tokens"], minimum=1)
    _number(resources["maximum_wall_seconds"], strict=True)
    if resources["include_failed_attempts"] is not True or resources["include_preparation"] is not True:
        raise ValueError("Resource gate must include failures and preparation")
    stages = _ids(resources["stage_coverage"], nonempty=True)
    if set(stages) != RESOURCE_STAGES:
        raise ValueError("Resource accounting must cover the whole implemented chain")
    return deepcopy(value)


def _verify_materials_manifest(manifest: object) -> dict:
    value = _keys(manifest, {
        "schema", "cases", "groups", "exposure_ledger", "independent_unit",
        "contains_generation_or_scoring_text", "materials_sha256", "exposure_ledger_complete",
    })
    if value["schema"] != MATERIALS_SCHEMA:
        raise ValueError("A new v2 material-grouping manifest is required")
    # Recompute both grouping and identity rather than trusting a forged groups list.
    expected = freeze_materials(value["cases"], exposure_ledger=value["exposure_ledger"],
                                exposure_ledger_complete=value["exposure_ledger_complete"])
    if value != expected:
        raise ValueError("Materials identity or transmitted grouping differs")
    return expected


def freeze_protocol(config: dict, materials_manifest: dict, *,
                    clarification_policies: dict[str, Callable]) -> dict:
    """Freeze the primary comparison without a six-cell factorial expansion."""
    value = _keys(config, {
        "protocol_id", "study_type", "analysis_kind", "memory_arms",
        "knowledge_condition", "knowledge_manifest_sha256",
        "clarification_policy_hashes", "managed_only_ux", "diagnostic_ablations",
        "frozen_before_generation", "already_generated_answers", "adoption_gates",
    }, {"human_design"})
    manifest = _verify_materials_manifest(materials_manifest)
    _identifier(value["protocol_id"])
    if not isinstance(value["study_type"], str) or value["study_type"] not in {"development", "confirmatory"}:
        raise ValueError("Study type must be declared")
    if not isinstance(value["analysis_kind"], str) or value["analysis_kind"] not in {"offline_paired", "human_parallel"}:
        raise ValueError("Independent analysis design must be declared")
    arms = _ids(value["memory_arms"], nonempty=True)
    if not PRIMARY_ARMS <= set(arms) or not set(arms) <= ARMS:
        raise ValueError("The competent raw and managed primary arms are required")
    if not isinstance(value["knowledge_condition"], str) or value["knowledge_condition"] not in {"K0", "K1"}:
        raise ValueError("Freeze one knowledge condition for this memory comparison")
    if value["knowledge_condition"] == "K1":
        _hash(value["knowledge_manifest_sha256"])
    elif value["knowledge_manifest_sha256"] is not None:
        raise ValueError("K0 cannot claim an injected knowledge corpus")
    policies = _keys(value["clarification_policy_hashes"], set(arms))
    for identity in policies.values():
        _hash(identity)
    actual_policies = _keys(clarification_policies, set(arms))
    for arm in arms:
        if policy_identity(actual_policies[arm]) != policies[arm]:
            raise ValueError("Configured policy hash does not identify the actual callable")
    features = _ids(value["managed_only_ux"])
    same_policy = (policies["raw_history_rag"] == policies["managed_memory_rag"]
                   and actual_policies["raw_history_rag"] is actual_policies["managed_memory_rag"])
    estimand = "memory_management" if same_policy and not features else "package_effect"
    ablations = value["diagnostic_ablations"]
    if not isinstance(ablations, list) or len(ablations) > 2:
        raise ValueError("At most two prespecified diagnostic ablations are permitted")
    ablation_ids = []
    for ablation in ablations:
        row = _keys(ablation, {"id", "mechanism"})
        ablation_ids.append(_identifier(row["id"]))
        _text(row["mechanism"])
    if len(ablations) != len(set(ablation_ids)):
        raise ValueError("Diagnostic ablation identities must be unique")
    if value["frozen_before_generation"] is not True or _integer(value["already_generated_answers"]) != 0:
        raise ValueError("Cannot retrospectively freeze a protocol after seeing answers")

    is_human = value["analysis_kind"] == "human_parallel"
    unit = "participant" if is_human else "connected_source_group"
    if is_human:
        design = _keys(value.get("human_design"), {
            "allocation", "independent_unit", "same_arm_all_visits",
            "co_primary_endpoints", "co_primary_ci_level",
        })
        if (design["allocation"] != "parallel_randomized"
                or design["independent_unit"] != "participant"
                or design["same_arm_all_visits"] is not True
                or design["co_primary_endpoints"] != ["task_success_fraction", "restricted_elapsed_seconds"]
                or type(design["co_primary_ci_level"]) not in {float, int}
                or design["co_primary_ci_level"] != 0.975):
            raise ValueError("Human study must preserve participant-level parallel co-primary design")
    elif "human_design" in value:
        raise ValueError("Offline source tasks are not randomized human participants")
    if value["study_type"] == "confirmatory":
        if not is_human and any(group["split"] != "blind_test" for group in manifest["groups"]):
            raise ValueError("Confirmatory offline results require prospectively independent blind groups")
        gates = _adoption_gates(value["adoption_gates"], unit=unit)
        if is_human and gates["cluster_interval"]["level"] != 0.975:
            raise ValueError("Each human co-primary endpoint retains its 97.5% interval")
    else:
        gates = None if value["adoption_gates"] is None else _adoption_gates(value["adoption_gates"], unit=unit)

    result = deepcopy(value)
    result.update({
        "schema": PROTOCOL_SCHEMA, "memory_arms": arms,
        "primary_contrast": "managed_memory_rag_minus_raw_history_rag",
        "estimand": estimand,
        "shared_clarification_function": same_policy,
        "displayed_check_counts_must_match": False,
        "independent_unit": unit,
        "materials_sha256": manifest["materials_sha256"],
        "adoption_gates": gates,
        "clinical_efficacy_claim": False,
        "provider_calls_by_this_module": 0,
    })
    result["protocol_sha256"] = canonical_hash(result)
    return result


def score_observation(*, truth_status: str, execution_status: str,
                      review_status: str, is_correct: bool | None = None,
                      cause: str = "unresolved", cause_evidence_ids: list[str] | None = None,
                      independent_cause_review: bool = False) -> dict:
    """Keep unknown truth, a failed attempt, and a pending rating distinct.

    ``truth_status='unknown'`` is a legitimate independently labeled answer
    target; it is not the absence of a reference. A technical failure scores zero
    operationally without asserting semantic falsity or guessing its cause.
    """
    if not isinstance(truth_status, str) or truth_status not in {"known", "unknown", "not_annotated"}:
        raise ValueError("Truth annotation status must be explicit")
    if not isinstance(execution_status, str) or execution_status not in {"success", "technical_failure", "timeout", "missing"}:
        raise ValueError("Execution status must be explicit")
    if not isinstance(review_status, str) or review_status not in {"complete", "pending_review", "unresolved", "not_required"}:
        raise ValueError("Review status must be explicit")
    _identifier(cause)
    if type(independent_cause_review) is not bool:
        raise ValueError("Independent cause-review status must be boolean")
    evidence = _ids([] if cause_evidence_ids is None else cause_evidence_ids)
    if cause != "unresolved" and (not independent_cause_review or not evidence):
        raise ValueError("A resolved cause needs independent evidence and adjudication")
    if is_correct is not None and type(is_correct) is not bool:
        raise ValueError("Semantic correctness cannot be coerced from a number or string")
    if execution_status != "success":
        if is_correct is not None:
            raise ValueError("A failed attempt is not an independently observed semantic answer")
        score = 0
        score_status = "operational_failure"
    elif review_status in {"pending_review", "unresolved"}:
        if is_correct is not None:
            raise ValueError("Pending or unresolved review cannot have a semantic verdict")
        score = None
        score_status = review_status
    elif review_status == "complete":
        if truth_status == "not_annotated" or type(is_correct) is not bool:
            raise ValueError("Completed semantic review requires a reference and boolean verdict")
        score = int(is_correct)
        score_status = "independently_reviewed"
    else:
        raise ValueError("A successful answer still requires independent review")
    return {
        "truth_status": truth_status, "execution_status": execution_status,
        "review_status": review_status, "semantic_correctness": is_correct,
        "operational_score": score, "score_status": score_status,
        "cause": cause, "cause_evidence_ids": evidence,
        "independent_cause_review": independent_cause_review,
    }


def summarize_observations(rows: list[dict]) -> dict:
    """Retain every planned row and block inference while ratings are pending."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("All planned observations must be supplied")
    verified = []
    for row in rows:
        value = _keys(row, {
            "truth_status", "execution_status", "review_status", "semantic_correctness",
            "operational_score", "score_status", "cause", "cause_evidence_ids",
            "independent_cause_review",
        })
        expected = score_observation(
            truth_status=value["truth_status"], execution_status=value["execution_status"],
            review_status=value["review_status"], is_correct=value["semantic_correctness"],
            cause=value["cause"], cause_evidence_ids=value["cause_evidence_ids"],
            independent_cause_review=value["independent_cause_review"],
        )
        if value != expected:
            raise ValueError("Stored score conflicts with its observation statuses")
        verified.append(expected)
    pending = sum(row["operational_score"] is None for row in verified)
    return {
        "planned_rows": len(verified),
        "technical_or_missing_failures": sum(row["execution_status"] != "success" for row in verified),
        "unknown_truth_rows": sum(row["truth_status"] == "unknown" for row in verified),
        "pending_or_unresolved_ratings": pending,
        "scored_rows": len(verified) - pending,
        "operational_fraction": None if pending else sum(row["operational_score"] for row in verified) / len(verified),
        "comparative_intervals_permitted": pending == 0,
        "denominator_excludes_failed_rows": False,
    }


def assess_offline_adoption(protocol: dict, *, effect: float, interval: list[float],
                            critical_errors: dict[str, int], resources: dict,
                            pending_reviews: int = 0) -> dict:
    """Apply frozen offline gates; never label a tiny positive change adoption.

    Human co-primary analysis remains in the original participant-analysis
    implementation, rather than borrowing this source-group decision helper.
    """
    value = deepcopy(protocol)
    identity = value.pop("protocol_sha256", None)
    if not isinstance(identity, str) or canonical_hash(value) != identity:
        raise ValueError("Frozen protocol identity differs")
    if value.get("schema") != PROTOCOL_SCHEMA or value.get("analysis_kind") != "offline_paired":
        raise ValueError("Use participant co-primary analysis for human outcomes")
    gates = _adoption_gates(value.get("adoption_gates"), unit="connected_source_group")
    if value["study_type"] != "confirmatory":
        raise ValueError("Development findings cannot pass a confirmatory adoption gate")
    _integer(pending_reviews)
    if pending_reviews:
        return {"status": "unresolved", "reasons": ["pending_independent_reviews"], "clinical_efficacy_claim": False}
    effect_value = _number(effect, minimum=-1)
    if effect_value > 1 or not isinstance(interval, list) or len(interval) != 2:
        raise ValueError("Effect and its group interval must be supplied")
    low, high = (_number(bound, minimum=-1) for bound in interval)
    if high > 1 or not low <= effect_value <= high:
        raise ValueError("Effect and interval are inconsistent")
    error_values = _keys(critical_errors, set(gates["critical_errors"]))
    resource_values = _keys(resources, {"provider_attempts", "total_tokens", "wall_seconds"})
    unknown = ["critical_error_observations:" + key for key, number in error_values.items() if number is None]
    unknown += ["resource_usage:" + key for key, number in resource_values.items() if number is None]
    if unknown:
        return {"status": "unresolved", "reasons": unknown, "clinical_efficacy_claim": False}
    reasons = []
    if effect_value < gates["meaningful_improvement"]["minimum_absolute_difference"]:
        reasons.append("effect_below_frozen_meaningful_difference")
    if low <= 0:
        reasons.append("group_interval_not_favorably_above_zero")
    for category, maximum in gates["critical_errors"].items():
        if _integer(error_values[category]) > maximum:
            reasons.append("critical_error_limit:" + category)
    caps = gates["resources"]
    for observed, maximum in [("provider_attempts", "maximum_provider_attempts"),
                              ("total_tokens", "maximum_total_tokens"),
                              ("wall_seconds", "maximum_wall_seconds")]:
        number = (_integer(resource_values[observed]) if observed != "wall_seconds"
                  else _number(resource_values[observed]))
        if number > caps[maximum]:
            reasons.append("resource_limit:" + observed)
    return {
        "status": "criteria_not_met" if reasons else "prespecified_offline_criteria_met",
        "reasons": reasons, "estimand": value["estimand"],
        "clinical_efficacy_claim": False,
    }
