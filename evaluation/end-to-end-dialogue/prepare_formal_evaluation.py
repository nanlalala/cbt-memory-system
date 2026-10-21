#!/usr/bin/env python3
"""Prepare separate benchmark cohorts in Colab; never generate or judge answers.

Small in-memory fixtures exercise the controls locally. Full dataset hashing,
loading and cohort writing are guarded to Colab by the production entry point.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation/memory-retrieval"))
from benchmark_adapters import RetrievalExample, load_locomo, load_longmemeval  # noqa: E402


SCHEMA_VERSION = "formal-cohort-manifest-v1"
PLAN_VERSION = "structured-memory-evaluation-plan-v1"
ARMS = ("none", "short_term", "short_plus_structured_candidate")
DEFAULT_PLAN = Path(__file__).with_name("formal_evaluation_plan.json")
HISTORY_FIELDS = frozenset({"dia_id", "role", "speaker", "text", "content"})
FINAL_CLAIM_REQUIREMENTS = [
    "all frozen cases evaluated", "predeclared independent-group minimum", "frozen reference-based evaluator and judge",
    "paired cluster confidence interval meets precision target", "directional improvement requires confidence-interval lower bound above zero",
    "at least meaningful improvement requires confidence-interval lower bound reaching minimum_effect",
    "technical failures retained and below threshold", "no clinical or authenticated-consent claims",
]


def stable_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def file_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def is_colab_runtime() -> bool:
    if not Path("/content").is_dir() or not os.environ.get("COLAB_RELEASE_TAG"):
        return False
    try:
        return bool(importlib.util.find_spec("google.colab"))
    except (ModuleNotFoundError, ValueError):
        return False


def require_colab_runtime() -> None:
    if not is_colab_runtime():
        raise RuntimeError("Full dataset preparation is Colab-only; local use is limited to --inspect-plan and tiny unit fixtures")


def validate_plan(plan: dict) -> None:
    if not isinstance(plan, dict):
        raise ValueError("evaluation plan must be a JSON object")
    if plan.get("schema_version") != PLAN_VERSION or plan.get("three_arms") != list(ARMS):
        raise ValueError("unsupported plan schema or memory conditions")
    memory = plan.get("memory_boundary", {})
    if memory.get("auto_confirm") is not False or memory.get("clinical_claims") is not False:
        raise ValueError("benchmarks cannot supply authentic user confirmation or clinical claims")
    configuration = plan.get("shared_response_configuration", {})
    model, accepted = configuration.get("model"), configuration.get("accepted_provider_models")
    if (not isinstance(model, str) or not model.strip() or not isinstance(accepted, list)
            or model not in accepted or any(not isinstance(value, str) or not value.strip() for value in accepted)):
        raise ValueError("response model and preapproved provider aliases must be frozen")
    if configuration.get("knowledge_condition") != "empty_in_every_arm_until_source_and_external_transmission_governance_approved":
        raise ValueError("knowledge must remain identically empty until a separate governed design is frozen")
    if configuration.get("max_response_tokens", 0) <= 0 or configuration.get("thinking_mode") not in {"disabled", "enabled", "provider_default"}:
        raise ValueError("response budget and thinking mode must be declared")
    statistics = plan.get("statistics", {})
    effect, minimum_groups = statistics.get("minimum_effect"), statistics.get("minimum_independent_groups")
    precision = statistics.get("precision_plan", {})
    if (not isinstance(effect, (int, float)) or isinstance(effect, bool) or not math.isfinite(effect)
            or not 0 < effect < 1 or type(minimum_groups) is not int or minimum_groups < 2):
        raise ValueError("predeclare a nonzero meaningful effect and minimum independent-group count")
    if (precision.get("method") != "paired_history_cluster_bootstrap"
            or not 0 < precision.get("confidence_level", 0) < 1
            or not 0 < precision.get("target_interval_half_width", 0) < 1
            or precision.get("resamples", 0) < 1000):
        raise ValueError("predeclare a paired cluster precision/power plan, not question-only independence")
    if statistics.get("pool_datasets") is not False or statistics.get("primary_contrast") != [ARMS[2], ARMS[1]]:
        raise ValueError("datasets must remain separate with the primary paired contrast frozen")
    failure = plan.get("completion_and_failure_policy", {})
    if (failure.get("all_frozen_examples_required") is not True
            or failure.get("optional_stopping") is not False
            or failure.get("post_result_case_exclusion") is not False
            or failure.get("failed_examples_remain_in_denominator") is not True
            or failure.get("unresolved_technical_failure_score") != 0
            or failure.get("retry_may_change_settings") is not False):
        raise ValueError("complete-cohort, fixed-setting failure handling must be prespecified")
    scoring = plan.get("scoring", {})
    if (scoring.get("reference_based_only_for_primary_endpoint") is not True
            or scoring.get("gold_visible_to_response_model") is not False
            or scoring.get("lexical_overlap_is_primary") is not False):
        raise ValueError("reference-based scoring must remain separate from answer generation")
    for key, spec in plan.get("datasets", {}).items():
        if not re.fullmatch(r"[0-9a-f]{64}", str(spec.get("sha256", ""))):
            raise ValueError(f"dataset {key} needs a pinned SHA-256")
        if not re.fullmatch(r"[0-9a-f]{40}", str(spec.get("revision", ""))):
            raise ValueError(f"dataset {key} needs an immutable upstream revision")
        if type(spec.get("bytes")) is not int or spec["bytes"] <= 0:
            raise ValueError(f"dataset {key} needs an exact byte count")
        if spec["revision"] not in spec.get("source", ""):
            raise ValueError(f"dataset {key} URL must include its pinned revision")


def validate_full_s_spec(spec: dict) -> None:
    if (spec.get("variant") != "S_full_cleaned_histories_not_oracle"
            or not spec.get("source", "").endswith("/longmemeval_s_cleaned.json")
            or spec.get("bytes", 0) < 100_000_000
            or spec.get("expected_released_examples") != 500
            or spec.get("expected_eligible_examples") != 500):
        raise ValueError("Formal LongMemEval retrieval requires the full pinned S500 history file, never the oracle subset")


def load_plan(path: Path) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    validate_plan(plan)
    return plan


def history_groups(examples: Sequence[RetrievalExample], dataset_key: str) -> dict[str, str]:
    """Conservative transitive grouping of shared sessions; never pool retrieval scopes."""
    if len({example.example_id for example in examples}) != len(examples):
        raise ValueError("duplicate benchmark question identity")
    if dataset_key == "locomo10":
        return {example.example_id: "locomo:" + example.example_id.split(":qa:")[0] for example in examples}
    parent = list(range(len(examples)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        parent[find(right)] = find(left)

    owner: dict[str, int] = {}
    for index, example in enumerate(examples):
        for session in example.sessions:
            tokens = ["id:" + str(session["session_id"]),
                      "history:" + stable_digest({"date": session.get("date"), "turns": sanitized_turns(session["turns"])})]
            for token in tokens:
                if token in owner:
                    union(index, owner[token])
                else:
                    owner[token] = index
    components: dict[int, list[str]] = {}
    for index, example in enumerate(examples):
        components.setdefault(find(index), []).append(example.example_id)
    labels = {root: "longmemeval:history-component:" + stable_digest(sorted(ids))[:20]
              for root, ids in components.items()}
    return {example.example_id: labels[find(index)] for index, example in enumerate(examples)}


def sanitized_turns(turns: Sequence[dict]) -> list[dict]:
    result = []
    for turn in turns:
        # Never forward answer/evidence/category/answerability fields hidden in a turn.
        clean = {key: value for key, value in turn.items() if key in HISTORY_FIELDS}
        if not isinstance(clean.get("content", clean.get("text")), str):
            raise ValueError("history turn is missing textual content")
        if any(not isinstance(value, str) for value in clean.values()):
            raise ValueError("allowed history fields must contain strings only")
        result.append(clean)
    return result


def _exposure_state(plan: dict, dataset_key: str, examples: Sequence[RetrievalExample], groups: dict[str, str],
                    exposure_ledger: dict | None) -> tuple[dict, list[str]]:
    base = plan["datasets"][dataset_key].get("development_exposure", {})
    added = (exposure_ledger or {}).get("datasets", {}).get(dataset_key, {})
    exposed_groups = set(base.get("group_ids", [])) | set(added.get("group_ids", []))
    example_ids = set(base.get("example_ids", [])) | set(added.get("example_ids", []))
    session_ids = set(base.get("session_ids", [])) | set(added.get("session_ids", []))
    for example in examples:
        if example.example_id in example_ids or any(str(session["session_id"]) in session_ids for session in example.sessions):
            exposed_groups.add(groups[example.example_id])
    observed_groups = set(groups.values())
    state = {
        "known_development_group_ids": sorted(exposed_groups),
        "overlapping_development_group_ids": sorted(exposed_groups & observed_groups),
        "review_complete": bool(added.get("exposure_review_complete", base.get("exposure_review_complete"))),
        "ledger_sha256": stable_digest(exposure_ledger) if exposure_ledger is not None else None,
        "definition": "Conversation/shared-history exposure, not merely repeated QA IDs; known exposure cannot be removed by an additional ledger",
    }
    return state, state["overlapping_development_group_ids"]


def _verify_pilot_exposure_source(plan: dict) -> None:
    exposure = plan["datasets"]["locomo10"]["development_exposure"]
    path = ROOT / exposure["source"]
    if file_identity(path)["sha256"] != exposure["source_sha256"]:
        raise ValueError("historical pilot exposure source has changed")
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    question_ids = {row["example_id"] for row in rows}
    group_ids = {"locomo:" + value.split(":qa:")[0] for value in question_ids}
    if len(question_ids) != exposure["question_count"] or group_ids != set(exposure["group_ids"]):
        raise ValueError("plan must preserve every historically exposed pilot conversation")


def _write_jsonl(path: Path, rows: Sequence[dict]) -> dict:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {**file_identity(path), "records": len(rows)}


def _claim_blockers(plan: dict, dataset_key: str, *, fixture_only: bool, role: str,
                    group_count: int, exposure: dict) -> list[str]:
    blockers = []
    if fixture_only:
        blockers.append("tiny_fixture_only_not_an_effectiveness_experiment")
    if role != "locked_external_test":
        blockers.append("exploratory_dataset_with_prior_development_exposure")
    if group_count < plan["statistics"]["minimum_independent_groups"]:
        blockers.append("too_few_independent_history_groups_for_predeclared_minimum")
    if not exposure["review_complete"]:
        blockers.append("development_exposure_review_not_complete")
    evaluator = plan["scoring"].get(dataset_key, {})
    if not re.fullmatch(r"[0-9a-f]{40}", str(evaluator.get("upstream_revision", ""))):
        blockers.append("official_reference_evaluator_revision_not_pinned")
    return blockers


def prepare_examples(
    examples: Sequence[RetrievalExample], dataset_key: str, plan: dict, source_identity: dict,
    output_dir: Path, *, role: str | None = None, exposure_ledger: dict | None = None,
    fixture_only: bool = True,
) -> dict:
    """Small deterministic fixture seam. Production callers must use prepare_dataset()."""
    validate_plan(plan)
    if dataset_key not in {"locomo10", "longmemeval_s_cleaned"}:
        raise ValueError("each supported dataset must be prepared and reported separately")
    if not fixture_only:
        require_colab_runtime()
    elif len(examples) > 10 or len(json.dumps([asdict(value) for value in examples])) > 1_000_000:
        raise ValueError("local fixture preparation is limited to ten tiny synthetic examples")
    spec = plan["datasets"][dataset_key]
    if dataset_key == "longmemeval_s_cleaned":
        validate_full_s_spec(spec)
    eligible = [example for example in examples if example.category not in spec.get("exclude_categories", [])]
    if not eligible:
        raise ValueError("cohort has no eligible examples")
    if not fixture_only:
        if source_identity.get("sha256") != spec["sha256"] or source_identity.get("bytes") != spec["bytes"]:
            raise ValueError("source identity does not match the frozen full dataset")
        if len(examples) != spec["expected_released_examples"] or len(eligible) != spec["expected_eligible_examples"]:
            raise ValueError("all predeclared benchmark examples must be retained; no cherry-picked subset")
        if "expected_abstention_examples" in spec and sum(value.should_abstain for value in eligible) != spec["expected_abstention_examples"]:
            raise ValueError("released abstention examples must be preserved")
    groups = history_groups(eligible, dataset_key)
    group_count = len(set(groups.values()))
    if not fixture_only and spec.get("expected_groups") is not None and group_count != spec["expected_groups"]:
        raise ValueError("unexpected number of conversation groups")
    exposure, overlap = _exposure_state(plan, dataset_key, eligible, groups, exposure_ledger)
    selected_role = role or spec["role"]
    if selected_role not in {"exploratory", "locked_external_test"}:
        raise ValueError("unknown cohort role")
    if selected_role == "locked_external_test" and overlap:
        raise ValueError("independent test refuses development-overlapping conversation/shared-history groups")
    if selected_role == "locked_external_test" and spec["role"] != "locked_external_test":
        raise ValueError("this dataset was preregistered as exploratory, not an independent external test")
    blockers = _claim_blockers(plan, dataset_key, fixture_only=fixture_only, role=selected_role,
                               group_count=group_count, exposure=exposure)
    generation, references = [], []
    for example in eligible:
        scope = ("locomo:" + example.example_id.split(":qa:")[0] if dataset_key == "locomo10"
                 else "longmemeval:" + example.example_id)
        generation.append({
            "example_id": example.example_id, "scope_id": scope, "group_id": groups[example.example_id],
            "query": example.query, "query_date": example.query_date,
            "sessions": [{"session_id": str(session["session_id"]), "date": session.get("date"),
                          "turns": sanitized_turns(session["turns"])} for session in example.sessions],
        })
        references.append({"example_id": example.example_id, "category": example.category,
                           "query": example.query, "reference_answer": example.answer,
                           "should_abstain": example.should_abstain})
    # Do not overwrite earlier locks, outcomes or half-prepared cohorts.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output cohort directory is not empty; preserve the old lock and use a new directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "generation_cases.jsonl": _write_jsonl(output_dir / "generation_cases.jsonl", generation),
        "scoring_references.jsonl": _write_jsonl(output_dir / "scoring_references.jsonl", references),
    }
    lock = {"schema_version": PLAN_VERSION, "plan_sha256": stable_digest(plan), "plan": plan,
            "dataset_key": dataset_key, "role": selected_role, "fixture_only": fixture_only,
            "source_identity": source_identity,
            "exposure_ledger": exposure_ledger, "exposure_ledger_sha256": exposure["ledger_sha256"]}
    (output_dir / "preregistration_lock.json").write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    files["preregistration_lock.json"] = file_identity(output_dir / "preregistration_lock.json")
    manifest = {
        "schema_version": SCHEMA_VERSION, "dataset_key": dataset_key, "role": selected_role,
        "fixture_only": fixture_only,
        "source": {**source_identity, "revision": spec["revision"], "source": spec["source"], "variant": spec["variant"]},
        "plan_sha256": stable_digest(plan), "three_arms": list(ARMS),
        "counts": {"released_examples": len(examples), "examples": len(eligible), "independent_groups": group_count,
                   "excluded_prespecified_examples": len(examples) - len(eligible),
                   "abstention_examples": sum(example.should_abstain for example in eligible),
                   "category_counts": dict(Counter(example.category for example in eligible))},
        "grouping_policy": spec["grouping"], "development_exposure": exposure,
        "development_exposed_group_ids": exposure["known_development_group_ids"],
        "preparation_eligible_for_confirmatory_test": not blockers,
        "confirmatory_claim_allowed": False,
        "claim_status": "preparation_only_effect_and_precision_not_yet_measured",
        "confirmatory_preparation_blockers": blockers,
        "final_claim_requirements": FINAL_CLAIM_REQUIREMENTS,
        "memory_boundary": plan["memory_boundary"], "statistics": plan["statistics"],
        "response_configuration": plan["shared_response_configuration"],
        "files": files,
    }
    manifest["cohort_sha256"] = stable_digest(manifest)
    (output_dir / "cohort_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def validate_cohort(cohort_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    """Validate artifact consistency against its frozen lock, not upstream authenticity.

    Returns the manifest, gold-free cases and separate scoring references. A
    modified master lock is not cryptographically authenticated; source release
    verification still belongs to preparation/audit against the upstream dataset.
    """
    cohort_dir = Path(cohort_dir)
    manifest_path = cohort_dir / "cohort_manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("cohort manifest must not be a symlink")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported cohort schema")
    unsigned = {key: value for key, value in manifest.items() if key != "cohort_sha256"}
    if manifest.get("cohort_sha256") != stable_digest(unsigned):
        raise ValueError("cohort manifest checksum mismatch")
    if type(manifest.get("fixture_only")) is not bool:
        raise ValueError("fixture-only mode must be an explicit boolean")
    if not manifest["fixture_only"]:
        require_colab_runtime()
    files = manifest.get("files", {})
    required_files = {"generation_cases.jsonl", "scoring_references.jsonl", "preregistration_lock.json"}
    if set(files) != required_files:
        raise ValueError("cohort must include exactly the declared cases, references and preregistration lock")
    for name in required_files:
        path = cohort_dir / name
        if path.is_symlink():
            raise ValueError("cohort member must not be a symlink")
        if manifest["fixture_only"] and path.stat().st_size > 1_000_000:
            raise ValueError("a fixture cohort cannot contain a large dataset")
        if file_identity(path) != {key: files[name].get(key) for key in ("sha256", "bytes")}:
            raise ValueError(f"cohort file checksum mismatch: {name}")
    lock = json.loads((cohort_dir / "preregistration_lock.json").read_text(encoding="utf-8"))
    plan = lock.get("plan")
    validate_plan(plan)
    if (lock.get("schema_version") != PLAN_VERSION or lock.get("plan_sha256") != stable_digest(plan)
            or manifest.get("plan_sha256") != stable_digest(plan)):
        raise ValueError("preregistration plan checksum mismatch")
    dataset_key, role = manifest.get("dataset_key"), manifest.get("role")
    if dataset_key not in {"locomo10", "longmemeval_s_cleaned"} or role not in {"exploratory", "locked_external_test"}:
        raise ValueError("unknown dataset or role in cohort")
    if any(lock.get(field) != manifest.get(field) for field in ("dataset_key", "role", "fixture_only")):
        raise ValueError("cohort role/dataset/mode does not match the frozen preregistration lock")
    ledger = lock.get("exposure_ledger")
    if lock.get("exposure_ledger_sha256") != (stable_digest(ledger) if ledger is not None else None):
        raise ValueError("development exposure ledger checksum mismatch")
    spec = plan["datasets"][dataset_key]
    if dataset_key == "longmemeval_s_cleaned":
        validate_full_s_spec(spec)
    expected_source = {**lock["source_identity"], "revision": spec["revision"], "source": spec["source"], "variant": spec["variant"]}
    if manifest.get("source") != expected_source:
        raise ValueError("cohort source does not match preregistration lock")
    if not manifest["fixture_only"] and any(lock["source_identity"].get(key) != spec[key] for key in ("sha256", "bytes")):
        raise ValueError("production cohort source is not the full pinned dataset")

    def rows(name):
        with (cohort_dir / name).open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    cases, references = rows("generation_cases.jsonl"), rows("scoring_references.jsonl")
    if (len(cases) != files["generation_cases.jsonl"].get("records")
            or len(references) != files["scoring_references.jsonl"].get("records")
            or not cases or len(cases) != len(references)):
        raise ValueError("cohort record counts do not match the locked files")
    if manifest["fixture_only"] and len(cases) > 10:
        raise ValueError("fixture cohort exceeds ten tiny examples")
    if len({row.get("example_id") for row in references}) != len(references):
        raise ValueError("duplicate scoring reference identity")
    reference_by_id = {row["example_id"]: row for row in references}
    examples = []
    for case in cases:
        if set(case) != {"example_id", "scope_id", "group_id", "query", "query_date", "sessions"}:
            raise ValueError("generation case contains unexpected or gold-label fields")
        reference = reference_by_id.get(case["example_id"])
        if reference is None or set(reference) != {"example_id", "category", "query", "reference_answer", "should_abstain"}:
            raise ValueError("generation/scoring IDs or reference schema do not match")
        if (case["query"] != reference["query"] or not isinstance(case["query"], str)
                or not isinstance(reference["reference_answer"], str) or not isinstance(reference["category"], str)
                or type(reference["should_abstain"]) is not bool):
            raise ValueError("cohort query or reference annotation has invalid types")
        if reference["category"] in spec.get("exclude_categories", []):
            raise ValueError("a prespecified excluded category entered the scored cohort")
        session_ids = []
        for session in case["sessions"]:
            if set(session) != {"session_id", "date", "turns"} or not isinstance(session["session_id"], str):
                raise ValueError("generation session contains unexpected annotation fields")
            session_ids.append(session["session_id"])
            if any(set(turn) - HISTORY_FIELDS for turn in session["turns"]):
                raise ValueError("gold labels or unexpected annotations found in generation history")
            sanitized_turns(session["turns"])
        if len(set(session_ids)) != len(session_ids):
            raise ValueError("duplicate session identity within a question")
        examples.append(RetrievalExample(
            example_id=case["example_id"], category=reference["category"], query=case["query"],
            answer=reference["reference_answer"], evidence_ids=(), sessions=tuple(case["sessions"]),
            should_abstain=reference["should_abstain"], query_date=case["query_date"],
        ))
    groups = history_groups(examples, dataset_key)
    for case in cases:
        expected_scope = ("locomo:" + case["example_id"].split(":qa:")[0] if dataset_key == "locomo10"
                          else "longmemeval:" + case["example_id"])
        if case["group_id"] != groups[case["example_id"]] or case["scope_id"] != expected_scope:
            raise ValueError("statistical group or retrieval scope does not match the actual history")
    exposure, overlap = _exposure_state(plan, dataset_key, examples, groups, ledger)
    if role == "locked_external_test" and (overlap or spec["role"] != "locked_external_test"):
        raise ValueError("independent test overlaps development groups or an exploratory dataset")
    group_count = len(set(groups.values()))
    expected_released = (manifest["counts"]["released_examples"] if manifest["fixture_only"]
                         else spec["expected_released_examples"])
    expected_counts = {
        "released_examples": expected_released, "examples": len(cases), "independent_groups": group_count,
        "excluded_prespecified_examples": expected_released - len(cases),
        "abstention_examples": sum(example.should_abstain for example in examples),
        "category_counts": dict(Counter(example.category for example in examples)),
    }
    if manifest.get("counts") != expected_counts or expected_released < len(cases):
        raise ValueError("cohort counts do not match actual rows/groups")
    if not manifest["fixture_only"]:
        if len(cases) != spec["expected_eligible_examples"] or (spec.get("expected_groups") is not None and group_count != spec["expected_groups"]):
            raise ValueError("production cohort does not preserve all planned cases/groups")
        if "expected_abstention_examples" in spec and expected_counts["abstention_examples"] != spec["expected_abstention_examples"]:
            raise ValueError("production cohort dropped declared abstention examples")
    blockers = _claim_blockers(plan, dataset_key, fixture_only=manifest["fixture_only"], role=role,
                               group_count=group_count, exposure=exposure)
    expected_metadata = {
        "three_arms": list(ARMS), "grouping_policy": spec["grouping"], "development_exposure": exposure,
        "development_exposed_group_ids": exposure["known_development_group_ids"],
        "preparation_eligible_for_confirmatory_test": not blockers, "confirmatory_claim_allowed": False,
        "claim_status": "preparation_only_effect_and_precision_not_yet_measured",
        "confirmatory_preparation_blockers": blockers, "final_claim_requirements": FINAL_CLAIM_REQUIREMENTS,
        "response_configuration": plan["shared_response_configuration"], "statistics": plan["statistics"],
        "memory_boundary": plan["memory_boundary"],
    }
    if any(manifest.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("cohort metadata/claim/response/statistics differ from frozen preregistration or actual history")
    return manifest, cases, references


def prepare_dataset(input_path: Path, dataset_key: str, plan_path: Path, output_dir: Path, *,
                    role: str | None = None, exposure_ledger_path: Path | None = None) -> dict:
    require_colab_runtime()
    plan = load_plan(plan_path)
    if dataset_key not in plan["datasets"]:
        raise ValueError("dataset is not registered in this frozen plan")
    spec = plan["datasets"][dataset_key]
    if dataset_key == "longmemeval_s_cleaned":
        validate_full_s_spec(spec)
    if input_path.stat().st_size != spec["bytes"]:
        raise ValueError("dataset byte count mismatch; oracle or incomplete downloads cannot qualify")
    identity = file_identity(input_path)
    if identity["sha256"] != spec["sha256"]:
        raise ValueError("dataset SHA-256 mismatch")
    _verify_pilot_exposure_source(plan)
    examples = load_locomo(input_path) if spec["adapter"] == "locomo" else load_longmemeval(input_path)
    ledger = json.loads(exposure_ledger_path.read_text(encoding="utf-8")) if exposure_ledger_path else None
    return prepare_examples(examples, dataset_key, plan, identity, output_dir,
                            role=role, exposure_ledger=ledger, fixture_only=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--dataset", choices=("locomo10", "longmemeval_s_cleaned"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--role", choices=("exploratory", "locked_external_test"))
    parser.add_argument("--exposure-ledger", type=Path)
    parser.add_argument("--inspect-plan", action="store_true")
    args = parser.parse_args()
    plan = load_plan(args.plan)
    if args.inspect_plan:
        print(json.dumps({"plan_sha256": stable_digest(plan), "datasets": plan["datasets"],
                          "statistics": plan["statistics"], "api_calls": 0,
                          "status": "plan inspection only; no dataset loaded or effect measured"}, indent=2))
        return
    if not args.dataset or args.input is None or args.output_dir is None:
        parser.error("provide --dataset, --input, --output-dir or use --inspect-plan")
    manifest = prepare_dataset(args.input, args.dataset, args.plan, args.output_dir,
                               role=args.role, exposure_ledger_path=args.exposure_ledger)
    print(json.dumps({key: manifest[key] for key in (
        "dataset_key", "role", "cohort_sha256", "counts", "preparation_eligible_for_confirmatory_test",
        "confirmatory_preparation_blockers", "claim_status")}, indent=2))


if __name__ == "__main__":
    main()
