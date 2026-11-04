#!/usr/bin/env python3
"""Offline deterministic scoring of the frozen, adapted memory comparison.

These are the upstream answer rules, not a claim of running the unmodified
benchmark leaderboard or measuring CBT treatment quality. No judge/API is used.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
import statistics
import string
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "end-to-end-dialogue"))
from execution_lock import exclusive_directory  # noqa: E402

VERSION = "objective-memory-deterministic-score-v1"
ARMS = ("short_term", "raw_history_rag", "managed_memory_rag")
METRIC_SOURCES = {
    "mab_substring_exact_match": {
        "repository": "https://github.com/HUST-AI-HYZ/MemoryAgentBench",
        "revision": "42977c66b215520da3a988c0838822af6cafe300",
        "path": "utils/eval_other_utils.py",
        "source_sha256": "d77976be409298970614d477a9d8003850caddb0510e56a7e821a037d98493a2",
        "functions": ["normalize_answer", "substring_exact_match_score", "parse_output", "default_post_process"],
    },
    "personamem_option_exact_match": {
        "repository": "https://github.com/bowen-upenn/PersonaMem",
        "revision": "caaae44b3f236b8751d499a770e94e5aecffcff1",
        "path": "inference.py",
        "source_sha256": "9b77115a0d92d255c4c6ef6a710f8975e178d7f0c902739a1538411e06b8a71a",
        "functions": ["extract_answer"],
    },
}


def stable_digest(value: object) -> str:
    # Match the runner's established JSON digest encoding exactly.
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def mab_normalize_answer(text: str) -> str:
    """Upstream order is lower, ASCII punctuation, English articles, whitespace."""
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def mab_parse_output(text: str, answer_prefix: str = "Answer:") -> str | None:
    # Preserve upstream first-line/prefix extraction, including its fallback.
    for pattern in (re.compile(f"(?:{answer_prefix})(.*)(?:\n|$)", re.IGNORECASE),
                    re.compile(r"(?:^)(.*)(?:\n|$)")):
        match = pattern.search(text)
        if match:
            return re.sub(f"^{re.escape(answer_prefix)}", "", match[1].strip(),
                          flags=re.IGNORECASE).strip()
    return None


def mab_substring_exact_match(prediction: str, accepted_answers: list[str]) -> tuple[float, str | None]:
    if not isinstance(accepted_answers, list) or not accepted_answers or any(
            not isinstance(value, str) or not value.strip() for value in accepted_answers):
        raise ValueError("MAB requires the released nonempty answer aliases")
    parsed = mab_parse_output(prediction)
    # Upstream default_post_process takes max of the full and parsed predictions.
    candidates = [prediction] + ([parsed] if parsed is not None else [])
    score = any(mab_normalize_answer(gold) in mab_normalize_answer(candidate)
                for candidate in candidates for gold in accepted_answers)
    return float(score), parsed


def personamem_option_exact_match(prediction: str, correct_option: str) -> tuple[float, str]:
    """Preserve PersonaMem's parentheses precedence and full-response fallback."""
    if not isinstance(correct_option, str):
        raise ValueError("PersonaMem requires a released a-d option label")
    correct = correct_option.lower().strip("() ")
    if correct not in {"a", "b", "c", "d"}:
        raise ValueError("PersonaMem requires a released a-d option label")

    def options(text: str) -> set[str]:
        text = text.lower()
        in_parens = re.findall(r"\(([a-d])\)", text)
        return set(in_parens) if in_parens else set(re.findall(r"\b([a-d])\b", text))

    full_response = prediction
    parsed = prediction.strip()
    if "<final_answer>" in parsed:
        parsed = parsed.split("<final_answer>")[-1].strip()
    if parsed.endswith("</final_answer>"):
        parsed = parsed[:-len("</final_answer>")].strip()
    score = options(parsed) == {correct} or options(full_response) == {correct}
    return float(score), parsed


def deterministic_score(prediction: str, reference: dict) -> tuple[float, str | None]:
    if not isinstance(prediction, str):
        raise ValueError("response hypothesis must be text")
    metric = reference.get("metric")
    if metric == "mab_substring_exact_match":
        return mab_substring_exact_match(prediction, reference.get("accepted_answers"))
    if metric == "personamem_option_exact_match":
        return personamem_option_exact_match(prediction, reference.get("correct_option"))
    raise ValueError(f"unsupported frozen deterministic metric: {metric}")


def _case_id(row: dict) -> str:
    value = row.get("case_id", row.get("example_id"))
    if not isinstance(value, str) or not value:
        raise ValueError("case ID must be nonempty text")
    if row.get("example_id", value) != value:
        raise ValueError("case_id and example_id differ")
    return value


def _quantile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def paired_cluster_intervals(groups: dict[str, list[float]], *, seed: int,
                             resamples: int) -> dict:
    """Resample entire shared histories; report both explicit weightings."""
    if len(groups) < 2:
        return {"group_equal_95_percent_interval": None, "micro_95_percent_interval": None,
                "interval_status": "not_estimable_fewer_than_two_groups"}
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples < 100:
        raise ValueError("cluster bootstrap needs at least 100 resamples")
    rng = random.Random(seed)
    stats = [(statistics.mean(groups[key]), sum(groups[key]), len(groups[key])) for key in sorted(groups)]
    macro, micro = [], []
    for _ in range(resamples):
        sample = rng.choices(stats, k=len(stats))
        macro.append(sum(value[0] for value in sample) / len(sample))
        micro.append(sum(value[1] for value in sample) / sum(value[2] for value in sample))
    macro.sort()
    micro.sort()
    return {"group_equal_95_percent_interval": [_quantile(macro, .025), _quantile(macro, .975)],
            "micro_95_percent_interval": [_quantile(micro, .025), _quantile(micro, .975)],
            "interval_status": "exploratory_percentile_cluster_bootstrap"}


def _summarize_subset(rows: list[dict], *, seed: int, bootstrap_samples: int) -> dict:
    conditions = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        groups = defaultdict(list)
        for row in selected:
            groups[row["group_id"]].append(row["score"])
        available = [row for row in selected if row["available"]]
        conditions[arm] = {
            "questions": len(selected), "independent_groups": len(groups),
            "accuracy_micro": statistics.mean(row["score"] for row in selected),
            "accuracy_group_equal": statistics.mean(statistics.mean(values) for values in groups.values()),
            "availability": len(available) / len(selected),
            "technical_failures": len(selected) - len(available),
            "accuracy_given_valid_response_diagnostic": (
                statistics.mean(row["score"] for row in available) if available else None),
            "logical_response_reported_total_tokens": sum(row.get("reported_usage", {}).get("total_tokens", 0) for row in available),
            "valid_responses_missing_total_token_usage": sum("total_tokens" not in row.get("reported_usage", {}) for row in available),
            "median_valid_response_api_latency_seconds": statistics.median(
                row["api_latency_seconds"] for row in available if row.get("api_latency_seconds") is not None)
                if any(row.get("api_latency_seconds") is not None for row in available) else None,
            "cost_boundary": "logical arm responses; shared-response tokens/latencies are reused, not separately billed or independently timed; actual unique-call cost is reported globally",
        }
    by_key = {(row["case_id"], row["arm"]): row for row in rows}
    case_ids = sorted({row["case_id"] for row in rows})
    contrasts = {}
    for baseline, treatment in itertools.combinations(ARMS, 2):
        groups = defaultdict(list)
        nonidentical_groups = defaultdict(list)
        identical = 0
        for case_id in case_ids:
            treated, control = by_key[(case_id, treatment)], by_key[(case_id, baseline)]
            delta = treated["score"] - control["score"]
            groups[treated["group_id"]].append(delta)
            shared = bool(treated.get("canonical_request_key")) and treated["canonical_request_key"] == control.get("canonical_request_key")
            identical += shared
            if not shared:
                nonidentical_groups[treated["group_id"]].append(delta)
        deltas = [value for values in groups.values() for value in values]
        mechanically_equal = identical == len(case_ids)
        interval = ({"group_equal_95_percent_interval": None, "micro_95_percent_interval": None,
                     "interval_status": "blocked_by_identical_inputs_and_shared_responses"}
                    if mechanically_equal else paired_cluster_intervals(groups, seed=seed, resamples=bootstrap_samples))
        nonidentical = [value for values in nonidentical_groups.values() for value in values]
        contrasts[f"{treatment}_minus_{baseline}"] = {
            "primary_contrast": (baseline, treatment) == ("raw_history_rag", "managed_memory_rag"),
            "questions": len(deltas), "independent_groups": len(groups),
            "identical_prompt_shared_response_cases": identical,
            "identical_prompt_shared_response_fraction": identical / len(case_ids),
            "nonidentical_prompt_cases": len(nonidentical),
            "result_type": ("by_construction_control_equivalence" if mechanically_equal else "exploratory_paired_comparison"),
            "inference_boundary": ("identical inputs share the same response by construction; zero difference is not empirical efficacy or statistical equivalence evidence"
                                   if mechanically_equal else "all frozen pairs remain primary; nonidentical-input subset is diagnostic and does not replace the cohort"),
            "mean_accuracy_delta_micro": statistics.mean(deltas),
            "mean_accuracy_delta_group_equal": statistics.mean(statistics.mean(values) for values in groups.values()),
            "wins": sum(value > 0 for value in deltas), "ties": sum(value == 0 for value in deltas),
            "losses": sum(value < 0 for value in deltas),
            **interval,
            "nonidentical_input_subset_diagnostic": {
                "questions": len(nonidentical), "independent_groups": len(nonidentical_groups),
                "mean_accuracy_delta_micro": statistics.mean(nonidentical) if nonidentical else None,
                "mean_accuracy_delta_group_equal": (
                    statistics.mean(statistics.mean(values) for values in nonidentical_groups.values()) if nonidentical else None),
                "inference": "descriptive subset only; not a replacement primary endpoint",
            },
        }
    return {"conditions": conditions, "paired_differences": contrasts}


def _trace_diagnostics(trace: dict) -> dict:
    """Only software-observable counts; missing gold evidence cannot locate blame."""
    if not isinstance(trace, dict):
        raise ValueError("prompt trace must be an object")
    allowed = ("included_short_ids", "included_long_ids", "included_memory_ids", "dropped_short_ids",
               "dropped_long_ids", "dropped_memory_ids", "retrieved_ids", "clarification_ids")
    trace = dict(trace)
    for source, target in (("budget_dropped_short_ids", "dropped_short_ids"),
                           ("budget_dropped_long_ids", "dropped_long_ids")):
        if source in trace:
            if target in trace and trace[target] != trace[source]:
                raise ValueError("conflicting budget-drop trace fields")
            trace[target] = trace[source]
    result = {}
    for key in allowed:
        if key in trace:
            if not isinstance(trace[key], list):
                raise ValueError(f"trace {key} must be a list")
            result[key] = trace[key]
            result[key.removesuffix("_ids") + "_count"] = len(trace[key])
    for key in ("first_failure_stage", "clarification_count", "user_confirmation_count",
                "preparation_seconds", "retrieval_seconds", "context_chars", "source_unit_count",
                "encoding", "current_use", "interactive_clarification"):
        if key in trace:
            result[key] = trace[key]
    return result


def _costs(completed: dict, call_outcomes: list[dict], call_attempts: list[dict] | None = None) -> dict:
    # Journal outcomes are one per actual attempt. A shared prompt has one cost,
    # while all three arms may refer to that answer for their paired comparison.
    charged = [row for row in completed.values() if row.get("charged_call", True)]
    token_keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    def has_usage(row: dict) -> bool:
        usage = row.get("usage")
        return isinstance(usage, dict) and any(isinstance(usage.get(key), (int, float)) for key in token_keys)
    known_usage = [row["usage"] for row in charged if has_usage(row)]
    latencies = [row["latency_seconds"] for row in charged
                 if isinstance(row.get("latency_seconds"), (float, int))]
    failed = [row for row in call_outcomes if row.get("status") != "success"]
    failed_usage = [row["usage"] for row in failed if has_usage(row)]
    token_totals = {key: sum(usage.get(key, 0) for usage in known_usage + failed_usage)
                    for key in token_keys}
    return {
        "recorded_provider_attempt_intents": len(call_attempts) if call_attempts is not None else None,
        "intents_without_outcome": max(0, len(call_attempts) - len(call_outcomes)) if call_attempts is not None else None,
        "successful_unique_provider_calls": len(charged),
        "failed_provider_attempts": len(failed),
        "reported_tokens_including_failed_attempts_when_supplied": token_totals,
        "failed_attempts_without_token_usage": len(failed) - len(failed_usage),
        "successful_calls_without_token_usage": len(charged) - len(known_usage),
        "median_successful_api_latency_seconds": statistics.median(latencies) if latencies else None,
        "sum_successful_api_latency_seconds": sum(latencies),
        "latency_boundary": "provider request/response time only; excludes preparation, extraction, indexing, retrieval and failed-call time unless separately reported",
        "deduplication_boundary": "actual canonical calls counted once; shared responses are not charged once per arm",
        "monetary_cost": None,
        "monetary_cost_status": "not inferred without a frozen provider price schedule and complete billable usage",
    }


def score_loaded_run(protocol: dict, prompts: list[dict], references: list[dict], completed: dict,
                     outcomes: dict, *, call_outcomes: list[dict] | None = None,
                     call_attempts: list[dict] | None = None,
                     bootstrap_samples: int = 10000) -> tuple[dict, list[dict]]:
    """Pure aggregation after the runner has verified every frozen artifact."""
    case_ids = protocol.get("case_ids", protocol.get("question_ids"))
    if not isinstance(case_ids, list) or not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("frozen protocol needs unique nonempty case IDs")
    expected = {(case_id, arm) for case_id in case_ids for arm in ARMS}
    prompt_map = {(_case_id(row), row["arm"]): row for row in prompts}
    reference_map = {_case_id(row): row for row in references}
    if len(prompt_map) != len(prompts) or set(prompt_map) != expected:
        raise ValueError("every frozen case needs exactly three prompt arms")
    if len(reference_map) != len(references) or set(reference_map) != set(case_ids):
        raise ValueError("references differ from frozen cases")
    if set(outcomes) != expected or not set(completed).issubset(expected):
        raise ValueError("response outcomes differ from frozen requests")
    for case_id, reference in reference_map.items():
        # Validate gold rules even when the whole arm technically failed.
        deterministic_score("", reference)
        for arm in ARMS:
            prompt = prompt_map[(case_id, arm)]
            for key in ("group_id", "dataset_key", "task"):
                if not reference.get(key) or prompt.get(key) != reference[key]:
                    raise ValueError(f"{key} differs between frozen arms/reference")
    policy = protocol.get("evaluation_policy", {})
    if "bootstrap_samples" in policy and policy["bootstrap_samples"] != bootstrap_samples:
        raise ValueError("bootstrap resamples differ from frozen evaluation policy")
    counts = Counter(row.get("status") for row in outcomes.values())
    allowed_status = {"success", "exhausted_failure", "pending", "inflight"}
    if not set(counts).issubset(allowed_status):
        raise ValueError("unknown request completion state")
    for key, outcome in outcomes.items():
        if (outcome["status"] == "success") != (key in completed):
            raise ValueError("successful outcome and validated response disagree")
    common = {
        "scoring_version": VERSION, "protocol_sha256": protocol.get("protocol_sha256", stable_digest(protocol)),
        "fixture_only": bool(protocol.get("fixture_only", False)),
        "result_origin": "synthetic_software_fixture" if protocol.get("fixture_only", False) else "frozen_public_benchmark_cohort",
        "cases": len(case_ids), "frozen_requests": len(expected),
        "completion_counts": dict(sorted(counts.items())),
        "primary_contrast": "managed_memory_rag_minus_raw_history_rag",
        "primary_weighting": "equal shared-history/persona groups within each dataset",
        "metric_sources": METRIC_SOURCES,
        "reported_provider_models": sorted({row["provider_model"] for row in completed.values()
                                             if isinstance(row.get("provider_model"), str)}),
        "provider_identity_boundary": "runner validates the frozen reported-name allowlist; underlying provider weights are not independently proven",
        "clinical_or_user_benefit_claim": False,
        "benchmark_boundary": "released answer rules on adapted three-arm contexts; not unmodified leaderboard scores or a CBT clinical outcome",
        "cost": _costs(completed, call_outcomes or [], call_attempts),
    }
    if counts["pending"] or counts["inflight"]:
        return {**common, "status": "partial", "final_effectiveness_result": False,
                "reason": "each frozen request must succeed or reach explicit terminal/exhausted failure before final scoring",
                "remaining_requests": counts["pending"] + counts["inflight"]}, []
    rows = []
    for case_id in case_ids:
        reference = reference_map[case_id]
        for arm in ARMS:
            key = case_id, arm
            response = completed.get(key)
            available = response is not None
            score, parsed = deterministic_score(response["hypothesis"], reference) if available else (0.0, None)
            trace = _trace_diagnostics(prompt_map[key].get("trace", {}))
            rows.append({
                "case_id": case_id, "group_id": reference["group_id"], "dataset_key": reference["dataset_key"],
                "task": reference["task"], "arm": arm, "metric": reference["metric"],
                "score": score, "available": available, "completion_status": outcomes[key]["status"],
                "parsed_prediction": parsed, "trace": trace,
                "reported_usage": response.get("usage", {}) if available else {},
                "api_latency_seconds": response.get("latency_seconds") if available else None,
                "charged_call": response.get("charged_call", True) if available else False,
                "failure_stage": ("provider_technical_failure" if not available else
                                  "none" if score else "answer_incorrect_stage_unresolved"),
                "stale_answer_diagnostic": None,
                "stale_answer_status": "not_computed_without_released_old_and_current_answer_annotations",
                "canonical_request_key": prompt_map[key].get("canonical_request_key",
                    response.get("canonical_request_key") if available else outcomes[key].get("canonical_request_key")),
                "deduplicated_call": response.get("deduplicated_call", False) if available else None,
            })
    datasets = {}
    seed = policy.get("bootstrap_seed", policy.get("seed", protocol.get("settings", {}).get("seed", 42)))
    for dataset_key in sorted({row["dataset_key"] for row in rows}):
        selected = [row for row in rows if row["dataset_key"] == dataset_key]
        datasets[dataset_key] = {
            **_summarize_subset(selected, seed=seed, bootstrap_samples=bootstrap_samples),
            "by_task": {task: _summarize_subset([row for row in selected if row["task"] == task],
                                                seed=seed, bootstrap_samples=bootstrap_samples)
                        for task in sorted({row["task"] for row in selected})},
        }
    return {**common, "status": "complete", "final_effectiveness_result": not bool(protocol.get("fixture_only", False)),
            "score_type": "upstream deterministic answer metrics; technical failures remain zero in the frozen denominator",
            "datasets": datasets, "independent_groups": len({(row["dataset_key"], row["group_id"]) for row in rows}),
            "software_diagnostics": {
                "answer_outcomes_by_stage": dict(Counter(row["failure_stage"] for row in rows)),
                "trace_counts": {key: sum(row["trace"].get(key, 0) for row in rows)
                                 for key in ("included_short_count", "included_long_count", "included_memory_count",
                                             "dropped_short_count", "dropped_long_count", "dropped_memory_count",
                                             "retrieved_count", "clarification_count", "user_confirmation_count")},
                "attribution_boundary": "only explicit software traces; answer errors are not assigned to extraction/retrieval/generation without released evidence annotations",
            },
            "bootstrap_samples": bootstrap_samples, "bootstrap_seed": seed,
            "confirmatory_claim": False,
            "inference_status": "exploratory; no independent-sample power or target-precision adequacy is inferred from QA count",
            "sampling_boundary": "dataset history/persona/context clusters, not independent clinical users; cluster intervals do not remove selection or benchmark dependence",
            "score_limitations": ["MAB substring matching may accept verbose, contradictory or answer-containing questions; it is not semantic/clinical correctness",
                                  "PersonaMem is released-option accuracy; no user answer is invented for clarification requests",
                                  "availability-conditioned accuracy is diagnostic; primary denominator includes technical failures",
                                  "no pooled cross-dataset accuracy or clinical effectiveness claim is produced"]}, rows


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _score_run_locked(run_dir: Path, *, write_outputs: bool = True,
                      bootstrap_samples: int = 10000) -> dict:
    from run_objective_comparison import _load_run_locked, _request_outcomes_locked  # noqa: PLC0415
    run_dir = Path(run_dir)
    protocol, prompts, references, completed = _load_run_locked(run_dir)
    outcomes = _request_outcomes_locked(run_dir, protocol, prompts, completed)
    summary, rows = score_loaded_run(protocol, prompts, references, completed, outcomes,
                                     call_outcomes=_read_jsonl(run_dir / "response_outcomes.jsonl"),
                                     call_attempts=_read_jsonl(run_dir / "response_attempts.jsonl"),
                                     bootstrap_samples=bootstrap_samples)
    summary["input_artifacts_sha256"] = {
        name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
        for name in ("protocol.json", "cases.jsonl", "inventory.jsonl", "prompts.jsonl", "references.jsonl", "responses.jsonl",
                     "response_attempts.jsonl", "response_outcomes.jsonl", "failures.jsonl")
        if (run_dir / name).exists()
    }
    summary["scoring_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if write_outputs:
        if summary["status"] == "complete":
            temporary = run_dir / "objective_scores.jsonl.tmp"
            temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            temporary.replace(run_dir / "objective_scores.jsonl")
            _atomic_json(run_dir / "objective_summary.json", summary)
            (run_dir / "objective_progress.json").unlink(missing_ok=True)
        else:
            # Stale or user-supplied derived files cannot masquerade as final.
            (run_dir / "objective_scores.jsonl").unlink(missing_ok=True)
            (run_dir / "objective_summary.json").unlink(missing_ok=True)
            _atomic_json(run_dir / "objective_progress.json", summary)
    return summary


def score_run(run_dir: Path, *, write_outputs: bool = True,
              bootstrap_samples: int = 10000) -> dict:
    with exclusive_directory(run_dir):
        return _score_run_locked(Path(run_dir), write_outputs=write_outputs,
                                 bootstrap_samples=bootstrap_samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    print(json.dumps(score_run(args.run_dir, bootstrap_samples=args.bootstrap_samples), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
