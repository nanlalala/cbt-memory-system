#!/usr/bin/env python3
"""Local answer proxies and blinded judge/export preparation, not official scores."""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path

from run_three_arm import ARMS, load_checkpoints, read_jsonl, stable_digest, write_jsonl
from benchmark_adapters import write_longmemeval_hypotheses


def normalized_tokens(value: str) -> list[str]:
    value = re.sub(r"\b(a|an|the)\b", " ", value.casefold())
    # Character-level CJK and word-level Latin is a proxy, not the official scorer.
    return re.findall(r"[a-z0-9]+|[\u3400-\u9fff]", value)


def answer_proxies(hypothesis: str, reference: str) -> dict[str, float]:
    predicted, gold = normalized_tokens(hypothesis), normalized_tokens(reference)
    exact = float(predicted == gold)
    if not predicted or not gold:
        return {"exact_match_proxy": exact, "token_f1_proxy": exact}
    overlap = sum((Counter(predicted) & Counter(gold)).values())
    precision, recall = overlap / len(predicted), overlap / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"exact_match_proxy": exact, "token_f1_proxy": f1}


def paired_interval(groups: dict[str, list[float]], *, seed: int, resamples: int = 2000) -> list[float]:
    """Resample conversations/scopes, not correlated questions as independent rows."""
    if not groups:
        return [0.0, 0.0]
    rng = random.Random(seed)
    scopes = sorted(groups)
    values = sorted(statistics.mean(value for scope in rng.choices(scopes, k=len(scopes)) for value in groups[scope])
                    for _ in range(resamples))
    return [values[int(0.025 * (resamples - 1))], values[int(0.975 * (resamples - 1))]]


def score_run(run_dir: Path) -> dict:
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    prompts = read_jsonl(run_dir / "prompts.jsonl")
    responses = load_checkpoints(run_dir / "responses.jsonl", prompts, stable_digest(protocol))
    references = read_jsonl(run_dir / "references.jsonl")
    reference_map = {row["example_id"]: row for row in references}
    expected_ids = protocol["question_ids"]
    expected_prompt_keys = {(example_id, arm) for example_id in expected_ids for arm in ARMS}
    if ({(row["example_id"], row["arm"]) for row in prompts} != expected_prompt_keys
            or len(prompts) != len(expected_prompt_keys)
            or stable_digest(prompts) != protocol["prompt_records_sha256"]):
        raise ValueError("prompt records do not match frozen IDs/conditions/checksum")
    scope_by_example = {row["example_id"]: row["scope_id"] for row in prompts}
    if (len(reference_map) != len(references) or set(reference_map) != set(expected_ids)
            or stable_digest(references) != protocol["references_sha256"]):
        raise ValueError("references do not match frozen question IDs/checksum")
    if len(responses) != len(expected_ids) * len(ARMS) or any(
        (example_id, arm) not in responses for example_id in expected_ids for arm in ARMS
    ):
        raise ValueError("all three successful responses are required per frozen example")
    reported_models = {row.get("provider_model") for row in responses.values() if row.get("provider_model")}
    if len(reported_models) > 1:
        raise ValueError("provider model changed across responses; fixed-model comparison is invalid")
    per_example = []
    for example_id in expected_ids:
        reference = reference_map[example_id]
        for arm in ARMS:
            result = responses[(example_id, arm)]
            per_example.append({"example_id": example_id, "arm": arm,
                                "category": reference["category"], "should_abstain": reference["should_abstain"],
                                **answer_proxies(result["hypothesis"], reference["reference_answer"]),
                                "latency_seconds": result["latency_seconds"], "usage": result["usage"]})
    write_jsonl(run_dir / "proxy_per_example.jsonl", per_example)
    summary = {"examples": len(expected_ids), "protocol_sha256": stable_digest(protocol),
               "score_type": "non-official lexical answer proxies",
               "oracle_history_warning": protocol["oracle_history_warning"],
               "official_evaluator_status": "not executed", "model_judge_status": "input prepared only",
               "reported_provider_models": sorted(reported_models),
               "provider_identity_status": "consistent reported name" if len(reported_models) == 1 and all(
                   row.get("provider_model") for row in responses.values()) else "unverified; provider omitted model identity",
               "conditions": {}, "paired_differences": {}}
    by_key = {(row["example_id"], row["arm"]): row for row in per_example}
    for arm in ARMS:
        subset = [row for row in per_example if row["arm"] == arm]
        arm_summary = {metric: statistics.mean(row[metric] for row in subset)
                       for metric in ("exact_match_proxy", "token_f1_proxy")}
        arm_summary["median_latency_seconds"] = statistics.median(row["latency_seconds"] for row in subset)
        arm_summary["reported_total_tokens"] = sum(row["usage"].get("total_tokens", 0) for row in subset)
        arm_summary["abstention_examples"] = sum(row["should_abstain"] for row in subset)
        arm_summary["abstention_token_f1_proxy"] = (statistics.mean(row["token_f1_proxy"] for row in subset if row["should_abstain"])
                                                      if arm_summary["abstention_examples"] else None)
        arm_summary["by_category"] = {}
        for category in sorted({row["category"] for row in subset}):
            category_rows = [row for row in subset if row["category"] == category]
            arm_summary["by_category"][category] = {
                "examples": len(category_rows),
                "token_f1_proxy": statistics.mean(row["token_f1_proxy"] for row in category_rows),
                "exact_match_proxy": statistics.mean(row["exact_match_proxy"] for row in category_rows),
            }
        summary["conditions"][arm] = arm_summary
    if protocol["dataset"] == "locomo":
        if any(row["category"] == "5" for row in references):
            raise ValueError("LoCoMo adversarial distractors cannot be scored as gold answers")
        summary["excluded_adversarial_distractors"] = protocol["excluded_category_5_examples"]
    for baseline, treatment in itertools.combinations(ARMS, 2):
        deltas = [by_key[(example_id, treatment)]["token_f1_proxy"] - by_key[(example_id, baseline)]["token_f1_proxy"]
                  for example_id in expected_ids]
        groups: dict[str, list[float]] = {}
        for example_id, delta in zip(expected_ids, deltas):
            groups.setdefault(scope_by_example[example_id], []).append(delta)
        summary["paired_differences"][f"{treatment}_minus_{baseline}"] = {
            "mean_token_f1_delta": statistics.mean(deltas),
            "conversation_cluster_bootstrap_95_percent_interval": paired_interval(groups, seed=protocol["settings"]["seed"]),
            "cluster_count": len(groups), "interval_interpretation": "exploratory; not a clinical significance claim",
            "wins": sum(value > 0 for value in deltas), "ties": sum(value == 0 for value in deltas),
            "losses": sum(value < 0 for value in deltas),
        }
    if protocol["dataset"] == "longmemeval":
        for arm in ARMS:
            write_longmemeval_hypotheses({key: responses[(key, arm)]["hypothesis"] for key in expected_ids},
                                       run_dir / f"official_longmemeval_{arm}.jsonl")
    judge_pairs, judge_mapping = [], []
    rng = random.Random(protocol["settings"]["seed"])
    for example_id in expected_ids:
        for pair_index, pair in enumerate(itertools.combinations(ARMS, 2)):
            shuffled = list(pair)
            rng.shuffle(shuffled)
            pair_id = f"{example_id}:pair:{pair_index}"
            judge_pairs.append({"pair_id": pair_id, "question": reference_map[example_id]["query"],
                                "reference_answer": reference_map[example_id]["reference_answer"],
                                "answer_A": responses[(example_id, shuffled[0])]["hypothesis"],
                                "answer_B": responses[(example_id, shuffled[1])]["hypothesis"],
                                "rubric": "Evaluate correctness/completeness against reference and uncertainty boundaries; do not reward citations/style/length alone."})
            judge_mapping.append({"pair_id": pair_id, "A": shuffled[0], "B": shuffled[1]})
    write_jsonl(run_dir / "model_judge_pairs.jsonl", judge_pairs)
    write_jsonl(run_dir / "model_judge_mapping.jsonl", judge_mapping)
    (run_dir / "proxy_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(score_run(args.run_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
