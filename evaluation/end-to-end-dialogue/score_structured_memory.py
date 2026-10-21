#!/usr/bin/env python3
"""Recompute diagnostic proxies; export official/judge inputs without claiming scores."""

from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
from pathlib import Path

from run_structured_memory import ARMS, BOUNDARY, load_run
from run_three_arm import read_jsonl, stable_digest, write_jsonl
from score_three_arm import answer_proxies, paired_interval
from benchmark_adapters import write_longmemeval_hypotheses


def score_structured_run(run_dir: Path, *, write_outputs: bool = True) -> dict:
    protocol, prompts, references, completed = load_run(run_dir)
    expected_count = len(protocol["question_ids"]) * len(ARMS)
    if len(completed) != expected_count:
        raise ValueError("all frozen questions need all three successful responses; incomplete runs cannot be scored")
    if any(reference.get("category") == "5" for reference in references) and protocol["dataset_key"] == "locomo10":
        raise ValueError("LoCoMo adversarial distractors are not gold answers")
    reference_map = {row["example_id"]: row for row in references}
    group_ids = {}
    for prompt in prompts:
        example_id = prompt["example_id"]
        if example_id in group_ids and group_ids[example_id] != prompt["group_id"]:
            raise ValueError("statistical group differs across conditions")
        group_ids[example_id] = prompt["group_id"]
    per_example = []
    for example_id in protocol["question_ids"]:
        reference = reference_map[example_id]
        for arm in ARMS:
            response = completed[(example_id, arm)]
            per_example.append({
                "example_id": example_id, "group_id": group_ids[example_id], "arm": arm,
                "category": reference["category"], "should_abstain": reference["should_abstain"],
                **answer_proxies(response["hypothesis"], reference["reference_answer"]),
                "api_latency_seconds": response["latency_seconds"], "usage": response["usage"],
            })
    summary = {
        "protocol_sha256": stable_digest(protocol), "dataset_key": protocol["dataset_key"],
        "examples": len(references), "responses": len(completed),
        "independent_groups": len(set(group_ids.values())),
        "score_type": "non-official lexical answer proxies; diagnostic only",
        "memory_boundary": BOUNDARY, "extraction_mode": protocol["extraction_mode"],
        "official_evaluator_status": "not executed",
        "model_judge_status": "input prepared only" if write_outputs else "not executed",
        "confirmatory_claim": False,
        "claim_blockers": ["official/reference-based semantic scoring not executed",
                           "candidate evidence is not confirmed user memory",
                           "statistical adequacy requires the separately frozen formal plan"],
        "cohort_claim_status": protocol["cohort_manifest"].get("claim_status"),
        "reported_provider_models": sorted({row["provider_model"] for row in completed.values()}),
        "provider_identity_status": "all reported names match preapproved allowlist; backend weights not independently proven",
        "failed_attempts_retained": len(read_jsonl(run_dir / "failures.jsonl")),
        "latency_boundary": "API response time only; excludes extraction, indexing and retrieval",
        "conditions": {}, "paired_differences": {},
    }
    by_key = {(row["example_id"], row["arm"]): row for row in per_example}
    for arm in ARMS:
        subset = [row for row in per_example if row["arm"] == arm]
        summary["conditions"][arm] = {
            "examples": len(subset),
            "exact_match_proxy": statistics.mean(row["exact_match_proxy"] for row in subset),
            "token_f1_proxy": statistics.mean(row["token_f1_proxy"] for row in subset),
            "median_api_latency_seconds": statistics.median(row["api_latency_seconds"] for row in subset),
            "reported_total_tokens": sum(row["usage"].get("total_tokens", 0) for row in subset),
            "by_category": {category: {
                "examples": sum(row["category"] == category for row in subset),
                "token_f1_proxy": statistics.mean(row["token_f1_proxy"] for row in subset if row["category"] == category),
            } for category in sorted({row["category"] for row in subset})},
            "abstention_examples": sum(row["should_abstain"] for row in subset),
        }
    for baseline, treatment in itertools.combinations(ARMS, 2):
        deltas = {
            example_id: by_key[(example_id, treatment)]["token_f1_proxy"] - by_key[(example_id, baseline)]["token_f1_proxy"]
            for example_id in protocol["question_ids"]
        }
        groups = {}
        for example_id, delta in deltas.items():
            groups.setdefault(group_ids[example_id], []).append(delta)
        summary["paired_differences"][f"{treatment}_minus_{baseline}"] = {
            "mean_token_f1_delta": statistics.mean(deltas.values()),
            "cluster_bootstrap_95_percent_interval": (
                paired_interval(groups, seed=protocol["settings"]["seed"]) if len(groups) >= 2 else None
            ),
            "interval_status": ("estimated_exploratory_cluster_bootstrap" if len(groups) >= 2
                                else "not_estimable_fewer_than_two_groups"),
            "cluster_count": len(groups), "interval_interpretation": "exploratory diagnostic, not proof of efficacy",
            "wins": sum(value > 0 for value in deltas.values()),
            "ties": sum(value == 0 for value in deltas.values()),
            "losses": sum(value < 0 for value in deltas.values()),
        }
    if not write_outputs:
        return summary
    write_jsonl(run_dir / "structured_proxy_per_example.jsonl", per_example)
    (run_dir / "structured_proxy_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if protocol["dataset_key"] == "longmemeval_s_cleaned":
        for arm in ARMS:
            write_longmemeval_hypotheses({example_id: completed[(example_id, arm)]["hypothesis"] for example_id in protocol["question_ids"]},
                                       run_dir / f"official_longmemeval_{arm}.jsonl")
    pairs, mappings = [], []
    rng = random.Random(protocol["settings"]["seed"])
    for example_id in protocol["question_ids"]:
        for pair_index, pair in enumerate(itertools.combinations(ARMS, 2)):
            order = list(pair)
            rng.shuffle(order)
            pair_id = f"{example_id}:pair:{pair_index}"
            pairs.append({"pair_id": pair_id, "question": reference_map[example_id]["query"],
                          "reference_answer": reference_map[example_id]["reference_answer"],
                          "answer_A": completed[(example_id, order[0])]["hypothesis"],
                          "answer_B": completed[(example_id, order[1])]["hypothesis"],
                          "evaluation_boundary": "reference-based model-assisted review input; not human or official evaluation",
                          "rubric": "Judge factual correctness and completeness against the reference; preserve uncertainty when evidence is absent. Do not reward citations, style or length alone. Give a brief reason."})
            mappings.append({"pair_id": pair_id, "A": order[0], "B": order[1]})
    write_jsonl(run_dir / "structured_model_judge_pairs.jsonl", pairs)
    write_jsonl(run_dir / "structured_model_judge_mapping.jsonl", mappings)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(score_structured_run(args.run_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
