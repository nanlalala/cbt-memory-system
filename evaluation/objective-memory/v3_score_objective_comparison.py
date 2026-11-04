"""Score v3 with unchanged official rules and separately frozen answer diagnostics.

Secondary committed/alias scores are format-sensitive diagnostics, not an
independent semantic or clinical gold standard. No generation/Secret is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import v3_run_objective_comparison as runner
from score_objective_comparison import score_loaded_run, _read_jsonl, _atomic_json
from v3_response_diagnostics import score_supplemental, policy_descriptor

VERSION = "v3-objective-scoring-wrapper-v1"


def _subset(rows):
    by_arm = {}
    for arm in runner.ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        if not selected:
            continue
        by_arm[arm] = {
            "questions": len(selected), "independent_groups": len({r["group_id"] for r in selected}),
            "official_accuracy": sum(r["score"] for r in selected) / len(selected),
            "exact_alias_or_option_accuracy": sum(r["supplemental"]["exact_alias_or_option_score"] for r in selected) / len(selected),
            "strict_committed_accuracy": sum(r["supplemental"]["strict_committed_score"] for r in selected) / len(selected),
            "official_positives_with_abstention_cue": sum(bool(r["supplemental"].get("official_positive_with_abstention_cue")) for r in selected),
            "official_positives_not_strict": sum(bool(r["supplemental"].get("official_positive_not_strict")) for r in selected),
            "technical_failures": sum(not r["available"] for r in selected),
            "denominator": "all frozen questions including terminal technical failures",
        }
    return by_arm


def _score_run_locked(run_dir: Path, *, write_outputs=True, bootstrap_samples=10000):
    run_dir = Path(run_dir)
    protocol, prompts, references, completed = runner._load_run_locked(run_dir)
    if protocol.get("supplemental_response_diagnostics") != policy_descriptor():
        raise ValueError("supplemental response policy differs from frozen v3 protocol")
    outcomes = runner._request_outcomes_locked(run_dir, protocol, prompts, completed)
    summary, rows = score_loaded_run(protocol, prompts, references, completed, outcomes,
        call_outcomes=_read_jsonl(run_dir / "response_outcomes.jsonl"),
        call_attempts=_read_jsonl(run_dir / "response_attempts.jsonl"), bootstrap_samples=bootstrap_samples)
    refs = {r["case_id"]: r for r in references}
    pmap = {(r["case_id"], r["arm"]): r for r in prompts}
    for row in rows:
        key = row["case_id"], row["arm"]
        response = completed.get(key)
        if response is None:
            supplemental = {"official_score": 0.0, "exact_alias_or_option_score": 0.0,
                            "strict_committed_score": 0.0, "format": {"unavailable": True},
                            "official_positive_with_abstention_cue": False, "official_positive_not_strict": False}
        else:
            supplemental = score_supplemental(response["hypothesis"], refs[row["case_id"]])
        if supplemental["official_score"] != row["score"]:
            raise ValueError("official score changed while applying supplemental diagnostic")
        row["supplemental"] = supplemental
        # Preserve transparent v3 stage/chunk coverage; neither this trace nor
        # an answer substring establishes sufficient evidence or causal blame.
        row["v3_source_trace"] = pmap[key].get("trace", {})
    summary["v3_scoring_wrapper"] = VERSION
    summary["supplemental_response_policy"] = policy_descriptor()
    if rows:
        exposure = protocol.get("dataset_manifest", {}).get("development_partition", {})
        development = set(exposure.get("development_case_ids", []))
        heldout = set(exposure.get("question_heldout_case_ids", []))
        partitions = {"all_frozen": rows}
        if exposure:
            partitions["development_exposed_questions"] = [r for r in rows if r["case_id"] in development]
            partitions["question_heldout_shared_histories"] = [r for r in rows if r["case_id"] in heldout]
        summary["v3_diagnostics"] = {
            "conditions": _subset(rows),
            "by_task": {task: _subset([r for r in rows if r["task"] == task]) for task in sorted({r["task"] for r in rows})},
            "question_exposure_partitions": {name: _subset(part) for name, part in partitions.items()},
            "boundary": "official rules unchanged; exact/strict metrics are predeclared string/format diagnostics, not independent semantic correctness; question-held-out histories remain shared and development exposed",
        }
    summary["input_artifacts_sha256"] = {
        name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
        for name in ("protocol.json", *runner.FROZEN_ROWS, "responses.jsonl", "response_attempts.jsonl", "response_outcomes.jsonl", "failures.jsonl")
        if (run_dir / name).exists()}
    summary["scoring_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if write_outputs:
        if summary["status"] == "complete":
            runner._replace_rows(run_dir / "objective_scores.jsonl", rows)
            _atomic_json(run_dir / "objective_summary.json", summary)
            (run_dir / "objective_progress.json").unlink(missing_ok=True)
        else:
            (run_dir / "objective_scores.jsonl").unlink(missing_ok=True)
            (run_dir / "objective_summary.json").unlink(missing_ok=True)
            _atomic_json(run_dir / "objective_progress.json", summary)
    return summary


def score_run(run_dir: Path, *, write_outputs=True, bootstrap_samples=10000):
    with runner.exclusive_directory(Path(run_dir)):
        return _score_run_locked(Path(run_dir), write_outputs=write_outputs, bootstrap_samples=bootstrap_samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(score_run(args.run_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
