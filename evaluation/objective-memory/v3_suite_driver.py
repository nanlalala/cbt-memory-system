"""Frozen hosted-Colab full-cohort planning, finite execution and aggregation.

Full mode fixes 800 MAB questions and 589 PersonaMem questions before any batch.
Shards are execution units, never new independent samples. No Secret is read by
this driver: generation inherits the existing temporary Colab environment only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import v3_run_objective_comparison as runner
import v3_score_objective_comparison as scoring
from objective_datasets import dataset_spec, download_registered_dataset, load_dataset_cases, stable_hash
from score_objective_comparison import _summarize_subset
from v3_persona_retrieval import plan_group_shards
from v3_response_diagnostics import development_partition, partition_from_exposed_cases

VERSION = "v3-full-cohort-suite-v1"
DATASETS = ("memoryagentbench_conflict", "personamem_v1")
API_BASE = "https://tokendance.space/gateway/v1"
MODEL = "deepseek-v4.1-flash"
ACCEPTED_MODEL = "deepseek-v4-1-flash-260910"
BOOTSTRAP_SEED = 20261104
BOOTSTRAP_SAMPLES = 10000


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    runner._replace_rows(path, [value])


def _read_json(path):
    return json.loads(Path(path).read_text())


def _codes():
    return {**runner._code_hashes(), str(Path(__file__).resolve().relative_to(runner.ROOT)): _sha(__file__)}


def _settings(dataset, *, max_context_chars=24000):
    return runner.Settings(model=MODEL, accepted_provider_models=(ACCEPTED_MODEL,), backend="lexical",
        thinking_mode="disabled", short_turns=16, top_k=12, short_chars=6000, long_chars=12000,
        max_context_chars=max_context_chars, max_response_tokens=128 if dataset == DATASETS[0] else 64)


def _capacity(dataset, cases):
    overhead = max(sum(len(m["content"]) for m in runner._messages(case, "", "")) for case in cases)
    cap = max(24000, overhead + 6000 + 12000)
    return {"maximum_observed_empty_context_characters": overhead, "shared_max_context_characters": cap,
            "policy": "full-cohort source-independent question/ALL-option overhead plus shared6000recent+12000older; no option truncation or outcome tuning",
            "settings": json.loads(json.dumps(asdict(_settings(dataset, max_context_chars=cap))))}


def _generation_identity(cases):
    # MAB reuses the same long context object across100questions. Hash each
    # actual object once, retaining its exact bytes plus each case's metadata;
    # avoid serializing repeated262k contexts into a multi-gigabyte JSON value.
    histories, rows = {}, []
    for case in cases:
        history = case.get("history", [])
        key = id(history)
        if key not in histories:
            histories[key] = runner.stable_digest(history)
        rows.append({**{k: v for k, v in case.items() if k != "history"}, "history_sha256": histories[key]})
    return {"case_metadata_and_exact_history_hashes_sha256": runner.stable_digest(rows),
            "history_hashing_policy": "per-case metadata plus SHA256ofexacthistory; same in-memory history object hashed once, no synthetic independent observations"}


def _exposure(path):
    path = Path(path)
    protocol_path = path.parent / "protocol.json"
    protocol = _read_json(protocol_path)
    if protocol.get("protocol_sha256") != runner.stable_digest({k: v for k, v in protocol.items() if k != "protocol_sha256"}):
        raise ValueError("exposed source protocol checksum mismatch")
    if path.name != "cases.jsonl" or _sha(path) != protocol["file_sha256"].get("cases.jsonl"):
        raise ValueError("exposed cases must match the frozen old run")
    cases = runner.read_jsonl(path)
    if [c["case_id"] for c in cases] != protocol["case_ids"]:
        raise ValueError("exposed source cohort mismatch")
    return cases, {"source_protocol_sha256": protocol["protocol_sha256"], "cases_sha256": _sha(path)}


def plan_cohorts(cohorts, exposed_cases, *, mode="full", strict_census=True):
    """Gold-free planning seam; strict_census=False is for tiny software tests."""
    if mode not in {"dev16", "full"}:
        raise ValueError("suite mode must be dev16 or full")
    if DATASETS[0] not in cohorts or (mode == "full" and DATASETS[1] not in cohorts):
        raise ValueError("missing required dataset cohort")
    old_mab = [c for c in exposed_cases if c.get("dataset_key") == DATASETS[0]]
    if strict_census and len(old_mab) != 16:
        raise ValueError("the declared development source must contain exactly 16 MAB cases")
    if len({c["case_id"] for c in old_mab}) != len(old_mab) or not old_mab:
        raise ValueError("empty or duplicate exposed development cases")
    full_mab = cohorts[DATASETS[0]]
    mab_partition = partition_from_exposed_cases(full_mab, old_mab)
    mab = full_mab if mode == "full" else [c for c in full_mab if c["case_id"] in {x["case_id"] for x in old_mab}]
    tasks = defaultdict(list)
    for case in mab:
        tasks[case["task"]].append(case["case_id"])
    if strict_census and (len(tasks) != 8 or len(mab) != (800 if mode == "full" else 16)
                         or any(len(ids) != (100 if mode == "full" else 2) for ids in tasks.values())):
        raise ValueError("MAB census/strata differ from frozen 8-task expectation")
    if len({c["group_id"] for c in mab}) != 1:
        raise ValueError("MAB variants must retain one conservative family")
    shards = [{"shard_id": f"mab-{n:02d}", "dataset_key": DATASETS[0], "task": task,
               "case_ids": sorted(ids), "case_count": len(ids),
               "group_ids": sorted({c["group_id"] for c in mab if c["task"] == task})}
              for n, (task, ids) in enumerate(sorted(tasks.items()))]
    capacity = _capacity(DATASETS[0], full_mab)
    datasets = {DATASETS[0]: {"case_ids": [c["case_id"] for c in mab], "case_count": len(mab),
                            "independent_groups": 1, "development_partition": mab_partition,
                            "settings": capacity["settings"], "capacity_census": capacity}}
    if mode == "full":
        persona = cohorts[DATASETS[1]]
        grouped = plan_group_shards(persona, shard_count=4,
            expected_cases=589 if strict_census else None, expected_groups=20 if strict_census else None)
        for shard in grouped["shards"]:
            shards.append({"shard_id": f"persona-{shard['shard']:02d}", "dataset_key": DATASETS[1],
                           "case_ids": shard["case_ids"], "case_count": shard["case_count"],
                           "group_ids": shard["group_ids"]})
        capacity = _capacity(DATASETS[1], persona)
        datasets[DATASETS[1]] = {"case_ids": [c["case_id"] for c in persona], "case_count": len(persona),
                               "independent_groups": len({c["group_id"] for c in persona}),
                               "development_partition": development_partition(persona, []),
                               "settings": capacity["settings"], "capacity_census": capacity, "group_shard_plan": grouped}
    for dataset, metadata in datasets.items():
        planned = [cid for shard in shards if shard["dataset_key"] == dataset for cid in shard["case_ids"]]
        if len(planned) != len(set(planned)) or set(planned) != set(metadata["case_ids"]):
            raise ValueError("shards must cover every selected question exactly once")
    return {"mode": mode, "datasets": datasets, "shards": shards,
            "fixture_only": not strict_census, "all_cases": sum(d["case_count"] for d in datasets.values()),
            "all_arm_rows": sum(d["case_count"] for d in datasets.values()) * len(runner.ARMS)}


def _load_plan(suite_dir):
    plan = _read_json(Path(suite_dir) / "suite_plan.json")
    if plan.get("version") != VERSION or plan.get("suite_plan_sha256") != runner.stable_digest({k: v for k, v in plan.items() if k != "suite_plan_sha256"}):
        raise ValueError("suite plan identity mismatch")
    if plan.get("code_sha256") != _codes():
        raise ValueError("suite code changed; preserve old suite with its matching executable")
    if plan.get("fixture_only"):
        raise ValueError("fixture suite cannot execute public-data/model stages")
    return plan


def prepare_suite(data_dir, suite_dir, exposed_cases, *, mode="full"):
    runner.require_colab()
    suite_dir = Path(suite_dir)
    with runner.exclusive_directory(suite_dir):
        existing = _load_plan(suite_dir) if (suite_dir / "suite_plan.json").exists() else None
        if existing and existing["mode"] != mode:
            raise ValueError("existing suite mode differs; do not replace its frozen plan")
        exposed, old_identity = _exposure(exposed_cases)
        loaded, refs, identities = {}, {}, {}
        for dataset in (DATASETS if mode == "full" else DATASETS[:1]):
            paths = download_registered_dataset(dataset, data_dir)
            source = paths.get("data") or paths.get("questions")
            cases, references = load_dataset_cases(dataset, source, context_path=paths.get("contexts"))
            loaded[dataset], refs[dataset] = cases, {r["case_id"]: r for r in references}
            spec = dataset_spec(dataset)
            identities[dataset] = {"dataset_key": dataset, "data_revision": spec["data_revision"],
                "registry_entry_sha256": stable_hash(spec),
                "input_files": {role: {"name": p.name, "bytes": p.stat().st_size, "sha256": _sha(p)} for role, p in paths.items()},
                "generation_identity": _generation_identity(cases),
                "scoring_references_sha256": runner.stable_digest(references)}
        plan = {"version": VERSION, **plan_cohorts(loaded, exposed, mode=mode),
                "api_base": API_BASE, "source_identities": identities, "development_source": old_identity,
                "code_sha256": _codes(),
                "execution_policy": {"maximum_workers": 4, "maximum_rounds_per_shard": 10,
                                     "maximum_calls_per_batch": 60, "maximum_attempts_per_request": runner.MAX_ATTEMPTS,
                                     "inflight_policy": "stop shard, explicit recovery only", "cross_shard_deduplication": False},
                "evaluation_policy": {"bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_samples": BOOTSTRAP_SAMPLES,
                                      "primary": "official equal-history-group accuracy per dataset; all planned terminal failures retained",
                                      "clinical_claim": False, "confirmatory_claim": False,
                                      "boundary": "MAB one family; Persona20 connected groups; execution shards are not independent histories; question-held-out source histories remain shared/development exposed"}}
        plan["suite_plan_sha256"] = runner.stable_digest(plan)
        if existing and plan != existing:
            raise ValueError("new preparation differs from the frozen suite plan")
        if not existing:
            _atomic_json(suite_dir / "suite_plan.json", plan)
        # Every dataset identity, selected case and shard is frozen above before
        # preparation starts; execute refuses until all shards validate.
        for shard in plan["shards"]:
            dataset, shard_id = shard["dataset_key"], shard["shard_id"]
            path = suite_dir / "shards" / shard_id
            mapping = {c["case_id"]: c for c in loaded[dataset]}
            if not (path / "protocol.json").exists():
                started = time.monotonic()
                receipt = runner.prepare_run([mapping[cid] for cid in shard["case_ids"]],
                    [refs[dataset][cid] for cid in shard["case_ids"]], path, runner._settings(plan["datasets"][dataset]["settings"]), API_BASE,
                    {**identities[dataset], "suite_plan_sha256": plan["suite_plan_sha256"],
                     "suite_shard_id": shard_id, "development_partition": plan["datasets"][dataset]["development_partition"]})
                _atomic_json(suite_dir / "progress" / (shard_id + "-preparation.json"),
                             {"shard_id": shard_id, "prepared": True, "preparation_seconds": time.monotonic() - started,
                              "protocol_sha256": receipt["protocol_sha256"]})
            _validate_shard(path, shard, plan)
        result = {"suite_plan_sha256": plan["suite_plan_sha256"], "prepared_shards": len(plan["shards"]),
                  "planned_cases": plan["all_cases"], "planned_arm_rows": plan["all_arm_rows"], "status": "all_shards_prepared"}
        _atomic_json(suite_dir / "preparation_summary.json", result)
        return result


def _validate_shard(path, shard, plan):
    protocol, prompts, _, _ = runner.load_run(path)
    dataset = shard["dataset_key"]
    if protocol["case_ids"] != shard["case_ids"] or len(prompts) != shard["case_count"] * len(runner.ARMS):
        raise ValueError("shard cohort differs from frozen suite")
    manifest = protocol["dataset_manifest"]
    if manifest.get("suite_plan_sha256") != plan["suite_plan_sha256"] or manifest.get("suite_shard_id") != shard["shard_id"]:
        raise ValueError("shard suite identity mismatch")
    if any(manifest.get(k) != v for k, v in plan["source_identities"][dataset].items()) or manifest.get("development_partition") != plan["datasets"][dataset]["development_partition"]:
        raise ValueError("shard source/exposure metadata differ from frozen suite")
    if protocol["settings"] != plan["datasets"][dataset]["settings"] or protocol["api_base"] != plan["api_base"]:
        raise ValueError("shard settings differ from frozen suite")
    return protocol


def _drive_shard(shard_id, run_dir, *, max_rounds, max_calls, status_function, generate_function, emit):
    """Finite scheduling seam; callers supply real validated runner functions."""
    for round_index in range(max_rounds):
        state = status_function(run_dir)
        if state["complete"]:
            return {"shard_id": shard_id, "stop_reason": "complete", "state": state}
        if state["arm_outcomes"]["inflight"]:
            return {"shard_id": shard_id, "stop_reason": "unresolved_inflight", "state": state}
        state = generate_function(run_dir, max_calls=max_calls)
        emit({"shard_id": shard_id, "round": round_index + 1, "maximum_rounds": max_rounds,
              "state": state, "timestamp_unix": time.time()})
        if state["complete"]:
            return {"shard_id": shard_id, "stop_reason": "complete", "state": state}
        if state.get("actual_calls_this_invocation", 0) == 0:
            return {"shard_id": shard_id, "stop_reason": "no_progress", "state": state}
    return {"shard_id": shard_id, "stop_reason": "finite_round_cap", "state": status_function(run_dir)}


def execute_suite(suite_dir, *, workers=4, max_rounds=10, max_calls_per_batch=60):
    runner.require_colab()
    if any(type(v) is not int for v in (workers, max_rounds, max_calls_per_batch)) or not 1 <= workers <= 4 or not 1 <= max_rounds <= 10 or not 1 <= max_calls_per_batch <= 60:
        raise ValueError("choose workers1–4, rounds1–10, calls1–60")
    suite_dir = Path(suite_dir)
    with runner.exclusive_directory(suite_dir):
        plan = _load_plan(suite_dir)
        for shard in plan["shards"]:
            _validate_shard(suite_dir / "shards" / shard["shard_id"], shard, plan)
        def worker(shard):
            shard_id = shard["shard_id"]
            try:
                result = _drive_shard(shard_id, suite_dir / "shards" / shard_id,
                    max_rounds=max_rounds, max_calls=max_calls_per_batch,
                    status_function=runner.status_run, generate_function=runner.generate_run,
                    emit=lambda value: _atomic_json(suite_dir / "progress" / (shard_id + "-latest.json"), value))
            except Exception as error:
                # Provider/raw transport bodies, endpoints and Secret values are
                # never echoed into process output or failure receipts.
                result = {"shard_id": shard_id, "stop_reason": "safe_stage_error", "error_type": type(error).__name__}
            _atomic_json(suite_dir / "progress" / (shard_id + "-stop.json"), result)
            return result
        results = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(worker, shard) for shard in plan["shards"]]
            for future in as_completed(futures):
                results.append(future.result())
                _atomic_json(suite_dir / "execution_progress.json", {"suite_plan_sha256": plan["suite_plan_sha256"],
                             "finished_workers": len(results), "planned_shards": len(futures),
                             "results": sorted(results, key=lambda r: r["shard_id"])})
        receipt = {"suite_plan_sha256": plan["suite_plan_sha256"], "all_shards_complete": all(r["stop_reason"] == "complete" for r in results),
                   "workers": workers, "max_rounds": max_rounds, "max_calls_per_batch": max_calls_per_batch,
                   "results": sorted(results, key=lambda r: r["shard_id"])}
        _atomic_json(suite_dir / "execution_summary.json", receipt)
        return receipt


def supplemental_group_summary(rows):
    """Report micro AND whole-group weighting; never average shard percentages."""
    result = scoring._subset(rows)
    for arm, metrics in result.items():
        selected = [row for row in rows if row["arm"] == arm]
        for key, field in (("exact_alias_or_option", "exact_alias_or_option_score"), ("strict_committed", "strict_committed_score")):
            groups = defaultdict(list)
            for row in selected:
                groups[row["group_id"]].append(row["supplemental"][field])
            metrics[key + "_accuracy_group_equal"] = statistics.mean(statistics.mean(values) for values in groups.values())
    return result


def _cost_summary(summaries):
    costs = [s["cost"] for s in summaries]
    result = {key: sum(c.get(key, 0) for c in costs) for key in (
        "recorded_provider_attempt_intents", "intents_without_outcome", "successful_unique_provider_calls",
        "failed_provider_attempts", "failed_attempts_without_token_usage", "successful_calls_without_token_usage",
        "sum_successful_api_latency_seconds")}
    result["reported_tokens_including_failed_attempts_when_supplied"] = {
        key: sum(c.get("reported_tokens_including_failed_attempts_when_supplied", {}).get(key, 0) for c in costs)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
    result.update({"monetary_cost": None, "cross_shard_deduplication": False,
                   "boundary": "sum actual canonical attempts within every shard; identical prompts in separate shards are separate requests; no preparation-time or currency-price inference"})
    return result


def score_suite(suite_dir):
    suite_dir = Path(suite_dir)
    with runner.exclusive_directory(suite_dir):
        plan = _load_plan(suite_dir)
        summaries, all_rows, shard_states = [], [], []
        for shard in plan["shards"]:
            path = suite_dir / "shards" / shard["shard_id"]
            _validate_shard(path, shard, plan)
            with runner.exclusive_directory(path):
                summary = scoring._score_run_locked(path)
                summaries.append(summary)
                if summary["status"] == "complete":
                    rows = runner.read_jsonl(path / "objective_scores.jsonl")
                    if {(r["case_id"], r["arm"]) for r in rows} != {(cid, arm) for cid in shard["case_ids"] for arm in runner.ARMS} or len(rows) != shard["case_count"] * len(runner.ARMS):
                        raise ValueError("verified shard score grid differs from planned suite")
                    all_rows.extend(rows)
                shard_states.append({"shard_id": shard["shard_id"], "status": summary["status"], "completion_counts": summary["completion_counts"]})
        counts = Counter()
        for summary in summaries:
            counts.update(summary["completion_counts"])
        common = {"version": VERSION, "suite_plan_sha256": plan["suite_plan_sha256"], "mode": plan["mode"],
                  "planned_cases": plan["all_cases"], "planned_arm_rows": plan["all_arm_rows"],
                  "shards": shard_states, "completion_counts": dict(counts), "cost": _cost_summary(summaries),
                  "clinical_claim": False, "confirmatory_claim": False}
        if any(s["status"] != "complete" for s in shard_states):
            result = {**common, "status": "partial", "final_effectiveness_result": False,
                      "reason": "all planned cohorts must be terminal before suite-level scores; no completed-shard average"}
            (suite_dir / "suite_summary.json").unlink(missing_ok=True)
            (suite_dir / "suite_scores.jsonl").unlink(missing_ok=True)
            _atomic_json(suite_dir / "suite_progress.json", result)
            return result
        if len(all_rows) != plan["all_arm_rows"]:
            raise ValueError("full suite score denominator mismatch")
        datasets = {}
        for dataset, metadata in plan["datasets"].items():
            rows = [row for row in all_rows if row["dataset_key"] == dataset]
            if {r["case_id"] for r in rows} != set(metadata["case_ids"]):
                raise ValueError("full dataset score identities differ from plan")
            def summarize(part):
                return {**_summarize_subset(part, seed=BOOTSTRAP_SEED, bootstrap_samples=BOOTSTRAP_SAMPLES),
                        "supplemental_conditions": supplemental_group_summary(part)}
            exposure = metadata["development_partition"]
            partitions = {"development_exposed_questions": set(exposure["development_case_ids"]),
                          "question_heldout_shared_histories": set(exposure["question_heldout_case_ids"])}
            datasets[dataset] = {**summarize(rows),
                "by_task": {task: summarize([r for r in rows if r["task"] == task]) for task in sorted({r["task"] for r in rows})},
                "question_exposure_partitions": {name: summarize(part) for name, ids in partitions.items()
                                                  if (part := [r for r in rows if r["case_id"] in ids])},
                "independence_boundary": "whole original groups across all shards; MAB one family, Persona20 groups; question-held-out is not independent history or participant holdout"}
        result = {**common, "status": "complete", "final_effectiveness_result": True, "datasets": datasets,
                  "bootstrap_samples": BOOTSTRAP_SAMPLES, "bootstrap_seed": BOOTSTRAP_SEED,
                  "primary_boundary": "exploratory official answer metrics; strict/exact strings are secondary, no pooled cross-dataset or clinical score"}
        runner._replace_rows(suite_dir / "suite_scores.jsonl", all_rows)
        _atomic_json(suite_dir / "suite_summary.json", result)
        (suite_dir / "suite_progress.json").unlink(missing_ok=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--suite-dir", type=Path, required=True)
    p.add_argument("--exposed-cases", type=Path, required=True)
    p.add_argument("--mode", choices=("dev16", "full"), default="full")
    for name in ("execute", "score"):
        q = sub.add_parser(name)
        q.add_argument("--suite-dir", type=Path, required=True)
        if name == "execute":
            q.add_argument("--workers", type=int, default=4)
            q.add_argument("--max-rounds", type=int, default=10)
            q.add_argument("--max-calls-per-batch", type=int, default=60)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare_suite(args.data_dir, args.suite_dir, args.exposed_cases, mode=args.mode)
        elif args.command == "execute":
            result = execute_suite(args.suite_dir, workers=args.workers, max_rounds=args.max_rounds, max_calls_per_batch=args.max_calls_per_batch)
        else:
            result = score_suite(args.suite_dir)
    except Exception as error:
        print(json.dumps({"status": "safe_stage_error", "error_type": type(error).__name__,
                          "note": "No raw transport text or credentials displayed; frozen artifacts retained"}))
        raise SystemExit(1) from None
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
