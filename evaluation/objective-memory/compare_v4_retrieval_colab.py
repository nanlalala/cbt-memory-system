#!/usr/bin/env python3
"""Read-only Colab comparison of frozen v3 MAB retrieval and candidate v4.

This makes no model requests and never changes the old protocol, journal or
score. The controlled candidate-request file contains original benchmark text:
keep it in Colab. Only summary.json, per_request_observations.jsonl and
diagnostic_manifest.json are intended for derived-only cloud reporting.
Changed requests are selected from source/prompt identity before old correctness
metadata is loaded. A changed input is not a corrected answer or a causal label.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

VERSION = "mab-v4-source-edge-offline-diagnostic-v1"
OLD_COMMIT = "1511706b3a4b0ac71416998b5b4d752a13c4e313"
PLAN = "940a2b3d8fa02aa683f7855f94931ff1d9710c7835b598c70fc201582ae8d568"
RAG_ARMS = ("raw_history_rag", "managed_memory_rag")


def sha(body):
    return hashlib.sha256(body).hexdigest()


def require_colab():
    try:
        available = importlib.util.find_spec("google.colab") is not None
    except (ModuleNotFoundError, ValueError):
        available = False
    if not (Path("/content").is_dir() and os.environ.get("COLAB_RELEASE_TAG") and available):
        raise RuntimeError("Genuine hosted Google Colab required; no experiment data was read")


def regular_bytes(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected a regular artifact file")
    return path.read_bytes()


def read_json(path):
    return json.loads(regular_bytes(path))


def rows(path):
    return [json.loads(line) for line in regular_bytes(path).splitlines() if line.strip()]


def git_clean_at(repo, commit):
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("Pin an exact executable commit")
    if subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() != commit:
        raise ValueError("Checkout revision differs from pinned identity")
    if subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD"], check=False).returncode:
        raise ValueError("Pinned checkout has tracked modifications")


def channel_snapshot(case, arm, documents, managed, settings, ranker, retrieve, runner):
    """Pure same-budget MAB renderer; old prompt SHA must reproduce exactly."""
    short = documents[-settings.short_turns:]
    short_ids = {item.document_id for item in short}
    older = [item for item in (managed if arm == RAG_ARMS[1] else documents)
             if item.document_id not in short_ids]
    _, recent_ids, short_dropped = runner.fit_turns(list(reversed(short)), settings.short_chars)
    recent_set = set(recent_ids)
    short_text, recent_ids, _ = runner.fit_turns(
        [item for item in short if item.document_id in recent_set], settings.short_chars)
    recent_seeds = [item for item in short if item.document_id in set(recent_ids)]
    selected, trace = retrieve(case["query"], older, ranker,
                               scope_id=case["subject_id"], seed_documents=recent_seeds)
    long_text, long_ids, long_dropped = runner.fit_turns(selected, settings.long_chars)
    messages = runner._messages(case, short_text, long_text)
    if sum(len(message["content"]) for message in messages) > settings.max_context_chars:
        raise ValueError("Candidate prompt exceeds the unchanged total character budget")
    return {"messages": messages, "prompt_sha256": runner.stable_digest(messages),
            "selected_ids": [item.document_id for item in selected],
            "included_short_ids": recent_ids, "included_long_ids": long_ids,
            "budget_dropped_short_ids": short_dropped, "budget_dropped_long_ids": long_dropped,
            "retrieval": trace}


def source_histories(state, runner):
    """Recover exact admitted MAB rows from the original frozen source ledger."""
    histories = defaultdict(list)
    for item in state["frozen_rows"]["inventory.jsonl"]:
        if item["storage_decision"].get("action") != "use_as_history":
            raise ValueError("Unexpected original source-store eligibility")
        histories[item["subject_id"]].append(item["source"])
    for subject, history in histories.items():
        history.sort(key=lambda r: (r["order"], r["session_id"], r["turn_id"]))
        if len({(r["session_id"], r["turn_id"]) for r in history}) != len(history):
            raise ValueError("Ambiguous original source inventory")
    result = []
    for metadata in state["frozen_rows"]["cases.jsonl"]:
        if metadata.get("dataset_key") != "memoryagentbench_conflict":
            raise ValueError("This diagnostic only accepts original MAB cases")
        history = histories[metadata["subject_id"]]
        if runner.stable_digest([metadata["subject_id"], history]) != metadata["history_inventory_sha256"]:
            raise ValueError("Source inventory does not reconstruct the original admitted history")
        case = {key: value for key, value in metadata.items() if key != "history_inventory_sha256"}
        case["history"] = history
        runner._validate_case(case)
        admitted, excluded = runner._eligible(case)
        if excluded or admitted != history:
            raise ValueError("MAB reconstructed eligibility differs from original inventory")
        result.append(case)
    return result


def write_rows(path, values):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in values))


def validate_import_origins(repo, code_paths):
    expected = defaultdict(set)
    for relative in code_paths:
        path = repo / relative
        if path.suffix == ".py":
            expected[path.stem].add(path.resolve())
    for name, module in list(sys.modules.items()):
        choices = expected.get(name.rsplit(".", 1)[-1])
        if choices and getattr(module, "__file__", None) and Path(module.__file__).resolve() not in choices:
            raise ValueError("A shared old-code dependency was imported from another checkout")


def validate_shard_binding(protocol, prompts, shard, plan, arms):
    dataset = shard["dataset_key"]
    if protocol["case_ids"] != shard["case_ids"] or len(prompts) != shard["case_count"] * len(arms):
        raise ValueError("Original shard cohort differs from the full suite")
    manifest = protocol["dataset_manifest"]
    if manifest.get("suite_plan_sha256") != plan["suite_plan_sha256"] or manifest.get("suite_shard_id") != shard["shard_id"]:
        raise ValueError("Original shard suite identity differs")
    if any(manifest.get(key) != value for key, value in plan["source_identities"][dataset].items()) or manifest.get("development_partition") != plan["datasets"][dataset]["development_partition"]:
        raise ValueError("Original shard source or exposure metadata differs")
    if protocol["settings"] != plan["datasets"][dataset]["settings"] or protocol["api_base"] != plan["api_base"]:
        raise ValueError("Original shard settings differ")


def validated_old_score_rows(state, saved_rows, scorer):
    """Scoring-only phase after source-only changed-input selection is frozen."""
    expected = set(state["outcomes"])
    saved = {(row["case_id"], row["arm"]): row for row in saved_rows}
    if len(saved) != len(saved_rows) or set(saved) != expected:
        raise ValueError("Original score metadata grid differs from validated source requests")
    references = {row["case_id"]: row for row in state["references"]}
    result = {}
    for key in sorted(expected):
        response = state["completed"][key]
        actual = scorer.deterministic_score(response["hypothesis"], references[key[0]])[0]
        row = saved[key]
        if row.get("available") is not True or row.get("completion_status") != "success" or row.get("score") != actual:
            raise ValueError("Original score metadata differs from read-only deterministic recomputation")
        result[key] = actual
    return result


def compare(args):
    require_colab()
    raw_roots = [Path(value).expanduser() for value in (args.old_repo, args.candidate_repo, args.suite, args.out)]
    if any(path.is_symlink() for path in raw_roots):
        raise ValueError("Symlinked root rejected")
    old_repo, new_repo, suite, out = [path.resolve() for path in raw_roots]
    if any(not path.is_relative_to(Path("/content")) for path in (old_repo, new_repo, suite, out)):
        raise ValueError("Source work must remain within genuine Colab")
    if old_repo == new_repo or out.exists() or out.is_relative_to(suite) or out.is_relative_to(old_repo) or out.is_relative_to(new_repo):
        raise ValueError("Use separate pinned checkouts and a fresh output outside original data/code")
    git_clean_at(old_repo, OLD_COMMIT)
    git_clean_at(new_repo, args.candidate_commit)
    candidate_path = new_repo / "evaluation/objective-memory/v4_source_retrieval.py"
    candidate_bytes = regular_bytes(candidate_path)
    pinned_bytes = subprocess.check_output(["git", "-C", str(new_repo), "show", args.candidate_commit + ":evaluation/objective-memory/v4_source_retrieval.py"])
    if candidate_bytes != pinned_bytes:
        raise ValueError("Candidate module differs from its tracked pinned bytes")
    pinned_helper = subprocess.check_output(["git", "-C", str(new_repo), "show", args.candidate_commit + ":evaluation/objective-memory/compare_v4_retrieval_colab.py"])
    if regular_bytes(Path(__file__).resolve()) != pinned_helper:
        raise ValueError("Diagnostic helper differs from its tracked pinned bytes")
    old_modules = old_repo / "evaluation/objective-memory"
    sys.path.insert(0, str(old_modules))
    runner = importlib.import_module("v3_run_objective_comparison")
    driver = importlib.import_module("v3_suite_driver")
    legacy = importlib.import_module("v3_source_retrieval")
    versions = importlib.import_module("v3_fact_versions")
    scorer = importlib.import_module("score_objective_comparison")
    for module in (runner, driver, legacy, versions, scorer):
        if not Path(module.__file__).resolve().is_relative_to(old_modules):
            raise ValueError("Old-code import origin differs; do not unload a foreign kernel module")
    validate_import_origins(old_repo, runner._code_hashes())
    spec = importlib.util.spec_from_file_location("candidate_v4_source_retrieval", candidate_path)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    # The candidate intentionally changes only source-edge gating. Shared fact
    # grammar, ranker and renderer remain the old, byte-verified dependencies.
    if candidate.parse_assertion is not versions.parse_assertion or candidate.HistoryRanker is not runner.HistoryRanker:
        raise ValueError("Candidate shared parser/ranker dependency identity differs")
    observations, requests, scoring_snapshots, source_hashes = [], [], [], {}
    changed_case_ids, stalled_case_ids = set(), set()
    with ExitStack() as locks:
        locks.enter_context(runner._readonly_existing_lock(suite))
        plan = driver._load_plan(suite)
        if plan.get("suite_plan_sha256") != PLAN or plan.get("all_cases") != 1389 or plan.get("all_arm_rows") != 4167:
            raise ValueError("Original full-suite identity differs")
        summary = read_json(suite / "suite_summary.json")
        if summary.get("status") != "complete" or summary.get("completion_counts") != {"success": 4167}:
            raise ValueError("Original full suite is not successfully terminal")
        shards = [shard for shard in plan["shards"] if shard["dataset_key"] == "memoryagentbench_conflict"]
        if len(shards) != 8 or sum(shard["case_count"] for shard in shards) != 800:
            raise ValueError("Original MAB census differs")
        for shard in shards:
            path = suite / "shards" / shard["shard_id"]
            if path.is_symlink():
                raise ValueError("Symlinked shard rejected")
            locks.enter_context(runner._readonly_existing_lock(path))
            state = runner.read_run_state_readonly(path)
            protocol = state["protocol"]
            validate_shard_binding(protocol, state["prompts"], shard, plan, runner.ARMS)
            validate_import_origins(old_repo, protocol["code_sha256"])
            if any(value["status"] != "success" for value in state["outcomes"].values()):
                raise ValueError("Original shard has unresolved/failed rows")
            for name in ("protocol.json", *runner.FROZEN_ROWS, "objective_scores.jsonl"):
                source_hashes[str((path / name).relative_to(suite))] = sha(regular_bytes(path / name))
            settings = runner._settings(protocol["settings"])
            if settings.backend != "lexical":
                raise ValueError("The frozen full MAB diagnostic expects its original lexical backend")
            original_prompts = {(p["case_id"], p["arm"]): p for p in state["prompts"]}
            old_ranker, new_ranker = runner.HistoryRanker(settings), runner.HistoryRanker(settings)
            history_views = {}
            for case in source_histories(state, runner):
                subject = case["subject_id"]
                if subject not in history_views:
                    admitted, _ = runner._eligible(case)
                    current, _ = versions.consolidate_rows(admitted)
                    history_views[subject] = ([runner._document(case, row) for row in admitted],
                                              [runner._document(case, row) for row in current])
                documents, managed = history_views[subject]
                for arm in RAG_ARMS:
                    key = (case["case_id"], arm)
                    old = channel_snapshot(case, arm, documents, managed, settings, old_ranker, legacy.retrieve_sources, runner)
                    frozen = original_prompts[key]
                    if old["prompt_sha256"] != frozen["prompt_sha256"] or old["messages"] != frozen["messages"]:
                        raise ValueError("Old prompt replay does not reproduce original frozen bytes")
                    new = channel_snapshot(case, arm, documents, managed, settings, new_ranker, candidate.retrieve_sources, runner)
                    changed = old["prompt_sha256"] != new["prompt_sha256"]
                    stalled = [round["round"] for round in old["retrieval"]["rounds"]
                               if round.get("candidate_count", 0) > 0 and not round.get("selected_ids")]
                    if changed:
                        changed_case_ids.add(case["case_id"])
                        requests.append({"case_id": case["case_id"], "arm": arm,
                                         "messages": new["messages"], "old_prompt_sha256": old["prompt_sha256"],
                                         "prompt_sha256": new["prompt_sha256"], "settings": protocol["settings"],
                                         "source_protocol_sha256": protocol["protocol_sha256"],
                                         "source_grammar": versions.PARSER_VERSION,
                                         "candidate_policy": candidate.VERSION})
                    if stalled:
                        stalled_case_ids.add(case["case_id"])
                    observations.append({"shard_id": shard["shard_id"], "case_id": case["case_id"],
                                         "arm": arm, "task": case["task"], "prompt_changed": changed,
                                         "old_prompt_sha256": old["prompt_sha256"], "new_prompt_sha256": new["prompt_sha256"],
                                         "old_selected_ids": old["selected_ids"], "new_selected_ids": new["selected_ids"],
                                         "old_included_long_ids": old["included_long_ids"], "new_included_long_ids": new["included_long_ids"],
                                         "old_budget_dropped_long_ids": old["budget_dropped_long_ids"], "new_budget_dropped_long_ids": new["budget_dropped_long_ids"],
                                         "old_nonempty_neighbor_rounds_with_no_admission": stalled,
                                         "stage_attribution": "unresolved_without_independent_evidence_annotations"})
            scoring_snapshots.append((state, path / "objective_scores.jsonl"))
        # Freeze the full source-only selection BEFORE looking at old scores.
        requests.sort(key=lambda row: (row["case_id"], row["arm"]))
        selection_identity = sha(json.dumps([[row["case_id"], row["arm"], row["prompt_sha256"]] for row in requests], sort_keys=True).encode())
        old_scores = {}
        for state, path in scoring_snapshots:
            for key, value in validated_old_score_rows(state, rows(path), scorer).items():
                if key in old_scores:
                    raise ValueError("Duplicate old per-question score")
                old_scores[key] = value
        for row in observations:
            row["old_official_score"] = old_scores[(row["case_id"], row["arm"])]
    if len(observations) != 1600 or len({row["case_id"] for row in observations}) != 800:
        raise ValueError("Complete MAB two-RAG-arm diagnostic grid differs")
    result = {"schema": VERSION, "old_executable": OLD_COMMIT, "old_suite_plan_sha256": PLAN,
              "candidate_commit": args.candidate_commit, "candidate_policy": candidate.policy_descriptor(),
              "candidate_module_sha256": sha(candidate_bytes), "diagnostic_helper_sha256": sha(Path(__file__).read_bytes()),
              "cases": 800, "compared_rag_arm_rows": 1600,
              "changed_cases": len(changed_case_ids), "changed_arm_rows": len(requests),
              "changed_case_ids": sorted(changed_case_ids), "observed_empty_graph_admission_cases": len(stalled_case_ids),
              "observed_empty_graph_admission_case_ids": sorted(stalled_case_ids),
              "old_officially_incorrect_changed_arm_rows": sum(row["prompt_changed"] and not row["old_official_score"] for row in observations),
              "selection_policy": "ALL changed prompt pairs from all 800 MAB cases; selection frozen before old correctness lookup",
              "selected_request_identity_sha256": selection_identity,
              "new_provider_calls": 0, "new_effectiveness_result": False,
              "causal_error_attribution": "unresolved without independent evidence annotations; changed input is not a fixed answer",
              "old_runs_modified": False, "clinical_claim": False,
              "full_source_processing_location": "genuine hosted Google Colab"}
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=out.name + ".tmp-", dir=out.parent))
    try:
        (staging / "summary.json").write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        write_rows(staging / "per_request_observations.jsonl", observations)
        write_rows(staging / "controlled_candidate_requests.jsonl", requests)
        manifest = {"schema": VERSION, "source_sha256": source_hashes,
                    "controlled_do_not_export": ["controlled_candidate_requests.jsonl"],
                    "exportable_derived_only": ["summary.json", "per_request_observations.jsonl", "diagnostic_manifest.json"],
                    "output_sha256": {p.name: sha(p.read_bytes()) for p in staging.iterdir() if p.is_file()}}
        (staging / "diagnostic_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        staging.rename(out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-repo", required=True)
    parser.add_argument("--candidate-repo", required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--out", required=True)
    print(json.dumps(compare(parser.parse_args()), ensure_ascii=False, sort_keys=True))
