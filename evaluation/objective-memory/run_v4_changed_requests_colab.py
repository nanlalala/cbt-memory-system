#!/usr/bin/env python3
"""Finite Colab-only development follow-up to the frozen v4 input diagnostic.

All 206 changed logical requests were selected from prompt identity before old
score lookup. Unchanged original replies are explicitly reused; changed replies
receive at most one new canonical attempt. Never run this against the old output
directory. No credential or raw provider error is persisted or displayed.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
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
from threading import Lock
import time

VERSION = "mab-v4-changed-input-development-response-v1"
CANDIDATE = "efd75116f985749a96a6070cb80c4b38d5c94dba"
OLD_EXECUTABLE = "1511706b3a4b0ac71416998b5b4d752a13c4e313"
PLAN = "940a2b3d8fa02aa683f7855f94931ff1d9710c7835b598c70fc201582ae8d568"
OLD_PLAN_FILE_SHA = "dbd41e4ea0fdf0fff3d357ad05ef9d72e546a5594481db518a31e93a13ea2f2e"
OLD_SUMMARY_FILE_SHA = "97a58e6000bb3d3ceaf8649487b412b02653d2eb114d62ae1fd02a32d6dd93aa"
OLD_METADATA_FILE_SHA = "9837a6825337898a441002abccde583417188f4d7edf4317869c6fe7c0962e36"
SELECTION = "9d3ee5c84ec55ee970780ffcfab19dd3f693a521adf8b37e2e9f9a38e7902e9f"
MODEL = "deepseek-v4.1-flash"
ALIAS = "deepseek-v4-1-flash-260910"
API = "https://tokendance.space/gateway/v1"
ARMS = ("short_term", "raw_history_rag", "managed_memory_rag")
TOKENS = ("prompt_tokens", "completion_tokens", "total_tokens")
MAX_ATTEMPTS = 206
MAX_OUTPUT = 128
CODE_FILES = ("evaluation/dialogue-benefit/followup_transport.py", "app/colab_provider.py",
              "evaluation/end-to-end-dialogue/run_three_arm.py", "evaluation/end-to-end-dialogue/execution_lock.py",
              "evaluation/objective-memory/score_objective_comparison.py",
              "evaluation/objective-memory/v3_response_diagnostics.py",
              "evaluation/objective-memory/v4_source_retrieval.py",
              "evaluation/objective-memory/compare_v4_retrieval_colab.py")


def stable(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def legacy_stable(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def data(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected regular source artifact")
    return path.read_bytes()


def digest(path):
    return hashlib.sha256(data(path)).hexdigest()


def read_json(path):
    return json.loads(data(path))


def rows(path, optional=False):
    if optional and not Path(path).exists():
        return []
    body = data(path)
    if body and not body.endswith(b"\n"):
        raise ValueError("Incomplete journal tail; no automatic repair or retry")
    return [json.loads(line) for line in body.splitlines() if line.strip()]


def atomic_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


def atomic_rows(path, values):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as stream:
        for row in values:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


def append(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def seal(value):
    return {**value, "record_sha256": stable(value)}


def verify_sealed(row):
    if row.get("record_sha256") != stable({k: v for k, v in row.items() if k != "record_sha256"}):
        raise ValueError("Journal record integrity differs")


def require_colab():
    try:
        present = importlib.util.find_spec("google.colab") is not None
    except (ModuleNotFoundError, ValueError):
        present = False
    if not (Path("/content").is_dir() and os.environ.get("COLAB_RELEASE_TAG") and present):
        raise RuntimeError("Genuine hosted Google Colab required before source or credential access")


@contextmanager
def source_lock(path):
    path = Path(path)
    if path.is_symlink() or (path / ".execution.lock").is_symlink():
        raise ValueError("Symlinked source lock")
    with (path / ".execution.lock").open("rb") as stream:
        fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def output_lock(path):
    lock = path / ".execution.lock"
    if lock.is_symlink():
        raise ValueError("Symlinked output lock")
    with lock.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def checked_module(name, path):
    if name in sys.modules:
        old = sys.modules[name]
        if Path(old.__file__).resolve() != path.resolve():
            raise ValueError("Foreign cached import; use a fresh subprocess")
        return old
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def numeric_usage(value):
    return {k: value[k] for k in TOKENS if isinstance(value, dict) and type(value.get(k)) is int and value[k] >= 0}


def verify_response(value):
    if (not isinstance(value, dict) or value.get("provider_model") != ALIAS or value.get("finish_reason") != "stop"
            or not isinstance(value.get("hypothesis"), str) or not value["hypothesis"].strip()):
        raise ValueError("Model identity or response integrity failure")
    latency = value.get("latency_seconds")
    if type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0 or not isinstance(value.get("usage"), dict):
        raise ValueError("Response latency or usage shape differs")
    return {"hypothesis": value["hypothesis"], "provider_model": ALIAS, "finish_reason": "stop",
            "latency_seconds": latency, "usage": numeric_usage(value["usage"])}


def request_body(messages):
    if not isinstance(messages, list) or not messages or any(set(row) != {"role", "content"} or row["role"] not in {"system", "user", "assistant"} or not isinstance(row["content"], str) for row in messages):
        raise ValueError("Unexpected request message structure")
    if sum(len(row["content"]) for row in messages) > 24000:
        raise ValueError("Request exceeds unchanged character ceiling")
    return {"messages": messages, "model": MODEL, "allowed_model": ALIAS, "api_base": API,
            "temperature": 0, "thinking_mode": "disabled", "max_response_tokens": MAX_OUTPUT}


def verify_old_settings(protocol, request_settings):
    settings = protocol["settings"]
    if (settings != request_settings or settings.get("model") != MODEL or settings.get("accepted_provider_models") != [ALIAS]
            or settings.get("max_response_tokens") != MAX_OUTPUT or settings.get("thinking_mode") != "disabled"
            or protocol.get("temperature") != 0 or protocol.get("api_base") != API):
        raise ValueError("Old and candidate generation settings differ")


def load_old_shard(path, shard, plan, source_hashes, code):
    protocol = read_json(path / "protocol.json")
    if protocol.get("protocol_sha256") != legacy_stable({k: v for k, v in protocol.items() if k != "protocol_sha256"}):
        raise ValueError("Original protocol hash differs")
    manifest = protocol["dataset_manifest"]
    dataset = shard["dataset_key"]
    if protocol["case_ids"] != shard["case_ids"] or manifest.get("suite_shard_id") != shard["shard_id"] or manifest.get("suite_plan_sha256") != PLAN:
        raise ValueError("Original shard identity differs")
    if any(manifest.get(k) != v for k, v in plan["source_identities"][dataset].items()) or manifest.get("development_partition") != plan["datasets"][dataset]["development_partition"]:
        raise ValueError("Original source or development partition differs")
    if protocol["settings"] != plan["datasets"][dataset]["settings"] or protocol["api_base"] != plan["api_base"]:
        raise ValueError("Original settings differ from suite")
    for relative, expected in protocol["code_sha256"].items():
        if relative in code and code[relative] != expected:
            raise ValueError("Shared candidate transport/scorer bytes differ from old frozen code")
    for name, expected in protocol["file_sha256"].items():
        source_hashes[path / name] = digest(path / name)
        if source_hashes[path / name] != expected:
            raise ValueError("Original source artifact differs")
    prompts = rows(path / "prompts.jsonl")
    prompt_map, canonical = {}, {}
    for row in prompts:
        key = (row["case_id"], row["arm"])
        if key in prompt_map or row["prompt_sha256"] != legacy_stable(row["messages"]) or row["canonical_request_key"] != row["prompt_sha256"]:
            raise ValueError("Original prompt grid or hash differs")
        prompt_map[key] = row
        canonical.setdefault(row["canonical_request_key"], row)
    expected_pairs = {(case, arm) for case in protocol["case_ids"] for arm in ARMS}
    if set(prompt_map) != expected_pairs:
        raise ValueError("Original full three-arm prompt grid differs")
    journals = {name: rows(path / name, optional=name == "failures.jsonl") for name in ("response_attempts.jsonl", "response_outcomes.jsonl", "responses.jsonl", "failures.jsonl")}
    for name in journals:
        if (path / name).exists():
            source_hashes[path / name] = digest(path / name)
    if journals["failures.jsonl"]:
        raise ValueError("The verified original run unexpectedly has failures")
    indexed = {}
    for name in ("response_attempts.jsonl", "response_outcomes.jsonl", "responses.jsonl"):
        by_key = {}
        for row in journals[name]:
            key = row.get("canonical_request_key")
            if key not in canonical or key in by_key or row.get("attempt_index") != 1 or any(row.get(k) != v for k, v in {"request_key": key, "protocol_sha256": protocol["protocol_sha256"], "prompt_sha256": key}.items()):
                raise ValueError("Original canonical attempt/response identity differs")
            by_key[key] = row
        if set(by_key) != set(canonical):
            raise ValueError("Original canonical journal is incomplete")
        indexed[name] = by_key
    responses = {}
    for key, row in indexed["responses.jsonl"].items():
        if indexed["response_outcomes.jsonl"][key].get("status") != "success" or row.get("response_sha256") != legacy_stable({k: v for k, v in row.items() if k != "response_sha256"}):
            raise ValueError("Original successful response integrity differs")
        verify_response(row)
        responses[key] = row
    source_hashes[path / "protocol.json"] = digest(path / "protocol.json")
    source_hashes[path / "objective_scores.jsonl"] = digest(path / "objective_scores.jsonl")
    return {"protocol": protocol, "prompts": prompt_map, "responses": responses,
            "reference_path": path / "references.jsonl", "score_path": path / "objective_scores.jsonl"}


class Journal:
    def __init__(self, run, protocol, requests, checker, *, max_new_calls=206, transport=None):
        self.run, self.protocol, self.requests, self.checker = Path(run), protocol, requests, checker
        if transport is not None and (not protocol.get("fixture_only") or len(requests) > 4):
            raise ValueError("Injected transport is limited to tiny explicit software fixtures")
        self.transport, self.max_new_calls, self.new_calls = transport, max_new_calls, 0
        self.mutex, self.stopped = Lock(), False
        self.attempts, self.outcomes = {}, {}
        for target, name in ((self.attempts, "attempts.jsonl"), (self.outcomes, "outcomes.jsonl")):
            for row in rows(self.run / name, optional=True):
                verify_sealed(row)
                key = row.get("request_key")
                if key not in requests or key in target or row.get("protocol_sha256") != protocol["protocol_sha256"]:
                    raise ValueError("New journal identity differs")
                if name == "attempts.jsonl" and row.get("attempt_index") != 1:
                    raise ValueError("Only one new attempt per canonical request is permitted")
                target[key] = row
        if set(self.outcomes) - set(self.attempts) or set(self.attempts) - set(self.outcomes):
            raise RuntimeError("Unresolved attempt blocks all new calls; no automatic retry")
        if len(self.attempts) > protocol["maximum_provider_attempts"]:
            raise ValueError("New journal exceeds finite budget")
        for outcome in self.outcomes.values():
            if outcome["status"] == "success":
                verify_response(outcome["response"])
            elif outcome["status"] == "protocol_validation_failure":
                raise RuntimeError("Previous identity/validation failure blocks continuation")
            elif outcome["status"] != "provider_failure":
                raise ValueError("Unknown provider outcome")

    def call(self, key):
        with self.mutex:
            if key in self.outcomes or self.stopped:
                return
            if key in self.attempts:
                raise RuntimeError("Canonical request is already in flight; never duplicate")
            if self.new_calls >= self.max_new_calls:
                return
            if len(self.attempts) >= self.protocol["maximum_provider_attempts"]:
                raise RuntimeError("Finite provider budget exhausted")
            self.checker()
            if self.transport is None:
                require_colab()
                if not os.environ.get("CBT_EVAL_API_KEY"):
                    raise RuntimeError("Colab credential unavailable before provider intent")
            intent = seal({"protocol_sha256": self.protocol["protocol_sha256"], "request_key": key,
                           "attempt_index": 1, "started_at_unix": time.time()})
            append(self.run / "attempts.jsonl", intent)
            self.attempts[key] = intent; self.new_calls += 1
        started = time.monotonic()
        response, fatal = None, False
        try:
            req = self.requests[key]
            if self.transport is not None:
                response = self.transport(req["messages"])
            else:
                response = self.provider.generate_response(req["messages"], model=MODEL, allowed_model=ALIAS,
                                                          api_base=API, key=os.environ["CBT_EVAL_API_KEY"], max_response_tokens=MAX_OUTPUT)
            response = verify_response(response)
            outcome = {"protocol_sha256": self.protocol["protocol_sha256"], "request_key": key,
                       "status": "success", "response": response}
        except Exception as error:
            fatal = isinstance(error, (ValueError, TypeError, KeyError)) or bool(getattr(error, "fatal", False))
            supplied = getattr(error, "reported_usage", None)
            if supplied is None and isinstance(response, dict):
                supplied = response.get("usage")
            usage = numeric_usage(supplied)
            outcome = {"protocol_sha256": self.protocol["protocol_sha256"], "request_key": key,
                       "status": "protocol_validation_failure" if fatal else "provider_failure",
                       "error_type": type(error).__name__, "reported_usage": usage,
                       "billable_usage_unknown_or_partial": len(usage) != 3,
                       "latency_seconds": time.monotonic() - started, "retry_sent": False}
        with self.mutex:
            record = seal(outcome)
            append(self.run / "outcomes.jsonl", record)
            self.outcomes[key] = record
            self.stopped = self.stopped or fatal

    def revalidate(self):
        for filename, memory in (("attempts.jsonl", self.attempts), ("outcomes.jsonl", self.outcomes)):
            loaded = {}
            for row in rows(self.run / filename, optional=True):
                verify_sealed(row)
                key = row["request_key"]
                if key in loaded:
                    raise ValueError("Duplicate persisted canonical journal key")
                loaded[key] = row
            if loaded != memory:
                raise ValueError("Persisted journal differs from validated in-memory execution")
        if set(self.attempts) != set(self.outcomes):
            raise RuntimeError("Provider attempt remains unresolved; no final result")

    def cost(self):
        successful = [row["response"] for row in self.outcomes.values() if row["status"] == "success"]
        failed = [row for row in self.outcomes.values() if row["status"] != "success"]
        latency = [row["latency_seconds"] for row in successful]
        return {"recorded_provider_attempts": len(self.attempts), "successful_canonical_calls": len(successful),
                "failed_attempts": len(failed), "unresolved_attempts": len(set(self.attempts) - set(self.outcomes)),
                "reported_tokens": {k: sum(row["usage"].get(k, 0) for row in successful) + sum(row.get("reported_usage", {}).get(k, 0) for row in failed) for k in TOKENS},
                "successful_calls_missing_usage_fields": {k: sum(k not in row["usage"] for row in successful) for k in TOKENS},
                "failed_attempts_missing_usage_fields": {k: sum(k not in row.get("reported_usage", {}) for row in failed) for k in TOKENS},
                "reported_provider_models": dict(Counter(row["provider_model"] for row in successful)),
                "median_successful_api_latency_seconds": statistics.median(latency) if latency else None,
                "sum_successful_api_latency_seconds": sum(latency), "monetary_cost": None,
                "boundary": "Only new canonical attempts; old reused replies are not billed again. Missing usage is unknown, not zero cost. API time is not wall time."}


def aggregate(score_rows):
    conditions = {}
    for arm in ARMS:
        chosen = [r for r in score_rows if r["arm"] == arm]
        conditions[arm] = {"questions": len(chosen), "official_correct": sum(r["score"] for r in chosen),
                           "official_accuracy": sum(r["score"] for r in chosen) / len(chosen),
                           "strict_format_correct": sum(r["strict_committed_score"] for r in chosen),
                           "new_response_rows": sum(r["provenance"] == "new_v4_response" for r in chosen),
                           "new_failure_rows": sum(r["completion_status"] == "provider_failure" for r in chosen),
                           "reused_old_response_rows": sum(r["provenance"] == "unchanged_prompt_reused_old_response" for r in chosen)}
    cases = defaultdict(dict)
    for row in score_rows:
        cases[row["case_id"]][row["arm"]] = row
    deltas = [value["managed_memory_rag"]["score"] - value["raw_history_rag"]["score"] for value in cases.values()]
    primary = {"accuracy_delta": sum(deltas) / len(deltas), "wins": sum(d > 0 for d in deltas),
               "ties": sum(d == 0 for d in deltas), "losses": sum(d < 0 for d in deltas),
               "independent_families": 1, "confidence_interval": None, "interval_status": "not_estimable_one_family"}
    return {"conditions": conditions, "managed_minus_raw": primary}


def score_complete(shards, pair_map, outcomes, scorer, diagnostic):
    output, controlled = [], []
    for shard in shards:
        refs = {row["case_id"]: row for row in rows(shard["reference_path"])}
        baseline = {(row["case_id"], row["arm"]): row for row in rows(shard["score_path"])}
        if len(baseline) != len(shard["prompts"]) or set(baseline) != set(shard["prompts"]):
            raise ValueError("Original score grid differs")
        for key, prompt in shard["prompts"].items():
            old_response = shard["responses"][prompt["canonical_request_key"]]
            reference = refs[key[0]]
            old_score = scorer.deterministic_score(old_response["hypothesis"], reference)[0]
            if baseline[key]["score"] != old_score:
                raise ValueError("Original score disagrees with independently recomputed source result")
            changed = key in pair_map
            if changed:
                outcome = outcomes[pair_map[key]["request_key"]]
                response = outcome.get("response") if outcome["status"] == "success" else None
                provenance = "new_v4_response" if response else "new_v4_terminal_failure"
                completion = "success" if response else "provider_failure"
            else:
                response, provenance, completion = old_response, "unchanged_prompt_reused_old_response", "success"
            answer = response["hypothesis"] if response else None
            score = scorer.deterministic_score(answer, reference)[0] if answer is not None else 0.0
            supplemental = diagnostic.score_supplemental(answer, reference) if answer is not None else {"strict_committed_score": 0.0, "exact_alias_or_option_score": 0.0}
            row = {"case_id": key[0], "arm": key[1], "dataset_key": "memoryagentbench_conflict", "task": prompt["task"],
                   "group_id": prompt["group_id"], "score": score, "old_v3_score": old_score,
                   "strict_committed_score": supplemental["strict_committed_score"],
                   "exact_alias_or_option_score": supplemental["exact_alias_or_option_score"],
                   "available": response is not None, "completion_status": completion, "provenance": provenance,
                   "prompt_changed": changed, "old_prompt_sha256": prompt["prompt_sha256"],
                   "prompt_sha256": pair_map[key]["prompt_sha256"] if changed else prompt["prompt_sha256"],
                   "new_request_key": pair_map[key]["request_key"] if changed else None,
                   "response_text_sha256": hashlib.sha256(answer.encode()).hexdigest() if answer is not None else None,
                   "error_cause": "none" if score else "provider_technical_failure" if response is None else "answer_incorrect_stage_unresolved"}
            output.append(row)
            controlled.append({**row, "original_hypothesis": answer, "provider_model": response.get("provider_model") if response else None})
    if len(output) != 2400 or len({r["case_id"] for r in output}) != 800:
        raise ValueError("Full 800-case three-arm counterfactual grid differs")
    return output, controlled


def run(args):
    require_colab()
    raw_paths = [Path(value).expanduser() for value in (args.candidate_repo, args.diagnostic_dir, args.old_suite, args.out)]
    if any(path.is_symlink() for path in raw_paths):
        raise ValueError("Symlinked roots rejected")
    repo, diag_dir, old_suite, out = [path.resolve() for path in raw_paths]
    if any(not path.is_relative_to(Path("/content")) for path in (repo, diag_dir, old_suite, out)) or out.is_relative_to(repo) or out.is_relative_to(old_suite) or out.is_relative_to(diag_dir):
        raise ValueError("Use separate controlled Colab paths")
    if args.candidate_commit != CANDIDATE or subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() != CANDIDATE or subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD"]).returncode:
        raise ValueError("Candidate executable checkout differs")
    if type(args.workers) is not int or not 1 <= args.workers <= 4 or not 1 <= args.max_new_calls <= MAX_ATTEMPTS:
        raise ValueError("Finite worker/call limits differ")
    code = {relative: digest(repo / relative) for relative in CODE_FILES}
    helper_hash = digest(Path(__file__))
    provider = checked_module("followup_transport", repo / "evaluation/dialogue-benefit/followup_transport.py")
    scorer = checked_module("score_objective_comparison", repo / "evaluation/objective-memory/score_objective_comparison.py")
    diagnostic = checked_module("v3_response_diagnostics", repo / "evaluation/objective-memory/v3_response_diagnostics.py")
    source_hashes, old_shards = {}, []
    with ExitStack() as locks:
        locks.enter_context(source_lock(old_suite))
        if digest(old_suite / "suite_plan.json") != OLD_PLAN_FILE_SHA or digest(old_suite / "suite_summary.json") != OLD_SUMMARY_FILE_SHA:
            raise ValueError("Original source plan/final summary differs from verified run")
        original_metadata_path = old_suite.parent / "derived-final-20261105/public_metadata.json"
        if digest(original_metadata_path) != OLD_METADATA_FILE_SHA:
            raise ValueError("Previously verified original metadata identity differs")
        original_metadata = read_json(original_metadata_path)
        source_hashes[original_metadata_path] = OLD_METADATA_FILE_SHA
        source_hashes[old_suite / "suite_plan.json"] = OLD_PLAN_FILE_SHA
        source_hashes[old_suite / "suite_summary.json"] = OLD_SUMMARY_FILE_SHA
        plan = read_json(old_suite / "suite_plan.json")
        if plan.get("suite_plan_sha256") != PLAN or legacy_stable({k: v for k, v in plan.items() if k != "suite_plan_sha256"}) != PLAN:
            raise ValueError("Original suite plan identity differs")
        manifest = read_json(diag_dir / "diagnostic_manifest.json")
        source_hashes[diag_dir / "diagnostic_manifest.json"] = digest(diag_dir / "diagnostic_manifest.json")
        for name, expected in manifest["output_sha256"].items():
            path = diag_dir / name
            if path.parent != diag_dir or digest(path) != expected:
                raise ValueError("Diagnostic output bytes differ")
            source_hashes[path] = expected
        summary = read_json(diag_dir / "summary.json")
        if (summary.get("candidate_commit") != CANDIDATE or summary.get("selected_request_identity_sha256") != SELECTION
                or summary.get("changed_cases") != 154 or summary.get("changed_arm_rows") != 206
                or summary.get("old_suite_plan_sha256") != PLAN or summary.get("new_provider_calls") != 0
                or summary.get("candidate_module_sha256") != code["evaluation/objective-memory/v4_source_retrieval.py"]
                or summary.get("diagnostic_helper_sha256") != code["evaluation/objective-memory/compare_v4_retrieval_colab.py"]):
            raise ValueError("Actual source-only diagnostic identity differs")
        for relative, expected in manifest["source_sha256"].items():
            path = old_suite / relative
            if not path.resolve().is_relative_to(old_suite) or digest(path) != expected:
                raise ValueError("Original source differs from diagnostic snapshot")
            source_hashes[path] = expected
        selected = rows(diag_dir / "controlled_candidate_requests.jsonl")
        if len(selected) != 206 or hashlib.sha256(json.dumps([[r["case_id"], r["arm"], r["prompt_sha256"]] for r in selected], sort_keys=True).encode()).hexdigest() != SELECTION:
            raise ValueError("Frozen source-only selected request cohort differs")
        for shard in plan["shards"]:
            if shard["dataset_key"] != "memoryagentbench_conflict":
                continue
            path = old_suite / "shards" / shard["shard_id"]
            locks.enter_context(source_lock(path))
            old_shards.append(load_old_shard(path, shard, plan, source_hashes, code))
        for path, expected in source_hashes.items():
            if path.is_relative_to(old_suite) and original_metadata["source_artifacts_sha256"].get(str(path.relative_to(old_suite.parent))) != expected:
                raise ValueError("Original data/journals differ from the previously verified full-run metadata")
        old_pairs = {key: (state, prompt) for state in old_shards for key, prompt in state["prompts"].items()}
        if len(old_pairs) != 2400:
            raise ValueError("Original full MAB grid differs")
        canonical, pairs = {}, {}
        allowed_fields = {"case_id", "arm", "messages", "old_prompt_sha256", "prompt_sha256", "settings", "source_protocol_sha256", "source_grammar", "candidate_policy"}
        for row in selected:
            key = (row["case_id"], row["arm"])
            if set(row) != allowed_fields or key in pairs or key not in old_pairs or key[1] not in ARMS[1:]:
                raise ValueError("Selected source-only request whitelist or grid differs")
            state, old = old_pairs[key]
            verify_old_settings(state["protocol"], row["settings"])
            if row["source_protocol_sha256"] != state["protocol"]["protocol_sha256"] or row["old_prompt_sha256"] != old["prompt_sha256"] or legacy_stable(row["messages"]) != row["prompt_sha256"] or row["prompt_sha256"] == old["prompt_sha256"]:
                raise ValueError("Changed source prompt identity differs")
            if row["source_grammar"] != "source-slot-version-view-v1" or row["candidate_policy"] != "source-edge-gated-expansion-v2":
                raise ValueError("Declared shared parser or retrieval policy differs")
            request = request_body(row["messages"])
            request_key = stable(request)
            if request_key in canonical and canonical[request_key] != request:
                raise ValueError("Canonical request digest collision")
            canonical.setdefault(request_key, request)
            pairs[key] = {"case_id": key[0], "arm": key[1], "request_key": request_key,
                          "prompt_sha256": row["prompt_sha256"], "old_prompt_sha256": row["old_prompt_sha256"]}
        protocol = {"schema": VERSION, "fixture_only": False, "candidate_commit": CANDIDATE,
                    "old_executable": OLD_EXECUTABLE, "old_suite_plan_sha256": PLAN, "source_only_selection_sha256": SELECTION,
                    "logical_changed_rows": 206, "changed_cases": 154, "maximum_provider_attempts": 206,
                    "canonical_new_requests": len(canonical), "maximum_workers": 4,
                    "model": MODEL, "allowed_model": ALIAS, "api_base": API, "max_response_tokens": 128,
                    "temperature": 0, "thinking_mode": "disabled", "maximum_attempts_per_canonical_request": 1,
                    "code_sha256": code, "helper_sha256": helper_hash,
                    "source_sha256": {str(p): value for p, value in source_hashes.items()},
                    "canonical_requests_sha256": stable(canonical), "logical_pairs_sha256": stable(list(pairs.values())),
                    "interpretation": "Development-exposed shared-retrieval repair; not new independent histories, users, CBT benefit or clinical outcome",
                    "temporal_provider_boundary": "Unchanged replies are reused original outputs, not contemporaneous reruns; returned-name matching is not independent attestation of unchanged provider weights",
                    "unchanged_policy": "Reuse original reply only for identical prompt and verified identical generation policy; no new call or new cost attributed"}
        protocol["protocol_sha256"] = stable(protocol)
        if out.exists():
            saved_requests = rows(out / "requests.jsonl")
            if (read_json(out / "protocol.json") != protocol
                    or len(saved_requests) != len(canonical)
                    or any(stable(r["request"]) != r["request_key"] for r in saved_requests)
                    or {r["request_key"]: r["request"] for r in saved_requests} != canonical
                    or stable(rows(out / "logical_changed_pairs.jsonl")) != protocol["logical_pairs_sha256"]
                    or digest(out / "executable_helper.py") != helper_hash):
                raise ValueError("Existing new-run identity differs; preserve it and choose a fresh directory")
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=out.name + ".tmp-", dir=out.parent))
            atomic_json(staging / "protocol.json", protocol)
            atomic_rows(staging / "requests.jsonl", [{"request_key": key, "request": request} for key, request in sorted(canonical.items())])
            atomic_rows(staging / "logical_changed_pairs.jsonl", list(pairs.values()))
            shutil.copyfile(Path(__file__), staging / "executable_helper.py")
            atomic_json(staging / "prepared_artifacts.json", {name: digest(staging / name) for name in ("protocol.json", "requests.jsonl", "logical_changed_pairs.jsonl", "executable_helper.py")})
            staging.rename(out)
        frozen_outputs = read_json(out / "prepared_artifacts.json")
        if set(frozen_outputs) != {"protocol.json", "requests.jsonl", "logical_changed_pairs.jsonl", "executable_helper.py"}:
            raise ValueError("Prepared output identity manifest differs")
        prepared_manifest_sha = digest(out / "prepared_artifacts.json")
        def unchanged_code():
            if digest(Path(__file__)) != helper_hash or any(digest(repo / relative) != expected for relative, expected in code.items()):
                raise ValueError("Frozen response code changed")
            if (digest(out / "prepared_artifacts.json") != prepared_manifest_sha
                    or any(digest(out / name) != expected for name, expected in frozen_outputs.items())
                    or read_json(out / "protocol.json") != protocol):
                raise ValueError("Frozen prepared request/protocol/helper files changed")
        with output_lock(out):
            unchanged_code()
            journal = Journal(out, protocol, canonical, unchanged_code, max_new_calls=args.max_new_calls)
            journal.provider = provider
            pending = sorted(set(canonical) - set(journal.outcomes))
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                list(executor.map(journal.call, pending))
            unchanged_code()
            journal.revalidate()
            if any(digest(path) != expected for path, expected in source_hashes.items()):
                raise ValueError("Original source or diagnostic files changed during bounded execution")
            complete = set(journal.outcomes) == set(canonical) and not journal.stopped
            result = {"schema": VERSION, "status": "complete" if complete else "partial_or_validation_blocked",
                      "protocol_sha256": protocol["protocol_sha256"], "candidate_commit": CANDIDATE,
                      "planned_cases": 800, "planned_three_arm_rows": 2400, "logical_changed_rows": 206,
                      "canonical_new_requests": len(canonical), "new_cost": journal.cost(),
                      "development_exposed": True, "independent_families": 1,
                      "clinical_claim": False, "causal_stage_attribution": "unresolved_without_independent_evidence_labels"}
            if complete:
                public, controlled = score_complete(old_shards, pairs, journal.outcomes, scorer, diagnostic)
                result.update(aggregate(public))
                result["by_task"] = {task: aggregate([r for r in public if r["task"] == task]) for task in sorted({r["task"] for r in public})}
                result["v3_to_v4_by_arm"] = {arm: {"old_correct": sum(r["old_v3_score"] for r in public if r["arm"] == arm),
                                                       "new_correct": sum(r["score"] for r in public if r["arm"] == arm),
                                                       "wins": sum(r["score"] > r["old_v3_score"] for r in public if r["arm"] == arm),
                                                       "losses": sum(r["score"] < r["old_v3_score"] for r in public if r["arm"] == arm)} for arm in ARMS}
                atomic_rows(out / "public_per_question_scores.jsonl", public)
                atomic_rows(out / "controlled_per_question_answers.jsonl", controlled)
            atomic_json(out / "public_summary.json", result)
            atomic_json(out / "SHA256SUMS.json", {p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name not in {".execution.lock", "SHA256SUMS.json"}})
            return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-repo", required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--diagnostic-dir", required=True)
    parser.add_argument("--old-suite", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-new-calls", type=int, default=206)
    args = parser.parse_args()
    try:
        result = run(args)
        print(json.dumps({key: result[key] for key in ("status", "protocol_sha256", "planned_cases", "planned_three_arm_rows", "logical_changed_rows", "canonical_new_requests", "new_cost")}, sort_keys=True))
    except Exception as error:
        print(json.dumps({"status": "safe_error", "error_type": type(error).__name__, "cloud_artifacts_preserved_at": args.out,
                          "provider_error_or_credential_displayed": False, "read_actual_attempt_journals_for_cost": True}))
        raise SystemExit(1)
