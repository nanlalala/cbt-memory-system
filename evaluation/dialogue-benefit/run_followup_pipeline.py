"""Finite, journalled fictional CBT follow-up comparison; never a human trial.

Actual preparation/calls run only in hosted Colab. Generation cannot read the
scoring ledger. An interrupted attempt without an outcome blocks further calls;
it is never silently retried. Original objective runs remain separate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from threading import Lock
import time

ROOT = Path(__file__).resolve().parents[2]
VERSION = "fictional-followup-pipeline-development-v1"
ARMS = ("short_term", "raw_history_rag", "managed_memory_rag")
MODEL = "deepseek-v4.1-flash"
ALIAS = "deepseek-v4-1-flash-260910"
API_BASE = "https://tokendance.space/gateway/v1"
MAX_CALLS = 320
OUTPUT_LIMITS = {"extraction": 4096, "response": 1024, "judge": 1024}
INPUT_FILES = ("generation_materials.json", "replay_events.jsonl", "scoring_references.jsonl", "manifest.json")


def stable(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_colab():
    importlib.import_module("followup_transport").checked_import("colab_provider", ROOT / "app").require_colab()


def modules():
    adapter = importlib.import_module("generation_adapter")
    scorer = importlib.import_module("score_followup_pipeline")
    return adapter, scorer


def code_hashes():
    paths = [*Path(__file__).parent.glob("*.py"), *list((ROOT / "modules/agent-integration").glob("*.py")),
             *list((ROOT / "modules/long-term-memory").rglob("*.py")), ROOT / "app/colab_provider.py",
             ROOT / "evaluation/end-to-end-dialogue/run_three_arm.py"]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))}


def read_json(path):
    if path.is_symlink():
        raise ValueError("symlinked experimental artifact")
    return json.loads(path.read_text())


def rows(path):
    if not path.exists():
        return []
    if path.is_symlink():
        raise ValueError("symlinked journal")
    body = path.read_bytes()
    if body and not body.endswith(b"\n"):
        raise ValueError("incomplete journal tail; retain it for explicit audit")
    return [json.loads(line) for line in body.splitlines() if line]


def atomic_json(path, value):
    body = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        stream.write(body); stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


def atomic_rows(path, values):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


def append(path, value):
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        stream.flush(); os.fsync(stream.fileno())


@contextmanager
def locked(path):
    if path.is_symlink():
        raise ValueError("symlinked run root")
    with (path / ".execution.lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class BudgetExhausted(RuntimeError):
    pass


class Journal:
    """Exactly one durable attempt per exact purpose/settings/messages identity."""

    def __init__(self, run, protocol, *, max_new_calls, transport=None):
        self.run, self.protocol = Path(run), protocol
        self.max_new_calls, self.new_calls = max_new_calls, 0
        self.transport = transport
        self.mutex = Lock()
        self.aborted = False
        self.requests = {}; self.attempts = {}; self.outcomes = {}
        for row in rows(self.run / "requests.jsonl"):
            request = row["request"]
            key = row["request_key"]
            if stable(request) != key or row["protocol_sha256"] != protocol["protocol_sha256"] or key in self.requests:
                raise ValueError("request journal identity mismatch")
            purpose = request.get("purpose")
            if purpose not in OUTPUT_LIMITS or {k: v for k, v in request.items() if k != "messages"} != {
                "purpose": purpose, "model": MODEL, "accepted_provider_models": [ALIAS], "api_base": API_BASE,
                "max_response_tokens": OUTPUT_LIMITS[purpose], "temperature": 0, "thinking_mode": "disabled"}:
                raise ValueError("request differs from frozen provider policy")
            self.requests[key] = request
        for collection, filename in ((self.attempts, "attempts.jsonl"), (self.outcomes, "outcomes.jsonl")):
            for row in rows(self.run / filename):
                key = row["request_key"]
                if key not in self.requests or key in collection or row["protocol_sha256"] != protocol["protocol_sha256"] or row.get("record_sha256") != stable({k: v for k, v in row.items() if k != "record_sha256"}):
                    raise ValueError("attempt/outcome integrity mismatch")
                collection[key] = row
        if set(self.outcomes) - set(self.attempts):
            raise ValueError("outcome without recorded intent")
        if set(self.attempts) - set(self.outcomes):
            raise RuntimeError("unresolved provider attempt; no automatic retry or additional calls")
        if len(self.attempts) > protocol["maximum_provider_attempts"]:
            raise ValueError("journal exceeds frozen call budget")
        for key, outcome in self.outcomes.items():
            if outcome["status"] not in {"success", "provider_failure"}:
                raise ValueError("unknown provider outcome")
            if outcome["status"] == "success":
                self._validate_response(outcome["response"])

    @staticmethod
    def _validate_response(response):
        if response.get("provider_model") != ALIAS or response.get("finish_reason") != "stop" or not isinstance(response.get("hypothesis"), str) or not response["hypothesis"].strip():
            raise ValueError("unexpected model or incomplete response")
        latency = response.get("latency_seconds")
        if not isinstance(latency, (float, int)) or isinstance(latency, bool) or not math.isfinite(latency) or latency < 0:
            raise ValueError("invalid provider latency")
        if not isinstance(response.get("usage"), dict):
            raise ValueError("usage must retain an explicit object")

    def call(self, messages, purpose):
        if self.transport is None:
            require_colab()
            if not os.environ.get("CBT_EVAL_API_KEY"):
                raise RuntimeError("Colab credential unavailable; no provider attempt recorded")
        request = {"purpose": purpose, "messages": messages, "model": MODEL,
                   "accepted_provider_models": [ALIAS], "api_base": API_BASE,
                   "max_response_tokens": OUTPUT_LIMITS[purpose], "temperature": 0, "thinking_mode": "disabled"}
        key = stable(request)
        if not isinstance(messages, list) or not messages or any(set(m) != {"role", "content"} or m["role"] not in {"system", "user", "assistant"} or not isinstance(m["content"], str) for m in messages):
            raise ValueError("invalid bounded role/content request")
        if sum(len(m["content"]) for m in messages) > 24000:
            raise ValueError("request exceeds fixed 24000-character ceiling")
        with self.mutex:
            if self.aborted:
                raise RuntimeError("provider validation failure blocks further calls")
            if key in self.outcomes:
                return key, self.outcomes[key]
            if key in self.attempts:
                raise RuntimeError("attempt already in flight; never duplicate")
            if self.new_calls >= self.max_new_calls or len(self.attempts) >= self.protocol["maximum_provider_attempts"]:
                raise BudgetExhausted("finite invocation/run call budget exhausted")
            if code_hashes() != self.protocol["code_sha256"]:
                raise ValueError("code changed before provider call")
            if key not in self.requests:
                append(self.run / "requests.jsonl", {"protocol_sha256": self.protocol["protocol_sha256"], "request_key": key, "request": request})
                self.requests[key] = request
            intent = {"protocol_sha256": self.protocol["protocol_sha256"], "request_key": key, "attempt_index": 1,
                      "purpose": purpose, "started_at_unix": time.time()}
            intent["record_sha256"] = stable(intent)
            append(self.run / "attempts.jsonl", intent)
            self.attempts[key] = intent; self.new_calls += 1
        response = None
        fatal_validation = False
        try:
            if self.transport is not None:
                response = self.transport(messages, purpose)
            else:
                require_colab()
                transport = importlib.import_module("followup_transport")
                response = transport.generate_response(messages, model=MODEL, allowed_model=ALIAS, api_base=API_BASE,
                                                       key=os.environ["CBT_EVAL_API_KEY"], max_response_tokens=OUTPUT_LIMITS[purpose])
            self._validate_response(response)
            usage = response.get("usage", {})
            response = {"hypothesis": response["hypothesis"], "provider_model": response["provider_model"],
                        "finish_reason": "stop", "latency_seconds": response["latency_seconds"],
                        "usage": {k: usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens") if type(usage.get(k)) is int and usage[k] >= 0}}
            outcome = {"protocol_sha256": self.protocol["protocol_sha256"], "request_key": key,
                       "status": "success", "response": response, "reported_usage_missing_or_partial": len(response["usage"]) != 3}
        except Exception as error:
            # No exception text, raw HTTP bodies, headers or credentials persist.
            outcome = {"protocol_sha256": self.protocol["protocol_sha256"], "request_key": key,
                       "status": "provider_failure", "error_type": type(error).__name__,
                       "reported_usage": None, "billable_usage_unknown": True, "retry_sent": False}
            supplied = getattr(error, "reported_usage", None)
            if not supplied and isinstance(response, dict):
                supplied = response.get("usage")
            if isinstance(supplied, dict):
                outcome["reported_usage"] = {k: supplied[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens") if type(supplied.get(k)) is int and supplied[k] >= 0}
                outcome["billable_usage_unknown"] = len(outcome["reported_usage"]) != 3
            latency = getattr(error, "latency_seconds", None)
            if isinstance(latency, (float, int)) and not isinstance(latency, bool) and math.isfinite(latency) and latency >= 0:
                outcome["latency_seconds"] = latency
            if isinstance(error, (ValueError, KeyError, TypeError)) or getattr(error, "fatal", False):
                fatal_validation = True
                outcome["status"] = "protocol_validation_failure"
        outcome["record_sha256"] = stable(outcome)
        with self.mutex:
            append(self.run / "outcomes.jsonl", outcome); self.outcomes[key] = outcome
            self.aborted = self.aborted or fatal_validation
        if fatal_validation:
            raise RuntimeError("provider validation failed; retained cost/intent and stopped without replacement")
        return key, outcome

    def cost(self):
        successful = [x["response"] for x in self.outcomes.values() if x["status"] == "success"]
        failed = [x for x in self.outcomes.values() if x["status"] != "success"]
        failure_usage = [x.get("reported_usage") or {} for x in failed]
        latencies = [r["latency_seconds"] for r in successful]
        return {"recorded_provider_attempts": len(self.attempts), "successful_canonical_calls": len(successful),
                "failed_attempts": len(self.outcomes) - len(successful), "unresolved_attempts": len(set(self.attempts) - set(self.outcomes)),
                "failed_attempts_without_usage": sum(not x for x in failure_usage),
                "failed_attempts_missing_usage_fields": {k: sum(k not in r for r in failure_usage) for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
                "reported_tokens": {k: sum(r["usage"].get(k, 0) for r in successful) + sum(r.get(k, 0) for r in failure_usage) for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
                "successful_calls_missing_usage_fields": {k: sum(k not in r["usage"] for r in successful) for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
                "reported_model_counts": {ALIAS: len(successful)}, "monetary_cost": None,
                "median_successful_api_latency_seconds": statistics.median(latencies) if latencies else None,
                "sum_successful_api_latency_seconds": sum(latencies),
                "latency_boundary": "provider request time only; not parallel wall time or end-to-end replay/indexing time",
                "per_purpose": {purpose: {"attempts": sum(self.requests[k]["purpose"] == purpose for k in self.attempts),
                                          "successful_calls": sum(self.requests[k]["purpose"] == purpose and x["status"] == "success" for k, x in self.outcomes.items()),
                                          "reported_tokens": {token: sum((x["response"]["usage"] if x["status"] == "success" else x.get("reported_usage") or {}).get(token, 0)
                                                                        for k, x in self.outcomes.items() if self.requests[k]["purpose"] == purpose) for token in ("prompt_tokens", "completion_tokens", "total_tokens")}}
                                for purpose in OUTPUT_LIMITS},
                "boundary": "actual canonical attempts; failed/missing usage unknown, never free; no frozen verified currency price"}


def load_protocol(run, *, include_scoring=False):
    protocol = read_json(run / "protocol.json")
    if protocol.get("protocol_sha256") != stable({k: v for k, v in protocol.items() if k != "protocol_sha256"}) or protocol["code_sha256"] != code_hashes():
        raise ValueError("frozen protocol or code identity changed")
    if not protocol["fixture_only"]:
        head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        if head != protocol["executable_commit"] or subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD"], check=False).returncode:
            raise ValueError("clean pinned executable checkout changed")
    for name, expected in protocol["source_files_sha256"].items():
        if name == "scoring_references.jsonl" and not include_scoring:
            continue
        if digest(run / "source_package" / name) != expected:
            raise ValueError("frozen source package changed")
    return protocol


def prepare(source, run, *, fixture=False):
    if not fixture:
        require_colab()
    source, run = Path(source), Path(run)
    if run.exists() or run.is_symlink() or source.is_symlink():
        raise ValueError("fresh independent run directory required")
    compiler = importlib.import_module("prepare_generation_materials")
    prepared_package = compiler.read_prepared_generation(source)
    package = prepared_package["generation"]
    manifest = prepared_package["manifest"]
    if manifest["compiler_sha256"] != digest(Path(compiler.__file__)) or manifest["original_ledger_file_sha256"] != digest(ROOT / "evaluation/dialogue-benefit/task_materials_draft.json"):
        raise ValueError("prepared package compiler/material identity differs")
    for name in INPUT_FILES:
        if name != "manifest.json" and digest(source / name) != manifest["artifact_sha256"][name]:
            raise ValueError("prepared artifact differs from original compilation")
    cases = package["tasks"]
    ids = [c["case_id"] for c in cases]
    if len(set(ids)) != len(ids) or (not fixture and len(cases) != 36):
        raise ValueError("expected distinct frozen fictional task census")
    run.mkdir(parents=True)
    (run / "source_package").mkdir()
    for name in INPUT_FILES:
        if (source / name).is_symlink():
            raise ValueError("source package symlink rejected")
        shutil.copyfile(source / name, run / "source_package" / name)
    executable = "fixture_only" if fixture else subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if not fixture and subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD"], check=False).returncode:
        raise ValueError("prepare requires a clean pinned executable checkout")
    protocol = {"version": VERSION, "fixture_only": fixture, "executable_commit": executable, "planned_cases": len(cases), "planned_arm_rows": len(cases) * 3,
                "case_ids": ids, "arms": list(ARMS), "maximum_provider_attempts": MAX_CALLS,
                "model": MODEL, "accepted_provider_models": [ALIAS], "api_base": API_BASE,
                "max_output_tokens_by_purpose": OUTPUT_LIMITS, "maximum_request_chars": 24000,
                "character_context_budget": 6000, "top_k": 12, "temperature": 0, "thinking_mode": "disabled",
                "compute_boundary": "response budgets matched across arms; managed preprocessing extraction is extra measured cost, not matched total compute",
                "memory_processing": "chronological source-event extraction, candidate updates and actual replayed deletion before later questions; source observations remain unconfirmed",
                "maximum_workers": 4, "automatic_retries": False, "code_sha256": code_hashes(),
                "source_files_sha256": {n: digest(run / "source_package" / n) for n in INPUT_FILES},
                "primary_contrast": "managed_memory_rag_minus_raw_history_rag",
                "study_type": "fully_exposed_researcher_authored_fictional_development_replay",
                "independent_human_reference_validators": 0, "human_participants": 0, "clinical_claim": False,
                "scoring_boundary": "mechanical control/citation observations and separately masked same-model-assisted draft-criterion checks; no independently assessed CBT efficacy",
                "old_runs_overwritten": False, "user_computer_used": False}
    protocol["protocol_sha256"] = stable(protocol)
    atomic_json(run / "protocol.json", protocol)
    return {"status": "prepared", "protocol_sha256": protocol["protocol_sha256"], "planned_cases": len(cases), "planned_arm_rows": len(cases) * 3, "provider_calls": 0}


def generate(run, *, max_new_calls=160, workers=4, fixture=False, transport=None):
    if not fixture:
        require_colab()
    run = Path(run)
    with locked(run):
        protocol = load_protocol(run)
        if protocol["fixture_only"] != fixture or (transport is not None and not fixture):
            raise ValueError("fixture transport cannot become actual evidence")
        if not 1 <= workers <= 4 or not 1 <= max_new_calls <= MAX_CALLS:
            raise ValueError("finite worker/call range required")
        adapter, _ = modules()
        compiler = importlib.import_module("prepare_generation_materials")
        package = read_json(run / "source_package/generation_materials.json")
        cases = package["tasks"]
        events = rows(run / "source_package/replay_events.jsonl")
        compiler.validate_generation_package(package, events)
        journal = Journal(run, protocol, max_new_calls=max_new_calls, transport=transport)
        prepared = []
        stage_failures = []
        extraction_outputs = {}; extraction_rows = []; extraction_keys = {}; event_availability = {}
        for case in cases:
            variant = case["variant"]
            variant_events = [event for event in events if event["variant"] == variant]
            for event in variant_events:
                if event["kind"] != "source":
                    continue
                source = event["source"]
                sid = source["source_id"]
                if source["source_order"] > case["history_prefix_source_order"] or sid in extraction_outputs:
                    continue
                # The pure adapter scopes this full controller stream to the
                # current source order and applies same-order privacy controls
                # BEFORE constructing any outbound request. Future text is never
                # copied into messages. Assistant/session-only events return [].
                extraction_messages = adapter.source_event_extraction_request(event, variant_events)
                if not extraction_messages:
                    extraction_outputs[sid] = None
                    continue
                try:
                    key, extraction = journal.call(extraction_messages, "extraction")
                except BudgetExhausted:
                    atomic_rows(run / "extraction_event_rows_partial.jsonl", extraction_rows)
                    atomic_json(run / "generation_progress.json", {"status": "partial", "prepared_cases": len(prepared), "cost": journal.cost(), "final_effectiveness_result": False})
                    return read_json(run / "generation_progress.json")
                event_availability[sid] = extraction["status"] == "success"
                extraction_keys[sid] = key
                extraction_outputs[sid] = extraction["response"]["hypothesis"] if event_availability[sid] else None
                extraction_rows.append({"variant": variant, "source_id": sid, "source_order": source["source_order"],
                                        "request_key": key, "prompt_sha256": stable(extraction_messages), "completion_status": extraction["status"]})
            relevant_ids = [event["source"]["source_id"] for event in variant_events if event["kind"] == "source" and event["source"]["source_order"] <= case["history_prefix_source_order"]]
            active_outputs = {sid: extraction_outputs[sid] for sid in relevant_ids if sid in extraction_outputs}
            available = all(event_availability[sid] for sid in relevant_ids if sid in event_availability)
            try:
                value = adapter.prepare_case(case, replay_events=variant_events, extraction_outputs=active_outputs)
            except (ValueError, TypeError, KeyError) as error:
                stage_failures.append({"case_id": case["case_id"], "stage": "extraction_or_preparation", "error_type": type(error).__name__,
                                       "extraction_request_keys": [extraction_keys[sid] for sid in relevant_ids if sid in extraction_keys]})
                available = False
                value = adapter.prepare_case(case, replay_events=variant_events, extraction_outputs={sid: None for sid in active_outputs})
            invalid_events = [event["source_id"] for event in value["trace"]["source_event_replay"]
                              if event.get("kind") == "source" and event.get("semantic_status") == "extraction_invalid"]
            if invalid_events:
                available = False
                stage_failures.append({"case_id": case["case_id"], "stage": "source_bound_extraction_validation",
                                       "source_ids": invalid_events, "causal_answer_failure_claimed": False})
            if value["trace"]["semantic_pipeline_status"] != "valid_event_extractions":
                available = False
            allowed_ids = set(case["visible_source_ids"])
            for arm in ARMS:
                prepared_arm = value["prepared_by_arm"][arm]
                if set(prepared_arm["trace"]["included_source_ids"]) - allowed_ids:
                    raise ValueError("excluded/future source would enter answer context; no call sent")
                if stable(prepared_arm["messages"]) != prepared_arm["prompt_sha256"] or prepared_arm["trace"]["context_characters"] > protocol["character_context_budget"]:
                    raise ValueError("prepared prompt identity/budget mismatch")
            prepared.append({"case_id": case["case_id"], "extraction_available": available,
                             "extraction_request_keys": [extraction_keys[sid] for sid in relevant_ids if sid in extraction_keys], "prepared": value})
        atomic_rows(run / "extraction_event_rows.jsonl", extraction_rows)
        if (run / "prepared_cases.jsonl").exists() and rows(run / "prepared_cases.jsonl") != prepared:
            raise ValueError("prepared case trace/prompt changed across resume")
        atomic_rows(run / "prepared_cases.jsonl", prepared)
        atomic_rows(run / "stage_failures.jsonl", stage_failures)
        jobs = []
        for item in prepared:
            for arm in ARMS:
                jobs.append((item["case_id"], arm, item["prepared"]["prepared_by_arm"][arm]["messages"]))
        def execute(job):
            try:
                key, outcome = journal.call(job[2], "response")
                return job[0], job[1], key, outcome
            except BudgetExhausted:
                return job[0], job[1], None, None
        unique = {}
        for job in jobs:
            unique.setdefault(stable(job[2]), job)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            unique_results = list(pool.map(execute, unique.values()))
        by_prompt = {prompt_key: result for prompt_key, result in zip(unique, unique_results)}
        results = [(cid, arm, *by_prompt[stable(messages)][2:]) for cid, arm, messages in jobs]
        pmap = {v["case_id"]: v for v in prepared}
        cmap = {v["case_id"]: v for v in cases}
        output = []
        for cid, arm, key, outcome in results:
            case, prep = cmap[cid], pmap[cid]
            output.append({"case_id": cid, "variant": case["variant"], "task_id": case["task_id"], "arm": arm,
                           "request_key": key, "completion_status": "pending" if outcome is None else outcome["status"],
                           "raw_response": outcome["response"]["hypothesis"] if outcome and outcome["status"] == "success" else None,
                           "response_sha256": hashlib.sha256(outcome["response"]["hypothesis"].encode()).hexdigest() if outcome and outcome["status"] == "success" else None,
                           "extraction_available": prep["extraction_available"],
                           "prepared": prep["prepared"]["prepared_by_arm"][arm]})
        complete = all(r["completion_status"] != "pending" for r in output)
        atomic_rows(run / ("generation_rows.jsonl" if complete else "generation_rows_partial.jsonl"), output)
        progress = {"status": "complete" if complete else "partial", "planned_arm_rows": len(jobs),
                    "terminal_arm_rows": sum(r["completion_status"] != "pending" for r in output), "stage_failure_cases": len(stage_failures),
                    "cost": journal.cost(), "clinical_claim": False, "task_success_independently_assessed": False}
        if complete:
            receipt = {"protocol_sha256": protocol["protocol_sha256"], "generation_rows_sha256": digest(run / "generation_rows.jsonl"), "prepared_cases_sha256": digest(run / "prepared_cases.jsonl"), "planned_arm_rows": len(jobs)}
            if (run / "generation_receipt.json").exists() and read_json(run / "generation_receipt.json") != receipt:
                raise ValueError("frozen generation receipt differs")
            atomic_json(run / "generation_receipt.json", receipt)
        atomic_json(run / "generation_progress.json", progress)
        return progress


def review(run, *, max_new_calls=160, fixture=False, transport=None):
    if not fixture:
        require_colab()
    run = Path(run)
    with locked(run):
        protocol = load_protocol(run, include_scoring=True)
        if protocol["fixture_only"] != fixture or (transport is not None and not fixture):
            raise ValueError("fixture transport cannot become actual evidence")
        receipt = read_json(run / "generation_receipt.json")
        if receipt["generation_rows_sha256"] != digest(run / "generation_rows.jsonl") or receipt["prepared_cases_sha256"] != digest(run / "prepared_cases.jsonl"):
            raise ValueError("generation snapshot changed before scoring")
        _, scorer = modules()
        cases = read_json(run / "source_package/generation_materials.json")["tasks"]
        case_map = {c["case_id"]: c for c in cases}
        references = rows(run / "source_package/scoring_references.jsonl")
        reference_map = {r["case_id"]: r for r in references}
        generated = rows(run / "generation_rows.jsonl")
        if len(generated) != protocol["planned_arm_rows"] or len(reference_map) != protocol["planned_cases"] or set(reference_map) != set(case_map):
            raise ValueError("complete frozen generation/reference grid required")
        journal = Journal(run, protocol, max_new_calls=max_new_calls, transport=transport)
        judge_requests = []
        for row in generated:
            if row["raw_response"] is not None and scorer.reference_allows_model_review(reference_map[row["case_id"]]):
                masked_id = stable([protocol["protocol_sha256"], row["case_id"], row["response_sha256"]])[:24]
                messages = scorer.build_judge_messages(reference_map[row["case_id"]], row["raw_response"], query=case_map[row["case_id"]]["query"], masked_id=masked_id)
                judge_requests.append({"masked_id": masked_id, "response_sha256": row["response_sha256"], "messages": messages, "prompt_sha256": stable(messages)})
        if (run / "judge_requests.jsonl").exists() and rows(run / "judge_requests.jsonl") != judge_requests:
            raise ValueError("frozen masked judge requests changed")
        atomic_rows(run / "judge_requests.jsonl", judge_requests)
        reviewed = []; complete = True
        for row in generated:
            case, reference = case_map[row["case_id"]], reference_map[row["case_id"]]
            value = dict(row)
            value["observations"] = scorer.score_observations(case, row["prepared"], row["raw_response"], extraction_available=row["extraction_available"])
            review_allowed = scorer.reference_allows_model_review(reference)
            judgment = scorer.unresolved_judgment("answer_unavailable" if review_allowed else "privacy_reference_withheld_from_model_review")
            if row["raw_response"] is not None and review_allowed:
                masked_id = stable([protocol["protocol_sha256"], row["case_id"], row["response_sha256"]])[:24]
                request = next(r for r in judge_requests if r["masked_id"] == masked_id)
                try:
                    key, outcome = journal.call(request["messages"], "judge")
                    value["judge_request_key"] = key
                    if outcome["status"] == "success":
                        try:
                            judgment = scorer.validate_judge_output(outcome["response"]["hypothesis"], reference, row["raw_response"])
                        except (ValueError, TypeError, KeyError):
                            judgment = scorer.unresolved_judgment("invalid_or_unbound_model_assessment")
                    else:
                        judgment = scorer.unresolved_judgment("judge_provider_failure")
                except BudgetExhausted:
                    complete = False
                    judgment = scorer.unresolved_judgment("judge_not_yet_attempted")
            value["judgment"] = judgment
            reviewed.append(value)
        atomic_rows(run / ("reviewed_rows.jsonl" if complete else "reviewed_rows_partial.jsonl"), reviewed)
        summary = scorer.summarize_rows(reviewed, cases, references=references, allow_partial=not complete)
        summary.update({"status": "complete" if complete else "partial", "protocol_sha256": protocol["protocol_sha256"],
                        "fixture_only": fixture, "cost": journal.cost(), "clinical_claim": False,
                        "independent_human_cbt_ratings": 0, "human_participants": 0})
        atomic_json(run / ("followup_summary.json" if complete else "review_progress.json"), summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--source-dir", type=Path, required=True); p.add_argument("--run-dir", type=Path, required=True)
    for name in ("generate", "review"):
        p = sub.add_parser(name); p.add_argument("--run-dir", type=Path, required=True); p.add_argument("--max-new-calls", type=int, default=160)
        if name == "generate": p.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "prepare": result = prepare(args.source_dir, args.run_dir)
    elif args.command == "generate": result = generate(args.run_dir, max_new_calls=args.max_new_calls, workers=args.workers)
    else: result = review(args.run_dir, max_new_calls=args.max_new_calls)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
