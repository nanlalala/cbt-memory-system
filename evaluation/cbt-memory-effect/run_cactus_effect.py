"""Finite source-quotation recall probes in released synthetic CBT dialogues.

This is prompted quotation recall, not a semantic factual/clinical gold standard.
The cue is disclosed to every arm; scoring requires the complete undisclosed
suffix and the actual source identifier. Calendar/task semantics are disabled by
the independently versioned ordinal adapter. Real corpus preparation and every
model request require genuine hosted Colab. Tiny fixtures exercise pure helpers.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import cactus_materials as materials
import controlled_replay_adapter as replay

VERSION = "cactus-source-quotation-recall-v1"
MODEL = "deepseek-v4.1-flash"
ALIAS = "deepseek-v4-1-flash-260910"
API_BASE = "https://tokendance.space/gateway/v1"
ARMS = ("short_term", "raw_history_rag", "managed_memory_rag")
MAX_ATTEMPTS = 5000
ANSWER_CHARS = 6000
EXTRACTION_CHARS = 18000
OUTPUT_TOKENS = {"extraction": 4096, "response": 1024}
MIN_SUFFIX_CHARS = 24
MIN_SUFFIX_WORDS = 5
MAX_TARGET_CHARS = 700
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
PERMISSIONS = {key: True for key in replay.PERMISSION_KEYS}


def stable(value):
    return materials.digest(value)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def text_sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def file_sha(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular non-symlinked artifact required")
    return materials.file_hash(path)


def strict_json(body):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate JSON field")
            value[key] = item
        return value
    return json.loads(body, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def read_json(path):
    file_sha(path)
    return strict_json(Path(path).read_text())


def read_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    file_sha(path)
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("incomplete journal tail retained for explicit audit")
    return [strict_json(line) for line in raw.splitlines() if line]


def atomic(path, value, *, jsonl=False):
    path = Path(path)
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError("symlinked experiment path")
    temp = path.with_name(path.name + ".tmp")
    if temp.exists() or temp.is_symlink():
        raise ValueError("interrupted atomic write retained; explicit audit required")
    body = ("".join(materials.canonical(row) + "\n" for row in value) if jsonl else
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    with temp.open("x") as stream:
        stream.write(body); stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


def append(path, value):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("symlinked journal")
    with path.open("a") as stream:
        stream.write(materials.canonical(value) + "\n"); stream.flush(); os.fsync(stream.fileno())


def signed(value):
    return {**value, "record_sha256": stable(value)}


def checked(row):
    if type(row) is not dict or row.get("record_sha256") != stable({
            key: value for key, value in row.items() if key != "record_sha256"}):
        raise ValueError("journal record checksum mismatch")


@contextmanager
def locked(run_dir):
    path = Path(run_dir)
    if path.is_symlink():
        raise ValueError("symlinked run")
    with (path / ".execution.lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def norm(text):
    return " ".join(text.split())


def choose_probes(cases):
    """Fixed early/middle/recent targets; unavailable targets are not replaced.

    All three distance probes share the latest frozen released prefix. Target
    distance therefore does not change the amount of visible history. Target
    choice precedes model execution. A target must have an unambiguous
    four-word cue and at least 24 characters/five words not disclosed by it.
    The whole target message is requested, avoiding invented sentence boundaries.
    """
    families = defaultdict(list)
    for case in cases:
        families[case["source_family_id"]].append(case)
    generation, gold, inventory = [], [], []
    product = replay._modules()
    for family in sorted(families):
        ordered = sorted(families[family], key=lambda c: c["history_cutoff_order_exclusive"])
        if len(ordered) != 3:
            raise ValueError("exactly three frozen source cutoffs per family required")
        for position in range(3):
            original = ordered[-1]
            case_id = "quote-" + stable([VERSION, original["case_id"], position])[:32]
            base = {"case_id": case_id, "source_family_id": family,
                    "position": ("early", "middle", "recent")[position],
                    "released_continuation_case_sha256": original["case_sha256"]}
            clients = [turn for turn in original["history"] if turn["role"] == "user"]
            if not clients:
                raise ValueError("released cutoff has no prior Client message")
            index = (0, len(clients) // 2, len(clients) - 1)[position]
            target = clients[index]
            permission = product.permissions_for_input(target["content"], PERMISSIONS)
            unavailable = None
            if len(original["history"]) > 64:
                unavailable = "source_prefix_exceeds_frozen_turn_limit"
            # A permission-excluded target never supplies even a cue or gold.
            if not all(permission[k] for k in ("save", "long_term_use", "response_external")):
                unavailable = "privacy_withheld"
            words = list(re.finditer(r"\S+", target["content"]))
            cue_end = words[3].end() if len(words) >= 4 else len(target["content"])
            cue = target["content"][:cue_end]
            suffix = target["content"][cue_end:]
            if unavailable is None and (len(suffix) < MIN_SUFFIX_CHARS or
                    len(re.findall(r"\S+", suffix)) < MIN_SUFFIX_WORDS or
                    len(target["content"]) > MAX_TARGET_CHARS):
                unavailable = "probe_target_outside_frozen_length_rule"
            if unavailable is None and sum(turn["content"].startswith(cue) for turn in clients) != 1:
                unavailable = "ambiguous_disclosed_cue"
            if unavailable:
                inventory.append({**base, "preparation_status": unavailable})
                continue
            query = ("Quotation-recall research probe, not a therapy request. Earlier, one of my Client messages "
                "began with this locator cue: " + json.dumps(cue, ensure_ascii=False) + ". "
                "Return that complete earlier Client message exactly, without paraphrasing or guessing. "
                "Use its source_id shown in the supplied evidence. Reply only as JSON with exactly "
                "{\"quote\":\"exact complete message or empty\",\"source_id\":\"evidence source id or empty\","
                "\"unknown\":false}. If unavailable set unknown=true and both strings empty. "
                "The locator cue alone is not the requested complete quotation.")
            case = {**{key: deepcopy(original[key]) for key in ("source_family_id", "history",
                        "history_cutoff_order_exclusive", "calendar_time_available", "source_kind")},
                    "case_id": case_id, "input": query,
                    "input_message_id": original["input_message_id"] + ":quote-probe-" + str(position),
                    # This stable source namespace permits exact extraction-call
                    # reuse across a family's separate frozen prefix databases.
                    "replay_identity": "released-family-" + stable(family)[:24]}
            case["case_sha256"] = stable(case)
            generation.append(case)
            gold.append({**base, "case_sha256": case["case_sha256"],
                "target_message_id": target["message_id"], "target_quote": target["content"],
                "disclosed_cue": cue, "undisclosed_suffix": suffix,
                "undisclosed_suffix_characters": len(suffix), "gold_kind": "literal_original_Client_message",
                "semantic_fact_gold": False, "clinical_gold": False})
            inventory.append({**base, "preparation_status": "ready", "case_sha256": case["case_sha256"]})
    return generation, gold, inventory


def score_quote(response, target, source_id):
    """Deterministic quotation identity only; no causal/semantic inference."""
    try:
        result = strict_json(response)
        if (type(result) is not dict or set(result) != {"quote", "source_id", "unknown"} or
                type(result["quote"]) is not str or type(result["source_id"]) is not str or
                type(result["unknown"]) is not bool):
            raise ValueError("invalid quotation envelope")
        if result["unknown"]:
            if result["quote"] or result["source_id"]:
                raise ValueError("unknown response carries an unsupported quotation")
            return {"status": "explicit_unknown", "literal_success": False,
                    "quote_identity_correct": None, "source_binding_correct": None}
        quote_ok = norm(result["quote"]) == norm(target["target_quote"])
        source_ok = result["source_id"] == source_id
        correct = quote_ok and source_ok
        return {"status": "correct" if correct else "wrong_quote_or_source_binding",
                "literal_success": correct, "quote_identity_correct": quote_ok,
                "source_binding_correct": source_ok,
                "error_subtype": None if correct else "source_binding_only" if quote_ok else
                    "quote_identity_only" if source_ok else "quote_and_source_binding"}
    except (ValueError, TypeError, KeyError):
        return {"status": "technical_failure", "reason": "invalid_quote_envelope",
                "literal_success": False, "quote_identity_correct": None, "source_binding_correct": None}


def code_hashes():
    paths = {Path(__file__), HERE / "cactus_materials.py", HERE / "controlled_replay_adapter.py",
             ROOT / "evaluation/dialogue-benefit/followup_transport.py"}
    for directory in (ROOT / "app", ROOT / "modules", ROOT / "evaluation/end-to-end-dialogue",
                      ROOT / "evaluation/memory-retrieval"):
        paths.update(directory.rglob("*.py")); paths.update(directory.rglob("*.json"))
    return {str(path.relative_to(ROOT)): file_sha(path) for path in sorted(paths)}


def settings():
    return {"model": MODEL, "accepted_provider_model": ALIAS, "api_base": API_BASE,
            "temperature": 0, "thinking_mode": "disabled", "output_tokens": OUTPUT_TOKENS}


def decision_rule():
    """Prospective research thresholds, never a tuning/stopping rule.

    Source families are resampling units, not independent participants. The
    resource adoption gate stays unresolved until attributable resources have
    an independently specified acceptable threshold and complete measurement.
    These are engineering research criteria, not clinical benefit thresholds.
    """
    return {"contrast": "managed_memory_rag minus raw_history_rag",
        "endpoint": "literal_success among the frozen ready probes; technical failures and unknown remain in denominator",
        "minimum_absolute_recall_improvement": 0.05,
        "paired_family_cluster_confidence_level": 0.975,
        "recall_improvement_interval_lower_bound_strictly_greater_than": 0,
        "maximum_wrong_quote_or_source_binding_increase_interval_upper_bound": 0.01,
        "maximum_privacy_or_source_permission_critical_errors": 0,
        "resampling": {"method": "paired family-cluster percentile bootstrap",
                       "seed": 20261109, "draws": 10000},
        "effect_gate_requires_all_ready_probes_finalized": True,
        "resource_gate": {"status": "unresolved", "threshold": None,
            "requires_complete_cost_and_attribution": True,
            "unknown_or_missing_cost_cannot_pass": True},
        "used_for_sampling_tuning_or_early_stopping": False,
        "clinical_threshold": False}


def revision():
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def prepare(source, run_dir, *, family_limit=200):
    materials.require_colab()
    source, run_dir = Path(source), Path(run_dir)
    if run_dir.exists() or run_dir.is_symlink():
        raise ValueError("preserve old run; new directory required")
    if source.stat().st_size != materials.SOURCE_BYTES or file_sha(source) != materials.SOURCE_SHA256:
        raise ValueError("released CACTUS identity differs")
    identity = materials.download_cactus(source)
    with source.open() as stream:
        released = strict_json(stream.read())
    compiled = materials.compile_rows(released, family_limit=family_limit, cutoff_count=3)
    generation, gold, inventory = choose_probes(compiled["generation"])
    run_dir.mkdir(parents=True, mode=0o700)
    (run_dir / "generation").mkdir(); (run_dir / "assessor_only").mkdir(); (run_dir / "cases").mkdir()
    atomic(run_dir / "generation/cases.jsonl", generation, jsonl=True)
    atomic(run_dir / "assessor_only/targets.jsonl", gold, jsonl=True)
    atomic(run_dir / "inventory.jsonl", inventory, jsonl=True)
    atomic(run_dir / "source_diagnostics.jsonl", compiled["diagnostics"], jsonl=True)
    files = {relative: file_sha(run_dir / relative) for relative in (
        "generation/cases.jsonl", "assessor_only/targets.jsonl", "inventory.jsonl", "source_diagnostics.jsonl")}
    maximum_extractions = sum(sum(turn["role"] == "user" for turn in c["history"]) for c in generation)
    protocol = {"version": VERSION, "fixture_only": False, "created_at_utc": stamp(),
        "git_head": revision(),
        "code_sha256": code_hashes(), "source_identity": identity, "files_sha256": files,
        "arms": list(ARMS), "primary_contrast": ["managed_memory_rag", "raw_history_rag"],
        "family_limit": family_limit, "planned_families": compiled["summary"]["selected_families"],
        "planned_probes": len(inventory), "ready_probes": len(generation),
        "planned_arm_rows": len(inventory) * len(ARMS), "clock_mode": "ordinal-only-v1",
        "max_outbound_characters": ANSWER_CHARS, "max_extraction_outbound_characters": EXTRACTION_CHARS,
        "maximum_provider_attempts": MAX_ATTEMPTS, "maximum_workers": 1,
        "naive_extraction_attempt_upper_bound": maximum_extractions,
        "naive_response_attempt_upper_bound": len(generation) * len(ARMS),
        "naive_total_attempt_upper_bound": maximum_extractions + len(generation) * len(ARMS),
        "naive_upper_bound_exceeds_call_limit": maximum_extractions + len(generation) * len(ARMS) > MAX_ATTEMPTS,
        "budget_note": "Exact extraction/response requests may share canonical receipts; total cap never expands. Pending rows retain denominator.",
        "settings": settings(), "research_decision_rule": decision_rule(),
        "target_design": "three target distances in the SAME latest frozen released prefix per source family",
        "released_cutoff_used_per_family": "latest eligible cutoff chosen by the frozen source compiler",
        "locator_words": 4, "minimum_undisclosed_suffix_characters": MIN_SUFFIX_CHARS,
        "minimum_undisclosed_suffix_words": MIN_SUFFIX_WORDS, "maximum_target_characters": MAX_TARGET_CHARS,
        "task_semantics": "prompted literal quotation recall; not semantic facts, task updates or clinical efficacy",
        "scoring": "whole original Client quote and actual source ID; whitespace normalization only",
        "current_query_is_synthetic_probe": True, "calendar_task_semantics_enabled": False,
        "independent_human_reviews": 0, "clinical_claim": False, "natural_cbt_continuation_scores": None,
        "source_families_are_independent_people": False, "retry_policy": "unknown send blocks whole run; never resend automatically"}
    protocol["protocol_sha256"] = stable(protocol)
    atomic(run_dir / "protocol.json", protocol)
    return protocol


class CallLimit(RuntimeError):
    pass


class UnknownSend(RuntimeError):
    pass


def numeric_usage(value):
    value = value if type(value) is dict else {}
    return {key: value[key] if type(value.get(key)) is int and value[key] >= 0 else None for key in TOKEN_FIELDS}


def request_payload(messages, purpose):
    return {"model": MODEL, "messages": messages, "temperature": 0,
            "max_tokens": OUTPUT_TOKENS[purpose], "thinking": {"type": "disabled"}}


class CallJournal:
    """One external send per exact purpose and wire, including extraction.

    Pure journal tests use a tiny explicit local fake; real run supplies only the
    fixed transport. Cached extraction results are reapplied through real worker
    source/role/schema validation, never installed directly as a product state.
    """
    def __init__(self, directory, protocol, *, max_new_calls, allow_unknown=False):
        self.directory, self.protocol = Path(directory), protocol
        self.max_new_calls, self.new_calls, self.blocked = max_new_calls, 0, False
        self.intents, self.outcomes = {}, {}
        for name, collection in (("call_intents.jsonl", self.intents), ("call_outcomes.jsonl", self.outcomes)):
            for row in read_rows(self.directory / name):
                checked(row)
                key = row.get("request_sha256")
                if key in collection or row.get("protocol_sha256") != protocol["protocol_sha256"]:
                    raise ValueError("duplicate or foreign request journal")
                if stable({"purpose": row["purpose"], "payload": row["request_payload"]}) != key:
                    raise ValueError("request identity mismatch")
                purpose, payload = row["purpose"], row["request_payload"]
                if (purpose not in OUTPUT_TOKENS or type(payload) is not dict or
                        payload != request_payload(payload.get("messages"), purpose)):
                    raise ValueError("request differs from frozen provider policy")
                cap = EXTRACTION_CHARS if purpose == "extraction" else ANSWER_CHARS
                if len(json.dumps(payload, ensure_ascii=True, allow_nan=False)) > cap:
                    raise ValueError("preserved request exceeds frozen ceiling")
                collection[key] = row
        if set(self.outcomes) - set(self.intents) or len(self.intents) > MAX_ATTEMPTS:
            raise ValueError("unbound outcome or frozen call cap exceeded")
        for key, row in self.outcomes.items():
            if row["purpose"] != self.intents[key]["purpose"] or row["request_payload"] != self.intents[key]["request_payload"]:
                raise ValueError("outcome contradicts intent")
            if row["status"] not in {"response_received", "response_rejected", "attempt_unknown"}:
                raise ValueError("invalid call status")
            if row["usage"] != numeric_usage(row["usage"]):
                raise ValueError("invalid usage")
            if row["status"] == "response_received" and (row.get("provider_model") != ALIAS or
                    row.get("finish_reason") != "stop" or type(row.get("response_text")) is not str or
                    not row["response_text"].strip() or row.get("failure_kind") is not None):
                raise ValueError("model receipt identity differs")
            if row["status"] != "response_received":
                if any(row.get(key) is not None for key in ("response_text", "provider_model", "finish_reason")):
                    raise ValueError("failed/unknown receipt contains a completed model result")
                allowed = {"provider_result_unknown"} if row["status"] == "attempt_unknown" else {
                    "provider_http_error", "invalid_response_shape", "model_identity_failure", "incomplete_response",
                    "response_size_limit", "invalid_response_json"}
                if row.get("failure_kind") not in allowed:
                    raise ValueError("unrecognized provider failure receipt")
            for field in ("api_latency_seconds", "observed_invocation_seconds"):
                value = row.get(field)
                if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                    raise ValueError("invalid measured latency")
        if set(self.intents) - set(self.outcomes) or any(row["status"] == "attempt_unknown" for row in self.outcomes.values()):
            self.blocked = True
            if not allow_unknown:
                raise UnknownSend("preserved unknown send blocks new calls and automatic retries")

    def invoke(self, messages, purpose, transport):
        if purpose not in OUTPUT_TOKENS or self.blocked:
            raise UnknownSend("frozen call path blocked")
        payload = request_payload(messages, purpose)
        cap = EXTRACTION_CHARS if purpose == "extraction" else ANSWER_CHARS
        if len(json.dumps(payload, ensure_ascii=True, allow_nan=False)) > cap:
            raise ValueError("complete escaped outbound exceeds frozen character ceiling")
        key = stable({"purpose": purpose, "payload": payload})
        if key in self.outcomes:
            row = self.outcomes[key]
            if row["status"] != "response_received":
                self.blocked = True
                raise UnknownSend("previous failed canonical request cannot be retried automatically")
            return row, True
        if self.new_calls >= self.max_new_calls or len(self.intents) >= self.protocol["maximum_provider_attempts"]:
            self.blocked = True
            raise CallLimit("frozen global/new-call limit reached before send")
        shared = {"protocol_sha256": self.protocol["protocol_sha256"], "request_sha256": key,
                  "purpose": purpose, "request_payload": payload}
        intent = signed({**shared, "intent_at_utc": stamp()})
        append(self.directory / "call_intents.jsonl", intent)
        self.intents[key] = intent; self.new_calls += 1
        started = time.monotonic()
        row = {**shared, "status": "attempt_unknown", "response_text": None, "provider_model": None,
               "finish_reason": None, "usage": numeric_usage(None), "api_latency_seconds": None,
               "failure_kind": None, "received_at_utc": None}
        try:
            response = transport(messages, purpose)
            if (type(response) is not dict or response.get("provider_model") != ALIAS or
                    response.get("finish_reason") != "stop" or type(response.get("hypothesis")) is not str):
                raise ValueError("unvalidated model response")
            row.update(status="response_received", response_text=response["hypothesis"],
                provider_model=response["provider_model"], finish_reason=response["finish_reason"],
                usage=numeric_usage(response.get("usage")), api_latency_seconds=response.get("latency_seconds"),
                received_at_utc=stamp())
        except Exception as error:
            kind = getattr(error, "failure_kind", None)
            known = kind in {"provider_http_error", "invalid_response_shape", "model_identity_failure",
                             "incomplete_response", "response_size_limit", "invalid_response_json"}
            row.update(status="response_rejected" if known else "attempt_unknown",
                failure_kind=kind if known else "provider_result_unknown",
                usage=numeric_usage(getattr(error, "reported_usage", None)),
                api_latency_seconds=getattr(error, "latency_seconds", None))
            self.blocked = True
        row["observed_invocation_seconds"] = time.monotonic() - started
        row = signed(row); append(self.directory / "call_outcomes.jsonl", row); self.outcomes[key] = row
        if row["status"] != "response_received":
            raise UnknownSend("preserved provider failure; inspect journal before continuing")
        return row, False

    def summary(self):
        values = list(self.outcomes.values())
        totals, missing = {}, {}
        for field in TOKEN_FIELDS:
            observed = [row["usage"][field] for row in values if row["usage"][field] is not None]
            totals[field] = sum(observed) if observed else None
            missing[field] = len(self.intents) - len(observed)
        api_latencies = [r["api_latency_seconds"] for r in values if r["api_latency_seconds"] is not None]
        invocation_latencies = [r["observed_invocation_seconds"] for r in values]
        return {"actual_provider_attempts": len(self.intents), "new_calls_this_invocation": self.new_calls,
                "attempts_by_purpose": dict(Counter(row["purpose"] for row in self.intents.values())),
                "delivery_statuses": dict(Counter(row["status"] for row in values)),
                "tokens_reported": totals, "attempts_missing_usage": missing, "monetary_cost": None,
                "known_api_latency_seconds_sum": sum(api_latencies) if api_latencies else None,
                "attempts_missing_api_latency": len(self.intents) - len(api_latencies),
                "observed_invocation_seconds_sum": sum(invocation_latencies) if invocation_latencies else None,
                "latency_scope": "model invocation time only; not observed participant task time",
                "unknown_sends": len(set(self.intents) - set(self.outcomes)) + sum(row["status"] == "attempt_unknown" for row in values)}


class ExtractionProvider:
    def __init__(self, journal, transport):
        self.journal, self.transport = journal, transport

    def preview_request(self, *, messages, purpose):
        if purpose != "extraction":
            raise ValueError("worker provider is extraction-only")
        payload = request_payload(messages, purpose)
        serialized = json.dumps(payload, ensure_ascii=True, allow_nan=False)
        return {"within_character_budget": len(serialized) <= EXTRACTION_CHARS,
                "serialized_length": len(serialized), "serialized_request": serialized,
                "requested_output_tokens": OUTPUT_TOKENS[purpose]}

    def __call__(self, *, messages, purpose):
        row, _ = self.journal.invoke(messages, purpose, self.transport)
        return replay._modules().ModelOutput(text=row["response_text"], provider_model=row["provider_model"],
            usage={key: value for key, value in row["usage"].items() if value is not None},
            live_model_or_network_used=True)


def load_protocol(run_dir):
    run_dir = Path(run_dir)
    protocol = read_json(run_dir / "protocol.json")
    if protocol.get("protocol_sha256") != stable({k: v for k, v in protocol.items() if k != "protocol_sha256"}):
        raise ValueError("frozen protocol checksum mismatch")
    if (protocol.get("version") != VERSION or protocol.get("git_head") != revision() or
            protocol.get("settings") != settings() or protocol.get("research_decision_rule") != decision_rule() or
            protocol.get("code_sha256") != code_hashes() or
            protocol.get("maximum_provider_attempts") != MAX_ATTEMPTS or protocol.get("arms") != list(ARMS) or
            protocol.get("max_outbound_characters") != ANSWER_CHARS or
            protocol.get("max_extraction_outbound_characters") != EXTRACTION_CHARS or
            protocol.get("maximum_workers") != 1 or protocol.get("clock_mode") != "ordinal-only-v1"):
        raise ValueError("source/protocol changed; preserve old run and prepare a new run")
    for relative, expected in protocol["files_sha256"].items():
        if file_sha(run_dir / relative) != expected:
            raise ValueError("frozen preparation input differs")
    return protocol


def _target_source_id(prepared, target):
    hits = [row["source_id"] for row in prepared["trace"]["source_receipts"]
            if row["message_id"] == target["target_message_id"] and row["role"] == "user"]
    if len(hits) != 1:
        raise ValueError("target has no unique actual Client source binding")
    return hits[0]


def _target_in_wire(context, quote, source_id):
    for message in context["messages"]:
        if message["content"].startswith("SOURCED MEMORY DATA ONLY."):
            for claim in strict_json(message["content"].split("\n", 1)[1])["claims"]:
                for span in claim["evidence"]:
                    if span["source_id"] == source_id and norm(span["text"]) == norm(quote):
                        return True
    return False


def analyze(run_dir, journal=None):
    run_dir = Path(run_dir); protocol = load_protocol(run_dir)
    journal = journal or CallJournal(run_dir, protocol, max_new_calls=0, allow_unknown=True)
    inventory = read_rows(run_dir / "inventory.jsonl")
    targets = {row["case_id"]: row for row in read_rows(run_dir / "assessor_only/targets.jsonl")}
    answers = read_rows(run_dir / "answers.jsonl")
    by_key = {}
    for row in answers:
        checked(row)
        key = (row["case_id"], row["arm"])
        if row.get("protocol_sha256") != protocol["protocol_sha256"] or key in by_key:
            raise ValueError("duplicate/foreign answer row")
        if row["arm"] not in ARMS or row["case_id"] not in targets:
            raise ValueError("answer outside the frozen generation inventory")
        if row["score"]["status"] == "policy_blocked":
            if row["request_sha256"] is not None or row["original_response"] is not None or row["score"].get("literal_success") is not False:
                raise ValueError("blocked row carries a completed answer")
        else:
            receipt = journal.outcomes.get(row.get("provider_request_sha256"))
            prepared = read_json(run_dir / "cases" / row["case_id"] / "prepared_contexts.json")["prepared"]
            context = prepared["contexts"][row["arm"]]
            target = targets[row["case_id"]]
            sid = _target_source_id(prepared, target)
            if (receipt is None or receipt["status"] != "response_received" or
                    row["request_sha256"] != context["request_sha256"] or
                    receipt["request_payload"] != request_payload(context["messages"], "response")):
                raise ValueError("answer lacks its exact preserved provider request/receipt")
            if row["score"]["status"] != "stale_discarded":
                if (row["original_response"] != receipt["response_text"] or
                        row["response_sha256"] != text_sha(row["original_response"]) or
                        row["score"] != score_quote(row["original_response"], target, sid)):
                    raise ValueError("score contradicts the original provider response")
            elif row.get("original_response") is not None or row["score"].get("literal_success") is not False:
                raise ValueError("stale publication contains an answer body/success")
        by_key[key] = row
    counters = {arm: Counter() for arm in ARMS}
    family_values = {arm: defaultdict(list) for arm in ARMS}
    pairs = Counter(); equal_wire = 0; available_pairs = 0
    for item in inventory:
        pair = []
        for arm in ARMS:
            row = by_key.get((item["case_id"], arm))
            status = row["score"]["status"] if row else (
                "pending" if item["preparation_status"] == "ready" else item["preparation_status"])
            success = bool(row and row["score"]["literal_success"])
            counters[arm][status] += 1
            family_values[arm][item["source_family_id"]].append(int(success))
            if arm in ("raw_history_rag", "managed_memory_rag"):
                pair.append(row)
        if all(pair) and all(row["request_sha256"] is not None for row in pair):
            available_pairs += 1; equal_wire += pair[0]["request_sha256"] == pair[1]["request_sha256"]
            raw, managed = (r["score"]["literal_success"] for r in pair)
            pairs["managed_win" if managed and not raw else "raw_win" if raw and not managed else "tie"] += 1
        else:
            pairs["pending_or_unavailable"] += 1
    summary = {"version": VERSION, "protocol_sha256": protocol["protocol_sha256"],
        "fixture_only": protocol.get("fixture_only") is True,
        "planned_families": protocol["planned_families"], "planned_probes": len(inventory),
        "planned_arm_rows": len(inventory) * len(ARMS), "answer_rows_recorded": len(answers),
        "by_arm": {arm: {"counts": dict(counter), "full_planned_denominator": len(inventory),
            "completion_adjusted_allplanned_recall_lower_bound": counter["correct"] / len(inventory) if inventory else None,
            "allplanned_wrong_quote_or_source_binding_rate": counter["wrong_quote_or_source_binding"] / len(inventory) if inventory else None,
            "ready_probe_denominator": protocol["ready_probes"],
            "completion_adjusted_ready_recall_lower_bound": counter["correct"] / protocol["ready_probes"] if protocol["ready_probes"] else None,
            "measured_ready_recall_rate": counter["correct"] / protocol["ready_probes"]
                if protocol["ready_probes"] and counter["pending"] == 0 else None,
            "explicit_unknown_rate": counter["explicit_unknown"] / len(inventory) if inventory else None,
            "accuracy_among_nonabstaining_valid_quotes": (counter["correct"] / (counter["correct"] + counter["wrong_quote_or_source_binding"]))
                if counter["correct"] + counter["wrong_quote_or_source_binding"] else None,
            "family_equal_completion_adjusted_recall_lower_bound": (sum(sum(v) / len(v) for v in family_values[arm].values()) /
                                    len(family_values[arm])) if family_values[arm] else None}
            for arm, counter in counters.items()},
        "raw_managed_pairs": dict(pairs), "recorded_pair_count": available_pairs,
        "same_wire_pairs": equal_wire,
        "same_wire_fraction_of_recorded_pairs": equal_wire / available_pairs if available_pairs else None,
        "cost": journal.summary(),
        "interpretation": "cue-assisted literal source recall only; not semantic facts, calendar updates, independent people or clinical outcomes",
        "effect_complete": len(answers) == sum(i["preparation_status"] == "ready" for i in inventory) * len(ARMS)}
    atomic(run_dir / "summary.json", summary)
    return summary


def run(run_dir, *, max_new_calls, max_cases):
    materials.require_colab()
    if type(max_new_calls) is not int or not 0 <= max_new_calls <= MAX_ATTEMPTS or type(max_cases) is not int or not 0 <= max_cases <= 600:
        raise ValueError("finite integer invocation limits required")
    run_dir = Path(run_dir)
    with locked(run_dir):
        protocol = load_protocol(run_dir)
        if protocol.get("fixture_only") is True:
            raise ValueError("fixture protocol cannot authorize a real model run")
        journal = CallJournal(run_dir, protocol, max_new_calls=max_new_calls)
        # Reject detectable historical response/score contradictions before any
        # new credential access or paid request, rather than only at export.
        analyze(run_dir, journal)
        if max_new_calls == 0 or max_cases == 0:
            return analyze(run_dir, journal)
        try:
            key = importlib.import_module("google.colab.userdata").get("Capstone")
        except Exception:
            raise RuntimeError("Colab Secret Capstone unavailable; no provider send") from None
        sys.path.insert(0, str(ROOT / "evaluation/dialogue-benefit"))
        transport_module = importlib.import_module("followup_transport")
        if Path(transport_module.__file__).resolve() != (ROOT / "evaluation/dialogue-benefit/followup_transport.py").resolve():
            raise RuntimeError("foreign cached transport rejected")
        def transport(messages, purpose):
            return transport_module.generate_response(messages, model=MODEL, allowed_model=ALIAS,
                api_base=API_BASE, key=key, max_response_tokens=OUTPUT_TOKENS[purpose])
        provider = ExtractionProvider(journal, transport)
        cases = read_rows(run_dir / "generation/cases.jsonl")
        targets = {row["case_id"]: row for row in read_rows(run_dir / "assessor_only/targets.jsonl")}
        prior_answers = {(row["case_id"], row["arm"]) for row in read_rows(run_dir / "answers.jsonl")}
        processed = 0
        budget_blocked = None
        try:
            for case in cases:
                if processed >= max_cases or journal.new_calls >= max_new_calls:
                    break
                if all((case["case_id"], arm) in prior_answers for arm in ARMS):
                    continue
                directory = run_dir / "cases" / case["case_id"]
                prepared_path = directory / "prepared_contexts.json"
                database = directory / "memory.sqlite"
                if prepared_path.exists():
                    saved = read_json(prepared_path)
                    if saved.get("study_case_sha256") != case["case_sha256"] or saved.get("protocol_sha256") != protocol["protocol_sha256"]:
                        raise ValueError("prepared context belongs to different frozen case")
                    prepared = saved["prepared"]
                    needed = {stable({"purpose": "response", "payload": request_payload(
                        prepared["contexts"][arm]["messages"], "response")}) for arm in ARMS
                        if (case["case_id"], arm) not in prior_answers}
                    needed -= set(journal.outcomes)
                    if journal.new_calls + len(needed) > max_new_calls or len(journal.intents) + len(needed) > MAX_ATTEMPTS:
                        budget_blocked = {"case_id": case["case_id"], "reason": "remaining_answer_calls_do_not_fit_frozen_batch", "upper_calls_needed": len(needed)}
                        break
                elif directory.exists():
                    raise UnknownSend("interrupted prefix preparation retained; explicit audit required before resume")
                else:
                    # Conservative bound prevents a voluntary batch boundary
                    # interrupting this prefix. Canonical sharing can lower cost.
                    upper = sum(t["role"] == "user" for t in case["history"]) + len(ARMS)
                    if journal.new_calls + upper > max_new_calls or len(journal.intents) + upper > MAX_ATTEMPTS:
                        budget_blocked = {"case_id": case["case_id"], "reason": "whole_prefix_upper_bound_does_not_fit_remaining_frozen_budget", "upper_calls_needed": upper}
                        break
                    directory.mkdir(mode=0o700)
                    replay_case = deepcopy(case)
                    replay_case["case_id"] = replay_case.pop("replay_identity")
                    replay_case.pop("case_sha256")
                    try:
                        prepared = replay.prepare_contexts(replay_case, database=database, approved_root=directory,
                            permissions=PERMISSIONS, config=replay.ReplayConfig(max_outbound_chars=ANSWER_CHARS,
                                max_response_tokens=OUTPUT_TOKENS["response"], clock_mode="ordinal-only-v1"),
                            extraction_provider=provider, request_metadata={"model": MODEL, "temperature": 0,
                                "thinking": {"type": "disabled"}, "max_tokens": OUTPUT_TOKENS["response"]})
                    except replay.ReplayBlocked as error:
                        for arm in ARMS:
                            append(run_dir / "answers.jsonl", signed({"protocol_sha256": protocol["protocol_sha256"],
                                "case_id": case["case_id"], "arm": arm, "request_sha256": None,
                                "score": {"status": "policy_blocked", "reason": error.reason,
                                          "literal_success": False}, "original_response": None}))
                        processed += 1
                        continue
                    if journal.blocked:
                        raise UnknownSend("worker encountered a failed/unknown/limited model path; no answer send")
                    atomic(prepared_path, {"protocol_sha256": protocol["protocol_sha256"],
                        "study_case_sha256": case["case_sha256"], "prepared": prepared})
                target = targets[case["case_id"]]
                sid = _target_source_id(prepared, target)
                for arm in ARMS:
                    if (case["case_id"], arm) in prior_answers:
                        continue
                    context = prepared["contexts"][arm]
                    admission = replay.begin_response_attempt(prepared, arm, database=database, approved_root=directory)
                    try:
                        receipt, shared = journal.invoke(context["messages"], "response", transport)
                        finished = None if admission.get("replayed") else replay.finish_response_attempt(prepared,
                            database=database, approved_root=directory, attempt_id=admission["attempt_id"],
                            outcome="success", response=receipt["response_text"],
                            usage={k: v for k, v in receipt["usage"].items() if v is not None} or None)
                    except Exception:
                        if not admission.get("replayed"):
                            replay.finish_response_attempt(prepared, database=database, approved_root=directory,
                                attempt_id=admission["attempt_id"], outcome="unknown", usage=None)
                        raise
                    publishable = admission.get("status") == "succeeded" if admission.get("replayed") else finished["publishable"]
                    body = receipt["response_text"] if publishable else None
                    score = score_quote(body, target, sid) if publishable else {
                        "status": "stale_discarded", "literal_success": False}
                    append(run_dir / "answers.jsonl", signed({"protocol_sha256": protocol["protocol_sha256"],
                        "case_id": case["case_id"], "arm": arm, "request_sha256": context["request_sha256"],
                        "provider_request_sha256": receipt["request_sha256"], "canonical_response_shared": shared,
                        "original_response": body, "response_sha256": text_sha(body) if body else None,
                        "score": score, "target_in_final_wire": _target_in_wire(context, target["target_quote"], sid),
                        "outbound_chars": context["outbound_chars"], "error_stage": "unresolved"}))
                    prior_answers.add((case["case_id"], arm))
                processed += 1
                print(json.dumps({"stage": "source_quote_case_complete", "completed_cases_this_invocation": processed,
                    "actual_model_calls_total": len(journal.intents), "actual_model_calls_this_invocation": journal.new_calls}), flush=True)
        finally:
            key = None
            atomic(run_dir / "last_invocation.json", {"protocol_sha256": protocol["protocol_sha256"],
                "maximum_new_calls": max_new_calls, "maximum_cases": max_cases,
                "processed_cases": processed, "budget_blocked": budget_blocked,
                "new_calls_recorded": journal.new_calls, "attempts_total": len(journal.intents)})
            analyze(run_dir, journal)
        return analyze(run_dir, journal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--run-dir", type=Path, required=True)
    prep.add_argument("--families", type=int, default=200)
    execute = commands.add_parser("run")
    execute.add_argument("--run-dir", type=Path, required=True)
    execute.add_argument("--max-new-calls", type=int, required=True)
    execute.add_argument("--max-cases", type=int, required=True)
    report = commands.add_parser("analyze")
    report.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        protocol = prepare(args.source, args.run_dir, family_limit=args.families)
        result = {"prepared": True, "protocol_sha256": protocol["protocol_sha256"],
                  "planned_probes": protocol["planned_probes"], "ready_probes": protocol["ready_probes"],
                  "maximum_provider_attempts": MAX_ATTEMPTS, "actual_model_calls": 0}
    elif args.command == "run":
        result = run(args.run_dir, max_new_calls=args.max_new_calls, max_cases=args.max_cases)
    else:
        result = analyze(args.run_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
