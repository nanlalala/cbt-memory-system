#!/usr/bin/env python3
"""Replay small, synthetic control events against a separately fixed ledger.

This is deterministic software acceptance, not clinical QA, a natural-language
extraction benchmark or evidence that managed memory helps real users. Both arms
receive the same explicitly annotated control events. The raw-history arm has
ordinary version, time, deletion and access controls; parity is a valid result.
No model, network client, embeddings or full benchmark data is loaded here.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
from typing import Callable


ROOT = Path(__file__).resolve().parents[2]
for relative in (
    "modules/long-term-memory/schema", "modules/long-term-memory/storage",
    "modules/long-term-memory/lifecycle", "modules/long-term-memory/retrieval",
    "modules/agent-integration",
):
    sys.path.insert(0, str(ROOT / relative))

from context_builder import build_context  # noqa: E402
from memory_manager import MemoryManager  # noqa: E402
from memory_models import MemoryRecord  # noqa: E402
from memory_retriever import MemoryRetriever, RetrievalPolicy  # noqa: E402
from sqlite_store import SQLiteMemoryStore  # noqa: E402


VERSION = "synthetic-control-replay-v2"
HERE = Path(__file__).resolve().parent
DEFAULT_EVENTS = HERE / "fixtures/control_events_v2.json"
DEFAULT_LEDGER = HERE / "fixtures/control_expected_v2.json"
TASKS = ("goals", "homework", "state", "corrections", "expiry", "deletion_isolation")
ARMS = ("raw_history_rag", "managed_memory_rag")
BOUNDARY = (
    "synthetic controlled software acceptance only; no model, natural-language "
    "extraction, free-response quality, real-user or clinical outcome evaluated"
)


def _time(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("event timestamps require a timezone")
    return result


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _record(source: dict, at: str) -> MemoryRecord:
    """Map an explicit operator event to storage, never infer its semantics."""
    kind = source.get("memory_type", "explicit_fact")
    payload = deepcopy(source.get("fields", {}))
    if source.get("slot_key"):
        payload["slot_key"] = source["slot_key"]
    return MemoryRecord.from_dict({
        "schema_version": "1.0", "memory_id": source["memory_id"],
        "user_id": source["user_id"], "memory_type": kind,
        "content": source["text"],
        "origin": "user_correction" if kind == "user_correction" else "explicit_user_statement",
        "confirmation_state": "unconfirmed", "lifecycle_status": "candidate",
        "sensitivity": source.get("sensitivity", "standard"),
        "confidence": 0.5, "importance": 0.5,
        "created_at": at, "observed_at": source.get("observed_at", at), "updated_at": at,
        "valid_from": source.get("valid_from", at), "valid_to": source.get("valid_to"),
        "source_evidence": {
            "session_id": source["session_id"], "turn_ids": [source["turn_id"]],
            "supporting_text": source["text"],
        },
        "structured_payload": payload,
        "supersedes_memory_id": source.get("supersedes_memory_id"),
    })


class RawHistoryControlStore:
    """Original text plus a minimal control log, independently of MemoryManager.

    Typed fields/slot IDs are supplied by the synthetic operator events equally
    to both arms. They are not extracted from text or discovered from the ledger.
    MemoryRecord is only an adapter for the shared retriever/context renderer.
    """

    def __init__(self) -> None:
        self.entries: dict[str, dict] = {}

    def close(self) -> None:
        pass

    def _require(self, user_id: str, memory_id: str) -> dict:
        entry = self.entries.get(memory_id)
        if entry is None or entry["source"]["user_id"] != user_id:
            raise KeyError("record not found for this user")
        return entry

    def apply(self, event: dict) -> None:
        op, at = event["op"], event["at"]
        _time(at)
        if op == "observe":
            source = deepcopy(event["record"])
            _record(source, at).validate()
            if source["memory_id"] in self.entries:
                raise ValueError("record ID already exists")
            self.entries[source["memory_id"]] = {
                "source": source, "created_at": at, "updated_at": at,
                "status": "candidate", "confirmation": "unconfirmed",
            }
        elif op == "confirm":
            if not event.get("consent_event_id", "").strip():
                raise ValueError("explicit synthetic confirmation event required")
            entry = self._require(event["user_id"], event["memory_id"])
            if entry["status"] != "candidate":
                raise ValueError("only candidate records can be confirmed")
            source = entry["source"]
            target_id = source.get("supersedes_memory_id")
            if target_id:
                target = self._require(event["user_id"], target_id)
                if target["status"] != "active":
                    raise ValueError("correction must target the active version")
                target.update(status="superseded", updated_at=at)
            elif source.get("slot_key"):
                for other in self.entries.values():
                    previous = other["source"]
                    if (other is not entry and other["status"] == "active"
                            and previous["user_id"] == source["user_id"]
                            and previous.get("memory_type", "explicit_fact") == source.get("memory_type", "explicit_fact")
                            and previous.get("slot_key") == source["slot_key"]):
                        other.update(status="superseded", updated_at=at)
            entry.update(status="active", updated_at=at,
                         confirmation="user_corrected" if target_id else "user_confirmed")
        elif op == "delete":
            self._require(event["user_id"], event["memory_id"])
            del self.entries[event["memory_id"]]
        elif op == "expire":
            for entry in self.entries.values():
                end = entry["source"].get("valid_to")
                if entry["status"] == "active" and end and _time(end) <= _time(at):
                    entry.update(status="expired", updated_at=at)
        else:
            raise ValueError(f"unsupported control event: {op}")

    def snapshot(self) -> list[MemoryRecord]:
        result = []
        for entry in self.entries.values():
            record = _record(entry["source"], entry["created_at"])
            record.lifecycle_status = entry["status"]
            record.confirmation_state = entry["confirmation"]
            record.updated_at = entry["updated_at"]
            result.append(record)
        return sorted(result, key=lambda record: record.memory_id)

    def eligible(self, user_id: str, *, as_of: str, allow_restricted: bool = False) -> list[MemoryRecord]:
        # Independently implement the declared half-open time/access rule. Do
        # not use the managed manager's output as a baseline or an oracle.
        now = _time(as_of)
        return [record for record in self.snapshot()
                if record.user_id == user_id and record.lifecycle_status == "active"
                and (allow_restricted or record.sensitivity != "restricted")
                and (not record.valid_from or _time(record.valid_from) <= now)
                and (not record.valid_to or now < _time(record.valid_to))]


class ManagedControlStore:
    def __init__(self) -> None:
        self.store = SQLiteMemoryStore()
        self.manager = MemoryManager(self.store)

    def close(self) -> None:
        self.store.close()

    def apply(self, event: dict) -> None:
        op, at = event["op"], event["at"]
        _time(at)
        if op == "observe":
            self.manager.ingest(_record(event["record"], at), actor="synthetic_control_operator")
        elif op == "confirm":
            self.manager.confirm_candidate(
                event["user_id"], event["memory_id"], consent_event_id=event["consent_event_id"],
                confirmation_turn_ids=[event["event_id"]], confirmed_at=at,
            )
        elif op == "delete":
            self.manager.delete(event["user_id"], event["memory_id"], deleted_at=at)
        elif op == "expire":
            self.manager.expire_due(as_of=at)
        else:
            raise ValueError(f"unsupported control event: {op}")

    def snapshot(self) -> list[MemoryRecord]:
        return sorted((record for uid in self.store.list_user_ids()
                       for record in self.store.list_for_user(uid)), key=lambda record: record.memory_id)

    def eligible(self, user_id: str, *, as_of: str, allow_restricted: bool = False) -> list[MemoryRecord]:
        return self.manager.eligible_context(user_id, as_of=as_of, allow_restricted=allow_restricted)


class _MemoryTags(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.open_count = 0
        self.invalid = False
        self.entries: list[dict] = []
        self.current: dict | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "memory":
            self.open_count += 1
            value = dict(attrs).get("id")
            if self.open_count != 1 or not value or len(attrs) != len(dict(attrs)):
                self.invalid = True
            self.ids.append(value or "")
            self.current = {"attrs": dict(attrs), "text": ""}
            self.entries.append(self.current)
        elif self.open_count:
            self.invalid = True

    def handle_endtag(self, tag):
        if tag == "memory":
            self.open_count -= 1
            if self.open_count != 0:
                self.invalid = True
            self.current = None
        elif self.open_count:
            self.invalid = True

    def handle_data(self, data):
        if self.open_count == 1 and self.current is not None:
            self.current["text"] += data


def _ids(records) -> list[str]:
    return sorted(record.memory_id for record in records)


def _query(engine, query: dict, expected: dict) -> dict:
    inventory = engine.snapshot()
    statuses = {record.memory_id: record.lifecycle_status for record in inventory}
    by_id = {record.memory_id: record for record in inventory}
    storage_ok = statuses == expected["storage_statuses"]
    for memory_id, fields in expected.get("fields", {}).items():
        record = by_id.get(memory_id)
        storage_ok &= record is not None and all(record.structured_payload.get(key) == value
                                                for key, value in fields.items())
    for memory_id, turn_ids in expected.get("source_turn_ids", {}).items():
        storage_ok &= memory_id in by_id and by_id[memory_id].source_evidence.turn_ids == turn_ids
    for memory_id, digest in expected.get("verbatim_text_sha256", {}).items():
        record = by_id.get(memory_id)
        storage_ok &= (record is not None and _sha(record.content.encode()) == digest
                       and _sha(record.source_evidence.supporting_text.encode()) == digest)
    eligible = engine.eligible(query["user_id"], as_of=query["as_of"],
                               allow_restricted=query.get("allow_restricted", False))
    top_k = query.get("top_k", 8)
    if type(top_k) is not int or not 1 <= top_k <= 32:
        raise ValueError("synthetic query top_k must be an integer 1–32")
    retrieved = MemoryRetriever(eligible, policy=RetrievalPolicy(top_k=top_k)).search(
        query["text"], user_id=query["user_id"], as_of=query["as_of"],
        allow_restricted=query.get("allow_restricted", False),
    )
    context = build_context(short_term_summary="Synthetic control replay query.",
                            personal_memories=retrieved, max_personal=top_k,
                            max_chars=query.get("max_chars", 6000))
    parser = _MemoryTags()
    parser.feed(context.prompt_context)
    trace_ok = not parser.invalid and parser.open_count == 0 and parser.ids == list(context.memory_ids)
    # The independent ledger binds the final rendered quote, not only an ID or
    # a hash of the storage row. HTMLParser decodes the renderer's escaped text
    # so exact source bytes can be checked even for <, &, and quotes.
    rendered_integrity = trace_ok
    rendered_hashes = {}
    for entry in parser.entries:
        memory_id = entry["attrs"].get("id", "")
        digest = _sha(entry["text"].encode())
        rendered_hashes[memory_id] = digest
        rendered_integrity &= digest == expected.get("verbatim_text_sha256", {}).get(memory_id)
        required_attrs = expected.get("rendered_memory_attributes", {}).get(memory_id)
        rendered_integrity &= required_attrs is not None and all(
            entry["attrs"].get(key) == value for key, value in (required_attrs or {}).items())
    forbidden_markers = [value for value in expected.get("forbidden_prompt_markers", ())
                         if value in unescape(context.prompt_context)]
    stages = {
        "storage": bool(storage_ok),
        "lifecycle_access": _ids(eligible) == sorted(expected["eligible_ids"]),
        "retrieval": sorted(item.record.memory_id for item in retrieved) == sorted(expected["retrieved_ids"]),
        "context_budget": sorted(context.dropped_memory_ids) == sorted(expected.get("dropped_ids", [])),
        "context_inclusion": (sorted(context.memory_ids) == sorted(expected["included_ids"])
                              and trace_ok and not forbidden_markers),
        "context_source_integrity": bool(rendered_integrity),
    }
    forbidden = set(expected.get("forbidden_ids", []))
    prohibited_ids = sorted(forbidden & ({item.record.memory_id for item in retrieved} | set(context.memory_ids)))
    return {
        "check_id": query["event_id"], "stages": stages, "passed": all(stages.values()),
        "first_failure_stage": next((name for name, passed in stages.items() if not passed), None),
        "storage_statuses": statuses, "eligible_ids": _ids(eligible),
        "retrieved_ids": [item.record.memory_id for item in retrieved],
        "included_ids": list(context.memory_ids), "dropped_ids": list(context.dropped_memory_ids),
        "source_turn_ids": {record.memory_id: list(record.source_evidence.turn_ids) for record in eligible},
        "prompt_sha256": _sha(context.prompt_context.encode()), "prompt_chars": len(context.prompt_context),
        "trace_matches_prompt": trace_ok, "prohibited_ids_exposed": prohibited_ids,
        "source_integrity_matches_prompt": bool(rendered_integrity),
        "rendered_text_sha256": rendered_hashes,
        "forbidden_marker_exposures": len(forbidden_markers),
    }


def load_spec(events_path: Path = DEFAULT_EVENTS, ledger_path: Path = DEFAULT_LEDGER) -> tuple[dict, dict]:
    # This local entry point is intentionally bounded to small synthetic fixtures.
    if max(events_path.stat().st_size, ledger_path.stat().st_size) > 256_000:
        raise ValueError("only small synthetic control fixtures may run locally")
    fixture, ledger = json.loads(events_path.read_text()), json.loads(ledger_path.read_text())
    if fixture.get("kind") != "synthetic_controlled_memory_events" or ledger.get("kind") != "independent_expected_control_ledger":
        raise ValueError("explicitly synthetic events and separate expected ledger are required")
    if fixture.get("version") != VERSION or ledger.get("version") != VERSION:
        raise ValueError("control fixture versions must match")
    if {case["task"] for case in fixture["cases"]} != set(TASKS):
        raise ValueError("all six declared task families are required")
    checks, seen_events = set(), set()
    for case in fixture["cases"]:
        if not any(event["op"] == "query" for event in case["events"]):
            raise ValueError("each control scenario requires at least one query assertion")
        previous_time = None
        for event in case["events"]:
            key = f"{case['case_id']}:{event['event_id']}"
            if key in seen_events:
                raise ValueError("event identifiers must be unique within a case")
            seen_events.add(key)
            timestamp = _time(event["at"])
            if previous_time is not None and timestamp < previous_time:
                raise ValueError("control events must be in chronological order")
            previous_time = timestamp
            if event["op"] not in {"observe", "confirm", "delete", "expire", "query"}:
                raise ValueError("unknown synthetic control operation")
            if event["op"] == "query":
                checks.add(key)
                if _time(event["as_of"]) != timestamp:
                    raise ValueError("this replay evaluates current state, not retrospective time travel")
    if checks != set(ledger["checks"]):
        raise ValueError("ledger must cover every query exactly once, without extra answers")
    if not set(ledger.get("operation_errors", {})).issubset(seen_events - checks):
        raise ValueError("expected operation errors must identify non-query events")
    return fixture, ledger


def run_replay(events_path: Path = DEFAULT_EVENTS, ledger_path: Path = DEFAULT_LEDGER,
               *, factories: dict[str, Callable] | None = None) -> dict:
    fixture, ledger = load_spec(events_path, ledger_path)
    factories = factories or {"raw_history_rag": RawHistoryControlStore, "managed_memory_rag": ManagedControlStore}
    if set(factories) != set(ARMS):
        raise ValueError("both controlled comparison arms are required")
    rows, operations = [], []
    for case in fixture["cases"]:
        for arm in ARMS:
            engine = factories[arm]()
            try:
                for event in case["events"]:
                    key = f"{case['case_id']}:{event['event_id']}"
                    if event["op"] == "query":
                        rows.append({"case_id": case["case_id"], "task": case["task"], "arm": arm,
                                     **_query(engine, event, ledger["checks"][key])})
                    else:
                        error_type = None
                        try:
                            engine.apply(event)
                        except Exception as exc:
                            error_type = type(exc).__name__
                        expected_error = ledger.get("operation_errors", {}).get(key)
                        operations.append({"case_id": case["case_id"], "task": case["task"], "arm": arm,
                                           "event_id": event["event_id"], "passed": error_type == expected_error,
                                           "expected_error_type": expected_error, "actual_error_type": error_type})
            finally:
                engine.close()
    summary = {
        "version": VERSION, "kind": "synthetic_software_mechanism_acceptance", "boundary": BOUNDARY,
        "clinical_or_user_benefit_result": False, "natural_language_extraction_evaluated": False,
        "model_or_network_calls": 0, "real_participants": 0,
        "independent_user_sample_size": None,
        "fixtures_sha256": _sha(events_path.read_bytes()), "expected_ledger_sha256": _sha(ledger_path.read_bytes()),
        "implementation_sha256": _sha(Path(__file__).read_bytes()),
        "comparison_controls": "same explicit synthetic events, lexical retrieval, context budget and access/time/deletion rules",
        "interpretation": "Both arms passing shows control compliance, not superiority of managed memory.",
        "scope_limits": ["sequential replay, not a new concurrency stress test",
                         "typed control events are supplied, not extracted from natural dialogue",
                         "current application state at each event; retrospective time-travel retrieval not evaluated",
                         "no free-text answer generation or clinical guidance evaluated"],
        "conditions": {}, "queries": rows, "operations": operations,
    }
    for arm in ARMS:
        subset = [row for row in rows if row["arm"] == arm]
        arm_ops = [row for row in operations if row["arm"] == arm]
        summary["conditions"][arm] = {
            "queries": len(subset), "passed_queries": sum(row["passed"] for row in subset),
            "operation_checks": len(arm_ops), "failed_operations": sum(not row["passed"] for row in arm_ops),
            "prohibited_exposure_queries": sum(bool(row["prohibited_ids_exposed"] or row["forbidden_marker_exposures"]) for row in subset),
            "by_task": {task: {
                "queries": sum(row["task"] == task for row in subset),
                "passed_queries": sum(row["passed"] for row in subset if row["task"] == task),
                "first_failures": {stage: sum(row["first_failure_stage"] == stage for row in subset if row["task"] == task)
                                   for stage in ("storage", "lifecycle_access", "retrieval", "context_budget", "context_inclusion", "context_source_integrity")},
            } for task in TASKS},
        }
    summary["acceptance_passed"] = all(row["passed"] for row in rows + operations)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_replay(args.events, args.ledger)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    if not report["acceptance_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
