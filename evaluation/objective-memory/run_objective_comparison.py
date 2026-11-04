#!/usr/bin/env python3
"""Frozen, gold-separated public QA comparison; full runs require hosted Colab.

This first representation is a source-verified *quotation* control.  It is not
semantic extraction.  Identical arm prompts share one explicitly accounted API
call, so sampling noise cannot masquerade as a memory-management contribution.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from contextlib import contextmanager
import fcntl
from datetime import datetime, timedelta, timezone
from html import escape
import json
import importlib.util
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[2]
for relative in (
    "evaluation/end-to-end-dialogue", "modules/long-term-memory/schema",
    "modules/long-term-memory/storage", "modules/long-term-memory/lifecycle",
    "modules/long-term-memory/retrieval",
):
    sys.path.insert(0, str(ROOT / relative))
from execution_lock import exclusive_directory  # noqa: E402
from run_three_arm import (Settings, HistoryTurn, HistoryRanker, SYSTEM_PROMPT,
                           stable_digest, digest_bytes, model_identity_policy,
                           validate_provider_model, validated_api_base, parse_date,
                           fit_turns, generate_response, read_jsonl, write_jsonl)  # noqa: E402
from memory_models import MemoryRecord, SourceEvidence  # noqa: E402
from memory_manager import MemoryManager  # noqa: E402
from sqlite_store import SQLiteMemoryStore  # noqa: E402
from memory_use_policy import assess_memory_use  # noqa: E402

VERSION = "objective-public-qa-source-index-v2.1"
ARMS = ("short_term", "raw_history_rag", "managed_memory_rag")
MAX_ATTEMPTS = 2
FROZEN_ROWS = ("cases.jsonl", "references.jsonl", "prompts.jsonl", "inventory.jsonl", "semantic_index.jsonl")
CASE_FIELDS = {
    "case_id", "group_id", "subject_id", "dataset_key", "task", "query", "options",
    "history", "events", "as_of", "ordering_kind", "history_scope_id",
    "persona_id", "source_context_id", "visible_context_end_exclusive", "topic",
    "source_question_index", "clustering_note",
}
HISTORY_FIELDS = {
    "turn_id", "session_id", "role", "text", "order", "observed_at", "sensitivity",
    "user_id", "valid_from", "valid_to", "lifecycle_status",
    "provenance_kind",
}
EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)


def require_colab() -> None:
    try:
        available = importlib.util.find_spec("google.colab") is not None
    except (ModuleNotFoundError, ValueError):
        available = False
    if not (Path("/content").is_dir() and os.environ.get("COLAB_RELEASE_TAG") and available):
        raise RuntimeError("Full dataset preparation and paid generation require hosted Google Colab")


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _append(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _replace_rows(path: Path, rows: list[dict]) -> None:
    """Atomically replace a derived diagnostic file, preserving old bytes on failure."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".diagnostic-", delete=False) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def _code_hashes() -> dict[str, str]:
    paths = [Path(__file__), ROOT / "evaluation/objective-memory/semantic_source_index.py", ROOT / "evaluation/end-to-end-dialogue/run_three_arm.py",
             ROOT / "evaluation/end-to-end-dialogue/execution_lock.py",
             ROOT / "evaluation/objective-memory/objective_datasets.py",
             ROOT / "evaluation/objective-memory/score_objective_comparison.py",
             ROOT / "evaluation/objective-memory/dataset_registry.json",
             ROOT / "evaluation/memory-retrieval/benchmark_adapters.py",
             ROOT / "evaluation/memory-retrieval/validate_external_benchmarks.py",
             ROOT / "evaluation/memory-retrieval/benchmark_manifest.json",
             ROOT / "modules/model_snapshots.py", ROOT / "modules/model_snapshots.json",
             ROOT / "modules/long-term-memory/schema/memory_models.py",
             ROOT / "modules/long-term-memory/storage/sqlite_store.py",
             ROOT / "modules/long-term-memory/lifecycle/memory_manager.py",
             ROOT / "modules/long-term-memory/retrieval/memory_use_policy.py",
             ROOT / "modules/long-term-memory/retrieval/memory_retriever.py"]
    return {str(path.relative_to(ROOT)): digest_bytes(path.read_bytes()) for path in paths}


def _settings(data: dict) -> Settings:
    result = Settings(**data)
    result.validate()
    for name in ("short_turns", "top_k", "short_chars", "long_chars", "max_context_chars", "max_response_tokens", "seed"):
        if type(getattr(result, name)) is not int:
            raise ValueError("budgets and seed must be integers")
    if type(result.minimum_relevance) not in (int, float) or not math.isfinite(result.minimum_relevance):
        raise ValueError("minimum relevance must be finite")
    return result


def _validate_case(case: dict) -> None:
    if not isinstance(case, dict) or set(case) - CASE_FIELDS:
        raise ValueError("case contains unknown fields; gold must remain in references")
    for key in ("case_id", "group_id", "subject_id", "dataset_key", "task", "query"):
        if not isinstance(case.get(key), str) or not case[key].strip():
            raise ValueError("case identifiers and question must be nonempty")
    if not isinstance(case.get("options", []), list) or any(not isinstance(x, str) for x in case.get("options", [])):
        raise ValueError("options must be a list of strings")
    if case.get("events"):
        raise ValueError("public QA runner does not infer lifecycle events; use the separate explicit-event replay")
    if not isinstance(case.get("history"), list):
        raise ValueError("history must be a list")
    keys = set()
    for row in case["history"]:
        if not isinstance(row, dict) or set(row) - HISTORY_FIELDS:
            raise ValueError("unknown history field")
        for key in ("turn_id", "session_id", "role"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError("source unit requires nonempty identifiers, role, and text")
        if not isinstance(row.get("text"), str):
            raise ValueError("source text must be a string")
        if type(row.get("order")) is not int or row["order"] < 0:
            raise ValueError("source order must be a nonnegative integer")
        key = (row["session_id"], row["turn_id"])
        if key in keys:
            raise ValueError("duplicate source unit ID")
        keys.add(key)
        if row.get("user_id", case["subject_id"]) != case["subject_id"]:
            raise ValueError("cross-subject source history is not permitted")
        if row.get("sensitivity", "standard") not in {"standard", "sensitive", "restricted"}:
            raise ValueError("unknown sensitivity")
        if row.get("lifecycle_status", "candidate") not in {"active", "candidate", "deleted", "expired", "superseded"}:
            raise ValueError("unknown lifecycle status")
        for field in ("observed_at", "valid_from", "valid_to"):
            if row.get(field):
                parse_date(row[field])
    if case.get("as_of"):
        parse_date(case["as_of"])


def _eligible(case: dict) -> tuple[list[dict], list[dict]]:
    admitted, excluded = [], []
    cutoff = parse_date(case["as_of"]) if case.get("as_of") else None
    for row in sorted(case["history"], key=lambda x: (x["order"], x["session_id"], x["turn_id"])):
        reason = None
        if not row["text"].strip():
            reason = "whitespace_only_source"
        elif row.get("sensitivity") == "restricted":
            reason = "restricted"
        elif row.get("lifecycle_status", "candidate") not in {"active", "candidate"}:
            reason = "inactive"
        elif cutoff and row.get("observed_at") and parse_date(row["observed_at"]) > cutoff:
            reason = "future"
        elif cutoff and row.get("valid_from") and parse_date(row["valid_from"]) > cutoff:
            reason = "not_yet_valid"
        elif cutoff and row.get("valid_to") and parse_date(row["valid_to"]) <= cutoff:
            reason = "expired"
        elif row.get("valid_to") and cutoff is None:
            raise ValueError("time-limited history requires an explicit as_of")
        if reason:
            excluded.append({"turn_id": row["turn_id"], "session_id": row["session_id"], "reason": reason})
        else:
            admitted.append(row)
    return admitted, excluded


def _document(case: dict, row: dict) -> HistoryTurn:
    return HistoryTurn(scope_id=case["subject_id"],
                       document_id=stable_digest([case["subject_id"], row["session_id"], row["turn_id"]]),
                       session_id=row["session_id"], date=row.get("observed_at") or f"source_order:{row['order']}",
                       speaker=row["role"], text=row["text"].strip())


def _managed_quotes(case: dict, admitted: list[dict]) -> tuple[list[HistoryTurn], list[dict]]:
    """Use real candidate storage/read policy; never invent confirmation events."""
    store = SQLiteMemoryStore()
    try:
        manager = MemoryManager(store)
        records, original, sources = [], {}, {}
        for row in admitted:
            doc = _document(case, row)
            # A schema clock is not a released calendar date. Never render it.
            stamp = parse_date(row["observed_at"]).isoformat() if row.get("observed_at") else (EPOCH + timedelta(seconds=row["order"])).isoformat()
            record = MemoryRecord(
                schema_version="1.0", memory_id=doc.document_id, user_id=case["subject_id"],
                memory_type="explicit_fact", content=doc.text,
                origin="system_observation",
                confirmation_state="unconfirmed", lifecycle_status="candidate",
                sensitivity=row.get("sensitivity", "standard"), confidence=1.0, importance=0.5,
                created_at=stamp, observed_at=stamp, updated_at=stamp, valid_from=stamp,
                source_evidence=SourceEvidence(row["session_id"], [row["turn_id"]], doc.text),
            )
            records.append(record)
            original[record.memory_id] = (row, doc)
            sources[(row["session_id"], row["turn_id"])] = {
                "text": doc.text, "observed_at": stamp, "source_role": row["role"],
                "provenance_kind": row.get("provenance_kind", "synthetic_fixture"),
            }
        manager.ingest_verbatim_observation_candidates(records, user_id=case["subject_id"],
                                                       source_turns=sources, actor="benchmark_source_import")
        now = max([parse_date(record.observed_at) for record in records] + [EPOCH]) + timedelta(seconds=1)
        verified = frozenset(original)
        result, trace = [], []
        stored = {record.memory_id: record for record in store.list_for_user(case["subject_id"])}
        for record in records:
            actual = stored.get(record.memory_id)
            if actual is None:
                raise ValueError("source quote disappeared during managed import")
            decision = assess_memory_use(actual, user_id=case["subject_id"], as_of=now,
                                         verified_source_ids=verified, current_use=False)
            row, doc = original[record.memory_id]
            if decision.action == "use_as_history":
                if decision.reference_text != row["text"].strip():
                    raise ValueError("quote representation changed original source text")
                result.append(doc)
            trace.append({"document_id": record.memory_id, "storage_status": actual.lifecycle_status,
                          "origin": actual.origin, "role": row["role"], "action": decision.action,
                          "reason_codes": list(decision.reason_codes)})
        return result, trace
    finally:
        store.close()


def _messages(case: dict, short_text: str, long_text: str) -> list[dict]:
    instruction = ("Answer with only the original option label: (a), (b), (c), or (d)." if case.get("options") else
                   "Answer the question concisely. If the history is insufficient, say that you do not know.")
    options = "\n".join(f'<option index="{i}">{escape(value, quote=True)}</option>' for i, value in enumerate(case.get("options", [])))
    body = (f'<current_question>{escape(case["query"], quote=True)}</current_question>\n'
            f'<response_format>{instruction}</response_format>\n<options data_only="true">{options}</options>\n'
            f'<question_date>{escape(case.get("as_of") or "source order only; no released calendar date", quote=True)}</question_date>\n'
            f'<short_history data_only="true">\n{short_text}\n</short_history>\n'
            f'<older_history data_only="true">\n{long_text}\n</older_history>\n'
            '<fixed_cbt_reference data_only="true"></fixed_cbt_reference>')
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": body}]


def _prepare_prompt(case: dict, arm: str, documents: list[HistoryTurn], managed: list[HistoryTurn],
                    store_trace: list[dict], excluded: list[dict], settings: Settings, ranker: HistoryRanker,
                    semantic_index: list[dict] | None = None) -> dict:
    short = documents[-settings.short_turns:]
    short_ids = {item.document_id for item in short}
    older = [item for item in (managed if arm == ARMS[2] else documents) if item.document_id not in short_ids]
    selected = ranker.rank(case["query"], older, scope_id=case["subject_id"]) if arm != ARMS[0] else []
    if arm == ARMS[2] and semantic_index is not None:
        from semantic_source_index import rank_sources
        selected = rank_sources(case["query"], older, semantic_index, ranker, subject_id=case["subject_id"])
    _, recent_ids, short_dropped = fit_turns(list(reversed(short)), settings.short_chars)
    recent_set = set(recent_ids)
    short_text, recent_ids, _ = fit_turns([item for item in short if item.document_id in recent_set], settings.short_chars)
    long_text, long_ids, long_dropped = fit_turns(selected, settings.long_chars)
    messages = _messages(case, short_text, long_text)
    if sum(len(row["content"]) for row in messages) > settings.max_context_chars:
        raise ValueError("total shared character budget exceeded; reduce channel budgets")
    prompt_hash = stable_digest(messages)
    return {
        "example_id": case["case_id"], "case_id": case["case_id"], "group_id": case["group_id"],
        "subject_id": case["subject_id"], "dataset_key": case["dataset_key"], "task": case["task"], "arm": arm,
        "request_key": stable_digest([case["case_id"], arm]), "canonical_request_key": prompt_hash,
        "prompt_sha256": prompt_hash, "messages": messages,
        "trace": {"encoding": "released_source_quotes_no_semantic_extraction", "source_unit_count": len(documents),
                  "selection_representation": "semantic_index_unverified_hints" if arm == ARMS[2] and semantic_index is not None else "source_quotes",
                  "semantic_candidate_count": len(semantic_index) if arm == ARMS[2] and semantic_index is not None else None,
                  "stored_quote_count": len(store_trace) if arm == ARMS[2] else None,
                  "storage_inventory_sha256": stable_digest(store_trace) if arm == ARMS[2] else None,
                  "storage": [row for row in store_trace if row["document_id"] in set(recent_ids + long_ids)] if arm == ARMS[2] else [],
                  "excluded_source_units": excluded,
                  "current_use": False, "interactive_clarification": False,
                  "retrieved_ids": [item.document_id for item in selected],
                  "included_short_ids": recent_ids, "included_long_ids": long_ids,
                  "budget_dropped_short_ids": short_dropped, "budget_dropped_long_ids": long_dropped},
    }


def prepare_run(cases: list[dict], references: list[dict], output_dir: Path, settings: Settings,
                api_base: str, dataset_manifest: dict, *, fixture_only: bool = False,
                semantic_index: list[dict] | None = None) -> dict:
    """Freeze a new run. Small explicitly labelled fixtures never become formal data."""
    if not fixture_only:
        require_colab()
    elif len(cases) > 10 or sum(len(json.dumps(x)) for x in cases) > 1024 * 1024:
        raise ValueError("fixture-only seam is limited to ten small synthetic cases")
    settings = _settings(asdict(settings))
    api_base = validated_api_base(api_base)
    if not cases or len({x.get("case_id") for x in cases}) != len(cases):
        raise ValueError("empty cohort or duplicate case IDs")
    if len(references) != len(cases) or {x.get("case_id") for x in references} != {x["case_id"] for x in cases}:
        raise ValueError("references must have exactly one row per frozen case")
    for case in cases:
        _validate_case(case)
    index_by_case = None
    if semantic_index is not None:
        if not isinstance(semantic_index, list):
            raise ValueError("semantic index must be a per-case list")
        index_by_case = {}
        for row in semantic_index:
            if not isinstance(row, dict) or set(row) != {"case_id", "candidates"} or row["case_id"] in index_by_case:
                raise ValueError("invalid or duplicate semantic index case")
            index_by_case[row["case_id"]] = row["candidates"]
        if set(index_by_case) != {c["case_id"] for c in cases}:
            raise ValueError("semantic index must cover exactly the frozen cohort, including empty candidate lists")
    output_dir = Path(output_dir)
    with exclusive_directory(output_dir):
        if (output_dir / "protocol.json").exists():
            raise ValueError("run already frozen; use a new output directory")
        if any((output_dir / name).exists() for name in FROZEN_ROWS):
            raise ValueError("partial preparation exists; choose a new output directory")
        prompts, inventory, seen_inventory, compact_cases, history_cache = [], [], set(), [], {}
        ranker = HistoryRanker(settings)
        for case in cases:
            admitted, excluded = _eligible(case)
            docs = [_document(case, row) for row in admitted]
            history_key = stable_digest([case["subject_id"], admitted])
            if history_key not in history_cache:
                history_cache[history_key] = _managed_quotes(case, admitted)
            managed, store_trace = history_cache[history_key]
            hints = None
            if index_by_case is not None:
                from semantic_source_index import validate_index
                hints = validate_index(index_by_case[case["case_id"]], documents=managed, subject_id=case["subject_id"])
            compact_cases.append({**{key: value for key, value in case.items() if key != "history"},
                                  "history_inventory_sha256": history_key})
            storage_by_id = {entry["document_id"]: entry for entry in store_trace}
            for row, doc in zip(admitted, docs):
                item = {"document_id": doc.document_id, "subject_id": case["subject_id"], "source": row,
                        "storage_decision": storage_by_id[doc.document_id]}
                key = stable_digest(item)
                if key not in seen_inventory:
                    inventory.append(item)
                    seen_inventory.add(key)
            prompts.extend(_prepare_prompt(case, arm, docs, managed, store_trace, excluded, settings, ranker, hints) for arm in ARMS)
        messages_by_key = {}
        for prompt in prompts:
            key = prompt["canonical_request_key"]
            if key in messages_by_key and messages_by_key[key] != prompt["messages"]:
                raise ValueError("canonical prompt digest collision")
            messages_by_key[key] = prompt["messages"]
        by_case = {case["case_id"]: {} for case in cases}
        for prompt in prompts:
            by_case[prompt["case_id"]][prompt["arm"]] = prompt["prompt_sha256"]
        equal = sum(row[ARMS[1]] == row[ARMS[2]] for row in by_case.values())
        equivalence = {"raw_managed_equal_cases": equal, "case_count": len(cases),
                       "raw_managed_equal_fraction": equal / len(cases),
                       "arm_requests": len(prompts), "unique_provider_requests": len(messages_by_key),
                       "policy": "exact_messages_hash_deduplicated_call",
                       "interpretation": "equivalent quote controls; shared response is not an independent observation"}
        rows_by_name = dict(zip(FROZEN_ROWS, (compact_cases, references, prompts, inventory, semantic_index or [])))
        for name, rows in rows_by_name.items():
            write_jsonl(output_dir / name, rows)
        code_hashes = _code_hashes()
        protocol = {
            "protocol_version": VERSION, "fixture_only": fixture_only, "arms": list(ARMS), "conditions": list(ARMS),
            "cohort": {"case_ids": [x["case_id"] for x in cases], "case_count": len(cases),
                       "group_ids": sorted({x["group_id"] for x in cases}), "group_count": len({x["group_id"] for x in cases})},
            "case_ids": [x["case_id"] for x in cases], "dataset_manifest": dataset_manifest,
            "settings": asdict(settings), "model_identity_policy": model_identity_policy(settings),
            "api_base": api_base, "temperature": 0, "code_sha256": code_hashes,
            "evaluator_identity": {"path": "evaluation/objective-memory/score_objective_comparison.py",
                                   "sha256": code_hashes.get("evaluation/objective-memory/score_objective_comparison.py"),
                                   "metric_version": "released-upstream-substring-exact-match-or-option-parser-v1"},
            "file_sha256": {name: digest_bytes((output_dir / name).read_bytes()) for name in FROZEN_ROWS},
            "maximum_attempts_per_request": MAX_ATTEMPTS, "deduplication": equivalence,
            "representation": {"version": "semantic_source_index_v1" if index_by_case is not None else "source_verified_quote_v1",
                               "semantic_candidate_artifact": "semantic_index.jsonl" if index_by_case is not None else None,
                               "semantic_boundary": "unverified index hints select eligible original quotes only; no summary confirmation, task agreement or version suppression is inferred",
                               "index_generation_provenance": "external artifact; gold-blind generation and independent semantic review not authenticated by this preparer" if index_by_case is not None else "not_applicable",
                               "index_generation_cost": "unreported; response-only costs are not total pipeline costs" if index_by_case is not None else "not_applicable",
                               "origin_policy": "all released data system_observation, including synthetic role=user; unconfirmed candidates, no user consent fabricated",
                               "current_use": False, "source_roles": "all released visible roles, retained equally in raw and managed",
                               "clock_policy": "released timestamps if present; otherwise internal source-order schema clock never rendered as calendar",
                               "events_policy": "no inferred events; explicit lifecycle events use separate replay"},
            "source_text_policy": "common boundary-whitespace trim and whitespace-only exclusion; inventory retains original source bytes",
            "budget_policy": "same character budgets and generation max_tokens; input token counts observed, not exact-token matched",
            "knowledge_policy": "empty in every arm; this is memory QA, not CBT clinical outcome",
            "evaluation_policy": {"primary": "group_equal_accuracy", "bootstrap_seed": 20261104,
                                  "bootstrap_samples": 10000, "independence_unit": "group_id",
                                  "power_status": "exploratory_no_prospective_power"},
            "retrieval_model_identity": getattr(ranker.encoder, "model_identity", None) if getattr(ranker, "encoder", None) else None,
        }
        protocol["protocol_sha256"] = stable_digest(protocol)
        _json(output_dir / "protocol.json", protocol)
        summary = {"protocol_sha256": protocol["protocol_sha256"], "fixture_only": fixture_only,
                   "cohort": protocol["cohort"], "deduplication": equivalence,
                   "effect_claim": "No semantic-memory or clinical benefit has been demonstrated by this preparation."}
        _json(output_dir / "preparation_summary.json", summary)
        return summary


def _canonical(prompts: list[dict]) -> dict[str, dict]:
    result = {}
    for prompt in prompts:
        result.setdefault(prompt["canonical_request_key"], prompt)
    return result


def _validate_response(row: dict, protocol: dict, prompt: dict) -> None:
    validate_provider_model(row.get("provider_model"), protocol["model_identity_policy"])
    if not isinstance(row.get("hypothesis"), str) or not row["hypothesis"].strip() or row.get("finish_reason") != "stop":
        raise ValueError("invalid completed response")
    if row.get("prompt_sha256") != prompt["prompt_sha256"] or row.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise ValueError("response differs from frozen prompt/protocol")
    if row.get("request_key") != prompt["canonical_request_key"] or row.get("canonical_request_key") != prompt["canonical_request_key"]:
        raise ValueError("response canonical request identity mismatch")
    if type(row.get("latency_seconds")) not in (float, int) or not math.isfinite(row["latency_seconds"]) or row["latency_seconds"] < 0:
        raise ValueError("invalid response latency")
    if not isinstance(row.get("usage"), dict):
        raise ValueError("invalid usage")
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if field in row["usage"] and (type(row["usage"][field]) is not int or row["usage"][field] < 0):
            raise ValueError("invalid usage counter")
    claimed = row.get("response_sha256")
    if claimed != stable_digest({key: value for key, value in row.items() if key != "response_sha256"}):
        raise ValueError("response integrity mismatch")


def _journals_locked(run_dir: Path, protocol: dict, prompts: list[dict], *,
                     repair_mirror: bool = True) -> tuple[dict, dict, dict]:
    canonical = _canonical(prompts)
    attempts, outcomes, responses = {}, {}, {}
    for row in read_jsonl(run_dir / "response_attempts.jsonl"):
        key, index = row.get("canonical_request_key"), row.get("attempt_index")
        if key not in canonical or type(index) is not int or not 1 <= index <= MAX_ATTEMPTS:
            raise ValueError("attempt journal differs from frozen requests")
        if row.get("request_key") != key or row.get("protocol_sha256") != protocol["protocol_sha256"] or row.get("prompt_sha256") != canonical[key]["prompt_sha256"]:
            raise ValueError("attempt identity mismatch")
        group = attempts.setdefault(key, {})
        if index in group or set(group) != set(range(1, index)):
            raise ValueError("duplicate or out-of-order attempt index")
        group[index] = row
    for row in read_jsonl(run_dir / "response_outcomes.jsonl"):
        key, index = row.get("canonical_request_key"), row.get("attempt_index")
        if index not in attempts.get(key, {}) or (key, index) in outcomes:
            raise ValueError("outcome without unique intent")
        identity = attempts[key][index]
        if any(row.get(field) != identity.get(field) for field in ("request_key", "protocol_sha256", "prompt_sha256")):
            raise ValueError("outcome identity mismatch")
        if row.get("status") not in {"success", "technical_failure", "identity_or_integrity_failure"} or type(row.get("retryable")) is not bool:
            raise ValueError("invalid attempt outcome")
        if row["status"] == "identity_or_integrity_failure":
            raise ValueError("run contains a model-identity or integrity failure; no formal scoring")
        outcomes[(key, index)] = row
    for row in read_jsonl(run_dir / "responses.jsonl"):
        key, index = row.get("canonical_request_key"), row.get("attempt_index")
        if key not in canonical or key in responses or index not in attempts.get(key, {}):
            raise ValueError("response has no unique frozen intent")
        _validate_response(row, protocol, canonical[key])
        responses[key] = row
    for (key, index), outcome in outcomes.items():
        if outcome["status"] == "success" and (key not in responses or responses[key]["attempt_index"] != index):
            raise ValueError("successful outcome has no matching response")
        if outcome["status"] != "success" and key in responses and responses[key]["attempt_index"] == index:
            raise ValueError("failed outcome has a successful response")
    for key, indexed in attempts.items():
        for index in sorted(indexed):
            if index > 1:
                prior = outcomes.get((key, index - 1))
                if prior is None or prior["status"] != "technical_failure" or not prior["retryable"]:
                    raise ValueError("attempt follows an unresolved, successful, or terminal attempt")
    failures = read_jsonl(run_dir / "failures.jsonl")
    expected_failures = [row for row in outcomes.values() if row["status"] != "success"]
    # outcomes is authoritative. A crash can interrupt its diagnostic mirror.
    if failures != expected_failures[:len(failures)]:
        raise ValueError("failure journal differs from outcome ledger")
    if failures != expected_failures and repair_mirror:
        _replace_rows(run_dir / "failures.jsonl", expected_failures)
    return attempts, outcomes, responses


def _load_run_locked(run_dir: Path, *, repair_mirror: bool = True) -> tuple[dict, list[dict], list[dict], dict]:
    run_dir = Path(run_dir)
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("protocol_version") != VERSION or protocol.get("arms") != list(ARMS):
        raise ValueError("unsupported objective protocol")
    digest = protocol.get("protocol_sha256")
    if digest != stable_digest({key: value for key, value in protocol.items() if key != "protocol_sha256"}):
        raise ValueError("frozen protocol hash mismatch")
    settings = _settings(protocol["settings"])
    if protocol.get("model_identity_policy") != model_identity_policy(settings) or protocol.get("maximum_attempts_per_request") != MAX_ATTEMPTS:
        raise ValueError("frozen policy mismatch")
    validated_api_base(protocol["api_base"])
    for relative, expected in protocol["code_sha256"].items():
        if relative not in _code_hashes() or digest_bytes((ROOT / relative).read_bytes()) != expected:
            raise ValueError("runner/evaluator code changed; freeze a new run")
    rows = {}
    for name in FROZEN_ROWS:
        path = run_dir / name
        if digest_bytes(path.read_bytes()) != protocol["file_sha256"].get(name):
            raise ValueError("frozen artifact changed")
        rows[name] = read_jsonl(path)
    cases, refs, prompts = rows["cases.jsonl"], rows["references.jsonl"], rows["prompts.jsonl"]
    if [x["case_id"] for x in cases] != protocol["case_ids"] or len(prompts) != len(cases) * len(ARMS):
        raise ValueError("frozen cohort differs from prompt grid")
    pairs, identities = set(), {}
    for prompt in prompts:
        pair = (prompt["example_id"], prompt["arm"])
        if pair in pairs or prompt["example_id"] not in protocol["case_ids"] or prompt["arm"] not in ARMS:
            raise ValueError("invalid prompt grid")
        pairs.add(pair)
        if stable_digest(prompt["messages"]) != prompt["prompt_sha256"] or prompt["canonical_request_key"] != prompt["prompt_sha256"]:
            raise ValueError("prompt hash mismatch")
        key = prompt["canonical_request_key"]
        if key in identities and identities[key] != prompt["messages"]:
            raise ValueError("canonical digest collision")
        identities[key] = prompt["messages"]
    _, outcomes, raw = _journals_locked(run_dir, protocol, prompts, repair_mirror=repair_mirror)
    canonical = _canonical(prompts)
    completed = {}
    for prompt in prompts:
        key = prompt["canonical_request_key"]
        response = raw.get(key)
        if response is None or (key, response["attempt_index"]) not in outcomes:
            continue
        owner = canonical[key]
        charged = prompt["request_key"] == owner["request_key"]
        completed[(prompt["example_id"], prompt["arm"])] = {
            **response, "example_id": prompt["example_id"], "case_id": prompt["case_id"], "arm": prompt["arm"],
            "group_id": prompt["group_id"], "dataset_key": prompt["dataset_key"], "task": prompt["task"],
            "request_key": prompt["request_key"], "canonical_request_key": key,
            "source_response_sha256": response["response_sha256"], "deduplicated_call": not charged,
            "charged_call": charged,
        }
    return protocol, prompts, refs, completed


def load_run(run_dir: Path) -> tuple[dict, list[dict], list[dict], dict]:
    with exclusive_directory(Path(run_dir)):
        return _load_run_locked(Path(run_dir))


def _request_outcomes_locked(run_dir: Path, protocol: dict, prompts: list[dict], completed: dict, *,
                             repair_mirror: bool = True) -> dict:
    attempts, outcomes, _ = _journals_locked(Path(run_dir), protocol, prompts, repair_mirror=repair_mirror)
    result = {}
    for prompt in prompts:
        pair, key = (prompt["example_id"], prompt["arm"]), prompt["canonical_request_key"]
        indices = attempts.get(key, {})
        last = max(indices, default=0)
        outcome = outcomes.get((key, last))
        if pair in completed:
            status = "success"
        elif last and outcome is None:
            status = "inflight"
        elif outcome and (not outcome["retryable"] or last >= MAX_ATTEMPTS):
            status = "exhausted_failure"
        else:
            status = "pending"
        result[pair] = {"status": status, "canonical_request_key": key, "attempt_count": last,
                        "error_type": outcome.get("error_type") if outcome else None}
    return result



_READONLY_OPTIONAL = ("response_attempts.jsonl", "response_outcomes.jsonl", "responses.jsonl", "failures.jsonl")


def _readonly_fingerprint(run_dir: Path) -> dict:
    """Fingerprint only the fixed input allowlist; no files are created."""
    result = {}
    for name in ("protocol.json", *FROZEN_ROWS, *_READONLY_OPTIONAL, ".execution.lock"):
        path = run_dir / name
        if path.is_symlink():
            raise ValueError("read-only artifact snapshots do not follow symlinked files")
        try:
            before = path.stat()
        except FileNotFoundError:
            result[name] = None
            continue
        body = path.read_bytes()
        after = path.stat()
        stamp = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        if stamp(before) != stamp(after):
            raise RuntimeError("experiment artifacts changed during read-only snapshot")
        result[name] = (stamp(after), digest_bytes(body))
    return result


@contextmanager
def _readonly_existing_lock(run_dir: Path):
    """Use an existing cooperating-writer lock read-only; never create one.

    Offline copies may omit the lock. Byte/identity fingerprints then reject
    observable concurrent changes instead of claiming a transaction lock.
    """
    lock_path = run_dir / ".execution.lock"
    if lock_path.is_symlink():
        raise ValueError("read-only snapshot lock cannot be a symlink")
    try:
        handle = lock_path.open("rb")
    except FileNotFoundError:
        yield
        return
    with handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("experiment directory is busy; no read-only result produced") from None
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_run_state_readonly(run_dir: Path) -> dict:
    """Validate a coherent artifact snapshot without repair or lock creation.

    Missing/short valid failure mirrors are derived from authoritative outcomes
    in memory. Contradictions still fail. No returned result is produced if the
    fixed artifacts change while they are read. This reader is suitable for a
    read-only artifact copy; it does not resume, recover or score a paid batch.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise ValueError("read-only snapshot requires an existing experiment directory")
    with _readonly_existing_lock(run_dir):
        before = _readonly_fingerprint(run_dir)
        protocol, prompts, references, completed = _load_run_locked(run_dir, repair_mirror=False)
        outcomes = _request_outcomes_locked(run_dir, protocol, prompts, completed, repair_mirror=False)
        frozen = {name: read_jsonl(run_dir / name) for name in FROZEN_ROWS}
        if before != _readonly_fingerprint(run_dir):
            raise RuntimeError("experiment artifacts changed during read-only snapshot")
        current_code = _code_hashes()
        if any(current_code.get(name) != expected for name, expected in protocol["code_sha256"].items()):
            raise ValueError("runner/evaluator code changed during read-only snapshot")
    return {"protocol": protocol, "prompts": prompts, "references": references,
            "completed": completed, "outcomes": outcomes, "frozen_rows": frozen}


def load_run_readonly(run_dir: Path) -> tuple[dict, list[dict], list[dict], dict]:
    state = read_run_state_readonly(run_dir)
    return state["protocol"], state["prompts"], state["references"], state["completed"]


def request_outcomes(run_dir: Path, protocol: dict | None = None, prompts: list[dict] | None = None,
                     completed: dict | None = None) -> dict:
    with exclusive_directory(Path(run_dir)):
        checked_protocol, checked_prompts, _, checked_completed = _load_run_locked(Path(run_dir))
        return _request_outcomes_locked(Path(run_dir), checked_protocol, checked_prompts, checked_completed)


def generate_run(run_dir: Path, *, max_calls: int, key: str | None = None) -> dict:
    require_colab()
    if type(max_calls) is not int or max_calls <= 0:
        raise ValueError("finite positive max_calls is required")
    secret = key or os.environ.get("CBT_EVAL_API_KEY", "")
    if not secret:
        raise ValueError("API key missing; configure a Colab secret")
    run_dir = Path(run_dir)
    with exclusive_directory(run_dir):
        protocol, prompts, _, completed = _load_run_locked(run_dir)
        if protocol["fixture_only"]:
            raise ValueError("fixture-only runs cannot call a real provider")
        settings = _settings(protocol["settings"])
        count = 0
        for canonical_key, prompt in _canonical(prompts).items():
            states = _request_outcomes_locked(run_dir, protocol, prompts, completed)
            state = states[(prompt["example_id"], prompt["arm"])]
            if state["status"] in {"success", "exhausted_failure", "inflight"}:
                continue
            if count >= max_calls:
                break
            intent = {"request_key": canonical_key, "canonical_request_key": canonical_key,
                      "protocol_sha256": protocol["protocol_sha256"], "prompt_sha256": prompt["prompt_sha256"],
                      "attempt_index": state["attempt_count"] + 1, "started_at_unix": time.time()}
            _append(run_dir / "response_attempts.jsonl", intent)
            count += 1
            try:
                response = generate_response(prompt, settings, api_base=protocol["api_base"], key=secret)
                row = {**response, **intent}
                row["response_sha256"] = stable_digest(row)
                _validate_response(row, protocol, prompt)
            except Exception as exc:
                identity = isinstance(exc, (ValueError, KeyError, TypeError))
                retryable = isinstance(exc, (URLError, TimeoutError, ConnectionError, RuntimeError))
                if isinstance(exc, HTTPError):
                    retryable = exc.code == 429 or 500 <= exc.code < 600
                outcome = {**intent, "status": "identity_or_integrity_failure" if identity else "technical_failure",
                           "retryable": retryable and not identity, "error_type": type(exc).__name__}
                _append(run_dir / "response_outcomes.jsonl", outcome)
                _append(run_dir / "failures.jsonl", outcome)
                if identity:
                    raise ValueError("provider identity or response integrity rejected; run cannot be formally scored") from None
            else:
                # Persistence is not a provider failure. If either write raises,
                # leave the intent unresolved instead of appending a conflicting
                # failure. A validated response already on disk is explicitly
                # recoverable without another paid request. Malformed/partial
                # journal writes are still rejected by the ordinary validator.
                _append(run_dir / "responses.jsonl", row)
                _append(run_dir / "response_outcomes.jsonl", {**intent, "status": "success", "retryable": False})
            protocol, prompts, _, completed = _load_run_locked(run_dir)
        return _status_locked(run_dir, protocol, prompts, completed, actual_calls_this_invocation=count)


def _status_locked(run_dir: Path, protocol: dict, prompts: list[dict], completed: dict, **extra: Any) -> dict:
    states = _request_outcomes_locked(run_dir, protocol, prompts, completed)
    counts = {status: sum(row["status"] == status for row in states.values()) for status in ("success", "pending", "inflight", "exhausted_failure")}
    attempts, _, raw = _journals_locked(run_dir, protocol, prompts)
    return {"protocol_sha256": protocol["protocol_sha256"], "arm_outcomes": counts,
            "arm_rows": len(prompts), "unique_provider_requests": len(_canonical(prompts)),
            "actual_provider_attempts": sum(len(x) for x in attempts.values()), "valid_provider_responses": len(raw),
            "complete": counts["pending"] == counts["inflight"] == 0,
            "deduplication": protocol["deduplication"], **extra}


def status_run(run_dir: Path) -> dict:
    with exclusive_directory(Path(run_dir)):
        protocol, prompts, _, completed = _load_run_locked(Path(run_dir))
        return _status_locked(Path(run_dir), protocol, prompts, completed)


def recover_interrupted(run_dir: Path, *, canonical_request_key: str, resolution: str) -> dict:
    """Explicitly reconcile an interrupted intent without sending another call.

    A valid locally recorded response can be committed. Otherwise the operator
    may close the unknown attempt as terminal technical failure: it remains in
    the cohort and attempted-call cost, never silently becomes a free retry.
    """
    if resolution not in {"use_recorded_response", "terminal_unknown_failure"}:
        raise ValueError("unknown interrupted-attempt resolution")
    run_dir = Path(run_dir)
    with exclusive_directory(run_dir):
        protocol, prompts, _, completed = _load_run_locked(run_dir)
        attempts, outcomes, responses = _journals_locked(run_dir, protocol, prompts)
        if canonical_request_key not in _canonical(prompts):
            raise ValueError("unknown canonical request")
        indexed = attempts.get(canonical_request_key, {})
        index = max(indexed, default=0)
        if not index or (canonical_request_key, index) in outcomes:
            raise ValueError("request has no unresolved latest intent")
        intent = indexed[index]
        response = responses.get(canonical_request_key)
        if resolution == "use_recorded_response":
            if response is None or response["attempt_index"] != index:
                raise ValueError("no validated recorded response for this intent")
            outcome = {**intent, "status": "success", "retryable": False,
                       "reconciliation": "explicit_valid_recorded_response_recovery"}
        else:
            if response is not None:
                raise ValueError("validated response exists; recover that response instead")
            outcome = {**intent, "status": "technical_failure", "retryable": False,
                       "error_type": "InterruptedAttemptOutcomeUnknown",
                       "reconciliation": "explicit_terminal_unknown_attempt_possible_provider_charge"}
        _append(run_dir / "response_outcomes.jsonl", outcome)
        # Recreate the diagnostic mirror; the outcome ledger remains authoritative.
        all_outcomes = read_jsonl(run_dir / "response_outcomes.jsonl")
        _replace_rows(run_dir / "failures.jsonl", [row for row in all_outcomes if row["status"] != "success"])
        protocol, prompts, _, completed = _load_run_locked(run_dir)
        return _status_locked(run_dir, protocol, prompts, completed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--context", type=Path)
    prepare.add_argument("--semantic-index", type=Path, help="Separately logged source-bound index JSONL; new v2 protocol only")
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--model", required=True)
    prepare.add_argument("--accepted-provider-model", action="append", default=[])
    prepare.add_argument("--api-base", required=True)
    prepare.add_argument("--backend", choices=("lexical", "dense", "hybrid"), default="lexical")
    prepare.add_argument("--thinking-mode", choices=("provider_default", "enabled", "disabled"), default="disabled")
    for name, default in (("short-turns", 16), ("top-k", 12), ("short-chars", 6000), ("long-chars", 12000), ("max-context-chars", 24000), ("max-response-tokens", 256)):
        prepare.add_argument("--" + name, type=int, default=default)
    generate = sub.add_parser("generate")
    generate.add_argument("--run-dir", type=Path, required=True)
    generate.add_argument("--max-calls", type=int, required=True)
    status = sub.add_parser("status")
    status.add_argument("--run-dir", type=Path, required=True)
    recover = sub.add_parser("recover")
    recover.add_argument("--run-dir", type=Path, required=True)
    recover.add_argument("--canonical-request-key", required=True)
    recover.add_argument("--resolution", required=True,
                         choices=("use_recorded_response", "terminal_unknown_failure"))
    args = parser.parse_args()
    if args.command == "prepare":
        require_colab()
        from objective_datasets import load_dataset_cases, dataset_spec, stable_hash
        cases, references = load_dataset_cases(args.dataset, args.input, context_path=args.context)
        files = [args.input] + ([args.context] if args.context else [])
        spec = dataset_spec(args.dataset)
        manifest = {"dataset_key": args.dataset, "data_revision": spec["data_revision"],
                    "registry_entry_sha256": stable_hash(spec),
                    "input_files": [{"name": x.name, "bytes": x.stat().st_size, "sha256": digest_bytes(x.read_bytes())} for x in files]}
        settings = Settings(model=args.model, accepted_provider_models=tuple(args.accepted_provider_model),
                            backend=args.backend, thinking_mode=args.thinking_mode, short_turns=args.short_turns,
                            top_k=args.top_k, short_chars=args.short_chars, long_chars=args.long_chars,
                            max_context_chars=args.max_context_chars, max_response_tokens=args.max_response_tokens)
        index = read_jsonl(args.semantic_index) if args.semantic_index else None
        result = prepare_run(cases, references, args.output_dir, settings, args.api_base, manifest, semantic_index=index)
    elif args.command == "generate":
        result = generate_run(args.run_dir, max_calls=args.max_calls)
    elif args.command == "recover":
        result = recover_interrupted(args.run_dir, canonical_request_key=args.canonical_request_key,
                                     resolution=args.resolution)
    else:
        result = status_run(args.run_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
