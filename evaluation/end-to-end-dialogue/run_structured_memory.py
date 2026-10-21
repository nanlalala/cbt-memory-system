#!/usr/bin/env python3
"""Colab-only batch preparation for research-only, unconfirmed candidate memories.

The default is an auditable verbatim-record baseline, not semantic extraction.
It never fabricates consent or promotes benchmark records to confirmed facts.
Actual model extraction is a separately frozen, question-independent stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError, URLError

from run_three_arm import (
    ROOT, SYSTEM_PROMPT, Settings, digest_bytes, fit_turns, freeze_json,
    generate_response, model_identity_policy, parse_date, partition_history,
    read_jsonl, scope_for, stable_digest, validate_provider_model,
    validated_api_base, write_jsonl,
)

for relative in (
    "modules/long-term-memory/extraction", "modules/long-term-memory/lifecycle",
    "modules/long-term-memory/storage", "modules/agent-integration",
):
    sys.path.insert(0, str(ROOT / relative))

from benchmark_adapters import RetrievalExample  # noqa: E402
from context_builder import build_context  # noqa: E402
from extraction_contract import (  # noqa: E402
    EXTRACTION_SYSTEM_PROMPT, ConversationTurn, build_extraction_request,
    parse_extraction_output,
)
from memory_manager import MemoryManager  # noqa: E402
from memory_models import MemoryRecord  # noqa: E402
from memory_retriever import MemoryRetriever, MultilingualE5Encoder, RetrievalPolicy  # noqa: E402
from sqlite_store import SQLiteMemoryStore  # noqa: E402


VERSION = "structured-candidate-three-arm-v1"
EXTRACTION_VERSION = "question-independent-candidate-extraction-v1"
ARMS = ("none", "short_term", "short_plus_structured_candidate")
CASE_FIELDS = {"example_id", "scope_id", "group_id", "query", "query_date", "sessions"}
TURN_FIELDS = {"dia_id", "role", "speaker", "text", "content"}
BOUNDARY = (
    "research-only unconfirmed candidate evidence; no user consent, no confirmation, "
    "no confirmed-memory or CBT/clinical effectiveness claim"
)
PREPROCESSING_POLICY = {
    "version": EXTRACTION_VERSION,
    "consent": "absent in external benchmark; candidates remain unconfirmed",
    "role_policy": "LongMemEval user turns only; LoCoMo peer statements preserve speaker labels",
    "gold_access": "none; source construction never receives question, answer or evidence labels",
    "source_policy": "exact user/peer quote, original speaker, source turn ID and session timestamp",
    "time_policy": "observed/created/valid-from fixed to source session; no inferred expiry",
    "retrieval_policy": "explicit research-only candidate lifecycle; production policy unchanged",
}


def require_colab() -> None:
    """Prevent production-sized batches or paid calls on a personal machine."""
    if not os.environ.get("COLAB_RELEASE_TAG") or not Path("/content").is_dir():
        raise RuntimeError("Batch/model execution is restricted to Google Colab; no local fallback")
    try:
        import google.colab  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Google Colab runtime is required; no local fallback") from exc


def file_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def streamed_digest(value: Any) -> str:
    """The exact stable_digest JSON encoding without a whole serialized copy."""
    digest = hashlib.sha256()
    for block in json.JSONEncoder(sort_keys=True, ensure_ascii=False).iterencode(value):
        digest.update(block.encode("utf-8"))
    return digest.hexdigest()


class CandidateInventoryWriter:
    """Stream JSONL and the existing canonical JSON-list fingerprint together."""

    def __init__(self, handle):
        self.handle = handle
        self.digest = hashlib.sha256(b"[")
        self.count = 0

    def add(self, row: dict) -> None:
        if self.count:
            self.digest.update(b", ")
        self.digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        if self.handle is not None:
            self.handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.count += 1

    def hexdigest(self) -> str:
        digest = self.digest.copy()
        digest.update(b"]")
        return digest.hexdigest()


def candidate_jsonl_digest(path: Path) -> tuple[str, int]:
    inventory = CandidateInventoryWriter(None)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                inventory.add(json.loads(line))
    return inventory.hexdigest(), inventory.count


def freeze_rows(path: Path, rows: Sequence[dict]) -> None:
    if path.exists():
        if read_jsonl(path) != list(rows):
            raise ValueError("frozen artifact differs; choose a new output directory")
    else:
        write_jsonl(path, rows)


def _append_row(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _request_key(request: dict) -> str:
    return request.get("source_sha256") or stable_digest([request["example_id"], request["arm"]])


def attempt_state(output_dir: Path, protocol: dict, requests: Sequence[dict], stage: str) -> tuple[dict, dict]:
    """Persist an intent before every paid call, including interrupted/unknown calls."""
    expected = {_request_key(row): row for row in requests}
    counts, failures = {}, {}
    protocol_sha = stable_digest(protocol)
    for row in read_jsonl(output_dir / f"{stage}_attempts.jsonl"):
        key = row.get("request_key")
        if (key not in expected or row.get("protocol_sha256") != protocol_sha
                or row.get("prompt_sha256") != expected[key]["prompt_sha256"]
                or row.get("attempt_index") != counts.get(key, 0) + 1
                or any(row.get(field) != expected[key].get(field) for field in ("example_id", "arm", "source_sha256"))):
            raise ValueError("attempt journal differs from frozen requests")
        counts[key] = row["attempt_index"]
        if counts[key] > protocol["maximum_attempts_per_request"]:
            raise ValueError("attempt journal exceeds frozen per-request retry cap")
    if sum(counts.values()) > protocol["maximum_total_calls"]:
        raise ValueError("attempt journal exceeds frozen total call budget")
    failure_path = output_dir / ("failures.jsonl" if stage == "response" else "extraction_failures.jsonl")
    for row in read_jsonl(failure_path):
        key, attempt = row.get("request_key"), row.get("attempt_index")
        if (key not in expected or row.get("protocol_sha256") != protocol_sha
                or type(attempt) is not int or not 1 <= attempt <= counts.get(key, 0)
                or (key, attempt) in failures
                or any(row.get(field) != expected[key].get(field) for field in ("example_id", "arm", "source_sha256"))):
            raise ValueError("failure journal differs from attempt intents")
        failures[(key, attempt)] = row
        if row.get("terminal_execution_failure"):
            raise ValueError("execution was invalidated by a recorded identity/integrity failure")
    return counts, failures


def journaled_response(output_dir: Path, protocol: dict, request: dict, settings: Settings,
                       *, key: str, stage: str, attempt_index: int,
                       expected_provider: str | None = None, validate_result=None) -> dict:
    request_key = _request_key(request)
    intent = {"request_key": request_key, "protocol_sha256": stable_digest(protocol),
              "prompt_sha256": request["prompt_sha256"], "attempt_index": attempt_index,
              "started_at_unix": time.time()}
    intent.update({field: request[field] for field in ("example_id", "arm", "source_sha256") if field in request})
    _append_row(output_dir / f"{stage}_attempts.jsonl", intent)
    try:
        result = generate_response(request, settings, api_base=protocol["api_base"], key=key)
        if expected_provider is not None and result["provider_model"] != expected_provider:
            raise ValueError("reported provider model changed across the frozen stage")
        if validate_result is not None:
            validate_result(result)
    except Exception as exc:
        terminal = isinstance(exc, ValueError)
        transient = ((isinstance(exc, HTTPError) and (exc.code == 429 or exc.code >= 500))
                     or isinstance(exc, (TimeoutError, ConnectionError))
                     or (isinstance(exc, URLError) and not isinstance(exc, HTTPError)))
        failure = {**intent, "error_type": type(exc).__name__, "retryable": bool(transient and not terminal),
                   "terminal_execution_failure": terminal}
        _append_row(output_dir / ("failures.jsonl" if stage == "response" else "extraction_failures.jsonl"), failure)
        raise
    return {**result, "attempt_index": attempt_index, "request_key": request_key}


def load_cohort(cohort_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    manifest = json.loads((cohort_dir / "cohort_manifest.json").read_text(encoding="utf-8"))
    require_workload_environment(manifest, cohort_dir)
    from prepare_formal_evaluation import validate_cohort
    manifest, cases, references = validate_cohort(cohort_dir)
    if manifest.get("three_arms") != list(ARMS):
        raise ValueError("cohort condition names do not match structured-candidate protocol")
    for case in cases:
        validate_case(case)
    return manifest, cases, references


def require_workload_environment(manifest: dict, directory: Path | None = None) -> None:
    if manifest.get("fixture_only") is not True:
        require_colab()
        return
    if manifest.get("counts", {}).get("examples", 11) > 10:
        raise ValueError("tiny fixture seam is limited to ten software examples")
    if directory is not None:
        size = sum((directory / name).stat().st_size for name in ("generation_cases.jsonl", "scoring_references.jsonl"))
        if size > 1_000_000:
            raise ValueError("tiny fixture seam is limited to one megabyte")


def validate_response_configuration(cohort: dict, settings: Settings, api_base: str) -> None:
    configuration = cohort.get("response_configuration", {})
    expected = {
        "model": settings.model, "temperature": 0,
        "max_response_tokens": settings.max_response_tokens, "thinking_mode": settings.thinking_mode,
        "long_context_maximum_characters": settings.long_chars,
        "total_context_maximum_characters": settings.max_context_chars,
        "retrieval_top_k": settings.top_k, "minimum_relevance": settings.minimum_relevance,
    }
    if any(configuration.get(name) != value for name, value in expected.items()):
        raise ValueError("runtime settings differ from frozen formal response configuration")
    if (validated_api_base(api_base) != configuration.get("api_base")
            or model_identity_policy(settings)["accepted_provider_models"] != configuration.get("accepted_provider_models")
            or configuration.get("short_window") != {"latest_eligible_sessions": 1, "maximum_turns": settings.short_turns, "maximum_characters": settings.short_chars}):
        raise ValueError("API/model aliases or short window differ from frozen formal plan")
    if settings.long_chars < 512:
        raise ValueError("candidate context budget must preserve data and usage boundaries")


def require_generation_cohort(cohort: dict) -> None:
    if cohort.get("fixture_only"):
        raise ValueError("synthetic software fixtures cannot be sent to paid benchmark generation")
    if (cohort.get("role") == "locked_external_test"
            and not cohort.get("preparation_eligible_for_confirmatory_test")):
        raise ValueError("locked external test is blocked by the formal preregistration/exposure review")


def validate_attempt_budget(protocol: dict, request_count: int) -> None:
    lock = protocol.get("preregistration_lock", {})
    retries = lock.get("plan", {}).get("completion_and_failure_policy", {}).get("transient_retry_limit")
    if (type(retries) is not int or retries < 0
            or protocol.get("maximum_attempts_per_request") != retries + 1
            or protocol.get("maximum_total_calls") != request_count * (retries + 1)):
        raise ValueError("execution call/retry budget differs from frozen preregistration")


def validate_case(case: dict) -> None:
    if set(case) != CASE_FIELDS or not all(isinstance(case.get(name), str) and case[name] for name in ("example_id", "scope_id", "group_id", "query")):
        raise ValueError("generation case must use the exact gold-free case schema")
    if not isinstance(case["sessions"], list):
        raise ValueError("sessions must be an array")
    for session in case["sessions"]:
        if set(session) != {"session_id", "date", "turns"} or not isinstance(session["turns"], list):
            raise ValueError("session schema contains unexpected metadata or labels")
        parse_date(session["date"])
        for turn in session["turns"]:
            if not isinstance(turn, dict) or not set(turn).issubset(TURN_FIELDS):
                raise ValueError("turn contains unexpected metadata or gold/evidence labels")
            text = turn.get("content", turn.get("text"))
            if not isinstance(text, str):
                raise ValueError("every turn must have text content")


def split_case(case: dict, dataset_key: str, settings: Settings):
    validate_case(case)
    dataset = "locomo" if dataset_key == "locomo10" else "longmemeval"
    # Deliberately do not construct this object from scoring_references.
    item = RetrievalExample(
        example_id=case["example_id"], category="not_used_for_generation",
        query=case["query"], answer="", evidence_ids=(),
        sessions=tuple(case["sessions"]), query_date=case["query_date"],
    )
    if case["scope_id"] != scope_for(item, dataset):
        raise ValueError("case scope does not match benchmark subject")
    return partition_history(item, dataset, settings)


def source_for(turns, dataset_key: str) -> dict:
    """Only source history enters this function; never a question or reference."""
    if not turns or len({(turn.scope_id, turn.session_id, turn.date) for turn in turns}) != 1:
        raise ValueError("a source must describe exactly one scoped session")
    first = turns[0]
    allowed = turns if dataset_key == "locomo10" else [turn for turn in turns if turn.speaker == "user"]
    source = {
        "scope_id": first.scope_id, "session_id": first.session_id,
        "observed_at": parse_date(first.date).isoformat().replace("+00:00", "Z"),
        "turns": [{"turn_id": turn.document_id, "speaker": turn.speaker, "text": turn.text} for turn in allowed if turn.text.strip()],
    }
    source["source_sha256"] = stable_digest(source)
    return source


def source_inventory(cases: Sequence[dict], dataset_key: str, settings: Settings) -> dict[str, dict]:
    sources: dict[str, dict] = {}
    for case in cases:
        _, older, _ = split_case(case, dataset_key, settings)
        by_session = defaultdict(list)
        for turn in older:
            by_session[turn.session_id].append(turn)
        for turns in by_session.values():
            source = source_for(turns, dataset_key)
            if source["turns"]:
                sources[source["source_sha256"]] = source
    return dict(sorted(sources.items(), key=lambda item: (
        item[1]["scope_id"], item[1]["observed_at"], item[1]["session_id"], item[0],
    )))


def extraction_messages(source: dict) -> list[dict]:
    turns = [ConversationTurn(row["turn_id"], "user", f'{row["speaker"]}: {row["text"]}') for row in source["turns"]]
    return [{"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": build_extraction_request(turns)}]


def extraction_requests(sources: dict[str, dict]) -> list[dict]:
    return [{"source_sha256": key, "source": source, "messages": extraction_messages(source),
             "prompt_sha256": stable_digest(extraction_messages(source))} for key, source in sources.items()]


def verbatim_output(source: dict) -> str:
    return json.dumps({"candidates": [
        {"memory_type": "explicit_fact", "content": f'{row["speaker"]}: {row["text"]}',
         "source_turn_ids": [row["turn_id"]], "supporting_text": f'{row["speaker"]}: {row["text"]}',
         "confidence": 0.5, "importance": 0.5}
        for row in source["turns"]
    ]}, ensure_ascii=False)


def parse_source_candidates(source: dict, raw_output: str) -> list[MemoryRecord]:
    payload = json.loads(raw_output)
    if not isinstance(payload, dict) or not isinstance(payload.get("candidates"), list):
        raise ValueError("extraction output must contain candidates")
    for index, item in enumerate(payload["candidates"]):
        # The extractor never chooses identity, ownership, time validity or consent.
        item["memory_id"] = f'candidate-{source["source_sha256"]}-{index}'
        item["observed_at"] = source["observed_at"]
        item["valid_from"] = source["observed_at"]
        item["valid_to"] = None
    turns = [ConversationTurn(row["turn_id"], "user", f'{row["speaker"]}: {row["text"]}') for row in source["turns"]]
    records = parse_extraction_output(
        json.dumps(payload, ensure_ascii=False), user_id=source["scope_id"],
        session_id=source["session_id"], turns=turns, extracted_at=source["observed_at"],
    )
    checked = []
    for record in records:
        # Normalize all enums/types through the full schema validator as well.
        record = MemoryRecord.from_dict(record.to_dict())
        if record.lifecycle_status != "candidate" or record.confirmation_state != "unconfirmed":
            raise ValueError("benchmark extraction cannot fabricate confirmation")
        checked.append(record)
    return checked


def ingest_source(store: SQLiteMemoryStore, source: dict, raw_output: str) -> list[MemoryRecord]:
    manager = MemoryManager(store)
    inserted = []
    for record in parse_source_candidates(source, raw_output):
        outcome = manager.ingest(record, actor="external_benchmark_candidate_not_user_consent")
        if outcome.action != "duplicate_rejected":
            inserted.append(record)
    return inserted


def ingest_verbatim_scope(store: SQLiteMemoryStore, sources: Sequence[dict]) -> list[MemoryRecord]:
    """One bounded scope; the manager independently validates the narrow import."""
    records, source_turns = [], {}
    scope_ids = {source["scope_id"] for source in sources}
    if len(scope_ids) != 1:
        raise ValueError("verbatim import must contain exactly one scope")
    for source in sources:
        records.extend(parse_source_candidates(source, verbatim_output(source)))
        for turn in source["turns"]:
            key = (source["session_id"], turn["turn_id"])
            value = {"text": f'{turn["speaker"]}: {turn["text"]}'.strip(),
                     "observed_at": source["observed_at"]}
            if key in source_turns and source_turns[key] != value:
                raise ValueError("a source turn cannot change text or timestamp within a scope")
            source_turns[key] = value
    results = MemoryManager(store).ingest_verbatim_candidates(
        records, user_id=next(iter(scope_ids)), source_turns=source_turns,
        actor="external_benchmark_candidate_not_user_consent",
    )
    return [record for record, result in zip(records, results) if result.action != "duplicate_rejected"]


def prepare_extraction_requests(cohort_dir: Path, output_dir: Path, settings: Settings, api_base: str) -> dict:
    manifest, cases, _ = load_cohort(cohort_dir)
    validate_response_configuration(manifest, settings, api_base)
    lock = json.loads((cohort_dir / "preregistration_lock.json").read_text(encoding="utf-8"))
    sources = source_inventory(cases, manifest["dataset_key"], settings)
    requests = extraction_requests(sources)
    if any(sum(len(message["content"]) for message in request["messages"]) > settings.max_context_chars for request in requests):
        raise ValueError("an extraction session exceeds the frozen input budget; use a separately declared splitting policy")
    protocol = {
        "protocol_version": EXTRACTION_VERSION, "cohort_sha256": manifest["cohort_sha256"],
        "cohort_manifest": manifest,
        "preregistration_lock": lock,
        "source_inventory_sha256": stable_digest(sources), "requests_sha256": stable_digest(requests),
        "model_identity_policy": model_identity_policy(settings), "settings": asdict(settings),
        "api_base": validated_api_base(api_base), "temperature": 0,
        "preprocessing_policy": PREPROCESSING_POLICY, "failure_policy": "stop; preserve failures and successful checkpoints; no post-hoc sample deletion",
        "maximum_attempts_per_request": 1 + lock["plan"]["completion_and_failure_policy"]["transient_retry_limit"],
        "maximum_total_calls": len(requests) * (1 + lock["plan"]["completion_and_failure_policy"]["transient_retry_limit"]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    freeze_json(output_dir / "extraction_protocol.json", protocol)
    freeze_rows(output_dir / "extraction_requests.jsonl", requests)
    return protocol


def execute_extraction(output_dir: Path, *, max_calls: int) -> dict:
    require_colab()
    protocol = json.loads((output_dir / "extraction_protocol.json").read_text())
    require_generation_cohort(protocol["cohort_manifest"])
    requests = read_jsonl(output_dir / "extraction_requests.jsonl")
    if protocol.get("protocol_version") != EXTRACTION_VERSION or stable_digest(requests) != protocol["requests_sha256"]:
        raise ValueError("extraction request/protocol mismatch")
    settings = Settings(**protocol["settings"])
    if protocol["model_identity_policy"] != model_identity_policy(settings):
        raise ValueError("extraction identity policy changed")
    validate_response_configuration(protocol["cohort_manifest"], settings, protocol["api_base"])
    validate_attempt_budget(protocol, len(requests))
    if any(sum(len(message["content"]) for message in request["messages"]) > settings.max_context_chars for request in requests):
        raise ValueError("extraction request exceeds frozen input-character budget")
    protocol_sha = stable_digest(protocol)
    completed = {}
    for row in read_jsonl(output_dir / "extraction_responses.jsonl"):
        key = row["source_sha256"]
        if key in completed or row.get("protocol_sha256") != protocol_sha:
            raise ValueError("invalid extraction checkpoint")
        validate_provider_model(row.get("provider_model"), protocol["model_identity_policy"])
        completed[key] = row
    expected = {row["source_sha256"]: row for row in requests}
    if not set(completed).issubset(expected):
        raise ValueError("unknown extraction checkpoint source")
    for key, row in completed.items():
        if row.get("prompt_sha256") != expected[key]["prompt_sha256"]:
            raise ValueError("extraction checkpoint prompt mismatch")
    if len({row["provider_model"] for row in completed.values()}) > 1:
        raise ValueError("extraction reported provider model changed")
    counts, failures = attempt_state(output_dir, protocol, requests, "extraction")
    for source_key, row in completed.items():
        if (row.get("attempt_index", 0) > counts.get(source_key, 0)
                or row.get("attempt_index", 0) <= 0
                or (source_key, row["attempt_index"]) in failures):
            raise ValueError("extraction checkpoint lacks a valid started attempt")
    pending = [row for row in requests if row["source_sha256"] not in completed]
    if max_calls <= 0:
        raise ValueError("set a positive explicit max-calls budget")
    key = os.environ.get("CBT_EVAL_API_KEY", "")
    if pending and not key:
        raise RuntimeError("CBT_EVAL_API_KEY must come from Colab Secrets/environment")
    calls = 0
    expected_provider = next((row["provider_model"] for row in completed.values()), None)
    with (output_dir / "extraction_responses.jsonl").open("a", encoding="utf-8") as handle:
        for request in pending:
            source_key = request["source_sha256"]
            previous = counts.get(source_key, 0)
            failure = failures.get((source_key, previous))
            if previous >= protocol["maximum_attempts_per_request"] or (failure and not failure.get("retryable")):
                continue
            if calls >= max_calls:
                break
            def validate_result(result):
                store = SQLiteMemoryStore()
                try:
                    ingest_source(store, request["source"], result["hypothesis"])
                finally:
                    store.close()
            result = journaled_response(output_dir, protocol, request, settings, key=key,
                                        stage="extraction", attempt_index=previous + 1,
                                        expected_provider=expected_provider, validate_result=validate_result)
            calls += 1
            counts[source_key] = previous + 1
            expected_provider = result["provider_model"]
            result.update({"source_sha256": request["source_sha256"], "prompt_sha256": request["prompt_sha256"], "protocol_sha256": protocol_sha})
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            completed[request["source_sha256"]] = result
    if len(completed) != len(requests):
        return {"sessions": len(requests), "completed": len(completed), "pending": len(requests) - len(completed),
                "status": "partial_checkpoint_only_no_extraction_artifact", "model_calls_this_invocation": calls}
    artifact = {"schema_version": EXTRACTION_VERSION, "protocol": protocol,
                "protocol_sha256": protocol_sha,
                "sessions": [{"source_sha256": request["source_sha256"], "prompt_sha256": request["prompt_sha256"],
                              "provider_model": completed[request["source_sha256"]]["provider_model"],
                              "raw_output": completed[request["source_sha256"]]["hypothesis"]} for request in requests]}
    freeze_json(output_dir / "extraction_artifact.json", artifact)
    return {"sessions": len(requests), "completed": len(completed), "artifact": str(output_dir / "extraction_artifact.json"), "consent": "not present; no confirmation performed"}


def validate_extraction_artifact(path: Path, sources: dict[str, dict], cohort: dict) -> dict[str, str]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    protocol = artifact.get("protocol", {})
    if (artifact.get("schema_version") != EXTRACTION_VERSION
            or protocol.get("protocol_version") != EXTRACTION_VERSION
            or artifact.get("protocol_sha256") != stable_digest(protocol)
            or protocol.get("cohort_sha256") != cohort["cohort_sha256"]
            or protocol.get("source_inventory_sha256") != stable_digest(sources)
            or protocol.get("requests_sha256") != stable_digest(extraction_requests(sources))
            or protocol.get("preprocessing_policy") != PREPROCESSING_POLICY):
        raise ValueError("extraction artifact does not match frozen cohort/source/preprocessing")
    settings = Settings(**protocol["settings"])
    if protocol.get("model_identity_policy") != model_identity_policy(settings):
        raise ValueError("invalid extraction model identity policy")
    validate_response_configuration(cohort, settings, protocol["api_base"])
    validate_attempt_budget(protocol, len(sources))
    sessions = artifact.get("sessions", [])
    if len(sessions) != len(sources) or {row["source_sha256"] for row in sessions} != set(sources):
        raise ValueError("extraction artifact must cover each source exactly once")
    outputs = {}
    for row in sessions:
        key = row["source_sha256"]
        if row.get("prompt_sha256") != stable_digest(extraction_messages(sources[key])):
            raise ValueError("extraction input differs from source-only prompt")
        validate_provider_model(row.get("provider_model"), protocol["model_identity_policy"])
        outputs[key] = row["raw_output"]
    if len({row["provider_model"] for row in sessions}) > 1:
        raise ValueError("extraction provider model changed across sessions")
    return outputs


class ScopeCachedEncoder:
    """Encode unchanged documents once per contiguous subject, not once per QA.

    Only one subject's vectors are retained, bounding memory on full LongMemEval-S.
    Dataset preparation keeps each subject contiguous. Model identity is unchanged.
    """

    def __init__(self, encoder):
        self.encoder = encoder
        self.model_identity = encoder.model_identity
        self.scope_id = None
        self.document_vectors = {}

    def set_scope(self, scope_id: str) -> None:
        if scope_id != self.scope_id:
            self.scope_id = scope_id
            self.document_vectors.clear()

    def encode(self, texts, *, is_query: bool):
        if is_query:
            return self.encoder.encode(texts, is_query=True)
        missing = list(dict.fromkeys(text for text in texts if text not in self.document_vectors))
        if missing:
            vectors = self.encoder.encode(missing, is_query=False)
            if len(vectors) != len(missing):
                raise ValueError("dense encoder returned the wrong document count")
            self.document_vectors.update(zip(missing, vectors))
        return [self.document_vectors[text] for text in texts]


def make_request(case: dict, dataset_key: str, arm: str, settings: Settings,
                 records: Sequence[MemoryRecord], *, encoder=None) -> dict:
    if arm not in ARMS:
        raise ValueError("unknown structured candidate arm")
    short, older, excluded = split_case(case, dataset_key, settings)
    eligible_source_ids = {turn.document_id for turn in older}
    eligible = []
    for record in records:
        if record.user_id != case["scope_id"]:
            raise ValueError("cross-subject candidate retrieval is prohibited")
        if not set(record.source_evidence.turn_ids).issubset(eligible_source_ids):
            continue
        if record.lifecycle_status != "candidate" or record.confirmation_state != "unconfirmed":
            raise ValueError("research candidate input cannot claim real user confirmation")
        eligible.append(record)
    selected_short = short if arm != "none" else []
    _, selected_short_ids, dropped_short_ids = fit_turns(list(reversed(selected_short)), settings.short_chars)
    short_text, short_ids, _ = fit_turns([turn for turn in selected_short if turn.document_id in selected_short_ids], settings.short_chars)
    policy = RetrievalPolicy(
        top_k=settings.top_k, minimum_relevance=settings.minimum_relevance,
        lexical_weight=0.0 if settings.backend == "dense" else (0.45 if encoder else 1.0),
        dense_weight=1.0 if settings.backend == "dense" else 0.55,
        recency_weight=0.0, importance_weight=0.0, confidence_weight=0.0,
        allowed_lifecycle=frozenset({"candidate"}),
    )
    as_of = case["query_date"] or (max((parse_date(turn.date) for turn in [*short, *older]), default=parse_date("2000-01-01T00:00:00Z")).isoformat())
    retrieved = MemoryRetriever(eligible, encoder=encoder, policy=policy).search(
        case["query"], user_id=case["scope_id"], as_of=parse_date(as_of).isoformat(),
    ) if arm == ARMS[2] else []
    # Same wrapper and budgets for all arms; only available history differs.
    # Keep the entire short window identical in the short and treatment arms.
    # The context builder budgets the separate candidate channel; it must not
    # re-truncate the short window depending on whether retrieval found a hit.
    context = build_context(short_term_summary="Short history is in the separate short_history block.",
                            personal_memories=retrieved, max_personal=settings.top_k,
                            max_chars=settings.long_chars)
    from html import escape
    user_prompt = (f'<current_question>{escape(case["query"], quote=True)}</current_question>\n'
                   f'<question_date>{escape(case["query_date"] or "QA after complete conversation", quote=True)}</question_date>\n'
                   f'<research_boundary>{BOUNDARY}</research_boundary>\n'
                   f'<short_history data_only="true">\n{short_text}\n</short_history>\n' + context.prompt_context)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
    if sum(len(message["content"]) for message in messages) > settings.max_context_chars:
        raise ValueError("context exceeds total fixed budget")
    included = set(context.memory_ids)
    return {"example_id": case["example_id"], "scope_id": case["scope_id"], "group_id": case["group_id"],
            "arm": arm, "messages": messages, "prompt_sha256": stable_digest(messages),
            "included_short_ids": short_ids, "dropped_short_ids": dropped_short_ids,
            "included_memory_ids": list(context.memory_ids), "dropped_memory_ids": list(context.dropped_memory_ids),
            "included_source_turn_ids": sorted({turn_id for item in retrieved if item.record.memory_id in included for turn_id in item.record.source_evidence.turn_ids}),
            "future_session_ids_excluded": excluded, "memory_boundary": BOUNDARY}


def prepare_structured_run(cohort_dir: Path, output_dir: Path, settings: Settings, api_base: str,
                           *, extraction_artifact: Path | None = None, device: str | None = None) -> dict:
    settings.validate()
    cohort, cases, references = load_cohort(cohort_dir)
    validate_response_configuration(cohort, settings, api_base)
    lock = json.loads((cohort_dir / "preregistration_lock.json").read_text(encoding="utf-8"))
    sources = source_inventory(cases, cohort["dataset_key"], settings)
    outputs = validate_extraction_artifact(extraction_artifact, sources, cohort) if extraction_artifact else None
    # Only one scope's SQLite rows, record objects and document vectors coexist.
    # The source inventory contains references to input strings; it is not copied
    # into a second all-subject candidate inventory.
    cases_by_scope, sources_by_scope = defaultdict(list), defaultdict(list)
    for case in cases:
        cases_by_scope[case["scope_id"]].append(case)
    for source in sources.values():
        sources_by_scope[source["scope_id"]].append(source)
    encoder = None
    if settings.backend != "lexical":
        require_colab()
        encoder = ScopeCachedEncoder(MultilingualE5Encoder(settings.dense_model, device=device))
    requests = []
    ordering = random.Random(settings.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_dir,
                                           prefix="candidate_memories-", suffix=".tmp", delete=False)
    temporary_path = Path(temporary.name)
    try:
        with temporary as handle:
            inventory = CandidateInventoryWriter(handle)
            for scope in sorted(cases_by_scope):
                store = SQLiteMemoryStore()
                try:
                    scoped_sources = sources_by_scope[scope]
                    if outputs is None:
                        records = ingest_verbatim_scope(store, scoped_sources) if scoped_sources else []
                    else:
                        # Arbitrary model summaries retain the ordinary manager
                        # decisions; only the exact-quote baseline is batched.
                        records = []
                        for source in scoped_sources:
                            records.extend(ingest_source(store, source, outputs[source["source_sha256"]]))
                    if encoder is not None:
                        encoder.set_scope(scope)
                    for case in sorted(cases_by_scope[scope], key=lambda row: row["example_id"]):
                        arms = list(ARMS)
                        ordering.shuffle(arms)
                        requests.extend(make_request(case, cohort["dataset_key"], arm, settings,
                                                     records, encoder=encoder) for arm in arms)
                    for record in records:
                        inventory.add({"scope_id": scope, "record": record.to_dict()})
                finally:
                    store.close()
                # Do not retain the previous scope while constructing the next.
                del records
            candidate_count = inventory.count
            candidate_sha256 = inventory.hexdigest()
            handle.flush()
            os.fsync(handle.fileno())
        code_paths = [Path(__file__), ROOT / "evaluation/end-to-end-dialogue/run_three_arm.py",
                      ROOT / "modules/long-term-memory/extraction/extraction_contract.py",
                      ROOT / "modules/long-term-memory/lifecycle/memory_manager.py",
                      ROOT / "modules/long-term-memory/retrieval/memory_retriever.py",
                      ROOT / "modules/agent-integration/context_builder.py"]
        protocol = {
            "protocol_version": VERSION, "dataset_key": cohort["dataset_key"],
            "cohort_manifest": cohort, "cohort_manifest_sha256": stable_digest(cohort),
            "preregistration_lock": lock, "preregistration_lock_sha256": stable_digest(lock),
            "question_ids": [case["example_id"] for case in cases], "conditions": list(ARMS),
            "settings": {**asdict(settings), "accepted_provider_models": list(settings.accepted_provider_models)},
            "model_identity_policy": model_identity_policy(settings), "api_base": validated_api_base(api_base),
            "temperature": 0, "system_prompt": SYSTEM_PROMPT,
            "memory_boundary": BOUNDARY, "preprocessing_policy": PREPROCESSING_POLICY,
            "extraction_mode": "model_extracted_unconfirmed_candidates" if outputs is not None else "verbatim_candidate_baseline_not_semantic_extraction",
            "extraction_artifact": file_identity(extraction_artifact) if extraction_artifact else None,
            "source_inventory_sha256": streamed_digest(sources), "candidate_records_sha256": candidate_sha256,
            "prompt_records_sha256": stable_digest(requests), "references_sha256": stable_digest(references),
            "retrieval_model_identity": getattr(encoder, "model_identity", None),
            "fixed_knowledge": "empty in all arms; governed sources unchanged",
            "failure_policy": "stop and retain checkpoints; incomplete runs cannot produce a formal summary; no failed example deletion",
            "code_sha256": {str(path.relative_to(ROOT)): digest_bytes(path.read_bytes()) for path in code_paths},
            "official_evaluator_status": "not executed", "model_judge_status": "not executed",
            "effectiveness_claim": "none; preparation is not a completed experiment",
            "maximum_attempts_per_request": 1 + lock["plan"]["completion_and_failure_policy"]["transient_retry_limit"],
            "maximum_total_calls": len(requests) * (1 + lock["plan"]["completion_and_failure_policy"]["transient_retry_limit"]),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        freeze_json(output_dir / "protocol.json", protocol)
        freeze_rows(output_dir / "prompts.jsonl", requests)
        freeze_rows(output_dir / "references.jsonl", references)
        candidate_path = output_dir / "candidate_memories.jsonl"
        if candidate_path.exists():
            if candidate_jsonl_digest(candidate_path) != (candidate_sha256, candidate_count):
                raise ValueError("frozen artifact differs; choose a new output directory")
        else:
            os.replace(temporary_path, candidate_path)
        summary = {"status": "prepared", "examples": len(cases), "requests": len(requests),
                   "candidate_records": candidate_count, "preprocessing_sessions": len(sources),
                   "protocol_sha256": stable_digest(protocol), "extraction_mode": protocol["extraction_mode"],
                   "memory_boundary": BOUNDARY, "model_calls": 0, "effectiveness_result": "not measured"}
        (output_dir / "preparation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return summary
    finally:
        temporary_path.unlink(missing_ok=True)


def validate_protocol(protocol: dict) -> Settings:
    if protocol.get("protocol_version") != VERSION or protocol.get("conditions") != list(ARMS):
        raise ValueError("unsupported structured candidate protocol")
    settings = Settings(**protocol["settings"])
    settings.validate()
    if protocol.get("model_identity_policy") != model_identity_policy(settings):
        raise ValueError("model identity differs from frozen settings")
    if protocol.get("memory_boundary") != BOUNDARY or protocol.get("preprocessing_policy") != PREPROCESSING_POLICY:
        raise ValueError("candidate/confirmation policy differs from frozen protocol")
    if (protocol.get("system_prompt") != SYSTEM_PROMPT or protocol.get("temperature") != 0
            or protocol.get("cohort_manifest_sha256") != stable_digest(protocol.get("cohort_manifest"))):
        raise ValueError("system prompt, temperature or cohort manifest differs from protocol")
    lock = protocol.get("preregistration_lock", {})
    if (protocol.get("preregistration_lock_sha256") != stable_digest(lock)
            or stable_digest(lock.get("plan")) != protocol["cohort_manifest"].get("plan_sha256")):
        raise ValueError("formal preregistration lock differs from cohort")
    validate_response_configuration(protocol["cohort_manifest"], settings, protocol["api_base"])
    validate_attempt_budget(protocol, len(protocol["question_ids"]) * len(ARMS))
    return settings


def load_run(run_dir: Path) -> tuple[dict, list[dict], list[dict], dict]:
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    validate_protocol(protocol)
    require_workload_environment(protocol["cohort_manifest"])
    prompts = read_jsonl(run_dir / "prompts.jsonl")
    references = read_jsonl(run_dir / "references.jsonl")
    reference_spec = protocol["cohort_manifest"]["files"]["scoring_references.jsonl"]
    if any(reference_spec.get(field) != value for field, value in file_identity(run_dir / "references.jsonl").items()):
        raise ValueError("run reference bytes no longer match the frozen complete cohort")
    for name, rows in (("prompt_records_sha256", prompts), ("references_sha256", references)):
        if stable_digest(rows) != protocol.get(name):
            raise ValueError("run artifact checksum differs from frozen protocol")
    if candidate_jsonl_digest(run_dir / "candidate_memories.jsonl")[0] != protocol.get("candidate_records_sha256"):
        raise ValueError("run artifact checksum differs from frozen protocol")
    expected = {(example_id, arm) for example_id in protocol["question_ids"] for arm in ARMS}
    actual = {(row["example_id"], row["arm"]) for row in prompts}
    if actual != expected or len(prompts) != len(expected):
        raise ValueError("prompt grid must contain all three conditions exactly once")
    if len(references) != len(protocol["question_ids"]) or {row["example_id"] for row in references} != set(protocol["question_ids"]):
        raise ValueError("references must match all frozen questions exactly once")
    if len(references) != protocol["cohort_manifest"]["counts"]["examples"]:
        raise ValueError("run must retain the entire predeclared cohort")
    for row in prompts:
        if row.get("prompt_sha256") != stable_digest(row["messages"]):
            raise ValueError("individual prompt checksum mismatch")
    by_key = {(row["example_id"], row["arm"]): row for row in prompts}
    counts, failures = attempt_state(run_dir, protocol, prompts, "response")
    completed = {}
    for row in read_jsonl(run_dir / "responses.jsonl"):
        key = (row["example_id"], row["arm"])
        if key in completed or key not in expected or row.get("protocol_sha256") != stable_digest(protocol) or row.get("prompt_sha256") != by_key[key]["prompt_sha256"]:
            raise ValueError("response checkpoint does not match frozen request")
        if not isinstance(row.get("hypothesis"), str) or not row["hypothesis"].strip() or row.get("finish_reason") != "stop":
            raise ValueError("response checkpoint is empty or unfinished")
        latency = row.get("latency_seconds")
        if type(latency) not in {int, float} or not math.isfinite(latency) or latency < 0:
            raise ValueError("response latency must be a nonnegative finite number")
        usage = row.get("usage")
        if not isinstance(usage, dict) or any(type(usage.get(name, 0)) is not int or usage.get(name, 0) < 0
                                              for name in ("prompt_tokens", "completion_tokens", "total_tokens")):
            raise ValueError("response token counts must be nonnegative integers")
        validate_provider_model(row.get("provider_model"), protocol["model_identity_policy"])
        request_key = _request_key(row)
        if (not protocol["cohort_manifest"].get("fixture_only")
                and (row.get("attempt_index", 0) > counts.get(request_key, 0)
                     or row.get("attempt_index", 0) <= 0
                     or (request_key, row["attempt_index"]) in failures)):
            raise ValueError("response checkpoint lacks a valid started attempt")
        completed[key] = row
    if len({row["provider_model"] for row in completed.values()}) > 1:
        raise ValueError("reported provider model changed across conditions")
    return protocol, prompts, references, completed


def generate_run(run_dir: Path, *, max_calls: int) -> dict:
    require_colab()
    protocol, prompts, _, completed = load_run(run_dir)
    require_generation_cohort(protocol["cohort_manifest"])
    settings = validate_protocol(protocol)
    pending = [row for row in prompts if (row["example_id"], row["arm"]) not in completed]
    if max_calls <= 0:
        raise ValueError("set a positive explicit max-calls budget")
    key = os.environ.get("CBT_EVAL_API_KEY", "")
    if pending and not key:
        raise RuntimeError("CBT_EVAL_API_KEY must come from Colab Secrets/environment")
    counts, failures = attempt_state(run_dir, protocol, prompts, "response")
    calls = 0
    expected_provider = next((row["provider_model"] for row in completed.values()), None)
    with (run_dir / "responses.jsonl").open("a", encoding="utf-8") as handle:
        for request in pending:
            request_key = _request_key(request)
            previous = counts.get(request_key, 0)
            failure = failures.get((request_key, previous))
            if previous >= protocol["maximum_attempts_per_request"] or (failure and not failure.get("retryable")):
                continue
            if calls >= max_calls:
                break
            result = journaled_response(run_dir, protocol, request, settings, key=key,
                                        stage="response", attempt_index=previous + 1,
                                        expected_provider=expected_provider)
            calls += 1
            counts[request_key] = previous + 1
            expected_provider = result["provider_model"]
            result.update({name: request[name] for name in ("example_id", "arm", "prompt_sha256")})
            result["protocol_sha256"] = stable_digest(protocol)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            completed[(request["example_id"], request["arm"])] = result
    return {"completed_responses": len(completed), "requests": len(prompts),
            "pending": len(prompts) - len(completed), "model_calls_this_invocation": calls,
            "status": "complete" if len(completed) == len(prompts) else "partial_checkpoint_only_no_effectiveness_result",
            "memory_boundary": BOUNDARY}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--accepted-provider-model", dest="accepted_provider_models", action="append", default=[])
    parser.add_argument("--backend", choices=("lexical", "dense", "hybrid"), default="lexical")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-response-tokens", type=int, default=4096)
    parser.add_argument("--thinking-mode", choices=("provider_default", "enabled", "disabled"), default="disabled")
    parser.add_argument("--extraction-artifact", type=Path)
    parser.add_argument("--prepare-extraction", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--max-calls", type=int, default=0)
    args = parser.parse_args()
    require_colab()
    settings = Settings(model=args.model, accepted_provider_models=tuple(args.accepted_provider_models),
                        backend=args.backend, max_response_tokens=args.max_response_tokens, thinking_mode=args.thinking_mode)
    api_base = validated_api_base(os.environ.get("CBT_EVAL_API_BASE", "https://api.openai.com/v1"))
    if args.extract and not args.prepare_extraction:
        parser.error("--extract requires --prepare-extraction to freeze source-only requests first")
    if args.prepare_extraction:
        protocol = prepare_extraction_requests(args.cohort_dir, args.output_dir, settings, api_base)
        print(json.dumps({"extraction_requests_sha256": protocol["requests_sha256"], "generated": False}))
        if args.extract:
            print(json.dumps(execute_extraction(args.output_dir, max_calls=args.max_calls), ensure_ascii=False, indent=2))
        return
    print(json.dumps(prepare_structured_run(args.cohort_dir, args.output_dir, settings, api_base,
                                          extraction_artifact=args.extraction_artifact, device=args.device), ensure_ascii=False, indent=2))
    if args.generate:
        print(json.dumps(generate_run(args.output_dir, max_calls=args.max_calls), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
