"""Bounded hosted-Colab source-only semantic-index stage, separate from QA.

No scoring references or model answers are read. Compact case metadata (which
can contain questions) is used only to prove which eligible source documents a
case owns; questions are never supplied to the extractor.
Exact source quotes authenticate provenance, not entailment of a model summary.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from urllib.error import HTTPError, URLError

import run_objective_comparison as qa
from execution_lock import exclusive_directories
from semantic_source_index import SYSTEM_PROMPT, validate_index

VERSION = "source-only-semantic-extraction-visible-history-v2"
MAX_ATTEMPTS = 2
FROZEN = ("sources.jsonl", "case_sources.jsonl", "extraction_requests.jsonl")
HERE = Path(__file__).resolve().parent


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _codes():
    paths = [Path(__file__), HERE / "semantic_source_index.py"]
    return {**qa._code_hashes(), **{str(p.relative_to(qa.ROOT)): _sha(p) for p in paths}}


def _rows(path):
    return qa.read_jsonl(Path(path))


def _write(path, rows):
    qa._replace_rows(Path(path), rows)


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def _visibility_identity(subject, sources):
    """Bind an extractor's complete visible history, not just cited sources."""
    return qa.stable_digest({"subject_id": subject, "visible_source_versions": sources})


def _source_inputs(source_run, case_ids=None):
    """Validate only protocol/inventory/cases; never open references or prompts."""
    source_run = Path(source_run)
    protocol = json.loads((source_run / "protocol.json").read_text())
    identity = protocol.get("protocol_sha256")
    if identity != qa.stable_digest({k: v for k, v in protocol.items() if k != "protocol_sha256"}):
        raise ValueError("source protocol digest mismatch")
    for name in ("inventory.jsonl", "cases.jsonl"):
        if _sha(source_run / name) != protocol["file_sha256"].get(name):
            raise ValueError("source-only input hash mismatch")
    inventory, cases = _rows(source_run / "inventory.jsonl"), _rows(source_run / "cases.jsonl")
    if [c["case_id"] for c in cases] != protocol["case_ids"]:
        raise ValueError("source case identities differ from protocol")
    if case_ids is not None:
        if not isinstance(case_ids, list) or not case_ids or len(set(case_ids)) != len(case_ids) or not set(case_ids) <= set(protocol["case_ids"]):
            raise ValueError("selection must contain unique existing source case IDs")
        cases = [c for c in cases if c["case_id"] in set(case_ids)]
    selected_subjects = {c["subject_id"] for c in cases}
    sources, by_subject = {}, {}
    for item in inventory:
        row, subject = item["source"], item["subject_id"]
        if subject not in selected_subjects:
            continue
        doc = qa._document({"subject_id": subject}, row)
        if doc.document_id != item["document_id"]:
            raise ValueError("source document ID mismatch")
        normalized = {"subject_id": subject, "document_id": doc.document_id, "source": row}
        if doc.document_id in sources and sources[doc.document_id] != normalized:
            raise ValueError("ambiguous source versions in inventory")
        sources[doc.document_id] = normalized
        by_subject.setdefault(subject, {})[doc.document_id] = row
    mappings, needed = [], set()
    for case in cases:
        subject = case["subject_id"]
        history = list(by_subject.get(subject, {}).values())
        if case.get("dataset_key") == "memoryagentbench_conflict" and case.get("history_scope_id"):
            # The adapter groups all variants as one statistical family but
            # stores each released context under its exact source scope.
            session_id = case["history_scope_id"] + ":released-sequence"
            history = [row for row in history if row["session_id"] == session_id]
        cutoff = case.get("visible_context_end_exclusive")
        if cutoff is not None:
            if type(cutoff) is not int or cutoff <= 0:
                raise ValueError("visible history prefix must have a positive exclusive boundary")
            if case.get("dataset_key") == "personamem_v1":
                context = case.get("source_context_id")
                if not isinstance(context, str) or not context:
                    raise ValueError("Persona visible prefix requires its released source context")
                history = [row for row in history if row["session_id"] == f"persona32k:{context}:released-sequence"]
            history = [row for row in history if row["order"] < cutoff]
        eligible, _ = qa._eligible({"history": history, "as_of": case.get("as_of")})
        # Refuse to guess at varying released prefix boundaries. This exact
        # history hash proves the reconstructed source set is the prepared one.
        if qa.stable_digest([subject, eligible]) != case["history_inventory_sha256"]:
            raise ValueError("cannot reconstruct exact visible history for case; use a cohort with one verifiable history per subject")
        ids = [qa._document({"subject_id": subject}, row).document_id for row in eligible]
        visible_sources = [sources[key] for key in ids]
        mappings.append({"case_id": case["case_id"], "subject_id": subject,
                         "history_inventory_sha256": case["history_inventory_sha256"], "source_document_ids": ids,
                         "visibility_sha256": _visibility_identity(subject, visible_sources)})
        needed.update(ids)
    selected = [item for key, item in sources.items() if key in needed]
    selected.sort(key=lambda x: (x["subject_id"], x["source"]["session_id"], x["source"]["order"], x["source"]["turn_id"]))
    return protocol, selected, mappings


def _requests(sources, chunk_chars, *, visibility_sha256=None):
    requests, current, subject, scope = [], [], None, None
    visibility_sha256 = visibility_sha256 or qa.stable_digest({"standalone_sources": sources})

    def payload(items):
        return json.dumps({"sources": [{"source_document_id": x["document_id"], "source_role": x["source"]["role"],
                            "date": x["source"].get("observed_at") or f"source_order:{x['source']['order']}",
                            "text": x["source"]["text"].strip()} for x in items]}, ensure_ascii=False)

    def finish():
        if current:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": payload(current)}]
            key = qa.stable_digest({"visibility_sha256": visibility_sha256, "messages": messages})
            requests.append({"request_key": key, "prompt_sha256": qa.stable_digest(messages), "subject_id": subject,
                             "visibility_sha256": visibility_sha256,
                             "source_document_ids": [x["document_id"] for x in current], "messages": messages})

    for item in sources:
        if len(payload([item])) > chunk_chars:
            raise ValueError("source document exceeds frozen character budget; refuse silent truncation")
        if current and (subject != item["subject_id"] or scope != item["source"]["session_id"] or len(current) >= 64 or len(payload(current + [item])) > chunk_chars):
            finish()
            current = []
        subject = item["subject_id"]
        scope = item["source"]["session_id"]
        current.append(item)
    finish()
    return requests


def _request_plan(sources, mappings, chunk_chars):
    """Share extraction only between cases with identical complete visibility."""
    by_id = {}
    for source in sources:
        key = source["document_id"]
        if key in by_id:
            raise ValueError("duplicate source identity in frozen index inputs")
        by_id[key] = source
    groups, case_ids = {}, set()
    for case in mappings:
        if case["case_id"] in case_ids:
            raise ValueError("duplicate visible-history case mapping")
        case_ids.add(case["case_id"])
        ids = case["source_document_ids"]
        if len(ids) != len(set(ids)) or not set(ids) <= set(by_id):
            raise ValueError("visible history has duplicate or unknown sources")
        visible = [by_id[key] for key in ids]
        if any(row["subject_id"] != case["subject_id"] for row in visible):
            raise ValueError("visible history crosses subjects")
        identity = _visibility_identity(case["subject_id"], visible)
        if identity != case["visibility_sha256"]:
            raise ValueError("visible source versions differ from case identity")
        if qa.stable_digest([case["subject_id"], [item["source"] for item in visible]]) != case["history_inventory_sha256"]:
            raise ValueError("visible source versions differ from original case history")
        group = (case["subject_id"], ids)
        if identity in groups and groups[identity] != group:
            raise ValueError("visible-history digest collision")
        groups[identity] = group
    requests = []
    for identity in sorted(groups):
        _, ids = groups[identity]
        requests.extend(_requests([by_id[key] for key in ids], chunk_chars, visibility_sha256=identity))
    return requests


def prepare(source_run, output_dir, settings, api_base, *, chunk_chars=12000, fixture_only=False, case_ids=None):
    if not fixture_only:
        qa.require_colab()
    if type(chunk_chars) is not int or not 1000 <= chunk_chars <= 12000:
        raise ValueError("choose source JSON chunk character budget 1000–12000")
    source_run, output_dir = Path(source_run), Path(output_dir)
    if output_dir.resolve().is_relative_to(source_run.resolve()):
        raise ValueError("index stage must be separate from source QA run")
    settings = qa._settings(asdict(settings))
    api_base = qa.validated_api_base(api_base)
    with exclusive_directories(source_run, output_dir):
        if any(p.name != ".execution.lock" for p in output_dir.iterdir()):
            raise ValueError("index stage directory is not empty; resume existing stage")
        source_protocol, sources, mappings = _source_inputs(source_run, case_ids)
        if fixture_only and (len(mappings) > 10 or sum(len(json.dumps(s)) for s in sources) > 1024 * 1024):
            raise ValueError("fixture seam limited to ten small synthetic cases")
        requests = _request_plan(sources, mappings, chunk_chars)
        for name, rows in zip(FROZEN, (sources, mappings, requests)):
            _write(output_dir / name, rows)
        protocol = {"version": VERSION, "fixture_only": fixture_only, "source_protocol_sha256": source_protocol["protocol_sha256"],
                    "source_input_hashes": {name: _sha(source_run / name) for name in ("protocol.json", "inventory.jsonl", "cases.jsonl")},
                    "settings": asdict(settings), "api_base": api_base, "model_identity_policy": qa.model_identity_policy(settings),
                    "code_sha256": _codes(), "file_sha256": {name: _sha(output_dir / name) for name in FROZEN},
                    "maximum_attempts_per_request": MAX_ATTEMPTS, "request_count": len(requests),
                    "source_chunk_chars": chunk_chars, "max_source_documents_per_request": 64,
                    "selected_case_ids": [m["case_id"] for m in mappings],
                    "questions_supplied_to_extractor": False, "references_read": False,
                    "semantic_entailment_validated": False, "clinical_claim": False,
                    "source_boundary": "each extractor request is bound to its complete visible-history identity and original source versions; only identical histories share results; original roles preserved; benchmark statements are data, not user consent"}
        protocol["protocol_sha256"] = qa.stable_digest(protocol)
        _json(output_dir / "index_protocol.json", protocol)
        return {"protocol_sha256": protocol["protocol_sha256"], "request_count": len(requests),
                "source_documents": len(sources), "cases": len(mappings), "model_calls": 0}


def _load(stage):
    protocol = json.loads((stage / "index_protocol.json").read_text())
    if protocol.get("version") != VERSION or protocol.get("maximum_attempts_per_request") != MAX_ATTEMPTS:
        raise ValueError("unsupported index protocol")
    if protocol["protocol_sha256"] != qa.stable_digest({k: v for k, v in protocol.items() if k != "protocol_sha256"}):
        raise ValueError("index protocol checksum mismatch")
    if protocol["code_sha256"] != _codes():
        raise ValueError("index stage code changed; retain old stage with matching code")
    for name in FROZEN:
        if _sha(stage / name) != protocol["file_sha256"][name]:
            raise ValueError("frozen index artifact changed")
    settings = qa._settings(protocol["settings"])
    if protocol["model_identity_policy"] != qa.model_identity_policy(settings):
        raise ValueError("index model policy mismatch")
    source_rows = _rows(stage / "sources.jsonl")
    mappings = _rows(stage / "case_sources.jsonl")
    if [case["case_id"] for case in mappings] != protocol["selected_case_ids"]:
        raise ValueError("visible-history mappings differ from frozen case selection")
    expected_requests = _request_plan(source_rows, mappings, protocol["source_chunk_chars"])
    sources = {x["document_id"]: x for x in source_rows}
    requests = _rows(stage / "extraction_requests.jsonl")
    if requests != expected_requests or len(requests) != protocol["request_count"]:
        raise ValueError("frozen extraction requests differ from exact visible histories")
    if len({r["request_key"] for r in requests}) != len(requests):
        raise ValueError("duplicate extraction requests")
    for row in requests:
        identity = qa.stable_digest({"visibility_sha256": row["visibility_sha256"], "messages": row["messages"]})
        if row["request_key"] != identity or row["prompt_sha256"] != qa.stable_digest(row["messages"]):
            raise ValueError("index request identity mismatch")
    return protocol, sources, requests, mappings


def _candidates(response, request, sources):
    raw = json.loads(response["hypothesis"])
    if not isinstance(raw, dict) or set(raw) != {"candidates"} or not isinstance(raw["candidates"], list):
        raise ValueError("extractor must return one candidates array")
    if len(raw["candidates"]) > 96:
        raise ValueError("too many extraction candidates")
    docs = [qa._document({"subject_id": request["subject_id"]}, sources[key]["source"]) for key in request["source_document_ids"]]
    by_id = {d.document_id: d for d in docs}
    normalized = []
    for item in raw["candidates"]:
        if not isinstance(item, dict) or set(item) != {"source_document_id", "memory_type", "summary", "supporting_text"}:
            raise ValueError("extractor candidate schema mismatch")
        doc = by_id.get(item["source_document_id"])
        if doc is None:
            raise ValueError("extractor cites a source outside its request")
        normalized.append({**item, "subject_id": request["subject_id"],
                           "source_text_sha256": hashlib.sha256(doc.text.encode()).hexdigest()})
    return validate_index(normalized, documents=docs, subject_id=request["subject_id"])


def _validate_response(row, request, protocol):
    qa.validate_provider_model(row.get("provider_model"), protocol["model_identity_policy"])
    if not isinstance(row.get("hypothesis"), str) or not row["hypothesis"].strip() or row.get("finish_reason") != "stop":
        raise ValueError("invalid completed extractor response")
    if row.get("response_sha256") != qa.stable_digest({k: v for k, v in row.items() if k != "response_sha256"}):
        raise ValueError("extractor response checksum mismatch")
    for field in ("latency_seconds",):
        if type(row.get(field)) not in (int, float) or not math.isfinite(row[field]) or row[field] < 0:
            raise ValueError("invalid extractor latency")
    if not isinstance(row.get("usage"), dict):
        raise ValueError("invalid extractor usage")
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if field in row["usage"] and (type(row["usage"][field]) is not int or row["usage"][field] < 0):
            raise ValueError("invalid extractor usage counter")


def _journals(stage, protocol, sources, requests):
    expected = {r["request_key"]: r for r in requests}
    intents, responses, outcomes = {}, {}, {}
    for name, target in (("index_attempts.jsonl", intents), ("index_responses.jsonl", responses), ("index_outcomes.jsonl", outcomes)):
        for row in _rows(stage / name):
            key, attempt = row.get("request_key"), row.get("attempt_index")
            if key not in expected or type(attempt) is not int or not 1 <= attempt <= MAX_ATTEMPTS:
                raise ValueError("index journal request/attempt mismatch")
            pair = key, attempt
            if pair in target:
                raise ValueError("duplicate index journal entry")
            if row.get("protocol_sha256") != protocol["protocol_sha256"] or row.get("prompt_sha256") != expected[key]["prompt_sha256"]:
                raise ValueError("index journal protocol/prompt mismatch")
            if target is intents:
                # Outcomes are loaded later; terminal/retry policy below.
                if any((key, n) not in intents for n in range(1, attempt)):
                    raise ValueError("out of order index attempt")
            elif pair not in intents:
                raise ValueError("index response/outcome without intent")
            if target is responses:
                _validate_response(row, expected[key], protocol)
            if target is outcomes:
                if row.get("status") not in {"success", "technical_failure", "invalid_index", "identity_failure", "terminal_unknown_failure"} or type(row.get("retryable")) is not bool:
                    raise ValueError("invalid index outcome")
                if row["status"] == "identity_failure":
                    raise ValueError("extractor identity/integrity failure; stage rejected")
                if row["status"] == "success":
                    if pair not in responses or row.get("candidates") != _candidates(responses[pair], expected[key], sources):
                        raise ValueError("index success differs from recorded response")
                if row["status"] == "invalid_index":
                    if pair not in responses:
                        raise ValueError("invalid-index outcome missing recorded response")
                    try:
                        _candidates(responses[pair], expected[key], sources)
                    except (ValueError, KeyError, TypeError):
                        pass
                    else:
                        raise ValueError("valid index response hidden by invalid-index outcome")
                if row["status"] == "technical_failure" and pair in responses:
                    raise ValueError("recorded provider response contradicts technical-failure outcome")
            target[pair] = row
    for (key, n) in intents:
        if n > 1 and ((key, n - 1) not in outcomes or not outcomes[(key, n - 1)]["retryable"]):
            raise ValueError("index attempt follows an inflight or terminal outcome")
    states = {}
    for key in expected:
        attempts = [n for k, n in intents if k == key]
        n = max(attempts, default=0)
        outcome = outcomes.get((key, n))
        status = ("pending" if not n else "inflight" if outcome is None else "success" if outcome["status"] == "success" else
                  "exhausted_failure" if n >= MAX_ATTEMPTS or not outcome["retryable"] else "pending")
        states[key] = {"status": status, "attempt_count": n}
    return states, intents, responses, outcomes


def _status(protocol, requests, journal):
    states, intents, responses, outcomes = journal
    counts = Counter(x["status"] for x in states.values())
    usage = {k: sum(r["usage"].get(k, 0) for r in responses.values()) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
    return {"protocol_sha256": protocol["protocol_sha256"], "requests": len(requests), "counts": dict(counts),
            "complete": counts["pending"] == counts["inflight"] == 0, "actual_provider_attempts": len(intents),
            "recorded_provider_responses": len(responses), "reported_usage_including_invalid_indexes": usage,
            "attempts_without_reported_usage": len(intents) - sum(bool(r["usage"]) for r in responses.values()),
            "failed_outcomes": sum(x["status"] != "success" for x in outcomes.values()),
            "unresolved_inflight": counts["inflight"], "monetary_cost": None,
            "cost_boundary": "all attempt intents retained; failed transport/truncation usage unavailable when provider adapter raises before returning; no zero-cost assumption",
            "semantic_entailment_validated": False, "clinical_claim": False}


def status(stage):
    stage = Path(stage)
    with qa.exclusive_directory(stage):
        protocol, sources, requests, _ = _load(stage)
        return _status(protocol, requests, _journals(stage, protocol, sources, requests))


def generate(stage, *, max_calls, key=None):
    qa.require_colab()
    if type(max_calls) is not int or not 1 <= max_calls <= 60:
        raise ValueError("choose finite batch of 1–60 attempted calls")
    key = key or os.environ.get("CBT_EVAL_API_KEY", "")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("configure existing Colab Secret; do not place credentials in artifacts")
    stage = Path(stage)
    with qa.exclusive_directory(stage):
        protocol, sources, requests, _ = _load(stage)
        if protocol["fixture_only"]:
            raise ValueError("synthetic fixture stages cannot call a real provider")
        settings, count = qa._settings(protocol["settings"]), 0
        for request in requests:
            states, _, _, _ = _journals(stage, protocol, sources, requests)
            state = states[request["request_key"]]
            if state["status"] != "pending" or count >= max_calls:
                continue
            intent = {"request_key": request["request_key"], "prompt_sha256": request["prompt_sha256"],
                      "protocol_sha256": protocol["protocol_sha256"], "attempt_index": state["attempt_count"] + 1,
                      "started_at_unix": time.time()}
            qa._append(stage / "index_attempts.jsonl", intent)
            count += 1
            started = time.monotonic()
            try:
                response = qa.generate_response(request, settings, api_base=protocol["api_base"], key=key)
                row = {**response, **intent}
                row["response_sha256"] = qa.stable_digest(row)
                _validate_response(row, request, protocol)
            except Exception as exc:
                identity = isinstance(exc, (ValueError, KeyError, TypeError))
                retryable = isinstance(exc, (URLError, TimeoutError, ConnectionError, RuntimeError))
                if isinstance(exc, HTTPError):
                    retryable = exc.code == 429 or 500 <= exc.code < 600
                outcome = {"status": "identity_failure" if identity else "technical_failure", "retryable": retryable and not identity,
                           "error_type": type(exc).__name__}
                qa._append(stage / "index_outcomes.jsonl", {**intent, **outcome, "elapsed_seconds": time.monotonic() - started})
                if identity:
                    raise ValueError("extractor identity or response integrity rejected") from None
                continue
            # Persistence is deliberately outside the provider error handler.
            # A full valid response may already be on disk when flush/fsync
            # raises. Keep that intent unresolved rather than recording a
            # contradictory provider failure or issuing another paid request.
            qa._append(stage / "index_responses.jsonl", row)
            try:
                candidates = _candidates(row, request, sources)
                outcome = {"status": "success", "retryable": False, "candidates": candidates}
            except (ValueError, KeyError, TypeError):
                outcome = {"status": "invalid_index", "retryable": False, "error_type": "InvalidSourceBoundIndex"}
            qa._append(stage / "index_outcomes.jsonl", {**intent, **outcome, "elapsed_seconds": time.monotonic() - started})
        result = _status(protocol, requests, _journals(stage, protocol, sources, requests))
        return {**result, "actual_calls_this_invocation": count}


def recover(stage, *, request_key, resolution):
    if resolution not in {"terminal_unknown_failure", "use_recorded_response"}:
        raise ValueError("explicit recovery resolution required")
    stage = Path(stage)
    with qa.exclusive_directory(stage):
        protocol, sources, requests, _ = _load(stage)
        states, intents, responses, _ = _journals(stage, protocol, sources, requests)
        if request_key not in states or states[request_key]["status"] != "inflight":
            raise ValueError("only unresolved inflight intents can be recovered")
        n = states[request_key]["attempt_count"]
        intent = intents[(request_key, n)]
        if resolution == "use_recorded_response":
            row = responses.get((request_key, n))
            if row is None:
                raise ValueError("no recorded provider response for recovery")
            request = next(r for r in requests if r["request_key"] == request_key)
            candidates = _candidates(row, request, sources)
            outcome = {"status": "success", "retryable": False, "candidates": candidates}
        else:
            outcome = {"status": "terminal_unknown_failure", "retryable": False}
        qa._append(stage / "index_outcomes.jsonl", {**intent, **outcome, "recovery": resolution})
        return _status(protocol, requests, _journals(stage, protocol, sources, requests))


def compile_index(stage):
    stage = Path(stage)
    with qa.exclusive_directory(stage):
        protocol, sources, requests, mappings = _load(stage)
        journal = _journals(stage, protocol, sources, requests)
        receipt = _status(protocol, requests, journal)
        if not receipt["complete"]:
            raise ValueError("index compile requires all frozen requests terminal; retain partial stage")
        requests_by_key = {request["request_key"]: request for request in requests}
        candidates = [c for outcome in journal[3].values() if outcome["status"] == "success" for c in outcome["candidates"]]
        candidates_by_visibility = {}
        for outcome in journal[3].values():
            if outcome["status"] == "success":
                identity = requests_by_key[outcome["request_key"]]["visibility_sha256"]
                candidates_by_visibility.setdefault(identity, []).extend(outcome["candidates"])
        rows = []
        for case in mappings:
            ids = set(case["source_document_ids"])
            selected = [c for c in candidates_by_visibility.get(case["visibility_sha256"], [])
                        if c["source_document_id"] in ids and c["subject_id"] == case["subject_id"]]
            docs = [qa._document({"subject_id": case["subject_id"]}, sources[key]["source"]) for key in ids]
            validate_index(selected, documents=docs, subject_id=case["subject_id"])
            rows.append({"case_id": case["case_id"], "candidates": selected})
        path = stage / "semantic_index.jsonl"
        if path.exists() and _rows(path) != rows:
            raise ValueError("existing compiled index differs from authoritative journals")
        _write(path, rows)
        manifest = {**receipt, "compiled_index_sha256": _sha(path), "cases": len(rows), "candidate_count": len(candidates),
                    "status": "complete_with_extraction_failures" if receipt["counts"].get("exhausted_failure") else "complete",
                    "failure_policy": "terminal source chunks contribute no candidates; failures remain in extraction denominator",
                    "summary_entailment_boundary": "exact support quotes checked; no independent semantic annotation or clinical inference"}
        _json(stage / "index_build_manifest.json", manifest)
        return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--source-run", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--accepted-provider-model", action="append", default=[])
    p.add_argument("--api-base", required=True)
    p.add_argument("--chunk-chars", type=int, default=12000)
    p.add_argument("--max-response-tokens", type=int, default=8192)
    p.add_argument("--case-id", action="append", help="Repeat to freeze an explicit source-only pilot subset before extraction")
    for name in ("generate", "status", "compile", "recover"):
        q = sub.add_parser(name)
        q.add_argument("--stage-dir", type=Path, required=True)
        if name == "generate":
            q.add_argument("--max-calls", type=int, required=True)
        if name == "recover":
            q.add_argument("--request-key", required=True)
            q.add_argument("--resolution", choices=("terminal_unknown_failure", "use_recorded_response"), required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        settings = qa.Settings(model=args.model, accepted_provider_models=tuple(args.accepted_provider_model),
                               max_response_tokens=args.max_response_tokens, thinking_mode="disabled", backend="lexical")
        result = prepare(args.source_run, args.output_dir, settings, args.api_base, chunk_chars=args.chunk_chars, case_ids=args.case_id)
    elif args.command == "generate":
        result = generate(args.stage_dir, max_calls=args.max_calls)
    elif args.command == "compile":
        result = compile_index(args.stage_dir)
    elif args.command == "recover":
        result = recover(args.stage_dir, request_key=args.request_key, resolution=args.resolution)
    else:
        result = status(args.stage_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
