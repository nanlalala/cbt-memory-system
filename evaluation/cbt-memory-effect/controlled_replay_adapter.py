"""Prepare source-faithful current-v2 contexts without answering prefix turns.

This is a research replay adapter, not the live dialogue entry point. It reuses
the actual durable store, product task rules, bounded worker, manager, retrieval
and evidence packer. It never calls ``send_message`` or generates an answer.
The caller must separately freeze the provider, perform persistent send admission
and collect the complete displayed reply before claiming an end-to-end result.

The dated mode preserves the product clock contract. The independently versioned
``ordinal-only-v1`` research mode uses surrogate timestamps internally for storage
and cutoff only, strips them at the wire boundary, and disables task/calendar and
deadline/refusal-time projection. It is not the full online product intervention.
Only tiny invented fixtures may execute outside hosted Colab.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
VERSION = "controlled-v2-context-replay-v2"
CLOCK_MODES = ("dated-sources-v1", "ordinal-only-v1")
ORDINAL_CLOCK_VERSION = "ordinal-only-v1"
SELECTION_POLICY = "raw_relevance_rank_then_semantic_supplement-v1"
# An implementation encoding, never a source date or a model-visible calendar.
_ORDER_EPOCH = datetime(2000, 1, 3, tzinfo=timezone.utc)
ARMS = ("short_term", "raw_history_rag", "managed_memory_rag")
PERMISSION_KEYS = {"save", "long_term_use", "response_external", "extraction_external"}
FORBIDDEN_INPUT_KEYS = {"patterns", "cbt_plan", "intake_form", "completion_criterion",
                        "memory_targets", "expected_current_state", "scoring_source_references",
                        "future_response", "answer", "gold", "old_score"}


def stable(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _modules():
    expected = ROOT / "modules/agent-integration/controlled_dialogue.py"
    sys.path.insert(0, str(expected.parent))
    module = importlib.import_module("controlled_dialogue")
    if Path(module.__file__).resolve() != expected.resolve():
        raise RuntimeError("foreign controlled-dialogue module; use a fresh process")
    return module


class ReplayBlocked(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class ReplayConfig:
    max_outbound_chars: int = 8000
    max_response_tokens: int = 1024
    max_history_turns: int = 64
    short_turns: int = 4
    top_k: int = 6
    maximum_extraction_calls: int = 64
    maximum_response_calls: int = 3
    clock_mode: str = "dated-sources-v1"

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for k, v in asdict(self).items() if k != "clock_mode"):
            raise ValueError("replay limits must be positive integers")
        if self.top_k != 6:
            raise ValueError("v1 managed product retrieval freezes top_k=6")
        if self.max_history_turns > 64 or self.maximum_extraction_calls > 64:
            raise ValueError("bounded replay permits at most 64 prefix turns/calls")
        if self.maximum_response_calls > 3 or self.clock_mode not in CLOCK_MODES:
            raise ValueError("unknown clock contract or more than three response calls")


class _ContextOnlyProvider:
    """Select external-use filtering without permitting an answer call."""
    def __call__(self, **kwargs):
        raise AssertionError("context replay must never invoke an answer provider")


def _ordinal_refusal_matches(text):
    """Reuse the product recognizer only as a boolean pre-write gate.

    Its mandatory clock argument is the internal encoding clock. The returned
    scope/expiry object is discarded: no calendar refusal state is established.
    Unknown applicability is conservatively blocked for the whole comparison.
    """
    return _modules().refusal_for_input(text, subject_id="ordinal:unbound-refusal-gate",
        as_of=_ORDER_EPOCH.isoformat(), timezone_name="UTC") is not None


def validate_case(case, config):
    """Validate the complete prefix before opening a database or running work."""
    if type(case) is not dict or FORBIDDEN_INPUT_KEYS.intersection(case):
        raise ValueError("gold-free generation case required")
    if type(case.get("case_id")) is not str or not case["case_id"]:
        raise ValueError("case_id required")
    if type(case.get("input")) is not str or not case["input"].strip():
        raise ValueError("nonempty current user input required")
    if type(case.get("input_message_id")) is not str or not case["input_message_id"]:
        raise ValueError("current input_message_id required")
    history = case.get("history")
    if type(history) is not list or len(history) > config.max_history_turns:
        raise ValueError("bounded released prefix required")
    ordinal = config.clock_mode == ORDINAL_CLOCK_VERSION
    if ordinal and (case.get("calendar_time_available") is not False or case.get("input_date") is not None):
        raise ValueError("ordinal-only mode requires explicitly undated sources")
    if not ordinal and (case.get("calendar_time_available") is False or case.get("input_date") is None):
        raise ReplayBlocked("undated_source_order_clock_and_rendering_contract_required")
    m = _modules()
    previous = None; ids = set(); orders = []
    for index, turn in enumerate(history):
        if type(turn) is not dict or FORBIDDEN_INPUT_KEYS.intersection(turn):
            raise ValueError("raw original turns only")
        if turn.get("role") not in {"user", "assistant"}:
            raise ValueError("unknown released speaker cannot be guessed")
        if type(turn.get("content")) is not str or not turn["content"]:
            raise ValueError("literal source text required")
        identifier = turn.get("message_id")
        if type(identifier) is not str or not identifier or identifier in ids:
            raise ValueError("unique source message IDs required")
        ids.add(identifier)
        if ordinal:
            if turn.get("date") is not None or "source_order" not in turn:
                raise ValueError("ordinal-only sources require null date and explicit source order")
        else:
            if turn.get("date") is None:
                raise ReplayBlocked("undated_source_order_clock_and_rendering_contract_required")
            stamp = m.moment(turn["date"])
            if previous is not None and stamp < previous:
                raise ValueError("released source dates must be chronological")
            previous = stamp
        order = turn.get("source_order", index)
        if type(order) is not int or not 0 <= order <= 1000000 or (orders and order <= orders[-1]):
            raise ValueError("strictly increasing source order required")
        orders.append(order)
        if turn["role"] == "user" and m.RISK_PATTERN.search(turn["content"]):
            raise ReplayBlocked("risk_source_requires_product_local_safety_path")
        if ordinal and turn["role"] == "user" and _ordinal_refusal_matches(turn["content"]):
            raise ReplayBlocked("ordinal_explicit_refusal_requires_separate_local_control_path")
        # Content controls need the actual previously displayed object identity;
        # a corpus prose instruction is not permission to guess that binding.
        if turn["role"] == "user" and re.match(r"\s*(?:删除|delete)\s+(?:source|span|object|representation)\b", turn["content"], re.I):
            raise ReplayBlocked("explicit_local_control_binding_requires_separate_replay")
    if case["input_message_id"] in ids:
        raise ValueError("current input duplicates a historical source ID")
    if ordinal:
        cutoff = case.get("history_cutoff_order_exclusive")
        if type(cutoff) is not int or not 0 <= cutoff <= 1000000 or (orders and cutoff <= orders[-1]):
            raise ValueError("ordinal current input needs a strictly later explicit cutoff order")
    elif previous is not None and m.moment(case["input_date"]) < previous:
        raise ValueError("current input precedes its visible source history")
    if not ordinal:
        m.moment(case["input_date"])
    if m.RISK_PATTERN.search(case["input"]):
        raise ReplayBlocked("risk_input_requires_product_local_safety_path")
    if ordinal and _ordinal_refusal_matches(case["input"]):
        raise ReplayBlocked("ordinal_explicit_refusal_requires_separate_local_control_path")
    if re.match(r"\s*(?:删除|delete)\s+(?:source|span|object|representation)\b", case["input"], re.I):
        raise ReplayBlocked("explicit_local_control_binding_requires_separate_replay")


def _source(session, turn, prefix, defaults, *, config):
    m = _modules()
    permissions = (m.permissions_for_input(turn["content"], defaults)
                   if turn["role"] == "user" else dict(defaults))
    iid = prefix + ":" + turn["message_id"]
    return {"user_id": session.user_id, "source_id": "source:" + iid,
            "input_id": iid, "role": turn["role"], "text": turn["content"],
            "source_time": (( _ORDER_EPOCH + timedelta(seconds=turn["source_order"])).isoformat()
                            if config.clock_mode == ORDINAL_CLOCK_VERSION else turn["date"]),
            "permissions": permissions}


def _commit(session, source, *, enqueue, temporal_semantics=True):
    """Reuse current product task/permission rules; never interpret assistant facts."""
    m = _modules(); task = None; events = []
    if source["role"] == "user" and temporal_semantics:
        # Resolve a late occurrence into the original period before selecting
        # a task, exactly as the current live source-first path does.
        period_id = None
        if any(m.verify_claim_binding(source["text"], 0, len(source["text"]),
                {"kind": "choice", "value": value}) for value in ("attempt", "cumulative")):
            reference = m.resolve_event_date_reference(source["text"], source_time=source["source_time"],
                task_instance_id="unselected-task", period_id=session._period(source["text"], source["source_time"]),
                timezone_name=session.timezone_name)
            if reference:
                period_id = reference["period_id"]
        task = session._task_for_input(source["text"], source["source_time"], period_id=period_id)
        events = session._events_for_input(source["text"], source, task)
        subject = task["payload"]["task_instance_id"] if task else "topic:general"
        refusal = m.refusal_for_input(source["text"], subject_id=subject,
                                     as_of=source["source_time"], timezone_name=session.timezone_name)
    else:
        subject = "topic:general"; refusal = None
    with session.store.transaction(immediate=True):
        if refusal:
            session.store.connection.execute(
                "INSERT INTO cp_refusals VALUES (?,?,?,?) ON CONFLICT(user_id,subject_id,kind) DO UPDATE SET value_json=excluded.value_json",
                (session.user_id, subject, refusal["kind"], json.dumps(refusal)))
        receipt = session.durable.submit_source(source, enqueue=enqueue and refusal is None,
            run_id=session.run_id, session_id=session.session_id)
        if task and receipt["source_status"] == "committed":
            session.store.connection.execute("INSERT OR IGNORE INTO cp_source_topics VALUES (?,?,?)",
                (session.user_id, source["source_id"], subject))
        if source["permissions"]["save"] is True and source["permissions"]["long_term_use"] is True and events:
            session._commit_task(task, events, source)
        elif events and task and task["status"] == "available":
            session.durable.stop_use_object(session.user_id, task["object_id"])
    if source["permissions"]["save"] is True:
        session._turns = [*session._turns, m.ConversationTurn(source["source_id"],
            source["role"], source["text"], source["source_time"])]
    if source["role"] == "assistant":
        session._set_meta("last_assistant_turn_id", source["source_id"])
    return receipt, task, events, refusal


def _quote_bundles(hits):
    m = _modules(); result = []
    for rank, hit in enumerate(hits):
        span = m.EvidenceSpan(hit.source_id, hit.span_id, hit.text, hit.start,
                              hit.end, hit.source_time, hit.role)
        identity = "history:" + hit.span_id
        claim = m.EvidenceClaim(identity, "history_quote", hit.text, (span,),
            support_status="source_quote", priority=-rank,
            origin="assistant_proposal" if hit.role == "assistant" else "explicit_user_statement")
        result.append(m.EvidenceBundle(identity, (claim,)))
    return result


def _preserve_managed_quote_rank(bundles, raw_hits):
    """All arms preserve the shared retriever's ranking at packing time.

    Semantic-only quotations supplement the ranked original-source channel;
    they do not displace its higher-ranked quotations via a priority bonus.
    Product rule-verified task fields in dated mode retain their own priority.
    """
    ranks = {(h.source_id, h.start, h.end): rank for rank, h in enumerate(raw_hits)}
    semantic_rank = 0; result = []
    for bundle in bundles:
        claims = []
        for claim in bundle.claims:
            if claim.field != "history_quote":
                claims.append(claim); continue
            matched = [ranks[(span.source_id, span.start, span.end)] for span in claim.spans
                       if (span.source_id, span.start, span.end) in ranks]
            if matched:
                priority = -min(matched)
            else:
                priority = -len(raw_hits) - semantic_rank
                semantic_rank += 1
            claims.append(replace(claim, priority=priority))
        result.append(replace(bundle, claims=tuple(claims)))
    return result


def _ordinal_managed_bundles(session, query, raw_search, source_pool):
    """Reuse candidate-pointer retrieval without inventing calendar validity.

    Model timestamps/validity are not known calendar truth. The research index
    view ignores those fields; the persisted unconfirmed record is unchanged.
    Only a complete currently permitted original quote can reach the wire.
    """
    m = _modules(); sources = {s["source_id"]: s for s in source_pool}
    bundles = _quote_bundles(raw_search.hits)
    seen = {(hit.source_id, hit.start, hit.end) for hit in raw_search.hits}
    candidates = []
    for obj in session._objects("candidate"):
        if obj["status"] != "available":
            continue
        record = m.construct_memory(obj["payload"]["record"], allow_legacy=False)
        if all(sid in sources and sources[sid]["permissions"]["response_external"] is True
               for sid in record.source_evidence.turn_ids):
            candidates.append(replace(record, valid_from=None, valid_to=None))
    hits = m.MemoryRetriever(candidates, policy=m.RetrievalPolicy(
        allowed_lifecycle=frozenset({"candidate"}), recency_weight=0,
        importance_weight=0, confidence_weight=0)).search(query, user_id=session.user_id) if candidates else []
    selected_ids = set()
    for hit in hits:
        record = hit.record
        for sid in record.source_evidence.turn_ids:
            span = m.restore_source_quote(sources[sid], pointer_text=record.source_evidence.supporting_text,
                span_id="semantic-source:" + record.memory_id + ":" + sid)
            if span is None or (sid, span.start, span.end) in seen:
                continue
            seen.add((sid, span.start, span.end)); selected_ids.add(sid)
            identity = "semantic:" + record.memory_id + ":" + sid
            bundles.append(m.EvidenceBundle(identity, (m.EvidenceClaim(identity, "history_quote", span.text,
                (span,), support_status="source_quote", priority=-len(raw_search.hits) - len(selected_ids)),)))
    trace = {**raw_search.trace, "semantic_candidate_count": len(candidates),
        "semantic_selected_source_ids": sorted(selected_ids),
        "candidate_calendar_validity": "disabled_for_ordinal_index_view_only",
        "task_temporal_semantics": "disabled"}
    return bundles, trace


def _ordinal_wire(package, source_orders, *, maximum_chars):
    """Replace internal schema-clock metadata with the released source order."""
    messages = []
    for message in package.messages:
        message = dict(message)
        if message["content"].startswith("SOURCED MEMORY DATA ONLY."):
            header, body = message["content"].split("\n", 1)
            packet = json.loads(body)
            for claim in packet["claims"]:
                if claim["field"] != "history_quote" or claim["support_status"] != "source_quote":
                    raise ValueError("ordinal mode cannot render current-state task claims")
                for span in claim["evidence"]:
                    span.pop("source_time", None)
                    span["source_order"] = source_orders[span["source_id"]]
            message["content"] = header + "\n" + json.dumps(packet, ensure_ascii=False, sort_keys=True, allow_nan=False)
        messages.append(message)
    payload = {**package.request_payload, "messages": messages}
    wire = json.dumps(payload, ensure_ascii=True, sort_keys=True, allow_nan=False)
    if len(wire) > maximum_chars:
        raise ValueError("ordinal-rendered complete request exceeds frozen character cap")
    return replace(package, messages=tuple(messages), request_payload=payload,
                   serialized_request=wire, char_count=len(wire))


def _check_for_view(session, state, query, stamp):
    """One common pure policy for eligible view-specific missing-quantity checks."""
    m = _modules()
    if not state or state["progress"]["target_count"] is not None or not re.search(r"几次|多少次|how many", query, re.I):
        return None
    request = m.CheckRequest(session.user_id, state["task_instance_id"], "target_count", None,
        "quantity_unknown", "这项练习本周期安排几次，目前还不清楚；你愿意现在说一下你的安排吗？", True)
    return m.decide_check(request, as_of=stamp, events=session._check_events(), refusals=session._refusals())


def _require_settled_extraction(durable, user_id, job_id):
    attempt = durable.get_attempt(user_id, "extract:" + job_id)
    if attempt and attempt["status"] in {"sending", "attempt_unknown", "dependency_unknown"}:
        raise ReplayBlocked("semantic_extraction_attempt_unknown_no_additional_calls")


def assert_evidence_integrity(package, sources, *, clock_mode="dated-sources-v1", source_orders=None):
    """Check final packed quotes/roles against the exact permitted snapshot."""
    by_id = {s["source_id"]: s for s in sources}
    for message in package.messages:
        if not message["content"].startswith("SOURCED MEMORY DATA ONLY."):
            continue
        claims = json.loads(message["content"].split("\n", 1)[1])["claims"]
        for claim in claims:
            for span in claim["evidence"]:
                source = by_id.get(span["source_id"])
                if source is None or source["permissions"]["response_external"] is not True:
                    raise ValueError("outbound evidence is not in the permitted source snapshot")
                if source["role"] != span["role"]:
                    raise ValueError("outbound evidence changed source role/time")
                if clock_mode == ORDINAL_CLOCK_VERSION:
                    if "source_time" in span or span.get("source_order") != (source_orders or {}).get(span["source_id"]):
                        raise ValueError("ordinal wire leaked a clock or changed source order")
                    if claim["field"] != "history_quote" or claim["support_status"] != "source_quote":
                        raise ValueError("ordinal wire must not assert current task facts")
                elif source["source_time"] != span["source_time"]:
                    raise ValueError("outbound evidence changed source role/time")
                if source["text"][span["start"]:span["end"]] != span["text"]:
                    raise ValueError("outbound evidence changed literal source text")
    return True


def prepare_contexts(case, *, database, approved_root, permissions, config=None,
                     extraction_provider=None, request_metadata=None, fixture_only=False):
    """Return three prepared requests and stage traces; make zero answer calls.

    Real corpus use and live semantic extraction require genuine hosted Colab.
    ``permissions`` explicitly defines benchmark access, not a claim of participant
    consent. Pass the same provider metadata that the eventual sender will use.
    A fresh database is required: restart/resume must use a separately reviewed
    orchestration protocol rather than silently regenerating extraction jobs.
    """
    config = config or ReplayConfig()
    if type(fixture_only) is not bool:
        raise ValueError("fixture flag must be a boolean")
    if not fixture_only:
        guard_path = ROOT / "app"
        sys.path.insert(0, str(guard_path))
        guard = importlib.import_module("colab_provider")
        if Path(guard.__file__).resolve() != (guard_path / "colab_provider.py").resolve():
            raise RuntimeError("foreign Colab guard module")
        guard.require_colab()
    elif extraction_provider is not None:
        raise ValueError("fixture-only replay does not admit a semantic provider")
    if type(permissions) is not dict or set(permissions) != PERMISSION_KEYS or any(
            type(value) is not bool for value in permissions.values()):
        raise ValueError("four explicit benchmark permission axes are required")
    validate_case(case, config)
    if Path(database).exists():
        raise ValueError("fresh replay database required; old evidence is never overwritten")
    m = _modules()
    prefix = "case-" + stable(case["case_id"])[:20]
    ordinal = config.clock_mode == ORDINAL_CLOCK_VERSION
    session = m.ControlledDialogueSession.open(database, approved_root=approved_root,
        user_id=prefix, run_id=VERSION, session_id=prefix + ":released-session",
        provider=extraction_provider or _ContextOnlyProvider(), background=False,
        limits=m.PrototypeLimits(max_outbound_chars=config.max_outbound_chars,
            max_response_tokens=config.max_response_tokens, max_worker_jobs_per_turn=1,
            max_session_turns=config.max_history_turns + 1),
        durable_limits=m.DurableLimits(max_run_calls=config.maximum_extraction_calls + config.maximum_response_calls,
            max_session_calls=config.maximum_extraction_calls + config.maximum_response_calls, max_queue_length=1, max_batch_size=1),
        permissions=permissions)
    receipts = []; worker_results = []; original_sources = []; source_orders = {}
    try:
        worker = m.BoundedMemoryWorker(database, approved_root, run_id=session.run_id,
            limits=session.durable_limits, provider=extraction_provider)
        for index, turn in enumerate(case["history"]):
            source = _source(session, turn, prefix, permissions, config=config)
            source_orders[source["source_id"]] = turn.get("source_order", index)
            receipt, _, _, _ = _commit(session, source, enqueue=turn["role"] == "user", temporal_semantics=not ordinal)
            receipts.append({"message_id": turn["message_id"], "role": turn["role"],
                "source_id": source["source_id"], "text_sha256": hashlib.sha256(source["text"].encode()).hexdigest(),
                "permissions": source["permissions"], "receipt": receipt})
            original_sources.append(source)
            if receipt.get("job_id"):
                worker_results.extend(worker.run_batch(user_id=session.user_id, maximum_jobs=1))
                _require_settled_extraction(session.durable, session.user_id, receipt["job_id"])
        current = _source(session, {"message_id": case["input_message_id"], "role": "user",
            "content": case["input"], "date": case.get("input_date"),
            "source_order": case.get("history_cutoff_order_exclusive", len(case["history"]))},
            prefix, permissions, config=config)
        source_orders[current["source_id"]] = case.get("history_cutoff_order_exclusive", len(case["history"]))
        stamp = current["source_time"]
        receipt, task, events, refusal = _commit(session, current, enqueue=False, temporal_semantics=not ordinal)
        original_sources.append(current)
        if refusal:
            raise ReplayBlocked("current_refusal_requires_local_preference_response_not_model")
        if current["permissions"]["response_external"] is not True:
            raise ReplayBlocked("current_response_external_not_permitted")
        if current["permissions"]["save"] is not True:
            raise ReplayBlocked("ephemeral_current_input_requires_live_entrypoint_contract")
        if task and events and receipt["source_status"] == "committed":
            task = {"object_id": task["object_id"], **session.durable.get_object_state(session.user_id, task["object_id"])}
        paused_sources, paused_subjects = session._paused_source_ids(stamp) if not ordinal else (set(), set())
        if task and task["payload"]["task_instance_id"] in paused_subjects:
            raise ReplayBlocked("current_topic_pause_requires_local_response_not_model")
        pool = [s for s in session.durable.list_sources(session.user_id, as_of=stamp)
                if s["source_id"] != current["source_id"] and s["source_id"] not in paused_sources]
        raw_search = m.StableSourceRetriever(pool).search(case["input"], user_id=session.user_id,
            source_cutoff=stamp, as_of=None if ordinal else stamp, for_external_response=True, top_k=config.top_k)
        raw = _quote_bundles(raw_search.hits)
        if ordinal:
            managed, managed_trace = _ordinal_managed_bundles(session, case["input"], raw_search, pool)
            state = None
        else:
            managed, managed_trace, state = session._bundles(case["input"], as_of=stamp,
                current_source=current, task=task, ephemeral_events=())
        managed = _preserve_managed_quote_rank(managed, raw_search.hits)
        recent_ids = {s["source_id"] for s in original_sources[:-1][-config.short_turns:]}
        short_search = m.StableSourceRetriever(pool).search(case["input"], user_id=session.user_id,
            source_cutoff=stamp, as_of=None if ordinal else stamp, for_external_response=True,
            allowed_source_ids=recent_ids, top_k=config.top_k)
        # Short-term is a secondary window-scoped retrieval comparator, not a
        # relabelled full-history condition. Its actual window is in the trace.
        short = _quote_bundles(short_search.hits)
        ordinal_instruction = ("\nThis is the separately frozen ordinal-only-v1 research comparison. "
            "Only released within-dialogue source order is known; no source calendar dates are available. "
            "Do not infer the current week, due dates, elapsed time, today's relation, or refusal expiry. "
            "All memories supplied here are historical quotations, not verified current task state. "
            "Preserve explicit date words in quotations without assigning calendar metadata." if ordinal else "")
        response_system = (m.SYSTEM.replace("Historical quotations are dated", "Historical quotations are source-ordered")
                           if ordinal else m.SYSTEM)
        base = [{"role": "system", "content": response_system + "\nKnowledge status: K0_empty" + ordinal_instruction},
                {"role": "user", "content": case["input"]}]
        packages = {}
        checks = {}
        for arm, bundles in zip(ARMS, (short, raw, managed)):
            package = m.build_outbound_messages(base, bundles,
                max_outbound_chars=config.max_outbound_chars,
                max_response_tokens=config.max_response_tokens, request_metadata=request_metadata, usage=None)
            if ordinal:
                package = _ordinal_wire(package, source_orders, maximum_chars=config.max_outbound_chars)
            assert_evidence_integrity(package, [s for s in original_sources
                if s["permissions"]["save"] and s["permissions"]["long_term_use"]],
                clock_mode=config.clock_mode, source_orders=source_orders)
            packages[arm] = {"messages": list(package.messages), "request_payload": package.request_payload,
                "serialized_request": package.serialized_request, "request_sha256": stable(package.request_payload),
                "outbound_chars": package.char_count, "included_claim_ids": list(package.included_claim_ids),
                "dropped_claims": package.dropped_claims, "source_ids": list(package.source_ids)}
            checks[arm] = _check_for_view(session, state if arm == "managed_memory_rag" else None,
                                        case["input"], stamp)
        trace = {"version": VERSION, "fixture_only": fixture_only, "case_id": case["case_id"],
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "case_sha256": stable(case), "config": asdict(config), "product_protocol": session.protocol,
            "user_id": session.user_id, "run_id": session.run_id, "session_id": session.session_id,
            "current_source_id": current["source_id"], "source_orders": source_orders,
            "clock_mode": config.clock_mode, "calendar_available": not ordinal,
            "context_selection_policy": SELECTION_POLICY,
            "timestamps_internal_only": ordinal,
            "task_temporal_semantics": "disabled" if ordinal else "product_dated_rules",
            "task_state_claims_allowed": not ordinal,
            "source_receipts": receipts, "current_source_receipt": receipt, "worker_results": worker_results,
            "retrieval": {"short_term": short_search.trace, "raw_history_rag": raw_search.trace,
                          "managed_memory_rag": managed_trace},
            "candidate_records": [{"memory_id": r.memory_id, "confirmation_state": r.confirmation_state,
                "lifecycle_status": r.lifecycle_status, "origin": r.origin} for r in session.store.list_for_user(session.user_id)],
            "shared_natural_check_policy_hash": m.POLICY_HASH, "proposed_checks_not_shown": checks,
            "actual_checks_shown": 0, "actual_answer_calls": 0,
            "semantic_extraction_mode": "live_provider" if extraction_provider else "source_only_no_model",
            "raw_managed_same_request": packages["raw_history_rag"]["request_sha256"] == packages["managed_memory_rag"]["request_sha256"],
            "same_permitted_source_pool": raw_search.trace["statistics_pool_fingerprint"] == managed_trace["statistics_pool_fingerprint"],
            "final_evidence_integrity_passed": True, "generation": session.durable.current_generation(session.user_id),
            "expected_task_versions": {task["object_id"]: task["version"]} if task and task.get("status") == "available" and state else {},
            "task_object_count": len(session._objects("task")),
            "ready_for_external_generation": False,
            "remaining_boundaries": ["freeze_actual_provider_and_budget_protocol", "durable_response_send_admission",
                "complete_reply_and_check_display_capture", "independent_memory_target_annotation"],
            "replay_scope": ("ordinal-only research semantic/quote channels; task calendar semantics disabled"
                if ordinal else "dated_sources_and_explicit_product_rules; no inferred proposal registration or retrospective check-display events"),
            "interpretation": "source/worker/context engineering replay only; no memory-benefit or clinical result"}
        # Bind each prepared request to authoritative body-free metadata. This
        # table contains no quotation/answer cache that deletion must later erase.
        with session.store.transaction(immediate=True):
            session.store.connection.execute("CREATE TABLE IF NOT EXISTS research_context_snapshots "
                "(user_id TEXT, request_sha256 TEXT, binding_json TEXT NOT NULL, PRIMARY KEY(user_id,request_sha256))")
            for context in packages.values():
                binding = _send_binding(trace, context)
                session.store.connection.execute("INSERT OR IGNORE INTO research_context_snapshots VALUES (?,?,?)",
                    (session.user_id, context["request_sha256"], json.dumps(binding, sort_keys=True)))
        return {"contexts": packages, "trace": trace}
    finally:
        session.close()


def _send_binding(trace, context):
    return {"adapter_sha256": trace["adapter_sha256"], "clock_mode": trace["clock_mode"],
        "user_id": trace["user_id"], "run_id": trace["run_id"], "session_id": trace["session_id"],
        "generation": trace["generation"], "current_source_id": trace["current_source_id"],
        "source_ids": sorted(set(context["source_ids"] + [trace["current_source_id"]])),
        "expected_task_versions": trace["expected_task_versions"], "source_orders": trace["source_orders"],
        "maximum_chars": trace["config"]["max_outbound_chars"],
        "maximum_output_tokens": trace["config"]["max_response_tokens"],
        "maximum_response_calls": trace["config"]["maximum_response_calls"],
        "request_sha256": context["request_sha256"]}


def _preflight_with_store(prepared, arm, durable):
    if arm not in ARMS:
        raise ValueError("unknown comparison condition")
    trace = prepared["trace"]; context = prepared["contexts"][arm]
    if trace["adapter_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("prepared context belongs to another adapter version")
    payload = context["request_payload"]
    wire = json.dumps(payload, ensure_ascii=True, sort_keys=True, allow_nan=False)
    if (wire != context["serialized_request"] or context["messages"] != payload["messages"]
            or context["outbound_chars"] != len(wire) or stable(payload) != context["request_sha256"]):
        raise ValueError("prepared outbound request identity changed")
    binding = _send_binding(trace, context)
    row = durable.store.connection.execute(
        "SELECT binding_json FROM research_context_snapshots WHERE user_id=? AND request_sha256=?",
        (trace["user_id"], context["request_sha256"])).fetchone()
    if row is None or json.loads(row[0]) != binding:
        raise ValueError("outbound request is not bound to the frozen source snapshot")
    if len(wire) > binding["maximum_chars"] or payload["max_tokens"] != binding["maximum_output_tokens"]:
        raise ValueError("complete outbound or output budget changed")
    if durable.current_generation(trace["user_id"]) != binding["generation"]:
        raise ValueError("withdrawal generation changed after context preparation")
    sources = []
    for sid in binding["source_ids"]:
        source = durable.get_source(trace["user_id"], sid)
        if source is None or source["permissions"]["response_external"] is not True:
            raise PermissionError("prepared source is no longer available for this response")
        sources.append(source)
    current = next(s for s in sources if s["source_id"] == trace["current_source_id"])
    if payload["messages"][-1] != {"role": "user", "content": current["text"]}:
        raise ValueError("current question does not match its committed source")
    assert_evidence_integrity(SimpleNamespace(messages=payload["messages"]), sources,
        clock_mode=trace["clock_mode"], source_orders=trace["source_orders"])
    for object_id, version in binding["expected_task_versions"].items():
        state = durable.get_object_state(trace["user_id"], object_id)
        if state["status"] != "available" or state["version"] != version:
            raise ValueError("task version changed after context preparation")
    return binding


def _open_replay_store(prepared, database, approved_root):
    m = _modules()
    limits = m.DurableLimits(**prepared["trace"]["product_protocol"]["durable_limits"])
    return m.DurableMemoryStore.open(database, approved_root=approved_root, run_limits=limits)


def preflight_outbound(prepared, arm, *, database, approved_root):
    """Recheck the exact wire, current permission, generation and source quotes.

    This read-only result is not send authorization: use ``begin_response_attempt``
    immediately at the actual send boundary. No model or network is invoked.
    """
    durable = _open_replay_store(prepared, database, approved_root)
    try:
        with durable.store.transaction(immediate=True):
            return _preflight_with_store(prepared, arm, durable)
    finally:
        durable.close()


def begin_response_attempt(prepared, arm, *, database, approved_root):
    """Persist canonical send intent atomically with the final source recheck.

    The research key is the exact request identity, so identical arm requests
    share one attempt. The current source must qualify for ordinary long-term
    external use; this route grants no ephemeral-current permission exception.
    No HTTP call occurs here. Any interrupted admitted attempt remains unknown
    and the store refuses automatic resend. The outer response journal owns
    eligible body retention and must discard a nonpublishable finish result.
    """
    durable = _open_replay_store(prepared, database, approved_root)
    try:
        with durable.store.transaction(immediate=True):
            binding = _preflight_with_store(prepared, arm, durable)
            request_key = "research-response:" + binding["request_sha256"]
            if durable.get_attempt(binding["user_id"], request_key) is None:
                count = durable.store.connection.execute(
                    "SELECT COUNT(*) FROM dm_attempts WHERE run_id=? AND user_id=? AND purpose='response'",
                    (binding["run_id"], binding["user_id"])).fetchone()[0]
                if count >= binding["maximum_response_calls"]:
                    raise _modules().BudgetExhausted("frozen research response call limit reached")
            admission = durable.begin_attempt(binding["user_id"],
                request_key, "response", binding["source_ids"],
                run_id=binding["run_id"], session_id=binding["session_id"],
                expected_generation=binding["generation"], expected_object_versions=binding["expected_task_versions"])
        return {**admission, "request_sha256": binding["request_sha256"], "actual_network_call_made": False}
    finally:
        durable.close()


def finish_response_attempt(prepared, *, database, approved_root, attempt_id,
                            outcome, response=None, usage=None):
    """Use the product's current-generation publication check after a send.

    Response text is not copied to an untracked replay cache. Return values with
    ``publishable=false`` must not be written as completed answer bodies by the
    caller. Missing usage stays unknown, not zero.
    """
    durable = _open_replay_store(prepared, database, approved_root)
    try:
        trace = prepared["trace"]
        attempts = [durable.get_attempt(trace["user_id"], "research-response:" + context["request_sha256"],
                    run_id=trace["run_id"], session_id=trace["session_id"])
                    for context in prepared["contexts"].values()]
        if not any(attempt and attempt["attempt_id"] == attempt_id for attempt in attempts):
            raise ValueError("response attempt does not belong to this prepared comparison")
        return durable.finish_attempt(attempt_id, outcome=outcome, response=response,
                                      usage=usage, retain_response=False)
    finally:
        durable.close()
