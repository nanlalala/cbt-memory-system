"""Deterministic fictional source replay through the real memory write path.

No provider is called here. Source text and dates come from an admitted ledger;
model summaries stay unconfirmed indexing candidates. Deletion removes source
quotes, semantic dependants and in-process buffers, not remote provider logs.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
for relative in ("modules/long-term-memory/schema", "modules/long-term-memory/storage",
                 "modules/long-term-memory/lifecycle", "modules/long-term-memory/extraction"):
    location = str(ROOT / relative)
    if location not in sys.path:
        sys.path.insert(0, location)
from extraction_contract import ConversationTurn, parse_extraction_output
from memory_manager import MemoryManager
from memory_models import MemoryRecord, SourceEvidence
from sqlite_store import SQLiteMemoryStore


def stable(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def moment(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("source time requires timezone")
    return parsed


def checked_source(source):
    required = {"source_id", "source_order", "visit", "date", "role", "text", "text_sha256", "storage_policy"}
    if set(source) - (required | {"valid_to"}) or not required <= set(source):
        raise ValueError("source fields differ from source-only contract")
    if not isinstance(source["source_id"], str) or not source["source_id"].strip():
        raise ValueError("source ID required")
    if type(source["source_order"]) is not int or source["source_order"] < 1 or type(source["visit"]) is not int:
        raise ValueError("source order/visit invalid")
    if source["role"] not in {"user", "assistant"} or source["storage_policy"] not in {"allowed", "session_only"}:
        raise ValueError("source role/storage policy invalid")
    if not isinstance(source["text"], str) or not source["text"].strip() or hashlib.sha256(source["text"].encode()).hexdigest() != source["text_sha256"]:
        raise ValueError("source quote hash differs")
    moment(source["date"])
    if source.get("valid_to"):
        if moment(source["valid_to"]) < moment(source["date"]):
            raise ValueError("source expiry precedes observation")
    return dict(source)


class SourceReplay:
    """Public deterministic ingestion seam; never impersonates confirmation."""

    def __init__(self, *, user_id):
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id required")
        self.user_id = user_id
        self.store = SQLiteMemoryStore()
        self.manager = MemoryManager(self.store)
        self.raw_sources = {}
        self.session_sources = {}
        self.source_metadata = {}
        self.record_dependencies = {}
        self.extraction_cache = {}
        self.extraction_cache_dependencies = {}
        self.events = []
        self.semantic_statuses = {}
        self.deleted_source_ids = set()
        self._visit = None

    def close(self):
        self.store.close()

    def quote_id(self, source_id):
        return "quote-" + stable([self.user_id, source_id])[:32]

    def state_hash(self):
        return stable({"records": [r.to_dict() for r in self.store.list_for_user(self.user_id)],
                       "raw": self.raw_sources, "session": self.session_sources,
                       "source_metadata": self.source_metadata,
                       "record_dependencies": {k: sorted(v) for k, v in self.record_dependencies.items()},
                       "extraction_cache_hashes": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in self.extraction_cache.items()},
                       "extraction_cache_dependencies": {k: sorted(v) for k, v in self.extraction_cache_dependencies.items()},
                       "deleted": sorted(self.deleted_source_ids)})

    def ingest_source(self, source, *, extraction_output=None, extraction_context_ids=(), extraction_attempted=False):
        source = checked_source(source)
        sid = source["source_id"]
        if sid in self.source_metadata or sid in self.deleted_source_ids:
            raise ValueError("source replay duplicate or resurrection")
        before = self.state_hash()
        if self._visit != source["visit"]:
            self.session_sources.clear()
            self._visit = source["visit"]
        self.session_sources[sid] = source
        self.source_metadata[sid] = {k: source[k] for k in ("source_id", "source_order", "visit", "date", "role", "text_sha256", "storage_policy")}
        writes = []
        if source["storage_policy"] == "allowed":
            self.raw_sources[sid] = source
            quote = MemoryRecord(
                "1.0", self.quote_id(sid), self.user_id, "explicit_fact", source["text"],
                "system_observation", "unconfirmed", "candidate", "standard", 1.0, 0.5,
                source["date"], source["date"], source["date"],
                SourceEvidence("visit-" + str(source["visit"]), [sid], source["text"]),
                valid_from=source["date"], valid_to=source.get("valid_to"),
            )
            if source.get("valid_to") or source["text"] != source["text"].strip():
                writes.append(self.manager.ingest(quote, actor="fictional_source_replay"))
                self.store.append_audit(user_id=self.user_id, memory_id=quote.memory_id,
                    action="fictional_source_quote_verified", occurred_at=source["date"], actor="fictional_source_replay",
                    details={"source_role": source["role"], "source_text_sha256": source["text_sha256"],
                             "user_consent": False, "validity_from_source_metadata": True})
            else:
                writes += self.manager.ingest_verbatim_observation_candidates(
                    [quote], user_id=self.user_id,
                    source_turns={(quote.source_evidence.session_id, sid): {
                        "text": source["text"], "observed_at": source["date"],
                        "source_role": source["role"], "provenance_kind": "fictional_ledger_observation"}},
                    actor="fictional_source_replay",
                )
            self.record_dependencies[quote.memory_id] = {sid}
            if source["role"] == "user":
                if extraction_output is None:
                    self.semantic_statuses[sid] = "extraction_unavailable" if extraction_attempted else "not_requested"
                else:
                    writes += self._ingest_semantic(source, extraction_output, extraction_context_ids)
        self.events.append({"kind": "source", "source_id": sid, "source_order": source["source_order"],
                            "source_date": source["date"], "source_text_sha256": source["text_sha256"],
                            "state_before_sha256": before, "state_after_sha256": self.state_hash(),
                            "writes": [asdict(item) for item in writes],
                            "semantic_status": self.semantic_statuses.get(sid, "not_eligible"),
                            "user_confirmation_created": False})

    def _ingest_semantic(self, source, raw, context_ids):
        sid = source["source_id"]
        try:
            value = json.loads(raw)
            if not isinstance(value, dict) or set(value) != {"candidates"} or not isinstance(value["candidates"], list) or len(value["candidates"]) > 12:
                raise ValueError("bounded candidates object required")
            cleaned = []
            link_source_ids = set()
            for index, candidate in enumerate(value["candidates"]):
                if not isinstance(candidate, dict) or candidate.get("source_turn_ids") != [sid]:
                    raise ValueError("event candidate must cite only the current user source")
                item = {key: candidate[key] for key in ("memory_type", "content", "source_turn_ids", "supporting_text", "confidence", "importance", "structured_payload", "sensitivity") if key in candidate}
                if not isinstance(item.get("content"), str) or not 1 <= len(item["content"]) <= 1000:
                    raise ValueError("bounded semantic indexing text required")
                payload = item.get("structured_payload", {})
                if not isinstance(payload, dict):
                    raise ValueError("candidate payload must be an object")
                item["structured_payload"] = dict(payload)
                item["observed_at"] = source["date"]
                item["valid_from"] = source["date"]
                item["valid_to"] = source.get("valid_to")
                item["tags"] = ["fictional_replay", "unconfirmed_model_index"]
                if item["memory_type"] == "cognitive_pattern":
                    item["structured_payload"]["evidence_count"] = 1
                for field, payload_field in (("target_source_id", "proposed_update_target_source_id"), ("assignment_source_id", "assignment_source_id")):
                    target = self._normalise_source_link(candidate, item["structured_payload"], source,
                                                         field=field, payload_field=payload_field)
                    if target is not None:
                        item["structured_payload"][payload_field] = target
                        if field != payload_field:
                            item["structured_payload"].pop(field, None)
                        link_source_ids.add(target)
                        item.setdefault("related_memory_ids", [])
                        if self.quote_id(target) not in item["related_memory_ids"]:
                            item["related_memory_ids"].append(self.quote_id(target))
                if item["memory_type"] == "user_correction":
                    target = item["structured_payload"].get("proposed_update_target_source_id")
                    if target is None:
                        raise ValueError("proposed correction requires a source-bound target")
                    item["supersedes_memory_id"] = self.quote_id(target)
                    item["structured_payload"]["corrected_content"] = item["content"]
                if item["memory_type"] == "assignment_outcome":
                    target = item["structured_payload"].get("assignment_source_id")
                    if target is None:
                        raise ValueError("assignment outcome requires a source-bound assignment target")
                    item["structured_payload"]["assignment_memory_id"] = self.quote_id(target)
                # Placement aliases cannot manufacture different memory IDs:
                # derive identity only after every accepted link is validated
                # and represented in its canonical payload field.
                item["memory_id"] = "candidate-" + stable([self.user_id, sid, index, item])[:32]
                cleaned.append(item)
            records = parse_extraction_output(json.dumps({"candidates": cleaned}, ensure_ascii=False),
                user_id=self.user_id, session_id="visit-" + str(source["visit"]),
                turns=[ConversationTurn(sid, "user", source["text"])], extracted_at=source["date"])
            writes = []
            dependencies = set(context_ids) | {sid} | link_source_ids
            if not dependencies <= set(self.raw_sources):
                raise ValueError("semantic context references unavailable sources")
            staged_dependencies = {}
            with self.store.transaction(immediate=True):
                for record in records:
                    if record.memory_type not in {"cognitive_pattern", "user_correction"}:
                        record = replace(record, origin="system_observation")
                    write = self.manager.ingest(record, actor="fictional_source_replay")
                    writes.append(write)
                    if write.action != "duplicate_rejected":
                        staged_dependencies[record.memory_id] = dependencies.copy()
            # In-memory provenance must commit with its SQL records. A later
            # candidate failure rolls back the transaction without phantom IDs.
            self.record_dependencies.update(staged_dependencies)
            self.extraction_cache[sid] = raw
            self.extraction_cache_dependencies[sid] = dependencies.copy()
            self.semantic_statuses[sid] = "valid_empty" if not records else "valid_candidates"
            return writes
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
            self.semantic_statuses[sid] = "extraction_invalid"
            return []

    def _normalise_source_link(self, candidate, payload, source, *, field, payload_field):
        """Accept documented field placements without weakening source checks.

        Each present placement must independently identify the same earlier,
        currently admitted user observation. A null, conflicting or invalid
        placement is not rescued by a valid value in another location. This is
        structural compatibility, not validation of the model's interpretation.
        """
        present = []
        if field in candidate:
            present.append(candidate[field])
        for name in dict.fromkeys((field, payload_field)):
            if name in payload:
                present.append(payload[name])
        if not present:
            return None
        for target in present:
            if not isinstance(target, str) or not target.strip():
                raise ValueError("candidate link target must be an explicit source ID")
            target_source = self.raw_sources.get(target)
            source_scope = re.fullmatch(r'([ABC])-S\d{2}', source['source_id'])
            target_scope = re.fullmatch(r'([ABC])-S\d{2}', target)
            if (target_source is None or target in self.deleted_source_ids
                    or target_source["role"] != "user"
                    or target_source["storage_policy"] != "allowed"
                    or target_source["source_order"] >= source["source_order"]
                    or moment(target_source["date"]) > moment(source["date"])
                    or (target_source.get("valid_to")
                        and moment(target_source["valid_to"]) <= moment(source["date"]))
                    or (source_scope and target_scope and source_scope[1] != target_scope[1])
                    or self.store.get(self.user_id, self.quote_id(target)) is None):
                raise ValueError("candidate link target must be a prior eligible user source")
        if any(target != present[0] for target in present[1:]):
            raise ValueError("candidate link target placements conflict")
        return present[0]

    def apply_control(self, control):
        required = {"event_id", "action", "target_source_ids", "as_of", "source_order", "before_task_id"}
        if set(control) != required or control["action"] not in {"delete", "do_not_store", "new_session"}:
            raise ValueError("controller event schema differs")
        before = self.state_hash()
        target_ids = set(control["target_source_ids"])
        if not target_ids <= set(self.source_metadata):
            raise ValueError("control targets unseen source")
        removed = []
        if control["action"] == "new_session":
            self.session_sources.clear()
        else:
            for record in self.store.list_for_user(self.user_id):
                if self.record_dependencies.get(record.memory_id, set(record.source_evidence.turn_ids)) & target_ids:
                    self.manager.delete(self.user_id, record.memory_id, deleted_at=control["as_of"])
                    self.record_dependencies.pop(record.memory_id, None)
                    removed.append(record.memory_id)
            for sid in target_ids:
                self.raw_sources.pop(sid, None)
                if control["action"] == "delete":
                    self.session_sources.pop(sid, None)
                    self.deleted_source_ids.add(sid)
            for sid, dependencies in list(self.extraction_cache_dependencies.items()):
                if dependencies & target_ids:
                    self.extraction_cache.pop(sid, None)
                    self.extraction_cache_dependencies.pop(sid, None)
        self.events.append({"kind": "control", "event_id": control["event_id"], "action": control["action"],
                            "target_source_ids": sorted(target_ids), "removed_memory_ids": sorted(removed),
                            "state_before_sha256": before, "state_after_sha256": self.state_hash(),
                            "remote_provider_log_erasure_claimed": False})

    def eligible_sources(self, *, as_of):
        now = moment(as_of)
        return {sid: source for sid, source in self.raw_sources.items()
                if moment(source["date"]) <= now and (not source.get("valid_to") or now < moment(source["valid_to"]))}
