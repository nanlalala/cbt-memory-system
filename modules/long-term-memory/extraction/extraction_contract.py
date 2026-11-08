"""Provider-independent extraction contract with evidence verification."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from memory_models import (
    ConfirmationState,
    LifecycleStatus,
    MemoryRecord,
    MemoryType,
    Origin,
    Sensitivity,
)
from memory_contract import construct_memory, decode_json_document


EXTRACTION_SYSTEM_PROMPT = """You extract candidate long-term memories from one conversation session.
Return JSON only: {"candidates": [...]}. Store only information supported by quoted user turns.
Do not store transient small talk, assistant suggestions, diagnoses, or unsupported interpretations.
Allowed memory types: explicit_fact, preference, episodic_event, emotion_record, goal,
cbt_assignment, assignment_outcome, cognitive_pattern, user_correction, safety_signal.
Every extracted record is an unconfirmed candidate until a separate user-confirmation step.
Cognitive patterns must additionally be labelled as agent_inference.
Safety signals must use restricted sensitivity. Include source_turn_ids and an exact supporting_text quote.
If nothing is appropriate, return {"candidates": []}."""


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    turn_id: str
    role: str
    text: str
    source_time: str | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_extraction_output(
    raw_output: str,
    *,
    user_id: str,
    session_id: str,
    turns: list[ConversationTurn],
    extracted_at: str | None = None,
    schema_version: str = "1.0",
    permissions: dict | None = None,
) -> list[MemoryRecord]:
    """Parse model output only after exact provenance checks against user turns."""

    parsed = decode_json_document(raw_output)
    if type(parsed) is not dict or schema_version not in {"1.0", "2.0"}:
        raise ValueError("extraction output must be an object with a supported schema version")
    if schema_version == "2.0" and set(parsed) != {"candidates"}:
        raise ValueError("unexpected extraction envelope fields")
    candidates = parsed.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("extraction output must contain a candidates array")

    user_turns = {turn.turn_id: turn for turn in turns if turn.role == "user"}
    timestamp = extracted_at or _now()
    records: list[MemoryRecord] = []
    for item in candidates:
        if type(item) is not dict:
            raise ValueError("each extraction candidate must be an object")
        if schema_version == "2.0":
            allowed = {"memory_id", "memory_type", "content", "source_turn_ids", "supporting_text",
                       "confidence", "importance", "sensitivity", "structured_payload", "tags",
                       "valid_from", "valid_to", "supersedes_memory_id", "related_memory_ids",
                       "memory_class", "category", "classification_status", "claim_bindings"}
            if set(item) - allowed:
                raise ValueError("unexpected extraction candidate fields")
            if "memory_id" in item and (type(item["memory_id"]) is not str or not item["memory_id"].strip()):
                raise ValueError("memory_id must be a nonempty string")
        source_ids = item.get("source_turn_ids", [])
        if type(source_ids) is not list or any(type(x) is not str or not x for x in source_ids):
            raise ValueError("source_turn_ids must contain string ids")
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_turn_ids must contain unique ids")
        if not source_ids or not set(source_ids).issubset(user_turns):
            raise ValueError("every candidate must cite existing user turn ids")
        supporting_text = item.get("supporting_text", "")
        if type(supporting_text) is not str:
            raise ValueError("supporting_text must be a string")
        supporting_text = supporting_text.strip()
        joined_sources = "\n".join(user_turns[turn_id].text for turn_id in source_ids)
        if not supporting_text or supporting_text not in joined_sources:
            raise ValueError("supporting_text must be an exact quote from cited user turns")

        memory_type = MemoryType(str(item["memory_type"]))
        if memory_type == MemoryType.COGNITIVE_PATTERN:
            origin = Origin.AGENT_INFERENCE.value
        elif memory_type == MemoryType.USER_CORRECTION:
            origin = Origin.USER_CORRECTION.value
        else:
            origin = Origin.EXPLICIT_USER_STATEMENT.value
        # An extraction model can identify provenance, but it cannot confirm that
        # its own summary is entailed by the quote.  All model-produced records
        # therefore remain candidates until a separate user-confirmation step.
        confirmation = ConfirmationState.UNCONFIRMED.value

        sensitivity = str(item.get("sensitivity", Sensitivity.STANDARD.value))
        if memory_type == MemoryType.SAFETY_SIGNAL:
            sensitivity = Sensitivity.RESTRICTED.value

        observed_at = str(item.get("observed_at", timestamp))
        payload = item.get("structured_payload", {})
        # Construct only through the shared schema/semantic entry point.  In
        # particular, do not coerce model strings into numbers or numeric IDs.
        record_data = dict(
            schema_version=schema_version,
            memory_id=item.get("memory_id") or f"mem-{uuid.uuid4().hex}",
            user_id=user_id,
            memory_type=memory_type.value,
            content=item["content"],
            origin=origin,
            confirmation_state=confirmation,
            lifecycle_status=LifecycleStatus.CANDIDATE.value,
            sensitivity=sensitivity,
            confidence=item.get("confidence", 0.8),
            importance=item.get("importance", 0.5),
            created_at=timestamp,
            observed_at=observed_at,
            updated_at=timestamp,
            source_evidence=dict(
                session_id=session_id,
                turn_ids=list(source_ids),
                supporting_text=supporting_text,
            ),
            structured_payload=payload,
            tags=item.get("tags", []),
            valid_from=item.get("valid_from"),
            valid_to=item.get("valid_to"),
            supersedes_memory_id=item.get("supersedes_memory_id"),
            related_memory_ids=item.get("related_memory_ids", []),
        )
        if schema_version == "2.0":
            source_times = {user_turns[turn_id].source_time for turn_id in source_ids}
            source_time = source_times.pop() if len(source_times) == 1 else None
            record_data.update(
                memory_id=f"mem-{uuid.uuid4().hex}",
                observed_at=source_time or timestamp,
                source_time=source_time,
                source_role="user",
                permissions=permissions if permissions is not None else {
                    "save": None, "long_term_use": None,
                    "response_external": None, "extraction_external": None,
                },
                representation_kind="structured_record",
                memory_class=item.get("memory_class"), category=item.get("category"),
                classification_status=item.get("classification_status", "pending"),
                semantic_support={"status": "not_checked", "method": None, "version": None},
                claim_bindings=item.get("claim_bindings", []),
            )
            # Span binding verifies only quotation existence, never entailment.
            for binding in record_data["claim_bindings"]:
                if (type(binding) is not dict or binding.get("turn_id") not in user_turns
                        or type(binding.get("start")) is not int or type(binding.get("end")) is not int):
                    raise ValueError("malformed extraction claim binding")
                source = user_turns[binding["turn_id"]].text
                if source[binding["start"]:binding["end"]] != binding.get("quote"):
                    raise ValueError("claim quote does not match its source span")
        record = construct_memory(record_data)
        records.append(record)
    return records


def build_extraction_request(turns: list[ConversationTurn]) -> str:
    return json.dumps(
        {"turns": [{"turn_id": item.turn_id, "role": item.role, "text": item.text} for item in turns]},
        ensure_ascii=False,
    )
