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
    SourceEvidence,
)


EXTRACTION_SYSTEM_PROMPT = """You extract candidate long-term memories from one conversation session.
Return JSON only: {"candidates": [...]}. Store only information supported by quoted user turns.
Do not store transient small talk, assistant suggestions, diagnoses, or unsupported interpretations.
Allowed memory types: explicit_fact, preference, episodic_event, emotion_record, goal,
cbt_assignment, assignment_outcome, cognitive_pattern, user_correction, safety_signal.
Cognitive patterns are agent_inference and must remain unconfirmed candidates.
Safety signals must use restricted sensitivity. Include source_turn_ids and an exact supporting_text quote.
If nothing is appropriate, return {"candidates": []}."""


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    turn_id: str
    role: str
    text: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_extraction_output(
    raw_output: str,
    *,
    user_id: str,
    session_id: str,
    turns: list[ConversationTurn],
    extracted_at: str | None = None,
) -> list[MemoryRecord]:
    """Parse model output only after exact provenance checks against user turns."""

    parsed = json.loads(raw_output)
    candidates = parsed.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("extraction output must contain a candidates array")

    user_turns = {turn.turn_id: turn for turn in turns if turn.role == "user"}
    timestamp = extracted_at or _now()
    records: list[MemoryRecord] = []
    for item in candidates:
        source_ids = item.get("source_turn_ids", [])
        if not source_ids or not set(source_ids).issubset(user_turns):
            raise ValueError("every candidate must cite existing user turn ids")
        supporting_text = str(item.get("supporting_text", "")).strip()
        joined_sources = "\n".join(user_turns[turn_id].text for turn_id in source_ids)
        if not supporting_text or supporting_text not in joined_sources:
            raise ValueError("supporting_text must be an exact quote from cited user turns")

        memory_type = MemoryType(str(item["memory_type"]))
        if memory_type == MemoryType.COGNITIVE_PATTERN:
            origin = Origin.AGENT_INFERENCE.value
            confirmation = ConfirmationState.UNCONFIRMED.value
        elif memory_type == MemoryType.USER_CORRECTION:
            origin = Origin.USER_CORRECTION.value
            confirmation = ConfirmationState.USER_CORRECTED.value
        else:
            origin = Origin.EXPLICIT_USER_STATEMENT.value
            confirmation = ConfirmationState.USER_CONFIRMED.value

        sensitivity = str(item.get("sensitivity", Sensitivity.STANDARD.value))
        if memory_type == MemoryType.SAFETY_SIGNAL:
            sensitivity = Sensitivity.RESTRICTED.value

        observed_at = str(item.get("observed_at", timestamp))
        payload = dict(item.get("structured_payload", {}))
        record = MemoryRecord(
            schema_version="1.0",
            memory_id=str(item.get("memory_id") or f"mem-{uuid.uuid4().hex}"),
            user_id=user_id,
            memory_type=memory_type.value,
            content=str(item["content"]).strip(),
            origin=origin,
            confirmation_state=confirmation,
            lifecycle_status=LifecycleStatus.CANDIDATE.value,
            sensitivity=sensitivity,
            confidence=float(item.get("confidence", 0.8)),
            importance=float(item.get("importance", 0.5)),
            created_at=timestamp,
            observed_at=observed_at,
            updated_at=timestamp,
            source_evidence=SourceEvidence(
                session_id=session_id,
                turn_ids=list(source_ids),
                supporting_text=supporting_text,
            ),
            structured_payload=payload,
            tags=[str(tag) for tag in item.get("tags", [])],
            valid_from=item.get("valid_from"),
            valid_to=item.get("valid_to"),
            supersedes_memory_id=item.get("supersedes_memory_id"),
            related_memory_ids=[str(value) for value in item.get("related_memory_ids", [])],
        )
        record.validate()
        records.append(record)
    return records


def build_extraction_request(turns: list[ConversationTurn]) -> str:
    return json.dumps(
        {"turns": [{"turn_id": item.turn_id, "role": item.role, "text": item.text} for item in turns]},
        ensure_ascii=False,
    )
