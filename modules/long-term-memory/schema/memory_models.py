"""Dependency-free data model and validation for Memory Schema v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class MemoryType(str, Enum):
    EXPLICIT_FACT = "explicit_fact"
    PREFERENCE = "preference"
    EPISODIC_EVENT = "episodic_event"
    EMOTION_RECORD = "emotion_record"
    GOAL = "goal"
    CBT_ASSIGNMENT = "cbt_assignment"
    ASSIGNMENT_OUTCOME = "assignment_outcome"
    COGNITIVE_PATTERN = "cognitive_pattern"
    USER_CORRECTION = "user_correction"
    SAFETY_SIGNAL = "safety_signal"


class Origin(str, Enum):
    EXPLICIT_USER_STATEMENT = "explicit_user_statement"
    USER_CORRECTION = "user_correction"
    AGENT_INFERENCE = "agent_inference"
    SYSTEM_OBSERVATION = "system_observation"


class ConfirmationState(str, Enum):
    UNCONFIRMED = "unconfirmed"
    USER_CONFIRMED = "user_confirmed"
    USER_CORRECTED = "user_corrected"
    NOT_APPLICABLE = "not_applicable"


class LifecycleStatus(str, Enum):
    CANDIDATE = "candidate"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"
    DELETED = "deleted"


class Sensitivity(str, Enum):
    STANDARD = "standard"
    SENSITIVE = "sensitive"
    RESTRICTED = "restricted"


def _enum_value(enum_class: type[Enum], value: str, field_name: str) -> str:
    try:
        return enum_class(value).value
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_class)
        raise ValueError(f"{field_name} must be one of: {allowed}") from exc


def _timestamp(value: str, field_name: str) -> str:
    if not value:
        raise ValueError(f"{field_name} is required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field_name} must include a timezone")
    return value


def _score(value: float, field_name: str) -> float:
    number = float(value)
    if not 0 <= number <= 1:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return number


@dataclass(slots=True)
class SourceEvidence:
    session_id: str
    turn_ids: list[str]
    supporting_text: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceEvidence":
        evidence = cls(
            session_id=str(data.get("session_id", "")).strip(),
            turn_ids=[str(item).strip() for item in data.get("turn_ids", [])],
            supporting_text=str(data.get("supporting_text", "")).strip(),
        )
        if not evidence.session_id or not evidence.turn_ids or not evidence.supporting_text:
            raise ValueError("source_evidence requires session_id, turn_ids and supporting_text")
        if any(not item for item in evidence.turn_ids):
            raise ValueError("source_evidence.turn_ids cannot contain empty values")
        return evidence


@dataclass(slots=True)
class MemoryRecord:
    schema_version: str
    memory_id: str
    user_id: str
    memory_type: str
    content: str
    origin: str
    confirmation_state: str
    lifecycle_status: str
    sensitivity: str
    confidence: float
    importance: float
    created_at: str
    observed_at: str
    updated_at: str
    source_evidence: SourceEvidence
    structured_payload: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    valid_from: str | None = None
    valid_to: str | None = None
    supersedes_memory_id: str | None = None
    related_memory_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRecord":
        required = {
            "schema_version",
            "memory_id",
            "user_id",
            "memory_type",
            "content",
            "origin",
            "confirmation_state",
            "lifecycle_status",
            "sensitivity",
            "confidence",
            "importance",
            "created_at",
            "observed_at",
            "updated_at",
            "source_evidence",
        }
        missing = sorted(required - data.keys())
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")

        record = cls(
            schema_version=str(data["schema_version"]),
            memory_id=str(data["memory_id"]).strip(),
            user_id=str(data["user_id"]).strip(),
            memory_type=_enum_value(MemoryType, str(data["memory_type"]), "memory_type"),
            content=str(data["content"]).strip(),
            origin=_enum_value(Origin, str(data["origin"]), "origin"),
            confirmation_state=_enum_value(
                ConfirmationState, str(data["confirmation_state"]), "confirmation_state"
            ),
            lifecycle_status=_enum_value(
                LifecycleStatus, str(data["lifecycle_status"]), "lifecycle_status"
            ),
            sensitivity=_enum_value(Sensitivity, str(data["sensitivity"]), "sensitivity"),
            confidence=_score(data["confidence"], "confidence"),
            importance=_score(data["importance"], "importance"),
            created_at=_timestamp(str(data["created_at"]), "created_at"),
            observed_at=_timestamp(str(data["observed_at"]), "observed_at"),
            updated_at=_timestamp(str(data["updated_at"]), "updated_at"),
            source_evidence=SourceEvidence.from_dict(data["source_evidence"]),
            structured_payload=dict(data.get("structured_payload", {})),
            tags=[str(item).strip() for item in data.get("tags", [])],
            valid_from=data.get("valid_from"),
            valid_to=data.get("valid_to"),
            supersedes_memory_id=data.get("supersedes_memory_id"),
            related_memory_ids=[str(item).strip() for item in data.get("related_memory_ids", [])],
        )
        record.validate()
        return record

    def validate(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("schema_version must be 1.0")
        if not self.memory_id or not self.user_id:
            raise ValueError("memory_id and user_id cannot be empty")
        if self.lifecycle_status != LifecycleStatus.DELETED.value and not self.content:
            raise ValueError("content cannot be empty unless the record is a deletion tombstone")
        if any(not item for item in self.tags + self.related_memory_ids):
            raise ValueError("tags and related_memory_ids cannot contain empty values")

        if self.valid_from:
            _timestamp(self.valid_from, "valid_from")
        if self.valid_to:
            _timestamp(self.valid_to, "valid_to")
        if self.valid_from and self.valid_to:
            start = datetime.fromisoformat(self.valid_from.replace("Z", "+00:00"))
            end = datetime.fromisoformat(self.valid_to.replace("Z", "+00:00"))
            if end < start:
                raise ValueError("valid_to cannot be earlier than valid_from")

        if self.origin == Origin.AGENT_INFERENCE.value and self.confirmation_state not in {
            ConfirmationState.UNCONFIRMED.value,
            ConfirmationState.USER_CONFIRMED.value,
        }:
            raise ValueError("agent inferences must remain unconfirmed until the user confirms them")

        if self.memory_type == MemoryType.COGNITIVE_PATTERN.value:
            if self.origin != Origin.AGENT_INFERENCE.value:
                raise ValueError("cognitive_pattern records must be marked as agent_inference")
            if "evidence_count" not in self.structured_payload:
                raise ValueError("cognitive_pattern requires structured_payload.evidence_count")

        if self.memory_type == MemoryType.USER_CORRECTION.value:
            if self.origin != Origin.USER_CORRECTION.value:
                raise ValueError("user_correction records require origin=user_correction")
            if not self.supersedes_memory_id:
                raise ValueError("user_correction requires supersedes_memory_id")
            if "corrected_content" not in self.structured_payload:
                raise ValueError("user_correction requires structured_payload.corrected_content")

        if self.memory_type == MemoryType.SAFETY_SIGNAL.value:
            if self.sensitivity != Sensitivity.RESTRICTED.value:
                raise ValueError("safety_signal records require restricted sensitivity")

        required_payload_fields = {
            MemoryType.EMOTION_RECORD.value: {"emotion", "intensity", "situation"},
            MemoryType.GOAL.value: {"description", "status"},
            MemoryType.CBT_ASSIGNMENT.value: {"description", "status"},
            MemoryType.ASSIGNMENT_OUTCOME.value: {"assignment_memory_id", "completion_status"},
        }
        expected = required_payload_fields.get(self.memory_type, set())
        absent = sorted(expected - self.structured_payload.keys())
        if absent:
            raise ValueError(
                f"{self.memory_type} is missing structured_payload fields: {', '.join(absent)}"
            )
        if self.memory_type == MemoryType.EMOTION_RECORD.value:
            intensity = self.structured_payload["intensity"]
            if not isinstance(intensity, int) or not 0 <= intensity <= 10:
                raise ValueError("emotion_record intensity must be an integer from 0 to 10")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

