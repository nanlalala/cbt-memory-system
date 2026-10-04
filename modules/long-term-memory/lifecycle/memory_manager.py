"""Memory write path and lifecycle rules for Memory Manager v1."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from memory_models import (
    ConfirmationState,
    LifecycleStatus,
    MemoryRecord,
    MemoryType,
    Origin,
    Sensitivity,
    is_valid_at,
)
from sqlite_store import SQLiteMemoryStore


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _normalise(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold()).strip()


@dataclass(slots=True)
class WriteResult:
    action: str
    memory_id: str
    related_memory_id: str | None = None
    reason: str = ""


class MemoryManager:
    """Apply deterministic safety and lifecycle rules before persistence."""

    def __init__(self, store: SQLiteMemoryStore) -> None:
        self.store = store

    def ingest(self, record: MemoryRecord, *, actor: str = "system") -> WriteResult:
        """Serialize predicate reads and writes for one complete ingest decision."""

        record.validate()
        with self.store.transaction(immediate=True):
            return self._ingest_locked(record, actor=actor)

    def _ingest_locked(self, record: MemoryRecord, *, actor: str) -> WriteResult:
        existing = self.store.get(record.user_id, record.memory_id)
        if existing:
            raise ValueError(f"memory_id already exists: {record.memory_id}")

        active = self.store.list_for_user(
            record.user_id,
            lifecycle_statuses={LifecycleStatus.ACTIVE.value, LifecycleStatus.CANDIDATE.value},
            memory_types={record.memory_type},
        )
        duplicate = next(
            (item for item in active if _normalise(item.content) == _normalise(record.content)), None
        )
        if duplicate:
            self.store.append_audit(
                user_id=record.user_id,
                memory_id=duplicate.memory_id,
                action="duplicate_rejected",
                occurred_at=record.updated_at,
                actor=actor,
                details={"rejected_memory_id": record.memory_id},
            )
            return WriteResult(
                "duplicate_rejected", record.memory_id, duplicate.memory_id, "normalised content match"
            )

        if record.memory_type == MemoryType.USER_CORRECTION.value:
            if record.confirmation_state != ConfirmationState.USER_CORRECTED.value:
                candidate = replace(
                    record,
                    lifecycle_status=LifecycleStatus.CANDIDATE.value,
                    confirmation_state=ConfirmationState.UNCONFIRMED.value,
                )
                self.store.atomic_write(
                    records=[candidate],
                    audits=[
                        {
                            "user_id": candidate.user_id,
                            "memory_id": candidate.memory_id,
                            "action": "correction_requires_confirmation",
                            "occurred_at": candidate.updated_at,
                            "actor": actor,
                            "details": {"target_memory_id": candidate.supersedes_memory_id},
                        }
                    ],
                )
                return WriteResult(
                    "correction_requires_confirmation",
                    candidate.memory_id,
                    candidate.supersedes_memory_id,
                )
            return self._apply_correction(record, actor=actor)

        conflict = self._find_slot_conflict(record, active)
        if conflict:
            candidate = replace(
                record,
                lifecycle_status=LifecycleStatus.CANDIDATE.value,
                confirmation_state=ConfirmationState.UNCONFIRMED.value,
            )
            self.store.atomic_write(
                records=[candidate],
                audits=[
                    {
                        "user_id": record.user_id,
                        "memory_id": record.memory_id,
                        "action": "conflict_requires_confirmation",
                        "occurred_at": record.updated_at,
                        "actor": actor,
                        "details": {"conflicts_with": conflict.memory_id},
                    }
                ],
            )
            return WriteResult(
                "conflict_requires_confirmation",
                record.memory_id,
                conflict.memory_id,
                "same slot_key with different content",
            )

        status = record.lifecycle_status
        if record.confirmation_state not in {
            ConfirmationState.USER_CONFIRMED.value,
            ConfirmationState.USER_CORRECTED.value,
        }:
            status = LifecycleStatus.CANDIDATE.value
        elif status == LifecycleStatus.CANDIDATE.value and record.origin in {
            Origin.EXPLICIT_USER_STATEMENT.value,
            Origin.USER_CORRECTION.value,
        }:
            status = LifecycleStatus.ACTIVE.value

        stored = replace(record, lifecycle_status=status)
        self.store.atomic_write(
            records=[stored],
            audits=[
                {
                    "user_id": stored.user_id,
                    "memory_id": stored.memory_id,
                    "action": "created",
                    "occurred_at": stored.updated_at,
                    "actor": actor,
                    "details": {"status": status, "origin": stored.origin},
                }
            ],
        )
        return WriteResult("created", stored.memory_id, reason=f"stored as {status}")

    def _find_slot_conflict(
        self, record: MemoryRecord, active: list[MemoryRecord]
    ) -> MemoryRecord | None:
        slot = record.structured_payload.get("slot_key")
        if not slot:
            return None
        return next(
            (
                item
                for item in active
                if item.structured_payload.get("slot_key") == slot
                and _normalise(item.content) != _normalise(record.content)
            ),
            None,
        )

    def _apply_correction(self, correction: MemoryRecord, *, actor: str) -> WriteResult:
        target_id = correction.supersedes_memory_id or ""
        target = self.store.get(correction.user_id, target_id)
        if target is None:
            raise ValueError("correction target does not exist for this user")
        if target.lifecycle_status != LifecycleStatus.ACTIVE.value:
            raise ValueError("correction target must be the active current version")

        timestamp = correction.updated_at
        superseded = replace(
            target,
            lifecycle_status=LifecycleStatus.SUPERSEDED.value,
            updated_at=timestamp,
        )
        active_correction = replace(
            correction,
            lifecycle_status=LifecycleStatus.ACTIVE.value,
            confirmation_state=ConfirmationState.USER_CORRECTED.value,
        )
        self.store.atomic_write(
            records=[superseded, active_correction],
            audits=[
                {
                    "user_id": correction.user_id,
                    "memory_id": target.memory_id,
                    "action": "superseded",
                    "occurred_at": timestamp,
                    "actor": actor,
                    "details": {"replacement_memory_id": correction.memory_id},
                },
                {
                    "user_id": correction.user_id,
                    "memory_id": correction.memory_id,
                    "action": "correction_applied",
                    "occurred_at": timestamp,
                    "actor": actor,
                    "details": {"supersedes": target.memory_id},
                },
            ],
        )
        return WriteResult("correction_applied", correction.memory_id, target.memory_id)

    def confirm_candidate(
        self,
        user_id: str,
        memory_id: str,
        *,
        consent_event_id: str,
        confirmation_turn_ids: list[str] | None = None,
        confirmed_at: str | None = None,
    ) -> MemoryRecord:
        if not consent_event_id.strip():
            raise ValueError("consent_event_id is required for user confirmation")
        with self.store.transaction(immediate=True):
            return self._confirm_candidate_locked(
                user_id,
                memory_id,
                consent_event_id=consent_event_id,
                confirmation_turn_ids=confirmation_turn_ids or [],
                confirmed_at=confirmed_at,
            )

    def _confirm_candidate_locked(
        self,
        user_id: str,
        memory_id: str,
        *,
        consent_event_id: str,
        confirmation_turn_ids: list[str],
        confirmed_at: str | None,
    ) -> MemoryRecord:
        record = self._require(user_id, memory_id)
        if record.lifecycle_status != LifecycleStatus.CANDIDATE.value:
            raise ValueError("only candidate memories can be confirmed")
        timestamp = confirmed_at or utc_now()
        if record.memory_type == MemoryType.USER_CORRECTION.value:
            correction = replace(
                record,
                confirmation_state=ConfirmationState.USER_CORRECTED.value,
                updated_at=timestamp,
            )
            self._apply_correction(correction, actor="user")
            self.store.append_audit(
                user_id=user_id,
                memory_id=memory_id,
                action="confirmation_evidence",
                occurred_at=timestamp,
                actor="user",
                details={
                    "consent_event_id": consent_event_id,
                    "confirmation_turn_ids": confirmation_turn_ids,
                },
            )
            return self._require(user_id, memory_id)

        slot = record.structured_payload.get("slot_key")
        changed: list[MemoryRecord] = []
        audits: list[dict] = []
        if slot:
            for current in self.store.list_for_user(
                user_id,
                lifecycle_statuses={LifecycleStatus.ACTIVE.value},
                memory_types={record.memory_type},
            ):
                if current.structured_payload.get("slot_key") == slot:
                    changed.append(
                        replace(
                            current,
                            lifecycle_status=LifecycleStatus.SUPERSEDED.value,
                            updated_at=timestamp,
                        )
                    )
                    audits.append(
                        {
                            "user_id": user_id,
                            "memory_id": current.memory_id,
                            "action": "superseded_after_confirmation",
                            "occurred_at": timestamp,
                            "actor": "user",
                            "details": {"replacement_memory_id": memory_id},
                        }
                    )
        confirmed = replace(
            record,
            confirmation_state=ConfirmationState.USER_CONFIRMED.value,
            lifecycle_status=LifecycleStatus.ACTIVE.value,
            updated_at=timestamp,
        )
        changed.append(confirmed)
        audits.append(
            {
                "user_id": user_id,
                "memory_id": memory_id,
                "action": "user_confirmed",
                "occurred_at": timestamp,
                "actor": "user",
                "details": {
                    "consent_event_id": consent_event_id,
                    "confirmation_turn_ids": confirmation_turn_ids,
                },
            }
        )
        self.store.atomic_write(records=changed, audits=audits)
        return confirmed

    def expire_due(self, *, as_of: str | None = None) -> list[str]:
        """Expire due records without racing a concurrent hard deletion."""

        with self.store.transaction(immediate=True):
            return self._expire_due_locked(as_of=as_of)

    def _expire_due_locked(self, *, as_of: str | None = None) -> list[str]:
        timestamp = as_of or utc_now()
        cutoff = _parse_time(timestamp)
        expired: list[str] = []
        changed: list[MemoryRecord] = []
        audits: list[dict] = []
        for user_id in self.store.list_user_ids():
            for record in self.store.list_for_user(
                user_id, lifecycle_statuses={LifecycleStatus.ACTIVE.value}
            ):
                if record.valid_to and _parse_time(record.valid_to) <= cutoff:
                    updated = replace(
                        record,
                        lifecycle_status=LifecycleStatus.EXPIRED.value,
                        updated_at=timestamp,
                    )
                    changed.append(updated)
                    audits.append(
                        {
                            "user_id": user_id,
                            "memory_id": record.memory_id,
                            "action": "expired",
                            "occurred_at": timestamp,
                            "actor": "system",
                            "details": {},
                        }
                    )
                    expired.append(record.memory_id)
        if changed:
            self.store.atomic_write(records=changed, audits=audits)
        return expired

    def delete(self, user_id: str, memory_id: str, *, deleted_at: str | None = None) -> bool:
        """Hard-delete one record and its audit event in one serialized decision."""

        with self.store.transaction(immediate=True):
            return self._delete_locked(user_id, memory_id, deleted_at=deleted_at)

    def _delete_locked(
        self, user_id: str, memory_id: str, *, deleted_at: str | None = None
    ) -> bool:
        record = self._require(user_id, memory_id)
        timestamp = deleted_at or utc_now()
        removed = self.store.atomic_write(
            deletes=[(user_id, memory_id)],
            audits=[
                {
                    "user_id": user_id,
                    "memory_id": memory_id,
                    "action": "hard_deleted",
                    "occurred_at": timestamp,
                    "actor": "user",
                    "details": {"memory_type": record.memory_type},
                }
            ],
        )
        return removed == 1

    def eligible_context(
        self,
        user_id: str,
        *,
        as_of: str | None = None,
        allow_restricted: bool = False,
        memory_types: set[str] | None = None,
    ) -> list[MemoryRecord]:
        timestamp = _parse_time(as_of or utc_now())
        records = self.store.list_for_user(
            user_id,
            lifecycle_statuses={LifecycleStatus.ACTIVE.value},
            memory_types=memory_types,
        )
        eligible = []
        for record in records:
            if record.sensitivity == Sensitivity.RESTRICTED.value and not allow_restricted:
                continue
            if not is_valid_at(record, timestamp):
                continue
            eligible.append(record)
        return eligible

    def _require(self, user_id: str, memory_id: str) -> MemoryRecord:
        record = self.store.get(user_id, memory_id)
        if record is None:
            raise KeyError("memory not found for this user")
        return record
