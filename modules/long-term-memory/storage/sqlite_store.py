"""SQLite persistence for user-scoped memory records and audit events."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from memory_models import MemoryRecord


class SQLiteMemoryStore:
    """Small reference store with strict user filtering on every read."""

    def __init__(self, database: str | Path = ":memory:") -> None:
        self.database = str(database)
        self.connection = sqlite3.connect(self.database, timeout=10.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA busy_timeout = 10000")
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    @contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        """Open a transaction, nesting safely inside an existing manager operation.

        ``immediate=True`` acquires SQLite's write reservation before predicate
        reads, preventing two writers from both deciding that the same slot is
        empty.
        """

        if self.connection.in_transaction:
            yield self.connection
            return
        try:
            self.connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield self.connection
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def _create_schema(self) -> None:
        with self.transaction() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    lifecycle_status TEXT NOT NULL,
                    sensitivity TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    record_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memories_user_status
                    ON memories(user_id, lifecycle_status);
                CREATE INDEX IF NOT EXISTS idx_memories_user_type
                    ON memories(user_id, memory_type);
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    details_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_audit_user_memory
                    ON audit_events(user_id, memory_id, event_id);
                """
            )

    def put(self, record: MemoryRecord) -> None:
        self.atomic_write(records=[record])

    def _put(self, connection: sqlite3.Connection, record: MemoryRecord) -> None:
        payload = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
        existing = connection.execute(
            "SELECT user_id FROM memories WHERE memory_id = ?", (record.memory_id,)
        ).fetchone()
        if existing and existing["user_id"] != record.user_id:
            raise PermissionError("memory_id belongs to a different user")
        connection.execute(
            """
            INSERT INTO memories(
                memory_id, user_id, memory_type, content, lifecycle_status,
                sensitivity, updated_at, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(memory_id) DO UPDATE SET
                memory_type=excluded.memory_type,
                content=excluded.content,
                lifecycle_status=excluded.lifecycle_status,
                sensitivity=excluded.sensitivity,
                updated_at=excluded.updated_at,
                record_json=excluded.record_json
            """,
            (
                record.memory_id,
                record.user_id,
                record.memory_type,
                record.content,
                record.lifecycle_status,
                record.sensitivity,
                record.updated_at,
                payload,
            ),
        )

    def get(self, user_id: str, memory_id: str) -> MemoryRecord | None:
        row = self.connection.execute(
            "SELECT record_json FROM memories WHERE user_id = ? AND memory_id = ?",
            (user_id, memory_id),
        ).fetchone()
        return MemoryRecord.from_dict(json.loads(row["record_json"])) if row else None

    def list_for_user(
        self,
        user_id: str,
        *,
        lifecycle_statuses: set[str] | None = None,
        memory_types: set[str] | None = None,
    ) -> list[MemoryRecord]:
        rows = self.connection.execute(
            "SELECT record_json FROM memories WHERE user_id = ? ORDER BY updated_at, memory_id",
            (user_id,),
        ).fetchall()
        records = [MemoryRecord.from_dict(json.loads(row["record_json"])) for row in rows]
        if lifecycle_statuses is not None:
            records = [item for item in records if item.lifecycle_status in lifecycle_statuses]
        if memory_types is not None:
            records = [item for item in records if item.memory_type in memory_types]
        return records

    def list_user_ids(self) -> list[str]:
        rows = self.connection.execute(
            "SELECT DISTINCT user_id FROM memories ORDER BY user_id"
        ).fetchall()
        return [row["user_id"] for row in rows]

    def hard_delete(self, user_id: str, memory_id: str) -> bool:
        return self.atomic_write(deletes=[(user_id, memory_id)]) == 1

    def append_audit(
        self,
        *,
        user_id: str,
        memory_id: str,
        action: str,
        occurred_at: str,
        actor: str,
        details: dict | None = None,
    ) -> None:
        self.atomic_write(
            audits=[
                {
                    "user_id": user_id,
                    "memory_id": memory_id,
                    "action": action,
                    "occurred_at": occurred_at,
                    "actor": actor,
                    "details": details or {},
                }
            ]
        )

    def atomic_write(
        self,
        *,
        records: list[MemoryRecord] | None = None,
        audits: list[dict[str, Any]] | None = None,
        deletes: list[tuple[str, str]] | None = None,
    ) -> int:
        """Commit a complete lifecycle transition and its audit trail together."""

        records = records or []
        audits = audits or []
        deletes = deletes or []
        for record in records:
            record.validate()
        deleted_count = 0
        with self.transaction() as connection:
            for record in records:
                self._put(connection, record)
            for user_id, memory_id in deletes:
                cursor = connection.execute(
                    "DELETE FROM memories WHERE user_id = ? AND memory_id = ?",
                    (user_id, memory_id),
                )
                deleted_count += cursor.rowcount
            for event in audits:
                connection.execute(
                    """
                    INSERT INTO audit_events(
                        user_id, memory_id, action, occurred_at, actor, details_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event["user_id"],
                        event["memory_id"],
                        event["action"],
                        event["occurred_at"],
                        event["actor"],
                        json.dumps(event.get("details", {}), ensure_ascii=False, sort_keys=True),
                    ),
                )
        return deleted_count

    def audit_for_user(self, user_id: str) -> list[dict]:
        rows = self.connection.execute(
            "SELECT * FROM audit_events WHERE user_id = ? ORDER BY event_id", (user_id,)
        ).fetchall()
        return [
            {
                "event_id": row["event_id"],
                "user_id": row["user_id"],
                "memory_id": row["memory_id"],
                "action": row["action"],
                "occurred_at": row["occurred_at"],
                "actor": row["actor"],
                "details": json.loads(row["details_json"]),
            }
            for row in rows
        ]
