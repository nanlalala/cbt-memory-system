"""Local, synthetic walkthrough of memory confirmation and deletion.

This is software demonstration code, not an outcome experiment or a clinical
chatbot.  No model, network, real patient data or crisis router is involved.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
# Keep this bootstrap bounded to the repository's existing modules.  It lets
# `python app/memory_demo.py` work before installing the optional research backends.
for relative in (
    "modules/long-term-memory/schema",
    "modules/long-term-memory/storage",
    "modules/long-term-memory/lifecycle",
    "modules/long-term-memory/extraction",
    "modules/long-term-memory/retrieval",
    "modules/agent-integration",
):
    module_path = str(ROOT / relative)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

from context_builder import ContextPackage, build_context  # noqa: E402
from extraction_contract import ConversationTurn, parse_extraction_output  # noqa: E402
from memory_manager import MemoryManager  # noqa: E402
from memory_models import MemoryRecord  # noqa: E402
from memory_retriever import MemoryRetriever  # noqa: E402
from sqlite_store import SQLiteMemoryStore  # noqa: E402


DISCLAIMER = (
    "CBT Memory System — SYNTHETIC SOFTWARE DEMO / 合成数据软件演示\n"
    "仅展示记忆状态与审计；不是治疗工具、聊天模型或安全路由。\n"
    "Only the supplied fictional fixtures are supported. Do not enter personal or clinical data."
)
INITIAL_QUOTE = "本周的目标是午饭后散步两次。"
CORRECTED_QUOTE = "我更正一下，本周的目标是午饭后散步三次，不是两次。"
CORRECTED_CONTENT = "本周的目标是午饭后散步三次。"
DEFAULT_QUERY = "我本周的散步目标是什么？"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class DemoSession:
    """One isolated fictional user's walkthrough backed by the real modules."""

    def __init__(self, store: SQLiteMemoryStore, *, run_id: str | None = None) -> None:
        self.store = store
        self.manager = MemoryManager(store)
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.user_id = f"synthetic-demo-{self.run_id}"
        self.initial_id = f"demo-{self.run_id}-goal"
        self.correction_id = f"demo-{self.run_id}-correction"

    def records(self) -> list[MemoryRecord]:
        return self.store.list_for_user(self.user_id)

    def stage_initial(self) -> MemoryRecord:
        """Parse a checked-in JSON-shaped fixture; do not claim live extraction."""

        raw = {
            "candidates": [
                {
                    "memory_id": self.initial_id,
                    "memory_type": "goal",
                    "content": INITIAL_QUOTE,
                    "source_turn_ids": [f"{self.run_id}-fixture-turn-1"],
                    "supporting_text": INITIAL_QUOTE,
                    "structured_payload": {
                        "description": INITIAL_QUOTE,
                        "status": "active",
                        "slot_key": "goal:weekly-walk",
                    },
                }
            ]
        }
        turn = ConversationTurn(f"{self.run_id}-fixture-turn-1", "user", INITIAL_QUOTE)
        record = parse_extraction_output(
            json.dumps(raw, ensure_ascii=False),
            user_id=self.user_id,
            session_id=f"{self.run_id}-fictional-session-1",
            turns=[turn],
        )[0]
        self.manager.ingest(record, actor="demo_fixture")
        return self.store.get(self.user_id, record.memory_id)

    def current_active(self) -> MemoryRecord:
        active = [item for item in self.records() if item.lifecycle_status == "active"]
        if len(active) != 1:
            raise ValueError("Demo needs exactly one confirmed active version first.")
        return active[0]

    def stage_correction(self) -> MemoryRecord:
        target = self.current_active()
        raw = {
            "candidates": [
                {
                    "memory_id": self.correction_id,
                    "memory_type": "user_correction",
                    "content": CORRECTED_CONTENT,
                    "source_turn_ids": [f"{self.run_id}-fixture-turn-2"],
                    "supporting_text": CORRECTED_QUOTE,
                    "supersedes_memory_id": target.memory_id,
                    "structured_payload": {
                        "corrected_content": CORRECTED_CONTENT,
                        "slot_key": "goal:weekly-walk",
                    },
                }
            ]
        }
        turn = ConversationTurn(f"{self.run_id}-fixture-turn-2", "user", CORRECTED_QUOTE)
        record = parse_extraction_output(
            json.dumps(raw, ensure_ascii=False),
            user_id=self.user_id,
            session_id=f"{self.run_id}-fictional-session-2",
            turns=[turn],
        )[0]
        self.manager.ingest(record, actor="demo_fixture")
        return self.store.get(self.user_id, record.memory_id)

    def confirm_command(
        self, memory_id: str, command: str, *, input_source: str = "interactive_input"
    ) -> MemoryRecord:
        """Bind a typed command (or explicitly labelled fixture) to consent audit."""

        if command.strip() != f"CONFIRM {memory_id}":
            raise ValueError("Confirmation cancelled: exact CONFIRM <memory-id> is required.")
        if input_source not in {"interactive_input", "scripted_fixture"}:
            raise ValueError("Unknown consent input source.")
        event_id = f"demo-consent-{uuid.uuid4().hex}"
        turn_id = f"demo-confirm-turn-{uuid.uuid4().hex}"
        timestamp = _now()
        with self.store.transaction(immediate=True):
            confirmed = self.manager.confirm_candidate(
                self.user_id,
                memory_id,
                consent_event_id=event_id,
                confirmation_turn_ids=[turn_id],
                confirmed_at=timestamp,
            )
            self.store.append_audit(
                user_id=self.user_id,
                memory_id=memory_id,
                action="demo_consent_event",
                occurred_at=timestamp,
                actor="demo_operator" if input_source == "interactive_input" else "scripted_fixture",
                details={
                    "consent_event_id": event_id,
                    "confirmation_turn_ids": [turn_id],
                    "input_source": input_source,
                    "command": command.strip(),
                    "scope": "synthetic_demo_only",
                },
            )
        return confirmed

    def retrieve(self, query: str = DEFAULT_QUERY) -> ContextPackage:
        retrieved = MemoryRetriever(self.records()).search(query, user_id=self.user_id)
        return build_context(
            short_term_summary="Fictional demo asks about the current weekly walking goal.",
            personal_memories=retrieved,
        )

    def delete_all_command(self, command: str) -> list[str]:
        """Delete every version created in this fictional run, not audit metadata."""

        if command.strip() != f"DELETE ALL {self.run_id}":
            raise ValueError("Deletion cancelled: exact DELETE ALL <run-id> is required.")
        deleted = []
        with self.store.transaction(immediate=True):
            for record in self.records():
                if self.manager.delete(self.user_id, record.memory_id):
                    deleted.append(record.memory_id)
        return deleted


def scripted_walkthrough(store: SQLiteMemoryStore, *, run_id: str | None = None) -> dict:
    """Software smoke test with simulated confirmations, never user/clinical evidence."""

    session = DemoSession(store, run_id=run_id)
    initial = session.stage_initial()
    before = session.retrieve()
    session.confirm_command(
        initial.memory_id, f"CONFIRM {initial.memory_id}", input_source="scripted_fixture"
    )
    after_initial = session.retrieve()
    correction = session.stage_correction()
    before_correction_confirmation = session.retrieve()
    session.confirm_command(
        correction.memory_id,
        f"CONFIRM {correction.memory_id}",
        input_source="scripted_fixture",
    )
    after_correction = session.retrieve()
    states_before_delete = {record.memory_id: record.lifecycle_status for record in session.records()}
    deleted = session.delete_all_command(f"DELETE ALL {session.run_id}")
    after_delete = session.retrieve()
    report = {
        "kind": "synthetic_software_smoke",
        "clinical_or_benchmark_result": False,
        "live_model_or_network_used": False,
        "consent_mode": "simulated_scripted_fixture_not_real_user_consent",
        "run_id": session.run_id,
        "initial_state": initial.lifecycle_status,
        "before_confirmation_memory_ids": list(before.memory_ids),
        "after_confirmation_memory_ids": list(after_initial.memory_ids),
        "before_correction_confirmation_memory_ids": list(before_correction_confirmation.memory_ids),
        "after_correction_memory_ids": list(after_correction.memory_ids),
        "states_before_delete": states_before_delete,
        "deleted_ids": deleted,
        "records_after_delete": len(session.records()),
        "after_delete_memory_ids": list(after_delete.memory_ids),
        "audit_actions": [event["action"] for event in store.audit_for_user(session.user_id)],
    }
    checks = [
        initial.lifecycle_status == "candidate",
        before.memory_ids == (),
        after_initial.memory_ids == (session.initial_id,),
        before_correction_confirmation.memory_ids == (session.initial_id,),
        after_correction.memory_ids == (session.correction_id,),
        states_before_delete == {session.initial_id: "superseded", session.correction_id: "active"},
        len(deleted) == 2,
        report["records_after_delete"] == 0,
        after_delete.memory_ids == (),
    ]
    report["smoke_passed"] = all(checks)
    if not report["smoke_passed"]:
        raise AssertionError(f"Demo software smoke failed: {report}")
    return report


def _show_candidate(record: MemoryRecord, output: Callable[[str], None]) -> None:
    output(f"ID: {record.memory_id}\nState: {record.lifecycle_status}\nCandidate: {record.content}")
    output(f"Exact fictional source quote: {record.source_evidence.supporting_text}")


def interactive_walkthrough(
    store: SQLiteMemoryStore,
    *,
    read: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
) -> None:
    session = DemoSession(store)
    output(DISCLAIMER)
    output("Confirmed input only authorizes this synthetic demo. No real health data is requested.")
    while True:
        output(
            "\n1 解析固定样例 → 候选 / Parse fixture\n"
            "2 查看候选并明确确认 / Review and confirm\n"
            "3 检索与上下文 / Retrieve context\n"
            "4 新建更正候选 / Stage correction\n"
            "5 硬删除本次演示所有版本 / Delete all demo versions\n"
            "6 状态与审计 / Status and audit\n0 退出 / Exit"
        )
        try:
            choice = read("Choose: ").strip()
            if choice == "0":
                return
            if choice == "1":
                _show_candidate(session.stage_initial(), output)
            elif choice == "2":
                candidates = [item for item in session.records() if item.lifecycle_status == "candidate"]
                if not candidates:
                    raise ValueError("No candidate. Choose 1 or 4 first.")
                for candidate in candidates:
                    _show_candidate(candidate, output)
                memory_id = read("Candidate ID: ").strip()
                if memory_id not in {item.memory_id for item in candidates}:
                    raise ValueError("Select a displayed candidate ID.")
                command = read(f"Type CONFIRM {memory_id} to confirm, anything else cancels: ")
                confirmed = session.confirm_command(memory_id, command)
                output(f"Confirmed: {confirmed.memory_id}; state={confirmed.lifecycle_status}")
            elif choice == "3":
                context = session.retrieve()
                output(f"Fixture query: {DEFAULT_QUERY}")
                output(f"Actually included memory IDs: {list(context.memory_ids)}")
                output(context.prompt_context)
                output("No answer was generated; this is the context handed to a future Agent.")
            elif choice == "4":
                _show_candidate(session.stage_correction(), output)
                output("The old active version stays current until you explicitly confirm this candidate.")
            elif choice == "5":
                output("Deletes all memory content from this run, including superseded versions. Content-free audit remains.")
                command = read(f"Type DELETE ALL {session.run_id}, anything else cancels: ")
                output(f"Hard-deleted IDs: {session.delete_all_command(command)}")
            elif choice == "6":
                output(json.dumps(
                    {
                        "records": [
                            {"id": item.memory_id, "state": item.lifecycle_status, "content": item.content}
                            for item in session.records()
                        ],
                        "audit": store.audit_for_user(session.user_id),
                    }, ensure_ascii=False, indent=2,
                ))
            else:
                output("Choose a menu number; do not enter real personal information.")
        except (KeyError, ValueError) as exc:
            output(f"Not changed: {exc}")
        except (EOFError, KeyboardInterrupt):
            output("\nDemo closed.")
            return


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic local memory lifecycle demo; no API required.")
    parser.add_argument("--scripted", action="store_true", help="Run a SOFTWARE SMOKE with simulated confirmation events.")
    parser.add_argument("--database", type=Path, help="Explicit optional SQLite path; default is ephemeral :memory:.")
    args = parser.parse_args(argv)
    if args.database:
        print(
            "PRIVACY WARNING: this optional database is plaintext, has no authentication, and keeps content-free audit metadata.\n"
            "Use fictional data only. Deleting rows is not guaranteed forensic erasure of SQLite files/backups.\n"
            f"Explicit persistent path: {args.database.resolve()}",
            file=sys.stderr,
        )
    store = SQLiteMemoryStore(args.database if args.database else ":memory:")
    try:
        if args.scripted:
            print(json.dumps(scripted_walkthrough(store), ensure_ascii=False, indent=2))
        else:
            interactive_walkthrough(store)
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
