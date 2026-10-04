from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for relative in (
    "modules/long-term-memory/schema",
    "modules/long-term-memory/storage",
    "modules/long-term-memory/lifecycle",
    "modules/long-term-memory/extraction",
    "evaluation/memory-retrieval",
):
    sys.path.insert(0, str(ROOT / relative))

from benchmark_adapters import (  # noqa: E402
    evidence_retrieval_metrics,
    load_locomo,
    load_longmemeval,
)
from extraction_contract import ConversationTurn, parse_extraction_output  # noqa: E402
from memory_manager import MemoryManager  # noqa: E402
from memory_models import MemoryRecord, SourceEvidence  # noqa: E402
from sqlite_store import SQLiteMemoryStore  # noqa: E402


def make_record(
    memory_id: str,
    *,
    user_id: str = "user-a",
    memory_type: str = "explicit_fact",
    content: str = "The seminar is Friday.",
    origin: str = "explicit_user_statement",
    confirmation_state: str = "user_confirmed",
    lifecycle_status: str = "candidate",
    sensitivity: str = "standard",
    payload: dict | None = None,
    valid_to: str | None = None,
    supersedes: str | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        schema_version="1.0",
        memory_id=memory_id,
        user_id=user_id,
        memory_type=memory_type,
        content=content,
        origin=origin,
        confirmation_state=confirmation_state,
        lifecycle_status=lifecycle_status,
        sensitivity=sensitivity,
        confidence=0.9,
        importance=0.7,
        created_at="2026-10-01T10:00:00Z",
        observed_at="2026-10-01T09:59:00Z",
        updated_at="2026-10-01T10:00:00Z",
        source_evidence=SourceEvidence(
            session_id="session-1",
            turn_ids=["turn-1"],
            supporting_text=content,
        ),
        structured_payload=payload or {},
        valid_from="2026-10-01T09:59:00Z",
        valid_to=valid_to,
        supersedes_memory_id=supersedes,
    )


class MemoryManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = SQLiteMemoryStore()
        self.manager = MemoryManager(self.store)

    def tearDown(self) -> None:
        self.store.close()

    def test_explicit_statement_becomes_active(self) -> None:
        result = self.manager.ingest(make_record("m1"))
        self.assertEqual(result.action, "created")
        self.assertEqual(self.store.get("user-a", "m1").lifecycle_status, "active")

    def test_unconfirmed_inference_stays_candidate(self) -> None:
        record = make_record(
            "m2",
            memory_type="cognitive_pattern",
            content="Possible all-or-nothing thinking.",
            origin="agent_inference",
            confirmation_state="unconfirmed",
            payload={"evidence_count": 2, "slot_key": "pattern:performance"},
        )
        self.manager.ingest(record)
        self.assertEqual(self.store.get("user-a", "m2").lifecycle_status, "candidate")
        self.assertEqual(self.manager.eligible_context("user-a"), [])
        confirmed = self.manager.confirm_candidate(
            "user-a",
            "m2",
            consent_event_id="consent-1",
            confirmation_turn_ids=["turn-confirm-1"],
            confirmed_at="2026-10-02T10:00:00Z",
        )
        self.assertEqual(confirmed.confirmation_state, "user_confirmed")
        self.assertEqual([item.memory_id for item in self.manager.eligible_context("user-a")], ["m2"])
        event = self.store.audit_for_user("user-a")[-1]
        self.assertEqual(event["details"]["consent_event_id"], "consent-1")
        self.assertEqual(event["details"]["confirmation_turn_ids"], ["turn-confirm-1"])

    def test_duplicate_is_rejected_without_second_record(self) -> None:
        self.manager.ingest(make_record("m1"))
        result = self.manager.ingest(make_record("m2", content="the seminar is friday"))
        self.assertEqual(result.action, "duplicate_rejected")
        self.assertIsNone(self.store.get("user-a", "m2"))

    def test_conflict_requires_confirmation(self) -> None:
        first = make_record("m1", payload={"slot_key": "event:seminar-date"})
        second = make_record(
            "m2", content="The seminar is Thursday.", payload={"slot_key": "event:seminar-date"}
        )
        self.manager.ingest(first)
        result = self.manager.ingest(second)
        self.assertEqual(result.action, "conflict_requires_confirmation")
        self.assertEqual(self.store.get("user-a", "m2").lifecycle_status, "candidate")
        self.manager.confirm_candidate(
            "user-a",
            "m2",
            consent_event_id="consent-2",
            confirmed_at="2026-10-02T10:00:00Z",
        )
        self.assertEqual(self.store.get("user-a", "m1").lifecycle_status, "superseded")
        self.assertEqual(self.store.get("user-a", "m2").lifecycle_status, "active")

    def test_correction_suppresses_old_memory(self) -> None:
        self.manager.ingest(make_record("old", payload={"slot_key": "event:seminar-date"}))
        correction = make_record(
            "correction",
            memory_type="user_correction",
            content="The seminar is Thursday, not Friday.",
            origin="user_correction",
            confirmation_state="user_corrected",
            payload={
                "corrected_content": "The seminar is Thursday.",
                "slot_key": "event:seminar-date",
            },
            supersedes="old",
        )
        self.manager.ingest(correction)
        self.assertEqual(self.store.get("user-a", "old").lifecycle_status, "superseded")
        ids = [item.memory_id for item in self.manager.eligible_context("user-a")]
        self.assertEqual(ids, ["correction"])

    def test_correction_cannot_target_another_user(self) -> None:
        self.manager.ingest(make_record("old", user_id="user-b"))
        correction = make_record(
            "correction",
            user_id="user-a",
            memory_type="user_correction",
            content="Corrected information.",
            origin="user_correction",
            confirmation_state="user_corrected",
            payload={"corrected_content": "Corrected information."},
            supersedes="old",
        )
        with self.assertRaisesRegex(ValueError, "does not exist"):
            self.manager.ingest(correction)

    def test_user_isolation(self) -> None:
        self.manager.ingest(make_record("m1", user_id="user-a"))
        self.assertEqual(self.manager.eligible_context("user-b"), [])
        with self.assertRaises(KeyError):
            self.manager.delete("user-b", "m1")

    def test_hard_delete_removes_content_but_keeps_content_free_audit(self) -> None:
        self.manager.ingest(make_record("m1", content="Private support detail."))
        self.assertTrue(self.manager.delete("user-a", "m1", deleted_at="2026-10-02T10:00:00Z"))
        self.assertIsNone(self.store.get("user-a", "m1"))
        audit = self.store.audit_for_user("user-a")[-1]
        self.assertEqual(audit["action"], "hard_deleted")
        self.assertNotIn("Private support detail", json.dumps(audit))

    def test_expiry_prevents_retrieval(self) -> None:
        self.manager.ingest(make_record("m1", valid_to="2026-10-02T00:00:00Z"))
        expired = self.manager.expire_due(as_of="2026-10-03T00:00:00Z")
        self.assertEqual(expired, ["m1"])
        self.assertEqual(self.manager.eligible_context("user-a"), [])

    def test_restricted_memory_requires_explicit_permission(self) -> None:
        record = make_record(
            "risk",
            memory_type="safety_signal",
            content="The user reported feeling unsafe last night.",
            sensitivity="restricted",
        )
        self.manager.ingest(record)
        self.assertEqual(self.manager.eligible_context("user-a"), [])
        self.assertEqual(
            [item.memory_id for item in self.manager.eligible_context("user-a", allow_restricted=True)],
            ["risk"],
        )


class ExtractionContractTests(unittest.TestCase):
    def test_exact_user_evidence_is_accepted(self) -> None:
        turns = [ConversationTurn("u1", "user", "I want to walk after lunch three times.")]
        raw = json.dumps(
            {
                "candidates": [
                    {
                        "memory_type": "goal",
                        "content": "Walk after lunch three times.",
                        "source_turn_ids": ["u1"],
                        "supporting_text": "walk after lunch three times",
                        "structured_payload": {
                            "description": "Walk after lunch three times.",
                            "status": "active",
                        },
                    }
                ]
            }
        )
        records = parse_extraction_output(
            raw,
            user_id="user-a",
            session_id="s1",
            turns=turns,
            extracted_at="2026-10-01T10:00:00Z",
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source_evidence.turn_ids, ["u1"])
        self.assertEqual(records[0].confirmation_state, "unconfirmed")

    def test_assistant_turn_cannot_support_user_memory(self) -> None:
        turns = [ConversationTurn("a1", "assistant", "Try walking after lunch.")]
        raw = json.dumps(
            {
                "candidates": [
                    {
                        "memory_type": "goal",
                        "content": "Walk after lunch.",
                        "source_turn_ids": ["a1"],
                        "supporting_text": "walking after lunch",
                        "structured_payload": {"description": "Walk.", "status": "active"},
                    }
                ]
            }
        )
        with self.assertRaisesRegex(ValueError, "user turn"):
            parse_extraction_output(
                raw, user_id="user-a", session_id="s1", turns=turns
            )

    def test_unentailed_model_summary_remains_unconfirmed_candidate(self) -> None:
        turns = [ConversationTurn("u1", "user", "I like walking after lunch.")]
        raw = json.dumps(
            {
                "candidates": [
                    {
                        "memory_id": "hallucinated",
                        "memory_type": "explicit_fact",
                        "content": "The user lives in Paris.",
                        "source_turn_ids": ["u1"],
                        "supporting_text": "I like walking",
                    }
                ]
            }
        )
        extracted = parse_extraction_output(
            raw, user_id="user-a", session_id="s1", turns=turns
        )[0]
        self.assertEqual(extracted.confirmation_state, "unconfirmed")
        store = SQLiteMemoryStore()
        self.addCleanup(store.close)
        result = MemoryManager(store).ingest(extracted)
        self.assertIn("candidate", result.reason)
        self.assertEqual(store.get("user-a", "hallucinated").lifecycle_status, "candidate")


class AtomicLifecycleTests(unittest.TestCase):
    def test_correction_cannot_target_superseded_version(self) -> None:
        store = SQLiteMemoryStore()
        self.addCleanup(store.close)
        manager = MemoryManager(store)
        manager.ingest(make_record("old", payload={"slot_key": "event:seminar-date"}))
        first = make_record(
            "corr-a",
            memory_type="user_correction",
            content="The seminar is Thursday.",
            origin="user_correction",
            confirmation_state="user_corrected",
            payload={"corrected_content": "The seminar is Thursday."},
            supersedes="old",
        )
        manager.ingest(first)
        stale = make_record(
            "corr-b",
            memory_type="user_correction",
            content="The seminar is Wednesday.",
            origin="user_correction",
            confirmation_state="user_corrected",
            payload={"corrected_content": "The seminar is Wednesday."},
            supersedes="old",
        )
        with self.assertRaisesRegex(ValueError, "active current version"):
            manager.ingest(stale)
        self.assertEqual(store.get("user-a", "corr-a").lifecycle_status, "active")
        self.assertIsNone(store.get("user-a", "corr-b"))

    def test_concurrent_slot_writers_leave_only_one_active_version(self) -> None:
        handle = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        handle.close()
        database = Path(handle.name)
        self.addCleanup(database.unlink)
        initial = SQLiteMemoryStore(database)
        initial.close()
        barrier = threading.Barrier(2)

        def write(memory_id: str, content: str) -> str:
            store = SQLiteMemoryStore(database)
            try:
                barrier.wait(timeout=5)
                result = MemoryManager(store).ingest(
                    make_record(
                        memory_id,
                        content=content,
                        payload={"slot_key": "event:seminar-date"},
                    )
                )
                return result.action
            finally:
                store.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(write, "concurrent-a", "The seminar is Friday."),
                pool.submit(write, "concurrent-b", "The seminar is Thursday."),
            ]
            actions = sorted(future.result() for future in futures)
        check = SQLiteMemoryStore(database)
        self.addCleanup(check.close)
        records = check.list_for_user("user-a")
        active = [item for item in records if item.lifecycle_status == "active"]
        candidates = [item for item in records if item.lifecycle_status == "candidate"]
        self.assertEqual(actions, ["conflict_requires_confirmation", "created"])
        self.assertEqual(len(active), 1)
        self.assertEqual(len(candidates), 1)

    def test_expiry_does_not_resurrect_concurrently_deleted_memory(self) -> None:
        handle = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        handle.close()
        database = Path(handle.name)
        self.addCleanup(database.unlink)
        initial = SQLiteMemoryStore(database)
        MemoryManager(initial).ingest(
            make_record("expiring", valid_to="2026-10-02T00:00:00Z")
        )
        initial.close()
        barrier = threading.Barrier(2)

        def expire() -> list[str]:
            store = SQLiteMemoryStore(database)
            try:
                barrier.wait(timeout=5)
                return MemoryManager(store).expire_due(as_of="2026-10-03T00:00:00Z")
            finally:
                store.close()

        def delete() -> bool:
            store = SQLiteMemoryStore(database)
            try:
                barrier.wait(timeout=5)
                try:
                    return MemoryManager(store).delete(
                        "user-a", "expiring", deleted_at="2026-10-03T00:00:01Z"
                    )
                except KeyError:
                    return False
            finally:
                store.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            expire_future = pool.submit(expire)
            delete_future = pool.submit(delete)
            expire_future.result()
            delete_future.result()

        check = SQLiteMemoryStore(database)
        self.addCleanup(check.close)
        self.assertIsNone(check.get("user-a", "expiring"))
        actions = [event["action"] for event in check.audit_for_user("user-a")]
        self.assertEqual(actions.count("hard_deleted"), 1)

    def test_concurrent_delete_writes_one_hard_delete_audit(self) -> None:
        handle = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
        handle.close()
        database = Path(handle.name)
        self.addCleanup(database.unlink)
        initial = SQLiteMemoryStore(database)
        MemoryManager(initial).ingest(make_record("delete-once"))
        initial.close()
        barrier = threading.Barrier(2)

        def delete() -> bool:
            store = SQLiteMemoryStore(database)
            try:
                barrier.wait(timeout=5)
                try:
                    return MemoryManager(store).delete("user-a", "delete-once")
                except KeyError:
                    return False
            finally:
                store.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = [future.result() for future in [pool.submit(delete), pool.submit(delete)]]

        self.assertEqual(sorted(outcomes), [False, True])
        check = SQLiteMemoryStore(database)
        self.addCleanup(check.close)
        actions = [event["action"] for event in check.audit_for_user("user-a")]
        self.assertEqual(actions.count("hard_deleted"), 1)

    def test_unconfirmed_correction_waits_then_applies_atomically(self) -> None:
        store = SQLiteMemoryStore()
        self.addCleanup(store.close)
        manager = MemoryManager(store)
        manager.ingest(make_record("old", payload={"slot_key": "event:seminar-date"}))
        correction = make_record(
            "correction",
            memory_type="user_correction",
            content="The seminar is Thursday, not Friday.",
            origin="user_correction",
            confirmation_state="unconfirmed",
            payload={"corrected_content": "The seminar is Thursday."},
            supersedes="old",
        )
        result = manager.ingest(correction)
        self.assertEqual(result.action, "correction_requires_confirmation")
        self.assertEqual(store.get("user-a", "old").lifecycle_status, "active")
        self.assertEqual(store.get("user-a", "correction").lifecycle_status, "candidate")
        manager.confirm_candidate(
            "user-a",
            "correction",
            consent_event_id="consent-correction",
            confirmation_turn_ids=["turn-confirm-correction"],
            confirmed_at="2026-10-02T10:00:00Z",
        )
        self.assertEqual(store.get("user-a", "old").lifecycle_status, "superseded")
        self.assertEqual(store.get("user-a", "correction").lifecycle_status, "active")

    def test_correction_rolls_back_when_replacement_insert_fails(self) -> None:
        store = SQLiteMemoryStore()
        self.addCleanup(store.close)
        manager = MemoryManager(store)
        manager.ingest(make_record("old", payload={"slot_key": "event:seminar-date"}))
        store.connection.execute(
            """
            CREATE TRIGGER fail_correction_insert
            BEFORE INSERT ON memories
            WHEN NEW.memory_id = 'correction'
            BEGIN
                SELECT RAISE(ABORT, 'simulated replacement failure');
            END;
            """
        )
        correction = make_record(
            "correction",
            memory_type="user_correction",
            content="The seminar is Thursday, not Friday.",
            origin="user_correction",
            confirmation_state="user_corrected",
            payload={"corrected_content": "The seminar is Thursday."},
            supersedes="old",
        )
        with self.assertRaisesRegex(Exception, "simulated replacement failure"):
            manager.ingest(correction)
        self.assertEqual(store.get("user-a", "old").lifecycle_status, "active")
        self.assertIsNone(store.get("user-a", "correction"))
        self.assertNotIn(
            "correction_applied",
            [event["action"] for event in store.audit_for_user("user-a")],
        )


class BenchmarkAdapterTests(unittest.TestCase):
    def _write_json(self, data: object) -> Path:
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(data, handle)
        handle.close()
        self.addCleanup(Path(handle.name).unlink)
        return Path(handle.name)

    def test_longmemeval_official_format_and_gold_evidence_metric(self) -> None:
        path = self._write_json(
            [
                {
                    "question_id": "q1",
                    "question_type": "knowledge-update",
                    "question": "When is the event now?",
                    "answer": "Thursday",
                    "question_date": "2026/10/03",
                    "haystack_session_ids": ["s1", "s2"],
                    "haystack_dates": ["2026/10/01", "2026/10/02"],
                    "haystack_sessions": [[], []],
                    "answer_session_ids": ["s2"],
                }
            ]
        )
        examples = load_longmemeval(path)
        metrics = evidence_retrieval_metrics(examples, {"q1": ["s2", "s1"]}, k=2)
        self.assertEqual(metrics["evidence_recall_at_2"], 1.0)
        self.assertEqual(metrics["mrr_at_2"], 1.0)

    def test_locomo_official_format_is_parsed(self) -> None:
        path = self._write_json(
            [
                {
                    "sample_id": "sample-1",
                    "conversation": {
                        "speaker_a": "A",
                        "speaker_b": "B",
                        "session_1": [{"speaker": "A", "dia_id": "d1", "text": "Hello"}],
                        "session_1_date_time": "1 January 2026",
                    },
                    "qa": [
                        {
                            "question": "Who spoke?",
                            "answer": "A",
                            "category": 1,
                            "evidence": ["d1"],
                        }
                    ],
                }
            ]
        )
        examples = load_locomo(path)
        self.assertEqual(examples[0].evidence_ids, ("d1",))
        self.assertEqual(len(examples[0].sessions), 1)

    def test_locomo_adversarial_answer_format_is_parsed(self) -> None:
        path = self._write_json(
            [
                {
                    "sample_id": "sample-adv",
                    "conversation": {
                        "session_1": [{"speaker": "A", "dia_id": "d1", "text": "Hello"}],
                        "session_1_date_time": "1 January 2026",
                    },
                    "qa": [
                        {
                            "question": "Unsupported premise?",
                            "adversarial_answer": "The premise is not supported.",
                            "category": 5,
                            "evidence": [],
                        }
                    ],
                }
            ]
        )
        examples = load_locomo(path)
        self.assertEqual(examples[0].answer, "The premise is not supported.")


if __name__ == "__main__":
    unittest.main()
