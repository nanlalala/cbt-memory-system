"""Tiny deterministic tests for external-source candidate bulk ingestion."""

import copy
import json
import sqlite3
import sys
import unittest
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for relative in ["modules/long-term-memory/schema", "modules/long-term-memory/storage",
                 "modules/long-term-memory/lifecycle"]:
    sys.path.insert(0, str(ROOT / relative))

from memory_manager import MemoryManager
from memory_models import MemoryRecord
from sqlite_store import SQLiteMemoryStore


OBSERVATION_METADATA = {"source_role", "provenance_kind", "source_session_id",
                        "source_turn_ids", "observation_only", "user_consent"}


def fixture(memory_id="observed-1", *, turn_id="turn-1", session="external-session",
            text="A quoted external observation.", source_role="context",
            provenance="synthetic_benchmark_document", sensitivity="standard"):
    timestamp = "2026-11-04T00:00:00Z"
    record = MemoryRecord.from_dict({
        "schema_version": "1.0", "memory_id": memory_id, "user_id": "scope-1",
        "memory_type": "explicit_fact", "content": text, "origin": "system_observation",
        "confirmation_state": "unconfirmed", "lifecycle_status": "candidate",
        "sensitivity": sensitivity, "confidence": 0.5, "importance": 0.5,
        "created_at": timestamp, "observed_at": timestamp, "updated_at": timestamp,
        "valid_from": timestamp,
        "source_evidence": {"session_id": session, "turn_ids": [turn_id],
                            "supporting_text": text},
    })
    source = {(session, turn_id): {"text": text, "observed_at": timestamp,
                                  "source_role": source_role, "provenance_kind": provenance}}
    return record, source


class ObservationBulkTests(unittest.TestCase):
    def setUp(self):
        self.store = SQLiteMemoryStore()
        self.manager = MemoryManager(self.store)

    def tearDown(self):
        self.store.close()

    def ingest(self, records, sources):
        return self.manager.ingest_verbatim_observation_candidates(
            records, user_id="scope-1", source_turns=sources, actor="fixture-import")

    def test_results_and_candidate_rows_match_serial_ingest(self):
        first, sources1 = fixture(text="The adjustment is -5 degrees.")
        second, sources2 = fixture("observed-2", turn_id="turn-2", text="The adjustment is +5 degrees.")
        repeated, sources3 = fixture("observed-3", turn_id="turn-3", text=first.content)
        replay = replace(first, memory_id="same-source-replay")
        records = [first, second, repeated, replay]
        serial = SQLiteMemoryStore()
        try:
            expected = [MemoryManager(serial).ingest(record, actor="fixture-import") for record in records]
            actual = self.ingest(records, {**sources1, **sources2, **sources3})
            self.assertEqual(actual, expected)
            self.assertEqual([item.action for item in actual],
                             ["created", "created", "created", "duplicate_rejected"])
            self.assertEqual([r.to_dict() for r in self.store.list_for_user("scope-1")],
                             [r.to_dict() for r in serial.list_for_user("scope-1")])
            actual_audits = self.store.audit_for_user("scope-1")
            for event in actual_audits:
                event["details"] = {key: value for key, value in event["details"].items()
                                    if key not in OBSERVATION_METADATA}
            self.assertEqual(actual_audits, serial.audit_for_user("scope-1"))
            self.assertEqual(self.manager.eligible_context("scope-1"), [])
        finally:
            serial.close()

    def test_all_source_roles_keep_provenance_and_never_claim_consent(self):
        records, sources = [], {}
        for index, role in enumerate(["user", "assistant", "system", "context"]):
            record, source = fixture(f"role-{role}", turn_id=f"turn-{index}",
                                     text="Same quoted content with separate provenance.",
                                     source_role=role, provenance=f"external_{role}",
                                     sensitivity="restricted" if role == "system" else "standard")
            records.append(record)
            sources.update(source)
        self.ingest(records, sources)
        self.assertEqual(len(self.store.list_for_user("scope-1")), 4)
        for record, audit in zip(records, self.store.audit_for_user("scope-1")):
            source = sources[(record.source_evidence.session_id, record.source_evidence.turn_ids[0])]
            self.assertEqual(audit["details"]["source_role"], source["source_role"])
            self.assertEqual(audit["details"]["provenance_kind"], source["provenance_kind"])
            self.assertTrue(audit["details"]["observation_only"])
            self.assertFalse(audit["details"]["user_consent"])
            self.assertEqual(audit["details"]["source_turn_ids"], record.source_evidence.turn_ids)
            stored = self.store.get("scope-1", record.memory_id)
            self.assertEqual(stored.origin, "system_observation")
            self.assertEqual(stored.confirmation_state, "unconfirmed")
            self.assertEqual(stored.lifecycle_status, "candidate")
            self.assertEqual(stored.sensitivity, record.sensitivity)
        self.assertNotIn(records[0].content, json.dumps(self.store.audit_for_user("scope-1")))

    def test_replayed_source_audit_retains_observation_metadata(self):
        record, sources = fixture(source_role="assistant", provenance="synthetic_assistant_quote")
        self.ingest([record], sources)
        result = self.ingest([replace(record, memory_id="replay")], sources)[0]
        self.assertEqual(result.action, "duplicate_rejected")
        audit = self.store.audit_for_user("scope-1")[-1]
        self.assertEqual(audit["details"]["source_role"], "assistant")
        self.assertEqual(audit["details"]["provenance_kind"], "synthetic_assistant_quote")
        self.assertFalse(audit["details"]["user_consent"])

    def test_summary_and_unverified_source_are_rejected_before_any_write(self):
        record, sources = fixture()
        invalid_records = [replace(record, content="An inferred summary of the quote."),
                           replace(record, updated_at="2026-11-05T00:00:00Z")]
        for invalid in invalid_records:
            with self.subTest(record=invalid.content), self.assertRaises(ValueError):
                self.ingest([invalid], sources)
        with self.assertRaises(ValueError):
            self.ingest([record], {})
        key = next(iter(sources))
        for field in ["text", "observed_at", "source_role", "provenance_kind"]:
            altered = copy.deepcopy(sources)
            altered[key].pop(field)
            with self.subTest(missing=field), self.assertRaises(ValueError):
                self.ingest([record], altered)
        self.assertEqual(self.store.list_for_user("scope-1"), [])
        self.assertEqual(self.store.audit_for_user("scope-1"), [])

    def test_source_role_and_provenance_require_valid_metadata(self):
        record, sources = fixture()
        key = next(iter(sources))
        for role in ["developer", "tool", "", None, ["user"]]:
            altered = copy.deepcopy(sources)
            altered[key]["source_role"] = role
            with self.subTest(role=role), self.assertRaises(ValueError):
                self.ingest([record], altered)
        for provenance in ["", "   ", None, {"pretend": "verified"}]:
            altered = copy.deepcopy(sources)
            altered[key]["provenance_kind"] = provenance
            with self.subTest(provenance=provenance), self.assertRaises(ValueError):
                self.ingest([record], altered)

    def test_only_unslotted_unconfirmed_observations_are_accepted(self):
        record, sources = fixture()
        for changes in [
            {"origin": "agent_inference"}, {"origin": "explicit_user_statement"},
            {"confirmation_state": "user_confirmed"}, {"lifecycle_status": "active"},
            {"user_id": "other-scope"}, {"structured_payload": {"slot_key": "current"}},
            {"tags": ["summary"]}, {"related_memory_ids": ["another"]},
            {"supersedes_memory_id": "previous"}, {"valid_to": "2027-01-01T00:00:00Z"},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.ingest([replace(record, **changes)], sources)
        self.assertEqual(self.store.list_for_user("scope-1"), [])

    def test_second_record_failure_rolls_back_candidates_and_all_audits(self):
        previous, prior_sources = fixture("previous", turn_id="previous-turn")
        self.ingest([previous], prior_sources)
        before_records = [r.to_dict() for r in self.store.list_for_user("scope-1")]
        before_audits = self.store.audit_for_user("scope-1")
        first, source1 = fixture("new-first", turn_id="new-turn-1")
        second, source2 = fixture("reject-second", turn_id="new-turn-2")
        self.store.connection.execute("""CREATE TRIGGER reject_fixture BEFORE INSERT ON memories
            WHEN NEW.memory_id = 'reject-second'
            BEGIN SELECT RAISE(ABORT, 'injected fixture failure'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            self.ingest([replace(previous, memory_id="replay-prior"), first, second],
                        {**prior_sources, **source1, **source2})
        self.assertEqual([r.to_dict() for r in self.store.list_for_user("scope-1")], before_records)
        self.assertEqual(self.store.audit_for_user("scope-1"), before_audits)
        self.assertFalse(self.store.connection.in_transaction)

    def test_existing_explicit_user_bulk_api_keeps_its_origin_boundary(self):
        observation, sources = fixture()
        with self.assertRaises(ValueError):
            self.manager.ingest_verbatim_candidates([observation], user_id="scope-1", source_turns=sources)
        user_quote = replace(observation, origin="explicit_user_statement")
        with self.assertRaises(ValueError):
            self.ingest([user_quote], sources)
        old_sources = {key: {"text": source["text"], "observed_at": source["observed_at"]}
                       for key, source in sources.items()}
        result = self.manager.ingest_verbatim_candidates([user_quote], user_id="scope-1", source_turns=old_sources)
        self.assertEqual(result[0].action, "created")
        self.assertNotIn("source_role", self.store.audit_for_user("scope-1")[0]["details"])


if __name__ == "__main__":
    unittest.main()
