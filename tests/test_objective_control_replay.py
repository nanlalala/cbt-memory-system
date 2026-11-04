"""Acceptance-oracle tests; fixtures do not represent real or clinical users."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation/objective-memory"))

import replay_memory_controls as replay  # noqa: E402
from memory_retriever import RetrievedMemory  # noqa: E402


class ObjectiveControlReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = replay.run_replay()

    def row(self, case_id, check_id, arm="managed_memory_rag"):
        return next(row for row in self.report["queries"]
                    if row["case_id"] == case_id and row["check_id"] == check_id and row["arm"] == arm)

    def altered_spec(self, *, alter_fixture=None, alter_ledger=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        fixture, ledger = replay.load_spec()
        if alter_fixture:
            alter_fixture(fixture)
        if alter_ledger:
            alter_ledger(ledger)
        events, expected = Path(temporary.name) / "events.json", Path(temporary.name) / "expected.json"
        events.write_text(json.dumps(fixture), encoding="utf-8")
        expected.write_text(json.dumps(ledger), encoding="utf-8")
        return events, expected

    def test_real_managed_pipeline_and_competent_raw_baseline_both_pass(self):
        self.assertTrue(self.report["acceptance_passed"])
        for arm in replay.ARMS:
            result = self.report["conditions"][arm]
            self.assertEqual((result["queries"], result["passed_queries"]), (26, 26))
            self.assertEqual(result["operation_checks"], 56)
            self.assertEqual(result["failed_operations"], 0)
            self.assertEqual(result["prohibited_exposure_queries"], 0)
            self.assertEqual(set(result["by_task"]), set(replay.TASKS))
        self.assertFalse(self.report["clinical_or_user_benefit_result"])
        self.assertFalse(self.report["natural_language_extraction_evaluated"])
        self.assertEqual(self.report["model_or_network_calls"], 0)
        self.assertEqual(self.report["real_participants"], 0)
        self.assertIsNone(self.report["independent_user_sample_size"])

    def test_confirmation_and_current_goal_are_distinct_assertions(self):
        self.assertEqual(self.row("goals", "unconfirmed-hidden")["included_ids"], [])
        self.assertEqual(self.row("goals", "pending-change")["included_ids"], ["goal-old"])
        self.assertEqual(self.row("goals", "current-goal")["included_ids"], ["goal-new"])
        self.assertEqual(self.row("goals", "current-goal")["source_turn_ids"],
                         {"goal-new": ["turn-goal-new"]})

    def test_no_outcome_is_invented_from_absence_of_a_report(self):
        before = self.row("homework", "no-outcome-invented")
        self.assertEqual(before["storage_statuses"], {"walk-assignment": "active"})
        self.assertEqual(before["included_ids"], ["walk-assignment"])
        after = self.row("homework", "explicit-partial")
        self.assertEqual(set(after["included_ids"]), {"walk-assignment", "walk-partial"})

    def test_exact_expiry_boundary_is_enforced_before_cleanup(self):
        before = self.row("expiry", "before-expiry")
        boundary = self.row("expiry", "exact-expiry-boundary")
        self.assertEqual(before["included_ids"], ["temporary-goal"])
        self.assertEqual(boundary["storage_statuses"], {"temporary-goal": "active"})
        self.assertEqual(boundary["eligible_ids"], [])
        self.assertEqual(boundary["included_ids"], [])
        self.assertEqual(self.row("expiry", "deleted-not-resurrected")["storage_statuses"], {})

    def test_current_correction_and_expected_control_rejections(self):
        self.assertEqual(self.row("corrections", "pending-correction")["included_ids"],
                         ["appointment-old"])
        self.assertEqual(self.row("corrections", "stale-target-rejected")["included_ids"],
                         ["appointment-new"])
        failures = [row for row in self.report["operations"] if row["expected_error_type"]]
        self.assertEqual(len(failures), 4)
        self.assertTrue(all(row["passed"] for row in failures))
        self.assertEqual({row["actual_error_type"] for row in failures}, {"ValueError", "KeyError"})

    def test_trace_matches_actual_whole_items_including_budget_drops(self):
        budget = self.row("goals", "budgeted-goal")
        self.assertEqual(budget["retrieved_ids"], ["goal-new"])
        self.assertEqual(budget["included_ids"], [])
        self.assertEqual(budget["dropped_ids"], ["goal-new"])
        self.assertLessEqual(budget["prompt_chars"], 512)
        self.assertTrue(all(row["trace_matches_prompt"] for row in self.report["queries"]))

    def test_user_and_sensitive_scope_are_checked_on_real_prompt(self):
        self.assertEqual(self.row("deletion_isolation", "default-sensitive-filter")["included_ids"],
                         ["alice-standard"])
        self.assertEqual(self.row("deletion_isolation", "other-user-own-record")["included_ids"],
                         ["bob-standard"])
        self.assertEqual(set(self.row("deletion_isolation", "explicit-sensitive-access")["included_ids"]),
                         {"alice-standard", "alice-restricted"})
        self.assertEqual(self.row("deletion_isolation", "delete-remains-final")["included_ids"], [])

    def test_oracle_rejects_changed_homework_status_without_a_new_user_report(self):
        def change(ledger):
            ledger["checks"]["homework:explicit-partial"]["fields"]["walk-partial"]["completion_status"] = "completed"
        events, ledger = self.altered_spec(alter_ledger=change)
        result = replay.run_replay(events, ledger)
        rows = [row for row in result["queries"] if row["check_id"] == "explicit-partial"]
        self.assertFalse(result["acceptance_passed"])
        self.assertEqual([row["first_failure_stage"] for row in rows], ["storage", "storage"])

    def test_oracle_detects_provenance_corruption(self):
        class CorruptProvenance(replay.RawHistoryControlStore):
            def snapshot(self):
                records = super().snapshot()
                for record in records:
                    record.source_evidence.turn_ids = ["invented-source"]
                return records
        report = replay.run_replay(factories={
            "raw_history_rag": CorruptProvenance, "managed_memory_rag": replay.ManagedControlStore,
        })
        self.assertFalse(report["acceptance_passed"])
        self.assertTrue(any(row["first_failure_stage"] == "storage"
                            for row in report["queries"] if row["arm"] == "raw_history_rag"))
        self.assertEqual(report["conditions"]["managed_memory_rag"]["passed_queries"], 26)

    def test_retrieval_has_irrelevant_distractors_zero_hits_and_more_than_top_k(self):
        capped = self.row("retrieval_quality", "rank-under-cap")
        single = self.row("retrieval_quality", "single-relevant")
        zero = self.row("retrieval_quality", "zero-relevance")
        self.assertEqual(len(capped["eligible_ids"]), 12)
        self.assertEqual(capped["retrieved_ids"], ["orchid-00", "orchid-01", "orchid-02"])
        self.assertEqual(single["retrieved_ids"], ["cobalt-target"])
        self.assertEqual(zero["retrieved_ids"], [])

    def test_return_all_mutation_fails_retrieval_acceptance(self):
        def return_everything(retriever, query, **kwargs):
            return [RetrievedMemory(record, 1.0, {"relevance": 1.0})
                    for record in retriever.records]
        with patch.object(replay.MemoryRetriever, "search", return_everything):
            report = replay.run_replay()
        self.assertFalse(report["acceptance_passed"])
        challenges = [r for r in report["queries"] if r["case_id"] == "retrieval_quality"]
        self.assertEqual(len(challenges), 6)
        self.assertTrue(all(r["first_failure_stage"] == "retrieval" for r in challenges))

    def test_ignoring_question_but_respecting_top_k_still_fails(self):
        def ignore_query(retriever, query, **kwargs):
            return [RetrievedMemory(record, 1.0, {"relevance": 1.0})
                    for record in retriever.records[:retriever.policy.top_k]]
        with patch.object(replay.MemoryRetriever, "search", ignore_query):
            report = replay.run_replay()
        self.assertFalse(report["acceptance_passed"])
        zero = [r for r in report["queries"] if r["case_id"] == "retrieval_quality"
                and r["check_id"] == "zero-relevance"]
        self.assertTrue(all(not r["stages"]["retrieval"] for r in zero))

    def test_preserved_ids_cannot_hide_replaced_final_prompt_body(self):
        real_builder = replay.build_context
        replaced = []
        def alter_body(**kwargs):
            result = real_builder(**kwargs)
            def corrupt(match):
                replaced.append(match.group(2))
                return match.group(1) + "Incorrect task status and strength." + match.group(3)
            prompt = re.sub(r"(<memory\b[^>]*>)(.*?)(</memory>)", corrupt,
                            result.prompt_context, flags=re.S)
            return replace(result, prompt_context=prompt)
        with patch.object(replay, "build_context", alter_body):
            report = replay.run_replay()
        self.assertGreater(len(replaced), 0)
        self.assertFalse(report["acceptance_passed"])
        changed = [r for r in report["queries"] if r["included_ids"]]
        self.assertTrue(all(r["trace_matches_prompt"] for r in changed))
        self.assertTrue(all(r["stages"]["storage"] and r["stages"]["retrieval"] for r in changed))
        self.assertTrue(all(r["first_failure_stage"] == "context_source_integrity" for r in changed))

    def test_changed_final_prompt_metadata_is_rejected_even_when_text_matches(self):
        real_builder = replay.build_context
        def alter_confirmation(**kwargs):
            result = real_builder(**kwargs)
            return replace(result, prompt_context=result.prompt_context.replace(
                'confirmation="user_confirmed"', 'confirmation="unconfirmed"'))
        with patch.object(replay, "build_context", alter_confirmation):
            report = replay.run_replay()
        self.assertFalse(report["acceptance_passed"])
        self.assertTrue(any(r["first_failure_stage"] == "context_source_integrity"
                            for r in report["queries"]))

    def test_escaped_source_text_is_verified_as_exact_decoded_quote(self):
        row = self.row("retrieval_quality", "single-relevant")
        _, ledger = replay.load_spec()
        self.assertTrue(row["source_integrity_matches_prompt"])
        self.assertEqual(row["rendered_text_sha256"]["cobalt-target"],
                         ledger["checks"]["retrieval_quality:single-relevant"]["verbatim_text_sha256"]["cobalt-target"])

    def test_unchanged_id_and_fields_cannot_hide_altered_verbatim_text(self):
        class CorruptContent(replay.RawHistoryControlStore):
            def snapshot(self):
                records = super().snapshot()
                for record in records:
                    if record.memory_id == "goal-new":
                        record.content = "Walking goal is twenty walks this week."
                return records
        report = replay.run_replay(factories={
            "raw_history_rag": CorruptContent, "managed_memory_rag": replay.ManagedControlStore,
        })
        current = next(row for row in report["queries"]
                       if row["arm"] == "raw_history_rag" and row["check_id"] == "current-goal")
        self.assertEqual(current["retrieved_ids"], ["goal-new"])
        self.assertEqual(current["first_failure_stage"], "storage")
        self.assertFalse(report["acceptance_passed"])

    def test_lifecycle_failure_is_not_hidden_by_retriever_defence(self):
        class UnfilteredRaw(replay.RawHistoryControlStore):
            def eligible(self, user_id, *, as_of, allow_restricted=False):
                return [record for record in self.snapshot() if record.lifecycle_status == "active"]
        report = replay.run_replay(factories={
            "raw_history_rag": UnfilteredRaw, "managed_memory_rag": replay.ManagedControlStore,
        })
        rows = [row for row in report["queries"]
                if row["arm"] == "raw_history_rag" and row["check_id"] == "default-sensitive-filter"]
        self.assertEqual(rows[0]["first_failure_stage"], "lifecycle_access")
        self.assertEqual(rows[0]["prohibited_ids_exposed"], [])
        self.assertFalse(report["acceptance_passed"])

    def test_actual_deleted_content_exposure_is_counted(self):
        class DeleteDoesNothing(replay.RawHistoryControlStore):
            def apply(self, event):
                if event["op"] == "delete" and event["event_id"] == "hard-delete":
                    return
                return super().apply(event)
        report = replay.run_replay(factories={
            "raw_history_rag": DeleteDoesNothing, "managed_memory_rag": replay.ManagedControlStore,
        })
        self.assertEqual(report["conditions"]["raw_history_rag"]["prohibited_exposure_queries"], 2)
        self.assertFalse(report["acceptance_passed"])

    def test_context_ledger_is_checked_independently_of_storage_and_retrieval(self):
        def change(ledger):
            ledger["checks"]["goals:current-goal"]["included_ids"] = []
        events, ledger = self.altered_spec(alter_ledger=change)
        result = replay.run_replay(events, ledger)
        rows = [row for row in result["queries"] if row["check_id"] == "current-goal"]
        self.assertEqual([row["first_failure_stage"] for row in rows],
                         ["context_inclusion", "context_inclusion"])

    def test_ledger_cannot_omit_failed_or_inconvenient_queries(self):
        def change(ledger):
            del ledger["checks"]["goals:current-goal"]
        events, ledger = self.altered_spec(alter_ledger=change)
        with self.assertRaisesRegex(ValueError, "every query"):
            replay.run_replay(events, ledger)

    def test_replay_does_not_claim_retrospective_state_reconstruction(self):
        def change(fixture):
            fixture["cases"][0]["events"][1]["as_of"] = "2025-01-01T00:00:00Z"
        events, ledger = self.altered_spec(alter_fixture=change)
        with self.assertRaisesRegex(ValueError, "retrospective"):
            replay.run_replay(events, ledger)


if __name__ == "__main__":
    unittest.main()
