"""Capture/provenance/privacy regressions using fictional offline fixtures only."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation/dialogue-benefit"))
import study_capture as capture


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


class StudyCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "fixture-study"
        self.protocol = ROOT / "evaluation/dialogue-benefit/protocol.json"
        self.ledger = ROOT / "evaluation/dialogue-benefit/task_materials_draft.json"
        self.seconds = 0

    def prepare(self, *, mode="fixture", ledger=True):
        return capture.prepare_study(self.root, self.protocol, mode=mode, seed=17,
                                     task_ledger_path=self.ledger if ledger else None)

    def event(self, kind, payload, *, participant="P001", visit=0, task="study", event_id=None):
        self.seconds += 1
        return {"event_id": event_id or f"fixture-event-{self.seconds:04d}", "participant_id": participant,
                "visit": visit, "task_id": task,
                "timestamp": (datetime(2026, 11, 5, tzinfo=timezone.utc) + timedelta(seconds=self.seconds)).isoformat(),
                "event_type": kind, "payload": payload}

    def enroll(self, *, participant="P001", real=False):
        event = self.event("enroll", {"consent_recorded": True, "nonclinical_fictional_tasks_only": True,
                                     "adult_eligible_attestation": True, "ethics_clearance_id": "fixture-clearance",
                                     "operator_id": "fixture-operator", "real_participant_attestation": real},
                           participant=participant)
        capture.record_event(self.root, event)
        return event

    def start_visit(self, visit=2):
        event = self.event("visit_start", {"instrument_sha256": "1" * 64, "operator_id": "fixture-operator"},
                           visit=visit, task="visit")
        capture.record_event(self.root, event)
        return event

    def start_task(self, task="T01", visit=2):
        event = self.event("task_start", {}, visit=visit, task=task)
        capture.record_event(self.root, event)
        return event

    def response(self, task="T01", visit=2):
        snapshot = capture.read_snapshot(self.root)
        variant = snapshot["allocation"]["task_variants"]["P001"]
        row = next(row for row in snapshot["task_ledger"]["tasks"]
                   if row["task_id"] == task and row["variant"] == variant)
        originals = {source["source_id"]: source for source in snapshot["task_ledger"]["sources"]
                     if source["variant"] == variant}
        history = [{"source_id": source_id, "role": originals[source_id]["role"], "date": originals[source_id]["date"],
                    "text": originals[source_id]["source_text"]} for source_id in row["visible_source_ids"]]
        return self.event("task_response", {"source_history": history, "current_query": row["task_query"],
                                           "response": "Explicitly fictional offline transport fixture.",
                                           "task_key_sha256": row["criterion_sha256"],
                                           "fictional_content_attestation": True, "response_origin": "offline_fixture"},
                          visit=visit, task=task)

    def observed_response(self, task="T01", visit=2):
        self.prepare()
        self.enroll()
        self.start_visit(visit)
        self.start_task(task, visit)
        event = self.response(task, visit)
        result = capture.record_event(self.root, event)
        return event, result

    def cost_payload(self):
        return {"attempt_id": "fixture-attempt-1", "status": "timeout", "model": "fixture-model",
                "prompt_tokens": None, "completion_tokens": None, "total_tokens": None,
                "cost_amount": None, "currency": None, "provider": None, "request_id": None,
                "model_revision": None, "cached_tokens": None, "price_reference_date": None,
                "stage": "generation", "latency_seconds": None}

    def test_planned_slots_are_not_people_and_allocation_is_concealed_balanced_and_frozen(self):
        status = self.prepare()
        self.assertEqual(status["actual_human_participants"], 0)
        self.assertEqual(status["enrolled_observation_slots"], 0)
        self.assertEqual(status["planned_allocation_slots"], 60)
        snapshot = capture.read_snapshot(self.root)
        assignments = snapshot["allocation"]["assignments"]
        self.assertEqual({arm: list(assignments.values()).count(arm) for arm in capture.ARMS},
                         dict.fromkeys(capture.ARMS, 20))
        for offset in range(0, 60, 6):
            block = [assignments[f"P{i:03d}"] for i in range(offset + 1, offset + 7)]
            self.assertEqual({arm: block.count(arm) for arm in capture.ARMS}, dict.fromkeys(capture.ARMS, 2))
        for arm in capture.ARMS:
            labels = [snapshot["allocation"]["task_variants"][slot] for slot in assignments if assignments[slot] == arm]
            self.assertEqual([labels.count(label) for label in "ABC"], [7, 7, 6])
        self.assertNotIn("randomization_seed", snapshot["manifest"])
        self.assertNotIn("assignments", snapshot["manifest"])
        self.assertTrue(snapshot["manifest"]["analysis_code_sha256"])
        with self.assertRaisesRegex(capture.CaptureError, "new study directory"):
            self.prepare()

    def test_human_enrollment_cannot_start_without_launch_clearance(self):
        self.prepare(mode="planned_human", ledger=False)
        with self.assertRaisesRegex(capture.CaptureError, "launch clearance"):
            self.enroll(real=True)
        self.assertEqual(capture.study_status(self.root)["actual_human_participants"], 0)

    def test_checkbox_launch_record_cannot_promote_draft_sources_to_human_gold(self):
        protocol = json.loads(self.protocol.read_text())
        ledger = json.loads(self.ledger.read_text())
        launch = {"ethics_clearance_id": "fixture-clearance", "consent_document_sha256": "1" * 64,
                  "task_ledger_sha256": capture.canonical_hash(ledger), "code_commit": "a" * 40,
                  "model_revision": "fixture-model", "prompt_sha256": "2" * 64, "analysis_sha256": "3" * 64,
                  "approved_data_policy_sha256": "4" * 64, "max_total_api_calls": 10,
                  "max_attempts_per_request": 1, "max_currency_cost": 1.0, "currency": "USD", "language": "en",
                  "responsible_investigator_id": "fictional-investigator", "completed_gates": protocol["launch_gates"],
                  "approval_attestation": True}
        launch_path = Path(self.temp.name) / "fictional-launch-checks-not-actual-clearance.json"
        launch_path.write_bytes(canonical_bytes(launch))
        capture.prepare_study(self.root, self.protocol, mode="planned_human", seed=17,
                              task_ledger_path=self.ledger, launch_record_path=launch_path)
        with self.assertRaisesRegex(capture.CaptureError, "independent human source validators"):
            self.enroll(real=True)
        self.assertEqual(capture.study_status(self.root)["actual_human_participants"], 0)

    def test_fixture_cannot_assert_real_humans_or_skip_concealed_enrollment_order_or_consent(self):
        self.prepare()
        with self.assertRaisesRegex(capture.CaptureError, "next concealed slot"):
            self.enroll(participant="P002")
        with self.assertRaisesRegex(capture.CaptureError, "real participants"):
            self.enroll(real=True)
        event = self.event("enroll", {"consent_recorded": False, "nonclinical_fictional_tasks_only": True,
                                     "adult_eligible_attestation": True, "ethics_clearance_id": "fixture",
                                     "operator_id": "fixture", "real_participant_attestation": False})
        with self.assertRaisesRegex(capture.CaptureError, "consent"):
            capture.record_event(self.root, event)
        self.enroll()
        self.assertEqual(capture.study_status(self.root)["fixture_observation_slots"], 1)
        self.assertEqual(capture.study_status(self.root)["actual_human_participants"], 0)

    def test_identical_retries_do_not_duplicate_and_conflicting_ids_fail(self):
        self.prepare()
        event = self.enroll()
        result = capture.record_event(self.root, event)
        self.assertEqual(result["status"], "already_recorded")
        event["payload"]["operator_id"] = "different-fixture-operator"
        with self.assertRaisesRegex(capture.CaptureError, "different content"):
            capture.record_event(self.root, event)
        self.assertEqual(capture.study_status(self.root)["collection_event_count"], 1)

    def test_partial_or_modified_journal_is_rejected_not_repaired_silently(self):
        self.prepare()
        self.enroll()
        path = self.root / "events.jsonl"
        original = path.read_bytes()
        path.write_bytes(original[:-1])
        with self.assertRaisesRegex(capture.CaptureError, "partial journal"):
            capture.read_snapshot(self.root)
        path.write_bytes(original.replace(b"fixture-operator", b"changed-operator"))
        with self.assertRaisesRegex(capture.CaptureError, "content/hash"):
            capture.read_snapshot(self.root)

    def test_private_source_artifact_changed_body_is_detected_even_with_preserved_ids(self):
        event, _ = self.observed_response()
        snapshot = capture.read_snapshot(self.root)
        row = snapshot["events"][-1]
        self.assertNotIn("source_history", row["event"]["payload"])
        self.assertEqual(row["response_content"]["source_history"], event["payload"]["source_history"])
        artifact = self.root / row["event"]["payload"]["artifact_path"]
        content = json.loads(artifact.read_text())
        content["source_history"][0]["text"] = "Altered fixture history with the same source ID."
        artifact.write_bytes(canonical_bytes(content) + b"\n")
        with self.assertRaisesRegex(capture.CaptureError, "content changed"):
            capture.read_snapshot(self.root)

    def test_wrong_query_variant_source_role_date_or_body_refused_before_capture(self):
        self.prepare()
        self.enroll()
        self.start_visit()
        self.start_task()
        original = self.response()
        for mutation in ("query", "key", "role", "date", "body", "missing_source"):
            event = deepcopy(original)
            event["event_id"] += f"-{mutation}"
            if mutation == "query":
                event["payload"]["current_query"] += " post-hoc changed question"
            elif mutation == "key":
                event["payload"]["task_key_sha256"] = "0" * 64
            elif mutation == "role":
                event["payload"]["source_history"][0]["role"] = "assistant"
            elif mutation == "date":
                event["payload"]["source_history"][0]["date"] = "2035-01-01T00:00:00Z"
            elif mutation == "body":
                event["payload"]["source_history"][0]["text"] += " edited"
            else:
                event["payload"]["source_history"].pop()
            with self.subTest(mutation=mutation), self.assertRaises(capture.CaptureError):
                capture.record_event(self.root, event)
        self.assertEqual(capture.study_status(self.root)["collection_event_count"], 3)

    def test_ledger_false_unicode_quote_refused_even_after_rehashing_criterion(self):
        ledger = json.loads(self.ledger.read_text())
        task = ledger["tasks"][0]
        task["source_reference"][0]["quote"] += " altered"
        task["criterion_sha256"] = capture.canonical_hash({key: value for key, value in task.items() if key != "criterion_sha256"})
        with self.assertRaisesRegex(capture.CaptureError, "Unicode source span"):
            capture._verify_task_ledger(ledger)

    def test_explicit_unknown_cost_is_not_zero_and_duplicate_attempt_is_refused(self):
        self.prepare()
        self.enroll()
        self.start_visit()
        payload = self.cost_payload()
        capture.record_event(self.root, self.event("cost", payload, visit=2, task="visit"))
        saved = capture.read_snapshot(self.root)["events"][-1]["event"]["payload"]
        self.assertIsNone(saved["total_tokens"])
        self.assertIsNone(saved["cost_amount"])
        with self.assertRaisesRegex(capture.CaptureError, "already counted"):
            capture.record_event(self.root, self.event("cost", payload, visit=2, task="visit"))
        missing = self.cost_payload()
        del missing["provider"]
        missing["attempt_id"] = "different-fixture-attempt"
        with self.assertRaisesRegex(capture.CaptureError, "payload fields"):
            capture.record_event(self.root, self.event("cost", missing, visit=2, task="visit"))

    def test_repeat_is_one_check_and_binds_source_issue_evidence_and_cooldown(self):
        self.prepare()
        self.enroll()
        self.start_visit()
        original = {"check_id": "check-1", "memory_subject_id": "goal-1", "source_version_sha256": "1" * 64,
                    "current_issue_id": "issue-1", "evidence_state_sha256": "2" * 64,
                    "repeated": False, "previous_check_id": None}
        capture.record_event(self.root, self.event("memory_check", original, visit=2, task="visit"))
        repeat = {**original, "check_id": "check-2", "repeated": True, "previous_check_id": "check-1"}
        for field, value in (("current_issue_id", "different-issue"), ("evidence_state_sha256", "3" * 64),
                             ("source_version_sha256", "4" * 64), ("previous_check_id", "missing-check")):
            with self.subTest(field=field), self.assertRaisesRegex(capture.CaptureError, "repeat check"):
                capture.record_event(self.root, self.event("memory_check", {**repeat, field: value}, visit=2, task="visit"))
        capture.record_event(self.root, self.event("memory_check", repeat, visit=2, task="visit"))
        checks = [row["event"] for row in capture.read_snapshot(self.root)["events"] if row["event"]["event_type"] == "memory_check"]
        self.assertEqual(len(checks), 2)
        self.assertEqual(sum(row["payload"]["repeated"] for row in checks), 1)

    def test_timing_requires_timezone_and_documented_breaks_never_remove_model_waiting(self):
        self.prepare()
        self.enroll()
        self.start_visit(1)
        with self.assertRaisesRegex(capture.CaptureError, "cannot be excluded"):
            capture.record_event(self.root, self.event("pause", {"reason_code": "model_waiting"}, visit=1, task="visit"))
        capture.record_event(self.root, self.event("pause", {"reason_code": "documented_break"}, visit=1, task="visit"))
        with self.assertRaisesRegex(capture.CaptureError, "open break"):
            capture.record_event(self.root, self.event("visit_end", {"status": "completed",
                                                                      "burden_observation_complete": True,
                                                                      "attempt_log_complete": True}, visit=1, task="visit"))
        capture.record_event(self.root, self.event("visit_end", {"status": "incomplete",
                                                                  "burden_observation_complete": False,
                                                                  "attempt_log_complete": False}, visit=1, task="visit"))
        bad = self.event("visit_start", {"instrument_sha256": "1" * 64, "operator_id": "fixture"}, visit=2, task="visit")
        bad["timestamp"] = "2026-11-06T12:00:00"
        with self.assertRaisesRegex(capture.CaptureError, "timezone"):
            capture.record_event(self.root, bad)

    def test_source_assessment_requires_response_key_and_actual_source_hash(self):
        event, saved = self.observed_response()
        assessment = {"status": "adjudicated", "correct": True, "source_verified": True,
                      "assessor_id": "fictional-assessor-1", "independent_attestation": True,
                      "blinding_attestation": True, "human_not_model_attestation": True,
                      "task_key_sha256": event["payload"]["task_key_sha256"],
                      "source_evidence_sha256": "0" * 64,
                      "task_response_event_sha256": saved["event_sha256"]}
        with self.assertRaisesRegex(capture.CaptureError, "source/task-key"):
            capture.record_event(self.root, self.event("task_assessment", assessment, visit=2, task="T01"))
        assessment["source_evidence_sha256"] = capture.canonical_hash(event["payload"]["source_history"])
        capture.record_event(self.root, self.event("task_assessment", assessment, visit=2, task="T01"))
        self.assertEqual(capture.study_status(self.root)["actual_human_participants"], 0)

    def test_after_seal_revocation_erases_response_and_retry_cannot_resurrect_it(self):
        event, _ = self.observed_response()
        before = capture.read_snapshot(self.root)
        artifact = self.root / before["events"][-1]["event"]["payload"]["artifact_path"]
        prefix = (self.root / "events.jsonl").read_bytes()
        capture.seal_study(self.root)
        withdrawal = self.event("withdraw", {"retain_collected_data": False})
        capture.record_event(self.root, withdrawal)
        after = capture.read_snapshot(self.root)
        self.assertFalse(artifact.exists())
        self.assertEqual((self.root / "events.jsonl").read_bytes(), prefix)
        self.assertIsNone(after["events"][-1]["response_content"])
        self.assertTrue(after["events"][-1]["erased"])
        self.assertNotEqual(before["consent_snapshot_sha256"], after["consent_snapshot_sha256"])
        self.assertEqual(capture.record_event(self.root, event)["status"], "already_recorded")
        self.assertFalse(artifact.exists())
        with self.assertRaisesRegex(capture.CaptureError, "sealed"):
            capture.record_event(self.root, self.event("failure", {"stage": "system", "reason_code": "fixture"}, visit=2, task="visit"))

    def test_revocation_commits_export_barrier_before_erasure_disk_error_and_retry_repairs(self):
        self.observed_response()
        withdrawal = self.event("withdraw", {"retain_collected_data": False})
        with patch.object(capture, "_erase_response_files", side_effect=OSError("fixture disk interruption")):
            with self.assertRaisesRegex(OSError, "disk interruption"):
                capture.record_event(self.root, withdrawal)
        interrupted = capture.read_snapshot(self.root)
        self.assertIn("P001", interrupted["withdrawals"])
        self.assertIsNone(interrupted["events"][-1]["response_content"])
        self.assertTrue((self.root / "restricted/responses/P001").exists())
        self.assertEqual(capture.record_event(self.root, withdrawal)["status"], "already_recorded")
        self.assertFalse((self.root / "restricted/responses/P001").exists())
        self.assertEqual(len(capture.read_snapshot(self.root)["revocations"]), 1)

    def test_corrupt_private_response_blocks_scoring_but_cannot_block_consent_revocation(self):
        self.observed_response()
        before = capture.read_snapshot(self.root)
        artifact = self.root / before["events"][-1]["event"]["payload"]["artifact_path"]
        artifact.write_text("unreadable fictional response bytes", encoding="utf-8")
        with self.assertRaises(capture.CaptureError):
            capture.read_snapshot(self.root)
        capture.record_event(self.root, self.event("withdraw", {"retain_collected_data": False}))
        after = capture.read_snapshot(self.root)
        self.assertFalse(artifact.exists())
        self.assertTrue(after["events"][-1]["erased"])
        self.assertIsNone(after["events"][-1]["response_content"])

    def test_changed_capture_source_cannot_append_outcomes_but_can_revoke_consent(self):
        self.prepare()
        self.enroll()
        real_file_hash = capture._file_hash
        def changed_executable(path):
            return "f" * 64 if Path(path) == Path(capture.__file__) else real_file_hash(path)
        with patch.object(capture, "_file_hash", side_effect=changed_executable):
            with self.assertRaisesRegex(capture.CaptureError, "capture executable changed"):
                self.start_visit()
            capture.record_event(self.root, self.event("withdraw", {"retain_collected_data": False}))
        self.assertIn("P001", capture.read_snapshot(self.root)["withdrawals"])

    def test_readonly_snapshot_never_creates_files_or_changes_frozen_inputs(self):
        self.prepare()
        self.enroll()
        before = {str(path.relative_to(self.root)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in self.root.rglob("*") if path.is_file()}
        capture.study_status(self.root)
        capture.read_snapshot(self.root)
        after = {str(path.relative_to(self.root)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_concurrent_export_read_lock_blocks_withdrawal_until_publication_finishes(self):
        self.prepare()
        self.enroll()
        withdrawal = self.event("withdraw", {"retain_collected_data": False})
        with capture.locked_snapshot(self.root):
            with self.assertRaisesRegex(capture.CaptureError, "directory is in use"):
                capture.record_event(self.root, withdrawal)
        capture.record_event(self.root, withdrawal)
        self.assertIn("P001", capture.read_snapshot(self.root)["withdrawals"])


if __name__ == "__main__":
    unittest.main()
