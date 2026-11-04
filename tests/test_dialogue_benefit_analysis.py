"""Scientific failure-mode checks using explicitly fictional software fixtures."""
from copy import deepcopy
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation/dialogue-benefit"))
import study_analysis as a
import study_capture as capture


def fixture_snapshot():
    source = {"source_id": "fixture-source", "source_text": "My current fictional goal is a five-minute walk."}
    source["source_text_sha256"] = a.text_hash(source["source_text"])
    source.update(role="user", date="2026-11-04", start=0, end=len(source["source_text"]), quote=source["source_text"])
    ledger = {"schema_version": "dialogue-benefit-task-ledger-v1", "independent_source_review": {"status": "draft", "reviewer_ids": []}, "tasks": []}
    for variant in ("A", "B", "C"):
        for i in range(1, 13):
            row = {"task_id": f"T{i:02}", "variant": variant, "completion_criterion": "Software fixture only: answer the current fictional goal from the supplied source.", "source_reference": [source]}
            row["criterion_sha256"] = a.digest(row)
            ledger["tasks"].append(row)
    return {"manifest": {"manifest_sha256": "fixture-manifest", "mode": "fixture", "planned_slots": 60,
        "task_ledger_sha256": a.digest(ledger),
        "visit_elapsed_cap_seconds": 900,
        "visit_tasks": {"1": [], "2": [f"T{i:02}" for i in range(1, 5)],
                        "3": [f"T{i:02}" for i in range(5, 9)], "4": [f"T{i:02}" for i in range(9, 13)]}},
        "allocation": {"assignments": {"P001": "managed_memory_rag", "P002": "raw_history_rag"},
                       "task_variants": {"P001": "A", "P002": "B"}},
        "task_ledger": ledger,
        "events": [], "sealed": {"snapshot_sha256": "fixture-seal"},
        "consent_snapshot_sha256": "fixture-consent", "withdrawals": {}}


def event(snapshot, event_type, *, p="P001", visit=2, task="T01", at="2026-11-05T00:00:00Z", payload=None, response=None):
    e = {"participant_id": p, "visit": visit, "task_id": task, "timestamp": at,
         "event_type": event_type, "payload": payload or {}}
    row = {"event_sha256": a.digest([len(snapshot["events"]), e]), "event": e}
    if response is not None:
        row["response_content"] = response
    snapshot["events"].append(row)
    return row


def one_response(snapshot):
    event(snapshot, "enroll", visit=0, task="study")
    source = {"source_id": "fixture-source", "role": "user", "date": "2026-11-04", "text": "My current fictional goal is a five-minute walk."}
    content = {"source_history": [source], "current_query": "What is the current fictional goal?",
               "response": "A five-minute walk.", "task_key_sha256": snapshot["task_ledger"]["tasks"][0]["criterion_sha256"]}
    event(snapshot, "task_response", response=content)
    event(snapshot, "task_end", payload={"status": "completed", "assistance_received": False})
    return content


def rater(rid):
    return {"rater_id": rid, "qualification_description": "Software fixture only; no actual professional has rated.",
        "qualified_cbt_training_attested": True, "independent_from_system_development": True,
        "arm_blinded": True, "human_not_model_attested": True,
        "qualification_verification": {"verified_by": "fixture-registry", "verified_at": "2026-11-05T00:00:00Z",
            "record_sha256": "0" * 64, "independent_verifier_attestation": True}}


def rating(packet, rid, correct=True):
    s = packet["source_history"][0]
    return {"packet_id": packet["packet_id"], "packet_sha256": packet["packet_sha256"],
        "source_history_sha256": packet["source_history_sha256"], "task_key_sha256": packet["task_key_sha256"],
        "rater_id": rid, "rating_role": "independent", "status": "adjudicated", "task_correct": correct,
        "source_verified": True, "source_evidence": [{"source_id": s["source_id"], "start": 0,
            "end": len(s["text"]), "source_text_sha256": a.text_hash(s["text"])}],
        "rubric": {k: 1 for k in a.RUBRIC_KEYS}, "rubric_reasons": {}, "suspected_unblinding": False}


class AnalysisTests(unittest.TestCase):
    def test_planned_slots_are_not_people_and_no_inference(self):
        summary, rows = a.analyze_snapshot(fixture_snapshot())
        self.assertEqual(summary["planned_slots"], 60)
        self.assertEqual(summary["enrolled_slots"], 0)
        self.assertEqual(summary["actual_human_participants"], 0)
        self.assertEqual(rows, [])
        self.assertEqual(summary["primary_exploratory_intervals"], {})
        self.assertIsNone(summary["cost"]["verified_monetary_cost"])

    def test_missing_tasks_and_visits_keep_denominators_without_free_time(self):
        snapshot = fixture_snapshot()
        event(snapshot, "enroll", visit=0, task="study")
        summary, rows = a.analyze_snapshot(snapshot)
        self.assertEqual(rows[0]["planned_followup_tasks"], 12)
        self.assertEqual(rows[0]["task_correctness"], 0)
        self.assertEqual(rows[0]["restricted_elapsed_seconds"], 3600)
        self.assertEqual(rows[0]["checks_observed"], 0)
        self.assertIsNone(rows[0]["checks_per_12_planned_tasks"])
        self.assertFalse(summary["clinical_efficacy_measured"])

    def test_pending_review_is_unresolved_not_automatic_incorrect(self):
        snapshot = fixture_snapshot()
        one_response(snapshot)
        summary, rows = a.analyze_snapshot(snapshot)
        self.assertIsNone(rows[0]["task_correctness"])
        self.assertEqual(rows[0]["task_correctness_identification_bounds"], [0, 1 / 12])
        self.assertIn("completed_task_correctness_reviews_unresolved", summary["inference_blockers"])

    def test_assisted_and_timed_out_visits_penalized_but_keep_observed_time(self):
        snapshot = fixture_snapshot()
        event(snapshot, "visit_start", task="study")
        event(snapshot, "task_end", at="2026-11-05T00:00:10Z", payload={"status": "completed", "assistance_received": True})
        event(snapshot, "visit_end", task="study", at="2026-11-05T00:00:20Z",
              payload={"status": "completed", "burden_observation_complete": True})
        values = a._visit_time([r["event"] for r in snapshot["events"]], 900)
        self.assertEqual(values["restricted_seconds"], 900)
        self.assertEqual(values["observed_seconds"], 20)
        snapshot["events"][-1]["event"]["payload"]["status"] = "timeout"
        self.assertEqual(a._visit_time([r["event"] for r in snapshot["events"]], 900)["observed_seconds"], 20)

    def test_documented_breaks_exclude_pause_not_model_wait(self):
        snapshot = fixture_snapshot()
        event(snapshot, "visit_start", task="study")
        event(snapshot, "pause", task="study", at="2026-11-05T00:00:10Z")
        event(snapshot, "resume", task="study", at="2026-11-05T00:00:15Z")
        event(snapshot, "visit_end", task="study", at="2026-11-05T00:00:25Z", payload={"status": "completed", "burden_observation_complete": True})
        self.assertEqual(a._visit_time([r["event"] for r in snapshot["events"]], 900)["observed_seconds"], 20)

    def test_repeated_checks_subset_and_unknown_attempt_usage_not_zero(self):
        snapshot = fixture_snapshot()
        event(snapshot, "enroll", visit=0, task="study")
        event(snapshot, "memory_check", payload={"repeated": False})
        event(snapshot, "memory_check", payload={"repeated": True})
        event(snapshot, "cost", payload={"status": "failure", "prompt_tokens": None, "completion_tokens": None, "total_tokens": None})
        summary, rows = a.analyze_snapshot(snapshot)
        self.assertEqual(rows[0]["checks_observed"], 2)
        self.assertEqual(rows[0]["repeated_checks_observed"], 1)
        self.assertEqual(summary["cost"]["failed_attempt_records"], 1)
        self.assertEqual(summary["cost"]["missing_usage_attempts"], 1)
        self.assertFalse(summary["cost"]["token_usage_complete_for_recorded_attempts"])

    def test_postseal_erasure_blocks_itt_and_is_visible_in_arm_flow(self):
        snapshot = fixture_snapshot()
        one_response(snapshot)
        snapshot["withdrawals"]["P001"] = {"retain_collected_data": False}
        summary, rows = a.analyze_snapshot(snapshot)
        self.assertEqual(rows, [])
        self.assertEqual(summary["participant_flow_by_arm"]["managed_memory_rag"]["consent_erased"], 1)
        self.assertIn("consent_erasure_prevents_complete_ITT_analysis", summary["inference_blockers"])

    def test_bootstrap_is_participant_unit_with_predeclared_familywise_level(self):
        result = a._difference_interval([.5, .75], [.25, .5], seed=20261105)
        self.assertEqual(result["difference"], .25)
        self.assertEqual(len(result["ci_97_5_percent"]), 2)
        self.assertEqual(result["replicates"], 10000)


class PacketTests(unittest.TestCase):
    def make_packet(self, directory):
        snapshot = fixture_snapshot()
        content = one_response(snapshot)
        output = Path(directory) / "packets"
        with patch.object(a, "_locked_study", return_value=nullcontext(snapshot)):
            manifest = a.export_blinded_packets("not-a-real-study", output)
        packet = json.loads((output / "blinded_packets.json").read_text())[0]
        review = {"packet_manifest_sha256": manifest["packet_manifest_sha256"],
                  "raters": [rater("fixture-r1"), rater("fixture-r2")],
                  "ratings": [rating(packet, "fixture-r1"), rating(packet, "fixture-r2")]}
        return snapshot, output, packet, review, content

    def write_review(self, output, review):
        path = output / "filled_fixture_ratings.json"
        path.write_text(json.dumps(review))
        return path

    def test_packet_preserves_exact_text_without_arm_participant_or_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, output, packet, review, content = self.make_packet(directory)
            self.assertEqual(packet["response"], content["response"])
            self.assertEqual(packet["source_history"], content["source_history"])
            self.assertNotIn("arm", packet)
            self.assertNotIn("participant_id", packet)
            self.assertNotIn("cost", packet)
            labels, summary = a.load_review_labels(snapshot, output, self.write_review(output, review))
            self.assertTrue(labels[("P001", "T01")])
            self.assertEqual(summary["actual_human_ratings"], 0)
            self.assertEqual(summary["pre_adjudication_domain_descriptives"]["source_fidelity"]["exact_agreement_pairs"], 1)

    def test_disagreement_remains_unresolved_until_third_human_adjudication(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, output, packet, review, _ = self.make_packet(directory)
            review["ratings"][1]["task_correct"] = False
            path = self.write_review(output, review)
            labels, _ = a.load_review_labels(snapshot, output, path)
            self.assertIsNone(labels[("P001", "T01")])
            review["raters"].append(rater("fixture-r3"))
            third = rating(packet, "fixture-r3", False)
            third.update(rating_role="adjudicator", adjudicates_rater_ids=["fixture-r1", "fixture-r2"])
            review["ratings"].append(third)
            labels, _ = a.load_review_labels(snapshot, output, self.write_review(output, review))
            self.assertFalse(labels[("P001", "T01")])

    def test_adjudicated_unscorable_zero_is_distinct_from_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, output, _, review, _ = self.make_packet(directory)
            for r in review["ratings"]:
                r.update(status="adjudicated_unscorable", task_correct=None, unscorable_reason="Fixture source does not determine the specified task.")
            labels, _ = a.load_review_labels(snapshot, output, self.write_review(output, review))
            summary, rows = a.analyze_snapshot(snapshot, labels=labels)
            self.assertEqual(rows[0]["task_correctness"], 0)
            self.assertNotIn("completed_task_correctness_reviews_unresolved", summary["inference_blockers"])

    def test_packet_mutation_and_duplicate_labels_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, output, _, review, _ = self.make_packet(directory)
            review["ratings"].append(deepcopy(review["ratings"][0]))
            with self.assertRaisesRegex(ValueError, "duplicate rating"):
                a.load_review_labels(snapshot, output, self.write_review(output, review))
            packets = json.loads((output / "blinded_packets.json").read_text())
            packets[0]["response"] = "A stale unsupported activity."
            (output / "blinded_packets.json").write_text(json.dumps(packets))
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                a.load_review_labels(snapshot, output, self.write_review(output, review))

    def test_stale_source_span_and_model_rater_cannot_become_expert_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, output, _, review, _ = self.make_packet(directory)
            review["ratings"][0]["source_evidence"][0]["source_text_sha256"] = "wrong-source-version"
            with self.assertRaisesRegex(ValueError, "text/version"):
                a.load_review_labels(snapshot, output, self.write_review(output, review))
            review["raters"][0]["human_not_model_attested"] = False
            with self.assertRaisesRegex(ValueError, "human rater"):
                a.load_review_labels(snapshot, output, self.write_review(output, review))

    def test_withdrawal_after_export_invalidates_packet_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, output, _, review, _ = self.make_packet(directory)
            snapshot["consent_snapshot_sha256"] = "revoked-after-export"
            with self.assertRaisesRegex(ValueError, "withdrawal snapshot"):
                a.load_review_labels(snapshot, output, self.write_review(output, review))

    def test_arbitrary_task_key_without_actual_frozen_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = fixture_snapshot()
            one_response(snapshot)
            snapshot["task_ledger"] = None
            with patch.object(a, "_locked_study", return_value=nullcontext(snapshot)):
                with self.assertRaisesRegex(ValueError, "actual predeclared frozen task ledger"):
                    a.export_blinded_packets("fixture", Path(directory) / "invalid")

    def test_analysis_identity_change_is_rejected(self):
        snapshot = fixture_snapshot()
        snapshot["manifest"]["analysis_code_sha256"] = "another-executable"
        with self.assertRaisesRegex(ValueError, "frozen study identity"):
            a.analyze_snapshot(snapshot)


class CaptureIntegrationTests(unittest.TestCase):
    def prepare(self, directory):
        root = Path(directory) / "study"
        package = Path(a.__file__).parent
        capture.prepare_study(root, package / "protocol.json", mode="fixture", seed=42,
                              task_ledger_path=package / "task_materials_draft.json")
        index = 0
        def record(kind, *, visit=2, task="T01", payload=None, second=0):
            nonlocal index
            index += 1
            e = {"event_id": f"fixture-{index}", "participant_id": "P001", "visit": visit,
                 "task_id": task, "timestamp": f"2026-11-05T00:00:{second:02}Z",
                 "event_type": kind, "payload": payload or {}}
            return capture.record_event(root, e)
        record("enroll", visit=0, task="study", payload={"consent_recorded": True,
            "nonclinical_fictional_tasks_only": True, "adult_eligible_attestation": True,
            "ethics_clearance_id": "software-fixture-no-human", "operator_id": "fixture-runner",
            "real_participant_attestation": False})
        record("visit_start", task="visit", payload={"instrument_sha256": "0" * 64, "operator_id": "fixture-runner"})
        record("task_start")
        snapshot = capture.read_snapshot(root)
        variant = snapshot["allocation"]["task_variants"]["P001"]
        ledger = snapshot["task_ledger"]
        task = next(t for t in ledger["tasks"] if t["variant"] == variant and t["task_id"] == "T01")
        sources = {s["source_id"]: s for s in ledger["sources"] if s["variant"] == variant}
        content = {"source_history": [{"source_id": sid, "role": sources[sid]["role"],
                    "date": sources[sid]["date"], "text": sources[sid]["source_text"]} for sid in task["visible_source_ids"]],
            "current_query": task["task_query"], "response": "Software test-double response; not an actual model or participant outcome.",
            "task_key_sha256": task["criterion_sha256"], "fictional_content_attestation": True,
            "response_origin": "offline_fixture"}
        record("task_response", payload=content)
        record("task_end", second=1, payload={"status": "completed", "burden_observation_complete": True, "assistance_received": False})
        record("visit_end", task="visit", second=2, payload={"status": "incomplete", "burden_observation_complete": False, "attempt_log_complete": True})
        capture.seal_study(root)
        return root, record

    def test_real_capture_to_blinded_packet_to_analysis_and_revocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root, record = self.prepare(directory)
            output = Path(directory) / "packets"
            exported = a.export_blinded_packets(root, output)
            self.assertEqual(exported["packet_count"], 1)
            packets = json.loads((output / "blinded_packets.json").read_text())
            self.assertIn("completion_criterion", packets[0]["task_criterion"])
            summary, rows = a.analyze(root)
            self.assertEqual(summary["actual_human_participants"], 0)
            self.assertIsNone(rows[0]["task_correctness"])
            record("withdraw", visit=0, task="study", second=3, payload={"retain_collected_data": False})
            summary, rows = a.analyze(root)
            self.assertEqual(rows, [])
            self.assertEqual(summary["consent_erased_slots"], 1)
            new = a.export_blinded_packets(root, Path(directory) / "after-revocation")
            self.assertEqual(new["packet_count"], 0)
            with self.assertRaisesRegex(ValueError, "withdrawal snapshot"):
                a.load_review_labels(capture.read_snapshot(root), output, output / "blank_ratings.json")

    def test_publication_and_revocation_are_serialized(self):
        with tempfile.TemporaryDirectory() as directory:
            root, record = self.prepare(directory)
            entered_write, permit_write, withdrawal_done = threading.Event(), threading.Event(), threading.Event()
            errors = []
            original = a._write_new
            def guarded(path, value):
                if path.name == "blinded_packets.json":
                    entered_write.set()
                    if not permit_write.wait(5):
                        raise RuntimeError("test synchronization timed out")
                original(path, value)
            def export():
                try:
                    a.export_blinded_packets(root, Path(directory) / "packets")
                except BaseException as error:
                    errors.append(error)
            def withdraw():
                try:
                    record("withdraw", visit=0, task="study", second=3, payload={"retain_collected_data": False})
                    withdrawal_done.set()
                except BaseException as error:
                    errors.append(error)
            with patch.object(a, "_write_new", side_effect=guarded):
                export_thread = threading.Thread(target=export)
                export_thread.start()
                self.assertTrue(entered_write.wait(5))
                withdrawal_thread = threading.Thread(target=withdraw)
                withdrawal_thread.start()
                self.assertFalse(withdrawal_done.wait(.05), "withdrawal interleaved with snapshot publication")
                permit_write.set()
                export_thread.join(5)
                withdrawal_thread.join(5)
            self.assertFalse(export_thread.is_alive())
            self.assertFalse(withdrawal_thread.is_alive())
            # Capture deliberately uses a nonblocking lock: the concurrent
            # revocation is explicitly rejected, never partially committed.
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], capture.CaptureError)
            self.assertIn("in use", str(errors[0]))
            self.assertFalse(withdrawal_done.is_set())
            record("withdraw", visit=0, task="study", second=3, payload={"retain_collected_data": False})
            self.assertFalse(capture.read_snapshot(root)["withdrawals"]["P001"]["retain_collected_data"])


if __name__ == "__main__":
    unittest.main()
