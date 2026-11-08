"""Invented receipts test paid-call accounting without credentials or a model."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation/cbt-response-review"))
SPEC = importlib.util.spec_from_file_location(
    "cbt_r1_r5_runner", ROOT / "evaluation/cbt-response-review/run_colab_review.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
analysis = runner.analysis


def packet(identity="BL-0001", response="You reported one attempt; we can keep that pace."):
    payload = {"history": [{"message_id": "u1", "role": "user", "content": "I tried the exercise once.",
                            "date": "2026-01-01"}],
               "input": "How should I continue?", "response": response}
    return {"canonical_id": identity, **payload, "packet_sha256": runner.stable(payload)}


def assessment():
    return {**{dimension: 4 for dimension in analysis.DIMENSIONS[:4]}, "R5": "pass",
            "evidence": [{"dimension": dimension, "source_quote": "I tried the exercise once.",
                          "response_quote": "You reported one attempt", "reason": "Invented receipt fixture, not an effect score."}
                         for dimension in analysis.DIMENSIONS]}


def response(value=None, *, reported_usage=None):
    return {"hypothesis": json.dumps(assessment() if value is None else value),
            "provider_model": runner.ALIAS, "finish_reason": "stop", "latency_seconds": 1.5,
            "usage": {} if reported_usage is None else reported_usage}


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in values))


def prepared(folder):
    folder = Path(folder)
    review = folder / "review"
    rubric = folder / "guide.md"
    rubric.write_text("Invented R1 R2 R3 R4 1-5 and R5 pass/fail fixture.\n")
    packets = [packet(), packet("BL-0002", "You reported one attempt; would keeping that pace help?")]
    write_rows(review / "reviewer/canonical_packets.jsonl", packets)
    controller = []
    for case, privacy in (("A:T01", False), ("A:T09", True)):
        for arm in analysis.ARMS:
            chosen = packets[1] if arm == "short_term" else packets[0]
            controller.append({"case_id": case, "arm": arm, "source_family_id": "invented-family",
                               "variant": "A", "task_family_id": case.split(":")[1],
                               "canonical_id": None if privacy else chosen["canonical_id"],
                               "packet_sha256": None if privacy else chosen["packet_sha256"],
                               "review_status": "privacy_withheld" if privacy else "pending"})
    write_rows(review / "controller/rows.jsonl", controller)
    manifest = {"schema": "invented-fixture", "seed": 20261109,
                "historical_executable_commit": "fixture-not-a-real-run",
                "historical_protocol_sha256": "fixture-not-a-real-protocol",
                "response_view": "answer_field_only",
                "counts": {"planned_rows": 6, "planned_tasks": 2, "reviewable_rows": 3,
                           "privacy_withheld_rows": 3, "unique_visible_review_objects": 2},
                "rating_protocol": {"rubric_frozen": True, "rubric_sha256": runner.sha(rubric),
                                    "packet_digest_fields": ["history", "input", "response"]},
                "output_sha256": {name: runner.sha(review / name) for name in (
                    "reviewer/canonical_packets.jsonl", "controller/rows.jsonl")}}
    (review / "manifest.json").write_text(json.dumps(manifest))
    run_dir = folder / "run"
    runner.prepare(review, run_dir, rubric)
    protocol, packet_map, guide = runner.load_protocol(run_dir)
    return review, rubric, run_dir, protocol, packet_map, guide


class FrozenPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.review, self.rubric, self.run_dir, self.protocol, self.packets, self.guide = prepared(self.folder)

    def test_prepare_has_zero_calls_and_preserves_complete_denominator(self):
        self.assertEqual(self.protocol["maximum_provider_attempts"], 2)
        self.assertEqual(self.protocol["planned_rows"], 6)
        self.assertEqual(self.protocol["privacy_withheld_rows"], 3)
        self.assertEqual(self.protocol["new_response_generation_calls"], 0)
        self.assertFalse((self.run_dir / "attempts.jsonl").exists())

    def test_input_and_rubric_changes_cannot_resume(self):
        for name in ("rubric.md", "canonical_packets.jsonl", "review_manifest.json"):
            path = self.run_dir / name
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            with self.subTest(name=name), self.assertRaises(ValueError):
                runner.load_protocol(self.run_dir)
            path.write_bytes(original)

    def test_protocol_changes_are_rejected_even_if_checksum_recomputed(self):
        changed = deepcopy(self.protocol)
        changed["settings"]["model"] = "unauthorized-substitute"
        changed.pop("protocol_sha256")
        changed["protocol_sha256"] = runner.stable(changed)
        (self.run_dir / "protocol.json").write_text(json.dumps(changed))
        with self.assertRaises(ValueError):
            runner.load_protocol(self.run_dir)

    def test_existing_run_cannot_be_overwritten(self):
        with self.assertRaises(ValueError):
            runner.prepare(self.review, self.run_dir, self.rubric)

    def test_unfrozen_or_different_rubric_cannot_start_a_review(self):
        self.rubric.write_text("Changed guide.\n")
        with self.assertRaises(ValueError):
            runner.prepare(self.review, self.folder / "new-run", self.rubric)
        self.assertFalse((self.folder / "new-run").exists())

    def test_only_visible_fields_enter_model_data(self):
        request = runner.request_for(self.packets["BL-0001"], self.guide)
        self.assertEqual(set(json.loads(request["messages"][1]["content"])), {"history", "input", "response"})
        for forbidden in ("arm", "completion_criterion", "old_criterion_result", "source_history"):
            self.assertNotIn(forbidden, json.loads(request["messages"][1]["content"]))

    def test_budget_checks_complete_escaped_outbound(self):
        with self.assertRaises(ValueError):
            runner.request_for(self.packets["BL-0001"], "汉" * runner.MAX_OUTBOUND_CHARS)

    def test_actual_host_guard_requires_colab_without_reading_a_secret(self):
        # Test the actual guard, never the paid run entry. Even running this
        # suite inside Colab cannot accidentally read Secret or send a request.
        try:
            runner.require_colab()
        except RuntimeError as error:
            self.assertIn("only inside Google Colab", str(error))
        else:
            self.skipTest("Hosted Colab passes; negative host check is inapplicable")
        self.assertFalse((self.run_dir / "attempts.jsonl").exists())


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        _, _, self.run_dir, self.protocol, self.packets, self.guide = prepared(self.folder)

    def journal(self):
        return runner.Journal(self.run_dir, self.protocol, self.packets, self.guide)

    def test_one_canonical_attempt_and_usage_do_not_repeat_on_resume(self):
        journal = self.journal()
        calls = []
        def invented(messages):
            calls.append(messages)
            return response(reported_usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})
        journal.process("BL-0001", invented)
        self.assertEqual(len(calls), 1)
        resumed = self.journal()
        with self.assertRaises(ValueError):
            resumed.process("BL-0001", invented)
        self.assertEqual(len(calls), 1)
        summary = resumed.write_summary()
        self.assertEqual(summary["paid_judge_attempts"], 1)
        self.assertEqual(summary["reported_tokens"]["total_tokens"], 120)
        self.assertEqual(summary["pending_canonical_objects"], 1)

    def test_missing_usage_remains_unknown_not_zero(self):
        journal = self.journal()
        journal.process("BL-0001", lambda messages: response())
        summary = journal.write_summary()
        self.assertIsNone(summary["reported_tokens"]["prompt_tokens"])
        self.assertEqual(summary["attempts_missing_usage_fields"]["total_tokens"], 1)
        self.assertIsNone(summary["monetary_cost"])

    def test_schema_or_unbound_evidence_failure_is_preserved_without_retry(self):
        value = assessment()
        value["evidence"][0]["source_quote"] = "unsupported fabricated source"
        journal = self.journal()
        journal.process("BL-0001", lambda messages: response(value))
        rating = runner.rows(self.run_dir / "ratings.jsonl")[0]
        self.assertEqual(rating["status"], "technical_failure")
        self.assertIsNone(rating["assessment"])
        self.assertEqual(journal.write_summary()["paid_judge_attempts"], 1)
        with self.assertRaises(ValueError):
            self.journal().process("BL-0001", lambda messages: response())

    def test_semantic_unresolved_is_not_a_safety_failure_or_technical_error(self):
        journal = self.journal()
        journal.process("BL-0001", lambda messages: response({"status": "unresolved", "reason": "Invented ambiguous source."}))
        rating = runner.rows(self.run_dir / "ratings.jsonl")[0]
        self.assertEqual(rating["status"], "unresolved")
        self.assertIsNone(rating["assessment"])
        self.assertEqual(journal.write_summary()["technical_failures"], 0)

    def test_timeout_preserves_unknown_send_and_blocks_whole_run(self):
        def lost(messages):
            raise TimeoutError("DO_NOT_EXPORT_SECRET_EXAMPLE")
        journal = self.journal()
        with self.assertRaises(runner.AttemptUnknown):
            journal.process("BL-0001", lost)
        with self.assertRaises(runner.AttemptUnknown):
            self.journal()
        summary = json.loads((self.run_dir / "execution_summary.json").read_text())
        self.assertEqual(summary["attempt_unknown"], 1)
        self.assertIsNone(summary["reported_tokens"]["total_tokens"])
        body = (self.run_dir / "outcomes.jsonl").read_text()
        self.assertNotIn("DO_NOT_EXPORT_SECRET_EXAMPLE", body)
        self.assertIn("attempt_unknown", body)
        self.assertEqual(len(runner.rows(self.run_dir / "attempts.jsonl")), 1)

    def test_interruption_after_intent_without_outcome_cannot_be_retried(self):
        journal = self.journal()
        def interrupted(messages):
            raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            journal.process("BL-0001", interrupted)
        self.assertFalse((self.run_dir / "outcomes.jsonl").exists())
        with self.assertRaises(runner.AttemptUnknown):
            self.journal()

    def test_http_failure_keeps_reported_usage_without_exposing_error_text(self):
        class InventedHTTPFailure(Exception):
            failure_kind = "provider_http_error"
            reported_usage = {"prompt_tokens": 18}
            latency_seconds = 0.7
            fatal = False
        def rejected(messages):
            raise InventedHTTPFailure("DO_NOT_EXPORT_SECRET_EXAMPLE")
        journal = self.journal()
        self.assertFalse(journal.process("BL-0001", rejected))
        summary = self.journal().write_summary()
        self.assertEqual(summary["reported_tokens"]["prompt_tokens"], 18)
        self.assertIsNone(summary["reported_tokens"]["completion_tokens"])
        self.assertNotIn("DO_NOT_EXPORT_SECRET_EXAMPLE", (self.run_dir / "outcomes.jsonl").read_text())

    def test_ratings_and_failure_mirrors_recover_missing_or_shortened_prefixes(self):
        journal = self.journal()
        journal.process("BL-0001", lambda messages: response({"bad": "schema"}))
        journal.process("BL-0002", lambda messages: response({"bad": "schema"}))
        expected_ratings = runner.rows(self.run_dir / "ratings.jsonl")
        expected_failures = runner.rows(self.run_dir / "failures.jsonl")
        for name in ("ratings.jsonl", "failures.jsonl"):
            path = self.run_dir / name
            path.unlink()
            self.journal()
            restored = runner.rows(path)
            runner.atomic(path, restored[:1], rows=True)
            self.journal()
            self.assertEqual(runner.rows(path), restored)
        self.assertEqual(runner.rows(self.run_dir / "ratings.jsonl"), expected_ratings)
        self.assertEqual(runner.rows(self.run_dir / "failures.jsonl"), expected_failures)

    def test_contradictory_mirror_is_rejected_instead_of_overwritten(self):
        journal = self.journal()
        journal.process("BL-0001", lambda messages: response())
        changed = runner.rows(self.run_dir / "ratings.jsonl")
        changed[0]["assessment"]["R1"] = 1
        runner.atomic(self.run_dir / "ratings.jsonl", changed, rows=True)
        with self.assertRaises(ValueError):
            self.journal()

    def test_tampered_journal_and_incomplete_tail_fail_closed(self):
        journal = self.journal()
        journal.process("BL-0001", lambda messages: response())
        path = self.run_dir / "outcomes.jsonl"
        original = path.read_bytes()
        path.write_bytes(original[:-1])
        with self.assertRaises(ValueError):
            self.journal()
        value = json.loads(original)
        value["usage"]["prompt_tokens"] = 999
        path.write_text(json.dumps(value) + "\n")
        with self.assertRaises(ValueError):
            self.journal()

    def test_changed_model_identity_records_failure_and_stops_batch(self):
        journal = self.journal()
        value = response()
        value["provider_model"] = "unapproved-model"
        self.assertTrue(journal.process("BL-0001", lambda messages: value))
        self.assertEqual(runner.rows(self.run_dir / "ratings.jsonl")[0]["status"], "technical_failure")

    def test_recomputed_checksum_cannot_hide_internal_response_contradictions(self):
        journal = self.journal()
        journal.process("BL-0001", lambda messages: response())
        original = runner.rows(self.run_dir / "outcomes.jsonl")[0]
        for mutation in ("score", "model", "delivery", "kind", "rater", "technical"):
            value = deepcopy(original)
            if mutation == "score":
                value["rating"]["assessment"]["R1"] = 1
            elif mutation == "model":
                value["provider_model"] = "different-backend"
            elif mutation == "delivery":
                value["delivery_status"] = "response_rejected"
            elif mutation == "kind":
                value["rating"]["review_kind"] = "human_independent"
            elif mutation == "rater":
                value["rating"]["rater_id"] = "new-rater"
            else:
                value["rating"].update(status="technical_failure", assessment=None,
                                       reason="invalid_judgment_schema_or_evidence")
                value["failure_kind"] = "invalid_judgment_schema_or_evidence"
            value.pop("record_sha256")
            runner.atomic(self.run_dir / "outcomes.jsonl", [runner.signed(value)], rows=True)
            for name in ("ratings.jsonl", "failures.jsonl"):
                (self.run_dir / name).unlink(missing_ok=True)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.journal()
            self.assertFalse((self.run_dir / "ratings.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
