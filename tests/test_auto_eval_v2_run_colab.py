"""Invented fixtures only: durable sends, frozen inputs and complete denominators."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation/cbt-memory-effect"))
from auto_eval_v2 import run_colab as runner

JUDGE = {"provider": "fixture", "model": "invented-model", "model_family": "invented-judge"}


def packet(case="raw"):
    return {"case_id": case, "source_family": "invented:family", "answer_model_family": "invented-answer",
            "history": [{"id": "h1", "role": "user", "content": "I paused that exercise."}],
            "input": "What could I try?", "response": "We can choose a smaller step together."}


def review():
    return {"R1": 4, "R2": 4, "R3": 4, "R4": 4, "R5": "pass", "evidence": [
        {"dimension": d, "source_id": "h1", "source_role": "user",
         "source_quote": "paused", "response_quote": "smaller step", "reason": "Accounts for the pause."}
        for d in ("R1", "R2", "R3", "R4", "R5")]}


class Transport:
    identity = JUDGE

    def __init__(self, result=None, *, error=False, mutate=False):
        self.result = review() if result is None else result
        self.calls, self.error, self.mutate = 0, error, mutate

    def __call__(self, payload):
        self.calls += 1
        if self.error:
            raise TimeoutError("do not record this exception body")
        if self.mutate:
            payload["messages"][0]["content"] = "mutated instruction"
        return {"text": json.dumps(self.result) if not isinstance(self.result, str) else self.result,
                "provider_model": JUDGE["model"],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name) / "new-run"

    def tearDown(self):
        self.tmp.cleanup()

    def prepare(self, packets=None, **kw):
        opts = {"fixture_only": True, "maximum_attempts": 10, "repeat_fraction": 0,
                "authorization_reference": "invented fixture test, no model"}
        opts.update(kw)
        return runner.prepare(packets or [packet()], JUDGE, self.folder, **opts)

    def run_fixture(self, transport, max_new_calls=10):
        return runner.run(self.folder, transport, max_new_calls=max_new_calls, fixture_only=True)

    def test_default_zero_attempt_cap_and_zero_invocation_limit(self):
        self.prepare(maximum_attempts=0)
        t = Transport()
        self.assertEqual(self.run_fixture(t)["actual_attempts"], 0)
        self.assertEqual(t.calls, 0)

    def test_completed_result_reuses_receipt_and_scores_are_bound(self):
        self.prepare()
        t = Transport()
        first, second = self.run_fixture(t), self.run_fixture(t)
        self.assertEqual(t.calls, 1)
        self.assertEqual(first["scored_unique_reviews"], 1)
        self.assertEqual(second["tokens_reported"]["total_tokens"], 15)
        self.assertEqual(second["items"][0]["review"]["assessment"]["R1"], 4)

    def test_invalid_json_is_retained_and_never_retried(self):
        self.prepare()
        t = Transport("invalid JSON")
        self.run_fixture(t)
        s = self.run_fixture(t)
        self.assertEqual(t.calls, 1)
        self.assertEqual(s["delivery_states"], {"invalid": 1})
        self.assertEqual(s["tokens_reported"]["total_tokens"], 15)

    def test_unknown_failure_blocks_all_further_calls(self):
        self.prepare([packet("one"), {**packet("two"), "response": "Another answer."}])
        failed = Transport(error=True)
        with self.assertRaises(runner.UnknownAttempt):
            self.run_fixture(failed)
        good = Transport()
        with self.assertRaises(runner.UnknownAttempt):
            self.run_fixture(good)
        self.assertEqual(failed.calls, 1)
        self.assertEqual(good.calls, 0)
        self.assertEqual(runner.summarize(self.folder, fixture_only=True)["unknown_attempts"], 1)

    def test_interrupted_intent_without_outcome_blocks(self):
        self.prepare()
        proto, _, plan = runner._load(self.folder, True)
        row = plan[0]
        runner._append(self.folder / "intents.jsonl", {
            "canonical_key": row["canonical_key"], "protocol_sha256": proto["protocol_sha256"],
            "payload_sha256": runner.digest(row["payload"])})
        good = Transport()
        with self.assertRaises(runner.UnknownAttempt):
            self.run_fixture(good)
        self.assertEqual(good.calls, 0)

    def test_identical_primary_answers_share_but_repeat_is_another_attempt(self):
        self.prepare([packet("one"), packet("two")], repeat_fraction=1)
        t = Transport()
        s = self.run_fixture(t)
        self.assertEqual(s["planned_items"], 4)
        self.assertEqual(s["actual_attempts"], 3)
        self.assertEqual(t.calls, 3)
        self.assertEqual(s["tokens_reported"]["total_tokens"], 45)

    def test_unresolved_is_received_not_a_completed_score(self):
        self.prepare()
        s = self.run_fixture(Transport({"status": "unresolved", "reason": "Not enough evidence."}))
        self.assertEqual(s["delivery_states"], {"received": 1})
        self.assertEqual(s["scored_unique_reviews"], 0)
        self.assertEqual(s["unresolved_unique_reviews"], 1)

    def test_foreign_or_duplicate_receipt_rejected_by_summary(self):
        self.prepare()
        self.run_fixture(Transport())
        rows = runner._rows(self.folder / "outcomes.jsonl")
        row = {k: v for k, v in rows[0].items() if k != "record_sha256"}
        runner._append(self.folder / "outcomes.jsonl", row)
        with self.assertRaises(ValueError):
            runner.summarize(self.folder, fixture_only=True)

    def test_orphan_foreign_protocol_receipt_rejected_by_summary(self):
        self.prepare()
        proto, _, plan = runner._load(self.folder, True)
        row = plan[0]
        runner._append(self.folder / "outcomes.jsonl", {
            "canonical_key": row["canonical_key"], "protocol_sha256": "foreign",
            "payload_sha256": runner.digest(row["payload"]), "status": "invalid"})
        with self.assertRaises(ValueError):
            runner.summarize(self.folder, fixture_only=True)

    def test_callback_mutation_cannot_change_frozen_plan_or_gain_score(self):
        self.prepare()
        before = (self.folder / "plan.json").read_bytes()
        s = self.run_fixture(Transport(mutate=True))
        self.assertEqual(before, (self.folder / "plan.json").read_bytes())
        self.assertEqual(s["delivery_states"], {"invalid": 1})
        self.assertEqual(s["scored_unique_reviews"], 0)

    def test_native_score_binding_rechecked_on_resume(self):
        self.prepare()
        self.run_fixture(Transport())
        row = runner._rows(self.folder / "outcomes.jsonl")[0]
        row.pop("record_sha256")
        row["review"]["assessment"]["R1"] = 5
        (self.folder / "outcomes.jsonl").unlink()
        runner._append(self.folder / "outcomes.jsonl", row)
        with self.assertRaises(ValueError):
            runner.summarize(self.folder, fixture_only=True)

    def test_invalid_delivery_cannot_bypass_native_binding_with_completed_scores(self):
        self.prepare()
        self.run_fixture(Transport())
        row = runner._rows(self.folder / "outcomes.jsonl")[0]
        row.pop("record_sha256")
        row["status"] = "invalid"
        row["review"]["assessment"]["R1"] = 5
        (self.folder / "outcomes.jsonl").unlink()
        runner._append(self.folder / "outcomes.jsonl", row)
        with self.assertRaises(ValueError):
            runner.summarize(self.folder, fixture_only=True)

    def test_missing_generation_remains_in_frozen_pair_denominator(self):
        comparisons = [{"case_id": "q1", "source_family": "invented:family",
                        "raw_case_id": "raw", "managed_case_id": None}]
        self.prepare(comparisons=comparisons)
        s = self.run_fixture(Transport())
        dim = s["paired_quality"]["dimensions"]["R1"]
        self.assertEqual(dim["planned_pairs"], 1)
        self.assertEqual(dim["scored_pairs"], 0)
        self.assertEqual(dim["missing_or_unresolved_pairs"], 1)

    def test_real_run_guard_precedes_source_reads(self):
        with patch.object(runner, "require_hosted", side_effect=RuntimeError("guard")):
            with patch.object(Path, "read_text", side_effect=AssertionError("source opened")):
                with self.assertRaisesRegex(RuntimeError, "guard"):
                    runner.run(self.folder, Transport())

    def test_changed_history_across_comparison_arms_rejected(self):
        p = packet("managed")
        p["history"] = []
        comparisons = [{"case_id": "q1", "source_family": "invented:family",
                        "raw_case_id": "raw", "managed_case_id": "managed"}]
        with self.assertRaises(ValueError):
            self.prepare([packet(), p], comparisons=comparisons)

    def test_paired_quality_and_repeat_consistency_are_separate(self):
        comparisons = [{"case_id": "q1", "source_family": "invented:family",
                        "raw_case_id": "raw", "managed_case_id": "managed"},
                       {"case_id": "q2", "source_family": "invented:missing",
                        "raw_case_id": None, "managed_case_id": None}]
        self.prepare([packet(), packet("managed")], comparisons=comparisons, repeat_fraction=1)
        s = self.run_fixture(Transport())
        d = s["paired_quality"]["dimensions"]["R1"]
        self.assertEqual(d["planned_pairs"], 2)
        self.assertEqual(d["scored_pairs"], 1)
        self.assertEqual(d["ties"], 1)
        self.assertEqual(s["actual_attempts"], 3)
        self.assertEqual(s["repeat_consistency"]["completed_repeat_pairs"], 2)
        self.assertIsNone(d["family_cluster_difference"]["ci95"])

    def test_calibration_uses_same_journal_and_scores_only_after_predictions(self):
        from auto_eval_v2.calibration import CTRS, SAFETY
        cp = {"case_id": "cal1", "source_family": "invented:family", "answer_model_family": "invented-answer",
              "messages": [{"role": "user" if i % 2 == 0 else "assistant", "content": "Invented text."}
                           for i in range(10)]}
        self.prepare([cp], task="calibration")
        g = [{"case_id": "cal1", "ctrs": {d: 3 for d in CTRS}, "safety": {d: False for d in SAFETY}}]
        with self.assertRaises(ValueError):
            runner.score_calibration(self.folder, g, fixture_only=True)
        result = {"ctrs": g[0]["ctrs"], "safety": g[0]["safety"], "reason": "Invented fixture."}
        s = self.run_fixture(Transport(result))
        self.assertEqual(s["task"], "calibration")
        self.assertEqual(s["scored_unique_reviews"], 1)
        scored = runner.score_calibration(self.folder, g, fixture_only=True)
        self.assertEqual(scored["calibration"]["ctrs"]["AGENDA"]["mae"], 0)


if __name__ == "__main__":
    unittest.main()
