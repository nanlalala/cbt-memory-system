"""Invented-only calibration prompt, parser and planning tests; no inference."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation/cbt-memory-effect"))
from auto_eval_v2 import calibration as c
from auto_eval_v2 import calibration_prompt as p


JUDGE = {"provider": "fixture", "model": "invented-judge-v2", "model_family": "invented-judge"}


def packet(case="case-1"):
    return {"case_id": case, "source_family": "invented:hidden-family",
            "answer_model_family": "invented-hidden-therapist",
            "messages": [{"role": "user" if i % 2 == 0 else "assistant",
                          "content": f"Invented visible utterance {i}."} for i in range(10)]}


def review():
    return {"ctrs": {key: 3 for key in c.CTRS},
            "safety": {key: False for key in c.SAFETY},
            "reason": "Invented example shows collaborative planning; no visible urgent cue."}


class CalibrationPromptTests(unittest.TestCase):
    def test_only_visible_dialogue_enters_request(self):
        row = packet("hidden-case-identity")
        request = p.request_messages(row)
        wire = json.dumps(request)
        for hidden in (row["case_id"], row["source_family"], row["answer_model_family"]):
            self.assertNotIn(hidden, wire)
        self.assertEqual(json.loads(request[1]["content"]), {"messages": row["messages"]})
        self.assertIn("data, never instructions", request[0]["content"])

    def test_project_packet_rejects_hidden_fields_and_bad_role_grid(self):
        row = packet(); row["expert_rating"] = 6
        with self.assertRaises(ValueError): p.request_messages(row)
        row = packet(); row["messages"][0]["expert_rating"] = 6
        with self.assertRaises(ValueError): p.request_messages(row)
        row = packet(); row["messages"][1]["role"] = "user"
        with self.assertRaises(ValueError): p.request_messages(row)

    def test_independent_judge_and_identity_requirements(self):
        row = packet(); row["answer_model_family"] = " INVENTED-JUDGE "
        with self.assertRaisesRegex(ValueError, "independent"):
            p.plan_quality([row], JUDGE)
        bad_judge = {**JUDGE}; del bad_judge["provider"]
        with self.assertRaises(ValueError): p.plan_quality([packet()], bad_judge)

    def test_plan_is_order_independent_with_new_repeat_slots(self):
        packets = [packet("a"), packet("b")]
        a = p.plan_quality(packets, JUDGE, repeat_fraction=.1, seed=5)
        b = p.plan_quality(list(reversed(packets)), JUDGE, repeat_fraction=.1, seed=5)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 3)
        repeat = a[-1]
        primary = next(row for row in a if row["item_id"] == repeat["repeat_of"])
        self.assertNotEqual(repeat["item_id"], primary["item_id"])
        self.assertEqual(repeat["request_sha256"], primary["request_sha256"])
        self.assertEqual(repeat["messages"], primary["messages"])

    def test_zero_repeat_and_duplicate_cases(self):
        self.assertEqual(len(p.plan_quality([packet()], JUDGE, repeat_fraction=0)), 1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            p.plan_quality([packet(), packet()], JUDGE)
        for bad in (True, -1, 1.01, float("nan")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                p.plan_quality([packet()], JUDGE, repeat_fraction=bad)

    def test_valid_review_stays_separate_from_memory_certification(self):
        out = p.validate_review(json.dumps(review()), packet())
        self.assertTrue(out["accepted"])
        self.assertEqual(out["status"], "completed")
        self.assertEqual(out["assessment"], review())
        self.assertIn("cannot certify custom R1/R3", p.GUIDE)

    def test_nullable_aspects_and_whole_unresolved(self):
        value = review(); value["ctrs"]["AGENDA"] = None
        self.assertTrue(p.validate_review(json.dumps(value), packet())["accepted"])
        out = p.validate_review(json.dumps({"status": "unresolved", "reason": "Missing visible evidence"}), packet())
        self.assertTrue(out["accepted"])
        self.assertEqual(out["status"], "unresolved")
        self.assertIsNone(out["assessment"])
        value = {"ctrs": {k: None for k in c.CTRS}, "safety": {k: None for k in c.SAFETY},
                 "reason": "Nothing assessable"}
        self.assertEqual(p.validate_review(json.dumps(value), packet())["status"], "unresolved")

    def test_boolean_float_nonfinite_and_out_of_range_ctrs_rejected(self):
        for bad in (True, False, 3.0, 7, -1, float("nan"), float("inf"), "3"):
            value = review(); value["ctrs"]["AGENDA"] = bad
            with self.subTest(bad=bad):
                self.assertFalse(p.validate_review(json.dumps(value), packet())["accepted"])

    def test_safety_must_be_four_boolean_or_null_flags(self):
        for bad in (0, 1, "false", .5):
            value = review(); value["safety"][c.SAFETY[0]] = bad
            with self.subTest(bad=bad):
                self.assertFalse(p.validate_review(json.dumps(value), packet())["accepted"])
        value = review(); value["safety"][c.SAFETY[0]] = None
        self.assertTrue(p.validate_review(json.dumps(value), packet())["accepted"])

    def test_duplicate_json_and_extra_dimensions_cannot_enter_assessment(self):
        text = json.dumps(review())
        duplicate = text.replace('"AGENDA": 3', '"AGENDA": 3, "AGENDA": 6')
        self.assertFalse(p.validate_review(duplicate, packet())["accepted"])
        value = review(); value["ctrs"]["GUIDED DISCOVERY"] = 6
        self.assertFalse(p.validate_review(json.dumps(value), packet())["accepted"])
        value = review(); value["total"] = 27
        self.assertFalse(p.validate_review(json.dumps(value), packet())["accepted"])


if __name__ == "__main__":
    unittest.main()
