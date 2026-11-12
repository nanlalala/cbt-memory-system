"""Invented text only: schema, evidence, blinding and position invariance."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parents[1] / "evaluation/cbt-memory-effect"))
from auto_eval_v2 import quality as q


def packet(case="invented-case"):
    return {"case_id": case, "source_family": "invented:one", "answer_model_family": "invented-answer",
        "history": [{"id": "u1", "role": "user", "content": "I paused the diary."},
                    {"id": "a2", "role": "assistant", "content": "You could try a brief walk."}],
        "input": "I still want to pause. What next?", "response": "We can leave the diary paused. What feels manageable?"}


JUDGE = {"provider": "fixture", "model": "invented-judge-v1", "model_family": "invented-judge"}


def review():
    return {"R1": 5, "R2": 4, "R3": 5, "R4": 4, "R5": "pass", "evidence": [
        {"dimension": d, "source_id": "u1", "source_role": "user", "source_quote": "paused the diary",
         "response_quote": "leave the diary paused", "reason": "Respects visible pause."}
        for d in q.DIMENSIONS]}


class QualityContractTests(unittest.TestCase):
    def test_gold_and_metadata_never_reach_judge(self):
        p = packet()
        msg = q.request_messages(p)
        visible = json.loads(msg[1]["content"])
        self.assertEqual(set(visible), {"history", "input", "response"})
        for hidden in (p["case_id"], p["source_family"], p["answer_model_family"]):
            self.assertNotIn(hidden, msg[1]["content"])
        p["future_therapist_answer"] = "Forged future content"
        with self.assertRaises(ValueError):
            q.request_messages(p)

    def test_source_role_and_quote_bindings(self):
        p, r = packet(), review()
        self.assertTrue(q.validate_review(json.dumps(r), p)["accepted"])
        for key, value in (("source_id", "a2"), ("source_role", "assistant"),
                           ("source_quote", "I completed the diary"),
                           ("response_quote", "You must restart")):
            bad = copy.deepcopy(r)
            bad["evidence"][0][key] = value
            self.assertFalse(q.validate_review(json.dumps(bad), p)["accepted"])
        r["evidence"][0].update(source_id="CURRENT", source_role="user", source_quote="want to pause")
        self.assertTrue(q.validate_review(json.dumps(r), p)["accepted"])

    def test_unsafe_schema_is_not_repaired_or_coerced(self):
        p, r = packet(), review()
        for invalid in (True, 5.0, "5", 0, 6, None):
            bad = {**r, "R1": invalid}
            self.assertFalse(q.validate_review(json.dumps(bad), p)["accepted"])
        text = json.dumps(r)
        self.assertFalse(q.validate_review(text.replace('"R1": 5', '"R1": 1, "R1": 5'), p)["accepted"])
        self.assertFalse(q.validate_review(text.replace('"R1": 5', '"R1": NaN'), p)["accepted"])
        self.assertFalse(q.validate_review(text + " junk", p)["accepted"])
        self.assertEqual(q.validate_review('{"status":"unresolved","reason":"missing context"}', p)["status"], "unresolved")

    def test_role_duplicates_and_reserved_current_are_rejected(self):
        for mutator in (lambda p: p["history"].append(p["history"][0]),
                        lambda p: p["history"][0].update(id="CURRENT"),
                        lambda p: p["history"][0].update(role="system")):
            p = packet(); mutator(p)
            with self.assertRaises(ValueError):
                q.validate_packet(p)

    def test_independence_repeats_and_identity_reuse(self):
        packets = [packet("invented-" + str(i)) for i in range(11)]
        plan = q.plan_quality(packets, JUDGE, repeat_fraction=.1, seed=2)
        self.assertEqual(len(plan), 13)
        self.assertEqual(plan, q.plan_quality(list(reversed(packets)), JUDGE, repeat_fraction=.1, seed=2))
        self.assertEqual(len({x["request_sha256"] for x in plan}), 1)
        self.assertEqual(len({x["item_id"] for x in plan}), len(plan))
        self.assertEqual(sum(x["repeat_of"] is not None for x in plan), 2)
        with self.assertRaises(ValueError):
            q.plan_quality([packet()], {**JUDGE, "model_family": " INVENTED-ANSWER "})
        for fraction in (True, float("nan"), -.1, 1.1):
            with self.assertRaises(ValueError):
                q.plan_quality([packet()], JUDGE, repeat_fraction=fraction)

    def test_pairwise_order_mapping_and_ties(self):
        first, second = packet("first"), packet("second")
        second["response"] = "Let's plan a new diary now."
        ab, ba = q.plan_pairwise(first, second, JUDGE)
        self.assertEqual(json.loads(ab["messages"][1]["content"])["history"], first["history"])
        self.assertEqual(q.validate_pairwise('{"winner":"A","reason":"preferred"}', ab)["winner"], "first")
        self.assertEqual(q.validate_pairwise('{"winner":"B","reason":"preferred"}', ba)["winner"], "first")
        self.assertEqual(q.validate_pairwise('{"winner":"tie","reason":"equal"}', ba)["winner"], "tie")
        # A-position preference in both orders is retained as a disagreement.
        self.assertNotEqual(q.validate_pairwise('{"winner":"A","reason":"preferred"}', ab)["winner"],
                            q.validate_pairwise('{"winner":"A","reason":"preferred"}', ba)["winner"])
        second["history"].reverse()
        with self.assertRaises(ValueError):
            q.plan_pairwise(first, second, JUDGE)


if __name__ == "__main__":
    unittest.main()
