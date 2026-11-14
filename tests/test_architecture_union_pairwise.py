"""Invented-data checks for the blinded, order-balanced pairwise judge.

No test reads a credential or calls a hosted provider. The source adapter is
mocked, while the real exact-wire Journal, parser and statistics are exercised.
"""
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation/cbt-memory-effect"))
from architecture_retest.journal import UnknownAttempt, rows
from architecture_union import pairwise


ARMS = ("raw_history_rag", "managed_memory_rag", "L1_raw_plus_items_compact",
        "L3_plus_fact_key", "L4_plus_pinned_profile")


def source_fixture():
    """Three fictional cases, with two cases sharing one source family."""
    packets, mapping = [], []
    for case_no in range(3):
        family = "fictional-family-" + str(case_no // 2)
        history = [
            {"id": "s1", "role": "user", "content": "FICTIONAL case %d: I avoid the bus." % case_no},
            {"id": "s2", "role": "assistant", "content": "FICTIONAL: Which part feels difficult?"},
            {"id": "s3", "role": "user", "content": "FICTIONAL: The crowded doorway worries me."}]
        for arm_no, arm in enumerate(ARMS):
            packet_id = "opaque-%d-%d" % (case_no, arm_no)
            packets.append({"case_id": packet_id, "source_family": family,
                            "answer_model_family": "deepseek", "history": history,
                            "input": "FICTIONAL current input %d: What could I try next?" % case_no,
                            "response": "FICTIONAL response option %d for situation %d." % (arm_no, case_no)})
            mapping.append({"case_id": "case-%d" % case_no, "source_family": family,
                            "arm": arm, "packet_id": packet_id, "status": "received"})
    return packets, mapping, {"protocol_sha256": "f" * 64}


def review(q1="tie", q2="tie", q3="tie", reason="Both responses are comparable."):
    return {"Q1": q1, "Q2": q2, "Q3": q3, "reason": reason}


def vote_for(value, candidate_is_a):
    if value == 0:
        return "tie"
    return "A" if (value > 0) == candidate_is_a else "B"


class FixtureTransport:
    fixture_only = True
    identity = pairwise.JUDGE_IDENTITY

    def __init__(self, replies=None, failure=False):
        self.replies, self.failure, self.sent = replies or {}, failure, []

    def __call__(self, payload):
        self.sent.append(payload)
        if self.failure:
            raise RuntimeError("invented provider transport failure")
        wire = json.dumps(payload, allow_nan=False)
        value = self.replies.get(sha256(wire.encode()).hexdigest(), review())
        text = value if isinstance(value, str) else json.dumps(value)
        return {"text": text, "provider_model": self.identity["returned_model"],
                "finish_reason": "stop", "wire_sha256": sha256(wire.encode()).hexdigest(),
                "usage": {"total_tokens": 1}}


class PairwiseJudge(unittest.TestCase):
    def prepare(self, root):
        folder = root / "new-pairwise"
        with patch.object(pairwise.union_run, "quality_packets", return_value=source_fixture()) as adapter:
            pairwise.prepare(root / "answered-pilot", folder, fixture_only=True)
        self.assertEqual(adapter.call_count, 1)
        self.assertEqual(adapter.call_args.args[0], root / "answered-pilot")
        return folder, json.loads((folder / "plan.json").read_text())

    def test_requests_are_blinded_and_keep_full_source_roles_and_both_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, plan = self.prepare(Path(tmp))
            self.assertEqual(len(plan), 18)
            self.assertEqual(len(json.loads((folder / "pairs.json").read_text())), 9)
            all_source = source_fixture()[0]
            for item in plan:
                payload = item["payload"]
                visible = "\n".join(message["content"] for message in payload["messages"])
                for forbidden in (*ARMS, "记忆", "RAG", "memory", "candidate_is_a", "source_family"):
                    self.assertNotIn(forbidden.lower(), visible.lower())
                self.assertEqual(payload["temperature"], 0)
                self.assertEqual(payload["thinking"], {"type": "disabled"})
                self.assertEqual(payload["max_tokens"], 1024)
                case_no = int(item["case_id"].split("-")[-1])
                packet = all_source[case_no * 5]
                for turn in packet["history"]:
                    self.assertIn(turn["content"], visible)
                    self.assertRegex(visible, r'"role"\s*:\s*"' + turn["role"] + '"')
                self.assertIn(packet["input"], visible)
                self.assertIn(packet["response"], visible)
                other_no = ARMS.index(item["arm"])
                self.assertIn(all_source[case_no * 5 + other_no]["response"], visible)
                self.assertEqual(json.loads(item["serialized_request"]), payload)
                self.assertEqual(sha256(item["serialized_request"].encode()).hexdigest(), item["wire_sha256"])

    def test_forward_reverse_orientation_and_disagreement_conversion(self):
        self.assertEqual(pairwise.orient_vote("A", True), 1)
        self.assertEqual(pairwise.orient_vote("B", True), -1)
        self.assertEqual(pairwise.orient_vote("A", False), -1)
        self.assertEqual(pairwise.orient_vote("B", False), 1)
        self.assertEqual(pairwise.orient_vote("tie", True), 0)
        for first_is_a in (True, False):
            first = pairwise.validate_review(json.dumps(review(*[vote_for(x, first_is_a) for x in (1, -1, 0)])))
            second = pairwise.validate_review(json.dumps(review(*[vote_for(x, not first_is_a) for x in (1, -1, 0)])))
            agreed = pairwise.consensus(first, second, first_is_a)
            self.assertEqual(agreed["status"], "completed")
            self.assertEqual(agreed["assessment"], {"Q1": 1, "Q2": -1, "Q3": 0})
            self.assertEqual(agreed["order_disagreement"], {"Q1": False, "Q2": False, "Q3": False})
        first = pairwise.validate_review(json.dumps(review("A", "A", "tie")))
        second = pairwise.validate_review(json.dumps(review("A", "tie", "tie")))
        disagreed = pairwise.consensus(first, second, True)
        self.assertEqual(disagreed["assessment"], {"Q1": 0, "Q2": 0, "Q3": 0})
        self.assertEqual(disagreed["order_disagreement"], {"Q1": True, "Q2": True, "Q3": False})
        self.assertEqual(pairwise.consensus(first, {"status": "invalid"}, True)["status"], "invalid")

    def test_strict_json_rejects_invalid_outputs_without_repair(self):
        good = json.dumps(review())
        self.assertEqual(pairwise.validate_review(good)["status"], "completed")
        bad = ["```json\n" + good + "\n```", "not json", "[]",
               json.dumps({**review(), "extra": 1}), json.dumps({"Q1": "A", "Q2": "B", "reason": "x"}),
               json.dumps(review("a")), json.dumps(review(reason="")), json.dumps(review(reason="One. Two. Three.")),
               '{"Q1":"A","Q1":"B","Q2":"tie","Q3":"tie","reason":"x"}',
               json.dumps(review()).replace('"tie"', 'NaN', 1)]
        for text in bad:
            with self.subTest(text=text):
                self.assertEqual(pairwise.validate_review(text)["status"], "invalid")

    def test_prepare_order_is_deterministic_and_exactly_reversed(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            folder, plan = self.prepare(Path(a))
            _, other = self.prepare(Path(b))
            self.assertEqual(plan, other)
            for pair in json.loads((folder / "pairs.json").read_text()):
                ordered = sorted([r for r in plan if r["pair_id"] == pair["pair_id"]], key=lambda r: r["order_index"])
                self.assertEqual([r["order_index"] for r in ordered], [0, 1])
                self.assertEqual(ordered[0]["candidate_is_a"], pair["first_candidate_is_a"])
                self.assertNotEqual(ordered[0]["candidate_is_a"], ordered[1]["candidate_is_a"])

    def test_cap_resume_and_finished_rerun_send_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, plan = self.prepare(Path(tmp))
            first = FixtureTransport()
            partial = pairwise.run(folder, first, max_new_calls=2, fixture_only=True)
            self.assertEqual(len(first.sent), 2)
            self.assertEqual(partial["reviewed"], 2)
            last = json.loads((folder / "last_invocation.json").read_text())
            self.assertEqual(last["new_calls"], 2)
            remaining = FixtureTransport()
            final = pairwise.run(folder, remaining, max_new_calls=100, fixture_only=True)
            self.assertEqual(len(first.sent) + len(remaining.sent), len(plan))
            self.assertEqual(final["review_status"], {"completed": len(plan)})
            self.assertEqual(final["valid_fraction"], 1.0)
            again = FixtureTransport()
            pairwise.run(folder, again, max_new_calls=0, fixture_only=True)
            self.assertEqual(again.sent, [])
            self.assertEqual(len(rows(folder / "intents.jsonl")), len(plan))
            self.assertEqual(len(rows(folder / "outcomes.jsonl")), len(plan))

    def test_manual_statistics_invalid_pair_and_order_disagreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, plan = self.prepare(Path(tmp))
            scripted = {}
            for item in plan:
                case_no = int(item["case_id"].split("-")[-1])
                value = 1 if case_no == 0 else -1 if case_no == 1 else 0
                if item["arm"] == "managed_memory_rag":
                    value = 1 if case_no < 2 else -1
                if item["arm"] == "L1_raw_plus_items_compact" and case_no == 2:
                    value = 1 if item["order_index"] == 0 else -1
                text = review(vote_for(value, item["candidate_is_a"]), "tie", vote_for(value, item["candidate_is_a"]))
                if item["arm"] == "L4_plus_pinned_profile" and case_no == 2 and item["order_index"] == 1:
                    text = "invalid fictional output"
                scripted[item["wire_sha256"]] = text
            transport = FixtureTransport(scripted)
            result = pairwise.run(folder, transport, max_new_calls=100, fixture_only=True)
            self.assertEqual(len(transport.sent), len(plan))
            self.assertEqual(result["review_status"], {"completed": 17, "invalid": 1})
            self.assertAlmostEqual(result["valid_fraction"], 17 / 18)
            for arm in ("L1_raw_plus_items_compact", "managed_memory_rag"):
                for dim in ("Q1", "Q3"):
                    stat = result["comparisons"][arm][dim]
                    expected = (3, 1, 1, 1) if arm == "L1_raw_plus_items_compact" else (3, 2, 0, 1)
                    self.assertEqual((stat["valid_pairs"], stat["wins"], stat["ties"], stat["losses"]), expected)
                    self.assertEqual(stat["net_win_rate"], 0 if arm == "L1_raw_plus_items_compact" else 1 / 3)
                    self.assertEqual(stat["source_families"], 2)
                    self.assertEqual(stat["ci95"], [0.0, 0.0] if arm == "L1_raw_plus_items_compact" else [-1.0, 1.0])
                q2 = result["comparisons"][arm]["Q2"]
                self.assertEqual((q2["wins"], q2["ties"], q2["losses"], q2["net_win_rate"]), (0, 3, 0, 0))
            self.assertEqual(result["comparisons"]["L1_raw_plus_items_compact"]["Q1"]["order_disagreements"], 1)
            self.assertEqual(result["comparisons"]["L1_raw_plus_items_compact"]["Q2"]["order_disagreements"], 0)
            for dim in ("Q1", "Q2", "Q3"):
                stat = result["comparisons"]["L4_plus_pinned_profile"][dim]
                self.assertEqual((stat["valid_pairs"], stat["invalid_pairs"]), (2, 1))
            again = FixtureTransport()
            pairwise.run(folder, again, max_new_calls=100, fixture_only=True)
            self.assertEqual(again.sent, [])  # invalid assessment is terminal, never repaired or retried

    def test_unknown_send_stops_and_is_never_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder, _ = self.prepare(Path(tmp))
            broken = FixtureTransport(failure=True)
            with self.assertRaises(UnknownAttempt):
                pairwise.run(folder, broken, max_new_calls=20, fixture_only=True)
            self.assertEqual(len(broken.sent), 1)
            self.assertEqual([r["status"] for r in rows(folder / "outcomes.jsonl")], ["unknown"])
            again = FixtureTransport()
            with self.assertRaises(UnknownAttempt):
                pairwise.run(folder, again, max_new_calls=20, fixture_only=True)
            self.assertEqual(again.sent, [])


if __name__ == "__main__":
    unittest.main()
