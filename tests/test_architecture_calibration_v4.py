"""Invented-only packet boundaries for the published-bank calibration adapter."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location("architecture_calibration_runner",
    ROOT / "evaluation/architecture-calibration/run_calibration.py")
adapter = module_from_spec(spec)
spec.loader.exec_module(adapter)


def invented(case, model="DEEPSEEK"):
    return {"id": case, "therapist_model": model, "patient_profile": "fixture-profile",
            "hidden_profile": "must never enter judge input",
            "expert_rating": {"private_fixture_label": "must never enter judge input"},
            "conversation": {"messages": [
                {"role": "user" if i % 2 == 0 else "assistant", "content": f"invented turn {i}",
                 "name": "speaker", "extra": "not projected"}
                for i in range(10)]}}


class CalibrationProjectionTests(unittest.TestCase):
    def test_visible_only_and_cross_family_exclusion(self):
        packets, excluded = adapter.project_packets([
            invented("source-one"), invented("same-family", "QWEN3_4B_LOCAL")])
        self.assertEqual(len(packets), 1)
        self.assertEqual(excluded, [{"case_id": "same-family", "reason": "same_therapist_and_judge_model_family"}])
        packet = packets[0]
        self.assertEqual(packet["source_family"], "published-profile:fixture-profile")
        self.assertEqual(packet["answer_model_family"], "deepseek")
        self.assertTrue(all(set(m) == {"role", "content"} for m in packet["messages"]))
        self.assertNotIn("must never", str(packet))

    def test_duplicate_or_unknown_model_fails_before_judging(self):
        with self.assertRaises(ValueError):
            adapter.project_packets([invented("duplicate"), invented("duplicate")])
        with self.assertRaises(ValueError):
            adapter.project_packets([invented("bad-model", "UNDECLARED")])

    def test_exact_ten_turn_source_contract_is_preserved(self):
        record = invented("short")
        record["conversation"]["messages"].pop()
        with self.assertRaises(ValueError):
            adapter.project_packets([record])


if __name__ == "__main__":
    unittest.main()
