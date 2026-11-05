"""Pure response fixtures, never network calls or fake Colab indicators."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("followup_transport_fixture", Path(__file__).resolve().parents[1] / "evaluation/dialogue-benefit/followup_transport.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class ResponseUsageTests(unittest.TestCase):
    def test_truncated_received_response_retains_all_numeric_billable_usage(self):
        payload = {"model": "fixture-allowed", "usage": {"prompt_tokens": 13, "completion_tokens": 9, "total_tokens": 22, "secret": "not exported"},
                   "choices": [{"finish_reason": "length", "message": {"content": "unfinished"}}]}
        with self.assertRaises(MOD.ResponseFailure) as caught:
            MOD.parse_received(payload, allowed_model="fixture-allowed", latency=0.1)
        self.assertEqual(caught.exception.reported_usage, {"prompt_tokens": 13, "completion_tokens": 9, "total_tokens": 22})
        self.assertFalse(caught.exception.fatal)

    def test_wrong_model_halts_but_keeps_known_usage(self):
        payload = {"model": "unapproved", "usage": {"total_tokens": 17}, "choices": []}
        with self.assertRaises(MOD.ResponseFailure) as caught:
            MOD.parse_received(payload, allowed_model="fixture-allowed", latency=0.1)
        self.assertTrue(caught.exception.fatal)
        self.assertEqual(caught.exception.reported_usage, {"total_tokens": 17})

    def test_missing_usage_stays_missing_and_bool_is_not_counted(self):
        payload = {"model": "fixture-allowed", "usage": {"total_tokens": True}, "choices": [{"finish_reason": "stop", "message": {"content": "fixture"}}]}
        self.assertEqual(MOD.parse_received(payload, allowed_model="fixture-allowed", latency=0.1)["usage"], {})


if __name__ == "__main__": unittest.main()
