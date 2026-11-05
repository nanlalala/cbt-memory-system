"""Small invented records only; no runtime-guard spoofing, models or datasets."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("followup_journal_under_test", ROOT / "evaluation/dialogue-benefit/run_followup_pipeline.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.protocol = {"protocol_sha256": "invented-fixture", "maximum_provider_attempts": 3, "code_sha256": MOD.code_hashes()}
        self.messages = [{"role": "user", "content": "Invented small fixture."}]

    def response(self, messages, purpose):
        self.assertEqual(len(MOD.rows(self.run / "attempts.jsonl")), 1)
        return {"hypothesis": "fixture answer", "provider_model": MOD.ALIAS, "finish_reason": "stop", "latency_seconds": 0.1,
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}

    def test_intent_before_call_and_exact_prompt_dedup_survives_resume(self):
        journal = MOD.Journal(self.run, self.protocol, max_new_calls=1, transport=self.response)
        key, result = journal.call(self.messages, "response")
        self.assertEqual(journal.call(self.messages, "response"), (key, result))
        loaded = MOD.Journal(self.run, self.protocol, max_new_calls=1, transport=lambda *_: self.fail("duplicate external call"))
        self.assertEqual(loaded.call(self.messages, "response"), (key, result))
        self.assertEqual(loaded.cost()["reported_tokens"]["total_tokens"], 10)

    def test_failure_redacts_error_and_preserves_unknown_cost(self):
        def failed(*_): raise RuntimeError("SECRET_SENTINEL must never persist")
        journal = MOD.Journal(self.run, self.protocol, max_new_calls=1, transport=failed)
        _, result = journal.call(self.messages, "response")
        self.assertEqual(result["status"], "provider_failure")
        self.assertTrue(result["billable_usage_unknown"])
        self.assertEqual(journal.cost()["failed_attempts_without_usage"], 1)
        self.assertNotIn("SECRET_SENTINEL", "".join(p.read_text() for p in self.run.iterdir()))

    def test_supplied_failure_usage_is_preserved_and_not_assumed_free(self):
        class KnownUsageFailure(RuntimeError):
            reported_usage = {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}
            latency_seconds = 0.2
        def failed(*_): raise KnownUsageFailure("do not store provider text")
        journal = MOD.Journal(self.run, self.protocol, max_new_calls=1, transport=failed)
        _, result = journal.call(self.messages, "response")
        self.assertEqual(result["reported_usage"]["total_tokens"], 15)
        self.assertFalse(result["billable_usage_unknown"])
        self.assertEqual(journal.cost()["reported_tokens"]["total_tokens"], 15)
        self.assertEqual(journal.cost()["failed_attempts_without_usage"], 0)

    def test_interrupted_intent_blocks_resume_and_does_not_spend(self):
        def interrupted(*_): raise KeyboardInterrupt()
        journal = MOD.Journal(self.run, self.protocol, max_new_calls=1, transport=interrupted)
        with self.assertRaises(KeyboardInterrupt): journal.call(self.messages, "response")
        with self.assertRaisesRegex(RuntimeError, "unresolved"):
            MOD.Journal(self.run, self.protocol, max_new_calls=1, transport=self.response)
        self.assertEqual(len(MOD.rows(self.run / "attempts.jsonl")), 1)

    def test_wrong_provider_identity_stops_and_is_not_replaced(self):
        def wrong(*_): return {**self.response(*_), "provider_model": "unapproved-fixture"}
        journal = MOD.Journal(self.run, self.protocol, max_new_calls=3, transport=wrong)
        with self.assertRaisesRegex(RuntimeError, "validation failed"): journal.call(self.messages, "response")
        with self.assertRaisesRegex(RuntimeError, "blocks further"): journal.call([{"role": "user", "content": "another"}], "response")
        self.assertEqual(len(MOD.rows(self.run / "attempts.jsonl")), 1)

    def test_corrupted_outcome_and_incomplete_tail_refuse_resume(self):
        journal = MOD.Journal(self.run, self.protocol, max_new_calls=1, transport=self.response)
        journal.call(self.messages, "response")
        path = self.run / "outcomes.jsonl"
        row = json.loads(path.read_text()); row["response"]["hypothesis"] = "tampered"
        path.write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(ValueError, "integrity"):
            MOD.Journal(self.run, self.protocol, max_new_calls=1, transport=self.response)
        path.write_text('{"unfinished":')
        with self.assertRaisesRegex(ValueError, "incomplete journal"):
            MOD.Journal(self.run, self.protocol, max_new_calls=1, transport=self.response)


if __name__ == "__main__": unittest.main()
