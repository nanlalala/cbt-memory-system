"""Tiny invented transport fixtures; no benchmark source/model/API use."""
import json
from pathlib import Path
import tempfile
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import unittest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation/objective-memory"))
import run_v4_changed_requests_colab as h


class FixtureFailure(RuntimeError):
    reported_usage = {"prompt_tokens": 17}


def response(model=h.ALIAS):
    return {"hypothesis": "Nova", "provider_model": model, "finish_reason": "stop",
            "latency_seconds": .25, "usage": {"prompt_tokens": 11, "completion_tokens": 1, "total_tokens": 12}}


class HarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="v4-tiny-fixture-")
        self.path = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def journal(self, texts=("fictional request",), *, budget=4, max_new=4, transport=lambda messages: response()):
        requests = {}
        for text in texts:
            body = h.request_body([{"role": "user", "content": text}])
            requests[h.stable(body)] = body
        protocol = {"fixture_only": True, "maximum_provider_attempts": budget, "protocol_sha256": "tiny-fixture"}
        return h.Journal(self.path, protocol, requests, lambda: None, max_new_calls=max_new, transport=transport)

    def test_exact_request_dedup_and_resume_do_not_repeat_provider_call(self):
        calls = []
        journal = self.journal(texts=("same", "same"), transport=lambda m: calls.append(m) or response())
        key = next(iter(journal.requests))
        journal.call(key); journal.call(key)
        self.assertEqual(len(calls), 1)
        self.assertEqual(journal.cost()["reported_tokens"]["total_tokens"], 12)
        resumed = self.journal(texts=("same",), transport=lambda m: self.fail("must reuse outcome"))
        resumed.call(key)
        self.assertEqual(resumed.new_calls, 0)

    def test_unknown_failure_usage_stays_terminal_and_not_free_or_retried(self):
        def fail(messages):
            raise FixtureFailure("do-not-persist-this-private-error")
        journal = self.journal(transport=fail)
        key = next(iter(journal.requests))
        journal.call(key); journal.call(key)
        self.assertEqual(journal.cost()["recorded_provider_attempts"], 1)
        self.assertEqual(journal.cost()["failed_attempts"], 1)
        self.assertEqual(journal.cost()["reported_tokens"]["prompt_tokens"], 17)
        self.assertEqual(journal.cost()["failed_attempts_missing_usage_fields"]["completion_tokens"], 1)
        self.assertNotIn("do-not-persist-this-private-error", (self.path / "outcomes.jsonl").read_text())

    def test_inflight_intent_blocks_constructor_without_automatic_recovery(self):
        journal = self.journal()
        key = next(iter(journal.requests))
        h.append(self.path / "attempts.jsonl", h.seal({"protocol_sha256": "tiny-fixture", "request_key": key, "attempt_index": 1}))
        with self.assertRaisesRegex(RuntimeError, "Unresolved"):
            self.journal()

    def test_identity_failure_stops_new_requests_and_retains_supplied_usage(self):
        journal = self.journal(texts=("first", "second"), transport=lambda m: response(model="wrong"))
        keys = list(journal.requests)
        journal.call(keys[0]); journal.call(keys[1])
        self.assertTrue(journal.stopped)
        self.assertEqual(journal.cost()["recorded_provider_attempts"], 1)
        self.assertEqual(journal.cost()["reported_tokens"]["total_tokens"], 12)
        with self.assertRaisesRegex(RuntimeError, "blocks continuation"):
            self.journal(texts=("first", "second"))

    def test_four_workers_respect_single_shared_finite_invocation_budget(self):
        journal = self.journal(texts=("one", "two", "three", "four"), budget=2, max_new=2)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(journal.call, journal.requests))
        self.assertEqual(journal.cost()["recorded_provider_attempts"], 2)

    def test_concurrent_duplicate_key_is_blocked_before_second_provider_send(self):
        started, release = Event(), Event()
        calls = []
        def blocked(messages):
            calls.append(messages); started.set()
            if not release.wait(2):
                raise RuntimeError("fixture release timeout")
            return response()
        journal = self.journal(transport=blocked)
        key = next(iter(journal.requests))
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(journal.call, key)
            self.assertTrue(started.wait(1))
            duplicate = pool.submit(journal.call, key)
            try:
                with self.assertRaisesRegex(RuntimeError, "already in flight"):
                    duplicate.result(timeout=1)
            finally:
                release.set()
            first.result(timeout=1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(h.rows(self.path / "attempts.jsonl")), 1)
        journal.revalidate()

    def test_final_reread_detects_persisted_journal_tampering(self):
        journal = self.journal()
        journal.call(next(iter(journal.requests)))
        outcome = h.rows(self.path / "outcomes.jsonl")[0]
        outcome["response"]["hypothesis"] = "Changed after generation"
        outcome = h.seal({k: v for k, v in outcome.items() if k != "record_sha256"})
        h.atomic_rows(self.path / "outcomes.jsonl", [outcome])
        with self.assertRaisesRegex(ValueError, "in-memory execution"):
            journal.revalidate()

    def test_request_reference_injection_and_changed_output_policy_rejected(self):
        with self.assertRaisesRegex(ValueError, "message structure"):
            h.request_body([{"role": "user", "content": "query", "accepted_answers": ["forbidden"]}])
        settings = {"model": h.MODEL, "accepted_provider_models": [h.ALIAS], "max_response_tokens": 128, "thinking_mode": "disabled"}
        proto = {"settings": settings, "temperature": 0, "api_base": h.API}
        h.verify_old_settings(proto, settings)
        with self.assertRaisesRegex(ValueError, "settings differ"):
            h.verify_old_settings(proto, {**settings, "max_response_tokens": 1024})

    def test_partial_tail_and_duplicate_attempt_index_are_not_repaired(self):
        (self.path / "attempts.jsonl").write_text('{"incomplete":')
        with self.assertRaisesRegex(ValueError, "Incomplete journal"):
            self.journal()

    def test_terminal_failures_remain_in_counterfactual_denominator(self):
        scores = []
        for arm, score, provenance, completion in (
            ("short_term", 0., "unchanged_prompt_reused_old_response", "success"),
            ("raw_history_rag", 1., "unchanged_prompt_reused_old_response", "success"),
            ("managed_memory_rag", 0., "new_v4_terminal_failure", "provider_failure")):
            scores.append({"case_id": "fictional", "arm": arm, "score": score,
                           "strict_committed_score": score, "provenance": provenance, "completion_status": completion})
        aggregate = h.aggregate(scores)
        self.assertEqual(aggregate["conditions"]["managed_memory_rag"]["questions"], 1)
        self.assertEqual(aggregate["conditions"]["managed_memory_rag"]["new_failure_rows"], 1)
        self.assertEqual(aggregate["managed_minus_raw"]["accuracy_delta"], -1.)
        self.assertIsNone(aggregate["managed_minus_raw"]["confidence_interval"])

    def test_non_colab_guard_precedes_invalid_source_and_credential_access(self):
        if Path("/content").is_dir():
            self.skipTest("CPU non-Colab guard check only")
        with self.assertRaisesRegex(RuntimeError, "before source or credential"):
            h.run(None)


if __name__ == "__main__":
    unittest.main()
