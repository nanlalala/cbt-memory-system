"""Tiny synthetic stage fixtures; provider mocked before any guard seam."""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation/objective-memory"))
import generate_semantic_index as stage


def source_fixture(directory):
    directory.mkdir()
    inventory, cases = [], []
    for scope in ("scope-a", "scope-b"):
        history = [{"turn_id": f"{scope}:turn:{n}", "session_id": f"{scope}:released-sequence", "role": "context",
                    "text": f"{scope} source fact {n}.", "order": n, "observed_at": None, "sensitivity": "standard"} for n in range(2)]
        case = {"case_id": scope + ":q", "subject_id": "one-family", "dataset_key": "memoryagentbench_conflict",
                "query": "QUESTION_SENTINEL_MUST_NOT_REACH_EXTRACTION", "as_of": None, "history_scope_id": scope,
                "history_inventory_sha256": stage.qa.stable_digest(["one-family", history])}
        cases.append(case)
        inventory.extend({"document_id": stage.qa._document(case, row).document_id, "subject_id": case["subject_id"], "source": row} for row in history)
    for name, rows in (("inventory.jsonl", inventory), ("cases.jsonl", cases)):
        stage._write(directory / name, rows)
    protocol = {"case_ids": [c["case_id"] for c in cases],
                "file_sha256": {name: stage._sha(directory / name) for name in ("inventory.jsonl", "cases.jsonl")}}
    protocol["protocol_sha256"] = stage.qa.stable_digest(protocol)
    stage._json(directory / "protocol.json", protocol)
    # No references or prompts file exists: preparation cannot rely on them.
    return cases


def fake_response(request, settings, **kwargs):
    docs = json.loads(request["messages"][1]["content"])["sources"]
    return {"hypothesis": json.dumps({"candidates": [{"source_document_id": d["source_document_id"], "memory_type": "explicit_fact",
                                                     "summary": "source-supported retrieval hint", "supporting_text": d["text"]} for d in docs]}),
            "provider_model": "fixture-model", "finish_reason": "stop", "latency_seconds": .01,
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}}


class SemanticGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source, self.output = self.root / "quote", self.root / "extract"
        self.cases = source_fixture(self.source)
        self.settings = stage.qa.Settings(model="fixture-model", backend="lexical", max_response_tokens=128, thinking_mode="disabled")

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, **kwargs):
        # Small fixtures only; every generation test separately mocks provider.
        with patch.object(stage.qa, "require_colab"):
            return stage.prepare(self.source, self.output, self.settings, "https://example.invalid/v1", **kwargs)

    def provider_context(self, **kwargs):
        stack = ExitStack()
        stack.enter_context(patch.object(stage.qa, "require_colab"))
        provider = stack.enter_context(patch.object(stage.qa, "generate_response", **kwargs))
        return stack, provider

    def test_two_scopes_one_subject_preserve_exact_case_boundary(self):
        receipt = self.prepare(case_ids=["scope-b:q"])
        self.assertEqual(receipt["source_documents"], 2)
        mappings = stage._rows(self.output / "case_sources.jsonl")
        self.assertEqual([m["case_id"] for m in mappings], ["scope-b:q"])
        text = (self.output / "extraction_requests.jsonl").read_text()
        self.assertNotIn("QUESTION_SENTINEL", text)
        self.assertNotIn("scope-a", text)
        self.assertIn('source_role', text)
        self.assertEqual(stage.status(self.output)["actual_provider_attempts"], 0)

    def test_wrong_visible_history_hash_rejected_before_calls(self):
        path = self.source / "cases.jsonl"
        cases = stage._rows(path)
        cases[0]["history_inventory_sha256"] = "bad"
        stage._write(path, cases)
        protocol = json.loads((self.source / "protocol.json").read_text())
        protocol["file_sha256"]["cases.jsonl"] = stage._sha(path)
        protocol.pop("protocol_sha256")
        protocol["protocol_sha256"] = stage.qa.stable_digest(protocol)
        stage._json(self.source / "protocol.json", protocol)
        with self.assertRaisesRegex(ValueError, "exact visible history"):
            self.prepare()

    def test_finite_batch_resume_compiles_source_bound_case_index(self):
        self.prepare()
        context, provider = self.provider_context(side_effect=fake_response)
        with context:
            first = stage.generate(self.output, max_calls=1, key="fixture-secret")
            self.assertFalse(first["complete"])
            with self.assertRaisesRegex(ValueError, "terminal"):
                stage.compile_index(self.output)
            second = stage.generate(self.output, max_calls=1, key="fixture-secret")
            self.assertTrue(second["complete"])
            stage.generate(self.output, max_calls=1, key="fixture-secret")
            self.assertEqual(provider.call_count, 2)
        receipt = stage.compile_index(self.output)
        self.assertEqual(receipt["candidate_count"], 4)
        self.assertEqual(receipt["reported_usage_including_invalid_indexes"]["total_tokens"], 60)
        index = stage._rows(self.output / "semantic_index.jsonl")
        self.assertEqual(len(index), 2)
        self.assertTrue(all(c["subject_id"] == "one-family" for row in index for c in row["candidates"]))
        self.assertNotIn("fixture-secret", "".join(p.read_text() for p in self.output.iterdir() if p.is_file()))

    def test_invalid_quote_retains_response_usage_and_terminal_empty_index(self):
        self.prepare(case_ids=["scope-a:q"])
        def invalid(*args, **kwargs):
            response = fake_response(*args, **kwargs)
            body = json.loads(response["hypothesis"])
            body["candidates"][0]["supporting_text"] = "unsupported quote"
            return {**response, "hypothesis": json.dumps(body)}
        context, provider = self.provider_context(side_effect=invalid)
        with context:
            status = stage.generate(self.output, max_calls=1, key="fixture-secret")
            stage.generate(self.output, max_calls=1, key="fixture-secret")
            self.assertEqual(provider.call_count, 1)
        self.assertEqual(status["counts"], {"exhausted_failure": 1})
        self.assertEqual(status["reported_usage_including_invalid_indexes"]["total_tokens"], 30)
        receipt = stage.compile_index(self.output)
        self.assertEqual(receipt["candidate_count"], 0)
        self.assertEqual(receipt["status"], "complete_with_extraction_failures")

    def test_retryable_failures_stop_at_two_attempts_without_cost_invention(self):
        self.prepare(case_ids=["scope-a:q"])
        context, provider = self.provider_context(side_effect=TimeoutError)
        with context:
            first = stage.generate(self.output, max_calls=60, key="fixture-secret")
            self.assertEqual(first["counts"], {"pending": 1})
            second = stage.generate(self.output, max_calls=60, key="fixture-secret")
            self.assertEqual(second["counts"], {"exhausted_failure": 1})
            stage.generate(self.output, max_calls=60, key="fixture-secret")
            self.assertEqual(provider.call_count, 2)
        self.assertEqual(second["attempts_without_reported_usage"], 2)
        self.assertIsNone(second["monetary_cost"])

    def test_inflight_not_retried_and_explicit_unknown_recovery_terminal(self):
        self.prepare(case_ids=["scope-a:q"])
        protocol, _, requests, _ = stage._load(self.output)
        request = requests[0]
        intent = {"request_key": request["request_key"], "prompt_sha256": request["prompt_sha256"],
                  "protocol_sha256": protocol["protocol_sha256"], "attempt_index": 1, "started_at_unix": 1}
        stage.qa._append(self.output / "index_attempts.jsonl", intent)
        context, provider = self.provider_context(side_effect=fake_response)
        with context:
            result = stage.generate(self.output, max_calls=1, key="fixture-secret")
            provider.assert_not_called()
        self.assertEqual(result["counts"], {"inflight": 1})
        result = stage.recover(self.output, request_key=request["request_key"], resolution="terminal_unknown_failure")
        self.assertEqual(result["counts"], {"exhausted_failure": 1})
        self.assertEqual(result["actual_provider_attempts"], 1)

    def test_request_tampering_and_source_budget_refuse_silent_change(self):
        self.prepare()
        (self.output / "extraction_requests.jsonl").write_text("[]")
        with self.assertRaisesRegex(ValueError, "frozen index artifact"):
            stage.status(self.output)
        source = {"subject_id": "u", "document_id": "d", "source": {"role": "context", "session_id": "s", "order": 0, "text": "x" * 12000}}
        with self.assertRaisesRegex(ValueError, "exceeds"):
            stage._requests([source], 12000)

    def test_recorded_response_recovery_has_no_new_provider_call(self):
        self.prepare(case_ids=["scope-a:q"])
        protocol, _, requests, _ = stage._load(self.output)
        request = requests[0]
        intent = {"request_key": request["request_key"], "prompt_sha256": request["prompt_sha256"],
                  "protocol_sha256": protocol["protocol_sha256"], "attempt_index": 1, "started_at_unix": 1}
        response = {**fake_response(request, self.settings), **intent}
        response["response_sha256"] = stage.qa.stable_digest(response)
        stage.qa._append(self.output / "index_attempts.jsonl", intent)
        stage.qa._append(self.output / "index_responses.jsonl", response)
        receipt = stage.recover(self.output, request_key=request["request_key"], resolution="use_recorded_response")
        self.assertEqual(receipt["actual_provider_attempts"], 1)
        self.assertEqual(receipt["counts"], {"success": 1})
        self.assertEqual(stage.compile_index(self.output)["candidate_count"], 2)


if __name__ == "__main__":
    unittest.main()
