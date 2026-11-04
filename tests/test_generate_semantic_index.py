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
from semantic_source_index import rank_sources


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


def temporal_source_fixture(directory, *, prefix=False, duplicate_early=False):
    """A future explanation changes what the same exact early quote could mean."""
    directory.mkdir()
    subject = "same-subject"
    session = "persona32k:c:released-sequence" if prefix else "same-session"
    texts = ["I wrote it down.", "Hello again.", "By it I meant the thought record."]
    times = ["2026-10-01T09:00:00Z", "2026-10-01T10:00:00Z", "2026-10-03T09:00:00Z"]
    history = [{"turn_id": f"t{n}", "session_id": session, "role": "user", "text": text,
                "order": n, "observed_at": None if prefix else times[n], "sensitivity": "standard"}
               for n, text in enumerate(texts)]
    cases = []
    for cid, count, cutoff in (("early", 2, "2026-10-02T00:00:00Z"), ("late", 3, "2026-10-04T00:00:00Z")):
        case = {"case_id": cid, "subject_id": subject, "dataset_key": "personamem_v1" if prefix else "synthetic",
                "query": "QUESTION_NOT_SUPPLIED", "as_of": None if prefix else cutoff,
                "history_inventory_sha256": stage.qa.stable_digest([subject, history[:count]])}
        if prefix:
            case.update(source_context_id="c", history_scope_id=f"persona32k:c:prefix:{count}",
                        visible_context_end_exclusive=count)
        cases.append(case)
    if duplicate_early:
        cases.append({**cases[0], "case_id": "same-visible-history-other-question", "query": "ANOTHER_QUERY_NOT_SUPPLIED"})
    inventory = [{"document_id": stage.qa._document({"subject_id": subject}, row).document_id,
                  "subject_id": subject, "source": row} for row in history]
    for name, rows in (("inventory.jsonl", inventory), ("cases.jsonl", cases)):
        stage._write(directory / name, rows)
    protocol = {"case_ids": [c["case_id"] for c in cases],
                "file_sha256": {name: stage._sha(directory / name) for name in ("inventory.jsonl", "cases.jsonl")}}
    protocol["protocol_sha256"] = stage.qa.stable_digest(protocol)
    stage._json(directory / "protocol.json", protocol)
    return cases, history


def future_sensitive_fake_response(request, settings, **kwargs):
    docs = json.loads(request["messages"][1]["content"])["sources"]
    first = next(d for d in docs if d["text"] == "I wrote it down.")
    summary = "thought record" if any("By it I meant" in d["text"] for d in docs) else "wrote something down"
    response = fake_response(request, settings, **kwargs)
    response["hypothesis"] = json.dumps({"candidates": [{"source_document_id": first["source_document_id"],
        "memory_type": "explicit_fact", "summary": summary, "supporting_text": first["text"]}]})
    return response


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

    def test_persistence_failure_after_valid_response_is_recoverable_without_paid_retry(self):
        for failure in ("response_postappend", "success_outcome_preappend"):
            with self.subTest(failure=failure):
                output = self.root / failure
                with patch.object(stage.qa, "require_colab"):
                    stage.prepare(self.source, output, self.settings, "https://example.invalid/v1", case_ids=["scope-a:q"])
                append = stage.qa._append
                raised = False

                def inject(path, row):
                    nonlocal raised
                    if not raised and failure == "response_postappend" and path.name == "index_responses.jsonl":
                        append(path, row)
                        raised = True
                        raise OSError("injected post-append flush failure")
                    if not raised and failure == "success_outcome_preappend" and path.name == "index_outcomes.jsonl" and row["status"] == "success":
                        raised = True
                        raise OSError("injected success-outcome persistence failure")
                    append(path, row)

                context, provider = self.provider_context(side_effect=fake_response)
                with context:
                    with patch.object(stage.qa, "_append", side_effect=inject), self.assertRaises(OSError):
                        stage.generate(output, max_calls=1, key="fixture-secret")
                    self.assertTrue(raised)
                    self.assertEqual(provider.call_count, 1)
                    self.assertEqual(stage._rows(output / "index_outcomes.jsonl"), [])
                    receipt = stage.status(output)
                    self.assertEqual(receipt["counts"], {"inflight": 1})
                    self.assertEqual(receipt["recorded_provider_responses"], 1)
                    self.assertEqual(receipt["failed_outcomes"], 0)
                    self.assertEqual(receipt["reported_usage_including_invalid_indexes"]["total_tokens"], 30)
                    stage.generate(output, max_calls=1, key="fixture-secret")
                    self.assertEqual(provider.call_count, 1)
                    protocol, _, requests, _ = stage._load(output)
                    receipt = stage.recover(output, request_key=requests[0]["request_key"], resolution="use_recorded_response")
                    self.assertEqual(receipt["counts"], {"success": 1})
                    self.assertEqual(receipt["actual_provider_attempts"], 1)
                    stage.generate(output, max_calls=1, key="fixture-secret")
                    self.assertEqual(provider.call_count, 1)
                self.assertEqual(stage.compile_index(output)["candidate_count"], 2)

    def _assert_temporal_isolation(self, *, prefix):
        source = self.root / ("prefix-source" if prefix else "date-source")
        cases, history = temporal_source_fixture(source, prefix=prefix, duplicate_early=True)
        joint, early = self.root / "joint", self.root / "early-only"
        with patch.object(stage.qa, "require_colab"):
            joint_receipt = stage.prepare(source, joint, self.settings, "https://example.invalid/v1")
            early_receipt = stage.prepare(source, early, self.settings, "https://example.invalid/v1", case_ids=["early"])
        self.assertEqual(joint_receipt["request_count"], 2)  # Identical early histories may share; late cannot.
        self.assertEqual(early_receipt["request_count"], 1)
        joint_mappings = {r["case_id"]: r for r in stage._rows(joint / "case_sources.jsonl")}
        early_mapping = stage._rows(early / "case_sources.jsonl")[0]
        identity = early_mapping["visibility_sha256"]
        self.assertEqual(identity, joint_mappings["early"]["visibility_sha256"])
        self.assertNotEqual(identity, joint_mappings["late"]["visibility_sha256"])
        self.assertEqual(identity, joint_mappings["same-visible-history-other-question"]["visibility_sha256"])
        joint_request = next(r for r in stage._rows(joint / "extraction_requests.jsonl") if r["visibility_sha256"] == identity)
        early_request = stage._rows(early / "extraction_requests.jsonl")[0]
        self.assertEqual(joint_request, early_request)
        self.assertNotIn("By it I meant", str(early_request["messages"]))
        self.assertNotIn("QUESTION", str(joint_request["messages"]))
        with patch.object(stage.qa, "require_colab"), patch.object(stage.qa, "generate_response", side_effect=future_sensitive_fake_response) as provider:
            self.assertTrue(stage.generate(joint, max_calls=60, key="fixture-secret")["complete"])
            self.assertTrue(stage.generate(early, max_calls=60, key="fixture-secret")["complete"])
            self.assertEqual(provider.call_count, 3)
        stage.compile_index(joint)
        stage.compile_index(early)
        joint_index = {r["case_id"]: r["candidates"] for r in stage._rows(joint / "semantic_index.jsonl")}
        early_index = stage._rows(early / "semantic_index.jsonl")[0]["candidates"]
        self.assertEqual(joint_index["early"], early_index)
        self.assertEqual(early_index[0]["summary"], "wrote something down")
        self.assertEqual(joint_index["late"][0]["summary"], "thought record")
        docs = [stage.qa._document({"subject_id": "same-subject"}, row) for row in history[:2]]
        ranker = stage.qa.HistoryRanker(stage.qa.Settings(model="fixture-model", backend="lexical", top_k=3, minimum_relevance=.01))
        self.assertEqual(rank_sources("thought record", docs, joint_index["early"], ranker, subject_id="same-subject"), [])
        self.assertEqual(rank_sources("thought record", docs, early_index, ranker, subject_id="same-subject"), [])
        self.assertEqual([d.text for d in rank_sources("thought record", docs, joint_index["late"], ranker, subject_id="same-subject")], ["I wrote it down."])

    def test_as_of_early_index_is_identical_alone_and_with_future_case(self):
        self._assert_temporal_isolation(prefix=False)

    def test_exclusive_prefix_early_index_is_identical_alone_and_with_future_case(self):
        self._assert_temporal_isolation(prefix=True)

    def test_conflicting_source_versions_are_rejected_instead_of_overwritten(self):
        path = self.source / "inventory.jsonl"
        inventory = stage._rows(path)
        conflict = {**inventory[0], "source": {**inventory[0]["source"], "text": "Changed same-ID source version."}}
        stage._write(path, [*inventory, conflict])
        protocol = json.loads((self.source / "protocol.json").read_text())
        protocol["file_sha256"]["inventory.jsonl"] = stage._sha(path)
        protocol.pop("protocol_sha256")
        protocol["protocol_sha256"] = stage.qa.stable_digest(protocol)
        stage._json(self.source / "protocol.json", protocol)
        with self.assertRaisesRegex(ValueError, "ambiguous source versions"):
            self.prepare()

    def test_previous_union_scope_protocol_is_not_silently_resumed(self):
        self.prepare()
        path = self.output / "index_protocol.json"
        protocol = json.loads(path.read_text())
        protocol["version"] = "source-only-semantic-extraction-v1"
        protocol.pop("protocol_sha256")
        protocol["protocol_sha256"] = stage.qa.stable_digest(protocol)
        stage._json(path, protocol)
        with self.assertRaisesRegex(ValueError, "unsupported index protocol"):
            stage.status(self.output)


if __name__ == "__main__":
    unittest.main()
