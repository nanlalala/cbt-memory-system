"""Tiny evaluator invariants; fixtures are not benchmark effectiveness data."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation/end-to-end-dialogue"))
import reference_judge as judge  # noqa: E402


class PromptTests(unittest.TestCase):
    def reference(self, category="single-session-user", abstain=False):
        return {"query": "Which notebook color was mentioned?", "reference_answer": "blue",
                "category": category, "should_abstain": abstain}

    def test_pinned_snapshot_preserves_source_identity_and_license(self):
        snapshot = judge.template_snapshot()
        self.assertEqual(snapshot["upstream_revision"], judge.UPSTREAM_REVISION)
        self.assertEqual(snapshot["source_git_blob_sha"], "4732f3772b04a2b9069121ade304e6320494abc2")
        self.assertIn("Copyright (c) 2024 Di Wu", snapshot["license_text"])

    def test_official_per_type_instructions_are_preserved(self):
        temporal = judge.reference_prompt(self.reference("temporal-reasoning"), "blue", "longmemeval_s_cleaned")
        update = judge.reference_prompt(self.reference("knowledge-update"), "blue", "longmemeval_s_cleaned")
        preference = judge.reference_prompt(self.reference("single-session-preference"), "blue", "longmemeval_s_cleaned")
        abstain = judge.reference_prompt(self.reference(abstain=True), "unknown", "longmemeval_s_cleaned")
        self.assertIn("do not penalize off-by-one errors", temporal)
        self.assertIn("previous information along with an updated answer", update)
        self.assertIn("does not need to reflect all the points", preference)
        self.assertIn("unanswerable question", abstain)
        self.assertTrue(all(text.endswith("Answer yes or no only.") for text in (temporal, update, preference, abstain)))

    def test_unknown_types_and_adversarial_locomo_fail_closed(self):
        with self.assertRaises(ValueError):
            judge.reference_prompt(self.reference("invented-type"), "blue", "longmemeval_s_cleaned")
        with self.assertRaises(ValueError):
            judge.reference_prompt(self.reference("5"), "blue", "locomo10")

    def test_yes_substrings_are_not_valid_labels(self):
        self.assertTrue(judge.parse_label(" YES \n"))
        self.assertFalse(judge.parse_label("no"))
        for value in (None, True, "yes and no", "no, but yes", "yes.", "perhaps"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                judge.parse_label(value)

    def test_changed_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            modified = Path(td) / "snapshot.json"
            modified.write_text(judge.SNAPSHOT.read_text() + " ")
            with patch.object(judge, "SNAPSHOT", modified), self.assertRaises(ValueError):
                judge.template_snapshot()


class JudgeRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.run_dir = self.base / "reader"
        self.run_dir.mkdir()
        self.output_dir = self.base / "judge"
        self.plan = json.loads(judge.DEFAULT_PLAN.read_text())
        self.plan["statistics"]["precision_plan"]["resamples"] = 1000
        self.plan_path = self.base / "plan.json"
        self.plan_path.write_text(json.dumps(self.plan))
        self.make_fixture([("fixture-q", "group-1")])
        self.colab = patch.object(judge, "require_colab")
        self.colab.start()
        self.addCleanup(self.colab.stop)
        self.loader = patch.object(judge, "load_run", side_effect=lambda _: self.loaded)
        self.loader_mock = self.loader.start()
        self.addCleanup(self.loader.stop)
        self.env = patch.dict(judge.os.environ, {"CBT_EVAL_API_KEY": "fixture-key-never-log"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def make_fixture(self, items):
        protocol = {"question_ids": [qid for qid, _ in items], "dataset_key": "longmemeval_s_cleaned",
                    "settings": {"model": self.plan["shared_response_configuration"]["model"]},
                    "maximum_attempts_per_request": 3, "maximum_total_calls": len(items) * 9,
                    "cohort_manifest": {"plan_sha256": judge.stable_digest(self.plan),
                                        "confirmatory_preparation_blockers": ["fixture is not research evidence"]}}
        prompts, references, completed = [], [], {}
        for qid, group in items:
            references.append({"example_id": qid, "query": "Which notebook color was mentioned?",
                               "reference_answer": "blue", "category": "single-session-user", "should_abstain": False})
            for arm in judge.ARMS:
                prompts.append({"example_id": qid, "arm": arm, "group_id": group,
                                "prompt_sha256": "fixture-reader-prompt"})
                completed[(qid, arm)] = {"example_id": qid, "arm": arm, "hypothesis": "blue",
                                         "protocol_sha256": judge.stable_digest(protocol),
                                         "prompt_sha256": "fixture-reader-prompt", "finish_reason": "stop",
                                         "provider_model": protocol["settings"]["model"], "usage": {}, "latency_seconds": 0.1}
        self.loaded = protocol, prompts, references, completed
        for name in judge.SOURCE_FILES:
            if name not in {"failures.jsonl", "response_attempts.jsonl"}:
                (self.run_dir / name).write_text(json.dumps({"tiny_fixture": name}) + "\n")

    def prepare(self, **kwargs):
        return judge.prepare_reference_judge(self.run_dir, self.output_dir, plan_path=self.plan_path, **kwargs)

    def fake_result(self, request, settings, **kwargs):
        return {"hypothesis": "yes", "provider_model": settings.model, "finish_reason": "stop",
                "usage": {"prompt_tokens": 100, "completion_tokens": 1, "total_tokens": 101}, "latency_seconds": 0.2}

    def execute(self, calls=3, result=None):
        with patch.object(judge, "generate_response", side_effect=result or self.fake_result) as transport:
            status = judge.execute_reference_judge(self.run_dir, self.output_dir, max_calls=calls)
        return status, transport

    def test_prepare_blinds_labels_and_does_not_generate(self):
        with patch.object(judge, "generate_response") as transport:
            result = self.prepare()
        transport.assert_not_called()
        self.assertEqual(result["judge_requests"], 3)
        requests = judge.read_jsonl(self.output_dir / "judge_requests.jsonl")
        for row in requests:
            self.assertNotIn("arm", row)
            self.assertNotIn("example_id", row)
            self.assertTrue(all(arm not in json.dumps(row["messages"]) for arm in judge.ARMS))
        self.assertIn("blue", requests[0]["messages"][1]["content"])
        self.assertNotIn("fixture-key-never-log", (self.output_dir / "judge_protocol.json").read_text())

    def test_non_colab_guard_precedes_artifact_or_key_access(self):
        with patch.object(judge, "require_colab", side_effect=RuntimeError("Colab required")):
            with self.assertRaises(RuntimeError):
                self.prepare()
            with self.assertRaises(RuntimeError):
                judge.execute_reference_judge(self.run_dir, self.output_dir, max_calls=1)
        self.loader_mock.assert_not_called()

    def test_synthetic_benchmark_fixtures_cannot_be_sent_to_paid_judge(self):
        self.loaded[0]["cohort_manifest"]["fixture_only"] = True
        self.prepare()
        with patch.object(judge, "generate_response") as transport, self.assertRaisesRegex(ValueError, "fixtures"):
            judge.execute_reference_judge(self.run_dir, self.output_dir, max_calls=3)
        transport.assert_not_called()

    def test_budget_and_checkpoint_resume_complete_exact_grid(self):
        self.prepare()
        first, transport = self.execute(calls=1)
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(first["successful_labels"], 1)
        with self.assertRaises(ValueError):
            judge.score_reference_judge(self.run_dir, self.output_dir)
        second, transport = self.execute(calls=10)
        self.assertEqual(transport.call_count, 2)
        self.assertEqual(second["status"], "complete")
        summary = judge.score_reference_judge(self.run_dir, self.output_dir)
        self.assertEqual(summary["scored_rows"], 3)
        self.assertFalse(summary["confirmatory_claim"])
        self.assertFalse(summary["predeclared_primary_checks"]["minimum_groups_met"])
        self.assertTrue(summary["same_reader_judge_name"])
        self.assertNotIn("fixture-key-never-log", json.dumps(summary))

    def test_consistently_wrong_reported_model_invalidates_run(self):
        self.prepare()
        def wrong(request, settings, **kwargs):
            return {**self.fake_result(request, settings), "provider_model": "wrong-model"}
        with self.assertRaises(ValueError):
            self.execute(calls=3, result=wrong)
        rows = judge.read_jsonl(self.output_dir / "judge_attempts.jsonl")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "identity_or_integrity_failure")
        with self.assertRaises(ValueError):
            judge.score_reference_judge(self.run_dir, self.output_dir)

    def test_invalid_label_is_retained_not_reinterpreted_as_yes(self):
        self.prepare()
        def invalid(request, settings, **kwargs):
            return {**self.fake_result(request, settings), "hypothesis": "no, but yes"}
        status, _ = self.execute(calls=1, result=invalid)
        self.assertEqual(status["successful_labels"], 0)
        row = judge.read_jsonl(self.output_dir / "judge_attempts.jsonl")[0]
        self.assertEqual(row["judge_text"], "no, but yes")
        self.assertEqual(row["error_type"], "InvalidJudgeLabel")
        self.assertNotIn("label", row)

    def test_exhausted_judge_failures_remain_zero_in_denominator(self):
        self.prepare()
        def failing(*args, **kwargs):
            raise TimeoutError("private transport text must not appear in output")
        for _ in range(3):
            self.execute(calls=3, result=failing)
        summary = judge.score_reference_judge(self.run_dir, self.output_dir)
        self.assertEqual(summary["scored_rows"], 3)
        self.assertEqual(summary["technical_failure_fraction"], 1)
        self.assertTrue(all(v["question_micro_accuracy"] == 0 for v in summary["conditions"].values()))
        status, transport = self.execute(calls=10)
        transport.assert_not_called()
        self.assertEqual(status["model_calls_this_invocation"], 0)
        self.assertNotIn("private transport text", (self.output_dir / "judge_attempts.jsonl").read_text())

    def test_missing_reader_rows_require_retry_evidence_and_explicit_finalization(self):
        protocol, _, _, completed = self.loaded
        missing = next(iter(completed))
        del completed[missing]
        with self.assertRaises(ValueError):
            self.prepare(finalize_reader_failures=True)
        attempts = [{"example_id": missing[0], "arm": missing[1], "protocol_sha256": judge.stable_digest(protocol),
                     "prompt_sha256": "fixture-reader-prompt", "request_key": judge.stable_digest(list(missing)),
                     "attempt_index": number} for number in range(1, 4)]
        judge.write_jsonl(self.run_dir / "response_attempts.jsonl", attempts)
        failures = [{**row, "error_type": "TimeoutError", "retryable": True,
                     "terminal_execution_failure": False} for row in attempts]
        judge.write_jsonl(self.run_dir / "failures.jsonl", failures)
        with self.assertRaises(ValueError):
            self.prepare()
        self.prepare(finalize_reader_failures=True)
        self.execute(calls=3)
        summary = judge.score_reference_judge(self.run_dir, self.output_dir)
        self.assertEqual(summary["scored_rows"], 3)
        self.assertAlmostEqual(summary["technical_failure_fraction"], 1 / 3)

    def test_crash_intent_is_charged_before_transport_and_caps_resume(self):
        self.prepare()
        def interrupted(*args, **kwargs):
            raise KeyboardInterrupt()
        for _ in range(3):
            with self.assertRaises(KeyboardInterrupt):
                self.execute(calls=1, result=interrupted)
        intents = judge.read_jsonl(self.output_dir / "judge_call_intents.jsonl")
        self.assertEqual([row["attempt"] for row in intents], [1, 2, 3])
        self.assertEqual(len({row["item_id"] for row in intents}), 1)
        status, transport = self.execute(calls=10)
        self.assertEqual(transport.call_count, 2)
        summary = judge.score_reference_judge(self.run_dir, self.output_dir)
        self.assertEqual(summary["scored_rows"], 3)
        self.assertAlmostEqual(summary["technical_failure_fraction"], 1 / 3)

    def test_nonretryable_reader_failure_can_finalize_after_one_validated_attempt(self):
        protocol, _, _, completed = self.loaded
        missing = next(iter(completed))
        del completed[missing]
        intent = {"example_id": missing[0], "arm": missing[1], "protocol_sha256": judge.stable_digest(protocol),
                  "prompt_sha256": "fixture-reader-prompt", "request_key": judge.stable_digest(list(missing)),
                  "attempt_index": 1}
        judge.write_jsonl(self.run_dir / "response_attempts.jsonl", [intent])
        judge.write_jsonl(self.run_dir / "failures.jsonl", [{**intent, "error_type": "RuntimeError", "retryable": False,
                                                          "terminal_execution_failure": False}])
        self.prepare(finalize_reader_failures=True)
        self.execute(calls=3)
        summary = judge.score_reference_judge(self.run_dir, self.output_dir)
        self.assertEqual(summary["scored_rows"], 3)
        self.assertAlmostEqual(summary["technical_failure_fraction"], 1 / 3)

    def test_changed_reader_artifacts_reject_resume_before_network(self):
        self.prepare()
        (self.run_dir / "responses.jsonl").write_text("tampered\n")
        with patch.object(judge, "generate_response") as transport, self.assertRaises(ValueError):
            judge.execute_reference_judge(self.run_dir, self.output_dir, max_calls=3)
        transport.assert_not_called()

    def test_modified_endpoint_cannot_redirect_key_to_other_host(self):
        self.prepare()
        path = self.output_dir / "judge_protocol.json"
        protocol = json.loads(path.read_text())
        protocol["api_base"] = "https://unapproved.example/v1"
        path.write_text(json.dumps(protocol))
        with patch.object(judge, "generate_response") as transport, self.assertRaises(ValueError):
            judge.execute_reference_judge(self.run_dir, self.output_dir, max_calls=3)
        transport.assert_not_called()

    def test_rehashed_group_mapping_tamper_is_rejected(self):
        self.prepare()
        mapping_path = self.output_dir / "judge_mapping.jsonl"
        mapping = judge.read_jsonl(mapping_path)
        mapping[0]["group_id"] = "fabricated-independent-person"
        judge.write_jsonl(mapping_path, mapping)
        path = self.output_dir / "judge_protocol.json"
        protocol = json.loads(path.read_text())
        protocol["mapping_sha256"] = judge.stable_digest(mapping)
        path.write_text(json.dumps(protocol))
        with self.assertRaises(ValueError):
            judge.score_reference_judge(self.run_dir, self.output_dir)

    def test_group_equal_endpoint_does_not_treat_repeated_questions_as_people(self):
        self.make_fixture([("q-a", "same-history"), ("q-b", "same-history"), ("q-c", "other-history")])
        self.prepare()
        mapping = {row["item_id"]: row for row in judge.read_jsonl(self.output_dir / "judge_mapping.jsonl")}
        def controlled(request, settings, **kwargs):
            item = mapping[request["item_id"]]
            correct = item["example_id"] == "q-c" and item["arm"] == judge.ARMS[2]
            return {**self.fake_result(request, settings), "hypothesis": "yes" if correct else "no"}
        self.execute(calls=9, result=controlled)
        summary = judge.score_reference_judge(self.run_dir, self.output_dir)
        self.assertEqual(summary["independent_groups"], 2)
        long = summary["conditions"][judge.ARMS[2]]
        self.assertAlmostEqual(long["question_micro_accuracy"], 1 / 3)
        self.assertAlmostEqual(long["group_equal_accuracy"], 0.5)

    def test_real_reader_failure_journal_and_recovered_response_remain_gradeable(self):
        # Use the actual reader/cohort validators here, rather than the loader
        # double used for the small evaluator-specific mutation tests above.
        import run_structured_memory as reader
        from benchmark_adapters import RetrievalExample
        from prepare_formal_evaluation import prepare_examples
        example = RetrievalExample(
            example_id="fixture-real-journal", category="single-session-user",
            query="Which notebook color was mentioned?", answer="blue", evidence_ids=("older",),
            query_date="2026-10-10T00:00:00Z", sessions=(
                {"session_id": "older", "date": "2026-10-01T00:00:00Z", "turns": [{"role": "user", "content": "My notebook is blue."}]},
                {"session_id": "recent", "date": "2026-10-09T00:00:00Z", "turns": [{"role": "user", "content": "Hello again."}]},
            ),
        )
        cohort = self.base / "actual-cohort"
        actual_run = self.base / "actual-reader"
        prepare_examples([example], "longmemeval_s_cleaned", self.plan,
                         {"sha256": "f" * 64, "bytes": 1}, cohort, role="exploratory", fixture_only=True)
        configuration = self.plan["shared_response_configuration"]
        settings = judge.Settings(model=configuration["model"], accepted_provider_models=tuple(configuration["accepted_provider_models"]),
                                  max_response_tokens=4096, thinking_mode="disabled")
        reader.prepare_structured_run(cohort, actual_run, settings, configuration["api_base"])
        protocol = json.loads((actual_run / "protocol.json").read_text())
        prompts = judge.read_jsonl(actual_run / "prompts.jsonl")
        with patch.object(reader, "generate_response", side_effect=TimeoutError("fixture failure")):
            with self.assertRaises(TimeoutError):
                reader.journaled_response(actual_run, protocol, prompts[0], settings,
                                          key="fixture-only", stage="response", attempt_index=1)
        responses = []
        with patch.object(reader, "generate_response", side_effect=self.fake_result):
            for index, request in enumerate(prompts):
                result = reader.journaled_response(actual_run, protocol, request, settings,
                                                   key="fixture-only", stage="response", attempt_index=2 if index == 0 else 1)
                result.update({name: request[name] for name in ("example_id", "arm", "prompt_sha256")})
                result["protocol_sha256"] = judge.stable_digest(protocol)
                responses.append(result)
        judge.write_jsonl(actual_run / "responses.jsonl", responses)
        actual_judge = self.base / "actual-judge"
        with patch.object(judge, "load_run", reader.load_run):
            result = judge.prepare_reference_judge(actual_run, actual_judge, plan_path=self.plan_path)
            self.assertEqual(result["judge_requests"], 3)
            judge_protocol = json.loads((actual_judge / "judge_protocol.json").read_text())
            self.assertIn("response_attempts.jsonl", judge_protocol["reader_file_identities"])
            # Fixture benchmark generations are forbidden. Write explicit tiny
            # software checkpoint fixtures to test the scorer, with no API call.
            intents, outcomes = [], []
            for request in judge.read_jsonl(actual_judge / "judge_requests.jsonl"):
                intent = {"item_id": request["item_id"], "attempt": 1,
                          "prompt_sha256": request["prompt_sha256"], "judge_protocol_sha256": judge.stable_digest(judge_protocol)}
                intents.append(intent)
                outcomes.append({**intent, "status": "success", "judge_text": "yes", "label": True,
                                 "provider_model": settings.model, "finish_reason": "stop", "usage": {}, "latency_seconds": 0})
            judge.write_jsonl(actual_judge / "judge_call_intents.jsonl", intents)
            judge.write_jsonl(actual_judge / "judge_attempts.jsonl", outcomes)
            summary = judge.score_reference_judge(actual_run, actual_judge)
            self.assertEqual(summary["scored_rows"], 3)
            self.assertEqual(summary["technical_failure_fraction"], 0)


if __name__ == "__main__":
    unittest.main()
