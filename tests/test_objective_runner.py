"""Tiny synthetic end-to-end runner fixtures; no models, API, or datasets."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation/objective-memory"))
import run_objective_comparison as runner
from objective_datasets import parse_personamem_rows, parse_mab_rows


def fixture():
    case = {"case_id": "fixture:q", "subject_id": "fixture:user", "group_id": "fixture:group",
            "dataset_key": "synthetic", "task": "recall", "query": "What colour is my notebook?",
            "options": [], "events": [], "as_of": None, "ordering_kind": "source_order_no_calendar",
            "history": [{"turn_id": "t1", "session_id": "s", "role": "system", "order": 0,
                         "text": "The notebook is blue.", "observed_at": None, "sensitivity": "standard"},
                        {"turn_id": "t2", "session_id": "s", "role": "user", "order": 1,
                         "text": "We discussed lunch.", "observed_at": None, "sensitivity": "standard"}]}
    reference = {"case_id": case["case_id"], "group_id": case["group_id"], "dataset_key": case["dataset_key"],
                 "task": case["task"], "metric": "mab_substring_exact_match", "accepted_answers": ["blue"]}
    return [case], [reference]


class ObjectiveRunnerTests(unittest.TestCase):
    def settings(self):
        return runner.Settings(model="fixture-model", accepted_provider_models=("fixture-alias",),
                               short_turns=1, top_k=3, minimum_relevance=0,
                               short_chars=1000, long_chars=1000, max_context_chars=6000,
                               max_response_tokens=64, thinking_mode="disabled")

    def prepare(self, path, cases=None, refs=None, *, fixture_only=True):
        if cases is None:
            cases, refs = fixture()
        with patch.object(runner, "require_colab"):
            return runner.prepare_run(cases, refs, path, self.settings(), "https://example.invalid/v1",
                                      {"synthetic_fixture": True}, fixture_only=fixture_only)

    def test_equal_quote_controls_are_explicitly_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            summary = self.prepare(path)
            self.assertEqual(summary["deduplication"]["raw_managed_equal_fraction"], 1)
            self.assertEqual(summary["deduplication"]["unique_provider_requests"], 2)
            protocol, prompts, refs, completed = runner.load_run(path)
            raw, managed = prompts[1:]
            self.assertEqual(raw["messages"], managed["messages"])
            self.assertEqual(raw["canonical_request_key"], managed["canonical_request_key"])
            self.assertNotEqual(raw["request_key"], managed["request_key"])
            self.assertEqual(completed, {})
            self.assertIn("system_observation", json.dumps(managed["trace"]))
            self.assertNotIn("accepted_answers", json.dumps(prompts))
            self.assertIn("internal source-order", protocol["representation"]["clock_policy"])
            self.assertNotIn("2000-01", json.dumps(prompts))
            self.assertIn("blue", refs[0]["accepted_answers"])

    def test_generation_charges_only_two_real_calls_for_three_arms(self):
        response = {"hypothesis": "blue", "provider_model": "fixture-alias", "finish_reason": "stop",
                    "latency_seconds": .1, "usage": {"prompt_tokens": 30, "completion_tokens": 1, "total_tokens": 31}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.prepare(path, fixture_only=False)
            with patch.object(runner, "require_colab"), patch.object(runner, "generate_response", return_value=response) as provider:
                status = runner.generate_run(path, max_calls=3, key="synthetic-fixture")
                self.assertEqual(provider.call_count, 2)
                self.assertTrue(status["complete"])
                runner.generate_run(path, max_calls=3, key="synthetic-fixture")
                self.assertEqual(provider.call_count, 2)
            _, _, _, completed = runner.load_run(path)
            self.assertEqual(len(completed), 3)
            self.assertEqual(sum(row["charged_call"] for row in completed.values()), 2)
            shared = [row for row in completed.values() if row["arm"] != "short_term"]
            self.assertEqual(shared[0]["source_response_sha256"], shared[1]["source_response_sha256"])
            self.assertEqual(sum(row["deduplicated_call"] for row in shared), 1)

    def test_unresolved_intent_never_automatically_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.prepare(path, fixture_only=False)
            protocol, prompts, _, _ = runner.load_run(path)
            prompt = prompts[0]
            intent = {"request_key": prompt["canonical_request_key"], "canonical_request_key": prompt["canonical_request_key"],
                      "protocol_sha256": protocol["protocol_sha256"], "prompt_sha256": prompt["prompt_sha256"],
                      "attempt_index": 1, "started_at_unix": 1}
            runner._append(path / "response_attempts.jsonl", intent)
            states = runner.request_outcomes(path)
            self.assertEqual(states[("fixture:q", "short_term")]["status"], "inflight")
            with patch.object(runner, "require_colab"), patch.object(runner, "generate_response", side_effect=TimeoutError) as provider:
                status = runner.generate_run(path, max_calls=2, key="synthetic-fixture")
                self.assertEqual(provider.call_count, 1)
                self.assertFalse(status["complete"])

    def test_explicit_recovery_closes_unknown_attempt_without_free_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.prepare(path, fixture_only=False)
            protocol, prompts, _, _ = runner.load_run(path)
            prompt = prompts[1]
            intent = {"request_key": prompt["canonical_request_key"], "canonical_request_key": prompt["canonical_request_key"],
                      "protocol_sha256": protocol["protocol_sha256"], "prompt_sha256": prompt["prompt_sha256"],
                      "attempt_index": 1, "started_at_unix": 1}
            runner._append(path / "response_attempts.jsonl", intent)
            status = runner.recover_interrupted(path, canonical_request_key=prompt["canonical_request_key"],
                                                resolution="terminal_unknown_failure")
            self.assertEqual(status["arm_outcomes"]["exhausted_failure"], 2)
            self.assertEqual(status["actual_provider_attempts"], 1)
            with patch.object(runner, "require_colab"), patch.object(runner, "generate_response", side_effect=TimeoutError) as provider:
                runner.generate_run(path, max_calls=4, key="synthetic-fixture")
                self.assertEqual(provider.call_count, 1)

    def test_recovery_commits_already_recorded_valid_response_without_call(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.prepare(path, fixture_only=False)
            protocol, prompts, _, _ = runner.load_run(path)
            prompt = prompts[1]
            intent = {"request_key": prompt["canonical_request_key"], "canonical_request_key": prompt["canonical_request_key"],
                      "protocol_sha256": protocol["protocol_sha256"], "prompt_sha256": prompt["prompt_sha256"],
                      "attempt_index": 1, "started_at_unix": 1}
            response = {**intent, "hypothesis": "blue", "provider_model": "fixture-alias", "finish_reason": "stop",
                        "latency_seconds": .1, "usage": {}}
            response["response_sha256"] = runner.stable_digest(response)
            runner._append(path / "response_attempts.jsonl", intent)
            runner._append(path / "responses.jsonl", response)
            self.assertEqual(runner.status_run(path)["arm_outcomes"]["inflight"], 2)
            status = runner.recover_interrupted(path, canonical_request_key=prompt["canonical_request_key"],
                                                resolution="use_recorded_response")
            self.assertEqual(status["arm_outcomes"]["success"], 2)
            self.assertEqual(status["actual_provider_attempts"], 1)
            self.assertEqual(status["valid_provider_responses"], 1)

    def test_exclusive_run_lock_blocks_second_executor_before_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.prepare(path, fixture_only=False)
            with runner.exclusive_directory(path), patch.object(runner, "require_colab"), patch.object(runner, "generate_response") as provider:
                with self.assertRaises(Exception) as error:
                    runner.generate_run(path, max_calls=1, key="synthetic-fixture")
                self.assertIn("busy", type(error.exception).__name__.lower())
                provider.assert_not_called()

    def test_transient_failures_keep_denominator_and_bounded_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.prepare(path, fixture_only=False)
            with patch.object(runner, "require_colab"), patch.object(runner, "generate_response", side_effect=TimeoutError) as provider:
                first = runner.generate_run(path, max_calls=10, key="synthetic-fixture")
                self.assertFalse(first["complete"])
                final = runner.generate_run(path, max_calls=10, key="synthetic-fixture")
                self.assertTrue(final["complete"])
                self.assertEqual(final["arm_outcomes"]["exhausted_failure"], 3)
                self.assertEqual(provider.call_count, 4)
                runner.generate_run(path, max_calls=10, key="synthetic-fixture")
                self.assertEqual(provider.call_count, 4)

    def test_wrong_model_identity_invalidates_run(self):
        response = {"hypothesis": "blue", "provider_model": "wrong-model", "finish_reason": "stop",
                    "latency_seconds": .1, "usage": {}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.prepare(path, fixture_only=False)
            with patch.object(runner, "require_colab"), patch.object(runner, "generate_response", return_value=response):
                with self.assertRaisesRegex(ValueError, "identity"):
                    runner.generate_run(path, max_calls=1, key="synthetic-fixture")
            with self.assertRaisesRegex(ValueError, "identity"):
                runner.load_run(path)

    def test_gold_fields_and_unknown_lifecycle_events_rejected(self):
        for field, value in (("accepted_answers", ["blue"]), ("events", [{"type": "delete"}])):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                cases, refs = fixture()
                cases[0][field] = value
                with self.assertRaises(ValueError):
                    self.prepare(Path(directory), cases, refs)

    def test_time_access_and_empty_source_filter_common_to_both_arms(self):
        cases, refs = fixture()
        case = cases[0]
        case["as_of"] = "2026-11-04T00:00:00Z"
        for index, patch_values in enumerate((
            {"sensitivity": "restricted"}, {"valid_to": case["as_of"]},
            {"observed_at": "2026-11-05T00:00:00Z"}, {"lifecycle_status": "deleted"}, {"text": "\n"},
        )):
            row = deepcopy(case["history"][0])
            row.update(patch_values)
            row.update(turn_id=f"excluded:{index}", order=index+2)
            case["history"].append(row)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.prepare(path, cases, refs)
            _, prompts, _, _ = runner.load_run(path)
            for prompt in prompts:
                self.assertEqual(len(prompt["trace"]["excluded_source_units"]), 5)
            self.assertEqual(prompts[1]["messages"], prompts[2]["messages"])

    def test_personamem_adapter_to_preparation_preserves_labels_and_roles(self):
        rows = [{"persona_id": "p", "shared_context_id": "c", "question_id": "q",
                 "question_type": "recall_user_shared_facts", "end_index_in_shared_context": "2",
                 "all_options": ["(a) blue", "(b) red", "(c) green", "(d) white"],
                 "correct_answer": "(a)", "user_question_or_message": "Which notebook?", "topic": "books"}]
        contexts = {"c": [{"role": "system", "content": "Synthetic user has a blue notebook."},
                           {"role": "assistant", "content": "We discussed books."},
                           {"role": "user", "content": "FUTURE HIDDEN"}]}
        cases, refs = parse_personamem_rows(rows, contexts)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.prepare(path, cases, refs)
            _, prompts, _, _ = runner.load_run(path)
            self.assertIn("original option label", prompts[0]["messages"][1]["content"])
            self.assertNotIn("FUTURE HIDDEN", json.dumps(prompts))
            self.assertEqual({row["role"] for row in prompts[2]["trace"]["storage"]}, {"system", "assistant"})

    def test_mab_adapter_blank_lines_and_boundary_whitespace(self):
        cases, refs = parse_mab_rows([{"metadata": {"source": "factconsolidation_sh_6k", "qa_pair_ids": ["q1"]},
                                      "context": "Here is a list of facts:\n\nThe notebook is blue.\n",
                                      "questions": ["Notebook colour?"], "answers": [["blue"]]}])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.prepare(path, cases, refs)
            _, prompts, _, _ = runner.load_run(path)
            self.assertEqual(prompts[1]["messages"], prompts[2]["messages"])
            self.assertTrue(all(row["origin"] == "system_observation" for row in prompts[2]["trace"]["storage"]))

    def test_frozen_artifact_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.prepare(path)
            with (path / "references.jsonl").open("a") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(ValueError, "artifact"):
                runner.load_run(path)

    def test_fixture_generation_and_invalid_call_budgets_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.prepare(path)
            with patch.object(runner, "require_colab"):
                with self.assertRaisesRegex(ValueError, "fixture-only"):
                    runner.generate_run(path, max_calls=1, key="synthetic-fixture")
                with self.assertRaisesRegex(ValueError, "finite positive"):
                    runner.generate_run(path, max_calls=0, key="synthetic-fixture")


if __name__ == "__main__":
    unittest.main()
