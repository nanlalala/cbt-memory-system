"""Paid-call persistence boundaries and read-only diagnostics, using fake providers.

Only tiny synthetic records are used. No dataset, real provider or credential is
loaded; guard patches are limited to exercising the fake-provider write path.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation/objective-memory"))
import run_objective_comparison as v2
import v3_run_objective_comparison as v3


def fixture():
    case = {"case_id": "fixture:q", "subject_id": "fixture:u", "group_id": "fixture:g",
            "dataset_key": "synthetic", "task": "recall", "query": "What colour is the notebook?",
            "options": [], "events": [], "as_of": None,
            "history": [{"session_id": "s", "turn_id": "t", "order": 0,
                         "role": "context", "text": "The notebook is blue."}]}
    ref = {"case_id": case["case_id"], "group_id": case["group_id"],
           "dataset_key": case["dataset_key"], "task": case["task"],
           "metric": "mab_substring_exact_match", "accepted_answers": ["blue"]}
    return case, ref


RESPONSE = {"hypothesis": "blue", "provider_model": "fixture-alias", "finish_reason": "stop",
            "latency_seconds": 0.1, "usage": {"prompt_tokens": 30, "completion_tokens": 1, "total_tokens": 31}}


def contents(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob("*") if p.is_file()}


class PersistenceAndReadonlyTests(unittest.TestCase):
    runners = (v2, v3)

    def prepare(self, runner, path):
        case, ref = fixture()
        settings = runner.Settings(model="fixture-model", accepted_provider_models=("fixture-alias",),
                                   short_turns=3, top_k=3, minimum_relevance=0,
                                   short_chars=1000, long_chars=1000, max_context_chars=6000,
                                   max_response_tokens=64, thinking_mode="disabled")
        with patch.object(runner, "require_colab"):
            runner.prepare_run([case], [ref], path, settings, "https://example.invalid/v1",
                               {"tiny_fake_provider_fixture": True}, fixture_only=False)
        protocol, prompts, _, _ = runner.load_run(path)
        self.assertEqual(len({p["canonical_request_key"] for p in prompts}), 1)
        return prompts[0]["canonical_request_key"]

    def test_saved_response_and_failed_success_commit_remain_recoverable_once(self):
        for runner in self.runners:
            with self.subTest(runner=runner.__name__), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                key = self.prepare(runner, path)
                original_append = runner._append
                injected = False

                def append(target, row):
                    nonlocal injected
                    if target.name == "response_outcomes.jsonl" and row.get("status") == "success" and not injected:
                        injected = True
                        raise OSError("synthetic success-outcome write failure")
                    return original_append(target, row)

                with patch.object(runner, "require_colab"), patch.object(runner, "generate_response", return_value=RESPONSE) as provider:
                    with patch.object(runner, "_append", side_effect=append):
                        with self.assertRaises(OSError):
                            runner.generate_run(path, max_calls=1, key="fake-fixture-only")
                    self.assertEqual(provider.call_count, 1)
                    self.assertEqual(len(runner.read_jsonl(path / "responses.jsonl")), 1)
                    self.assertEqual(runner.read_jsonl(path / "response_outcomes.jsonl"), [])
                    self.assertEqual(runner.read_jsonl(path / "failures.jsonl"), [])
                    self.assertEqual(runner.status_run(path)["arm_outcomes"]["inflight"], 3)
                    recovered = runner.recover_interrupted(path, canonical_request_key=key,
                                                           resolution="use_recorded_response")
                    self.assertEqual(recovered["arm_outcomes"]["success"], 3)
                    self.assertEqual(recovered["actual_provider_attempts"], 1)
                    self.assertEqual(recovered["valid_provider_responses"], 1)
                    with self.assertRaisesRegex(ValueError, "no unresolved"):
                        runner.recover_interrupted(path, canonical_request_key=key, resolution="use_recorded_response")
                    runner.generate_run(path, max_calls=1, key="fake-fixture-only")
                    self.assertEqual(provider.call_count, 1)

    def test_response_write_error_is_not_recorded_as_provider_failure(self):
        for runner in self.runners:
            with self.subTest(runner=runner.__name__), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                key = self.prepare(runner, path)
                original_append = runner._append

                def append(target, row):
                    if target.name == "responses.jsonl":
                        raise OSError("synthetic response write failure")
                    return original_append(target, row)

                with patch.object(runner, "require_colab"), patch.object(runner, "generate_response", return_value=RESPONSE) as provider:
                    with patch.object(runner, "_append", side_effect=append), self.assertRaises(OSError):
                        runner.generate_run(path, max_calls=1, key="fake-fixture-only")
                    self.assertEqual(runner.read_jsonl(path / "response_outcomes.jsonl"), [])
                    self.assertEqual(runner.status_run(path)["arm_outcomes"]["inflight"], 3)
                    closed = runner.recover_interrupted(path, canonical_request_key=key, resolution="terminal_unknown_failure")
                    self.assertEqual(closed["arm_outcomes"]["exhausted_failure"], 3)
                    self.assertEqual(closed["actual_provider_attempts"], 1)
                    self.assertEqual(provider.call_count, 1)

    def test_committed_success_then_sync_error_does_not_cause_duplicate_call(self):
        for runner in self.runners:
            with self.subTest(runner=runner.__name__), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                self.prepare(runner, path)
                original_append = runner._append

                def append(target, row):
                    original_append(target, row)
                    if target.name == "response_outcomes.jsonl" and row.get("status") == "success":
                        raise OSError("synthetic post-write sync failure")

                with patch.object(runner, "require_colab"), patch.object(runner, "generate_response", return_value=RESPONSE) as provider:
                    with patch.object(runner, "_append", side_effect=append), self.assertRaises(OSError):
                        runner.generate_run(path, max_calls=1, key="fake-fixture-only")
                    self.assertEqual(runner.status_run(path)["arm_outcomes"]["success"], 3)
                    runner.generate_run(path, max_calls=1, key="fake-fixture-only")
                    self.assertEqual(provider.call_count, 1)

    def test_partial_response_line_is_rejected_without_fabricated_outcome(self):
        for runner in self.runners:
            with self.subTest(runner=runner.__name__), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                self.prepare(runner, path)
                original_append = runner._append

                def append(target, row):
                    if target.name == "responses.jsonl":
                        target.write_text('{"partial":')
                        raise OSError("synthetic interrupted partial write")
                    return original_append(target, row)

                with patch.object(runner, "require_colab"), patch.object(runner, "generate_response", return_value=RESPONSE) as provider:
                    with patch.object(runner, "_append", side_effect=append), self.assertRaises(OSError):
                        runner.generate_run(path, max_calls=1, key="fake-fixture-only")
                    self.assertEqual(runner.read_jsonl(path / "response_outcomes.jsonl"), [])
                    with self.assertRaises(json.JSONDecodeError):
                        runner.status_run(path)
                    self.assertEqual(provider.call_count, 1)

    def failed_fixture(self, runner, path):
        self.prepare(runner, path)
        with patch.object(runner, "require_colab"), patch.object(runner, "generate_response", side_effect=TimeoutError):
            runner.generate_run(path, max_calls=1, key="fake-fixture-only")
            runner.generate_run(path, max_calls=1, key="fake-fixture-only")

    def test_readonly_snapshot_never_repairs_mirror_or_creates_lock(self):
        for runner in self.runners:
            with self.subTest(runner=runner.__name__), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                self.failed_fixture(runner, path)
                (path / "failures.jsonl").unlink()
                (path / ".execution.lock").unlink()
                before = contents(path)
                with patch.object(runner, "_replace_rows", side_effect=AssertionError("unexpected write")), \
                     patch.object(runner, "exclusive_directory", side_effect=AssertionError("unexpected writer lock")):
                    state = runner.read_run_state_readonly(path)
                self.assertEqual({r["status"] for r in state["outcomes"].values()}, {"exhausted_failure"})
                self.assertEqual(set(state["frozen_rows"]), set(runner.FROZEN_ROWS))
                self.assertEqual(contents(path), before)
                self.assertFalse((path / "failures.jsonl").exists())
                self.assertFalse((path / ".execution.lock").exists())

    def test_readonly_snapshot_accepts_valid_short_mirror_but_rejects_contradiction(self):
        for runner in self.runners:
            with self.subTest(runner=runner.__name__), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                self.failed_fixture(runner, path)
                failures = runner.read_jsonl(path / "failures.jsonl")
                self.assertEqual(len(failures), 2)
                runner._replace_rows(path / "failures.jsonl", failures[:1])
                before = contents(path)
                runner.read_run_state_readonly(path)
                self.assertEqual(contents(path), before)
                changed = dict(failures[0], error_type="tampered")
                runner._replace_rows(path / "failures.jsonl", [changed])
                with self.assertRaisesRegex(ValueError, "failure journal differs"):
                    runner.read_run_state_readonly(path)

    def test_readonly_snapshot_uses_existing_lock_and_rejects_observable_race(self):
        for runner in self.runners:
            with self.subTest(runner=runner.__name__), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                self.prepare(runner, path)
                with runner.exclusive_directory(path), self.assertRaisesRegex(RuntimeError, "busy"):
                    runner.read_run_state_readonly(path)
                original = runner._request_outcomes_locked

                def change(*args, **kwargs):
                    value = original(*args, **kwargs)
                    target = path / "cases.jsonl"
                    target.write_text(target.read_text() + "\n")
                    return value

                with patch.object(runner, "_request_outcomes_locked", side_effect=change), \
                     self.assertRaisesRegex(RuntimeError, "changed during read-only"):
                    runner.read_run_state_readonly(path)


if __name__ == "__main__":
    unittest.main()
