"""Synthetic, dependency-light ownership tests; no model or network calls."""

from __future__ import annotations

import multiprocessing
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation/end-to-end-dialogue"))
sys.path.insert(0, str(ROOT / "tests"))
from execution_lock import EvaluationDirectoryBusy, exclusive_directories, exclusive_directory
import run_structured_memory as runner
import reference_judge as judge
import test_structured_memory as fixture_module


def _competing_worker(directory, stage, entered, release, outcomes, calls):
    """A real separate process, with only the transport replaced by a mock."""
    def transport(request, settings, **kwargs):
        calls.put(request["prompt_sha256"])
        entered.set()
        if not release.wait(10):
            raise TimeoutError("synthetic worker synchronization timed out")
        return {"hypothesis": runner.verbatim_output(request["source"]) if stage == "extraction" else "blue",
                "provider_model": settings.model, "finish_reason": "stop",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                "latency_seconds": 0.01}
    try:
        with patch.object(runner, "require_colab"), patch.object(runner, "require_generation_cohort"), \
                patch.object(runner, "generate_response", side_effect=transport), \
                patch.dict(runner.os.environ, {"CBT_EVAL_API_KEY": "synthetic-never-log"}):
            result = (runner.execute_extraction if stage == "extraction" else runner.generate_run)(
                Path(directory), max_calls=1)
        outcomes.put(("completed", result))
    except EvaluationDirectoryBusy:
        outcomes.put(("busy_before_transport", None))
    except Exception as exc:
        outcomes.put(("unexpected", type(exc).__name__))


class DirectoryLockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_other_thread_is_rejected_and_lock_inode_survives(self):
        directory = self.root / "same"
        results = []
        def contender():
            try:
                with exclusive_directory(directory):
                    results.append("entered")
            except EvaluationDirectoryBusy:
                results.append("busy")
        with exclusive_directory(directory):
            inode = (directory / ".execution.lock").stat().st_ino
            thread = threading.Thread(target=contender)
            thread.start()
            thread.join(5)
            self.assertFalse(thread.is_alive())
        self.assertEqual(results, ["busy"])
        with exclusive_directory(directory):
            self.assertEqual((directory / ".execution.lock").stat().st_ino, inode)

    def test_exception_releases_lock_and_nested_acquisition_refuses(self):
        directory = self.root / "same"
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            with exclusive_directory(directory):
                with self.assertRaises(EvaluationDirectoryBusy):
                    with exclusive_directory(directory):
                        self.fail("reentrant lock must not silently share ownership")
                raise RuntimeError("synthetic failure")
        with exclusive_directory(directory):
            pass

    def test_distinct_directories_are_independent_and_partial_acquisition_releases(self):
        first, second = self.root / "a", self.root / "b"
        with exclusive_directory(second):
            with exclusive_directory(first):
                pass
            with self.assertRaises(EvaluationDirectoryBusy):
                with exclusive_directories(first, second):
                    self.fail("occupied directory must reject")
            with exclusive_directory(first):
                pass

    def test_judge_execution_and_scoring_refuse_before_state_or_transport(self):
        reader, output = self.root / "reader", self.root / "judge"
        with exclusive_directory(reader), patch.object(judge, "require_colab"), \
                patch.object(judge, "_load_judge") as load, patch.object(judge, "generate_response") as transport:
            with self.assertRaises(EvaluationDirectoryBusy):
                judge.execute_reference_judge(reader, output, max_calls=1)
            with self.assertRaises(EvaluationDirectoryBusy):
                judge.score_reference_judge(reader, output)
            load.assert_not_called()
            transport.assert_not_called()


class ConcurrentExecutionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.StructuredMemoryTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.context = multiprocessing.get_context("fork")

    def compete(self, directory, stage):
        entered, release = self.context.Event(), self.context.Event()
        outcomes, calls = self.context.Queue(), self.context.Queue()
        processes = []
        try:
            first = self.context.Process(target=_competing_worker,
                args=(str(directory), stage, entered, release, outcomes, calls))
            processes.append(first)
            first.start()
            self.assertTrue(entered.wait(8), "first worker must reserve before transport")
            second = self.context.Process(target=_competing_worker,
                args=(str(directory), stage, entered, release, outcomes, calls))
            processes.append(second)
            second.start()
            self.assertEqual(outcomes.get(timeout=8)[0], "busy_before_transport")
            second.join(5)
            self.assertEqual(second.exitcode, 0)
            self.assertIsInstance(calls.get(timeout=5), str)
            release.set()
            self.assertEqual(outcomes.get(timeout=8)[0], "completed")
            first.join(5)
            self.assertEqual(first.exitcode, 0)
            self.assertTrue(calls.empty(), "the rejected process must never invoke transport")
        finally:
            release.set()
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(5)
            outcomes.close()
            calls.close()

    def test_two_reader_processes_do_not_duplicate_request_and_resume_is_valid(self):
        run = self.fixture.prepared()
        self.compete(run, "response")
        protocol, prompts, _, complete = runner.load_run(run)
        self.assertEqual(len(complete), 1)
        attempts = runner.read_jsonl(run / "response_attempts.jsonl")
        self.assertEqual([row["attempt_index"] for row in attempts], [1])
        self.assertLessEqual(len(attempts), protocol["maximum_attempts_per_request"])
        with patch.object(runner, "require_colab"), patch.object(runner, "require_generation_cohort"), \
                patch.dict(runner.os.environ, {"CBT_EVAL_API_KEY": "synthetic-never-log"}), \
                patch.object(runner, "generate_response", return_value={
                    "hypothesis": "blue", "provider_model": "test-model", "finish_reason": "stop",
                    "usage": {}, "latency_seconds": 0.01}) as transport:
            resumed = runner.generate_run(run, max_calls=3)
        self.assertEqual(transport.call_count, 2)
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(len(runner.load_run(run)[3]), len(prompts))
        attempts = runner.read_jsonl(run / "response_attempts.jsonl")
        self.assertEqual(len({row["request_key"] for row in attempts}), 3)
        self.assertEqual([row["attempt_index"] for row in attempts], [1, 1, 1])

    def test_two_extraction_processes_share_budget_and_sequential_resume_works(self):
        cohort = self.fixture.cohort()
        output = self.fixture.base / "extraction"
        runner.prepare_extraction_requests(cohort, output, self.fixture.settings, self.fixture.api_base)
        self.compete(output, "extraction")
        attempts = runner.read_jsonl(output / "extraction_attempts.jsonl")
        self.assertEqual([row["attempt_index"] for row in attempts], [1])
        with patch.object(runner, "require_colab"), patch.object(runner, "require_generation_cohort"), \
                patch.object(runner, "generate_response") as transport:
            resumed = runner.execute_extraction(output, max_calls=1)
        transport.assert_not_called()
        self.assertEqual(resumed["completed"], 1)
        self.assertEqual(len(runner.read_jsonl(output / "extraction_responses.jsonl")), 1)

    def test_preparation_cannot_write_a_directory_held_by_an_executor(self):
        cohort = self.fixture.cohort()
        output = self.fixture.base / "run"
        with exclusive_directory(output), patch.object(runner, "_prepare_structured_run_locked") as prepare:
            with self.assertRaises(EvaluationDirectoryBusy):
                runner.prepare_structured_run(cohort, output, self.fixture.settings, self.fixture.api_base)
            prepare.assert_not_called()
        with exclusive_directory(output), patch.object(runner, "_prepare_extraction_requests_locked") as prepare:
            with self.assertRaises(EvaluationDirectoryBusy):
                runner.prepare_extraction_requests(cohort, output, self.fixture.settings, self.fixture.api_base)
            prepare.assert_not_called()

    def test_missing_or_changed_lock_identity_refuses_execution_before_transport(self):
        run = self.fixture.prepared()
        protocol = runner.json.loads((run / "protocol.json").read_text())
        self.assertIn("evaluation/end-to-end-dialogue/execution_lock.py", protocol["code_sha256"])
        protocol.pop("execution_lock_identity")
        (run / "protocol.json").write_text(runner.json.dumps(protocol))
        with patch.object(runner, "require_colab"), patch.object(runner, "generate_response") as transport:
            with self.assertRaisesRegex(ValueError, "execution lock"):
                runner.generate_run(run, max_calls=1)
            transport.assert_not_called()
        extraction = self.fixture.base / "extract"
        runner.prepare_extraction_requests(self.fixture.base / "cohort", extraction,
                                           self.fixture.settings, self.fixture.api_base)
        protocol = runner.json.loads((extraction / "extraction_protocol.json").read_text())
        protocol["execution_lock_identity"]["code_sha256"] = "0" * 64
        (extraction / "extraction_protocol.json").write_text(runner.json.dumps(protocol))
        with patch.object(runner, "require_colab"), patch.object(runner, "generate_response") as transport:
            with self.assertRaisesRegex(ValueError, "execution lock"):
                runner.execute_extraction(extraction, max_calls=1)
            transport.assert_not_called()


if __name__ == "__main__":
    unittest.main()
