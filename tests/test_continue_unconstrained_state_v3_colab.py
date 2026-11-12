"""Tiny process/metadata fixtures only; no real child, model, dataset or Colab."""
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "evaluation/cbt-memory-effect/continue_unconstrained_state_v3_colab.py"
SPEC = importlib.util.spec_from_file_location("owned_continuation_fixture_tests", PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def stat_line(state="S", ticks=M.BEFORE_START_TICKS):
    fields = ["0"] * 20
    fields[0], fields[19] = state, ticks
    return str(M.BEFORE_PID) + " (toy python) " + " ".join(fields)


class OwnedProcessTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.proc = Path(self.directory.name)
        self.pid = self.proc / str(M.BEFORE_PID)

    def make_process(self, *, state="S", ticks=M.BEFORE_START_TICKS, argv=M.BEFORE_ARGV):
        self.pid.mkdir()
        (self.pid / "stat").write_text(stat_line(state, ticks))
        (self.pid / "cmdline").write_bytes(b"\x00".join(a.encode() for a in argv) + b"\x00")
        (self.pid / "exe").symlink_to(Path(sys.executable).resolve())

    def test_exact_owned_process_and_absent_or_zombie_process_are_distinguished(self):
        self.assertEqual(M.owned_before_status(self.proc)["state"], "exited")
        self.make_process()
        self.assertEqual(M.owned_before_status(self.proc)["state"], "running")
        (self.pid / "stat").write_text(stat_line("Z"))
        self.assertEqual(M.owned_before_status(self.proc)["state"], "exited")

    def test_reused_pid_or_changed_command_is_rejected(self):
        self.make_process(ticks="999999")
        with self.assertRaisesRegex(M.StopContinuation, "pid_reused"):
            M.owned_before_status(self.proc)
        (self.pid / "stat").write_text(stat_line())
        (self.pid / "cmdline").write_bytes(b"python3\x00unrelated.py\x00")
        with self.assertRaisesRegex(M.StopContinuation, "command_or_executable_changed"):
            M.owned_before_status(self.proc)

    def test_missing_executable_for_live_pid_is_not_misreported_as_exit(self):
        self.make_process()
        (self.pid / "exe").unlink()
        with self.assertRaisesRegex(M.StopContinuation, "live_owned_process_identity"):
            M.owned_before_status(self.proc)

    def test_wait_timeout_does_not_kill_or_start_children(self):
        journal = SimpleNamespace(record=Mock())
        with patch.object(M, "owned_before_status", return_value={"state": "running"}):
            with patch.object(M.time, "monotonic", side_effect=[0, M.WAIT_SECONDS + 1]):
                with patch.object(M.subprocess, "Popen") as start:
                    with self.assertRaisesRegex(M.StopContinuation, "wait_120_minutes_expired"):
                        M.wait_for_before(journal, "a" * 64)
                    start.assert_not_called()


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.study = self.root / "invented-study"
        self.study.mkdir()
        self.logs = {phase: self.root / (phase + ".log") for phase in M.LOGS}
        for target, value in (("ROOT", self.study), ("CONTENT", self.root), ("LOGS", self.logs),
                              ("MARKER", self.root / "marker.json"), ("JOURNAL", self.root / "journal.jsonl")):
            mock = patch.object(M, target, value)
            mock.start()
            self.addCleanup(mock.stop)

    def test_child_intent_is_durable_before_spawn_and_existing_log_prevents_retry(self):
        journal = SimpleNamespace(record=Mock())
        child = SimpleNamespace(pid=12345, returncode=0, poll=lambda: 0)
        def spawn(argv, **kwargs):
            self.assertEqual(journal.record.call_args.args[0], "child_start_intent")
            self.assertEqual(journal.record.call_args.kwargs["phase"], "after")
            self.assertEqual(argv[-3:], ["run-side", "--side", "after"])
            return child
        with patch.object(M, "check_identity"), patch.object(M.subprocess, "Popen", side_effect=spawn) as start:
            M.run_phase("after", journal, "a" * 64)
            with self.assertRaisesRegex(M.StopContinuation, "existing_phase_log"):
                M.run_phase("after", journal, "a" * 64)
            self.assertEqual(start.call_count, 1)
        self.assertEqual([call.args[0] for call in journal.record.call_args_list],
                         ["child_start_intent", "child_started", "child_exited"])

    def test_unknown_child_start_keeps_log_and_never_retries(self):
        journal = SimpleNamespace(record=Mock())
        with patch.object(M, "check_identity"), patch.object(M.subprocess, "Popen", side_effect=OSError("toy failure")) as start:
            with self.assertRaises(OSError):
                M.run_phase("after", journal, "a" * 64)
            with self.assertRaisesRegex(M.StopContinuation, "existing_phase_log"):
                M.run_phase("after", journal, "a" * 64)
            self.assertEqual(start.call_count, 1)
        self.assertTrue(self.logs["after"].exists())
        self.assertEqual(journal.record.call_args.args[0], "child_start_or_wait_failed_or_unknown")

    def test_verification_failure_stops_before_after_and_marker_remains_exclusive(self):
        with patch.object(M, "check_identity"), patch.object(M, "owned_before_status", return_value={"state": "exited"}):
            with patch.object(M, "wait_for_before"), patch.object(M, "run_phase", side_effect=M.StopContinuation("toy authentication failed")) as phases:
                with self.assertRaisesRegex(M.StopContinuation, "authentication failed"):
                    M.run("a" * 64)
                self.assertEqual([call.args[0] for call in phases.call_args_list], ["verify-before"])
                with self.assertRaisesRegex(M.StopContinuation, "no_automatic_resume"):
                    M.run("a" * 64)
        records = [json.loads(x) for x in M.JOURNAL.read_text().splitlines()]
        self.assertEqual(records[-1]["event"], "stopped_no_kill_no_automatic_retry")
        self.assertEqual(records[-1]["processes_killed"], 0)
        previous = None
        for record in records:
            self.assertEqual(record["previous_record_sha256"], previous)
            body = {k: v for k, v in record.items() if k != "record_sha256"}
            expected = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            self.assertEqual(record["record_sha256"], expected)
            previous = expected

    def test_complete_sequence_authenticates_before_scoring_and_records_completion(self):
        def phase(name, journal, source):
            if name == "score":
                reports = self.study / "reports"
                reports.mkdir()
                (reports / "summary.json").write_text(json.dumps({"version": "unconstrained-state-development-study-v3",
                    "main_logical_rows": 96, "new_study_unique_generations": 57, "study_cap": 57}))
        with patch.object(M, "check_identity"), patch.object(M, "owned_before_status", return_value={"state": "exited"}):
            with patch.object(M, "wait_for_before"), patch.object(M, "run_phase", side_effect=phase) as phases:
                with redirect_stdout(io.StringIO()):
                    M.run("a" * 64)
        self.assertEqual([call.args[0] for call in phases.call_args_list], ["verify-before", "after", "verify-after", "score"])
        records = [json.loads(x) for x in M.JOURNAL.read_text().splitlines()]
        self.assertEqual(records[-1]["event"], "completed")
        self.assertEqual(records[-1]["new_budget"], 0)

    def test_host_guard_precedes_marker_or_child_access(self):
        with patch.object(M, "require_hosted", side_effect=M.StopContinuation("not hosted")):
            with patch.object(M.subprocess, "Popen") as child:
                with self.assertRaisesRegex(M.StopContinuation, "not hosted"):
                    M.run("a" * 64)
                child.assert_not_called()
        self.assertFalse(M.MARKER.exists())


if __name__ == "__main__":
    unittest.main()
