"""Software-only recovery admission checks; no Colab, sockets or model calls."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[1] / "evaluation/architecture-rerun/cloud_resume_professional7.py"
SPEC = importlib.util.spec_from_file_location("professional7_recovery_unit", SOURCE)
resume = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resume)


class RecoveryAdmissionTests(unittest.TestCase):
    def test_missing_hosted_runtime_rejects_before_reading_or_restoring(self):
        with patch.object(resume, "_hosted", side_effect=RuntimeError("unit missing runtime")), \
                patch.object(resume, "_reader_status") as reader, patch.object(resume, "_parent") as parent:
            with self.assertRaises(RuntimeError):
                resume.run({})
            reader.assert_not_called()
            parent.assert_not_called()

    def test_a_repeated_invocation_only_returns_its_retained_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = {"state": "stopped", "nonce": "fixture-trace", "automatic_retry": False}
            (root / "entry.json").write_text(json.dumps(original))
            with patch.object(resume, "ROOT", root), patch.object(resume, "_hosted"), \
                    patch.object(resume, "_reader_status") as reader, patch.object(resume, "_parent") as parent:
                self.assertEqual(resume.run({}), original)
                reader.assert_not_called()
                parent.assert_not_called()

    def test_partial_reader_tail_is_never_silently_repaired(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "outcomes.jsonl"
            original = b'{"status":"unknown"}'
            path.write_bytes(original)
            with self.assertRaises(ValueError):
                resume._rows(path)
            self.assertEqual(path.read_bytes(), original)

    def test_failed_source_only_worker_cannot_launch_grading_or_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "recovery"
            registrations = []
            gate = object()
            broker = SimpleNamespace(register_job=None, _request=gate)
            original_register = lambda name, limits: registrations.append((name, limits))
            class FixtureTransport:
                pass
            original_call = lambda self, payload: self.fail("fixture should never call provider")
            launch = Mock(return_value=SimpleNamespace(poll=lambda: 1, returncode=1, pid=12345))
            with patch.object(resume, "ROOT", root), patch.object(resume, "_hosted"), \
                    patch.object(resume, "_reader_status", return_value={"fixture_unit_only": True}), \
                    patch.object(resume, "_parent", return_value=(broker, FixtureTransport, original_call, original_register)), \
                    patch.object(resume, "_checkout"), \
                    patch.object(resume.runpy, "run_path", return_value={"launch_worker": launch}):
                result = resume.run({})
                self.assertEqual(result["state"], "stopped")
                self.assertFalse(result["automatic_retry"])
                self.assertEqual(registrations, [(resume.PREPARE_JOB, {resume.MODEL: 0})])
                self.assertEqual(launch.call_count, 1)
                self.assertEqual(launch.call_args.kwargs["stage"], "prepare")
                self.assertEqual(launch.call_args.kwargs["max_new_calls"], 0)
                self.assertIs(broker._request, gate)
                resume.run({})
                self.assertEqual(launch.call_count, 1)

    def test_timeout_never_claims_the_child_has_stopped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "recovery"
            broker = SimpleNamespace(register_job=None)
            class FixtureTransport:
                pass
            parent = (broker, FixtureTransport, lambda self, payload: None, lambda name, limits: None)
            launch = Mock(return_value=SimpleNamespace(pid=24680))
            with patch.object(resume, "ROOT", root), patch.object(resume, "_hosted"), \
                    patch.object(resume, "_reader_status", return_value={"fixture_unit_only": True}), \
                    patch.object(resume, "_parent", return_value=parent), patch.object(resume, "_checkout"), \
                    patch.object(resume.runpy, "run_path", return_value={"launch_worker": launch}), \
                    patch.object(resume, "_wait", side_effect=TimeoutError("fixture live child")):
                result = resume.run({})
                self.assertEqual(result["state"], "child_may_still_be_running")
                self.assertEqual(result["active_worker"]["pid"], 24680)
                self.assertFalse(result["automatic_retry"])
                resume.run({})
                self.assertEqual(launch.call_count, 1)


if __name__ == "__main__":
    unittest.main()
