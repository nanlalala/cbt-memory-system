"""Smoke the real bounded CLI, including explicit storage and run identity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "app" / "controlled_memory_demo.py"


class ControlledMemoryDemoTests(unittest.TestCase):
    def test_cli_replays_contracts_and_rejects_overwriting_the_run(self):
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / "demo.sqlite"
            command = [sys.executable, str(CLI), "--database", str(database),
                       "--approved-root", folder, "--run-id", "smoke-test", "--scripted"]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(len(report["scenarios"]), 5)
            self.assertGreaterEqual(report["assertion_count"], 22)
            self.assertTrue(all(item["passed"] for scenario in report["scenarios"]
                                for item in scenario["assertions"]))
            self.assertEqual(report["api_calls"], 0)
            self.assertEqual(report["mock_provider_calls"], 6)
            self.assertIs(report["live_model"], False)
            self.assertIs(report["effectiveness_claim"], False)
            encoded = json.dumps(report["protocol"], ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":"), allow_nan=False).encode()
            self.assertEqual(report["protocol_hash"], hashlib.sha256(encoded).hexdigest())
            before = hashlib.sha256(database.read_bytes()).hexdigest()
            repeat = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(repeat.returncode, 1)
            self.assertEqual(repeat.stdout, "")
            self.assertIn("existing runs are never overwritten", json.loads(repeat.stderr)["reason"])
            self.assertEqual(before, hashlib.sha256(database.read_bytes()).hexdigest())

    def test_cli_requires_explicit_storage_identity_and_scripted_mode(self):
        missing = subprocess.run([sys.executable, str(CLI), "--scripted"], cwd=ROOT,
                                 capture_output=True, text=True, timeout=30)
        self.assertEqual(missing.returncode, 2)
        self.assertIn("--database", missing.stderr)
        self.assertIn("--approved-root", missing.stderr)
        self.assertIn("--run-id", missing.stderr)
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / "must-not-open.sqlite"
            unscripted = subprocess.run([sys.executable, str(CLI), "--database", str(database),
                                        "--approved-root", folder, "--run-id", "smoke-test"],
                                       cwd=ROOT, capture_output=True, text=True, timeout=30)
            self.assertEqual(unscripted.returncode, 2)
            self.assertFalse(database.exists())


if __name__ == "__main__":
    unittest.main()
