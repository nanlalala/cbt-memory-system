from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("local_memory_demo", ROOT / "app/memory_demo.py")
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


class LocalDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = demo.SQLiteMemoryStore()
        self.addCleanup(self.store.close)
        self.session = demo.DemoSession(self.store, run_id="test-demo")

    def test_scripted_walkthrough_is_only_a_software_smoke(self) -> None:
        result = demo.scripted_walkthrough(self.store, run_id="scripted-demo")
        self.assertTrue(result["smoke_passed"])
        self.assertEqual(result["kind"], "synthetic_software_smoke")
        self.assertFalse(result["clinical_or_benchmark_result"])
        self.assertFalse(result["live_model_or_network_used"])
        self.assertEqual(result["records_after_delete"], 0)
        events = self.store.audit_for_user("synthetic-demo-scripted-demo")
        consents = [event for event in events if event["action"] == "demo_consent_event"]
        self.assertEqual(len(consents), 2)
        self.assertTrue(all(event["details"]["input_source"] == "scripted_fixture" for event in consents))

    def test_typed_confirmation_is_required_and_binds_audit_event(self) -> None:
        candidate = self.session.stage_initial()
        self.assertEqual(self.session.retrieve().memory_ids, ())
        with self.assertRaisesRegex(ValueError, "exact CONFIRM"):
            self.session.confirm_command(candidate.memory_id, "yes")
        self.assertEqual(self.store.get(self.session.user_id, candidate.memory_id).lifecycle_status, "candidate")
        confirmed = self.session.confirm_command(candidate.memory_id, f"CONFIRM {candidate.memory_id}")
        self.assertEqual(confirmed.lifecycle_status, "active")
        events = self.store.audit_for_user(self.session.user_id)
        consent = next(event for event in events if event["action"] == "demo_consent_event")
        confirmation = next(event for event in events if event["action"] == "user_confirmed")
        self.assertEqual(consent["details"]["input_source"], "interactive_input")
        self.assertEqual(consent["details"]["consent_event_id"], confirmation["details"]["consent_event_id"])
        self.assertEqual(consent["details"]["confirmation_turn_ids"], confirmation["details"]["confirmation_turn_ids"])

    def test_correction_waits_for_confirmation_and_delete_removes_all_versions(self) -> None:
        initial = self.session.stage_initial()
        self.session.confirm_command(initial.memory_id, f"CONFIRM {initial.memory_id}")
        correction = self.session.stage_correction()
        self.assertEqual(correction.supersedes_memory_id, initial.memory_id)
        self.assertEqual(self.session.retrieve().memory_ids, (initial.memory_id,))
        self.session.confirm_command(correction.memory_id, f"CONFIRM {correction.memory_id}")
        self.assertEqual(self.session.retrieve().memory_ids, (correction.memory_id,))
        self.assertEqual(self.store.get(self.session.user_id, initial.memory_id).lifecycle_status, "superseded")
        with self.assertRaisesRegex(ValueError, "exact DELETE ALL"):
            self.session.delete_all_command("delete")
        self.assertEqual(len(self.session.records()), 2)
        self.assertEqual(len(self.session.delete_all_command(f"DELETE ALL {self.session.run_id}")), 2)
        self.assertEqual(self.session.records(), [])
        self.assertEqual(self.session.retrieve().memory_ids, ())
        audit = json.dumps(self.store.audit_for_user(self.session.user_id), ensure_ascii=False)
        self.assertNotIn(demo.INITIAL_QUOTE, audit)
        self.assertNotIn(demo.CORRECTED_CONTENT, audit)

    def test_cli_scripted_runs_without_api_key_or_model(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "app/memory_demo.py"), "--scripted"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        payload = json.loads(result.stdout)
        self.assertTrue(payload["smoke_passed"])
        self.assertEqual(payload["consent_mode"], "simulated_scripted_fixture_not_real_user_consent")


if __name__ == "__main__":
    unittest.main()
