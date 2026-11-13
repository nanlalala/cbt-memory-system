"""Pure metadata and invented-byte checks; never emulate hosted Colab."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation/architecture-rerun"))
SPEC = importlib.util.spec_from_file_location("lme_source_recovery_test",
    ROOT / "evaluation/architecture-rerun/longmemeval_source_recovery.py")
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


class LongMemEvalSourceRecoveryTests(unittest.TestCase):
    def test_original_full_source_pin_and_finite_original_budget(self):
        pin = recovery.source_pin()
        self.assertEqual(pin["bytes"], 277383467)
        self.assertIn(recovery.REVISION, pin["source"])
        plan = recovery.finite_plan({"planned_answers":1500, "preparation_chunks":18866})
        self.assertEqual((plan["maximum_deepseek_calls"], plan["maximum_qwen_calls"],
                          plan["maximum_total_calls"]), (39232, 1650, 40882))
        self.assertFalse(plan["paid_execution_authorized"])
        self.assertEqual(plan["new_provider_attempts"], 0)
        altered = recovery.sources.formal.load_plan(recovery.sources.formal.DEFAULT_PLAN)
        altered["datasets"][recovery.DATASET]["source"] = "https://example.invalid/oracle.json"
        with patch.object(recovery.sources.formal, "load_plan", return_value=altered):
            with self.assertRaisesRegex(ValueError, "source pin differs"):
                recovery.source_pin()

    def test_existing_three_reviewed_changes_keep_original_v1_behavior(self):
        original = {"code_sha256":recovery.full.code_hashes()}
        for name, (old_hash, _) in recovery.continuation.V1_AST.items():
            original["code_sha256"][name] = old_hash
        original["code_sha256"][recovery.continuation.GUARD_PATH] = recovery.continuation.OLD_GUARD_HASH
        changes = recovery.scientific_compatibility(original)
        self.assertEqual(set(changes), set(recovery.continuation.V1_AST) | {recovery.continuation.GUARD_PATH})
        for name, (_, ast_hash) in recovery.continuation.V1_AST.items():
            self.assertEqual(changes[name]["original_v1_ast_sha256"], ast_hash)
        original["code_sha256"]["evaluation/architecture-rerun/formal_full.py"] = "0"*64
        with self.assertRaisesRegex(ValueError, "original rendering code changed"):
            recovery.scientific_compatibility(original)

    def test_source_and_roster_drift_or_paid_attempt_cannot_be_relabelled_source_only(self):
        cases = [{"case_id":"invented-only"}]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "histories.json").write_text("invented frozen history bytes\n")
            (folder / "cases.json").write_text(json.dumps(cases))
            original = {"files_sha256":{"histories.json":recovery.sources.hash_file(folder / "histories.json")},
                        "code_sha256":recovery.full.code_hashes(), "created_at_utc":"old",
                        "cases_sha256":recovery.digest(cases)}
            rebuilt = {**original, "created_at_utc":"new"}
            rebuilt["protocol_sha256"] = recovery.digest(rebuilt)
            (folder / "protocol.json").write_text(json.dumps(rebuilt))
            with patch.object(recovery, "CASES_SHA256", recovery.digest(cases)):
                self.assertEqual(recovery.verify_rebuilt(folder, original), rebuilt)
                (folder / "histories.json").write_text("changed\n")
                with self.assertRaisesRegex(ValueError, "bytes differ"):
                    recovery.verify_rebuilt(folder, original)
                (folder / "histories.json").write_text("invented frozen history bytes\n")
                (folder / "cases.json").write_text(json.dumps([{"case_id":"different"}]))
                with self.assertRaisesRegex(ValueError, "case roster differs"):
                    recovery.verify_rebuilt(folder, original)
                (folder / "cases.json").write_text(json.dumps(cases))
                (folder / "intents.jsonl").write_text("{}\n")
                with self.assertRaisesRegex(ValueError, "provider attempts"):
                    recovery.verify_rebuilt(folder, original)


if __name__ == "__main__":
    unittest.main()
