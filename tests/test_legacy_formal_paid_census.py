"""Offline legacy-schema census: count all attempts without changing labels."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("legacy_census_unit", ROOT / "evaluation/architecture-rerun/cloud_continue_non_mab.py")
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


class LegacyFormalCensusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.protocol = {"version": "formal-new-readout-full-v1", "provider": driver.ANSWER_IDENTITY,
            "dataset_key": "locomo10", "case_count": 1540, "planned_answers": 4620,
            "preparation_chunks": 169, "maximum_attempts": 4958}
        self.protocol["protocol_sha256"] = driver._digest(self.protocol)
        self.sealed = self.protocol["protocol_sha256"]
        self.rows = []
        for number in range(3319):
            purpose = "extraction" if number < 169 else "support_check" if number < 337 else "answer"
            payload = {"model": driver.ANSWER_IDENTITY["model"],
                       "messages": [{"role": "user", "content": "invented fixture " + str(number)}]}
            wire = json.dumps(payload, allow_nan=False)
            self.rows.append({"protocol_sha256": self.sealed, "case_id": "invented",
                "purpose": purpose, "payload": payload, "serialized_request": wire,
                "key": driver._digest({"purpose": purpose, "payload": payload}),
                "wire_sha256": driver.sha256(wire.encode()).hexdigest(), "wire_chars": len(wire)})
        self.write()

    def write(self):
        (self.folder / "protocol.json").write_text(json.dumps(self.protocol))
        (self.folder / "intents.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.rows))

    def census(self):
        with patch.object(driver, "LEGACY_LOCOMO_PROTOCOL_SHA256", self.sealed):
            return driver._paid_census(self.folder, driver.ANSWER_IDENTITY, legacy_formal=True)

    def test_only_explicit_exact_legacy_schema_can_omit_fixture_field(self):
        self.assertEqual(self.census()["intents"], 3319)
        self.assertTrue(self.census()["legacy_missing_fixture_field"])
        with self.assertRaises(ValueError):
            driver._paid_census(self.folder, driver.ANSWER_IDENTITY)
        self.protocol["case_count"] = 1539
        self.protocol["protocol_sha256"] = driver._digest({k: v for k, v in self.protocol.items() if k != "protocol_sha256"})
        self.write()
        with self.assertRaises(ValueError):
            self.census()

    def test_all_original_attempts_are_counted_and_any_fixture_or_wire_tamper_rejected(self):
        original = dict(self.rows[-1])
        for change in ({"fixture_only": True}, {"wire_sha256": "0" * 64}, {"key": "0" * 64}, {"wire_chars": 1}):
            with self.subTest(change=change):
                self.rows[-1] = {**original, **change}
                self.write()
                with self.assertRaises(ValueError):
                    self.census()
        self.rows[-1] = original
        self.rows.pop()
        self.write()
        with self.assertRaises(ValueError):
            self.census()

    def test_protocol_checksum_and_duplicate_ledger_are_not_relaxed(self):
        self.protocol["maximum_attempts"] = 9999
        self.write()
        with self.assertRaises(ValueError):
            self.census()
        self.protocol["maximum_attempts"] = 4958
        self.rows[-1] = dict(self.rows[0])
        self.write()
        with self.assertRaises(ValueError):
            self.census()


if __name__ == "__main__":
    unittest.main()
