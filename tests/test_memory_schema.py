from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_DIR = ROOT / "modules" / "long-term-memory" / "schema"
sys.path.insert(0, str(SCHEMA_DIR))

from memory_models import MemoryRecord  # noqa: E402
from validate_memory_data import validate_examples, validate_test_set  # noqa: E402


EXAMPLES = ROOT / "modules" / "long-term-memory" / "examples" / "memory_records.json"
TEST_SET = ROOT / "evaluation" / "memory-retrieval" / "memory_extraction_test_set.jsonl"


class MemorySchemaTests(unittest.TestCase):
    def test_example_records_validate(self) -> None:
        self.assertEqual(validate_examples(EXAMPLES), 4)

    def test_multi_session_test_set_validates(self) -> None:
        self.assertEqual(validate_test_set(TEST_SET), 10)

    def test_inference_cannot_be_disguised_as_explicit_fact(self) -> None:
        data = json.loads(EXAMPLES.read_text(encoding="utf-8"))[2]
        data["origin"] = "explicit_user_statement"
        with self.assertRaisesRegex(ValueError, "agent_inference"):
            MemoryRecord.from_dict(data)

    def test_safety_memory_must_be_restricted(self) -> None:
        data = json.loads(EXAMPLES.read_text(encoding="utf-8"))[0]
        data["memory_type"] = "safety_signal"
        with self.assertRaisesRegex(ValueError, "restricted"):
            MemoryRecord.from_dict(data)

    def test_correction_requires_target(self) -> None:
        data = json.loads(EXAMPLES.read_text(encoding="utf-8"))[3]
        data["supersedes_memory_id"] = None
        with self.assertRaisesRegex(ValueError, "supersedes_memory_id"):
            MemoryRecord.from_dict(data)

    def test_emotion_intensity_range(self) -> None:
        data = json.loads(EXAMPLES.read_text(encoding="utf-8"))[1]
        data["structured_payload"]["intensity"] = 11
        with self.assertRaisesRegex(ValueError, "0 to 10"):
            MemoryRecord.from_dict(data)


if __name__ == "__main__":
    unittest.main()
