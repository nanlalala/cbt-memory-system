"""Validate Memory Schema v1 examples and extraction test cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from memory_models import MemoryRecord, MemoryType


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXAMPLES = ROOT / "examples" / "memory_records.json"
DEFAULT_TEST_SET = ROOT.parents[1] / "evaluation" / "memory-retrieval" / "memory_extraction_test_set.jsonl"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_examples(path: Path) -> int:
    records = load_json(path)
    if not isinstance(records, list) or not records:
        raise ValueError("example file must contain a non-empty JSON array")
    parsed = [MemoryRecord.from_dict(item) for item in records]
    ids = [item.memory_id for item in parsed]
    if len(ids) != len(set(ids)):
        raise ValueError("example memory_id values must be unique")
    return len(parsed)


def validate_case(case: dict[str, Any]) -> None:
    required = {"case_id", "title", "sessions", "expected_memories", "must_not_store"}
    missing = sorted(required - case.keys())
    if missing:
        raise ValueError(f"{case.get('case_id', '<unknown>')}: missing {', '.join(missing)}")
    if len(case["sessions"]) < 2:
        raise ValueError(f"{case['case_id']}: each case must contain at least two sessions")
    turn_ids: set[str] = set()
    for session in case["sessions"]:
        if not session.get("session_id") or not session.get("timestamp") or not session.get("turns"):
            raise ValueError(f"{case['case_id']}: malformed session")
        for turn in session["turns"]:
            if turn.get("role") not in {"user", "assistant"} or not turn.get("turn_id"):
                raise ValueError(f"{case['case_id']}: malformed turn")
            if turn["turn_id"] in turn_ids:
                raise ValueError(f"{case['case_id']}: duplicate turn_id {turn['turn_id']}")
            turn_ids.add(turn["turn_id"])
    for expected in case["expected_memories"]:
        if expected.get("memory_type") not in {item.value for item in MemoryType}:
            raise ValueError(f"{case['case_id']}: unknown expected memory_type")
        if not set(expected.get("source_turn_ids", [])).issubset(turn_ids):
            raise ValueError(f"{case['case_id']}: expected memory points to an unknown turn")


def validate_test_set(path: Path) -> int:
    cases = load_jsonl(path)
    if not cases:
        raise ValueError("test set cannot be empty")
    for case in cases:
        validate_case(case)
    ids = [case["case_id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case_id values must be unique")
    return len(cases)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--test-set", type=Path, default=DEFAULT_TEST_SET)
    args = parser.parse_args()
    example_count = validate_examples(args.examples)
    case_count = validate_test_set(args.test_set)
    print(f"Validated {example_count} memory records and {case_count} multi-session cases.")


if __name__ == "__main__":
    main()
