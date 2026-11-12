"""Tiny invented metadata only: no model, tokenizer, data, provider or Colab."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "evaluation/cbt-memory-effect/run_prefill_pilot_v2_colab.py"
spec = importlib.util.spec_from_file_location("new_prefill_runtime_tests", PATH)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def stamp(value):
    return {**value, "record_sha256": R.reader.digest(value)}


def tiny_ledger(n, *, missing_last=False):
    rows = []
    for i in range(n):
        rows.append(stamp({"event": "generation_intent", "canonical_key": str(i)}))
        if not (missing_last and i == n - 1):
            rows.append(stamp({"event": "generation_outcome", "canonical_key": str(i), "status": "published"}))
    return rows


class NewPrefillRuntimeMetadataTests(unittest.TestCase):
    def test_independent_61_cap_and_unknown_intent_fail_closed(self):
        with patch.object(R.reader, "read_rows", return_value=tiny_ledger(61)):
            intents, outcomes = R.ledger("invented")
            self.assertEqual((len(intents), len(outcomes)), (61, 61))
        with patch.object(R.reader, "read_rows", return_value=tiny_ledger(62)):
            with self.assertRaisesRegex(ValueError, "61_budget"):
                R.ledger("invented")
        with patch.object(R.reader, "read_rows", return_value=tiny_ledger(1, missing_last=True)):
            with self.assertRaisesRegex(RuntimeError, "no_automatic_retry"):
                R.ledger("invented")

    def test_canonical_isolation_and_within_study_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "study_protocol.json"
            protocol.write_text('{"study":"new"}')
            args = ({"model": "invented"}, {"grammar": "v2"}, {"messages": []})
            one = R.generation_key(root, *args)
            self.assertEqual(one, R.generation_key(root, *args))
            self.assertNotEqual(one, R.generation_key(root, args[0], {"grammar": "free"}, args[2]))
            protocol.write_text('{"study":"another"}')
            self.assertNotEqual(one, R.generation_key(root, *args))

    def test_old_cache_is_rejected_before_any_cache_file_access(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(R.reader, "_cache") as read_cache:
            with self.assertRaisesRegex(ValueError, "old_or_foreign"):
                R.cached(Path(directory) / "new", {"cache_path": str(Path(directory) / "old/canonical/x.json")})
            read_cache.assert_not_called()

    def test_shared_receipt_not_double_counted_and_missing_usage_not_zero(self):
        result = R.cost({"x": {"status": "published", "usage": {"total_tokens": 9}, "generation_wall_seconds": 2}}, ["x", "x"])
        self.assertEqual(result["unique_admitted_attempts"], 1)
        self.assertEqual(result["observed_tokens"]["total_tokens"], 9)
        self.assertEqual(result["answer_path_wall_seconds_sum"], 2)
        self.assertIsNone(result["observed_tokens"]["prompt_tokens"])
        self.assertEqual(result["usage_missing_attempts"]["prompt_tokens"], 1)
        failed = R.cost({"x": {"status": "unknown", "usage": None}}, ["x"])
        self.assertIsNone(failed["observed_tokens"]["total_tokens"])
        self.assertIsNone(failed["answer_path_wall_seconds_sum"])

    def test_host_guard_runs_before_initialization_reads_or_creates_files(self):
        with tempfile.TemporaryDirectory() as directory:
            new = Path(directory) / "new"
            with patch.object(R.reader, "require_colab", side_effect=RuntimeError("not_colab")):
                with self.assertRaisesRegex(RuntimeError, "not_colab"):
                    R.initialize(new, old_root=Path(directory) / "old")
            self.assertFalse(new.exists())


if __name__ == "__main__":
    unittest.main()
