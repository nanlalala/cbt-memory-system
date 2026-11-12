"""Tiny invented metadata only: no model, tokenizer, data, provider or Colab."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from contextlib import ExitStack

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "evaluation/cbt-memory-effect/unconstrained_state_runtime_v3.py"
spec = importlib.util.spec_from_file_location("new_unconstrained_runtime_tests", PATH)
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


class NewUnconstrainedRuntimeMetadataTests(unittest.TestCase):
    def test_independent_57_cap_and_unknown_intent_fail_closed(self):
        with patch.object(R.reader, "read_rows", return_value=tiny_ledger(57)):
            intents, outcomes = R.ledger("invented")
            self.assertEqual((len(intents), len(outcomes)), (57, 57))
        with patch.object(R.reader, "read_rows", return_value=tiny_ledger(58)):
            with self.assertRaisesRegex(ValueError, "57_budget"):
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
                    R.initialize(new, old_root=Path(directory) / "old", failed_v2_root=Path(directory) / "failed-v2")
            self.assertFalse(new.exists())

    def test_new_root_cannot_be_nested_inside_either_historical_study(self):
        with tempfile.TemporaryDirectory() as directory:
            old, failed = Path(directory) / "old", Path(directory) / "failed"
            for prior in (old, failed):
                with patch.object(R, "guard"), patch.object(R.reader, "read_json") as read:
                    with self.assertRaisesRegex(FileExistsError, "new_independent"):
                        R.initialize(prior / "new", old_root=old, failed_v2_root=failed)
                    read.assert_not_called()
                self.assertFalse((prior / "new").exists())


class UnconstrainedSourceBoundGenerationTests(unittest.TestCase):
    """Exercise journals/cache/adapter with a one-line invented fixture only."""
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "new-v3"
        self.root.mkdir()
        R.reader._write_new(self.root / "study_protocol.json", {"invented": "new-v3"})
        payload = {"messages": [{"role": "user", "content": "Invented toy fixture"}], "max_tokens": 64}
        self.rows = [{"case_id": "invented-1", "arm": arm,
            "request_payload": payload, "request_sha256": R.reader.digest(payload)} for arm in R.ARMS]
        self.result = {"response_text": "answer: 12", "response_sha256": R.reader.hashlib.sha256(b"answer: 12").hexdigest(),
            "finish_reason": "eos", "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            "input_tensor_sha256": "invented-input-tensor"}
        self.gates = SimpleNamespace(identity=lambda: {"invented": "source-gates"},
            preflight=Mock(return_value={"invented": "source-fence"}),
            begin=Mock(return_value={"replayed": False}),
            finish=Mock(return_value={"publishable": True}))

    def fixture_patches(self):
        stack = ExitStack()
        stack.enter_context(patch.object(R, "study"))
        stack.enter_context(patch.object(R.reader, "_model_identity", return_value={"invented": "fixed-model"}))
        stack.enter_context(patch.object(R, "input_preflight", return_value={
            "input_tokens": 3, "input_tensor_sha256": "invented-input-tensor"}))
        generate = stack.enter_context(patch.object(R.reader, "_generate", return_value=self.result))
        return stack, generate

    def test_one_raw_generation_is_shared_but_every_logical_row_has_own_source_fences(self):
        output = self.root / "answers"
        stack, generate = self.fixture_patches()
        with stack:
            summary = R.generate_rows(self.root, output, self.rows, None, None,
                purpose="invented-fixture", gates=self.gates)
            answers, _ = R.authenticate(self.root, output, self.rows)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(self.gates.begin.call_count, 2)
        self.assertEqual(self.gates.finish.call_count, 2)
        self.assertGreater(self.gates.preflight.call_count, 4)
        self.assertEqual(summary["owned_generation_cost"]["unique_admitted_attempts"], 1)
        self.assertEqual(len(answers), 2)
        for answer in answers:
            self.assertEqual(answer["response_text"], "answer: 12")
            self.assertEqual(answer["answer_adapter"]["canonical_response_text"], '{"answer":12}')
        _, outcomes = R.ledger(self.root)
        cache = R.cached(self.root, next(iter(outcomes.values())))
        self.assertEqual(cache["response_text"], "answer: 12")
        self.assertNotIn("answer_adapter", cache)

    def test_recomputed_adapter_rejects_tamper_even_with_internally_updated_checksums(self):
        output = self.root / "answers"
        stack, _ = self.fixture_patches()
        with stack:
            R.generate_rows(self.root, output, self.rows, None, None, purpose="invented", gates=self.gates)
            answers = R.reader.read_rows(output / "answers_unscored.jsonl")
            answers[0]["answer_adapter"]["canonical_response_text"] = '{"answer":13}'
            answers = [{key: value for key, value in a.items() if key != "record_sha256"} for a in answers]
            (output / "answers_unscored.jsonl").write_text("".join(R.reader.canonical(stamp(a)) + "\n" for a in answers))
            summary = R.reader.read_json(output / "completion_summary.json")
            summary["answers_sha256"] = R.reader.file_hash(output / "answers_unscored.jsonl")
            (output / "completion_summary.json").write_text(R.reader.canonical(summary))
            with self.assertRaisesRegex(ValueError, "adapter_differs"):
                R.authenticate(self.root, output, self.rows)

    def test_source_withdrawal_after_generation_discards_output_and_never_retries(self):
        output = self.root / "answers"
        def source_check(row):
            if self.gates.preflight.call_count == 4:
                raise PermissionError("invented-withdrawal")
            return {}
        self.gates.preflight.side_effect = source_check
        stack, generate = self.fixture_patches()
        with stack:
            with self.assertRaisesRegex(PermissionError, "withdrawal"):
                R.generate_rows(self.root, output, self.rows[:1], None, None, purpose="invented", gates=self.gates)
            with self.assertRaisesRegex(FileExistsError, "no_automatic_resume"):
                R.generate_rows(self.root, output, self.rows[:1], None, None, purpose="invented", gates=self.gates)
        self.assertEqual(generate.call_count, 1)
        self.assertFalse((output / "answers_unscored.jsonl").exists())
        events = R.reader.read_rows(self.root / "generation_ledger.jsonl")
        self.assertEqual([e["event"] for e in events], ["generation_intent", "generation_outcome"])
        self.assertEqual(events[-1]["status"], "stale_discarded")
        with self.assertRaisesRegex(RuntimeError, "no_automatic_retry"):
            R.ledger(self.root)


if __name__ == "__main__":
    unittest.main()
