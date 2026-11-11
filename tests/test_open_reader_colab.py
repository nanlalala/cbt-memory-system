"""Tiny fake-model fixtures; no weights, real dataset, network or credentials."""
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "evaluation/cbt-memory-effect/open_reader_colab.py"
spec = importlib.util.spec_from_file_location("open_reader_colab_tested", PATH)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class Vector:
    def __init__(self, values): self.values = values
    def __getitem__(self, key):
        result = self.values[key]
        return Vector(result) if isinstance(key, slice) else result
    def tolist(self): return self.values


class Tensor:
    device = "cpu"
    def __init__(self, n=3): self.n = n
    @property
    def shape(self): return (1, self.n)
    def tolist(self): return [list(range(self.n))]
    def to(self, device): return self


class Tokenizer:
    eos_token_id = 99
    chat_template = "fixture-chat-template"
    def __init__(self, n=3): self.n = n
    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": Tensor(self.n), "attention_mask": Tensor(self.n)}
    def decode(self, ids, **kwargs): return "yes"


class Model:
    device = "cpu"
    training = False
    generation_config = SimpleNamespace(eos_token_id=99, to_dict=lambda: {"eos_token_id": 99})
    def __init__(self): self.calls = 0
    def eval(self): self.training = False; return self
    def parameters(self): yield SimpleNamespace(dtype="torch.float32")
    def generate(self, *, input_ids, **kwargs):
        self.calls += 1
        return [Vector(list(range(input_ids.n)) + [11, 99])]


class Gates:
    def __init__(self, reject_at=None):
        self.preflights = 0; self.begins = []; self.finishes = []; self.reject_at = reject_at
    def identity(self): return {"fixture_gate_code_sha256": "a" * 64}
    def preflight(self, row):
        self.preflights += 1
        if self.preflights == self.reject_at: raise PermissionError("fixture withdrawal")
        return {"current": True}
    def begin(self, row):
        self.begins.append((row["case_id"], row["arm"]))
        return {"attempt_id": str(len(self.begins)), "replayed": False}
    def finish(self, row, admission, **kwargs):
        self.finishes.append(kwargs)
        return {"publishable": True}


class OpenReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.model_root = self.root / "model"; self.model_root.mkdir()
        r._write_new(self.model_root / "load_protocol.json", {
            "model_id": r.MODEL_ID, "model_revision": r.MODEL_REVISION,
            "maximum_generated_answers": 64, "maximum_input_tokens": 4096,
            "max_new_tokens": 64, "sampling": False, "external_inference_calls": 0})
        r._write_new(self.model_root / "model_load_receipt.json", {
            "stage": "open_reader_model_loaded", "model_revision": r.MODEL_REVISION,
            "device": "cpu", "dtype": "torch.float32"})
        r._write_new(self.model_root / "model_files.json", {"revision": r.MODEL_REVISION, "files": []})
        self.fake_torch = patch.dict(sys.modules, {"torch": SimpleNamespace(inference_mode=nullcontext)})
        self.fake_torch.start()
    def tearDown(self):
        self.fake_torch.stop(); self.tmp.cleanup()
    def row(self, case="one", arm="raw_history_rag", text="tiny invented question"):
        messages = [{"role": "system", "content": "Answer from history."},
                    {"role": "user", "content": text}]
        payload = {"messages": messages, "max_tokens": 64}
        return {"case_id": case, "arm": arm, "messages": messages,
                "request_payload": payload, "request_sha256": r.digest(payload)}
    def prepare(self, rows, name="prepared"):
        folder = self.root / name; folder.mkdir()
        (folder / "requests.jsonl").write_text("".join(r.canonical(row) + "\n" for row in rows))
        r._write_new(folder / "protocol.json", {"requests_file_sha256": r.file_hash(folder / "requests.jsonl")})
        return folder
    def run_reader(self, prepared, model=None, tokenizer=None, gates=None, output="output", cap=64):
        return r.run_open_reader(prepared, model or Model(), tokenizer or Tokenizer(),
            self.model_root, self.root / output, max_new_generations=cap,
            gates=gates or Gates(), fixture_only=True)

    def test_identical_arms_share_one_generation_but_gate_every_row(self):
        prepared = self.prepare([self.row(), self.row(arm="managed_memory_rag")])
        model, gates = Model(), Gates()
        result = self.run_reader(prepared, model=model, gates=gates)
        self.assertTrue(result["complete"])
        self.assertEqual((model.calls, result["global_generated_attempts"]), (1, 1))
        self.assertEqual(result["generation_cost_owning_run"]["observed_tokens"]["total_tokens"], 5)
        self.assertEqual(len(gates.begins), 2)
        self.assertEqual(len(gates.finishes), 2)
        answers = r.read_rows(self.root / "output/answers_unscored.jsonl")
        self.assertEqual([a["canonical_generation_shared"] for a in answers], [False, True])
        self.assertTrue(all(a["response_text"] == "yes" and not a["scored"] for a in answers))

    def test_bounded_resume_completes_without_repeating_existing_generation(self):
        prepared = self.prepare([self.row(), self.row(arm="managed_memory_rag"),
                                 self.row(case="two", text="another invented question")])
        model = Model()
        first = self.run_reader(prepared, model=model, cap=1)
        self.assertEqual((first["logical_completed"], model.calls), (2, 1))
        second = self.run_reader(prepared, model=model, cap=1)
        self.assertTrue(second["complete"])
        self.assertEqual(model.calls, 2)
        third = self.run_reader(prepared, model=model, cap=1)
        self.assertTrue(third["complete"]); self.assertEqual(model.calls, 2)

    def test_missing_gate_cannot_create_output(self):
        prepared = self.prepare([self.row()])
        with self.assertRaisesRegex(ValueError, "source_gate"):
            r.run_open_reader(prepared, Model(), Tokenizer(), self.model_root,
                              self.root / "output", fixture_only=True)
        self.assertFalse((self.root / "output").exists())

    def test_unknown_global_intent_blocks_without_generation(self):
        prepared = self.prepare([self.row()]); model = Model()
        r._append(self.model_root / "generation_ledger.jsonl", {
            "event": "generation_intent", "canonical_key": "unknown"})
        with self.assertRaisesRegex(RuntimeError, "automatic_retry"):
            self.run_reader(prepared, model=model)
        self.assertEqual(model.calls, 0)

    def test_partial_journal_tail_blocks(self):
        prepared = self.prepare([self.row()]); model = Model()
        (self.model_root / "generation_ledger.jsonl").write_bytes(b'{"event":')
        with self.assertRaisesRegex(RuntimeError, "incomplete_journal"):
            self.run_reader(prepared, model=model)
        self.assertEqual(model.calls, 0)

    def test_changed_prepared_request_fails_before_generation(self):
        prepared = self.prepare([self.row()]); model = Model()
        with (prepared / "requests.jsonl").open("a") as stream: stream.write("\n")
        with self.assertRaisesRegex(ValueError, "requests_file_changed"):
            self.run_reader(prepared, model=model)
        self.assertEqual(model.calls, 0)

    def test_withdrawal_after_generation_discards_body_and_blocks_retry(self):
        prepared = self.prepare([self.row()]); model = Model()
        with self.assertRaises(PermissionError):
            self.run_reader(prepared, model=model, gates=Gates(reject_at=3))
        self.assertEqual(model.calls, 1)
        self.assertFalse((self.root / "output/answers_unscored.jsonl").exists())
        last = r.read_rows(self.model_root / "generation_ledger.jsonl")[-1]
        self.assertEqual(last["status"], "stale_discarded")
        self.assertIsNone(last["response_text"])
        with self.assertRaisesRegex(RuntimeError, "automatic_retry"):
            self.run_reader(prepared, model=model)
        self.assertEqual(model.calls, 1)

    def test_global_ceiling_applies_across_run_directories(self):
        prepared = self.prepare([self.row()]); model = Model()
        for n in range(64):
            key = f"prior-{n}"
            r._append(self.model_root / "generation_ledger.jsonl", {"event": "generation_intent", "canonical_key": key})
            r._append(self.model_root / "generation_ledger.jsonl", {
                "event": "generation_outcome", "canonical_key": key, "status": "published"})
        result = self.run_reader(prepared, model=model)
        self.assertFalse(result["complete"])
        self.assertEqual((result["logical_completed"], model.calls), (0, 0))

    def test_cache_tampering_is_not_reused(self):
        model = Model()
        self.run_reader(self.prepare([self.row()]), model=model)
        cache = next((self.root / "output/canonical").glob("*.json"))
        cache.write_text('{}\n')
        another = self.prepare([self.row(case="another")], name="another-prepared")
        with self.assertRaisesRegex(ValueError, "canonical_cache_changed"):
            self.run_reader(another, model=model, output="another-output")
        self.assertEqual(model.calls, 1)

    def test_generation_cannot_receive_assessor_fields(self):
        row = self.row(); row["gold"] = "fixture-only reference"
        with self.assertRaisesRegex(ValueError, "assessor_fields"):
            self.run_reader(self.prepare([row]))

    def test_gpu_or_training_mode_is_rejected(self):
        prepared = self.prepare([self.row()]); model = Model(); model.device = "cuda"
        with self.assertRaisesRegex(ValueError, "cpu_float32"):
            self.run_reader(prepared, model=model)

    def test_cache_redundant_receipt_fields_reconcile(self):
        path = self.root / "canonical/key.json"
        value = {"canonical_key": "key", "response_text": "yes",
                 "response_sha256": r.hashlib.sha256(b"yes").hexdigest(),
                 "usage": {"total_tokens": 5}, "finish_reason": "eos"}
        r._write_new(path, value)
        outcome = {"canonical_key": "key", "cache_path": str(path), "cache_sha256": r.file_hash(path),
                   "response_sha256": value["response_sha256"], "usage": {"total_tokens": 99},
                   "finish_reason": "eos"}
        with self.assertRaisesRegex(ValueError, "receipt_disagreement"):
            r._cache(outcome)

    def test_reuse_in_another_run_has_zero_new_cost(self):
        model = Model()
        self.run_reader(self.prepare([self.row()]), model=model)
        prepared = self.prepare([self.row(case="another")], name="another-prepared")
        result = self.run_reader(prepared, model=model, output="another-output")
        self.assertEqual(model.calls, 1)
        self.assertEqual(result["generation_cost_owning_run"]["canonical_attempts"], 0)
        self.assertEqual(result["referenced_canonical_cost_nonadditive"]["canonical_attempts"], 1)
        self.assertEqual(result["reused_canonical_receipts_from_other_runs"], 1)
        model.device = "cpu"; model.training = True
        with self.assertRaisesRegex(ValueError, "cpu_float32"):
            self.run_reader(prepared, model=model)

    def test_live_guard_runs_before_creating_output(self):
        with patch.object(r, "require_colab", side_effect=RuntimeError("genuine_hosted_colab_required")):
            with self.assertRaisesRegex(RuntimeError, "genuine_hosted_colab"):
                r.run_open_reader(self.root / "does-not-exist", Model(), Tokenizer(), self.model_root,
                                  self.root / "output", gates=Gates())
        self.assertFalse((self.root / "output").exists())

    def test_cli_guard_precedes_cache_or_gates(self):
        with patch.object(r, "require_colab", side_effect=RuntimeError("genuine_hosted_colab_required")):
            with self.assertRaisesRegex(RuntimeError, "genuine_hosted_colab"):
                r.main(["run", "--prepared-dir", str(self.root / "absent"),
                        "--model-root", str(self.model_root), "--output", str(self.root / "output")])
        self.assertFalse((self.model_root / "model_load_events.jsonl").exists())

    def test_cli_rejects_cached_foreign_planner(self):
        foreign = SimpleNamespace(__file__="/unrelated/controlled_open_memory.py")
        with patch.dict(sys.modules, {"controlled_open_memory": foreign}):
            with self.assertRaisesRegex(RuntimeError, "foreign_controlled"):
                r._gate_module()

    def test_cli_accepts_official_protocol_without_optional_fixture_field(self):
        folder = PATH.parent
        sys.path.insert(0, str(folder))
        import controlled_open_memory as planner
        prepared = self.root / "official-cli-fixture"
        planner.prepare(prepared, fixture_only=True, case_limit=1)
        protocol, rows, request_hash = r._requests(prepared)
        self.assertNotIn("fixture_only", protocol)
        gates = planner.make_gates(prepared)
        proxy = SimpleNamespace(make_gates=lambda _folder: gates)
        argv = ["run", "--prepared-dir", "/content/fixture-prepared",
                "--model-root", "/content/fixture-model", "--output", "/content/fixture-output"]
        result = {"run_protocol_sha256": "fixture", "logical_completed": 2, "complete": True}
        with patch.object(r, "require_colab"), patch.object(r, "_gate_module", return_value=proxy), \
                patch.object(r, "_requests", return_value=(protocol, rows, request_hash)), \
                patch.object(r, "_locked", return_value=nullcontext()), \
                patch.object(r, "_global_state", return_value=({}, {})), \
                patch.object(r, "_load_cached_model", return_value=(Model(), Tokenizer(), {"setup_id": "fixture", "setup_seconds": 0})) as load, \
                patch.object(r, "run_open_reader", return_value=result) as reader, patch.object(r, "_append"):
            self.assertEqual(r.main(argv), result)
            load.assert_called_once(); reader.assert_called_once()
        # An explicit fixture marker still cannot enter the real CLI.
        with patch.object(r, "require_colab"), patch.object(r, "_gate_module", return_value=proxy), \
                patch.object(r, "_requests", return_value=({**protocol, "fixture_only": True}, rows, request_hash)), \
                patch.object(r, "_load_cached_model") as load:
            with self.assertRaisesRegex(ValueError, "nonfixture"):
                r.main(argv)
            load.assert_not_called()

    def test_offline_reload_checks_exact_bytes_and_preserves_old_receipt(self):
        snapshot = self.root / "cache" / r.MODEL_REVISION
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_bytes(b'{"fixture":true}')
        (snapshot / "model.safetensors").write_bytes(b'tiny-invented-fixture-not-model-weights')
        files = [{"name": p.name, "bytes": p.stat().st_size, "sha256": r.file_hash(p)}
                 for p in snapshot.iterdir()]
        (self.model_root / "model_files.json").write_text(r.canonical({"revision": r.MODEL_REVISION, "files": files}) + "\n")
        old_receipt = (self.model_root / "model_load_receipt.json").read_bytes()
        calls = []
        model, tokenizer = Model(), Tokenizer()
        model.config = SimpleNamespace(_name_or_path=str(snapshot))
        tokenizer.name_or_path = str(snapshot)
        def download(*args, **kwargs): calls.append(("download", kwargs)); return str(snapshot)
        def load_model(*args, **kwargs): calls.append(("model", kwargs)); return model
        def load_tokenizer(*args, **kwargs): calls.append(("tokenizer", kwargs)); return tokenizer
        libs = {"huggingface_hub": SimpleNamespace(snapshot_download=download),
                "torch": SimpleNamespace(float32="torch.float32", set_num_threads=lambda n: None),
                "transformers": SimpleNamespace(
                    AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
                    AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer))}
        with patch.object(r, "require_colab"), patch.dict(sys.modules, libs), patch.dict(r.os.environ):
            actual_model, actual_tokenizer, receipt = r._load_cached_model(self.model_root)
        self.assertIs(actual_model, model); self.assertIs(actual_tokenizer, tokenizer)
        self.assertTrue(all(kwargs.get("local_files_only") is True for _, kwargs in calls))
        self.assertFalse(calls[0][1]["token"])
        self.assertFalse(calls[1][1]["trust_remote_code"])
        self.assertEqual(receipt["answer_generations_during_setup"], 0)
        self.assertEqual((self.model_root / "model_load_receipt.json").read_bytes(), old_receipt)
        events = r.read_rows(self.model_root / "model_load_events.jsonl")
        self.assertEqual([e["event"] for e in events], ["model_reload_intent", "model_reload_completed"])


if __name__ == "__main__": unittest.main()
