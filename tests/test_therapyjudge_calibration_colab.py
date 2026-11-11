"""Invented fixture dialogues and fake models only; no real source or weights."""
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "evaluation/cbt-memory-effect/therapyjudge_calibration_colab.py"
spec = importlib.util.spec_from_file_location("therapyjudge_calibration_tested", PATH)
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
r = c.reader


class Vector:
    def __init__(self, values): self.values = values
    def __getitem__(self, key):
        value = self.values[key]
        return Vector(value) if isinstance(key, slice) else value
    def tolist(self): return self.values


class Tensor:
    device = "cpu"
    def __init__(self, n): self.n = n
    @property
    def shape(self): return (1, self.n)
    def tolist(self): return [list(range(self.n))]
    def to(self, device): return self


class Tokenizer:
    eos_token_id, chat_template = 99, "invented-fixture-template"
    def __init__(self, text=None, n=3):
        self.text = text if text is not None else json.dumps([3] * 11 + [0] * 4)
        self.n = n
    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": Tensor(self.n), "attention_mask": Tensor(self.n)}
    def decode(self, ids, **kwargs): return self.text


class Model:
    device, training = "cpu", False
    generation_config = SimpleNamespace(eos_token_id=99, to_dict=lambda: {"eos_token_id": 99})
    def __init__(self): self.calls = 0
    def parameters(self): yield SimpleNamespace(dtype="torch.float32")
    def generate(self, *, input_ids, **kwargs):
        self.calls += 1
        return [Vector(list(range(input_ids.n)) + [11, 99])]


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory(); self.root = Path(self.tmp.name)
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
        self.bank = self.make_bank()
        self.prepared = self.root / "prepared"
        c.prepare(self.bank, self.prepared, fixture_only=True)
    def tearDown(self): self.fake_torch.stop(); self.tmp.cleanup()

    def make_bank(self, name="bank", extra=""):
        bank = self.root / name; source = bank / "assessor_only/therapyjudgebench.jsonl"
        source.parent.mkdir(parents=True)
        rows = []
        for n in range(1, 22):
            rows.append({"id": f"Conversation {n}", "session_id": f"invented-session-{n}",
                "patient_profile": "PROFILEHIDDEN-exposed" if n < 16 else f"PROFILEHIDDEN-{n}",
                "conversation": {"messages": [{"role": "user" if i % 2 == 0 else "assistant",
                    "content": f"Invented fixture {n} turn {i}. {extra}"} for i in range(10)]},
                "expert_rating": {"ctrs_ratings": dict.fromkeys(c.CTRS, 3),
                                  "safety_ethics": dict.fromkeys(c.SAFETY, False)}})
        source.write_text("".join(json.dumps(row) + "\n" for row in rows))
        r._write_new(bank / "source_bank_manifest.json", {
            "files": [{"filename": source.name, "sha256": r.file_hash(source)}]})
        return bank

    def run_model(self, model=None, tokenizer=None, output="output", cap=5, prepared=None):
        return c.run_calibration(prepared or self.prepared, model or Model(), tokenizer or Tokenizer(),
            self.model_root, self.root / output, max_new_generations=cap, fixture_only=True)

    def prior_cache(self, n):
        key = f"fixture-prior-{n}"
        path = self.root / "prior/canonical" / (key + ".json")
        value = {"canonical_key": key, "response_text": "invented prior",
            "response_sha256": hashlib.sha256(b"invented prior").hexdigest(),
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}, "finish_reason": "eos"}
        r._write_new(path, value)
        r._append(self.model_root / "generation_ledger.jsonl", {
            "event": "generation_intent", "canonical_key": key, "output": str(self.root / "prior")})
        r._append(self.model_root / "generation_ledger.jsonl", {
            "event": "generation_outcome", "canonical_key": key, "status": "published",
            "cache_path": str(path), "cache_sha256": r.file_hash(path),
            "response_sha256": value["response_sha256"], "usage": value["usage"], "finish_reason": "eos",
            "generation_wall_seconds": 0.01})
        return path

    def test_gold_free_deterministic_profile_disjoint_selection(self):
        p = r.read_json(self.prepared / "protocol.json")
        rows = r.read_rows(self.prepared / "requests.jsonl")
        self.assertEqual(len(rows), 5)
        self.assertEqual(len({x["profile_group_sha256"] for x in rows}), 5)
        self.assertFalse(set(p["selected_profile_groups"]) & set(p["exposed_profile_groups"]))
        raw = (self.prepared / "requests.jsonl").read_text()
        self.assertNotIn("PROFILEHIDDEN", raw)
        self.assertNotIn("expert_rating", raw)
        again = self.root / "again"; c.prepare(self.bank, again, fixture_only=True)
        self.assertEqual((self.prepared / "requests.jsonl").read_bytes(), (again / "requests.jsonl").read_bytes())
        self.assertTrue(p["source_schema_audit_previously_saw_labels"])

    def test_projection_never_decodes_assessor_only_values(self):
        source = self.bank / "assessor_only/therapyjudgebench.jsonl"
        decode = json.loads; observed = []
        def checked(value, *args, **kwargs):
            observed.append(value)
            self.assertNotIn("ctrs_ratings", value)
            self.assertNotIn("safety_ethics", value)
            return decode(value, *args, **kwargs)
        with patch.object(c.json, "loads", side_effect=checked): c._project_line(source.read_bytes().splitlines()[0])
        self.assertEqual(len(observed), 4)

    def test_resume_freezes_all_before_scoring_and_keeps_zero_positive_f1_na(self):
        model = Model()
        first = self.run_model(model=model, cap=2)
        self.assertFalse(first["complete"]); self.assertEqual(model.calls, 2)
        with self.assertRaisesRegex(ValueError, "regular_json_file_required"):
            c.score(self.prepared, self.root / "output", self.model_root, fixture_only=True)
        final = self.run_model(model=model, cap=3)
        self.assertTrue(final["complete"]); self.assertEqual(model.calls, 5)
        report = c.score(self.prepared, self.root / "output", self.model_root, fixture_only=True)
        self.assertEqual(report["valid_prediction_arrays"], 5)
        self.assertEqual(report["ctrs_pooled"]["mae_valid_predictions"], 0)
        self.assertEqual(report["generation_cost_unique_selected_receipts_nonadditive"]["canonical_attempts"], 5)
        self.assertEqual(report["generation_cost_owning_run"]["canonical_attempts"], 5)
        self.assertEqual(report["generation_cost_last_invocation"]["canonical_attempts"], 3)
        self.assertEqual(report["per_dimension"][c.SAFETY[0]]["exact_agreement_all_nonmissing_gold"], 1)
        self.assertIsNone(report["per_dimension"][c.SAFETY[0]]["f1"])
        self.run_model(model=model)
        self.assertEqual(model.calls, 5)

    def test_malformed_arrays_are_retained_and_scored_as_invalid(self):
        self.run_model(tokenizer=Tokenizer("Here is the rating: [3, 3]"))
        report = c.score(self.prepared, self.root / "output", self.model_root, fixture_only=True)
        self.assertEqual(report["invalid_prediction_arrays"], 5)
        self.assertEqual(report["ctrs_pooled"]["coverage"], 0)
        self.assertEqual(report["ctrs_pooled"]["mae_with_maximum_six_point_penalty_for_missing_predictions"], 6)
        self.assertTrue(all("Here is" in a["response_text"] for a in r.read_rows(self.root / "output/answers_unscored.jsonl")))

    def test_five_attempt_purpose_ceiling_is_shared_across_outputs(self):
        self.run_model()
        bank = self.make_bank("newbank", "A different invented detail.")
        new = self.root / "newprepared"; c.prepare(bank, new, fixture_only=True)
        model = Model(); result = self.run_model(model=model, output="newoutput", prepared=new)
        self.assertEqual(model.calls, 0)
        self.assertFalse(result["complete"])
        self.assertEqual(result["calibration_lifetime_generated_attempts"], 5)

    def test_remaining_global_budget_is_respected(self):
        for n in range(63): self.prior_cache(n)
        model = Model(); result = self.run_model(model=model)
        self.assertEqual(model.calls, 1)
        self.assertEqual(result["global_generated_attempts"], 64)
        self.assertFalse(result["complete"])

    def test_unknown_intent_blocks_before_model_generate(self):
        r._append(self.model_root / "generation_ledger.jsonl", {"event": "generation_intent", "canonical_key": "unknown"})
        model = Model()
        with self.assertRaisesRegex(RuntimeError, "automatic_retry"): self.run_model(model=model)
        self.assertEqual(model.calls, 0)

    def test_all_existing_canonical_receipts_verified_even_when_unreferenced(self):
        path = self.prior_cache(1); path.write_text("{}\n")
        model = Model()
        with self.assertRaisesRegex(ValueError, "canonical_cache_changed"): self.run_model(model=model)
        self.assertEqual(model.calls, 0)

    def test_full_batch_token_preflight_admits_no_long_input_attempt(self):
        model = Model()
        with self.assertRaisesRegex(ValueError, "no_truncation_no_reselection"):
            self.run_model(model=model, tokenizer=Tokenizer(n=4097))
        self.assertEqual(model.calls, 0)
        self.assertFalse((self.model_root / "generation_ledger.jsonl").exists())

    def test_changed_source_blocks_before_generation(self):
        source = self.bank / "assessor_only/therapyjudgebench.jsonl"
        with source.open("a") as stream: stream.write("\n")
        model = Model()
        with self.assertRaisesRegex(ValueError, "source_bank_file_binding_changed"): self.run_model(model=model)
        self.assertEqual(model.calls, 0)

    def test_partial_auxiliary_journal_tail_blocks_before_generation(self):
        self.run_model(cap=0)
        with (self.root / "output/invocations.jsonl").open("ab") as stream: stream.write(b'{"partial":')
        model = Model()
        with self.assertRaisesRegex(RuntimeError, "incomplete_journal_tail"): self.run_model(model=model)
        self.assertEqual(model.calls, 0)

    def test_full_budget_cache_recovers_missing_logical_answer_without_generation(self):
        self.run_model()
        path = self.root / "output/answers_unscored.jsonl"
        lines = path.read_bytes().splitlines(keepends=True)
        path.write_bytes(b"".join(lines[:-1]))
        (self.root / "output/completion_summary.json").unlink()
        model = Model(); result = self.run_model(model=model)
        self.assertTrue(result["complete"])
        self.assertEqual(model.calls, 0)
        self.assertEqual(result["calibration_lifetime_generated_attempts"], 5)
        self.assertTrue(r.read_rows(path)[-1]["canonical_generation_shared"])

    def test_failed_generation_is_logged_and_cannot_be_automatically_retried(self):
        model = Model()
        def broken(**kwargs):
            model.calls += 1
            raise RuntimeError("invented interrupted generation")
        model.generate = broken
        with self.assertRaisesRegex(RuntimeError, "invented interrupted"): self.run_model(model=model)
        self.assertEqual(model.calls, 1)
        records = r.read_rows(self.model_root / "generation_ledger.jsonl")
        self.assertEqual(records[-1]["status"], "calibration_failed_or_unknown")
        self.assertIsNone(records[-1]["response_text"])
        fresh = Model()
        with self.assertRaisesRegex(RuntimeError, "automatic_retry"): self.run_model(model=fresh)
        self.assertEqual(fresh.calls, 0)

    def test_scorer_validates_exact_answer_receipts_before_released_labels(self):
        self.run_model()
        path = self.root / "output/answers_unscored.jsonl"
        answers = r.read_rows(path)
        answers[0]["response_text"] = json.dumps([6] * 11 + [0] * 4)
        answers[0]["response_sha256"] = hashlib.sha256(answers[0]["response_text"].encode()).hexdigest()
        path.unlink()
        for row in answers: r._append(path, {k: v for k, v in row.items() if k != "record_sha256"})
        summary = r.read_json(self.root / "output/completion_summary.json")
        summary["answers_sha256"] = r.file_hash(path)
        (self.root / "output/completion_summary.json").unlink()
        r._write_new(self.root / "output/completion_summary.json", summary)
        with self.assertRaisesRegex(ValueError, "answer_cache_disagreement"):
            c.score(self.prepared, self.root / "output", self.model_root, fixture_only=True)

    def test_strict_array_schema_rejects_boolean_decimal_and_markup(self):
        valid = [3] * 11 + [0] * 4
        self.assertEqual(c.parse_prediction(json.dumps(valid)), valid)
        self.assertIsNone(c.parse_prediction("```json\n" + json.dumps(valid) + "\n```"))
        for value in (True, 3.0, 7, -1):
            row = valid.copy(); row[0] = value
            self.assertIsNone(c.parse_prediction(json.dumps(row)))
        row = valid.copy(); row[-1] = 2
        self.assertIsNone(c.parse_prediction(json.dumps(row)))

    def test_missing_gold_and_abstentions_have_explicit_denominators(self):
        m = c._metrics([(None, 6), (3.5, 3), (2, None)])
        self.assertEqual((m["released_label_nonmissing"], m["released_label_missing"]), (2, 1))
        self.assertEqual(m["coverage"], 0.5)
        self.assertEqual(m["mae_valid_predictions"], 0.5)
        self.assertEqual(m["mae_with_maximum_six_point_penalty_for_missing_predictions"], 3.25)
        s = c._metrics([(True, None), (True, 1), (False, 0)], safety=True)
        self.assertEqual((s["tp"], s["fn_including_abstentions"], s["tn"]), (1, 1, 1))
        self.assertAlmostEqual(s["f1"], 2 / 3)

    def test_real_guard_precedes_source_io(self):
        with patch.object(r, "require_colab", side_effect=RuntimeError("not Colab")):
            with self.assertRaisesRegex(RuntimeError, "not Colab"):
                c.prepare(self.root / "does-not-exist", self.root / "never-created")
        self.assertFalse((self.root / "never-created").exists())


if __name__ == "__main__": unittest.main()
