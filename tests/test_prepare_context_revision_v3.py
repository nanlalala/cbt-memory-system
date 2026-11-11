"""Small SQLite fixture integration only: no corpus, Secrets or provider."""
from datetime import timedelta
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_context_revision_v3", ROOT / "evaluation/cbt-memory-effect/prepare_context_revision_v3.py")
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)
P = M.product_modules(ROOT)


class CheckpointDryRunTests(unittest.TestCase):
    def fixture(self, root, *, delete_first=False):
        old = root / "old-v2"; case_dir = old / "cases/probe-0"; case_dir.mkdir(parents=True)
        (old / "generation").mkdir()
        history = [{"message_id": "turn-0", "source_order": 0, "role": "user", "content": "I have not agreed to thought records."},
            {"message_id": "turn-1", "source_order": 1, "role": "assistant", "content": "We can discuss your preference."},
            {"message_id": "turn-2", "source_order": 2, "role": "user", "content": "I prefer discussing the nervousness."}]
        permissions = {key: True for key in ("save", "long_term_use", "response_external", "extraction_external")}
        session = P.ControlledDialogueSession.open(str(case_dir / "memory.sqlite"), approved_root=str(case_dir),
            user_id="fixture-user", background=False)
        receipts = []
        for turn in history:
            source = {"user_id": "fixture-user", "source_id": "s:" + turn["message_id"], "input_id": "i:" + turn["message_id"],
                "text": turn["content"], "role": turn["role"], "source_time": (M.EPOCH + timedelta(seconds=turn["source_order"])).isoformat(),
                "permissions": permissions}
            session.durable.submit_source(source, enqueue=False)
            receipts.append({"message_id": turn["message_id"], "source_id": source["source_id"],
                "role": source["role"], "text_sha256": M.hashlib.sha256(source["text"].encode()).hexdigest()})
            if turn["role"] == "user":
                record = P._source_record(source, "fixture-session", memory_id="candidate-" + turn["message_id"],
                    content=source["text"], representation="structured_record")
                session.durable.put_effect("fixture-user", "fixture-effect-" + turn["message_id"], "memory:" + record.memory_id,
                    expected_version=0, payload={"kind": "candidate", "record": record.to_dict()},
                    dependencies=[{"source_id": source["source_id"], "start": 0, "end": len(source["text"])}])
        if delete_first:
            session.delete_content("source", "s:turn-0")
        session.close()
        cases = []
        for i, query in enumerate(["thought records", "nervousness", "discuss preference"]):
            case = {"case_id": "probe-" + str(i), "source_family_id": "fixture-family", "history": history,
                "history_cutoff_order_exclusive": 3, "input": "What did I say about " + query + "?",
                "retrieval_query": query}
            cases.append({**case, "case_sha256": M.digest(case)})
        generation = old / "generation/cases.jsonl"
        generation.write_text("".join(json.dumps(row) + "\n" for row in cases))
        protocol = {"version": "cactus-source-quotation-recall-v2", "clock_mode": "ordinal-only-v1",
            "files_sha256": {"generation/cases.jsonl": M.file_hash(generation)},
            "code_sha256": {"modules/agent-integration/controlled_dialogue.py": M.file_hash(ROOT / "modules/agent-integration/controlled_dialogue.py")},
            "settings": {"model": "fixture-model-only", "output_tokens": {"response": 128}}}
        protocol["protocol_sha256"] = M.digest(protocol)
        (old / "protocol.json").write_text(json.dumps(protocol))
        (case_dir / "prepared_contexts.json").write_text(json.dumps({"protocol_sha256": protocol["protocol_sha256"],
            "study_case_sha256": cases[0]["case_sha256"], "prepared": {"trace": {"user_id": "fixture-user", "source_receipts": receipts}}}))
        return old

    def test_three_probe_zero_call_freeze_preserves_original_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); old = self.fixture(root)
            before = {str(p): M.file_hash(p) for p in old.rglob("*") if p.is_file()}
            with patch.object(M, "require_colab", side_effect=AssertionError("fixture must not use cloud bypass")):
                result = M.prepare(old, root / "v3-development", product_repo=ROOT, fixture_only=True)
            self.assertEqual(result["prepared_cases"], 3)
            self.assertEqual(result["actual_model_calls"], 0)
            self.assertFalse(result["secret_access"])
            self.assertFalse(result["gold_read"])
            self.assertTrue(result["not_send_ready"])
            self.assertIsNone(result["effect_scores"])
            self.assertEqual(before, {str(p): M.file_hash(p) for p in old.rglob("*") if p.is_file()})
            traces = M.read_rows(root / "v3-development/context_rank_pack_traces.jsonl")
            self.assertEqual(traces[0]["candidate_index_trace"]["eligible_candidate_count"], 2)
            self.assertTrue(all(row["not_send_ready"] for row in traces))

    def test_deleted_source_and_candidate_do_not_reenter_compact_packets(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); old = self.fixture(root, delete_first=True)
            M.prepare(old, root / "v3-development", product_repo=ROOT, fixture_only=True)
            rows = M.read_rows(root / "v3-development/context_rank_pack_traces.jsonl")
            for row in rows:
                for context in row["assembly"]["contexts"].values():
                    self.assertNotIn("s:turn-0", context["source_dependencies"])
                    self.assertNotIn("I have not agreed to thought records.", json.dumps(context["messages"]))
                self.assertEqual(row["candidate_index_trace"]["eligible_candidate_count"], 1)

    def test_changed_frozen_input_or_reused_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); old = self.fixture(root)
            out = root / "v3-development"; out.mkdir()
            with self.assertRaises(ValueError):
                M.prepare(old, out, product_repo=ROOT, fixture_only=True)
            (old / "generation/cases.jsonl").write_text("{}\n")
            with self.assertRaises(ValueError):
                M.prepare(old, root / "v3-new", product_repo=ROOT, fixture_only=True)

    def test_optional_encoder_shared_between_channels_and_tracks_local_work_only(self):
        class FakePinnedEncoder:
            model_identity = {"repo_id": "fixture-model", "revision": "fixture-pinned-only"}
            def encode(self, texts, *, is_query):
                return [[1.0, 0.0] for _ in texts]
        original_raw = P.StableSourceRetriever
        original_memory = P.MemoryRetriever
        raw_encoders = []; candidate_encoders = []
        def raw_factory(*args, **kwargs):
            raw_encoders.append(kwargs.get("encoder"))
            return original_raw(*args, **kwargs)
        def candidate_factory(*args, **kwargs):
            candidate_encoders.append(kwargs.get("encoder"))
            return original_memory(*args, **kwargs)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); old = self.fixture(root)
            with patch.object(P, "StableSourceRetriever", side_effect=raw_factory), patch.object(P, "MemoryRetriever", side_effect=candidate_factory), patch("socket.create_connection", side_effect=AssertionError("no network in fixture")):
                result = M.prepare(old, root / "dense-fixture", product_repo=ROOT, fixture_only=True,
                    encoder_name="multilingual-e5-small", encoder_factory=FakePinnedEncoder)
            self.assertEqual(len(raw_encoders), 3)
            self.assertEqual(len(candidate_encoders), 3)
            self.assertTrue(all(value is raw_encoders[0] for value in raw_encoders + candidate_encoders))
            work = result["local_embedding_work"]
            self.assertEqual(work["calls_attempted"], 12)
            self.assertEqual(work["calls_completed"], 12)
            self.assertEqual(work["query_texts_attempted"], 6)
            self.assertEqual(work["document_texts_attempted"], 15)
            self.assertEqual(work["texts_attempted"], 21)
            self.assertEqual(work["embedding_api_requests"], 0)
            self.assertGreaterEqual(work["local_embedding_seconds_sum"], 0)
            self.assertEqual(result["actual_model_calls"], 0)
            protocol = M.read_json(root / "dense-fixture/protocol.json")
            self.assertEqual(protocol["retrieval"]["encoder_model_identity"], FakePinnedEncoder.model_identity)
            self.assertTrue(protocol["retrieval"]["same_encoder_instance_for_both_channels"])
            rows = M.read_rows(root / "dense-fixture/context_rank_pack_traces.jsonl")
            self.assertTrue(all(row["candidate_index_trace"]["ranked_index_hits"] for row in rows))
            self.assertTrue(all(row["not_send_ready"] for row in rows))

    def test_decimal_or_numpy_float32_vectors_produce_native_serializable_traces(self):
        from decimal import Decimal
        try:
            import numpy as np
            scalar = np.float32
        except ImportError:
            scalar = Decimal
        class ScalarEncoder:
            model_identity = {"repo_id": "fixture-scalar", "revision": "fixture-pinned-only"}
            def encode(self, texts, *, is_query):
                return [[scalar("1.0"), scalar("0.0")] for _ in texts]
        tracked = M.TrackedEncoder(ScalarEncoder())
        vectors = tracked.encode(["fixture"], is_query=True)
        self.assertTrue(all(type(component) is float for vector in vectors for component in vector))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); old = self.fixture(root)
            result = M.prepare(old, root / "scalar-fixture", product_repo=ROOT, fixture_only=True,
                encoder_name="multilingual-e5-small", encoder_factory=ScalarEncoder)
            json.dumps(result, allow_nan=False)
            for name in ("protocol.json", "context_rank_pack_traces.jsonl", "summary.json"):
                path = root / "scalar-fixture" / name
                values = M.read_rows(path) if path.suffix == ".jsonl" else [M.read_json(path)]
                for value in values:
                    json.dumps(value, allow_nan=False)
            self.assertEqual(result["prepared_cases"], 3)
            self.assertEqual(result["actual_provider_calls"], 0)

    def test_nonfinite_encoder_vectors_fail_before_completed_encoding_or_output(self):
        class InvalidEncoder:
            model_identity = {"repo_id": "fixture-invalid"}
            def encode(self, texts, *, is_query):
                return [[float("nan"), 0.0] for _ in texts]
        tracked = M.TrackedEncoder(InvalidEncoder())
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            tracked.encode(["fixture"], is_query=True)
        self.assertEqual(tracked.calls_attempted, 1)
        self.assertEqual(tracked.calls_completed, 0)
        self.assertEqual(tracked.texts_completed, 0)

    def test_encoder_loading_failure_is_not_silently_replaced_with_lexical(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); old = self.fixture(root)
            def fail():
                raise RuntimeError("pinned snapshot failure")
            with self.assertRaisesRegex(RuntimeError, "pinned snapshot failure"):
                M.prepare(old, root / "must-not-exist", product_repo=ROOT, fixture_only=True,
                    encoder_name="multilingual-e5-small", encoder_factory=fail)
            self.assertFalse((root / "must-not-exist").exists())

    def test_encoder_override_restricted_to_fixtures_and_unknown_encoder_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(M, "require_colab", return_value=None):
                with self.assertRaisesRegex(ValueError, "tiny fixture"):
                    M.prepare(folder, Path(folder) / "out", product_repo=ROOT,
                        encoder_name="multilingual-e5-small", encoder_factory=lambda: None)
            with self.assertRaisesRegex(ValueError, "unknown encoder"):
                M.prepare(folder, Path(folder) / "out", product_repo=ROOT, fixture_only=True,
                    encoder_name="moving-latest-model")

    def test_real_entry_requires_hosted_colab_without_exposing_a_cli_fixture_switch(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(M, "require_colab", side_effect=RuntimeError("not hosted")):
                with self.assertRaisesRegex(RuntimeError, "not hosted"):
                    M.prepare(Path(folder), Path(folder) / "v3", product_repo=ROOT)


if __name__ == "__main__":
    unittest.main()
