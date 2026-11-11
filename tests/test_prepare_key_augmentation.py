"""Invented SQLite fixtures only; no public corpus or model download."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("prepare_key_augmentation", ROOT / "evaluation/cbt-memory-effect/prepare_key_augmentation.py")
K = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(K)
FIXTURE_SPEC = importlib.util.spec_from_file_location("key_fixture_support", ROOT / "tests/test_prepare_context_revision_v3.py")
F = importlib.util.module_from_spec(FIXTURE_SPEC); FIXTURE_SPEC.loader.exec_module(F)


class FakeEncoder:
    model_identity = {"repo_id": "fixture-only-pinned", "revision": "fixture"}
    def encode(self, texts, *, is_query):
        # Local deterministic fixture embeddings, never a downloaded model.
        return [[1.0, float("nervousness" in text)] for text in texts]


class KeyAugmentationTests(unittest.TestCase):
    def fixture(self, root, **kwargs):
        return F.CheckpointDryRunTests().fixture(root, **kwargs)

    def add_candidate(self, old, memory_id, content, *, source_ids=("s:turn-0",), record_overrides=None):
        directory = old / "cases/probe-0"
        session = F.P.ControlledDialogueSession.open(str(directory / "memory.sqlite"),
            approved_root=str(directory), user_id="fixture-user", background=False)
        source = session.durable.get_source("fixture-user", "s:turn-0")
        record = F.P._source_record(source, "fixture-session", memory_id=memory_id,
            content=content, representation="structured_record")
        data = record.to_dict(); data["source_evidence"]["turn_ids"] = list(source_ids)
        data.update(record_overrides or {})
        record = F.P.construct_memory(data, allow_legacy=False)
        session.durable.put_effect("fixture-user", "key-effect-" + memory_id, "memory:" + memory_id,
            expected_version=0, payload={"kind": "candidate", "record": record.to_dict()},
            dependencies=[{"source_id": sid, "start": 0,
                           "end": len(session.durable.get_source("fixture-user", sid)["text"])} for sid in source_ids])
        session.close()

    def test_full_three_condition_dryrun_has_no_candidate_prose_on_wire_or_gold_access(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); old = self.fixture(root, extra_turns=7)
            marker = "INDEX_ONLY_INFERENCE_MUST_NOT_APPEAR_AS_USER_FACT"
            self.add_candidate(old, "aa-marker", marker)
            gold = old / "assessor_only/targets.jsonl"; gold.parent.mkdir(); gold.write_text("private scoring sentinel")
            before = {str(p): K.base.file_hash(p) for p in old.rglob("*") if p.is_file()}
            real_read = Path.read_bytes
            def read_guard(path):
                if "assessor_only" in path.parts:
                    raise AssertionError("gold must not be read")
                return real_read(path)
            with patch.object(Path, "read_bytes", read_guard), patch("socket.create_connection", side_effect=AssertionError("no network in fixture")):
                summary = K.prepare(old, root / "key-dev", product_repo=ROOT, fixture_only=True, encoder_factory=FakeEncoder)
            self.assertEqual(summary["prepared_cases"], 3)
            self.assertEqual(summary["actual_provider_calls"], 0)
            self.assertIsNone(summary["effect_scores"])
            self.assertTrue(summary["development_only"])
            self.assertEqual(before, {str(p): K.base.file_hash(p) for p in old.rglob("*") if p.is_file()})
            rows = K.base.read_rows(root / "key-dev/traces.jsonl")
            for row in rows:
                self.assertEqual(set(row["contexts"]), {"raw_original", "managed_rrf", "managed_key_augmentation"})
                recent = [context["recent_source_ids"] for context in row["contexts"].values()]
                self.assertTrue(all(ids == recent[0] for ids in recent))
                for context in row["contexts"].values():
                    wire = json.dumps(context["messages"], ensure_ascii=False)
                    self.assertNotIn(marker, wire)
                    self.assertNotIn(K.KEY_SEPARATOR, wire)
                    self.assertLessEqual(context["outbound_chars"], 6000)
                    self.assertEqual(len(context["serialized_request"]), context["outbound_chars"])
                    json.dumps(context, allow_nan=False)
            protocol = K.base.read_json(root / "key-dev/protocol.json")
            self.assertEqual(protocol["policy"], K.POLICY)
            self.assertTrue(all(row["status"] == "unknown" for row in protocol["index_truncation"]["augmented"].values()))

    def test_source_key_limits_dedup_and_single_source_binding_are_fixed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); old = self.fixture(root)
            self.add_candidate(old, "aa-marker", "same index description")
            self.add_candidate(old, "ab-duplicate", "same index description")
            self.add_candidate(old, "ac-multi", "cannot attach this to every source", source_ids=("s:turn-0", "s:turn-2"))
            self.add_candidate(old, "zz-too-long", "X" * 900)
            protocol, cases, prepared, db, hashes = K.load_checkpoint(old, ROOT)
            with K.base.readonly_durable(db, F.P) as durable:
                sources, _ = K.base.checkpoint_sources(durable, prepared, cases[0])
                original, augmented, manifest = K.build_keys(durable, F.P, sources, "fixture-user")
            self.assertTrue(all(augmented[sid].startswith(text) for sid, text in original.items()))
            self.assertEqual(augmented["s:turn-0"].count("same index description"), 1)
            reasons = {row["reason"] for row in manifest["excluded"]}
            self.assertIn("duplicate_content_for_source", reasons)
            self.assertIn("multiple_or_missing_source_binding", reasons)
            self.assertIn("fixed_source_augmentation_limit", reasons)
            self.assertTrue(all(len(row["candidate_ids"]) <= 2 and row["added_candidate_characters"] <= 800
                                for row in manifest["sources"].values()))

    def test_augmented_keys_match_candidate_lifecycle_and_restricted_admission(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); old = self.fixture(root)
            self.add_candidate(old, "aa-expired", "EXPIRED_INDEX_DESCRIPTION", record_overrides={"lifecycle_status": "expired"})
            self.add_candidate(old, "ab-restricted", "RESTRICTED_INDEX_DESCRIPTION", record_overrides={"sensitivity": "restricted"})
            _, cases, prepared, db, _ = K.load_checkpoint(old, ROOT)
            with K.base.readonly_durable(db, F.P) as durable:
                sources, _ = K.base.checkpoint_sources(durable, prepared, cases[0])
                _, keys, manifest = K.build_keys(durable, F.P, sources, "fixture-user")
            self.assertNotIn("EXPIRED_INDEX_DESCRIPTION", " ".join(keys.values()))
            self.assertNotIn("RESTRICTED_INDEX_DESCRIPTION", " ".join(keys.values()))
            self.assertEqual({row["reason"] for row in manifest["excluded"]}, {"non_candidate_lifecycle", "restricted_candidate"})

    def test_deleted_sources_cannot_enter_augmented_index(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); old = self.fixture(root, delete_first=True)
            K.prepare(old, root / "deleted-key-dev", product_repo=ROOT, fixture_only=True, encoder_factory=FakeEncoder)
            rows = K.base.read_rows(root / "deleted-key-dev/traces.jsonl")
            for row in rows:
                self.assertNotIn("s:turn-0", [hit["source_id"] for hit in row["augmented_key_ranking"]["ranking"]])
                self.assertTrue(all("s:turn-0" not in context["source_dependencies"] for context in row["contexts"].values()))

    def test_same_ranker_identical_keys_and_detectable_truncation_estimate(self):
        encoder = K.base.TrackedEncoder(FakeEncoder())
        keys = {"a": "Original report nervousness", "b": "Original thought record"}
        first = K.rank_keys(keys, "nervousness", product=F.P, encoder=encoder)
        second = K.rank_keys(deepcopy(keys), "nervousness", product=F.P, encoder=encoder)
        self.assertEqual(first, second)
        class Tokenizer:
            def __call__(self, text, **kwargs):
                self.kwargs = kwargs
                return {"input_ids": list(range(len(text.split())))}
        fake = FakeEncoder(); fake.model = type("Model", (), {"max_seq_length": 3, "tokenizer": Tokenizer()})()
        report = K.truncation_report(K.base.TrackedEncoder(fake), {"a": "one two three four"})
        self.assertTrue(report["a"]["may_exceed_encoder_limit"])
        self.assertFalse(report["a"]["actual_embedding_truncation_verified"])

    def test_original_key_rank_reproduces_v3_original_source_rank_on_fixture(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); old = self.fixture(root, extra_turns=7)
            _, cases, prepared, db, _ = K.load_checkpoint(old, ROOT)
            encoder = K.base.TrackedEncoder(FakeEncoder())
            with K.base.readonly_durable(db, F.P) as durable:
                sources, _ = K.base.checkpoint_sources(durable, prepared, cases[0])
            keys = {source["source_id"]: source["text"] for source in sources}
            for case in cases:
                query = case["retrieval_query"]
                ranked, _ = K.rank_keys(keys, query, product=F.P, encoder=encoder)
                original = F.P.StableSourceRetriever(sources, encoder=encoder).search(query,
                    user_id="fixture-user", source_cutoff=(K.base.EPOCH + K.base.timedelta(seconds=10)).isoformat(),
                    as_of=None, for_external_response=True, top_k=10, lexical_k=10, dense_k=10)
                self.assertEqual(ranked, [hit.source_id for hit in original.hits])

    def test_live_entry_does_not_accept_fake_encoder_or_unhosted_work(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(K.base, "require_colab", side_effect=RuntimeError("not hosted")):
                with self.assertRaisesRegex(RuntimeError, "not hosted"):
                    K.prepare(folder, Path(folder) / "new", product_repo=ROOT)
            with patch.object(K.base, "require_colab", return_value=None):
                with self.assertRaisesRegex(ValueError, "fixtures"):
                    K.prepare(folder, Path(folder) / "new", product_repo=ROOT, encoder_factory=FakeEncoder)

    def test_memoization_reduces_physical_work_and_preserves_all_nine_fixture_wires(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); old = self.fixture(root, extra_turns=7)
            plain = K.prepare(old, root / "plain", product_repo=ROOT, fixture_only=True,
                              encoder_factory=FakeEncoder)
            cached = K.prepare(old, root / "memoized", product_repo=ROOT, fixture_only=True,
                               encoder_factory=FakeEncoder, memoize_embeddings=True)
            a = K.base.read_rows(root / "plain/traces.jsonl")
            b = K.base.read_rows(root / "memoized/traces.jsonl")
            for old_row, new_row in zip(a, b):
                for arm in old_row["contexts"]:
                    self.assertEqual(old_row["contexts"][arm], new_row["contexts"][arm])
            self.assertEqual(plain["local_embedding_work"]["texts_attempted"],
                             cached["local_embedding_work"]["texts_attempted"])
            self.assertLess(cached["physical_embedding_work"]["physical_encode_texts_attempted"],
                            plain["physical_embedding_work"]["physical_encode_texts_attempted"])
            self.assertGreater(cached["physical_embedding_work"]["cache_hit_texts"], 0)
            self.assertFalse(plain["embedding_cache"]["enabled"])
            self.assertTrue(cached["embedding_cache"]["enabled"])
            self.assertIsNone(cached["reference_comparison"]["all_nine_wire_hashes_and_source_orders_equal"])

    def test_reference_comparison_checks_all_wire_hashes_and_source_orders(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); old = self.fixture(root, extra_turns=7)
            K.prepare(old, root / "plain", product_repo=ROOT, fixture_only=True, encoder_factory=FakeEncoder)
            rows = K.base.read_rows(root / "plain/traces.jsonl")
            reference = root / "fixture-v1-reference"; reference.mkdir()
            protocol = {"version": "source-key-augmentation-development-v1", "fixture_only": True,
                "code_sha256": {str(Path(K.__file__).resolve()): K.base.file_hash(K.__file__)},
                "source_checkpoint_hashes": {str(old / "protocol.json"): K.base.file_hash(old / "protocol.json")}}
            protocol["protocol_sha256"] = K.base.digest(protocol)
            (reference / "protocol.json").write_text(json.dumps(protocol))
            (reference / "traces.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = K.compare_reference(reference, rows)
            self.assertTrue(result["all_nine_wire_hashes_and_source_orders_equal"])
            self.assertEqual(len(result["comparisons"]), 9)
            spoofed = deepcopy(rows); context = next(iter(spoofed[0]["contexts"].values()))
            context["request_sha256"] = "spoofed-fixture-hash"
            with self.assertRaisesRegex(ValueError, "canonical wire"):
                K.compare_reference(reference, spoofed)
            changed = deepcopy(rows); context = next(iter(changed[0]["contexts"].values()))
            context["request_payload"]["messages"][0]["content"] += " A changed fixture instruction."
            context["messages"] = context["request_payload"]["messages"]
            context["serialized_request"] = K.assembly.canonical(context["request_payload"])
            context["outbound_chars"] = len(context["serialized_request"])
            context["request_sha256"] = K.hashlib.sha256(context["serialized_request"].encode()).hexdigest()
            self.assertFalse(K.compare_reference(reference, changed)["all_nine_wire_hashes_and_source_orders_equal"])
            dependency = old / "protocol.json"
            dependency.write_text("changed fixture dependency")
            with self.assertRaisesRegex(ValueError, "dependency changed"):
                K.compare_reference(reference, rows)


if __name__ == "__main__":
    unittest.main()
