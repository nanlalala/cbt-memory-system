from __future__ import annotations

import json
import sys
import tempfile
import unittest
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules/cbt-knowledge-rag"))

from cbt_rag_v1 import Retriever, evaluate, export_examples, load_corpus, load_knowledge_config, main  # noqa: E402
from source_registry import SourcePolicyError, SourceRegistry  # noqa: E402
from test_source_registry import fixture_config, fixture_payload, raw_fixture_chunk  # noqa: E402


def two_source_registry() -> SourceRegistry:
    payload = fixture_payload()
    other = deepcopy(payload["sources"][0])
    other.update(id="other", title="Second original fixture")
    payload["sources"].append(other)
    return SourceRegistry(payload)


def chunks_for(registry: SourceRegistry, *source_ids: str) -> list[dict]:
    return [registry.stamp_chunk({
        **raw_fixture_chunk(), "chunk_id": source_id + ":one", "source_id": source_id,
        "title": registry.source(source_id)["title"],
    }) for source_id in source_ids]


class KnowledgeConfigTests(unittest.TestCase):
    def test_selection_is_immutable_and_fingerprint_changes_with_selection(self) -> None:
        registry = two_source_registry()
        config = fixture_config(registry)
        with self.assertRaises(FrozenInstanceError):
            config.selected_source_ids = ("other",)
        first = config.chunker_config
        first["target_chars"] = 7
        self.assertNotEqual(config.chunker_config["target_chars"], 7)
        self.assertNotEqual(config.fingerprint, fixture_config(registry, ("other",)).fingerprint)
        self.assertNotEqual(fixture_config(registry, ("synthetic", "other")).fingerprint,
                            fixture_config(registry, ("other", "synthetic")).fingerprint)

    def test_approved_but_unselected_source_is_rejected_by_retriever(self) -> None:
        registry = two_source_registry()
        restricted = fixture_config(registry)
        broad = fixture_config(registry, ("synthetic", "other"))
        chunks = chunks_for(registry, "synthetic", "other")
        with self.assertRaisesRegex(SourcePolicyError, "unselected"):
            Retriever(chunks, download_models=False, registry=registry,
                      knowledge_config=restricted, manifest=broad.corpus_manifest(chunks, registry))

    def test_manifest_binds_source_set_registry_config_count_and_corpus(self) -> None:
        registry = two_source_registry()
        config = fixture_config(registry)
        chunks = chunks_for(registry, "synthetic")
        baseline = config.corpus_manifest(chunks, registry)
        changes = {
            "source_ids": ["other"], "selected_source_ids": ["other"],
            "registry_fingerprint": "0" * 64, "config_fingerprint": "0" * 64,
            "corpus_fingerprint": "0" * 64, "chunk_count": 99,
            "schema_version": 1, "chunker_config": {"target_chars": 3},
        }
        for field, value in changes.items():
            with self.subTest(field=field), self.assertRaisesRegex(SourcePolicyError, "manifest"):
                Retriever(chunks, download_models=False, registry=registry,
                          knowledge_config=config, manifest={**baseline, field: value})

    def test_missing_selected_source_and_duplicate_chunks_are_rejected(self) -> None:
        registry = two_source_registry()
        config = fixture_config(registry, ("synthetic", "other"))
        with self.assertRaisesRegex(SourcePolicyError, "source set"):
            config.corpus_manifest(chunks_for(registry, "synthetic"), registry)
        single = fixture_config(registry)
        chunks = chunks_for(registry, "synthetic")
        with self.assertRaisesRegex(SourcePolicyError, "duplicate"):
            single.corpus_manifest(chunks + chunks, registry)

    def test_manifest_is_required_and_loader_rejects_stale_selection(self) -> None:
        registry = two_source_registry()
        config = fixture_config(registry)
        chunks = chunks_for(registry, "synthetic")
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            path = temporary / "chunks.jsonl"
            path.write_text("\n".join(json.dumps(chunk) for chunk in chunks), encoding="utf-8")
            with self.assertRaisesRegex(SourcePolicyError, "manifest"):
                load_corpus(path, registry=registry, knowledge_config=config)
            (temporary / "corpus_manifest.json").write_text(json.dumps(config.corpus_manifest(chunks, registry)), encoding="utf-8")
            self.assertEqual(load_corpus(path, registry=registry, knowledge_config=config), chunks)
            with self.assertRaises(SourcePolicyError):
                load_corpus(path, registry=registry, knowledge_config=fixture_config(registry, ("other",)))

    def test_metadata_tamper_is_rejected_at_existing_corpus_load(self) -> None:
        registry = two_source_registry()
        config = fixture_config(registry)
        chunks = chunks_for(registry, "synthetic")
        manifest = config.corpus_manifest(chunks, registry)
        chunks[0]["citation"] = "Forged publication p. 999"
        with self.assertRaises(SourcePolicyError):
            Retriever(chunks, download_models=False, registry=registry, knowledge_config=config, manifest=manifest)


class NoModelKnowledgeEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = two_source_registry()
        self.config = fixture_config(self.registry)
        chunks = chunks_for(self.registry, "synthetic")
        self.retriever = Retriever(chunks, download_models=False, registry=self.registry,
                                   knowledge_config=self.config,
                                   manifest=self.config.corpus_manifest(chunks, self.registry))
        self.questions = [{"question_id": "test-q", "query": "blue test notebook",
                           "category": "test", "expected_sources": ["synthetic"]}]

    def test_lexical_evaluate_and_export_run_without_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch("cbt_rag_v1.RESULTS", Path(directory)):
            summary = evaluate(self.retriever, self.questions)
            self.assertEqual(set(summary["method"]), {"tfidf", "bm25", "lexical_hybrid"})
            export_examples(self.retriever, self.questions)
            self.assertTrue((Path(directory) / "lexical_hybrid_examples.csv").exists())
            self.assertTrue((Path(directory) / "retrieval_summary.json").exists())
            self.assertFalse((Path(directory) / "hybrid_rerank_examples.csv").exists())

    def test_unavailable_dense_modes_fail_with_clear_value_error(self) -> None:
        for mode in ("dense", "hybrid", "hybrid_rerank", "not_a_mode"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "unavailable"):
                    self.retriever.rank("notebook", mode)
                with self.assertRaisesRegex(ValueError, "unavailable"):
                    evaluate(self.retriever, self.questions, modes=[mode])
                with self.assertRaisesRegex(ValueError, "unavailable"):
                    export_examples(self.retriever, self.questions, mode=mode)

    def test_mutated_retriever_metadata_is_not_exported_or_evaluated(self) -> None:
        self.retriever.chunks[0]["risk_level"] = "high"
        with self.assertRaises(SourcePolicyError):
            evaluate(self.retriever, self.questions)
        with self.assertRaises(SourcePolicyError):
            export_examples(self.retriever, self.questions)

    def test_valid_replacement_corpus_cannot_reuse_an_existing_index(self) -> None:
        self.retriever.rank("blue test notebook", "tfidf", 1)
        replacement = self.registry.stamp_chunk({**raw_fixture_chunk(), "text": "Wholly different verified yellow diary."})
        self.retriever.chunks = [replacement]
        self.retriever.manifest = self.config.corpus_manifest([replacement], self.registry)
        # Even updating the public label cannot turn old vectors/rank caches
        # into a new index. This tests accidental runtime drift, not Python ACLs.
        self.retriever.corpus_fingerprint = self.retriever.manifest["corpus_fingerprint"]
        with self.assertRaisesRegex(SourcePolicyError, "changed after indexing"):
            self.retriever.rank("blue test notebook", "tfidf", 1)
        with self.assertRaisesRegex(SourcePolicyError, "changed after indexing"):
            self.retriever.evidence_cards([0])

    def test_modified_index_texts_are_rejected_even_with_unchanged_manifest(self) -> None:
        self.retriever.texts[0] = "unrelated text"
        with self.assertRaisesRegex(SourcePolicyError, "indexed texts changed"):
            self.retriever.rank("notebook", "tfidf", 1)

    def test_no_models_cli_evaluate_also_exports_without_loading_models(self) -> None:
        payload = fixture_payload()
        registry = SourceRegistry(payload)
        chunks = chunks_for(registry, "synthetic")
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            (temporary / "review.json").write_text(json.dumps(payload), encoding="utf-8")
            selection = temporary / "selection.yaml"
            selection.write_text("source_registry: review.json\nsources:\n  - id: synthetic\n", encoding="utf-8")
            config = load_knowledge_config(selection, registry=registry)
            (temporary / "chunks.jsonl").write_text("\n".join(json.dumps(chunk) for chunk in chunks), encoding="utf-8")
            (temporary / "corpus_manifest.json").write_text(json.dumps(config.corpus_manifest(chunks, registry)), encoding="utf-8")
            (temporary / "rag_eval_set.jsonl").write_text("\n".join(json.dumps(question) for question in self.questions), encoding="utf-8")
            with patch("cbt_rag_v1.DATA", temporary), patch("cbt_rag_v1.RESULTS", temporary), \
                 patch("cbt_rag_v1.ROOT", temporary), patch.dict("os.environ", {"CBT_KB_CONFIG": str(selection)}), \
                 patch.object(sys, "argv", ["cbt_rag_v1.py", "evaluate", "--no-models"]), \
                 patch("cbt_rag_v1.prepare_model_snapshot", side_effect=AssertionError("no model download allowed")):
                main()
            self.assertTrue((temporary / "lexical_hybrid_examples.csv").exists())
            frozen = json.loads((temporary / "evaluation_run_manifest.json").read_text())
            self.assertEqual(frozen["verified_model_identities"], {})


if __name__ == "__main__":
    unittest.main()
