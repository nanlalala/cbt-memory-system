from __future__ import annotations

import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules/cbt-knowledge-rag"))

from cbt_rag_v1 import (  # noqa: E402
    embedding_cache_fingerprint,
    load_embedding_cache,
    save_embedding_cache,
)


class EmbeddingCacheTests(unittest.TestCase):
    def test_same_chunk_count_with_changed_text_invalidates_cache(self) -> None:
        first = embedding_cache_fingerprint(["chunk A", "chunk B"], "model-v1")
        changed = embedding_cache_fingerprint(["chunk A", "changed B"], "model-v1")
        self.assertNotEqual(first, changed)

    def test_cache_requires_matching_corpus_and_model_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            embedding_path = Path(directory) / "embeddings.npy"
            metadata_path = Path(directory) / "embeddings.meta.json"
            fingerprint = embedding_cache_fingerprint(["one", "two"], "model-v1")
            embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]])
            save_embedding_cache(
                embedding_path,
                metadata_path,
                embeddings,
                fingerprint=fingerprint,
                model_name="model-v1",
            )
            loaded = load_embedding_cache(
                embedding_path,
                metadata_path,
                expected_fingerprint=fingerprint,
                expected_count=2,
            )
            self.assertTrue(np.array_equal(loaded, embeddings))
            self.assertIsNone(
                load_embedding_cache(
                    embedding_path,
                    metadata_path,
                    expected_fingerprint=embedding_cache_fingerprint(
                        ["one", "changed"], "model-v1"
                    ),
                    expected_count=2,
                )
            )

    def test_revision_files_runtime_and_preprocessing_change_cache_identity(self) -> None:
        identity = {
            "repo_id": "same-model", "revision": "a" * 40,
            "files": {"tokenizer.json": {"sha256": "a" * 64, "size": 1}},
            "runtime": {"sentence-transformers": "test-v1"},
        }
        preprocessing = {"prefix": "passage: ", "normalization": True}
        baseline = embedding_cache_fingerprint(["same text"], "same-model",
                                               model_identity=identity, preprocessing=preprocessing)
        variants = []
        revision = deepcopy(identity)
        revision["revision"] = "b" * 40
        variants.append(revision)
        files = deepcopy(identity)
        files["files"]["tokenizer.json"]["sha256"] = "b" * 64
        variants.append(files)
        runtime = deepcopy(identity)
        runtime["runtime"]["sentence-transformers"] = "test-v2"
        variants.append(runtime)
        for changed in variants:
            self.assertNotEqual(baseline, embedding_cache_fingerprint(
                ["same text"], "same-model", model_identity=changed, preprocessing=preprocessing))
        self.assertNotEqual(baseline, embedding_cache_fingerprint(
            ["same text"], "same-model", model_identity=identity, preprocessing={"prefix": "query: "}))

    def test_torn_or_modified_embedding_bytes_do_not_load_under_valid_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            embedding_path = Path(directory) / "embeddings.npy"
            metadata_path = Path(directory) / "embeddings.meta.json"
            save_embedding_cache(embedding_path, metadata_path, np.asarray([[1.0, 0.0]]),
                                 fingerprint="fixture-fingerprint", model_name="fixture-model")
            with embedding_path.open("wb") as handle:
                np.save(handle, np.asarray([[0.0, 1.0]]))
            self.assertIsNone(load_embedding_cache(embedding_path, metadata_path,
                                                  expected_fingerprint="fixture-fingerprint", expected_count=1))

    def test_retriever_uses_verified_local_safetensors_and_no_reranker_fallback(self) -> None:
        from cbt_rag_v1 import DENSE_MODEL, RERANK_MODEL, Retriever
        from source_registry import SourceRegistry
        from test_source_registry import fixture_chunk, fixture_config, fixture_payload

        registry = SourceRegistry(fixture_payload())
        config = fixture_config(registry)
        chunks = [fixture_chunk(registry)]
        manifest = config.corpus_manifest(chunks, registry)
        dense = Mock()
        dense.encode.return_value = np.asarray([[1.0, 0.0]])
        dense_loader = Mock(return_value=dense)
        rerank_loader = Mock(return_value=Mock())
        fake_package = SimpleNamespace(SentenceTransformer=dense_loader, CrossEncoder=rerank_loader)
        snapshots = [SimpleNamespace(path=Path("/fixture/dense"), identity={"revision": "a" * 40}),
                     SimpleNamespace(path=Path("/fixture/reranker"), identity={"revision": "b" * 40})]
        with tempfile.TemporaryDirectory() as directory, patch("cbt_rag_v1.CACHE", Path(directory)), \
             patch.dict(sys.modules, {"sentence_transformers": fake_package}), \
             patch("cbt_rag_v1.prepare_model_snapshot", side_effect=snapshots) as prepare:
            retriever = Retriever(
                chunks, registry=registry, knowledge_config=config, manifest=manifest,
                dense_snapshot_path=Path("/preloaded/dense"), rerank_snapshot_path=Path("/preloaded/reranker"),
                model_local_files_only=True,
            )
            self.assertEqual(prepare.call_args_list, [
                call(DENSE_MODEL, local_path=Path("/preloaded/dense"), local_files_only=True),
                call(RERANK_MODEL, local_path=Path("/preloaded/reranker"), local_files_only=True),
            ])
        for loader in (dense_loader, rerank_loader):
            self.assertTrue(loader.call_args.kwargs["local_files_only"])
            self.assertFalse(loader.call_args.kwargs["trust_remote_code"])
            self.assertEqual(loader.call_args.kwargs["model_kwargs"], {"use_safetensors": True})
        self.assertEqual(retriever.model_identities["dense"], snapshots[0].identity)
        self.assertNotIn("/preloaded", json.dumps(retriever.model_identities))
        self.assertEqual(retriever.corpus_fingerprint, manifest["corpus_fingerprint"])
        self.assertIn("hybrid_rerank", retriever.available_modes())
        rerank_loader.side_effect = RuntimeError("pinned reranker cannot load")
        with tempfile.TemporaryDirectory() as directory, patch("cbt_rag_v1.CACHE", Path(directory)), \
             patch.dict(sys.modules, {"sentence_transformers": fake_package}), \
             patch("cbt_rag_v1.prepare_model_snapshot", side_effect=snapshots), \
             self.assertRaisesRegex(RuntimeError, "pinned reranker"):
            Retriever(chunks, registry=registry, knowledge_config=config, manifest=manifest)


if __name__ == "__main__":
    unittest.main()
