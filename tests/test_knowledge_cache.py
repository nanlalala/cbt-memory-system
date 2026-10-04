from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
