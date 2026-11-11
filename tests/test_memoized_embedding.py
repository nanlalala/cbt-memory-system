"""Tiny deterministic encoder fixtures, without model loading or network."""
from array import array
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("memoized_embedding", Path(__file__).resolve().parents[1] /
    "evaluation/cbt-memory-effect/memoized_embedding.py")
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


class Encoder:
    model_identity = {"repo": "fixture", "revision": "pinned"}
    model = object()
    def __init__(self): self.calls = []; self.last = None
    def encode(self, texts, *, is_query):
        self.calls.append((list(texts), is_query))
        self.last = [array("f", [len(text), int(is_query)]) for text in texts]
        return self.last


class MemoizedEmbeddingTests(unittest.TestCase):
    def test_exact_identity_query_flag_and_text_key_with_batch_dedup(self):
        encoder = Encoder(); memo = M.MemoizedEmbedding(encoder)
        result = memo.encode(["same", "same", " same"], is_query=False)
        memo.encode(["same", " same"], is_query=False)
        memo.encode(["same"], is_query=True)
        self.assertEqual(encoder.calls, [(["same", " same"], False), (["same"], True)])
        self.assertEqual(result[0], result[1])
        stats = memo.statistics()
        self.assertEqual(stats["logical_encode_calls_attempted"], 3)
        self.assertEqual(stats["logical_encode_texts_attempted"], 6)
        self.assertEqual(stats["physical_encode_calls_attempted"], 2)
        self.assertEqual(stats["physical_encode_texts_attempted"], 3)
        self.assertEqual(stats["cache_hit_texts"], 3)
        self.assertEqual(stats["cached_entries"], 3)
        self.assertIs(memo.model, encoder.model)

    def test_return_and_delegate_mutation_cannot_change_cached_vector_or_dtype(self):
        encoder = Encoder(); memo = M.MemoizedEmbedding(encoder)
        first = memo.encode(["same"], is_query=False)
        self.assertIsInstance(first[0], array)
        first[0][0] = 999; encoder.last[0][0] = 888
        second = memo.encode(["same"], is_query=False)
        self.assertEqual(second[0], array("f", [4, 0]))
        self.assertIsInstance(second[0], array)

    def test_new_instance_has_no_cross_run_cache_and_identity_change_rejected(self):
        encoder = Encoder(); first = M.MemoizedEmbedding(encoder)
        first.encode(["same"], is_query=False)
        second = M.MemoizedEmbedding(encoder)
        second.encode(["same"], is_query=False)
        self.assertEqual(len(encoder.calls), 2)
        encoder.model_identity = {"repo": "fixture", "revision": "changed"}
        with self.assertRaisesRegex(ValueError, "identity changed"):
            first.encode(["same"], is_query=False)

    def test_failed_batch_does_not_install_partial_cache(self):
        class Bad(Encoder):
            def encode(self, texts, *, is_query): return []
        memo = M.MemoizedEmbedding(Bad())
        with self.assertRaisesRegex(ValueError, "output count"):
            memo.encode(["same"], is_query=False)
        stats = memo.statistics()
        self.assertEqual(stats["physical_encode_calls_attempted"], 1)
        self.assertEqual(stats["physical_encode_calls_completed"], 0)
        self.assertEqual(stats["cached_entries"], 0)


if __name__ == "__main__": unittest.main()
