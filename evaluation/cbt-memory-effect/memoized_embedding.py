"""Per-run, in-memory exact embedding reuse; no I/O or model loading.

The caller supplies its already pinned local encoder. Cached vectors retain
their original numeric dtype, and are copied on both storage and retrieval.
"""
from copy import deepcopy
import json
import time

VERSION = "exact-text-embedding-memoization-v1"


def _identity(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


class MemoizedEmbedding:
    def __init__(self, encoder):
        if not getattr(encoder, "model_identity", None):
            raise ValueError("explicit pinned model identity required")
        self.encoder = encoder
        self.model_identity = deepcopy(encoder.model_identity)
        self._identity = _identity(self.model_identity)
        self._cache = {}
        self._counts = {key: 0 for key in (
            "logical_encode_calls_attempted", "logical_encode_calls_completed",
            "logical_encode_texts_attempted", "logical_encode_texts_completed",
            "physical_encode_calls_attempted", "physical_encode_calls_completed",
            "physical_encode_texts_attempted", "physical_encode_texts_completed",
            "cache_hit_texts", "existing_cache_hit_texts", "batch_duplicate_texts")}
        self._elapsed = 0.0

    @property
    def model(self):
        """Expose the underlying tokenizer model without changing its settings."""
        return getattr(self.encoder, "model", None)

    def encode(self, texts, *, is_query):
        if type(is_query) is not bool or isinstance(texts, (str, bytes)):
            raise ValueError("explicit query flag and text sequence required")
        texts = list(texts)
        if any(type(text) is not str for text in texts):
            raise ValueError("each embedding input must be exact text")
        if (_identity(self.encoder.model_identity) != self._identity or
                _identity(self.model_identity) != self._identity):
            raise ValueError("embedding model identity changed during this run")
        self._counts["logical_encode_calls_attempted"] += 1
        self._counts["logical_encode_texts_attempted"] += len(texts)
        keys = [(self._identity, is_query, text) for text in texts]
        missing = []; missing_keys = set()
        for key in keys:
            if key in self._cache:
                self._counts["existing_cache_hit_texts"] += 1
            elif key in missing_keys:
                self._counts["batch_duplicate_texts"] += 1
            else:
                missing.append(key); missing_keys.add(key)
        self._counts["cache_hit_texts"] += len(keys) - len(missing)
        if missing:
            self._counts["physical_encode_calls_attempted"] += 1
            self._counts["physical_encode_texts_attempted"] += len(missing)
            started = time.perf_counter()
            try:
                vectors = self.encoder.encode([key[2] for key in missing], is_query=is_query)
                if len(vectors) != len(missing):
                    raise ValueError("encoder output count differs from missed text count")
                copied = [deepcopy(vector) for vector in vectors]
                # A failed batch is never partially installed into the cache.
                self._cache.update(zip(missing, copied))
                self._counts["physical_encode_calls_completed"] += 1
                self._counts["physical_encode_texts_completed"] += len(missing)
            finally:
                self._elapsed += time.perf_counter() - started
        result = [deepcopy(self._cache[key]) for key in keys]
        self._counts["logical_encode_calls_completed"] += 1
        self._counts["logical_encode_texts_completed"] += len(texts)
        return result

    def statistics(self):
        return {**self._counts, "physical_encode_seconds_sum": self._elapsed,
                "cached_entries": len(self._cache)}
