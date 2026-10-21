"""Small software equivalence tests, not benchmark or effectiveness results."""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for relative in ("modules/long-term-memory/schema", "modules/long-term-memory/retrieval"):
    sys.path.insert(0, str(ROOT / relative))

from memory_models import MemoryRecord, SourceEvidence  # noqa: E402
from memory_retriever import MemoryRetriever, RetrievalPolicy, _lexical_scores, _tokens  # noqa: E402


def original_lexical_scores(query, documents):
    """The pre-optimization dense-vocabulary implementation, kept as a reference."""
    tokenized = [_tokens(query), *(_tokens(value) for value in documents)]
    if not tokenized[0]:
        return [0.0] * len(documents)
    document_count = len(documents)
    vocabulary = set(tokenized[0])
    for values in tokenized[1:]:
        vocabulary.update(values)
    document_frequency = {
        token: sum(token in set(values) for values in tokenized[1:]) for token in vocabulary
    }
    idf = {token: math.log((1 + document_count) / (1 + document_frequency[token])) + 1
           for token in vocabulary}

    def vector(values):
        counts = {}
        for token in values:
            counts[token] = counts.get(token, 0) + 1
        total = len(values) or 1
        return {token: count / total * idf[token] for token, count in counts.items()}

    query_vector, scores = vector(tokenized[0]), []
    for values in tokenized[1:]:
        document_vector = vector(values)
        left = [query_vector.get(token, 0.0) for token in vocabulary]
        right = [document_vector.get(token, 0.0) for token in vocabulary]
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        scores.append(numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0)
    return scores


class LexicalScoreEquivalenceTests(unittest.TestCase):
    def assert_equivalent(self, query, documents):
        expected = original_lexical_scores(query, documents)
        actual = _lexical_scores(query, documents)
        self.assertEqual(len(actual), len(expected))
        for index, (old, new) in enumerate(zip(expected, actual)):
            self.assertTrue(math.isfinite(new))
            self.assertAlmostEqual(old, new, delta=1e-12, msg=f"document {index}: {query!r}")
        old_order = sorted(range(len(expected)), key=lambda index: (-expected[index], index))
        positions = {index: position for position, index in enumerate(
            sorted(range(len(actual)), key=lambda index: (-actual[index], index)))}
        for position, first in enumerate(old_order):
            for second in old_order[position + 1:]:
                # Equal or roundoff-close scores are ties, not ranking evidence.
                if expected[first] - expected[second] > 1e-12:
                    self.assertLess(positions[first], positions[second])

    def test_fixed_multilingual_repetition_and_query_only_terms(self):
        documents = [
            "My walking goal is ten minutes after lunch.",
            "walking walking after lunch with tea",
            "午饭后散步十分钟，这是本周目标。",
            "本周散步目标 walking ten minutes",
            "散步散步散步，目标。", "Tea TEA tea 123_abc", "unrelated blue notebook",
            "", "the and is my", "🙂!?", "café naïve résumé", "独立内容",
        ]
        for query in ("walking goal after lunch", "本周的散步目标是什么？", "walking 散步十分钟",
                      "tea tea tea 123_abc", "query_only_token 午饭", "café résumé"):
            with self.subTest(query=query):
                self.assert_equivalent(query, documents)

    def test_empty_query_documents_and_stopword_only_vectors(self):
        for query, documents in (
            ("", []), ("", ["goal", ""]), ("the and my", ["goal", ""]),
            ("goal", []), ("goal", ["", "the and", "🙂", "another"]),
            ("！🙂", ["散步", "walking"]), ("散步", ["散步", "散步"]),
        ):
            with self.subTest(query=query, documents=documents):
                self.assert_equivalent(query, documents)

    def test_deterministic_tiny_random_corpora_match_scores_and_nontied_ranks(self):
        generator = random.Random(20261021)
        words = ["walk", "walking", "TEA", "goal", "lunch", "repeat", "123_abc",
                 "午饭", "散步", "本周目标", "茶", "the", "and", "🙂", "!", ""]
        for index in range(120):
            query = " ".join(generator.choices(words, k=generator.randrange(15)))
            documents = [" ".join(generator.choices(words, k=generator.randrange(20)))
                         for _ in range(generator.randrange(10))]
            with self.subTest(case=index):
                self.assert_equivalent(query, documents)

    def test_actual_retrieval_keeps_scores_components_and_nontied_order(self):
        timestamp = "2026-10-01T00:00:00Z"
        records = [MemoryRecord(
            schema_version="1.0", memory_id=f"record-{index}", user_id="synthetic-user",
            memory_type="explicit_fact", content=text, origin="explicit_user_statement",
            confirmation_state="unconfirmed", lifecycle_status="candidate", sensitivity="standard",
            confidence=0.5, importance=0.5, created_at=timestamp, observed_at=timestamp,
            updated_at=timestamp, valid_from=timestamp,
            source_evidence=SourceEvidence("synthetic-session", [f"turn-{index}"], text),
        ) for index, text in enumerate((
            "walking goal ten minutes", "walking ten minutes after lunch", "tea after lunch",
            "散步十分钟，本周目标", "blue notebook", "walking walking ten minutes",
        ))]
        policy = RetrievalPolicy(top_k=10, lexical_weight=1.0, dense_weight=0.0,
                                 recency_weight=0.0, importance_weight=0.0, confidence_weight=0.0,
                                 minimum_relevance=0.0, allowed_lifecycle=frozenset({"candidate"}))
        retriever = MemoryRetriever(records, policy=policy)
        for query in ("walking goal ten minutes", "本周散步目标", "tea after lunch"):
            with self.subTest(query=query):
                with patch("memory_retriever._lexical_scores", original_lexical_scores):
                    expected = retriever.search(query, user_id="synthetic-user", as_of=timestamp)
                actual = retriever.search(query, user_id="synthetic-user", as_of=timestamp)
                expected_by_id = {item.record.memory_id: item for item in expected}
                for item in actual:
                    old = expected_by_id[item.record.memory_id]
                    self.assertAlmostEqual(item.score, old.score, delta=1e-12)
                    for key in old.components:
                        self.assertAlmostEqual(item.components[key], old.components[key], delta=1e-12)
                positions = {item.record.memory_id: index for index, item in enumerate(actual)}
                for index, first in enumerate(expected):
                    for second in expected[index + 1:]:
                        if first.score - second.score > 1e-12:
                            self.assertLess(positions[first.record.memory_id], positions[second.record.memory_id])


if __name__ == "__main__":
    unittest.main()
