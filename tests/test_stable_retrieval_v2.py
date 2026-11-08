from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for relative in ("modules/long-term-memory/schema", "modules/long-term-memory/retrieval"):
    sys.path.insert(0, str(ROOT / relative))

from stable_retrieval import StableSourceRetriever


CUTOFF = "2026-11-08T12:00:00Z"


def source(source_id, text, **kwargs):
    return {
        "source_id": source_id, "user_id": "alice", "text": text,
        "source_time": "2026-11-01T12:00:00Z", "role": "user",
        "permissions": {"save": True, "long_term_use": True,
                        "response_external": True, "extraction_external": True},
        **kwargs,
    }


class SemanticEncoder:
    def encode(self, texts, *, is_query):
        return [[1., 0.] if ("fitness" in text or "running" in text) else [0., 1.] for text in texts]


class StableRetrievalTests(unittest.TestCase):
    def search(self, retriever, query="walking record", **kwargs):
        return retriever.search(query, user_id="alice", source_cutoff=CUTOFF,
                                minimum_relevance=.001, **kwargs)

    def test_soft_routing_and_extra_spans_do_not_refit_source_statistics(self):
        text = "walking twice; thought record paused"
        split = text.index(";")
        first = {"span_id": "walk", "start": 0, "end": split, "category": "V4"}
        second = {"span_id": "pause", "start": split + 2, "end": len(text), "category": "V4"}
        other = source("s2", "thought record was discussed")
        raw = StableSourceRetriever([source("s1", text, spans=[first]), other])
        managed = StableSourceRetriever([source("s1", text, spans=[first, second]), other])
        a = self.search(raw, categories=["F1"])
        b = self.search(managed, categories=["V4"])
        self.assertEqual(a.trace["statistics_pool_fingerprint"], b.trace["statistics_pool_fingerprint"])
        raw_walk = next(hit for hit in a if hit.span_id == "walk")
        managed_walk = next(hit for hit in b if hit.span_id == "walk")
        self.assertEqual(raw_walk.components["lexical"], managed_walk.components["lexical"])

    def test_derived_notes_do_not_enter_or_change_the_quotation_pool(self):
        original = source("s1", "walking after lunch")
        note = source("derived", "walking " * 50, representation_kind="semantic_summary")
        a = self.search(StableSourceRetriever([original]))
        b = self.search(StableSourceRetriever([original, note]))
        self.assertEqual(a.trace["statistics_pool_fingerprint"], b.trace["statistics_pool_fingerprint"])
        self.assertEqual([h.source_id for h in b], ["s1"])

    def test_dense_only_match_is_recalled_without_lexical_gate(self):
        retriever = StableSourceRetriever([source("sport", "running"), source("art", "painting")], encoder=SemanticEncoder())
        result = self.search(retriever, "fitness")
        self.assertEqual(result.trace["lexical_recall"], [])
        self.assertEqual([h.source_id for h in result], ["sport"])
        self.assertEqual(result.hits[0].components["lexical"], 0)

    def test_multiple_necessary_spans_from_one_source_survive(self):
        text = "散步两次；记录暂停"
        retriever = StableSourceRetriever([source("s1", text, spans=[
            {"span_id": "walk", "start": 0, "end": 4},
            {"span_id": "record", "start": 5, "end": 9},
        ])])
        result = self.search(retriever, "散步记录")
        self.assertEqual({h.span_id for h in result}, {"walk", "record"})
        self.assertEqual(result.trace["unique_source_count"], 1)
        self.assertEqual(result.trace["selected_span_count"], 2)

    def test_late_report_cannot_cross_source_cutoff_using_early_event_time(self):
        report = source("late", "walking was completed", source_time="2026-11-09T00:00:00Z",
                        event_time="2026-10-01T00:00:00Z")
        result = self.search(StableSourceRetriever([report]), as_of="2026-10-02T00:00:00Z", query_mode="history")
        self.assertEqual(len(result), 0)
        self.assertEqual(result.trace["hard_excluded_counts"]["after_source_cutoff"], 1)

    def test_history_does_not_restore_revoked_current_permission(self):
        retriever = StableSourceRetriever([source("s1", "walking twice")])
        result = self.search(retriever, query_mode="history", as_of="2026-11-01T12:00:00Z",
                             current_permissions={"s1": {"long_term_use": False}})
        self.assertEqual(len(result), 0)
        self.assertEqual(result.trace["statistics_source_count"], 0)

    def test_user_scope_and_revocation_precede_pool_statistics(self):
        retriever = StableSourceRetriever([
            source("own", "walking"), source("gone", "walking"),
            source("other", "walking", user_id="bob"),
        ])
        result = self.search(retriever, revoked_source_ids=["gone"])
        self.assertEqual(result.trace["statistics_source_ids"], ["own"])
        self.assertNotIn("other", str(result.trace))

    def test_unknown_permissions_do_not_grant_read_or_external_use(self):
        unresolved = source("unknown", "walking", permissions={"save": True})
        restricted = source("local", "walking", permissions={"save": True, "long_term_use": True,
                                                               "response_external": None, "extraction_external": None})
        retriever = StableSourceRetriever([unresolved, restricted])
        self.assertEqual([h.source_id for h in self.search(retriever)], ["local"])
        self.assertEqual(len(self.search(retriever, for_external_response=True)), 0)
        with self.assertRaises(ValueError):
            StableSourceRetriever([source("bad", "walking", permissions={"long_term_use": "true"})])

    def test_unknown_legacy_role_or_time_is_not_filled_from_clock(self):
        retriever = StableSourceRetriever([source("role", "walking", role=None),
                                           source("time", "walking", source_time=None)])
        self.assertEqual(len(self.search(retriever)), 0)

    def test_state_applicability_is_separate_from_visibility(self):
        text = "walking twice"
        retriever = StableSourceRetriever([source("s1", text, spans=[{
            "span_id": "old", "start": 0, "end": len(text),
            "valid_from": "2026-11-01T00:00:00Z", "valid_to": "2026-11-02T00:00:00Z",
        }])])
        historical = self.search(retriever, query_mode="history", as_of="2026-11-01T12:00:00Z")
        current = self.search(retriever, query_mode="current", as_of="2026-11-02T00:00:00Z")
        self.assertEqual(len(historical), 1)
        self.assertEqual(len(current), 0)
        self.assertEqual(historical.trace["statistics_source_count"], current.trace["statistics_source_count"])

    def test_exact_unicode_spans_and_conflicting_source_snapshots_are_rejected(self):
        bad = source("s1", "散步两次", spans=[{"start": 0, "end": 4, "text": "散步三次"}])
        with self.assertRaises(ValueError):
            self.search(StableSourceRetriever([bad]))
        with self.assertRaises(ValueError):
            StableSourceRetriever([source("s1", "walking"), source("s1", "different")])

    def test_no_relevant_source_returns_empty(self):
        result = self.search(StableSourceRetriever([source("s1", "walking after lunch")]), "database configuration")
        self.assertEqual(result.hits, ())


if __name__ == "__main__":
    unittest.main()
