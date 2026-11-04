"""Bounded shared retrieval follows source entities, never answer labels."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation/objective-memory"))
sys.path.insert(0, str(ROOT / "evaluation/end-to-end-dialogue"))
from run_three_arm import HistoryTurn, HistoryRanker, Settings
from v3_source_retrieval import retrieve_sources, normalize_entity


def doc(key, text, *, scope="fixture", session="s", order=0):
    return HistoryTurn(scope_id=scope, document_id=key, session_id=session,
                       date=f"source_order:{order}", speaker="context", text=text)


class SourceGraphTests(unittest.TestCase):
    def ranker(self, top_k=3):
        return HistoryRanker(Settings(model="fixture", backend="lexical", top_k=top_k, minimum_relevance=.01))

    def test_source_objects_recover_chain_missing_from_direct_query(self):
        documents = [doc("person", "Orion is a citizen of Aurelia."),
                     doc("country", "The capital of Aurelia is Nova."),
                     doc("city", "The continent of Nova is Coasta.")]
        ranker = self.ranker()
        self.assertEqual([d.document_id for d in ranker.rank("Orion citizen", documents, scope_id="fixture")], ["person"])
        selected, trace = retrieve_sources("Orion citizen", documents, ranker, scope_id="fixture")
        self.assertEqual(selected, documents)
        self.assertEqual(trace["rounds"][1]["entities"], ["aurelia"])
        self.assertEqual(trace["rounds"][2]["entities"], ["nova"])
        self.assertEqual(len(trace["rounds"]), 3)

    def test_shared_policy_does_not_rewrite_quotes_or_exceed_budget(self):
        documents = [doc("person", "Orion is a citizen of Aurelia."),
                     doc("country", "The capital of Aurelia is Nova."),
                     doc("city", "The continent of Nova is Coasta."),
                     doc("extra", "Orion has several historical records.")]
        selected, _ = retrieve_sources("Orion citizen", documents, self.ranker(2), scope_id="fixture")
        self.assertLessEqual(len(selected), 2)
        self.assertEqual(len({d.document_id for d in selected}), len(selected))
        self.assertTrue(all(any(d is original for original in documents) for d in selected))
        repeated, _ = retrieve_sources("Orion citizen", documents, self.ranker(2), scope_id="fixture")
        self.assertEqual(selected, repeated)

    def test_unparsed_sources_remain_eligible_and_fill_unused_slots(self):
        documents = [doc("a", "Orion remembers a complicated childhood."),
                     doc("b", "Orion discussed an ambiguous earlier account.")]
        selected, trace = retrieve_sources("Orion", documents, self.ranker(3), scope_id="fixture")
        self.assertEqual({d.document_id for d in selected}, {"a", "b"})
        self.assertEqual(len(trace["rounds"]), 1)

    def test_cross_scope_rejected_instead_of_silently_accessed(self):
        documents = [doc("a", "Orion is a citizen of Aurelia.", scope="someone-else")]
        with self.assertRaisesRegex(ValueError, "cross-subject"):
            retrieve_sources("Orion", documents, self.ranker(), scope_id="fixture")

    def test_exact_source_replacement_invalidates_graph_cache(self):
        documents = [doc("a", "Orion is a citizen of Aurelia."),
                     doc("b", "The capital of Aurelia is Nova.")]
        ranker = self.ranker()
        selected, _ = retrieve_sources("Orion citizen", documents, ranker, scope_id="fixture")
        self.assertEqual(len(selected), 2)
        changed = [replace(documents[0], text="Orion is a citizen of Belora."), documents[1]]
        selected, trace = retrieve_sources("Orion citizen", changed, ranker, scope_id="fixture")
        self.assertEqual([d.document_id for d in selected], ["a"])
        self.assertEqual(trace["rounds"][1]["entities"], ["belora"])

    def test_entity_normalization_is_exact_not_substring_matching(self):
        self.assertEqual(normalize_entity(" The Aurelia. "), "aurelia")
        documents = [doc("a", "Orion is a citizen of Aurelia."),
                     doc("b", "The capital of Greater Aurelia is Nova.")]
        selected, _ = retrieve_sources("Orion citizen", documents, self.ranker(), scope_id="fixture")
        self.assertEqual([d.document_id for d in selected], ["a"])

    def test_common_recent_seed_opens_older_chain_without_returning_seed(self):
        old = doc("old-identity", "Orion is a citizen of Aurelia.", order=0)
        downstream = doc("country", "The capital of Belora is Nova.", order=1)
        recent = doc("new-identity", "Orion is a citizen of Belora.", order=2)
        ranker = self.ranker(3)
        # Version consolidation can remove the old identity from the managed
        # older view; the shared current source must still open Belora's link.
        self.assertEqual(ranker.rank("Orion citizenship", [downstream], scope_id="fixture"), [])
        selected, trace = retrieve_sources("Orion citizenship", [downstream], ranker,
                                           scope_id="fixture", seed_documents=[recent])
        self.assertEqual(selected, [downstream])
        self.assertEqual(trace["common_recent_seed_ids"], ["new-identity"])
        self.assertNotIn(recent.document_id, trace["selected_ids"])
        # The raw arm receives exactly the same common recent seed policy.
        raw, raw_trace = retrieve_sources("Orion citizenship", [old, downstream], self.ranker(3),
                                          scope_id="fixture", seed_documents=[recent])
        self.assertIn(downstream, raw)
        self.assertEqual(raw_trace["common_recent_seed_ids"], trace["common_recent_seed_ids"])

    def test_seed_ownership_duplicates_and_channel_overlap_are_rejected(self):
        recent = doc("recent", "Orion is a citizen of Belora.")
        foreign = replace(recent, scope_id="foreign")
        for older, seeds, error in (([], [foreign], "cross-subject"),
                                    ([], [recent, recent], "unique recent"),
                                    ([recent], [recent], "separate")):
            with self.assertRaisesRegex(ValueError, error):
                retrieve_sources("Orion", older, self.ranker(), scope_id="fixture", seed_documents=seeds)


if __name__ == "__main__":
    unittest.main()
