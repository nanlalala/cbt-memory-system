"""Candidate shared repair: query dilution must not erase a validated source edge.

All sources are tiny invented assertions, not benchmark inputs or answer labels.
The legacy miss is retained as an observed regression, not rewritten history.
"""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation/objective-memory"))
sys.path.insert(0, str(ROOT / "evaluation/end-to-end-dialogue"))
from run_three_arm import HistoryRanker, HistoryTurn, Settings
from v3_source_retrieval import retrieve_sources as retrieve_v3
from v4_source_retrieval import retrieve_sources, policy_descriptor


def source(key, text, scope="fictional"):
    return HistoryTurn(scope, key, "fictional-session", "source_order:0", "context", text)


class ExactSourceEdgeRepairTests(unittest.TestCase):
    def ranker(self, top_k=12, threshold=.05):
        return HistoryRanker(Settings(model="fixture", backend="lexical",
                                      top_k=top_k, minimum_relevance=threshold))

    def test_long_question_does_not_erase_known_same_scope_source_edge(self):
        documents = [source("identity", "Orion is a citizen of Aurelia."),
                     source("capital", "The capital of Aurelia is Nova.")]
        query = "Orion citizen " + " ".join("extra" + str(i) for i in range(35))
        old, old_trace = retrieve_v3(query, documents, self.ranker(), scope_id="fictional")
        self.assertEqual([d.document_id for d in old], ["identity"])
        self.assertEqual(old_trace["rounds"][1]["candidate_count"], 1)
        self.assertEqual(old_trace["rounds"][1]["selected_ids"], [])
        ranker = self.ranker()
        repaired, trace = retrieve_sources(query, documents, ranker, scope_id="fictional")
        self.assertEqual(repaired, documents)
        self.assertEqual(trace["rounds"][1]["candidate_ids"], ["capital"])
        self.assertEqual(trace["rounds"][1]["selected_ids"], ["capital"])
        self.assertEqual(ranker.settings.minimum_relevance, .05)

    def test_edge_admission_does_not_turn_off_direct_seed_relevance(self):
        documents = [source("a", "Orion is a citizen of Aurelia."),
                     source("b", "The capital of Aurelia is Nova.")]
        selected, trace = retrieve_sources("unrelated", documents, self.ranker(), scope_id="fictional")
        self.assertEqual(selected, [])
        self.assertEqual(trace["initial_ranked_ids"], [])

    def test_non_linked_zero_overlap_sources_do_not_enter_via_disabled_gate(self):
        documents = [source("a", "Orion is a citizen of Aurelia."),
                     source("b", "The capital of Aurelia is Nova."),
                     source("unrelated", "The country of Selene is Belora.")]
        query = "Orion citizen " + " ".join("extra" + str(i) for i in range(35))
        selected, _ = retrieve_sources(query, documents, self.ranker(), scope_id="fictional")
        self.assertEqual([d.document_id for d in selected], ["a", "b"])
        self.assertTrue(all(any(d is original for original in documents) for d in selected))

    def test_shared_budget_and_recent_channel_exclusion_still_apply(self):
        recent = source("recent", "Orion is a citizen of Aurelia.")
        documents = [source("b", "The capital of Aurelia is Nova."),
                     source("c", "The continent of Nova is Coasta.")]
        selected, trace = retrieve_sources("Orion citizen", documents, self.ranker(top_k=1),
                                           scope_id="fictional", seed_documents=[recent])
        self.assertEqual([d.document_id for d in selected], ["b"])
        self.assertNotIn("recent", trace["selected_ids"])
        self.assertEqual(trace["common_recent_seed_ids"], ["recent"])

    def test_cross_scope_and_same_channel_source_aliases_fail_closed(self):
        own = source("a", "Orion is a citizen of Aurelia.")
        foreign = source("b", "The capital of Aurelia is Nova.", scope="other")
        with self.assertRaisesRegex(ValueError, "cross-subject"):
            retrieve_sources("Orion citizen", [own, foreign], self.ranker(), scope_id="fictional")
        with self.assertRaisesRegex(ValueError, "separate"):
            retrieve_sources("Orion citizen", [own], self.ranker(), scope_id="fictional", seed_documents=[own])

    def test_unsupported_dense_backend_is_rejected_before_any_model_load(self):
        ranker = HistoryRanker(Settings(model="fixture", backend="dense"))
        with self.assertRaisesRegex(ValueError, "lexical retrieval only"):
            retrieve_sources("Orion", [source("a", "Orion is a citizen of Aurelia.")],
                             ranker, scope_id="fictional")
        self.assertIsNone(ranker.encoder)

    def test_policy_does_not_claim_an_executed_effect_or_managed_only_treatment(self):
        policy = policy_descriptor()
        self.assertTrue(policy["shared_repair_not_managed_treatment"])
        self.assertIn("no new benchmark effectiveness result", policy["evaluation_status"])
        self.assertFalse(policy["facts_rewritten"])
        self.assertFalse(policy["gold_or_external_knowledge_used"])


if __name__ == "__main__":
    unittest.main()
