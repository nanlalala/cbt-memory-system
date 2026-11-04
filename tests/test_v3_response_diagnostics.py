"""Frozen supplemental metrics catch refusal/list credit without rewriting gold."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation/objective-memory"))
from v3_response_diagnostics import (
    development_partition, partition_from_exposed_cases, policy_descriptor,
    response_format_diagnostics, score_supplemental,
)


class ResponseDiagnosticTests(unittest.TestCase):
    def mab(self, aliases):
        return {"metric": "mab_substring_exact_match", "accepted_answers": aliases}

    def test_official_refusal_containing_gold_keeps_point_but_not_strict_commitment(self):
        row = score_supplemental("I don't know. The history lists Jack Dorsey and Bernard Arnault.", self.mab(["Bernard Arnault"]))
        self.assertEqual(row["official_score"], 1)
        self.assertEqual(row["strict_committed_score"], 0)
        self.assertTrue(row["official_positive_with_abstention_cue"])

    def test_full_alias_not_answer_substring_required(self):
        ref = self.mab(["Paris"])
        self.assertEqual(score_supplemental("Answer: Paris", ref)["strict_committed_score"], 1)
        verbose = score_supplemental("The capital is Paris.", ref)
        self.assertEqual(verbose["official_score"], 1)
        self.assertEqual(verbose["exact_alias_or_option_score"], 0)

    def test_disjunctive_guesses_not_exact_single_alias(self):
        row = score_supplemental("Paris or London", self.mab(["Paris"]))
        self.assertEqual(row["official_score"], 1)
        self.assertEqual(row["strict_committed_score"], 0)
        self.assertTrue(row["format"]["disjunction_cue_detected"])

    def test_alias_with_conjunction_is_not_rejected_as_guessed_list(self):
        row = score_supplemental("Bosnia and Herzegovina", self.mab(["Bosnia and Herzegovina"]))
        self.assertEqual(row["strict_committed_score"], 1)

    def test_upstream_normalization_preserved(self):
        self.assertEqual(score_supplemental("The Beatles.", self.mab(["Beatles"]))["strict_committed_score"], 1)
        self.assertEqual(score_supplemental("Unknown", self.mab(["Somebody"]))["official_score"], 0)

    def test_format_flags_are_gold_blind_and_no_entailment_claim(self):
        row = response_format_diagnostics("A confident but wrong entity")
        self.assertTrue(row["short_single_line_without_abstention"])
        self.assertNotIn("score", row)
        self.assertIsNone(row["bare_option_label"])
        self.assertTrue(response_format_diagnostics("UNKNOWN")["abstention_cue_detected"])
        self.assertEqual(response_format_diagnostics("Paris\nLondon")["nonempty_line_count"], 2)
        self.assertFalse(response_format_diagnostics("")["short_single_line_without_abstention"])
        self.assertFalse(response_format_diagnostics("x" * 161)["short_single_line_without_abstention"])

    def test_personamem_requires_bare_option_for_secondary_only(self):
        ref = {"metric": "personamem_option_exact_match", "correct_option": "(b)"}
        self.assertEqual(score_supplemental("(b)", ref)["strict_committed_score"], 1)
        self.assertEqual(score_supplemental("Answer: B", ref)["strict_committed_score"], 1)
        self.assertIsNone(response_format_diagnostics("(b")["bare_option_label"])
        self.assertIsNone(response_format_diagnostics("b)")["bare_option_label"])
        verbose = score_supplemental("(b), because the preference changed.", ref)
        self.assertEqual(verbose["official_score"], 1)
        self.assertEqual(verbose["strict_committed_score"], 0)

    def test_policy_is_jsonable_and_retains_interpretation(self):
        policy = policy_descriptor()
        json.dumps(policy)
        self.assertIn("unchanged", policy["official_metric"])
        self.assertIn("not semantic", policy["interpretation"])


class DevelopmentPartitionTests(unittest.TestCase):
    def case(self, cid, query):
        return {"case_id": cid, "query": query, "dataset_key": "memoryagentbench_conflict"}

    def test_variants_and_same_query_expanded_conservatively(self):
        cases = [self.case("mab:factconsolidation_sh_6k_no1", "Who leads Example?"),
                 self.case("mab:factconsolidation_sh_32k_no1", "A renumbered query?"),
                 self.case("mab:factconsolidation_sh_64k_no99", "  WHO leads Example? "),
                 self.case("mab:factconsolidation_mh_6k_no1", "A different multi-hop task?"),
                 self.case("mab:factconsolidation_sh_6k_no2", "Unseen question?")]
        result = development_partition(cases, [cases[0]["case_id"]])
        self.assertEqual(result["development_count"], 2)
        self.assertEqual(result["question_heldout_count"], 3)
        self.assertFalse(result["group_independence_claim"])
        self.assertIn("same_normalized_question_text", result["development_reasons"][cases[2]["case_id"]])

    def test_missing_exposure_and_duplicate_cases_refused(self):
        cases = [self.case("q", "question")]
        with self.assertRaises(ValueError):
            development_partition(cases, ["missing"])
        with self.assertRaises(ValueError):
            development_partition(cases * 2, [])

    def test_same_ordinal_is_not_identity_and_different_ordinals_can_be_same_question(self):
        cases = [self.case("mab:factconsolidation_sh_6k_no80", "What position does Lisa Leslie play?"),
                 self.case("mab:factconsolidation_sh_32k_no38", " WHAT position does Lisa Leslie play？ "),
                 self.case("mab:factconsolidation_sh_32k_no80", "What nationality is Another Person?")]
        result = development_partition(cases, [cases[0]["case_id"]])
        self.assertEqual(result["development_case_ids"], [cases[0]["case_id"], cases[1]["case_id"]])
        self.assertEqual(result["question_heldout_case_ids"], [cases[2]["case_id"]])

    def test_frozen_exposure_metadata_must_match_queries(self):
        case = self.case("q1", "Which city?")
        result = partition_from_exposed_cases([case], [dict(case)])
        self.assertEqual(result["development_count"], 1)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            partition_from_exposed_cases([case], [{**case, "query": "Which country?"}])


if __name__ == "__main__":
    unittest.main()
