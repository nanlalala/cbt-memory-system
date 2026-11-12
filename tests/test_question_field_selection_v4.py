"""Tiny invented fixtures only: no dataset, tokenizer, model or credentials."""
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules/agent-integration"))
from evidence_bundles import EvidenceBundle, EvidenceClaim, EvidenceSpan
from question_field_selection_v4 import (question_field_contract, select_question_field,
                                         select_question_fields)


class QuestionIntentTests(unittest.TestCase):
    def test_bilingual_count_axes_are_distinct(self):
        for question in ("What is my current target count for breathing?",
                         "How many times did I agree to complete breathing?",
                         "我目前同意的呼吸练习目标次数是多少？"):
            self.assertEqual(select_question_field(question), "target_count")
        for question in ("How many times have I completed breathing so far?",
                         "What is the cumulative reported count for breathing?",
                         "我总共完成了几次呼吸练习？"):
            self.assertEqual(select_question_field(question), "reported_count")

    def test_practice_outcome_is_not_progress(self):
        for question in ("What was my reported outcome for breathing?",
                         "Was breathing helpful?", "呼吸练习有没有帮助？",
                         "练习的效果是什么？"):
            self.assertEqual(select_question_field(question), "outcome")
        self.assertEqual(question_field_contract("Was breathing helpful?")["value_kind"], "status")

    def test_symptom_change_and_severity_are_not_practice_outcome(self):
        for question in ("Has my anxiety improved?", "What is my symptom severity?",
                         "How has my mood changed?", "我的焦虑有没有改善？",
                         "焦虑程度是多少？"):
            self.assertIsNone(select_question_field(question))

    def test_multi_axis_and_nonquery_preserve_complete_context(self):
        for question in ("Was it helpful, and how many times did I complete it?",
                         "How many times was the practice helpful?",
                         "How many did I complete versus my target count?",
                         "练习的效果和完成次数分别是多少？",
                         "I found breathing helpful.", "我完成了呼吸练习。",
                         "Please summarize my situation."):
            self.assertIsNone(select_question_field(question), question)

    def test_agreement_and_participation_remain_separate(self):
        self.assertEqual(select_question_field("Did I agree to breathing?"), "acceptance")
        self.assertEqual(select_question_field("我有没有同意试试呼吸练习？"), "acceptance")
        self.assertEqual(select_question_field("Am I currently paused?"), "engagement")
        self.assertEqual(select_question_field("我现在是否暂停了练习？"), "engagement")

    def test_no_arm_or_label_input_and_invalid_question_rejected(self):
        with self.assertRaises(TypeError):
            select_question_field({"expected_answer": 5})
        self.assertEqual(question_field_contract("What should we discuss?"), {
            "version": question_field_contract("What should we discuss?")["version"],
            "field": None, "value_kind": "unrestricted_scalar"})


class ClaimSelectionTests(unittest.TestCase):
    def setUp(self):
        self.text = "I completed breathing four times. It was unhelpful."
        self.source = EvidenceSpan("fictional-source-a", "full-a", self.text, 0, len(self.text),
                                   "2026-01-05T12:00:00+00:00", "user")
        self.quote = EvidenceClaim("quote-a", "history_quote", self.text, (self.source,),
                                   support_status="source_quote")
        self.progress = EvidenceClaim("progress-a", "reported_count", 4, (self.source,),
                                      support_status="rule_verified")
        self.outcome = EvidenceClaim("outcome-a", "outcome", "unhelpful", (self.source,),
                                     support_status="rule_verified")
        self.pause = EvidenceClaim("pause-a", "engagement", "paused", (self.source,),
                                   support_status="rule_verified")
        self.unknown = EvidenceClaim("unknown-a", "target_count", None, (self.source,),
                                     support_status="rule_verified")
        self.version = EvidenceClaim("version-a", "version", 9, (self.source,),
                                     support_status="rule_verified")
        self.bundles = (EvidenceBundle("task-a", (self.progress, self.outcome, self.pause,
                                                  self.unknown, self.version), "breathing-a"),
                        EvidenceBundle("source-a", (self.quote,)))

    def test_outcome_query_withholds_progress_but_preserves_original_and_controls(self):
        chosen = select_question_fields("Was breathing helpful?", self.bundles)
        self.assertEqual(chosen.field, "outcome")
        self.assertEqual(chosen.omitted_claim_ids, ("progress-a",))
        claims = {c.claim_id: c for b in chosen.bundles for c in b.claims}
        for original in (self.quote, self.outcome, self.pause, self.unknown, self.version):
            self.assertIs(claims[original.claim_id], original)
        self.assertEqual(self.bundles[0].claims[0], self.progress)

    def test_count_query_withholds_outcome_and_retains_count(self):
        chosen = select_question_fields("How many times have I completed breathing?", self.bundles)
        self.assertEqual(chosen.omitted_claim_ids, ("outcome-a",))
        self.assertIn(self.progress, chosen.bundles[0].claims)

    def test_uncovered_source_provenance_cannot_disappear(self):
        other = EvidenceSpan("fictional-source-b", "full-b", "It was helpful.", 0, 15,
                             "2026-01-06T12:00:00+00:00", "user")
        extra = EvidenceClaim("outcome-b", "outcome", "helpful", (other,),
                              support_status="rule_verified")
        bundles = (*self.bundles, EvidenceBundle("task-b", (extra,), "breathing-b"))
        chosen = select_question_fields("How many times have I completed breathing?", bundles)
        self.assertIn("outcome-b", chosen.retained_for_provenance)
        self.assertIn(extra, chosen.bundles[-1].claims)

    def test_subspan_coverage_is_exact_and_role_time_sensitive(self):
        short = EvidenceSpan("fictional-source-a", "sub-a", "unhelpful", 40, 49,
                             self.source.source_time, "user")
        # Invalid offset is valid structurally but not covered at its asserted
        # source position. Similar words elsewhere do not authorize removal.
        self.assertNotEqual(self.text[short.start:short.end], short.text)
        claim = EvidenceClaim("wrong-offset", "outcome", "unhelpful", (short,),
                              support_status="rule_verified")
        chosen = select_question_fields("How many times have I completed breathing?",
                                       (*self.bundles, EvidenceBundle("bad", (claim,))))
        self.assertIn("wrong-offset", chosen.retained_for_provenance)
        real_start = self.text.index("unhelpful")
        precise = EvidenceSpan("fictional-source-a", "sub-real", "unhelpful", real_start,
                               real_start + 9, self.source.source_time, "user")
        real = EvidenceClaim("good-offset", "outcome", "unhelpful", (precise,),
                             support_status="rule_verified")
        selected = select_question_fields("How many times have I completed breathing?",
                                         (*self.bundles, EvidenceBundle("good", (real,))))
        self.assertIn("good-offset", selected.omitted_claim_ids)

    def test_all_alternative_evidence_must_be_retained(self):
        other = EvidenceSpan("fictional-source-b", "full-b", "Breathing was helpful.")
        claim = EvidenceClaim("two-alternatives", "outcome", "helpful",
                              evidence_options=((self.source,), (other,)),
                              support_status="rule_verified")
        selected = select_question_fields("How many times have I completed breathing?",
                                         (*self.bundles, EvidenceBundle("alternatives", (claim,))))
        self.assertIn("two-alternatives", selected.retained_for_provenance)

    def test_raw_quotation_only_arm_is_identical(self):
        bundles = (self.bundles[1],)
        selected = select_question_fields("Was breathing helpful?", bundles)
        self.assertIs(selected.bundles[0], bundles[0])
        self.assertEqual(selected.omitted_claim_ids, ())

    def test_ambiguous_intent_is_noop(self):
        selected = select_question_fields("Was it helpful and how many did I complete?", self.bundles)
        self.assertEqual(selected.bundles, self.bundles)
        for old, new in zip(self.bundles, selected.bundles):
            self.assertIs(old, new)

    def test_unknown_and_unverified_fields_not_erased(self):
        statuses = [None, "unknown", "blocked", "deleted", "expired", "revoked"]
        controls = tuple(EvidenceClaim("barrier-" + str(i), "outcome", value, (self.source,),
                                       support_status="rule_verified") for i, value in enumerate(statuses))
        inferred = EvidenceClaim("unverified", "outcome", "helpful", (self.source,),
                                  support_status="unverified", origin="agent_inference")
        selected = select_question_fields("How many times have I completed breathing?",
                                         (*self.bundles, EvidenceBundle("barriers", (*controls, inferred))))
        self.assertEqual(selected.bundles[-1].claims, (*controls, inferred))


if __name__ == "__main__":
    unittest.main()
