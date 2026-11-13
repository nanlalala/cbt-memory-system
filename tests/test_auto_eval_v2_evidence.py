"""Invented facts exercise accounting; no experimental data or inference."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("auto_v2_evidence", ROOT / "evaluation/cbt-memory-effect/auto_eval_v2/evidence.py")
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)


def source(sid, order, text, **changes):
    return {"source_id": sid, "source_order": order, "role": "user", "content": text,
            "retained": True, "use_allowed": True, **changes}


def gold(case="case-1", family="family-a", **changes):
    return {"case_id": case, "source_family": family, "label_provenance": "rule_verified",
            "required_facts": [{"id": "paused", "allowed_source_ids": ["u2"]}],
            "forbidden_or_stale_fact_ids": ["continuing"], **changes}


SOURCES = [source("u1", 1, "I will continue the exercise."),
           source("u2", 2, "I have paused the exercise.")]


def claim(ident="c1", fact="paused", sid="u2", quote="paused the exercise", verdict="supported"):
    return {"claim_id": ident, "fact_id": fact, "source_ids": [sid] if sid else [], "verdict": verdict,
            "evidence": [{"source_id": sid, "quote": quote}] if quote is not None else []}


def adjudication(claims=None, status="answered", provenance="rule_verified"):
    return {"status": status, "label_provenance": provenance,
            "claims": [claim()] if claims is None else claims}


def score(g=None, a=None, sources=None, **kw):
    return evidence.score_memory(g or gold(), a or adjudication(), SOURCES if sources is None else sources, 2, **kw)


class EvidenceAccountingTests(unittest.TestCase):
    def test_new_and_old_together_still_count_stale(self):
        result = score(a=adjudication([claim(), claim("c2", "continuing", "u1", "continue the exercise")]))
        self.assertEqual(result["metrics"]["recall"], 1)
        self.assertEqual(result["metrics"]["precision"], .5)
        self.assertEqual(result["counts"]["stale"], 1)
        self.assertAlmostEqual(result["metrics"]["f1"], 2 / 3)

    def test_repeated_correct_fact_does_not_hide_an_invention(self):
        bad = claim("invented", None, None, None, "unsupported")
        result = score(a=adjudication([claim(), claim("repeat"), bad]))
        self.assertEqual(result["counts"]["raw_claims"], 3)
        self.assertEqual(result["counts"]["deduplicated_facts_or_unmapped_claims"], 2)
        self.assertEqual(result["metrics"]["precision"], .5)

    def test_repeated_conflicting_verdict_uses_worst(self):
        result = score(a=adjudication([claim(), claim("bad", verdict="contradicted")]))
        self.assertEqual(result["metrics"]["recall"], 0)
        self.assertEqual(result["counts"]["contradicted"], 1)

    def test_source_id_is_not_semantic_entailment(self):
        unsupported = claim(verdict="contradicted")
        result = score(a=adjudication([unsupported]))
        self.assertEqual(result["metrics"]["precision"], 0)
        self.assertTrue(result["semantic_verdicts_supplied_not_inferred"])

    def test_required_fact_bound_to_unallowed_source_is_not_recovered(self):
        result = score(a=adjudication([claim(sid="u1", quote="continue the exercise")]))
        self.assertEqual(result["metrics"]["recall"], 0)
        self.assertTrue(result["claim_observations"][0]["allowed_fact_binding_mismatch"])

    def test_missing_and_abstained_answers_do_not_gain_precision(self):
        for status in ("abstained", "generation_failed"):
            with self.subTest(status=status):
                result = score(a=adjudication([], status))
                self.assertIsNone(result["metrics"]["precision"])
                self.assertEqual(result["metrics"]["recall"], 0)
                self.assertEqual(result["metrics"]["f1"], 0)
                self.assertFalse(result["complete_answer"])

    def test_no_gold_evidence_is_na_and_abstention_is_separate(self):
        result = score(g=gold(required_facts=[], forbidden_or_stale_fact_ids=[]),
                       a=adjudication([], "abstained"))
        self.assertIsNone(result["metrics"]["retrieval_recall"])
        self.assertIsNone(result["metrics"]["f1"])
        self.assertTrue(result["no_evidence_abstention"])
        self.assertNotIn("safety", result["metrics"])

    def test_retrieval_is_not_answer_use(self):
        result = score(a=adjudication([]), retrieved_source_ids=["u2", "u2"])
        self.assertEqual(result["metrics"]["retrieval_recall"], 1)
        self.assertEqual(result["metrics"]["recall"], 0)

    def test_alternative_evidence_is_not_all_required(self):
        g = gold(required_facts=[{"id": "paused", "allowed_source_ids": ["u2", "u3"]}])
        sources = SOURCES + [source("u3", 2, "The exercise is still paused.")]
        result = score(g=g, sources=sources, retrieved_source_ids=["u2"])
        self.assertEqual(result["metrics"]["retrieval_recall"], 1)
        self.assertEqual(result["metrics"]["allowed_evidence_id_coverage"], .5)

    def test_future_deleted_denied_sources_cannot_be_used(self):
        for update in ({"source_order": 3}, {"retained": False}, {"use_allowed": False}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                sources = deepcopy(SOURCES)
                sources[1].update(update)
                score(sources=sources)

    def test_literal_evidence_and_claim_schema_are_strict(self):
        mutations = [lambda c: c.update(evidence=[]),
                     lambda c: c["evidence"][0].update(quote="never said"),
                     lambda c: c.update(source_ids=["u1"]),
                     lambda c: c.update(verdict="correct")]
        for mutate in mutations:
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                c = claim(); mutate(c); score(a=adjudication([c]))

    def test_provenance_is_explicit_and_not_upgraded_to_gold(self):
        result = score(a=adjudication(provenance="model_proposed"))
        self.assertTrue(result["model_proposed_labels_present"])
        self.assertEqual(result["adjudication_provenance"], "model_proposed")
        with self.assertRaises(ValueError):
            score(a=adjudication(provenance="expert"))

    def test_ids_permissions_and_orders_are_not_coerced(self):
        for updates in ({"source_order": True}, {"retained": 1}, {"role": "system"}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                sources = deepcopy(SOURCES); sources[0].update(updates); score(sources=sources)
        with self.assertRaises(ValueError):
            score(a={"status": [], "label_provenance": "rule_verified", "claims": []})

    def test_strict_quote_only_normalizes_whitespace(self):
        result = evidence.exact_quote_score("I paused\nthis.", "u2", " I  paused this. ", "u2")
        self.assertTrue(result["fully_correct"])
        self.assertFalse(evidence.exact_quote_score("I paused.", "u2", "i paused.", "u2")["fully_correct"])
        self.assertFalse(evidence.exact_quote_score("I paused.", "u2", "I paused.", "U2")["fully_correct"])


class FixedGridTests(unittest.TestCase):
    def test_missing_rows_stay_in_completion_grid_not_precision(self):
        refs = [gold(), gold("case-2", "family-b")]
        raw1 = {**score(), "arm": "raw"}
        managed1 = {**score(), "arm": "managed"}
        raw2 = {**score(g=refs[1]), "arm": "raw"}
        report = evidence.paired_report(refs, [raw1, managed1, raw2])
        self.assertEqual(report["planned_arm_rows"], 4)
        self.assertEqual(report["valid_paired_questions"], 1)
        self.assertEqual(report["statuses_by_arm"]["managed"]["missing_row"], 1)
        self.assertEqual(report["conditional_completed_pairs"]["mean_difference"], 0)
        self.assertEqual(report["completion_adjusted_fixed_grid"]["mean_difference"], -.5)
        p = evidence.paired_report(refs, [raw1, managed1, raw2], metric="precision")
        self.assertIsNone(p["completion_adjusted_fixed_grid"])

    def test_one_family_has_no_fake_interval(self):
        report = evidence.paired_report([gold()], [{**score(), "arm": a} for a in ("raw", "managed")])
        self.assertIsNone(report["conditional_completed_pairs"]["ci95"])
        self.assertEqual(report["conditional_completed_pairs"]["source_families"], 1)

    def test_generation_failure_is_not_a_completed_conditional_pair(self):
        failed = score(a=adjudication([], "generation_failed"))
        report = evidence.paired_report([gold()], [{**score(), "arm": "raw"}, {**failed, "arm": "managed"}])
        self.assertEqual(report["valid_paired_questions"], 0)
        self.assertEqual(report["completion_adjusted_fixed_grid"]["mean_difference"], -1)

    def test_family_bootstrap_is_seeded_and_case_grid_is_fixed(self):
        refs = [gold(), gold("case-2", "family-b")]
        rows = [{**score(g=g), "arm": a} for g in refs for a in ("raw", "managed")]
        a = evidence.paired_report(refs, rows)
        b = evidence.paired_report(refs, rows)
        self.assertEqual(a, b)
        self.assertEqual(a["conditional_completed_pairs"]["ci95"], [0, 0])
        self.assertEqual(a["conditional_completed_pairs"]["source_families"], 2)

    def test_unplanned_duplicate_or_nonfinite_results_are_rejected(self):
        row = {**score(), "arm": "raw"}
        for rows in ([row, row], [{**row, "case_id": "other"}],
                     [{**row, "metrics": {"f1": float("nan")}}],
                     [{**row, "metrics": {"f1": True}}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                evidence.paired_report([gold()], rows)


if __name__ == "__main__":
    unittest.main()
