"""Failure-mode checks on tiny fictional scoring fixtures, not effect evidence."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation/dialogue-benefit"))
import score_followup_pipeline as scoring


SOURCE_TEXT = "A fictional user reported practicing once, not three times."


def source(sid="A-S01", text=SOURCE_TEXT, **extra):
    return {"source_id": sid, "source_text": text,
            "source_text_sha256": scoring.text_hash(text), "quote": text,
            "start": 0, "end": len(text), "role": "user", "date": "2026-01-01", **extra}


def reference():
    return {"case_id": "A:T01", "completion_criterion": "Fictional fixture: distinguish reported practice from the plan.",
            "criterion_sha256": "f" * 64, "source_reference": [source()],
            "arm": "MUST_NOT_ENTER_JUDGE", "variant_parameters": {"secret": "MUST_NOT_ENTER_JUDGE"}}


def response(check=None):
    return json.dumps({"answer": "You reported practicing once.", "used_source_ids": ["A-S01"],
                       "clarification": check})


def judgment(result="met"):
    return json.dumps({"criterion_result": result, "source_quotes": [{"source_id": "A-S01", "quote": "practicing once"}],
                       "response_quotes": ["practicing once"], "reason": "Fictional evidence supports the wording."})


def case():
    return {"case_id": "A:T01", "variant": "A", "query": "What was actually done?",
            "history": [{"source_id": "A-S01", "text": SOURCE_TEXT, "text_sha256": scoring.text_hash(SOURCE_TEXT)}],
            "exclusions": [{"source_id": "A-S99", "reason": "deleted"}]}


def row(arm, *, raw=None, review=None, extracted=True):
    raw = response() if raw is None else raw
    return {"case_id": "A:T01", "variant": "A", "task_id": "T01", "arm": arm,
            "raw_response": raw, "completion_status": "success", "extraction_available": extracted,
            "observations": scoring.score_observations(case(), {"trace": {"included_source_ids": ["A-S01"]}}, raw,
                                                        extraction_available=extracted),
            "judgment": review or scoring.validate_judge_output(judgment(), reference(), raw)}


class FollowupScoringTests(unittest.TestCase):
    def test_envelope_rejects_duplicate_json_and_unbounded_or_hidden_fields(self):
        for raw in ('{"answer":"a","answer":"b","used_source_ids":[],"clarification":null}',
                    '{"answer":"a","used_source_ids":[],"clarification":null,"arm":"raw"}',
                    '{"answer":"a","used_source_ids":["A-S01","A-S01"],"clarification":null}',
                    '{"answer":"a","used_source_ids":[],"clarification":NaN}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                scoring.parse_answer_envelope(raw)

    def test_judge_only_sees_explicit_scoring_projection_and_no_condition(self):
        messages = scoring.build_judge_messages(reference(), response(), query=case()["query"], masked_id="review-opaque")
        data = json.loads(messages[1]["content"])
        self.assertNotIn("MUST_NOT_ENTER_JUDGE", messages[1]["content"])
        self.assertNotIn("arm", data)
        self.assertTrue(data["scoring_only_not_generation_input"])
        self.assertEqual(data["response_sha256"], scoring.text_hash(response()))
        self.assertIn("not clinical", messages[0]["content"])

    def test_scoring_reference_cannot_be_used_with_a_changed_query(self):
        ref = {**reference(), "query_sha256": scoring.text_hash(case()["query"]), "materials_sha256": "e" * 64}
        scoring.build_judge_messages(ref, response(), query=case()["query"], masked_id="review-only")
        with self.assertRaises(ValueError):
            scoring.build_judge_messages(ref, response(), query="A different question?", masked_id="review-only")

    def test_excluded_reference_values_are_never_sent_to_a_model_scorer(self):
        ref = reference()
        ref["source_reference"] = [source("A-S99", "Fictional deleted value: red kite.", permitted_absence_reason="deleted")]
        raw = json.dumps({"answer": "That information is excluded.", "used_source_ids": [], "clarification": None})
        review = json.dumps({"criterion_result": "met", "source_quotes": [{"source_id": "A-S99", "quote": "deleted value"}],
                             "response_quotes": ["information is excluded"], "reason": "Deletion behavior in this fictional fixture."})
        self.assertFalse(scoring.reference_allows_model_review(ref))
        with self.assertRaises(ValueError):
            scoring.build_judge_messages(ref, raw, query="Is it deleted?", masked_id="review-only")
        with self.assertRaises(ValueError):
            scoring.validate_judge_output(review, ref, raw)
        self.assertTrue(scoring.reference_allows_model_review(reference()))
        for reason in ("dont_save", "expired", "access_restricted"):
            with self.subTest(reason=reason):
                ref["source_reference"][0]["permitted_absence_reason"] = reason
                self.assertFalse(scoring.reference_allows_model_review(ref))
                with self.assertRaises(ValueError):
                    scoring.build_judge_messages(ref, raw, query="A follow-up?", masked_id="review-only")

    def test_forged_source_response_quotes_and_empty_evidence_rejected(self):
        base = json.loads(judgment())
        mutations = []
        bad = deepcopy(base); bad["source_quotes"][0]["source_id"] = "A-S404"; mutations.append(bad)
        bad = deepcopy(base); bad["source_quotes"][0]["quote"] = "practiced three times"; mutations.append(bad)
        bad = deepcopy(base); bad["response_quotes"] = ["practiced three times"]; mutations.append(bad)
        bad = deepcopy(base); bad["source_quotes"] = []; mutations.append(bad)
        for item in mutations:
            with self.subTest(item=item), self.assertRaises(ValueError):
                scoring.validate_judge_output(json.dumps(item), reference(), response())

    def test_quote_validity_does_not_claim_entailment_or_causal_attribution(self):
        raw = json.dumps({"answer": "The user completed everything?", "used_source_ids": ["A-S01"], "clarification": None})
        obs = scoring.score_observations(case(), {"trace": {"included_source_ids": ["A-S01"]}}, raw)
        self.assertTrue(obs["answer_envelope_valid"])
        self.assertEqual(obs["cited_unknown_source_ids"], [])
        self.assertIsNone(obs["semantic_correctness"])
        self.assertIn("unresolved", obs["causal_failure_stage"])
        self.assertEqual(obs["question_mark_cue_count"], 1)
        self.assertFalse(obs["model_check_emitted"])
        self.assertFalse(obs["participant_interruptions_measured"])

    def test_missing_context_inclusion_is_unknown_and_bad_ids_remain_visible(self):
        raw = json.dumps({"answer": "Fictional output.", "used_source_ids": ["A-S99", "A-S404"], "clarification": None})
        obs = scoring.score_observations(case(), {"trace": {}}, raw)
        self.assertIsNone(obs["context_excluded_source_ids"])
        self.assertEqual(obs["cited_excluded_source_ids"], ["A-S99"])
        self.assertEqual(obs["cited_unknown_source_ids"], ["A-S404", "A-S99"])
        self.assertIsNone(obs["cited_ids_not_in_supplied_context"])

    def test_partially_truncated_source_is_not_falsely_reported_as_unavailable(self):
        trace = {"included_source_ids": [],
                 "short_source_inclusion_unknown_due_to_truncation": ["A-S01"]}
        obs = scoring.score_observations(case(), {"trace": trace}, response())
        self.assertFalse(obs["context_source_ids_available"])
        self.assertIsNone(obs["cited_ids_not_in_supplied_context"])
        self.assertEqual(obs["known_fully_included_source_ids"], [])
        self.assertEqual(obs["short_source_inclusion_unknown_due_to_truncation"], ["A-S01"])
        self.assertEqual(obs["cited_unknown_source_ids"], [])

    def test_partial_grid_never_produces_comparative_scores(self):
        with self.assertRaises(ValueError):
            scoring.summarize_rows([row(scoring.ARMS[0])], [case()], references=[reference()])
        summary = scoring.summarize_rows([row(scoring.ARMS[0])], [case()], references=[reference()], allow_partial=True)
        self.assertEqual(summary["conditions"], {})
        self.assertFalse(summary["final_effectiveness_result"])

    def test_complete_grid_with_unattempted_judge_is_still_partial(self):
        rows = [row(arm) for arm in scoring.ARMS]
        rows[1]["judgment"] = scoring.unresolved_judgment("judge_not_yet_attempted")
        with self.assertRaises(ValueError):
            scoring.summarize_rows(rows, [case()], references=[reference()])
        summary = scoring.summarize_rows(rows, [case()], references=[reference()], allow_partial=True)
        self.assertEqual(summary["pending_generation_or_judge_rows"], 1)
        self.assertEqual(summary["conditions"], {})

    def test_failed_extraction_and_missing_answer_review_remain_denominators(self):
        rows = [row(arm) for arm in scoring.ARMS]
        failed = rows[2]
        failed["raw_response"] = None
        failed["extraction_available"] = False
        failed["observations"] = scoring.score_observations(case(), {"trace": {}}, None, extraction_available=False)
        failed["judgment"] = scoring.unresolved_judgment("provider_failure")
        summary = scoring.summarize_rows(rows, [case()], references=[reference()])
        m = summary["conditions"]["managed_memory_rag"]
        self.assertEqual(m["planned_tasks"], 1)
        self.assertEqual(m["model_assisted_criterion_counts"]["unresolved"], 1)
        self.assertEqual(m["provider_responses_available"], 0)
        self.assertEqual(m["managed_extraction_unavailable_tasks"], 1)
        self.assertEqual(m["model_assisted_met_fraction_all_planned_tasks"], 0)
        self.assertEqual(summary["actual_human_participants"], 0)
        self.assertEqual(summary["comparative_intervals"], {})

    def test_judgment_cannot_move_to_another_response_or_change_grade_silently(self):
        rows = [row(arm) for arm in scoring.ARMS]
        rows[0]["raw_response"] = response().replace("once", "twice")
        with self.assertRaises(ValueError):
            scoring.summarize_rows(rows, [case()], references=[reference()])
        rows = [row(arm) for arm in scoring.ARMS]
        rows[0]["judgment"]["criterion_result"] = "violated"
        with self.assertRaises(ValueError):
            scoring.summarize_rows(rows, [case()], references=[reference()])

    def test_judgment_cannot_move_between_cases_with_identical_responses(self):
        first, second = case(), {**case(), "case_id": "B:T01", "variant": "B"}
        first_ref = {**reference(), "completion_criterion": "Fictional source must be excluded.",
                     "source_reference": [source("A-S01", "Remove fictional private source.")]}
        second_ref = {**reference(), "case_id": "B:T01", "completion_criterion": "Fictional permitted goal must remain usable.",
                      "source_reference": [source("B-S01", "Retain fictional allowed practice.")]}
        raw = json.dumps({"answer": "That information is excluded.", "used_source_ids": [], "clarification": None})
        reviews = []
        for ref in (first_ref, second_ref):
            evidence = ref["source_reference"][0]
            model = json.dumps({"criterion_result": "met", "source_quotes": [{"source_id": evidence["source_id"], "quote": evidence["quote"]}],
                                "response_quotes": ["information is excluded"], "reason": "Fictional fixture assessment only."})
            reviews.append(scoring.validate_judge_output(model, ref, raw))
        values = []
        for index, item in enumerate((first, second)):
            for arm in scoring.ARMS:
                values.append({**row(arm), "case_id": item["case_id"], "raw_response": raw, "judgment": reviews[index]})
        scoring.summarize_rows(values, [first, second], references=[first_ref, second_ref])
        values[-1]["judgment"] = reviews[0]
        with self.assertRaises(ValueError):
            scoring.summarize_rows(values, [first, second], references=[first_ref, second_ref])

    def test_resolved_summary_requires_actual_reference_binding(self):
        with self.assertRaises(ValueError):
            scoring.summarize_rows([row(arm) for arm in scoring.ARMS], [case()])

    def test_blank_packets_hide_model_grades_and_keep_allocation_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "review"
            metadata = scoring.export_blinded_packets([case()], [reference()], [row(arm) for arm in scoring.ARMS], out)
            self.assertEqual(metadata["packets"], 3)
            self.assertEqual(metadata["independent_ratings_received"], 0)
            packets = [json.loads(line) for line in (out / "blank_packets.jsonl").read_text().splitlines()]
            self.assertTrue(all("arm" not in p and "judgment" not in p for p in packets))
            self.assertTrue((out / "restricted_allocation/condition_key.json").is_file())
            with (out / "blank_ratings.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.reader(stream))
            self.assertTrue(all(len(record) == len(rows[0]) for record in rows))
            self.assertTrue(all(not any(record[2:]) for record in rows[1:]))
            with self.assertRaises(ValueError):
                scoring.export_blinded_packets([case()], [reference()], [row(arm) for arm in scoring.ARMS], out)

    def test_privacy_excluded_references_are_withheld_from_default_blank_packets(self):
        ref = reference()
        ref["source_reference"] = [source("A-S99", "PRIVATE_FIXTURE_SENTINEL", permitted_absence_reason="deleted")]
        values = [row(arm) for arm in scoring.ARMS]
        for item in values:
            item["judgment"] = scoring.unresolved_judgment("privacy_reference_withheld_from_model_review")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "review"
            metadata = scoring.export_blinded_packets([case()], [ref], values, out)
            self.assertEqual(metadata["packets"], 0)
            self.assertEqual(metadata["planned_arm_rows"], 3)
            self.assertEqual(metadata["privacy_excluded_reference_packets_withheld"], 3)
            self.assertEqual(metadata["unavailable_answers_not_exported"], 0)
            self.assertTrue(all("PRIVATE_FIXTURE_SENTINEL" not in path.read_text(encoding="utf-8-sig")
                                for path in out.rglob("*") if path.is_file()))


if __name__ == "__main__":
    unittest.main()
