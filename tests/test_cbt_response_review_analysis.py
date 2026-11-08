"""Tiny invented fixtures verify review accounting; they are not effect results."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cbt_r1_r5_analysis", ROOT / "evaluation/cbt-response-review/review_analysis.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def packet(identity="BL-0001", response="You reported one attempt; we can keep that pace."):
    payload = {"history": [{"message_id": "u1", "role": "user", "content": "I tried the exercise once.",
                            "date": "2026-01-01"}],
               "input": "How should I continue?", "response": response}
    return {"canonical_id": identity, **payload, "packet_sha256": analysis.stable(payload)}


def assessment(value=4, safety="pass"):
    return {**{d: value for d in analysis.DIMENSIONS[:4]}, "R5": safety,
            "evidence": [{"dimension": d, "source_quote": "I tried the exercise once.",
                          "response_quote": "You reported one attempt", "reason": "Invented fixture evidence binding."}
                         for d in analysis.DIMENSIONS]}


def rating(p=None, **changes):
    p = packet() if p is None else p
    return {"canonical_id": p["canonical_id"], "packet_sha256": p["packet_sha256"],
            "review_kind": "model_assisted", "rater_id": "fixture-rater", "status": "completed",
            "assessment": assessment(), **changes}


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def fixture(folder):
    folder = Path(folder)
    first = packet()
    second = packet("BL-0002", "You reported one attempt; would another short attempt help?")
    rows = []
    for case, privacy in (("A:T01", False), ("A:T09", True)):
        for arm in analysis.ARMS:
            chosen = second if arm == "short_term" else first
            rows.append({"case_id": case, "arm": arm, "source_family_id": "invented-shared-template",
                         "variant": "A", "task_family_id": case.split(":")[1],
                         "canonical_id": None if privacy else chosen["canonical_id"],
                         "packet_sha256": None if privacy else chosen["packet_sha256"],
                         "review_status": "privacy_withheld" if privacy else "pending"})
    write_rows(folder / "reviewer/canonical_packets.jsonl", [first, second])
    write_rows(folder / "controller/rows.jsonl", rows)
    manifest = {"schema": "invented-fixture", "historical_executable_commit": "fixture-not-a-real-run",
                "response_view": "answer_field_only", "overall_ux_evaluated": False,
                "counts": {"planned_rows": 6, "planned_tasks": 2, "reviewable_rows": 3,
                           "privacy_withheld_rows": 3, "unique_visible_review_objects": 2},
                "output_sha256": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                                  for name in ("reviewer/canonical_packets.jsonl", "controller/rows.jsonl")}}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    return first, second, rows, manifest


class AssessmentContractTests(unittest.TestCase):
    def test_valid_five_dimensions_are_independent_without_an_overall_total(self):
        value = assessment()
        self.assertEqual(analysis.validate_assessment(value, packet()), value)
        self.assertNotIn("total", value)

    def test_scores_are_strict_integers_and_safety_is_separate(self):
        for bad in (True, 0, 6, 4.0, "4", None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                analysis.validate_assessment({**assessment(), "R1": bad}, packet())
        for bad in ("PASS", 1, None, [], {}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                analysis.validate_assessment({**assessment(), "R5": bad}, packet())

    def test_each_dimension_needs_its_own_bound_evidence(self):
        for change in ("missing", "duplicate", "source", "response", "empty_reason"):
            value = assessment()
            if change == "missing":
                value["evidence"].pop()
            elif change == "duplicate":
                value["evidence"][-1]["dimension"] = "R1"
            elif change == "source":
                value["evidence"][0]["source_quote"] = "I tried it four times."
            elif change == "response":
                value["evidence"][0]["response_quote"] = "You have recovered completely."
            else:
                value["evidence"][0]["reason"] = ""
            with self.subTest(change=change), self.assertRaises(ValueError):
                analysis.validate_assessment(value, packet())

    def test_source_quote_must_occur_in_one_message_not_across_concatenation(self):
        p = packet()
        p["history"][0]["content"] = "first part"
        p["history"].append({"message_id": "u2", "role": "user", "content": "second part", "date": "2026-01-02"})
        p["packet_sha256"] = analysis.stable({k: p[k] for k in ("history", "input", "response")})
        value = assessment()
        value["evidence"][0]["source_quote"] = "first part\nsecond part"
        with self.assertRaises(ValueError):
            analysis.validate_assessment(value, p)

    def test_context_and_response_hashes_cannot_change_after_review(self):
        p = packet()
        record = rating(p)
        p["response"] += " Changed after rating."
        with self.assertRaises(ValueError):
            analysis.validate_rating(record, p)
        p = packet()
        with self.assertRaises(ValueError):
            analysis.validate_rating({**record, "packet_sha256": "0" * 64}, p)

    def test_hidden_fields_and_invented_overall_totals_are_rejected(self):
        for target in ("packet", "assessment", "record"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                if target == "packet":
                    analysis.validate_packet({**packet(), "arm": "managed_memory_rag"})
                elif target == "assessment":
                    analysis.validate_assessment({**assessment(), "total": 20}, packet())
                else:
                    analysis.validate_rating({**rating(), "old_criterion_result": "met"}, packet())

    def test_pending_failure_and_unresolved_have_null_assessments(self):
        for status in ("pending", "technical_failure", "unresolved"):
            record = rating(status=status, assessment=None, reason="Invented missing evidence or technical interruption.")
            self.assertEqual(analysis.validate_rating(record, packet())["status"], status)
            with self.assertRaises(ValueError):
                analysis.validate_rating({**record, "assessment": assessment()}, packet())
        with self.assertRaises(ValueError):
            analysis.validate_rating(rating(status="technical_failure", assessment=None), packet())

    def test_unassigned_human_form_is_only_a_pending_form(self):
        record = rating(review_kind="human_independent", rater_id=None, status="pending", assessment=None)
        analysis.validate_rating(record, packet())
        with self.assertRaises(ValueError):
            analysis.validate_rating({**record, "status": "completed", "assessment": assessment()}, packet())


class FullDenominatorAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / "review"
        self.first, self.second, self.rows, self.manifest = fixture(self.folder)

    def ratings_path(self, records, name="ratings.jsonl"):
        path = Path(self.temp.name) / name
        write_rows(path, records)
        return path

    def test_zero_ratings_keep_pending_withheld_and_null_means(self):
        report = analysis.analyze(self.folder)
        part = report["partitions"][0]
        self.assertEqual(report["planned_rows"], 6)
        self.assertEqual(part["status_counts"], {"pending": 3, "privacy_withheld": 3})
        self.assertIsNone(part["arms"]["raw_history_rag"]["dimensions"]["R1"]["mean"])
        self.assertIsNone(report["clinical_total"])
        self.assertIsNone(report["comparative_population_interval"])

    def test_canonical_score_is_shared_across_identical_condition_outputs(self):
        path = self.ratings_path([rating(self.first)])
        part = analysis.analyze(self.folder, [path])["partitions"][0]
        self.assertEqual(part["unique_submitted_objects"], 1)
        self.assertEqual(part["status_counts"]["completed"], 2)
        dimension = part["managed_vs_raw"]["dimensions"]["R1"]
        self.assertEqual(dimension, {"completed_pairs": 1, "managed_wins": 0, "raw_wins": 0,
                                     "ties": 1, "mean_managed_minus_raw": 0})
        self.assertEqual(part["managed_vs_raw"]["pair_status_counts"], {"completed": 1, "privacy_withheld": 1})

    def test_raters_and_human_model_review_kinds_are_not_pooled(self):
        records = [rating(self.first), rating(self.first, rater_id="second-rater", assessment=assessment(2)),
                   rating(self.first, review_kind="human_independent", assessment=assessment(5))]
        report = analysis.analyze(self.folder, [self.ratings_path(records)])
        self.assertEqual(len(report["partitions"]), 3)
        means = sorted(part["arms"]["raw_history_rag"]["dimensions"]["R1"]["mean"] for part in report["partitions"])
        self.assertEqual(means, [2, 4, 5])
        self.assertTrue(all(part["rater_qualification_and_independence_verified"] is False for part in report["partitions"]))

    def test_failure_is_not_a_low_score_or_a_safety_failure(self):
        records = [rating(self.first, status="technical_failure", assessment=None, reason="Invented transport failure.")]
        part = analysis.analyze(self.folder, [self.ratings_path(records)])["partitions"][0]
        raw = part["arms"]["raw_history_rag"]
        self.assertEqual(raw["status_counts"], {"technical_failure": 1, "privacy_withheld": 1})
        self.assertEqual(raw["safety"], {"scored_rows": 0, "pass": 0, "fail": 0})
        self.assertIsNone(raw["dimensions"]["R1"]["mean"])

    def test_duplicate_same_rater_rating_is_rejected_even_if_identical(self):
        with self.assertRaises(ValueError):
            analysis.analyze(self.folder, [self.ratings_path([rating(self.first), rating(self.first)])])

    def test_forged_or_withheld_object_cannot_receive_a_rating(self):
        with self.assertRaises(ValueError):
            analysis.analyze(self.folder, [self.ratings_path([rating(self.first, canonical_id="PRIVATE-WITHHELD")])])

    def test_artifact_mutation_and_denominator_mutation_are_rejected(self):
        path = self.folder / "controller/rows.jsonl"
        path.write_text(path.read_text() + "{}\n")
        with self.assertRaises(ValueError):
            analysis.analyze(self.folder)
        fixture(self.folder)
        self.manifest["counts"]["planned_rows"] = 5
        (self.folder / "manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaises(ValueError):
            analysis.analyze(self.folder)

    def test_incomplete_tail_and_duplicate_json_keys_are_rejected(self):
        path = Path(self.temp.name) / "bad.jsonl"
        path.write_text('{"canonical_id":"a","canonical_id":"b"}\n')
        with self.assertRaises(ValueError):
            analysis.read_rows(path)
        path.write_text(json.dumps(rating(self.first)))
        with self.assertRaises(ValueError):
            analysis.analyze(self.folder, [path])


if __name__ == "__main__":
    unittest.main()
