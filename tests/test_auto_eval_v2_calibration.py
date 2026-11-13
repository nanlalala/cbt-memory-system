"""Invented labels exercise arithmetic, missingness and case-family isolation."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("auto_calibration_v2",
    Path(__file__).resolve().parents[1] / "evaluation/cbt-memory-effect/auto_eval_v2/calibration.py")
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)


def labels(case="a", score=3, positive=False):
    return {"case_id": case, "ctrs": {key: score for key in c.CTRS},
            "safety": {key: positive for key in c.SAFETY}}


def prediction(gold, **changes):
    result = {**deepcopy(gold), "status": "completed"}
    result.update(changes)
    return result


class CalibrationTests(unittest.TestCase):
    def test_ctrs_keeps_exact_nine_and_excludes_two(self):
        self.assertEqual(len(c.CTRS), 9)
        self.assertNotIn("GUIDED DISCOVERY", c.CTRS)
        self.assertNotIn("TECHNIQUE APPLICATION", c.CTRS)

    def test_perfect_inverse_and_tied_rank_correlation(self):
        self.assertEqual(c.spearman([0, 1, 2], [0, 1, 2]), 1)
        self.assertEqual(c.spearman([0, 1, 2], [2, 1, 0]), -1)
        # Ranks [1.5,1.5,3] vs [1,2.5,2.5], Pearson of ranks = .5.
        self.assertAlmostEqual(c.spearman([0, 0, 2], [0, 2, 2]), .5)

    def test_undefined_is_null_and_not_zero(self):
        self.assertIsNone(c.spearman([], []))
        self.assertIsNone(c.spearman([2], [3]))
        self.assertIsNone(c.spearman([2, 2], [1, 4]))

    def test_invalid_numeric_scores_are_rejected(self):
        for bad in (True, False, float("nan"), float("inf"), -1, 7, "3"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                row = labels(score=bad)
                c.calibrate([row], [])

    def test_safety_rejects_numeric_boolean_substitutes(self):
        for bad in (0, 1, "false", 0.0):
            row = labels()
            row["safety"][c.SAFETY[0]] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                c.calibrate([row], [])

    def test_nullable_scores_and_absolute_bias(self):
        gold = [labels("a", 1), labels("b", 3), labels("c", None)]
        pred = [prediction(labels("a", 3)), prediction(labels("b", 4)), prediction(labels("c", 6))]
        out = c.calibrate(gold, pred)["ctrs"]["AGENDA"]
        self.assertEqual(out["n_planned"], 3)
        self.assertEqual(out["n_compared"], 2)
        self.assertEqual(out["n_gold_missing"], 1)
        self.assertEqual(out["mae"], 1.5)
        self.assertEqual(out["mean_bias_prediction_minus_expert"], 1.5)

    def test_missing_invalid_unresolved_keep_fixed_denominator(self):
        gold = [labels(case) for case in "abcd"]
        pred = [prediction(gold[0]), {"case_id": "b", "status": "invalid"},
                {"case_id": "c", "status": "unresolved"}]
        out = c.calibrate(gold, pred)
        self.assertEqual(out["prediction_status_counts"],
                         {"completed": 1, "unresolved": 1, "invalid": 1, "missing": 1})
        self.assertEqual(out["ctrs"]["AGENDA"]["n_planned"], 4)
        self.assertEqual(out["ctrs"]["AGENDA"]["n_prediction_missing_for_available_gold"], 3)

    def test_null_predictions_are_not_zero(self):
        gold = [labels()]
        out = c.calibrate(gold, [prediction(labels(score=None))])["ctrs"]["AGENDA"]
        self.assertEqual(out["n_prediction_missing_for_available_gold"], 1)
        self.assertIsNone(out["mae"])

    def test_confusion_matrix_and_missing_positive_bound(self):
        gold = [labels("tp", positive=True), labels("fn", positive=True),
                labels("fp"), labels("tn"), labels("missing", positive=True)]
        pred = [prediction(labels("tp", positive=True)), prediction(labels("fn")),
                prediction(labels("fp", positive=True)), prediction(labels("tn"))]
        out = c.calibrate(gold, pred)["safety"][c.SAFETY[0]]
        self.assertEqual([out[k] for k in ("tp", "fp", "fn", "tn")], [1, 1, 1, 1])
        self.assertEqual(out["gold_positive_support"], 3)
        self.assertEqual(out["n_missing_prediction_positive"], 1)
        self.assertEqual(out["precision"], .5)
        self.assertEqual(out["recall"], .5)
        self.assertEqual(out["f1"], .5)
        self.assertEqual(out["missing_positive_recall_bounds"], [1 / 3, 2 / 3])

    def test_zero_positive_does_not_produce_fake_safety_recall(self):
        gold = [labels("a"), labels("b")]
        out = c.calibrate(gold, [prediction(row) for row in gold])["safety"][c.SAFETY[0]]
        self.assertEqual(out["gold_positive_support"], 0)
        self.assertEqual(out["tn"], 2)
        self.assertIsNone(out["recall"])
        self.assertIsNone(out["f1"])

    def test_no_positive_with_false_positive_still_exposes_error(self):
        out = c.calibrate([labels()], [prediction(labels(positive=True))])["safety"][c.SAFETY[0]]
        self.assertEqual(out["fp"], 1)
        self.assertEqual(out["precision"], 0)
        self.assertIsNone(out["recall"])

    def test_unknown_duplicate_or_unplanned_case_rejected(self):
        with self.assertRaises(ValueError):
            c.calibrate([labels()], [prediction(labels("other"))])
        with self.assertRaises(ValueError):
            c.calibrate([labels(), labels()], [])
        with self.assertRaises(ValueError):
            c.calibrate([labels()], [prediction(labels()), prediction(labels())])

    def test_no_claim_of_new_expert_reviews_or_interrater_recomputation(self):
        out = c.calibrate([labels()], [])
        self.assertFalse(out["inter_rater_reliability_recomputed"])
        self.assertFalse(out["validates_project_specific_R1_R3_or_longitudinal_scores"])


class GroupAndProjectionTests(unittest.TestCase):
    def metadata(self):
        return [{"case_id": case, "patient_profile": profile}
                for case, profile in zip("abcde", ("p7-1", "p7-2", "p8", "p9", "p10"))]

    def mapping(self):
        return {"p7-1": "shared", "p7-2": "shared", "p8": "f8", "p9": "f9", "p10": "f10"}

    def test_sibling_variants_never_cross_groups_and_exposure_propagates(self):
        out = c.split_groups(self.metadata(), self.mapping(), ["a"], seed=12)
        rows = {r["case_id"]: r for r in out["assignments"]}
        self.assertEqual(rows["a"]["split"], "development")
        self.assertEqual(rows["b"]["split"], "development")
        self.assertNotIn("shared", out["heldout_families"])
        self.assertTrue(out["heldout_families"])

    def test_deterministic_under_metadata_order(self):
        a = c.split_groups(self.metadata(), self.mapping(), ["a"], seed=42)
        b = c.split_groups(list(reversed(self.metadata())), self.mapping(), ["a"], seed=42)
        self.assertEqual(a, b)

    def test_missing_lineage_fails_closed_instead_of_profile_heuristic(self):
        mapping = self.mapping(); del mapping["p7-2"]
        with self.assertRaisesRegex(ValueError, "lineage_missing"):
            c.split_groups(self.metadata(), mapping)

    def test_unknown_exposure_or_too_few_unexposed_families_rejected(self):
        with self.assertRaises(ValueError):
            c.split_groups(self.metadata(), self.mapping(), ["not-a-case"])
        with self.assertRaises(ValueError):
            c.split_groups(self.metadata(), self.mapping(), ["a", "c", "d"])

    def test_metadata_projection_rejects_hidden_labels(self):
        rows = self.metadata(); rows[0]["expert_rating"] = "not allowed"
        with self.assertRaisesRegex(ValueError, "metadata_projection"):
            c.split_groups(rows, self.mapping())

    def record(self):
        return {"id": "internal name", "patient_profile": "hidden profile", "expert_rating": "HIDDEN GOLD",
                "therapist_model": "hidden model", "conversation": {"messages": [
                    {"role": "user" if i % 2 == 0 else "assistant", "content": f"visible text {i}",
                     "name": "excluded", "expert_score": 6} for i in range(10)]}}

    def test_projection_hides_gold_and_profile_and_model_metadata(self):
        out = c.project_dialogue(self.record())
        text = json.dumps(out)
        self.assertNotIn("HIDDEN GOLD", text)
        self.assertNotIn("hidden profile", text)
        self.assertNotIn("hidden model", text)
        self.assertNotIn("expert_score", text)
        self.assertEqual(len(out["messages"]), 10)
        self.assertEqual(set(out), {"messages", "input_sha256"})

    def test_gold_does_not_change_projected_request_fingerprint(self):
        row = self.record()
        before = c.project_dialogue(row)
        row["expert_rating"] = {"completely": "different"}
        self.assertEqual(before, c.project_dialogue(row))

    def test_projection_requires_five_ordered_exchanges(self):
        row = self.record(); row["conversation"]["messages"][1]["role"] = "user"
        with self.assertRaises(ValueError): c.project_dialogue(row)
        row = self.record(); row["conversation"]["messages"] *= 2
        with self.assertRaises(ValueError): c.project_dialogue(row)

    def test_known_source_manifest_identity_is_bound_without_file_reads(self):
        row = {"repository": c.REPOSITORY, "revision": c.REVISION, "dataset_version": "2.0.0",
               "files": [{"filename": "therapyjudgebench.jsonl", "sha256": c.SOURCE_SHA256}]}
        self.assertEqual(c.verify_source_manifest(row, c.SOURCE_SHA256)["source_sha256"], c.SOURCE_SHA256)
        row["revision"] = "new-version"
        with self.assertRaises(ValueError): c.verify_source_manifest(row, c.SOURCE_SHA256)


if __name__ == "__main__":
    unittest.main()
