"""Invented, unreviewed annotation fixtures only. No real dataset or model."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "evaluation/cbt-memory-effect/annotation_contract"
SPEC = importlib.util.spec_from_file_location("natural_cbt_annotation_contract", CONTRACT / "validate.py")
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


class NaturalSourceAnnotationTests(unittest.TestCase):
    def fixture(self, index=0):
        return json.loads((CONTRACT / "fixtures.jsonl").read_text().splitlines()[index])

    def test_all_fixtures_remain_unreviewed_and_progress_survives_pause(self):
        for i in range(3):
            row = self.fixture(i)
            report = M.validate(row["annotation"], row["case"])
            self.assertTrue(report["structurally_valid"])
            self.assertFalse(report["semantic_gold_available"])
            self.assertFalse(report["semantic_review_certified_by_validator"])
            self.assertIsNone(report["user_or_professional_scores"])
        state = {r["axis"]: r["value"] for r in self.fixture()["annotation"]["expected_current_state"]}
        self.assertEqual(state["reported_count"], 1)
        self.assertEqual(state["participation"], "paused")
        self.assertIsNone(state["reported_outcome"])

    def test_existing_pending_row_extends_into_new_identity_without_mutation(self):
        row = self.fixture(); case = row["case"]
        pending = {"case_id": case["case_id"], "case_sha256": case["case_sha256"],
            "annotation_status": "pending_independent_source_annotation", "memory_targets": None,
            "expected_current_state": None, "semantic_gold_available": False, "future_counselor_answer_is_gold": False}
        original = deepcopy(pending)
        annotation = M.draft_from_pending(pending, case)
        self.assertEqual(pending, original)
        self.assertEqual(annotation["source_family_id"], case["source_family_id"])
        self.assertFalse(M.validate(annotation, case)["semantic_gold_available"])

    def test_span_hash_offsets_and_future_evidence_fail(self):
        row = self.fixture(); ann = row["annotation"]; case = row["case"]
        ann["memory_targets"][0]["evidence"][0]["end"] = 9999
        with self.assertRaisesRegex(ValueError, "Unicode offsets"):
            M.validate(ann, case)
        row = self.fixture(); row["annotation"]["memory_targets"][0]["evidence"][0]["source_text_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source identity"):
            M.validate(row["annotation"], row["case"])

    def test_assistant_suggestion_cannot_become_known_user_consent(self):
        row = self.fixture(2)
        target = row["annotation"]["memory_targets"][1]
        target.update(value=True, truth_status="known", evidence=deepcopy(row["annotation"]["memory_targets"][0]["evidence"]))
        with self.assertRaisesRegex(ValueError, "assistant wording"):
            M.validate(row["annotation"], row["case"])

    def test_unknown_is_distinct_from_unannotated_or_guessed(self):
        row = self.fixture(2)
        row["annotation"]["memory_targets"][1]["value"] = "probably agreed"
        with self.assertRaisesRegex(ValueError, "stay null"):
            M.validate(row["annotation"], row["case"])

    def test_latest_state_cannot_reuse_superseded_old_fact(self):
        row = self.fixture(1)
        state = row["annotation"]["expected_current_state"][0]
        state["basis_fact_ids"], state["superseded_fact_ids"] = state["superseded_fact_ids"], state["basis_fact_ids"]
        with self.assertRaisesRegex(ValueError, "precedes its superseded"):
            M.validate(row["annotation"], row["case"])

    def test_current_input_and_locator_labels_do_not_enter_memory_denominator(self):
        row = self.fixture(); target = row["annotation"]["memory_targets"][0]
        target["source_scope"] = "current_input"
        with self.assertRaisesRegex(ValueError, "historical facts"):
            M.validate(row["annotation"], row["case"])
        row = self.fixture(); row["case"]["locator_cue"] = "word prefix"
        row["case"]["case_sha256"] = M.case_hash(row["case"]); row["annotation"]["case_sha256"] = row["case"]["case_sha256"]
        with self.assertRaisesRegex(ValueError, "locator"):
            M.validate(row["annotation"], row["case"])

    def test_fixture_identity_cannot_be_relabeled_as_real_gold(self):
        row = self.fixture()
        row["annotation"]["fixture_only"] = False
        with self.assertRaisesRegex(ValueError, "fixture identity"):
            M.validate(row["annotation"], row["case"])

    def test_model_proposal_or_missing_human_registry_never_becomes_independent_gold(self):
        row = self.fixture(); ann = row["annotation"]
        # A synthetic test of the production identity branch, never a saved
        # annotation or a claim that any human reviewed these fixture facts.
        row["case"]["fixture_only"] = False
        row["case"]["source_kind"] = "released_synthetic_cbt_dialogue"
        row["case"]["case_sha256"] = M.case_hash(row["case"])
        ann["case_sha256"] = row["case"]["case_sha256"]
        ann.update(fixture_only=False, annotation_status="independently_reviewed", semantic_gold_available=True)
        ann["review"].update(label_origin="model_proposal", reviewer_ids=["reviewer-a", "reviewer-b"],
            agreement="agreed", human_record_refs=["record-a", "record-b"])
        with self.assertRaisesRegex(ValueError, "verified human"):
            M.validate(ann, row["case"])
        ann["review"]["label_origin"] = "human_draft"
        with self.assertRaisesRegex(ValueError, "external human reviewer registry"):
            M.validate(ann, row["case"])


if __name__ == "__main__":
    unittest.main()
