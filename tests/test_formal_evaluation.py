"""Tiny fixtures only: test cohort controls without a real large benchmark/model."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation/end-to-end-dialogue"))

from prepare_formal_evaluation import (  # noqa: E402
    DEFAULT_PLAN, _verify_pilot_exposure_source, file_identity, history_groups,
    load_plan, prepare_dataset, prepare_examples, stable_digest, validate_cohort, validate_full_s_spec, validate_plan,
)
from benchmark_adapters import RetrievalExample  # noqa: E402


def example(question_id="q1", *, session_id="s1", text="I prefer walking after lunch.", category="single-session-user"):
    return RetrievalExample(
        example_id=question_id, category=category, query="What activity do I prefer?",
        answer="GOLD_ANSWER_MUST_NOT_ENTER_GENERATION",
        evidence_ids=("GOLD_EVIDENCE_MUST_NOT_ENTER_GENERATION",),
        query_date="2026-09-10T10:00:00Z",
        sessions=({"session_id": session_id, "date": "2026-09-01T10:00:00Z", "turns": [
            {"role": "user", "content": text, "answer": "ANNOTATION_MUST_BE_REMOVED",
             "evidence": "GOLD_LABEL_REMOVED", "category": "HIDDEN_LABEL_REMOVED"}]},),
    )


class FormalEvaluationPreparationTests(unittest.TestCase):
    def setUp(self):
        self.plan = load_plan(DEFAULT_PLAN)

    def prepare(self, rows, directory, *, dataset="longmemeval_s_cleaned", **kwargs):
        return prepare_examples(rows, dataset, self.plan, {"sha256": "0" * 64, "bytes": 10},
                                Path(directory) / "cohort", fixture_only=True, **kwargs)

    def test_full_s_has_immutable_metadata_and_oracle_cannot_qualify(self):
        spec = self.plan["datasets"]["longmemeval_s_cleaned"]
        self.assertEqual(spec["bytes"], 277383467)
        self.assertEqual(spec["sha256"], "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442")
        self.assertIn(spec["revision"], spec["source"])
        validate_full_s_spec(spec)
        oracle = {**spec, "bytes": 15388478, "source": spec["source"].replace("longmemeval_s_cleaned.json", "longmemeval_oracle.json")}
        with self.assertRaisesRegex(ValueError, "never the oracle"):
            validate_full_s_spec(oracle)

    def test_full_dataset_preparation_refuses_local_computer_before_loading_any_data(self):
        with patch("prepare_formal_evaluation.is_colab_runtime", return_value=False), \
             patch("prepare_formal_evaluation.load_longmemeval") as loader:
            with self.assertRaisesRegex(RuntimeError, "Colab-only"):
                prepare_dataset(Path("no-file-needed"), "longmemeval_s_cleaned", DEFAULT_PLAN, Path("unused"))
            loader.assert_not_called()

    def test_history_groups_merge_shared_sessions_transitively_not_qa_count(self):
        first = example("q1", session_id="shared")
        second = example("q2", session_id="shared", text="A distinct turn in an identically identified session.")
        third = example("q3", session_id="different-id", text="A distinct turn in an identically identified session.")
        fourth = example("q4", session_id="isolated", text="I prefer cycling outdoors.")
        groups = history_groups([first, second, third, fourth], "longmemeval_s_cleaned")
        self.assertEqual(groups["q1"], groups["q2"])
        self.assertEqual(groups["q2"], groups["q3"])
        self.assertNotEqual(groups["q1"], groups["q4"])
        self.assertEqual(len(set(groups.values())), 2)

    def test_generation_is_gold_free_and_not_user_confirmed(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.prepare([example()], directory)
            target = Path(directory) / "cohort"
            generation = (target / "generation_cases.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("GOLD_", generation)
            self.assertNotIn("ANNOTATION_MUST_BE_REMOVED", generation)
            self.assertNotIn("HIDDEN_LABEL_REMOVED", generation)
            row = json.loads(generation)
            self.assertEqual(row["scope_id"], "longmemeval:q1")
            self.assertEqual(set(row), {"example_id", "scope_id", "group_id", "query", "query_date", "sessions"})
            self.assertIn("GOLD_ANSWER", (target / "scoring_references.jsonl").read_text())
            self.assertFalse(manifest["memory_boundary"]["auto_confirm"])
            self.assertEqual(manifest["memory_boundary"]["representation"], "research_only_unconfirmed_candidate_evidence")
            self.assertTrue(manifest["fixture_only"])
            self.assertFalse(manifest["confirmatory_claim_allowed"])

    def test_overlap_is_rejected_at_group_level_even_when_question_ids_differ(self):
        rows = [example("conv-42:qa:999", session_id="s", category="1")]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "development-overlapping"):
                self.prepare(rows, directory, dataset="locomo10", role="locked_external_test")
            self.assertFalse((Path(directory) / "cohort").exists())

    def test_locomo_exploratory_excludes_only_prespecified_distractors(self):
        rows = [example("conv-42:qa:999", category="1"), example("conv-42:qa:1000", category="5")]
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.prepare(rows, directory, dataset="locomo10")
            self.assertEqual(manifest["role"], "exploratory")
            self.assertEqual(manifest["counts"]["examples"], 1)
            self.assertEqual(manifest["counts"]["independent_groups"], 1)
            self.assertEqual(manifest["counts"]["excluded_prespecified_examples"], 1)
            self.assertFalse(manifest["preparation_eligible_for_confirmatory_test"])
            self.assertIn("locomo:conv-42", manifest["development_exposure"]["overlapping_development_group_ids"])

    def test_additional_ledger_cannot_remove_known_pilot_exposure(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.prepare([example("conv-42:qa:999", category="1")], directory, dataset="locomo10",
                                    exposure_ledger={"datasets": {"locomo10": {"group_ids": []}}})
            self.assertEqual(len(manifest["development_exposed_group_ids"]), 9)
            self.assertIn("locomo:conv-42", manifest["development_exposure"]["overlapping_development_group_ids"])

    def test_shared_history_exposure_propagates_to_entire_independent_test_cluster(self):
        rows = [example("q1"), example("q2")]
        ledger = {"datasets": {"longmemeval_s_cleaned": {"example_ids": ["q1"], "exposure_review_complete": True}}}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "development-overlapping"):
                self.prepare(rows, directory, exposure_ledger=ledger)

    def test_too_few_groups_blocks_confirmatory_preparation_and_lock_binds_every_file(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.prepare([example()], directory)
            self.assertIn("too_few_independent_history_groups_for_predeclared_minimum", manifest["confirmatory_preparation_blockers"])
            target = Path(directory) / "cohort"
            checksum = manifest.pop("cohort_sha256")
            self.assertEqual(checksum, stable_digest(manifest))
            for name, spec in manifest["files"].items():
                self.assertEqual(file_identity(target / name), {key: spec[key] for key in ("sha256", "bytes")})
            lock = json.loads((target / "preregistration_lock.json").read_text())
            self.assertEqual(lock["plan_sha256"], stable_digest(self.plan))
            with self.assertRaisesRegex(ValueError, "not empty"):
                self.prepare([example()], directory)

    def test_no_effect_precision_or_posthoc_exclusion_plan_is_rejected(self):
        for field, value in (("minimum_effect", None), ("minimum_independent_groups", 1)):
            plan = copy.deepcopy(self.plan)
            plan["statistics"][field] = value
            with self.assertRaises(ValueError):
                validate_plan(plan)
        plan = copy.deepcopy(self.plan)
        plan["completion_and_failure_policy"]["optional_stopping"] = True
        with self.assertRaisesRegex(ValueError, "prespecified"):
            validate_plan(plan)

    def test_published_pilot_exposure_source_still_matches_the_nine_declared_conversations(self):
        _verify_pilot_exposure_source(self.plan)

    def test_public_validator_recomputes_lock_rows_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            original = self.prepare([example()], directory)
            manifest, cases, references = validate_cohort(Path(directory) / "cohort")
            self.assertEqual(manifest, original)
            self.assertEqual(len(cases), 1)
            self.assertEqual(references[0]["reference_answer"], "GOLD_ANSWER_MUST_NOT_ENTER_GENERATION")

    def test_validator_rejects_changed_claim_even_if_manifest_self_hash_is_recomputed(self):
        with tempfile.TemporaryDirectory() as directory:
            self.prepare([example()], directory)
            path = Path(directory) / "cohort/cohort_manifest.json"
            manifest = json.loads(path.read_text())
            manifest["confirmatory_claim_allowed"] = True
            manifest["cohort_sha256"] = stable_digest({key: value for key, value in manifest.items() if key != "cohort_sha256"})
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "metadata/claim"):
                validate_cohort(path.parent)

    def test_validator_rejects_forged_group_after_file_and_manifest_hashes_are_updated(self):
        with tempfile.TemporaryDirectory() as directory:
            self.prepare([example()], directory)
            target = Path(directory) / "cohort"
            path = target / "generation_cases.jsonl"
            case = json.loads(path.read_text())
            case["group_id"] = "pretend-this-is-an-independent-subject"
            path.write_text(json.dumps(case) + "\n", encoding="utf-8")
            manifest_path = target / "cohort_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["files"][path.name].update(file_identity(path))
            manifest["cohort_sha256"] = stable_digest({key: value for key, value in manifest.items() if key != "cohort_sha256"})
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "statistical group"):
                validate_cohort(target)

    def test_validator_rejects_preregistration_lock_file_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            self.prepare([example()], directory)
            target = Path(directory) / "cohort"
            lock_path = target / "preregistration_lock.json"
            lock = json.loads(lock_path.read_text())
            lock["plan"]["statistics"]["minimum_effect"] = 0.9
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                validate_cohort(target)


if __name__ == "__main__":
    unittest.main()
