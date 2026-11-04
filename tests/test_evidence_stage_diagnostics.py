"""Evidence gaps, sufficient alternatives, and refusal to infer causal blame."""
import hashlib
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation/objective-memory"))
from diagnose_evidence_stages import observe, validate_annotations


class EvidenceDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.prompt = {"case_id": "q", "subject_id": "u", "arm": "raw_history_rag", "trace": {
            "encoding": "released_source_quotes_no_semantic_extraction", "included_short_ids": [],
            "included_long_ids": [], "retrieved_ids": [], "interactive_clarification": False}}
        self.label = {"case_id": "q", "protocol_sha256": "frozen", "reviewer_id": "fixture-author",
                      "annotation_origin": "developer_fixture", "coverage": "complete_sufficient_alternatives",
                      "alternatives": [[{"document_id": "d1", "start": 0, "end": 4,
                                         "source_text_sha256": hashlib.sha256(b"blue").hexdigest()}]]}

    def test_without_labels_no_stage_blame(self):
        row = observe(self.prompt, None)
        self.assertEqual(row["causal_attribution"], "unresolved")
        self.assertIsNone(row["user_effort_seconds"])
        self.assertEqual(row["clarification_requests"], 0)
        self.assertEqual(row["evidence_observation"], "unresolved_no_evidence_annotation")

    def test_retrieval_gap_and_budget_gap_are_distinct_observations(self):
        row = observe(self.prompt, self.label)
        self.assertEqual(row["evidence_observation"], "annotated_evidence_not_fully_retrieved")
        self.prompt["trace"]["retrieved_ids"] = ["d1"]
        row = observe(self.prompt, self.label)
        self.assertEqual(row["evidence_observation"], "annotated_evidence_retrieved_but_context_incomplete")
        self.assertEqual(row["causal_attribution"], "unresolved")

    def test_joint_evidence_requires_all_documents(self):
        self.label["alternatives"][0].append({"document_id": "d2"})
        self.prompt["trace"]["included_long_ids"] = ["d1"]
        self.assertFalse(observe(self.prompt, self.label)["stages"]["context"])
        self.label["alternatives"].append([{"document_id": "d1"}])
        self.assertTrue(observe(self.prompt, self.label)["stages"]["context"])

    def test_recent_only_gap_is_not_retriever_failure(self):
        self.prompt["arm"] = "short_term"
        row = observe(self.prompt, self.label)
        self.assertEqual(row["evidence_observation"], "annotated_evidence_outside_recent_context")
        self.assertEqual(row["stages"]["retrieval"], "not_applicable_recent_only")

    def test_source_and_protocol_binding(self):
        inventory = [{"document_id": "d1", "subject_id": "u", "source": {"text": "blue"}}]
        labels = validate_annotations([self.label], {"protocol_sha256": "frozen"}, [self.prompt], inventory)
        self.assertIn("q", labels)
        inventory[0]["source"]["text"] = "blue edited"
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_annotations([self.label], {"protocol_sha256": "frozen"}, [self.prompt], inventory)

    def test_partial_labels_and_cross_subject_labels_rejected(self):
        inventory = [{"document_id": "d1", "subject_id": "someone-else", "source": {"text": "blue"}}]
        with self.assertRaisesRegex(ValueError, "cross-subject"):
            validate_annotations([self.label], {"protocol_sha256": "frozen"}, [self.prompt], inventory)
        self.label["coverage"] = "partial"
        with self.assertRaisesRegex(ValueError, "partial"):
            validate_annotations([self.label], {"protocol_sha256": "frozen"}, [self.prompt], inventory)

    def test_semantic_encoding_cannot_be_misdiagnosed_as_whole_quotes(self):
        self.prompt["trace"]["encoding"] = "semantic_summary"
        with self.assertRaisesRegex(ValueError, "whole-source"):
            observe(self.prompt, self.label)


if __name__ == "__main__":
    unittest.main()
