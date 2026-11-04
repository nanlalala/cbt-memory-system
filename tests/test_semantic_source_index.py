"""Index hints may alter retrieval, never provenance, eligibility or fact status."""
from copy import deepcopy
from dataclasses import replace
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation/objective-memory"))
import run_objective_comparison as runner
from semantic_source_index import validate_index, rank_sources


class SemanticSourceIndexTests(unittest.TestCase):
    def setUp(self):
        self.doc = runner.HistoryTurn(scope_id="u", document_id="d", session_id="s", date="2026-10-01",
                                      speaker="user", text="I will write one down after the meeting.")
        self.row = {"subject_id": "u", "source_document_id": "d", "source_text_sha256": hashlib.sha256(self.doc.text.encode()).hexdigest(),
                    "memory_type": "cbt_assignment", "summary": "thought record homework plan", "supporting_text": self.doc.text}
        self.ranker = runner.HistoryRanker(runner.Settings(model="fixture", backend="lexical", top_k=1, minimum_relevance=.01))

    def test_semantic_index_changes_selection_but_renders_original_quote(self):
        checked = validate_index([self.row], documents=[self.doc], subject_id="u")
        self.assertEqual(self.ranker.rank("thought homework", [self.doc], scope_id="u"), [])
        selected = rank_sources("thought homework", [self.doc], checked, self.ranker, subject_id="u")
        self.assertEqual(selected, [self.doc])
        self.assertNotIn("thought record", selected[0].text)

    def test_deleted_future_or_missing_source_never_returns_via_index(self):
        with self.assertRaisesRegex(ValueError, "ineligible"):
            validate_index([self.row], documents=[], subject_id="u")
        self.assertEqual(rank_sources("thought homework", [], [self.row], self.ranker, subject_id="u"), [])

    def test_cross_subject_source_tampering_and_false_quote_rejected(self):
        for field, value, error in (("subject_id", "another", "cross-subject"),
                                     ("source_text_sha256", "bad", "checksum"),
                                     ("supporting_text", "I completed it", "exact supporting"),
                                     ("memory_type", "diagnosis", "unsupported")):
            row = {**self.row, field: value}
            with self.assertRaisesRegex(ValueError, error):
                validate_index([row], documents=[self.doc], subject_id="u")

    def test_multiple_candidates_do_not_consume_separate_top_k_slots(self):
        row = {**self.row, "memory_type": "goal"}
        checked = validate_index([self.row, row], documents=[self.doc], subject_id="u")
        self.assertEqual(len(rank_sources("thought homework", [self.doc], checked, self.ranker, subject_id="u")), 1)

    def test_preparer_freezes_real_content_difference_and_keeps_gold_out(self):
        case = {"case_id": "q", "subject_id": "u", "group_id": "g", "dataset_key": "synthetic", "task": "recall",
                "query": "thought homework", "options": [], "events": [], "as_of": None,
                "history": [{"turn_id": "t1", "session_id": "s", "role": "user", "order": 0, "text": self.doc.text},
                            {"turn_id": "t2", "session_id": "s", "role": "user", "order": 1, "text": "Hello again."}]}
        source = runner._document(case, case["history"][0])
        index = [{"case_id": "q", "candidates": [{**self.row, "source_document_id": source.document_id}]}]
        refs = [{"case_id": "q", "group_id": "g", "dataset_key": "synthetic", "task": "recall", "metric": "mab_substring_exact_match", "accepted_answers": ["scoring-only-gold"]}]
        settings = runner.Settings(model="fixture", backend="lexical", short_turns=1, top_k=1, minimum_relevance=.01)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            result = runner.prepare_run([case], refs, path, settings, "https://example.invalid/v1", {"fixture": True}, fixture_only=True, semantic_index=index)
            self.assertEqual(result["deduplication"]["raw_managed_equal_cases"], 0)
            protocol, prompts, _, _ = runner.load_run(path)
            self.assertNotEqual(prompts[1]["messages"], prompts[2]["messages"])
            self.assertIn(self.doc.text, prompts[2]["messages"][1]["content"])
            self.assertNotIn("scoring-only-gold", str(prompts))
            self.assertNotIn(self.row["summary"], str(prompts))
            self.assertEqual(protocol["representation"]["version"], "semantic_source_index_v1")
            self.assertIn("unreported", protocol["representation"]["index_generation_cost"])
            self.assertEqual(prompts[2]["trace"]["storage"][0]["storage_status"], "candidate")
            with self.assertRaises(ValueError):
                runner.prepare_run([case], refs, path / "missing", settings, "https://example.invalid/v1", {}, fixture_only=True, semantic_index=[])


if __name__ == "__main__":
    unittest.main()
