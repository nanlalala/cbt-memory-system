"""Gold-free, whole-arm Persona integration fixtures; no APIs/data download."""
from pathlib import Path
from dataclasses import replace
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation/objective-memory"))
import v3_run_objective_comparison as runner
from v3_persona_retrieval import QUERY_POLICY


def fixture():
    case = {"case_id": "persona:q", "group_id": "persona:g", "subject_id": "persona:g",
            "dataset_key": "personamem_v1", "task": "preference", "query": "What should I choose?",
            "options": ["(a) swimming", "(b) walking", "(c) coding", "(d) sailing"],
            "events": [], "as_of": None, "visible_context_end_exclusive": 2,
            "source_context_id": "fixture", "history_scope_id": "fixture:prefix:2",
            "history": [{"turn_id": "t0", "session_id": "s", "order": 0, "role": "user",
                         "text": "I prefer swimming at sunset."},
                        {"turn_id": "t1", "session_id": "s", "order": 1, "role": "assistant",
                         "text": "Good morning."}]}
    ref = {"case_id": "persona:q", "group_id": "persona:g", "dataset_key": "personamem_v1",
           "task": "preference", "metric": "personamem_option_exact_match", "correct_option": "a"}
    settings = runner.Settings(model="fixture-model", accepted_provider_models=("fixture-alias",),
                               short_turns=1, top_k=3, minimum_relevance=.01,
                               short_chars=1200, long_chars=1600, max_context_chars=6000)
    return case, ref, settings


class PersonaRunnerIntegrationTests(unittest.TestCase):
    def test_shared_option_query_finds_topic_and_keeps_quote_control_equivalent(self):
        case, ref, settings = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            runner.prepare_run([case], [ref], path, settings, "https://example.invalid/v1", {}, fixture_only=True)
            protocol, prompts, _, _ = runner.load_run(path)
            recent, raw, managed = prompts
            self.assertNotIn("swimming at sunset", recent["messages"][1]["content"])
            self.assertIn("swimming at sunset", raw["messages"][1]["content"])
            self.assertEqual(raw["messages"], managed["messages"])
            self.assertEqual(raw["canonical_request_key"], managed["canonical_request_key"])
            self.assertEqual(raw["trace"]["retrieval_query_policy"], QUERY_POLICY)
            self.assertEqual(protocol["deduplication"]["raw_managed_equal_cases"], 1)
            self.assertTrue(managed["trace"]["storage"])
            self.assertTrue(raw["trace"]["included_source_spans"])

    def test_released_prefix_guard_precedes_source_use_even_for_deleted_rows(self):
        case, ref, settings = fixture()
        case["history"].append({"turn_id": "future", "session_id": "s", "order": 2,
                                "role": "user", "text": "Future hidden preference", "lifecycle_status": "deleted"})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "exclusive|boundary|prefix"):
                runner.prepare_run([case], [ref], Path(directory), settings,
                                   "https://example.invalid/v1", {}, fixture_only=True)

    def test_runner_renders_relevant_original_span_instead_of_dropping_whole_long_message(self):
        case, ref, settings = fixture()
        source = "lunch " * 600 + "I prefer swimming at sunset. " + "lunch " * 600
        case["history"][0]["text"] = source
        settings = replace(settings, minimum_relevance=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            runner.prepare_run([case], [ref], path, settings, "https://example.invalid/v1", {}, fixture_only=True)
            _, prompts, _, _ = runner.load_run(path)
            raw = next(p for p in prompts if p["arm"] == "raw_history_rag")
            self.assertIn("I prefer swimming at sunset.", raw["messages"][1]["content"])
            spans = [s for s in raw["trace"]["included_source_spans"]
                     if s["chunk_id"] in raw["trace"]["included_long_ids"]]
            self.assertTrue(spans)
            self.assertTrue(all(s["end"] - s["start"] <= 1200 for s in spans))
            self.assertGreater(len(source), settings.long_chars)


if __name__ == "__main__":
    unittest.main()
