"""Small synthetic v3 integration fixtures; no dataset/model/API or guard bypass."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation/objective-memory"))
import v3_run_objective_comparison as runner
import v3_score_objective_comparison as scorer


def fixture():
    case = {"case_id": "fixture:q", "subject_id": "fixture:u", "group_id": "fixture:g",
            "dataset_key": "memoryagentbench_conflict", "task": "fixture-task",
            "query": "What is the official language of Arcadia?", "options": [], "events": [], "as_of": None,
            "history": [{"session_id": "scope", "turn_id": "old", "order": 0, "role": "context", "text": "The official language of Arcadia is Alpha."},
                        {"session_id": "scope", "turn_id": "new", "order": 1, "role": "context", "text": "The official language of Arcadia is Delta."},
                        {"session_id": "scope", "turn_id": "recent", "order": 2, "role": "context", "text": "The capital of Borealis is East Bay."}]}
    ref = {"case_id": "fixture:q", "group_id": "fixture:g", "dataset_key": case["dataset_key"], "task": case["task"],
           "metric": "mab_substring_exact_match", "accepted_answers": ["gold-only-needle"]}
    return case, ref


class V3RunnerIntegrationTests(unittest.TestCase):
    def settings(self):
        return runner.Settings(model="fixture", accepted_provider_models=("fixture-alias",), backend="lexical",
                               short_turns=1, top_k=4, short_chars=1500, long_chars=2000, max_context_chars=7000,
                               max_response_tokens=64, minimum_relevance=0, thinking_mode="disabled")

    def prepare(self, path, case=None, ref=None):
        if case is None:
            case, ref = fixture()
        return runner.prepare_run([case], [ref], path, self.settings(), "https://example.invalid/v1", {}, fixture_only=True)

    def test_version_view_changes_sources_without_changing_shared_instructions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            receipt = self.prepare(path)
            protocol, prompts, _, _ = runner.load_run(path)
            self.assertEqual(protocol["protocol_version"], "objective-public-qa-version-view-v3.1")
            self.assertEqual(len({p["messages"][0]["content"] for p in prompts}), 1)
            self.assertIn("fictional world", prompts[0]["messages"][0]["content"])
            self.assertIn("Alpha", prompts[1]["messages"][1]["content"])
            self.assertNotIn("Alpha", prompts[2]["messages"][1]["content"])
            self.assertIn("Delta", prompts[2]["messages"][1]["content"])
            self.assertNotIn("gold-only-needle", str(prompts))
            self.assertEqual(receipt["deduplication"]["raw_managed_equal_cases"], 0)
            inventory = runner.read_jsonl(path / "inventory.jsonl")
            self.assertEqual(len(inventory), 3)
            self.assertEqual({r["storage_decision"]["storage_status"] for r in inventory}, {"candidate"})
            self.assertIn("superseded", {r["benchmark_version_decision"]["status"] for r in inventory})

    def test_unknown_grammar_is_not_erased_and_restricted_sources_never_enter_context(self):
        case, ref = fixture()
        case["history"].insert(0, {"session_id": "scope", "turn_id": "free", "order": 0, "role": "context", "text": "Arcadia might use a language nobody knows."})
        case["history"][1]["sensitivity"] = "restricted"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory);self.prepare(path, case, ref)
            _, prompts, _, _ = runner.load_run(path)
            self.assertNotIn("Alpha", str(prompts))
            self.assertIn("might use", prompts[2]["messages"][1]["content"])
            self.assertTrue(prompts[2]["trace"]["excluded_source_units"])

    def test_persona_shared_spans_and_all_choices_preserve_equivalence(self):
        case, ref = fixture()
        case.update(dataset_key="personamem_v1", visible_context_end_exclusive=3, options=["(a) Alpha", "(b) Delta", "(c) East", "(d) West"])
        ref.update(dataset_key="personamem_v1", metric="personamem_option_exact_match", correct_option="(b)")
        ref.pop("accepted_answers")
        case["history"][0]["text"] = "Arcadia " + "old discussion " * 150 + "Alpha."
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory);self.prepare(path, case, ref)
            protocol, prompts, _, _ = runner.load_run(path)
            self.assertEqual(prompts[1]["messages"], prompts[2]["messages"])
            self.assertIn("all_four", prompts[2]["trace"]["retrieval_query_policy"])
            self.assertIn("source_span", prompts[2]["messages"][1]["content"])
            self.assertNotIn("fictional world", prompts[2]["messages"][0]["content"])
            self.assertTrue(prompts[2]["trace"]["storage"])
            self.assertIn("exact source spans", protocol["shared_adaptations"]["personamem"])

    def test_frozen_code_and_source_tampering_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);self.prepare(path)
            data=runner.read_jsonl(path / "inventory.jsonl");data[0]["source"]["text"] += " altered"
            runner.write_jsonl(path / "inventory.jsonl", data)
            with self.assertRaisesRegex(ValueError,"frozen artifact"):
                runner.load_run(path)

    def test_scoring_keeps_official_substring_and_reports_uncommitted_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);self.prepare(path)
            protocol,prompts,_,_=runner.load_run(path)
            for key,prompt in runner._canonical(prompts).items():
                intent={"request_key":key,"canonical_request_key":key,"protocol_sha256":protocol["protocol_sha256"],
                        "prompt_sha256":key,"attempt_index":1,"started_at_unix":1}
                response={**intent,"hypothesis":"I don't know. gold-only-needle or an alternative.","provider_model":"fixture-alias",
                          "finish_reason":"stop","latency_seconds":0.1,"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}
                response["response_sha256"]=runner.stable_digest(response)
                runner._append(path/'response_attempts.jsonl',intent)
                runner._append(path/'responses.jsonl',response)
                runner._append(path/'response_outcomes.jsonl',{**intent,"status":"success","retryable":False})
            summary=scorer.score_run(path)
            self.assertEqual(summary["status"],"complete")
            self.assertTrue(summary["fixture_only"])
            for arm in runner.ARMS:
                row=summary['v3_diagnostics']['conditions'][arm]
                self.assertEqual(row['official_accuracy'],1)
                self.assertEqual(row['strict_committed_accuracy'],0)
                self.assertEqual(row['official_positives_with_abstention_cue'],1)

    def test_removed_latest_source_does_not_restore_previous_slot_value(self):
        for update in ({"lifecycle_status": "deleted"}, {"valid_to": "2020-01-01T00:00:00Z"}):
            case, ref = fixture()
            case["as_of"] = "2026-01-01T00:00:00Z"
            case["history"][1].update(update)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                with self.assertRaisesRegex(ValueError, "opaque version tombstone"):
                    self.prepare(path, case, ref)
                self.assertFalse((path / "protocol.json").exists())

    def test_graph_seeds_are_only_common_recent_sources_that_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            case, ref = fixture()
            settings = self.settings()
            settings = runner.Settings(**{**runner.asdict(settings), "short_chars": 1})
            path = Path(directory)
            runner.prepare_run([case],[ref],path,settings,"https://example.invalid/v1",{},fixture_only=True)
            _,prompts,_,_=runner.load_run(path)
            for prompt in prompts[1:]:
                self.assertEqual(prompt['trace']['included_short_ids'], [])
                self.assertEqual(prompt['trace']['retrieval']['common_recent_seed_ids'], [])

    def test_fixture_cannot_start_paid_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);self.prepare(path)
            # This cloud container is intentionally not hosted Colab; the real
            # guard is exercised rather than patched or spoofed for the test.
            with self.assertRaises(RuntimeError):
                runner.generate_run(path,max_calls=1,key="synthetic-placeholder-not-a-secret")


if __name__ == '__main__':unittest.main()
