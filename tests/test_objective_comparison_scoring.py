"""Small deterministic scoring fixtures, not evidence of benchmark performance."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation/objective-memory"))

from score_objective_comparison import (  # noqa: E402
    ARMS, deterministic_score, mab_normalize_answer, mab_parse_output,
    mab_substring_exact_match, paired_cluster_intervals,
    personamem_option_exact_match, score_loaded_run,
    score_run,
)


def fixture(groups=("history-1", "history-1", "history-2")):
    protocol = {"case_ids": [f"q{i}" for i in range(len(groups))], "settings": {"seed": 3},
                "evaluation_policy": {"bootstrap_samples": 200, "bootstrap_seed": 19}}
    prompts, refs, responses, outcomes = [], [], {}, {}
    for i, group in enumerate(groups):
        case_id = f"q{i}"
        common = {"case_id": case_id, "group_id": group, "task": "factconsolidation_sh_6k",
                  "dataset_key": "memoryagentbench_conflict"}
        refs.append({**common, "metric": "mab_substring_exact_match", "accepted_answers": ["Thursday"]})
        for arm in ARMS:
            key = case_id, arm
            prompts.append({**common, "arm": arm, "trace": {"included_memory_ids": ["m1"], "dropped_memory_ids": ["m2"]}})
            responses[key] = {"hypothesis": "Thursday", "usage": {"total_tokens": 10},
                              "latency_seconds": 1., "provider_model": "mock", "charged_call": True}
            outcomes[key] = {"status": "success"}
    return protocol, prompts, refs, responses, outcomes


class UpstreamDeterministicMetricTests(unittest.TestCase):
    def test_mab_normalization_keeps_official_ascii_rules_and_order(self):
        self.assertEqual(mab_normalize_answer("The RED-PENCIL,  a  name!"), "redpencil name")
        self.assertEqual(mab_normalize_answer("Straße ＋5"), "straße ＋5")
        # Official metric removes ASCII signs; this is a metric limitation,
        # distinct from the memory store's semantics-preserving duplicate rule.
        self.assertEqual(mab_normalize_answer("-5"), mab_normalize_answer("+5"))

    def test_mab_substring_alias_max_not_token_overlap(self):
        self.assertEqual(mab_substring_exact_match("The meeting is THURS.", ["Friday", "Thurs"])[0], 1.)
        self.assertEqual(mab_substring_exact_match("red", ["red pencil"])[0], 0.)
        self.assertEqual(mab_substring_exact_match("redpencil", ["red pencil"])[0], 0.)

    def test_mab_official_parse_and_full_response_max(self):
        text = "Reasoning first.\nAnswer: Answer: Thursday\nend"
        self.assertEqual(mab_parse_output(text), "Thursday")
        self.assertEqual(mab_substring_exact_match(text, ["Thursday"])[0], 1.)
        # Full response can match even when parsed first line does not.
        self.assertEqual(mab_substring_exact_match("Nothing\nThursday", ["Thursday"])[0], 1.)

    def test_mab_empty_or_missing_aliases_are_invalid_references(self):
        for value in (None, [], [""], [7]):
            with self.assertRaises(ValueError):
                mab_substring_exact_match("anything", value)

    def test_personamem_released_option_parser(self):
        self.assertEqual(personamem_option_exact_match("(C)", "c")[0], 1.)
        self.assertEqual(personamem_option_exact_match("c", "(c)")[0], 1.)
        self.assertEqual(personamem_option_exact_match("c d", "c")[0], 0.)
        self.assertEqual(personamem_option_exact_match("The third choice", "c")[0], 0.)

    def test_personamem_parentheses_override_bare_options(self):
        self.assertEqual(personamem_option_exact_match("a and b, select (c)", "c")[0], 1.)
        self.assertEqual(personamem_option_exact_match("c but (a)", "c")[0], 0.)
        self.assertEqual(personamem_option_exact_match("(a), (c)", "c")[0], 0.)

    def test_personamem_final_tag_and_upstream_full_response_fallback(self):
        self.assertEqual(personamem_option_exact_match("(a) was considered. <final_answer>(c)</final_answer>", "c")[0], 1.)
        # Preserve the upstream fallback even when a later bare letter differs.
        self.assertEqual(personamem_option_exact_match("(a) earlier <final_answer>b</final_answer>", "a")[0], 1.)
        score, parsed = personamem_option_exact_match("<final_answer>(a)</final_answer><final_answer>(c)</final_answer>", "c")
        self.assertEqual((score, parsed), (1., "(c)"))

    def test_clarification_gets_no_invented_user_answer(self):
        reference = {"metric": "personamem_option_exact_match", "correct_option": "c"}
        self.assertEqual(deterministic_score("Could you remind me of your preference?", reference)[0], 0.)

    def test_unknown_metric_and_invalid_option_fail_closed(self):
        with self.assertRaises(ValueError):
            deterministic_score("x", {"metric": "freeform_judge"})
        with self.assertRaises(ValueError):
            personamem_option_exact_match("e", "e")


class PairedObjectiveAggregationTests(unittest.TestCase):
    def score(self, data):
        return score_loaded_run(*data, bootstrap_samples=200)

    def test_complete_has_all_arms_per_task_and_no_clinical_claim(self):
        summary, rows = self.score(fixture())
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(len(rows), 9)
        self.assertFalse(summary["clinical_or_user_benefit_claim"])
        self.assertFalse(summary["confirmatory_claim"])
        dataset = summary["datasets"]["memoryagentbench_conflict"]
        self.assertIn("factconsolidation_sh_6k", dataset["by_task"])
        self.assertEqual(dataset["conditions"][ARMS[0]]["accuracy_micro"], 1.)

    def test_failure_is_zero_in_denominator_and_availability_separate(self):
        data = fixture()
        failed = "q0", ARMS[2]
        data[3].pop(failed)
        data[4][failed] = {"status": "exhausted_failure"}
        summary, rows = self.score(data)
        stats = summary["datasets"]["memoryagentbench_conflict"]["conditions"][ARMS[2]]
        self.assertAlmostEqual(stats["accuracy_micro"], 2 / 3)
        self.assertAlmostEqual(stats["availability"], 2 / 3)
        self.assertEqual(stats["accuracy_given_valid_response_diagnostic"], 1.)
        row = next(row for row in rows if (row["case_id"], row["arm"]) == failed)
        self.assertEqual((row["score"], row["available"], row["failure_stage"]),
                         (0., False, "provider_technical_failure"))

    def test_pending_and_inflight_allow_progress_only(self):
        for status in ("pending", "inflight"):
            data = fixture()
            key = "q2", ARMS[1]
            data[3].pop(key)
            data[4][key] = {"status": status}
            summary, rows = self.score(data)
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["remaining_requests"], 1)
            self.assertFalse(summary["final_effectiveness_result"])
            self.assertNotIn("datasets", summary)
            self.assertEqual(rows, [])

    def test_micro_and_group_equal_do_not_inflate_shared_history(self):
        data = fixture()
        for case_id in ("q0", "q1"):
            data[3][(case_id, ARMS[2])]["hypothesis"] = "Wrong"
        summary, _ = self.score(data)
        dataset = summary["datasets"]["memoryagentbench_conflict"]
        self.assertAlmostEqual(dataset["conditions"][ARMS[2]]["accuracy_micro"], 1 / 3)
        self.assertEqual(dataset["conditions"][ARMS[2]]["accuracy_group_equal"], .5)
        contrast = dataset["paired_differences"]["managed_memory_rag_minus_raw_history_rag"]
        self.assertEqual(contrast["independent_groups"], 2)
        self.assertEqual(contrast["mean_accuracy_delta_group_equal"], -.5)
        self.assertAlmostEqual(contrast["mean_accuracy_delta_micro"], -2 / 3)

    def test_one_shared_context_has_no_confidence_interval(self):
        data = fixture(groups=("same", "same", "same"))
        summary, _ = self.score(data)
        contrast = summary["datasets"]["memoryagentbench_conflict"]["paired_differences"]["managed_memory_rag_minus_raw_history_rag"]
        self.assertEqual(contrast["independent_groups"], 1)
        self.assertIsNone(contrast["group_equal_95_percent_interval"])
        self.assertIn("fewer_than_two", contrast["interval_status"])

    def test_bootstrap_is_deterministic_and_keeps_cluster_pairing(self):
        values = {"history-a": [-1., -1.], "history-b": [1.]}
        first = paired_cluster_intervals(values, seed=6, resamples=200)
        self.assertEqual(first, paired_cluster_intervals(values, seed=6, resamples=200))
        self.assertEqual(first["group_equal_95_percent_interval"], [-1., 1.])

    def test_wrong_answers_do_not_fabricate_error_stage_or_stale_gold(self):
        data = fixture()
        data[3][("q0", ARMS[2])]["hypothesis"] = "Friday"
        _, rows = self.score(data)
        row = next(row for row in rows if row["case_id"] == "q0" and row["arm"] == ARMS[2])
        self.assertEqual(row["failure_stage"], "answer_incorrect_stage_unresolved")
        self.assertIsNone(row["stale_answer_diagnostic"])
        self.assertEqual(row["trace"]["included_memory_count"], 1)

    def test_shared_canonical_response_cost_is_charged_once(self):
        data = fixture(groups=("one",))
        for arm in ARMS:
            data[3][("q0", arm)].update(canonical_request_key="shared", charged_call=arm == ARMS[0],
                                        deduplicated_call=arm != ARMS[0])
        summary, _ = self.score(data)
        self.assertEqual(summary["cost"]["successful_unique_provider_calls"], 1)
        self.assertEqual(summary["cost"]["reported_tokens_including_failed_attempts_when_supplied"]["total_tokens"], 10)
        contrast = summary["datasets"]["memoryagentbench_conflict"]["paired_differences"]["managed_memory_rag_minus_raw_history_rag"]
        self.assertEqual(contrast["result_type"], "by_construction_control_equivalence")
        self.assertEqual(contrast["identical_prompt_shared_response_fraction"], 1.)
        self.assertIsNone(contrast["group_equal_95_percent_interval"])
        self.assertEqual(contrast["interval_status"], "blocked_by_identical_inputs_and_shared_responses")

    def test_mixed_input_equivalence_keeps_all_pairs_primary(self):
        data = fixture()
        for case_id in ("q0", "q1"):
            for prompt in data[1]:
                if prompt["case_id"] == case_id and prompt["arm"] in ARMS[1:]:
                    prompt["canonical_request_key"] = case_id + "shared"
        data[3][("q2", ARMS[2])]["hypothesis"] = "Wrong"
        summary, _ = self.score(data)
        contrast = summary["datasets"]["memoryagentbench_conflict"]["paired_differences"]["managed_memory_rag_minus_raw_history_rag"]
        self.assertEqual(contrast["questions"], 3)
        self.assertEqual(contrast["identical_prompt_shared_response_cases"], 2)
        self.assertEqual(contrast["nonidentical_input_subset_diagnostic"]["questions"], 1)
        self.assertAlmostEqual(contrast["mean_accuracy_delta_micro"], -1 / 3)
        self.assertEqual(contrast["nonidentical_input_subset_diagnostic"]["mean_accuracy_delta_micro"], -1.)

    def test_failed_attempt_usage_is_included_and_unknown_usage_explicit(self):
        summary, _ = score_loaded_run(*fixture(), bootstrap_samples=200,
                                      call_outcomes=[{"status": "technical_failure", "usage": {"total_tokens": 7}},
                                                     {"status": "technical_failure"}])
        self.assertEqual(summary["cost"]["reported_tokens_including_failed_attempts_when_supplied"]["total_tokens"], 97)
        self.assertEqual(summary["cost"]["failed_attempts_without_token_usage"], 1)

    def test_references_outcomes_group_and_metric_cannot_drift(self):
        mutations = (
            lambda d: d[1].pop(),
            lambda d: d[2].pop(),
            lambda d: d[4].pop(("q0", ARMS[0])),
            lambda d: d[1][0].update(group_id="different-history"),
            lambda d: d[2][0].update(metric="unapproved_model_judge"),
            lambda d: d[4][("q0", ARMS[0])].update(status="pending"),
        )
        for mutate in mutations:
            data = deepcopy(fixture())
            mutate(data)
            with self.assertRaises(ValueError):
                self.score(data)

    def test_bootstrap_settings_must_match_frozen_policy(self):
        with self.assertRaisesRegex(ValueError, "frozen evaluation"):
            score_loaded_run(*fixture(), bootstrap_samples=300)


class FrozenArtifactScoringTests(unittest.TestCase):
    """Real prepare/load/score code with tiny synthetic journals, zero requests."""

    def prepare(self, directory):
        import run_objective_comparison as runner
        cases, refs = [], []
        for i in range(2):
            common = {"case_id": f"synthetic-q{i}", "group_id": f"synthetic-history{i}",
                      "subject_id": f"synthetic-subject{i}", "dataset_key": "memoryagentbench_conflict",
                      "task": "factconsolidation_sh_6k"}
            cases.append({**common, "query": "What is my walking target?", "options": [], "events": [],
                          "history": [
                              {"turn_id": "old", "session_id": "first", "role": "user", "order": 0,
                               "text": "My walking target is Thursday."},
                              {"turn_id": "new", "session_id": "second", "role": "user", "order": 1,
                               "text": f"Unrelated synthetic note {i}."}]})
            refs.append({**common, "metric": "mab_substring_exact_match", "accepted_answers": ["Thursday"]})
        runner.prepare_run(cases, refs, directory, runner.Settings(model="mock", short_turns=1),
                           "https://example.invalid/v1", {"synthetic_fixture": True}, fixture_only=True)
        return runner

    def journal(self, directory, runner, *, terminal=False, wrong_model=False):
        protocol, prompts, _, _ = runner.load_run(directory)
        canonical = {}
        for prompt in prompts:
            canonical.setdefault(prompt["canonical_request_key"], prompt)
        for canonical_key, prompt in canonical.items():
            intent = {"request_key": canonical_key, "canonical_request_key": canonical_key,
                      "prompt_sha256": prompt["prompt_sha256"], "protocol_sha256": protocol["protocol_sha256"],
                      "attempt_index": 1, "started_at_unix": 0.}
            runner._append(directory / "response_attempts.jsonl", intent)
            if not terminal:
                row = {**intent, "hypothesis": "Thursday", "provider_model": "wrong" if wrong_model else "mock",
                       "finish_reason": "stop", "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
                       "latency_seconds": .1}
                row["response_sha256"] = runner.stable_digest(row)
                runner._append(directory / "responses.jsonl", row)
            outcome = {**intent, "status": "technical_failure" if terminal else "success", "retryable": False}
            if terminal:
                outcome["error_type"] = "SyntheticTransportError"
            runner._append(directory / "response_outcomes.jsonl", outcome)
            if terminal:
                runner._append(directory / "failures.jsonl", outcome)
        return protocol, len(canonical)

    def test_offline_recompute_binds_raw_artifacts_and_canonical_cost(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            runner = self.prepare(directory)
            protocol, unique = self.journal(directory, runner)
            summary = score_run(directory)
            self.assertEqual(summary["status"], "complete")
            self.assertTrue(summary["fixture_only"])
            self.assertEqual(summary["result_origin"], "synthetic_software_fixture")
            self.assertFalse(summary["final_effectiveness_result"])
            self.assertEqual(summary["protocol_sha256"], protocol["protocol_sha256"])
            self.assertEqual(summary["cost"]["successful_unique_provider_calls"], unique)
            self.assertEqual(summary["cost"]["reported_tokens_including_failed_attempts_when_supplied"]["total_tokens"], unique * 5)
            self.assertIn("references.jsonl", summary["input_artifacts_sha256"])
            self.assertIn("inventory.jsonl", summary["input_artifacts_sha256"])
            self.assertEqual(len((directory / "objective_scores.jsonl").read_text().splitlines()), 6)
            self.assertEqual(summary, score_run(directory, write_outputs=False))

    def test_partial_cannot_publish_stale_final_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.prepare(directory)
            (directory / "objective_summary.json").write_text('{"status":"complete"}')
            (directory / "objective_scores.jsonl").write_text('{}\n')
            summary = score_run(directory)
            self.assertEqual(summary["status"], "partial")
            self.assertFalse((directory / "objective_summary.json").exists())
            self.assertFalse((directory / "objective_scores.jsonl").exists())
            self.assertEqual(json.loads((directory / "objective_progress.json").read_text()), summary)

    def test_terminal_failures_form_complete_zero_score_cohort(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            runner = self.prepare(directory)
            _, unique = self.journal(directory, runner, terminal=True)
            summary = score_run(directory)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["completion_counts"], {"exhausted_failure": 6})
            self.assertEqual(summary["cost"]["failed_provider_attempts"], unique)
            stats = summary["datasets"]["memoryagentbench_conflict"]["conditions"][ARMS[2]]
            self.assertEqual((stats["accuracy_micro"], stats["availability"]), (0., 0.))

    def test_wrong_model_or_reference_tamper_rejected_before_scoring(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            runner = self.prepare(directory)
            self.journal(directory, runner, wrong_model=True)
            with self.assertRaises(ValueError):
                score_run(directory)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            runner = self.prepare(directory)
            self.journal(directory, runner)
            with (directory / "references.jsonl").open("a") as handle:
                handle.write('{}\n')
            with self.assertRaisesRegex(ValueError, "frozen artifact"):
                score_run(directory)


if __name__ == "__main__":
    unittest.main()
