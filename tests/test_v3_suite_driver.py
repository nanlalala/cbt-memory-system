"""Synthetic full-suite planning/finite scheduling checks; no data or API."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation/objective-memory"))
import v3_suite_driver as driver


def tiny_cohorts():
    mab = [{"case_id": f"mab:{task}:{n}", "group_id": "one-family", "subject_id": "one-family",
            "dataset_key": "memoryagentbench_conflict", "task": task, "query": f"Who has item {n}?",
            "options": [], "history": [], "events": [], "as_of": None}
           for task in ("fact_sh_a", "fact_mh_b") for n in range(3)]
    persona = [{"case_id": f"persona:{group}:{n}", "group_id": f"connected:{group}", "subject_id": f"p:{group}",
                "dataset_key": "personamem_v1", "task": "preference", "query": "What might I like?",
                "options": ["(a) tea", "(b) coffee", "(c) water", "(d) milk"], "history": [], "events": [], "as_of": None}
               for group in range(4) for n in range(group + 1)]
    return {"memoryagentbench_conflict": mab, "personamem_v1": persona}


def state(*, complete=False, inflight=0, calls=1):
    return {"complete": complete, "arm_outcomes": {"inflight": inflight}, "actual_calls_this_invocation": calls}


class SuiteDriverTests(unittest.TestCase):
    def setUp(self):
        self.cohorts = tiny_cohorts()
        self.exposed = [self.cohorts[driver.DATASETS[0]][0], self.cohorts[driver.DATASETS[0]][1]]

    def test_plan_has_disjoint_exhaustive_shards_and_whole_persona_groups(self):
        plan = driver.plan_cohorts(self.cohorts, self.exposed, strict_census=False)
        self.assertEqual(plan["all_cases"], 16)
        self.assertEqual(plan["all_arm_rows"], 48)
        ids = [cid for shard in plan["shards"] for cid in shard["case_ids"]]
        self.assertEqual(len(ids), len(set(ids)))
        group_shards = {}
        for shard in plan["shards"]:
            if shard["dataset_key"] == "personamem_v1":
                for group in shard["group_ids"]:
                    self.assertNotIn(group, group_shards)
                    group_shards[group] = shard["shard_id"]
        self.assertEqual(len(group_shards), 4)

    def test_dev_selects_exact_old_ids_exposure_expands_normalized_variants(self):
        plan = driver.plan_cohorts(self.cohorts, self.exposed, mode="dev16", strict_census=False)
        self.assertEqual(set(plan["datasets"][driver.DATASETS[0]]["case_ids"]), {r["case_id"] for r in self.exposed})
        partition = plan["datasets"][driver.DATASETS[0]]["development_partition"]
        self.assertEqual(partition["development_count"], 4)
        self.assertEqual(partition["question_heldout_count"], 2)
        self.assertNotIn(driver.DATASETS[1], plan["datasets"])

    def test_production_census_rejects_tiny_fixtures_before_any_execution(self):
        with self.assertRaisesRegex(ValueError, "16 MAB"):
            driver.plan_cohorts(self.cohorts, self.exposed)

    def test_full_option_capacity_is_common_and_not_silently_truncated(self):
        cohorts = deepcopy(self.cohorts)
        cohorts[driver.DATASETS[1]][-1]["options"][3] = "(d) " + "z" * 30000
        plan = driver.plan_cohorts(cohorts, self.exposed, strict_census=False)
        data = plan["datasets"][driver.DATASETS[1]]
        census = data["capacity_census"]
        self.assertGreater(data["settings"]["max_context_chars"], 48000)
        self.assertEqual(data["settings"]["max_context_chars"], census["maximum_observed_empty_context_characters"] + 18000)
        self.assertEqual(data["settings"]["max_response_tokens"], 64)
        self.assertEqual(plan["datasets"][driver.DATASETS[0]]["settings"]["max_response_tokens"], 128)

    def test_execution_round_cap_is_finite_and_each_batch_bounded(self):
        calls, receipts = [], []
        def generate(path, *, max_calls):
            calls.append(max_calls)
            return state(calls=max_calls)
        receipt = driver._drive_shard("fixture", Path("unused"), max_rounds=10, max_calls=60,
            status_function=lambda path: state(), generate_function=generate, emit=receipts.append)
        self.assertEqual(receipt["stop_reason"], "finite_round_cap")
        self.assertEqual(calls, [60] * 10)
        self.assertEqual(len(receipts), 10)
        self.assertEqual(receipts[-1]["round"], 10)

    def test_inflight_is_not_recovered_or_retried(self):
        calls = []
        receipt = driver._drive_shard("fixture", Path("unused"), max_rounds=10, max_calls=60,
            status_function=lambda path: state(inflight=1), generate_function=lambda *a, **k: calls.append(1), emit=lambda value: None)
        self.assertEqual(receipt["stop_reason"], "unresolved_inflight")
        self.assertEqual(calls, [])

    def test_success_and_no_progress_stop_without_extra_calls(self):
        calls = []
        def generate(path, *, max_calls):
            calls.append(1)
            return state(complete=True)
        receipt = driver._drive_shard("fixture", Path("unused"), max_rounds=10, max_calls=60,
            status_function=lambda path: state(), generate_function=generate, emit=lambda value: None)
        self.assertEqual(receipt["stop_reason"], "complete")
        self.assertEqual(len(calls), 1)
        receipt = driver._drive_shard("fixture", Path("unused"), max_rounds=10, max_calls=60,
            status_function=lambda path: state(), generate_function=lambda *a, **k: state(calls=0), emit=lambda value: None)
        self.assertEqual(receipt["stop_reason"], "no_progress")

    def test_secondary_group_mean_uses_original_groups_not_shard_weights(self):
        rows = [{"arm": "raw_history_rag", "group_id": group, "score": value, "available": True,
                 "supplemental": {"exact_alias_or_option_score": value, "strict_committed_score": value}}
                for group, value in [("large", 1), ("large", 1), ("large", 1), ("small", 0)]]
        metrics = driver.supplemental_group_summary(rows)["raw_history_rag"]
        self.assertEqual(metrics["strict_committed_accuracy"], .75)
        self.assertEqual(metrics["strict_committed_accuracy_group_equal"], .5)

    def test_costs_sum_actual_shard_attempts_without_global_dedup(self):
        summaries = [{"cost": {"recorded_provider_attempt_intents": 2, "successful_unique_provider_calls": 2,
                                  "reported_tokens_including_failed_attempts_when_supplied": {"total_tokens": 20}}}] * 2
        result = driver._cost_summary(summaries)
        self.assertEqual(result["recorded_provider_attempt_intents"], 4)
        self.assertEqual(result["reported_tokens_including_failed_attempts_when_supplied"]["total_tokens"], 40)
        self.assertFalse(result["cross_shard_deduplication"])
        self.assertIsNone(result["monetary_cost"])


if __name__ == "__main__":
    unittest.main()
