"""Synthetic contract cases, without participants, models, or outcome claims."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation/dialogue-benefit"))
import v2_followup_contract as contract


def case(cid="A", *, split="development", sources=None, mothers=None,
         parents=None, exposed=False, reasons=None):
    result = {
        "case_id": cid, "split": split,
        "source_group_ids": ["source-" + cid] if sources is None else sources,
        "mother_ids": ["mother-" + cid] if mothers is None else mothers,
        "parent_case_ids": [] if parents is None else parents,
        "exposed": exposed,
    }
    if reasons is not None:
        result["exposure_reasons"] = reasons
    return result


def manifest(*, blind=False):
    return contract.freeze_materials(
        [case(split="blind_test" if blind else "development")],
        exposure_ledger=[], exposure_ledger_complete=True,
    )


def gates():
    # Deliberately small fictional values exercise gates, not study thresholds.
    return {
        "meaningful_improvement": {
            "minimum_absolute_difference": .1,
            "justification": "Explicit synthetic fixture only; not a clinical or adoption recommendation.",
        },
        "cluster_interval": {
            "method": "cluster_bootstrap", "unit": "connected_source_group",
            "level": .95, "replicates": 100, "seed": 17,
            "favorable_bound_rule": "lower_bound_above_zero",
        },
        "critical_errors": {"deleted_content_return": 0, "unsupported_current_state": 0},
        "resources": {
            "maximum_provider_attempts": 20, "maximum_total_tokens": 1000,
            "maximum_wall_seconds": 60,
            "include_failed_attempts": True, "include_preparation": True,
            "stage_coverage": sorted(contract.RESOURCE_STAGES),
        },
    }


def shared_policy(view):
    return view


def alternate_policy(view):
    return {"view": view}


def freeze(value, materials):
    actual = {arm: shared_policy for arm in contract.ARMS}
    if value["clarification_policy_hashes"]["managed_memory_rag"] == contract.policy_identity(alternate_policy):
        actual["managed_memory_rag"] = alternate_policy
    return contract.freeze_protocol(value, materials, clarification_policies=actual)


def config(*, confirmatory=False, human=False):
    value = {
        "protocol_id": "synthetic-contract-fixture",
        "study_type": "confirmatory" if confirmatory else "development",
        "analysis_kind": "human_parallel" if human else "offline_paired",
        "memory_arms": ["short_term", "raw_history_rag", "managed_memory_rag"],
        "knowledge_condition": "K0", "knowledge_manifest_sha256": None,
        "clarification_policy_hashes": {arm: contract.policy_identity(shared_policy) for arm in contract.ARMS},
        "managed_only_ux": [], "diagnostic_ablations": [],
        "frozen_before_generation": True, "already_generated_answers": 0,
        "adoption_gates": gates() if confirmatory else None,
    }
    if human:
        value["human_design"] = {
            "allocation": "parallel_randomized", "independent_unit": "participant",
            "same_arm_all_visits": True,
            "co_primary_endpoints": ["task_success_fraction", "restricted_elapsed_seconds"],
            "co_primary_ci_level": .975,
        }
        if confirmatory:
            frozen = value["adoption_gates"]
            frozen["cluster_interval"]["unit"] = "participant"
            frozen["cluster_interval"]["level"] = .975
            del frozen["cluster_interval"]["favorable_bound_rule"]
            frozen["cluster_interval"]["favorable_bound_rules"] = {
                "task_success_fraction": "lower_bound_above_zero",
                "restricted_elapsed_seconds": "upper_bound_below_zero",
            }
            frozen["meaningful_improvement"] = {
                "task_success_fraction": {"minimum_absolute_difference": .1, "justification": "Fictional fixture."},
                "restricted_elapsed_seconds": {"minimum_absolute_difference": 5, "justification": "Fictional fixture."},
            }
    return value


class MaterialGroupingTests(unittest.TestCase):
    def test_transitive_mothers_derivatives_and_shared_sources_form_one_group(self):
        rows = [case("A", exposed=True), case("B", mothers=["mother-A"]),
                case("C", parents=["B"]), case("D", sources=["source-C"])]
        result = contract.freeze_materials(rows)
        self.assertEqual(len(result["groups"]), 1)
        self.assertEqual(result["groups"][0]["case_ids"], ["A", "B", "C", "D"])
        self.assertTrue(result["groups"][0]["exposed"])
        self.assertEqual(result["independent_unit"], "connected_source_group")

    def test_normalized_manifest_is_independent_of_input_order(self):
        a = case("A", sources=["shared", "first"])
        b = case("B", parents=["A"])
        forward = contract.freeze_materials([a, b])
        reverse = contract.freeze_materials([b, a])
        self.assertEqual(forward, reverse)

    def test_source_and_indirect_mother_overlap_cannot_cross_splits(self):
        for shared in ("source", "mother", "parent"):
            a = case("A")
            b = case("B", split="blind_test")
            if shared == "source":
                b["source_group_ids"] = a["source_group_ids"]
            elif shared == "mother":
                b["mother_ids"] = a["mother_ids"]
            else:
                b["parent_case_ids"] = ["A"]
            with self.subTest(shared=shared), self.assertRaisesRegex(ValueError, "crosses splits"):
                contract.freeze_materials([a, b], exposure_ledger=[], exposure_ledger_complete=True)

    def test_exposure_propagates_from_historical_mother_and_ledger_bridges(self):
        ledger = [{"case_ids": [], "source_group_ids": [], "mother_ids": ["mother-A", "mother-B"],
                   "reason": "Earlier derivative responses were inspected."}]
        merged = contract.freeze_materials([case("A"), case("B")], exposure_ledger=ledger,
                                           exposure_ledger_complete=True)
        self.assertEqual(len(merged["groups"]), 1)
        self.assertTrue(merged["groups"][0]["exposed"])
        with self.assertRaisesRegex(ValueError, "Blind test"):
            contract.freeze_materials([case("A", split="blind_test")], exposure_ledger=ledger,
                                      exposure_ledger_complete=True)
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            contract.freeze_materials([case("A"), case("B", split="blind_test")], exposure_ledger=ledger,
                                      exposure_ledger_complete=True)

    def test_blind_test_requires_complete_inventory_and_exposure_ledger(self):
        with self.assertRaisesRegex(ValueError, "complete exposure ledger"):
            contract.freeze_materials([case(split="blind_test")])
        with self.assertRaisesRegex(ValueError, "Uninventoried parent"):
            contract.freeze_materials([case(parents=["missing"])])
        with self.assertRaisesRegex(ValueError, "cycle"):
            contract.freeze_materials([case("A", parents=["B"]), case("B", parents=["A"])])

    def test_cannot_hide_exposure_or_coerce_ids_and_flags(self):
        mutations = [case(exposed=False, reasons=["Already inspected."]),
                     {**case(), "case_id": 1}, {**case(), "exposed": 0},
                     {**case(), "source_group_ids": ["same", "same"]},
                     {**case(), "source_text": "Forbidden nonwhitelisted generation text."}]
        for row in mutations:
            with self.subTest(row=row), self.assertRaises(ValueError):
                contract.freeze_materials([row])


class ProtocolFreezeTests(unittest.TestCase):
    def test_common_policy_is_memory_comparison_even_when_check_counts_differ(self):
        frozen = freeze(config(), manifest())
        self.assertEqual(frozen["estimand"], "memory_management")
        self.assertTrue(frozen["shared_clarification_function"])
        self.assertFalse(frozen["displayed_check_counts_must_match"])
        self.assertEqual(frozen["provider_calls_by_this_module"], 0)

    def test_actual_policy_hash_and_managed_only_ux_change_estimand(self):
        first = contract.policy_identity(case)
        self.assertEqual(first, contract.policy_identity(case))
        self.assertNotEqual(first, contract.policy_identity(config))
        for change in ("policy", "ux"):
            value = config()
            if change == "policy":
                value["clarification_policy_hashes"]["managed_memory_rag"] = contract.policy_identity(alternate_policy)
            else:
                value["managed_only_ux"] = ["new_followup_panel"]
            frozen = freeze(value, manifest())
            self.assertEqual(frozen["estimand"], "package_effect")

    def test_at_most_two_diagnostic_ablations_and_one_knowledge_condition(self):
        value = config()
        value["diagnostic_ablations"] = [{"id": str(i), "mechanism": "Fixture."} for i in range(3)]
        with self.assertRaisesRegex(ValueError, "At most two"):
            freeze(value, manifest())
        value["diagnostic_ablations"] = value["diagnostic_ablations"][:2]
        freeze(value, manifest())
        value["knowledge_condition"] = ["K0", "K1"]
        with self.assertRaises(ValueError):
            freeze(value, manifest())
        value = config(); value["condition_cells"] = list(range(6))
        with self.assertRaises(ValueError):
            freeze(value, manifest())

    def test_confirmatory_requires_gates_before_answers_and_blind_lineage(self):
        value = config(confirmatory=True)
        freeze(value, manifest(blind=True))
        with self.assertRaisesRegex(ValueError, "blind groups"):
            freeze(value, manifest())
        for missing in gates():
            bad = deepcopy(value)
            del bad["adoption_gates"][missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                freeze(bad, manifest(blind=True))
        value["already_generated_answers"] = 1
        with self.assertRaisesRegex(ValueError, "retrospectively"):
            freeze(value, manifest(blind=True))

    def test_resigned_fake_group_manifest_is_recomputed_and_rejected(self):
        source = manifest()
        source["groups"][0]["case_ids"] = ["made-up"]
        source["materials_sha256"] = contract.canonical_hash({k: v for k, v in source.items() if k != "materials_sha256"})
        with self.assertRaisesRegex(ValueError, "grouping differs"):
            freeze(config(), source)

    def test_human_parallel_participant_co_primary_contract_is_preserved(self):
        frozen = freeze(config(human=True), manifest())
        self.assertEqual(frozen["independent_unit"], "participant")
        self.assertEqual(frozen["human_design"]["co_primary_ci_level"], .975)
        bad = config(human=True)
        bad["human_design"]["same_arm_all_visits"] = False
        with self.assertRaises(ValueError):
            freeze(bad, manifest())
        bad = config(human=True)
        bad["human_design"]["co_primary_ci_level"] = .95
        with self.assertRaises(ValueError):
            freeze(bad, manifest())
        bad = config(human=True, confirmatory=True)
        bad["adoption_gates"]["cluster_interval"]["unit"] = "question"
        with self.assertRaisesRegex(ValueError, "analysis unit"):
            freeze(bad, manifest())
        bad = config(human=True, confirmatory=True)
        del bad["adoption_gates"]["meaningful_improvement"]["restricted_elapsed_seconds"]
        with self.assertRaises(ValueError):
            freeze(bad, manifest())
        proper = freeze(config(human=True, confirmatory=True), manifest())
        self.assertEqual(proper["adoption_gates"]["cluster_interval"]["favorable_bound_rules"]["restricted_elapsed_seconds"],
                         "upper_bound_below_zero")
        bad = config(human=True, confirmatory=True)
        bad["adoption_gates"]["cluster_interval"]["favorable_bound_rules"]["restricted_elapsed_seconds"] = "lower_bound_above_zero"
        with self.assertRaisesRegex(ValueError, "lower time"):
            freeze(bad, manifest())


class ObservationSemanticsTests(unittest.TestCase):
    def test_unknown_truth_is_a_valid_target_not_missing_reference(self):
        correct_unknown = contract.score_observation(truth_status="unknown", execution_status="success",
                                                     review_status="complete", is_correct=True)
        self.assertEqual(correct_unknown["operational_score"], 1)
        self.assertEqual(correct_unknown["cause"], "unresolved")
        with self.assertRaises(ValueError):
            contract.score_observation(truth_status="not_annotated", execution_status="success",
                                       review_status="complete", is_correct=True)

    def test_technical_failure_scores_zero_without_semantic_verdict(self):
        failed = contract.score_observation(truth_status="unknown", execution_status="technical_failure",
                                            review_status="not_required")
        self.assertEqual(failed["operational_score"], 0)
        self.assertIsNone(failed["semantic_correctness"])
        self.assertEqual(failed["cause"], "unresolved")
        with self.assertRaises(ValueError):
            contract.score_observation(truth_status="known", execution_status="technical_failure",
                                       review_status="complete", is_correct=False)

    def test_pending_rating_stays_null_and_blocks_comparative_intervals(self):
        pending = contract.score_observation(truth_status="known", execution_status="success",
                                             review_status="pending_review")
        failure = contract.score_observation(truth_status="known", execution_status="timeout",
                                             review_status="not_required")
        summary = contract.summarize_observations([pending, failure])
        self.assertEqual(summary["planned_rows"], 2)
        self.assertEqual(summary["technical_or_missing_failures"], 1)
        self.assertIsNone(summary["operational_fraction"])
        self.assertFalse(summary["comparative_intervals_permitted"])
        success = contract.score_observation(truth_status="known", execution_status="success",
                                             review_status="complete", is_correct=True)
        self.assertEqual(contract.summarize_observations([success, failure])["operational_fraction"], .5)
        pending["operational_score"] = 0
        with self.assertRaises(ValueError):
            contract.summarize_observations([pending, failure])

    def test_cause_needs_independent_evidence_not_answer_string_match(self):
        kwargs = {"truth_status": "known", "execution_status": "success",
                  "review_status": "complete", "is_correct": False, "cause": "retrieval"}
        with self.assertRaises(ValueError):
            contract.score_observation(**kwargs)
        with self.assertRaises(ValueError):
            contract.score_observation(**kwargs, cause_evidence_ids=["source-1"])
        result = contract.score_observation(**kwargs, cause_evidence_ids=["source-1"], independent_cause_review=True)
        self.assertEqual(result["cause"], "retrieval")


class AdoptionGateTests(unittest.TestCase):
    def setUp(self):
        self.protocol = freeze(config(confirmatory=True), manifest(blind=True))
        self.errors = {"deleted_content_return": 0, "unsupported_current_state": 0}
        self.resources = {"provider_attempts": 10, "total_tokens": 500, "wall_seconds": 20}

    def assess(self, **overrides):
        args = {"effect": .2, "interval": [.05, .3], "critical_errors": self.errors,
                "resources": self.resources}
        args.update(overrides)
        return contract.assess_offline_adoption(self.protocol, **args)

    def test_tiny_positive_effect_and_interval_crossing_do_not_establish_adoption(self):
        self.assertEqual(self.assess()["status"], "prespecified_offline_criteria_met")
        small = self.assess(effect=.02, interval=[.01, .04])
        self.assertIn("effect_below_frozen_meaningful_difference", small["reasons"])
        crossed = self.assess(interval=[-.01, .3])
        self.assertIn("group_interval_not_favorably_above_zero", crossed["reasons"])
        self.assertFalse(crossed["clinical_efficacy_claim"])

    def test_errors_resource_limits_and_unknown_usage_cannot_be_hidden(self):
        error = self.assess(critical_errors={**self.errors, "deleted_content_return": 1})
        self.assertIn("critical_error_limit:deleted_content_return", error["reasons"])
        costly = self.assess(resources={**self.resources, "provider_attempts": 21})
        self.assertIn("resource_limit:provider_attempts", costly["reasons"])
        unknown = self.assess(resources={**self.resources, "total_tokens": None})
        self.assertEqual(unknown["status"], "unresolved")
        self.assertIn("resource_usage:total_tokens", unknown["reasons"])
        self.assertEqual(self.assess(pending_reviews=1)["status"], "unresolved")

    def test_modified_frozen_gates_and_human_source_group_reuse_are_rejected(self):
        self.protocol["adoption_gates"]["meaningful_improvement"]["minimum_absolute_difference"] = .001
        with self.assertRaisesRegex(ValueError, "identity differs"):
            self.assess()
        human = freeze(config(human=True), manifest())
        with self.assertRaisesRegex(ValueError, "participant"):
            contract.assess_offline_adoption(human, effect=.2, interval=[.05, .3],
                                            critical_errors=self.errors, resources=self.resources)


if __name__ == "__main__":
    unittest.main()
