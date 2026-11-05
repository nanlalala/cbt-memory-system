"""Tiny invented cases exercise the offline diagnostic's source-only seam."""
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
import os
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation/objective-memory"))
import v3_run_objective_comparison as runner
import v3_source_retrieval as old_policy
import v4_source_retrieval as new_policy
from compare_v4_retrieval_colab import channel_snapshot, source_histories, compare
from compare_v4_retrieval_colab import validate_shard_binding, validated_old_score_rows
import score_objective_comparison as scorer


def case():
    text = ["Orion is a citizen of Aurelia.", "The capital of Aurelia is Nova.",
            "A neutral recent observation."]
    history = [{"session_id": "fixture", "turn_id": str(i), "order": i,
                "role": "context", "text": value} for i, value in enumerate(text)]
    return {"case_id": "fictional-q", "group_id": "fictional-family", "subject_id": "fictional-family",
            "dataset_key": "memoryagentbench_conflict", "task": "fictional-mh",
            "query": "Orion citizen " + " ".join("extra" + str(i) for i in range(35)),
            "options": [], "events": [], "as_of": None, "history": history}


class OfflineRetrievalDiagnosticTests(unittest.TestCase):
    def test_same_budget_replay_preserves_old_prompt_but_candidate_can_change_sources(self):
        current = case()
        docs = [runner._document(current, source) for source in current["history"]]
        settings = runner.Settings(model="fixture", short_turns=1, minimum_relevance=.05)
        args = (current, "raw_history_rag", docs, docs, settings)
        first = channel_snapshot(*args, runner.HistoryRanker(settings), old_policy.retrieve_sources, runner)
        old = runner._prepare_prompt(current, "raw_history_rag", docs, docs, [], [], settings,
                                     runner.HistoryRanker(settings))
        self.assertEqual(first["messages"], old["messages"])
        self.assertEqual(first["prompt_sha256"], old["prompt_sha256"])
        second = channel_snapshot(*args, runner.HistoryRanker(settings), new_policy.retrieve_sources, runner)
        self.assertNotEqual(first["prompt_sha256"], second["prompt_sha256"])
        self.assertEqual(second["included_short_ids"], first["included_short_ids"])
        self.assertEqual(len(second["included_long_ids"]), 2)
        self.assertNotIn("gold", second)

    def test_history_restoration_binds_exact_source_order_bytes_and_identity(self):
        current = case()
        compact = {key: value for key, value in current.items() if key != "history"}
        compact["history_inventory_sha256"] = runner.stable_digest([current["subject_id"], current["history"]])
        inventory = [{"subject_id": current["subject_id"], "source": source,
                      "storage_decision": {"action": "use_as_history"}} for source in current["history"]]
        state = {"frozen_rows": {"cases.jsonl": [compact], "inventory.jsonl": inventory}}
        self.assertEqual(source_histories(state, runner), [current])
        altered = deepcopy(state)
        altered["frozen_rows"]["inventory.jsonl"][0]["source"]["text"] = "Replaced original source."
        with self.assertRaisesRegex(ValueError, "original admitted history"):
            source_histories(altered, runner)

    def test_mixed_or_duplicate_same_family_sources_are_not_silently_merged(self):
        current = case()
        compact = {key: value for key, value in current.items() if key != "history"}
        compact["history_inventory_sha256"] = runner.stable_digest([current["subject_id"], current["history"]])
        inventory = [{"subject_id": current["subject_id"], "source": source,
                      "storage_decision": {"action": "use_as_history"}} for source in current["history"]]
        state = {"frozen_rows": {"cases.jsonl": [compact], "inventory.jsonl": inventory + [deepcopy(inventory[0])]}}
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            source_histories(state, runner)

    def test_non_colab_rejected_before_any_input_or_git_access(self):
        if Path("/content").is_dir() and os.environ.get("COLAB_RELEASE_TAG"):
            self.skipTest("This guard check is for the non-Colab CPU test environment")
        with self.assertRaisesRegex(RuntimeError, "no experiment data was read"):
            compare(Namespace(old_repo="/does-not-exist", candidate_repo="/also-absent",
                              candidate_commit="x", suite="/absent", out="/absent-output"))

    def test_suite_binding_rejects_changed_source_settings_or_exposure(self):
        shard = {"dataset_key": "fixture", "case_ids": ["q"], "case_count": 1, "shard_id": "mab-fixture"}
        dataset = {"settings": {"top_k": 3}, "development_partition": {"development_case_ids": []}}
        plan = {"suite_plan_sha256": "plan", "datasets": {"fixture": dataset},
                "source_identities": {"fixture": {"data_revision": "source-revision"}}, "api_base": "https://example.invalid"}
        manifest = {"suite_plan_sha256": "plan", "suite_shard_id": "mab-fixture",
                    "data_revision": "source-revision", "development_partition": {"development_case_ids": []}}
        protocol = {"case_ids": ["q"], "dataset_manifest": manifest,
                    "settings": {"top_k": 3}, "api_base": "https://example.invalid"}
        validate_shard_binding(protocol, [{}, {}, {}], shard, plan, ("a", "b", "c"))
        for key, value in (("suite_shard_id", "other"), ("data_revision", "other"),
                           ("development_partition", {"development_case_ids": ["q"]})):
            changed = deepcopy(protocol)
            changed["dataset_manifest"][key] = value
            with self.assertRaises(ValueError):
                validate_shard_binding(changed, [{}, {}, {}], shard, plan, ("a", "b", "c"))
        changed = deepcopy(protocol)
        changed["settings"]["top_k"] = 6
        with self.assertRaises(ValueError):
            validate_shard_binding(changed, [{}, {}, {}], shard, plan, ("a", "b", "c"))

    def test_old_score_metadata_is_not_trusted_without_deterministic_recomputation(self):
        state = {"outcomes": {("q", "raw"): {"status": "success"}},
                 "completed": {("q", "raw"): {"hypothesis": "Nova"}},
                 "references": [{"case_id": "q", "metric": "mab_substring_exact_match", "accepted_answers": ["Nova"]}]}
        saved = [{"case_id": "q", "arm": "raw", "available": True, "completion_status": "success", "score": 1.0}]
        self.assertEqual(validated_old_score_rows(state, saved, scorer), {("q", "raw"): 1.0})
        tampered = deepcopy(saved)
        tampered[0]["score"] = 0.0
        with self.assertRaisesRegex(ValueError, "deterministic recomputation"):
            validated_old_score_rows(state, tampered, scorer)
        with self.assertRaisesRegex(ValueError, "grid"):
            validated_old_score_rows(state, saved + saved, scorer)


if __name__ == "__main__":
    unittest.main()
