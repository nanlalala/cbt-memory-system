"""Tiny invented fixtures; no released dataset, Secret or model calls."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "evaluation/cbt-memory-effect"
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location("run_cactus_effect", HERE / "run_cactus_effect.py")
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def cases():
    rows = M.materials.invented_fixture_rows()
    for family, row in enumerate(rows):
        turns = []
        for index in range(7):
            turns.extend([f"Counselor: FICTIONAL research question {index}.",
                f"Client: Topic{family}_{index} I clearly remember this invented experience about practice, thoughts, "
                "and a small example that is original to this fictional source."])
        turns.append("Counselor: FICTIONAL held-out reply.")
        row["dialogue"] = "\n".join(turns)
    return M.materials.compile_rows(rows, family_limit=2, fixture_only=True)["generation"]


class ProbeTests(unittest.TestCase):
    def test_frozen_dependencies_include_nested_schema_and_provider_contracts(self):
        frozen = M.code_hashes()
        for dependency in ("app/colab_provider.py",
                "modules/long-term-memory/schema/memory_record_v2.schema.json",
                "modules/long-term-memory/schema/memory_record.schema.json",
                "modules/cbt-knowledge-rag/source_registry.json",
                "evaluation/dialogue-benefit/followup_transport.py",
                "evaluation/end-to-end-dialogue/run_three_arm.py"):
            self.assertEqual(frozen[dependency], M.file_sha(ROOT / dependency))

    def test_fixed_positions_have_undisclosed_suffix_and_no_assessor_fields(self):
        generation, gold, inventory = M.choose_probes(cases())
        self.assertEqual((len(generation), len(gold), len(inventory)), (6, 6, 6))
        self.assertEqual({r["position"] for r in inventory}, {"early", "middle", "recent"})
        targets = {row["case_id"]: row for row in gold}
        for case in generation:
            target = targets[case["case_id"]]
            self.assertNotIn(target["target_quote"], case["input"])
            self.assertNotIn(target["undisclosed_suffix"], case["input"])
            self.assertGreaterEqual(len(target["undisclosed_suffix"]), 24)
            self.assertGreaterEqual(len(target["undisclosed_suffix"].split()), 5)
            self.assertTrue({"target_quote", "target_message_id", "expected_current_state", "gold"}.isdisjoint(case))

    def test_family_source_namespace_is_stable_without_combining_people(self):
        generation, _, _ = M.choose_probes(cases())
        by_family = {}
        for case in generation:
            by_family.setdefault(case["source_family_id"], set()).add(case["replay_identity"])
        self.assertEqual(len(by_family), 2)
        self.assertTrue(all(len(values) == 1 for values in by_family.values()))
        self.assertEqual(len(set.union(*by_family.values())), 2)

    def test_distance_probes_share_identical_latest_history(self):
        generation, gold, _ = M.choose_probes(cases())
        grouped = {}
        for case in generation:
            grouped.setdefault(case["source_family_id"], []).append(case)
        targets = {r["case_id"]: r for r in gold}
        for family in grouped.values():
            self.assertEqual(len({M.stable(c["history"]) for c in family}), 1)
            self.assertEqual(len({c["history_cutoff_order_exclusive"] for c in family}), 1)
            self.assertEqual(len({targets[c["case_id"]]["target_message_id"] for c in family}), 3)
            self.assertEqual(len({c["input_message_id"] for c in family}), 3)

    def test_short_target_retains_denominator_and_is_not_replaced(self):
        values = cases()
        for case in values:
            first = next(turn for turn in case["history"] if turn["role"] == "user")
            first["content"] = "Only the cue exists."
        generation, gold, inventory = M.choose_probes(values)
        bad = [row for row in inventory if row["preparation_status"] != "ready"]
        self.assertEqual(len(inventory), 6)
        self.assertEqual(len(bad), 2)
        self.assertTrue(all(row["case_id"] not in {c["case_id"] for c in generation} for row in bad))
        self.assertEqual(len(gold), 4)

    def test_privacy_target_never_contributes_cue_or_gold(self):
        values = cases()
        for case in values:
            first = next(turn for turn in case["history"] if turn["role"] == "user")
            first["content"] = "Don't save this. FICTIONAL_PRIVATE_TARGET_NOT_TO_BE_SENT."
        generation, gold, inventory = M.choose_probes(values)
        withheld = {row["case_id"] for row in inventory if row["preparation_status"] == "privacy_withheld"}
        self.assertEqual(len(withheld), 2)
        self.assertTrue(withheld.isdisjoint(c["case_id"] for c in generation))
        self.assertNotIn("FICTIONAL_PRIVATE_TARGET_NOT_TO_BE_SENT", json.dumps(gold))

    def test_literal_scoring_requires_full_quote_and_actual_source_id(self):
        _, targets, _ = M.choose_probes(cases()); target = targets[0]
        good = {"quote": target["target_quote"], "source_id": "actual-source-id", "unknown": False}
        self.assertTrue(M.score_quote(json.dumps(good), target, "actual-source-id")["literal_success"])
        bad = deepcopy(good); bad["quote"] = target["disclosed_cue"]
        self.assertEqual(M.score_quote(json.dumps(bad), target, "actual-source-id")["status"], "wrong_quote_or_source_binding")
        bad = deepcopy(good); bad["source_id"] = "original-turn-id-not-provider-id"
        self.assertFalse(M.score_quote(json.dumps(bad), target, "actual-source-id")["literal_success"])
        self.assertEqual(M.score_quote(json.dumps(bad), target, "actual-source-id")["error_subtype"], "source_binding_only")
        self.assertEqual(M.score_quote('{"quote":"","source_id":"","unknown":true}', target, "s")["status"], "explicit_unknown")

    def test_duplicate_json_and_unknown_with_claim_are_technical_failures(self):
        target = {"target_quote": "FICTIONAL original quote"}
        self.assertEqual(M.score_quote('{"quote":"x","quote":"y","source_id":"s","unknown":false}', target, "s")["status"], "technical_failure")
        self.assertEqual(M.score_quote('{"quote":"invented","source_id":"s","unknown":true}', target, "s")["status"], "technical_failure")


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.protocol = {"protocol_sha256": "fictional-test-protocol", "maximum_provider_attempts": 5000}
        self.calls = []

    def fake(self, messages, purpose):
        self.calls.append((messages, purpose))
        return {"hypothesis": "FICTIONAL response", "provider_model": M.ALIAS,
                "finish_reason": "stop", "usage": {"total_tokens": 17}, "latency_seconds": .01}

    def test_exact_requests_share_one_costed_call_and_missing_usage_is_unknown(self):
        journal = M.CallJournal(self.path, self.protocol, max_new_calls=2)
        first, shared = journal.invoke([{"role": "user", "content": "FICTIONAL input"}], "extraction", self.fake)
        second, shared2 = journal.invoke([{"role": "user", "content": "FICTIONAL input"}], "extraction", self.fake)
        self.assertFalse(shared); self.assertTrue(shared2); self.assertEqual(first, second)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(journal.summary()["tokens_reported"], {"prompt_tokens": None, "completion_tokens": None, "total_tokens": 17})
        restored = M.CallJournal(self.path, self.protocol, max_new_calls=0)
        _, reused = restored.invoke([{"role": "user", "content": "FICTIONAL input"}], "extraction", self.fake)
        self.assertTrue(reused); self.assertEqual(len(self.calls), 1)

    def test_unknown_delivery_blocks_resume_and_new_requests(self):
        def lost(messages, purpose):
            self.calls.append(True)
            raise TimeoutError("FICTIONAL dropped result")
        journal = M.CallJournal(self.path, self.protocol, max_new_calls=2)
        with self.assertRaises(M.UnknownSend):
            journal.invoke([{"role": "user", "content": "FICTIONAL input"}], "response", lost)
        with self.assertRaises(M.UnknownSend):
            M.CallJournal(self.path, self.protocol, max_new_calls=2)
        audit = M.CallJournal(self.path, self.protocol, max_new_calls=0, allow_unknown=True)
        self.assertEqual(audit.summary()["unknown_sends"], 1)
        self.assertIsNone(audit.summary()["tokens_reported"]["total_tokens"])
        self.assertEqual(len(self.calls), 1)

    def test_batch_cap_is_checked_before_another_intent(self):
        journal = M.CallJournal(self.path, self.protocol, max_new_calls=1)
        journal.invoke([{"role": "user", "content": "FICTIONAL first"}], "response", self.fake)
        with self.assertRaises(M.CallLimit):
            journal.invoke([{"role": "user", "content": "FICTIONAL second"}], "response", self.fake)
        self.assertEqual(len(M.read_rows(self.path / "call_intents.jsonl")), 1)

    def test_failed_known_response_preserves_usage_and_is_not_retried(self):
        class KnownFailure(Exception):
            failure_kind = "provider_http_error"
            reported_usage = {"prompt_tokens": 12}
            latency_seconds = .1
        def rejected(messages, purpose):
            self.calls.append(True); raise KnownFailure()
        journal = M.CallJournal(self.path, self.protocol, max_new_calls=2)
        with self.assertRaises(M.UnknownSend):
            journal.invoke([{"role": "user", "content": "FICTIONAL input"}], "response", rejected)
        restored = M.CallJournal(self.path, self.protocol, max_new_calls=2)
        with self.assertRaises(M.UnknownSend):
            restored.invoke([{"role": "user", "content": "FICTIONAL input"}], "response", self.fake)
        self.assertEqual(restored.summary()["tokens_reported"]["prompt_tokens"], 12)
        self.assertEqual(len(self.calls), 1)

    def test_incomplete_tail_and_foreign_model_are_rejected(self):
        journal = M.CallJournal(self.path, self.protocol, max_new_calls=1)
        journal.invoke([{"role": "user", "content": "FICTIONAL input"}], "response", self.fake)
        path = self.path / "call_outcomes.jsonl"
        records = M.read_rows(path); record = records[0]
        record["provider_model"] = "other-model"
        record = M.signed({k: v for k, v in record.items() if k != "record_sha256"})
        path.write_text(json.dumps(record) + "\n")
        with self.assertRaisesRegex(ValueError, "identity"):
            M.CallJournal(self.path, self.protocol, max_new_calls=1)
        path.write_text('{"unfinished":')
        with self.assertRaisesRegex(ValueError, "tail"):
            M.CallJournal(self.path, self.protocol, max_new_calls=1)


class AnalysisBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        (self.path / "generation").mkdir(); (self.path / "assessor_only").mkdir(); (self.path / "cases").mkdir()
        generation, gold, inventory = M.choose_probes(cases())
        self.generation, self.gold = generation, gold
        for relative, values in (("generation/cases.jsonl", generation), ("assessor_only/targets.jsonl", gold),
                                 ("inventory.jsonl", inventory)):
            M.atomic(self.path / relative, values, jsonl=True)
        self.protocol = {"version": M.VERSION, "fixture_only": True, "git_head": M.revision(),
            "settings": M.settings(), "code_sha256": M.code_hashes(), "maximum_provider_attempts": 5000,
            "maximum_workers": 1, "arms": list(M.ARMS), "max_outbound_characters": M.ANSWER_CHARS,
            "max_extraction_outbound_characters": M.EXTRACTION_CHARS, "clock_mode": "ordinal-only-v1",
            "planned_families": 2, "ready_probes": 6, "research_decision_rule": M.decision_rule(),
            "files_sha256": {relative: M.file_sha(self.path / relative) for relative in (
                "generation/cases.jsonl", "assessor_only/targets.jsonl", "inventory.jsonl")}}
        self.protocol["protocol_sha256"] = M.stable(self.protocol)
        M.atomic(self.path / "protocol.json", self.protocol)

    def test_pending_has_null_measured_accuracy_and_full_grid(self):
        summary = M.analyze(self.path)
        self.assertEqual(summary["planned_arm_rows"], 18)
        self.assertFalse(summary["effect_complete"])
        self.assertTrue(summary["fixture_only"])
        for values in summary["by_arm"].values():
            self.assertEqual(values["counts"]["pending"], 6)
            self.assertIsNone(values["measured_ready_recall_rate"])
            self.assertEqual(values["completion_adjusted_allplanned_recall_lower_bound"], 0)

    def test_recomputed_checksum_does_not_allow_provider_policy_change(self):
        modified = deepcopy(self.protocol); modified["settings"]["model"] = "different-model"
        modified["protocol_sha256"] = M.stable({k: v for k, v in modified.items() if k != "protocol_sha256"})
        (self.path / "protocol.json").write_text(json.dumps(modified))
        with self.assertRaisesRegex(ValueError, "protocol changed"):
            M.load_protocol(self.path)

    def test_recomputed_checksum_does_not_change_prospective_benefit_threshold(self):
        modified = deepcopy(self.protocol)
        modified["research_decision_rule"]["minimum_absolute_recall_improvement"] = 0
        modified["protocol_sha256"] = M.stable({k: v for k, v in modified.items() if k != "protocol_sha256"})
        (self.path / "protocol.json").write_text(json.dumps(modified))
        with self.assertRaisesRegex(ValueError, "protocol changed"):
            M.load_protocol(self.path)

    def test_score_is_recomputed_from_original_response_and_actual_source(self):
        case, target = self.generation[0], self.gold[0]
        directory = self.path / "cases" / case["case_id"]; directory.mkdir()
        replay_case = deepcopy(case); replay_case["case_id"] = replay_case.pop("replay_identity")
        replay_case.pop("case_sha256")
        prepared = M.replay.prepare_contexts(replay_case, database=directory / "memory.sqlite", approved_root=directory,
            permissions=M.PERMISSIONS, config=M.replay.ReplayConfig(clock_mode="ordinal-only-v1", max_outbound_chars=6000),
            request_metadata={"model": M.MODEL, "temperature": 0, "thinking": {"type": "disabled"}, "max_tokens": 1024}, fixture_only=True)
        M.atomic(directory / "prepared_contexts.json", {"protocol_sha256": self.protocol["protocol_sha256"],
            "study_case_sha256": case["case_sha256"], "prepared": prepared})
        sid = M._target_source_id(prepared, target)
        body = json.dumps({"quote": target["target_quote"], "source_id": sid, "unknown": False})
        def fake(messages, purpose):
            return {"hypothesis": body, "provider_model": M.ALIAS, "finish_reason": "stop", "usage": {}}
        journal = M.CallJournal(self.path, self.protocol, max_new_calls=1)
        context = prepared["contexts"]["raw_history_rag"]
        receipt, _ = journal.invoke(context["messages"], "response", fake)
        answer = {"protocol_sha256": self.protocol["protocol_sha256"], "case_id": case["case_id"],
            "arm": "raw_history_rag", "request_sha256": context["request_sha256"],
            "provider_request_sha256": receipt["request_sha256"], "original_response": body,
            "response_sha256": M.text_sha(body), "score": M.score_quote(body, target, sid)}
        M.append(self.path / "answers.jsonl", M.signed(answer))
        self.assertEqual(M.analyze(self.path)["by_arm"]["raw_history_rag"]["counts"]["correct"], 1)
        answer["score"]["literal_success"] = False
        (self.path / "answers.jsonl").write_text(json.dumps(M.signed(answer)) + "\n")
        with self.assertRaisesRegex(ValueError, "contradicts"):
            M.analyze(self.path)


if __name__ == "__main__":
    unittest.main()
