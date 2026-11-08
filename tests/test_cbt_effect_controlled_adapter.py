"""Invented fixture evidence only, with no public dataset or model calls."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("controlled_replay_adapter",
    ROOT / "evaluation/cbt-memory-effect/controlled_replay_adapter.py")
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)
PERMISSIONS = {k: True for k in M.PERMISSION_KEYS}


class ControlledReplayTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def case(self):
        return {"case_id": "invented-cbt-case", "calendar_time_available": True,
            "history": [
                {"message_id": "c0", "role": "assistant", "content": "You might try a thought record.",
                 "date": "2026-11-03T08:00:00Z", "source_order": 0},
                {"message_id": "u1", "role": "user", "content": "I agreed to do thought record 2 times.",
                 "date": "2026-11-03T08:01:00Z", "source_order": 1},
                {"message_id": "c2", "role": "assistant", "content": "That is a proposed practice, not proof you completed it.",
                 "date": "2026-11-03T08:02:00Z", "source_order": 2},
                {"message_id": "u3", "role": "user", "content": "thought record 累计做了一次。",
                 "date": "2026-11-04T09:00:00Z", "source_order": 3}],
            "input": "How many thought record times did I agree to do?",
            "input_message_id": "u4", "input_date": "2026-11-05T09:00:00Z"}

    def run_case(self, case=None, **kwargs):
        return M.prepare_contexts(case or self.case(),
            database=Path(self.directory.name) / "fixture.sqlite", approved_root=self.directory.name,
            permissions=PERMISSIONS, fixture_only=True, **kwargs)

    def test_replay_preserves_literal_speakers_without_prefix_answers(self):
        result = self.run_case()
        trace = result["trace"]
        self.assertEqual(trace["actual_answer_calls"], 0)
        self.assertEqual(len(trace["source_receipts"]), 4)
        self.assertEqual([s["role"] for s in trace["source_receipts"]], ["assistant", "user", "assistant", "user"])
        self.assertEqual(len(trace["worker_results"]), 2)
        self.assertTrue(all(r["semantic_status"] == "source_only_no_model" for r in trace["worker_results"]))
        self.assertTrue(trace["same_permitted_source_pool"])
        self.assertFalse(trace["ready_for_external_generation"])
        self.assertTrue(trace["final_evidence_integrity_passed"])
        self.assertTrue(all(r["confirmation_state"] == "unconfirmed" for r in trace["candidate_records"]))

    def test_same_answer_prompt_question_metadata_and_full_request_budget(self):
        result = self.run_case(request_metadata={"model": "invented-model", "temperature": 0, "max_tokens": 1024})
        contexts = result["contexts"]
        for context in contexts.values():
            self.assertEqual(context["messages"][0], contexts["raw_history_rag"]["messages"][0])
            self.assertEqual(context["messages"][-1], {"role": "user", "content": self.case()["input"]})
            self.assertEqual(context["request_payload"]["model"], "invented-model")
            self.assertEqual(len(context["serialized_request"]), context["outbound_chars"])
            self.assertLessEqual(context["outbound_chars"], 8000)
        self.assertIn("task:", contexts["managed_memory_rag"]["serialized_request"])
        self.assertNotIn('"field": "target_count"', json.dumps(contexts["raw_history_rag"]["messages"]))

    def test_undated_cactus_is_blocked_before_database_write(self):
        case = self.case()
        case["calendar_time_available"] = False
        case.pop("input_date")
        for turn in case["history"]:
            turn["date"] = None
        with self.assertRaisesRegex(M.ReplayBlocked, "source_order_clock"):
            self.run_case(case)
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])

    def test_assistant_statement_is_not_user_acceptance_or_candidate(self):
        case = self.case()
        case["history"] = [{"message_id": "c0", "role": "assistant",
            "content": "I agreed to do thought record 9 times.",
            "date": "2026-11-03T08:00:00Z", "source_order": 0}]
        result = self.run_case(case)
        self.assertEqual(result["trace"]["candidate_records"], [])
        self.assertEqual(result["trace"]["worker_results"], [])
        self.assertNotIn('"field": "target_count"', result["contexts"]["managed_memory_rag"]["serialized_request"])

    def test_no_save_user_body_does_not_enter_context_or_worker(self):
        case = self.case()
        case["history"][1]["content"] = "Don't save this. FICTIONAL_NOSAVE_ZEBRA"
        case["input"] = "What about FICTIONAL_NOSAVE_ZEBRA?"
        result = self.run_case(case)
        receipt = result["trace"]["source_receipts"][1]
        self.assertEqual(receipt["receipt"]["source_status"], "not_saved")
        self.assertEqual(len(result["trace"]["worker_results"]), 1)
        for context in result["contexts"].values():
            self.assertNotIn("Don't save this.", context["serialized_request"])

    def test_user_risk_does_not_enter_source_or_queue(self):
        case = self.case()
        case["history"][1]["content"] = "I want to kill myself"
        with self.assertRaisesRegex(M.ReplayBlocked, "local_safety"):
            self.run_case(case)
        self.assertEqual(list(Path(self.directory.name).iterdir()), [])

    def test_gold_and_future_fields_are_rejected(self):
        for key in ("patterns", "gold", "future_response", "memory_targets"):
            case = self.case(); case[key] = "not allowed"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "gold-free"):
                self.run_case(case)

    def test_fresh_database_cannot_overwrite_an_old_replay(self):
        first = self.run_case()
        with self.assertRaisesRegex(ValueError, "fresh replay database"):
            self.run_case()
        self.assertEqual(first["trace"]["actual_answer_calls"], 0)

    def test_outbound_quote_mutation_is_rejected(self):
        m = M._modules()
        source = {"source_id": "s1", "text": "I did one thought record.", "role": "user",
            "source_time": "2026-11-03T08:00:00Z", "permissions": PERMISSIONS}
        span = m.EvidenceSpan("s1", "s1:full", source["text"], source_time=source["source_time"])
        claim = m.EvidenceClaim("c1", "history_quote", span.text, (span,), support_status="source_quote")
        package = m.build_outbound_messages([{"role": "user", "content": "thought record"}],
            [m.EvidenceBundle("b1", (claim,))], max_outbound_chars=4000)
        self.assertTrue(M.assert_evidence_integrity(package, [source]))
        mutated = deepcopy(source); mutated["role"] = "assistant"
        with self.assertRaisesRegex(ValueError, "role/time"):
            M.assert_evidence_integrity(package, [mutated])

    def test_fixture_cannot_enable_semantic_provider(self):
        with self.assertRaisesRegex(ValueError, "does not admit"):
            self.run_case(extraction_provider=lambda **kwargs: None)

    def ordinal_case(self):
        case = self.case()
        case["calendar_time_available"] = False
        case.pop("input_date")
        case["history_cutoff_order_exclusive"] = 4
        for turn in case["history"]:
            turn["date"] = None
        case["history"][1]["content"] += " On 2024-02-03 I tried another thought record."
        return case

    def ordinal(self, case=None, **kwargs):
        return self.run_case(case or self.ordinal_case(),
            config=M.ReplayConfig(clock_mode="ordinal-only-v1", **kwargs))

    def test_ordinal_wire_has_no_internal_date_but_preserves_original_date_words(self):
        result = self.ordinal()
        trace = result["trace"]
        self.assertFalse(trace["calendar_available"])
        self.assertTrue(trace["timestamps_internal_only"])
        self.assertEqual(trace["task_temporal_semantics"], "disabled")
        self.assertEqual(trace["task_object_count"], 0)
        self.assertEqual(trace["actual_answer_calls"], 0)
        self.assertTrue(trace["same_permitted_source_pool"])
        for context in result["contexts"].values():
            self.assertNotIn("2000-01-03", context["serialized_request"])
            self.assertNotIn('"source_time"', context["serialized_request"])
            self.assertNotIn('"as_of"', context["serialized_request"])
            self.assertNotIn('"field": "target_count"', context["serialized_request"])
            self.assertIn("2024-02-03", context["serialized_request"])
            self.assertLessEqual(len(context["serialized_request"]), 8000)
            for message in context["messages"]:
                if message["content"].startswith("SOURCED MEMORY DATA ONLY."):
                    for claim in json.loads(message["content"].split("\n", 1)[1])["claims"]:
                        self.assertEqual(claim["field"], "history_quote")
                        for span in claim["evidence"]:
                            self.assertIn("source_order", span)
                            self.assertNotIn("source_time", span)

    def test_ordinal_disables_task_pause_due_calendar_projections(self):
        case = self.ordinal_case()
        case["history"][3]["content"] = "I agreed to do thought record 7 times next week."
        result = self.ordinal(case)
        self.assertEqual(result["trace"]["task_object_count"], 0)
        self.assertEqual(result["trace"]["expected_task_versions"], {})
        self.assertTrue(all(v is None for v in result["trace"]["proposed_checks_not_shown"].values()))
        durable = M._open_replay_store(result, Path(self.directory.name) / "fixture.sqlite", self.directory.name)
        try:
            self.assertEqual(durable.store.connection.execute("SELECT COUNT(*) FROM cp_refusals").fetchone()[0], 0)
        finally:
            durable.close()

    def test_ordinal_requires_explicit_null_dates_and_source_order(self):
        case = self.ordinal_case(); case["history"][0].pop("source_order")
        with self.assertRaisesRegex(ValueError, "source order"):
            self.ordinal(case)
        case = self.ordinal_case(); case["history"][0]["date"] = "2026-11-01T10:00:00Z"
        with self.assertRaisesRegex(ValueError, "null date"):
            self.ordinal(case)

    def test_current_permission_and_wire_rechecked_at_send_boundary(self):
        result = self.ordinal()
        database = Path(self.directory.name) / "fixture.sqlite"
        binding = M.preflight_outbound(result, "raw_history_rag", database=database, approved_root=self.directory.name)
        self.assertEqual(binding["clock_mode"], "ordinal-only-v1")
        changed = deepcopy(result)
        changed["contexts"]["raw_history_rag"]["serialized_request"] += " "
        with self.assertRaisesRegex(ValueError, "identity changed"):
            M.preflight_outbound(changed, "raw_history_rag", database=database, approved_root=self.directory.name)
        durable = M._open_replay_store(result, database, self.directory.name)
        try:
            durable.revoke(result["trace"]["user_id"], "source", binding["source_ids"][0])
        finally:
            durable.close()
        with self.assertRaisesRegex(ValueError, "generation changed"):
            M.preflight_outbound(result, "raw_history_rag", database=database, approved_root=self.directory.name)

    def test_same_canonical_request_does_not_admit_another_inflight_send(self):
        result = self.ordinal()
        database = Path(self.directory.name) / "fixture.sqlite"
        self.assertTrue(result["trace"]["raw_managed_same_request"])
        admission = M.begin_response_attempt(result, "raw_history_rag", database=database, approved_root=self.directory.name)
        self.assertFalse(admission["actual_network_call_made"])
        with self.assertRaises(M._modules().AttemptUnknown):
            M.begin_response_attempt(result, "managed_memory_rag", database=database, approved_root=self.directory.name)
        finished = M.finish_response_attempt(result, database=database, approved_root=self.directory.name,
            attempt_id=admission["attempt_id"], outcome="success", response={"text": "invented fixture response"})
        self.assertTrue(finished["publishable"])
        self.assertFalse(finished["response_retained"])
        replay = M.begin_response_attempt(result, "managed_memory_rag", database=database, approved_root=self.directory.name)
        self.assertTrue(replay["replayed"])

    def test_withdrawal_after_admission_discards_received_body(self):
        result = self.ordinal()
        database = Path(self.directory.name) / "fixture.sqlite"
        admission = M.begin_response_attempt(result, "raw_history_rag", database=database, approved_root=self.directory.name)
        durable = M._open_replay_store(result, database, self.directory.name)
        try:
            durable.revoke(result["trace"]["user_id"], "source", result["trace"]["current_source_id"])
        finally:
            durable.close()
        finished = M.finish_response_attempt(result, database=database, approved_root=self.directory.name,
            attempt_id=admission["attempt_id"], outcome="success", response={"text": "WITHDRAWN_FIXTURE_REPLY"})
        self.assertFalse(finished["publishable"])
        self.assertIsNone(finished["response"])

    def test_response_limit_cannot_spend_unused_extraction_allowance(self):
        result = self.ordinal(short_turns=1, maximum_response_calls=1)
        database = Path(self.directory.name) / "fixture.sqlite"
        self.assertNotEqual(result["contexts"]["short_term"]["request_sha256"],
                            result["contexts"]["raw_history_rag"]["request_sha256"])
        admission = M.begin_response_attempt(result, "raw_history_rag", database=database, approved_root=self.directory.name)
        M.finish_response_attempt(result, database=database, approved_root=self.directory.name,
            attempt_id=admission["attempt_id"], outcome="success", response={"text": "invented response"})
        with self.assertRaisesRegex(M._modules().BudgetExhausted, "response call limit"):
            M.begin_response_attempt(result, "short_term", database=database, approved_root=self.directory.name)

    def test_unknown_extraction_intent_prevents_additional_background_work(self):
        result = self.ordinal()
        database = Path(self.directory.name) / "fixture.sqlite"
        source_receipt = next(r for r in result["trace"]["source_receipts"] if r["receipt"].get("job_id"))
        durable = M._open_replay_store(result, database, self.directory.name)
        try:
            durable.begin_attempt(result["trace"]["user_id"], "extract:" + source_receipt["receipt"]["job_id"],
                "extraction", [source_receipt["source_id"]], run_id=result["trace"]["run_id"],
                session_id=result["trace"]["session_id"])
            with self.assertRaisesRegex(M.ReplayBlocked, "unknown_no_additional_calls"):
                M._require_settled_extraction(durable, result["trace"]["user_id"], source_receipt["receipt"]["job_id"])
        finally:
            durable.close()

    def test_ranked_original_quote_survives_tight_budget_despite_later_source_id(self):
        m = M._modules()
        def hit(sid, text):
            return SimpleNamespace(source_id=sid,span_id=sid+":full",text=text,start=0,end=len(text),
                source_time="2026-11-03T08:00:00Z",role="user")
        high = hit("z-high-relevance", "TARGET " + "evidence "*20)
        low = hit("a-low-relevance", "OTHER " + "material "*20)
        bundles = M._quote_bundles([high,low])
        base = [{"role":"user","content":"TARGET recall"}]
        cap = m.build_outbound_messages(base,[bundles[0]],max_outbound_chars=10000).char_count
        packed = m.build_outbound_messages(base,bundles,max_outbound_chars=cap)
        self.assertEqual(packed.included_claim_ids,("history:z-high-relevance:full",))
        self.assertIn("history:a-low-relevance:full",packed.dropped_claims)

    def test_semantic_supplement_cannot_displace_highest_ranked_original_quote(self):
        m = M._modules()
        def hit(sid, text):
            return SimpleNamespace(source_id=sid,span_id=sid+":full",text=text,start=0,end=len(text),
                source_time="2026-11-03T08:00:00Z",role="user")
        raw = hit("z-high-relevance","TARGET "+"evidence "*20)
        semantic = hit("a-semantic-only","OTHER "+"material "*20)
        raw_bundle = M._quote_bundles([raw])[0]
        semantic_bundle = M._quote_bundles([semantic])[0]
        semantic_bundle = replace(semantic_bundle,bundle_id="semantic:a",claims=(replace(semantic_bundle.claims[0],priority=99),))
        ranked = M._preserve_managed_quote_rank([semantic_bundle,raw_bundle],[raw])
        base=[{"role":"user","content":"TARGET recall"}]
        cap=m.build_outbound_messages(base,[raw_bundle],max_outbound_chars=10000).char_count
        packed=m.build_outbound_messages(base,ranked,max_outbound_chars=cap)
        self.assertEqual(packed.included_claim_ids,("history:z-high-relevance:full",))

    def test_ordinal_explicit_refusal_blocks_history_or_current_before_any_write(self):
        for text in ("Never ask me about thought record again.","Stop asking about walking.",
                     "Not today.","今天不谈想法记录。","以后别问这件事。","现在不要问。"):
            for target in ("history","current"):
                with self.subTest(text=text,target=target):
                    case=self.ordinal_case()
                    if target=="history":case["history"][1]["content"]=text
                    else:case["input"]=text
                    with self.assertRaisesRegex(M.ReplayBlocked,"ordinal_explicit_refusal"):
                        self.ordinal(case)
                    self.assertEqual(list(Path(self.directory.name).iterdir()),[])

    def test_ordinal_assistant_quotation_does_not_become_user_refusal(self):
        case=self.ordinal_case();case["history"][0]["content"]="Not today; never ask me again."
        result=self.ordinal(case)
        self.assertEqual(result["trace"]["source_receipts"][0]["role"],"assistant")


if __name__ == "__main__":
    unittest.main()
