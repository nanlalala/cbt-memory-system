"""Invented miniature histories only; no benchmark data/model/API calls."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation/objective-memory"))
from persona_management_adapter import (ARMS, _clock, prepare_scope_plan,
    extraction_requests, replay_scope_prefix, prepare_arms, stable, MemoryManager)
from run_three_arm import Settings
from v3_persona_retrieval import PersonaChunkRetriever
from v3_run_objective_comparison import _messages


def case(prefix=4, *, identity="one", context="c1", persona="p1", rows=None):
    rows = rows or [("user", "I enjoy chess and quiet libraries."),
                    ("assistant", "Could a quiet swimming pool suit you?"),
                    ("user", "Now I prefer a lively pool and swimming."),
                    ("user", "Hello again, can we continue?")]
    return {"case_id": f"{identity}:{prefix}", "subject_id": "same-statistical-group",
        "group_id": "same-statistical-group", "dataset_key": "personamem_v1",
        "task": "preference", "query": "Which activity does the person prefer?",
        "options": ["(a) swimming", "(b) chess", "(c) running", "(d) painting"],
        "events": [], "as_of": None, "visible_context_end_exclusive": prefix,
        "persona_id": persona, "source_context_id": context,
        "history_scope_id": f"{context}:prefix:{prefix}",
        "history": [{"turn_id": f"{context}:t{i}", "session_id": context,
            "role": role, "text": text, "order": i}
            for i, (role, text) in enumerate(rows[:prefix])]}


def settings(**kwargs):
    return Settings(model="fixture", short_turns=1, top_k=3,
        short_chars=1600, long_chars=4000, max_context_chars=10000,
        minimum_relevance=kwargs.pop("minimum_relevance", .01), **kwargs)


def empty_outputs(plan):
    return {r["window_id"]: '{"candidates":[]}' for r in extraction_requests(plan)}


def candidate(source, *, content="swimming pool", memory_type="preference", **extra):
    return {"memory_type": memory_type, "content": content,
        "source_turn_ids": [source["chunk_id"]], "supporting_text": source["text"].strip(),
        "structured_payload": {}, **extra}


class PersonaManagerAdapterTests(unittest.TestCase):
    def plan(self, cases=None, **kwargs):
        return prepare_scope_plan(cases or [case()], **kwargs)

    def test_quote_only_exact_v3_baseline_prompts_roles_ties_oov_and_zero(self):
        fixtures = [case(), case(rows=[("system", "keep source data separate"),
            ("user", "pear apple"), ("assistant", "apple pear"),
            ("user", "today")]), case(rows=[("user", "same same"),
            ("user", "same same"), ("user", "same same"), ("user", "hello")])]
        for fixture in fixtures:
            for query, minimum in (("swimming newword", .01), ("same", 0.), ("unseen-only", .01)):
                fixture = copy.deepcopy(fixture)
                fixture["query"] = query
                config = settings(minimum_relevance=minimum)
                plan = self.plan([fixture], chunk_chars=128)
                prepared = prepare_arms(fixture, plan=plan,
                    extraction_outputs=empty_outputs(plan), settings=asdict(config))
                raw, managed = (prepared["prepared_by_arm"][arm] for arm in ARMS)
                short, long, trace = PersonaChunkRetriever(config, chunk_chars=128).prepare_case(fixture)
                self.assertEqual(raw["messages"], _messages(fixture, short, long))
                self.assertEqual(raw["messages"], managed["messages"])
                self.assertEqual(trace["retrieved_ids"], managed["trace"]["retrieved_ids"])
                self.assertTrue(managed["trace"]["pipeline_valid"])

    def test_extraction_payload_has_only_sources_never_questions_options_gold(self):
        fixture = case()
        plan = self.plan([fixture])
        requests = extraction_requests(plan)
        for request in requests:
            text = json.dumps(request["messages"])
            self.assertNotIn(fixture["query"], text)
            self.assertNotIn(fixture["options"][0], text)
            self.assertNotIn("correct_option", text)
            self.assertNotIn("2000-01", text)
            self.assertEqual(request["request_sha256"], stable(request["messages"]))
        bad = copy.deepcopy(fixture)
        bad["correct_option"] = "a"
        with self.assertRaises(ValueError):
            self.plan([bad])

    def test_windows_stop_at_all_cutoffs_and_future_outputs_cannot_enter_store(self):
        first, last = case(2), case(4)
        plan = self.plan([last, first], window_chars=256, prior_context_chars=128, chunk_chars=128)
        scope = plan["scopes"][0]
        self.assertTrue(all(not (min(next(s["source_order"] for s in scope["sources"] if s["chunk_id"] == c)
                 for c in w["current_chunk_ids"]) < 2 < w["available_at_cutoff"])
                 for w in scope["windows"]))
        outputs = empty_outputs(plan)
        for window in scope["windows"]:
            if window["available_at_cutoff"] > 2:
                outputs[window["window_id"]] = "future malformed and secret"
        replay = replay_scope_prefix(plan, outputs, scope_id=scope["scope_id"], cutoff=2)
        try:
            self.assertTrue(all(s["source_order"] < 2 for s in replay.sources.values()))
            self.assertTrue(all(s["status"] != "extraction_invalid" for s in replay.statuses))
            self.assertTrue(all(r.observed_at < _clock(2) for r in replay.store.list_for_user(replay.user_id)))
        finally:
            replay.close()

    def test_statistical_group_is_not_access_scope_and_prefixes_consistent(self):
        a, b = case(identity="a"), case(identity="b", context="c2", persona="p2")
        b["history"][0]["text"] = "A private separate sailing preference."
        plan = self.plan([a, b])
        self.assertEqual(len(plan["scopes"]), 2)
        self.assertNotEqual(plan["case_bindings"][a["case_id"]]["scope_id"],
                            plan["case_bindings"][b["case_id"]]["scope_id"])
        result = prepare_arms(a, plan=plan, extraction_outputs=empty_outputs(plan), settings=settings())
        self.assertNotIn("sailing", json.dumps(result["prepared_by_arm"][ARMS[1]]["messages"]))
        revised = copy.deepcopy(a)
        revised["case_id"] = "revised"
        revised["history"][0]["text"] = "changed silently"
        with self.assertRaisesRegex(ValueError, "prefixes disagree"):
            self.plan([a, revised])

    def test_semantic_candidate_writes_real_store_unconfirmed_and_renders_only_source(self):
        fixture = case()
        fixture["query"] = "Which chess or swimming activity?"
        plan = self.plan([fixture])
        scope = plan["scopes"][0]
        source = next(s for s in scope["sources"] if s["source_order"] == 0)
        output = {scope["windows"][0]["window_id"]: json.dumps({"candidates": [candidate(source,
            content="swimming swimming swimming made-up indexing label")]})}
        replay = replay_scope_prefix(plan, output, scope_id=scope["scope_id"], cutoff=4)
        try:
            records = replay.store.list_for_user(replay.user_id)
            semantic = [r for r in records if r.memory_id.startswith("persona-index-")]
            self.assertEqual(len(semantic), 1)
            self.assertEqual(semantic[0].confirmation_state, "unconfirmed")
            self.assertEqual(semantic[0].lifecycle_status, "candidate")
            result = prepare_arms(fixture, plan=plan, extraction_outputs=output, settings=settings(), replay=replay)
            managed = result["prepared_by_arm"][ARMS[1]]
            self.assertNotIn("made-up indexing label", json.dumps(managed["messages"]))
            self.assertTrue(managed["trace"]["retrieval_candidates"])
            self.assertTrue(any(o["kind"] == "semantic_candidate" and o["action"] == "created"
                                for o in managed["trace"]["source_event_operations"]))
            self.assertFalse(managed["trace"]["semantic_summaries_rendered"])
            self.assertFalse(result["trace"]["answer_reingestion"])
        finally:
            replay.close()

    def test_correction_links_proposed_not_applied_and_invalid_targets_rejected(self):
        fixture = case()
        plan = self.plan([fixture])
        scope = plan["scopes"][0]
        first = next(s for s in scope["sources"] if s["source_order"] == 0)
        assistant = next(s for s in scope["sources"] if s["source_order"] == 1)
        update = next(s for s in scope["sources"] if s["source_order"] == 2)
        wid = scope["windows"][0]["window_id"]
        for target, valid in ((first["chunk_id"], True), (assistant["chunk_id"], False),
                              (update["chunk_id"], False), ("foreign", False), (None, False)):
            item = candidate(update, memory_type="user_correction", structured_payload={"target_source_id": target})
            replay = replay_scope_prefix(plan, {wid: json.dumps({"candidates": [item]})},
                                         scope_id=scope["scope_id"], cutoff=4)
            try:
                self.assertEqual(replay.statuses[0]["status"], "valid_candidates" if valid else "extraction_invalid")
                if valid:
                    self.assertTrue(any(o["action"] == "correction_requires_confirmation" for o in replay.operations))
                    self.assertEqual(replay.store.get(replay.user_id, first["chunk_id"]).lifecycle_status, "candidate")
            finally:
                replay.close()
        conflict = candidate(update, memory_type="user_correction", target_source_id=first["chunk_id"],
                             structured_payload={"proposed_update_target_source_id": assistant["chunk_id"]})
        replay = replay_scope_prefix(plan, {wid: json.dumps({"candidates": [conflict]})}, scope_id=scope["scope_id"], cutoff=4)
        self.assertEqual(replay.statuses[0]["status"], "extraction_invalid")
        replay.close()

    def test_invalid_extraction_retains_quotes_failure_status_and_source_spans(self):
        fixture = case()
        plan = self.plan([fixture])
        result = prepare_arms(fixture, plan=plan, extraction_outputs={}, settings=settings())
        managed = result["prepared_by_arm"][ARMS[1]]
        self.assertFalse(managed["trace"]["pipeline_valid"])
        self.assertTrue(managed["trace"]["invalid_window_ids"])
        self.assertEqual(managed["messages"], result["prepared_by_arm"][ARMS[0]]["messages"])
        for span in managed["trace"]["included_source_spans"]:
            self.assertEqual(span["end"] - span["start"], span["original_end"] - span["original_start"])
            self.assertTrue(span["complete"])

    def test_eligibility_before_extraction_and_unknown_dates_not_invented(self):
        fixture = case()
        fixture["history"][0]["sensitivity"] = "restricted"
        fixture["history"][1]["lifecycle_status"] = "deleted"
        plan = self.plan([fixture])
        payloads = json.dumps(extraction_requests(plan))
        self.assertNotIn("quiet libraries", payloads)
        self.assertNotIn("Could a quiet", payloads)
        result = prepare_arms(fixture, plan=plan, extraction_outputs=empty_outputs(plan), settings=settings())
        for arm in ARMS:
            body = json.dumps(result["prepared_by_arm"][arm]["messages"])
            self.assertNotIn("2000-01", body)
            self.assertIn("source_order:2", body)
            self.assertEqual(len(result["prepared_by_arm"][arm]["trace"]["excluded_source_units"]), 2)

    def test_boundaries_whitespace_and_full_source_hashes_match_legacy_spans(self):
        fixture = case(rows=[("user", "   swimming " * 50), ("assistant", "hello"),
                            ("user", "chess"), ("user", "again")])
        plan = self.plan([fixture], chunk_chars=128)
        result = prepare_arms(fixture, plan=plan, extraction_outputs=empty_outputs(plan), settings=settings())
        self.assertEqual(result["prepared_by_arm"][ARMS[0]]["messages"],
                         result["prepared_by_arm"][ARMS[1]]["messages"])
        source = next(s for s in plan["scopes"][0]["sources"] if s["source_order"] == 0)
        self.assertGreater(source["original_leading_trim"], 0)
        self.assertNotEqual(source["source_text_sha256"], source["original_source_text_sha256"])
        from persona_management_scoring import verify_source_bindings
        for arm in ARMS:
            trace = result["prepared_by_arm"][arm]["trace"]
            self.assertEqual(len(trace["source_catalog"]), len(fixture["history"]))
            verify_source_bindings(fixture, trace, fixture=True)

    def test_many_same_source_indexes_do_not_change_corpus_or_crowd_source_topk(self):
        fixture = case()
        plan = self.plan([fixture])
        scope = plan["scopes"][0]
        source = next(s for s in scope["sources"] if s["source_order"] == 0)
        items = [candidate(source, content=source["text"],
                           structured_payload={"slot_key": f"independent-proposal-{i}"}) for i in range(12)]
        outputs = {scope["windows"][0]["window_id"]: json.dumps({"candidates": items})}
        prepared = prepare_arms(fixture, plan=plan, extraction_outputs=outputs, settings=settings())
        raw, managed = (prepared["prepared_by_arm"][arm] for arm in ARMS)
        self.assertEqual(raw["messages"], managed["messages"])
        self.assertEqual(len(managed["trace"]["retrieval_index_aliases"]), 12)
        self.assertEqual(len(managed["trace"]["retrieved_ids"]), len(set(managed["trace"]["retrieved_ids"])))
        self.assertEqual(managed["trace"]["accepted_candidate_count"], 12)

    def test_semantic_transaction_rollback_has_no_phantom_provenance_or_operations(self):
        fixture = case()
        plan = self.plan([fixture])
        scope = plan["scopes"][0]
        source = next(s for s in scope["sources"] if s["source_order"] == 0)
        outputs = {scope["windows"][0]["window_id"]: json.dumps({"candidates": [
            candidate(source, content="a source-supported preference index"),
            candidate(source, content="a second episodic index", memory_type="episodic_event")]})}
        original = MemoryManager.ingest
        calls = []
        def injected(manager, record, **kwargs):
            calls.append(record.memory_id)
            if len(calls) == 2:
                raise ValueError("injected_second_write_failure")
            return original(manager, record, **kwargs)
        with patch.object(MemoryManager, "ingest", injected):
            replay = replay_scope_prefix(plan, outputs, scope_id=scope["scope_id"], cutoff=4)
        try:
            self.assertEqual(replay.statuses[0]["status"], "extraction_invalid")
            self.assertEqual(replay.statuses[0]["state_before_sha256"], replay.statuses[0]["state_after_sha256"])
            self.assertTrue(all(not key.startswith("persona-index-") for key in replay.dependencies))
            self.assertTrue(all(not record.memory_id.startswith("persona-index-") for record in replay.store.list_for_user(replay.user_id)))
            self.assertTrue(all(op["kind"] == "source_quote" for op in replay.operations))
        finally:
            replay.close()

    def test_serialized_request_cap_handles_many_tiny_messages_without_dropping_sources(self):
        fixture = case(300, rows=[("user", "x") for _ in range(300)])
        plan = self.plan([fixture])
        requests = extraction_requests(plan)
        self.assertGreater(len(requests), 1)
        self.assertTrue(all(sum(len(m["content"]) for m in r["messages"]) <= 24000 for r in requests))
        scope = plan["scopes"][0]
        observed = [cid for window in scope["windows"] for cid in window["current_chunk_ids"]]
        self.assertEqual(observed, [s["chunk_id"] for s in scope["sources"]])

    def test_unknown_branch_identity_and_expiry_do_not_borrow_cached_sources(self):
        fixture = case()
        other = copy.deepcopy(fixture)
        other["case_id"] = "other-branch"
        other["history_scope_id"] = "same-context-but-different-authorized-branch"
        plan = self.plan([fixture, other])
        self.assertEqual(len(plan["scopes"]), 2)
        fixture["as_of"] = "2026-11-05T00:00:00Z"
        fixture["history"][0]["valid_to"] = "2026-11-04T00:00:00Z"
        plan = self.plan([fixture])
        self.assertNotIn("quiet libraries", json.dumps(extraction_requests(plan)))

    def test_invalid_scalar_or_enum_cannot_poison_persisted_candidate_json(self):
        fixture = case()
        plan = self.plan([fixture])
        scope = plan["scopes"][0]
        source = next(s for s in scope["sources"] if s["source_order"] == 0)
        wid = scope["windows"][0]["window_id"]
        for override in ({"confidence": 3.}, {"importance": float("nan")},
                         {"sensitivity": "invented-policy"}):
            outputs = {wid: json.dumps({"candidates": [candidate(source, **override)]})}
            replay = replay_scope_prefix(plan, outputs, scope_id=scope["scope_id"], cutoff=4)
            try:
                self.assertEqual(replay.statuses[0]["status"], "extraction_invalid")
                self.assertEqual(len(replay.store.list_for_user(replay.user_id)), 4)
                self.assertTrue(all(op["kind"] == "source_quote" for op in replay.operations))
            finally:
                replay.close()

    def test_source_sensitivity_is_not_downgraded_by_quote_or_model_index(self):
        fixture = case()
        fixture["history"][0]["sensitivity"] = "sensitive"
        plan = self.plan([fixture])
        scope = plan["scopes"][0]
        source = next(s for s in scope["sources"] if s["source_order"] == 0)
        outputs = {scope["windows"][0]["window_id"]: json.dumps({"candidates": [
            candidate(source, sensitivity="standard")]})}
        replay = replay_scope_prefix(plan, outputs, scope_id=scope["scope_id"], cutoff=4)
        try:
            bound = [r for r in replay.store.list_for_user(replay.user_id)
                     if replay.anchors[r.memory_id] == source["chunk_id"]]
            self.assertEqual(len(bound), 2)
            self.assertTrue(all(r.sensitivity == "sensitive" for r in bound))
        finally:
            replay.close()


if __name__ == "__main__":
    unittest.main()
