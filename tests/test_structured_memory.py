"""Tiny synthetic software regressions, never benchmark-effectiveness results."""

from __future__ import annotations

import copy
import json
import os
import random
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation/end-to-end-dialogue"))

from run_three_arm import Settings, read_jsonl, stable_digest, write_jsonl  # noqa: E402
from benchmark_adapters import RetrievalExample  # noqa: E402
from prepare_formal_evaluation import DEFAULT_PLAN, load_plan, prepare_examples  # noqa: E402
from run_structured_memory import (  # noqa: E402
    ARMS, CandidateInventoryWriter, ScopeCachedEncoder, attempt_state, candidate_jsonl_digest,
    execute_extraction, extraction_messages,
    generate_run, ingest_source, load_cohort, load_run, make_request,
    parse_source_candidates, prepare_extraction_requests, prepare_structured_run, require_colab,
    source_inventory, streamed_digest, validate_case, validate_extraction_artifact, verbatim_output,
)
from score_structured_memory import score_structured_run  # noqa: E402
from memory_manager import MemoryManager  # noqa: E402
from sqlite_store import SQLiteMemoryStore  # noqa: E402


def fixture(**changes):
    return replace(RetrievalExample(
        example_id="fixture-a", category="single-session-user",
        query="What is my walking goal?", answer="SECRET_GOLD_NOT_FOR_GENERATION",
        evidence_ids=("SECRET_GOLD_EVIDENCE",), query_date="2026-10-10T00:00:00Z",
        sessions=(
            {"session_id": "old", "date": "2026-10-01T00:00:00Z", "turns": [
                {"role": "user", "content": "My walking goal is ten minutes after lunch."},
                {"role": "assistant", "content": "ASSISTANT_ONLY_OLD_CONTENT"}]},
            {"session_id": "recent", "date": "2026-10-09T00:00:00Z", "turns": [
                {"role": "user", "content": "I watched a film today."}]},
            {"session_id": "future", "date": "2026-10-11T00:00:00Z", "turns": [
                {"role": "user", "content": "SECRET_FUTURE_INFORMATION"}]},
        ),
    ), **changes)


def verbatim_sources():
    sources = []
    for session, date, texts in (
        ("old-a", "2026-10-01T00:00:00Z", ["Walking at noon.", "WALKING at noon!", "Tea after breakfast."]),
        ("old-b", "2026-10-02T00:00:00Z", ["Tea after breakfast!", "读书 after lunch.  "]),
    ):
        source = {"scope_id": "synthetic-scope", "session_id": session, "observed_at": date,
                  "turns": [{"turn_id": f"{session}-{index}", "speaker": "user", "text": text}
                            for index, text in enumerate(texts)]}
        source["source_sha256"] = stable_digest(source)
        sources.append(source)
    records = [record for source in sources for record in parse_source_candidates(source, verbatim_output(source))]
    turns = {(source["session_id"], turn["turn_id"]):
             {"text": f'{turn["speaker"]}: {turn["text"]}'.strip(), "observed_at": source["observed_at"]}
             for source in sources for turn in source["turns"]}
    return sources, records, turns


class StructuredMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.plan = load_plan(DEFAULT_PLAN)
        self.plan["shared_response_configuration"].update({"model": "test-model", "accepted_provider_models": ["test-model"]})
        self.settings = Settings(model="test-model", max_response_tokens=4096, thinking_mode="disabled")
        self.api_base = self.plan["shared_response_configuration"]["api_base"]

    def tearDown(self):
        self.temp.cleanup()

    def cohort(self, name="cohort", examples=None, plan=None):
        path = self.base / name
        prepare_examples(examples or [fixture()], "longmemeval_s_cleaned", plan or self.plan,
                         {"sha256": "f" * 64, "bytes": 1}, path,
                         role="exploratory", fixture_only=True)
        return path

    def prepared(self, name="run", cohort=None, settings=None):
        path = self.base / name
        prepare_structured_run(cohort or self.cohort(), path, settings or self.settings, self.api_base)
        return path

    def complete_fixture_responses(self, run_dir, *, model="test-model"):
        protocol = json.loads((run_dir / "protocol.json").read_text())
        rows = [{"example_id": row["example_id"], "arm": row["arm"],
                 "prompt_sha256": row["prompt_sha256"], "protocol_sha256": stable_digest(protocol),
                 "hypothesis": "SYNTHETIC TEST RESPONSE ONLY", "provider_model": model,
                 "finish_reason": "stop", "latency_seconds": 0.01,
                 "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
                for row in read_jsonl(run_dir / "prompts.jsonl")]
        write_jsonl(run_dir / "responses.jsonl", rows)
        return rows

    def test_preparation_uses_real_manager_but_never_fabricates_confirmation(self):
        run_dir = self.prepared()
        memories = read_jsonl(run_dir / "candidate_memories.jsonl")
        self.assertEqual(len(memories), 1)
        record = memories[0]["record"]
        self.assertEqual((record["lifecycle_status"], record["confirmation_state"]), ("candidate", "unconfirmed"))
        self.assertIn("user: My walking goal", record["source_evidence"]["supporting_text"])
        self.assertEqual(record["observed_at"], "2026-10-01T00:00:00Z")
        self.assertIn("verbatim_candidate_baseline", json.loads((run_dir / "protocol.json").read_text())["extraction_mode"])

    def test_verbatim_bulk_matches_serial_records_dedup_results_and_audits(self):
        _, records, turns = verbatim_sources()
        serial, bulk = SQLiteMemoryStore(), SQLiteMemoryStore()
        try:
            existing = replace(records[0], memory_id="earlier-active", lifecycle_status="active",
                               confirmation_state="user_confirmed")
            for store in (serial, bulk):
                store.put(existing)
            expected = [MemoryManager(serial).ingest(record, actor="synthetic-test") for record in records]
            with patch.object(bulk, "list_for_user", wraps=bulk.list_for_user) as reads:
                actual = MemoryManager(bulk).ingest_verbatim_candidates(
                    records, user_id="synthetic-scope", source_turns=turns, actor="synthetic-test")
                self.assertEqual(reads.call_count, 1)
            self.assertEqual(actual, expected)
            self.assertEqual([record.to_dict() for record in bulk.list_for_user("synthetic-scope")],
                             [record.to_dict() for record in serial.list_for_user("synthetic-scope")])
            self.assertEqual(bulk.audit_for_user("synthetic-scope"), serial.audit_for_user("synthetic-scope"))
            self.assertEqual([result.action for result in actual],
                             ["duplicate_rejected", "duplicate_rejected", "created", "duplicate_rejected", "created"])
        finally:
            serial.close()
            bulk.close()

    def test_verbatim_bulk_rejects_nonverbatim_slots_confirmation_and_missing_sources(self):
        _, records, turns = verbatim_sources()
        record = records[0]
        invalid = [
            replace(record, content="An unsupported summary"),
            replace(record, source_evidence=replace(record.source_evidence, supporting_text="Walking")),
            replace(record, source_evidence=replace(record.source_evidence, turn_ids=["not-a-source"])),
            replace(record, source_evidence=replace(record.source_evidence, turn_ids=["a", "b"])),
            replace(record, source_evidence=replace(record.source_evidence, session_id="")),
            replace(record, memory_type="preference"), replace(record, origin="agent_inference"),
            replace(record, lifecycle_status="active"), replace(record, confirmation_state="user_confirmed"),
            replace(record, structured_payload={"slot_key": "goal"}), replace(record, tags=["inferred"]),
            replace(record, related_memory_ids=["other"]), replace(record, supersedes_memory_id="other"),
            replace(record, user_id="other-user"), replace(record, valid_from=None),
            replace(record, valid_to="2026-11-01T00:00:00Z"),
            replace(record, observed_at="2026-10-03T00:00:00Z"),
        ]
        store = SQLiteMemoryStore()
        try:
            for bad in invalid:
                with self.subTest(record=bad):
                    with self.assertRaises(ValueError):
                        MemoryManager(store).ingest_verbatim_candidates(
                            [records[-1], bad], user_id="synthetic-scope", source_turns=turns)
                    self.assertEqual(store.list_for_user("synthetic-scope"), [])
                    self.assertEqual(store.audit_for_user("synthetic-scope"), [])
            with self.assertRaises(ValueError):
                MemoryManager(store).ingest_verbatim_candidates([record], user_id="synthetic-scope", source_turns={})
        finally:
            store.close()

    def test_verbatim_bulk_write_failure_rolls_back_all_records_and_audits(self):
        _, records, turns = verbatim_sources()
        store = SQLiteMemoryStore()
        try:
            store.connection.execute("""CREATE TRIGGER fail_batch BEFORE INSERT ON memories
                WHEN NEW.content = 'user: Tea after breakfast.'
                BEGIN SELECT RAISE(ABORT, 'synthetic interruption'); END""")
            with self.assertRaisesRegex(Exception, "synthetic interruption"):
                MemoryManager(store).ingest_verbatim_candidates(records, user_id="synthetic-scope", source_turns=turns)
            self.assertEqual(store.list_for_user("synthetic-scope"), [])
            self.assertEqual(store.audit_for_user("synthetic-scope"), [])
        finally:
            store.close()

    def test_verbatim_bulk_id_collision_cannot_overwrite_another_user(self):
        _, records, turns = verbatim_sources()
        store = SQLiteMemoryStore()
        try:
            foreign = replace(records[2], user_id="other-user")
            store.put(foreign)
            with self.assertRaises(PermissionError):
                MemoryManager(store).ingest_verbatim_candidates(records, user_id="synthetic-scope", source_turns=turns)
            self.assertEqual(store.list_for_user("synthetic-scope"), [])
            self.assertEqual(store.audit_for_user("synthetic-scope"), [])
            self.assertEqual(store.get("other-user", foreign.memory_id).to_dict(), foreign.to_dict())
        finally:
            store.close()

    def test_streamed_inventory_hash_preserves_original_canonical_list_format(self):
        rows = [{"scope_id": "中文", "record": {"z": "quote \\\"", "a": [1, None]}},
                {"record": {"text": "é\n第二行"}, "scope_id": "other"}]
        path = self.base / "inventory.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            writer = CandidateInventoryWriter(handle)
            for row in rows:
                writer.add(row)
        self.assertEqual(writer.hexdigest(), stable_digest(rows))
        self.assertEqual(candidate_jsonl_digest(path), (stable_digest(rows), len(rows)))
        self.assertEqual(streamed_digest({"sources": rows}), stable_digest({"sources": rows}))

    def test_scoped_preparation_preserves_serial_prompts_inventory_and_source_hash(self):
        cohort = self.cohort(examples=[fixture(), fixture(example_id="fixture-b")])
        manifest, cases, _ = load_cohort(cohort)
        sources = source_inventory(cases, manifest["dataset_key"], self.settings)
        serial = SQLiteMemoryStore()
        by_scope = {}
        try:
            for source in sources.values():
                by_scope.setdefault(source["scope_id"], []).extend(ingest_source(serial, source, verbatim_output(source)))
        finally:
            serial.close()
        expected_inventory = [{"scope_id": scope, "record": record.to_dict()}
                              for scope in sorted(by_scope) for record in by_scope[scope]]
        expected_prompts, ordering = [], random.Random(self.settings.seed)
        for case in sorted(cases, key=lambda row: (row["scope_id"], row["example_id"])):
            arms = list(ARMS)
            ordering.shuffle(arms)
            expected_prompts.extend(make_request(case, manifest["dataset_key"], arm, self.settings,
                                                 by_scope[case["scope_id"]]) for arm in arms)
        closed_scopes = []
        class ScopedStore(SQLiteMemoryStore):
            def close(self):
                closed_scopes.append(self.list_user_ids())
                super().close()
        with patch("run_structured_memory.SQLiteMemoryStore", ScopedStore):
            run_dir = self.prepared(cohort=cohort)
        self.assertEqual(len(closed_scopes), 2)
        self.assertTrue(all(len(scopes) == 1 for scopes in closed_scopes))
        self.assertEqual(read_jsonl(run_dir / "prompts.jsonl"), expected_prompts)
        self.assertEqual(read_jsonl(run_dir / "candidate_memories.jsonl"), expected_inventory)
        protocol = json.loads((run_dir / "protocol.json").read_text())
        self.assertEqual(protocol["candidate_records_sha256"], stable_digest(expected_inventory))
        self.assertEqual(protocol["source_inventory_sha256"], stable_digest(sources))
        self.assertEqual(list(run_dir.glob("*.tmp")), [])

    def test_one_shared_history_group_does_not_produce_a_bootstrap_interval(self):
        cohort = self.cohort(examples=[fixture(), fixture(example_id="fixture-b")])
        run_dir = self.prepared(cohort=cohort)
        responses = self.complete_fixture_responses(run_dir)
        references = {row["example_id"]: row for row in read_jsonl(run_dir / "references.jsonl")}
        for row in responses:
            if row["arm"] == ARMS[2]:
                row["hypothesis"] = references[row["example_id"]]["reference_answer"]
        write_jsonl(run_dir / "responses.jsonl", responses)
        with patch("score_structured_memory.paired_interval", side_effect=AssertionError("one group cannot be bootstrapped")):
            summary = score_structured_run(run_dir, write_outputs=False)
        self.assertEqual(summary["examples"], 2)
        self.assertEqual(summary["independent_groups"], 1)
        comparison = summary["paired_differences"][f"{ARMS[2]}_minus_{ARMS[1]}"]
        self.assertEqual(comparison["mean_token_f1_delta"], 1.0)
        self.assertIsNone(comparison["cluster_bootstrap_95_percent_interval"])
        self.assertEqual(comparison["cluster_count"], 1)
        self.assertEqual(comparison["interval_status"], "not_estimable_fewer_than_two_groups")

    def test_gold_future_and_assistant_only_content_do_not_enter_long_candidates(self):
        run_dir = self.prepared()
        text = (run_dir / "prompts.jsonl").read_text()
        for value in ("SECRET_GOLD", "SECRET_FUTURE_INFORMATION", "ASSISTANT_ONLY_OLD_CONTENT"):
            self.assertNotIn(value, text)
        prompts = read_jsonl(run_dir / "prompts.jsonl")
        treatment = next(row for row in prompts if row["arm"] == ARMS[2])
        self.assertTrue(treatment["included_memory_ids"])
        self.assertEqual(treatment["future_session_ids_excluded"], ["future"])
        self.assertIn("unconfirmed", treatment["messages"][1]["content"])

    def test_short_window_and_system_prompt_are_identical_between_history_arms(self):
        prompts = {row["arm"]: row for row in read_jsonl(self.prepared() / "prompts.jsonl")}
        self.assertEqual(prompts[ARMS[1]]["included_short_ids"], prompts[ARMS[2]]["included_short_ids"])
        def short(row):
            return row["messages"][1]["content"].split('<short_history data_only="true">')[1].split("</short_history>")[0]
        self.assertEqual(short(prompts[ARMS[1]]), short(prompts[ARMS[2]]))
        self.assertEqual(len({row["messages"][0]["content"] for row in prompts.values()}), 1)
        self.assertEqual(prompts[ARMS[0]]["included_short_ids"], [])
        self.assertEqual(prompts[ARMS[1]]["included_memory_ids"], [])

    def test_different_question_or_gold_cannot_change_source_preprocessing(self):
        first = self.prepared("run-a", self.cohort("a"))
        second = self.prepared("run-b", self.cohort("b", [fixture(query="A DIFFERENT QUESTION", answer="DIFFERENT_GOLD")]))
        self.assertEqual(read_jsonl(first / "candidate_memories.jsonl"), read_jsonl(second / "candidate_memories.jsonl"))

    def test_extraction_requests_contain_source_only_and_exact_quotes(self):
        cohort = self.cohort()
        output = self.base / "extraction"
        prepare_extraction_requests(cohort, output, self.settings, self.api_base)
        request = read_jsonl(output / "extraction_requests.jsonl")[0]
        raw = json.dumps(request)
        self.assertNotIn("What is my walking goal", raw)
        self.assertNotIn("SECRET_GOLD", raw)
        self.assertNotIn("SECRET_FUTURE", raw)
        store = SQLiteMemoryStore()
        try:
            with self.assertRaisesRegex(ValueError, "exact quote"):
                ingest_source(store, request["source"], json.dumps({"candidates": [{
                    "memory_type": "explicit_fact", "content": "Invented", "source_turn_ids": [request["source"]["turns"][0]["turn_id"]],
                    "supporting_text": "THIS QUOTE DOES NOT EXIST"}]}))
        finally:
            store.close()

    def test_model_artifact_requires_source_hash_prompt_and_preapproved_identity(self):
        cohort = self.cohort()
        output = self.base / "extraction"
        protocol = prepare_extraction_requests(cohort, output, self.settings, self.api_base)
        request = read_jsonl(output / "extraction_requests.jsonl")[0]
        artifact = {"schema_version": protocol["protocol_version"], "protocol": protocol,
                    "protocol_sha256": stable_digest(protocol), "sessions": [{
                        "source_sha256": request["source_sha256"], "prompt_sha256": request["prompt_sha256"],
                        "provider_model": "WRONG_MODEL", "raw_output": '{"candidates": []}'}]}
        path = self.base / "artifact.json"
        path.write_text(json.dumps(artifact))
        manifest, cases, _ = load_cohort(cohort)
        sources = source_inventory(cases, manifest["dataset_key"], self.settings)
        with self.assertRaisesRegex(ValueError, "allowlist"):
            validate_extraction_artifact(path, sources, manifest)

    def test_frozen_cohort_or_preregistration_tamper_is_rejected(self):
        cohort = self.cohort()
        lock = json.loads((cohort / "preregistration_lock.json").read_text())
        lock["plan"]["statistics"]["minimum_effect"] = 0.001
        (cohort / "preregistration_lock.json").write_text(json.dumps(lock))
        with self.assertRaises(ValueError):
            prepare_structured_run(cohort, self.base / "run", self.settings, self.api_base)

    def test_runtime_response_model_budget_or_alias_changes_are_rejected(self):
        cohort = self.cohort()
        for index, settings in enumerate((replace(self.settings, model="other-model"),
                                          replace(self.settings, max_response_tokens=1024),
                                          replace(self.settings, accepted_provider_models=("unexpected-alias",)))):
            with self.assertRaisesRegex(ValueError, "frozen formal"):
                prepare_structured_run(cohort, self.base / f"run-{index}", settings, self.api_base)

    def test_candidate_cannot_be_retrieved_across_subjects(self):
        run_dir = self.prepared()
        from memory_models import MemoryRecord
        record = MemoryRecord.from_dict(read_jsonl(run_dir / "candidate_memories.jsonl")[0]["record"])
        record.user_id = "ANOTHER_SUBJECT"
        manifest, cases, _ = load_cohort(self.base / "cohort")
        with self.assertRaisesRegex(ValueError, "cross-subject"):
            make_request(cases[0], manifest["dataset_key"], ARMS[2], self.settings, [record])

    def test_budget_dropped_memory_is_not_reported_as_included(self):
        plan = copy.deepcopy(self.plan)
        plan["shared_response_configuration"]["long_context_maximum_characters"] = 900
        settings = replace(self.settings, long_chars=900)
        value = fixture(sessions=(
            {"session_id": "old", "date": "2026-10-01T00:00:00Z", "turns": [{"role": "user", "content": "My walking goal is " + "walking " * 300}]},
            {"session_id": "recent", "date": "2026-10-09T00:00:00Z", "turns": [{"role": "user", "content": "Hello"}]},
        ))
        run_dir = self.prepared(cohort=self.cohort(examples=[value], plan=plan), settings=settings)
        treatment = next(row for row in read_jsonl(run_dir / "prompts.jsonl") if row["arm"] == ARMS[2])
        self.assertFalse(treatment["included_memory_ids"])
        self.assertTrue(treatment["dropped_memory_ids"])
        self.assertFalse(treatment["included_source_turn_ids"])

    def test_dense_cache_encodes_documents_once_and_clears_between_subjects(self):
        class Encoder:
            model_identity = {"fixed": True}
            def __init__(self):
                self.calls = []
            def encode(self, texts, *, is_query):
                self.calls.append((list(texts), is_query))
                return [[1.0, 0.0] for _ in texts]
        underlying = Encoder()
        cached = ScopeCachedEncoder(underlying)
        cached.set_scope("first")
        cached.encode(["same", "same", "other"], is_query=False)
        cached.encode(["same", "other"], is_query=False)
        cached.encode(["query"], is_query=True)
        self.assertEqual(underlying.calls, [(["same", "other"], False), (["query"], True)])
        cached.set_scope("second")
        cached.encode(["same"], is_query=False)
        self.assertEqual(underlying.calls[-1], (["same"], False))

    def test_proxy_scoring_requires_complete_grid_and_correct_model(self):
        run_dir = self.prepared()
        with self.assertRaisesRegex(ValueError, "all three"):
            score_structured_run(run_dir)
        self.complete_fixture_responses(run_dir, model="consistently-wrong-model")
        with self.assertRaisesRegex(ValueError, "allowlist"):
            score_structured_run(run_dir)
        self.complete_fixture_responses(run_dir)
        result = score_structured_run(run_dir)
        self.assertEqual(result["responses"], 3)
        self.assertFalse(result["confirmatory_claim"])
        self.assertEqual(result["official_evaluator_status"], "not executed")
        self.assertTrue((run_dir / f"official_longmemeval_{ARMS[2]}.jsonl").exists())
        pairs = read_jsonl(run_dir / "structured_model_judge_pairs.jsonl")
        self.assertNotIn("arm", pairs[0])

    def test_prompt_or_candidate_file_tampering_prevents_scores(self):
        run_dir = self.prepared()
        self.complete_fixture_responses(run_dir)
        (run_dir / "candidate_memories.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_run(run_dir)

    def test_tiny_fixtures_cannot_be_sent_to_paid_generation(self):
        run_dir = self.prepared()
        with patch("run_structured_memory.require_colab"), patch("run_structured_memory.generate_response") as provider:
            with self.assertRaisesRegex(ValueError, "fixtures"):
                generate_run(run_dir, max_calls=1)
            provider.assert_not_called()

    def test_colab_guard_refuses_personal_machine_without_importing_models(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "restricted to Google Colab"):
                require_colab()

    def test_gold_label_in_case_or_turn_schema_is_rejected(self):
        _, cases, _ = load_cohort(self.cohort())
        case = copy.deepcopy(cases[0])
        case["sessions"][0]["turns"][0]["has_answer"] = True
        with self.assertRaisesRegex(ValueError, "gold/evidence"):
            validate_case(case)

    def fake_result(self, model="test-model"):
        return {"hypothesis": "SOFTWARE TEST ONLY", "provider_model": model, "finish_reason": "stop", "latency_seconds": 0.01,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    def test_explicit_per_invocation_cap_preserves_full_frozen_cohort(self):
        run_dir = self.prepared()
        with patch("run_structured_memory.require_colab"), patch("run_structured_memory.require_generation_cohort"), \
             patch.dict(os.environ, {"CBT_EVAL_API_KEY": "synthetic-unit-test-key"}), \
             patch("run_structured_memory.generate_response", return_value=self.fake_result()) as provider:
            result = generate_run(run_dir, max_calls=1)
        self.assertEqual(provider.call_count, 1)
        self.assertEqual(result["requests"], 3)
        self.assertEqual(result["completed_responses"], 1)
        self.assertEqual(len(read_jsonl(run_dir / "response_attempts.jsonl")), 1)

    def test_transient_retries_are_bounded_across_manual_resumes(self):
        run_dir = self.prepared()
        with patch("run_structured_memory.require_colab"), patch("run_structured_memory.require_generation_cohort"), \
             patch.dict(os.environ, {"CBT_EVAL_API_KEY": "synthetic-unit-test-key"}), \
             patch("run_structured_memory.generate_response", side_effect=HTTPError("https://example.invalid", 429, "", {}, None)) as provider:
            for _ in range(3):
                with self.assertRaises(HTTPError):
                    generate_run(run_dir, max_calls=1)
            # First request exhausted; the fourth invocation moves to another
            # frozen request instead of charging an unlimited fourth retry.
            with self.assertRaises(HTTPError):
                generate_run(run_dir, max_calls=1)
        attempts = read_jsonl(run_dir / "response_attempts.jsonl")
        self.assertEqual([row["attempt_index"] for row in attempts], [1, 2, 3, 1])
        self.assertNotEqual(attempts[0]["request_key"], attempts[-1]["request_key"])
        self.assertEqual(provider.call_count, 4)

    def test_identity_failure_is_terminal_across_resumes_and_does_not_log_secrets(self):
        run_dir = self.prepared()
        with patch("run_structured_memory.require_colab"), patch("run_structured_memory.require_generation_cohort"), \
             patch.dict(os.environ, {"CBT_EVAL_API_KEY": "synthetic-unit-test-key"}), \
             patch("run_structured_memory.generate_response", side_effect=ValueError("SECRET_PROVIDER_BODY")) as provider:
            with self.assertRaises(ValueError):
                generate_run(run_dir, max_calls=1)
            with self.assertRaisesRegex(ValueError, "invalidated"):
                generate_run(run_dir, max_calls=1)
        self.assertEqual(provider.call_count, 1)
        self.assertNotIn("SECRET_PROVIDER_BODY", (run_dir / "failures.jsonl").read_text())
        self.assertNotIn("synthetic-unit-test-key", (run_dir / "failures.jsonl").read_text())

    def test_provider_alias_change_stops_before_persisting_changed_model(self):
        plan = copy.deepcopy(self.plan)
        plan["shared_response_configuration"]["accepted_provider_models"] = ["test-model", "approved-alias"]
        settings = replace(self.settings, accepted_provider_models=("approved-alias",))
        run_dir = self.prepared(cohort=self.cohort(plan=plan), settings=settings)
        with patch("run_structured_memory.require_colab"), patch("run_structured_memory.require_generation_cohort"), \
             patch.dict(os.environ, {"CBT_EVAL_API_KEY": "synthetic-unit-test-key"}), \
             patch("run_structured_memory.generate_response", side_effect=[self.fake_result(), self.fake_result("approved-alias")]):
            with self.assertRaisesRegex(ValueError, "changed"):
                generate_run(run_dir, max_calls=3)
        self.assertEqual(len(read_jsonl(run_dir / "responses.jsonl")), 1)

    def test_extraction_failure_intent_is_retained_and_terminal_resume_is_blocked(self):
        cohort = self.cohort()
        output = self.base / "extract"
        prepare_extraction_requests(cohort, output, self.settings, self.api_base)
        with patch("run_structured_memory.require_colab"), patch("run_structured_memory.require_generation_cohort"), \
             patch.dict(os.environ, {"CBT_EVAL_API_KEY": "synthetic-unit-test-key"}), \
             patch("run_structured_memory.generate_response", return_value=self.fake_result()) as provider:
            with self.assertRaises(json.JSONDecodeError):
                execute_extraction(output, max_calls=1)
            with self.assertRaisesRegex(ValueError, "invalidated"):
                execute_extraction(output, max_calls=1)
        self.assertEqual(provider.call_count, 1)
        self.assertEqual(len(read_jsonl(output / "extraction_attempts.jsonl")), 1)
        self.assertEqual(len(read_jsonl(output / "extraction_failures.jsonl")), 1)
        self.assertFalse((output / "extraction_artifact.json").exists())

    def test_valid_model_extraction_artifact_stays_unconfirmed(self):
        cohort = self.cohort()
        output = self.base / "extract"
        prepare_extraction_requests(cohort, output, self.settings, self.api_base)
        request = read_jsonl(output / "extraction_requests.jsonl")[0]
        turn = request["source"]["turns"][0]
        raw = json.dumps({"candidates": [{"memory_type": "explicit_fact", "content": "Walking lasts ten minutes.",
                                           "source_turn_ids": [turn["turn_id"]], "supporting_text": turn["text"]}]})
        with patch("run_structured_memory.require_colab"), patch("run_structured_memory.require_generation_cohort"), \
             patch.dict(os.environ, {"CBT_EVAL_API_KEY": "synthetic-unit-test-key"}), \
             patch("run_structured_memory.generate_response", return_value={**self.fake_result(), "hypothesis": raw}):
            result = execute_extraction(output, max_calls=1)
        self.assertEqual(result["completed"], 1)
        run_dir = self.base / "model-run"
        with patch.object(MemoryManager, "ingest_verbatim_candidates", side_effect=AssertionError("model summaries cannot use bulk import")):
            prepare_structured_run(cohort, run_dir, self.settings, self.api_base,
                                   extraction_artifact=output / "extraction_artifact.json")
        records = read_jsonl(run_dir / "candidate_memories.jsonl")
        self.assertEqual(records[0]["record"]["confirmation_state"], "unconfirmed")
        self.assertEqual(json.loads((run_dir / "protocol.json").read_text())["extraction_mode"], "model_extracted_unconfirmed_candidates")


if __name__ == "__main__":
    unittest.main()
