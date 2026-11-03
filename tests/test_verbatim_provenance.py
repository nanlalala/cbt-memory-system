"""Small source-fidelity regressions; no dataset download, model or API calls."""

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation/end-to-end-dialogue"))

from prepare_formal_evaluation import DEFAULT_PLAN, load_plan, prepare_examples
from run_three_arm import Settings, read_jsonl
from benchmark_adapters import RetrievalExample
from run_structured_memory import (
    PREPROCESSING_POLICY, VERSION, load_run, parse_source_candidates,
    prepare_structured_run, stable_digest, verbatim_output,
)
from memory_manager import MemoryManager, _normalise
from sqlite_store import SQLiteMemoryStore


def source(session, text):
    result = {"scope_id": "synthetic-scope", "session_id": session,
              "observed_at": "2026-10-01T00:00:00Z",
              "turns": [{"turn_id": session + "-u1", "speaker": "user", "text": text}]}
    result["source_sha256"] = stable_digest(result)
    return result


def records_and_evidence(sources):
    records = [record for item in sources
               for record in parse_source_candidates(item, verbatim_output(item))]
    evidence = {(item["session_id"], turn["turn_id"]): {
        "text": f'{turn["speaker"]}: {turn["text"]}'.strip(),
        "observed_at": item["observed_at"],
    } for item in sources for turn in item["turns"]}
    return records, evidence


class VerbatimProvenanceTests(unittest.TestCase):
    def test_signed_numbers_and_operators_remain_distinct_in_both_write_paths(self):
        texts = ["My temperature adjustment is -5 degrees.",
                 "My temperature adjustment is +5 degrees.",
                 "My ratio is 1/2.", "My ratio is 12."]
        records, evidence = records_and_evidence([
            source("source-" + str(index), text) for index, text in enumerate(texts)])
        self.assertNotEqual(_normalise(texts[0]), _normalise(texts[1]))
        self.assertNotEqual(_normalise(texts[2]), _normalise(texts[3]))
        for batched in (False, True):
            with self.subTest(batched=batched):
                store = SQLiteMemoryStore()
                try:
                    manager = MemoryManager(store)
                    for record in records:
                        if batched:
                            result = manager.ingest_verbatim_candidates([record],
                                user_id="synthetic-scope", source_turns={
                                    (record.source_evidence.session_id, record.source_evidence.turn_ids[0]):
                                    {"text": record.content, "observed_at": record.observed_at}})[0]
                        else:
                            result = manager.ingest(record)
                        self.assertEqual(result.action, "created")
                    self.assertEqual(len(store.list_for_user("synthetic-scope")), len(texts))
                finally:
                    store.close()

    def test_repeated_text_preserves_each_source_but_same_source_replay_is_rejected(self):
        records, evidence = records_and_evidence([
            source("first-session", "I walk after lunch."),
            source("second-session", "I walk after lunch.")])
        replay = replace(records[0], memory_id="replayed-first-observation")
        serial, bulk = SQLiteMemoryStore(), SQLiteMemoryStore()
        try:
            expected = [MemoryManager(serial).ingest(record, actor="fixture")
                        for record in [*records, replay]]
            actual = MemoryManager(bulk).ingest_verbatim_candidates([*records, replay],
                user_id="synthetic-scope", source_turns=evidence, actor="fixture")
            self.assertEqual(actual, expected)
            self.assertEqual([result.action for result in actual],
                             ["created", "created", "duplicate_rejected"])
            self.assertEqual([record.to_dict() for record in bulk.list_for_user("synthetic-scope")],
                             [record.to_dict() for record in serial.list_for_user("synthetic-scope")])
            self.assertEqual(bulk.audit_for_user("synthetic-scope"), serial.audit_for_user("synthetic-scope"))
            self.assertEqual({record.source_evidence.session_id for record in bulk.list_for_user("synthetic-scope")},
                             {"first-session", "second-session"})
            # Re-extracting the original turn later is still a replay, not a
            # new observation merely because the extractor's clock changed.
            later_replay = replace(replay, memory_id="later-replay",
                                   observed_at="2026-10-03T00:00:00Z")
            self.assertEqual(MemoryManager(serial).ingest(later_replay).action, "duplicate_rejected")
        finally:
            serial.close()
            bulk.close()

    def test_full_preparation_retains_both_signed_facts_and_both_source_traces(self):
        plan = load_plan(DEFAULT_PLAN)
        plan["shared_response_configuration"].update(model="test-model", accepted_provider_models=["test-model"])
        example = RetrievalExample(example_id="signed-fixture", category="single-session-user",
            query="What were my temperature adjustments?", answer="SCORING_ONLY",
            evidence_ids=("SCORING_ONLY_EVIDENCE",), query_date="2026-10-10T00:00:00Z",
            sessions=tuple({"session_id": name, "date": date, "turns": [{"role": "user", "content": text}]}
                for name, date, text in (
                    ("minus", "2026-10-01T00:00:00Z", "My temperature adjustment is -5 degrees."),
                    ("plus", "2026-10-02T00:00:00Z", "My temperature adjustment is +5 degrees."),
                    ("recent", "2026-10-09T00:00:00Z", "Today I read a book."))))
        with tempfile.TemporaryDirectory() as temporary:
            cohort, run = Path(temporary) / "cohort", Path(temporary) / "run"
            prepare_examples([example], "longmemeval_s_cleaned", plan,
                {"sha256": "f" * 64, "bytes": 1}, cohort, role="exploratory", fixture_only=True)
            settings = Settings(model="test-model", max_response_tokens=4096, thinking_mode="disabled")
            prepare_structured_run(cohort, run, settings, plan["shared_response_configuration"]["api_base"])
            rows = read_jsonl(run / "candidate_memories.jsonl")
            self.assertEqual({row["record"]["source_evidence"]["session_id"] for row in rows}, {"minus", "plus"})
            request = next(row for row in read_jsonl(run / "prompts.jsonl")
                           if row["arm"] == "short_plus_structured_candidate")
            prompt = json.dumps(request["messages"])
            self.assertIn("-5 degrees", prompt)
            self.assertIn("+5 degrees", prompt)
            self.assertEqual(len(request["included_memory_ids"]), 2)
            self.assertEqual(len(request["included_source_turn_ids"]), 2)
            protocol = json.loads((run / "protocol.json").read_text())
            self.assertEqual(protocol["protocol_version"], VERSION)
            self.assertIn("duplicate_policy", protocol["preprocessing_policy"])
            self.assertEqual(protocol["preprocessing_policy"], PREPROCESSING_POLICY)
            protocol["protocol_version"] = "structured-candidate-three-arm-v1"
            (run / "protocol.json").write_text(json.dumps(protocol))
            with self.assertRaisesRegex(ValueError, "unsupported structured candidate protocol"):
                load_run(run)


if __name__ == "__main__":
    unittest.main()
