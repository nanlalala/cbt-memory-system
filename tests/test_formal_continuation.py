"""Invented receipt fixtures: no benchmark data, runtime emulation or API calls."""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"evaluation/architecture-rerun"))
import formal_continuation as continuation
import formal_full as full
import formal_judge as judge


class LexicalFixture:
    def rank(self, query, documents):
        return list(range(len(documents)))


class AnswerTransport:
    fixture_only = True
    identity = full.IDENTITY

    def __init__(self):
        self.calls, self.fail_next = 0, False

    def __call__(self, payload):
        self.calls += 1
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("invented timeout after send")
        system = payload["messages"][0]["content"]
        if system == full.EXTRACT:
            source = json.loads(payload["messages"][-1]["content"])["sources"][0]
            text = json.dumps({"items":[{"item_id":"m0", "section":"case_state", "origin":"benchmark_source",
                "value":{"subject":"Aster", "attribute":"residence", "reported_value":"reported living in Cedar",
                    "temporal_scope":"historical_source_report", "source_dates":[source["source_date"]]},
                "evidence":[{"source_id":source["source_id"], "quote":source["content"]}]}]})
        elif system == full.VERIFY:
            text = json.dumps({"verdicts":[{"item_id":"m0", "supported":True, "reason":"invented supported report"}]})
        else:
            text = "Cedar."
        return {"text":text, "provider_model":self.identity["returned_model"], "finish_reason":"stop",
            "wire_sha256":sha256(json.dumps(payload, allow_nan=False).encode()).hexdigest(),
            "usage":{"prompt_tokens":10, "completion_tokens":1, "total_tokens":11}}


class JudgeTransport:
    fixture_only = True
    identity = {"provider":"https://tokendance.space/gateway/v1", "model":"independent-fixture",
                "returned_model":"independent-fixture-returned", "model_family":"qwen"}

    def __init__(self):
        self.calls = 0

    def __call__(self, payload):
        self.calls += 1
        return {"text":"yes", "provider_model":self.identity["returned_model"], "finish_reason":"stop",
            "wire_sha256":sha256(json.dumps(payload, allow_nan=False).encode()).hexdigest(), "usage":{}}


def parent_fixture(folder):
    folder.mkdir()
    history = [{"source_id":"invented-old", "source_order":0, "session_id":"old", "role":"context",
                "source_speaker":"Aster", "content":"I live in Cedar.", "date":"2026/01/01"}]
    history += [{"source_id":"invented-recent-"+str(i), "source_order":i+1, "session_id":"recent", "role":"context",
        "source_speaker":"Aster", "content":"Invented recent unrelated turn "+str(i), "date":"2026/02/01"} for i in range(17)]
    sources = full.source_fragments(history)
    cid = full.digest(sources)
    chunks = [{"case_id":"preparation:"+cid, "chunk_id":cid, "sources":sources}]
    cases = [{"case_id":"invented-case-"+str(i), "history_key":"invented-history", "source_family_id":"invented-family",
        "input":"Where did Aster report living"+"?"*(i+1), "query_date":"2026/02/02", "source_cutoff_order_inclusive":17} for i in range(2)]
    histories = {"invented-history":{"history_key":"invented-history", "original_history":history, "sources":sources, "chunk_ids":[cid]}}
    refs = [{"case_id":c["case_id"], "query":c["input"], "reference_answer":"Cedar", "category":"1",
        "should_abstain":False} for c in cases]
    for name, value in (("cases.json",cases), ("histories.json",histories), ("chunks.json",chunks),
            ("references.json",refs), ("upstream_source_manifest.json",{}),
            ("mandatory_packet_preflight.json",full.mandatory_packet_preflight(cases,histories))):
        full.atomic(folder/name,value)
    protocol = {"version":full.VERSION, "provider":full.IDENTITY, "case_count":2, "arms":list(full.ARMS),
        "planned_answers":6, "preparation_chunks":1, "dataset_key":"locomo10", "maximum_attempts":8,
        "settings":full.SETTINGS, "code_sha256":full.code_hashes(), "cases_sha256":full.digest(cases),
        "request_limits":{"answer":24000,"preparation":48000}, "mandatory_packet_preflight_passed":True,
        "files_sha256":{name:continuation._hash(folder/name) for name in continuation.STATIC_FILES if name!="cases.json"}}
    protocol["protocol_sha256"] = full.digest(protocol)
    full.atomic(folder/"protocol.json",protocol)
    (folder/"prepared_chunks").mkdir()
    transport = AnswerTransport()
    journal = full.Journal(folder,protocol,transport,8)
    prepared = full._prepare_chunk(chunks[0],folder,journal)
    inventory = continuation._inventory(histories["invented-history"],{cid:prepared})
    for index,case in enumerate(cases):
        dest = folder/"cases"/full.digest(case["case_id"])[:32]
        dest.mkdir(parents=True)
        full.atomic(dest/"memory_projection_inventory.json",inventory)
        packets = full.prepare_case_requests(case,histories["invented-history"],inventory["version_groups"],LexicalFixture(),threading.Lock())
        results = []
        for arm in full.ARMS:
            packet = packets[arm]
            packet["trace"]["preparation_coverage"] = {"planned_chunks":1,"failed_chunk_ids":[], "prepared_chunks":1,"partial_supported_memory":False}
            full.atomic(dest/(arm+".request.json"),packet)
            if index == 1:
                if arm == "no_rag":
                    transport.fail_next = True
                    try:
                        journal.call(case,"answer",packet["payload"])
                    except full.UnknownAttempt:
                        pass
                continue
            receipt = journal.call(case,"answer",packet["payload"])
            answer = {"case_id":case["case_id"], "arm":arm,"source_family_id":case["source_family_id"],
                "status":"received", "response":receipt["text"], "request_key":receipt["key"],
                "provider_model":receipt["provider_model"], "usage":receipt["usage"]}
            full.atomic(dest/(arm+".answer.json"),answer)
            results.append(answer)
        if index == 0:
            full.atomic(dest/"completed.json",{"case_id":case["case_id"], "rows":results,"failed_preparation_chunks":[]})
    return cases, transport.calls


class ContinuationTest(unittest.TestCase):
    def test_source_only_recount_has_no_paid_authorization_and_keeps_actual_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)/"parent"
            parent_fixture(parent)
            result = continuation.inspect_source_only(parent,fixture_only=True)
            self.assertEqual(result["planned_cases"],2)
            self.assertEqual(result["preparation_chunks"],1)
            self.assertEqual(result["maximum_attempts_from_actual_sources"],8)
            self.assertEqual(result["maximum_independent_judge_calls"],7)
            self.assertEqual(result["model_calls"],0)
            self.assertFalse(result["paid_execution_authorized"])

    def test_additive_v2_compatibility_never_allows_a_v1_change(self):
        name = "modules/agent-integration/cbt_projection_contract.py"
        original = (ROOT/name).read_text()
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp)/name
            copy.parent.mkdir(parents=True)
            copy.write_text(original)
            with patch.object(full,"ROOT",Path(tmp)):
                self.assertEqual(continuation._v1_ast_compatibility(name,continuation.V1_AST[name][0]),continuation.V1_AST[name][1])
                copy.write_text(original.replace('len(value["items"]) > 16','len(value["items"]) > 15',1))
                with self.assertRaisesRegex(ValueError,"V1 projection behavior changed"):
                    continuation._v1_ast_compatibility(name,continuation.V1_AST[name][0])

    def test_quarantined_send_is_never_retried_and_receipts_keep_parent_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, output = Path(tmp)/"parent", Path(tmp)/"new"
            cases, _ = parent_fixture(parent)
            original = continuation._snapshot(parent)
            protocol = continuation.prepare(parent,output,fixture_only=True)
            self.assertEqual(protocol["inherited_received_receipts"],5)
            self.assertEqual(protocol["inherited_preparation_receipts"],2)
            self.assertEqual(protocol["inherited_answer_rows"],3)
            self.assertEqual(protocol["maximum_attempts"],2)
            transport = AnswerTransport()
            with patch.object(full,"DenseRetriever",side_effect=AssertionError("no embedding during status-only check")):
                status = continuation.run(output,transport,max_new_calls=0,fixture_only=True)
            self.assertFalse(status["complete"])
            self.assertEqual(transport.calls,0)
            summary = continuation.run(output,transport,max_new_calls=1,dense=LexicalFixture(),fixture_only=True)
            self.assertFalse(summary["complete"])
            self.assertEqual(transport.calls,1)
            summary = continuation.run(output,transport,max_new_calls=2,dense=LexicalFixture(),fixture_only=True)
            self.assertTrue(summary["complete"])
            self.assertEqual(transport.calls,2)
            self.assertEqual(summary["received_answers"],5)
            self.assertEqual(summary["technical_missing_answers"],1)
            self.assertEqual(summary["quarantined_unknown_attempts"],1)
            self.assertFalse(summary["all_provider_answers_received"])
            with patch.object(full,"DenseRetriever",side_effect=AssertionError("completed run needs no embedding")):
                continuation.run(output,transport,max_new_calls=10,fixture_only=True)
            self.assertEqual(transport.calls,2)
            self.assertEqual(continuation._snapshot(parent),original)
            parent_hash = continuation._json(parent/"protocol.json")["protocol_sha256"]
            self.assertTrue(all(r["protocol_sha256"]==parent_hash for r in full.rows(output/"inherited/outcomes.jsonl")))
            self.assertTrue(all(r["protocol_sha256"]==protocol["protocol_sha256"] for r in full.rows(output/"outcomes.jsonl")))
            self.assertFalse(set(r["key"] for r in full.rows(output/"intents.jsonl")) & set(r["key"] for r in full.rows(output/"inherited/intents.jsonl")))

    def test_existing_grader_uses_original_denominator_and_missing_is_a_technical_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, output, graded = Path(tmp)/"parent", Path(tmp)/"new", Path(tmp)/"grade"
            parent_fixture(parent)
            continuation.prepare(parent,output,fixture_only=True)
            with self.assertRaisesRegex(ValueError,"reader is incomplete"):
                judge.prepare(output,graded,JudgeTransport.identity,fixture_only=True)
            continuation.run(output,AnswerTransport(),max_new_calls=2,dense=LexicalFixture(),fixture_only=True)
            frozen = judge.prepare(output,graded,JudgeTransport.identity,fixture_only=True)
            self.assertEqual(frozen["planned_answers"],6)
            self.assertEqual(frozen["primary_grading_rows"],5)
            self.assertEqual(frozen["repeat_rows"],1)
            self.assertEqual(frozen["maximum_attempts"],6)
            summary = judge.run(graded,JudgeTransport(),max_new_calls=6,fixture_only=True)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["scored_denominator"],6)
            self.assertEqual(summary["conditions"]["no_rag"]["question_micro_accuracy"],0.5)
            self.assertEqual(summary["conditions"]["raw_history_rag"]["question_micro_accuracy"],1)

    def test_inherited_or_quarantine_mutation_blocks_before_any_send(self):
        for mutation in ("receipt", "failure", "request"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                parent, output = Path(tmp)/"parent", Path(tmp)/"new"
                cases, _ = parent_fixture(parent)
                continuation.prepare(parent,output,fixture_only=True)
                if mutation == "receipt":
                    path = output/"inherited/outcomes.jsonl"
                    path.write_text(path.read_text().replace("Cedar.","Fabricated."))
                elif mutation == "request":
                    path = next((output/"cases").glob("*/*.answer.json"))
                    value = continuation._json(path)
                    value["response"] = "Fabricated."
                    path.write_text(json.dumps(value))
                else:
                    path = output/"cases"/full.digest(cases[1]["case_id"])[:32]/"no_rag.failure.json"
                    value = continuation._json(path)
                    value["request_key"] = "invented-unbound-key"
                    path.write_text(json.dumps(value))
                transport = AnswerTransport()
                with self.assertRaises(ValueError):
                    # A pre-admission audit must reject even a corrupted
                    # quarantine marker in a case without a completion marker.
                    continuation.run(output,transport,max_new_calls=2,dense=LexicalFixture(),fixture_only=True)
                self.assertEqual(transport.calls,0)

    def test_new_unknown_stops_resume_and_cannot_join_the_original_quarantine(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, output = Path(tmp)/"parent", Path(tmp)/"new"
            parent_fixture(parent)
            continuation.prepare(parent,output,fixture_only=True)
            transport = AnswerTransport()
            transport.fail_next = True
            summary = continuation.run(output,transport,max_new_calls=2,dense=LexicalFixture(),fixture_only=True)
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["new_unknown_receipts"],1)
            calls = transport.calls
            with self.assertRaises(full.UnknownAttempt):
                continuation.run(output,transport,max_new_calls=2,dense=LexicalFixture(),fixture_only=True)
            self.assertEqual(transport.calls,calls)

    def test_relabeling_parent_receipt_as_new_is_rejected_before_send(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, output = Path(tmp)/"parent", Path(tmp)/"new"
            parent_fixture(parent)
            protocol = continuation.prepare(parent,output,fixture_only=True)
            intent = full.rows(parent/"intents.jsonl")[0]
            receipt = next(r for r in full.rows(parent/"outcomes.jsonl") if r["key"]==intent["key"])
            intent["protocol_sha256"] = receipt["protocol_sha256"] = protocol["protocol_sha256"]
            (output/"intents.jsonl").write_text(full.canonical(intent)+"\n")
            (output/"outcomes.jsonl").write_text(full.canonical(receipt)+"\n")
            transport = AnswerTransport()
            with self.assertRaisesRegex(ValueError,"original request was resent"):
                continuation.run(output,transport,max_new_calls=2,dense=LexicalFixture(),fixture_only=True)
            self.assertEqual(transport.calls,0)

    def test_rehashed_larger_cap_still_fails_derived_remaining_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, output = Path(tmp)/"parent", Path(tmp)/"new"
            parent_fixture(parent)
            continuation.prepare(parent,output,fixture_only=True)
            protocol = continuation._json(output/"protocol.json")
            protocol["maximum_attempts"] += 1
            protocol["protocol_sha256"] = full.digest({k:v for k,v in protocol.items() if k!="protocol_sha256"})
            full.atomic(output/"protocol.json",protocol)
            transport = AnswerTransport()
            with self.assertRaisesRegex(ValueError,"provenance or cap differs"):
                continuation.run(output,transport,max_new_calls=3,dense=LexicalFixture(),fixture_only=True)
            self.assertEqual(transport.calls,0)

    def test_fixture_mode_cannot_be_used_with_a_real_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent, output = Path(tmp)/"parent", Path(tmp)/"new"
            parent_fixture(parent)
            continuation.prepare(parent,output,fixture_only=True)
            transport = AnswerTransport()
            transport.fixture_only = False
            with self.assertRaisesRegex(ValueError,"cannot invoke a real provider"):
                continuation.run(output,transport,max_new_calls=2,dense=LexicalFixture(),fixture_only=True)
            self.assertEqual(transport.calls,0)


if __name__ == "__main__":
    unittest.main()
