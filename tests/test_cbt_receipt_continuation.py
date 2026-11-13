"""No-network continuation checks against complete invented source/receipt runs."""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"evaluation/cbt-memory-effect"))
from architecture_retest import continuation, runner, memory_assessment as memory
from architecture_retest import receipt_resolver as rr
from architecture_retest.journal import UnknownAttempt,atomic,canonical,digest,rows

IDENTITY={"provider":"https://tokendance.space/gateway/v1","model":"qwen3.5-plus",
    "returned_model":"qwen3.5-plus-2026-04-20","model_family":"qwen"}


class Dense:
    def rank(self,query,documents):
        return list(range(len(documents)))


class FixtureTransport:
    fixture_only=True
    def __init__(self,identity,malformed_support=False):
        self.identity=identity
        self.calls=[]
        self.malformed_support=malformed_support

    def __call__(self,body):
        self.calls.append(body)
        system=body["messages"][0]["content"]
        if system.startswith("Extract only reusable"):
            source=json.loads(body["messages"][-1]["content"])["history"][0]
            good={"item_id":"m-good","section":"intervention_learning","value":"The user reports one practice.",
                "origin":"explicit_user_statement","evidence":[{"source_id":source["source_id"],"quote":source["content"]}]}
            value={"items":[good,{**good,"item_id":"m-bad","evidence":[{"source_id":source["source_id"],"quote":"Not literally present."}]}]}
        elif system.startswith("Independently assess whether"):
            data=json.loads(body["messages"][-1]["content"])
            value={"verdicts":[{"item_id":[] if self.malformed_support else i["item_id"],
                "supported":True,"reason":"Supported by the invented source."} for i in data["items"]]}
        elif system==memory.TARGETS:
            source=json.loads(body["messages"][-1]["content"])["history"][0]
            value={"targets":[{"id":"f0","kind":"required","proposition":"The user reported one practice.",
                "evidence":[{"source_id":source["source_id"],"quote":source["content"]}]}]}
        elif system==memory.VERIFY:
            data=json.loads(body["messages"][-1]["content"])
            value={"verdicts":[{"item_id":t["id"],"supported":True,"reason":"Relevant source-supported fixture."} for t in data["targets"]]}
        elif system==memory.ADJUDICATE:
            value={"status":"answered","claims":[]}
        else:
            value=None
        text=json.dumps(value) if value is not None else "An invented continuation response " + digest(body)[:12] + "."
        return {"text":text,"provider_model":self.identity["returned_model"],"finish_reason":"stop",
            "usage":{"total_tokens":1},"wire_sha256":sha256(json.dumps(body).encode()).hexdigest()}


def old_fixture(root):
    source=[{"case_id":"invented-case","source_family_id":"invented-family","input":"What could I reflect on next?",
        "calendar_time_available":False,"history":[{"source_id":"s"+str(i),"source_order":i,
            "role":"user" if i%2==0 else "assistant","content":"I completed one practice." if i==0 else "An invented dialogue turn "+str(i)+"."}
            for i in range(8)]}]
    natural,assessor=root/"old_reader",root/"old_assessor"
    runner.prepare_case_pool(source,natural,{"archive":"invented fixture"},fixture_only=True)
    memory.prepare(natural,assessor,IDENTITY,fixture_only=True)
    a,q=FixtureTransport(runner.IDENTITY),FixtureTransport(IDENTITY)
    result=runner.run(natural,a,max_new_calls=5,workers=1,dense=Dense(),fixture_only=True,
        assessment_folder=assessor,assessment_transport=q,assessment_max_new_calls=5)
    assert result["arms"]["managed_memory_rag"]["received"]==0
    assert result["arms"]["no_rag"]["received"]==1
    return natural,assessor


def write_rows(path,values):
    path.write_text("".join(canonical(value)+"\n" for value in values))


class ContinuationTests(unittest.TestCase):
    def test_repair_only_missing_managed_reuses_received_and_keeps_old_files(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw); old,assessment=old_fixture(root)
            originals={p:rr.file_hash(p) for directory in (old,assessment) for p in directory.rglob("*") if p.is_file()}
            budget=continuation.prepare(old,assessment,root/"new",fixture_only=True)
            self.assertEqual(budget["planned_cases"],1)
            self.assertEqual(budget["maximum_new_generation_requests"],2)
            self.assertEqual(budget["maximum_new_assertion_requests"],1)
            answer,judge=FixtureTransport(runner.IDENTITY),FixtureTransport(IDENTITY)
            result=continuation.run(root/"new",answer,judge,max_new_calls=2,assessment_max_new_calls=1,
                workers=1,dense=Dense(),fixture_only=True)
            self.assertEqual(len(answer.calls),2)
            self.assertEqual(len(judge.calls),1)
            self.assertTrue(all(value["received"]==1 for value in result["arms"].values()))
            self.assertEqual(result["attempts"],2)
            self.assertEqual(result["inherited_received_receipts"],3)
            self.assertTrue(all(rr.file_hash(path)==value for path,value in originals.items()))
            audit=rr.read_json(root/"new/reader/cases/invented-case/projections.json")
            self.assertEqual(audit["binding_rejected_count"],1)
            self.assertEqual([s["semantic_status"] for s in audit["candidate_statuses"]],["supported","not_eligible"])
            accepted,unbound=runner.verified_answers(root/"new/reader")
            self.assertEqual((len(accepted),unbound),(3,[]))
            for arm in ("no_rag","raw_history_rag"):
                self.assertEqual(rr.read_json(old/"cases/invented-case"/(arm+".answer.json")),
                    rr.read_json(root/"new/reader/cases/invented-case"/(arm+".answer.json")))
            costs=rr.read_json(root/"new/memory/summary.json")["request_accounting"]
            self.assertEqual(costs["inherited_received_requests"],4)
            self.assertEqual(costs["new_attempts"],1)
            # Repeating the complete continuation admits no extra calls.
            continuation.run(root/"new",answer,judge,max_new_calls=0,assessment_max_new_calls=0,
                workers=1,dense=Dense(),fixture_only=True)
            self.assertEqual((len(answer.calls),len(judge.calls)),(2,1))

    def test_exact_frozen_post_answer_unknown_keeps_case_targets_and_other_arms(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw); old,assessment=old_fixture(root)
            intents=rows(assessment/"intents.jsonl")
            raw_answer=rr.read_json(old/"cases/invented-case/raw_history_rag.answer.json")
            intent=next(r for r in intents if r["purpose"]=="memory_adjudication" and
                json.loads(r["payload"]["messages"][-1]["content"])["response"]==raw_answer["response"])
            outcomes=rows(assessment/"outcomes.jsonl")
            for row in outcomes:
                if row["key"]==intent["key"]:
                    row.update(status="unknown",text=None)
            write_rows(assessment/"outcomes.jsonl",outcomes)
            allowed={intent["key"]:{"case_id":intent["case_id"],"wire_sha256":intent["wire_sha256"]}}
            with patch.dict(rr.FROZEN_ASSESSMENTS,allowed,clear=True):
                budget=continuation.prepare(old,assessment,root/"new",fixture_only=True)
                self.assertEqual(budget["inherited_frozen_unavailable_assessments"],1)
                a,q=FixtureTransport(runner.IDENTITY),FixtureTransport(IDENTITY)
                result=continuation.run(root/"new",a,q,max_new_calls=2,assessment_max_new_calls=1,
                    workers=1,dense=Dense(),fixture_only=True)
                self.assertTrue(all(value["received"]==1 for value in result["arms"].values()))
                unavailable=rr.read_json(root/"new/memory/assessor_only/invented-case/raw_history_rag.unavailable.json")
                self.assertEqual(unavailable["assessor_request_key"],intent["key"])
                self.assertFalse(unavailable["resend_permitted"])
                self.assertTrue((root/"new/memory/assessor_only/invented-case/targets.json").exists())
                self.assertFalse((root/"new/memory/assessor_only/invented-case/raw_history_rag.score.json").exists())
                self.assertEqual(len(q.calls),1)
                self.assertFalse(any(digest({"purpose":"memory_adjudication","payload":p})==intent["key"] for p in q.calls))
                summary=rr.read_json(root/"new/memory/summary.json")
                self.assertEqual(summary["frozen_target_cases"],1)
                self.assertEqual(summary["arms"]["raw_history_rag"]["planned_answers"],1)
                self.assertEqual(summary["arms"]["raw_history_rag"]["scored_answers"],0)
                self.assertEqual(summary["inherited_frozen_unavailable_requests"],1)

    def test_other_unknown_remains_fail_closed_without_new_directory(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw); old,assessment=old_fixture(root)
            outcomes=rows(assessment/"outcomes.jsonl")
            outcomes[0]["status"]="unknown"
            write_rows(assessment/"outcomes.jsonl",outcomes)
            with self.assertRaises(UnknownAttempt):
                continuation.prepare(old,assessment,root/"new",fixture_only=True)
            self.assertFalse((root/"new").exists())

    def test_target_file_must_reconstruct_from_genuine_pre_answer_receipts(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw); old,assessment=old_fixture(root)
            path=assessment/"assessor_only/invented-case/targets.json"
            frozen=rr.read_json(path)
            frozen["targets"][0]["proposition"]="A fabricated replacement target."
            atomic(path,frozen)
            with self.assertRaisesRegex(ValueError,"frozen labels differ"):
                continuation.prepare(old,assessment,root/"new",fixture_only=True)

    def test_modified_original_journal_fails_before_any_new_send(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw); old,assessment=old_fixture(root)
            continuation.prepare(old,assessment,root/"new",fixture_only=True)
            with (old/"outcomes.jsonl").open("a") as stream:
                stream.write("\n")
            a,q=FixtureTransport(runner.IDENTITY),FixtureTransport(IDENTITY)
            with self.assertRaisesRegex(ValueError,"immutable journal file changed"):
                continuation.run(root/"new",a,q,max_new_calls=2,assessment_max_new_calls=1,fixture_only=True,dense=Dense())
            self.assertEqual((a.calls,q.calls),([],[]))

    def test_malformed_new_support_is_local_failure_with_candidate_audit(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw); old,assessment=old_fixture(root)
            continuation.prepare(old,assessment,root/"new",fixture_only=True)
            a,q=FixtureTransport(runner.IDENTITY,malformed_support=True),FixtureTransport(IDENTITY)
            result=continuation.run(root/"new",a,q,max_new_calls=2,assessment_max_new_calls=1,
                fixture_only=True,dense=Dense(),workers=1)
            self.assertEqual(result["completed_cases"],1)
            self.assertEqual(result["arms"]["managed_memory_rag"]["received"],0)
            self.assertEqual(len(a.calls),1)
            audit=rr.read_json(root/"new/reader/cases/invented-case/projections.json")
            self.assertEqual(audit["candidate_statuses"][0]["semantic_status"],"unavailable")
            self.assertEqual(audit["candidate_statuses"][1]["semantic_status"],"not_eligible")

    def test_quality_reader_uses_same_inherited_receipt_resolver(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw); old,assessment=old_fixture(root)
            continuation.prepare(old,assessment,root/"new",fixture_only=True)
            continuation.run(root/"new",FixtureTransport(runner.IDENTITY),FixtureTransport(IDENTITY),
                max_new_calls=2,assessment_max_new_calls=1,fixture_only=True,dense=Dense(),workers=1)
            spec=importlib.util.spec_from_file_location("continuation_test_quality",ROOT/"evaluation/architecture-rerun/quality_span_judge.py")
            quality=importlib.util.module_from_spec(spec);spec.loader.exec_module(quality)
            result=quality.prepare(root/"new/reader",root/"quality",IDENTITY,fixture_only=True)
            self.assertEqual((result["full_grid_primary"],result["available_response_packets"]),(3,3))
            self.assertEqual(result["planned_judge_requests"],4)
            manifest=rr.read_json(root/"new/reader/receipt_import.json")
            self.assertEqual(manifest["source_protocol"]["protocol_sha256"],rr.read_json(old/"protocol.json")["protocol_sha256"])
            # An imported old receipt is never placed in the new paid journal.
            own={r["key"] for r in rows(root/"new/reader/intents.jsonl")}
            self.assertTrue(own.isdisjoint({e["intent"]["key"] for e in manifest["entries"]}))


if __name__=="__main__":
    unittest.main()
