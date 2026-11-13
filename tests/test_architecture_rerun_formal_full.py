"""Tiny invented generic QA fixtures; no corpus, Colab emulation or paid call."""
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("formal_new_readout",ROOT/"evaluation/architecture-rerun/formal_full.py")
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class LexicalFixture:
    def rank(self,query,documents):
        return list(range(len(documents)))


def history_row(index,text,date="2026/01/01",session="old"):
    return {"source_id":"invented:"+str(index),"source_order":index,"session_id":session,
            "role":"context","source_speaker":"Aster","content":text,"date":date}


def candidate(item_id,source,value):
    return {"item_id":item_id,"section":"case_state","origin":"benchmark_source",
        "value":{"subject":"Aster","attribute":"residence","reported_value":value,
                 "temporal_scope":"historical_source_report","source_dates":[source["source_date"]]},
        "evidence":[{"source_id":source["source_id"],"text":source["content"],"start":0,"end":len(source["content"])}]}


class FormalFullReadoutTest(unittest.TestCase):
    def test_mandatory_preflight_keeps_full_roster_and_blocks_oversized_current_before_paid_preparation(self):
        good={"case_id":"invented-fit","source_family_id":"invented-family","history_key":"empty",
              "input":"What was reported?","query_date":"2026/01/02","source_cutoff_order_inclusive":-1}
        oversized={**good,"case_id":"invented-overflow","input":"x"*runner.SETTINGS["answer_chars"]}
        result=runner.mandatory_packet_preflight([good,oversized],{"empty":{"original_history":[],"sources":[]}})
        self.assertEqual((result["planned_cases"],result["fit_cases"],result["blocked_cases"]),(2,1,1))
        self.assertFalse(result["all_fit"])
        self.assertEqual(result["model_calls"],0)
        self.assertEqual([r["case_id"] for r in result["rows"]],[good["case_id"],oversized["case_id"]])
        packet=runner.assemble(good,[],[],[],[],"raw")
        self.assertEqual(result["rows"][0]["wire_chars"],len(json.dumps(packet["payload"])))
        self.assertEqual(result["rows"][0]["wire_sha256"],runner.sha256(json.dumps(packet["payload"]).encode()).hexdigest())

    def test_partition_keeps_all_characters_and_every_chunk_has_full_wire_budget(self):
        original=history_row(0,"Invented context 数据。 "*700)
        sources=runner.source_fragments([original])
        self.assertGreater(len(sources),1)
        self.assertEqual("".join(s["content"] for s in sources),original["content"])
        for source in sources:
            self.assertEqual(source["content"],original["content"][source["parent_start"]:source["parent_end"]])
            self.assertEqual(source["parent_source_id"],original["source_id"])
            self.assertEqual(source["source_speaker"],"Aster")
        chunks=runner.source_chunks(sources)
        self.assertEqual([s["source_id"] for c in chunks for s in c],[s["source_id"] for s in sources])
        for chunk in chunks:
            packet=runner.payload(runner.p.request(runner.EXTRACT,{"sources":chunk}),"extraction")
            self.assertLessEqual(len(json.dumps(packet)),runner.SETTINGS["extraction_chunk_wire_chars"])
            self.assertNotIn("current_question",json.dumps(packet))

    def test_cross_chunk_versions_remain_historical_and_processed_values_reach_actual_messages(self):
        history=[history_row(0,"I live in Maple.","2026/01/01"),
                 history_row(1,"I moved from Maple to Cedar.","2026/02/01")]
        history += [history_row(i,f"Invented filler turn {i}.","2026/03/01","recent") for i in range(2,24)]
        sources=runner.source_fragments(history)
        groups=runner.historical_version_groups([candidate("chunk1:m0",sources[0],"reported living in Maple"),
            candidate("chunk2:m0",sources[1],"reported moving from Maple to Cedar")],sources)
        self.assertEqual(len(groups),1)
        self.assertEqual(len(groups[0]["value"]["historical_versions"]),2)
        self.assertEqual({e["source_id"] for e in groups[0]["evidence"]},{sources[0]["source_id"],sources[1]["source_id"]})
        case={"case_id":"invented-question","source_family_id":"invented-family","input":"Where did Aster move?",
              "query_date":"2026/03/02","source_cutoff_order_inclusive":23}
        packets=runner.prepare_case_requests(case,{"original_history":history,"sources":sources},groups,LexicalFixture(),threading.Lock())
        self.assertEqual(set(packets),set(runner.ARMS))
        managed=packets["managed_memory_rag"]
        dynamic=json.loads(managed["payload"]["messages"][-2]["content"].split("\n",1)[1])
        self.assertIn("historical_versions",dynamic["case_state"][0]["value"])
        self.assertIn("reported moving from Maple to Cedar",json.dumps(dynamic))
        self.assertIn("reported living in Maple",json.dumps(dynamic))
        self.assertEqual(len(managed["trace"]["fragment_parent_mapping"]),2)
        none=json.loads(packets["no_rag"]["payload"]["messages"][-1]["content"])
        self.assertTrue(none["recent_source_reports"])
        raw=json.loads(packets["raw_history_rag"]["payload"]["messages"][-2]["content"].split("\n",1)[1])
        self.assertTrue(all(s["source_speaker"]=="Aster" for s in raw["source_reports"]))
        for packet in packets.values():
            messages=packet["payload"]["messages"]
            self.assertEqual([m["role"] for m in messages],["system","user","user"])
            self.assertEqual(sum(case["input"] in m["content"] for m in messages),1)
            self.assertLessEqual(len(json.dumps(packet["payload"])),runner.SETTINGS["answer_chars"])

    def test_fabricated_projection_date_is_rejected_before_support_call(self):
        source=runner.source_fragments([history_row(0,"I moved to Cedar.")])[0]
        item=candidate("m0",source,"reported moving to Cedar")
        item["evidence"]=[{"source_id":source["source_id"],"quote":source["content"]}]
        item["value"]["source_dates"]=["2099/12/31"]
        with self.assertRaisesRegex(ValueError,"date"):
            runner._generic_candidates(json.dumps({"items":[item]}),[source])

    def test_derived_cache_and_answer_mutations_are_rejected_by_receipt_reconstruction(self):
        history=[history_row(0,"I live in Cedar.")]
        sources=runner.source_fragments(history)
        item=candidate("m0",sources[0],"reported living in Cedar")
        item["evidence"]=[{"source_id":sources[0]["source_id"],"quote":sources[0]["content"]}]
        class Receipts:
            def call(self,case,purpose,payload):
                text=json.dumps({"items":[item]}) if purpose=="extraction" else json.dumps({"verdicts":[
                    {"item_id":"m0","supported":True,"reason":"invented supported historical report"}]})
                if purpose=="answer":
                    text="Cedar."
                return {"key":runner.digest({"purpose":purpose,"payload":payload}),"text":text,
                        "provider_model":runner.IDENTITY["returned_model"],"usage":{"total_tokens":2}}
        journal=Receipts()
        chunk={"case_id":"preparation:invented","chunk_id":"invented","sources":sources}
        case={"case_id":"invented-question","source_family_id":"invented-family","input":"Where did Aster live?",
              "query_date":"2026/01/02","source_cutoff_order_inclusive":0}
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            (folder/"prepared_chunks").mkdir()
            result=runner._prepare_chunk(chunk,folder,journal)
            saved=folder/"prepared_chunks/invented.json"
            changed=json.loads(saved.read_text())
            changed["accepted"][0]["value"]["reported_value"]="fabricated relocation"
            saved.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,"exact extraction/verifier receipts"):
                runner._prepare_chunk(chunk,folder,journal)
            saved.write_text(json.dumps(result))
            groups=runner.historical_version_groups(result["accepted"],sources)
            inventory={"version_groups":groups,"failed_chunks":[],"automatic_support_assessment":True,"all_versions_preserved":True}
            dest=folder/"cases"/runner.digest(case["case_id"])[:32]
            dest.mkdir(parents=True)
            (dest/"memory_projection_inventory.json").write_text(json.dumps(inventory))
            packets=runner.prepare_case_requests(case,{"original_history":history,"sources":sources},groups,LexicalFixture(),threading.Lock())
            for arm,packet in packets.items():
                (dest/(arm+".request.json")).write_text(json.dumps(packet))
                receipt=journal.call(case,"answer",packet["payload"])
                answer={"case_id":case["case_id"],"arm":arm,"source_family_id":case["source_family_id"],
                    "status":"received","response":receipt["text"],"request_key":receipt["key"],
                    "provider_model":receipt["provider_model"],"usage":receipt["usage"]}
                (dest/(arm+".answer.json")).write_text(json.dumps(answer))
            record={"original_history":history,"sources":sources}
            self.assertEqual(len(runner._validate_case_outputs(case,record,folder,journal,inventory)),3)
            answer_path=dest/"raw_history_rag.answer.json"
            changed=json.loads(answer_path.read_text())
            changed["response"]="fabricated answer"
            answer_path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,"exact formal request/receipt"):
                runner._validate_case_outputs(case,record,folder,journal,inventory)

    def test_budget_rejections_remain_in_external_selection_diagnostics(self):
        history=[history_row(0,"Aster reported a project.")]
        sources=runner.source_fragments(history)
        item=candidate("large",sources[0],"long invented processed report "*1000)
        case={"case_id":"invented-budget","source_family_id":"invented-family","input":"What project?",
              "query_date":"2026/01/02","source_cutoff_order_inclusive":0}
        packet=runner.prepare_case_requests(case,{"original_history":history,"sources":sources},[item],LexicalFixture(),threading.Lock())["managed_memory_rag"]
        self.assertEqual(packet["trace"]["selection_diagnostics"]["selected_item_ids"],[])
        self.assertIn("large",packet["trace"]["selection_diagnostics"]["omitted"])

    def test_full_fixture_controller_resumes_complete_without_another_send(self):
        history=[history_row(0,"I live in Cedar.")]
        sources=runner.source_fragments(history)
        item=candidate("m0",sources[0],"reported living in Cedar")
        item["evidence"]=[{"source_id":sources[0]["source_id"],"quote":sources[0]["content"]}]
        cid=runner.digest(sources)
        case={"case_id":"invented-controller","source_family_id":"invented-family","input":"Where does Aster report living?",
              "query_date":"2026/01/02","source_cutoff_order_inclusive":0,"history_key":"invented"}
        class Transport:
            identity=runner.IDENTITY
            calls=0
            def __call__(self,payload):
                self.calls+=1
                system=payload["messages"][0]["content"]
                text=json.dumps({"items":[item]}) if system==runner.EXTRACT else json.dumps({"verdicts":[
                    {"item_id":"m0","supported":True,"reason":"invented supported historical report"}]})
                if system not in (runner.EXTRACT,runner.VERIFY):
                    text="Cedar."
                return {"text":text,"provider_model":runner.IDENTITY["returned_model"],"finish_reason":"stop",
                        "wire_sha256":runner.sha256(json.dumps(payload,allow_nan=False).encode()).hexdigest(),
                        "usage":{"total_tokens":2}}
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            cases=[case]
            protocol={"provider":runner.IDENTITY,"case_count":1,"planned_answers":3,"preparation_chunks":1,
                "dataset_key":"invented","maximum_attempts":5,"cases_sha256":runner.digest(cases),
                "request_limits":{"answer":24000,"preparation":48000}}
            protocol["protocol_sha256"]=runner.digest(protocol)
            (folder/"protocol.json").write_text(json.dumps(protocol))
            (folder/"cases.json").write_text(json.dumps(cases))
            (folder/"histories.json").write_text(json.dumps({"invented":{"original_history":history,"sources":sources,"chunk_ids":[cid]}}))
            (folder/"chunks.json").write_text(json.dumps([{"case_id":"preparation:"+cid,"chunk_id":cid,"sources":sources}]))
            transport=Transport()
            with patch.object(runner,"load",return_value=(folder,protocol,cases)):
                summary=runner.run(folder,transport,max_new_calls=5,dense=LexicalFixture())
                self.assertTrue(summary["complete"])
                self.assertEqual({v["received"] for v in summary["arms"].values()},{1})
                first_calls=transport.calls
                summary=runner.run(folder,transport,max_new_calls=0,dense=LexicalFixture())
                self.assertTrue(summary["complete"])
                self.assertEqual(transport.calls,first_calls)

    def test_read_only_audit_preserves_invalid_receipt_failure_and_unknown_distinction(self):
        history=[history_row(0,"I live in Cedar.")]
        sources=runner.source_fragments(history)
        chunk={"case_id":"preparation:invented-invalid","chunk_id":"invented-invalid","sources":sources}
        request=runner.payload(runner.p.request(runner.EXTRACT,{"sources":sources}),"extraction")
        key=runner.digest({"purpose":"extraction","payload":request})
        wire=json.dumps(request)
        protocol={"protocol_sha256":"invented-protocol","maximum_attempts":2,"provider":runner.IDENTITY}
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            (folder/"prepared_chunks").mkdir()
            intent={"key":key,"purpose":"extraction","payload":request,"serialized_request":wire,
                "wire_sha256":runner.sha256(wire.encode()).hexdigest(),"wire_chars":len(wire),
                "protocol_sha256":"invented-protocol"}
            (folder/"intents.jsonl").write_text(json.dumps(intent)+"\n")
            outcome={"key":key,"status":"invalid","protocol_sha256":"invented-protocol"}
            (folder/"outcomes.jsonl").write_text(json.dumps(outcome)+"\n")
            audit=runner.ReceiptAudit(folder,protocol)
            result=runner._prepare_chunk(chunk,folder,audit)
            self.assertEqual(result["status"],"preparation_failed")
            self.assertEqual(runner._prepare_chunk(chunk,folder,audit,persist=False),result)
            outcome["status"]="unknown"
            (folder/"outcomes.jsonl").write_text(json.dumps(outcome)+"\n")
            audit=runner.ReceiptAudit(folder,protocol)
            with self.assertRaises(runner.UnknownAttempt):
                runner._prepare_chunk(chunk,folder,audit,persist=False)


if __name__=="__main__":
    unittest.main()
