"""Invented input contracts for the hosted replication, without real models."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"evaluation/cbt-memory-effect"))
from architecture_retest import projections as p
from architecture_retest import runner
from architecture_retest import memory_assessment
from architecture_retest.journal import Journal,UnknownAttempt,atomic,digest
import cactus_materials


class FakeDense:
    def rank(self,query,documents):
        return list(range(len(documents)))


class RetestContract(unittest.TestCase):
    def test_binding_and_verdict_are_separate_and_hypotheses_stay_tentative(self):
        sources=[dict(source_id="s0",role="user",content="I completed one thought record.")]
        item=dict(item_id="m0",section="intervention_learning",value="The user reports one completed record.",
                  origin="explicit_user_statement",evidence=[dict(source_id="s0",quote=sources[0]["content"])])
        bound=p.bind_candidates(json.dumps({"items":[item]}),sources)
        self.assertEqual(bound[0]["evidence"][0]["start"],0)
        self.assertEqual(p.checked_items('{"verdicts":[{"item_id":"m0","supported":false,"reason":"overstated"}]}',bound)[0],[])
        with self.assertRaises(ValueError):
            p.checked_items('{"verdicts":[]}',bound)
        item["origin"]="agent_inference"
        with self.assertRaises(ValueError):
            p.bind_candidates(json.dumps({"items":[item]}),sources)

    def test_duplicate_concurrent_requests_share_receipt_and_unknown_blocks_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            cases=[{"case_id":"invented:case"}]
            atomic(folder/"cases.json",cases)
            identity={"provider":"fixture","model":"request-model","returned_model":"actual-alias","model_family":"fixture"}
            protocol={"provider":identity,"cases_sha256":digest(cases),"maximum_attempts":2,
                      "request_limits":{"answer":1000,"preparation":1000}}
            protocol["protocol_sha256"]=digest(protocol)
            class Transport:
                identity=protocol["provider"]
                calls=0
                def __call__(self,payload):
                    self.calls+=1
                    time.sleep(.01)
                    return dict(text="ok",provider_model="actual-alias",finish_reason="stop",usage={"total_tokens":2},
                                wire_sha256=sha256(json.dumps(payload).encode()).hexdigest())
            t=Transport()
            journal=Journal(folder,protocol,t,2)
            body={"model":"request-model","messages":[]}
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(journal.call,cases[0],"answer",body) for _ in range(2)]
                outcomes=[f.result() for f in futures]
            self.assertEqual(t.calls,1)
            self.assertEqual(outcomes[0],outcomes[1])
            self.assertEqual(outcomes[0]["finish_reason"],"stop")
            again=Journal(folder,protocol,t,0)
            self.assertEqual(again.call(cases[0],"answer",body)["text"],"ok")
            class Broken(Transport):
                def __call__(self,payload):
                    raise TimeoutError("unknown provider send")
            j=Journal(folder,protocol,Broken(),1)
            with self.assertRaises(UnknownAttempt):
                j.call(cases[0],"answer",{"messages":["different"]})
            with self.assertRaises(UnknownAttempt):
                Journal(folder,protocol,t,2)

    def test_three_conditions_share_the_actual_complete_cbt_policy_and_recent_dialogue(self):
        compiled=cactus_materials.compile_rows(cactus_materials.invented_fixture_rows(),family_limit=1,fixture_only=True)
        case=compiled["generation"][-1]
        sources=runner.source_views(case)
        selected=sources[0]
        item=dict(item_id="m",section="case_state",value="A deliberately processed fixture representation.",
                  origin="benchmark_source",evidence=[dict(source_id=selected["source_id"],start=0,
                      end=len(selected["content"]),text=selected["content"])])
        plans=runner.prepare_case_requests(case,sources,[item],FakeDense(),__import__('threading').Lock())
        self.assertEqual(set(plans),set(runner.ARMS))
        for row in plans.values():
            self.assertIn("payload",row)
            self.assertEqual(row["payload"]["messages"][-1],{"role":"user","content":case["input"]})
            self.assertNotIn('2000-01-',json.dumps(row["payload"]))
        self.assertEqual(tuple(runner.ARMS),("no_rag","raw_history_rag","managed_memory_rag"))
        api,_=runner._api()
        systems=[row["payload"]["messages"][0] for row in plans.values()]
        self.assertTrue(all(system=={"role":"system","content":api.CBT_POLICY} for system in systems))
        self.assertIn("You are a CBT psychological counselor",api.CBT_POLICY)
        self.assertEqual(plans["no_rag"]["trace"]["selected_source_ids"],[])
        self.assertEqual(plans["no_rag"]["trace"]["recent_source_ids"],
                         plans["raw_history_rag"]["trace"]["recent_source_ids"])
        self.assertEqual(plans["no_rag"]["trace"]["recent_source_ids"],
                         plans["managed_memory_rag"]["trace"]["recent_source_ids"])
        self.assertNotEqual(plans["raw_history_rag"]["payload"],plans["managed_memory_rag"]["payload"])

    def test_received_partial_case_counts_without_completed_marker(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            atomic(folder/"protocol.json",{"case_count":3,"source_families":1})
            dest=folder/"cases"/"invented";dest.mkdir(parents=True)
            body={"messages":[]}
            atomic(dest/"raw_history_rag.answer.json",{"case_id":"invented","arm":"raw_history_rag","status":"received",
                "request_key":"r","response":"ok","provider_model":"alias"})
            atomic(dest/"raw_history_rag.request.json",{"payload":body})
            atomic(folder/"intents.jsonl",{"key":"r","payload":body})
            atomic(folder/"outcomes.jsonl",{"key":"r","status":"received","text":"ok","provider_model":"alias"})
            summary=runner.summarize(folder,fixture_only=True)
            self.assertEqual(summary["completed_cases"],0)
            self.assertEqual(summary["arms"]["raw_history_rag"],{"planned":3,"received":1})

    def test_missing_assessments_keep_the_fixed_necessary_fact_denominator(self):
        from auto_eval_v2 import evidence
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            cases=[{"case_id":"c"+str(i),"source_family_id":"invented:family"} for i in range(3)]
            atomic(root/"cases.json",cases)
            for i in range(2):
                dest=root/"assessor_only"/("c"+str(i));dest.mkdir(parents=True)
                gold={"case_id":"c"+str(i),"source_family":"invented:family","label_provenance":"model_proposed",
                    "required_facts":[{"id":"f","allowed_source_ids":["s"]}],"forbidden_or_stale_fact_ids":[]}
                atomic(dest/"targets.json",{"gold":gold})
                if i==0:
                    score=evidence.score_memory(gold,{"status":"answered","label_provenance":"model_proposed",
                        "claims":[{"claim_id":"a","fact_id":"f","source_ids":["s"],"verdict":"supported",
                            "evidence":[{"source_id":"s","quote":"one"}]}]},
                        [{"source_id":"s","source_order":0,"role":"user","content":"one",
                          "retained":True,"use_allowed":True}],0)
                    atomic(dest/"raw_history_rag.score.json",{"score":score})
            assessor=memory_assessment.Assessor.__new__(memory_assessment.Assessor)
            assessor.folder=root
            report=assessor.summarize()
            self.assertEqual(report["target_unavailable_cases"],1)
            raw=report["arms"]["raw_history_rag"]
            self.assertEqual(raw["conditional_necessary_fact_coverage"],1)
            self.assertEqual(raw["necessary_fact_coverage"],.5)
            self.assertEqual(raw["required"],2)
            self.assertEqual(report["arms"]["managed_memory_rag"]["required"],2)
            paired=report["paired"]["raw_history_rag__managed_memory_rag"]["recall"]
            self.assertEqual(paired["completion_adjusted_applicable_questions"],2)
            self.assertEqual(paired["completion_adjusted_fixed_grid"]["mean_difference"],-.5)

    def test_fixture_assessor_rejects_a_real_transport(self):
        with tempfile.TemporaryDirectory() as folder:
            atomic(Path(folder)/"protocol.json",{})
            with self.assertRaisesRegex(ValueError,"fixture assessor"):
                memory_assessment.Assessor(folder,object(),max_new_calls=1,fixture_only=True)

    def test_recovered_pool_preserves_actual_dates_and_excludes_old_gold(self):
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root)/"pool"
            source=[{"case_id":"old-case","source_family_id":"invented:dated","input":"What is the plan?",
                "input_date":"2026-11-05T09:00:00Z","calendar_time_available":True,"old_gold":"hidden",
                "history":[{"source_id":"s0","role":"user","content":"I agreed to two practices.",
                    "date":"2026-11-03T09:00:00Z","source_order":0}]}]
            protocol=runner.prepare_case_pool(source,folder,{"archive":"invented fixture"},fixture_only=True)
            case=json.loads((folder/"cases.json").read_text())[0]
            self.assertNotIn("old_gold",case)
            self.assertEqual(case["input_date"],source[0]["input_date"])
            sources=runner.source_views(case)
            package=runner.assemble(case,sources,[],[],"raw")
            block=json.loads(package.messages[-2]["content"].split('\n',1)[1])
            self.assertEqual(block["current_input_source"]["source_time"],source[0]["input_date"])
            self.assertTrue(json.loads((folder/"source_preflight.json").read_text())[0]["extraction_within_cap"])
            self.assertEqual(protocol["maximum_attempts"],5)

    def test_full_fixture_pipeline_freezes_hidden_targets_and_scores_actual_answers(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            source=root/"invented.json"
            source.write_text(json.dumps(cactus_materials.invented_fixture_rows()))
            folder=root/"natural"
            runner.prepare(source,folder,family_limit=1,fixture_only=True)
            independent={"provider":"fixture","model":"judge-request","returned_model":"judge-receipt","model_family":"qwen"}
            memory_assessment.prepare(folder,root/"assessment",independent,fixture_only=True)
            class Transport:
                fixture_only=True
                def __init__(self,identity):
                    self.identity=identity
                def __call__(self,body):
                    system=body["messages"][0]["content"]
                    data=json.loads(body["messages"][-1]["content"]) if body["messages"][-1]["content"].startswith('{') else {}
                    if system.startswith("Extract only reusable"):
                        s=next(s for s in data["history"] if s["role"]=="user")
                        value={"items":[{"item_id":"m0","section":"case_state","value":"Processed fixture summary.",
                            "origin":"explicit_user_statement","evidence":[{"source_id":s["source_id"],"quote":s["content"]}]}]}
                    elif system.startswith("Independently assess whether"):
                        value={"verdicts":[{"item_id":i["item_id"],"supported":True,"reason":"fixture source-supported"} for i in data["items"]]}
                    elif system.startswith("Read the visible history"):
                        s=next(s for s in data["history"] if s["role"]=="user")
                        value={"targets":[{"id":"f0","kind":"required","proposition":"An invented source fact.",
                               "evidence":[{"source_id":s["source_id"],"quote":s["content"]}]}]}
                    elif system.startswith("Independently check the proposed"):
                        value={"verdicts":[{"item_id":"f0","supported":True,"reason":"fixture supported"}]}
                    elif system.startswith("Assess one anonymous answer"):
                        value={"status":"answered","claims":[]}
                    else:
                        value=None
                        # Targets must already have been frozen before generation.
                        self_test=list((root/"assessment/assessor_only").glob('*/targets.json'))
                        if not self_test:
                            raise AssertionError("answer sent before target freeze")
                    return dict(text=json.dumps(value) if value is not None else "A fixture response.",
                        provider_model=self.identity["returned_model"],finish_reason="stop",usage={"total_tokens":3},
                        wire_sha256=sha256(json.dumps(body).encode()).hexdigest())
            summary=runner.run(folder,Transport(runner.IDENTITY),max_new_calls=15,workers=2,dense=FakeDense(),fixture_only=True,
                assessment_folder=root/"assessment",assessment_transport=Transport(independent),assessment_max_new_calls=15)
            self.assertEqual(summary["completed_cases"],3)
            self.assertTrue(all(r["received"]==3 for r in summary["arms"].values()))
            assessment=json.loads((root/"assessment/summary.json").read_text())
            self.assertEqual(assessment["label_provenance"],"model_proposed")
            self.assertEqual(assessment["arms"]["raw_history_rag"]["necessary_fact_coverage"],0)
            for request_file in (folder/"cases").glob('*/*.request.json'):
                self.assertNotIn('An invented source fact',request_file.read_text())
            processed=list((folder/"cases").glob('*/managed_memory_rag.request.json'))
            self.assertTrue(all('Processed fixture summary' in p.read_text() for p in processed))
