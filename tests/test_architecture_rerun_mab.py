"""Tiny invented regression cases for the full-cohort readout adapter."""
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("architecture_mab_rerun", ROOT / "evaluation/architecture-rerun/mab_full.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
from objective_datasets import parse_mab_rows


class ArchitectureMabTest(unittest.TestCase):
    def test_processed_fact_fields_and_source_roles_reach_the_new_renderer(self):
        context = "The capital of Elar is Oldtown.\nThe capital of Elar is Newtown.\n"
        context += "The country of Newtown is Terra.\n"
        context += "".join(f"The country of Filler{i} is Land{i}.\n" for i in range(22))
        cases, _ = parse_mab_rows([{"context":context, "questions":["What is the capital of Elar?"],
            "answers":[["Newtown"]], "metadata":{"source":"factconsolidation_sh_6k", "qa_pair_ids":["fictional"]}}])
        settings = runner.Settings(model=runner.MODEL, accepted_provider_models=(runner.RETURNED_MODEL,),
            backend="lexical", thinking_mode="disabled", short_turns=16, top_k=12,
            short_chars=6000, long_chars=12000, max_context_chars=24000, max_response_tokens=128)
        packets, _ = runner.prepare_case(cases[0], runner.HistoryRanker(settings))
        by_arm = {p["arm"]:p for p in packets}
        raw = json.loads(by_arm["raw_history_rag"]["request"]["messages"][-2]["content"].split("\n",1)[1])
        managed = json.loads(by_arm["managed_memory_rag"]["request"]["messages"][-2]["content"].split("\n",1)[1])
        self.assertTrue(all(s["role"] == "context" and "source_order" in s for s in raw["source_reports"]))
        values = [p["value"] for p in managed["case_state"]]
        self.assertIn({"subject":"Elar", "relation":"capital", "object":"Newtown", "source_order":1,
                       "basis":"synthetic_benchmark_source_grammar"}, values)
        self.assertNotIn("Oldtown", json.dumps(managed))
        self.assertIn("Oldtown", json.dumps(raw))
        for packet in packets:
            messages = packet["request"]["messages"]
            self.assertEqual([m["role"] for m in messages], ["system","user","user"])
            self.assertEqual(sum("What is the capital of Elar?" in m["content"] for m in messages), 1)
            self.assertLessEqual(packet["wire_chars"], runner.MAX_CHARS)

    def test_journal_rejects_payload_substitution_and_blocks_unknown_before_secret(self):
        request = {"model":runner.MODEL,"messages":[],"temperature":0,"max_tokens":128}
        packet = {"request_sha256":"fictional-key","request":request}
        protocol = {"protocol_sha256":"fictional-protocol"}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"protocol.json").write_text(json.dumps(protocol))
            intent = {"event":"intent","protocol_sha256":"fictional-protocol",
                      "request_sha256":"fictional-key","request":request}
            runner.append(root/"journal.jsonl",intent)
            with patch.object(runner.old,"require_colab"), patch.object(runner,"load",return_value=(protocol,[packet])):
                with self.assertRaisesRegex(RuntimeError,"unresolved"):
                    runner.generate(root,1)
            packet["request"] = {"model":"changed"}
            with self.assertRaisesRegex(ValueError,"payload changed"):
                runner.ledger_state(root,[packet])

    def test_parallel_failure_drains_the_already_admitted_requests_without_resending(self):
        requests = [{'request_sha256':str(i),'request':{'messages':[{'role':'user','content':str(i)}]}}
                    for i in range(10)]
        protocol = {'protocol_sha256':'fictional-protocol','maximum_provider_attempts':10}
        def fake_transport(messages, **kwargs):
            if messages[0]['content']=='0':
                raise TimeoutError('invented uncertain provider result')
            time.sleep(0.02)
            return {'hypothesis':'value','provider_model':runner.RETURNED_MODEL,'finish_reason':'stop',
                    'usage':{'prompt_tokens':1,'completion_tokens':1,'total_tokens':2},'latency_seconds':0.02}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'protocol.json').write_text(json.dumps(protocol))
            with patch.object(runner.old,'require_colab'),patch.object(runner,'load',return_value=(protocol,requests)), \
                    patch.object(runner,'generate_response',side_effect=fake_transport):
                with self.assertRaisesRegex(RuntimeError,'in-flight requests drained'):
                    runner.generate(root,10,key='fictional-unit-test-placeholder')
            intents,outcomes=runner.ledger_state(root,requests)
            self.assertEqual(set(intents),set(outcomes))
            self.assertEqual(len(intents),runner.WORKERS)
            self.assertEqual(sum(r['event']=='unknown' for r in outcomes.values()),1)
            self.assertEqual(sum(r['event']=='received' for r in outcomes.values()),runner.WORKERS-1)
