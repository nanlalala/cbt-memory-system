"""Tiny invented fixtures for blind reference grading and full denominators."""
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("architecture_formal_judge", ROOT / "evaluation/architecture-rerun/formal_judge.py")
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)

IDENTITY = {"provider":"https://tokendance.space/gateway/v1", "model":"independent-fixture-model",
            "returned_model":"independent-fixture-returned", "model_family":"qwen"}


class Transport:
    fixture_only = True
    identity = IDENTITY

    def __init__(self, text="yes"):
        self.text, self.calls = text, 0

    def __call__(self, payload):
        self.calls += 1
        return {"text":self.text, "provider_model":self.identity["returned_model"], "finish_reason":"stop",
                "wire_sha256":sha256(json.dumps(payload,allow_nan=False).encode()).hexdigest(),
                "usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}


def reader_fixture(folder):
    folder.mkdir()
    cases = [{"case_id":"invented-"+str(i), "source_family_id":"invented-family-"+str(i),
              "history_key":"invented-history", "input":"What city did the fictional person mention?",
              "query_date":"2026-01-01"} for i in range(2)]
    references = [{"case_id":c["case_id"], "query":c["input"], "reference_answer":"FictionalCity"+str(i),
                   "category":"single-session-user", "should_abstain":False} for i,c in enumerate(cases)]
    protocol = {"version":"invented-software-fixture", "dataset_key":"longmemeval_s_cleaned",
                "case_count":len(cases), "arms":list(judge.ARMS), "cases_sha256":judge.digest(cases),
                "provider":{"returned_model":"deepseek-fixture-returned"}}
    protocol["protocol_sha256"] = judge.digest(protocol)
    judge.atomic(folder/"protocol.json",protocol)
    judge.atomic(folder/"cases.json",cases)
    judge.atomic(folder/"references.json",references)
    for i,c in enumerate(cases):
        dest = folder/"cases"/judge.digest(c["case_id"])[:32]
        dest.mkdir(parents=True)
        for arm in judge.ARMS:
            judge.atomic(dest/(arm+".answer.json"), {"case_id":c["case_id"],"arm":arm,"status":"received",
                "response":"FictionalCity"+str(i), "request_key":"invented-receipt-"+str(i),
                "provider_model":"deepseek-fixture-returned", "usage":{}})
    return cases


class FormalJudgeTest(unittest.TestCase):
    def test_complete_blind_grid_repeat_and_receipt_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/"source", Path(tmp)/"judge"
            reader_fixture(source)
            protocol = judge.prepare(source,output,IDENTITY,fixture_only=True)
            self.assertEqual(protocol["planned_answers"],6)
            self.assertEqual(protocol["repeat_rows"],1)
            requests = judge._json(output/"requests.json")
            self.assertEqual(len(requests),7)
            for row in requests:
                payload = json.dumps(row["payload"])
                for arm in judge.ARMS:
                    self.assertNotIn(arm,payload)
            transport = Transport()
            summary = judge.run(output,transport,max_new_calls=7,fixture_only=True)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["scored_denominator"],6)
            self.assertEqual(summary["repeatability"]["label_agreement"],1)
            self.assertEqual(summary["conditions"]["raw_history_rag"]["question_micro_accuracy"],1)
            self.assertEqual(summary["paired_contrasts"]["primary_managed_minus_raw"]["cluster_bootstrap_95_interval"],[0,0])
            calls = transport.calls
            judge.run(output,transport,max_new_calls=0,fixture_only=True)
            self.assertEqual(transport.calls,calls)
            self.assertLess(calls,7)  # same anonymous answer/rubric reuses the exact receipt
            retained = next((output/"scores").glob("*.json"))
            edited = judge._json(retained)
            edited["score"] = 0
            retained.write_text(json.dumps(edited))
            with self.assertRaisesRegex(ValueError,"changed after strict parsing"):
                judge.summarize(output,fixture_only=True)

    def test_missing_arm_blocks_freeze_and_same_family_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/"source", Path(tmp)/"judge"
            cases = reader_fixture(source)
            path = source/"cases"/judge.digest(cases[0]["case_id"])[:32]/"raw_history_rag.answer.json"
            path.unlink()
            with self.assertRaisesRegex(ValueError,"reader is incomplete"):
                judge.prepare(source,output,IDENTITY,fixture_only=True)
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError,"independent"):
                judge.prepare(source,output,{**IDENTITY,"model_family":"deepseek"},fixture_only=True)

    def test_invalid_label_is_zero_and_terminal_reader_failure_stays_in_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/"source", Path(tmp)/"judge"
            cases = reader_fixture(source)
            dest = source/"cases"/judge.digest(cases[0]["case_id"])[:32]
            (dest/"no_rag.answer.json").unlink()
            judge.atomic(dest/"no_rag.failure.json", {"case_id":cases[0]["case_id"],"arm":"no_rag",
                                                  "status":"preparation_failed"})
            judge.prepare(source,output,IDENTITY,fixture_only=True)
            summary = judge.run(output,Transport("maybe"),max_new_calls=20,fixture_only=True)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["scored_denominator"],6)
            for arm in judge.ARMS:
                self.assertEqual(summary["conditions"][arm]["question_micro_accuracy"],0)
                self.assertEqual(summary["conditions"][arm]["technical_failures_or_pending"],2)


if __name__ == "__main__":
    unittest.main()
