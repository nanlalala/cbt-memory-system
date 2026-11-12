"""Fictional dated product replay and the preserved direct JSON scoring rule."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("architecture_controlled",ROOT/"evaluation/architecture-rerun/controlled.py")
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ControlledArchitectureTest(unittest.TestCase):
    def case(self):
        texts=["我同意本周做三次想法记录。","想法记录累计做了两次。","想法记录累计更正为一次。"]
        return {"case_id":"invented-dated-correction","source_family_id":"invented-family",
            "calendar_time_available":True,"input":"想法记录累计做了几次？","input_message_id":"current",
            "input_date":"2026-11-05T09:00:00Z","history":[{"message_id":str(i),"role":"user","content":text,
                "source_order":i,"date":f"2026-11-0{3+i}T08:00:00Z"} for i,text in enumerate(texts)]}

    def test_real_task_correction_is_processed_in_actual_request_and_send_is_revocable(self):
        with tempfile.TemporaryDirectory() as temporary:
            packets=runner.prepare_one(self.case(),Path(temporary)/"product",fixture_only=True)
            managed=packets["managed_memory_rag"]
            data=json.loads(managed["payload"]["messages"][-2]["content"].split("\n",1)[1])
            claims={c["field"]:c["value"] for c in data["claims"] if c.get("task_instance_id")}
            self.assertEqual(claims["reported_count"],1)
            self.assertTrue(managed["trace"]["task_projection_present"])
            self.assertEqual(managed["trace"]["clock_mode"],"dated-sources-v1")
            self.assertTrue(all("source_time" in s for s in data["recent_sources"]))
            for packet in packets.values():
                runner.preflight(packet)
                self.assertLessEqual(len(json.dumps(packet["payload"])),runner.MAX_CHARS)
                self.assertEqual(packet["payload"]["max_tokens"],64)
                self.assertEqual(sum(m["content"]==self.case()["input"] for m in packet["payload"]["messages"]),1)
            m=runner.replay._modules()
            store=m.DurableMemoryStore.open(managed["database"],approved_root=managed["approved_root"])
            try:
                sid=next(iter(managed["binding"]["source_hashes"]))
                store.revoke(managed["binding"]["user_id"],"stop_use",sid)
            finally:
                store.close()
            with self.assertRaisesRegex(ValueError,"generation|revoked|changed"):
                runner.preflight(managed)

    def test_original_metric_accepts_only_single_key_exact_type_and_value(self):
        self.assertTrue(runner.strict_result('{"answer":2}',2)["correct"])
        self.assertEqual(runner.strict_result('{"answer":"2"}',2)["reason"],"output_type_error")
        self.assertEqual(runner.strict_result('{"answer":true}',1)["reason"],"malformed_output")
        self.assertEqual(runner.strict_result('{"answer":2,"answer":3}',3)["reason"],"malformed_output")
        self.assertEqual(runner.strict_result('```json\n{"answer":2}\n```',2)["reason"],"malformed_output")
        self.assertEqual(runner.strict_result('{"answer":2}',None)["reason"],"unsupported_assertion")


if __name__=="__main__":
    unittest.main()
