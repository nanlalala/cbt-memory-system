"""Explicit software fixtures for indexed quality grading; no hosted claims/API."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "quality_span_integration_fixture", ROOT / "evaluation/architecture-rerun/quality_span_judge.py")
grade = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = grade
SPEC.loader.exec_module(grade)


def packet():
    return {"case_id": "fictional", "source_family": "fictional-family", "answer_model_family": "deepseek",
        "history": [{"id": "fictional-source-0", "role": "user", "content": "甲😀\u2028乙\n" * 170},
                    {"id": "fictional-source-1", "role": "assistant", "content": "A fictional proposal, not consent."}],
        "input": "Fictional current question.", "response": "Fictional supportive reply. " * 40}


def review(source_id="H0000_0000", response_id="A0000"):
    value = {"R1": 4, "R2": 3, "R3": 4, "R4": 4, "R5": "pass"}
    value["evidence"] = [{"dimension": d, "source_refs": [source_id],
        "response_refs": [response_id], "disposition": "supported", "reason": "Fictional fixture evidence."}
        for d in grade.DIMENSIONS]
    return value


class FixtureTransport:
    """Synthetic receipt generator used only with fixture_only=True."""
    fixture_only = True
    identity = dict(grade.JUDGE_IDENTITY)

    def __init__(self):
        self.calls = []
        self._lock = threading.Lock()

    def __call__(self, payload):
        with self._lock:
            number = len(self.calls)
            self.calls.append(deepcopy(payload))
        visible = json.loads(payload["messages"][1]["content"])
        source_id = visible["history"][0]["segments"][0]["id"]
        response_id = visible["response"]["segments"][0]["id"]
        return {"text": json.dumps(review(source_id, response_id)),
            "provider_model": self.identity["returned_model"], "finish_reason": "stop",
            "response_id": "fictional-fixture-" + str(number),
            "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            "wire_sha256": sha256(json.dumps(payload, allow_nan=False).encode()).hexdigest()}


def source_fixture(folder):
    """Two invented source cases, with one explicitly missing managed answer."""
    folder.mkdir()
    cases = [{"case_id": "fictional-" + str(i), "source_family_id": "family-" + str(i),
        "history": [{"source_id": "source-" + str(i), "message_id": "m0", "role": "user",
                     "content": "Fictional original report " + str(i) + "."}],
        "input": "Fictional follow-up " + str(i) + "."} for i in range(2)]
    protocol = {"fixture_only": True, "arms": list(grade.ARMS),
        "provider": dict(grade.ANSWER_IDENTITY), "cases_sha256": grade.digest(cases),
        "case_count": len(cases), "code_sha256": {
            "evaluation/cbt-memory-effect/auto_eval_v2/quality.py":
                grade.file_hash(ROOT / "evaluation/cbt-memory-effect/auto_eval_v2/quality.py")}}
    protocol["protocol_sha256"] = grade.digest(protocol)
    grade.atomic(folder / "protocol.json", protocol)
    grade.atomic(folder / "cases.json", cases)
    intents, outcomes = [], []
    for i, case in enumerate(cases):
        dest = folder / "cases" / case["case_id"]
        dest.mkdir(parents=True)
        completed = []
        for j, arm in enumerate(grade.ARMS):
            if (i, j) == (0, 2):
                completed.append({"case_id": case["case_id"], "arm": arm, "status": "preparation_failed"})
                continue
            payload = {"model": grade.ANSWER_IDENTITY["model"], "messages": [
                {"role": "system", "content": "Fictional fixture only."},
                {"role": "user", "content": case["input"]}], "max_tokens": 32, "temperature": 0}
            # Distinct fixture replies prevent primary-response cache sharing.
            payload["messages"][1]["content"] += " " + str(j)
            wire = json.dumps(payload)
            key = grade.digest({"purpose": "answer", "payload": payload})
            intent = {"key": key, "purpose": "answer", "payload": payload,
                "serialized_request": wire, "wire_sha256": sha256(wire.encode()).hexdigest(),
                "protocol_sha256": protocol["protocol_sha256"], "case_id": case["case_id"]}
            text = "Fictional reply " + str(i) + ":" + str(j) + "."
            receipt = {"key": key, "status": "received", "protocol_sha256": protocol["protocol_sha256"],
                "provider_model": grade.ANSWER_IDENTITY["returned_model"], "finish_reason": "stop",
                "wire_sha256": intent["wire_sha256"], "text": text}
            answer = {"case_id": case["case_id"], "arm": arm, "status": "received", "response": text,
                "provider_model": receipt["provider_model"], "request_key": key}
            grade.atomic(dest / (arm + ".request.json"), {"payload": payload})
            grade.atomic(dest / (arm + ".answer.json"), answer)
            intents.append(intent)
            outcomes.append(receipt)
            completed.append(answer)
        grade.atomic(dest / "completed.json", {"case_id": case["case_id"], "rows": completed})
    for name, data in (("intents.jsonl", intents), ("outcomes.jsonl", outcomes)):
        (folder / name).write_text("".join(grade.canonical(row) + "\n" for row in data))
    return cases


class QualitySpanIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="indexed-quality-fixture-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def prepared(self):
        reader, output = self.root / "reader", self.root / "quality"
        source_fixture(reader)
        result = grade.prepare(reader, output, grade.JUDGE_IDENTITY, fixture_only=True)
        return reader, output, result

    def test_lossless_unicode_offsets_and_roles(self):
        original = packet()
        visible, manifest = grade.segment_packet(original)
        for source, rendered in zip(original["history"], visible["history"]):
            self.assertEqual("".join(s["text"] for s in rendered["segments"]), source["content"])
            for segment in rendered["segments"]:
                bound = manifest["sources"][segment["id"]]
                self.assertEqual(bound["role"], source["role"])
                self.assertEqual(bound["text"], source["content"][bound["start"]:bound["end"]])
        self.assertEqual("".join(s["text"] for s in visible["response"]["segments"]), original["response"])

    def test_original_rubric_is_reused_without_new_score_threshold(self):
        self.assertEqual(grade.RUBRIC, grade.quality.GUIDE.split("Return JSON with exactly R1", 1)[0])
        self.assertTrue(grade.validate_review(json.dumps(review()), packet())["accepted"])

    def test_unknown_duplicate_wrong_channel_and_wrong_dimension_are_rejected(self):
        for change in ("unknown", "duplicate", "channel", "dimension"):
            with self.subTest(change=change):
                value = review()
                evidence = value["evidence"][0]
                if change == "unknown": evidence["source_refs"] = ["not-a-real-segment"]
                if change == "duplicate": evidence["source_refs"] *= 2
                if change == "channel": evidence["source_refs"] = ["A0000"]
                if change == "dimension": evidence["dimension"] = "Contextual Accuracy"
                self.assertEqual(grade.validate_review(json.dumps(value), packet())["status"], "invalid")

    def test_missing_responses_stay_in_full_grid_and_repeats_are_real_calls(self):
        reader, output, preflight = self.prepared()
        self.assertEqual(preflight["full_grid_primary"], 6)
        self.assertEqual(preflight["available_response_packets"], 5)
        self.assertEqual(preflight["planned_judge_requests"], 6)
        transport = FixtureTransport()
        report = grade.run(output, transport, max_new_calls=7, workers=2, fixture_only=True)
        self.assertEqual(len(transport.calls), 6)
        self.assertTrue(report["complete"])
        self.assertFalse(report["all_primary_scores_valid"])
        self.assertEqual(report["dimensions"]["R1"]["managed_memory_rag"]["planned"], 2)
        self.assertEqual(report["dimensions"]["R1"]["managed_memory_rag"]["scored"], 1)
        self.assertEqual(report["repeat_consistency"]["paired_repeats"], 1)
        purposes = [r["purpose"] for r in grade.rows(output / "intents.jsonl")]
        self.assertEqual(sum(p.startswith("quality_v2_repeat:") for p in purposes), 1)
        grade.run(output, transport, max_new_calls=0, fixture_only=True)
        self.assertEqual(len(transport.calls), 6)

    def test_source_mutation_blocks_before_any_provider_call(self):
        reader, output, _ = self.prepared()
        cases = grade._json(reader / "cases.json")
        cases[0]["input"] += " changed after freeze"
        grade.atomic(reader / "cases.json", cases)
        transport = FixtureTransport()
        with self.assertRaises(ValueError):
            grade.run(output, transport, max_new_calls=7, fixture_only=True)
        self.assertEqual(transport.calls, [])

    def test_frozen_payload_mutation_blocks_before_any_provider_call(self):
        _, output, _ = self.prepared()
        plan = grade._json(output / "plan.json")
        plan[0]["payload"]["messages"][1]["content"] += " unapproved mutation"
        grade.atomic(output / "plan.json", plan)
        transport = FixtureTransport()
        with self.assertRaises(ValueError):
            grade.run(output, transport, max_new_calls=7, fixture_only=True)
        self.assertEqual(transport.calls, [])


if __name__ == "__main__":
    unittest.main()
