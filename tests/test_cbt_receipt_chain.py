"""Invented multi-hop receipt inheritance; no provider, browser or credentials."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"evaluation/cbt-memory-effect"))
sys.path.insert(0, str(Path(__file__).parent))
from test_cbt_receipt_continuation import old_fixture, FixtureTransport, Dense, IDENTITY, write_rows
from architecture_retest import continuation, runner, memory_assessment as memory
from architecture_retest import receipt_resolver as rr
from architecture_retest import receipt_chain
from architecture_retest.journal import InvalidReceipt, UnknownAttempt, atomic, digest, rows

CHAIN_VERSION = "cbt-exact-receipt-chain-import-v2"


def snapshot(*folders):
    return {path:rr.file_hash(path) for folder in folders for path in folder.rglob("*") if path.is_file()}


def first_continuation(root, old, assessor):
    first = root/"first"
    budget = continuation.prepare(old, assessor, first, fixture_only=True)
    a, q = FixtureTransport(runner.IDENTITY), FixtureTransport(IDENTITY)
    result = continuation.run(first, a, q, max_new_calls=budget["maximum_new_generation_requests"],
        assessment_max_new_calls=budget["maximum_new_assertion_requests"], workers=1, dense=Dense(), fixture_only=True)
    if not all(value["received"]==1 for value in result["arms"].values()):
        raise AssertionError("invented first continuation did not complete its three answers")
    return first


def truncated_first_continuation(root):
    """An actually journalled invented length receipt, never a real API call."""
    cases = [{"case_id":"invented-case","source_family_id":"invented-family",
        "input":"What could I reflect on next?","calendar_time_available":False,
        "history":[{"source_id":"s"+str(i),"source_order":i,
            "role":"user" if i%2==0 else "assistant",
            "content":"I completed one practice." if i==0 else "An invented dialogue turn "+str(i)+"."}
            for i in range(8)]}]
    old, assessor = root/"empty_reader", root/"empty_assessor"
    runner.prepare_case_pool(cases, old, {"archive":"invented empty fixture"}, fixture_only=True)
    memory.prepare(old, assessor, IDENTITY, fixture_only=True)
    for source in (old, assessor):
        for name in ("intents.jsonl","outcomes.jsonl"):
            (source/name).touch()
    first = root/"truncated_first"
    continuation.prepare(old, assessor, first, fixture_only=True)
    # Preserve the observed older policy: a length receipt stopped admission.
    # Later continuation policies must not retroactively change this source.
    for leaf in ("reader","memory"):
        protocol = rr.read_json(first/leaf/"protocol.json")
        protocol.pop("terminal_receipt_policy", None)
        protocol["protocol_sha256"] = digest({key:value for key,value in protocol.items() if key!="protocol_sha256"})
        atomic(first/leaf/"protocol.json", protocol)

    class LengthTransport(FixtureTransport):
        def __call__(self, body):
            result = super().__call__(body)
            if body["messages"][0]["content"].startswith("Extract only reusable"):
                result.update(finish_reason="length", text='{"items":[{"item_id":"never-use-truncated"')
            return result

    answer, judge = LengthTransport(runner.IDENTITY), FixtureTransport(IDENTITY)
    continuation.run(first, answer, judge, max_new_calls=5, assessment_max_new_calls=5,
        workers=1, dense=Dense(), fixture_only=True)
    intent, outcome = rows(first/"reader/intents.jsonl")[0], rows(first/"reader/outcomes.jsonl")[0]
    if (len(answer.calls)!=1 or outcome["status"]!="invalid" or outcome["finish_reason"]!="length"
            or outcome["error_type"]!="InvalidReceipt" or intent["purpose"]!="extraction"):
        raise AssertionError("invented first continuation did not retain its exact truncated extraction")
    allow = {intent["key"]:{name:(intent if name in intent else outcome)[name]
        for name in ("case_id","purpose","protocol_sha256","wire_sha256","provider_model","finish_reason","error_type")}}
    return old, assessor, first, intent, outcome, allow


class ReceiptChainTests(unittest.TestCase):
    def test_exact_truncated_extraction_stays_paid_invalid_and_only_other_arms_continue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, assessor, first, intent, outcome, allow = truncated_first_continuation(root)
            before = snapshot(old, assessor, first)
            second = root/"second"
            with patch.dict(receipt_chain.KNOWN_INVALID_EXTRACTIONS, allow, clear=True):
                budget = continuation.prepare(first/"reader", first/"memory", second, fixture_only=True)
                self.assertEqual(budget["inherited_generation_received"], 0)
                self.assertEqual(budget["inherited_generation_terminal_invalid"], 1)
                self.assertEqual(budget["maximum_new_generation_requests"], 4)
                manifest = rr.read_json(second/"reader/receipt_import.json")
                self.assertEqual(manifest["entries"], [])
                self.assertEqual(manifest["unavailable"], [])
                self.assertEqual(manifest["terminal_invalid"], [{"intent":intent,"receipt":outcome}])
                self.assertEqual(sum(row["paid_attempts_counted"] for row in manifest["paid_attempt_origins"]), 1)
                self.assertEqual(sum(row["invalid_preserved"] for row in manifest["paid_attempt_origins"]), 1)
                resolver = rr.ReceiptResolver(second/"reader", rr.read_json(second/"reader/protocol.json"))
                case = rr.read_json(second/"reader/cases.json")[0]
                with self.assertRaises(receipt_chain.FrozenInvalidReceipt) as frozen:
                    resolver.call(case, "extraction", json.loads(intent["serialized_request"]))
                self.assertIsInstance(frozen.exception, InvalidReceipt)
                answer, judge = FixtureTransport(runner.IDENTITY), FixtureTransport(IDENTITY)
                result = continuation.run(second, answer, judge, max_new_calls=2, assessment_max_new_calls=2,
                    workers=1, dense=Dense(), fixture_only=True)
                self.assertEqual(len(answer.calls), 2)
                self.assertFalse(any(body==intent["payload"] for body in answer.calls))
                self.assertTrue(all(not body["messages"][0]["content"].startswith("Extract only reusable")
                    for body in answer.calls))
                self.assertEqual(result["arms"]["no_rag"]["received"], 1)
                self.assertEqual(result["arms"]["raw_history_rag"]["received"], 1)
                self.assertEqual(result["arms"]["managed_memory_rag"]["received"], 0)
                failure = rr.read_json(second/"reader/cases/invented-case/memory_preparation_failure.json")
                self.assertEqual(failure["error_type"], "FrozenInvalidReceipt")
                self.assertFalse((second/"reader/cases/invented-case/managed_memory_rag.answer.json").exists())
                self.assertEqual([row["purpose"] for row in rows(second/"reader/intents.jsonl")], ["answer","answer"])
                self.assertTrue(all(rr.file_hash(path)==expected for path,expected in before.items()))

    def test_truncated_extraction_exception_rejects_other_key_wire_purpose_finish_and_unknown(self):
        for changed in ("key","wire_sha256","purpose","finish_reason","unknown_status"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _, _, first, intent, _, allow = truncated_first_continuation(root)
                if changed=="key":
                    allow = {"different-invented-key":allow[intent["key"]]}
                elif changed=="unknown_status":
                    outcomes = rows(first/"reader/outcomes.jsonl")
                    outcomes[0].update(status="unknown",text=None)
                    write_rows(first/"reader/outcomes.jsonl",outcomes)
                else:
                    allow[intent["key"]][changed] = {"wire_sha256":"0"*64,"purpose":"answer","finish_reason":"stop"}[changed]
                with patch.dict(receipt_chain.KNOWN_INVALID_EXTRACTIONS, allow, clear=True), self.assertRaises(UnknownAttempt):
                    continuation.prepare(first/"reader", first/"memory", root/"second", fixture_only=True)
                self.assertFalse((root/"second").exists())

    def test_second_hop_reuses_all_original_and_first_receipts_without_relabelling_or_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, assessor = old_fixture(root)
            first = first_continuation(root, old, assessor)
            before = snapshot(old, assessor, first)
            second = root/"second"
            budget = continuation.prepare(first/"reader", first/"memory", second, fixture_only=True)
            self.assertEqual((budget["maximum_new_generation_requests"], budget["maximum_new_assertion_requests"]), (0,0))
            for leaf, original in (("reader", old), ("memory", assessor)):
                manifest = rr.read_json(second/leaf/"receipt_import.json")
                self.assertEqual(manifest["version"], CHAIN_VERSION)
                expected = {}
                for source in (original, first/leaf):
                    source_intents = {row["key"]:row for row in rows(source/"intents.jsonl")}
                    for outcome in rows(source/"outcomes.jsonl"):
                        expected[outcome["key"]] = {"intent":source_intents[outcome["key"]],"receipt":outcome}
                actual = {entry["intent"]["key"]:entry for entry in manifest["entries"]}
                self.assertEqual(actual, expected)
                self.assertEqual(len(actual), 5)
                for entry in actual.values():
                    protocol_hash = entry["intent"]["protocol_sha256"]
                    self.assertIn(protocol_hash, manifest["source_protocols"])
                    self.assertEqual(entry["receipt"]["protocol_sha256"], protocol_hash)
                self.assertIsNone(manifest["legacy_inherited_target_receipts"])
                origins = manifest["paid_attempt_origins"]
                self.assertEqual(len(origins), 2)
                self.assertEqual(sum(item["paid_attempts_counted"] for item in origins), 5)
                for item in origins:
                    self.assertEqual(item["intents_file_sha256"], rr.file_hash(Path(item["source_folder"])/"intents.jsonl"))
                self.assertTrue(manifest["external_attempts_not_in_receipt_chain_require_separate_driver_debit"])
            a, q = FixtureTransport(runner.IDENTITY), FixtureTransport(IDENTITY)
            result = continuation.run(second, a, q, max_new_calls=0, assessment_max_new_calls=0,
                workers=1, dense=Dense(), fixture_only=True)
            self.assertTrue(all(value["received"]==1 for value in result["arms"].values()))
            self.assertEqual((a.calls,q.calls), ([],[]))
            self.assertTrue(all(rr.file_hash(path)==expected for path, expected in before.items()))

    def test_legacy_target_receipt_is_flattened_once_with_its_own_original_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, original_assessor = old_fixture(root)
            assessor = root/"legacy_assessor"
            memory.prepare(old, assessor, IDENTITY, fixture_only=True, inherited_folder=original_assessor)
            q = FixtureTransport(IDENTITY)
            engine = memory.Assessor(assessor, q, max_new_calls=4, fixture_only=True)
            case = rr.read_json(old/"cases.json")[0]
            engine.freeze_targets(case, runner.source_views(case))
            self.assertEqual(len(q.calls), 1)
            first = first_continuation(root, old, assessor)
            continuation.prepare(first/"reader", first/"memory", root/"second", fixture_only=True)
            manifest = rr.read_json(root/"second/memory/receipt_import.json")
            target = next(row for row in rows(original_assessor/"intents.jsonl") if row["purpose"]=="fact_targets")
            matches = [entry for entry in manifest["entries"] if entry["intent"]["key"]==target["key"]]
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0]["intent"], target)
            self.assertIn(target["protocol_sha256"], manifest["source_protocols"])
            self.assertIsNone(manifest["legacy_inherited_target_receipts"])
            legacy = manifest["legacy_receipts_outside_current_local_envelope"]
            self.assertEqual(len(legacy), 1)
            self.assertEqual((legacy[0]["received_receipts"],legacy[0]["new_paid_attempts_counted"]), (1,0))
            a, q = FixtureTransport(runner.IDENTITY), FixtureTransport(IDENTITY)
            continuation.run(root/"second", a, q, max_new_calls=0, assessment_max_new_calls=0,
                workers=1, dense=Dense(), fixture_only=True)
            self.assertEqual((a.calls,q.calls), ([],[]))

    def test_changed_ancestor_journal_fails_before_any_second_hop_send(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, assessor = old_fixture(root)
            first = first_continuation(root, old, assessor)
            continuation.prepare(first/"reader", first/"memory", root/"second", fixture_only=True)
            with (old/"outcomes.jsonl").open("a") as stream: stream.write("\n")
            a, q = FixtureTransport(runner.IDENTITY), FixtureTransport(IDENTITY)
            with self.assertRaises(ValueError):
                continuation.run(root/"second", a, q, max_new_calls=0, assessment_max_new_calls=0,
                    workers=1, dense=Dense(), fixture_only=True)
            self.assertEqual((a.calls,q.calls), ([],[]))

    def test_local_journal_cannot_overlap_inherited_keys_or_duplicate_received_rows(self):
        for overlap in (True,False):
            with self.subTest(overlap=overlap), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                old, assessor = old_fixture(root)
                first = first_continuation(root, old, assessor)
                source = old if overlap else first/"reader"
                duplicate_intent, duplicate_outcome = rows(source/"intents.jsonl")[0], rows(source/"outcomes.jsonl")[0]
                for name, duplicate in (("intents.jsonl",duplicate_intent),("outcomes.jsonl",duplicate_outcome)):
                    write_rows(first/"reader"/name, rows(first/"reader"/name)+[duplicate])
                with self.assertRaises((ValueError,UnknownAttempt)):
                    continuation.prepare(first/"reader", first/"memory", root/"second", fixture_only=True)
                self.assertFalse((root/"second").exists())

    def test_new_local_unknown_cannot_be_promoted_to_historical_unavailable(self):
        for leaf in ("reader","memory"):
            with self.subTest(leaf=leaf), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                old, assessor = old_fixture(root)
                first = first_continuation(root, old, assessor)
                intent = rows(first/leaf/"intents.jsonl")[0]
                outcomes = rows(first/leaf/"outcomes.jsonl")
                outcomes[0].update(status="unknown", text=None)
                write_rows(first/leaf/"outcomes.jsonl", outcomes)
                allowed = {intent["key"]:{"case_id":intent["case_id"],"wire_sha256":intent["wire_sha256"]}}
                with patch.dict(rr.FROZEN_ASSESSMENTS, allowed, clear=True), self.assertRaises(UnknownAttempt):
                    continuation.prepare(first/"reader", first/"memory", root/"second", fixture_only=True)
                self.assertFalse((root/"second").exists())

    def test_frozen_original_unknown_stays_unavailable_and_never_resent_through_two_hops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, assessor = old_fixture(root)
            raw_answer = rr.read_json(old/"cases/invented-case/raw_history_rag.answer.json")
            intent = next(row for row in rows(assessor/"intents.jsonl") if row["purpose"]=="memory_adjudication" and
                json.loads(row["payload"]["messages"][-1]["content"])["response"]==raw_answer["response"])
            outcomes = rows(assessor/"outcomes.jsonl")
            for row in outcomes:
                if row["key"]==intent["key"]: row.update(status="unknown",text=None)
            write_rows(assessor/"outcomes.jsonl", outcomes)
            allowed = {intent["key"]:{"case_id":intent["case_id"],"wire_sha256":intent["wire_sha256"]}}
            with patch.dict(rr.FROZEN_ASSESSMENTS, allowed, clear=True):
                first = first_continuation(root, old, assessor)
                budget = continuation.prepare(first/"reader", first/"memory", root/"second", fixture_only=True)
                self.assertEqual((budget["maximum_new_generation_requests"],budget["maximum_new_assertion_requests"]), (0,0))
                manifest = rr.read_json(root/"second/memory/receipt_import.json")
                self.assertEqual(len(manifest["unavailable"]), 1)
                self.assertEqual(manifest["unavailable"][0]["intent"], intent)
                a,q = FixtureTransport(runner.IDENTITY), FixtureTransport(IDENTITY)
                continuation.run(root/"second", a, q, max_new_calls=0, assessment_max_new_calls=0,
                    workers=1, dense=Dense(), fixture_only=True)
                unavailable = rr.read_json(root/"second/memory/assessor_only/invented-case/raw_history_rag.unavailable.json")
                self.assertEqual(unavailable["assessor_request_key"], intent["key"])
                self.assertIs(unavailable["resend_permitted"], False)
                self.assertEqual((a.calls,q.calls), ([],[]))


if __name__=="__main__":
    unittest.main()
