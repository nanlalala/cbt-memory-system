"""Terminal length is missing evidence, never a retry or a successful answer.

All providers below are explicit invented fixtures. No hosted flag, credential,
network call or clinical example is created by these tests.
"""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "evaluation/cbt-memory-effect")]
from architecture_retest import continuation, memory_assessment as memory, runner
from architecture_retest.journal import (Journal, InvalidReceipt, UnknownAttempt,
    TerminalLengthReceipt, TERMINAL_RECEIPT_POLICY, atomic, canonical, digest,
    is_terminal_length, rows)
from test_cbt_receipt_continuation import Dense, FixtureTransport, IDENTITY
from test_quality_span_judge_integration import source_fixture, FixtureTransport as QualityFixtureTransport

SPEC = importlib.util.spec_from_file_location("terminal_quality_v3_fixture",
    ROOT / "evaluation/architecture-rerun/quality_span_judge_v3.py")
quality = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = quality
SPEC.loader.exec_module(quality)


class ScriptedTransport:
    fixture_only = True
    identity = dict(runner.IDENTITY)

    def __init__(self, fault=None):
        self.calls = []
        self.fault = fault

    def __call__(self, payload):
        self.calls.append(deepcopy(payload))
        receipt = {"text": "An invented complete response.",
            "provider_model": self.identity["returned_model"], "finish_reason": "stop",
            "usage": {"prompt_tokens": 17, "completion_tokens": 3, "total_tokens": 20},
            "wire_sha256": sha256(json.dumps(payload, allow_nan=False).encode()).hexdigest()}
        if len(self.calls) == 1:
            receipt.update(text="{\"unfinished\":", finish_reason="length")
            if self.fault == "model": receipt["provider_model"] = "not-the-frozen-model"
            if self.fault == "wire": receipt["wire_sha256"] = "0" * 64
            if self.fault == "text": receipt["text"] = None
            if self.fault == "finish": receipt["finish_reason"] = "content_filter"
            if self.fault == "network": raise OSError("invented non-network transport failure")
        return receipt


def journal_fixture(root, policy=True):
    root.mkdir()
    cases = [{"case_id": "invented-one"}, {"case_id": "invented-two"}]
    protocol = {"provider": dict(runner.IDENTITY), "maximum_attempts": 10,
        "request_limits": {"answer": 96000, "preparation": 96000},
        "cases_sha256": digest(cases)}
    if policy:
        protocol["terminal_receipt_policy"] = TERMINAL_RECEIPT_POLICY
    protocol["protocol_sha256"] = digest(protocol)
    atomic(root / "cases.json", cases)
    atomic(root / "protocol.json", protocol)
    requests = [runner.payload([{"role": "user", "content": "Invented request " + str(i)}], "answer")
        for i in range(2)]
    return cases, protocol, requests


class LengthAtFirstMatch(FixtureTransport):
    def __init__(self, identity, predicate):
        super().__init__(identity)
        self.predicate, self.length_sent = predicate, False

    def __call__(self, body):
        receipt = super().__call__(body)
        if not self.length_sent and self.predicate(body["messages"][0]["content"]):
            self.length_sent = True
            receipt.update(text="{\"unfinished\":", finish_reason="length")
        return receipt


def continuation_fixture(root):
    """A newly versioned continuation from two unexecuted invented sources."""
    source = [{"case_id": "invented-case-" + str(i), "source_family_id": "invented-family-" + str(i),
        "input": "What could I reflect on next, invented case " + str(i) + "?",
        "calendar_time_available": False,
        "history": [{"source_id": "s" + str(i) + "-" + str(j), "source_order": j,
            "role": "user" if j % 2 == 0 else "assistant",
            "content": "I completed one practice." if j == 0 else "An invented dialogue turn " + str(j) + "."}
            for j in range(8)]} for i in range(2)]
    natural, assessor, output = root / "source_reader", root / "source_memory", root / "continuation"
    runner.prepare_case_pool(source, natural, {"archive": "invented fixture only"}, fixture_only=True)
    memory.prepare(natural, assessor, IDENTITY, fixture_only=True)
    # These source fixtures deliberately contain zero executed requests.
    for directory in (natural, assessor):
        for name in ("intents.jsonl", "outcomes.jsonl"):
            (directory / name).write_text("")
    continuation.prepare(natural, assessor, output, fixture_only=True)
    return output


class JournalTerminalTests(unittest.TestCase):
    def test_exact_length_is_preserved_never_resent_and_does_not_stop_other_requests(self):
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw) / "journal"
            cases, protocol, requests = journal_fixture(folder)
            transport = ScriptedTransport()
            journal = Journal(folder, protocol, transport, 2)
            with self.assertRaises(TerminalLengthReceipt) as caught:
                journal.call(cases[0], "answer", requests[0])
            self.assertFalse(journal.stop.is_set())
            outcome = rows(folder / "outcomes.jsonl")[0]
            self.assertEqual(outcome["status"], "invalid")
            self.assertEqual(outcome["text"], "{\"unfinished\":")
            self.assertEqual(outcome["usage"], {"prompt_tokens": 17, "completion_tokens": 3, "total_tokens": 20})
            self.assertEqual(caught.exception.request_key, outcome["key"])
            self.assertEqual(caught.exception.source_protocol_sha256, protocol["protocol_sha256"])
            self.assertEqual(caught.exception.wire_sha256, outcome["wire_sha256"])
            with self.assertRaises(TerminalLengthReceipt):
                journal.call(cases[0], "answer", requests[0])
            self.assertEqual(len(transport.calls), 1)
            self.assertEqual(journal.call(cases[1], "answer", requests[1])["status"], "received")
            self.assertEqual(len(transport.calls), 2)
            reopened = Journal(folder, protocol, transport, 0)
            with self.assertRaises(TerminalLengthReceipt):
                reopened.call(cases[0], "answer", requests[0])
            self.assertEqual(len(transport.calls), 2)
            self.assertFalse(reopened.stop.is_set())

    def test_absent_opt_in_keeps_original_fail_closed_behavior(self):
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw) / "journal"
            cases, protocol, requests = journal_fixture(folder, policy=False)
            transport = ScriptedTransport()
            journal = Journal(folder, protocol, transport, 2)
            with self.assertRaises(UnknownAttempt):
                journal.call(cases[0], "answer", requests[0])
            self.assertTrue(journal.stop.is_set())
            with self.assertRaises(UnknownAttempt):
                journal.call(cases[1], "answer", requests[1])
            self.assertEqual(len(transport.calls), 1)

    def test_wrong_metadata_and_unknown_transport_failure_are_never_terminal_exceptions(self):
        for fault in ("model", "wire", "text", "finish", "network"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as raw:
                folder = Path(raw) / "journal"
                cases, protocol, requests = journal_fixture(folder)
                transport = ScriptedTransport(fault)
                journal = Journal(folder, protocol, transport, 2)
                with self.assertRaises(UnknownAttempt):
                    journal.call(cases[0], "answer", requests[0])
                self.assertTrue(journal.stop.is_set())
                self.assertFalse(is_terminal_length(rows(folder / "outcomes.jsonl")[0], protocol,
                    rows(folder / "intents.jsonl")[0]["wire_sha256"]))
                with self.assertRaises(UnknownAttempt):
                    journal.call(cases[1], "answer", requests[1])
                self.assertEqual(len(transport.calls), 1)

    def test_retained_terminal_row_cannot_be_reclassified_after_metadata_changes(self):
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw) / "journal"
            cases, protocol, requests = journal_fixture(folder)
            transport = ScriptedTransport()
            journal = Journal(folder, protocol, transport, 2)
            with self.assertRaises(TerminalLengthReceipt):
                journal.call(cases[0], "answer", requests[0])
            values = rows(folder / "outcomes.jsonl")
            values[0]["wire_sha256"] = "b" * 64
            (folder / "outcomes.jsonl").write_text("".join(canonical(v) + "\n" for v in values))
            with self.assertRaises(UnknownAttempt):
                Journal(folder, protocol, transport, 2)
            self.assertEqual(len(transport.calls), 1)


class RunnerTerminalTests(unittest.TestCase):
    def run_fixture(self, root, answer, judge):
        output = continuation_fixture(root)
        report = continuation.run(output, answer, judge, max_new_calls=10, assessment_max_new_calls=10,
            workers=1, dense=Dense(), fixture_only=True)
        return output, report

    def test_length_answer_is_missing_and_later_arms_and_cases_continue_without_retry(self):
        with tempfile.TemporaryDirectory() as raw:
            answer = LengthAtFirstMatch(runner.IDENTITY,
                lambda system: not system.startswith(("Extract only reusable", "Independently assess whether")))
            judge = FixtureTransport(IDENTITY)
            output, report = self.run_fixture(Path(raw), answer, judge)
            self.assertEqual(report["completed_cases"], 2)
            self.assertEqual([report["arms"][a]["received"] for a in runner.ARMS], [1, 2, 2])
            first = output / "reader/cases/invented-case-0"
            missing = json.loads((first / "no_rag.terminal_invalid.json").read_text())
            self.assertFalse(missing["resend_permitted"])
            self.assertFalse((first / "no_rag.answer.json").exists())
            self.assertTrue((first / "raw_history_rag.answer.json").exists())
            self.assertTrue((output / "reader/cases/invented-case-1/managed_memory_rag.answer.json").exists())
            before = len(answer.calls), len(judge.calls)
            continuation.run(output, answer, judge, max_new_calls=0, assessment_max_new_calls=0,
                workers=1, dense=Dense(), fixture_only=True)
            self.assertEqual((len(answer.calls), len(judge.calls)), before)

    def test_length_target_preparation_keeps_answers_and_next_case_assessments(self):
        with tempfile.TemporaryDirectory() as raw:
            answer = FixtureTransport(runner.IDENTITY)
            judge = LengthAtFirstMatch(IDENTITY, lambda system: system == memory.TARGETS)
            output, report = self.run_fixture(Path(raw), answer, judge)
            self.assertTrue(judge.length_sent)
            self.assertTrue(all(report["arms"][a]["received"] == 2 for a in runner.ARMS))
            first = output / "memory/assessor_only/invented-case-0"
            second = output / "memory/assessor_only/invented-case-1"
            self.assertTrue((first / "target_unavailable.json").exists())
            self.assertFalse((first / "targets.json").exists())
            self.assertFalse(any(first.glob("*.score.json")))
            self.assertTrue(all((second / (a + ".score.json")).exists() for a in runner.ARMS))
            self.assertEqual(sum(r.get("finish_reason") == "length"
                for r in rows(output / "memory/outcomes.jsonl")), 1)

    def test_length_assertion_score_does_not_discard_answers_or_other_arms(self):
        with tempfile.TemporaryDirectory() as raw:
            answer = FixtureTransport(runner.IDENTITY)
            judge = LengthAtFirstMatch(IDENTITY, lambda system: system == memory.ADJUDICATE)
            output, report = self.run_fixture(Path(raw), answer, judge)
            self.assertTrue(all(report["arms"][a]["received"] == 2 for a in runner.ARMS))
            first = output / "memory/assessor_only/invented-case-0"
            self.assertFalse((first / "no_rag.score.json").exists())
            self.assertFalse(json.loads((first / "no_rag.unavailable.json").read_text())["resend_permitted"])
            self.assertTrue((first / "raw_history_rag.score.json").exists())
            self.assertTrue((first / "managed_memory_rag.score.json").exists())
            summary = json.loads((output / "memory/summary.json").read_text())
            self.assertEqual([summary["arms"][a]["scored_answers"] for a in runner.ARMS], [1, 2, 2])

    def test_length_memory_extraction_keeps_raw_arms_and_next_case(self):
        with tempfile.TemporaryDirectory() as raw:
            answer = LengthAtFirstMatch(runner.IDENTITY, lambda system: system.startswith("Extract only reusable"))
            output, report = self.run_fixture(Path(raw), answer, FixtureTransport(IDENTITY))
            self.assertEqual(report["completed_cases"], 2)
            self.assertEqual([report["arms"][a]["received"] for a in runner.ARMS], [2, 2, 1])
            first = output / "reader/cases/invented-case-0"
            self.assertTrue((first / "memory_preparation_failure.json").exists())
            self.assertTrue((first / "no_rag.answer.json").exists())
            self.assertTrue((first / "raw_history_rag.answer.json").exists())
            self.assertFalse((first / "managed_memory_rag.answer.json").exists())
            self.assertTrue((output / "reader/cases/invented-case-1/managed_memory_rag.answer.json").exists())

    def test_descendant_continuation_preserves_terminal_missing_answer_without_resend(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            answer = LengthAtFirstMatch(runner.IDENTITY,
                lambda system: not system.startswith(("Extract only reusable", "Independently assess whether")))
            output, report = self.run_fixture(root, answer, FixtureTransport(IDENTITY))
            descendant = root / "descendant"
            plan = continuation.prepare(output / "reader", output / "memory", descendant, fixture_only=True)
            self.assertEqual(plan["inherited_generation_terminal_invalid"], 1)
            self.assertEqual(plan["maximum_new_generation_requests"], 0)
            fresh_answer, fresh_judge = FixtureTransport(runner.IDENTITY), FixtureTransport(IDENTITY)
            repeated = continuation.run(descendant, fresh_answer, fresh_judge,
                max_new_calls=0, assessment_max_new_calls=0, workers=1, dense=Dense(), fixture_only=True)
            self.assertEqual((fresh_answer.calls, fresh_judge.calls), ([], []))
            self.assertEqual(repeated["arms"], report["arms"])


class LengthFirstQuality(QualityFixtureTransport):
    def __call__(self, payload):
        receipt = super().__call__(payload)
        if len(self.calls) == 1:
            receipt.update(text="{\"unfinished\":", finish_reason="length")
        return receipt


class WrongWireLengthQuality(LengthFirstQuality):
    def __call__(self, payload):
        receipt = super().__call__(payload)
        if len(self.calls) == 1:
            receipt["wire_sha256"] = "b" * 64
        return receipt


class QualityTerminalTests(unittest.TestCase):
    def test_length_grade_is_retained_as_invalid_while_other_grades_finish(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            reader, output = root / "reader", root / "quality_v3"
            source_fixture(reader)
            preflight = quality.prepare(reader, output, quality.JUDGE_IDENTITY, fixture_only=True)
            transport = LengthFirstQuality()
            report = quality.run(output, transport, preflight["planned_judge_requests"], workers=1, fixture_only=True)
            self.assertTrue(report["complete"])
            self.assertFalse(report["all_primary_scores_valid"])
            self.assertEqual(report["planned_primary"], 6)
            self.assertEqual(len(transport.calls), 6)
            outcomes = rows(output / "outcomes.jsonl")
            self.assertEqual(sum(r["status"] == "invalid" for r in outcomes), 1)
            self.assertEqual(sum(r["status"] == "received" for r in outcomes), 5)
            retained = rows(output / "reviews.jsonl")
            self.assertEqual(len(retained), 6)
            self.assertEqual(sum(r["review"]["status"] == "invalid" for r in retained), 1)
            self.assertEqual(sum(r["review"]["status"] == "completed" for r in retained), 5)
            self.assertEqual(json.loads((output / "last_invocation.json").read_text())["errors"], [])
            repeated = quality.run(output, transport, 0, workers=1, fixture_only=True)
            self.assertEqual(len(transport.calls), 6)
            self.assertEqual(repeated["grade_status_by_arm"], report["grade_status_by_arm"])

    def test_wrong_wire_length_is_not_converted_into_a_missing_grade_and_halts(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            reader, output = root / "reader", root / "quality_v3"
            source_fixture(reader)
            preflight = quality.prepare(reader, output, quality.JUDGE_IDENTITY, fixture_only=True)
            transport = WrongWireLengthQuality()
            report = quality.run(output, transport, preflight["planned_judge_requests"], workers=1, fixture_only=True)
            self.assertFalse(report["complete"])
            self.assertEqual(len(transport.calls), 1)
            self.assertEqual(rows(output / "reviews.jsonl"), [])
            self.assertTrue(json.loads((output / "last_invocation.json").read_text())["errors"])
            self.assertEqual(len(rows(output / "outcomes.jsonl")), 1)


if __name__ == "__main__":
    unittest.main()
