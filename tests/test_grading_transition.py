"""Admission-only quality-format transition: no sockets or model providers."""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock
import uuid


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "grading_transition_fixture", ROOT / "evaluation/architecture-rerun/grading_transition.py")
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)

GUIDE = "Fictional legacy quality guide. 字面证据引用。"
GUIDE_SHA = sha256(GUIDE.encode()).hexdigest()
MODEL = "qwen3.5-plus"
JOB = "cbt-v6-full"
IDENTITY = {"provider": "https://tokendance.space/gateway/v1", "model": MODEL,
            "returned_model": "qwen3.5-plus-2026-04-20", "model_family": "qwen"}


class FixtureRelay:
    """A delegate spy; never represents a scientific model receipt."""

    fixture_only = True

    def __init__(self):
        self.identities = {MODEL: dict(IDENTITY)}
        self._jobs = {JOB: {MODEL: 5032}, "quality-v2": {MODEL: 24}}
        self._mutex = threading.RLock()
        self._seen = set()
        self._admitted = Counter()
        self.calls = []
        self.result = {"status": "fixture_delegate_only"}

    def _request(self, request):
        self.calls.append(request)
        return self.result


def envelope(*, guide=GUIDE, job=JOB):
    payload = {"max_tokens": 2048, "messages": [
        {"content": guide, "role": "system"},
        {"content": "Fictional private text must not enter the operational audit.", "role": "user"}],
        "model": MODEL, "temperature": 0, "thinking": {"type": "disabled"}}
    wire = json.dumps(payload, allow_nan=False)
    return {"request_id": str(uuid.uuid4()), "job_id": job,
            "identity": dict(IDENTITY), "payload": payload,
            "wire_sha256": sha256(wire.encode()).hexdigest(), "fixture_only": True}


class GradingTransitionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="grading-transition-")
        self.addCleanup(self.temporary.cleanup)
        self.audit = Path(self.temporary.name) / "audit"
        self.relay = FixtureRelay()

    def install(self, **kwargs):
        return gate.install_quality_transition(self.relay, self.audit,
            system_prompt_sha256=GUIDE_SHA, fixture_only=True, **kwargs)

    def records(self):
        path = self.audit / "deferred.jsonl"
        return [json.loads(line) for line in path.read_bytes().split(b"\n") if line] if path.exists() else []

    def test_exact_legacy_job_is_deferred_before_delegate_or_budget_admission(self):
        self.install()
        request = envelope()
        before = deepcopy(request)
        result = self.relay._request(request)
        self.assertEqual(result["status"], "deferred")
        self.assertEqual(result["request_id"], request["request_id"])
        self.assertEqual(result["reason"], "superseded_evidence_format_v2")
        self.assertIs(result["provider_called"], False)
        self.assertEqual(request, before)
        self.assertEqual(self.relay.calls, [])
        self.assertEqual(self.relay._seen, set())
        self.assertEqual(self.relay._admitted, Counter())
        records = self.records()
        self.assertEqual(len(records), 1)
        for name in ("request_id", "job_id", "wire_sha256"):
            self.assertEqual(records[0][name], request[name])
        self.assertEqual(records[0]["model"], MODEL)
        self.assertIs(records[0]["provider_called"], False)

    def test_generation_memory_and_new_quality_jobs_delegate_verbatim(self):
        self.install()
        requests = [envelope(guide="Memory assessment, distinct from quality."),
                    envelope(job="quality-v2"), envelope(guide=GUIDE + " ")]
        for request in requests:
            original_wire = json.dumps(request["payload"], allow_nan=False)
            result = self.relay._request(request)
            self.assertIs(result, self.relay.result)
            self.assertIs(self.relay.calls[-1], request)
            self.assertEqual(json.dumps(request["payload"], allow_nan=False), original_wire)
        self.assertEqual(self.records(), [])

    def test_same_wire_distinct_requests_have_separate_deferrals_not_a_response_cache(self):
        self.install()
        first, second = envelope(), envelope()
        self.assertEqual(first["wire_sha256"], second["wire_sha256"])
        self.relay._request(first)
        self.relay._request(second)
        self.assertEqual({r["request_id"] for r in self.records()},
                         {first["request_id"], second["request_id"]})
        self.assertEqual(self.relay.calls, [])

    def test_a_deferred_uuid_is_rejected_on_replay_without_another_audit_row(self):
        self.install()
        request = envelope()
        self.relay._request(request)
        audit_before = (self.audit / "deferred.jsonl").read_bytes()
        with self.assertRaises(ValueError):
            self.relay._request(deepcopy(request))
        self.assertEqual((self.audit / "deferred.jsonl").read_bytes(), audit_before)
        self.assertEqual(self.relay.calls, [])

    def test_old_relay_journal_is_retained_byte_for_byte(self):
        journal = Path(self.temporary.name) / "old-events.jsonl"
        retained = b'{"event":"retained-original-relay-record"}\n'
        journal.write_bytes(retained)
        self.relay.events = journal
        self.install()
        self.relay._request(envelope())
        self.assertEqual(journal.read_bytes(), retained)

    def test_install_is_idempotent_and_does_not_rewrite_manifest(self):
        first = self.install()
        wrapper = self.relay._request
        manifest_before = (self.audit / "installation.json").read_bytes()
        second = self.install()
        self.assertEqual(first, second)
        self.assertIs(self.relay._request, wrapper)
        self.assertEqual((self.audit / "installation.json").read_bytes(), manifest_before)

    def test_changed_scope_hash_or_audit_directory_cannot_replace_installed_gate(self):
        self.install()
        original = self.relay._request
        cases = [dict(system_prompt_sha256="0" * 64), dict(job_id="quality-v2"),
                 dict(audit_directory=Path(self.temporary.name) / "other")]
        for changes in cases:
            values = dict(system_prompt_sha256=GUIDE_SHA, fixture_only=True,
                          audit_directory=self.audit)
            values.update(changes)
            audit = values.pop("audit_directory")
            with self.subTest(changes=changes), self.assertRaises((ValueError, RuntimeError)):
                gate.install_quality_transition(self.relay, audit, **values)
        self.assertIs(self.relay._request, original)

    def test_already_admitted_uuid_is_not_misreported_as_never_sent(self):
        self.install()
        request = envelope()
        self.relay._seen.add(request["request_id"])
        try:
            result = self.relay._request(request)
        except (ValueError, RuntimeError):
            pass
        else:
            self.assertNotEqual(result.get("status"), "deferred")
        self.assertEqual(self.relay.calls, [])
        self.assertEqual(self.records(), [])

    def test_invalid_matching_uuid_wire_or_identity_cannot_claim_valid_deferral(self):
        self.install()
        requests = []
        for key, value in (("request_id", "not-a-uuid"), ("wire_sha256", "0" * 64),
                           ("fixture_only", False)):
            request = envelope()
            request[key] = value
            requests.append(request)
        foreign = envelope()
        foreign["identity"]["returned_model"] = "different-model"
        requests.append(foreign)
        for request in requests:
            with self.subTest(request_key=request["request_id"]):
                try:
                    result = self.relay._request(request)
                except (ValueError, RuntimeError):
                    pass
                else:
                    self.assertNotEqual(result.get("status"), "deferred")
        self.assertEqual(self.relay.calls, [])
        self.assertEqual(self.records(), [])

    def test_untrusted_user_text_and_credentials_are_absent_from_operational_audit(self):
        self.install()
        self.relay._request(envelope())
        raw = b"\n".join(path.read_bytes() for path in self.audit.iterdir() if path.is_file())
        self.assertNotIn(b"Fictional private text", raw)
        self.assertNotIn(GUIDE.encode(), raw)
        self.assertNotIn(b"serialized_request", raw)
        self.assertNotIn(b'"payload"', raw)
        record = self.records()[0]
        hashes = [value for key, value in record.items() if key.endswith("sha256")]
        self.assertGreaterEqual(len(hashes), 3)
        self.assertTrue(all(isinstance(v, str) and len(v) == 64 for v in hashes))

    def test_parallel_deferrals_are_complete_physical_lf_records(self):
        self.install()
        requests = [envelope() for _ in range(24)]
        errors = []
        def one(request):
            try:
                self.relay._request(request)
            except Exception as exc:
                errors.append(exc)
        workers = [threading.Thread(target=one, args=(request,)) for request in requests]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(3)
            self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        records = self.records()
        self.assertEqual(len(records), len(requests))
        self.assertEqual({r["request_id"] for r in records}, {r["request_id"] for r in requests})
        self.assertEqual(self.relay.calls, [])
        self.assertTrue((self.audit / "deferred.jsonl").read_bytes().endswith(b"\n"))

    def test_audit_fsync_failure_is_fail_closed_before_provider(self):
        self.install()
        with mock.patch.object(gate.os, "fsync", side_effect=OSError("fixture disk failure")):
            try:
                result = self.relay._request(envelope())
            except (OSError, RuntimeError):
                pass
            else:
                self.assertNotIn(result.get("status"), {"received", "deferred"})
        self.assertEqual(self.relay.calls, [])
        self.assertEqual(self.relay._seen, set())
        with self.assertRaises(RuntimeError):
            self.relay._request(envelope())
        self.assertEqual(self.relay.calls, [])

    def test_valid_saved_audit_is_restored_without_reclassifying_or_replaying_requests(self):
        self.install()
        request = envelope()
        self.relay._request(request)
        saved = (self.audit / "deferred.jsonl").read_bytes()
        original_installation = (self.audit / "installation.json").read_bytes()
        replacement = FixtureRelay()
        gate.install_quality_transition(replacement, self.audit,
            system_prompt_sha256=GUIDE_SHA, fixture_only=True)
        with self.assertRaises(ValueError):
            replacement._request(request)
        self.assertEqual(replacement.calls, [])
        self.assertEqual((self.audit / "deferred.jsonl").read_bytes(), saved)
        self.assertEqual((self.audit / "installation.json").read_bytes(), original_installation)
        next_request = envelope()
        self.assertEqual(replacement._request(next_request)["status"], "deferred")
        self.assertEqual(len(self.records()), 2)

    def test_saved_audit_hash_tampering_is_rejected_before_installation(self):
        self.install()
        self.relay._request(envelope())
        path = self.audit / "deferred.jsonl"
        row = json.loads(path.read_text())
        row["wire_sha256"] = "0" * 64
        path.write_text(json.dumps(row) + "\n")
        replacement = FixtureRelay()
        original = replacement._request
        with self.assertRaises(ValueError):
            gate.install_quality_transition(replacement, self.audit,
                system_prompt_sha256=GUIDE_SHA, fixture_only=True)
        self.assertEqual(replacement._request, original)
        self.assertEqual(replacement.calls, [])

    def test_hosted_default_does_not_accept_a_fixture_relay(self):
        with self.assertRaises((ValueError, RuntimeError)):
            gate.install_quality_transition(self.relay, self.audit,
                system_prompt_sha256=GUIDE_SHA)

    def test_fixture_mode_cannot_wrap_a_real_relay(self):
        self.relay.fixture_only = False
        with self.assertRaises((ValueError, RuntimeError)):
            self.install()

    def test_malformed_prompt_fingerprint_and_unregistered_job_are_rejected(self):
        for fingerprint in ("", "z" * 64, "0" * 63, 12):
            with self.subTest(fingerprint=fingerprint), self.assertRaises((ValueError, TypeError)):
                gate.install_quality_transition(self.relay, self.audit,
                    system_prompt_sha256=fingerprint, fixture_only=True)
        self.relay._jobs = {}
        with self.assertRaises((ValueError, RuntimeError)):
            self.install()

    def test_symlinked_audit_directory_is_not_followed(self):
        target = Path(self.temporary.name) / "target"
        target.mkdir()
        self.audit.symlink_to(target, target_is_directory=True)
        with self.assertRaises((ValueError, RuntimeError)):
            self.install()
        self.assertEqual(list(target.iterdir()), [])

    def test_an_inflight_delegate_finishes_unchanged_while_future_quality_is_deferred(self):
        started, release = threading.Event(), threading.Event()
        result = []
        def slow_delegate(request):
            self.relay.calls.append(request)
            started.set()
            if not release.wait(3):
                raise RuntimeError("fixture release timeout")
            return self.relay.result
        self.relay._request = slow_delegate
        original_request = envelope(guide="Already admitted memory assessment.")
        worker = threading.Thread(target=lambda: result.append(self.relay._request(original_request)))
        worker.start()
        self.assertTrue(started.wait(2))
        try:
            self.install()
            self.assertEqual(self.relay._request(envelope())["status"], "deferred")
            self.assertTrue(worker.is_alive())
        finally:
            release.set()
            worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(result, [self.relay.result])
        self.assertEqual(self.relay.calls, [original_request])


if __name__ == "__main__":
    unittest.main()
