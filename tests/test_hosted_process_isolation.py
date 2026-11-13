"""Local-only regression tests for the hosted process/credential boundary."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock
import uuid


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "architecture_hosted_relay", ROOT / "evaluation/architecture-rerun/hosted_relay.py"
)
relay_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = relay_module
SPEC.loader.exec_module(relay_module)

IDENTITY = {
    "provider": "https://tokendance.space/gateway/v1",
    "model": "deepseek-v4.1-flash",
    "returned_model": "deepseek-v4-1-flash-260910",
    "model_family": "deepseek",
}
MODEL = IDENTITY["model"]


def unix_socket_available():
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM):
            return True
    except (AttributeError, OSError):
        return False


def payload():
    # Both outer keys and nested message keys intentionally have non-sorted order.
    return {
        "max_tokens": 64,
        "messages": [
            {"content": "Fictional local fixture. 保留顺序。", "role": "system"},
            {"content": "Return the fictional answer.", "role": "user"},
        ],
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "model": MODEL,
    }


class FixtureTransport:
    """Records the actual serialization at the final, credential-owning hop."""

    fixture_only = True

    def __init__(self):
        self.identity = dict(IDENTITY)
        self.wires = []
        self.receipts = []
        self._lock = threading.Lock()

    def __call__(self, request):
        wire = json.dumps(request, allow_nan=False)
        with self._lock:
            number = len(self.wires) + 1
            receipt = {
                "text": f"Fictional reply {number}.",
                "provider_model": IDENTITY["returned_model"],
                "finish_reason": "stop",
                "response_id": f"fixture-response-{number}",
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
                "wire_sha256": sha256(wire.encode()).hexdigest(),
            }
            self.wires.append(wire)
            self.receipts.append(receipt)
            return deepcopy(receipt)


@unittest.skipUnless(unix_socket_available(), "platform disallows AF_UNIX sockets; core admission tests still run")
class HostedRelayFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="relay-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "private"
        self.provider = FixtureTransport()
        self.relay = relay_module.start_relay(
            self.root, {MODEL: self.provider}, fixture_only=True
        )
        self.addCleanup(self.relay.close)

    def transport(self, job="fixture-job", *, cap=4, identity=None):
        self.relay.register_job(job, {MODEL: cap})
        return relay_module.RelayTransport(
            dict(IDENTITY if identity is None else identity),
            self.relay.endpoint,
            job,
            fixture_only=True,
        )

    def send_envelope(self, envelope):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(5)
            connection.connect(self.relay.endpoint)
            connection.sendall((json.dumps(envelope, allow_nan=False) + "\n").encode())
            with connection.makefile("rb") as response:
                return json.loads(response.readline())

    def envelope(self, *, fixture_only=True):
        request = payload()
        return {
            "request_id": str(uuid.uuid4()),
            "job_id": "fixture-job",
            "identity": dict(IDENTITY),
            "payload": request,
            "wire_sha256": sha256(json.dumps(request, allow_nan=False).encode()).hexdigest(),
            "fixture_only": fixture_only,
        }

    def test_disk_payload_order_and_provider_receipt_survive_verbatim(self):
        request_path = Path(self.temporary.name) / "request.json"
        request_path.write_text(json.dumps(payload(), ensure_ascii=False), encoding="utf-8")
        disk_request = json.loads(request_path.read_text(encoding="utf-8"))
        original_wire = json.dumps(disk_request, allow_nan=False)

        receipt = self.transport()(disk_request)

        self.assertEqual(self.provider.wires, [original_wire])
        self.assertEqual(receipt, self.provider.receipts[0])
        self.assertEqual(set(receipt), {
            "text", "provider_model", "finish_reason", "response_id", "usage", "wire_sha256"
        })
        self.assertEqual(receipt["wire_sha256"], sha256(original_wire.encode()).hexdigest())
        self.assertEqual(json.dumps(disk_request, allow_nan=False), original_wire)
        self.assertEqual(list(json.loads(self.provider.wires[0])["messages"][0]), ["content", "role"])

    def test_identical_repeated_judge_requests_make_two_provider_calls(self):
        transport = self.transport(job="independent-judge", cap=2)
        first = transport(payload())
        second = transport(payload())

        self.assertEqual(len(self.provider.wires), 2)
        self.assertEqual(self.provider.wires[0], self.provider.wires[1])
        self.assertNotEqual(first["response_id"], second["response_id"])
        self.assertEqual(first, self.provider.receipts[0])
        self.assertEqual(second, self.provider.receipts[1])
        # Exceeding the job cap cannot create a third real provider invocation.
        with self.assertRaises(Exception):
            transport(payload())
        self.assertEqual(len(self.provider.wires), 2)

    def test_unknown_job_is_rejected_before_provider_call(self):
        transport = relay_module.RelayTransport(
            dict(IDENTITY), self.relay.endpoint, "unregistered", fixture_only=True
        )
        with self.assertRaises(Exception):
            transport(payload())
        self.assertEqual(self.provider.wires, [])

    def test_fixture_mode_mismatch_is_rejected_before_provider_call(self):
        self.relay.register_job("fixture-job", {MODEL: 1})
        reply = self.send_envelope(self.envelope(fixture_only=False))
        self.assertEqual(reply["status"], "rejected")
        self.assertEqual(self.provider.wires, [])

    def test_hosted_client_does_not_fabricate_runtime_to_reach_fixture_relay(self):
        self.relay.register_job("fixture-job", {MODEL: 1})
        with self.assertRaises(Exception):
            transport = relay_module.RelayTransport(
                dict(IDENTITY), self.relay.endpoint, "fixture-job", fixture_only=False
            )
            transport(payload())
        self.assertEqual(self.provider.wires, [])

    def test_duplicate_request_uuid_is_rejected_without_replaying_or_caching(self):
        self.relay.register_job("fixture-job", {MODEL: 2})
        envelope = self.envelope()
        first = self.send_envelope(envelope)
        repeated = self.send_envelope(envelope)

        self.assertEqual(first["status"], "received")
        self.assertEqual(first["receipt"], self.provider.receipts[0])
        self.assertEqual(repeated["status"], "rejected")
        self.assertNotIn("receipt", repeated)
        self.assertEqual(len(self.provider.wires), 1)

    def test_model_identity_mismatch_is_rejected_before_provider_call(self):
        changed_identity = {**IDENTITY, "returned_model": "unapproved-returned-alias"}
        with self.assertRaises(Exception):
            self.transport(identity=changed_identity)(payload())
        self.assertEqual(self.provider.wires, [])

    def test_payload_policy_rejections_do_not_reach_provider(self):
        transport = self.transport(cap=1)
        invalid_requests = [
            {**payload(), "model": "unapproved-model"},
            {**payload(), "temperature": 1},
            {**payload(), "thinking": {"type": "enabled"}},
            {**payload(), "max_tokens": 4097},
            {**payload(), "messages": [{"role": "user", "content": 17}]},
        ]
        for request in invalid_requests:
            with self.subTest(request=request):
                with self.assertRaises(Exception):
                    transport(request)
                self.assertEqual(self.provider.wires, [])

        # Invalid requests must not reserve the one permitted provider attempt.
        self.assertEqual(transport(payload()), self.provider.receipts[0])
        self.assertEqual(len(self.provider.wires), 1)

    def test_journal_pairs_unique_intent_and_outcome_ids_without_cache(self):
        transport = self.transport(cap=2)
        transport(payload())
        transport(payload())

        journal = self.root / "events.jsonl"
        raw = journal.read_bytes()
        self.assertTrue(raw.endswith(b"\n"))
        events = [json.loads(line) for line in raw.splitlines()]
        intents = [row for row in events if row.get("event") == "intent"]
        outcomes = [row for row in events if row.get("event") == "outcome"]
        self.assertEqual(len(intents), 2)
        self.assertEqual(len(outcomes), 2)
        intent_ids = [row["request_id"] for row in intents]
        outcome_ids = [row["request_id"] for row in outcomes]
        self.assertEqual(len(set(intent_ids)), 2)
        self.assertCountEqual(intent_ids, outcome_ids)
        for request_id in intent_ids:
            matching = [row["event"] for row in events if row.get("request_id") == request_id]
            self.assertEqual(matching, ["intent", "outcome"])
        json.dumps(self.relay.snapshot(), allow_nan=False)

    def test_relay_root_and_unix_socket_are_private(self):
        self.assertEqual(stat.S_IMODE(self.root.stat().st_mode), 0o700)
        endpoint = Path(self.relay.endpoint)
        self.assertTrue(stat.S_ISSOCK(endpoint.stat().st_mode))
        self.assertEqual(stat.S_IMODE(endpoint.stat().st_mode), 0o600)


class ChildEnvironmentTests(unittest.TestCase):
    def test_secret_names_are_removed_case_insensitively_without_mutating_input(self):
        environment = {
            "PATH": "/usr/bin:/bin",
            "COLAB_RELEASE_TAG": "existing-nonsecret-marker",
            "LANG": "C.UTF-8",
            "API_KEY": "fixture-key",
            "Api_Token": "fixture-token",
            "service_secret": "fixture-secret",
            "DB_PASSWORD": "fixture-password",
            "GOOGLE_APPLICATION_CREDENTIALS": "/fixture/credentials.json",
            "CAPSTONE_ENDPOINT": "fixture-endpoint",
            "HTTP_AUTHORIZATION": "fixture-authorization",
            "BEARER_VALUE": "fixture-bearer",
            "PYTHONPATH": "/fixture/old-checkout",
            "PYTHONHOME": "/fixture/old-python",
            "PYTHONSTARTUP": "/fixture/startup.py",
        }
        original = dict(environment)

        result = relay_module.sanitized_child_environment(environment)

        self.assertEqual(environment, original)
        self.assertIsNot(result, environment)
        self.assertEqual(result["PATH"], environment["PATH"])
        self.assertEqual(result["COLAB_RELEASE_TAG"], environment["COLAB_RELEASE_TAG"])
        for name in environment.keys() - {"PATH", "COLAB_RELEASE_TAG", "LANG"}:
            self.assertNotIn(name, result)

    def test_environment_does_not_invent_a_hosted_runtime_marker(self):
        self.assertNotIn("COLAB_RELEASE_TAG", relay_module.sanitized_child_environment({"PATH": "/bin"}))
        self.assertNotIn("COLAB_RELEASE_TAG", relay_module.sanitized_child_environment({}))

    def test_real_python_subprocess_imports_fresh_source_not_parent_module_cache(self):
        module_name = "_hosted_process_fixture_sentinel"
        with tempfile.TemporaryDirectory(prefix="worker-import-test-") as temporary:
            source = Path(temporary) / f"{module_name}.py"
            source.write_text("VALUE = 'parent-cached'\n", encoding="utf-8")
            spec = importlib.util.spec_from_file_location(module_name, source)
            parent_module = importlib.util.module_from_spec(spec)
            prior_module = sys.modules.get(module_name)
            sys.modules[module_name] = parent_module
            try:
                spec.loader.exec_module(parent_module)
                source.write_text("VALUE = 'fresh-worker-source'\n", encoding="utf-8")
                environment = relay_module.sanitized_child_environment({
                    "PATH": os.defpath,
                    "PARENT_SENTINEL_SECRET": "fixture-only-secret",
                })
                script = (
                    "import importlib,json,os,sys; "
                    "sys.path.insert(0,sys.argv[1]); "
                    f"cached={module_name!r} in sys.modules; "
                    f"sentinel=importlib.import_module({module_name!r}); "
                    "print(json.dumps({'value':sentinel.VALUE,'cached':cached,'pid':os.getpid(),"
                    "'secret_present':'PARENT_SENTINEL_SECRET' in os.environ,"
                    "'hosted_marker_present':'COLAB_RELEASE_TAG' in os.environ}))"
                )
                completed = subprocess.run(
                    [sys.executable, "-I", "-B", "-c", script, temporary],
                    env=environment, check=True, text=True, capture_output=True, timeout=15,
                )
                result = json.loads(completed.stdout)
                self.assertEqual(parent_module.VALUE, "parent-cached")
                self.assertEqual(result["value"], "fresh-worker-source")
                self.assertFalse(result["cached"])
                self.assertNotEqual(result["pid"], os.getpid())
                self.assertFalse(result["secret_present"])
                self.assertFalse(result["hosted_marker_present"])
            finally:
                if prior_module is None:
                    sys.modules.pop(module_name, None)
                else:
                    sys.modules[module_name] = prior_module


class RelayAdmissionWithoutSocketsTests(unittest.TestCase):
    """Exercise real admission/journal logic even in socket-restricted CI."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="relay-core-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "private"
        self.provider = FixtureTransport()
        patcher = mock.patch.object(relay_module.Relay, "_start_server", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.relay = relay_module.start_relay(self.root, {MODEL: self.provider}, fixture_only=True)

    def envelope(self, request=None):
        request = payload() if request is None else request
        return {"request_id": str(uuid.uuid4()), "job_id": "fixture-job", "identity": dict(IDENTITY),
                "payload": request, "wire_sha256": sha256(json.dumps(request, allow_nan=False).encode()).hexdigest(),
                "fixture_only": True}

    def events(self):
        return [json.loads(line) for line in self.relay.events.read_bytes().split(b"\n") if line]

    def test_actual_admission_preserves_exact_wire_and_distinct_repeats(self):
        self.relay.register_job("fixture-job", {MODEL: 2})
        first = self.relay._request(self.envelope())
        second = self.relay._request(self.envelope())
        self.assertEqual(first["status"], "received")
        self.assertNotEqual(first["receipt"]["response_id"], second["receipt"]["response_id"])
        self.assertEqual(self.provider.wires, [json.dumps(payload(), allow_nan=False)] * 2)
        self.assertEqual(len([row for row in self.events() if row["event"] == "intent"]), 2)
        with self.assertRaises(ValueError):
            self.relay._request(self.envelope())
        self.assertEqual(len(self.provider.wires), 2)

    def test_invalid_message_or_wire_has_no_intent_or_provider_call(self):
        self.relay.register_job("fixture-job", {MODEL: 1})
        for request in [
            {**payload(), "messages": []},
            {**payload(), "messages": [{"role": "user", "content": 17}]},
            {**payload(), "messages": [{"role": "tool", "content": "no actual call"}]},
            {**payload(), "messages": [{"role": "system", "content": "x", "name": "extra"}]},
            {**payload(), "temperature": False},
            {**payload(), "extra": "unfrozen field"},
        ]:
            with self.subTest(request=request), self.assertRaises(ValueError):
                self.relay._request(self.envelope(request))
        changed_hash = self.envelope()
        changed_hash["wire_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.relay._request(changed_hash)
        self.assertEqual(self.provider.wires, [])
        self.assertEqual([row["event"] for row in self.events()], ["register"])

    def test_zero_cap_and_unknown_jobs_fail_before_provider(self):
        self.relay.register_job("fixture-job", {MODEL: 0})
        for job in ("fixture-job", "unregistered"):
            envelope = {**self.envelope(), "job_id": job}
            with self.subTest(job=job), self.assertRaises(ValueError):
                self.relay._request(envelope)
        self.assertEqual(self.provider.wires, [])

    def test_restoring_received_receipts_counts_budget_and_never_replays(self):
        self.relay.register_job("fixture-job", {MODEL: 2})
        envelope = self.envelope()
        self.relay._request(envelope)
        restored = relay_module.start_relay(self.root, {MODEL: self.provider}, fixture_only=True)
        self.assertEqual(restored.snapshot()["jobs"]["fixture-job"][MODEL]["admitted"], 1)
        with self.assertRaises(ValueError):
            restored._request(envelope)
        restored._request(self.envelope())
        self.assertEqual(len(self.provider.wires), 2)
        with self.assertRaises(ValueError):
            restored._request(self.envelope())

    def test_failed_parent_attempt_stays_counted_and_exception_text_is_not_exported(self):
        self.relay.register_job("fixture-job", {MODEL: 1})
        def fail(_):
            raise RuntimeError("fixture-only private transport detail")
        self.relay.transports[MODEL] = fail
        envelope = self.envelope()
        result = self.relay._request(envelope)
        self.assertEqual(result["status"], "error")
        self.assertNotIn("private transport detail", json.dumps(result))
        self.assertNotIn("private transport detail", self.relay.events.read_text())
        restored = relay_module.start_relay(self.root, {MODEL: self.provider}, fixture_only=True)
        with self.assertRaises(ValueError):
            restored._request(self.envelope())
        self.assertEqual(self.provider.wires, [])

    def test_parallel_admission_never_exceeds_registered_cap(self):
        from concurrent.futures import ThreadPoolExecutor
        self.relay.register_job("fixture-job", {MODEL: 3})
        def invoke(_):
            try:
                return self.relay._request(self.envelope())["status"]
            except ValueError:
                return "rejected"
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(invoke, range(16)))
        self.assertEqual(results.count("received"), 3)
        self.assertEqual(results.count("rejected"), 13)
        self.assertEqual(len(self.provider.wires), 3)

    def test_retained_partial_record_and_duplicate_outcome_are_rejected(self):
        self.relay.register_job("fixture-job", {MODEL: 1})
        self.relay._request(self.envelope())
        original = self.relay.events.read_bytes()
        with self.relay.events.open("ab") as stream:
            stream.write(original.splitlines()[-1] + b"\n")
        with self.assertRaises(ValueError):
            relay_module.start_relay(self.root, {MODEL: self.provider}, fixture_only=True)
        self.relay.events.write_bytes(original + b'{"partial"')
        with self.assertRaises(ValueError):
            relay_module.start_relay(self.root, {MODEL: self.provider}, fixture_only=True)

    def test_symlinked_root_is_rejected_before_resolution(self):
        link = self.root.parent / "linked"
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            relay_module.start_relay(link, {MODEL: self.provider}, fixture_only=True)


if __name__ == "__main__":
    unittest.main()
