"""Private local transport for isolated workers in the same genuine Colab VM.

Only the parent kernel owns provider transports/credentials. RPC never transfers
credentials, retries a request, or caches a response. Identical judge repeats are
separate provider calls. Scientific request/receipt journals remain in each run.

Usage in the existing hosted kernel (no secret is passed to launch_worker)::

    api = runpy.run_path(str(new_repo / "evaluation/architecture-rerun/hosted_relay.py"))
    relay = api["start_relay"]("/content/cbt-relay-v1", {
        answer.identity["model"]: answer, judge.identity["model"]: judge})
    relay.register_job("prepare", {answer.identity["model"]: 0,
                                   judge.identity["model"]: 0})
    process = api["launch_worker"](relay, repo=new_repo, job_id="prepare",
        action="cbt-prepare", run_root="/content/cbt-isolated-rerun",
        worker_root="/content/cbt-isolated-prepare-worker",
        inherited_memory_root="/content/cbt-v4-cbt-full/professional7_memory")

Wait for process.poll() == 0 and inspect complete.json before starting the paid
job. Register the paid job with the already approved finite model allowances;
use a new worker_root and action="cbt-run" with the same run_root. Source-only
prepare is not billed. Each helper directory records exact copied helper hashes,
selected commit and process ID. A judge-only worker can select an older frozen
checkout without modifying it: action="formal-judge", cohort="locomo1540",
max_new_calls=5082 and the existing source/generation run_root. Its selected code
must match the retained study protocol. No LongMemEval action is implied here.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import socket
import socketserver
import subprocess
import sys
import threading
import uuid

VERSION = "hosted-kernel-relay-v1"
MAX_PACKET_BYTES = 12_000_000
MAX_WIRE_CHARS = 96_000
RECEIPT_KEYS = {"text", "provider_model", "finish_reason", "response_id", "usage", "wire_sha256"}
PROVIDER = "https://tokendance.space/gateway/v1"


class RelayError(RuntimeError):
    pass


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _hosted():
    if not (os.environ.get("COLAB_RELEASE_TAG") and Path("/content").is_dir()
            and importlib.util.find_spec("google.colab") is not None):
        raise RuntimeError("genuine hosted Colab required; no runtime flags are fabricated")


def _json(value):
    return json.dumps(value, allow_nan=False)


def _parse(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def _identity(value):
    if (type(value) is not dict or set(value) != {"provider", "model", "returned_model", "model_family"}
            or value["provider"] != PROVIDER or not all(type(v) is str and v for v in value.values())):
        raise ValueError("fixed provider identity required")
    return dict(value)


def _private_directory(path, fixture_only):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("symlinked relay directory")
    path = path.resolve()
    if not fixture_only and (not path.is_relative_to("/content") or path == Path("/content")):
        raise ValueError("a named private /content directory is required")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


def sanitized_child_environment(environment=None):
    """Preserve real runtime configuration but never pass credential-like vars."""
    source = os.environ if environment is None else environment
    forbidden = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "CAPSTONE", "AUTH", "BEARER")
    return {str(key): str(value) for key, value in source.items()
            if str(key).upper() not in {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"}
            and not any(word in str(key).upper() for word in forbidden)}


class Relay:
    def __init__(self, root, transports_by_model, *, fixture_only=False):
        if type(fixture_only) is not bool:
            raise ValueError("explicit Boolean fixture mode required")
        if not fixture_only:
            _hosted()
        self.fixture_only = bool(fixture_only)
        self.root = _private_directory(root, fixture_only)
        self.endpoint = str(self.root / "api.sock")
        if len(self.endpoint.encode()) >= 104:
            raise ValueError("Unix socket path is too long")
        self.transports, self.identities = {}, {}
        for model, transport in transports_by_model.items():
            identity = _identity(getattr(transport, "identity", None))
            if model != identity["model"] or not callable(transport):
                raise ValueError("transport/model mismatch")
            self.transports[model], self.identities[model] = transport, identity
        if not self.transports:
            raise ValueError("at least one parent-owned transport is required")
        self._mutex = threading.RLock()
        self._jobs, self._seen, self._admitted = {}, set(), Counter()
        self.events = self.root / "events.jsonl"
        self._restore()
        self._server = None
        self._start_server()

    def _start_server(self):
        if Path(self.endpoint).exists():
            raise ValueError("existing relay socket; do not replace an active endpoint")
        relay = self
        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                self.connection.settimeout(600)
                try:
                    line = self.rfile.readline(MAX_PACKET_BYTES + 1)
                    if len(line) > MAX_PACKET_BYTES or not line.endswith(b"\n"):
                        raise ValueError("incomplete or oversized local request")
                    request = _parse(line)
                    reply = relay._request(request)
                except Exception as exc:
                    reply = {"status": "rejected", "error_type": type(exc).__name__}
                encoded = (_json(reply) + "\n").encode()
                if len(encoded) > MAX_PACKET_BYTES:
                    encoded = b'{"status":"error","error_type":"OversizedReceipt"}\n'
                try:
                    self.wfile.write(encoded)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    # The real outcome is already fsynced. No resend occurs.
                    return
        class Server(socketserver.ThreadingUnixStreamServer):
            daemon_threads = False
            block_on_close = True
        self._server = Server(self.endpoint, Handler)
        os.chmod(self.endpoint, 0o600)
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="colab-parent-api-relay", daemon=True)
        self._thread.start()

    def _restore(self):
        if not self.events.exists():
            return
        if self.events.is_symlink():
            raise ValueError("symlinked relay events")
        intents, outcomes = {}, set()
        # Iterate physical LF records; legal Unicode separators inside text do
        # not split JSON. Retained full wires can exceed RAM if loaded at once.
        with self.events.open("rb") as stream:
            for line in stream:
                if not line.endswith(b"\n"):
                    raise ValueError("incomplete relay journal retained for reconciliation")
                row = _parse(line)
                if row.get("fixture_only") is not self.fixture_only:
                    raise ValueError("relay fixture/hosted provenance mismatch")
                event = row["event"]
                if event == "register":
                    job, limits = row["job_id"], row["limits"]
                    self._validate_limits(job, limits)
                    old = self._jobs.get(job)
                    if old is not None and old != limits:
                        raise ValueError("conflicting prior relay limits")
                    self._jobs[job] = limits
                elif event == "intent":
                    request_id, job, model = row["request_id"], row["job_id"], row["model"]
                    if (request_id in self._seen or str(uuid.UUID(request_id)) != request_id or
                            row["identity"] != self.identities.get(model)):
                        raise ValueError("duplicate or mismatched retained relay intent")
                    cap = self._jobs.get(job, {}).get(model)
                    if cap is None or self._admitted[(job, model)] >= cap:
                        raise ValueError("retained relay intent exceeds its registered allowance")
                    wire = row["serialized_request"]
                    if (sha256(wire.encode()).hexdigest() != row["wire_sha256"] or
                            _parse(wire) != row["payload"] or row["payload"].get("model") != model):
                        raise ValueError("retained relay request identity differs")
                    self._seen.add(request_id)
                    self._admitted[(job, model)] += 1
                    intents[request_id] = (job, model, row["wire_sha256"])
                elif event == "outcome":
                    request_id = row["request_id"]
                    if (request_id in outcomes or intents.get(request_id) !=
                            (row["job_id"], row["model"], row["wire_sha256"])):
                        raise ValueError("unmatched or duplicate retained relay outcome")
                    if row["status"] == "received":
                        receipt = row["receipt"]
                        if (type(receipt) is not dict or set(receipt) != RECEIPT_KEYS or
                                receipt["wire_sha256"] != row["wire_sha256"] or
                                receipt["provider_model"] != self.identities[row["model"]]["returned_model"]):
                            raise ValueError("retained relay receipt differs")
                    elif row["status"] != "unknown":
                        raise ValueError("unrecognized retained relay outcome status")
                    outcomes.add(request_id)
                else:
                    raise ValueError("unknown retained relay event")
        # Unknown prior requests stay counted; nothing is automatically replayed.

    def _append(self, event):
        row = {**event, "fixture_only": self.fixture_only, "at_utc": _utc()}
        with self.events.open("a", encoding="utf-8") as stream:
            stream.write(_json(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _validate_limits(self, job_id, limits):
        if (type(job_id) is not str or not job_id or len(job_id) > 160 or
                type(limits) is not dict or not limits or
                any(model not in self.transports or type(cap) is not int or cap < 0
                    for model, cap in limits.items())):
            raise ValueError("explicit known job/model finite limits are required")

    def register_job(self, job_id, limits):
        self._validate_limits(job_id, limits)
        with self._mutex:
            prior = self._jobs.get(job_id)
            if prior is not None:
                if prior != limits:
                    raise ValueError("registered limits are immutable")
                return
            self._append({"event": "register", "job_id": job_id, "limits": dict(limits)})
            self._jobs[job_id] = dict(limits)

    def _request(self, request):
        if (type(request) is not dict or set(request) != {
                "request_id", "job_id", "identity", "payload", "wire_sha256", "fixture_only"}
                or request["fixture_only"] is not self.fixture_only):
            raise ValueError("local request schema/mode mismatch")
        request_id = request["request_id"]
        if type(request_id) is not str or str(uuid.UUID(request_id)) != request_id:
            raise ValueError("a canonical unique request UUID is required")
        identity = _identity(request["identity"])
        payload = request["payload"]
        model, job = identity["model"], request["job_id"]
        if (type(job) is not str or type(payload) is not dict or
                identity != self.identities.get(model) or payload.get("model") != model or
                set(payload) != {"model", "messages", "temperature", "thinking", "max_tokens"} or
                type(payload.get("temperature")) not in {int, float} or
                payload.get("temperature") != 0 or payload.get("thinking") != {"type": "disabled"} or
                type(payload.get("max_tokens")) is not int or not 1 <= payload["max_tokens"] <= 4096):
            raise ValueError("payload/provider/sampling mismatch")
        messages = payload["messages"]
        if (type(messages) is not list or not messages or any(
                type(message) is not dict or set(message) != {"role", "content"} or
                message["role"] not in {"system", "user", "assistant"} or
                type(message["content"]) is not str for message in messages)):
            raise ValueError("fixed text-message schema required")
        wire = _json(payload)
        wire_hash = sha256(wire.encode()).hexdigest()
        if request["wire_sha256"] != wire_hash or len(wire) > MAX_WIRE_CHARS:
            raise ValueError("actual local/provider wire identity or budget mismatch")
        with self._mutex:
            cap = self._jobs.get(job, {}).get(model)
            if request_id in self._seen:
                raise ValueError("duplicate request UUID; no automatic replay")
            if cap is None or self._admitted[(job, model)] >= cap:
                raise ValueError("job/model call allowance absent or exhausted")
            self._append({"event": "intent", "request_id": request_id, "job_id": job,
                          "model": model, "identity": identity, "payload": payload,
                          "serialized_request": wire, "wire_sha256": wire_hash})
            self._seen.add(request_id)
            self._admitted[(job, model)] += 1
        try:
            receipt = self.transports[model](payload)
            if (type(receipt) is not dict or set(receipt) != RECEIPT_KEYS or
                    receipt.get("wire_sha256") != wire_hash or
                    receipt.get("provider_model") != identity["returned_model"]):
                raise ValueError("parent provider receipt/wire identity differs")
            # Validate serialization before recording/returning the actual receipt.
            _json(receipt)
            outcome = {"event": "outcome", "status": "received", "request_id": request_id,
                       "job_id": job, "model": model, "wire_sha256": wire_hash,
                       "receipt": receipt}
            with self._mutex:
                self._append(outcome)
            return {"status": "received", "request_id": request_id, "receipt": receipt}
        except Exception as exc:
            with self._mutex:
                self._append({"event": "outcome", "status": "unknown", "request_id": request_id,
                              "job_id": job, "model": model, "wire_sha256": wire_hash,
                              "error_type": type(exc).__name__})
            # Exception text/locals may contain confidential transport details.
            return {"status": "error", "request_id": request_id, "error_type": type(exc).__name__}

    def snapshot(self):
        with self._mutex:
            return {"version": VERSION, "fixture_only": self.fixture_only, "endpoint": self.endpoint,
                    "jobs": {job: {model: {"maximum": cap,
                        "admitted": self._admitted[(job, model)]} for model, cap in caps.items()}
                        for job, caps in self._jobs.items()}}

    def close(self):
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()  # drains already admitted handler threads
        if Path(self.endpoint).exists():
            Path(self.endpoint).unlink()


def start_relay(root, transports_by_model, *, fixture_only=False):
    return Relay(root, transports_by_model, fixture_only=fixture_only)


class RelayTransport:
    def __init__(self, identity, endpoint, job_id, *, fixture_only=False, timeout=600):
        if type(fixture_only) is not bool:
            raise ValueError("explicit Boolean fixture mode required")
        if not fixture_only:
            _hosted()
        self.identity, self.endpoint, self.job_id = _identity(identity), str(endpoint), str(job_id)
        self.fixture_only, self.timeout = bool(fixture_only), timeout
        if not fixture_only and not Path(self.endpoint).resolve().is_relative_to("/content"):
            raise ValueError("hosted relay must remain in /content")

    def __call__(self, payload):
        wire = _json(payload)
        request_id = str(uuid.uuid4())
        envelope = {"request_id": request_id, "job_id": self.job_id, "identity": self.identity,
                    "payload": payload, "wire_sha256": sha256(wire.encode()).hexdigest(),
                    "fixture_only": self.fixture_only}
        raw = (_json(envelope) + "\n").encode()
        if len(raw) > MAX_PACKET_BYTES:
            raise ValueError("oversized local envelope")
        # Exactly one connect/send/read attempt. Network ambiguity is not retried.
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(self.timeout)
            connection.connect(self.endpoint)
            connection.sendall(raw)
            with connection.makefile("rb") as response:
                line = response.readline(MAX_PACKET_BYTES + 1)
        if len(line) > MAX_PACKET_BYTES or not line.endswith(b"\n"):
            raise RelayError("incomplete local response; no automatic retry")
        value = _parse(line)
        if value.get("status") != "received" or value.get("request_id") != request_id:
            raise RelayError("parent relay rejected or could not reconcile this request")
        receipt = value["receipt"]
        if (type(receipt) is not dict or set(receipt) != RECEIPT_KEYS or
                receipt["wire_sha256"] != envelope["wire_sha256"] or
                receipt["provider_model"] != self.identity["returned_model"]):
            raise RelayError("local receipt identity differs")
        return receipt


def launch_worker(relay, *, repo, job_id, action, run_root, worker_root,
                  stage="all", inherited_memory_root=None, cohort="locomo1540",
                  max_new_calls=None, worker_source=None, fixture_only=False):
    """Launch an isolated interpreter without exporting provider credentials.

    Helper scripts are byte-copied outside project roots, with explicit hashes.
    This permits a new orchestration helper to grade an unchanged older checkout.
    """
    if not fixture_only:
        _hosted()
    if fixture_only:
        raise ValueError("fixture launcher does not execute real study actions")
    if action not in {"cbt-prepare", "cbt-run", "formal-judge"}:
        raise ValueError("unknown isolated worker action")
    limits = relay.snapshot()["jobs"].get(job_id)
    if limits is None:
        raise ValueError("register the finite job/model limits before launching")
    if action == "cbt-prepare" and any(row["maximum"] != 0 for row in limits.values()):
        raise ValueError("source-only preparation must have zero provider allowances")
    if action != "cbt-prepare" and not any(row["maximum"] > row["admitted"] for row in limits.values()):
        raise ValueError("worker job has no remaining provider allowance")
    repo, run_root = Path(repo).resolve(), Path(run_root).resolve()
    if not repo.is_relative_to("/content") or not run_root.is_relative_to("/content"):
        raise ValueError("selected checkout and run must remain in /content")
    if not all((repo / name).is_dir() for name in ("modules/agent-integration", "evaluation/cbt-memory-effect")):
        raise ValueError("selected checkout is unavailable")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        raise ValueError("selected checkout must be clean before worker launch")
    root = _private_directory(worker_root, False)
    source = Path(worker_source).resolve() if worker_source else Path(__file__).with_name("hosted_worker.py").resolve()
    helper = Path(__file__).resolve()
    scripts = {"hosted_worker.py": source.read_bytes(), "hosted_relay.py": helper.read_bytes()}
    for name, content in scripts.items():
        target = root / name
        if target.exists() and (target.is_symlink() or target.read_bytes() != content):
            raise ValueError("isolated worker helper changed; use a new worker directory")
        if not target.exists():
            target.write_bytes(content)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    manifest = {"version": VERSION, "created_at_utc": _utc(), "selected_repo": str(repo),
                "selected_commit": commit, "job_id": job_id, "action": action, "stage": stage,
                "run_root": str(run_root), "socket": relay.endpoint, "cohort": cohort,
                "max_new_calls": max_new_calls,
                "inherited_memory_root": str(inherited_memory_root) if inherited_memory_root else None,
                "helper_sha256": {name: sha256(content).hexdigest() for name, content in scripts.items()},
                "credentials": "provider key remains only in the parent Colab kernel",
                "execution": "independent Python process in the same genuine hosted Colab VM",
                "relay_limits": limits}
    manifest_path = root / "execution_manifest.json"
    if manifest_path.exists():
        raise ValueError("worker directory already has an execution manifest")
    manifest_path.write_text(_json(manifest) + "\n")
    argv = [sys.executable, "-B", str(root / "hosted_worker.py"), "--manifest", str(manifest_path)]
    with (root / "worker.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(argv, cwd=root, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, env=sanitized_child_environment(),
                                   start_new_session=True)
    (root / "process.json").write_text(_json({"pid": process.pid, "argv": argv,
        "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest()}) + "\n")
    return process
