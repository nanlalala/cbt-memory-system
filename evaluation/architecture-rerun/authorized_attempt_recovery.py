"""One explicit, auditable authorization for exact prior unknown attempts.

No import reads credentials, calls a provider, changes the old invocation or
claims that its unknown requests were free, rejected, or answered.
"""
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import threading

VERSION = "hosted-non-mab-authorized-attempt-recovery-v1"
ROOT = "/content/capstone-non-mab-restored-continuation-20261113-v4"
PREVIOUS_ROOT = "/content/capstone-non-mab-restored-continuation-20261113-v3"
PREVIOUS_NONCE = "fea6979d-faa8-4f40-9849-f0b4320fdcf9"
PREVIOUS_COMMIT = "cb224d323ab2eaee586822500ba1b152d6c0f3be"
AUTHORIZATION_TEXT_SHA256 = "b02da2d784f500db593df48f0b40503b114348afa0f1307de80a0ae8c2c94cb3"
MODEL = "deepseek-v4.1-flash"
JUDGE = "qwen3.5-plus"
ANSWER_IDENTITY = {"provider": "https://tokendance.space/gateway/v1", "model": MODEL,
                   "model_family": "deepseek", "returned_model": "deepseek-v4-1-flash-260910"}
ERROR_ARCHIVE = "/content/Capstone_Non_MAB_V3_Provider_Errors_PRIVATE_20261113.zip"
ERROR_ARCHIVE_SHA256 = "4d63b9170a983f41b32521cf6aa7db7ca260ad1391bdd20fd7c8a0833cfb3182"
ALLOWLIST = (
    {"cohort": "professional7", "case_id": "cbt-audio-video-24-cutoff-29", "purpose": "support_check",
     "key": "f2acc6beec912e5e6a88ad599fe27a2ef23a65d7c230d28d5e00e4063151ba96",
     "wire_sha256": "11209b26a88cdd10bf383985bf4d9dbc747e624217ee9a9083e86015a4f085b4",
     "request_id": "a5b87194-d38f-4290-ae4b-ab3d3f5ef48c",
     "protocol_sha256": "e2df4de4f8d78860287304f2b4d5d2daa68a0d99a6a3d1b0ddbcbacf39b6142d"},
    {"cohort": "professional7", "case_id": "cbt-audio-video-2-cutoff-20", "purpose": "support_check",
     "key": "af9abb7bd1e340bf704302b6fa175e163195c381dad2d8fa6ca0ea4019f54b88",
     "wire_sha256": "b5228d396f3c143b46085beaf32d4899027cf9f12051f809b383ad369a837c0d",
     "request_id": "f045d218-a768-4a95-bbce-a0d6c963552f",
     "protocol_sha256": "e2df4de4f8d78860287304f2b4d5d2daa68a0d99a6a3d1b0ddbcbacf39b6142d"},
    {"cohort": "cactus600", "case_id": "cactus-0115002036e1b7c3f2dbe79f81180635", "purpose": "support_check",
     "key": "0cf64ee653742ff36e1f38c36d1760952b096a3319bf02458a6de8f49469a175",
     "wire_sha256": "4c5bbd761208359032d711f06862d3facb6bc8e119fb358fcbb2cdd246c74229",
     "request_id": "d0b0f9cf-01b1-47cd-ac03-70554e6d3fa0",
     "protocol_sha256": "11fe33142fe476a09ee9082186b3b2b36d9d2f112494e5947c0cc8a8e65962d2"},
)
OLD_FORBIDDEN_WIRES = {
    "b6406640f48bc3d1d860b4290ee381fb72d40c83c6ff3572e9b7d19df8cebe5e",
    "41d5ab159a302e7cf8a5ac601bbf7e103128f0589b01738d2137d743ecb39c85",
    "0c7edebb9739470171c8985f21596b80052651e98d9292646559213edef4d728",
    "78bea788536d653b88f5d2036a39b42b1f352991473c582c474d494c90943fb7",
    "cba2b4c779b7efa63d534221722203dc482d0b67f059f9fe72fd3f4d5310d409",
    "2596c0bd08730995bc1b94b3ec72ba3c4c25501907173093bd61f615e98e17ab",
}


def digest(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    path = Path(path)
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("regular frozen recovery file required")
    result = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    file_hash(path)
    return json.loads(Path(path).read_text())


def rows(path):
    path = Path(path)
    if not path.exists():
        return []
    file_hash(path)
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ValueError("incomplete frozen recovery journal")
    return [json.loads(line) for line in data.split(b"\n") if line]


def _without_clock(value):
    if isinstance(value, dict):
        return {key: _without_clock(item) for key, item in value.items() if key != "verified_at_utc"}
    if isinstance(value, list):
        return [_without_clock(item) for item in value]
    return value


def _worker_is_running(pid, manifest):
    if type(pid) is not int or pid <= 0:
        raise ValueError("recorded recovery worker PID required")
    try:
        return str(manifest).encode() in (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return False


def validate(previous_root, namespace, *, sources, restoration, repo, zero_proof_validator, user_authorization_path):
    authorization = read(user_authorization_path)
    if (type(authorization.get("user_instruction")) is not str
            or sha256(authorization["user_instruction"].encode()).hexdigest() != AUTHORIZATION_TEXT_SHA256
            or authorization.get("previous_nonce") != PREVIOUS_NONCE
            or authorization.get("authorized_action") != "one_new_attempt_per_exact_listed_wire"):
        raise ValueError("the specific new user authorization is required")
    previous, repo = Path(previous_root), Path(repo)
    entry, status = read(previous / "entry.json"), read(previous / "status.json")
    if (entry.get("nonce") != PREVIOUS_NONCE or status.get("nonce") != PREVIOUS_NONCE
            or entry.get("version") != "hosted-non-mab-restored-connection-v3"
            or entry.get("selected_commit") != PREVIOUS_COMMIT or entry.get("sources") != sources
            or entry.get("mab_accessed") is not False or entry.get("allow_longmemeval_paid") is not False
            or status.get("status") not in {"finished", "stopped_without_retry"}):
        raise ValueError("the exact paused v3 invocation is required")
    stable = ("source_restore_archive_sha256", "source_manifest_sha256", "source_restoration_receipt",
              "source_restoration_receipt_sha256", "restored_original_files_verified", "restored_original_bytes_verified")
    if any(entry.get("source_restoration", {}).get(key) != restoration.get(key) for key in stable):
        raise ValueError("the restored original sources changed")
    prior = zero_proof_validator()
    if _without_clock(prior) != _without_clock(entry.get("preflight_predecessor")):
        raise ValueError("the two zero-call predecessor proofs changed")
    state = namespace.get("_capstone_non_mab_continuation_v1")
    if isinstance(state, dict) and state.get("nonce") == PREVIOUS_NONCE:
        thread, process = state.get("thread"), state.get("process")
        if ((isinstance(thread, threading.Thread) and thread.is_alive())
                or (process is not None and callable(getattr(process, "poll", None)) and process.poll() is None)):
            raise RuntimeError("the paused v3 controller/worker must first finish")
    if (previous / "relay/api.sock").exists() or (previous / "relay/api.sock").is_symlink():
        raise RuntimeError("the old relay is still active")
    if file_hash(ERROR_ARCHIVE) != ERROR_ARCHIVE_SHA256:
        raise ValueError("the preserved v3 audit archive differs")
    controls = {"entry.json", "status.json", "relay/events.jsonl", "operator_pause.json", "operator_quiescence.json",
                "source_preparation.json", "finite_budget_plan.json"}
    for name in controls:
        file_hash(previous / name)
    relay_rows = rows(previous / "relay/events.jsonl")
    intents = [row for row in relay_rows if row.get("event") == "intent"]
    outcomes = [row for row in relay_rows if row.get("event") == "outcome"]
    if (any(row.get("event") not in {"register", "intent", "outcome"} or row.get("fixture_only") is not False for row in relay_rows)
            or len(intents) != 3 or len(outcomes) != 3
            or any(row.get("status") != "unknown" for row in outcomes)):
        raise ValueError("only the three frozen unknown provider attempts are authorized")
    allowed = {row["wire_sha256"]: row for row in ALLOWLIST}
    if len(allowed) != 3 or set(allowed) & OLD_FORBIDDEN_WIRES:
        raise ValueError("the old six unknown sends cannot be included in this grant")
    by_request = {row.get("request_id"): row for row in outcomes}
    if (len(by_request) != 3 or {row.get("request_id") for row in intents} != {row["request_id"] for row in ALLOWLIST}
            or len({row.get("wire_sha256") for row in intents}) != 3
            or {row.get("wire_sha256") for row in intents} != set(allowed)
            or set(by_request) != {row["request_id"] for row in ALLOWLIST}):
        raise ValueError("duplicate frozen provider outcomes")
    for intent in intents:
        permit = allowed.get(intent.get("wire_sha256"))
        wire, payload = intent.get("serialized_request"), intent.get("payload")
        if (permit is None or intent.get("request_id") != permit["request_id"] or intent.get("model") != MODEL
                or intent.get("identity") != ANSWER_IDENTITY
                or intent.get("job_id") != PREVIOUS_NONCE + ":" + permit["cohort"] + "-continuation"
                or type(wire) is not str or sha256(wire.encode()).hexdigest() != permit["wire_sha256"]
                or json.loads(wire) != payload or payload.get("model") != MODEL
                or digest({"purpose": permit["purpose"], "payload": payload}) != permit["key"]):
            raise ValueError("a prior provider attempt does not match its exact authorized wire")
        outcome = by_request[permit["request_id"]]
        if any(outcome.get(key) != intent.get(key) for key in ("job_id", "model", "wire_sha256")):
            raise ValueError("the prior unknown outcome differs from its intent")
    child_intents, child_outcomes, frozen_protocols, cohort_protocols = {}, {}, {}, {}
    for cohort in ("professional7", "cactus600", "locomo1540"):
        folders = [previous / "studies" / cohort]
        if cohort != "locomo1540":
            folders = [folders[0] / "reader", folders[0] / "memory"]
        for folder in folders:
            for filename, target in (("intents.jsonl", child_intents), ("outcomes.jsonl", child_outcomes)):
                path = folder / filename
                for row in rows(path):
                    if row.get("key") in target:
                        raise ValueError("duplicate prior new-run journal key")
                    target[row["key"]] = row
                if path.exists():
                    controls.add(str(path.relative_to(previous)))
        protocol_path = previous / "studies" / cohort / ("reader/protocol.json" if cohort != "locomo1540" else "protocol.json")
        protocol = read(protocol_path)
        if protocol.get("protocol_sha256") != digest({key: value for key, value in protocol.items() if key != "protocol_sha256"}):
            raise ValueError("previous scientific protocol checksum changed")
        cohort_protocols[cohort] = protocol["protocol_sha256"]
        for name, expected in protocol.get("code_sha256", {}).items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or file_hash(repo / name) != expected:
                raise ValueError("scientific renderer, scoring or source code changed")
        if not protocol.get("code_sha256"):
            raise ValueError("frozen scientific source hashes required")
        frozen_protocols[str(protocol_path.relative_to(previous))] = file_hash(protocol_path)
    if set(child_intents) != {row["key"] for row in ALLOWLIST} or set(child_outcomes) != set(child_intents):
        raise ValueError("the child ledgers contain another or outstanding attempt")
    for permit in ALLOWLIST:
        intent, outcome = child_intents[permit["key"]], child_outcomes[permit["key"]]
        if (cohort_protocols[permit["cohort"]] != permit["protocol_sha256"]
                or any(intent.get(key) != permit[key] for key in ("case_id", "purpose", "protocol_sha256", "wire_sha256"))
                or any(outcome.get(key) != permit[key] for key in ("case_id", "purpose", "protocol_sha256"))
                or outcome.get("status") != "unknown"
                or type(intent.get("serialized_request")) is not str
                or sha256(intent["serialized_request"].encode()).hexdigest() != permit["wire_sha256"]
                or json.loads(intent["serialized_request"]) != intent.get("payload")
                or digest({"purpose": permit["purpose"], "payload": intent.get("payload")}) != permit["key"]):
            raise ValueError("the original child request/protocol binding differs")
    for manifest_path in (previous / "workers").rglob("execution_manifest.json"):
        manifest = read(manifest_path)
        process_path = manifest_path.parent / "process.json"
        process = read(process_path)
        if (manifest.get("nonce") != PREVIOUS_NONCE or manifest.get("selected_commit") != PREVIOUS_COMMIT
                or process.get("manifest_sha256") != file_hash(manifest_path)
                or _worker_is_running(process.get("pid"), manifest_path)):
            raise RuntimeError("a previous invocation worker differs or is still active")
        controls.update((str(manifest_path.relative_to(previous)), str(process_path.relative_to(previous))))
    return {"version": VERSION, "previous_run_root": str(previous), "previous_nonce": PREVIOUS_NONCE,
        "previous_commit": PREVIOUS_COMMIT, "previous_status_preserved": "unknown",
        "previous_cost_status": "unknown", "provider_http_status_claim": None,
        "user_authorization": {"authorization_text_sha256": AUTHORIZATION_TEXT_SHA256,
                               "private_manifest_path": str(Path(user_authorization_path).resolve()),
                               "private_manifest_sha256": file_hash(user_authorization_path),
                               "scope": "one_new_attempt_per_exact_listed_wire; full original cohorts continue"},
        "retry_allowlist": list(ALLOWLIST), "prior_paid_attempts": {MODEL: 3, JUDGE: 0},
        "previous_error_archive_sha256": ERROR_ARCHIVE_SHA256,
        "frozen_scientific_protocol_files_sha256": frozen_protocols,
        "previous_files_sha256": {name: file_hash(previous / name) for name in sorted(controls)},
        "zero_call_predecessor": prior, "old_root_preserved": True}


def assert_current(proof):
    previous = Path(proof["previous_run_root"])
    if (previous / "relay/api.sock").exists() or (previous / "relay/api.sock").is_symlink():
        raise RuntimeError("the old relay reappeared")
    for name, expected in proof["previous_files_sha256"].items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or file_hash(previous / name) != expected:
            raise ValueError("frozen prior recovery evidence changed")
    authorization = proof["user_authorization"]
    if file_hash(authorization["private_manifest_path"]) != authorization["private_manifest_sha256"]:
        raise ValueError("the explicit private user authorization changed")


def make_relay(base_class, root, transports, proof, *, nonce):
    """Parent-only global circuit breaker and exact-once new-attempt audit."""
    permits = {row["wire_sha256"]: row for row in proof["retry_allowlist"]}
    if not isinstance(nonce, str) or not nonce or nonce == PREVIOUS_NONCE:
        raise ValueError("a distinct new invocation nonce is required")
    root = Path(root)
    if (root / "events.jsonl").exists() or (root / "admission_halted.json").exists():
        raise RuntimeError("existing relay evidence cannot be recreated for another attempt")
    class AuthorizedAttemptRelay(base_class):
        def __init__(self, *args, **kwargs):
            self.admission_stopped = False
            self._retried_wires, self._retry_context = set(), {}
            super().__init__(*args, **kwargs)

        def _validate_limits(self, job_id, limits):
            if self.admission_stopped:
                raise RuntimeError("new unknown stopped all provider admission")
            return super()._validate_limits(job_id, limits)

        def _append(self, event):
            event = dict(event)
            if event.get("event") == "intent":
                if self.admission_stopped:
                    raise RuntimeError("new unknown stopped all provider admission")
                assert_current(proof)
                wire = event.get("wire_sha256")
                if wire in OLD_FORBIDDEN_WIRES:
                    raise ValueError("historical six unknown sends are never authorized for retry")
                permit = permits.get(wire)
                if permit is not None:
                    if (wire in self._retried_wires or event.get("model") != MODEL
                            or event.get("job_id") != nonce + ":" + permit["cohort"] + "-continuation"
                            or digest({"purpose": permit["purpose"], "payload": event.get("payload")}) != permit["key"]):
                        raise ValueError("authorized prior wire may have only one matching new attempt")
                    context = {"previous_nonce": PREVIOUS_NONCE, **permit,
                               "authorization_text_sha256": AUTHORIZATION_TEXT_SHA256,
                               "attempt_kind": "new_authorized_attempt"}
                    event["authorized_retry_of"] = context
                    self._retried_wires.add(wire)
                    self._retry_context[event["request_id"]] = context
            elif event.get("event") == "outcome":
                if event.get("request_id") in self._retry_context:
                    event["authorized_retry_of"] = self._retry_context[event["request_id"]]
                if event.get("status") == "unknown":
                    self.admission_stopped = True
            result = super()._append(event)
            if event.get("event") == "outcome" and event.get("status") == "unknown":
                marker = self.root / "admission_halted.json"
                marker.write_text(json.dumps({"status": "halted_on_new_unknown", "automatic_retry": False,
                    "request_id": event.get("request_id"), "wire_sha256": event.get("wire_sha256"),
                    "already_admitted_requests_may_finish": True}, sort_keys=True) + "\n")
            return result

        def snapshot(self):
            return {**super().snapshot(), "admission_stopped": self.admission_stopped,
                    "new_authorized_retry_count": len(self._retried_wires)}
    return AuthorizedAttemptRelay(root, transports)
