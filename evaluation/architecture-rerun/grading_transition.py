"""Defer a superseded grading format before an existing relay admits a send.

This is an operational transition, not a change to a frozen generator, rubric,
receipt or protocol. Existing requests drain through their original dispatch.
Only a new request for the exact named old job/model/system prompt is deferred.
The old child can record a transport failure; the separate fsynced audit binds
that failure to a known pre-provider non-send. Nothing is automatically retried.

Usage in the genuine parent Colab kernel::

    transition = runpy.run_path(str(new_repo / 'evaluation/architecture-rerun/grading_transition.py'))
    transition['install_quality_transition'](existing_relay,
        '/content/cbt-v1-quality-transition', system_prompt_sha256=old_guide_sha)

The existing relay instance, frozen files, provider transports and credentials
are retained. All nonmatching requests use the untouched original dispatcher.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import uuid

VERSION = "cbt-quality-format-transition-v1"
REASON = "superseded_evidence_format_v2"
JOB_ID = "cbt-v6-full"
MODEL = "qwen3.5-plus"
PROVIDER = "https://tokendance.space/gateway/v1"
_SOURCE_PATH = Path(__file__).resolve()
_SOURCE_SHA256 = sha256(_SOURCE_PATH.read_bytes()).hexdigest()
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _digest(value):
    return sha256(_json(value).encode()).hexdigest()


def _hosted():
    if not (os.environ.get("COLAB_RELEASE_TAG") and Path("/content").is_dir()
            and importlib.util.find_spec("google.colab") is not None):
        raise RuntimeError("genuine hosted Colab required; no runtime flag is fabricated")


def _directory(value, fixture_only):
    path = Path(value).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("symlinked transition audit directory")
    path = path.resolve()
    if not fixture_only and (not path.is_relative_to("/content") or path == Path("/content")):
        raise ValueError("a named private /content audit directory is required")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


def _write_installation(path, value):
    body = (_json(value) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())


def _append_audit(state, row):
    path = state["audit_path"]
    if path.is_symlink() or (path.stat().st_size if path.exists() else 0) != state["audit_size"]:
        raise ValueError("transition audit changed outside the gate")
    row = {**row, "previous_sha256": state["previous_sha256"]}
    row["event_sha256"] = _digest(row)
    body = (_json(row) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "ab") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("transition audit must be a regular file")
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    state["audit_size"] += len(body)
    state["previous_sha256"] = row["event_sha256"]


def _load_audit(state):
    path, install = state["audit_path"], state["installation"]
    if path.is_symlink():
        raise ValueError("symlinked transition audit")
    if not path.exists():
        return
    with path.open("rb") as stream:
        for raw in stream:
            if not raw.endswith(b"\n"):
                raise ValueError("incomplete transition audit retained")
            row = json.loads(raw)
            checksum = row.pop("event_sha256")
            request_id = row["request_id"]
            if (checksum != _digest(row) or row["previous_sha256"] != state["previous_sha256"]
                    or row["installation_sha256"] != install["installation_sha256"]
                    or row["rule_sha256"] != install["rule_sha256"]
                    or row["module_sha256"] != install["module_sha256"]
                    or row["fixture_only"] is not install["fixture_only"]
                    or row["event"] != "deferred_before_provider"
                    or row["provider_called"] is not False or row["reason"] != REASON
                    or row["job_id"] != JOB_ID or row["model"] != MODEL
                    or str(uuid.UUID(request_id)) != request_id
                    or request_id in state["deferred_ids"] or not _HEX.fullmatch(row["wire_sha256"])):
                raise ValueError("transition audit identity differs")
            state["deferred_ids"].add(request_id)
            state["audit_size"] += len(raw)
            state["previous_sha256"] = checksum


def _matches(request, system_prompt_sha256):
    if type(request) is not dict or request.get("job_id") != JOB_ID:
        return False
    payload, identity = request.get("payload"), request.get("identity")
    if type(payload) is not dict or type(identity) is not dict:
        return False
    if payload.get("model") != MODEL and identity.get("model") != MODEL:
        return False
    messages = payload.get("messages")
    return (type(messages) is list and bool(messages) and type(messages[0]) is dict
            and messages[0].get("role") == "system"
            and type(messages[0].get("content")) is str
            and sha256(messages[0]["content"].encode()).hexdigest() == system_prompt_sha256)


def _validate_deferred_request(request, relay, fixture_only):
    if (set(request) != {"request_id", "job_id", "identity", "payload", "wire_sha256", "fixture_only"}
            or request["fixture_only"] is not fixture_only):
        raise ValueError("transition request schema/mode mismatch")
    request_id, payload = request["request_id"], request["payload"]
    if type(request_id) is not str or str(uuid.UUID(request_id)) != request_id:
        raise ValueError("canonical request UUID required")
    if (request["identity"] != relay.identities[MODEL]
            or payload.get("model") != MODEL
            or set(payload) != {"model", "messages", "temperature", "thinking", "max_tokens"}
            or type(payload.get("temperature")) not in {int, float} or payload["temperature"] != 0
            or payload.get("thinking") != {"type": "disabled"}
            or type(payload.get("max_tokens")) is not int or not 1 <= payload["max_tokens"] <= 4096):
        raise ValueError("transition payload/provider/sampling mismatch")
    if any(type(m) is not dict or set(m) != {"role", "content"}
           or m["role"] not in {"system", "user", "assistant"}
           or type(m["content"]) is not str for m in payload["messages"]):
        raise ValueError("transition text-message schema differs")
    # This deliberately matches the actual parent provider's serialization.
    wire = json.dumps(payload, allow_nan=False)
    wire_sha256 = sha256(wire.encode()).hexdigest()
    if len(wire) > 96000 or request["wire_sha256"] != wire_sha256:
        raise ValueError("transition actual wire identity or budget differs")
    if request_id in relay._seen:
        raise ValueError("previously admitted UUID must never be reclassified as a non-send")
    return request_id, wire_sha256


def install_quality_transition(relay, audit_directory, *, system_prompt_sha256,
                               job_id=JOB_ID, model=MODEL, fixture_only=False):
    """Install one immutable, exact old-quality deferral on an existing relay.

    No old relay or child journal is rewritten. Returned JSON contains only
    public operational metadata. A deferred UUID is never reused or resent.
    """
    if type(fixture_only) is not bool or getattr(relay, "fixture_only", None) is not fixture_only:
        raise ValueError("explicit matching fixture/hosted relay mode required")
    if not fixture_only:
        _hosted()
    if sha256(_SOURCE_PATH.read_bytes()).hexdigest() != _SOURCE_SHA256:
        raise ValueError("transition code changed after loading")
    if job_id != JOB_ID or model != MODEL or type(system_prompt_sha256) is not str or not _HEX.fullmatch(system_prompt_sha256):
        raise ValueError("only the exact old CBT job/model and a complete system SHA are allowed")
    identity = getattr(relay, "identities", {}).get(MODEL)
    if (type(identity) is not dict or identity.get("provider") != PROVIDER
            or identity.get("model") != MODEL or identity.get("model_family") != "qwen"
            or MODEL not in getattr(relay, "_jobs", {}).get(JOB_ID, {})
            or not callable(getattr(relay, "_request", None))):
        raise ValueError("the already registered fixed old quality relay job is required")
    root = _directory(audit_directory, fixture_only)
    rule = {"job_id": JOB_ID, "model": MODEL,
            "system_prompt_sha256": system_prompt_sha256, "reason": REASON}
    rule_sha256 = _digest(rule)
    with relay._mutex:
        existing = getattr(relay, "_quality_format_transition", None)
        if existing is not None:
            if (existing["root"] != root or existing["installation"]["rule_sha256"] != rule_sha256
                    or existing["installation"]["module_sha256"] != _SOURCE_SHA256
                    or relay._request is not existing["dispatcher"]):
                raise ValueError("installed transition is immutable")
            return dict(existing["installation"])
        path = root / "installation.json"
        if path.exists():
            if path.is_symlink():
                raise ValueError("symlinked transition installation")
            installation = json.loads(path.read_text())
            unsigned = {k:v for k,v in installation.items() if k != "installation_sha256"}
            if (installation.get("installation_sha256") != _digest(unsigned)
                    or installation.get("version") != VERSION or installation.get("rule") != rule
                    or installation.get("rule_sha256") != rule_sha256
                    or installation.get("module_sha256") != _SOURCE_SHA256
                    or installation.get("fixture_only") is not fixture_only):
                raise ValueError("retained transition installation differs")
        else:
            installation = {"version": VERSION, "rule": rule, "rule_sha256": rule_sha256,
                "module_sha256": _SOURCE_SHA256, "module_path": str(_SOURCE_PATH),
                "fixture_only": fixture_only, "installation_id": str(uuid.uuid4()),
                "installed_at_utc": datetime.now(timezone.utc).isoformat(),
                "old_protocols_and_journals_changed": False, "automatic_retry": False,
                "credential_access": False}
            installation["installation_sha256"] = _digest(installation)
            _write_installation(path, installation)
        original = relay._request
        state = {"root": root, "installation": installation,
            "audit_path": root / "deferred.jsonl", "audit_size": 0,
            "previous_sha256": installation["installation_sha256"],
            "deferred_ids": set(), "failed_closed": False, "original": original}
        _load_audit(state)
        if state["deferred_ids"] & relay._seen:
            raise ValueError("retained deferred UUID was also admitted by the relay")

        def dispatch(request):
            if not _matches(request, system_prompt_sha256):
                return original(request)
            with relay._mutex:
                if state["failed_closed"]:
                    raise RuntimeError("transition audit failed; matching quality sends remain blocked")
                request_id, wire_hash = _validate_deferred_request(request, relay, fixture_only)
                if request_id in state["deferred_ids"]:
                    raise ValueError("deferred UUID cannot be replayed")
                event = {"event": "deferred_before_provider", "request_id": request_id,
                    "job_id": JOB_ID, "model": MODEL, "wire_sha256": wire_hash,
                    "rule_sha256": rule_sha256, "module_sha256": _SOURCE_SHA256,
                    "installation_sha256": installation["installation_sha256"],
                    "fixture_only": fixture_only, "provider_called": False, "reason": REASON,
                    "at_utc": datetime.now(timezone.utc).isoformat()}
                try:
                    _append_audit(state, event)
                except Exception:
                    state["failed_closed"] = True
                    raise
                state["deferred_ids"].add(request_id)
                return {"status": "deferred", "request_id": request_id,
                        "reason": REASON, "provider_called": False}

        state["dispatcher"] = dispatch
        relay._quality_format_transition = state
        relay._request = dispatch
        return dict(installation)
