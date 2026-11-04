"""Private, offline capture for a *planned* four-visit nonclinical study.

This module does not run a model, recruit people, verify ethics approval, or make
the demonstration app suitable for treatment.  Planned allocation slots are not
participants.  ``fixture`` runs remain explicitly synthetic at every entry point.
Free text is permitted only in declared fictional task-response artifacts; the
append-only event chain stores their hashes, not their contents. Consent revocation
remains available after collection is sealed and prevents new exports of erased
responses. Previously issued copies and provider logs require separate handling.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import shutil
import tempfile
from typing import Any


SCHEMA_VERSION = "dialogue-benefit-capture-v1"
ARMS = ("short_term", "raw_history_rag", "managed_memory_rag")
VISIT_TASKS = {
    "1": [],
    "2": [f"T{i:02d}" for i in range(1, 5)],
    "3": [f"T{i:02d}" for i in range(5, 9)],
    "4": [f"T{i:02d}" for i in range(9, 13)],
}
ZERO_HASH = "0" * 64
ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}\Z")
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
BASE_KEYS = {"event_id", "participant_id", "visit", "task_id", "timestamp", "event_type", "payload"}
EVENT_TYPES = {
    "enroll", "withdraw", "visit_start", "visit_end", "pause", "resume",
    "task_start", "task_end", "task_response", "task_assessment", "memory_check",
    "user_edit", "explicit_confirm", "cost", "failure",
}


class CaptureError(ValueError):
    """Invalid, contradictory, or incompatible study input; no model was called."""


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise CaptureError("timestamp must be an explicit timezone-aware ISO string")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CaptureError("invalid timestamp") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise CaptureError("timestamp must include timezone")
    return result.astimezone(timezone.utc)


def _identifier(value: Any, name: str) -> None:
    if not isinstance(value, str) or not ID_RE.fullmatch(value):
        raise CaptureError(f"{name} must be a pseudonymous code, not free text")


def _hash(value: Any, name: str) -> None:
    if not isinstance(value, str) or not HASH_RE.fullmatch(value):
        raise CaptureError(f"{name} must be a SHA-256 hex string")


def _bool(value: Any, name: str) -> None:
    if type(value) is not bool:
        raise CaptureError(f"{name} must be an explicit boolean")


def _keys(value: Any, required: set[str], optional: set[str] | None = None) -> None:
    if not isinstance(value, dict) or set(value) - required - (optional or set()) or required - set(value):
        raise CaptureError("payload fields do not match this frozen event schema")


def _atomic_write(path: Path, data: bytes) -> None:
    """Replace one complete file; an interrupted append never creates a partial line."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, name = tempfile.mkstemp(prefix="capture-write-", dir=path.parent)
    temporary = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _lock(root: Path, *, exclusive: bool):
    # Prepared once and never unlinked; read-only analysis creates no input files.
    path = root / ".capture.lock"
    if not path.is_file() or path.is_symlink():
        raise CaptureError("not a complete prepared study directory")
    with path.open("rb") as handle:
        try:
            fcntl.flock(handle.fileno(), (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CaptureError("study directory is in use; no capture was attempted") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_json(path: Path) -> Any:
    try:
        if path.is_symlink():
            raise CaptureError("study files must not be symbolic links")
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CaptureError(f"cannot read a complete study file: {path.name}") from exc


def _read_chain(path: Path) -> list[dict]:
    try:
        if path.is_symlink():
            raise CaptureError("journal must not be a symbolic link")
        content = path.read_bytes()
    except OSError as exc:
        raise CaptureError("missing study journal") from exc
    if content and not content.endswith(b"\n"):
        raise CaptureError("partial journal line; preserve the file and investigate")
    previous = ZERO_HASH
    result = []
    seen = set()
    for index, line in enumerate(content.splitlines(), 1):
        try:
            row = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise CaptureError("malformed journal line") from exc
        _keys(row, {"sequence", "previous_sha256", "event_sha256", "event"})
        if type(row["sequence"]) is not int or row["sequence"] != index or row["previous_sha256"] != previous:
            raise CaptureError("journal sequence/hash chain mismatch")
        expected = canonical_hash({key: value for key, value in row.items() if key != "event_sha256"})
        if row["event_sha256"] != expected:
            raise CaptureError("journal event content/hash mismatch")
        event = row["event"]
        _keys(event, BASE_KEYS)
        if event["event_id"] in seen:
            raise CaptureError("duplicate journal event ID")
        seen.add(event["event_id"])
        previous = expected
        result.append(row)
    return result


def _head(rows: list[dict]) -> str:
    return rows[-1]["event_sha256"] if rows else ZERO_HASH


def _snapshot_identity(manifest: dict, rows: list[dict]) -> str:
    return canonical_hash({"manifest_sha256": manifest["manifest_sha256"],
                           "event_count": len(rows), "event_head_sha256": _head(rows)})


def _verify_launch_record(protocol: dict, record: dict) -> None:
    required = {"ethics_clearance_id", "consent_document_sha256", "task_ledger_sha256", "code_commit",
                "model_revision", "prompt_sha256", "analysis_sha256", "approved_data_policy_sha256",
                "max_total_api_calls", "max_attempts_per_request", "max_currency_cost", "currency",
                "language", "responsible_investigator_id", "completed_gates", "approval_attestation"}
    _keys(record, required)
    for name in ("consent_document_sha256", "task_ledger_sha256", "prompt_sha256", "analysis_sha256",
                 "approved_data_policy_sha256"):
        _hash(record[name], name)
    for name in ("ethics_clearance_id", "model_revision", "language", "responsible_investigator_id"):
        _identifier(record[name], name)
    if not isinstance(record["code_commit"], str) or not re.fullmatch(r"[0-9a-f]{40}", record["code_commit"]):
        raise CaptureError("launch requires a frozen source commit")
    for name in ("max_total_api_calls", "max_attempts_per_request"):
        if type(record[name]) is not int or record[name] < 1:
            raise CaptureError("launch requires finite positive call/retry caps")
    if (type(record["max_currency_cost"]) not in (int, float) or
            not math.isfinite(record["max_currency_cost"]) or record["max_currency_cost"] <= 0):
        raise CaptureError("launch requires a finite positive currency cap")
    if not isinstance(record["currency"], str) or not re.fullmatch("[A-Z]{3}", record["currency"]):
        raise CaptureError("launch requires an explicit currency")
    if (not isinstance(record["completed_gates"], list) or
            set(record["completed_gates"]) != set(protocol.get("launch_gates", [])) or
            not protocol.get("launch_gates")):
        raise CaptureError("all preregistered launch gates require explicit independent clearance")
    if record["approval_attestation"] is not True:
        raise CaptureError("launch approval attestation is absent; software cannot grant approval")


def _verify_task_ledger(ledger: dict) -> None:
    if not isinstance(ledger, dict) or ledger.get("schema_version") != "dialogue-benefit-task-ledger-v1":
        raise CaptureError("unsupported task ledger")
    source_rows = ledger.get("sources")
    if not isinstance(source_rows, list) or not source_rows:
        raise CaptureError("task ledger requires the original full fictional source ledger")
    sources = {}
    for source in source_rows:
        if not isinstance(source, dict) or source.get("variant") not in {"A", "B", "C"}:
            raise CaptureError("invalid source variant")
        _identifier(source.get("source_id"), "source_id")
        if source.get("role") not in {"user", "assistant", "system_observation"}:
            raise CaptureError("invalid source role")
        _time(source.get("date"))
        text = source.get("source_text")
        if not isinstance(text, str) or not text.strip():
            raise CaptureError("source ledger requires exact original source text")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != source.get("source_text_sha256"):
            raise CaptureError("source ledger text/hash mismatch")
        key = (source["variant"], source["source_id"])
        if key in sources:
            raise CaptureError("duplicate source ID/version within a task variant")
        sources[key] = source
    rows = ledger.get("tasks")
    if not isinstance(rows, list) or len(rows) != 36:
        raise CaptureError("task ledger must freeze all 12 tasks in each of three variants")
    expected = {(f"T{i:02d}", variant) for i in range(1, 13) for variant in "ABC"}
    found = set()
    for row in rows:
        if not isinstance(row, dict):
            raise CaptureError("invalid task ledger row")
        pair = (row.get("task_id"), row.get("variant"))
        if pair not in expected or pair in found:
            raise CaptureError("task ledger has missing, duplicate or out-of-scope tasks")
        found.add(pair)
        if (not isinstance(row.get("completion_criterion"), str) or
                not row["completion_criterion"].strip() or not isinstance(row.get("source_reference"), list) or
                not row["source_reference"]):
            raise CaptureError("task criteria and actual source references must precede response generation")
        for source in row["source_reference"]:
            if not isinstance(source, dict) or "source_id" not in source or "source_text_sha256" not in source:
                raise CaptureError("ledger references require source IDs and exact source hashes")
            _identifier(source["source_id"], "source_id")
            _hash(source["source_text_sha256"], "source_text_sha256")
            original = sources.get((row["variant"], source["source_id"]))
            if original is None or any(source.get(field) != original[field]
                                       for field in ("source_text_sha256", "role", "date")):
                raise CaptureError("task reference does not match the independently reviewable original source")
            start, end = source.get("start"), source.get("end")
            if (type(start) is not int or type(end) is not int or start < 0 or end <= start or
                    end > len(original["source_text"]) or source.get("quote") != original["source_text"][start:end]):
                raise CaptureError("task reference does not match its exact Unicode source span")
            if source.get("permitted_absence_reason") not in {None, "deleted", "dont_save", "expired"}:
                raise CaptureError("source absence must be preregistered as deletion, optout or expiry")
        if not isinstance(row.get("task_query"), str) or not row["task_query"].strip():
            raise CaptureError("task query must be fixed separately from scoring criteria")
        visible = row.get("visible_source_ids")
        if not isinstance(visible, list) or len(set(visible)) != len(visible):
            raise CaptureError("task requires a fixed visible source inventory")
        if any((row["variant"], source_id) not in sources for source_id in visible):
            raise CaptureError("task visible history includes an unknown source")
        if row.get("criterion_sha256") != canonical_hash({key: value for key, value in row.items() if key != "criterion_sha256"}):
            raise CaptureError("task criterion identity mismatch")
    if found != expected:
        raise CaptureError("task ledger does not cover all fixed variants")


def prepare_study(root: Path, protocol_path: Path, *, mode: str = "planned_human", seed: int | None = None,
                  launch_record_path: Path | None = None, task_ledger_path: Path | None = None) -> dict:
    """Freeze a future allocation; this function enrolls exactly zero people."""
    root, protocol_path = Path(root).absolute(), Path(protocol_path)
    if mode not in {"planned_human", "fixture"}:
        raise CaptureError("mode must distinguish planned_human from fixture")
    if root.exists() or root.is_symlink():
        raise CaptureError("use a new study directory; existing studies are never overwritten")
    protocol = _read_json(protocol_path)
    allocation_spec = protocol.get("allocation", {})
    visits = {str(row["visit"]): row["task_ids"] for row in protocol.get("visits", [])}
    endpoints = {row.get("id"): row for row in protocol.get("endpoints", {}).get("co_primary", [])}
    if (protocol.get("protocol_id") != "dialogue-benefit-four-visit-v1" or
            allocation_spec.get("planned_slots") != 60 or allocation_spec.get("planned_per_arm") != 20 or
            allocation_spec.get("arms") != list(ARMS) or visits != VISIT_TASKS or
            allocation_spec.get("permuted_blocks") is not True or allocation_spec.get("same_arm_all_visits") is not True or
            endpoints.get("task_success_fraction", {}).get("participant_denominator") != 12 or
            endpoints.get("restricted_elapsed_seconds", {}).get("visit_cap_seconds") != 900):
        raise CaptureError("protocol is incompatible with the frozen four-visit capture schema")
    launch = _read_json(Path(launch_record_path)) if launch_record_path else None
    ledger = _read_json(Path(task_ledger_path)) if task_ledger_path else None
    if ledger is not None:
        _verify_task_ledger(ledger)
    if launch is not None:
        _verify_launch_record(protocol, launch)
        if ledger is None or launch["task_ledger_sha256"] != canonical_hash(ledger):
            raise CaptureError("human launch record must bind the actual frozen task ledger")
    if seed is None:
        seed = secrets.randbits(128)
    if type(seed) is not int or seed < 0:
        raise CaptureError("randomization seed must be a nonnegative integer")
    rng = random.Random(seed)
    schedule = []
    # Ten concealed balanced blocks; each participant remains in one arm.
    for _ in range(10):
        block = list(ARMS) * 2
        rng.shuffle(block)
        schedule.extend(block)
    slots = [f"P{number:03d}" for number in range(1, 61)]
    assignments = dict(zip(slots, schedule))
    variants = {}
    for arm in ARMS:
        labels = ["A"] * 7 + ["B"] * 7 + ["C"] * 6
        rng.shuffle(labels)
        variants.update(zip([slot for slot in slots if assignments[slot] == arm], labels))
    allocation = {"assignments": assignments, "task_variants": variants, "randomization_seed": seed,
                  "seed_commitment_sha256": canonical_hash({"seed": seed}), "permuted_block_size": 6}
    analysis_path = Path(__file__).with_name("study_analysis.py")
    analysis_bytes = analysis_path.read_bytes() if analysis_path.is_file() else None
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_id": root.name,
        "mode": mode,
        "created_at": _now(),
        "protocol_sha256": canonical_hash(protocol),
        "protocol_file_sha256": hashlib.sha256(_json_bytes(protocol) + b"\n").hexdigest(),
        "source_code_sha256": _file_hash(Path(__file__)),
        "analysis_code_sha256": hashlib.sha256(analysis_bytes).hexdigest() if analysis_bytes is not None else None,
        "allocation_sha256": canonical_hash(allocation),
        "allocation_seed_commitment_sha256": allocation["seed_commitment_sha256"],
        "launch_record_sha256": canonical_hash(launch) if launch is not None else None,
        "task_ledger_sha256": canonical_hash(ledger) if ledger is not None else None,
        "planned_slots": 60, "planned_per_arm": 20, "arms": list(ARMS), "slots": slots,
        "visit_tasks": VISIT_TASKS, "visit_elapsed_cap_seconds": 900,
        "planned_slots_are_enrolled_participants": False,
        "participant_text_public_export_permitted": False,
        "privacy_boundary": "pseudonymous_metadata_retained_for_audit;revocation_excludes_analysis_and_erases_private_response_content",
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="study-prepare-", dir=root.parent))
    try:
        (temporary / "restricted/responses").mkdir(parents=True, mode=0o700)
        (temporary / "frozen").mkdir(mode=0o700)
        _atomic_write(temporary / "restricted/allocation.json", _json_bytes(allocation) + b"\n")
        _atomic_write(temporary / "frozen/protocol.json", _json_bytes(protocol) + b"\n")
        _atomic_write(temporary / "frozen/study_capture.py", Path(__file__).read_bytes())
        if analysis_bytes is not None:
            _atomic_write(temporary / "frozen/study_analysis.py", analysis_bytes)
        if ledger is not None:
            _atomic_write(temporary / "frozen/task_ledger.json", _json_bytes(ledger) + b"\n")
        if launch is not None:
            _atomic_write(temporary / "restricted/launch_record.json", _json_bytes(launch) + b"\n")
        _atomic_write(temporary / "events.jsonl", b"")
        _atomic_write(temporary / "revocations.jsonl", b"")
        _atomic_write(temporary / ".capture.lock", b"")
        _atomic_write(temporary / "manifest.json", _json_bytes(manifest) + b"\n")
        if root.exists():
            raise CaptureError("concurrent study preparation; use another directory")
        os.rename(temporary, root)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return study_status(root)


def _read_unlocked(root: Path, *, load_responses: bool = True) -> dict:
    manifest = _read_json(root / "manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise CaptureError("unsupported study schema")
    if canonical_hash({key: value for key, value in manifest.items() if key != "manifest_sha256"}) != manifest.get("manifest_sha256"):
        raise CaptureError("manifest identity mismatch")
    protocol = _read_json(root / "frozen/protocol.json")
    if (canonical_hash(protocol) != manifest["protocol_sha256"] or
            _file_hash(root / "frozen/protocol.json") != manifest["protocol_file_sha256"] or
            _file_hash(root / "frozen/study_capture.py") != manifest["source_code_sha256"]):
        raise CaptureError("frozen protocol/source identity mismatch")
    if manifest.get("analysis_code_sha256") is not None:
        if _file_hash(root / "frozen/study_analysis.py") != manifest["analysis_code_sha256"]:
            raise CaptureError("frozen analysis code identity mismatch")
    allocation = _read_json(root / "restricted/allocation.json")
    if canonical_hash(allocation) != manifest["allocation_sha256"]:
        raise CaptureError("concealed allocation identity mismatch")
    if set(allocation.get("assignments", {})) != set(manifest["slots"]):
        raise CaptureError("allocation does not match planned slots")
    if any(list(allocation["assignments"].values()).count(arm) != 20 for arm in ARMS):
        raise CaptureError("allocation no longer has 20 planned slots per arm")
    launch = None
    if manifest["launch_record_sha256"] is not None:
        launch = _read_json(root / "restricted/launch_record.json")
        if canonical_hash(launch) != manifest["launch_record_sha256"]:
            raise CaptureError("frozen launch record changed")
        _verify_launch_record(protocol, launch)
    ledger = None
    if manifest.get("task_ledger_sha256") is not None:
        ledger = _read_json(root / "frozen/task_ledger.json")
        _verify_task_ledger(ledger)
        if canonical_hash(ledger) != manifest["task_ledger_sha256"]:
            raise CaptureError("frozen task ledger changed")
    rows = _read_chain(root / "events.jsonl")
    revocations = _read_chain(root / "revocations.jsonl")
    withdrawals = {}
    for row in revocations:
        event = row["event"]
        if event["event_type"] != "withdraw" or event["participant_id"] not in manifest["slots"]:
            raise CaptureError("invalid consent revocation journal")
        _keys(event["payload"], {"retain_collected_data"})
        _bool(event["payload"]["retain_collected_data"], "retain_collected_data")
        old = withdrawals.get(event["participant_id"])
        if old and not old["retain_collected_data"] and event["payload"]["retain_collected_data"]:
            raise CaptureError("erased study data must never be reactivated")
        withdrawals[event["participant_id"]] = {
            **event["payload"], "event_sha256": row["event_sha256"], "timestamp": event["timestamp"]}
    if len({row["event"]["event_id"] for row in rows + revocations}) != len(rows) + len(revocations):
        raise CaptureError("event ID repeated across consent and outcome journals")
    sealed = _read_json(root / "seal.json") if (root / "seal.json").exists() else None
    if sealed is not None:
        if (sealed.get("snapshot_sha256") != _snapshot_identity(manifest, rows) or
                sealed.get("event_count") != len(rows) or sealed.get("event_head_sha256") != _head(rows) or
                sealed.get("manifest_sha256") != manifest["manifest_sha256"]):
            raise CaptureError("sealed collection was changed")
    for row in rows:
        event = row["event"]
        if event["participant_id"] not in manifest["slots"] or event["event_type"] not in EVENT_TYPES - {"withdraw"}:
            raise CaptureError("journal is not a valid study collection")
        if event["event_type"] == "task_response":
            metadata = event["payload"]
            _keys(metadata, {"artifact_path", "artifact_sha256", "task_key_sha256", "source_history_sha256"})
            erased = withdrawals.get(event["participant_id"], {}).get("retain_collected_data") is False
            row["erased"] = erased
            if load_responses and not erased:
                path = _response_path(root, event)
                if _file_hash(path) != metadata["artifact_sha256"]:
                    raise CaptureError("private task response content changed or is unavailable")
                content = _read_json(path)
                if (canonical_hash(content.get("source_history")) != metadata["source_history_sha256"] or
                        content.get("task_key_sha256") != metadata["task_key_sha256"]):
                    raise CaptureError("task response source/key binding changed")
                row["response_content"] = content
            else:
                row["response_content"] = None
    return {"manifest": manifest, "allocation": allocation, "events": rows, "revocations": revocations,
            "withdrawals": withdrawals, "sealed": sealed, "protocol": protocol, "task_ledger": ledger,
            "launch_record": launch,
            "snapshot_sha256": _snapshot_identity(manifest, rows),
            "revocation_head_sha256": _head(revocations),
            "consent_snapshot_sha256": canonical_hash({"collection": _snapshot_identity(manifest, rows),
                                                        "revocation_head": _head(revocations)})}


@contextmanager
def locked_snapshot(root: Path):
    """Hold a shared consent/capture lock through a private export's publication."""
    root = Path(root)
    with _lock(root, exclusive=False):
        yield _read_unlocked(root)


def read_snapshot(root: Path) -> dict:
    """Read and verify current consent as well as a possibly sealed collection.

    Consumers must use ``response_content`` for private task text and must bind
    exports/scores to ``consent_snapshot_sha256``. A collection seal alone is not
    permission to reuse an erased participant's text.
    """
    with locked_snapshot(root) as snapshot:
        return snapshot


def _response_path(root: Path, event: dict) -> Path:
    expected = f"restricted/responses/{event['participant_id']}/{event['event_id']}.json"
    if event["payload"].get("artifact_path") != expected:
        raise CaptureError("response artifact is outside its participant scope")
    path = root / expected
    if any(parent.is_symlink() for parent in (root, root / "restricted", root / "restricted/responses", path.parent, path)):
        raise CaptureError("response artifacts must not follow symbolic links")
    return path


def _events_for(snapshot: dict, participant_id: str) -> list[dict]:
    return [row for row in snapshot["events"] if row["event"]["participant_id"] == participant_id]


def _validate_response(payload: dict, mode: str) -> None:
    _keys(payload, {"source_history", "current_query", "response", "task_key_sha256",
                    "fictional_content_attestation", "response_origin"})
    if payload["fictional_content_attestation"] is not True:
        raise CaptureError("real personal/clinical free text is not supported by this capture instrument")
    if payload["response_origin"] not in {"live_model", "offline_fixture", "human_observed_ui"}:
        raise CaptureError("response origin must be declared")
    if mode == "fixture" and payload["response_origin"] != "offline_fixture":
        raise CaptureError("fixture mode cannot assert a live model or a real observed response")
    if mode == "planned_human" and payload["response_origin"] == "offline_fixture":
        raise CaptureError("offline test doubles cannot become human study outcomes")
    _hash(payload["task_key_sha256"], "task_key_sha256")
    for field in ("current_query", "response"):
        if not isinstance(payload[field], str) or not payload[field].strip() or len(payload[field]) > 30000:
            raise CaptureError("task text must be nonempty and bounded")
    history = payload["source_history"]
    if not isinstance(history, list) or not history or len(history) > 500:
        raise CaptureError("bounded fictional source history is required")
    ids = set()
    for source in history:
        _keys(source, {"source_id", "role", "date", "text"})
        _identifier(source["source_id"], "source_id")
        if source["source_id"] in ids:
            raise CaptureError("source IDs must be unique in this exact visible history")
        ids.add(source["source_id"])
        if source["role"] not in {"user", "assistant", "system_observation"}:
            raise CaptureError("source role is invalid")
        _time(source["date"])
        if not isinstance(source["text"], str) or not source["text"].strip() or len(source["text"]) > 30000:
            raise CaptureError("source text must be nonempty and bounded")


def _validate_event(snapshot: dict, event: dict, *, response_content: dict | None = None) -> None:
    _keys(event, BASE_KEYS)
    _identifier(event["event_id"], "event_id")
    if event["participant_id"] not in snapshot["manifest"]["slots"]:
        raise CaptureError("participant is not a planned pseudonymous slot")
    if type(event["visit"]) is not int or event["visit"] not in range(5):
        raise CaptureError("visit must be 0 (study) or 1..4")
    if event["event_type"] not in EVENT_TYPES:
        raise CaptureError("unknown event type")
    moment = _time(event["timestamp"])
    participant_id, kind, payload = event["participant_id"], event["event_type"], event["payload"]
    rows = _events_for(snapshot, participant_id)
    events = [row["event"] for row in rows]
    all_participant_events = events + [row["event"] for row in snapshot["revocations"]
                                      if row["event"]["participant_id"] == participant_id]
    if all_participant_events and moment < max(_time(old["timestamp"]) for old in all_participant_events):
        raise CaptureError("participant timestamps cannot move backwards")
    enrolled = any(old["event_type"] == "enroll" for old in events)
    if kind == "enroll":
        if event["visit"] != 0 or event["task_id"] != "study" or enrolled:
            raise CaptureError("enrollment is recorded once at study level")
        already_enrolled = {row["event"]["participant_id"] for row in snapshot["events"]
                            if row["event"]["event_type"] == "enroll"}
        next_slot = next((slot for slot in snapshot["manifest"]["slots"] if slot not in already_enrolled), None)
        if participant_id != next_slot:
            raise CaptureError("enroll the next concealed slot; selective arm allocation is refused")
        _keys(payload, {"consent_recorded", "nonclinical_fictional_tasks_only", "adult_eligible_attestation",
                        "ethics_clearance_id", "operator_id", "real_participant_attestation"})
        for name in ("consent_recorded", "nonclinical_fictional_tasks_only", "adult_eligible_attestation"):
            if payload[name] is not True:
                raise CaptureError("consent, adult eligibility and fictional task scope are required")
        _identifier(payload["operator_id"], "operator_id")
        _identifier(payload["ethics_clearance_id"], "ethics_clearance_id")
        _bool(payload["real_participant_attestation"], "real_participant_attestation")
        if payload["real_participant_attestation"] != (snapshot["manifest"]["mode"] == "planned_human"):
            raise CaptureError("fixture observations cannot be represented as real participants")
        if snapshot["manifest"]["mode"] == "planned_human":
            if snapshot["manifest"]["launch_record_sha256"] is None:
                raise CaptureError("human enrollment is blocked: frozen launch clearance and identities are absent")
            if snapshot["manifest"].get("task_ledger_sha256") is None:
                raise CaptureError("human enrollment requires actual frozen task criteria and source references")
            if snapshot["manifest"].get("analysis_code_sha256") is None:
                raise CaptureError("human enrollment requires a frozen analysis executable")
            source_review = snapshot["task_ledger"].get("independent_source_review", {})
            reviewers = source_review.get("reviewer_ids", [])
            if (source_review.get("status") != "independently_validated" or not isinstance(reviewers, list) or
                    len(reviewers) < 2 or len(set(reviewers)) != len(reviewers)):
                raise CaptureError("draft task references require at least two independent human source validators before enrollment")
            for reviewer in reviewers:
                _identifier(reviewer, "source reviewer ID")
            _hash(source_review.get("reviewer_verification_sha256"), "source reviewer verification record")
            if source_review.get("human_and_independence_attestation") is not True:
                raise CaptureError("source review requires a documented independent human review attestation")
            if _time(source_review.get("completed_at")) > moment:
                raise CaptureError("independent source review must be complete before enrollment")
        return
    if not enrolled:
        raise CaptureError("no observation may precede documented enrollment")
    if kind == "withdraw":
        if event["visit"] != 0 or event["task_id"] != "study":
            raise CaptureError("withdrawal is a study-level consent event")
        _keys(payload, {"retain_collected_data"})
        _bool(payload["retain_collected_data"], "retain_collected_data")
        if snapshot["withdrawals"].get(participant_id, {}).get("retain_collected_data") is False:
            raise CaptureError("erased data cannot be reactivated or recollected in this study")
        return
    if participant_id in snapshot["withdrawals"]:
        raise CaptureError("participant withdrew; new collection is prohibited")
    if event["visit"] == 0:
        raise CaptureError("observations require a visit")
    task_ids = snapshot["manifest"]["visit_tasks"][str(event["visit"])]
    if event["task_id"] not in task_ids + ["visit"]:
        raise CaptureError("task is outside this frozen visit")
    visit_events = [old for old in events if old["visit"] == event["visit"]]
    visit_started = any(old["event_type"] == "visit_start" for old in visit_events)
    visit_ended = any(old["event_type"] == "visit_end" for old in visit_events)
    if kind == "visit_start":
        if event["task_id"] != "visit" or visit_started:
            raise CaptureError("visit starts exactly once")
        if any(old["visit"] > event["visit"] for old in events):
            raise CaptureError("visits cannot be backfilled after a later visit")
        _keys(payload, {"instrument_sha256", "operator_id"})
        _hash(payload["instrument_sha256"], "instrument_sha256")
        _identifier(payload["operator_id"], "operator_id")
        return
    if not visit_started:
        raise CaptureError("visit_start must precede visit observations")
    if kind not in {"task_assessment", "cost", "failure"} and visit_ended:
        raise CaptureError("visit is already closed")
    if kind == "visit_end":
        if event["task_id"] != "visit":
            raise CaptureError("visit_end is a visit-level event")
        _keys(payload, {"status", "burden_observation_complete", "attempt_log_complete"})
        if payload["status"] not in {"completed", "timeout", "incomplete", "assisted"}:
            raise CaptureError("unknown visit completion status")
        _bool(payload["burden_observation_complete"], "burden_observation_complete")
        _bool(payload["attempt_log_complete"], "attempt_log_complete")
        pauses = [old["event_type"] for old in visit_events if old["event_type"] in {"pause", "resume"}]
        if pauses and pauses[-1] == "pause" and payload["status"] == "completed":
            raise CaptureError("an open break makes elapsed time unresolved; resume or mark an incomplete visit")
        if payload["status"] == "completed":
            ended = {old["task_id"] for old in visit_events if old["event_type"] == "task_end"}
            if ended != set(task_ids):
                raise CaptureError("completed visit requires all planned task_end observations")
        return
    if kind in {"pause", "resume"}:
        if event["task_id"] != "visit":
            raise CaptureError("documented breaks are visit-level events")
        _keys(payload, {"reason_code"})
        if payload["reason_code"] not in {"documented_break", "scheduled_break"}:
            raise CaptureError("model waiting or technical delay cannot be excluded as a break")
        pauses = [old["event_type"] for old in visit_events if old["event_type"] in {"pause", "resume"}]
        if (kind == "pause" and pauses and pauses[-1] == "pause") or (kind == "resume" and (not pauses or pauses[-1] != "pause")):
            raise CaptureError("break timestamps must alternate pause then resume")
        return
    if kind in {"task_start", "task_end", "task_response", "task_assessment"}:
        if event["task_id"] == "visit":
            raise CaptureError("task observations need a planned task ID")
        same_task = [old for old in visit_events if old["task_id"] == event["task_id"]]
        if kind != "task_assessment" and any(old["event_type"] == "task_end" for old in same_task):
            raise CaptureError("task is already closed")
        if kind == "task_start":
            _keys(payload, set())
            if any(old["event_type"] == "task_start" for old in same_task):
                raise CaptureError("task starts exactly once")
            return
        if not any(old["event_type"] == "task_start" for old in same_task):
            raise CaptureError("task_start must precede task observations")
        responses = [row for row in rows if row["event"]["visit"] == event["visit"]
                     and row["event"]["task_id"] == event["task_id"] and row["event"]["event_type"] == "task_response"]
        if kind == "task_response":
            if responses:
                raise CaptureError("one frozen final response per planned task; attempts belong in cost events")
            _validate_response(response_content if response_content is not None else payload, snapshot["manifest"]["mode"])
            if snapshot["task_ledger"] is not None:
                variant = snapshot["allocation"]["task_variants"][participant_id]
                criterion = next(row for row in snapshot["task_ledger"]["tasks"]
                                 if row["task_id"] == event["task_id"] and row["variant"] == variant)
                content = response_content if response_content is not None else payload
                if content["task_key_sha256"] != criterion["criterion_sha256"]:
                    raise CaptureError("response task key is not its predeclared task/variant criterion")
                if content["current_query"] != criterion["task_query"]:
                    raise CaptureError("task query differs from the fixed arm-independent task")
                captured_sources = {source["source_id"]: source for source in content["source_history"]}
                if set(captured_sources) != set(criterion["visible_source_ids"]):
                    raise CaptureError("captured source history differs from the predeclared eligible task history")
                originals = {source["source_id"]: source for source in snapshot["task_ledger"]["sources"]
                             if source["variant"] == variant}
                for source_id, captured in captured_sources.items():
                    original = originals[source_id]
                    if (captured["text"] != original["source_text"] or captured["role"] != original["role"] or
                            captured["date"] != original["date"]):
                        raise CaptureError("captured source body/role/date differs from its frozen source version")
                for reference in criterion["source_reference"]:
                    if reference["source_id"] not in captured_sources and not reference.get("permitted_absence_reason"):
                        raise CaptureError("required source reference is absent without a frozen exclusion reason")
            return
        if kind == "task_end":
            _keys(payload, {"status", "burden_observation_complete", "assistance_received"})
            if payload["status"] not in {"completed", "technical_failure", "assisted", "incomplete"}:
                raise CaptureError("invalid task completion status")
            _bool(payload["burden_observation_complete"], "burden_observation_complete")
            _bool(payload["assistance_received"], "assistance_received")
            if payload["status"] == "completed" and not responses:
                raise CaptureError("completed task requires the actual final response artifact")
            if payload["status"] == "assisted" and not payload["assistance_received"]:
                raise CaptureError("assisted task must retain the assistance flag")
            return
        _keys(payload, {"status", "correct", "source_verified", "assessor_id", "independent_attestation",
                        "blinding_attestation", "human_not_model_attestation", "task_key_sha256",
                        "source_evidence_sha256", "task_response_event_sha256"}, {"assessment_role"})
        if not responses or payload["task_response_event_sha256"] != responses[0]["event_sha256"]:
            raise CaptureError("assessment is not bound to the exact frozen task response")
        expected = responses[0]["event"]["payload"]
        if (payload["task_key_sha256"] != expected["task_key_sha256"] or
                payload["source_evidence_sha256"] != expected["source_history_sha256"]):
            raise CaptureError("assessment source/task-key identity mismatch")
        if payload["status"] not in {"rated", "adjudicated", "unresolved"}:
            raise CaptureError("unknown evidence assessment status")
        if payload["correct"] is not None and type(payload["correct"]) is not bool:
            raise CaptureError("correct must be true, false or unresolved null")
        if payload["status"] == "unresolved" and payload["correct"] is not None:
            raise CaptureError("unresolved evidence cannot be assigned a correctness label")
        for name in ("source_verified", "independent_attestation", "blinding_attestation", "human_not_model_attestation"):
            _bool(payload[name], name)
        if payload["correct"] is not None and not payload["source_verified"]:
            raise CaptureError("a correctness label requires verified sources, not substring overlap")
        _identifier(payload["assessor_id"], "assessor_id")
        if payload.get("assessment_role", "primary_rater") not in {"primary_rater", "adjudicator"}:
            raise CaptureError("unknown assessor role")
        if any(old["event_type"] == "task_assessment" and old["payload"]["assessor_id"] == payload["assessor_id"]
               for old in same_task):
            raise CaptureError("one immutable label per assessor/task; adjudication needs a separate assessor")
        return
    if kind == "memory_check":
        _keys(payload, {"check_id", "memory_subject_id", "source_version_sha256", "current_issue_id",
                        "evidence_state_sha256", "repeated", "previous_check_id"})
        for name in ("check_id", "memory_subject_id", "current_issue_id"):
            _identifier(payload[name], name)
        _hash(payload["source_version_sha256"], "source_version_sha256")
        _hash(payload["evidence_state_sha256"], "evidence_state_sha256")
        _bool(payload["repeated"], "repeated")
        previous_checks = [old for old in events if old["event_type"] == "memory_check"]
        if any(old["payload"]["check_id"] == payload["check_id"] for old in previous_checks):
            raise CaptureError("check ID is already recorded")
        if payload["repeated"]:
            previous = next((old for old in previous_checks if old["payload"]["check_id"] == payload["previous_check_id"]), None)
            if (previous is None or previous["payload"]["memory_subject_id"] != payload["memory_subject_id"] or
                    previous["payload"]["source_version_sha256"] != payload["source_version_sha256"] or
                    previous["payload"]["current_issue_id"] != payload["current_issue_id"] or
                    previous["payload"]["evidence_state_sha256"] != payload["evidence_state_sha256"] or
                    (moment - _time(previous["timestamp"])).total_seconds() > 7 * 86400):
                raise CaptureError("repeat check must bind an earlier unchanged source version within the frozen cooldown")
        elif payload["previous_check_id"] is not None:
            raise CaptureError("a nonrepeat check must not claim a previous-check link")
        return
    if kind == "user_edit":
        _keys(payload, {"target_id", "action"})
        _identifier(payload["target_id"], "target_id")
        if payload["action"] not in {"correction", "deletion", "expiry", "dont_save", "draft", "review"}:
            raise CaptureError("unknown user edit action")
        return
    if kind == "explicit_confirm":
        _keys(payload, {"target_id"})
        _identifier(payload["target_id"], "target_id")
        return
    if kind == "cost":
        _keys(payload, {"attempt_id", "status", "model", "prompt_tokens", "completion_tokens", "total_tokens",
                        "cost_amount", "currency", "provider", "request_id", "model_revision", "cached_tokens",
                        "price_reference_date", "stage", "latency_seconds"})
        _identifier(payload["attempt_id"], "attempt_id")
        _identifier(payload["model"], "model")
        for name in ("provider", "request_id", "model_revision"):
            if payload[name] is not None:
                _identifier(payload[name], name)
        if any(row["event"]["event_type"] == "cost" and row["event"]["payload"]["attempt_id"] == payload["attempt_id"]
               for row in snapshot["events"]):
            raise CaptureError("provider attempt already counted")
        if payload["status"] not in {"success", "failure", "timeout", "unknown"}:
            raise CaptureError("all provider attempts, including failures, need an explicit status")
        for name in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens"):
            value = payload[name]
            if value is not None and (type(value) is not int or value < 0):
                raise CaptureError("usage must be a nonnegative integer or explicit unknown null")
        if all(payload[name] is not None for name in ("prompt_tokens", "completion_tokens", "total_tokens")):
            if payload["prompt_tokens"] + payload["completion_tokens"] != payload["total_tokens"]:
                raise CaptureError("reported token totals disagree")
        amount = payload["cost_amount"]
        if amount is not None and (type(amount) not in (int, float) or not math.isfinite(amount) or amount < 0):
            raise CaptureError("cost must be finite, nonnegative or explicitly unknown")
        if amount is not None and (not isinstance(payload["currency"], str) or not re.fullmatch("[A-Z]{3}", payload["currency"])):
            raise CaptureError("known cost requires a currency")
        if amount is None and payload["currency"] is not None:
            if not isinstance(payload["currency"], str) or not re.fullmatch("[A-Z]{3}", payload["currency"]):
                raise CaptureError("currency must be an ISO-style code or null")
        if payload["cached_tokens"] is not None and payload["prompt_tokens"] is not None:
            if payload["cached_tokens"] > payload["prompt_tokens"]:
                raise CaptureError("cached tokens cannot exceed reported prompt tokens")
        if payload["price_reference_date"] is not None:
            value = payload["price_reference_date"]
            if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                raise CaptureError("price reference date must be YYYY-MM-DD or explicit unknown null")
            try:
                datetime.fromisoformat(value)
            except ValueError as exc:
                raise CaptureError("invalid price reference date") from exc
        if payload["latency_seconds"] is not None:
            value = payload["latency_seconds"]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise CaptureError("latency must be nonnegative or null")
        if payload["stage"] not in {"extraction", "updates", "indexing_embedding", "retrieval", "generation"}:
            raise CaptureError("cost stage is outside the protocol")
        return
    if kind == "failure":
        _keys(payload, {"stage", "reason_code"})
        if payload["stage"] not in {"source_capture", "extraction", "update", "retrieval", "context", "answer", "assessment", "system"}:
            raise CaptureError("failure stage must describe an observed technical failure")
        _identifier(payload["reason_code"], "reason_code")


def _append_chain(path: Path, rows: list[dict], event: dict) -> dict:
    row = {"sequence": len(rows) + 1, "previous_sha256": _head(rows), "event": event}
    row["event_sha256"] = canonical_hash(row)
    # Rewriting the complete prefix atomically preserves every earlier byte.
    _atomic_write(path, path.read_bytes() + _json_bytes(row) + b"\n")
    return row


def record_event(root: Path, event: dict) -> dict:
    """Validate one finite observation, append once, or accept an identical retry."""
    root = Path(root)
    event = json.loads(_json_bytes(event))  # immutable caller-independent copy
    with _lock(root, exclusive=True):
        _keys(event, BASE_KEYS)
        # An unavailable/corrupt response must block scoring, never revocation.
        snapshot = _read_unlocked(root, load_responses=event["event_type"] != "withdraw")
        if (event["event_type"] != "withdraw" and
                _file_hash(Path(__file__)) != snapshot["manifest"]["source_code_sha256"]):
            raise CaptureError("capture executable changed; use frozen source or a new study directory")
        response_content = event["payload"] if event["event_type"] == "task_response" else None
        if response_content is not None:
            _identifier(event["event_id"], "event_id")
            if event["participant_id"] not in snapshot["manifest"]["slots"]:
                raise CaptureError("unknown participant slot")
            _validate_response(response_content, snapshot["manifest"]["mode"])
            artifact_bytes = _json_bytes(response_content) + b"\n"
            event["payload"] = {
                "artifact_path": f"restricted/responses/{event['participant_id']}/{event['event_id']}.json",
                "artifact_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
                "task_key_sha256": response_content["task_key_sha256"],
                "source_history_sha256": canonical_hash(response_content["source_history"]),
            }
        all_rows = snapshot["events"] + snapshot["revocations"]
        existing = next((row for row in all_rows if row["event"]["event_id"] == event["event_id"]), None)
        if existing is not None:
            if existing["event"] != event:
                raise CaptureError("event ID already exists with different content")
            # Crucially, a retry after revocation never recreates a response file.
            if event["event_type"] == "withdraw" and not event["payload"]["retain_collected_data"]:
                _erase_response_files(root, event["participant_id"])
            return {"status": "already_recorded", "event_sha256": existing["event_sha256"],
                    "synthetic_fixture": snapshot["manifest"]["mode"] == "fixture"}
        if snapshot["sealed"] is not None and event["event_type"] != "withdraw":
            raise CaptureError("collection is sealed; only consent revocation remains available")
        _validate_event(snapshot, event, response_content=response_content)
        if event["event_type"] == "withdraw":
            row = _append_chain(root / "revocations.jsonl", snapshot["revocations"], event)
            if not event["payload"]["retain_collected_data"]:
                _erase_response_files(root, event["participant_id"])
        else:
            if response_content is not None:
                path = _response_path(root, event)
                if path.exists() and path.read_bytes() != artifact_bytes:
                    raise CaptureError("orphan response artifact disagrees with retried event")
                if not path.exists():
                    _atomic_write(path, artifact_bytes)
            row = _append_chain(root / "events.jsonl", snapshot["events"], event)
        return {"status": "recorded", "event_sha256": row["event_sha256"],
                "synthetic_fixture": snapshot["manifest"]["mode"] == "fixture"}


def _erase_response_files(root: Path, participant_id: str) -> None:
    directory = root / "restricted/responses" / participant_id
    if directory.is_symlink():
        raise CaptureError("refusing to follow a symlink during response erasure")
    if directory.exists():
        shutil.rmtree(directory)


def repair_revocation_erasure(root: Path) -> dict:
    """Retry local erasure after a disk error; consent already blocks all exports."""
    root = Path(root)
    with _lock(root, exclusive=True):
        snapshot = _read_unlocked(root, load_responses=False)
        erased = []
        for participant_id, withdrawal in snapshot["withdrawals"].items():
            if not withdrawal["retain_collected_data"]:
                _erase_response_files(root, participant_id)
                erased.append(participant_id)
        return {"status": "erasure_rechecked", "participants": erased,
                "consent_snapshot_sha256": snapshot["consent_snapshot_sha256"],
                "issued_copies_or_provider_logs_deleted": False}


def study_status(root: Path) -> dict:
    snapshot = read_snapshot(root)
    enrolled = {row["event"]["participant_id"] for row in snapshot["events"] if row["event"]["event_type"] == "enroll"}
    fixture = snapshot["manifest"]["mode"] == "fixture"
    return {
        "study_id": snapshot["manifest"]["study_id"], "mode": snapshot["manifest"]["mode"],
        "planned_allocation_slots": 60, "enrolled_observation_slots": len(enrolled),
        "actual_human_participants": 0 if fixture else len(enrolled),
        "fixture_observation_slots": len(enrolled) if fixture else 0,
        "withdrawn_slots": len(snapshot["withdrawals"]),
        "erased_response_slots": sum(not row["retain_collected_data"] for row in snapshot["withdrawals"].values()),
        "collection_event_count": len(snapshot["events"]),
        "revocation_event_count": len(snapshot["revocations"]),
        "sealed": snapshot["sealed"] is not None,
        "manifest_sha256": snapshot["manifest"]["manifest_sha256"],
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "consent_snapshot_sha256": snapshot["consent_snapshot_sha256"],
        "human_enrollment_launch_clearance_present": snapshot["manifest"]["launch_record_sha256"] is not None,
        "clinical_effectiveness_established": False,
        "model_or_network_calls_performed_by_capture": 0,
    }


def seal_study(root: Path) -> dict:
    """Freeze collection, including incomplete/missing visits; do not fabricate them."""
    root = Path(root)
    with _lock(root, exclusive=True):
        snapshot = _read_unlocked(root)
        if _file_hash(Path(__file__)) != snapshot["manifest"]["source_code_sha256"]:
            raise CaptureError("capture executable changed")
        if snapshot["sealed"] is not None:
            return snapshot["sealed"]
        sealed = {"schema_version": SCHEMA_VERSION, "sealed_at": _now(),
                  "manifest_sha256": snapshot["manifest"]["manifest_sha256"],
                  "event_count": len(snapshot["events"]), "event_head_sha256": _head(snapshot["events"]),
                  "snapshot_sha256": snapshot["snapshot_sha256"],
                  "mode": snapshot["manifest"]["mode"],
                  "sealing_implies_task_completion": False,
                  "postseal_consent_revocation_permitted": True}
        _atomic_write(root / "seal.json", _json_bytes(sealed) + b"\n")
        return sealed


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="freeze 60 planned slots; enroll no people")
    prepare.add_argument("--study-dir", type=Path, required=True)
    prepare.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    prepare.add_argument("--mode", choices=["planned_human", "fixture"], default="planned_human")
    prepare.add_argument("--seed", type=int)
    prepare.add_argument("--launch-record", type=Path)
    prepare.add_argument("--task-ledger", type=Path)
    for command in ("record", "enroll", "withdraw", "status", "seal", "repair-erasure"):
        child = sub.add_parser(command)
        child.add_argument("--study-dir", type=Path, required=True)
        if command in {"record", "enroll", "withdraw"}:
            child.add_argument("--event-json", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare_study(args.study_dir, args.protocol, mode=args.mode, seed=args.seed,
                                   launch_record_path=args.launch_record, task_ledger_path=args.task_ledger)
        elif args.command in {"record", "enroll", "withdraw"}:
            event = _read_json(args.event_json)
            if args.command == "enroll" and event.get("event_type") != "enroll":
                raise CaptureError("enroll command requires an enrollment event")
            if args.command == "withdraw" and event.get("event_type") != "withdraw":
                raise CaptureError("withdraw command requires a consent revocation event")
            result = record_event(args.study_dir, event)
        elif args.command == "seal":
            result = seal_study(args.study_dir)
        elif args.command == "repair-erasure":
            result = repair_revocation_erasure(args.study_dir)
        else:
            result = study_status(args.study_dir)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    except (CaptureError, OSError) as exc:
        raise SystemExit(f"Capture refused: {exc}") from None


if __name__ == "__main__":
    _main()
