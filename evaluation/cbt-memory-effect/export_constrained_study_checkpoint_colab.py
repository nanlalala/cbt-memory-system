"""Export the independent constrained study from hosted Colab, without inference.

Only the fixed study tree and explicitly named recovery evidence are captured.
Unsettled/failed calls are retained as live_partial, never called a final study.
The exporter never imports runners, loads models, retries calls, or reads weights.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
from io import BytesIO
import importlib.util
import json
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import stat
import tempfile
import time
import zipfile


ROOT = Path("/content/cbt-constrained-state-study-20261112-v2")
CONTENT = Path("/content")
VERSION = "constrained-state-development-study-v2"
MAX_BYTES = 256_000_000
MAX_FILE_BYTES = 16_000_000
MAX_FILES = 3000
ROOT_FILES = {
    "study_protocol.json", "main_freeze.json", "generation_ledger.jsonl", "load_protocol.json",
    "model_files.json", "model_load_receipt.json", "model_load_events.jsonl",
    "prepare_before.log", "prepare_after.log", "source_gate_audit_before.log", "source_gate_audit_after.log",
}
SUBTREES = {"pilot_prepared", "pilot_answers", "source_prepared", "prepared", "source_gate_audits",
            "answers_before", "answers_after", "reports"}
LOCKS = {".model_setup.lock", ".generation.lock"}
EXTENSIONS = {".json", ".jsonl", ".sqlite", ".log"}
FORBIDDEN = {".git", ".github", ".ssh", ".config", ".cache", ".env", "credentials", "secrets"}
CODE_NAMES = {"run_prefill_pilot_v2_colab.py", "open_reader_colab.py", "constrained_prefill_pilot_v2.py",
              "constrained_json_prefill_v2.py", "constrained_json_grammar.py", "constrained_open_reader.py"}
OLD_ARCHIVES = {
    "complete_62_archive_sha256": "8cee7b783915c8da05f428c2c45ab5e0e1f8c9507a21c7946ebcf317f5887e33",
    "supplement_64_archive_sha256": "6f67748b220724bb4b510092e229c42e262b458ef67896fd196633e0757174c1",
}
RESTORE_PATH = CONTENT / "cbt-open-state-restore-receipt-20261112-v1.json"
RESTORE_SHA = "3e313f7e7e70e3eb36704c190b5cf36496a71d3e6e59efa1d947449ee267f635"
BOOTSTRAP_PATH = CONTENT / "cbt-prefill-recovery-bootstrap-20261112-v2.py"
BOOTSTRAP_SHA = "626c7d7877283897ba0c0d9e8b327171405616830b52f8ac5b8af18bc94ca25c"
REACQUISITION_PATH = CONTENT / "cbt-model-reacquisition-20261112-v1.json"
OPTIONAL_RECOVERY = (
    CONTENT / "cbt-prefill-recovery-bootstrap-20261112-v2.log",
    CONTENT / "cbt-prefill-recovery-bootstrap-20261112-v1.py",
    CONTENT / "cbt-prefill-recovery-bootstrap-20261112-v1.log",
    CONTENT / "cbt-prefill-recovery-bootstrap-20261112-v3.py",
    CONTENT / "cbt-prefill-recovery-bootstrap-20261112-v3.log",
    CONTENT / "cbt-model-download-failure-20261112-v1.json",
)
OPTIONAL_RECOVERY_SOURCE_SHA256 = {
    str(CONTENT / "cbt-prefill-recovery-bootstrap-20261112-v3.py"):
        "4a643e1bb050c06a6d6ca41e2374f057981c98a795198ceab05044849b631808",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate_json_key")
        value[key] = item
    return value


def parse(data):
    return json.loads(data, object_pairs_hook=_unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))


def guard():
    try:
        valid = (bool(os.environ.get("COLAB_RELEASE_TAG")) and CONTENT.is_dir()
                 and not CONTENT.is_symlink() and importlib.util.find_spec("google.colab") is not None)
    except (ImportError, ValueError, AttributeError):
        valid = False
    if not valid:
        raise RuntimeError("genuine_hosted_colab_required_before_any_io")


def safe_member(name, *, recovery=False):
    if (not isinstance(name, str) or not name or "\\" in name or "\x00" in name
            or name.startswith("/") or any(p in ("", ".", "..") for p in name.split("/"))):
        raise ValueError("unsafe_archive_member")
    path = PurePosixPath(name)
    if any(p.lower() in FORBIDDEN for p in path.parts):
        raise ValueError("secret_or_repository_configuration_prohibited")
    allowed = EXTENSIONS | ({".py"} if recovery else set())
    if path.suffix not in allowed:
        raise ValueError("only_named_study_artifacts_no_weights_allowed")
    return path


def content_path(value):
    raw = str(value)
    if not raw.startswith("/content/") or "\\" in raw or "\x00" in raw:
        raise ValueError("absolute_content_path_required")
    parts = raw[len("/content/"):].split("/")
    if any(p in ("", ".", "..") or p.lower() in FORBIDDEN for p in parts):
        raise ValueError("unsafe_content_path")
    return CONTENT.joinpath(*parts)


def output_path(value, suffix):
    path = content_path(value)
    if (path.parent != CONTENT or path.suffix != suffix
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", path.name)):
        raise ValueError("new_direct_content_archive_or_receipt_required")
    return path


def open_regular(path, *, missing=False):
    """Open every ancestor without following symlinks."""
    path = content_path(path)
    directory = os.open(CONTENT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parts = path.relative_to(CONTENT).parts
        for part in parts[:-1]:
            following = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = following
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            os.close(fd)
            raise ValueError("regular_file_required")
        return fd
    except FileNotFoundError:
        if missing:
            return None
        raise
    finally:
        os.close(directory)


def read_bytes(path, *, missing=False):
    fd = open_regular(path, missing=missing)
    if fd is None:
        return None
    with os.fdopen(fd, "rb") as stream:
        if os.fstat(stream.fileno()).st_size > MAX_FILE_BYTES:
            raise ValueError("artifact_file_bound_exceeded")
        data = stream.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("artifact_file_bound_exceeded")
    return data


def check_optional_recovery_source(path, data):
    expected = OPTIONAL_RECOVERY_SOURCE_SHA256.get(str(path))
    if expected is not None and sha(data) != expected:
        raise ValueError("pinned_optional_recovery_source_sha256_changed:" + str(path))


def discover(root):
    if str(root) != str(ROOT):
        raise ValueError("fixed_independent_study_root_required")
    if root.is_symlink() or not root.is_dir():
        raise ValueError("regular_fixed_study_directory_required")
    found, pending = [], [(root, ())]
    directory_count = 0
    while pending:
        directory, prefix = pending.pop()
        directory_count += 1
        if directory_count > MAX_FILES:
            raise ValueError("study_directory_count_bound_exceeded")
        for item in sorted(os.scandir(directory), key=lambda x: x.name):
            parts = prefix + (item.name,)
            name = "/".join(parts)
            if item.is_symlink():
                raise ValueError("symlinks_prohibited:" + name)
            if not prefix and item.name in LOCKS:
                if not item.is_file(follow_symlinks=False):
                    raise ValueError("regular_lock_file_required")
                continue
            if item.is_dir(follow_symlinks=False):
                if not prefix and item.name not in SUBTREES:
                    raise ValueError("unlisted_study_subtree:" + name)
                if any(part.lower() in FORBIDDEN for part in parts):
                    raise ValueError("repository_or_secret_directory_prohibited")
                pending.append((Path(item.path), parts))
            elif item.is_file(follow_symlinks=False):
                if item.name.endswith((".sqlite-wal", ".sqlite-shm", ".sqlite-journal")):
                    base = Path(item.path.rsplit("-", 1)[0])
                    if base.is_symlink() or not base.is_file():
                        raise ValueError("orphan_or_unsafe_sqlite_sidecar")
                    # SQLite's read-only backup incorporates committed WAL data.
                    # Never archive a sidecar as an independently restorable DB.
                    continue
                safe_member(name)
                if not prefix and item.name not in ROOT_FILES:
                    raise ValueError("unlisted_root_artifact:" + name)
                if prefix and parts[0] not in SUBTREES:
                    raise ValueError("unlisted_study_subtree")
                if item.stat(follow_symlinks=False).st_size > MAX_FILE_BYTES:
                    raise ValueError("artifact_file_bound_exceeded")
                found.append((name, Path(item.path)))
                if len(found) > MAX_FILES:
                    raise ValueError("study_file_count_bound_exceeded")
            else:
                raise ValueError("only_regular_files_and_directories_allowed")
    return sorted(found)


def checked_rows(data, *, optional=False):
    if data is None or not data:
        if optional:
            return []
        raise ValueError("nonempty_journal_required")
    if not data.endswith(b"\n"):
        raise ValueError("incomplete_jsonl_tail")
    rows = []
    for line in data.splitlines():
        if not line.strip():
            raise ValueError("blank_journal_record")
        row = parse(line)
        if not isinstance(row, dict):
            raise ValueError("journal_object_required")
        body = {k: v for k, v in row.items() if k != "record_sha256"}
        if sha(canonical(body).encode()) != row.get("record_sha256"):
            raise ValueError("journal_record_checksum_changed")
        rows.append(row)
    return rows


def _object(files, name):
    if name not in files:
        raise ValueError("missing_metadata:" + name)
    value = parse(files[name])
    if not isinstance(value, dict):
        raise ValueError("metadata_object_required:" + name)
    return value


def inspect_state(files):
    """Only metadata/journal authentication; preserves raw failed records too."""
    state = {"stage": "live_partial", "terminal": False, "all_intents_published": False,
             "new_attempts": None, "published_outcomes": None, "issues": [],
             "maximum_new_study_attempts": 61, "automatic_resume_or_resend_authorized": False}
    try:
        protocol = _object(files, "study_protocol.json")
        if (protocol.get("version") != VERSION or protocol.get("study_root") != str(ROOT)
                or protocol.get("maximum_new_study_attempts") != 61
                or protocol.get("planned_pilot_attempts") != 4
                or protocol.get("planned_main_unique_attempts") != 57):
            raise ValueError("study_identity_or_fixed_cap_changed")
        intents, outcomes = {}, {}
        for row in checked_rows(files.get("generation_ledger.jsonl"), optional=True):
            kind = row.get("event")
            target = intents if kind == "generation_intent" else outcomes if kind == "generation_outcome" else None
            key = row.get("canonical_key")
            if target is None or not isinstance(key, str) or key in target:
                raise ValueError("unknown_or_duplicate_study_event")
            target[key] = row
        state.update(new_attempts=len(intents), published_outcomes=sum(x.get("status") == "published" for x in outcomes.values()),
                     outcome_statuses={status: sum(r.get("status") == status for r in outcomes.values())
                                       for status in sorted({str(r.get("status")) for r in outcomes.values()})})
        if (len(intents) > 61 or set(intents) != set(outcomes)
                or any(type(x.get("attempt_number")) is not int for x in intents.values())
                or sorted(x.get("attempt_number", -1) for x in intents.values()) != list(range(1, len(intents) + 1))
                or any(x.get("status") != "published" for x in outcomes.values())):
            raise ValueError("unsettled_failed_or_over_budget_journal_no_retry")
        for key, row in outcomes.items():
            cache_path = content_path(row.get("cache_path", ""))
            if not cache_path.is_relative_to(ROOT) or cache_path.parent.name != "canonical" or cache_path.name != key + ".json":
                raise ValueError("foreign_or_unbound_study_cache")
            relative = str(cache_path.relative_to(ROOT))
            if relative not in files or sha(files[relative]) != row.get("cache_sha256"):
                raise ValueError("published_cache_missing_or_hash_changed")
            cache = _object(files, relative)
            text = cache.get("response_text")
            usage = cache.get("usage") or {}
            if (cache.get("canonical_key") != key or not isinstance(text, str)
                    or sha(text.encode()) != row.get("response_sha256")
                    or any(type(usage.get(field)) is not int or usage[field] < 0
                           for field in ("prompt_tokens", "completion_tokens", "total_tokens"))
                    or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]
                    or any(cache.get(k) != row.get(k) for k in ("response_sha256", "usage", "finish_reason"))):
                raise ValueError("published_response_or_usage_binding_changed")
        setups, completed = set(), set()
        for row in checked_rows(files.get("model_load_events.jsonl"), optional=True):
            event, setup = row.get("event"), row.get("setup_id")
            target = setups if event == "model_reload_intent" else completed if event == "model_reload_completed" else None
            if target is None or not isinstance(setup, str) or setup in target:
                raise ValueError("failed_unknown_or_duplicate_model_setup_event")
            target.add(setup)
        if setups != completed:
            raise ValueError("unsettled_model_setup")
        if intents and not completed:
            raise ValueError("published_calls_require_new_study_model_setup_receipt")
        state["all_intents_published"] = True
        state["completed_new_model_setups"] = len(completed)
        state["stage"] = "settled_incomplete_study"

        def complete(section, expected):
            summary = _object(files, section + "/completion_summary.json")
            run = _object(files, section + "/run_protocol.json")
            answers = checked_rows(files.get(section + "/answers_unscored.jsonl"))
            if section == "pilot_answers":
                planned = parse(files["pilot_prepared/requests.json"])
            else:
                side = section.removeprefix("answers_")
                raw = files["prepared/" + side + "/requests.jsonl"]
                if not raw.endswith(b"\n"):
                    raise ValueError("incomplete_prepared_request_journal")
                planned = [parse(line) for line in raw.splitlines()]
            if (summary.get("complete") is not True or summary.get("logical_completed") != expected
                    or len(answers) != expected
                    or not isinstance(planned, list) or len(planned) != expected
                    or run.get("rows_sha256") != sha(canonical(planned).encode())
                    or len(run.get("actual_tensor_preflight", [])) != expected
                    or summary.get("answers_sha256") != sha(files[section + "/answers_unscored.jsonl"])
                    or summary.get("run_protocol_sha256") != sha(canonical(run).encode())
                    or run.get("study_protocol_sha256") != sha(files["study_protocol.json"])):
                raise ValueError("incomplete_or_unbound_logical_answers:" + section)
            for wanted, answer, preflight in zip(planned, answers, run["actual_tensor_preflight"]):
                key = answer.get("canonical_key")
                if key not in outcomes or answer.get("status") != "completed":
                    raise ValueError("logical_answer_has_no_published_outcome")
                cache = parse(files[str(content_path(outcomes[key]["cache_path"]).relative_to(ROOT))])
                if (answer.get("run_protocol_sha256") != sha(canonical(run).encode())
                        or answer.get("case_id") != wanted.get("case_id")
                        or answer.get("arm") != wanted.get("arm", wanted.get("decode_mode", "prefill_v2"))
                        or answer.get("request_sha256") != wanted.get("request_sha256")
                        or wanted.get("request_sha256") != sha(canonical(wanted.get("request_payload")).encode())
                        or intents[key].get("request_sha256") != wanted.get("request_sha256")
                        or answer.get("input_tensor_sha256") != preflight.get("input_tensor_sha256")
                        or (answer.get("usage") or {}).get("prompt_tokens") != preflight.get("input_tokens")
                        or any(answer.get(k) != cache.get(k) for k in ("response_text", "response_sha256", "usage", "finish_reason", "input_tensor_sha256"))):
                    raise ValueError("logical_answer_disagrees_with_original_cache")
            return answers

        if len(intents) < 4:
            return state
        pilot_answers = complete("pilot_answers", 4)
        score = _object(files, "pilot_answers/pilot_score.json")
        schema, correct = score.get("prefill_schema_valid"), score.get("prefill_basic_readability_correct")
        if (type(schema) is not int or type(correct) is not int or not 0 <= schema <= 2 or not 0 <= correct <= 2
                or score.get("prefill_denominator") != 2 or score.get("ready_for_main") is not (schema == correct == 2)):
            raise ValueError("pilot_gate_report_not_consistent")
        if (score.get("prepared_protocol_sha256") != sha(files["pilot_prepared/protocol.json"])
                or [(row.get("case_id"), row.get("decode_mode"), row.get("response_sha256")) for row in score.get("rows", [])]
                != [(row.get("case_id"), row.get("decode_mode", row.get("arm")), row.get("response_sha256")) for row in pilot_answers]):
            raise ValueError("pilot_gate_report_not_bound_to_original_preparation_and_answers")
        state["pilot_gate"] = {"prefill_schema_valid": schema, "prefill_basic_readability_correct": correct,
                               "denominator": 2, "passed": schema == correct == 2}
        if schema != 2 or correct != 2:
            if len(intents) != 4 or "main_freeze.json" in files or any(n.startswith(("answers_before/", "answers_after/")) for n in files):
                raise ValueError("main_activity_after_failed_pilot_gate")
            state.update(stage="pilot_gate_rejected_settled", terminal=True)
            return state
        if len(intents) == 61:
            before = complete("answers_before", 48)
            after = complete("answers_after", 48)
            if {row["canonical_key"] for row in pilot_answers + before + after} != set(intents):
                raise ValueError("completed_logical_grid_does_not_cover_every_admitted_call")
            freeze = _object(files, "main_freeze.json")
            if freeze.get("study_protocol_sha256") != sha(files["study_protocol.json"]):
                raise ValueError("main_freeze_protocol_binding_changed")
            # Reports are preserved as raw bytes. Their changing presentation/schema
            # does not become part of the immutable pilot code identity.
            for name in ("before_strict.json", "after_strict.json", "resources.json", "summary.json"):
                if "reports/" + name not in files:
                    raise ValueError("completed_answers_require_saved_report:" + name)
            state.update(stage="completed_study_settled", terminal=True,
                         main_reports="Raw bytes preserved. Terminal status authenticates settled calls and complete logical answers; it does not independently validate scores or require a particular report schema.")
        return state
    except (ValueError, TypeError, KeyError, UnicodeError) as error:
        state.update(stage="live_partial", terminal=False)
        state["issues"].append(str(error))
        return state


def reference_identity(files):
    """Validate only the frozen pilot/runtime identity, never freeze a future main renderer."""
    protocol = _object(files, "study_protocol.json")
    evidence = []
    inherited = protocol.get("inherited_identity_sha256", {})
    inherited_names = {"load_protocol.json", "model_files.json", "model_load_receipt.json"}
    if not isinstance(inherited, dict) or set(inherited) != inherited_names:
        raise ValueError("exact_three_inherited_model_identity_files_required")
    for name, expected in inherited.items():
        if name not in files or sha(files[name]) != expected:
            raise ValueError("inherited_model_identity_changed")
    historical = protocol.get("historical_reference_sha256", {})
    required_history = {str(CONTENT / "cbt-open-reader-20261112-v1" / name)
                        for name in inherited_names | {"generation_ledger.jsonl"}}
    if not isinstance(historical, dict) or set(historical) != required_history:
        raise ValueError("exact_four_old_run_reference_paths_required")
    for name, expected in historical.items():
        path = content_path(name)
        if path.parent != CONTENT / "cbt-open-reader-20261112-v1" or path.name not in {
            "generation_ledger.jsonl", "load_protocol.json", "model_files.json", "model_load_receipt.json"
        }:
            raise ValueError("unlisted_old_run_reference")
        if sha(read_bytes(path)) != expected:
            raise ValueError("old_64_reference_changed")
        evidence.append({"path": str(path), "sha256": expected, "copied": False})
    codes = protocol.get("code_sha256", {})
    if not isinstance(codes, dict) or {Path(p).name for p in codes} != CODE_NAMES:
        raise ValueError("frozen_pilot_runtime_code_set_changed")
    for name, expected in codes.items():
        path = content_path(name)
        if path.name not in CODE_NAMES or path.parent.parts[-2:] != ("evaluation", "cbt-memory-effect"):
            raise ValueError("unlisted_runtime_code_path")
        if sha(read_bytes(path)) != expected:
            raise ValueError("frozen_pilot_runtime_code_changed")
        evidence.append({"path": str(path), "sha256": expected, "copied": False})
    return evidence


def capture_sqlite(source, temporary):
    # Read-only source; backup bytes intentionally need not match the live DB hash.
    destination = Path(temporary) / "snapshot.sqlite"
    if destination.exists():
        destination.unlink()
    deadline = time.monotonic() + 30

    def bounded_progress(status, remaining, total):
        if time.monotonic() > deadline:
            raise TimeoutError("readonly_sqlite_backup_time_bound_exceeded")

    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True, timeout=10) as original:
        with sqlite3.connect(destination) as copied:
            original.backup(copied, pages=128, progress=bounded_progress, sleep=0.1)
    if destination.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("sqlite_snapshot_bound_exceeded")
    return destination.read_bytes()


def write_new(path, data):
    parent = os.open(CONTENT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(parent)
    fd = open_regular(path)
    digest = hashlib.sha256()
    count = 0
    with os.fdopen(fd, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            count += len(block)
            digest.update(block)
    if count != len(data) or digest.hexdigest() != sha(data):
        raise RuntimeError("new_export_file_postwrite_verification_failed")


def export(archive, receipt, *, model_reacquisition_sha256):
    guard()
    archive, receipt = output_path(archive, ".zip"), output_path(receipt, ".json")
    if not re.fullmatch(r"[0-9a-f]{64}", model_reacquisition_sha256):
        raise ValueError("actual_model_reacquisition_receipt_sha256_required")
    for path in (archive, receipt):
        if os.path.lexists(path):
            raise FileExistsError("preserve_previous_export_use_new_paths:" + str(path))
    members = discover(ROOT)
    planned_bytes = sum(source.stat().st_size for _, source in members)
    if planned_bytes > MAX_BYTES:
        raise ValueError("study_uncompressed_size_bound_exceeded")
    started = datetime.now(timezone.utc).isoformat()
    files, records, originals, missing, lock_status = {}, [], {}, [], {}
    with ExitStack() as stack:
        for name in sorted(LOCKS):
            fd = open_regular(ROOT / name, missing=True)
            if fd is None:
                lock_status[name] = "absent"
                continue
            stack.callback(os.close, fd)
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                stack.callback(fcntl.flock, fd, fcntl.LOCK_UN)
                lock_status[name] = "held_shared_no_generation_or_setup_writer"
            except BlockingIOError:
                lock_status[name] = "active_writer_live_partial"
        members = discover(ROOT)
        with tempfile.TemporaryDirectory(prefix="cbt-constrained-export-", dir=CONTENT) as temporary:
            for name, source in members:
                if source.suffix == ".sqlite":
                    # Validate every source ancestor before SQLite opens its URI.
                    fd = open_regular(source)
                    os.close(fd)
                    data = capture_sqlite(source, temporary)
                    method = "consistent_sqlite_backup_from_read_only_connection"
                else:
                    data = read_bytes(source)
                    originals[str(source)] = sha(data)
                    method = "exact_bytes_read_during_capture"
                member = "study/" + name
                safe_member(member)
                files[name] = data
                records.append({"path": member, "source_path": str(source), "bytes": len(data),
                                "sha256": sha(data), "method": method})
        payloads = {"study/" + name: data for name, data in files.items()}
        references = ((RESTORE_PATH, RESTORE_SHA), (BOOTSTRAP_PATH, BOOTSTRAP_SHA),
                      (REACQUISITION_PATH, model_reacquisition_sha256))
        for source, expected in references:
            data = read_bytes(source)
            if sha(data) != expected:
                raise ValueError("pinned_recovery_evidence_sha256_changed:" + str(source))
            member = "recovery/" + source.name
            safe_member(member, recovery=True)
            payloads[member] = data
            records.append({"path": member, "source_path": str(source), "bytes": len(data),
                            "sha256": sha(data), "method": "exact_bytes_read_during_capture"})
            originals[str(source)] = sha(data)
        for source in OPTIONAL_RECOVERY:
            data = read_bytes(source, missing=True)
            if data is None:
                missing.append(str(source))
                continue
            check_optional_recovery_source(source, data)
            member = "recovery/" + source.name
            safe_member(member, recovery=True)
            payloads[member] = data
            records.append({"path": member, "source_path": str(source), "bytes": len(data),
                            "sha256": sha(data), "method": "exact_bytes_read_during_capture"})
            originals[str(source)] = sha(data)
        total = sum(len(data) for data in payloads.values())
        if total > MAX_BYTES:
            raise ValueError("combined_uncompressed_size_bound_exceeded")
        state = inspect_state(files)
        try:
            identity = reference_identity(files)
        except (ValueError, TypeError, KeyError, OSError) as error:
            identity = []
            state.update(stage="live_partial", terminal=False)
            state["issues"].append("identity_validation:" + str(error))
        changed = []
        for source, expected in originals.items():
            current = read_bytes(source, missing=True)
            if current is None or sha(current) != expected:
                changed.append(source)
        membership_changed = {name for name, _ in discover(ROOT)} != {name for name, _ in members}
        if changed or membership_changed or any(status != "held_shared_no_generation_or_setup_writer" for status in lock_status.values()):
            state.update(stage="live_partial", terminal=False)
            state["issues"].append("source_changed_or_existing_writer_lock_not_held")
        manifest = {"version": "constrained-study-private-checkpoint-v1", "study_root": str(ROOT),
            "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_state": state, "files": sorted(records, key=lambda row: row["path"]),
            "uncompressed_artifact_bytes": total, "old_64_archive_references": OLD_ARCHIVES,
            "restore_receipt_sha256": RESTORE_SHA, "bootstrap_v2_sha256": BOOTSTRAP_SHA,
            "optional_recovery_source_expected_sha256": OPTIONAL_RECOVERY_SOURCE_SHA256,
            "model_reacquisition_receipt_sha256": model_reacquisition_sha256,
            "verified_frozen_runtime_and_historical_references": identity,
            "missing_optional_recovery_files": missing, "existing_runtime_lock_observations": lock_status,
            "changed_source_files_during_capture": changed, "collection_is_not_atomic_across_files": True,
            "study_file_membership_changed_during_capture": membership_changed,
            "sqlite_backups_individually_consistent": True,
            "sqlite_wal_shm_journal_sidecars_not_copied_separately": True,
            "sqlite_snapshot_hashes_do_not_assert_original_live_database_byte_identity": True,
            "exporter_code_sha256": sha(Path(__file__).read_bytes()),
            "new_model_calls": 0, "new_model_loads": 0, "weights_credentials_repository_configuration_included": False,
            "old_archives_copied": False, "inherited_setup_receipts_are_nonadditive_historical_references": True,
            "bootstrap_v1_failure": "Optional original v1 script/log preserve a Python parse failure before restore or model generation; no generation slot, retry, or model-unknown event is inferred; unmeasured elapsed time remains null.",
            "bootstrap_v1_elapsed_seconds": None, "automatic_resume_or_resend_authorized": False}
        manifest_data = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
        if total + len(manifest_data) > MAX_BYTES:
            raise ValueError("manifest_inclusive_uncompressed_size_bound_exceeded")
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
            for name, data in sorted(payloads.items()):
                info = zipfile.ZipInfo(name)
                info.external_attr = (stat.S_IFREG | 0o600) << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                zipped.writestr(info, data)
            zipped.writestr("checkpoint_manifest.json", manifest_data)
        archive_data = buffer.getvalue()
        result = {"version": manifest["version"], "archive": str(archive), "bytes": len(archive_data),
                  "sha256": sha(archive_data), "manifest_sha256": sha(manifest_data), "artifact_files": len(records),
                  "checkpoint_state": state, "new_model_calls": 0, "new_model_loads": 0,
                  "automatic_resume_or_resend_authorized": False}
        # Both destinations were checked before capture; O_EXCL also prevents races.
        write_new(archive, archive_data)
        write_new(receipt, (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--model-reacquisition-sha256", required=True,
                        help="SHA256 actually observed after the named reacquisition receipt is written")
    args = parser.parse_args()
    print(json.dumps(export(args.archive, args.receipt,
                            model_reacquisition_sha256=args.model_reacquisition_sha256), sort_keys=True))
