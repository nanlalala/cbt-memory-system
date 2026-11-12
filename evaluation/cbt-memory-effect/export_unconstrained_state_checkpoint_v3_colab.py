"""Export the independent v3 unconstrained study from hosted Colab without inference.

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
import math
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import stat
import tempfile
import time
import zipfile


ROOT = Path("/content/cbt-unconstrained-state-study-20261112-v3")
CONTENT = Path("/content")
VERSION = "unconstrained-state-development-study-v3"
STUDY_PROTOCOL_SHA256 = "118bccbf288105cafbe4c88e6b582f0742b1b28056c8d06a5cf7860296841d16"
FROZEN_HELPER_COMMIT = "8d1b0f7e6688dc480ad4ef282a80b8c1fd4916fb"
MAX_BYTES = 256_000_000
MAX_FILE_BYTES = 16_000_000
MAX_FILES = 3000
ROOT_FILES = {
    "study_protocol.json", "main_freeze.json", "generation_ledger.jsonl", "load_protocol.json",
    "model_files.json", "model_load_receipt.json", "model_load_events.jsonl",
    "prepare_before.log", "prepare_after.log", "render_before.log", "render_after.log",
    "source_gate_audit_before.log", "source_gate_audit_after.log",
}
SUBTREES = {"source_prepared", "prepared", "source_gate_audits",
            "answers_before", "answers_after", "reports"}
LOCKS = {".model_setup.lock", ".generation.lock"}
EXTENSIONS = {".json", ".jsonl", ".sqlite", ".log"}
FORBIDDEN = {".git", ".github", ".ssh", ".config", ".cache", ".env", "credentials", "secrets"}
FROZEN_CODE_SHA256 = {
    "unconstrained_state_runtime_v3.py": "f717f70a0ecc15bb9f43a89cfae2cd983ed8e51acfff9c2b255b8ba59a67510e",
    "unconstrained_state_comparison_v3_colab.py": "4a350371c52dc34ac5c5f7c7d7880e2fd70d13ec450698b10c0af5e6d43812bb",
    "common_answer_adapter_v3.py": "545ee307f07576f34bcc8780b3b263b4ca8118690a283764a0e8374e9316302e",
    "common_retrieval_policy_v1.py": "f4bbdca9bf9c0fefe0ba210aa7ad3bab6c53c9cd48e531f1b1c09a832347390a",
    "open_reader_colab.py": "fd37280f4679bb8e5bf8f4bf8ccfc4339c0530ae193df389ba91bee3176dfae1",
}
PRIOR_ARCHIVES = {
    "complete_62_archive_sha256": "8cee7b783915c8da05f428c2c45ab5e0e1f8c9507a21c7946ebcf317f5887e33",
    "supplement_64_archive_sha256": "6f67748b220724bb4b510092e229c42e262b458ef67896fd196633e0757174c1",
    "failed_v2_four_pilot_archive_sha256": "09d32eb1eb8616c2fffbfc2bdfdab5618c8ffb5f226a8759008d37ee28326ab2",
    "transport_archive_sha256": "d4d4530e5db9a94f5187a13fece3c4af96d1b4d90ba96d9d92cc6720049e6f51",
    "continuation_four_pilot_and_recovery_bundle_sha256": "247ba6385853ce20c8a971dc6930ca1bc68988b5b1141f5efbe865b447946ef4",
}
RESTORE_PATH = CONTENT / "cbt-open-state-restore-receipt-20261112-v1.json"
RESTORE_SHA = "3e313f7e7e70e3eb36704c190b5cf36496a71d3e6e59efa1d947449ee267f635"
REACQUISITION_PATH = CONTENT / "cbt-model-reacquisition-20261112-v1.json"
REACQUISITION_SHA = "678011720f92a4578167de10ec383c91163a093b92adfddf6daeb55974848b60"
OLD_ROOT = CONTENT / "cbt-open-reader-20261112-v1"
FAILED_V2_ROOT = CONTENT / "cbt-constrained-state-study-20261112-v2"
V2_REFERENCES = {
    "study_protocol.json", "generation_ledger.jsonl", "pilot_prepared/protocol.json",
    "pilot_prepared/requests.json", "pilot_prepared/assessor_only/targets.json",
    "pilot_answers/run_protocol.json", "pilot_answers/answers_unscored.jsonl",
    "pilot_answers/completion_summary.json", "pilot_answers/pilot_score.json",
}
DECODING = {"version": "unconstrained-greedy-common-adapter-reader-v3", "constraint": None,
            "assistant_prefix": None, "max_new_tokens": 64, "do_sample": False}
ARMS = {"raw_history_rag", "managed_memory_rag"}
OPTIONAL_WORKER_ARTIFACTS = (
    CONTENT / "cbt-unconstrained-v3-before-20261112.log",
    CONTENT / "cbt-unconstrained-v3-after-20261112.log",
    CONTENT / "cbt-unconstrained-v3-verify-before-20261112.log",
    CONTENT / "cbt-unconstrained-v3-verify-after-20261112.log",
    CONTENT / "cbt-unconstrained-v3-score-20261112.log",
    CONTENT / "cbt-unconstrained-v3-orchestrator-20261112-v1.py",
    CONTENT / "cbt-unconstrained-v3-orchestrator-20261112-v1.log",
    CONTENT / "cbt-unconstrained-v3-orchestrator-20261112-v1.marker.json",
    CONTENT / "cbt-unconstrained-v3-orchestrator-20261112-v1.journal.jsonl",
)


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


def digest(value):
    return sha(canonical(value).encode())


def bounded_size(current, additional):
    total = current + additional
    if total > MAX_BYTES:
        raise ValueError("cumulative_captured_byte_bound_exceeded")
    return total


def capture_observations(state, locks, changed, membership_changed):
    state = {**state, "issues": list(state["issues"])}
    if (changed or membership_changed or set(locks) != LOCKS
            or any(value != "held_shared_no_generation_or_setup_writer" for value in locks.values())):
        state.update(stage="live_partial", terminal=False)
        state["issues"].append("source_changed_or_runtime_writer_not_excluded")
    return state


def finite_time(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def plain_rows(data):
    if not data or not data.endswith(b"\n"):
        raise ValueError("complete_nonempty_request_journal_required")
    result = [parse(line) for line in data.splitlines()]
    if any(not isinstance(row, dict) for row in result):
        raise ValueError("request_object_required")
    return result


def study_relative(value):
    path = content_path(value)
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError("owned_v3_study_path_required")
    return str(path.relative_to(ROOT))


def adapter_expected(raw, domain):
    """Independent output-only acceptance audit; never read questions or targets."""
    integer = re.compile(r"(?:0|[1-9][0-9]{0,3})\Z", re.ASCII)

    def number(token):
        if integer.fullmatch(token) is None:
            raise ValueError("noncanonical_integer")
        return int(token)

    def reject_number(token):
        raise ValueError("noninteger_number")

    def decode(text):
        return json.loads(text, object_pairs_hook=_unique, parse_int=number,
                          parse_float=reject_number, parse_constant=reject_number)

    def allowed(value):
        if (value is None and domain["null_allowed"] is True
                or type(value) is int and domain["integer_minimum"] <= value <= domain["integer_maximum"]
                or type(value) is str and value in domain["string_enum"]):
            return value
        raise ValueError("outside_domain")

    def scalar(text):
        if integer.fullmatch(text):
            return int(text), "integer"
        if text in domain["string_enum"]:
            return text, "status"
        if text == "null":
            return allowed(None), "null"
        if text.startswith('"'):
            value = decode(text)
            if type(value) is str and value in domain["string_enum"]:
                return value, "json_quoted_status"
        raise ValueError("unsupported_scalar")

    try:
        text = raw.strip()
        if text.startswith("{"):
            obj = decode(text)
            if type(obj) is not dict or set(obj) != {"answer"}:
                raise ValueError("single_answer_key_required")
            value, rule = allowed(obj["answer"]), "single_answer_json"
        elif text.startswith("answer:"):
            value, form = scalar(text[len("answer:"):].strip())
            rule = "answer_prefix_" + form
        else:
            value, form = scalar(text)
            rule = form if form == "json_quoted_status" else "bare_" + form
        return True, canonical({"answer": value}), rule
    except (ValueError, TypeError, KeyError, RecursionError):
        return False, None, "adapter_rejected"


def adapter_binding(raw, adapted, identity):
    """Authenticate immutable raw and canonical sidecars; no target/scorer access."""
    if not isinstance(adapted, dict) or type(adapted.get("accepted")) is not bool:
        raise ValueError("adapter_object_and_explicit_acceptance_required")
    if adapted.get("raw_response_text") != raw or adapted.get("raw_response_sha256") != sha(raw.encode()):
        raise ValueError("adapter_raw_response_binding_changed")
    if not isinstance(adapted.get("rule"), str) or not isinstance(adapted.get("reason"), str):
        raise ValueError("adapter_rule_and_reason_required")
    accepted, canonical_text, rule = adapter_expected(raw, identity["policy"]["domain"])
    if (adapted["accepted"] is not accepted or adapted.get("canonical_response_text") != canonical_text
            or adapted["rule"] != rule):
        raise ValueError("adapter_acceptance_or_value_differs_from_frozen_output_only_rule")
    if adapted["accepted"]:
        text = adapted.get("canonical_response_text")
        if not isinstance(text, str) or sha(text.encode()) != adapted.get("canonical_response_sha256"):
            raise ValueError("adapter_canonical_checksum_changed")
        value = parse(text)
        if type(value) is not dict or set(value) != {"answer"} or canonical(value) != text:
            raise ValueError("adapter_single_compact_answer_object_required")
        answer = value["answer"]
        domain = identity["policy"]["domain"]
        if not (answer is None and domain["null_allowed"] is True
                or type(answer) is int and domain["integer_minimum"] <= answer <= domain["integer_maximum"]
                or type(answer) is str and answer in domain["string_enum"]):
            raise ValueError("adapter_value_outside_frozen_domain")
    elif (adapted.get("canonical_response_text") is not None
          or adapted.get("canonical_response_sha256") is not None or adapted["rule"] != "adapter_rejected"):
        raise ValueError("adapter_rejection_must_not_be_a_null_answer")


def validate_freeze(files, protocol):
    freeze = _object(files, "main_freeze.json")
    if (freeze.get("version") != VERSION or freeze.get("study_protocol_sha256") != sha(files["study_protocol.json"])
            or freeze.get("main_stage_code_sha256") != FROZEN_CODE_SHA256["unconstrained_state_comparison_v3_colab.py"]
            or freeze.get("output_adapter_identity") != protocol["output_adapter_identity"]
            or freeze.get("decoding_identity") != DECODING or freeze.get("main_logical_rows") != 96
            or freeze.get("main_unique_payloads") != 57 or freeze.get("total_study_cap") != 57
            or freeze.get("no_new_pilot_claim") is not True
            or freeze.get("source_databases_are_live_mutable_authorities") is not True
            or freeze.get("live_database_relocation") is not False
            or set(freeze.get("sections", {})) != {"before", "after"}):
        raise ValueError("frozen_v3_grid_or_live_database_boundary_changed")
    for side, section in freeze["sections"].items():
        source, fresh = "source_prepared/" + side, "prepared/" + side
        if (section.get("source_prepared") != str(ROOT / source)
                or section.get("fresh_prepared") != str(ROOT / fresh)
                or section.get("source_gate_audit_path") != str(ROOT / "source_gate_audits" / (side + ".json"))):
            raise ValueError("frozen_source_preparation_path_changed")
        bindings = {source + "/protocol.json": "source_protocol_sha256",
                    source + "/requests.jsonl": "source_requests_sha256",
                    fresh + "/protocol.json": "fresh_protocol_sha256", fresh + "/requests.jsonl": "requests_sha256",
                    fresh + "/assessor_only/targets.json": "targets_file_sha256",
                    fresh + "/generation_cases.json": "case_file_sha256",
                    "source_gate_audits/" + side + ".json": "source_gate_audit_sha256"}
        for name, key in bindings.items():
            if name not in files or sha(files[name]) != section.get(key):
                raise ValueError("frozen_static_artifact_changed:" + name)
        prepared = _object(files, fresh + "/protocol.json")
        if (prepared.get("renderer_code_sha256") != FROZEN_CODE_SHA256["common_retrieval_policy_v1.py"]
                or prepared.get("common_policy_sha256") != digest(prepared.get("common_policy"))):
            raise ValueError("frozen_common_renderer_policy_identity_changed")
        for key in ("renderer_code_sha256", "common_policy_sha256", "request_mapping_sha256"):
            if prepared.get(key) != section.get(key):
                raise ValueError("common_source_policy_mapping_changed")
        if prepared.get("request_mapping_sha256") != digest(prepared.get("request_mappings")):
            raise ValueError("ordered_request_mapping_checksum_changed")
        if prepared.get("source_snapshot_sha256") != digest(prepared.get("source_files_sha256")):
            raise ValueError("static_source_snapshot_checksum_changed")
        for group, hashes in ((source, prepared.get("source_files_sha256")),
                              (fresh, prepared.get("prepared_files_sha256"))):
            if not isinstance(hashes, dict) or not hashes:
                raise ValueError("nonempty_static_source_manifest_required")
            for relative, expected in hashes.items():
                safe_member(relative)
                # Live DBs/control anchors deliberately are not frozen static files.
                if (Path(relative).suffix == ".sqlite" or relative.endswith(".sqlite.control.json")
                        or ".cbt-memory-control" in PurePosixPath(relative).parts):
                    raise ValueError("live_database_or_control_in_static_manifest")
                name = group + "/" + relative
                if name not in files or sha(files[name]) != expected:
                    raise ValueError("static_preparation_checksum_changed:" + name)
        rows = plain_rows(files[fresh + "/requests.jsonl"])
        dbs = {study_relative(row["database_path"]) for row in rows}
        if len(dbs) != 24 or any(not db.startswith(source + "/") or not db.endswith(".sqlite") or db not in files for db in dbs):
            raise ValueError("authoritative_source_database_snapshot_missing_or_relocated")
        audit = _object(files, "source_gate_audits/" + side + ".json")
        expected = {"version": VERSION + "-source-gate-audit", "side": side,
                    "study_protocol_sha256": sha(files["study_protocol.json"]),
                    "main_stage_code_sha256": freeze["main_stage_code_sha256"],
                    "all_preflight_passed": True, "checked_row_count": 48,
                    "new_model_calls": 0, "generation_admissions": 0, "automatic_retry": False}
        expected.update({key: section[key] for key in ("checkout", "fresh_prepared", "fresh_protocol_sha256",
            "requests_sha256", "source_code_identity", "new_database_count", "source_prepared",
            "source_protocol_sha256", "source_requests_sha256", "renderer_code_sha256",
            "common_policy_sha256", "request_mapping_sha256")})
        checks = audit.get("ordered_checks", [])
        if (any(audit.get(key) != value for key, value in expected.items()) or len(checks) != 48
                or [{k: row.get(k) for k in ("case_id", "arm", "request_sha256")} for row in checks]
                   != [{k: row.get(k) for k in ("case_id", "arm", "request_sha256")} for row in rows]
                or any(not re.fullmatch(r"[0-9a-f]{64}", row.get("source_binding_sha256", "")) for row in checks)):
            raise ValueError("source_gate_audit_binding_changed")
    return freeze


def inspect_state(files):
    state = {"stage": "live_partial", "terminal": False, "all_intents_published": False,
             "new_attempts": None, "published_outcomes": None, "issues": [],
             "maximum_new_study_attempts": 57, "new_pilot_attempts": 0,
             "automatic_resume_or_resend_authorized": False,
             "adapter_authentication_scope": "Raw/canonical checksums, acceptance, canonical typed value and accepted-form rule are independently authenticated from raw text only. Original adapter failure reasons and target scores are preserved without importing or re-executing the frozen adapter or scorer."}
    try:
        protocol = _object(files, "study_protocol.json")
        if (sha(files["study_protocol.json"]) != STUDY_PROTOCOL_SHA256
                or protocol.get("version") != VERSION or protocol.get("study_root") != str(ROOT)
                or protocol.get("maximum_new_study_attempts") != 57 or protocol.get("planned_pilot_attempts") != 0
                or protocol.get("planned_main_unique_attempts") != 57 or protocol.get("planned_main_logical_rows") != 96
                or protocol.get("failed_v2_gate_unchanged_and_closed") is not True
                or protocol.get("decoding_policy") != DECODING):
            raise ValueError("frozen_v3_protocol_or_cap_changed")
        adapter = protocol["output_adapter_identity"]
        if (adapter.get("code_sha256") != FROZEN_CODE_SHA256["common_answer_adapter_v3.py"]
                or adapter.get("policy_sha256") != digest(adapter.get("policy"))):
            raise ValueError("output_adapter_identity_changed")
        intents, outcomes, caches = {}, {}, {}
        for row in checked_rows(files.get("generation_ledger.jsonl"), optional=True):
            kind, key = row.get("event"), row.get("canonical_key")
            target = intents if kind == "generation_intent" else outcomes if kind == "generation_outcome" else None
            if target is None or not isinstance(key, str) or key in target:
                raise ValueError("unknown_or_duplicate_study_event")
            target[key] = row
        state.update(new_attempts=len(intents), published_outcomes=sum(r.get("status") == "published" for r in outcomes.values()))
        if (len(intents) > 57 or set(intents) != set(outcomes)
                or any(type(row.get("attempt_number")) is not int for row in intents.values())
                or sorted(row["attempt_number"] for row in intents.values()) != list(range(1, len(intents) + 1))
                or any(row.get("status") != "published" for row in outcomes.values())):
            raise ValueError("unsettled_failed_or_over_budget_journal_no_retry")
        for key, row in outcomes.items():
            relative = study_relative(row.get("cache_path", ""))
            owner = study_relative(intents[key].get("output", ""))
            if owner not in ("answers_before", "answers_after") or relative != owner + "/canonical/" + key + ".json":
                raise ValueError("foreign_or_unbound_raw_cache")
            if relative not in files or sha(files[relative]) != row.get("cache_sha256"):
                raise ValueError("raw_cache_missing_or_checksum_changed")
            cache = _object(files, relative)
            text, usage = cache.get("response_text"), cache.get("usage")
            if (cache.get("canonical_key") != key or not isinstance(text, str) or not isinstance(usage, dict)
                    or sha(text.encode()) != row.get("response_sha256")
                    or any(type(usage.get(k)) is not int or usage[k] < 0 for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
                    or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]
                    or usage["prompt_tokens"] > 4096 or usage["completion_tokens"] > 64
                    or any(cache.get(k) != row.get(k) for k in ("response_sha256", "usage", "finish_reason"))
                    or not finite_time(row.get("generation_wall_seconds"))):
                raise ValueError("raw_response_usage_or_timing_binding_changed")
            owner_run = _object(files, owner + "/run_protocol.json")
            if intents[key].get("run_protocol_sha256") != digest(owner_run):
                raise ValueError("canonical_owner_run_checksum_changed")
            caches[key] = cache
        setups, completed = {}, {}
        events = checked_rows(files.get("model_load_events.jsonl"), optional=True)
        for event in events:
            target = setups if event.get("event") == "model_reload_intent" else completed if event.get("event") == "model_reload_completed" else None
            key = event.get("setup_id")
            if target is None or not isinstance(key, str) or key in target:
                raise ValueError("failed_unknown_or_duplicate_model_setup_event")
            target[key] = event
        if set(setups) != set(completed) or intents and not completed:
            raise ValueError("unsettled_or_missing_new_study_model_setup")
        for event in completed.values():
            identity = event["model_identity"]
            if not finite_time(event.get("setup_seconds")):
                raise ValueError("new_setup_timing_required")
            for name, key in (("load_protocol.json", "load_protocol_sha256"), ("model_files.json", "model_files_sha256"), ("model_load_receipt.json", "load_receipt_sha256")):
                if identity.get(key) != sha(files[name]):
                    raise ValueError("new_setup_model_metadata_changed")
        state.update(all_intents_published=True, completed_new_model_setups=len(completed), stage="settled_incomplete_study")
        if len(intents) != 57:
            return state
        freeze = validate_freeze(files, protocol)
        all_answers = []
        for side in ("before", "after"):
            section, directory = freeze["sections"][side], "answers_" + side
            rows = plain_rows(files["prepared/" + side + "/requests.jsonl"])
            answers = checked_rows(files.get(directory + "/answers_unscored.jsonl"))
            run, summary = _object(files, directory + "/run_protocol.json"), _object(files, directory + "/completion_summary.json")
            metadata = {"side": side, "main_freeze_sha256": sha(files["main_freeze.json"]),
                        "fresh_prepared_protocol_sha256": section["fresh_protocol_sha256"],
                        **{k: section[k] for k in ("renderer_code_sha256", "common_policy_sha256", "request_mapping_sha256")},
                        "all_cases_exposed_development_only": True}
            if (len(rows) != 48 or len(answers) != 48 or len({(r["case_id"], r["arm"]) for r in rows}) != 48
                    or len(run.get("actual_tensor_preflight", [])) != 48 or run.get("rows_sha256") != digest(rows)
                    or run.get("study_protocol_sha256") != sha(files["study_protocol.json"])
                    or run.get("code_sha256") != protocol["code_sha256"] or run.get("decoding_identity") != DECODING
                    or not any(run.get("model") == event["model_identity"] for event in completed.values())
                    or run.get("output_adapter_identity") != adapter or run.get("metadata") != metadata
                    or run.get("gate_identity") != section["source_code_identity"]
                    or summary.get("complete") is not True or summary.get("logical_completed") != 48
                    or summary.get("logical_planned") != 48 or summary.get("answers_sha256") != sha(files[directory + "/answers_unscored.jsonl"])
                    or summary.get("run_protocol_sha256") != digest(run)):
                raise ValueError("incomplete_or_unbound_48_row_side:" + side)
            for wanted, answer, preflight in zip(rows, answers, run["actual_tensor_preflight"]):
                key = digest({"new_study_protocol_sha256": sha(files["study_protocol.json"]), "model": run["model"],
                              "decoding": DECODING, "payload": wanted["request_payload"], "max_new_tokens": 64, "do_sample": False})
                if (key not in caches or answer.get("canonical_key") != key or answer.get("case_id") != wanted["case_id"]
                        or wanted["request_payload"].get("max_tokens") != 64
                        or len(json.dumps(wanted["request_payload"], ensure_ascii=True, sort_keys=True, allow_nan=False)) > 6000
                        or answer.get("arm") != wanted["arm"] or wanted["arm"] not in ARMS
                        or answer.get("request_sha256") != wanted["request_sha256"]
                        or wanted["request_sha256"] != digest(wanted["request_payload"])
                        or intents[key].get("request_sha256") != wanted["request_sha256"]
                        or answer.get("run_protocol_sha256") != digest(run)
                        or answer.get("status") != "completed" or answer.get("source_checks_passed") is not True
                        or any(answer.get(k) != v for k, v in caches[key].items())
                        or answer.get("input_tensor_sha256") != preflight.get("input_tensor_sha256")
                        or (answer.get("usage") or {}).get("prompt_tokens") != preflight.get("input_tokens")
                        or not finite_time(answer.get("output_adaptation_wall_seconds"))):
                    raise ValueError("logical_raw_answer_source_input_or_cache_binding_changed")
                adapter_binding(answer["response_text"], answer.get("answer_adapter"), adapter)
            report = _object(files, "reports/" + side + "_metrics.json")
            if (report.get("answers_sha256") != sha(files[directory + "/answers_unscored.jsonl"])
                    or report.get("main_freeze_sha256") != sha(files["main_freeze.json"])
                    or report.get("output_adapter_identity") != adapter):
                raise ValueError("saved_metric_report_binding_changed")
            all_answers.extend(answers)
        if {a["canonical_key"] for a in all_answers} != set(intents):
            raise ValueError("logical_grid_does_not_cover_all_57_intents")
        if (sum(row.get("output") == str(ROOT / "answers_before") for row in intents.values()) != 48
                or sum(row.get("output") == str(ROOT / "answers_after") for row in intents.values()) != 9):
            raise ValueError("fixed_before48_after9_unique_admission_order_changed")
        summary, resources = _object(files, "reports/summary.json"), _object(files, "reports/resources.json")
        if any(summary.get(k) != v for k, v in {"version": VERSION, "main_logical_rows": 96, "main_unique_generations": 57,
                                               "pilot_generations": 0, "new_study_unique_generations": 57, "study_cap": 57}.items()):
            raise ValueError("saved_final_summary_fixed_counts_changed")
        usage = {k: sum(row["usage"][k] for row in outcomes.values()) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
        wall, adaptation = sum(r["generation_wall_seconds"] for r in outcomes.values()), sum(a["output_adaptation_wall_seconds"] for a in all_answers)
        cost = resources["new_study_unique_generation_cost"]
        if (cost.get("observed_tokens") != usage or cost.get("unique_admitted_attempts") != 57
                or not math.isclose(cost.get("answer_path_wall_seconds_sum", -1), wall, rel_tol=1e-12, abs_tol=1e-9)
                or not math.isclose(resources.get("output_adaptation_wall_seconds_sum", -1), adaptation, rel_tol=1e-12, abs_tol=1e-9)
                or resources.get("actual_new_model_setup_events") != events):
            raise ValueError("saved_resource_cost_binding_changed")
        state.update(stage="completed_study_settled", terminal=True, logical_answers=96,
                     observed_tokens=usage, unique_answer_path_wall_seconds_sum=wall,
                     logical_output_adaptation_wall_seconds_sum=adaptation,
                     new_model_setup_seconds_sum=sum(e["setup_seconds"] for e in completed.values()))
        return state
    except (ValueError, TypeError, KeyError, UnicodeError, AttributeError, OverflowError) as error:
        state.update(stage="live_partial", terminal=False)
        state["issues"].append(str(error))
        return state


def reference_identity(files):
    protocol = _object(files, "study_protocol.json")
    inherited = {"load_protocol.json", "model_files.json", "model_load_receipt.json"}
    if set(protocol.get("inherited_identity_sha256", {})) != inherited:
        raise ValueError("exact_three_inherited_model_identity_files_required")
    for name, expected in protocol["inherited_identity_sha256"].items():
        if sha(files[name]) != expected:
            raise ValueError("inherited_model_identity_changed")
    history = protocol.get("historical_reference_sha256", {})
    required = {str(OLD_ROOT / name) for name in inherited | {"generation_ledger.jsonl"}} | {str(FAILED_V2_ROOT / name) for name in V2_REFERENCES}
    if set(history) != required:
        raise ValueError("exact_thirteen_old64_and_failed_v2_references_required")
    evidence, code_payloads = [], {}
    for source, expected in history.items():
        if sha(read_bytes(content_path(source))) != expected:
            raise ValueError("historical_reference_changed")
        evidence.append({"path": source, "sha256": expected, "copied": False})
    codes = protocol.get("code_sha256", {})
    if (len(codes) != 5 or {Path(p).name: value for p, value in codes.items()} != FROZEN_CODE_SHA256
            or len({str(Path(p).parent) for p in codes}) != 1):
        raise ValueError("exact_five_pinned_v3_runtime_code_files_required")
    for source, expected in codes.items():
        path = content_path(source)
        if path.parent.parts[-2:] != ("evaluation", "cbt-memory-effect"):
            raise ValueError("unlisted_frozen_code_location")
        data = read_bytes(path)
        if sha(data) != expected:
            raise ValueError("frozen_v3_code_changed")
        code_payloads[path.name] = data
        evidence.append({"path": source, "sha256": expected, "copied": True, "archive_path": "frozen_code/" + path.name})
    return evidence, code_payloads

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


def export(archive, receipt):
    guard()
    archive, receipt = output_path(archive, ".zip"), output_path(receipt, ".json")
    for path in (archive, receipt):
        if os.path.lexists(path):
            raise FileExistsError("preserve_previous_export_use_new_paths:" + str(path))
    members = discover(ROOT)
    if sum(path.stat().st_size for _, path in members) > MAX_BYTES:
        raise ValueError("study_uncompressed_size_bound_exceeded")
    started = datetime.now(timezone.utc).isoformat()
    files, records, originals, missing, lock_status = {}, [], {}, [], {}
    captured_bytes = 0
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
        with tempfile.TemporaryDirectory(prefix="cbt-unconstrained-v3-export-", dir=CONTENT) as temporary:
            for name, source in members:
                if source.suffix == ".sqlite":
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
                captured_bytes = bounded_size(captured_bytes, len(data))
                files[name] = data
                records.append({"path": member, "source_path": str(source), "bytes": len(data),
                                "sha256": sha(data), "method": method,
                                "live_mutable_authority_snapshot": source.suffix == ".sqlite"})
        payloads = {"study/" + name: data for name, data in files.items()}
        for source, expected in ((RESTORE_PATH, RESTORE_SHA), (REACQUISITION_PATH, REACQUISITION_SHA)):
            data = read_bytes(source)
            if sha(data) != expected:
                raise ValueError("pinned_recovery_evidence_changed:" + str(source))
            member = "recovery/" + source.name
            safe_member(member)
            captured_bytes = bounded_size(captured_bytes, len(data))
            payloads[member] = data
            originals[str(source)] = sha(data)
            records.append({"path": member, "source_path": str(source), "bytes": len(data),
                            "sha256": sha(data), "method": "exact_bytes_read_during_capture"})
        for source in OPTIONAL_WORKER_ARTIFACTS:
            data = read_bytes(source, missing=True)
            if data is None:
                missing.append(str(source))
                continue
            member = "worker_evidence/" + source.name
            safe_member(member, recovery=source.suffix == ".py")
            captured_bytes = bounded_size(captured_bytes, len(data))
            payloads[member] = data
            originals[str(source)] = sha(data)
            records.append({"path": member, "source_path": str(source), "bytes": len(data),
                            "sha256": sha(data), "method": "exact_bytes_read_during_capture"})
        state = inspect_state(files)
        try:
            evidence, codes = reference_identity(files)
        except (ValueError, TypeError, KeyError, OSError, AttributeError) as error:
            evidence, codes = [], {}
            state.update(stage="live_partial", terminal=False)
            state["issues"].append("identity_validation:" + str(error))
        for item in evidence:
            originals[item["path"]] = item["sha256"]
        for name, data in codes.items():
            member = "frozen_code/" + name
            safe_member(member, recovery=True)
            captured_bytes = bounded_size(captured_bytes, len(data))
            payloads[member] = data
            source = next(item["path"] for item in evidence if item.get("archive_path") == member)
            records.append({"path": member, "source_path": source, "bytes": len(data), "sha256": sha(data),
                            "method": "exact_bytes_of_pinned_code_not_imported_or_executed"})
        total = sum(len(data) for data in payloads.values())
        if total > MAX_BYTES or len(payloads) > MAX_FILES:
            raise ValueError("combined_artifact_count_or_size_bound_exceeded")
        changed = []
        for source, expected in originals.items():
            try:
                data = read_bytes(source, missing=True)
            except OSError:
                data = None
            if data is None or sha(data) != expected:
                changed.append(source)
        membership_changed = {name for name, _ in discover(ROOT)} != {name for name, _ in members}
        state = capture_observations(state, lock_status, changed, membership_changed)
        manifest = {
            "version": "unconstrained-state-v3-private-checkpoint-v1", "study_root": str(ROOT),
            "study_protocol_sha256": STUDY_PROTOCOL_SHA256, "frozen_helper_commit": FROZEN_HELPER_COMMIT,
            "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_state": state, "files": sorted(records, key=lambda row: row["path"]),
            "uncompressed_artifact_bytes": total, "prior_archive_references": PRIOR_ARCHIVES,
            "restore_receipt_sha256": RESTORE_SHA, "model_reacquisition_receipt_sha256": REACQUISITION_SHA,
            "verified_frozen_code_and_historical_references": evidence,
            "missing_optional_worker_artifacts": missing, "existing_runtime_lock_observations": lock_status,
            "changed_source_files_during_capture": changed, "study_file_membership_changed_during_capture": membership_changed,
            "collection_is_not_atomic_across_files": True, "sqlite_backups_individually_consistent": True,
            "live_database_boundary": "Source-prepared SQLite databases and controls are live mutable authorities at their original paths. Static source hashes exclude those authorities; archive SQLite hashes identify individually consistent backup bytes, not original live database bytes or a relocatable continuation.",
            "sqlite_wal_shm_journal_sidecars_not_copied_separately": True,
            "raw_model_outputs_preserved_without_repair": True,
            "inherited_setup_receipts_are_nonadditive_historical_references": True,
            "old64_and_failed_v2_four_costs_not_added_to_v3": True,
            "new_model_calls": 0, "new_model_loads": 0, "old_archives_copied": False,
            "weights_credentials_repository_configuration_included": False,
            "automatic_resume_or_resend_authorized": False, "exporter_code_sha256": sha(Path(__file__).read_bytes()),
        }
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
        data = buffer.getvalue()
        result = {"version": manifest["version"], "archive": str(archive), "bytes": len(data), "sha256": sha(data),
                  "manifest_sha256": sha(manifest_data), "artifact_files": len(records), "checkpoint_state": state,
                  "new_model_calls": 0, "new_model_loads": 0, "automatic_resume_or_resend_authorized": False}
        write_new(archive, data)
        write_new(receipt, (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.archive, args.receipt), sort_keys=True))
