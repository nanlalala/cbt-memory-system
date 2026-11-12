"""Restore the two pinned private checkpoints in genuine hosted Colab only.

This is an artifact restore, with zero model calls, downloads, package installs,
or automatic retries. Every archive byte, mapping and existing destination is
checked before creating anything. Historical load receipts are not new loads.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile


CONTENT = Path("/content")
MAX_BYTES = 256_000_000
MAX_FILE_BYTES = 16_000_000
BASE_SHA256 = "8cee7b783915c8da05f428c2c45ab5e0e1f8c9507a21c7946ebcf317f5887e33"
PILOT_SHA256 = "6f67748b220724bb4b510092e229c42e262b458ef67896fd196633e0757174c1"
BASE_BYTES = 6_277_389
PILOT_BYTES = 30_569
ROOTS = {
    "discarded_prepared_before_v1": "cbt-open-state-prepared-before-20261112-v1",
    "discarded_prepared_after_v1": "cbt-open-state-prepared-after-20261112-v1",
    "prepared_before": "cbt-open-state-prepared-before-20261112-v2",
    "prepared_after": "cbt-open-state-prepared-after-20261112-v2",
    "answers_before": "cbt-open-state-answers-before-20261112-v2",
    "answers_after": "cbt-open-state-answers-after-20261112-v2",
    "equip": "cbt-equip-professional-sourcebank-20261112-v1",
    "therapyjudge": "therapyjudgebench-v2-source-freeze-20261112",
    "calibration_prepared": "therapyjudge-five-calibration-prepared-v1",
    "calibration_answers": "therapyjudge-five-calibration-results-v1",
    "model_metadata": "cbt-open-reader-20261112-v1",
}
MODEL_METADATA = {"load_protocol.json", "model_files.json", "model_load_receipt.json",
                  "model_load_events.jsonl", "generation_ledger.jsonl"}
REPORTS = {
    "cbt-open-state-before-after-freeze-20261112-v2.json",
    "cbt-open-state-final-source-fence-after-20261112-v2.json",
    "cbt-open-state-final-source-fence-before-20261112-v2.json",
    "cbt-open-state-primary-score-after-20261112-v2.json",
    "cbt-open-state-primary-score-before-20261112-v2.json",
    "cbt-open-state-resource-audit-57-20261112-v2.json",
    "cbt-open-state-resource-audit-62-20261112-v2.json",
    "cbt-open-state-secondary-format-after-20261112-v2.json",
    "cbt-open-state-secondary-format-before-20261112-v2.json",
    "cbt-open-state-reader-after-20261112-v2.log",
    "cbt-open-state-reader-before-20261112-v2.log",
    "therapyjudge-five-calibration-token-preflight-v1.json",
    "therapyjudge-five-calibration-token-preflight-v2.json",
    "cbt-open-state-v1-abort-before-generation-20261112.json",
}
EXTENSIONS = {".json", ".jsonl", ".sqlite", ".pdf", ".html", ".md", ".log"}
SUPERSEDED = {"model_metadata/generation_ledger.jsonl", "model_metadata/model_load_events.jsonl"}
PILOT_ROOTS = {"prepared": "cbt-constrained-format-pilot-prepared-20261112-v1",
               "answers": "cbt-constrained-format-pilot-results-20261112-v1",
               "model_metadata": ROOTS["model_metadata"]}
TOKEN_AUDIT = "cbt-constrained-tokenization-audit-20261112-v1.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def parse_json(data):
    return json.loads(data, object_pairs_hook=_unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))


def require_hosted_colab():
    # Check before reading archives, destinations, or creating any paths.
    try:
        valid = (bool(os.environ.get("COLAB_RELEASE_TAG")) and CONTENT.is_dir()
                 and not CONTENT.is_symlink()
                 and importlib.util.find_spec("google.colab") is not None)
    except (ImportError, ValueError, AttributeError):
        valid = False
    if not valid:
        raise RuntimeError("genuine_hosted_colab_required_before_any_io")


def member_path(name):
    if (not isinstance(name, str) or not name or "\\" in name or "\x00" in name
            or name.startswith("/") or any(p in ("", ".", "..") for p in name.split("/"))):
        raise ValueError("unsafe_archive_member")
    path = PurePosixPath(name)
    forbidden = {".git", ".github", ".ssh", ".config", ".cache", ".env", "credentials", "secrets"}
    if (any(p.lower() in forbidden for p in path.parts)
            or (path.suffix not in EXTENSIONS and path.name != "LICENSE")):
        raise ValueError("unsupported_or_secret_or_weight_member")
    return path


def content_path(value):
    raw = str(value)
    if "\\" in raw or "\x00" in raw or not raw.startswith("/content/"):
        raise ValueError("absolute_hosted_content_path_required")
    relative = raw[len("/content/"):]
    if any(p in ("", ".", "..") for p in relative.split("/")):
        raise ValueError("unsafe_content_path")
    return CONTENT / relative


def receipt_path(value):
    path = content_path(value)
    relative = path.relative_to(CONTENT)
    if (len(relative.parts) != 1
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.json", relative.name)):
        raise ValueError("receipt_must_be_a_new_direct_content_json_file")
    return path


def _open_parent(path, *, create=False):
    """Walk from /content with no symlink following, including during writes."""
    relative = content_path(path).relative_to(CONTENT)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open(CONTENT, flags)
    try:
        for part in relative.parts[:-1]:
            try:
                following = os.open(part, flags, dir_fd=fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(part, 0o700, dir_fd=fd)
                except FileExistsError:
                    pass
                following = os.open(part, flags, dir_fd=fd)
            os.close(fd)
            fd = following
        return fd, relative.name
    except BaseException:
        os.close(fd)
        raise


def read_regular(path, *, limit=MAX_FILE_BYTES):
    fd, name = _open_parent(path)
    try:
        item = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
        try:
            info = os.fstat(item)
            if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
                raise ValueError("bounded_regular_file_required")
            with os.fdopen(item, "rb", closefd=False) as stream:
                data = stream.read(limit + 1)
            if len(data) > limit:
                raise ValueError("file_exceeds_limit")
            return data
        finally:
            os.close(item)
    finally:
        os.close(fd)


def read_archive(path, *, expected_sha, expected_bytes, manifest_name, expected_count):
    archive = read_regular(path, limit=MAX_BYTES)
    if len(archive) != expected_bytes or sha(archive) != expected_sha:
        raise ValueError("pinned_archive_bytes_or_sha256_changed")
    # The verified bytes are opened, never a second unverified read of the path.
    from io import BytesIO
    with zipfile.ZipFile(BytesIO(archive)) as zipped:
        infos = zipped.infolist()
        if len(infos) != expected_count + 1 or len({i.filename for i in infos}) != len(infos):
            raise ValueError("duplicate_or_unexpected_archive_entry_count")
        total = 0
        for info in infos:
            member_path(info.filename)
            mode = info.external_attr >> 16
            if (info.is_dir() or stat.S_IFMT(mode) not in (0, stat.S_IFREG)
                    or info.flag_bits & 1 or info.file_size > MAX_FILE_BYTES):
                raise ValueError("only_bounded_unencrypted_regular_zip_members_allowed")
            total += info.file_size
        if total > MAX_BYTES:
            raise ValueError("uncompressed_archive_limit_exceeded")
        manifest = parse_json(zipped.read(manifest_name))
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != expected_count:
            raise ValueError("manifest_file_count_changed")
        records, payloads = {}, {}
        for row in files:
            if not isinstance(row, dict):
                raise ValueError("manifest_record_required")
            name = row.get("path")
            member_path(name)
            if name in records or name == manifest_name:
                raise ValueError("duplicate_or_self_listed_manifest_member")
            if type(row.get("bytes")) is not int or not 0 <= row["bytes"] <= MAX_FILE_BYTES:
                raise ValueError("manifest_byte_count_invalid")
            data = zipped.read(name)
            if len(data) != row["bytes"] or sha(data) != row.get("sha256"):
                raise ValueError("manifest_artifact_size_or_sha256_changed")
            records[name], payloads[name] = row, data
        if set(zipped.namelist()) != set(records) | {manifest_name}:
            raise ValueError("unmanifested_archive_member")
        return manifest, records, payloads, total


def destination_for(name, *, pilot=False):
    member = member_path(name)
    if pilot and name == "tokenization_only_audit.json":
        return CONTENT / TOKEN_AUDIT
    parts = member.parts
    if len(parts) < 2:
        raise ValueError("named_artifact_root_required")
    group, rest = parts[0], parts[1:]
    roots = PILOT_ROOTS if pilot else ROOTS
    if group == "reports" and not pilot:
        if len(rest) != 1 or rest[0] not in REPORTS:
            raise ValueError("unlisted_report")
        return CONTENT / rest[0]
    if group not in roots:
        raise ValueError("unlisted_artifact_root")
    if group == "model_metadata" and (len(rest) != 1 or rest[0] not in MODEL_METADATA
                                      or (pilot and name not in SUPERSEDED)):
        raise ValueError("unlisted_model_metadata_no_weights_allowed")
    return CONTENT / roots[group] / Path(*rest)


def checked_rows(data):
    if not data or not data.endswith(b"\n"):
        raise ValueError("complete_nonempty_jsonl_required")
    rows = []
    for line in data.splitlines():
        if not line.strip():
            raise ValueError("blank_jsonl_record")
        row = parse_json(line)
        if not isinstance(row, dict):
            raise ValueError("journal_object_required")
        body = {k: v for k, v in row.items() if k != "record_sha256"}
        if sha(canonical(body).encode()) != row.get("record_sha256"):
            raise ValueError("journal_record_checksum_changed")
        rows.append(row)
    return rows


def validate_extension(base, extended, *, base_count, final_count):
    old, new = checked_rows(base), checked_rows(extended)
    if (len(old) != base_count or len(new) != final_count
            or not extended.startswith(base) or new[:len(old)] != old):
        raise ValueError("supplement_is_not_exact_append_only_extension")
    return new


def validate_ledger(rows, expected_attempts, plan):
    intents, outcomes = {}, {}
    for row in rows:
        event, key = row.get("event"), row.get("canonical_key")
        target = intents if event == "generation_intent" else outcomes if event == "generation_outcome" else None
        if target is None or not isinstance(key, str) or key in target:
            raise ValueError("unknown_or_duplicate_generation_record")
        target[key] = row
    if (len(intents) != expected_attempts or set(intents) != set(outcomes)
            or sorted(r.get("attempt_number", -1) for r in intents.values()) != list(range(1, expected_attempts + 1))):
        raise ValueError("generation_attempts_not_complete_and_bound")
    for key, row in outcomes.items():
        cache_path = str(content_path(row.get("cache_path", "")))
        cache_location = Path(cache_path)
        if cache_location.parent.name != "canonical" or cache_location.name != key + ".json":
            raise ValueError("unexpected_terminal_cache_path")
        if row.get("status") != "published" or cache_path not in plan:
            raise ValueError("unsettled_generation_or_missing_terminal_cache")
        data = plan[cache_path]["data"]
        cache = parse_json(data)
        if sha(data) != row.get("cache_sha256") or cache.get("canonical_key") != key:
            raise ValueError("terminal_cache_hash_or_identity_changed")
        response = cache.get("response_text")
        if not isinstance(response, str) or sha(response.encode()) != cache.get("response_sha256"):
            raise ValueError("terminal_response_checksum_changed")
        if any(cache.get(k) != row.get(k) for k in ("response_sha256", "usage", "finish_reason")):
            raise ValueError("terminal_response_receipt_changed")


def make_plan(base_manifest, base_records, base_data, pilot_manifest, pilot_records, pilot_data):
    if (base_manifest.get("stage") != "completed_memory_audit"
            or base_manifest.get("weights_credentials_repository_configuration_included") is not False
            or pilot_manifest.get("no_secrets_or_weights") is not True
            or pilot_manifest.get("global_attempts") != 64
            or pilot_manifest.get("original_complete_62_archive_sha256") != BASE_SHA256):
        raise ValueError("checkpoint_scope_or_original_binding_changed")
    plan, extensions = {}, []
    for pilot, records, payloads in ((False, base_records, base_data), (True, pilot_records, pilot_data)):
        for name, row in records.items():
            target = str(destination_for(name, pilot=pilot))
            if not pilot:
                if row.get("source_path") != target or row.get("method") not in {
                    "exact_bytes_read_during_capture", "consistent_sqlite_backup_from_read_only_connection"
                }:
                    raise ValueError("original_source_mapping_or_copy_method_changed")
            if target in plan:
                if not pilot or name not in SUPERSEDED:
                    raise ValueError("unexpected_destination_collision")
                counts = (124, 128) if name.endswith("generation_ledger.jsonl") else (6, 8)
                rows = validate_extension(plan[target]["data"], payloads[name],
                                          base_count=counts[0], final_count=counts[1])
                extensions.append({"path": target, "original_bytes": len(plan[target]["data"]),
                                   "original_sha256": sha(plan[target]["data"]),
                                   "restored_bytes": len(payloads[name]), "restored_sha256": sha(payloads[name]),
                                   "original_records": counts[0], "restored_records": len(rows),
                                   "exact_byte_prefix_verified": True})
            plan[target] = {"data": payloads[name], "record": row, "archive": "pilot" if pilot else "base"}
    if len(extensions) != 2:
        raise ValueError("both_documented_metadata_extensions_required")
    ledger = str(CONTENT / ROOTS["model_metadata"] / "generation_ledger.jsonl")
    validate_ledger(checked_rows(plan[ledger]["data"]), 64, plan)
    return plan, extensions


def preflight_existing(plan, receipt):
    receipt = receipt_path(receipt)
    if str(receipt) in plan:
        raise ValueError("receipt_must_be_separate_from_restored_artifacts")
    for target, item in plan.items():
        try:
            current = read_regular(target)
        except FileNotFoundError:
            item["already_present"] = False
        else:
            if current != item["data"]:
                raise FileExistsError("existing_artifact_mismatch_preserved:" + target)
            item["already_present"] = True
    try:
        fd, name = _open_parent(receipt)
    except FileNotFoundError:
        return
    try:
        try:
            os.stat(name, dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise FileExistsError("receipt_exists_use_a_new_path")
    finally:
        os.close(fd)


def write_new(path, data):
    parent, name = _open_parent(path, create=True)
    try:
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        try:
            with os.fdopen(fd, "wb", closefd=False) as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(fd)
    finally:
        os.close(parent)


def restore(base_archive, pilot_archive, receipt):
    require_hosted_colab()
    base_archive, pilot_archive = map(content_path, (base_archive, pilot_archive))
    receipt = receipt_path(receipt)
    if len({str(base_archive), str(pilot_archive), str(receipt)}) != 3:
        raise ValueError("distinct_archive_and_receipt_paths_required")
    base = read_archive(base_archive, expected_sha=BASE_SHA256, expected_bytes=BASE_BYTES,
                        manifest_name="checkpoint_manifest.json", expected_count=522)
    pilot = read_archive(pilot_archive, expected_sha=PILOT_SHA256, expected_bytes=PILOT_BYTES,
                         manifest_name="manifest.json", expected_count=13)
    if base[3] + pilot[3] > MAX_BYTES:
        raise ValueError("combined_uncompressed_limit_exceeded")
    plan, extensions = make_plan(*base[:3], *pilot[:3])
    if any(str(p) in plan for p in (base_archive, pilot_archive)):
        raise ValueError("input_archive_destination_collision")
    preflight_existing(plan, receipt)
    # No writes occur above this point. Existing differing artifacts always fail.
    restored = skipped = 0
    for target, item in sorted(plan.items()):
        if item["already_present"]:
            skipped += 1
        else:
            write_new(target, item["data"])
            restored += 1
    for target, item in plan.items():
        if read_regular(target) != item["data"]:
            raise RuntimeError("post_restore_artifact_verification_failed")
    record = {
        "version": "cbt-hosted-colab-pinned-checkpoint-restore-v1",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_archive": {"path": str(base_archive), "sha256": BASE_SHA256, "bytes": BASE_BYTES},
        "pilot_archive": {"path": str(pilot_archive), "sha256": PILOT_SHA256, "bytes": PILOT_BYTES},
        "original_checkpoint_manifest": base[0], "pilot_checkpoint_manifest": pilot[0],
        "append_only_metadata_extensions": extensions,
        "tokenization_audit_mapping": {"archive_path": "tokenization_only_audit.json",
                                      "restored_path": str(CONTENT / TOKEN_AUDIT),
                                      "mapping_provenance": "original path recorded in this session; not specified by pilot manifest"},
        "restored_artifacts": restored, "already_identical_artifacts": skipped,
        "verified_destination_count": len(plan), "all_destinations_verified_after_restore": True,
        "sqlite_provenance": "Manifest bytes are individually consistent SQLite snapshots. Original live database byte identity is not asserted.",
        "historical_model_metadata_only": True, "new_model_load_receipt_created": False,
        "weights_restored": False, "new_model_calls": 0,
        "automatic_resume_or_resend_authorized": False,
        "code_checkouts_restored": False,
    }
    write_new(receipt, (json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())
    return {"receipt": str(receipt), "receipt_sha256": sha(read_regular(receipt)),
            "verified_destination_count": len(plan), "restored_artifacts": restored,
            "already_identical_artifacts": skipped, "global_completed_attempts": 64,
            "new_model_calls": 0, "new_model_load_receipt_created": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-archive", required=True)
    parser.add_argument("--pilot-archive", required=True)
    parser.add_argument("--receipt", required=True, help="A new /content/*.json path; never overwritten")
    args = parser.parse_args()
    print(json.dumps(restore(args.base_archive, args.pilot_archive, args.receipt), sort_keys=True))
