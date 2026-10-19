#!/usr/bin/env python3
"""Verify and recompute the historical pilot offline; never execute archived code.

Checksums establish byte consistency with the supplied manifest, not authenticity,
semantic correctness, clinical benefit or a verified provider/model identity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import stat
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_three_arm import ARMS, MANIFEST, stable_digest  # noqa: E402
from score_three_arm import answer_proxies, score_run  # noqa: E402
from benchmark_adapters import load_locomo  # noqa: E402


DEFAULT_MANIFEST = HERE / "pilot_20261019_audit.json"
PUBLIC_CSV = "pilot_20261019_scores.csv"
PUBLIC_SUMMARY = "pilot_20261019_summary.json"
CSV_COLUMNS = ("example_id", "arm", "category", "exact_match_proxy",
               "token_f1_proxy", "latency_seconds", "total_tokens")
RAW_FILES = ("protocol.json", "prompts.jsonl", "references.jsonl", "responses.jsonl")
REQUIRED_MEMBERS = {
    *("run/" + name for name in RAW_FILES),
    "run/proxy_summary.json", "run/proxy_per_example.jsonl", "run_info.json",
    "scripts/run_three_arm.py", "scripts/score_three_arm.py",
    "ATTRIBUTION.md", "LICENSE_LOCOMO.txt", "SHA256SUMS",
}
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_MEMBER_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_MEMBERS = 128
PINNED_LOCOMO = MANIFEST["benchmarks"]["locomo10"]
HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _json(data: bytes) -> Any:
    return json.loads(data, object_pairs_hook=_unique_object)


def _jsonl(data: bytes) -> list[dict]:
    return [_json(line) for line in data.splitlines() if line.strip()]


def _safe_name(name: str) -> str:
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or "\x00" in name or PurePosixPath(name).is_absolute()
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError("unsafe relative member/file path")
    return name


def _relative_file(directory: Path, name: str) -> Path:
    result = (directory / _safe_name(name)).resolve()
    if not result.is_relative_to(directory.resolve()):
        raise ValueError("file path escapes manifest directory")
    return result


def _record(record: dict) -> None:
    if (not isinstance(record, dict) or not isinstance(record.get("sha256"), str)
            or not HASH_RE.fullmatch(record["sha256"])
            or type(record.get("bytes")) is not int or record["bytes"] < 0):
        raise ValueError("invalid hash/byte-count record")


def _check_bytes(data: bytes, record: dict, label: str) -> None:
    _record(record)
    if len(data) != record["bytes"] or _sha(data) != record["sha256"]:
        raise ValueError(f"byte count or SHA-256 mismatch: {label}")


def _read_file(path: Path, limit: int) -> bytes:
    if path.stat().st_size > limit:
        raise ValueError("payload exceeds configured size limit")
    return path.read_bytes()


def _archive_members(path: Path, manifest: dict) -> dict[str, bytes]:
    _check_bytes(_read_file(path, MAX_ARCHIVE_BYTES), manifest["archive"], "archive")
    declared = manifest.get("members")
    if not isinstance(declared, dict) or not REQUIRED_MEMBERS.issubset(declared):
        raise ValueError("manifest does not declare required audit members")
    for name, record in declared.items():
        _safe_name(name)
        _record(record)
    contents = {}
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) > MAX_MEMBERS:
            raise ValueError("too many ZIP members")
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError("duplicate ZIP member")
        if set(names) != set(declared):
            raise ValueError("ZIP member set differs from manifest")
        if sum(entry.file_size for entry in entries) > MAX_TOTAL_BYTES:
            raise ValueError("total uncompressed payload exceeds size limit")
        for entry in entries:
            _safe_name(entry.filename)
            if (entry.is_dir() or entry.flag_bits & 1
                    or stat.S_ISLNK(entry.external_attr >> 16)):
                raise ValueError("directories, encrypted members and symlinks are not allowed")
            if entry.file_size > MAX_MEMBER_BYTES:
                raise ValueError("ZIP member payload exceeds size limit")
            data = archive.read(entry)
            _check_bytes(data, declared[entry.filename], entry.filename)
            contents[entry.filename] = data
    sums = {}
    for line in contents["SHA256SUMS"].decode("utf-8").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        if not match:
            raise ValueError("malformed SHA256SUMS line")
        checksum, name = match.groups()
        _safe_name(name)
        if name in sums:
            raise ValueError("duplicate SHA256SUMS member")
        sums[name] = checksum
    if set(sums) != set(contents) - {"SHA256SUMS"}:
        raise ValueError("SHA256SUMS must cover every member except itself")
    if any(_sha(contents[name]) != checksum for name, checksum in sums.items()):
        raise ValueError("SHA256SUMS mismatch")
    return contents


def _keyed(rows: list[dict], *, label: str) -> dict[tuple[str, str], dict]:
    result = {}
    for row in rows:
        key = (row["example_id"], row["arm"])
        if key in result:
            raise ValueError(f"duplicate example/arm in {label}")
        result[key] = row
    return result


def _number_equal(actual: Any, expected: Any, label: str) -> None:
    if (isinstance(actual, bool) or not isinstance(actual, (float, int))
            or not math.isfinite(actual) or actual != expected):
        raise ValueError(f"recomputed metric mismatch: {label}")


def _check_dataset(path: Path, protocol: dict, references: list[dict]) -> None:
    data = _read_file(path, MAX_MEMBER_BYTES)
    _check_bytes(data, PINNED_LOCOMO, "pinned LoCoMo dataset")
    if _sha(data) != protocol.get("input_sha256"):
        raise ValueError("dataset does not match execution input hash")
    examples = {example.example_id: example for example in load_locomo(path)}
    for reference in references:
        example = examples.get(reference["example_id"])
        if (example is None or example.category == "5"
                or reference["query"] != example.query
                or reference["reference_answer"] != example.answer
                or reference["category"] != example.category
                or reference["should_abstain"] != example.should_abstain):
            raise ValueError("reference differs from pinned released dataset")


def verify_audit(manifest_path: Path = DEFAULT_MANIFEST, *, archive_path: Path | None = None,
                 dataset_path: Path | None = None) -> dict:
    """Recompute historical lexical metrics without model/network calls or ZIP execution."""
    manifest_path = Path(manifest_path)
    manifest = _json(_read_file(manifest_path, 1024 * 1024))
    if (manifest.get("schema_version") != 1 or not isinstance(manifest.get("pilot_id"), str)
            or not manifest["pilot_id"]
            or manifest.get("model_identity_mode") != "legacy_unverified"):
        raise ValueError("unsupported manifest or historical identity mode")
    directory = manifest_path.resolve().parent
    default_archive = _relative_file(directory, manifest["archive"]["path"])
    contents = _archive_members(Path(archive_path) if archive_path is not None else default_archive, manifest)
    public_records = manifest.get("public_files")
    if not isinstance(public_records, dict) or not {PUBLIC_CSV, PUBLIC_SUMMARY}.issubset(public_records):
        raise ValueError("public CSV and aggregate summary must be declared")
    public = {}
    for name, record in public_records.items():
        if "/" in _safe_name(name):
            raise ValueError("public files must be in the manifest directory")
        public[name] = _read_file(_relative_file(directory, name), MAX_MEMBER_BYTES)
        _check_bytes(public[name], record, name)
    protocol = _json(contents["run/protocol.json"])
    references = _jsonl(contents["run/references.jsonl"])
    responses = _keyed(_jsonl(contents["run/responses.jsonl"]), label="responses")
    metrics = _keyed(_jsonl(contents["run/proxy_per_example.jsonl"]), label="archived metrics")
    reference_map = {row["example_id"]: row for row in references}
    original_summary = _json(contents["run/proxy_summary.json"])
    info = _json(contents["run_info.json"])
    execution = manifest.get("execution", {})
    if (execution.get("protocol_sha256") != stable_digest(protocol)
            or execution.get("input_sha256") != protocol.get("input_sha256")
            or execution.get("code_commit") != info.get("code_commit")):
        raise ValueError("execution metadata/hash mismatch")
    runner_hash = protocol.get("code_sha256", {}).get("evaluation/end-to-end-dialogue/run_three_arm.py")
    if runner_hash != _sha(contents["scripts/run_three_arm.py"]):
        raise ValueError("archived runner differs from frozen code hash")
    if protocol.get("dataset") != "locomo":
        raise ValueError("this audit manifest is for the historical LoCoMo pilot only")
    if dataset_path is not None:
        _check_dataset(Path(dataset_path), protocol, references)
    # Only these four data files enter the temporary score directory. Do not
    # extract the archive, import its snapshots, or create new model responses.
    with tempfile.TemporaryDirectory(prefix="cbt-pilot-audit-") as temporary:
        run_dir = Path(temporary)
        for name in RAW_FILES:
            (run_dir / name).write_bytes(contents["run/" + name])
        recomputed = score_run(run_dir, legacy_audit=True, write_outputs=False)
        if set(path.name for path in run_dir.iterdir()) != set(RAW_FILES):
            raise ValueError("offline scoring unexpectedly wrote extra artifacts")
    if recomputed.get("legacy_identity_unverified") is not True:
        raise ValueError("historical audit cannot approve model identity retrospectively")
    for field in ("examples", "protocol_sha256", "conditions", "paired_differences",
                  "excluded_adversarial_distractors", "reported_provider_models"):
        if original_summary.get(field) != recomputed[field]:
            raise ValueError(f"archived summary differs from recomputation: {field}")
    reader = csv.DictReader(io.StringIO(public[PUBLIC_CSV].decode("utf-8"), newline=""))
    if reader.fieldnames != list(CSV_COLUMNS):
        raise ValueError("unexpected public CSV columns")
    rows = list(reader)
    csv_rows = _keyed(rows, label="public CSV")
    if set(csv_rows) != set(responses) or set(metrics) != set(responses):
        raise ValueError("CSV/archived metrics do not match all frozen responses")
    if Counter(row["arm"] for row in rows) != {arm: recomputed["examples"] for arm in ARMS}:
        raise ValueError("each question must have all three experiment arms")
    for key, response in responses.items():
        row, metric = csv_rows[key], metrics[key]
        reference = reference_map[key[0]]
        if response.get("finish_reason") != "stop":
            raise ValueError("a non-normal response must not enter the historical scores")
        if (row["category"] != reference["category"] or metric["category"] != reference["category"]
                or metric.get("should_abstain") != reference["should_abstain"]):
            raise ValueError("category mismatch")
        scores = answer_proxies(response["hypothesis"], reference["reference_answer"])
        for field, expected in {**scores, "latency_seconds": response["latency_seconds"]}.items():
            _number_equal(float(row[field]), expected, f"CSV {field}")
            _number_equal(metric[field], expected, f"archived {field}")
        tokens = response.get("usage", {}).get("total_tokens")
        if type(tokens) is not int or tokens < 0 or int(row["total_tokens"]) != tokens or metric["usage"] != response["usage"]:
            raise ValueError("provider-reported token usage mismatch")
    aggregate = _json(public[PUBLIC_SUMMARY])
    for field, expected in {
        "examples": recomputed["examples"], "successful_responses": len(responses),
        "protocol_sha256": stable_digest(protocol), "input_sha256": protocol["input_sha256"],
        "code_commit": execution["code_commit"],
    }.items():
        if aggregate.get(field) != expected:
            raise ValueError(f"public aggregate metadata mismatch: {field}")
    prompts = _jsonl(contents["run/prompts.jsonl"])
    optional_metadata = {
        "distinct_conversation_scopes": len({row["scope_id"] for row in prompts}),
        "normal_stop_responses": sum(row.get("finish_reason") == "stop" for row in responses.values()),
        "reasoning_tokens_zero_responses": sum(
            row.get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens") == 0
            for row in responses.values()),
        "requested_model": protocol["settings"]["model"],
        "prompt_records_sha256": protocol["prompt_records_sha256"],
        "excluded_adversarial_distractors": recomputed["excluded_adversarial_distractors"],
        "abstention_examples_in_scored_pilot": sum(row["should_abstain"] for row in references),
        "oracle_history_warning": protocol["oracle_history_warning"],
        "legacy_identity_unverified": True,
        "model_judge_executed": False,
    }
    optional_metadata.update({name: value for name, value in protocol["settings"].items()
                              if name in {"short_turns", "top_k", "minimum_relevance", "short_chars", "long_chars",
                                          "max_context_chars", "max_response_tokens", "thinking_mode", "seed"}})
    optional_metadata["retrieval_backend"] = protocol["settings"].get("backend")
    if "temperature" in protocol:
        optional_metadata["temperature"] = protocol["temperature"]
    for category, count in Counter(row["category"] for row in references).items():
        optional_metadata[f"sample_category_{category}_examples"] = count
    if len(recomputed["reported_provider_models"]) == 1:
        optional_metadata["reported_provider_model"] = recomputed["reported_provider_models"][0]
    if "manifest_spec" in protocol:
        optional_metadata["source_upstream_commit"] = protocol["manifest_spec"].get("upstream_commit")
        optional_metadata["eligible_release_examples"] = (protocol["manifest_spec"]["expected_qa_examples"]
                                                          - protocol["excluded_category_5_examples"])
    if "scored_round" in info:
        optional_metadata["generation_attempts_in_scored_run"] = info["scored_round"]["generation_attempts"]
        optional_metadata["automatic_retries_in_scored_run"] = info["scored_round"]["automatic_retries"]
    if "known_total_http_attempts_including_auth_failures" in info:
        optional_metadata["known_total_http_attempts_including_excluded_rounds_and_auth_failure"] = info["known_total_http_attempts_including_auth_failures"]
    prior_rounds = info.get("prior_rounds", {})
    if "authentication_failures" in prior_rounds:
        optional_metadata["prior_authentication_failures_excluded"] = prior_rounds["authentication_failures"]
    for prior in prior_rounds.get("rounds", []):
        budget = prior["max_tokens"]
        optional_metadata[f"prior_provider_default_{budget}_valid_responses_excluded"] = prior["completed"]
        optional_metadata[f"prior_provider_default_{budget}_empty_responses_excluded"] = prior["empty"]
    for field, expected in optional_metadata.items():
        if field in aggregate and aggregate[field] != expected:
            raise ValueError(f"public aggregate metadata mismatch: {field}")
    for arm, values in recomputed["conditions"].items():
        for field in ("exact_match_proxy", "token_f1_proxy", "median_latency_seconds", "reported_total_tokens"):
            _number_equal(aggregate.get(f"{arm}_{field}"), values[field], f"aggregate {arm}/{field}")
    for comparison, values in recomputed["paired_differences"].items():
        for field in ("mean_token_f1_delta", "wins", "ties", "losses"):
            _number_equal(aggregate.get(f"{comparison}_{field}"), values[field], f"aggregate {comparison}/{field}")
    primary = "short_plus_long_term_minus_short_term"
    paired = recomputed["paired_differences"][primary]
    if "paired_interval_cluster_count" in aggregate:
        _number_equal(aggregate["paired_interval_cluster_count"], paired["cluster_count"], "aggregate interval cluster count")
    for suffix, expected in zip(("lower", "upper"), paired["conversation_cluster_bootstrap_95_percent_interval"]):
        _number_equal(aggregate.get(f"{primary}_conversation_cluster_bootstrap_95_percent_{suffix}"), expected, f"aggregate interval {suffix}")
    return {
        "offline": True, "api_calls": 0, "recomputed": True,
        "pilot_id": manifest["pilot_id"], "examples": recomputed["examples"],
        "responses": len(responses), "members_verified": len(contents),
        "public_files_verified": len(public), "dataset_reference_check": dataset_path is not None,
        "execution_mode": "legacy_audit_only", "legacy_only": True,
        "legacy_identity_unverified": True, "model_identity_mode": "legacy_unverified",
        "official_evaluator_status": "not executed", "model_judge_executed": False,
        "score_type": "non-official lexical answer proxies",
        "checksum_limit": "Hashes check byte consistency, not authenticity, semantic correctness or provider/model identity; this manifest is not a cryptographic identity signature.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--archive", type=Path, help="Override the manifest-relative ZIP path")
    parser.add_argument("--dataset", type=Path, help="Optional local pinned LoCoMo JSON for gold/reference checking")
    args = parser.parse_args(argv)
    try:
        result = verify_audit(args.manifest, archive_path=args.archive, dataset_path=args.dataset)
    except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as exc:
        # Never echo dataset, prompts, hypotheses or credentials on validation errors.
        print(json.dumps({"offline": True, "api_calls": 0, "recomputed": False,
                          "error": type(exc).__name__, "detail": "Audit validation failed; check manifest, archive and public artifacts."}), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
