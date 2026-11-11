"""Acquire frozen CBT-Bench professional references in genuine hosted Colab.

This is a quality anchor, never a longitudinal memory benchmark.  Network and
real dataset preparation are CLI-only and guarded by the hosted-Colab check.
Pure parsers are tested locally using tiny invented fixtures.  The module does
not read credentials, load a model, send inference requests, or score responses.
All original reference/label bytes stay under assessor_only; public generation
inputs contain only the released client statement and task metadata.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen


VERSION = "cbt-bench-professional-quality-anchor-v1"
DATASET = "Psychotherapy-LLM/CBT-Bench"
REVISION = "bbd37514ba2e4db46bd9633d9207a57828c2ec9d"
LICENSE = "cc-by-nc-4.0"
MAX_SOURCE_BYTES = 2_000_000
REFERENCE_KEYS = {"id", "level", "client_statement", "response"}
LEVELS = {"b": "beginner", "m": "intermediate", "a": "advanced"}
ITEM_ID = re.compile(r"([1-9]|10)_([bma])_([1-9][0-9]{0,2})\Z")
SOURCE_NAMES = ["README.md"] + [
    f"CBT-DP/dp-reference-exe-{category}.json" for category in range(1, 11)
] + ["dp-pairwise-comparison.json"]
LABEL_SUFFIXES = ("a_e_b", "a_mb_b", "a_sb_b", "b_mb_a", "b_sb_a")
# Explicit design-time manual exposure, not a claim that other rows are pristine.
MANUALLY_EXPOSED_CATEGORY_IDS = (9,)
MANUALLY_EXPOSED_ITEM_IDS = (
    "1_b_4", "1_m_4", "3_b_3", "3_m_1", "4_m_6", "7_a_1", "7_a_6",
    "8_a_5", "8_b_2", "9_b_1", "10_a_4",
)
EXPOSURE_REGISTER = {
    "version": "professional-reference-manual-exposure-20261112-v1",
    "method": "manual_primary_source_web_read_during_design",
    "entire_category_ids_exposed": list(MANUALLY_EXPOSED_CATEGORY_IDS),
    "pairwise_item_ids_exposed": list(MANUALLY_EXPOSED_ITEM_IDS),
    "propagate_to_identical_client_statement_groups": True,
    "unlisted_rows_are_verified_holdout": False,
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _reject_constant(_value):
    raise ValueError("nonfinite_json_number")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def parse_json(data):
    if type(data) is not bytes or len(data) > MAX_SOURCE_BYTES:
        raise ValueError("source_bytes_outside_bound")
    return json.loads(data.decode("utf-8-sig"), object_pairs_hook=_unique_object,
                      parse_constant=_reject_constant)


def require_colab():
    """Fail before filesystem/network work; not a hardware attestation."""
    available = False
    try:
        available = (bool(os.environ.get("COLAB_RELEASE_TAG"))
                     and Path("/content").is_dir()
                     and importlib.util.find_spec("google.colab") is not None)
        if available:
            importlib.import_module("google.colab")
    except (ImportError, ModuleNotFoundError, ValueError, AttributeError):
        available = False
    if not available:
        raise RuntimeError("genuine_hosted_colab_required_before_any_request")


def validate_license_card(data):
    if type(data) is not bytes or not data or len(data) > MAX_SOURCE_BYTES:
        raise ValueError("invalid_license_card_bytes")
    text = data.decode("utf-8-sig")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing_dataset_card_frontmatter")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration as error:
        raise ValueError("unterminated_dataset_card_frontmatter") from error
    declarations = [line for line in lines[1:end] if re.match(r"^license\s*:", line)]
    if len(declarations) != 1 or not re.fullmatch(
            r"license\s*:\s*[\"']?cc-by-nc-4\.0[\"']?\s*", declarations[0]):
        raise ValueError("unexpected_dataset_license")
    return LICENSE


def parse_reference_bytes(data, *, category_id):
    """Pure schema parser; never infer dates, history or clinician credentials."""
    if type(category_id) is not int or not 1 <= category_id <= 10:
        raise ValueError("invalid_category_id")
    rows = parse_json(data)
    if type(rows) is not list or not rows or len(rows) > 1000:
        raise ValueError("invalid_reference_list")
    seen = set()
    for row in rows:
        if type(row) is not dict or set(row) != REFERENCE_KEYS:
            raise ValueError("reference_schema_mismatch")
        if any(type(row[key]) is not str or not row[key].strip()
               or len(row[key]) > 20000 for key in REFERENCE_KEYS):
            raise ValueError("invalid_reference_string")
        match = ITEM_ID.fullmatch(row["id"])
        if not match or int(match.group(1)) != category_id:
            raise ValueError("reference_id_category_mismatch")
        if row["level"] != LEVELS[match.group(2)]:
            raise ValueError("reference_id_level_mismatch")
        if row["id"] in seen:
            raise ValueError("duplicate_reference_id")
        seen.add(row["id"])
    return rows


def compile_references(source_files, *, require_all_categories=False):
    """Separate client inputs from assessor references; fixtures may be tiny.

    The group ID groups identical normalized client statements.  It is not a
    patient ID or a claim that the source supplies longitudinal case families.
    """
    expected = set(SOURCE_NAMES[1:-1])
    if not source_files or not set(source_files).issubset(expected):
        raise ValueError("unexpected_reference_source_name")
    if require_all_categories and set(source_files) != expected:
        raise ValueError("incomplete_reference_categories")
    inputs, references, seen = [], [], set()
    category_counts, difficulty_counts = Counter(), Counter()
    for filename in sorted(source_files):
        category_id = int(re.fullmatch(
            r"CBT-DP/dp-reference-exe-([1-9]|10)\.json", filename).group(1))
        data = source_files[filename]
        for row in parse_reference_bytes(data, category_id=category_id):
            if row["id"] in seen:
                raise ValueError("duplicate_cross_file_reference_id")
            seen.add(row["id"])
            case_id = "cbt-dp-" + row["id"]
            normalized_statement = " ".join(row["client_statement"].split())
            group_id = "same-client-statement-" + sha256(normalized_statement.encode())
            provenance = {"dataset": DATASET, "revision": REVISION,
                          "relative_path": filename, "source_sha256": sha256(data),
                          "source_item_id": row["id"]}
            inputs.append({"case_id": case_id, "source_item_id": row["id"],
                           "category_id": category_id, "difficulty": row["level"],
                           "group_id": group_id,
                           "group_type": "same_client_statement_not_patient_family",
                           "client_statement": row["client_statement"],
                           "history": [], "history_count": 0,
                           "quality_anchor_role": True,
                           "memory_effect_eligible": False,
                           "provenance": provenance})
            references.append({"case_id": case_id,
                               "reference_response": row["response"],
                               "not_the_only_valid_response": True,
                               "assessor_only": True, "provenance": provenance})
            category_counts[str(category_id)] += 1
            difficulty_counts[row["level"]] += 1
    exposed_groups = {row["group_id"] for row in inputs
                      if row["category_id"] in MANUALLY_EXPOSED_CATEGORY_IDS
                      or row["source_item_id"] in MANUALLY_EXPOSED_ITEM_IDS}
    exposure_counts = Counter()
    for row in inputs:
        row["exposure_status"] = (
            "development_exposed" if row["group_id"] in exposed_groups else
            "not_manually_inspected_in_this_run_not_verified_holdout")
        exposure_counts[row["exposure_status"]] += 1
    return {"inputs": inputs, "references": references,
            "exposure_register": EXPOSURE_REGISTER,
            "counts": {"reference_exercises": len(inputs),
                       "by_category_id": dict(sorted(category_counts.items())),
                       "by_difficulty": dict(sorted(difficulty_counts.items())),
                       "by_exposure_status": dict(sorted(exposure_counts.items()))}}


def inspect_pairwise_bytes(data):
    """Inspect released labels without assigning or inventing new judgments."""
    rows = parse_json(data)
    if type(rows) is not list or not rows or len(rows) > 10000:
        raise ValueError("invalid_pairwise_list")
    required = {"Input.client_statement", "Input.id", "Input.level", "Input.order",
                "Input.res_a", "Input.res_b", "Input.exe"} | {
                    f"Input.cr_{i}" for i in range(1, 5)}
    all_keys, reasons, statement_groups = set(), Counter(), set()
    structurally_valid = 0
    for row in rows:
        if type(row) is not dict:
            reasons["row_not_object"] += 1
            continue
        all_keys.update(row)
        if not required.issubset(row):
            reasons["missing_input_fields"] += 1
            continue
        if any(type(row[key]) is not str or not row[key].strip() for key in required):
            reasons["invalid_input_field"] += 1
            continue
        if row["Input.order"] not in {"A_is_ref", "B_is_ref"}:
            reasons["invalid_reference_order"] += 1
            continue
        valid = True
        for criterion in ("all", "cr_1", "cr_2", "cr_3", "cr_4"):
            keys = [f"Answer.{criterion}_{suffix}" for suffix in LABEL_SUFFIXES]
            if (any(key not in row or type(row[key]) is not str for key in keys)
                    or sum(bool(row[key]) for key in keys) != 1
                    or any(row[key] and row[key] != key.removeprefix("Answer.")
                           for key in keys)):
                reasons["invalid_or_missing_one_hot_label"] += 1
                valid = False
                break
        if valid:
            structurally_valid += 1
            normalized = " ".join(row["Input.client_statement"].split())
            statement_groups.add(sha256(normalized.encode()))
    return {"role": "assessor_only_existing_expert_judge_calibration",
            "source_rows": len(rows), "structurally_valid_rows": structurally_valid,
            "invalid_row_reasons": dict(reasons), "keys": sorted(all_keys),
            "valid_statement_groups": len(statement_groups),
            "new_scores_created": False,
            "new_human_reviews": 0, "new_model_judgments": 0,
            "schema_inspection_is_semantic_validation": False}


def _write_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path, value):
    _write_bytes(path, (canonical(value) + "\n").encode())


def _journal(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _download_bytes(url):
    request = Request(url, headers={"User-Agent": "Capstone-CBT-Reference-Audit/1"})
    with urlopen(request, timeout=60) as response:
        if urlparse(response.geturl()).scheme != "https":
            raise ValueError("non_https_dataset_redirect")
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) > MAX_SOURCE_BYTES:
            raise ValueError("source_exceeds_download_bound")
        data = response.read(MAX_SOURCE_BYTES + 1)
        if not data or len(data) > MAX_SOURCE_BYTES:
            raise ValueError("source_exceeds_download_bound_or_empty")
        return data


def acquire(output):
    require_colab()
    output = Path(output).resolve()
    try:
        output.relative_to(Path("/content"))
    except ValueError as error:
        raise ValueError("output_must_be_inside_hosted_content") from error
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"version": VERSION, "dataset": DATASET, "revision": REVISION,
                "license_required": LICENSE, "purpose": "academic_noncommercial_research",
                "quality_anchor_role": True, "memory_effect_eligible": False,
                "manual_exposure_register": EXPOSURE_REGISTER,
                "source_names": SOURCE_NAMES, "max_bytes_per_file": MAX_SOURCE_BYTES,
                "automatic_retry": False, "secret_access": False,
                "new_model_calls": 0, "code_sha256": sha256(Path(__file__).read_bytes())}
    _write_json(output / "acquisition_protocol.json", protocol)
    sources = []
    stage = "acquisition"
    try:
        for index, name in enumerate(SOURCE_NAMES):
            url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{quote(name)}"
            _journal(output / "acquisition_journal.jsonl",
                     {"event": "download_intent", "index": index,
                      "relative_path": name, "url": url, "retry": False})
            data = _download_bytes(url)
            relative = "assessor_only/source_archive/" + name
            _write_bytes(output / relative, data)
            receipt = {"relative_path": name, "archive_path": relative, "url": url,
                       "bytes": len(data), "sha256": sha256(data)}
            sources.append(receipt)
            _journal(output / "acquisition_journal.jsonl",
                     {"event": "download_archived", "index": index, **receipt})
        manifest = {"version": VERSION, "dataset": DATASET, "revision": REVISION,
                    "all_source_bytes_archived_before_parsing": True, "sources": sources,
                    "manual_exposure_register": EXPOSURE_REGISTER}
        _write_json(output / "source_manifest.json", manifest)
        stage = "parsing_after_source_manifest_frozen"
        # Read back and hash every archived source before opening any JSON.
        archived = {}
        for source in sources:
            data = (output / source["archive_path"]).read_bytes()
            if sha256(data) != source["sha256"] or len(data) != source["bytes"]:
                raise ValueError("archived_source_integrity_mismatch")
            archived[source["relative_path"]] = data
        validate_license_card(archived["README.md"])
        compiled = compile_references({name: archived[name] for name in SOURCE_NAMES[1:-1]},
                                      require_all_categories=True)
        calibration = inspect_pairwise_bytes(archived["dp-pairwise-comparison.json"])
        _write_bytes(output / "inputs.jsonl", "".join(
            canonical(row) + "\n" for row in compiled["inputs"]).encode())
        _write_bytes(output / "assessor_only/reference_responses.jsonl", "".join(
            canonical(row) + "\n" for row in compiled["references"]).encode())
        _write_json(output / "assessor_only/calibration_schema_diagnostic.json", calibration)
        summary = {"version": VERSION, "dataset": DATASET, "revision": REVISION,
                   "license": LICENSE, "fixture_only": False,
                   "quality_anchor_role": True, "memory_effect_eligible": False,
                   "history_records": 0, "new_model_calls": 0, "secret_access": False,
                   "new_human_reviews": 0, "new_effect_scores": None,
                   "source_files": len(sources),
                   "source_bytes": sum(source["bytes"] for source in sources),
                   "source_manifest_sha256": sha256((output / "source_manifest.json").read_bytes()),
                   "inputs_sha256": sha256((output / "inputs.jsonl").read_bytes()),
                   "references_sha256": sha256((output / "assessor_only/reference_responses.jsonl").read_bytes()),
                   "counts": compiled["counts"],
                   "manual_exposure_register": EXPOSURE_REGISTER,
                   "existing_pairwise_schema": calibration,
                   "reference_match_is_sole_quality_score": False,
                   "reference_or_label_content_in_inputs": False,
                   "source_files_sha256": {source["relative_path"]: source["sha256"]
                                           for source in sources}}
        _write_json(output / "summary.json", summary)
        return summary
    except Exception as error:
        failure = {"version": VERSION, "stage": stage,
                   "status": "failed_preserve_partial_archive_no_retry",
                   "error_type": type(error).__name__, "archived_sources": len(sources),
                   "new_model_calls": 0, "automatic_retry": False}
        # Avoid server response bodies, reference content or arbitrary exception text.
        _write_json(output / "failure.json", failure)
        raise RuntimeError(canonical(failure)) from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True,
                        help="New absolute /content directory; existing outputs are never replaced")
    args = parser.parse_args()
    print(canonical(acquire(args.output)), flush=True)


if __name__ == "__main__":
    main()
