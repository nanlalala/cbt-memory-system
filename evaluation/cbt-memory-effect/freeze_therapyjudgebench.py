"""Acquire the canonical TherapyJudgeBench v2 source bank in hosted Colab only.

No model calls, credentials, label generation, retry or generation-input export.
Run once with a fresh /content directory; failed attempts remain journalled.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

VERSION = "therapyjudgebench-v2-source-freeze-v1"
REPOSITORY = "therapygym/therapyjudgebench"
REVISION = "fba2e5ad29356417cde50479d823c6cf0b316754"
DATA_SHA256 = "ade12985a5224e425805cfb198b3a3fa338727a158f1446fa5073a40fdf6fd01"
FILES = ("README.md", "LICENSE", "DATASHEET.md", "croissant.json", "therapyjudgebench.jsonl")
CTRS = ("AGENDA", "FEEDBACK", "UNDERSTANDING", "INTERPERSONAL EFFECTIVENESS",
        "COLLABORATION", "PACING", "GUIDED DISCOVERY", "FOCUSING", "STRATEGY",
        "TECHNIQUE APPLICATION", "HOMEWORK")
SAFETY = ("GIVING MEDICAL ADVICE", "FAILURE TO ADDRESS CRISIS", "FAILURE TO ADDRESS ABUSE",
          "FAILURE TO ADDRESS FUNCTIONAL IMPAIRMENT")
NULL_SAFETY_IDS = {f"Conversation {i}" for i in (28, 33, 52, 53, 60)}
EXPOSED_IDS = {f"Conversation {i}" for i in range(1, 16)}
MAX_BYTES = 4_000_000


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require_colab():
    # Operational guard, not cryptographic runtime attestation.
    try:
        valid = (os.environ.get("COLAB_RELEASE_TAG") and Path("/content").is_dir()
                 and importlib.util.find_spec("google.colab") is not None)
    except (ImportError, ModuleNotFoundError, ValueError):
        valid = False
    if not valid:
        raise RuntimeError("genuine_hosted_colab_required_before_any_io_or_request")


def _check_url(url, filename):
    p = urlsplit(url)
    allowed_paths = {
        f"/datasets/{REPOSITORY}/resolve/{REVISION}/{filename}",
        f"/api/resolve-cache/datasets/{REPOSITORY}/{REVISION}/{filename}",
    }
    if (p.scheme != "https" or p.hostname != "huggingface.co" or p.username or p.password
            or p.port not in (None, 443) or p.fragment or unquote(p.path) not in allowed_paths):
        raise ValueError("redirect_outside_exact_official_pinned_artifact")


class _PinnedRedirect(HTTPRedirectHandler):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename

    def redirect_request(self, request, fp, code, message, headers, newurl):
        _check_url(urljoin(request.full_url, newurl), self.filename)
        return super().redirect_request(request, fp, code, message, headers, newurl)


def download(filename):
    url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{filename}"
    request = Request(url, headers={"User-Agent": "Capstone-Professional-Source-Freeze/1",
                                    "Accept-Encoding": "identity"})
    with build_opener(_PinnedRedirect(filename)).open(request, timeout=45) as response:
        _check_url(response.geturl(), filename)
        size = response.headers.get("Content-Length")
        if size is not None and not 0 < int(size) <= MAX_BYTES:
            raise ValueError("declared_file_size_outside_bound")
        if response.headers.get("Content-Encoding", "identity").lower() not in ("identity", ""):
            raise ValueError("encoded_response_not_exact_bytes")
        data = response.read(MAX_BYTES + 1)
        if not data or len(data) > MAX_BYTES or (size is not None and len(data) != int(size)):
            raise ValueError("empty_oversized_or_incomplete_file")
        if data.lstrip().lower().startswith((b"<!doctype html", b"<html")):
            raise ValueError("html_error_or_access_page")
        return data, {"source_url": url, "final_url": response.geturl(),
                      "etag": response.headers.get("ETag")}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def parse_json(data):
    def invalid(value):
        raise ValueError("non_standard_json_number:" + value)
    return json.loads(data, object_pairs_hook=_object, parse_constant=invalid)


def _score(value):
    return value is None or (type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 6)


def audit_records(records, *, require_release_counts=True):
    ids, sessions, profiles, models, null_ids = set(), set(), set(), set(), set()
    ctrs = {key: Counter() for key in CTRS}
    safety = {key: Counter() for key in SAFETY}
    for row in records:
        ident = row["id"]
        if ident in ids or row["session_id"] in sessions:
            raise ValueError("duplicate_conversation_or_session")
        ids.add(ident); sessions.add(row["session_id"])
        profiles.add(row["patient_profile"]); models.add(row["therapist_model"])
        if (type(row["num_turns"]) is not int or row["num_turns"] != 5
                or row["patient_model"] != "O3_MINI"
                or row["therapist_style"] != "professional_cbt_therapist"):
            raise ValueError("unexpected_dialogue_configuration")
        messages = row["conversation"]["messages"]
        if len(messages) != 10 or any(
                m["role"] != ("user" if i % 2 == 0 else "assistant")
                or not isinstance(m["content"], str) or not m["content"].strip()
                for i, m in enumerate(messages)):
            raise ValueError("expected_ten_ordered_alternating_messages")
        labels = row["expert_rating"]
        if set(labels["ctrs_ratings"]) != set(CTRS) or set(labels["safety_ethics"]) != set(SAFETY):
            raise ValueError("v2_exact_label_schema_required_do_not_mix_v1")
        for key, value in labels["ctrs_ratings"].items():
            if not _score(value):
                raise ValueError("invalid_ctrs_score")
            ctrs[key]["null" if value is None else "nonmissing"] += 1
        if not _score(labels["overall_ctrs_score"]):
            raise ValueError("invalid_overall_score")
        values = list(labels["safety_ethics"].values())
        if any(v is None for v in values) and not all(v is None for v in values):
            raise ValueError("partly_missing_revised_safety_record")
        if all(v is None for v in values):
            null_ids.add(ident)
        for key, value in labels["safety_ethics"].items():
            if value is not None and type(value) is not bool:
                raise ValueError("safety_must_be_boolean_or_null")
            safety[key]["null" if value is None else "positive" if value else "negative"] += 1
    if require_release_counts and (
            ids != {f"Conversation {i}" for i in range(1, 117)} or len(profiles) != 18
            or len(models) != 8 or sum(v["nonmissing"] for v in ctrs.values()) != 993
            or null_ids != NULL_SAFETY_IDS
            or [safety[k]["positive"] for k in SAFETY] != [0, 26, 4, 49]):
        raise ValueError("canonical_v2_release_counts_mismatch")
    exposed_profiles = {r["patient_profile"] for r in records if r["id"] in EXPOSED_IDS}
    groups = [{"id": r["id"], "patient_profile": r["patient_profile"],
               "exposure": "development_exposed" if r["patient_profile"] in exposed_profiles
                           else "unexposed_candidate_not_sealed_test"} for r in records]
    return {"dialogues": len(ids), "patient_profiles": len(profiles), "therapist_models": len(models),
            "ctrs_by_dimension": {k: dict(v) for k, v in ctrs.items()},
            "safety_by_dimension": {k: dict(v) for k, v in safety.items()},
            "medical_advice_has_zero_positive_support": safety[SAFETY[0]]["positive"] == 0,
            "missing_values_must_be_excluded_per_dimension": True}, groups


def freeze(output):
    require_colab()
    output = Path(output).resolve()
    if not output.is_relative_to(Path("/content")) or output == Path("/content"):
        raise ValueError("fresh_output_directory_under_content_required")
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    source_dir = output / "assessor_only"
    source_dir.mkdir(mode=0o700)
    journal = output / "acquisition_journal.jsonl"

    def write(path, data):
        with path.open("xb") as stream:
            os.chmod(path, 0o600)
            stream.write(data); stream.flush(); os.fsync(stream.fileno())

    def event(**values):
        with journal.open("a", encoding="utf-8") as stream:
            os.chmod(journal, 0o600)
            stream.write(canonical({"at_utc": datetime.now(timezone.utc).isoformat(), **values}) + "\n")
            stream.flush(); os.fsync(stream.fileno())

    files, raw = [], {}
    for filename in FILES:
        event(event="attempt_started", filename=filename, revision=REVISION)
        try:
            data, metadata = download(filename)
            write(source_dir / filename, data)
            raw[filename] = data
            files.append({"filename": filename, "bytes": len(data), "sha256": digest(data), **metadata})
            event(event="attempt_completed", filename=filename, bytes=len(data), sha256=digest(data))
        except Exception as exc:
            event(event="attempt_failed", filename=filename, error_type=type(exc).__name__)
            raise
    if digest(raw["therapyjudgebench.jsonl"]) != DATA_SHA256:
        raise ValueError("published_jsonl_sha256_mismatch")
    card = raw["README.md"].decode("utf-8")
    if "2.0.0" not in card or "CC BY 4.0" not in card:
        raise ValueError("canonical_card_version_or_license_missing")
    if b"Attribution 4.0 International" not in raw["LICENSE"]:
        raise ValueError("canonical_cc_by_4_license_missing")
    croissant = parse_json(raw["croissant.json"])
    if not any(item.get("sha256") == DATA_SHA256 for item in croissant["distribution"]):
        raise ValueError("publisher_checksum_metadata_mismatch")
    records = [parse_json(line) for line in raw["therapyjudgebench.jsonl"].decode("utf-8").splitlines() if line.strip()]
    audit, groups = audit_records(records)
    write(source_dir / "schema_audit.json", (canonical(audit) + "\n").encode())
    write(source_dir / "profile_exposure_groups.json", (canonical(groups) + "\n").encode())
    manifest = {"protocol": VERSION, "repository": REPOSITORY, "revision": REVISION,
                "dataset_version": "2.0.0", "license": "CC-BY-4.0", "files": files,
                "collector_sha256": digest(Path(__file__).read_bytes()),
                "dialogue_origin": "SIMULATED_LLM_patient_and_therapist_conversations",
                "rating_origin": "published_consolidated_labels_by_two_licensed_CBT_trained_humans",
                "double_annotation_policy": "20_percent_independently_double_annotated_remainder_single",
                "individual_rater_records_available": False,
                "complete_sources_and_ratings_assessor_only": True, "generation_inputs_created": False,
                "new_oursystem_human_quality_scores_created": False, "model_calls": 0,
                "split_policy": "group_by_patient_profile_all_profiles_of_exposed_Conversation_1_to_15_are_development",
                "holdout_status": "remaining_profiles_are_candidates_not_a_sealed_test",
                "role": "judge_calibration_source_bank_not_longitudinal_memory_effect_result"}
    write(output / "source_bank_manifest.json", (canonical(manifest) + "\n").encode())
    event(event="freeze_validated", dialogues=audit["dialogues"], patient_profiles=audit["patient_profiles"])
    return {"status": "source_bank_frozen_not_model_evaluated", "output": str(output), **audit}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(canonical(freeze(parser.parse_args().output)))
