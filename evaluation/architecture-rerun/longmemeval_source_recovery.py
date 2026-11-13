"""Rebuild the original complete LongMemEval S500 sources in hosted Colab.

This entry only downloads and transforms pinned public benchmark sources. It
does not initialize a transport, read a Secret, prepare model-derived memories,
generate answers, grade answers, or authorize the separate paid experiment.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import formal_full as full
import formal_sources as sources
import formal_continuation as continuation
from architecture_retest.journal import atomic, digest, utc
from cbt_runtime_guard import assert_repository_runtime

VERSION = "longmemeval-original-s500-source-recovery-v1"
ROOT = Path("/content/capstone-longmemeval500-source-recovery-20261113-v1")
DATASET = "longmemeval_s_cleaned"
REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
URL = ("https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/"
       + REVISION + "/longmemeval_s_cleaned.json")
RAW_BYTES = 277383467
RAW_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
REFERENCE_FILE_SHA256 = "2b9eb936b8a39217d2d0ceb84010e2356cfd9cae9c78483759a0624c7a5ec3af"
REFERENCE_PROTOCOL_SHA256 = "ba567ab509b1045975ecef49e9a9cd51280413618d1fcbbda94a0e475591f564"
CASES_SHA256 = "e0747b301cf57422e9320af92e0cedab75e52e6232e5b4a0655eea66ebace10e"


def _read(path):
    path = Path(path)
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("regular source-recovery file required")
    return json.loads(path.read_text())


def original_protocol(path):
    """Read the actual archived metadata, never invent a replacement protocol."""
    if sources.hash_file(path) != REFERENCE_FILE_SHA256:
        raise ValueError("the exact archived LongMemEval protocol file is required")
    value = _read(path)
    if (value.get("protocol_sha256") != REFERENCE_PROTOCOL_SHA256
            or digest({k:v for k,v in value.items() if k != "protocol_sha256"}) != REFERENCE_PROTOCOL_SHA256
            or value.get("dataset_key") != DATASET or value.get("cases_sha256") != CASES_SHA256
            or value.get("case_count") != 500 or value.get("source_histories") != 500
            or value.get("preparation_chunks") != 18866 or value.get("planned_answers") != 1500
            or value.get("maximum_attempts") != 39232
            or value.get("settings") != full.SETTINGS or value.get("provider") != full.IDENTITY
            or value.get("arms") != list(full.ARMS)):
        raise ValueError("the original complete LongMemEval plan differs")
    return value


def source_pin():
    spec = sources.formal.load_plan(sources.formal.DEFAULT_PLAN)["datasets"][DATASET]
    if (spec.get("source") != URL or spec.get("revision") != REVISION
            or spec.get("bytes") != RAW_BYTES or spec.get("sha256") != RAW_SHA256
            or spec.get("expected_released_examples") != 500
            or spec.get("expected_eligible_examples") != 500 or spec.get("exclude_categories") != []):
        raise ValueError("the immutable original complete S500 source pin differs")
    return {key:spec[key] for key in ("source", "revision", "bytes", "sha256")}


def scientific_compatibility(original):
    current = full.code_hashes()
    if set(original["code_sha256"]) != set(current):
        raise ValueError("the original scientific code roster differs")
    changes = {}
    for name, old in original["code_sha256"].items():
        if old == current[name]:
            continue
        if name == continuation.GUARD_PATH and old == continuation.OLD_GUARD_HASH:
            reason = {"reason":"runtime origin check only; no request representation change"}
        else:
            reason = {"reason":"additive CBT V2 API only; complete original V1 module AST retained",
                      "original_v1_ast_sha256":continuation._v1_ast_compatibility(name, old)}
        changes[name] = {"original_sha256":old, "current_sha256":current[name], **reason}
    return changes


def verify_rebuilt(folder, original):
    """Reject any roster, source, reference, chunking or message-preflight drift."""
    folder = Path(folder)
    protocol = _read(folder / "protocol.json")
    if protocol.get("protocol_sha256") != digest({k:v for k,v in protocol.items() if k != "protocol_sha256"}):
        raise ValueError("rebuilt protocol checksum differs")
    # Only creation time and explicitly reviewed source-code hashes can differ.
    ignored = {"created_at_utc", "code_sha256", "protocol_sha256"}
    if ({k:v for k,v in protocol.items() if k not in ignored}
            != {k:v for k,v in original.items() if k not in ignored}
            or protocol.get("code_sha256") != full.code_hashes()):
        raise ValueError("rebuilt original source plan or representation differs")
    for name, expected in original["files_sha256"].items():
        if sources.hash_file(folder / name) != expected:
            raise ValueError("rebuilt original source/reference bytes differ: " + name)
    cases = _read(folder / "cases.json")
    if digest(cases) != CASES_SHA256:
        raise ValueError("rebuilt complete original case roster differs")
    if any((folder / name).exists() and (folder / name).stat().st_size
           for name in ("intents.jsonl", "outcomes.jsonl")):
        raise ValueError("source-only recovery cannot contain provider attempts")
    return protocol


def finite_plan(original):
    primary = original["planned_answers"]
    repeats = (primary + 9) // 10
    extraction = original["preparation_chunks"]
    return {"case_count":500, "source_histories":500, "arms":list(full.ARMS),
        "preparation_chunks":extraction, "maximum_extraction_calls":extraction,
        "maximum_support_verification_calls":extraction, "planned_answer_calls":primary,
        "maximum_deepseek_calls":extraction*2+primary,
        "maximum_primary_qwen_grades":primary, "maximum_qwen_repeat_grades":repeats,
        "maximum_qwen_calls":primary+repeats, "maximum_total_calls":extraction*2+primary*2+repeats,
        "independent_judge_rule":"existing formal_judge: reference grading plus outcome-blind 10% repeats",
        "technical_reader_failure_rule":"retain 1500 answer-row denominator; grade only received answers",
        "new_provider_attempts":0, "paid_execution_authorized":False,
        "budget_status":"source_only; separate explicit paid entry and budget review required"}


def prepare(reference_protocol_path):
    full.require_hosted()
    assert_repository_runtime(full.ROOT)
    original = original_protocol(reference_protocol_path)
    pin, changes = source_pin(), scientific_compatibility(original)
    root = ROOT.resolve()
    if root != ROOT or root == Path("/content") or not root.is_relative_to("/content"):
        raise ValueError("the fixed independent hosted source-recovery directory is required")
    receipt_path = root / "source_recovery_receipt.json"
    if receipt_path.exists():
        previous = _read(receipt_path)
        protocol = verify_rebuilt(root / "study", original)
        if (previous.get("version") != VERSION
                or previous.get("rebuilt_protocol_sha256") != protocol["protocol_sha256"]
                or previous.get("original_protocol_file_sha256") != REFERENCE_FILE_SHA256
                or previous.get("source_pin") != pin or previous.get("scientific_compatibility") != changes):
            raise ValueError("the retained source-recovery receipt differs")
        return previous
    root.mkdir(parents=True, exist_ok=True)
    retained = root / "reference_protocol.original.json"
    if retained.exists():
        original_protocol(retained)
    else:
        shutil.copyfile(reference_protocol_path, retained)
    atomic(root / "source_preparation_status.json", {"version":VERSION, "status":"preparing",
        "started_or_resumed_at_utc":utc(), "model_calls":0, "paid_execution_authorized":False})
    raw = sources.source_file(DATASET, root / "raw")
    source_dir, study = root / "sources", root / "study"
    if not (source_dir / "source_manifest.json").exists():
        sources.prepare(DATASET, root / "raw", source_dir, existing=raw)
    if not (study / "protocol.json").exists():
        full.prepare(source_dir, study)
    protocol = verify_rebuilt(study, original)
    plan = finite_plan(original)
    atomic(root / "finite_original_plan.json", plan)
    receipt = {"version":VERSION, "status":"source_only_complete", "verified_at_utc":utc(),
        "source_pin":pin, "raw_source_path":str(raw), "source_directory":str(source_dir),
        "rebuilt_study_directory":str(study), "original_protocol_path":str(retained),
        "original_protocol_file_sha256":REFERENCE_FILE_SHA256,
        "original_protocol_sha256":REFERENCE_PROTOCOL_SHA256,
        "rebuilt_protocol_sha256":protocol["protocol_sha256"], "scientific_compatibility":changes,
        "all_original_source_artifact_hashes_match":True, "cases_sha256":CASES_SHA256,
        "case_count":500, "source_histories":500, "preparation_chunks":18866,
        "planned_answers":1500, "mandatory_packet_preflight_all_fit":True,
        "model_derived_memory_prepared":False, "model_calls":0, "secret_access":False,
        "paid_execution_started":False, "paid_execution_authorized":False,
        "existing_v4_invocation_changed":False, "finite_plan":plan}
    atomic(receipt_path, receipt)
    atomic(root / "source_preparation_status.json", {"version":VERSION, "status":"source_only_complete",
        "verified_at_utc":utc(), "receipt_sha256":sources.hash_file(receipt_path), "model_calls":0,
        "paid_execution_authorized":False})
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-protocol", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.reference_protocol), ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
