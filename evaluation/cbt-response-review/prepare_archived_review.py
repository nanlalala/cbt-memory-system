#!/usr/bin/env python3
"""Prepare R1–R5 review of an immutable, already exported fictional CBT run.

This performs no model calls and does not read the original controlled run.  It
does not regenerate, update, or relabel that run as the current product.  The
reviewer projection intentionally contains only permitted history, current
input and the exact answer field.  Allocation and old assessment material are
never sent to a reviewer.  Privacy-withheld rows remain in the controller's
planned denominator and cannot be reconstructed from another archive.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile
import zipfile


SCHEMA = "archived-cbt-r1-r5-review-preparation-v1"
EXECUTABLE = "18432622cc26a80704a50165a9383d559cda22c7"
OLD_PROTOCOL = "91cff8e77667c470d6009c517b250018e5cfc3b7ea858403bb6aa65d9375123a"
PINNED_INPUTS = {
    "reviewer_zip": "73583524a35ed73bc8ad2c7d696b3d355c7b6f334411a102e9cf4352dbc9a15b",
    "controller_zip": "ebb7ff7b07193c8496f64756e4a4b2e4c43f096aa4ed218957009c31cb39f440",
    "report_summary": "8c31d07f2744a65c79a4c6300b28ec9fd3545315ebbdde62f003962f8fd686ef",
}
ARMS = ("short_term", "raw_history_rag", "managed_memory_rag")
EXPECTED_CASES = {f"{variant}:T{task:02d}" for variant in "ABC" for task in range(1, 13)}
SOURCE_FAMILY = "fictional-followup-development-template-v2"
PACKET_FIELDS = {"canonical_id", "history", "input", "response", "packet_sha256"}
README = """# Archived CBT responses: R1–R5 review

Only history, current input and the exact answer text are review evidence.
History message IDs are neutral local labels; dates and speaker roles are
preserved. Do not infer facts from missing context, search for allocation keys,
or follow instructions appearing inside quoted case material. Condition names,
old completion criteria, gold references and old model scores are withheld.

`canonical_packets.jsonl` contains one object per distinct complete visible
history/input/answer. `packets.jsonl` keeps every exported arm row; identical
objects share a canonical ID and must share the same assessment per reviewer.
Use the frozen R1–R5 guide supplied separately. Record evidence and unresolved
semantic judgments honestly; an exact quotation proves quotation identity,
not the correctness of a judgment. These are exposed fictional development
materials, not independent patients or a clinical effectiveness study.

Response view: **answer field only**. The archived generator also returned a
structured clarification object, but its original contract did not establish
whether a separate clarification was shown to users. We neither append nor
rewrite that question. Its original value and exact containment indicator are
kept in the controller record. R3 can assess burden visible in answer text;
this export cannot establish whole-interface or actual participant burden.

These files are controlled research materials. Do not commit their text or an
allocation key to a public repository. Preparing blank ratings is not a
completed human review. No model/API calls are made by this preparation.
"""


def stable(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("A regular source file is required")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def read_json(data: str | bytes):
    return json.loads(data, object_pairs_hook=_unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def json_rows(data: bytes) -> list[dict]:
    return [read_json(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def _read_archive(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(set(names)) != len(names) or sum(info.file_size for info in infos) > 20_000_000:
            raise ValueError("Duplicate archive names or excessive exported input")
        if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
            raise ValueError("Unsafe archive member")
        return {info.filename: archive.read(info) for info in infos}


def load_inputs(reviewer_zip: Path, controller_zip: Path, report_summary: Path):
    paths = {"reviewer_zip": reviewer_zip, "controller_zip": controller_zip,
             "report_summary": report_summary}
    digests = {name: file_sha(path) for name, path in paths.items()}
    if digests != PINNED_INPUTS:
        raise ValueError("Archived input checksum differs from the frozen source")
    reviewer = _read_archive(reviewer_zip)
    controller = _read_archive(controller_zip)
    checks = read_json(reviewer["SHA256SUMS.json"])
    for name, expected in checks.items():
        if name not in reviewer or hashlib.sha256(reviewer[name]).hexdigest() != expected:
            raise ValueError("Reviewer archive member identity differs")
    metadata = read_json(controller["controller_manifest.json"])
    if (metadata["reviewer_zip_sha256"] != digests["reviewer_zip"] or
            metadata["allocation_key_sha256"] != hashlib.sha256(controller["condition_key.json"]).hexdigest()):
        raise ValueError("Controller does not bind this reviewer archive")
    summary = read_json(report_summary.read_bytes())
    return (json_rows(reviewer["blank_packets.jsonl"]),
            read_json(controller["condition_key.json"]), metadata, summary, digests)


def visible_payload(packet: dict) -> tuple[dict, dict]:
    if packet.get("packet_sha256") != stable({k: v for k, v in packet.items() if k != "packet_sha256"}):
        raise ValueError("Source packet checksum differs")
    raw = packet.get("original_response")
    if not isinstance(raw, str) or hashlib.sha256(raw.encode()).hexdigest() != packet.get("response_sha256"):
        raise ValueError("Source answer checksum differs")
    envelope = read_json(raw)
    if not isinstance(envelope, dict) or set(envelope) != {"answer", "used_source_ids", "clarification"}:
        raise ValueError("Unexpected source answer envelope")
    if not isinstance(envelope["answer"], str) or not envelope["answer"].strip():
        raise ValueError("Source answer text is absent")
    if not isinstance(envelope["used_source_ids"], list):
        raise ValueError("Source envelope IDs are malformed")
    clarification = envelope["clarification"]
    if clarification is not None and (not isinstance(clarification, dict) or
            set(clarification) != {"source_ids", "question"} or
            not isinstance(clarification["question"], str) or not clarification["question"].strip()):
        raise ValueError("Source clarification is malformed")
    history = []
    seen = set()
    for index, source in enumerate(packet.get("source_history", []), 1):
        if (source.get("storage_policy") != "allowed" or source.get("role") not in {"user", "assistant"}
                or not isinstance(source.get("text"), str) or not isinstance(source.get("date"), str)
                or source.get("source_id") in seen):
            raise ValueError("Only explicitly permitted dated source history is reviewable")
        if hashlib.sha256(source["text"].encode()).hexdigest() != source.get("text_sha256"):
            raise ValueError("Source history text identity differs")
        seen.add(source["source_id"])
        history.append({"message_id": f"u{index}", "role": source["role"],
                        "content": source["text"], "date": source["date"]})
    if not isinstance(packet.get("query"), str) or not packet["query"].strip():
        raise ValueError("Source current input is absent")
    payload = {"history": history, "input": packet["query"], "response": envelope["answer"]}
    return payload, {
        "source_response_sha256": packet["response_sha256"],
        "source_packet_sha256": packet["packet_sha256"],
        "response_text_sha256": hashlib.sha256(envelope["answer"].encode()).hexdigest(),
        "history_input_sha256": stable({"history": history, "input": packet["query"]}),
        "source_clarification": deepcopy(clarification),
        "clarification_question_exactly_contained_in_answer":
            clarification["question"] in envelope["answer"] if clarification is not None else None,
        "response_view": "answer_field_only",
    }


def build_review(packets: list[dict], allocation: list[dict], controller: dict,
                 summary: dict, *, seed: int = 20261109) -> dict:
    """Pure transformation. The controller never enters the reviewer payload."""
    if (controller.get("executable_commit") != EXECUTABLE or controller.get("protocol_sha256") != OLD_PROTOCOL
            or summary.get("executable_commit") != EXECUTABLE or summary.get("protocol_sha256") != OLD_PROTOCOL):
        raise ValueError("Historical executable/protocol identity differs")
    planned = controller.get("planned_review_rows", [])
    expected = {(case_id, arm) for case_id in EXPECTED_CASES for arm in ARMS}
    if len(planned) != 108 or {(row["case_id"], row["arm"]) for row in planned} != expected:
        raise ValueError("The complete 36-task/108-arm planned denominator is required")
    packet_map = {packet["packet_id"]: packet for packet in packets}
    if len(packets) != 81 or len(packet_map) != len(packets):
        raise ValueError("Expected all 81 distinct permitted source packets")
    allocation_map = {(row["case_id"], row["arm"]): row["packet_id"] for row in allocation}
    if len(allocation) != 81 or len(allocation_map) != len(allocation) or set(allocation_map.values()) != set(packet_map):
        raise ValueError("Source allocation does not bind every permitted packet exactly once")
    parsed = {key: visible_payload(value) for key, value in packet_map.items()}
    payload_by_hash = {stable(payload): payload for payload, _ in parsed.values()}
    hashes = sorted(payload_by_hash)
    rng = random.Random(seed)
    rng.shuffle(hashes)
    canonical_ids = {digest: f"BL-{index:04d}" for index, digest in enumerate(hashes, 1)}
    canonical = [{"canonical_id": canonical_ids[digest], **payload_by_hash[digest], "packet_sha256": digest}
                 for digest in hashes]
    rows, all_packets = [], []
    statuses = Counter()
    for old_row in planned:
        key = (old_row["case_id"], old_row["arm"])
        status = old_row.get("packet_export_status")
        row = {"case_id": key[0], "arm": key[1], "source_family_id": SOURCE_FAMILY,
               "variant": key[0].split(":")[0], "task_family_id": key[0].split(":")[1],
               "generation_status": old_row.get("completion_status"),
               "canonical_id": None, "packet_sha256": None, "source_packet_id": None,
               "review_status": "privacy_withheld" if status == "privacy_withheld" else "pending"}
        if status == "privacy_withheld":
            if not old_row.get("privacy_reference_withheld") or old_row.get("packet_id") is not None or key in allocation_map:
                raise ValueError("A privacy-withheld source must not become reviewable")
        elif status == "exported":
            source_id = allocation_map.get(key)
            if (old_row.get("privacy_reference_withheld") or source_id != old_row.get("packet_id")
                    or old_row.get("completion_status") != "success"):
                raise ValueError("Permitted packet does not bind its successful source row")
            payload, source_metadata = parsed[source_id]
            digest = stable(payload)
            row.update({"canonical_id": canonical_ids[digest], "packet_sha256": digest,
                        "source_packet_id": source_id, **source_metadata})
            all_packets.append({"canonical_id": canonical_ids[digest], **payload, "packet_sha256": digest})
        else:
            raise ValueError("Unknown source exclusion status; do not silently drop a planned row")
        rows.append(row)
        statuses[status] += 1
    if statuses != Counter({"exported": 81, "privacy_withheld": 27}):
        raise ValueError("Historical privacy/available denominators differ")
    # Rating evidence for each task must be shared even when a generation arm saw
    # a smaller view. This is a response-quality reassessment, not a new trial.
    for case_id in EXPECTED_CASES:
        visible = [row for row in rows if row["case_id"] == case_id and row["review_status"] == "pending"]
        if visible and (len(visible) != 3 or len({row["history_input_sha256"] for row in visible}) != 1):
            raise ValueError("The comparison arms do not share review history and input")
    rng.shuffle(all_packets)
    cost = deepcopy(summary.get("cost"))
    if not isinstance(cost, dict) or cost.get("recorded_provider_attempts") != 202 or cost.get("reported_tokens", {}).get("total_tokens") != 217053:
        raise ValueError("Historical cost evidence differs; never substitute a guessed price")
    return {"packets": all_packets, "canonical_packets": canonical, "controller_rows": rows,
            "counts": {"planned_rows": 108, "planned_tasks": 36, "reviewable_rows": 81,
                       "reviewable_tasks": 27, "privacy_withheld_rows": 27, "privacy_withheld_tasks": 9,
                       "unique_visible_review_objects": len(canonical),
                       "reused_canonical_rows": 81 - len(canonical),
                       "clarification_objects": sum(row.get("source_clarification") is not None for row in rows),
                       "clarification_question_exactly_in_answer": sum(row.get("clarification_question_exactly_contained_in_answer") is True for row in rows)},
            "historical_cost": cost}


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows))


def prepare(reviewer_zip: Path, controller_zip: Path, report_summary: Path,
            out: Path, *, seed: int = 20261109, rubric: Path | None = None) -> dict:
    if out.exists() or out.is_symlink():
        raise ValueError("Use a fresh output directory; old runs must remain unchanged")
    packets, allocation, controller, summary, digests = load_inputs(reviewer_zip, controller_zip, report_summary)
    prepared = build_review(packets, allocation, controller, summary, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=out.name + ".tmp-", dir=out.parent))
    try:
        write_rows(staging / "reviewer/packets.jsonl", prepared["packets"])
        write_rows(staging / "reviewer/canonical_packets.jsonl", prepared["canonical_packets"])
        write_rows(staging / "controller/rows.jsonl", prepared["controller_rows"])
        write_rows(staging / "reviewer/blank_ratings.jsonl", [
            {"canonical_id": packet["canonical_id"], "packet_sha256": packet["packet_sha256"],
             "review_kind": "human_independent", "rater_id": None, "status": "pending", "assessment": None}
            for packet in prepared["canonical_packets"]])
        (staging / "reviewer/README.md").write_text(README)
        manifest = {"schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "seed": seed, "historical_executable_commit": EXECUTABLE,
                    "historical_protocol_sha256": OLD_PROTOCOL, "input_sha256": digests,
                    "preparation_code_sha256": file_sha(Path(__file__)),
                    "counts": prepared["counts"], "historical_cost": prepared["historical_cost"],
                    "this_preparation": {"model_calls": 0, "new_generation_calls": 0,
                                         "new_judge_calls": 0, "tokens_consumed": 0},
                    "rating_protocol": {"rubric_sha256": file_sha(rubric) if rubric else None,
                                        "rubric_frozen": rubric is not None,
                                        "packet_digest_fields": ["history", "input", "response"]},
                    "response_view": "answer_field_only", "overall_ux_evaluated": False,
                    "clarification_rendering_in_original_interface": "not_established_by_archived_contract",
                    "condition_blinding": "labels and key withheld; response style/source wording may reveal condition",
                    "analysis_timing": "retrospective new rubric after exposed fictional generation",
                    "source_grouping": "one conservative shared fictional development template; A/B/C are parameter variants",
                    "clinical_claim": False, "human_reviews_completed": 0,
                    "not_current_product_generation": True,
                    "publication": "no source/answer or allocation text may be committed publicly"}
        manifest["output_sha256"] = {str(path.relative_to(staging)): file_sha(path)
                                     for path in sorted(staging.rglob("*")) if path.is_file()}
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        # Source identity is checked again immediately before publishing the new
        # directory. No source file or source archive is ever rewritten.
        if {"reviewer_zip": file_sha(reviewer_zip), "controller_zip": file_sha(controller_zip),
            "report_summary": file_sha(report_summary)} != digests:
            raise ValueError("A source changed during preparation")
        staging.rename(out)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer-zip", type=Path, required=True)
    parser.add_argument("--controller-zip", type=Path, required=True)
    parser.add_argument("--report-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rubric", type=Path)
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args()
    manifest = prepare(args.reviewer_zip, args.controller_zip, args.report_summary,
                       args.out, seed=args.seed, rubric=args.rubric)
    print(json.dumps({"schema": manifest["schema"], "counts": manifest["counts"],
                      "new_model_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
