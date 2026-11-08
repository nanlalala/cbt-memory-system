#!/usr/bin/env python3
"""Validate and describe R1–R5 single-answer reviews without making model calls.

Scores are descriptive, source-bound assessments, not clinical competence or
outcome measurements. Literal quotations bind an assessment to supplied text;
they do not independently establish its semantic validity or an error's cause.
Conditions are restored only here, after single-answer review. All planned rows
remain visible, and each declared review kind/rater is analyzed separately.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics


VERSION = "cbt-response-r1-r5-descriptive-analysis-v1"
DIMENSIONS = ("R1", "R2", "R3", "R4", "R5")
ARMS = ("short_term", "raw_history_rag", "managed_memory_rag")
REVIEW_KINDS = {"model_assisted", "human_independent"}
RATING_STATUSES = {"completed", "pending", "technical_failure", "unresolved"}
ID_RE = re.compile(r"[A-Za-z0-9_.:/-]{1,256}\Z")
HASH_RE = re.compile(r"[0-9a-f]{64}\Z")


def stable(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def read_json(data: str | bytes):
    return json.loads(data, object_pairs_hook=_unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def _read(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("A regular artifact file is required")
    body = path.read_bytes()
    if len(body) > 100_000_000:
        raise ValueError("Excessive review artifact")
    return body


def read_rows(path: Path) -> list[dict]:
    body = _read(path)
    if body and not body.endswith(b"\n"):
        raise ValueError("Incomplete JSONL tail; preserve it for explicit recovery")
    values = [read_json(line) for line in body.splitlines() if line.strip()]
    if any(not isinstance(value, dict) for value in values):
        raise ValueError("JSONL rows must be objects")
    return values


def _id(value, label="identifier") -> str:
    if not isinstance(value, str) or not ID_RE.fullmatch(value):
        raise ValueError("Invalid " + label)
    return value


def _text(value, label, *, maximum=200_000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError("Missing or excessive " + label)
    return value


def validate_packet(packet: dict) -> dict:
    if not isinstance(packet, dict) or set(packet) != {
        "canonical_id", "history", "input", "response", "packet_sha256"
    }:
        raise ValueError("Packet must contain only anonymous visible review fields")
    _id(packet["canonical_id"], "canonical ID")
    history = packet["history"]
    if not isinstance(history, list) or len(history) > 20_000:
        raise ValueError("Invalid visible history")
    seen = set()
    for message in history:
        if not isinstance(message, dict) or set(message) != {"message_id", "role", "content", "date"}:
            raise ValueError("Unexpected history fields")
        identity = _id(message["message_id"], "message ID")
        if identity in seen or message["role"] not in {"user", "assistant"}:
            raise ValueError("Duplicate message ID or invalid speaker role")
        seen.add(identity)
        _text(message["content"], "history content")
        _text(message["date"], "source date", maximum=100)
    _text(packet["input"], "current input")
    _text(packet["response"], "response")
    payload = {field: packet[field] for field in ("history", "input", "response")}
    if packet["packet_sha256"] != stable(payload):
        raise ValueError("Packet content hash differs")
    return packet


def validate_assessment(assessment: dict, packet: dict) -> dict:
    """Require valid values and five independent, literal evidence bindings.

    This verifies schema and quotations only. It deliberately does not assign
    scores, infer clinical quality or upgrade quotation validity into causality.
    """
    validate_packet(packet)
    if not isinstance(assessment, dict) or set(assessment) != {*DIMENSIONS, "evidence"}:
        raise ValueError("Assessment requires R1–R5 and evidence, without an overall total")
    for dimension in DIMENSIONS[:4]:
        if type(assessment[dimension]) is not int or not 1 <= assessment[dimension] <= 5:
            raise ValueError("R1–R4 must be strict integers from 1 to 5")
    if not isinstance(assessment["R5"], str) or assessment["R5"] not in {"pass", "fail"}:
        raise ValueError("R5 must be pass or fail")
    evidence = assessment["evidence"]
    if not isinstance(evidence, list) or len(evidence) != len(DIMENSIONS):
        raise ValueError("Each dimension needs its own evidence object")
    sources = [packet["input"], *(item["content"] for item in packet["history"])]
    seen = set()
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {
            "dimension", "source_quote", "response_quote", "reason"
        }:
            raise ValueError("Unexpected evidence fields")
        dimension = item["dimension"]
        if not isinstance(dimension, str) or dimension not in DIMENSIONS or dimension in seen:
            raise ValueError("Duplicate or unknown evidence dimension")
        seen.add(dimension)
        quote = _text(item["source_quote"], "source quote", maximum=20_000)
        response_quote = _text(item["response_quote"], "response quote", maximum=20_000)
        _text(item["reason"], "dimension justification", maximum=8_000)
        if not any(quote in source for source in sources):
            raise ValueError("Source quote does not occur in one visible message/input")
        if response_quote not in packet["response"]:
            raise ValueError("Response quote does not occur in the original visible answer")
    return assessment


def validate_rating(record: dict, packet: dict) -> dict:
    if not isinstance(record, dict) or not {
        "canonical_id", "packet_sha256", "review_kind", "rater_id", "status", "assessment"
    } <= set(record) or set(record) - {
        "canonical_id", "packet_sha256", "review_kind", "rater_id", "status", "assessment", "reason"
    }:
        raise ValueError("Unexpected rating envelope")
    validate_packet(packet)
    if record["canonical_id"] != packet["canonical_id"] or record["packet_sha256"] != packet["packet_sha256"]:
        raise ValueError("Rating does not bind the frozen visible packet")
    if (not isinstance(record["review_kind"], str) or record["review_kind"] not in REVIEW_KINDS or
            not isinstance(record["status"], str) or record["status"] not in RATING_STATUSES):
        raise ValueError("Unknown review kind or rating status")
    if record["rater_id"] is None:
        if record["status"] != "pending":
            raise ValueError("Only an unassigned pending form may lack a rater")
    else:
        _id(record["rater_id"], "rater ID")
    if "reason" in record:
        _text(record["reason"], "status reason", maximum=8_000)
    if record["status"] == "completed":
        validate_assessment(record["assessment"], packet)
    elif record["assessment"] is not None:
        raise ValueError("Non-completed review must retain null assessment")
    if record["status"] in {"technical_failure", "unresolved"} and not record.get("reason"):
        raise ValueError("Failure/unresolved status needs a reason")
    return record


def _load_review(review_dir: Path):
    if review_dir.is_symlink():
        raise ValueError("Symlinked review directory")
    manifest = read_json(_read(review_dir / "manifest.json"))
    if not isinstance(manifest, dict) or any((review_dir / name).is_symlink() for name in ("reviewer", "controller")):
        raise ValueError("Invalid manifest or symlinked artifact directory")
    names = ("reviewer/canonical_packets.jsonl", "controller/rows.jsonl")
    for name in names:
        body = _read(review_dir / name)
        if manifest.get("output_sha256", {}).get(name) != hashlib.sha256(body).hexdigest():
            raise ValueError("Frozen review artifact identity differs: " + name)
    packets = {}
    hashes = set()
    for packet in read_rows(review_dir / names[0]):
        validate_packet(packet)
        if packet["canonical_id"] in packets or packet["packet_sha256"] in hashes:
            raise ValueError("Duplicate canonical packet or noncanonical duplicate payload")
        packets[packet["canonical_id"]] = packet
        hashes.add(packet["packet_sha256"])
    rows = read_rows(review_dir / names[1])
    identities = set()
    cases = defaultdict(set)
    used = set()
    for row in rows:
        for field in ("case_id", "source_family_id", "variant", "task_family_id"):
            _id(row.get(field), field)
        identity = (row["case_id"], row.get("arm"))
        if identity in identities or row.get("arm") not in ARMS:
            raise ValueError("Duplicate controller case/arm or unknown condition")
        identities.add(identity)
        cases[row["case_id"]].add(row["arm"])
        if row.get("review_status") == "privacy_withheld":
            if row.get("canonical_id") is not None or row.get("packet_sha256") is not None:
                raise ValueError("A privacy-withheld row cannot be reviewed")
        elif row.get("review_status") == "pending":
            packet = packets.get(row.get("canonical_id"))
            if packet is None or packet["packet_sha256"] != row.get("packet_sha256"):
                raise ValueError("Controller row does not bind a visible packet")
            used.add(row["canonical_id"])
        else:
            raise ValueError("Unknown controller review status")
    if any(arms != set(ARMS) for arms in cases.values()) or used != set(packets):
        raise ValueError("Incomplete condition grid or unbound canonical packets")
    counts = manifest.get("counts", {})
    actual = {"planned_rows": len(rows), "planned_tasks": len(cases),
              "reviewable_rows": sum(row["review_status"] == "pending" for row in rows),
              "privacy_withheld_rows": sum(row["review_status"] == "privacy_withheld" for row in rows),
              "unique_visible_review_objects": len(packets)}
    if any(counts.get(key) != value for key, value in actual.items()):
        raise ValueError("Manifest denominator differs from frozen controller rows")
    return manifest, packets, rows


def _mean(values):
    return statistics.mean(values) if values else None


def _partition(rows, ratings, *, review_kind, rater_id):
    expanded = []
    for row in rows:
        rating = ratings.get(row["canonical_id"]) if row["canonical_id"] is not None else None
        status = "privacy_withheld" if row["review_status"] == "privacy_withheld" else (
            rating["status"] if rating else "pending")
        expanded.append({key: row[key] for key in (
            "case_id", "arm", "source_family_id", "variant", "task_family_id", "canonical_id", "packet_sha256")}
                        | {"status": status, "scores": {d: rating["assessment"][d] for d in DIMENSIONS}
                           if status == "completed" else {d: None for d in DIMENSIONS}})
    arms = {}
    for arm in ARMS:
        selected = [row for row in expanded if row["arm"] == arm]
        completed = [row for row in selected if row["status"] == "completed"]
        statuses = Counter(row["status"] for row in selected)
        arms[arm] = {"planned_rows": len(selected), "status_counts": dict(statuses),
                     "dimensions": {d: {"scored_rows": len(completed),
                                          "mean": _mean([row["scores"][d] for row in completed])}
                                    for d in DIMENSIONS[:4]},
                     "safety": {"scored_rows": len(completed),
                                "pass": sum(row["scores"]["R5"] == "pass" for row in completed),
                                "fail": sum(row["scores"]["R5"] == "fail" for row in completed)}}
    grid = defaultdict(dict)
    for row in expanded:
        grid[row["case_id"]][row["arm"]] = row
    pair_statuses = Counter()
    pairs = []
    for case_id, grid_rows in sorted(grid.items()):
        raw, managed = grid_rows["raw_history_rag"], grid_rows["managed_memory_rag"]
        statuses = {raw["status"], managed["status"]}
        status = next((s for s in ("privacy_withheld", "technical_failure", "unresolved", "pending") if s in statuses), "completed")
        pair_statuses[status] += 1
        if status == "completed":
            pairs.append((raw, managed))
    dimensions = {}
    for dimension in DIMENSIONS[:4]:
        differences = [managed["scores"][dimension] - raw["scores"][dimension] for raw, managed in pairs]
        dimensions[dimension] = {"completed_pairs": len(pairs),
                                 "managed_wins": sum(v > 0 for v in differences),
                                 "raw_wins": sum(v < 0 for v in differences),
                                 "ties": sum(v == 0 for v in differences),
                                 "mean_managed_minus_raw": _mean(differences)}
    safety_pairs = Counter((raw["scores"]["R5"], managed["scores"]["R5"]) for raw, managed in pairs)
    return {"review_kind": review_kind, "rater_id": rater_id,
            "rater_qualification_and_independence_verified": False,
            "unique_submitted_objects": len(ratings),
            "planned_rows": len(rows), "status_counts": dict(Counter(row["status"] for row in expanded)),
            "arms": arms,
            "managed_vs_raw": {"planned_pairs": len(grid), "pair_status_counts": dict(pair_statuses),
                               "dimensions": dimensions,
                               "safety_pairs": {f"raw_{a}__managed_{b}": safety_pairs[(a, b)]
                                                for a in ("pass", "fail") for b in ("pass", "fail")}},
            "rows": expanded}


def analyze(review_dir: Path, ratings_paths=()) -> dict:
    manifest, packets, rows = _load_review(Path(review_dir))
    groups = defaultdict(dict)
    unassigned = set()
    for path in ratings_paths:
        for record in read_rows(Path(path)):
            packet = packets.get(record.get("canonical_id"))
            if packet is None:
                raise ValueError("Rating names an unknown or privacy-withheld object")
            validate_rating(record, packet)
            if record["rater_id"] is None:
                key = (record["review_kind"], record["canonical_id"])
                if key in unassigned:
                    raise ValueError("Duplicate unassigned pending form")
                unassigned.add(key)
                continue
            key = (record["review_kind"], record["rater_id"])
            if record["canonical_id"] in groups[key]:
                raise ValueError("Duplicate rating for the same kind/rater/canonical object")
            groups[key][record["canonical_id"]] = record
    partitions = [_partition(rows, value, review_kind=key[0], rater_id=key[1])
                  for key, value in sorted(groups.items())]
    if not partitions:
        partitions = [_partition(rows, {}, review_kind="unassigned", rater_id=None)]
    return {"schema": VERSION, "preparation_schema": manifest.get("schema"),
            "historical_executable_commit": manifest.get("historical_executable_commit"),
            "historical_protocol_sha256": manifest.get("historical_protocol_sha256"),
            "rubric_sha256": manifest.get("rating_protocol", {}).get("rubric_sha256"),
            "rubric_frozen": manifest.get("rating_protocol", {}).get("rubric_frozen", False),
            "planned_rows": len(rows), "unique_review_objects": len(packets),
            "privacy_withheld_rows": sum(row["review_status"] == "privacy_withheld" for row in rows),
            "source_families": len({row["source_family_id"] for row in rows}),
            "variants": len({row["variant"] for row in rows}),
            "unassigned_pending_forms": len(unassigned),
            "response_view": manifest.get("response_view", "unknown"),
            "overall_ux_evaluated": manifest.get("overall_ux_evaluated", False),
            "historical_cost": manifest.get("historical_cost"),
            "new_generation_calls_by_analysis": 0, "new_judge_calls_by_analysis": 0,
            "partitions": partitions, "clinical_total": None, "comparative_population_interval": None,
            "limitations": [
                "Retrospective reassessment of exposed fictional development outputs; not current product generation.",
                "Conditions are masked during scoring, but answer wording may reveal a condition.",
                "Exact quotations verify binding only, not independent semantic correctness or stage-error causality.",
                "Declared human review does not establish reviewer qualifications or independence.",
                "Raters and review kinds are reported separately; no pooled expert or clinical total is computed.",
                "Related parameter variants and repeated tasks are not independent patients; no population interval is inferred.",
                "Pending, technical failure, unresolved and privacy-withheld rows retain null scores and planned denominators.",
                "Observed answer text alone does not measure actual whole-interface user burden or treatment outcomes.",
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("analyze")
    command.add_argument("--review-dir", type=Path, required=True)
    command.add_argument("--ratings", type=Path, action="append", default=[])
    command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.review_dir, args.ratings)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("Use a new analysis output; do not overwrite previous evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"schema": VERSION, "planned_rows": report["planned_rows"],
                      "unique_review_objects": report["unique_review_objects"],
                      "privacy_withheld_rows": report["privacy_withheld_rows"],
                      "review_partitions": len(report["partitions"]),
                      "new_model_calls": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
