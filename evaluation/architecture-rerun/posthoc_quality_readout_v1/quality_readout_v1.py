"""Post-hoc, zero-call readout. Original strict results are never overwritten.

The only score-carrier conversion is dimension_scores.Rn -> Rn. Values,
types, references, dispositions and reasons are never repaired or inferred.
Evidence validity below is structural source binding, not semantic validation.
"""
from hashlib import sha256
from collections import Counter
import json

VERSION = "cbt-quality-posthoc-readout-v1"
DIMENSIONS = ("R1", "R2", "R3", "R4", "R5")
EVIDENCE_FIELDS = {"dimension", "source_refs", "response_refs", "disposition", "reason"}
ARMS = ("no_rag", "raw_history_rag", "managed_memory_rag")


def _empty(status):
    return {d: {"rating_status": status, "rating": None, "rating_source_path": None,
        "evidence_status": status, "evidence_errors": [], "evidence_source_path": None}
        for d in DIMENSIONS}


def _evidence(row, dimension, rating, manifest):
    """The original per-row requirements, without cross-dimension rejection."""
    errors = []
    if type(row) is not dict or set(row) != EVIDENCE_FIELDS or row.get("dimension") != dimension:
        return ["ordered_exact_evidence_fields_required"]
    reason = row["reason"]
    if type(reason) is not str or not reason.strip() or len(reason) > 8000:
        errors.append("bounded_explanation_required")
    refs_valid = True
    for key, channel in (("source_refs", "sources"), ("response_refs", "responses")):
        refs = row[key]
        if (type(refs) is not list or len(refs) > 12
                or any(type(r) is not str or r not in manifest[channel] for r in refs)
                or len(set(refs)) != len(refs)):
            errors.append(key + ":distinct_known_references_required")
            refs_valid = False
    if not refs_valid:
        return errors
    disposition = row["disposition"]
    if disposition == "supported":
        if not row["source_refs"] or not row["response_refs"]:
            errors.append("observed_judgment_requires_source_and_response_references")
    elif disposition == "omission":
        if not row["source_refs"] or row["response_refs"]:
            errors.append("omission_requires_source_references_and_no_response_reference")
    elif disposition == "no_visible_safety_cue":
        if dimension != "R5" or rating != "pass" or not row["source_refs"]:
            errors.append("explicit_no_visible_cue_only_for_R5_pass")
    else:
        errors.append("unknown_evidence_disposition")
    return errors


def readout(text, packet, api):
    """Read unchanged ratings and diagnose each evidence row independently.

    ``api`` is the frozen quality_span_judge.py runpy dictionary. No transport
    is constructed. Invalid JSON, duplicate keys and Markdown fences stay
    unparsed. Missing numerical ratings are never inferred from evidence.
    """
    original = api["validate_review"](text, packet)
    result = {"version": VERSION, "raw_sha256": sha256(text.encode()).hexdigest(),
        "original_strict_status": original["status"], "parser_status": "unparsed",
        "transforms": [], "dimensions": _empty("unparsed"),
        "evidence_interpretation": "structural binding under original row rules; not semantic verification"}
    try:
        value = api["quality"]._strict_json(text)
    except (ValueError, TypeError, KeyError):
        return result
    if type(value) is not dict:
        result["parser_status"] = "unsupported_shape"
        return result
    result["dimensions"] = _empty("missing")
    if set(value) == {"status", "reason"} and value.get("status") == "unresolved":
        result["parser_status"] = "unresolved"
        return result
    evidence_map = None
    if set(value) == set(DIMENSIONS) | {"evidence"}:
        ratings, evidence, prefix = value, value["evidence"], "$"
        result["parser_status"] = "flat"
    elif (set(value) == {"dimension_scores", "evidence"}
            and type(value["dimension_scores"]) is dict
            and set(value["dimension_scores"]) == set(DIMENSIONS)):
        ratings, evidence, prefix = value["dimension_scores"], value["evidence"], "$.dimension_scores"
        result["parser_status"] = "dimension_scores"
        result["transforms"] = [{"operation": "unwrap_explicit_score_carrier",
                                  "source_path": prefix, "destination_path": "$"}]
    elif (set(value) == set(DIMENSIONS) and all(type(value[d]) is dict
            and set(value[d]) == EVIDENCE_FIELDS for d in DIMENSIONS)):
        # This observed shape carries evidence only. There is no score carrier.
        ratings, evidence, prefix, evidence_map = {}, None, None, value
        result["parser_status"] = "evidence_only_map"
    else:
        result["parser_status"] = "unsupported_shape"
        return result
    _, manifest = api["segment_packet"](packet)
    result["evidence_container_status"] = ("evidence_only_map" if evidence_map is not None else
        "five_rows" if type(evidence) is list and len(evidence) == 5 else "invalid_container")
    for index, dimension in enumerate(DIMENSIONS):
        record = result["dimensions"][dimension]
        if dimension in ratings:
            rating = ratings[dimension]
            valid = (type(rating) is str and rating in {"pass", "fail"}) if dimension == "R5" else (
                type(rating) is int and 1 <= rating <= 5)
            record.update(rating_status="available" if valid else "invalid",
                rating=rating if valid else None, rating_source_path=prefix + "." + dimension)
        if evidence_map is not None:
            row, path = evidence_map[dimension], "$." + dimension
        elif type(evidence) is list and index < len(evidence):
            row, path = evidence[index], "$.evidence[" + str(index) + "]"
        else:
            if type(evidence) is not list:
                record.update(evidence_status="invalid", evidence_errors=["evidence_list_required"])
            continue
        errors = _evidence(row, dimension, record["rating"], manifest)
        if type(evidence) is list and sum(type(r) is dict and r.get("dimension") == dimension for r in evidence) > 1:
            errors.append("duplicate_dimension_evidence")
        record.update(evidence_status="invalid" if errors else "valid", evidence_errors=errors,
                      evidence_source_path=path)
    return result


def summarize_readouts(rows, mapping):
    """Full-grid coverage and same-case comparisons; never impute a score.

    Numerical summaries are conditional on explicitly reported availability.
    Per-row evidence binding is a separate restriction, not a new rating.
    Repeats never enter primary counts or matched comparisons.
    """
    primary = [r for r in rows if r["primary"]]
    by_packet = {r["packet_id"]: r for r in primary}
    by_case = {}
    if len(by_packet) != len(primary):
        raise ValueError("duplicate primary readout packet")
    expected_packets = [r["packet_id"] for r in mapping if r["packet_id"] is not None]
    if len(set(expected_packets)) != len(expected_packets) or set(by_packet) != set(expected_packets):
        raise ValueError("original mapped packets and primary readouts differ")
    for item in mapping:
        case = by_case.setdefault(item["case_id"], {})
        if item["arm"] not in ARMS or item["arm"] in case:
            raise ValueError("duplicate or unknown original arm")
        case[item["arm"]] = item
    if any(set(case) != set(ARMS) or len({r["source_family"] for r in case.values()}) != 1
           for case in by_case.values()):
        raise ValueError("complete original three-arm source-family grid required")

    def state(item, dimension):
        return None if item["packet_id"] is None else by_packet[item["packet_id"]]["dimensions"][dimension]

    def number(value):
        if type(value) is str and value in {"pass", "fail"}:
            return int(value == "pass")
        if type(value) is int and 1 <= value <= 5:
            return value
        raise ValueError("only unchanged explicit in-range ratings can be summarized")

    def reasons(item, dimension, qualified):
        record = state(item, dimension)
        if record is None:
            return [item["arm"] + ":generation_not_received"]
        missing = []
        if record["rating_status"] != "available":
            missing.append(item["arm"] + ":rating_" + record["rating_status"])
        if qualified and record["evidence_status"] != "valid":
            missing.append(item["arm"] + ":evidence_" + record["evidence_status"])
        return missing

    arms = {}
    for arm in ARMS:
        items = [case[arm] for case in by_case.values()]
        arm_result = {"planned": len(items), "generation_received": sum(i["packet_id"] is not None for i in items),
            "original_strict_completed": sum(i["packet_id"] is not None and
                by_packet[i["packet_id"]]["original_strict_status"] == "completed" for i in items), "dimensions": {}}
        for dimension in DIMENSIONS:
            dimension_result = {field + "_counts": dict(Counter(
                state(i, dimension)[field] if state(i, dimension) is not None else "generation_not_received"
                for i in items)) for field in ("rating_status", "evidence_status")}
            for name, qualified in (("raw_rating", False), ("rating_and_evidence_valid", True)):
                values = [state(i, dimension)["rating"] for i in items if not reasons(i, dimension, qualified)]
                dimension_result[name] = {"planned": len(items), "available": len(values),
                    "coverage": len(values) / len(items) if items else None,
                    "distribution": dict(Counter(str(v) for v in values)),
                    "conditional_mean": sum(map(number, values)) / len(values) if values else None}
            arm_result["dimensions"][dimension] = dimension_result
        arms[arm] = arm_result

    contrasts = {}
    for name, baseline in (("primary_managed_minus_raw", ARMS[1]), ("auxiliary_managed_minus_no_rag", ARMS[0])):
        contrast = {"baseline": baseline, "treatment": ARMS[2], "planned_pairs": len(by_case), "dimensions": {}}
        for dimension in DIMENSIONS:
            contrast["dimensions"][dimension] = {}
            for condition, qualified in (("raw_rating", False), ("rating_and_evidence_valid", True)):
                paired, missing_reasons, planned_families = [], Counter(), Counter()
                for case in by_case.values():
                    left, right = case[baseline], case[ARMS[2]]
                    family = left["source_family"]
                    planned_families[family] += 1
                    unavailable = reasons(left, dimension, qualified) + reasons(right, dimension, qualified)
                    if unavailable:
                        missing_reasons.update(unavailable)
                    else:
                        a, b = number(state(left, dimension)["rating"]), number(state(right, dimension)["rating"])
                        paired.append((family, a, b))
                differences = [b - a for _, a, b in paired]
                family_differences = {}
                for family, a, b in paired:
                    family_differences.setdefault(family, []).append(b - a)
                contrast["dimensions"][dimension][condition] = {
                    "planned_pairs": len(by_case), "matched_pairs": len(paired),
                    "missing_pairs": len(by_case) - len(paired),
                    "matched_pair_coverage": len(paired) / len(by_case) if by_case else None,
                    "baseline_conditional_mean": sum(a for _, a, _ in paired) / len(paired) if paired else None,
                    "treatment_conditional_mean": sum(b for _, _, b in paired) / len(paired) if paired else None,
                    "mean_paired_difference": sum(differences) / len(differences) if differences else None,
                    "wins": sum(d > 0 for d in differences), "losses": sum(d < 0 for d in differences),
                    "ties": sum(d == 0 for d in differences), "missing_reason_counts": dict(missing_reasons),
                    "planned_source_families": len(planned_families), "matched_source_families": len(family_differences),
                    "fully_matched_source_families": sum(len(ds) == planned_families[f] for f, ds in family_differences.items()),
                    "conditional_family_equal_difference": sum(sum(ds) / len(ds) for ds in family_differences.values())
                        / len(family_differences) if family_differences else None,
                    "units": "pass_rate_difference" if dimension == "R5" else "rating_points",
                    "missing_reason_counts_may_overlap": True}
        contrasts[name] = contrast
    return {"primary_planned": len(mapping), "planned_cases": len(by_case), "arms": arms, "contrasts": contrasts,
        "interpretation": "post-hoc unchanged model ratings with explicit coverage; not clinical efficacy; conditional means do not cover missing rows"}


def main():
    """Read a completed frozen grade directory; emit a separate JSON to stdout."""
    import argparse
    from pathlib import Path
    import runpy
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo")
    parser.add_argument("grade")
    args = parser.parse_args()
    repo, grade = Path(args.repo).resolve(), Path(args.grade).resolve()
    sys.path[:0] = [str(repo / p) for p in ("evaluation/architecture-rerun",
        "evaluation/cbt-memory-effect", "modules/agent-integration")]
    api = runpy.run_path(str(repo / "evaluation/architecture-rerun/quality_span_judge.py"))
    folder, protocol, artifacts = api["_load"](grade, False)
    intents = {r["key"]: r for r in api["rows"](folder / "intents.jsonl")}
    outcomes = {r["key"]: r for r in api["rows"](folder / "outcomes.jsonl")}
    api["_checked_reviews"](artifacts, api["rows"](folder / "reviews.jsonl"), intents, outcomes)
    packets = {p["case_id"]: p for p in artifacts["cases.json"]}
    rows = []
    for item in artifacts["plan.json"]:
        purpose = "quality_v2" if item["repeat_of"] is None else "quality_v2_repeat:" + item["item_id"]
        key = api["digest"]({"purpose": purpose, "payload": item["payload"]})
        receipt = outcomes.get(key)
        if receipt is None or receipt.get("status") != "received":
            raise ValueError("offline readout requires all planned original received receipts")
        rows.append({"item_id": item["item_id"], "packet_id": item["case_id"],
            "primary": item["repeat_of"] is None, "request_key": key,
            **readout(receipt["text"], packets[item["case_id"]], api)})
    summary = summarize_readouts(rows, artifacts["controller_arm_mapping.json"])
    print(json.dumps({"version": VERSION, "posthoc": True, "model_calls": 0,
        "original_results_overwritten": False, "protocol_sha256": protocol["protocol_sha256"],
        "parser_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "repeat_and_primary_rows": len(rows), "summary": summary, "rows": rows}, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    main()
