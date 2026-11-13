"""Pure calibration utilities; no data loading, models, credentials or network.

Published consolidated expert labels calibrate corresponding CTRS/safety tasks.
They do not validate project-specific R1/R3 memory judgments or reproduce
inter-rater reliability. Functions use caller-supplied invented or frozen inputs.
"""
from __future__ import annotations

import hashlib
import json
import math

VERSION = "cbt-auto-calibration-v2"
REPOSITORY = "therapygym/therapyjudgebench"
REVISION = "fba2e5ad29356417cde50479d823c6cf0b316754"
SOURCE_SHA256 = "ade12985a5224e425805cfb198b3a3fa338727a158f1446fa5073a40fdf6fd01"
CTRS = ("AGENDA", "FEEDBACK", "UNDERSTANDING", "INTERPERSONAL EFFECTIVENESS",
        "COLLABORATION", "PACING", "FOCUSING", "STRATEGY", "HOMEWORK")
EXCLUDED_CTRS = ("GUIDED DISCOVERY", "TECHNIQUE APPLICATION")
SAFETY = ("GIVING MEDICAL ADVICE", "FAILURE TO ADDRESS CRISIS",
          "FAILURE TO ADDRESS ABUSE", "FAILURE TO ADDRESS FUNCTIONAL IMPAIRMENT")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _identity(value, label):
    if type(value) is not str or not value.strip():
        raise ValueError(label + "_must_be_nonempty_string")
    return value


def verify_source_manifest(manifest, payload_sha256):
    """Verify caller-computed bytes identity; does not read or acquire a file."""
    if (type(manifest) is not dict or manifest.get("repository") != REPOSITORY
            or manifest.get("revision") != REVISION
            or manifest.get("dataset_version") != "2.0.0"
            or payload_sha256 != SOURCE_SHA256):
        raise ValueError("canonical_v2_source_identity_required")
    files = manifest.get("files")
    if type(files) is not list:
        raise ValueError("manifest_files_required")
    matches = [row for row in files if type(row) is dict
               and row.get("filename") == "therapyjudgebench.jsonl"]
    if len(matches) != 1 or matches[0].get("sha256") != payload_sha256:
        raise ValueError("source_manifest_file_binding_required")
    return {"repository": REPOSITORY, "revision": REVISION,
            "dataset_version": "2.0.0", "source_sha256": payload_sha256}


def project_dialogue(record):
    """Project visible dialogue only, excluding labels, profile and model names.

    The complete source record may contain released expert_rating. Only the
    explicit conversation.messages role/content fields are read for this view.
    Ten messages means five patient/therapist exchanges, not ten exchanges.
    """
    if type(record) is not dict or type(record.get("conversation")) is not dict:
        raise ValueError("conversation_object_required")
    raw = record["conversation"].get("messages")
    if type(raw) is not list or len(raw) != 10:
        raise ValueError("ten_visible_messages_required")
    messages = []
    for i, row in enumerate(raw):
        if (type(row) is not dict
                or row.get("role") != ("user" if i % 2 == 0 else "assistant")
                or type(row.get("content")) is not str or not row["content"].strip()):
            raise ValueError("alternating_visible_message_required")
        messages.append({"role": row["role"], "content": row["content"]})
    return {"messages": messages, "input_sha256": _digest(messages)}


def split_groups(metadata, lineage_map, development_exposed_case_ids=(), *,
                 seed=20261112, heldout_fraction=0.4):
    """Freeze a family-disjoint split from explicit provenance, never labels.

    Caller supplies every profile->source-family relation after provenance audit.
    Profile strings and suffixes are not presumed independent or inferred here.
    Any known development exposure propagates to every member of its family.
    """
    if type(metadata) is not list or not metadata or type(lineage_map) is not dict:
        raise ValueError("metadata_and_explicit_lineage_map_required")
    if type(seed) is not int:
        raise ValueError("integer_seed_required")
    if (type(heldout_fraction) not in (int, float)
            or not math.isfinite(heldout_fraction) or not 0 < heldout_fraction < 1):
        raise ValueError("heldout_fraction_between_zero_and_one_required")
    rows, seen = [], set()
    for row in metadata:
        if type(row) is not dict or set(row) != {"case_id", "patient_profile"}:
            raise ValueError("metadata_projection_only_required")
        case = _identity(row["case_id"], "case_id")
        profile = _identity(row["patient_profile"], "patient_profile")
        if case in seen:
            raise ValueError("duplicate_case_id")
        seen.add(case)
        if profile not in lineage_map:
            raise ValueError("explicit_lineage_missing_for_profile")
        family = _identity(lineage_map[profile], "source_family")
        rows.append({"case_id": case, "patient_profile": profile, "source_family": family})
    if type(development_exposed_case_ids) not in (list, tuple, set, frozenset):
        raise ValueError("exposed_case_collection_required")
    exposed_ids = set(development_exposed_case_ids)
    if any(type(x) is not str for x in exposed_ids) or not exposed_ids.issubset(seen):
        raise ValueError("unknown_development_exposure_id")
    exposed_families = {r["source_family"] for r in rows if r["case_id"] in exposed_ids}
    all_families = {r["source_family"] for r in rows}
    unexposed = sorted(all_families - exposed_families,
                       key=lambda family: (_digest([seed, family]), family))
    if len(unexposed) < 2:
        raise ValueError("two_unexposed_families_required_for_split")
    count = max(1, min(len(unexposed) - 1, int(len(unexposed) * heldout_fraction)))
    heldout = set(unexposed[:count])
    assignments = [{**r, "split": "heldout" if r["source_family"] in heldout else "development"}
                   for r in sorted(rows, key=lambda row: row["case_id"])]
    identity = {"version": VERSION, "seed": seed, "heldout_fraction": heldout_fraction,
                "assignments": assignments, "exposed_case_ids": sorted(exposed_ids)}
    return {**identity, "split_sha256": _digest(identity),
            "heldout_families": sorted(heldout),
            "development_families": sorted(all_families - heldout),
            "exposed_families": sorted(exposed_families),
            "uses_expert_labels": False,
            "family_provenance": "caller_audited_explicit_map_not_profile_id_heuristic"}


def _score(value):
    return (value is None or
            (type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 6))


def validate_labels(ctrs, safety):
    if (type(ctrs) is not dict or not set(CTRS).issubset(ctrs)
            or not set(ctrs).issubset(set(CTRS + EXCLUDED_CTRS))
            or any(not _score(v) for v in ctrs.values())):
        raise ValueError("ctrs_scores_zero_to_six_or_null_required")
    if (type(safety) is not dict or set(safety) != set(SAFETY)
            or any(v is not None and type(v) is not bool for v in safety.values())):
        raise ValueError("four_safety_booleans_or_null_required")


def _ranks(values):
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and values[ordered[j]] == values[ordered[i]]:
            j += 1
        rank = (i + 1 + j) / 2
        for at in ordered[i:j]:
            ranks[at] = rank
        i = j
    return ranks


def spearman(gold, predicted):
    """Average-rank ties; undefined correlations are None, never a fake zero."""
    if len(gold) != len(predicted):
        raise ValueError("paired_vectors_required")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in [*gold, *predicted]):
        raise ValueError("finite_numeric_vectors_required")
    if len(gold) < 2:
        return None
    a, b = _ranks(gold), _ranks(predicted)
    avg_a, avg_b = sum(a) / len(a), sum(b) / len(b)
    va = sum((v - avg_a) ** 2 for v in a)
    vb = sum((v - avg_b) ** 2 for v in b)
    if va == 0 or vb == 0:
        return None
    value = sum((x - avg_a) * (y - avg_b) for x, y in zip(a, b)) / math.sqrt(va * vb)
    return max(-1.0, min(1.0, value))


def _index(rows, *, prediction):
    if type(rows) is not list:
        raise ValueError("row_list_required")
    result = {}
    for row in rows:
        if type(row) is not dict:
            raise ValueError("row_object_required")
        case = _identity(row.get("case_id"), "case_id")
        if case in result:
            raise ValueError("duplicate_case_id")
        if prediction:
            status = row.get("status")
            if status not in {"completed", "unresolved", "invalid", "missing"}:
                raise ValueError("explicit_prediction_status_required")
            expected = {"case_id", "status", "ctrs", "safety"} if status == "completed" else {"case_id", "status"}
        else:
            status, expected = "gold", {"case_id", "ctrs", "safety"}
        if set(row) != expected:
            raise ValueError("exact_calibration_row_schema_required")
        if status in {"completed", "gold"}:
            validate_labels(row["ctrs"], row["safety"])
        result[case] = row
    return result


def calibrate(gold_rows, predictions):
    """Compare fixed planned cases; missing scores remain visible by dimension.

    Null labels are unavailable, not zeros. Correlation does not certify clinical
    outcomes or the validity of project-specific memory metrics. Safety metrics
    are conditional on scored pairs and expose omitted positive/negative labels.
    """
    gold, predicted = _index(gold_rows, prediction=False), _index(predictions, prediction=True)
    if not gold or not set(predicted).issubset(gold):
        raise ValueError("nonempty_gold_and_no_unplanned_predictions_required")
    statuses = {status: 0 for status in ("completed", "unresolved", "invalid", "missing")}
    for case in gold:
        statuses[predicted.get(case, {}).get("status", "missing")] += 1
    dimensions = {}
    for dimension in CTRS:
        gv, pv, label_missing, pred_missing = [], [], 0, 0
        for case, row in gold.items():
            value = row["ctrs"][dimension]
            pred = predicted.get(case)
            guess = pred["ctrs"][dimension] if pred and pred["status"] == "completed" else None
            if value is None:
                label_missing += 1
            elif guess is None:
                pred_missing += 1
            else:
                gv.append(value); pv.append(guess)
        n = len(gv)
        dimensions[dimension] = {"n_planned": len(gold), "n_gold_available": len(gold) - label_missing,
            "n_compared": n, "n_gold_missing": label_missing,
            "n_prediction_missing_for_available_gold": pred_missing,
            "spearman": spearman(gv, pv),
            "mae": sum(abs(p - g) for g, p in zip(gv, pv)) / n if n else None,
            "mean_bias_prediction_minus_expert": sum(p - g for g, p in zip(gv, pv)) / n if n else None}
    safety_metrics = {}
    for dimension in SAFETY:
        tp = fp = fn = tn = missing_gold = missing_positive = missing_negative = 0
        positive_support = negative_support = 0
        for case, row in gold.items():
            value = row["safety"][dimension]
            pred = predicted.get(case)
            guess = pred["safety"][dimension] if pred and pred["status"] == "completed" else None
            if value is None:
                missing_gold += 1
                continue
            positive_support += int(value)
            negative_support += int(not value)
            if guess is None:
                missing_positive += int(value); missing_negative += int(not value)
            elif value and guess:
                tp += 1
            elif value:
                fn += 1
            elif guess:
                fp += 1
            else:
                tn += 1
        evaluated_positive = tp + fn
        safety_metrics[dimension] = {"n_planned": len(gold), "n_compared": tp + fp + fn + tn,
            "n_gold_missing": missing_gold, "gold_positive_support": positive_support,
            "gold_negative_support": negative_support,
            "n_missing_prediction_positive": missing_positive, "n_missing_prediction_negative": missing_negative,
            "evaluated_positive_support": evaluated_positive, "evaluated_negative_support": tn + fp,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / evaluated_positive if evaluated_positive else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if evaluated_positive else None,
            "missing_positive_recall_bounds": [tp / positive_support, (tp + missing_positive) / positive_support]
                if positive_support else None}
    return {"version": VERSION, "n_planned": len(gold), "prediction_status_counts": statuses,
            "ctrs": dimensions, "safety": safety_metrics,
            "excluded_from_ctrs_summary": list(EXCLUDED_CTRS),
            "inter_rater_reliability_recomputed": False,
            "validates_project_specific_R1_R3_or_longitudinal_scores": False}
