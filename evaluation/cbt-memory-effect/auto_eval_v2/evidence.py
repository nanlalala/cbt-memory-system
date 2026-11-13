"""Evidence accounting for adjudicated CBT-memory claims (no model or IO).

This module checks *bindings*, not semantic entailment. An external rule or
assessor must supply each verdict. Literal quotation and a matching fact ID do
not establish that a claim is true. ``label_provenance`` is mandatory on both
the reference and adjudication; model-proposed labels are never called gold.

Gold schema: case_id, source_family, label_provenance, required_facts
[{id, allowed_source_ids}], forbidden_or_stale_fact_ids.
Source schema: source_id, source_order, role, content, retained, use_allowed.
Adjudication: status (answered/abstained/generation_failed), label_provenance,
claims [{claim_id, fact_id|null, source_ids, verdict, evidence}]. Each evidence
entry has source_id and a nonempty literal quote. Supported claims need at
least one binding. A fact ID denotes one *specific state-bearing proposition*,
so an old and corrected fact must have different IDs.
"""
from __future__ import annotations

from collections import Counter
import math
import random

VERSION = "cbt-source-evidence-accounting-v2"
PROVENANCE = frozenset({"published_gold", "rule_verified", "model_proposed"})
VERDICTS = frozenset({"supported", "unsupported", "contradicted", "stale"})
STATUSES = frozenset({"answered", "abstained", "generation_failed"})


def _keys(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(label + "_schema")


def _text(value, label):
    if type(value) is not str or not value.strip():
        raise ValueError(label + "_nonempty_string")
    return value


def _strings(values, label, *, nonempty=False):
    if type(values) is not list or (nonempty and not values):
        raise ValueError(label + "_list")
    if any(type(v) is not str or not v.strip() for v in values) or len(set(values)) != len(values):
        raise ValueError(label + "_unique_strings")
    return set(values)


def validate_gold(gold):
    _keys(gold, {"case_id", "source_family", "label_provenance", "required_facts",
                 "forbidden_or_stale_fact_ids"}, "reference")
    _text(gold["case_id"], "case_id")
    _text(gold["source_family"], "source_family")
    if type(gold["label_provenance"]) is not str or gold["label_provenance"] not in PROVENANCE:
        raise ValueError("reference_provenance")
    if type(gold["required_facts"]) is not list:
        raise ValueError("required_facts_list")
    required = {}
    for fact in gold["required_facts"]:
        _keys(fact, {"id", "allowed_source_ids"}, "required_fact")
        ident = _text(fact["id"], "fact_id")
        if ident in required:
            raise ValueError("duplicate_required_fact")
        required[ident] = _strings(fact["allowed_source_ids"], "allowed_source_ids", nonempty=True)
    stale = _strings(gold["forbidden_or_stale_fact_ids"], "stale_fact_ids")
    if set(required) & stale:
        raise ValueError("same_fact_cannot_be_required_and_stale")
    return required, stale


def validate_sources(sources, cutoff_order):
    if type(cutoff_order) is not int or cutoff_order < 0 or type(sources) is not list:
        raise ValueError("sources_and_cutoff")
    ledger = {}
    for source in sources:
        _keys(source, {"source_id", "source_order", "role", "content", "retained", "use_allowed"}, "source")
        sid = _text(source["source_id"], "source_id")
        if sid in ledger:
            raise ValueError("duplicate_source_id")
        if type(source["source_order"]) is not int or source["source_order"] < 0:
            raise ValueError("source_order_integer")
        if type(source["role"]) is not str or source["role"] not in {"user", "assistant"}:
            raise ValueError("source_role")
        _text(source["content"], "source_content")
        if type(source["retained"]) is not bool or type(source["use_allowed"]) is not bool:
            raise ValueError("source_permissions_boolean")
        ledger[sid] = source
    return ledger


def _eligible(sid, ledger, cutoff_order):
    if sid not in ledger:
        raise ValueError("unknown_source_id")
    s = ledger[sid]
    if s["source_order"] > cutoff_order or not s["retained"] or not s["use_allowed"]:
        raise ValueError("ineligible_future_deleted_or_denied_source")
    return s


def score_memory(gold, adjudication, sources, cutoff_order, *, retrieved_source_ids=()):
    """Score explicit judgments, deduplicating repeated state-bearing fact IDs.

    A stale ID overrides an erroneous ``supported`` verdict. Repetition cannot
    inflate precision: each non-null fact ID is one unit, with its worst verdict
    retained. Fact IDs outside the required set can count as correct extra facts
    only under an explicit supported verdict plus eligible literal evidence.
    No new semantic judgments are made here. Unsafe/future evidence raises an
    error; a caller must retain that error in its planned denominator.
    """
    required, stale = validate_gold(gold)
    ledger = validate_sources(sources, cutoff_order)
    for allowed in required.values():
        for sid in allowed:
            _eligible(sid, ledger, cutoff_order)
    retrieved = set()
    for sid in retrieved_source_ids:
        _text(sid, "retrieved_source_id")
        _eligible(sid, ledger, cutoff_order)
        retrieved.add(sid)
    _keys(adjudication, {"status", "label_provenance", "claims"}, "adjudication")
    status = adjudication["status"]
    if (type(status) is not str or status not in STATUSES
            or type(adjudication["label_provenance"]) is not str
            or adjudication["label_provenance"] not in PROVENANCE):
        raise ValueError("adjudication_status_or_provenance")
    if type(adjudication["claims"]) is not list:
        raise ValueError("claims_list")
    if status != "answered" and adjudication["claims"]:
        raise ValueError("nonanswer_cannot_have_claims")
    seen, units, observations = set(), {}, []
    rank = {"supported": 0, "unsupported": 1, "contradicted": 2, "stale": 3}
    for claim in adjudication["claims"]:
        _keys(claim, {"claim_id", "fact_id", "source_ids", "verdict", "evidence"}, "claim")
        ident = _text(claim["claim_id"], "claim_id")
        if ident in seen:
            raise ValueError("duplicate_claim_id")
        seen.add(ident)
        fact = claim["fact_id"]
        if fact is not None:
            _text(fact, "claim_fact_id")
        if type(claim["verdict"]) is not str or claim["verdict"] not in VERDICTS:
            raise ValueError("claim_verdict")
        source_ids = _strings(claim["source_ids"], "claim_source_ids")
        for sid in source_ids:
            _eligible(sid, ledger, cutoff_order)
        if type(claim["evidence"]) is not list:
            raise ValueError("claim_evidence_list")
        quoted_ids = set()
        for ev in claim["evidence"]:
            _keys(ev, {"source_id", "quote"}, "evidence")
            sid = _text(ev["source_id"], "evidence_source_id")
            source = _eligible(sid, ledger, cutoff_order)
            quote = _text(ev["quote"], "literal_quote")
            if quote not in source["content"] or sid not in source_ids:
                raise ValueError("literal_evidence_binding")
            quoted_ids.add(sid)
        verdict = claim["verdict"]
        if verdict == "supported" and not quoted_ids:
            raise ValueError("supported_claim_needs_literal_evidence")
        if fact in stale:
            verdict = "stale"
        binding_mismatch = verdict == "supported" and fact in required and not (quoted_ids & required[fact])
        if binding_mismatch:
            verdict = "unsupported"
        key = ("fact", fact) if fact is not None else ("unmapped_claim", ident)
        if key not in units or rank[verdict] > rank[units[key]]:
            units[key] = verdict
        observations.append({"claim_id": ident, "fact_id": fact, "effective_verdict": verdict,
                             "allowed_fact_binding_mismatch": binding_mismatch})
    counts = Counter(units.values())
    correct_required = sorted(f for f in required if units.get(("fact", f)) == "supported")
    precision = counts["supported"] / len(units) if units else None
    recall = len(correct_required) / len(required) if required else None
    if required and not units:
        f1 = 0.0
    elif precision is not None and recall is not None:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    else:
        f1 = None
    gold_sources = set().union(*required.values()) if required else set()
    return {
        "version": VERSION, "case_id": gold["case_id"], "source_family": gold["source_family"],
        "status": status, "complete_answer": status == "answered",
        "generation_completed": status != "generation_failed",
        "reference_provenance": gold["label_provenance"],
        "adjudication_provenance": adjudication["label_provenance"],
        "model_proposed_labels_present": "model_proposed" in (gold["label_provenance"], adjudication["label_provenance"]),
        "semantic_verdicts_supplied_not_inferred": True,
        "metrics": {"precision": precision, "recall": recall, "f1": f1,
                    "retrieval_recall": sum(bool(allowed & retrieved) for allowed in required.values()) / len(required)
                        if required else None,
                    "allowed_evidence_id_coverage": len(gold_sources & retrieved) / len(gold_sources)
                        if gold_sources else None},
        "retrieval_recall_definition": "required facts with at least one allowed evidence source retrieved / required facts",
        "allowed_evidence_id_coverage_definition": "retrieved allowed IDs / all allowed IDs; alternatives are not all necessary",
        "counts": {"raw_claims": len(seen), "deduplicated_facts_or_unmapped_claims": len(units),
                   "supported": counts["supported"], "unsupported": counts["unsupported"],
                   "contradicted": counts["contradicted"], "stale": counts["stale"],
                   "required": len(required), "correct_required": len(correct_required),
                   "omitted_required": len(required) - len(correct_required)},
        "correct_required_fact_ids": correct_required,
        "omitted_required_fact_ids": sorted(set(required) - set(correct_required)),
        "no_evidence_abstention": status == "abstained" and not required,
        "claim_observations": observations,
    }


def exact_quote_score(expected_quote, expected_source_id, answer_quote, answer_source_id):
    """Whitespace-only normalization; punctuation, case and ID stay exact."""
    _text(expected_quote, "expected_quote")
    _text(expected_source_id, "expected_source_id")
    if type(answer_quote) is not str or type(answer_source_id) is not str:
        return {"quote_correct": False, "source_correct": False, "fully_correct": False}
    quote = " ".join(expected_quote.split()) == " ".join(answer_quote.split())
    source = expected_source_id == answer_source_id
    return {"quote_correct": quote, "source_correct": source, "fully_correct": quote and source}


def _cluster_interval(differences, *, draws, seed):
    groups = {}
    for family, diff in differences:
        groups.setdefault(family, []).append(diff)
    estimate = sum(diff for _, diff in differences) / len(differences) if differences else None
    if len(groups) < 2:
        return {"mean_difference": estimate, "ci95": None, "source_families": len(groups),
                "interval_status": "at_least_two_source_families_required"}
    rng, keys, samples = random.Random(seed), sorted(groups), []
    for _ in range(draws):
        values = [v for key in rng.choices(keys, k=len(keys)) for v in groups[key]]
        samples.append(sum(values) / len(values))
    samples.sort()
    def quantile(p):
        at = p * (len(samples) - 1)
        lo, hi = math.floor(at), math.ceil(at)
        return samples[lo] + (samples[hi] - samples[lo]) * (at - lo)
    return {"mean_difference": estimate, "ci95": [quantile(.025), quantile(.975)],
            "source_families": len(groups), "interval_status": "family_cluster_percentile_bootstrap"}


def paired_report(planned_cases, scored_rows, *, metric="f1", arms=("raw", "managed"),
                  bootstrap_draws=1000, seed=20261112):
    """Keep the complete fixed question grid, including missing outputs.

    ``planned_cases`` are gold-schema references. ``scored_rows`` contain the
    output of score_memory plus ``arm``. Conditional paired means are distinct
    from completion-adjusted means (missing answer/assessment contributes zero
    to recall/F1 only, never to precision). Nothing here scores CBT or safety.
    """
    if metric not in {"precision", "recall", "f1"} or len(arms) != 2 or len(set(arms)) != 2:
        raise ValueError("paired_metric_or_arms")
    if type(bootstrap_draws) is not int or not 100 <= bootstrap_draws <= 100000 or type(seed) is not int:
        raise ValueError("bootstrap_configuration")
    planned = {}
    for gold in planned_cases:
        required, _ = validate_gold(gold)
        ident = gold["case_id"]
        if ident in planned:
            raise ValueError("duplicate_planned_case")
        planned[ident] = (gold["source_family"], bool(required))
    found = {}
    for row in scored_rows:
        ident, arm = row.get("case_id"), row.get("arm")
        if ident not in planned or arm not in arms or (ident, arm) in found:
            raise ValueError("unplanned_or_duplicate_scored_row")
        if (row.get("source_family") != planned[ident][0] or type(row.get("status")) is not str
                or row["status"] not in STATUSES):
            raise ValueError("scored_row_identity_or_status")
        value = row.get("metrics", {}).get(metric)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("metric_must_be_finite_proportion_or_null")
        if row["status"] == "generation_failed" and value is not None and value != 0:
            raise ValueError("generation_failure_cannot_have_positive_metric")
        found[ident, arm] = row
    statuses = {arm: Counter() for arm in arms}
    conditional, adjusted, details = [], [], []
    for ident, (family, has_required) in planned.items():
        values = []
        for arm in arms:
            row = found.get((ident, arm))
            statuses[arm][row["status"] if row else "missing_row"] += 1
            values.append(row["metrics"].get(metric) if row else None)
        if (all(v is not None for v in values)
                and all(found[ident, a]["status"] != "generation_failed" for a in arms)):
            conditional.append((family, values[1] - values[0]))
        adjusted_values = None
        if metric in {"recall", "f1"} and has_required:
            adjusted_values = [v if v is not None else 0.0 for v in values]
            adjusted.append((family, adjusted_values[1] - adjusted_values[0]))
        details.append({"case_id": ident, "source_family": family,
                        "values_in_arm_order": values, "completion_adjusted_values": adjusted_values})
    return {"version": VERSION, "metric": metric, "arm_order": list(arms),
            "difference_direction": arms[1] + " minus " + arms[0], "planned_questions": len(planned),
            "planned_arm_rows": 2 * len(planned), "observed_rows": len(found),
            "statuses_by_arm": {a: dict(statuses[a]) for a in arms},
            "valid_paired_questions": len(conditional),
            "conditional_completed_pairs": _cluster_interval(conditional, draws=bootstrap_draws, seed=seed),
            "completion_adjusted_applicable_questions": len(adjusted),
            "completion_adjusted_fixed_grid": _cluster_interval(adjusted, draws=bootstrap_draws, seed=seed)
                if metric != "precision" else None,
            "completion_adjusted_interpretation": "missing outputs score zero for applicable recall/F1; not completed effect evidence",
            "bootstrap_draws": bootstrap_draws, "bootstrap_seed": seed, "per_question": details}
