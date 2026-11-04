"""Read-only, per-request evidence diagnostics, separate from frozen scoring.

Annotations are scoring-side artifacts, never inputs to preparation/generation.
An observed evidence gap is not a causal explanation of an incorrect answer.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from run_objective_comparison import (_load_run_locked, _request_outcomes_locked,
                                      exclusive_directory, stable_digest)
from score_objective_comparison import deterministic_score

VERSION = "evidence-stage-observations-v1"


def validate_annotations(rows, protocol, prompts, inventory):
    """Bind exact source spans and complete alternative evidence sets to a run.

    Independent review is declared metadata, not authenticated by this script.
    No answer-string search is used to invent evidence labels.
    """
    sources = {r["document_id"]: r for r in inventory}
    subjects = {p["case_id"]: p["subject_id"] for p in prompts}
    result = {}
    for row in rows:
        cid = row["case_id"]
        if cid in result or cid not in subjects:
            raise ValueError("duplicate or unknown annotation case")
        if row.get("protocol_sha256") != protocol["protocol_sha256"]:
            raise ValueError("annotation protocol mismatch")
        if row.get("coverage") != "complete_sufficient_alternatives":
            raise ValueError("partial evidence labels cannot diagnose missing evidence")
        if row.get("annotation_origin") not in {"released", "human_review", "developer_fixture", "model_proposed"}:
            raise ValueError("declare annotation origin")
        if not isinstance(row.get("reviewer_id"), str) or not row["reviewer_id"].strip():
            raise ValueError("annotation reviewer identity required")
        alternatives = row.get("alternatives")
        if not isinstance(alternatives, list) or not alternatives:
            raise ValueError("nonempty sufficient evidence alternatives required")
        for alternative in alternatives:
            if not isinstance(alternative, list) or not alternative:
                raise ValueError("empty evidence alternative")
            for span in alternative:
                source = sources.get(span["document_id"])
                if source is None or source["subject_id"] != subjects[cid]:
                    raise ValueError("unknown or cross-subject evidence source")
                text = source["source"]["text"]
                if span.get("source_text_sha256") != hashlib.sha256(text.encode()).hexdigest():
                    raise ValueError("evidence source checksum mismatch")
                start, end = span.get("start"), span.get("end")
                if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
                    raise ValueError("invalid evidence span")
        result[cid] = row
    return result


def observe(prompt, annotation):
    trace = prompt["trace"]
    stages = {"extraction": "external_index_not_semantically_validated" if trace.get("selection_representation") == "semantic_index_unverified_hints" else "not_run_quote_control",
              "semantic_update": "not_run"}
    base = {"stages": stages, "causal_attribution": "unresolved",
            "clarification_requests": 0 if trace.get("interactive_clarification") is False else None,
            "user_effort_seconds": None, "repeated_questions": None,
            "burden_boundary": "static QA has no participant; zero interactive checks is not measured user effort"}
    if annotation is None:
        return {**base, "evidence_observation": "unresolved_no_evidence_annotation",
                "annotation_origin": None}
    included = set(trace.get("included_short_ids", []) + trace.get("included_long_ids", []))
    retrieved = set(trace.get("retrieved_ids", [])) | set(trace.get("included_short_ids", []))
    stored = {r["document_id"] for r in trace.get("storage", []) if r.get("action") == "use_as_history"}
    alternatives = [{span["document_id"] for span in alt} for alt in annotation["alternatives"]]
    # Whole document inclusion is only a quote-control observation. It is not
    # valid for semantic summaries or future renderers that truncate documents.
    if trace.get("encoding") != "released_source_quotes_no_semantic_extraction":
        raise ValueError("diagnostic requires whole-source quotation encoding")
    if any(alt <= included for alt in alternatives):
        observation = "sufficient_annotated_source_documents_in_context"
    elif prompt["arm"] == "short_term":
        observation = "annotated_evidence_outside_recent_context"
    elif any(alt <= retrieved for alt in alternatives):
        observation = "annotated_evidence_retrieved_but_context_incomplete"
    else:
        observation = "annotated_evidence_not_fully_retrieved"
    stages.update({"retrieval": "not_applicable_recent_only" if prompt["arm"] == "short_term" else
                   any(alt <= retrieved for alt in alternatives),
                   "context": any(alt <= included for alt in alternatives),
                   "storage": "selected_documents_only_not_full_inventory" if prompt["arm"] == "managed_memory_rag" else
                   "not_applicable_raw_history"})
    return {**base, "evidence_observation": observation, "annotation_origin": annotation["annotation_origin"],
            "reviewer_id": annotation["reviewer_id"], "annotation_sha256": stable_digest(annotation),
            "selected_stored_source_ids": sorted(stored),
            "evidence_boundary": "source-document coverage under declared labels; neither semantic sufficiency nor causal blame is proven"}


def diagnose(run_dir, annotations=()):
    run_dir = Path(run_dir)
    with exclusive_directory(run_dir):
        protocol, prompts, references, completed = _load_run_locked(run_dir)
        inventory = [json.loads(line) for line in (run_dir / "inventory.jsonl").read_text().splitlines()]
        labels = validate_annotations(annotations, protocol, prompts, inventory)
        outcomes = _request_outcomes_locked(run_dir, protocol, prompts, completed)
        refs = {r["case_id"]: r for r in references}
        rows = []
        for prompt in prompts:
            key = prompt["case_id"], prompt["arm"]
            response = completed.get(key)
            score, parsed = deterministic_score(response["hypothesis"], refs[key[0]]) if response else (None, None)
            rows.append({"case_id": key[0], "arm": key[1], "group_id": prompt["group_id"],
                         "task": prompt["task"], "status": outcomes[key]["status"], "score": score,
                         "parsed_prediction": parsed, "canonical_request_key": prompt["canonical_request_key"],
                         **observe(prompt, labels.get(key[0]))})
        terminal = all(r["status"] in {"success", "exhausted_failure"} for r in rows)
        return {"version": VERSION, "protocol_sha256": protocol["protocol_sha256"],
                "complete": terminal, "final_effectiveness_result": False,
                "request_counts": dict(Counter(r["status"] for r in rows)),
                "annotated_cases": len(labels), "rows": len(rows),
                "observations": dict(Counter(r["evidence_observation"] for r in rows)),
                "boundary": "diagnostic ledger; use frozen scorer for final cohort accuracy; no causal or clinical claim"}, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.resolve().is_relative_to(args.run_dir.resolve()):
        raise ValueError("diagnostics must be outside the frozen run")
    labels = [json.loads(s) for s in args.annotations.read_text().splitlines()] if args.annotations else []
    summary, rows = diagnose(args.run_dir, labels)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "stage_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "per_request_diagnostics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
