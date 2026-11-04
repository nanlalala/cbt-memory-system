"""Read-only source/version-bound diagnostics, separate from frozen scoring.

Annotations are scoring-side artifacts, never inputs to preparation/generation.
Evidence coverage is checked against actual rendered quotations, not IDs alone.
An observed evidence gap is not a causal explanation of an incorrect answer.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import run_objective_comparison as runner
from score_objective_comparison import deterministic_score

VERSION = "evidence-stage-observations-v2"
stable_digest = runner.stable_digest


def visible_source_views(cases, prompts, inventory):
    """Reconstruct each exact prepared visible history or refuse ambiguity.

    A document ID alone does not identify a version or a question's visibility.
    The original history digest is the final authority for reconstruction; an
    unknown scope/prefix combination is never guessed into an evidence label.
    """
    case_map = {c["case_id"]: c for c in cases}
    if len(case_map) != len(cases) or set(case_map) != {p["case_id"] for p in prompts}:
        raise ValueError("case visibility metadata differs from prompt grid")
    sources, by_subject = {}, {}
    for item in inventory:
        did, subject, row = item["document_id"], item["subject_id"], item["source"]
        identity = {"subject_id": subject, "source": row}
        if did in sources:
            if sources[did] != identity:
                raise ValueError("conflicting source versions share one document ID")
            continue
        if runner._document({"subject_id": subject}, row).document_id != did:
            raise ValueError("inventory source document identity mismatch")
        sources[did] = identity
        by_subject.setdefault(subject, []).append(row)
    views = {}
    for cid, case in case_map.items():
        subject = case["subject_id"]
        if any(p["subject_id"] != subject for p in prompts if p["case_id"] == cid):
            raise ValueError("case and prompt subjects differ")
        history = by_subject.get(subject, [])
        scope = case.get("history_scope_id")
        if case.get("dataset_key") == "memoryagentbench_conflict" and scope:
            history = [r for r in history if r["session_id"] == scope + ":released-sequence"]
        elif case.get("dataset_key") == "personamem_v1":
            context = case.get("source_context_id")
            end = case.get("visible_context_end_exclusive")
            if not context or type(end) is not int or end <= 0:
                raise ValueError("released exclusive source prefix metadata required")
            history = [r for r in history if r["session_id"] == f"persona32k:{context}:released-sequence" and r["order"] < end]
        admitted, _ = runner._eligible({"history": history, "as_of": case.get("as_of")})
        digest = stable_digest([subject, admitted])
        if digest != case.get("history_inventory_sha256"):
            raise ValueError("cannot reconstruct exact case-visible history/version")
        view = {}
        for row in admitted:
            doc = runner._document(case, row)
            view[doc.document_id] = {"source": row, "document": doc,
                                     "source_text_sha256": hashlib.sha256(row["text"].encode()).hexdigest()}
        views[cid] = {"history_inventory_sha256": digest, "sources": view}
    return views


def validate_annotations(rows, protocol, prompts, inventory, cases=None, *, views=None):
    """Bind annotations to case visibility, exact source version and span.

    Independent review and sufficient alternatives are declared metadata, not
    authenticated by this script. No answer-string search invents source labels.
    """
    if views is None:
        if cases is None:
            raise ValueError("case visibility metadata required for evidence annotations")
        views = visible_source_views(cases, prompts, inventory)
    result = {}
    for row in rows:
        cid = row["case_id"]
        if cid in result or cid not in views:
            raise ValueError("duplicate or unknown annotation case")
        if row.get("protocol_sha256") != protocol["protocol_sha256"]:
            raise ValueError("annotation protocol mismatch")
        view = views[cid]
        if row.get("history_inventory_sha256") != view["history_inventory_sha256"]:
            raise ValueError("annotation case-visible history identity mismatch")
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
                source = view["sources"].get(span["document_id"])
                if source is None:
                    raise ValueError("evidence source is outside current case-visible history")
                text = source["source"]["text"]
                if span.get("source_text_sha256") != source["source_text_sha256"]:
                    raise ValueError("evidence source checksum/version mismatch")
                start, end = span.get("start"), span.get("end")
                if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
                    raise ValueError("invalid evidence span")
        result[cid] = row
    return result


def _actual_context(prompt, view):
    """Verify each rendered source version and return covered raw-text ranges.

    Known quote and source-span encodings are supported. Unknown encodings yield
    unknown coverage, rather than a positive answer based only on traced IDs.
    """
    encoding = prompt["trace"].get("encoding")
    if encoding not in {"released_source_quotes_no_semantic_extraction", "exact_source_spans_no_summary"}:
        return None
    messages = prompt.get("messages")
    if not isinstance(messages, list):
        raise ValueError("actual prompt messages required for evidence coverage")
    user_messages = [m["content"] for m in messages if m.get("role") == "user"]
    if len(user_messages) != 1:
        raise ValueError("expected one actual question/history message")
    try:
        root = ET.fromstring("<root>" + user_messages[0] + "</root>")
    except ET.ParseError as exc:
        raise ValueError("cannot verify actual rendered source context") from exc
    included, spans, found = {}, {}, set()
    for channel, trace_field in (("short_history", "included_short_ids"), ("older_history", "included_long_ids")):
        nodes = root.findall(channel)
        if len(nodes) != 1:
            raise ValueError("actual prompt history channel missing or repeated")
        ids = []
        for node in nodes[0]:
            if list(node):
                raise ValueError("source quotation contains unescaped nested markup")
            did = node.get("id")
            if not did or did in found:
                raise ValueError("duplicate or missing actual context source identity")
            found.add(did)
            ids.append(did)
            source_id = did if encoding == "released_source_quotes_no_semantic_extraction" else node.get("source_id")
            source = view["sources"].get(source_id)
            if source is None:
                raise ValueError("actual prompt source is outside case-visible history")
            row, doc = source["source"], source["document"]
            if node.get("date") != doc.date or node.get("speaker") != doc.speaker:
                raise ValueError("actual prompt source role/date version mismatch")
            left = len(row["text"]) - len(row["text"].lstrip())
            if encoding == "released_source_quotes_no_semantic_extraction":
                if node.tag != "turn" or node.text != doc.text:
                    raise ValueError("actual prompt quotation differs from source version")
                start, end = 0, len(doc.text)
            else:
                if node.tag != "source_span":
                    raise ValueError("unexpected actual source-span markup")
                try:
                    start, end = int(node.attrib["start"]), int(node.attrib["end"])
                except (KeyError, ValueError) as exc:
                    raise ValueError("invalid actual source-span offsets") from exc
                if not 0 <= start < end <= len(doc.text) or node.text != doc.text[start:end]:
                    raise ValueError("actual context span differs from source version")
                if did != stable_digest([doc.scope_id, doc.document_id, doc.session_id, doc.date, doc.speaker, start, end, node.text]):
                    raise ValueError("actual source-span identity mismatch")
            included.setdefault(source_id, []).append((left + start, left + end))
            spans[did] = source_id
        expected = prompt["trace"].get(trace_field)
        if not isinstance(expected, list) or ids != expected:
            raise ValueError("actual context identities/order differ from trace")
    return {"covered_ranges": included, "rendered_to_source": spans}


def _span_covered(span, ranges):
    """Allow exact adjacent rendered spans jointly to cover one annotation."""
    start, end = span["start"], span["end"]
    for left, right in sorted(ranges.get(span["document_id"], [])):
        if right <= start:
            continue
        if left > start:
            return False
        start = max(start, right)
        if start >= end:
            return True
    return False


def observe(prompt, annotation, *, view=None):
    trace = prompt["trace"]
    representation = trace.get("selection_representation")
    stages = {"extraction": "external_index_not_semantically_validated" if representation == "semantic_index_unverified_hints" else
              "syntactic_source_slot_view_not_semantic_extraction" if representation == "source_only_latest_slot_view" else "not_run_quote_control",
              "semantic_update": "syntactic_benchmark_view_not_product_memory_update" if representation == "source_only_latest_slot_view" else "not_run"}
    base = {"stages": stages, "causal_attribution": "unresolved",
            "clarification_requests": 0 if trace.get("interactive_clarification") is False else None,
            "user_effort_seconds": None, "repeated_questions": None,
            "burden_boundary": "static QA has no participant; zero interactive checks is not measured user effort"}
    if annotation is None:
        return {**base, "evidence_observation": "unresolved_no_evidence_annotation", "annotation_origin": None}
    if view is None:
        raise ValueError("verified per-case source view required for coverage")
    actual = _actual_context(prompt, view)
    if actual is None:
        return {**base, "evidence_observation": "unresolved_unsupported_context_encoding",
                "annotation_origin": annotation["annotation_origin"], "context_coverage": None}
    alternatives = annotation["alternatives"]
    context_complete = any(all(_span_covered(span, actual["covered_ranges"]) for span in alt) for alt in alternatives)
    if trace.get("encoding") == "released_source_quotes_no_semantic_extraction":
        retrieved = set(trace.get("retrieved_ids", [])) | set(trace.get("included_short_ids", []))
        if not retrieved <= set(view["sources"]):
            raise ValueError("retrieval trace includes source outside case-visible history")
        retrieval_complete = any({s["document_id"] for s in alt} <= retrieved for alt in alternatives)
    else:
        # Included spans authenticate what reached context. Traces do not retain
        # every dropped chunk body, so do not infer upstream span sufficiency.
        retrieval_complete = True if context_complete else None
    stored = {r["document_id"] for r in trace.get("storage", []) if r.get("action") == "use_as_history"}
    if context_complete:
        observation = "sufficient_annotated_source_spans_in_actual_context"
    elif prompt["arm"] == "short_term":
        observation = "annotated_evidence_outside_recent_context"
    elif retrieval_complete is True:
        observation = "annotated_evidence_retrieved_but_context_incomplete"
    elif retrieval_complete is None:
        observation = "unresolved_retrieved_span_versions_not_fully_recorded"
    else:
        observation = "annotated_evidence_not_fully_retrieved"
    stages.update({"retrieval": "not_applicable_recent_only" if prompt["arm"] == "short_term" else retrieval_complete,
                   "context": context_complete,
                   "storage": "selected_documents_only_not_full_inventory" if prompt["arm"] == "managed_memory_rag" else "not_applicable_raw_history"})
    return {**base, "evidence_observation": observation, "annotation_origin": annotation["annotation_origin"],
            "reviewer_id": annotation["reviewer_id"], "annotation_sha256": stable_digest(annotation),
            "history_inventory_sha256": view["history_inventory_sha256"],
            "selected_stored_source_ids": sorted(stored),
            "evidence_boundary": "byte-verified actual source spans under declared labels; neither independent semantic sufficiency nor causal blame is proven"}


def _reader_for(run_dir):
    protocol_path = Path(run_dir) / "protocol.json"
    if protocol_path.is_symlink():
        raise ValueError("read-only diagnosis refuses a symlink protocol")
    version = json.loads(protocol_path.read_text())["protocol_version"]
    if version == runner.VERSION:
        return runner
    import v3_run_objective_comparison as v3
    if version == v3.VERSION:
        return v3
    raise ValueError("unsupported protocol; use its matching pinned read-only reader")


def diagnose(run_dir, annotations=()):
    # The combined reader snapshots cases/inventory/prompts/journals together.
    # It neither creates a lock file nor writes a diagnostic mirror into input.
    state = _reader_for(run_dir).read_run_state_readonly(Path(run_dir))
    protocol, prompts, references = state["protocol"], state["prompts"], state["references"]
    completed, outcomes = state["completed"], state["outcomes"]
    views = visible_source_views(state["frozen_rows"]["cases.jsonl"], prompts, state["frozen_rows"]["inventory.jsonl"])
    labels = validate_annotations(annotations, protocol, prompts, (), views=views)
    refs = {r["case_id"]: r for r in references}
    rows = []
    for prompt in prompts:
        key = prompt["case_id"], prompt["arm"]
        response = completed.get(key)
        score, parsed = deterministic_score(response["hypothesis"], refs[key[0]]) if response else (None, None)
        rows.append({"case_id": key[0], "arm": key[1], "group_id": prompt["group_id"],
                     "task": prompt["task"], "status": outcomes[key]["status"], "score": score,
                     "parsed_prediction": parsed, "canonical_request_key": prompt["canonical_request_key"],
                     **observe(prompt, labels.get(key[0]), view=views[key[0]])})
    terminal = all(r["status"] in {"success", "exhausted_failure"} for r in rows)
    canonical = {}
    for prompt in prompts:
        key = prompt["case_id"], prompt["arm"]
        identity = prompt["canonical_request_key"]
        value = (outcomes[key]["status"], outcomes[key]["attempt_count"])
        if identity in canonical and canonical[identity] != value:
            raise ValueError("shared canonical request has inconsistent arm states")
        canonical[identity] = value
    return {"version": VERSION, "diagnostic_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "protocol_sha256": protocol["protocol_sha256"], "complete": terminal, "final_effectiveness_result": False,
            "arm_row_counts": dict(Counter(r["status"] for r in rows)),
            "canonical_request_counts": dict(Counter(value[0] for value in canonical.values())),
            "recorded_provider_attempt_intents": sum(value[1] for value in canonical.values()),
            "unique_canonical_requests": len(canonical), "annotated_cases": len(labels), "rows": len(rows),
            "observations": dict(Counter(r["evidence_observation"] for r in rows)),
            "boundary": "read-only diagnostic ledger; use frozen scorer for final cohort accuracy; no causal or clinical claim"}, rows


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
