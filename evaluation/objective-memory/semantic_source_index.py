"""Source-bound semantic retrieval indexes; summaries never become reply facts.

Index generation is a separate, paid, logged stage. This module only validates
its artifacts and maps ranked candidates back to eligible original quotations.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace

VERSION = "semantic-source-index-v1"
TYPES = frozenset({"explicit_fact", "preference", "episodic_event", "emotion_record",
                   "goal", "cbt_assignment", "assignment_outcome", "cognitive_pattern", "user_correction"})
SYSTEM_PROMPT = '''Extract retrieval index candidates from the supplied historical source documents.
Return JSON only: {"candidates": [{"source_document_id": "...", "memory_type": "goal",
"summary": "...", "supporting_text": "exact contiguous source quote"}]}.
Use only supplied documents. Preserve negation, uncertainty, time, task completion and corrections.
Do not infer diagnoses. Never mark a summary as confirmed or overwrite historical evidence.
Allowed types: explicit_fact, preference, episodic_event, emotion_record, goal, cbt_assignment,
assignment_outcome, cognitive_pattern, user_correction. Summaries are unverified search hints only.
Do not extract assistant suggestions as user agreements. Return an empty list if unsupported.'''


def validate_index(rows, *, documents, subject_id):
    sources = {d.document_id: d for d in documents}
    checked, seen = [], set()
    for row in rows:
        allowed = {"subject_id", "source_document_id", "source_text_sha256", "memory_type", "summary", "supporting_text"}
        if not isinstance(row, dict) or set(row) != allowed:
            raise ValueError("semantic index has unknown or missing fields")
        if row["subject_id"] != subject_id:
            raise ValueError("cross-subject semantic index")
        doc = sources.get(row["source_document_id"])
        if doc is None:
            # Never let a model/index resurrect deleted, expired, restricted or
            # future sources. Callers pass only the common eligible history.
            raise ValueError("semantic index cites ineligible or missing source")
        if row["source_text_sha256"] != hashlib.sha256(doc.text.encode()).hexdigest():
            raise ValueError("semantic index source checksum mismatch")
        if row["memory_type"] not in TYPES:
            raise ValueError("unsupported semantic type")
        if not isinstance(row["summary"], str) or not row["summary"].strip() or len(row["summary"]) > 2000:
            raise ValueError("semantic summary must be bounded nonempty text")
        quote = row["supporting_text"]
        if not isinstance(quote, str) or not quote.strip() or quote not in doc.text:
            raise ValueError("semantic index needs an exact supporting source quote")
        identity = (doc.document_id, row["memory_type"], row["summary"], quote)
        if identity in seen:
            raise ValueError("duplicate semantic candidate")
        seen.add(identity)
        checked.append(dict(row))
    return checked


def rank_sources(query, eligible_older, index_rows, ranker, *, subject_id):
    """Rank unverified summaries, but render only authenticated source history.

    Both arms use the same ranker/settings. A summary affects selection only;
    it is never inserted into the answer model's context or given fact status.
    Multiple index candidates for a source are merged before ranking so they
    cannot use up separate top-k slots. No implicit version suppression occurs.
    """
    sources = {d.document_id: d for d in eligible_older}
    grouped = {}
    for row in index_rows:
        if row["source_document_id"] in sources:
            grouped.setdefault(row["source_document_id"], []).append(row)
    hints = [replace(sources[key], text="\n".join(
        f'{r["memory_type"]}: {r["summary"]}' for r in candidates)) for key, candidates in grouped.items()]
    ranked = ranker.rank(query, hints, scope_id=subject_id)
    return [sources[d.document_id] for d in ranked]
