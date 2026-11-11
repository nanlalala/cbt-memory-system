"""Select candidate-index supplements by distinct permitted original sources.

The ranker must see the full eligible candidate pool before calling this helper.
Unconfirmed candidate text is an index only: returned values are complete user
quotations resolved from the caller's current source snapshot. This is neither
semantic fact verification nor a replacement for durable send/publication gates.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

from evidence_bundles import EvidenceSpan, restore_source_quote

VERSION = "distinct-new-source-candidate-selection-v1"
REQUIRED_PERMISSIONS = ("save", "long_term_use", "response_external")


@dataclass(frozen=True)
class SourceSupplement:
    memory_id: str
    span: EvidenceSpan
    candidate_rank: int
    score: float
    candidate_source_dependencies: tuple[str, ...]


def select_source_supplements(ranked_hits, *, sources, user_id,
                              already_present_source_ids=(), max_new_sources=3):
    """Apply the limit after source binding, coverage exclusion and deduplication.

    ``ranked_hits`` is the complete relevance-ranked output of MemoryRetriever,
    not its former top-three record slice. ``sources`` must already have been
    fetched under current user, cutoff, pause and withdrawal controls. Every
    dependency of a multi-source candidate is checked, even if its locator only
    resolves in one source. Those dependencies accompany the supplement so the
    caller can fence the actual packed claim before sending and publishing it.
    No value, date, consent or current state is inferred here.
    """
    if not isinstance(sources, Mapping) or not isinstance(user_id, str) or not user_id:
        raise ValueError("current source map and explicit user ID required")
    if not isinstance(ranked_hits, Sequence) or isinstance(ranked_hits, (str, bytes)):
        raise ValueError("complete ranked candidate sequence required")
    if type(max_new_sources) is not int or max_new_sources < 1:
        raise ValueError("positive distinct source budget required")
    if (not isinstance(already_present_source_ids, (Sequence, set, frozenset)) or
            isinstance(already_present_source_ids, (str, bytes)) or
            any(type(sid) is not str or not sid for sid in already_present_source_ids)):
        raise ValueError("already-present sources require explicit IDs")
    present = set(already_present_source_ids)
    selected = []; seen_memories = set(); decisions = []
    for rank, hit in enumerate(ranked_hits, 1):
        record = hit.record
        memory_id = record.memory_id
        reason = None
        dependencies = tuple(dict.fromkeys(record.source_evidence.turn_ids))
        if memory_id in seen_memories:
            reason = "duplicate_candidate_identity"
        seen_memories.add(memory_id)
        if reason is None and (record.user_id != user_id or record.source_role != "user"):
            reason = "candidate_subject_or_role"
        if reason is None and (record.lifecycle_status != "candidate" or record.sensitivity == "restricted"):
            reason = "candidate_lifecycle_or_sensitivity"
        if reason is None and any(record.permissions.get(key) is not True for key in REQUIRED_PERMISSIONS):
            reason = "candidate_permission"
        if reason is None and not dependencies:
            reason = "missing_source_binding"
        if reason is None and (type(hit.score) not in (int, float) or not math.isfinite(hit.score)):
            reason = "invalid_rank_score"
        if reason is None:
            for sid in dependencies:
                source = sources.get(sid)
                if (not isinstance(source, Mapping) or source.get("source_id") != sid or
                        source.get("user_id") != user_id or source.get("role") != "user" or
                        not isinstance(source.get("permissions"), Mapping) or
                        any(source["permissions"].get(key) is not True
                            for key in REQUIRED_PERMISSIONS)):
                    reason = "unavailable_source_dependency"
                    break
        row = {"memory_id": memory_id if record.user_id == user_id else None, "candidate_rank": rank,
               "candidate_source_dependencies": list(dependencies) if reason is None else [],
               "selected_source_ids": []}
        if reason is not None:
            decisions.append({**row, "decision": "excluded", "reason": reason})
            continue
        resolvable = []
        for sid in dependencies:
            if sid in present:
                continue
            span = restore_source_quote(sources[sid],
                pointer_text=record.source_evidence.supporting_text,
                span_id="semantic-source:" + memory_id + ":" + sid)
            if span is not None:
                resolvable.append(span)
        if not resolvable:
            reason = ("sources_already_present" if all(sid in present for sid in dependencies)
                      else "no_new_exact_source_locator")
            decisions.append({**row, "decision": "excluded", "reason": reason})
            continue
        if len(selected) >= max_new_sources:
            decisions.append({**row, "decision": "excluded", "reason": "distinct_source_budget"})
            continue
        for span in resolvable:
            if len(selected) >= max_new_sources:
                break
            present.add(span.source_id)
            selected.append(SourceSupplement(memory_id, span, rank, float(hit.score), dependencies))
            row["selected_source_ids"].append(span.source_id)
        decisions.append({**row, "decision": "selected", "reason": "new_permitted_original_sources"})
    return tuple(selected), {"version": VERSION,
        "budget_unit": "distinct_new_original_user_sources", "maximum_new_sources": max_new_sources,
        "ranked_candidate_count": len(ranked_hits), "selected_source_count": len(selected),
        "selected_source_ids": [item.span.source_id for item in selected],
        "candidate_index_text_is_response_evidence": False, "decisions": decisions}
