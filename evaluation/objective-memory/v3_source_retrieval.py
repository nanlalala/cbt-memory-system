"""Bounded source-entity expansion shared by raw and version-managed retrieval.

This is a source-grammar graph, not a knowledge graph supplied with benchmark
answers. Only parsed objects of already selected source assertions open a next
hop. Returned documents retain their exact original quotation and provenance.
"""
from __future__ import annotations

import hashlib
import math
import re
import unicodedata

from v3_fact_versions import parse_assertion

VERSION = "source-entity-expansion-v1"
MAX_ROUNDS = 3
MAX_CACHE_ENTRIES = 16


def policy_descriptor():
    return {"version": VERSION, "max_rounds": MAX_ROUNDS,
            "direct_seed_reserve": "ceil(top_k / 3)",
            "links": "selected assertion object -> same-scope assertion subject, normalized exact entity match",
            "allocation": "one third per expansion round; fill unused capacity from original lexical ranking",
            "short_context_seeds": "same-question-ranked common recent sources may open links into older history; never returned or charged against older top_k",
            "budget": "top_k total unique source documents, unchanged shared character budgets",
            "facts_rewritten": False, "gold_or_external_knowledge_used": False}


def normalize_entity(text):
    value = unicodedata.normalize("NFKC", text).casefold().strip()
    value = re.sub(r"\s+", " ", value)
    value = value.strip(" \t\r\n.,;:!?\"'")
    return re.sub(r"^(?:the|an|a)\s+", "", value)


def _graph(documents, ranker, scope_id):
    digest = hashlib.sha256(scope_id.encode())
    for doc in documents:
        # Including both exact text and identity prevents a same-length or
        # same-source-ID replacement from accidentally reusing an older graph.
        for value in (doc.document_id, doc.session_id, doc.date, doc.speaker, doc.text):
            block = str(value).encode()
            digest.update(len(block).to_bytes(8, "big"))
            digest.update(block)
    key = digest.hexdigest()
    cache = getattr(ranker, "_v3_source_graph_cache", None)
    if cache is None:
        cache = {}
        ranker._v3_source_graph_cache = cache
    if key not in cache:
        parsed, by_subject = {}, {}
        for doc in documents:
            assertion = parse_assertion(doc.text)
            if assertion is not None:
                parsed[doc.document_id] = assertion
                by_subject.setdefault(normalize_entity(assertion.subject), []).append(doc)
        if len(cache) >= MAX_CACHE_ENTRIES:
            cache.pop(next(iter(cache)))
        cache[key] = (parsed, by_subject)
    return cache[key]


def retrieve_sources(query, documents, ranker, *, scope_id, seed_documents=()):
    """Retrieve at most the ranker's top_k original same-scope source units.

    Version consolidation, if used, happens before this function. Raw and
    managed arms call this exact function with the same policy and ranker
    settings. No relation, entity, answer or source priority comes from gold.
    """
    documents, seed_documents = list(documents), list(seed_documents)
    if any(doc.scope_id != scope_id for doc in documents + seed_documents):
        raise ValueError("cross-subject source graph retrieval is prohibited")
    if len({doc.document_id for doc in documents}) != len(documents):
        raise ValueError("source graph requires unique source document IDs")
    if len({doc.document_id for doc in seed_documents}) != len(seed_documents):
        raise ValueError("source graph requires unique recent seed document IDs")
    if {doc.document_id for doc in documents} & {doc.document_id for doc in seed_documents}:
        raise ValueError("recent seed documents must be separate from older retrieval documents")
    top_k = ranker.settings.top_k
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("source graph requires positive integer top_k")
    initial = ranker.rank(query, documents, scope_id=scope_id)
    per_round = math.ceil(top_k / MAX_ROUNDS)
    recent_seeds = ranker.rank(query, seed_documents, scope_id=scope_id)[:per_round]
    trace = {"policy_version": VERSION, "initial_ranked_ids": [d.document_id for d in initial],
             "common_recent_seed_ids": [d.document_id for d in recent_seeds],
             "rounds": [], "fallback_fill_ids": [], "selected_ids": []}
    if not initial and not recent_seeds:
        return [], trace
    parsed, by_subject = _graph(documents, ranker, scope_id)
    # Recent sources already occupy the shared short channel. Their source
    # objects can identify older follow-up facts even if the older corpus has
    # no lexical match for the question's starting entity. They never enter the
    # older result list or consume its top_k slots.
    parsed = dict(parsed)
    for doc in recent_seeds:
        assertion = parse_assertion(doc.text)
        if assertion is not None:
            parsed[doc.document_id] = assertion
    selected = initial[:per_round]
    selected_ids = {d.document_id for d in selected}
    trace["rounds"].append({"round": 0, "kind": "direct_query", "selected_ids": [d.document_id for d in selected],
                             "common_recent_seed_ids": [d.document_id for d in recent_seeds]})
    frontier, expanded_entities = selected + recent_seeds, set()
    for depth in range(1, MAX_ROUNDS):
        if len(selected) >= top_k:
            break
        entities = sorted({normalize_entity(parsed[d.document_id].object)
                           for d in frontier if d.document_id in parsed}
                          - expanded_entities - {""})
        expanded_entities.update(entities)
        if not entities:
            break
        neighbors = {doc.document_id: doc for entity in entities for doc in by_subject.get(entity, [])
                     if doc.document_id not in selected_ids}
        if not neighbors:
            trace["rounds"].append({"round": depth, "kind": "source_object_links", "entities": entities,
                                     "candidate_count": 0, "selected_ids": []})
            break
        # The question is unchanged; the appended text contains only entity
        # strings read from selected source assertions, never answer labels.
        expanded_query = query + " " + " ".join(entities)
        ranked = ranker.rank(expanded_query, [neighbors[key] for key in sorted(neighbors)], scope_id=scope_id)
        frontier = ranked[:min(per_round, top_k - len(selected))]
        selected.extend(frontier)
        selected_ids.update(d.document_id for d in frontier)
        trace["rounds"].append({"round": depth, "kind": "source_object_links", "entities": entities,
                                 "candidate_count": len(neighbors), "selected_ids": [d.document_id for d in frontier]})
        if not frontier:
            break
    for doc in initial:
        if len(selected) >= top_k:
            break
        if doc.document_id not in selected_ids:
            selected.append(doc)
            selected_ids.add(doc.document_id)
            trace["fallback_fill_ids"].append(doc.document_id)
    trace["selected_ids"] = [d.document_id for d in selected]
    return selected, trace
