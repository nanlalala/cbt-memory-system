"""Candidate shared source-edge retrieval repair for a NEW protocol only.

This is a source-grammar graph, not a knowledge graph supplied with benchmark
answers. Only parsed objects of already selected source assertions open a next
hop. Returned documents retain their exact original quotation and provenance.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import math
import re
import unicodedata

from v3_fact_versions import parse_assertion
from run_three_arm import HistoryRanker

VERSION = "source-edge-gated-expansion-v2"
MAX_ROUNDS = 3
MAX_CACHE_ENTRIES = 16


def policy_descriptor():
    return {"version": VERSION, "max_rounds": MAX_ROUNDS,
            "direct_seed_reserve": "ceil(top_k / 3)",
            "links": "selected assertion object -> same-scope assertion subject, normalized exact entity match",
            "allocation": "one third per expansion round; fill unused capacity from original lexical ranking",
            "short_context_seeds": "same-question-ranked common recent sources may open links into older history; never returned or charged against older top_k",
            "budget": "top_k total unique source documents, unchanged shared character budgets",
            "linked_relevance_gate": "same-scope exact source-object -> subject edge; zero generic query threshold only after that edge is validated",
            "shared_repair_not_managed_treatment": True,
            "evaluation_status": "candidate repair; no new benchmark effectiveness result",
            "supported_backends": ["lexical"],
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
    cache = getattr(ranker, "_v4_source_graph_cache", None)
    if cache is None:
        cache = {}
        ranker._v4_source_graph_cache = cache
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


def _rank_linked_candidates(query, documents, ranker, *, scope_id):
    """Rank an already validated exact-edge neighborhood without re-gating it.

    The question threshold still governs all direct seeds. A source edge is a
    distinct admitted relevance signal: generic full-question cosine must not
    erase it merely because an unrelated query token enlarged the norm. A
    separate ranker avoids mutating the caller's frozen settings or caches.
    Scores only order the same source documents; total top-k stays unchanged.
    This candidate currently supports only the original lexical MAB backend;
    dense cosine can be negative, so a zero threshold is not equivalent there.
    """
    if ranker.settings.backend != "lexical":
        raise ValueError("Candidate source-edge repair is validated for lexical retrieval only")
    linked = getattr(ranker, "_v4_exact_edge_ranker", None)
    settings = replace(ranker.settings, minimum_relevance=0.0)
    if linked is None or linked.settings != settings:
        linked = HistoryRanker(settings, device=getattr(ranker, "device", None))
        ranker._v4_exact_edge_ranker = linked
    selected = linked.rank(query, documents, scope_id=scope_id)
    # Each neighborhood is finite, but a long prepared suite must not retain
    # every historical neighborhood's corpus arrays indefinitely.
    for cache in (linked.lexical_cache, linked.dense_cache):
        while len(cache) > MAX_CACHE_ENTRIES:
            cache.pop(next(iter(cache)))
    return selected


def retrieve_sources(query, documents, ranker, *, scope_id, seed_documents=()):
    """Candidate v4 repair; never resume a frozen v3 run with this function.

    Retrieve at most the ranker's top_k original same-scope source units.

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
    if ranker.settings.backend != "lexical":
        raise ValueError("Candidate source-edge repair is validated for lexical retrieval only")
    initial = ranker.rank(query, documents, scope_id=scope_id)
    per_round = math.ceil(top_k / MAX_ROUNDS)
    recent_seeds = ranker.rank(query, seed_documents, scope_id=scope_id)[:per_round]
    trace = {"policy_version": VERSION, "initial_ranked_ids": [d.document_id for d in initial],
             "common_recent_seed_ids": [d.document_id for d in recent_seeds],
             "rounds": [], "fallback_fill_ids": [], "selected_ids": [],
             "direct_seed_minimum_relevance": ranker.settings.minimum_relevance,
             "linked_candidate_minimum_generic_query_relevance": 0.0,
             "cause_attribution": "source-path observation only; no independently annotated answer sufficiency"}
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
        ranked = _rank_linked_candidates(expanded_query, [neighbors[key] for key in sorted(neighbors)], ranker, scope_id=scope_id)
        frontier = ranked[:min(per_round, top_k - len(selected))]
        selected.extend(frontier)
        selected_ids.update(d.document_id for d in frontier)
        trace["rounds"].append({"round": depth, "kind": "source_object_links", "entities": entities,
                                 "candidate_count": len(neighbors),
                                 "candidate_ids": sorted(neighbors),
                                 "edge_gate": "exact_same_scope_source_object_subject_match",
                                 "ranked_candidate_ids": [d.document_id for d in ranked],
                                 "selected_ids": [d.document_id for d in frontier]})
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
