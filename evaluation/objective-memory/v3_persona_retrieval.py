"""Lossless source-span retrieval for a new PersonaMem protocol.

This is a *shared retrieval repair*, not a semantic-memory treatment. Apply the
same chunk size, ranker, recent-message window and character budgets to both raw
and managed arms. It never reads labels, calls a model or downloads data.

PersonaMem's released exclusive message boundary must be applied before this
module receives source documents. ``prepare_case`` reuses that adapter's already
bounded case and common lifecycle eligibility checks. Exact chunk offsets retain
the original role/source identity; a chunk is quoted history, not a user fact.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from html import escape
from typing import Sequence

from run_objective_comparison import CASE_FIELDS, _document, _eligible, _validate_case, require_colab
from run_three_arm import HistoryRanker, HistoryTurn, Settings, stable_digest

VERSION = "persona-source-span-retrieval-v1"
QUERY_POLICY = "current_question_plus_all_four_released_options-v1"


def retrieval_query(case: dict) -> str:
    """Use all released choices symmetrically, never the correct-option label.

    Options are already part of every arm's current question. Their wording can
    supply topics missing from an underspecified question. This fixed policy is
    shared by raw and managed retrieval, so it is not a management treatment.
    """
    if set(case) - CASE_FIELDS or case.get("dataset_key") != "personamem_v1":
        raise ValueError("retrieval query accepts gold-free PersonaMem cases only")
    query, options = case.get("query"), case.get("options")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("current question must be nonempty")
    if not isinstance(options, list) or len(options) != 4 or any(not isinstance(o, str) or not o.strip() for o in options):
        raise ValueError("retain all four nonempty released options")
    return "\n".join([query, *options])


def plan_group_shards(cases: Sequence[dict], *, shard_count: int = 4,
                      expected_cases: int | None = None,
                      expected_groups: int | None = None) -> dict:
    """Freeze whole connected groups across finite, independent writer dirs.

    The balancing input is question count only, never correctness, options or
    source text. Sorting makes the result independent of source file row order.
    This plans execution; it does not manufacture additional independent users.
    """
    if type(shard_count) is not int or shard_count <= 0:
        raise ValueError("positive integer shard count required")
    groups, seen = {}, set()
    for case in cases:
        if set(case) - CASE_FIELDS or case.get("dataset_key") != "personamem_v1":
            raise ValueError("group planner accepts gold-free PersonaMem cases only")
        case_id, group_id = case.get("case_id"), case.get("group_id")
        if any(not isinstance(x, str) or not x.strip() for x in (case_id, group_id)) or case_id in seen:
            raise ValueError("missing group identity or duplicate case")
        seen.add(case_id)
        groups.setdefault(group_id, []).append(case_id)
    if not groups or shard_count > len(groups):
        raise ValueError("each execution shard needs at least one whole group")
    if expected_cases is not None and len(seen) != expected_cases:
        raise ValueError("question census differs from frozen full-cohort expectation")
    if expected_groups is not None and len(groups) != expected_groups:
        raise ValueError("group census differs from frozen full-cohort expectation")
    shards = [{"shard": n, "group_ids": [], "case_ids": []} for n in range(shard_count)]
    for group_id in sorted(groups, key=lambda g: (-len(groups[g]), stable_digest(g))):
        target = min(shards, key=lambda s: (len(s["case_ids"]), len(s["group_ids"]), s["shard"]))
        target["group_ids"].append(group_id)
        target["case_ids"].extend(sorted(groups[group_id]))
    for shard in shards:
        shard["group_ids"].sort()
        shard["case_ids"].sort()
        shard["case_count"] = len(shard["case_ids"])
        shard["group_count"] = len(shard["group_ids"])
    plan = {"version": VERSION, "policy": "whole connected groups; question-count balance fixed before generation",
            "case_count": len(seen), "independent_groups": len(groups), "shards": shards,
            "scoring_labels_used": False, "questions_are_independent_users": False}
    plan["plan_sha256"] = stable_digest(plan)
    return plan


@dataclass(frozen=True, slots=True)
class SourceChunk:
    chunk_id: str
    source_document_id: str
    scope_id: str
    session_id: str
    date: str
    speaker: str
    start: int
    end: int
    source_position: int
    text: str

    def ranking_document(self) -> HistoryTurn:
        return HistoryTurn(self.scope_id, self.chunk_id, self.session_id,
                           self.date, self.speaker, self.text)


def split_source(document: HistoryTurn, *, source_position: int, max_chars: int = 1200) -> tuple[SourceChunk, ...]:
    """Partition source text exactly; no summaries, normalization or overlap.

    Prefer a late paragraph/word boundary but fall back to a Unicode character
    cut. Joining the chunks' text reconstructs the supplied source text exactly.
    Offsets are Unicode character indices, not bytes or tokenizer counts.
    """
    if type(max_chars) is not int or max_chars < 128:
        raise ValueError("chunk character budget must be an integer >= 128")
    if type(source_position) is not int or source_position < 0:
        raise ValueError("source position must be nonnegative")
    if not isinstance(document.text, str):
        raise ValueError("source text must be a string")
    result, start = [], 0
    while start < len(document.text):
        end = min(len(document.text), start + max_chars)
        if end < len(document.text):
            floor = start + max_chars // 2
            boundary = max(document.text.rfind("\n", floor, end),
                           document.text.rfind(" ", floor, end))
            if boundary >= floor:
                end = boundary + 1
        text = document.text[start:end]
        identity = stable_digest([document.scope_id, document.document_id,
                                  document.session_id, document.date,
                                  document.speaker, start, end, text])
        result.append(SourceChunk(identity, document.document_id, document.scope_id,
                                  document.session_id, document.date, document.speaker,
                                  start, end, source_position, text))
        start = end
    return tuple(result)


def render_chunk(chunk: SourceChunk) -> str:
    return (f'<source_span id="{escape(chunk.chunk_id, quote=True)}" '
            f'source_id="{escape(chunk.source_document_id, quote=True)}" '
            f'start="{chunk.start}" end="{chunk.end}" '
            f'date="{escape(chunk.date, quote=True)}" '
            f'speaker="{escape(chunk.speaker, quote=True)}">'
            f'{escape(chunk.text, quote=True)}</source_span>')


def fit_chunks(chunks: Sequence[SourceChunk], budget: int, *, chronological: bool = True) -> tuple[str, list[str], list[str]]:
    """Select whole spans in priority order, then optionally show chronology."""
    if type(budget) is not int or budget <= 0:
        raise ValueError("positive character budget required")
    if len({c.chunk_id for c in chunks}) != len(chunks):
        raise ValueError("duplicate source chunk")
    selected, dropped, used = [], [], 0
    for chunk in chunks:
        cost = len(render_chunk(chunk)) + bool(selected)
        if used + cost <= budget:
            selected.append(chunk)
            used += cost
        else:
            dropped.append(chunk.chunk_id)
    if chronological:
        selected.sort(key=lambda x: (x.source_position, x.start, x.chunk_id))
    text = "\n".join(map(render_chunk, selected))
    if len(text) > budget:
        raise AssertionError("rendered source spans exceeded character budget")
    return text, [c.chunk_id for c in selected], dropped


class PersonaChunkRetriever:
    """Bounded caches keyed by exact visible source bytes and subject identity."""

    def __init__(self, settings: Settings, *, chunk_chars: int = 1200,
                 max_source_cache: int = 4096, max_corpus_cache: int = 4):
        settings.validate()
        if type(chunk_chars) is not int or chunk_chars < 128:
            raise ValueError("chunk character budget must be an integer >= 128")
        if any(type(n) is not int or n <= 0 for n in (max_source_cache, max_corpus_cache)):
            raise ValueError("cache limits must be positive integers")
        self.settings, self.chunk_chars = settings, chunk_chars
        self.max_source_cache, self.max_corpus_cache = max_source_cache, max_corpus_cache
        self.ranker = HistoryRanker(settings)
        self.source_cache: OrderedDict[str, tuple[SourceChunk, ...]] = OrderedDict()
        self.corpus_cache: OrderedDict[str, None] = OrderedDict()

    def chunks(self, documents: Sequence[HistoryTurn], *, scope_id: str) -> list[SourceChunk]:
        if any(d.scope_id != scope_id for d in documents):
            raise ValueError("cross-subject source chunking prohibited")
        if len({d.document_id for d in documents}) != len(documents):
            raise ValueError("duplicate source document")
        result = []
        for position, document in enumerate(documents):
            key = stable_digest([asdict(document), position, self.chunk_chars])
            if key not in self.source_cache:
                self.source_cache[key] = split_source(document, source_position=position,
                                                       max_chars=self.chunk_chars)
                while len(self.source_cache) > self.max_source_cache:
                    self.source_cache.popitem(last=False)
            self.source_cache.move_to_end(key)
            result.extend(self.source_cache[key])
        return result

    def rank(self, query: str, chunks: Sequence[SourceChunk], *, scope_id: str) -> list[SourceChunk]:
        if any(c.scope_id != scope_id for c in chunks):
            raise ValueError("cross-subject source retrieval prohibited")
        if self.settings.backend != "lexical":
            # The existing dense adapter lazily loads model weights. This path
            # cannot load those weights in an ordinary personal-computer runtime.
            require_colab()
        documents = [c.ranking_document() for c in chunks]
        key = stable_digest([scope_id, self.settings.dense_model,
                             [f"{d.speaker}: {d.text}" for d in documents]])
        if key not in self.corpus_cache:
            while len(self.corpus_cache) >= self.max_corpus_cache:
                old, _ = self.corpus_cache.popitem(last=False)
                self.ranker.lexical_cache.pop(old, None)
                self.ranker.dense_cache.pop(old, None)
            self.corpus_cache[key] = None
        self.corpus_cache.move_to_end(key)
        ranked = self.ranker.rank(query, documents, scope_id=scope_id)
        by_id = {c.chunk_id: c for c in chunks}
        return [by_id[d.document_id] for d in ranked]

    def channels(self, query: str, documents: Sequence[HistoryTurn], *, scope_id: str,
                 include_older: bool = True) -> tuple[str, str, dict]:
        """Return common recent/source channels using only currently visible docs.

        Source split is still the last N *messages*, not the last N chunks.
        The recent channel selects the newest fitting spans then displays them
        chronologically. The older channel ranks chunks and displays survivors
        chronologically. No inference about answer sufficiency is made here.
        """
        all_chunks = self.chunks(documents, scope_id=scope_id)
        short_ids = {d.document_id for d in documents[-self.settings.short_turns:]}
        short_chunks = [c for c in all_chunks if c.source_document_id in short_ids]
        older_chunks = [c for c in all_chunks if c.source_document_id not in short_ids]
        ranked = self.rank(query, older_chunks, scope_id=scope_id) if include_older else []
        short_text, included_short, dropped_short = fit_chunks(
            list(reversed(short_chunks)), self.settings.short_chars)
        long_text, included_long, dropped_long = fit_chunks(ranked, self.settings.long_chars)
        selected = set(included_short + included_long)
        trace = {"retrieval_version": VERSION, "encoding": "exact_source_spans_no_summary",
                 "source_unit_count": len(documents), "source_chunk_count": len(all_chunks),
                 "chunk_chars": self.chunk_chars, "chunk_offsets": "Unicode characters; exclusive end",
                 "short_window_unit": "original source messages",
                 "retrieved_ids": [c.chunk_id for c in ranked],
                 "included_short_ids": included_short, "included_long_ids": included_long,
                 "budget_dropped_short_ids": dropped_short, "budget_dropped_long_ids": dropped_long,
                 "included_source_spans": [{"chunk_id": c.chunk_id,
                     "source_document_id": c.source_document_id, "start": c.start,
                     "end": c.end, "role": c.speaker} for c in all_chunks if c.chunk_id in selected],
                 "source_visibility": "already bounded eligible history only; cache never adds documents",
                 "current_use": False, "interactive_clarification": False}
        return short_text, long_text, trace

    def prepare_case(self, case: dict, *, include_older: bool = True) -> tuple[str, str, dict]:
        """Consume a gold-free bounded adapter case; reject unknown/gold fields."""
        _validate_case(case)
        if case.get("dataset_key") != "personamem_v1":
            raise ValueError("this adapter is for released PersonaMem cases")
        cutoff = case.get("visible_context_end_exclusive")
        if type(cutoff) is not int or cutoff <= 0 or any(r["order"] >= cutoff for r in case["history"]):
            raise ValueError("source exceeds released exclusive message boundary")
        admitted, excluded = _eligible(case)
        documents = [_document(case, row) for row in admitted]
        short, long, trace = self.channels(retrieval_query(case), documents,
                                           scope_id=case["subject_id"], include_older=include_older)
        trace["excluded_source_units"] = excluded
        trace["visible_context_end_exclusive"] = case.get("visible_context_end_exclusive")
        trace["retrieval_query_policy"] = QUERY_POLICY
        return short, long, trace
