"""Source-only retrieval with a stable, permission-scoped lexical statistics pool.

This module does not establish semantic support, consent, or user confirmation.
Its inputs must be original source records. Derived notes are not original
documents and cannot alter the quotation channel's document frequencies.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from memory_retriever import DenseEncoder, _tokens


PERMISSION_FIELDS = (
    "save", "long_term_use", "response_external", "extraction_external",
)
_ORIGINAL_REPRESENTATIONS = {"source", "source_quote", "raw", "verbatim"}


def _time(value: str, name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} requires a timezone-aware timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} requires an ISO-8601 timestamp") from exc
    if result.tzinfo is None:
        raise ValueError(f"{name} requires a timezone-aware timestamp")
    return result


def _permissions(value: Any) -> dict[str, bool | None]:
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("permissions must be a mapping")
    unknown = set(value) - set(PERMISSION_FIELDS)
    if unknown:
        raise ValueError("unknown permission fields: " + ", ".join(sorted(unknown)))
    result: dict[str, bool | None] = {}
    for name in PERMISSION_FIELDS:
        item = value.get(name)
        if type(item) is not bool and item is not None:
            raise ValueError(f"permission {name} must be boolean or null (unknown)")
        result[name] = item
    return result


def _integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True, slots=True)
class SourceSpanHit:
    source_id: str
    span_id: str
    start: int
    end: int
    text: str
    source_time: str
    role: str
    category: str | None
    score: float
    components: dict[str, float]
    applicable: bool = True
    object_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SourceSearchResult:
    hits: tuple[SourceSpanHit, ...]
    trace: dict[str, Any]

    def __iter__(self):
        return iter(self.hits)

    def __len__(self) -> int:
        return len(self.hits)

    def __getitem__(self, index):
        return self.hits[index]


def _norm_vector(vector: Sequence[float], *, expected: int | None = None) -> tuple[float, ...]:
    result = tuple(float(item) for item in vector)
    if not result or (expected is not None and len(result) != expected):
        raise ValueError("encoder returned inconsistent vector dimensions")
    if any(not math.isfinite(item) for item in result):
        raise ValueError("encoder returned non-finite vector values")
    norm = math.sqrt(sum(item * item for item in result))
    return tuple(item / norm for item in result) if norm else result


class StableSourceRetriever:
    """Independently recall lexical/dense spans over an eligible raw-source pool.

    ``source_cutoff`` controls when a report became visible. ``as_of`` only
    controls the requested state applicability. Permissions are evaluated for
    this read, never rewound to ``as_of``. A caller using external generation
    must set ``for_external_response=True`` as well as enforce its send gate.
    """

    def __init__(
        self,
        sources: Sequence[Mapping[str, Any]],
        *,
        encoder: DenseEncoder | None = None,
    ) -> None:
        self.encoder = encoder
        self.sources: list[dict[str, Any]] = []
        seen: dict[tuple[str, str], str] = {}
        for raw in sources:
            if not isinstance(raw, Mapping):
                raise ValueError("source must be a mapping")
            source = dict(raw)
            for key in ("source_id", "user_id", "text"):
                if not isinstance(source.get(key), str) or not source[key]:
                    raise ValueError(f"source {key} must be a nonempty string")
            source["permissions"] = _permissions(source.get("permissions"))
            # An unresolved legacy role/time remains unresolved and is excluded
            # at read time; no current clock or inferred user role is filled in.
            source.setdefault("role", "unknown")
            source.setdefault("source_time", "unknown")
            if source["role"] is not None and not isinstance(source["role"], str):
                raise ValueError("source role must be a string or null")
            if source["source_time"] is not None and not isinstance(source["source_time"], str):
                raise ValueError("source_time must be a string or null")
            identity = (source["user_id"], source["source_id"])
            signature = json.dumps(source, sort_keys=True, ensure_ascii=True)
            if identity in seen:
                if signature != seen[identity]:
                    raise ValueError("conflicting versions of one source in retrieval snapshot")
                continue
            seen[identity] = signature
            self.sources.append(source)

    @staticmethod
    def _spans(source: Mapping[str, Any]) -> list[dict[str, Any]]:
        raw_spans = source.get("spans")
        if raw_spans is None:
            raw_spans = [{"span_id": source["source_id"] + ":full", "start": 0,
                          "end": len(source["text"])}]
        if not isinstance(raw_spans, (list, tuple)):
            raise ValueError("source spans must be a list")
        result = []
        seen_ids: set[str] = set()
        for index, raw in enumerate(raw_spans):
            if not isinstance(raw, Mapping):
                raise ValueError("source span must be a mapping")
            span = dict(raw)
            start = _integer(span.get("start"), "span.start")
            end = _integer(span.get("end"), "span.end")
            if not start < end <= len(source["text"]):
                raise ValueError("source span is out of bounds or empty")
            span_id = span.get("span_id", source["source_id"] + f":span:{index}")
            if not isinstance(span_id, str) or not span_id or span_id in seen_ids:
                raise ValueError("source span IDs must be nonempty and unique")
            seen_ids.add(span_id)
            text = source["text"][start:end]
            if "text" in span and span["text"] != text:
                raise ValueError("source span text does not match exact Unicode offsets")
            span.update(start=start, end=end, span_id=span_id, text=text)
            for name in ("valid_from", "valid_to"):
                if span.get(name) is not None:
                    _time(span[name], "span." + name)
            if span.get("valid_from") is not None and span.get("valid_to") is not None:
                if _time(span["valid_to"], "span.valid_to") < _time(span["valid_from"], "span.valid_from"):
                    raise ValueError("span valid_to precedes valid_from")
            result.append(span)
        return result

    def search(
        self,
        query: str,
        *,
        user_id: str,
        source_cutoff: str,
        as_of: str | None = None,
        query_mode: str = "mixed",
        allowed_source_ids: set[str] | None = None,
        revoked_source_ids: Sequence[str] = (),
        current_permissions: Mapping[str, Mapping[str, Any]] | None = None,
        categories: Sequence[str] | None = None,
        top_k: int = 8,
        lexical_k: int = 12,
        dense_k: int = 12,
        minimum_relevance: float = .05,
        for_external_response: bool = False,
    ) -> SourceSearchResult:
        if not isinstance(query, str) or not isinstance(user_id, str) or not user_id:
            raise ValueError("query and user_id must be strings")
        if query_mode not in {"current", "history", "mixed"}:
            raise ValueError("query_mode must be current, history or mixed")
        cutoff = _time(source_cutoff, "source_cutoff")
        applicable_at = _time(as_of, "as_of") if as_of is not None else None
        for name, value in (("top_k", top_k), ("lexical_k", lexical_k), ("dense_k", dense_k)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(minimum_relevance) not in {int, float} or not 0 <= minimum_relevance <= 1:
            raise ValueError("minimum_relevance must be between zero and one")
        if type(for_external_response) is not bool:
            raise ValueError("for_external_response must be boolean")
        for name, values in (("categories", categories), ("allowed_source_ids", allowed_source_ids),
                             ("revoked_source_ids", revoked_source_ids)):
            if values is not None and (isinstance(values, (str, bytes)) or
                                       any(not isinstance(item, str) or not item for item in values)):
                raise ValueError(f"{name} must contain nonempty string IDs")
        revoked = set(revoked_source_ids)
        route_categories = set(categories or ())
        current_permissions = current_permissions or {}
        pool: list[dict[str, Any]] = []
        excluded: Counter[str] = Counter()
        for source in self.sources:
            if source["user_id"] != user_id:
                continue  # Do not expose other users' source IDs in a trace.
            reason = None
            if source.get("representation_kind", "source") not in _ORIGINAL_REPRESENTATIONS:
                reason = "derived_representation"
            elif source["source_id"] in revoked or source.get("deleted") is True or source.get("blocked") is True:
                reason = "revoked_or_blocked"
            elif source.get("lifecycle_status") in {"deleted", "blocked"}:
                reason = "revoked_or_blocked"
            elif allowed_source_ids is not None and source["source_id"] not in allowed_source_ids:
                reason = "outside_allowed_scope"
            elif source["role"] not in {"user", "assistant"}:
                reason = "unknown_role"
            elif source["source_time"] is None or source["source_time"] == "unknown":
                reason = "unknown_source_time"
            elif _time(source["source_time"], "source_time") > cutoff:
                reason = "after_source_cutoff"
            permissions = source["permissions"]
            if source["source_id"] in current_permissions:
                permissions = _permissions({**permissions, **current_permissions[source["source_id"]]})
            if reason is None and permissions["long_term_use"] is not True:
                reason = "current_use_not_permitted"
            if reason is None and for_external_response and permissions["response_external"] is not True:
                reason = "current_external_use_not_permitted"
            if reason is not None:
                excluded[reason] += 1
                continue
            pool.append(source)

        pool.sort(key=lambda source: source["source_id"])
        pool_identity = [{"source_id": s["source_id"], "sha256": hashlib.sha256(s["text"].encode()).hexdigest()}
                         for s in pool]
        fingerprint = hashlib.sha256(json.dumps(pool_identity, sort_keys=True).encode()).hexdigest()
        trace: dict[str, Any] = {
            "version": "stable-source-retrieval-v2",
            "statistics_pool_fingerprint": fingerprint,
            "statistics_source_count": len(pool),
            "statistics_source_ids": [s["source_id"] for s in pool],
            "source_cutoff": source_cutoff, "state_as_of": as_of,
            "permission_time": "current_read_policy_not_rewound",
            "query_mode": query_mode, "hard_excluded_counts": dict(excluded),
            "soft_categories": sorted(route_categories), "candidates": [],
            "unique_source_count": 0, "lexical_recall": [], "dense_recall": [],
        }
        if not pool or not query.strip():
            return SourceSearchResult((), trace)

        # Statistics are over complete original sources, once per source. The
        # span set, soft category routing and derived paraphrases never refit it.
        document_counts = [Counter(_tokens(source["text"])) for source in pool]
        df = Counter(token for counts in document_counts for token in counts)
        size = len(pool)
        def idf(token: str) -> float:
            return math.log((1 + size) / (1 + df[token])) + 1
        def vector(text: str) -> dict[str, float]:
            counts = Counter(_tokens(text)); total = sum(counts.values()) or 1
            return {token: count / total * idf(token) for token, count in counts.items()}
        qv = vector(query); qnorm = math.sqrt(sum(value * value for value in qv.values()))
        candidates: list[tuple[dict[str, Any], dict[str, Any], bool]] = []
        for source in pool:
            for span in self._spans(source):
                applicable = True
                if applicable_at is not None:
                    if span.get("valid_from") is not None and _time(span["valid_from"], "span.valid_from") > applicable_at:
                        applicable = False
                    if span.get("valid_to") is not None and applicable_at >= _time(span["valid_to"], "span.valid_to"):
                        applicable = False
                if query_mode != "mixed" and not applicable:
                    continue
                candidates.append((source, span, applicable))
        lexical: list[float] = []
        for _, span, _ in candidates:
            sv = vector(span["text"]); norm = math.sqrt(sum(value * value for value in sv.values()))
            numerator = sum(value * sv.get(token, 0) for token, value in qv.items())
            lexical.append(numerator / (qnorm * norm) if qnorm and norm else 0.0)
        dense = [0.0] * len(candidates)
        if self.encoder is not None and candidates:
            qencoded = self.encoder.encode([query], is_query=True)
            if len(qencoded) != 1:
                raise ValueError("encoder did not return exactly one query vector")
            qdense = _norm_vector(qencoded[0])
            vectors = self.encoder.encode([span["text"] for _, span, _ in candidates], is_query=False)
            if len(vectors) != len(candidates):
                raise ValueError("encoder did not return every source span vector")
            dense = [max(0.0, sum(a * b for a, b in zip(qdense, _norm_vector(v, expected=len(qdense)))))
                     for v in vectors]

        def recall(scores: list[float], limit: int) -> list[int]:
            eligible = [index for index, score in enumerate(scores) if score > 0 and score >= minimum_relevance]
            return sorted(eligible, key=lambda i: (-scores[i], candidates[i][0]["source_id"], candidates[i][1]["span_id"]))[:limit]
        lex_ids = recall(lexical, lexical_k)
        dense_ids = recall(dense, dense_k) if self.encoder is not None else []
        lex_ranks = {index: rank for rank, index in enumerate(lex_ids, 1)}
        dense_ranks = {index: rank for rank, index in enumerate(dense_ids, 1)}
        trace["lexical_recall"] = [candidates[i][1]["span_id"] for i in lex_ids]
        trace["dense_recall"] = [candidates[i][1]["span_id"] for i in dense_ids]
        hits: list[SourceSpanHit] = []
        for index in sorted(set(lex_ids) | set(dense_ids)):
            source, span, applicable = candidates[index]
            rrf = (1 / (60 + lex_ranks[index]) if index in lex_ranks else 0.0)
            rrf += (1 / (60 + dense_ranks[index]) if index in dense_ranks else 0.0)
            category = span.get("category", source.get("category"))
            route_bonus = .001 if category in route_categories else 0.0
            score = rrf + route_bonus
            hit = SourceSpanHit(
                source_id=source["source_id"], span_id=span["span_id"], start=span["start"], end=span["end"],
                text=span["text"], source_time=source["source_time"], role=source["role"], category=category,
                score=score, components={"lexical": lexical[index], "dense": dense[index],
                                         "rank_fusion": rrf, "soft_route_bonus": route_bonus},
                applicable=applicable, object_id=span.get("object_id"),
            )
            hits.append(hit)
        hits.sort(key=lambda hit: (-hit.score, hit.source_id, hit.span_id))
        selected = tuple(hits[:top_k])
        trace["candidates"] = [{"source_id": hit.source_id, "span_id": hit.span_id,
                                "components": hit.components, "selected": hit in selected}
                               for hit in hits]
        trace["unique_source_count"] = len({hit.source_id for hit in selected})
        trace["selected_span_count"] = len(selected)
        return SourceSearchResult(selected, trace)
