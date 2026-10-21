"""User-scoped long-term memory retrieval with transparent score components.

The reference implementation is dependency-free.  It provides a lexical baseline
for local tests and accepts any dense encoder implementing ``encode``.  Production
experiments can therefore use multilingual E5 without making PyTorch a runtime
requirement for the memory manager.
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, Sequence

from memory_models import LifecycleStatus, MemoryRecord, Sensitivity, is_valid_at


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u3400-\u9fff]+", re.UNICODE)
STOPWORDS = {
    "a", "an", "and", "are", "do", "does", "for", "how", "i", "in", "is", "it",
    "my", "of", "on", "the", "to", "was", "what", "when", "where", "who", "why",
}


class DenseEncoder(Protocol):
    def encode(self, texts: Sequence[str], *, is_query: bool) -> Sequence[Sequence[float]]: ...


class MultilingualE5Encoder:
    """Lazy E5 adapter loading only a byte-verified, fixed model snapshot."""

    def __init__(
        self, model_name: str = "intfloat/multilingual-e5-small", device: str | None = None,
        *, snapshot_path: Path | str | None = None, local_files_only: bool = False,
    ):
        shared_modules = str(Path(__file__).resolve().parents[2])
        if shared_modules not in sys.path:
            sys.path.insert(0, shared_modules)
        from model_snapshots import prepare_model_snapshot
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised in Colab
            raise RuntimeError("Install sentence-transformers to use the E5 backend") from exc
        snapshot = prepare_model_snapshot(
            model_name, local_path=snapshot_path, local_files_only=local_files_only,
        )
        self.model = SentenceTransformer(
            str(snapshot.path), device=device, local_files_only=True,
            trust_remote_code=False, model_kwargs={"use_safetensors": True},
        )
        self.model_identity = snapshot.identity

    def encode(self, texts: Sequence[str], *, is_query: bool) -> Sequence[Sequence[float]]:
        prefix = "query: " if is_query else "passage: "
        return self.model.encode(
            [prefix + text for text in texts],
            normalize_embeddings=True,
            show_progress_bar=False,
        )


@dataclass(frozen=True, slots=True)
class RetrievalPolicy:
    top_k: int = 3
    lexical_weight: float = 0.55
    dense_weight: float = 0.45
    recency_weight: float = 0.08
    importance_weight: float = 0.04
    confidence_weight: float = 0.03
    minimum_relevance: float = 0.12
    recency_half_life_days: float = 180.0
    allowed_lifecycle: frozenset[str] = frozenset({LifecycleStatus.ACTIVE.value})


@dataclass(frozen=True, slots=True)
class RetrievedMemory:
    record: MemoryRecord
    score: float
    components: dict[str, float] = field(default_factory=dict)


def _tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for value in TOKEN_RE.findall(text):
        if re.fullmatch(r"[\u3400-\u9fff]+", value):
            # Character unigrams and bigrams make the dependency-free fallback
            # usable for unsegmented CJK text and mixed-language queries.
            tokens.extend(value)
            tokens.extend(value[index : index + 2] for index in range(len(value) - 1))
        else:
            folded = value.casefold()
            if folded not in STOPWORDS:
                tokens.append(folded)
    return tokens


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _lexical_scores(query: str, documents: Sequence[str]) -> list[float]:
    """The same smoothed TF-IDF cosine, computed over nonzero weights only."""

    query_counts = Counter(_tokens(query))
    if not query_counts:
        return [0.0] * len(documents)
    document_counts = [Counter(_tokens(value)) for value in documents]
    document_count = len(documents)
    # Each document contributes once per token. Rebuilding set(values) for every
    # vocabulary/document pair dominated large scopes. Sparse counts preserve
    # exactly the original DF formula.
    document_frequency = Counter(token for counts in document_counts for token in counts)
    idf = {
        token: math.log((1 + document_count) / (1 + document_frequency[token])) + 1
        for token in query_counts.keys() | document_frequency.keys()
    }

    def vector(counts: Counter) -> dict[str, float]:
        total = sum(counts.values()) or 1
        return {token: count / total * idf[token] for token, count in counts.items()}

    query_vector = vector(query_counts)
    query_norm = math.sqrt(sum(value * value for value in query_vector.values()))
    scores: list[float] = []
    for counts in document_counts:
        document_vector = vector(counts)
        document_norm = math.sqrt(sum(value * value for value in document_vector.values()))
        numerator = sum(query_vector[token] * document_vector[token]
                        for token in query_vector.keys() & document_vector.keys())
        scores.append(numerator / (query_norm * document_norm) if document_norm else 0.0)
    return scores


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class MemoryRetriever:
    """Retrieve only eligible memories owned by the active user.

    Relevance gating is applied to the lexical/dense score before recency,
    importance and confidence boosts.  A recent but unrelated memory therefore
    cannot enter the prompt merely because it is recent.
    """

    def __init__(
        self,
        records: Sequence[MemoryRecord],
        *,
        encoder: DenseEncoder | None = None,
        policy: RetrievalPolicy | None = None,
    ) -> None:
        self.records = list(records)
        self.encoder = encoder
        self.policy = policy or RetrievalPolicy()

    def search(
        self,
        query: str,
        *,
        user_id: str,
        as_of: str | None = None,
        allow_restricted: bool = False,
        memory_types: set[str] | None = None,
    ) -> list[RetrievedMemory]:
        now = _parse_time(as_of) if as_of else datetime.now(timezone.utc)
        candidates = [
            record
            for record in self.records
            if record.user_id == user_id
            and record.lifecycle_status in self.policy.allowed_lifecycle
            and (memory_types is None or record.memory_type in memory_types)
            and (allow_restricted or record.sensitivity != Sensitivity.RESTRICTED.value)
            and is_valid_at(record, now)
        ]
        if not candidates or not query.strip():
            return []

        texts = [record.content for record in candidates]
        lexical = _lexical_scores(query, texts)
        dense = [0.0] * len(candidates)
        if self.encoder is not None:
            query_vector = self.encoder.encode([query], is_query=True)[0]
            document_vectors = self.encoder.encode(texts, is_query=False)
            dense = [max(0.0, _cosine(query_vector, vector)) for vector in document_vectors]

        relevance_weight = self.policy.lexical_weight + (
            self.policy.dense_weight if self.encoder is not None else 0.0
        )
        ranked: list[RetrievedMemory] = []
        for index, record in enumerate(candidates):
            raw_relevance = (
                self.policy.lexical_weight * lexical[index]
                + (self.policy.dense_weight * dense[index] if self.encoder is not None else 0.0)
            ) / relevance_weight
            if raw_relevance < self.policy.minimum_relevance:
                continue
            age_days = max(0.0, (now - _parse_time(record.updated_at)).total_seconds() / 86400)
            recency = math.exp(-math.log(2) * age_days / self.policy.recency_half_life_days)
            score = (
                raw_relevance
                + self.policy.recency_weight * recency
                + self.policy.importance_weight * record.importance
                + self.policy.confidence_weight * record.confidence
            )
            ranked.append(
                RetrievedMemory(
                    record=record,
                    score=score,
                    components={
                        "lexical": lexical[index],
                        "dense": dense[index],
                        "relevance": raw_relevance,
                        "recency": recency,
                        "importance": record.importance,
                        "confidence": record.confidence,
                    },
                )
            )
        ranked.sort(key=lambda item: (-item.score, item.record.memory_id))
        return ranked[: self.policy.top_k]
