"""A fail-closed bridge from the existing CBT corpus into dialogue context.

Importing this module loads no model and performs no network work. K0 and K1
share a versioned query/route policy. K1 can only expose existing, reviewed,
byte-pinned corpus chunks; project-authored fixtures are labelled separately.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


KNOWLEDGE_ROOT = Path(__file__).resolve().parents[1] / "cbt-knowledge-rag"
if str(KNOWLEDGE_ROOT) not in sys.path:
    sys.path.insert(0, str(KNOWLEDGE_ROOT))
from source_registry import KnowledgeConfig, SourcePolicyError, SourceRegistry  # noqa: E402


QUERY_POLICY_VERSION = "cbt-knowledge-query-v2.0"
# These are retrieval expansions, not therapeutic recommendations. They are
# fixed independently of memory arm, source labels, scoring answers or gold.
TASK_ROUTES = {
    "homework_review": "homework practice review barriers behavioral experiment",
    "homework_followup": "homework practice review barriers behavioral experiment",
    "goal_review": "goal action plan agenda progress",
    "goal_followup": "goal action plan agenda progress",
    "cognitive_restructuring": "automatic thoughts evidence alternative interpretation thought record",
    "thought_record": "situation automatic thoughts emotion evidence thought record",
    "emotion_followup": "emotion situation automatic thoughts behavior",
    "skill_application": "cognitive behavioral skills behavioral experiment practice",
    "cbt_question": "cognitive behavioral therapy",
    "knowledge_request": "cognitive behavioral therapy",
    "reflection": "",
    "general": "",
}
_CBT_CUE = re.compile(
    r"\b(?:cbt|cognitive|homework|thought\s+record|behavioral\s+experiment)\b|"
    r"认知行为|想法记录|思维记录|行为实验|认知重构|自动想法|咨询作业|练习复盘",
    re.IGNORECASE,
)
_PAGE_CITATION = re.compile(r"\bpp?\.\s*(\d+)(?:\s*[-–—]\s*(\d+))?", re.IGNORECASE)
_ZH_PAGE_CITATION = re.compile(r"第\s*(\d+)(?:\s*[-–—至]\s*(\d+))?\s*页")
_STOPWORDS = frozenset({"the", "of", "to", "and", "a", "in", "is", "for", "it", "you", "your", "an", "with"})


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _tokens(text: str) -> Counter:
    latin = [token for token in re.findall(r"[a-z0-9]+", text.casefold()) if token not in _STOPWORDS]
    chinese: list[str] = []
    for run in re.findall(r"[\u3400-\u9fff]+", text):
        chinese.extend(run)
        chinese.extend(run[i:i + 2] for i in range(len(run) - 1))
    return Counter(latin + chinese)


def _relevance(query: str, text: str) -> float:
    left, right = _tokens(query), _tokens(text)
    denominator = math.sqrt(sum(x*x for x in left.values()) * sum(x*x for x in right.values()))
    return sum(count * right[token] for token, count in left.items()) / denominator if denominator else 0.0


@dataclass(frozen=True)
class KnowledgeSelection:
    items: tuple[dict[str, Any], ...]
    status: str
    policy_hash: str
    evidence_status: str
    reason: str
    latency_ms: float | None = None
    corpus_fingerprint: str | None = None

    @property
    def has_support(self) -> bool:
        """Only a populated packet claims that external context was supplied."""
        return self.status == "retrieved" and bool(self.items)


@dataclass(frozen=True, slots=True)
class _SourceSnapshot:
    """One immutable version used both for source identity and page parsing."""
    source_id: str
    path: Path
    data: bytes
    sha256: str


class KnowledgeAdapter:
    """Use a prebuilt Retriever without relaxing its existing source checks.

    ``fixture_only`` is solely for project-authored, fixture-scoped corpora. It
    never turns the seven disabled old sources on, nor establishes real K1.
    Source bytes and corpus chunks are snapshotted before ranking. Source paths
    are rechecked after ranking and packet construction; the packet is built
    from the original version, not from a later path read. This is an immutable
    read snapshot, not a filesystem-wide lock or a promise that external writers
    cannot update a path after the completed retrieval.
    """

    def __init__(self, retriever: Any = None, *, registry: SourceRegistry | None = None,
                 mode: str = "K0", minimum_relevance: float = 0.08,
                 source_root: str | Path | None = None, top_k: int = 3,
                 retrieval_mode: str = "lexical_hybrid", fixture_only: bool = False):
        if mode not in {"K0", "K1"}:
            raise ValueError("knowledge mode must be K0 or K1")
        if type(top_k) is not int or not 1 <= top_k <= 10:
            raise ValueError("knowledge top_k must be an integer from 1 to 10")
        if type(minimum_relevance) not in {int, float} or not math.isfinite(minimum_relevance) or not 0 <= minimum_relevance <= 1:
            raise ValueError("minimum_relevance must be finite and between zero and one")
        if type(fixture_only) is not bool or type(retrieval_mode) is not str or not retrieval_mode:
            raise ValueError("explicit retrieval mode and fixture flag required")
        self.retriever = retriever
        self.registry = registry if registry is not None else SourceRegistry.from_path()
        self.mode = mode
        self.minimum_relevance = float(minimum_relevance)
        self.source_root = Path(source_root) if source_root is not None else KNOWLEDGE_ROOT
        self.top_k = top_k
        self.retrieval_mode = retrieval_mode
        self.fixture_only = fixture_only
        # Mode and memory condition are deliberately excluded: this identifies
        # the common query policy, not whether an external channel is enabled.
        self.policy_hash = _digest({"version": QUERY_POLICY_VERSION, "routes": TASK_ROUTES,
                                    "top_k": top_k, "retrieval_mode": retrieval_mode,
                                    "minimum_relevance": self.minimum_relevance,
                                    "relevance": "query_body_token_cosine_v1"})

    def _selection(self, status: str, reason: str, *, items=(), latency_ms=None, corpus=None) -> KnowledgeSelection:
        evidence_status = ("fixture_not_real_k1" if self.fixture_only else
                           "retrieved_external_material_not_clinical_validation" if items else
                           "no_external_knowledge_support")
        return KnowledgeSelection(tuple(items), status, self.policy_hash, evidence_status,
                                  reason, latency_ms, corpus)

    def _query(self, query: str, task_kind: str) -> str | None:
        if type(query) is not str or not query.strip():
            raise ValueError("knowledge query must be a nonempty string")
        if task_kind not in TASK_ROUTES:
            raise ValueError("unknown knowledge task route")
        expansion = TASK_ROUTES[task_kind]
        if not expansion and not _CBT_CUE.search(query):
            return None
        return query.strip() + ("\n" + expansion if expansion else "")

    def _source_path(self, source: dict) -> Path:
        relative = Path(source["file"])
        root = self.source_root.resolve(strict=True)
        path = root / relative
        if relative.is_absolute() or path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
            raise SourcePolicyError("source file must remain under the selected corpus root")
        return path

    def _validate_corpus(self) -> tuple[list[dict], str, dict[str, int], tuple[_SourceSnapshot, ...]]:
        retriever = self.retriever
        if retriever is None:
            raise SourcePolicyError("no reviewed corpus retriever is configured")
        validate = getattr(retriever, "validate_corpus", None)
        if not callable(validate) or not callable(getattr(retriever, "rank", None)):
            raise SourcePolicyError("retriever must expose its corpus validation contract")
        validate()
        own_registry = getattr(retriever, "registry", None)
        config, manifest = getattr(retriever, "knowledge_config", None), getattr(retriever, "manifest", None)
        chunks = getattr(retriever, "chunks", None)
        if not isinstance(own_registry, SourceRegistry) or own_registry.fingerprint != self.registry.fingerprint:
            raise SourcePolicyError("knowledge adapter and retriever registries disagree")
        if not isinstance(config, KnowledgeConfig) or type(chunks) is not list or not chunks:
            raise SourcePolicyError("a selected, nonempty, bound corpus is required")
        config.validate_manifest(chunks, manifest, self.registry)
        corpus = manifest["corpus_fingerprint"]
        if getattr(retriever, "corpus_fingerprint", None) != corpus:
            raise SourcePolicyError("retriever corpus identity differs from manifest")
        counts, snapshots = {}, []
        for source_id in sorted({chunk["source_id"] for chunk in chunks}):
            source = config.require_selected(source_id, self.registry)
            path = self._source_path(source)
            # Read once. Hashing one path read and parsing a second path read
            # would let a concurrent replacement change the actual PDF version.
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if digest != source['content_sha256']:
                raise SourcePolicyError('source file differs from pinned bytes')
            snapshots.append(_SourceSnapshot(source_id, path, data, digest))
            fixture_source = source["role"] == "test_only" or source["coverage"] == "fixture_only"
            if fixture_source and not self.fixture_only:
                raise SourcePolicyError("test fixtures require an explicit nonempirical channel")
            if self.fixture_only:
                if not (source["role"] == "test_only" and source["coverage"] == "fixture_only"
                        and source["license"]["status"] == "project_authored"):
                    raise SourcePolicyError("fixture mode accepts only explicit project test fixtures")
                if source.get("page_count") is None:
                    raise SourcePolicyError("fixture source requires an explicit fixture page count")
                counts[source_id] = source["page_count"]
            else:
                from pypdf import PdfReader
                if not data[:8].startswith(b"%PDF-"):
                    raise SourcePolicyError("pinned knowledge source is not a PDF")
                try:
                    count = len(PdfReader(io.BytesIO(data)).pages)
                except Exception as exc:
                    raise SourcePolicyError("pinned PDF structure could not be verified") from exc
                if count < 1 or (source.get("page_count") is not None and source["page_count"] != count):
                    raise SourcePolicyError("registry page count differs from pinned PDF")
                counts[source_id] = count
        for chunk in chunks:
            self.registry.validate_chunk(chunk)
            if chunk["page_end"] > counts[chunk["source_id"]]:
                raise SourcePolicyError("chunk page range exceeds pinned document")
            match = _PAGE_CITATION.search(chunk["citation"]) or _ZH_PAGE_CITATION.search(chunk["citation"])
            if match is None or (int(match[1]), int(match[2] or match[1])) != (chunk["page_start"], chunk["page_end"]):
                raise SourcePolicyError("citation page range disagrees with chunk metadata")
        return deepcopy(chunks), corpus, counts, tuple(snapshots)

    def _verify_source_snapshots(self, snapshots: tuple[_SourceSnapshot, ...]) -> None:
        """Check live source admission without replacing the frozen read data."""
        for snapshot in snapshots:
            source = self.registry.require_enabled(snapshot.source_id)
            path = self._source_path(source)
            if path != snapshot.path or source['content_sha256'] != snapshot.sha256:
                raise SourcePolicyError('source version changed during retrieval')
            if hashlib.sha256(path.read_bytes()).hexdigest() != snapshot.sha256:
                raise SourcePolicyError('source bytes changed during retrieval')

    def retrieve(self, query: str, *, task_kind: str = "general", external: bool = True) -> KnowledgeSelection:
        if type(external) is not bool:
            raise ValueError("external must be boolean")
        routed_query = self._query(query, task_kind)
        if self.mode == "K0":
            return self._selection("disabled", "knowledge_condition_k0")
        if routed_query is None:
            return self._selection("no_hit", "task_does_not_request_cbt_material")
        if not any(source["retrieval_enabled"] for source in self.registry.sources):
            return self._selection("unavailable", "no_approved_pinned_sources")
        try:
            chunks, corpus, _, snapshots = self._validate_corpus()
            indices, latency = self.retriever.rank(routed_query, self.retrieval_mode,
                                                   k=max(10, self.top_k * 4))
            if type(indices) is not list or any(type(index) is not int or not 0 <= index < len(chunks) for index in indices):
                raise SourcePolicyError("retrieval indices must refer to the bound corpus")
            if len(set(indices)) != len(indices):
                raise SourcePolicyError("duplicate ranked chunk indices")
            if type(latency) not in {int, float} or not math.isfinite(latency) or latency < 0:
                raise SourcePolicyError("retrieval latency must be finite and nonnegative")
            self.retriever.validate_corpus()
            if self.retriever.chunks != chunks:
                raise SourcePolicyError("corpus changed while retrieval was running")
            self._verify_source_snapshots(snapshots)
            items = []
            for index in indices:
                chunk = chunks[index]
                score = _relevance(routed_query, chunk["text"])
                if score < self.minimum_relevance:
                    continue
                if external:
                    self.registry.require_use(chunk["source_id"], "external_model_transmission")
                card = self.registry.evidence_card(chunk, retrieval_rank=len(items) + 1,
                                                   max_excerpt_chars=len(chunk["text"]),
                                                   for_external_model=external).to_dict()
                items.append({"source_id": chunk["source_id"], "chunk_id": chunk["chunk_id"],
                              "text": card["excerpt"], "citation": card["citation"],
                              "page_start": card["page_start"], "page_end": card["page_end"],
                              "relevance": score, "chunk_sha256": card["chunk_sha256"],
                              "source_sha256": card["source_sha256"],
                              "evidence_status": "fixture_not_real_k1" if self.fixture_only else card["evidence_kind"]})
                if len(items) == self.top_k:
                    break
            # Evidence cards use only the copied, hash-bound chunks. Recheck
            # live paths once the complete packet is ready, so callbacks or
            # updates during card construction cannot escape the same request.
            self._verify_source_snapshots(snapshots)
            if not items:
                return self._selection("no_hit", "no_chunk_passed_relevance_gate", latency_ms=float(latency), corpus=corpus)
            return self._selection("retrieved", "reviewed_relevant_chunks", items=items,
                                   latency_ms=float(latency), corpus=corpus)
        except (SourcePolicyError, OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError):
            # Do not expose provider/path errors as evidence or silently claim
            # textbook grounding when the channel is unavailable.
            return self._selection("unavailable", "knowledge_source_or_corpus_check_failed")
