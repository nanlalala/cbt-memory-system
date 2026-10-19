"""Fail-closed source policy and provenance for the research knowledge layer.

Public availability, a sample chapter, or possession of a PDF is not permission.
The registry records a project review decision; it is not a legal verifier.
"""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_REGISTRY = Path(__file__).resolve().parent / "source_registry.json"
APPROVED_LICENSE_STATUSES = frozenset({
    "verified_open_license", "verified_permission", "project_authored",
})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CHUNK_FIELDS = frozenset({
    "chunk_id", "source_id", "title", "section", "page_start", "page_end",
    "role", "access", "coverage", "cbt_topics", "risk_level", "text", "citation",
})
PROVENANCE_FIELDS = frozenset({
    "source_registry_fingerprint", "source_version_fingerprint", "source_sha256",
    "chunk_text_sha256", "chunk_sha256", "chunk_schema_version",
})


class SourcePolicyError(ValueError):
    """A source or chunk is not approved for this frozen research run."""


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class KnowledgeConfig:
    """Immutable selection and normalized build settings, not the whole registry.

    Hashes detect integrity/configuration changes; they do not authenticate authors
    or prove that a PDF actually contains the text attributed to it.
    """

    selected_source_ids: tuple[str, ...]
    registry_fingerprint: str
    normalized_config_json: str
    chunker_config_json: str

    def __post_init__(self) -> None:
        if not isinstance(self.selected_source_ids, tuple) or not self.selected_source_ids:
            raise SourcePolicyError("knowledge selection must be a nonempty immutable tuple")
        if any(not isinstance(value, str) or not value for value in self.selected_source_ids):
            raise SourcePolicyError("knowledge source IDs must be nonempty strings")
        if len(set(self.selected_source_ids)) != len(self.selected_source_ids):
            raise SourcePolicyError("duplicate knowledge source selection")
        if not SHA256_RE.fullmatch(self.registry_fingerprint):
            raise SourcePolicyError("knowledge config requires a registry fingerprint")
        payload = json.loads(self.normalized_config_json)
        if not isinstance(payload, dict) or not isinstance(json.loads(self.chunker_config_json), dict):
            raise SourcePolicyError("knowledge config documents must be JSON objects")
        entries = payload.get("sources")
        if not isinstance(entries, list) or any(not isinstance(item, dict) or set(item) != {"id"} for item in entries):
            raise SourcePolicyError("knowledge config requires id-only source entries")
        if tuple(item["id"] for item in entries) != self.selected_source_ids:
            raise SourcePolicyError("normalized config and selected source IDs disagree")

    @classmethod
    def from_payload(
        cls, payload: dict[str, Any], registry: "SourceRegistry", *,
        chunker_config: dict[str, Any],
    ) -> "KnowledgeConfig":
        if not isinstance(payload, dict) or not isinstance(payload.get("sources"), list):
            raise SourcePolicyError("knowledge config must contain a sources list")
        if any(not isinstance(item, dict) or set(item) != {"id"} for item in payload["sources"]):
            raise SourcePolicyError("source selection accepts id-only entries")
        selected = tuple(item["id"] for item in payload["sources"])
        for source_id in selected:
            registry.source(source_id)
        return cls(
            selected, registry.fingerprint,
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
            json.dumps(chunker_config, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint({
            "schema_version": 2, "selected_source_ids": self.selected_source_ids,
            "registry_fingerprint": self.registry_fingerprint,
            "config": json.loads(self.normalized_config_json),
            "chunker": self.chunker_config,
        })

    @property
    def chunker_config(self) -> dict[str, Any]:
        return json.loads(self.chunker_config_json)

    def validate_registry(self, registry: "SourceRegistry") -> None:
        if self.registry_fingerprint != registry.fingerprint:
            raise SourcePolicyError("knowledge config registry fingerprint mismatch")

    def require_selected(self, source_id: str, registry: "SourceRegistry") -> dict[str, Any]:
        self.validate_registry(registry)
        if source_id not in self.selected_source_ids:
            raise SourcePolicyError(f"{source_id}: approved but unselected source")
        return registry.require_enabled(source_id)

    def enabled_source_ids(self, registry: "SourceRegistry") -> tuple[str, ...]:
        self.validate_registry(registry)
        return tuple(source_id for source_id in self.selected_source_ids
                     if registry.source(source_id)["retrieval_enabled"])

    def corpus_manifest(self, chunks: list[dict[str, Any]], registry: "SourceRegistry") -> dict[str, Any]:
        self.validate_registry(registry)
        for chunk in chunks:
            if not isinstance(chunk, dict):
                raise SourcePolicyError("corpus chunks must be JSON objects")
            self.require_selected(chunk.get("source_id", ""), registry)
            registry.validate_chunk(chunk)
        source_ids = self.enabled_source_ids(registry)
        if {chunk["source_id"] for chunk in chunks} != set(source_ids):
            raise SourcePolicyError("corpus source set differs from enabled selected sources")
        if len({chunk["chunk_id"] for chunk in chunks}) != len(chunks):
            raise SourcePolicyError("duplicate corpus chunk ID")
        return {
            "schema_version": 2, "registry_fingerprint": registry.fingerprint,
            "config_fingerprint": self.fingerprint,
            "selected_source_ids": list(self.selected_source_ids),
            "source_ids": list(source_ids), "chunker_config": self.chunker_config,
            "chunk_count": len(chunks),
            "corpus_fingerprint": _fingerprint({
                "config_fingerprint": self.fingerprint,
                "ordered_chunk_digests": [chunk["chunk_sha256"] for chunk in chunks],
            }),
        }

    def validate_manifest(self, chunks: list[dict[str, Any]], manifest: dict[str, Any], registry: "SourceRegistry") -> None:
        if not isinstance(manifest, dict) or manifest != self.corpus_manifest(chunks, registry):
            raise SourcePolicyError("corpus manifest mismatch: rebuild with the selected frozen config")


@dataclass(frozen=True)
class EvidenceCard:
    """A traceable retrieved excerpt, not a claim of clinical validation."""

    evidence_id: str
    source_id: str
    title: str
    source_url: str
    source_version: str
    source_sha256: str
    registry_fingerprint: str
    chunk_text_sha256: str
    chunk_sha256: str
    license_id: str
    license_status: str
    license_evidence_url: str
    approved_uses: tuple[str, ...]
    external_model_transmission_enabled: bool
    corpus_redistribution_enabled: bool
    usage_limits: tuple[str, ...]
    section: str
    page_start: int
    page_end: int
    citation: str
    excerpt: str
    excerpt_truncated: bool
    retrieval_rank: int | None = None
    evidence_kind: str = "retrieved_excerpt_not_clinical_validation"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SourceRegistry:
    def __init__(self, payload: dict[str, Any]):
        self._payload = deepcopy(payload)
        if payload.get("schema_version") != 1 or not isinstance(payload.get("sources"), list):
            raise SourcePolicyError("registry must have schema_version=1 and sources list")
        self._sources: dict[str, dict[str, Any]] = {}
        for source in self._payload["sources"]:
            required = {"id", "title", "file", "role", "access", "coverage",
                        "source_url", "version", "content_sha256", "retrieval_enabled", "license"}
            if not required.issubset(source):
                raise SourcePolicyError("source is missing registry fields")
            source_id = source["id"]
            if not isinstance(source_id, str) or not source_id or source_id in self._sources:
                raise SourcePolicyError("source IDs must be nonempty and unique")
            if type(source["retrieval_enabled"]) is not bool:
                raise SourcePolicyError(f"{source_id}: retrieval_enabled must be a boolean")
            if source.get("page_count") is not None and (type(source["page_count"]) is not int or source["page_count"] < 1):
                raise SourcePolicyError(f"{source_id}: page_count must be a positive integer")
            license_info = source["license"]
            if not isinstance(license_info, dict) or not {
                "id", "status", "evidence_url", "reviewed_on", "approved_by",
                "approval_reference", "approved_uses", "usage_limits",
            }.issubset(license_info):
                raise SourcePolicyError(f"{source_id}: incomplete license review")
            if not isinstance(license_info["approved_uses"], list) or not isinstance(license_info["usage_limits"], list):
                raise SourcePolicyError(f"{source_id}: approved_uses and usage_limits must be lists")
            self._sources[source_id] = source
        self.fingerprint = _fingerprint(self._payload)
        # An enable flag alone must never bypass absent review or source pin.
        for source_id, source in self._sources.items():
            if source["retrieval_enabled"]:
                self.require_enabled(source_id)

    @classmethod
    def from_path(cls, path: str | Path = DEFAULT_REGISTRY) -> "SourceRegistry":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def sources(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(source) for source in self._sources.values())

    def source(self, source_id: str) -> dict[str, Any]:
        if source_id not in self._sources:
            raise SourcePolicyError(f"unregistered knowledge source: {source_id}")
        return deepcopy(self._sources[source_id])

    def require_enabled(self, source_id: str) -> dict[str, Any]:
        source = self.source(source_id)
        if not source["retrieval_enabled"]:
            raise SourcePolicyError(f"{source_id}: retrieval disabled; {source.get('disabled_reason', 'review pending')}")
        license_info = source["license"]
        if license_info["status"] not in APPROVED_LICENSE_STATUSES or "retrieval" not in license_info["approved_uses"]:
            raise SourcePolicyError(f"{source_id}: no reviewed retrieval authorization")
        if not all(license_info[key] for key in (
            "evidence_url", "reviewed_on", "approved_by", "approval_reference",
        )):
            raise SourcePolicyError(f"{source_id}: license approval lacks audit evidence")
        if not isinstance(source["content_sha256"], str) or not SHA256_RE.fullmatch(source["content_sha256"]):
            raise SourcePolicyError(f"{source_id}: exact source bytes must be SHA-256 pinned")
        return source

    def verify_source_file(self, source_id: str, path: str | Path) -> str:
        source = self.require_enabled(source_id)
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if digest != source["content_sha256"]:
            raise SourcePolicyError(f"{source_id}: source file differs from pinned bytes")
        return digest

    def require_use(self, source_id: str, use: str) -> dict[str, Any]:
        """Retrieval, external transmission and redistribution are separate approvals."""
        source = self.require_enabled(source_id)
        if use not in source["license"]["approved_uses"]:
            raise SourcePolicyError(f"{source_id}: {use} is not approved")
        return source

    def source_version_fingerprint(self, source_id: str) -> str:
        return _fingerprint(self.source(source_id))

    def stamp_chunk(self, chunk: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(chunk, dict):
            raise SourcePolicyError("chunk must be a JSON object")
        if PROVENANCE_FIELDS.intersection(chunk):
            raise SourcePolicyError("refusing to re-stamp existing chunk provenance; rebuild from pinned source")
        source = self.require_enabled(chunk["source_id"])
        result = deepcopy(chunk)
        self._validate_chunk_schema(result, source)
        result.update({
            "chunk_schema_version": 2,
            "source_registry_fingerprint": self.fingerprint,
            "source_version_fingerprint": self.source_version_fingerprint(source["id"]),
            "source_sha256": source["content_sha256"],
            "chunk_text_sha256": text_sha256(result["text"]),
        })
        result["chunk_sha256"] = _fingerprint(result)
        return result

    def _validate_chunk_schema(self, chunk: dict[str, Any], source: dict[str, Any]) -> None:
        if not CHUNK_FIELDS.issubset(chunk) or set(chunk) - CHUNK_FIELDS - PROVENANCE_FIELDS:
            raise SourcePolicyError("incomplete or unknown chunk schema fields")
        for key in CHUNK_FIELDS - {"page_start", "page_end", "cbt_topics"}:
            if not isinstance(chunk[key], str) or not chunk[key].strip():
                raise SourcePolicyError(f"invalid chunk {key}")
        if not chunk["chunk_id"].startswith(source["id"] + ":"):
            raise SourcePolicyError("chunk ID must be scoped to its source")
        if any(type(chunk[key]) is not int for key in ("page_start", "page_end")):
            raise SourcePolicyError("chunk page numbers must be integers")
        if not 1 <= chunk["page_start"] <= chunk["page_end"]:
            raise SourcePolicyError("invalid chunk page range")
        if source.get("page_count") is not None and chunk["page_end"] > source["page_count"]:
            raise SourcePolicyError("chunk page range exceeds source page count")
        if not isinstance(chunk["cbt_topics"], list) or any(
            not isinstance(topic, str) or not topic for topic in chunk["cbt_topics"]
        ) or len(set(chunk["cbt_topics"])) != len(chunk["cbt_topics"]):
            raise SourcePolicyError("chunk topics must be a unique list of strings")
        if chunk["risk_level"] not in {"low", "medium", "high"}:
            raise SourcePolicyError("invalid chunk risk level")
        if any(chunk[key] != source[key] for key in ("title", "role", "access", "coverage")):
            raise SourcePolicyError("chunk source metadata disagrees with registry")

    def validate_chunk(self, chunk: dict[str, Any]) -> None:
        if not isinstance(chunk, dict):
            raise SourcePolicyError("chunk must be a JSON object")
        source = self.require_enabled(chunk.get("source_id", ""))
        self._validate_chunk_schema(chunk, source)
        expected = {
            "chunk_schema_version": 2,
            "source_registry_fingerprint": self.fingerprint,
            "source_version_fingerprint": self.source_version_fingerprint(source["id"]),
            "source_sha256": source["content_sha256"],
            "chunk_text_sha256": text_sha256(chunk.get("text", "")),
            "chunk_sha256": _fingerprint({key: value for key, value in chunk.items() if key != "chunk_sha256"}),
        }
        if any(chunk.get(key) != value for key, value in expected.items()):
            raise SourcePolicyError(f"{source['id']}: legacy or stale chunk provenance; rebuild corpus")

    def corpus_fingerprint(self, chunks: list[dict[str, Any]], *, chunker_config: dict[str, Any] | None = None) -> str:
        for chunk in chunks:
            self.validate_chunk(chunk)
        return _fingerprint({
            "registry_fingerprint": self.fingerprint,
            "chunker_config": chunker_config or {},
            "ordered_chunks": [(chunk["chunk_id"], _fingerprint(chunk)) for chunk in chunks],
        })

    def evidence_card(
        self, chunk: dict[str, Any], *, retrieval_rank: int | None = None,
        max_excerpt_chars: int = 600, for_external_model: bool = False,
    ) -> EvidenceCard:
        if type(max_excerpt_chars) is not int or max_excerpt_chars <= 0:
            raise ValueError("max_excerpt_chars must be positive")
        if retrieval_rank is not None and (type(retrieval_rank) is not int or retrieval_rank <= 0):
            raise ValueError("retrieval_rank must be a positive integer")
        self.validate_chunk(chunk)
        source = self.source(chunk["source_id"])
        if for_external_model:
            self.require_use(source["id"], "external_model_transmission")
        license_info = source["license"]
        text = chunk["text"]
        return EvidenceCard(
            evidence_id=chunk["chunk_id"], source_id=source["id"], title=source["title"],
            source_url=source["source_url"], source_version=source["version"],
            source_sha256=source["content_sha256"], registry_fingerprint=self.fingerprint,
            chunk_text_sha256=chunk["chunk_text_sha256"], license_id=license_info["id"],
            chunk_sha256=chunk["chunk_sha256"],
            license_status=license_info["status"], license_evidence_url=license_info["evidence_url"],
            approved_uses=tuple(license_info["approved_uses"]),
            external_model_transmission_enabled="external_model_transmission" in license_info["approved_uses"],
            corpus_redistribution_enabled="corpus_redistribution" in license_info["approved_uses"],
            usage_limits=tuple(license_info["usage_limits"]), section=chunk["section"],
            page_start=chunk["page_start"], page_end=chunk["page_end"], citation=chunk["citation"],
            excerpt=text[:max_excerpt_chars], excerpt_truncated=len(text) > max_excerpt_chars,
            retrieval_rank=retrieval_rank,
        )


if __name__ == "__main__":
    registry = SourceRegistry.from_path()
    print(json.dumps({
        "registry_fingerprint": registry.fingerprint,
        "sources": [{"id": source["id"], "version": source["version"],
                     "retrieval_enabled": source["retrieval_enabled"],
                     "license_status": source["license"]["status"],
                     "disabled_reason": source.get("disabled_reason")}
                    for source in registry.sources],
    }, ensure_ascii=False, indent=2))
