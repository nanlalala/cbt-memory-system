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


class SourcePolicyError(ValueError):
    """A source or chunk is not approved for this frozen research run."""


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
        source = self.require_enabled(chunk["source_id"])
        result = deepcopy(chunk)
        result.update({
            "source_registry_fingerprint": self.fingerprint,
            "source_version_fingerprint": self.source_version_fingerprint(source["id"]),
            "source_sha256": source["content_sha256"],
            "chunk_text_sha256": text_sha256(result["text"]),
        })
        return result

    def validate_chunk(self, chunk: dict[str, Any]) -> None:
        source = self.require_enabled(chunk.get("source_id", ""))
        expected = {
            "source_registry_fingerprint": self.fingerprint,
            "source_version_fingerprint": self.source_version_fingerprint(source["id"]),
            "source_sha256": source["content_sha256"],
            "chunk_text_sha256": text_sha256(chunk.get("text", "")),
        }
        if any(chunk.get(key) != value for key, value in expected.items()):
            raise SourcePolicyError(f"{source['id']}: legacy or stale chunk provenance; rebuild corpus")
        if chunk.get("title") != source["title"] or chunk.get("role") != source["role"]:
            raise SourcePolicyError(f"{source['id']}: chunk source metadata disagrees with registry")

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
        if max_excerpt_chars <= 0:
            raise ValueError("max_excerpt_chars must be positive")
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
