from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules/cbt-knowledge-rag"))

from source_registry import KnowledgeConfig, SourcePolicyError, SourceRegistry  # noqa: E402


def fixture_payload() -> dict:
    return {
        "schema_version": 1,
        "registry_version": "synthetic-test-v1",
        "sources": [{
            "id": "synthetic", "title": "Original test fixture", "file": "fixture.pdf",
            "role": "test_only", "access": "project_authored", "coverage": "fixture_only",
            "source_url": "https://example.invalid/fixture", "version": "fixture-v1",
            "content_sha256": hashlib.sha256(b"fixture bytes").hexdigest(),
            "retrieval_enabled": True,
            "license": {
                "id": "test-fixture-only", "status": "project_authored",
                "evidence_url": "https://example.invalid/fixture-review",
                "reviewed_on": "2026-10-14", "approved_by": "test-reviewer",
                "approval_reference": "synthetic fixture, no clinical content",
                "approved_uses": ["retrieval"], "usage_limits": ["Test fixture only"],
            },
        }],
    }


def raw_fixture_chunk() -> dict:
    return {
        "chunk_id": "synthetic:one", "source_id": "synthetic", "title": "Original test fixture",
        "role": "test_only", "section": "Test", "page_start": 1, "page_end": 1,
        "citation": "Original test fixture, p. 1", "cbt_topics": [], "text": "Remember the blue test notebook.",
        "access": "project_authored", "coverage": "fixture_only", "risk_level": "low",
    }


def fixture_chunk(registry: SourceRegistry) -> dict:
    return registry.stamp_chunk(raw_fixture_chunk())


def fixture_config(registry: SourceRegistry, ids: tuple[str, ...] = ("synthetic",)) -> KnowledgeConfig:
    from cbt_rag_v1 import CORPUS_CHUNKER_CONFIG, with_model_config
    return KnowledgeConfig.from_payload(with_model_config({"sources": [{"id": source_id} for source_id in ids]}), registry,
                                        chunker_config=CORPUS_CHUNKER_CONFIG)


class SourceRegistryTests(unittest.TestCase):
    def test_real_sources_default_to_disabled_pending_review(self) -> None:
        registry = SourceRegistry.from_path()
        self.assertEqual(len(registry.sources), 7)
        self.assertTrue(all(not source["retrieval_enabled"] for source in registry.sources))
        for source in registry.sources:
            with self.assertRaises(SourcePolicyError):
                registry.require_enabled(source["id"])

    def test_enable_flag_without_review_does_not_authorize_source(self) -> None:
        payload = fixture_payload()
        payload["sources"][0]["license"]["status"] = "permission_pending"
        with self.assertRaisesRegex(SourcePolicyError, "authorization"):
            SourceRegistry(payload)

    def test_enable_requires_exact_content_pin_and_audit_evidence(self) -> None:
        for key in ("content_sha256", "approval_reference"):
            payload = fixture_payload()
            if key == "content_sha256":
                payload["sources"][0][key] = None
            else:
                payload["sources"][0]["license"][key] = None
            with self.assertRaises(SourcePolicyError):
                SourceRegistry(payload)

    def test_unknown_disabled_and_legacy_chunks_are_rejected(self) -> None:
        registry = SourceRegistry(fixture_payload())
        chunk = fixture_chunk(registry)
        bad_source = deepcopy(chunk)
        bad_source["source_id"] = "not_registered"
        legacy = deepcopy(chunk)
        del legacy["source_registry_fingerprint"]
        for invalid in (bad_source, legacy):
            with self.assertRaises(SourcePolicyError):
                registry.validate_chunk(invalid)
        payload = fixture_payload()
        payload["sources"][0]["retrieval_enabled"] = False
        with self.assertRaisesRegex(SourcePolicyError, "disabled"):
            SourceRegistry(payload).validate_chunk(chunk)

    def test_changed_version_or_text_invalidates_chunk(self) -> None:
        registry = SourceRegistry(fixture_payload())
        chunk = fixture_chunk(registry)
        changed = deepcopy(chunk)
        changed["text"] = "changed text"
        with self.assertRaises(SourcePolicyError):
            registry.validate_chunk(changed)
        payload = fixture_payload()
        payload["sources"][0]["version"] = "fixture-v2"
        with self.assertRaises(SourcePolicyError):
            SourceRegistry(payload).validate_chunk(chunk)

    def test_source_bytes_must_match_pinned_hash(self) -> None:
        registry = SourceRegistry(fixture_payload())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.pdf"
            path.write_bytes(b"fixture bytes")
            registry.verify_source_file("synthetic", path)
            path.write_bytes(b"different bytes")
            with self.assertRaisesRegex(SourcePolicyError, "pinned"):
                registry.verify_source_file("synthetic", path)

    def test_corpus_fingerprint_binds_chunk_order_and_configuration(self) -> None:
        registry = SourceRegistry(fixture_payload())
        one = fixture_chunk(registry)
        two = registry.stamp_chunk({**raw_fixture_chunk(), "chunk_id": "synthetic:two", "text": "second fixture"})
        baseline = registry.corpus_fingerprint([one, two], chunker_config={"size": 100})
        self.assertNotEqual(baseline, registry.corpus_fingerprint([two, one], chunker_config={"size": 100}))
        self.assertNotEqual(baseline, registry.corpus_fingerprint([one, two], chunker_config={"size": 200}))

    def test_evidence_card_has_provenance_bounded_excerpt_and_limits(self) -> None:
        registry = SourceRegistry(fixture_payload())
        card = registry.evidence_card(fixture_chunk(registry), retrieval_rank=1, max_excerpt_chars=8).to_dict()
        self.assertEqual(card["excerpt"], "Remember")
        self.assertTrue(card["excerpt_truncated"])
        self.assertEqual(card["registry_fingerprint"], registry.fingerprint)
        self.assertEqual(card["source_version"], "fixture-v1")
        self.assertEqual(card["license_status"], "project_authored")
        self.assertEqual(card["usage_limits"], ("Test fixture only",))
        self.assertEqual(card["evidence_kind"], "retrieved_excerpt_not_clinical_validation")
        self.assertFalse(card["external_model_transmission_enabled"])
        self.assertFalse(card["corpus_redistribution_enabled"])

    def test_local_retrieval_does_not_authorize_external_transmission(self) -> None:
        registry = SourceRegistry(fixture_payload())
        registry.require_use("synthetic", "retrieval")
        with self.assertRaisesRegex(SourcePolicyError, "external_model_transmission"):
            registry.evidence_card(fixture_chunk(registry), for_external_model=True)
        with self.assertRaisesRegex(SourcePolicyError, "corpus_redistribution"):
            registry.require_use("synthetic", "corpus_redistribution")
        payload = fixture_payload()
        payload["sources"][0]["license"]["approved_uses"].append("external_model_transmission")
        approved = SourceRegistry(payload)
        card = approved.evidence_card(fixture_chunk(approved), for_external_model=True)
        self.assertTrue(card.external_model_transmission_enabled)
        self.assertFalse(card.corpus_redistribution_enabled)

    def test_retriever_rejects_disabled_legacy_chunks_before_model_loading(self) -> None:
        from cbt_rag_v1 import Retriever
        with self.assertRaises(SourcePolicyError):
            Retriever([{"source_id": "beck_cbt_3e_sample"}], download_models=False)
        registry = SourceRegistry(fixture_payload())
        chunk = fixture_chunk(registry)
        config = fixture_config(registry)
        retriever = Retriever([chunk], download_models=False, registry=registry,
                              knowledge_config=config, manifest=config.corpus_manifest([chunk], registry))
        ids, _ = retriever.rank("blue notebook", "tfidf", 1)
        self.assertEqual(ids, [0])
        self.assertEqual(retriever.evidence_cards(ids)[0]["evidence_id"], chunk["chunk_id"])
        with self.assertRaises(SourcePolicyError):
            retriever.assert_external_transmission(ids)

    def test_build_excludes_disabled_sources_before_pdf_read(self) -> None:
        from cbt_rag_v1 import build_corpus
        registry = SourceRegistry.from_path()
        with patch("cbt_rag_v1.make_chunks") as extract:
            with self.assertRaisesRegex(SourcePolicyError, "No approved"):
                build_corpus(registry=registry)
            extract.assert_not_called()

    def test_build_includes_only_approved_sources_and_writes_frozen_manifest(self) -> None:
        from cbt_rag_v1 import CORPUS_CHUNKER_CONFIG, build_corpus
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            pdf_path = temporary / "fixture.pdf"
            pdf_path.write_bytes(b"fixture bytes")
            payload = fixture_payload()
            payload["sources"][0]["file"] = str(pdf_path)
            disabled = deepcopy(payload["sources"][0])
            disabled["id"] = "disabled"
            disabled["retrieval_enabled"] = False
            payload["sources"].append(disabled)
            registry = SourceRegistry(payload)
            config = temporary / "selection.yaml"
            config.write_text("sources:\n  - id: synthetic\n  - id: disabled\n", encoding="utf-8")
            with patch("cbt_rag_v1.DATA", temporary), patch("cbt_rag_v1.make_chunks", return_value=[raw_fixture_chunk()]) as extract:
                chunks = build_corpus(config, registry=registry)
            self.assertEqual(extract.call_count, 1)
            self.assertEqual({chunk["source_id"] for chunk in chunks}, {"synthetic"})
            manifest = json.loads((temporary / "corpus_manifest.json").read_text())
            self.assertEqual(manifest["registry_fingerprint"], registry.fingerprint)
            self.assertEqual(manifest["corpus_fingerprint"], fixture_config(registry, ("synthetic", "disabled")).corpus_manifest(chunks, registry)["corpus_fingerprint"])

    def test_alternate_selection_config_uses_same_registry_for_retrieval(self) -> None:
        from cbt_rag_v1 import configured_registry, load_knowledge_config, Retriever
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            (temporary / "review.json").write_text(json.dumps(fixture_payload()), encoding="utf-8")
            config = temporary / "selection.yaml"
            config.write_text("source_registry: review.json\nsources:\n  - id: synthetic\n", encoding="utf-8")
            with patch.dict("os.environ", {"CBT_KB_CONFIG": str(config)}):
                registry = configured_registry()
                config = load_knowledge_config(registry=registry)
                chunks = [fixture_chunk(registry)]
                retriever = Retriever(chunks, download_models=False,
                                      manifest=config.corpus_manifest(chunks, registry))
            self.assertEqual(retriever.registry.fingerprint, registry.fingerprint)

    def test_every_evidence_metadata_field_is_integrity_bound(self) -> None:
        registry = SourceRegistry(fixture_payload())
        original = fixture_chunk(registry)
        changes = {
            "section": "Forged section", "page_start": 2, "page_end": 3,
            "citation": "Fabricated citation", "cbt_topics": ["suicide_safety"],
            "risk_level": "high", "text": "Altered text", "chunk_id": "synthetic:other",
            "title": "Fake source", "role": "safety", "access": "other", "coverage": "other",
        }
        for field, value in changes.items():
            with self.subTest(field=field), self.assertRaises(SourcePolicyError):
                registry.evidence_card({**original, field: value})

    def test_invalid_metadata_schema_is_rejected_before_stamping(self) -> None:
        registry = SourceRegistry(fixture_payload())
        for changes in ({"page_start": 0}, {"page_start": True}, {"page_start": 3, "page_end": 2},
                        {"cbt_topics": "unsafe"}, {"cbt_topics": ["one", "one"]},
                        {"risk_level": "critical"}, {"citation": ""}, {"text": []}):
            with self.subTest(changes=changes), self.assertRaises(SourcePolicyError):
                registry.stamp_chunk({**raw_fixture_chunk(), **changes})

    def test_loaded_or_legacy_chunks_are_not_silently_restamped(self) -> None:
        registry = SourceRegistry(fixture_payload())
        with self.assertRaisesRegex(SourcePolicyError, "re-stamp"):
            registry.stamp_chunk(fixture_chunk(registry))
        legacy = raw_fixture_chunk()
        legacy["chunk_text_sha256"] = "0" * 64
        with self.assertRaisesRegex(SourcePolicyError, "re-stamp"):
            registry.stamp_chunk(legacy)


if __name__ == "__main__":
    unittest.main()
