from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules/agent-integration"))
from knowledge_adapter_v2 import KnowledgeAdapter  # noqa: E402
from source_registry import KnowledgeConfig, SourceRegistry  # noqa: E402


class FixtureRetriever:
    """Project-authored mechanical fixture, not an empirical knowledge result."""

    def __init__(self, registry, chunks):
        self.registry = registry
        self.chunks = deepcopy(chunks)
        self.knowledge_config = KnowledgeConfig.from_payload(
            {"sources": [{"id": source["id"]} for source in registry.sources]}, registry,
            chunker_config={"fixture": "v2"},
        )
        self.manifest = self.knowledge_config.corpus_manifest(self.chunks, registry)
        self.corpus_fingerprint = self.manifest["corpus_fingerprint"]
        self.calls = []
        self.after_rank = None
        self.indices = list(range(len(chunks)))

    def validate_corpus(self):
        self.knowledge_config.validate_manifest(self.chunks, self.manifest, self.registry)

    def rank(self, query, mode, k=10):
        self.calls.append((query, mode, k))
        if self.after_rank:
            self.after_rank()
        return self.indices[:k], 2.5


class KnowledgeAdapterTests(unittest.TestCase):
    def fixture(self, *, external=True, texts=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        directory = Path(tmp.name)
        data = b"project authored fixture bytes; no actual clinical material"
        (directory / "fixture.pdf").write_bytes(data)
        payload = {
            "schema_version": 1, "registry_version": "knowledge-fixture-v2",
            "sources": [{
                "id": "fixture", "title": "Project fixture", "file": "fixture.pdf",
                "role": "test_only", "access": "project_authored", "coverage": "fixture_only",
                "source_url": "https://example.invalid/fixture", "version": "v2",
                "content_sha256": hashlib.sha256(data).hexdigest(), "retrieval_enabled": True,
                "page_count": 2,
                "license": {
                    "id": "test-fixture-only", "status": "project_authored",
                    "evidence_url": "https://example.invalid/review", "reviewed_on": "2026-11-08",
                    "approved_by": "test fixture", "approval_reference": "original engineering fixture",
                    "approved_uses": ["retrieval"] + (["external_model_transmission"] if external else []),
                    "usage_limits": ["Engineering checks only"],
                },
            }],
        }
        registry = SourceRegistry(payload)
        texts = texts or ["Review homework practice and barriers together."]
        chunks = [registry.stamp_chunk({
            "chunk_id": f"fixture:{i}", "source_id": "fixture", "title": "Project fixture",
            "section": "Original engineering example", "page_start": i + 1, "page_end": i + 1,
            "role": "test_only", "access": "project_authored", "coverage": "fixture_only",
            "cbt_topics": ["homework_review"], "risk_level": "low", "text": text,
            "citation": f"Project fixture, p. {i + 1}",
        }) for i, text in enumerate(texts)]
        retriever = FixtureRetriever(registry, chunks)
        return registry, retriever, directory

    def adapter(self, **kwargs):
        registry, retriever, directory = self.fixture(**kwargs)
        adapter = KnowledgeAdapter(retriever, registry=registry, mode="K1", source_root=directory,
                                   fixture_only=True)
        return adapter, retriever, directory

    def test_all_seven_real_sources_stay_unavailable_in_k1(self):
        result = KnowledgeAdapter(mode="K1").retrieve("CBT想法记录怎么使用？", task_kind="thought_record")
        self.assertEqual(result.status, "unavailable")
        self.assertEqual(result.reason, "no_approved_pinned_sources")
        self.assertFalse(result.has_support)
        self.assertEqual(result.items, ())
        self.assertEqual(result.evidence_status, "no_external_knowledge_support")

    def test_k0_never_calls_knowledge_retriever(self):
        _, retriever, directory = self.adapter()
        adapter = KnowledgeAdapter(retriever, registry=retriever.registry, mode="K0",
                                   source_root=directory, fixture_only=True)
        result = adapter.retrieve("review homework", task_kind="homework_review")
        self.assertEqual(result.status, "disabled")
        self.assertEqual(retriever.calls, [])

    def test_unrelated_task_exits_without_using_rank(self):
        adapter, retriever, _ = self.adapter()
        result = adapter.retrieve("Explain Python recursion", task_kind="general")
        self.assertEqual(result.status, "no_hit")
        self.assertEqual(result.reason, "task_does_not_request_cbt_material")
        self.assertEqual(retriever.calls, [])

    def test_query_strategy_is_identical_for_k0_k1(self):
        adapter, retriever, directory = self.adapter()
        k0 = KnowledgeAdapter(retriever, registry=retriever.registry, mode="K0",
                              source_root=directory, fixture_only=True)
        self.assertEqual(adapter.policy_hash, k0.policy_hash)
        adapter.retrieve("这次作业怎么回顾？", task_kind="homework_review")
        self.assertIn("homework practice review barriers", retriever.calls[0][0])

    def test_retrieved_fixture_has_consistent_chunk_source_pages_and_label(self):
        adapter, _, _ = self.adapter()
        result = adapter.retrieve("review homework", task_kind="homework_review")
        self.assertEqual(result.status, "retrieved")
        self.assertTrue(result.has_support)
        self.assertEqual(result.evidence_status, "fixture_not_real_k1")
        item = result.items[0]
        self.assertEqual((item["source_id"], item["chunk_id"], item["page_start"], item["page_end"]),
                         ("fixture", "fixture:0", 1, 1))
        self.assertEqual(item["citation"], "Project fixture, p. 1")
        self.assertGreater(item["relevance"], 0)
        self.assertEqual(item["evidence_status"], "fixture_not_real_k1")

    def test_relevance_gate_can_return_no_hit_after_ranking(self):
        adapter, retriever, _ = self.adapter(texts=["Blue notebook beside the window."])
        result = adapter.retrieve("homework practice", task_kind="homework_review")
        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(result.status, "no_hit")
        self.assertEqual(result.items, ())
        self.assertFalse(result.has_support)

    def test_local_permission_does_not_authorize_external_context(self):
        adapter, _, _ = self.adapter(external=False)
        outside = adapter.retrieve("homework practice", task_kind="homework_review", external=True)
        self.assertEqual(outside.status, "unavailable")
        self.assertFalse(outside.has_support)
        local = adapter.retrieve("homework practice", task_kind="homework_review", external=False)
        self.assertEqual(local.status, "retrieved")

    def test_changed_source_bytes_block_old_index(self):
        adapter, retriever, directory = self.adapter()
        (directory / "fixture.pdf").write_bytes(b"changed bytes")
        result = adapter.retrieve("homework", task_kind="homework_review")
        self.assertEqual(result.status, "unavailable")
        self.assertEqual(retriever.calls, [])

    def test_stamped_but_wrong_citation_pages_are_rejected(self):
        adapter, retriever, _ = self.adapter()
        raw = {key: value for key, value in retriever.chunks[0].items()
               if key not in {"source_registry_fingerprint", "source_version_fingerprint", "source_sha256",
                              "chunk_text_sha256", "chunk_sha256", "chunk_schema_version"}}
        raw["citation"] = "Project fixture, p. 2"
        retriever.chunks = [retriever.registry.stamp_chunk(raw)]
        retriever.manifest = retriever.knowledge_config.corpus_manifest(retriever.chunks, retriever.registry)
        retriever.corpus_fingerprint = retriever.manifest["corpus_fingerprint"]
        result = adapter.retrieve("homework", task_kind="homework_review")
        self.assertEqual(result.status, "unavailable")
        self.assertEqual(retriever.calls, [])

    def test_changed_manifest_is_not_silently_restamped(self):
        adapter, retriever, _ = self.adapter()
        retriever.manifest["chunk_count"] = 900
        result = adapter.retrieve("homework", task_kind="homework_review")
        self.assertEqual(result.status, "unavailable")
        self.assertEqual(retriever.calls, [])

    def test_out_of_range_rank_and_duplicate_indices_fail_closed(self):
        adapter, retriever, _ = self.adapter()
        for indices in [[5], [0, 0], [True]]:
            retriever.indices = indices
            with self.subTest(indices=indices):
                result = adapter.retrieve("homework", task_kind="homework_review")
                self.assertEqual(result.status, "unavailable")
                self.assertEqual(result.items, ())

    def test_corpus_mutation_during_retrieval_cannot_escape_validation(self):
        adapter, retriever, _ = self.adapter()
        retriever.after_rank = lambda: retriever.chunks[0].update(text="a mutated quotation")
        result = adapter.retrieve("homework", task_kind="homework_review")
        self.assertEqual(result.status, "unavailable")
        self.assertFalse(result.has_support)

    def test_fixture_mode_is_explicit_and_real_pdf_validation_is_not_bypassed(self):
        _, retriever, directory = self.adapter()
        adapter = KnowledgeAdapter(retriever, registry=retriever.registry, mode="K1", source_root=directory)
        result = adapter.retrieve("homework", task_kind="homework_review")
        self.assertEqual(result.status, "unavailable")
        self.assertNotEqual(result.evidence_status, "fixture_not_real_k1")

    def test_ranked_irrelevant_first_chunk_does_not_hide_later_relevant_chunk(self):
        adapter, _, _ = self.adapter(texts=["Blue notebook beside the window.", "Review homework practice and barriers."])
        result = adapter.retrieve("review homework", task_kind="homework_review")
        self.assertEqual(result.status, "retrieved")
        self.assertEqual([item["chunk_id"] for item in result.items], ["fixture:1"])
        self.assertEqual(result.items[0]["page_start"], 2)

    def test_invalid_public_arguments_do_not_coerce(self):
        for value in [True, "0.1", float("nan"), -0.1]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    KnowledgeAdapter(minimum_relevance=value)
        adapter = KnowledgeAdapter()
        for query, task_kind in [("", "general"), ("CBT", "invented_route")]:
            with self.assertRaises(ValueError):
                adapter.retrieve(query, task_kind=task_kind)


if __name__ == "__main__":
    unittest.main()
