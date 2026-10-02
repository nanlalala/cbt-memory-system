from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for relative in (
    "modules/long-term-memory/schema",
    "modules/long-term-memory/retrieval",
    "modules/agent-integration",
    "evaluation/memory-retrieval",
):
    sys.path.insert(0, str(ROOT / relative))

from benchmark_adapters import RetrievalExample  # noqa: E402
from context_builder import KnowledgeEvidence, build_context  # noqa: E402
from memory_models import MemoryRecord, SourceEvidence  # noqa: E402
from memory_retriever import MemoryRetriever, RetrievalPolicy  # noqa: E402
from run_memory_retrieval_benchmark import documents_for_example  # noqa: E402


def record(
    memory_id: str,
    content: str,
    *,
    user_id: str = "alice",
    status: str = "active",
    sensitivity: str = "standard",
    updated_at: str = "2026-09-01T10:00:00Z",
) -> MemoryRecord:
    return MemoryRecord(
        schema_version="1.0",
        memory_id=memory_id,
        user_id=user_id,
        memory_type="explicit_fact",
        content=content,
        origin="explicit_user_statement",
        confirmation_state="user_confirmed",
        lifecycle_status=status,
        sensitivity=sensitivity,
        confidence=0.9,
        importance=0.7,
        created_at=updated_at,
        observed_at=updated_at,
        updated_at=updated_at,
        source_evidence=SourceEvidence("s1", ["u1"], content),
    )


class MemoryRetrieverTests(unittest.TestCase):
    def test_user_lifecycle_and_sensitivity_boundaries(self) -> None:
        retriever = MemoryRetriever(
            [
                record("own", "Presentation meeting is Friday"),
                record("other", "Presentation meeting is Thursday", user_id="bob"),
                record("old", "Presentation meeting used to be Monday", status="superseded"),
                record("risk", "Presentation meeting caused a safety concern", sensitivity="restricted"),
            ],
            policy=RetrievalPolicy(minimum_relevance=0.01, top_k=10),
        )
        results = retriever.search(
            "When is my presentation meeting?", user_id="alice", as_of="2026-10-02T10:00:00Z"
        )
        self.assertEqual([item.record.memory_id for item in results], ["own"])

    def test_relevance_gate_can_return_zero_memories(self) -> None:
        retriever = MemoryRetriever(
            [record("m1", "I prefer walking after lunch")],
            policy=RetrievalPolicy(minimum_relevance=0.1),
        )
        self.assertEqual(
            retriever.search(
                "How do I configure a database?", user_id="alice", as_of="2026-10-02T10:00:00Z"
            ),
            [],
        )

    def test_semantic_relevance_precedes_recency_boost(self) -> None:
        retriever = MemoryRetriever(
            [
                record("relevant", "My active goal is to walk after lunch", updated_at="2026-01-01T10:00:00Z"),
                record("recent", "I watched a film yesterday", updated_at="2026-10-01T10:00:00Z"),
            ],
            policy=RetrievalPolicy(minimum_relevance=0.05, top_k=2),
        )
        results = retriever.search(
            "What is my walking goal?", user_id="alice", as_of="2026-10-02T10:00:00Z"
        )
        self.assertEqual([item.record.memory_id for item in results], ["relevant"])


class ContextBuilderTests(unittest.TestCase):
    def test_channels_are_separate_and_traceable(self) -> None:
        retrieved = MemoryRetriever(
            [record("goal-1", "My goal is to walk after lunch")],
            policy=RetrievalPolicy(minimum_relevance=0.01),
        ).search("walking goal", user_id="alice", as_of="2026-10-02T10:00:00Z")
        package = build_context(
            short_term_summary="The user is reviewing this week's assignment.",
            personal_memories=retrieved,
            professional_evidence=[KnowledgeEvidence("beck-11", "Review homework collaboratively.", 0.9, "p. 221")],
        )
        self.assertIn("[PERSONAL MEMORY", package.prompt_context)
        self.assertIn("[CBT KNOWLEDGE", package.prompt_context)
        self.assertEqual(package.memory_ids, ("goal-1",))
        self.assertEqual(package.knowledge_ids, ("beck-11",))

    def test_urgent_safety_bypasses_normal_context(self) -> None:
        package = build_context(short_term_summary="anything", urgent_safety=True)
        self.assertTrue(package.safety_bypass)
        self.assertNotIn("PERSONAL MEMORY", package.prompt_context)


class BenchmarkDocumentTests(unittest.TestCase):
    def test_locomo_uses_official_dialogue_ids(self) -> None:
        example = RetrievalExample(
            example_id="sample:qa:0",
            category="1",
            query="What happened?",
            answer="A walk",
            evidence_ids=("D1:1",),
            sessions=(
                {
                    "session_id": "sample:session_1",
                    "date": "1 Jan 2026",
                    "turns": ({"dia_id": "D1:1", "speaker": "A", "text": "I took a walk."},),
                },
            ),
        )
        documents = documents_for_example(example, "locomo")
        self.assertEqual(documents[0].document_id, "D1:1")


if __name__ == "__main__":
    unittest.main()
