"""Assemble short-term, personal-memory and professional-evidence context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from memory_retriever import RetrievedMemory


@dataclass(frozen=True, slots=True)
class KnowledgeEvidence:
    source_id: str
    text: str
    relevance: float
    citation: str = ""


@dataclass(frozen=True, slots=True)
class ContextPackage:
    prompt_context: str
    memory_ids: tuple[str, ...]
    knowledge_ids: tuple[str, ...]
    safety_bypass: bool = False


def build_context(
    *,
    short_term_summary: str,
    personal_memories: Sequence[RetrievedMemory] = (),
    professional_evidence: Sequence[KnowledgeEvidence] = (),
    urgent_safety: bool = False,
    max_personal: int = 3,
    max_professional: int = 3,
    max_chars: int = 6000,
) -> ContextPackage:
    """Keep retrieval channels explicit and never replace urgent safety handling."""

    if urgent_safety:
        return ContextPackage(
            prompt_context="URGENT_SAFETY_ROUTE: Follow the deterministic safety protocol before normal dialogue.",
            memory_ids=(),
            knowledge_ids=(),
            safety_bypass=True,
        )

    selected_memories = list(personal_memories[:max_personal])
    selected_evidence = list(professional_evidence[:max_professional])
    sections = [
        "[CURRENT SESSION]\n" + (short_term_summary.strip() or "No session summary provided."),
    ]
    if selected_memories:
        lines = []
        for item in selected_memories:
            record = item.record
            lines.append(
                f"- [{record.memory_id}] {record.content} "
                f"(type={record.memory_type}; observed={record.observed_at}; "
                f"confirmation={record.confirmation_state}; relevance={item.components.get('relevance', 0):.3f})"
            )
        sections.append("[PERSONAL MEMORY — context, not professional evidence]\n" + "\n".join(lines))
    if selected_evidence:
        lines = [
            f"- [{item.source_id}] {item.text}"
            + (f" ({item.citation})" if item.citation else "")
            for item in selected_evidence
        ]
        sections.append("[CBT KNOWLEDGE — professional reference, use only when relevant]\n" + "\n".join(lines))
    sections.append(
        "[USAGE RULES]\n"
        "Use retrieved items only when relevant to the current turn. "
        "Do not treat inferred memories as confirmed facts. "
        "Prefer the latest user correction. Do not diagnose."
    )
    prompt = "\n\n".join(sections)
    if len(prompt) > max_chars:
        prompt = prompt[: max_chars - 20].rstrip() + "\n[CONTEXT TRUNCATED]"
    return ContextPackage(
        prompt_context=prompt,
        memory_ids=tuple(item.record.memory_id for item in selected_memories),
        knowledge_ids=tuple(item.source_id for item in selected_evidence),
    )
