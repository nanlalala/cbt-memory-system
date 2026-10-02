![CBT Memory System — a hierarchical memory framework for repeated CBT reflections](assets/cbt-memory-system-header.jpg)

Research prototype for **A Hierarchical Memory Framework for Psychotherapy Reflection and Care Tracking**.

## Project overview

This project investigates how a CBT reflective-journaling agent can form, store, retrieve, update, correct and forget memories across repeated sessions. It is designed as a research and educational prototype, not as a diagnostic tool or a replacement for therapists.

The central research question is:

> How can an AI agent use short-term and long-term memory to maintain continuity, track CBT goals and assignments, respect user corrections, and remain safe across multiple sessions?

## Overall architecture

```mermaid
flowchart TD
    U["User reflection or follow-up"] --> APP["Reflective journaling interface"]
    APP --> R["Safety and task router"]

    R -->|Urgent risk| S["Safety response and referral"]
    S --> APP

    R -->|Normal flow| STM["Short-term state"]
    R --> KR["CBT Knowledge RAG"]
    R --> MR["Long-term Memory RAG"]

    STM --> C["Context builder"]
    KR --> C
    MR --> C
    C --> A["Response agent"]
    A --> APP

    APP --> X["Memory candidate extraction"]
    X --> L["Memory lifecycle manager"]
    L --> DB[("Long-term memory store")]
    DB --> MR
```

The system uses two separate but coordinated retrieval channels:

- **CBT Knowledge RAG** retrieves professional CBT evidence, methods and safety guidance.
- **Long-term Memory RAG** retrieves relevant user-specific history, including emotions, goals, events, assignments, outcomes, cognitive patterns and corrections.

Deterministic safety routing runs before both retrieval channels. Qualified knowledge and memory results are combined with the short-term session state by the context builder. Either retriever may return no context when its results are not sufficiently relevant.

The conversation also has a separate memory write path: supported information is extracted as a candidate, validated, versioned and managed through update, correction and forgetting rules before it becomes retrievable.

## Architecture components

| Component | Responsibility |
|---|---|
| Safety and task router | Detects urgent risk and decides whether knowledge or memory retrieval is required |
| Short-term memory | Maintains the recent conversation, current emotion, active goal, thought record and assignment |
| CBT Knowledge RAG | Retrieves grounded professional evidence and safety references |
| Long-term Memory RAG | Stores and retrieves user-specific, time-sensitive and correctable memories |
| Memory lifecycle manager | Handles consolidation, conflict, correction, superseding, expiry and deletion |
| Context builder | Validates, separates and budgets short-term state, personal memory and professional evidence |
| Response agent | Generates a supportive response while respecting safety and role boundaries |

See the [complete architecture document](docs/architecture.md) for the overall design and all module-level diagrams on one page.

## Repository structure

```text
cbt-memory-system/
├── docs/
│   └── architecture.md                 # Overall and module-level diagrams
├── modules/
│   ├── cbt-knowledge-rag/              # Professional knowledge and safety references
│   ├── long-term-memory/
│   │   ├── schema/                     # Typed memory records and metadata
│   │   ├── extraction/                 # Session-to-memory candidates
│   │   ├── storage/                    # User-scoped records and indexes
│   │   ├── retrieval/                  # Query, search, reranking and gating
│   │   └── lifecycle/                  # Update, correction, conflict and forgetting
│   └── agent-integration/              # Routing and context assembly
├── evaluation/
│   ├── knowledge-retrieval/
│   ├── memory-retrieval/
│   ├── end-to-end-dialogue/
│   └── safety/
└── app/                                # Journal, assignments and memory controls
```

## Experimental design

The main comparison will include:

1. no cross-session memory;
2. short-term memory only;
3. short-term plus manageable long-term memory.

The response model, CBT Knowledge RAG, safety rules and evaluation scenarios will remain fixed across these conditions so that differences can be attributed to the memory framework.

## Current status

The CBT Knowledge RAG module is complete. Memory Schema v1 is now implemented with provenance-aware records, executable validation, structured examples and a ten-case synthetic multi-session extraction set.

Current retrieval results:

| Metric | Result |
|---|---:|
| Recall@5 | **0.80** |
| Recall@10 | **0.84** |
| MRR@10 | **0.583** |
| Safety Recall@5 | **1.00** |

An early eight-scenario pilot found nearly identical dialogue scores for no RAG (7.94/8) and Reference RAG v2 (7.88/8). In the later 16-scenario full-corpus comparison, hybrid-and-reranked RAG scored 11.69/12 versus 11.38/12 for no RAG, with six RAG wins, two no-RAG wins and eight ties. The observed difference remains modest, so CBT Knowledge RAG is treated as a reference layer for grounding, citations, professional boundaries and safety support rather than proof of general conversational improvement. These are model-assisted development signals, not human clinical ratings.

The full no-RAG versus RAG comparison will be repeated after Long-term Memory RAG and Agent integration are complete. The end-to-end evaluation will assess dialogue quality, cross-session continuity, assignment tracking, professional boundaries, safety, latency and token cost.

## Four-week implementation plan

| Week | Main goal | Key tasks | Deliverables |
|---|---|---|---|
| **Week 1 — complete** | Memory schema and data preparation | Define short- and long-term memory types; design schemas for emotions, events, goals, assignments, cognitive patterns and corrections; prepare multi-session test cases | Memory Schema v1, four structured examples and ten multi-session extraction cases |
| **Week 2** | Memory writing and lifecycle management | Implement candidate extraction, validation, deduplication, conflict detection, updating, user correction, expiry and forgetting | Memory Manager v1, correction workflow and unit-test results |
| **Week 3** | Long-term Memory RAG and Agent integration | Build user-scoped memory indexes; implement semantic retrieval, temporal reranking and relevance gating; integrate both RAG channels with short-term context | Long-term Memory RAG v1, dual-retrieval pipeline and an end-to-end prototype |
| **Week 4** | Comparative evaluation and reporting | Compare the three memory conditions; evaluate continuity, latest-state accuracy, assignment tracking, correction, safety, latency and token cost | Evaluation results, error analysis, system demo and progress report |

## Documentation

- [System architecture](docs/architecture.md)
- [CBT Knowledge RAG report](modules/cbt-knowledge-rag/CBT_RAG_v1_Report.md)
- [Full-corpus dialogue comparison](modules/cbt-knowledge-rag/CBT_RAG_v4_Full_Comparison.ipynb)
- [Limited-corpus recovery experiment](modules/cbt-knowledge-rag/CBT_RAG_v3_Expanded_Evaluation.md)
