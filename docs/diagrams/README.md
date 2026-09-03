# Architecture diagram set

These diagrams present the CBT Memory System at two levels: one overall system view followed by detailed views of each major block. Diagram labels are in English for presentation and supervisor review.

## Diagram index

1. [Safety and task routing](01-safety-task-routing.md)
2. [CBT Knowledge RAG](02-cbt-knowledge-rag.md)
3. [Short-term memory](03-short-term-memory.md)
4. [Long-term Memory RAG](04-long-term-memory-rag.md)
5. [Memory schema](05-memory-schema.md)
6. [Memory lifecycle](06-memory-lifecycle.md)
7. [Agent context assembly](07-agent-context-assembly.md)
8. [Evaluation framework](08-evaluation-framework.md)

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

### Core idea

The system has two retrieval channels. CBT Knowledge RAG retrieves professional evidence and safety guidance, while Long-term Memory RAG retrieves relevant user-specific history. Deterministic safety routing runs before both. The response agent receives only qualified context, and the conversation can also create or update long-term memories through a separate write path.
