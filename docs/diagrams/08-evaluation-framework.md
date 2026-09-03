# Evaluation framework

```mermaid
flowchart TD
    K["Knowledge retrieval evaluation"] --> R["Integrated results"]
    M["Memory retrieval evaluation"] --> R
    D["End-to-end dialogue evaluation"] --> R
    S["Safety evaluation"] --> R

    K --> KM["Recall@K, MRR, precision, citation support"]
    M --> MM["Recall@K, latest-state accuracy, correction compliance"]
    D --> DM["Continuity, homework tracking, contextual correctness"]
    S --> SM["Crisis handling, role boundaries, sensitive-memory leakage"]

    C1["No cross-session memory"] --> D
    C2["Short-term memory only"] --> D
    C3["Short-term plus Long-term Memory RAG"] --> D

    F["Frozen response model, CBT RAG, safety rules and scenarios"] --> D
    R --> REP["Model-assisted development report and later human review"]
```

The main experiment changes only cross-session memory availability and management. Knowledge RAG, the response model, safety rules and scenarios remain fixed so that differences can be attributed to the memory framework.
