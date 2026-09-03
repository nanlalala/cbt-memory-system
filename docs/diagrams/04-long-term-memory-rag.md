# Long-term Memory RAG

```mermaid
flowchart TD
    subgraph WRITE["Write path"]
        S["Session summary and supported spans"] --> X["Memory candidate extraction"]
        UC["User confirmation or correction"] --> X
        X --> SC["Typed schema and provenance"]
        SC --> V["Validation, sensitivity and conflict checks"]
        V --> DB[("Versioned user memory store and embeddings")]
    end

    subgraph READ["Retrieval path"]
        T["Current task and dialogue"] --> Q["Task-focused memory query"]
        Q --> F["User, lifecycle and sensitivity filters"]
        F --> R["Semantic and structured retrieval"]
        R --> RR["Temporal and semantic reranking"]
        RR --> G{"Relevant and valid?"}
        G -->|No| Z["Return zero memories"]
        G -->|Yes| M["Return a small memory set with provenance"]
    end

    DB --> F
    DB --> R
    M --> C["Personal memory context"]
    Z --> C
```

**Write path:** creates structured, traceable and correctable memories.  
**Read path:** retrieves only memories that are relevant, valid and permitted for the current task.

Memory ranking considers semantic relevance, recency, importance, confidence, user confirmation and lifecycle status.
