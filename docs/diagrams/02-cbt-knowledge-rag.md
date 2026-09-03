# CBT Knowledge RAG

```mermaid
flowchart TD
    subgraph OFF["Offline knowledge preparation"]
        S["WHO, VA, CCI and official samples"] --> P["PDF extraction and cleaning"]
        P --> CH["Page-aware chunks and metadata"]
        CH --> IDX["BM25 index and multilingual embeddings"]
    end

    subgraph ON["Online retrieval"]
        Q["Chinese or English task query"] --> QR["Multilingual query builder"]
        QR --> B["BM25 lexical retrieval"]
        QR --> E["Multilingual E5 dense retrieval"]
        B --> F["Reciprocal-rank fusion"]
        E --> F
        F --> RR["Cross-encoder reranking"]
        RR --> G{"Relevant enough?"}
        G -->|No| Z["Return zero chunks"]
        G -->|Yes| TOP["Return up to three cited chunks"]
    end

    IDX --> B
    IDX --> E
    CH --> TOP
    TOP --> C["Professional evidence context"]
    Z --> C
```

**Input:** a task-focused retrieval query.  
**Output:** zero to three relevant, cited CBT evidence chunks.

This module supports professional grounding, role boundaries and safety references. It remains fixed across the main memory-system comparison.
