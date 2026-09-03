# Safety and task routing

```mermaid
flowchart TD
    I["Current user message and recent context"] --> D["Deterministic risk checks"]
    D --> Q{"Urgent or imminent risk?"}

    Q -->|Yes| P["Safety-first response"]
    P --> H["Encourage immediate human or emergency support"]
    H --> O["Return response without normal RAG flow"]

    Q -->|No| T["Task classifier"]
    T --> K{"Professional CBT knowledge needed?"}
    T --> M{"Cross-session memory needed?"}

    K -->|Yes| KR["Build CBT knowledge query"]
    K -->|No| K0["No knowledge context"]
    M -->|Yes| MR["Build memory query"]
    M -->|No| M0["No memory context"]

    KR --> C["Context builder"]
    K0 --> C
    MR --> C
    M0 --> C
```

**Input:** current message, recent turns and deterministic safety signals.  
**Output:** a safety-first response or separate retrieval decisions for professional knowledge and long-term memory.

Safety detection is not delegated to similarity search. RAG may supply grounded safety guidance, but it does not decide whether a crisis exists.
