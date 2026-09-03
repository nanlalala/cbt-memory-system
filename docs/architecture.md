# System architecture

[Back to project overview](../README.md)

This page contains the overall architecture and all module-level diagrams in one continuous document.

## 1. Overall system

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

The system combines current-session state with two independent retrieval channels. Safety routing is executed before normal retrieval.

## 2. Safety and task routing

```mermaid
flowchart TD
    I["Current message and recent context"] --> D["Deterministic risk checks"]
    D --> Q{"Urgent or imminent risk?"}
    Q -->|Yes| P["Safety-first response"]
    P --> H["Immediate human or emergency support"]
    Q -->|No| T["Task classifier"]
    T --> K{"CBT knowledge needed?"}
    T --> M{"Long-term memory needed?"}
    K -->|Yes| KR["Build knowledge query"]
    K -->|No| K0["No knowledge context"]
    M -->|Yes| MR["Build memory query"]
    M -->|No| M0["No memory context"]
    KR --> C["Context builder"]
    K0 --> C
    MR --> C
    M0 --> C
```

Crisis detection is not delegated to vector retrieval. Either normal retrieval channel may return no context.

## 3. CBT Knowledge RAG

```mermaid
flowchart TD
    S["Professional CBT sources"] --> P["PDF extraction and cleaning"]
    P --> CH["Cited chunks and metadata"]
    CH --> IDX["BM25 and multilingual embeddings"]
    Q["Chinese or English task query"] --> QR["Multilingual query builder"]
    QR --> B["BM25 retrieval"]
    QR --> E["Multilingual E5 retrieval"]
    IDX --> B
    IDX --> E
    B --> F["Rank fusion"]
    E --> F
    F --> RR["Cross-encoder reranking"]
    RR --> G{"Relevant enough?"}
    G -->|No| Z["Zero chunks"]
    G -->|Yes| TOP["Up to three cited chunks"]
```

This channel provides professional evidence, role boundaries and safety references.

## 4. Short-term memory

```mermaid
flowchart TD
    W["Recent conversation window"] --> U["Session-state updater"]
    U --> T["Current topic and emotion"]
    U --> R["Active thought record"]
    U --> G["Current goal and assignment"]
    U --> N["Unresolved items"]
    T --> S["Structured session state"]
    R --> S
    G --> S
    N --> S
    S --> C["Context builder"]
    S --> SUM["Rolling session summary"]
    SUM --> X["Long-term memory candidates"]
    UC["User correction"] --> U
```

Short-term memory represents what is active now. Only selected, supported information becomes a long-term memory candidate.

## 5. Long-term Memory RAG

```mermaid
flowchart TD
    subgraph WRITE["Write path"]
        S["Session summary and supported spans"] --> X["Candidate extraction"]
        UC["User confirmation or correction"] --> X
        X --> SC["Typed schema and provenance"]
        SC --> V["Validation and conflict checks"]
        V --> DB[("Versioned user memory store")]
    end
    subgraph READ["Retrieval path"]
        T["Current task"] --> Q["Memory query"]
        Q --> F["User, lifecycle and sensitivity filters"]
        F --> R["Semantic and structured retrieval"]
        R --> RR["Temporal and semantic reranking"]
        RR --> G{"Relevant and valid?"}
        G -->|No| Z["Zero memories"]
        G -->|Yes| M["Selected memories with provenance"]
    end
    DB --> F
    DB --> R
```

The write path creates traceable, correctable memories. The read path retrieves only relevant, valid and permitted memories.

## 6. Memory schema

```mermaid
classDiagram
    class BaseMemory {
        +memory_id
        +user_id
        +content
        +source_session_id
        +timestamps
        +confidence
        +confirmation_state
        +lifecycle_status
        +sensitivity
    }
    class ExplicitFact
    class EpisodicEvent
    class EmotionRecord
    class Goal
    class CBTAssignment
    class AssignmentOutcome
    class CognitivePattern
    class UserCorrection
    BaseMemory <|-- ExplicitFact
    BaseMemory <|-- EpisodicEvent
    BaseMemory <|-- EmotionRecord
    BaseMemory <|-- Goal
    BaseMemory <|-- CBTAssignment
    BaseMemory <|-- AssignmentOutcome
    BaseMemory <|-- CognitivePattern
    BaseMemory <|-- UserCorrection
    CBTAssignment "1" --> "0..*" AssignmentOutcome : produces
    UserCorrection --> BaseMemory : corrects
```

Every memory retains provenance, time, confidence, confirmation, lifecycle and sensitivity metadata. User statements remain distinguishable from model inferences.

## 7. Memory lifecycle

```mermaid
flowchart TD
    I["New candidate, correction, time or deletion signal"] --> M["Match existing memories"]
    M --> D{"Lifecycle decision"}
    D -->|New| C["Create active version"]
    D -->|Duplicate| G["Merge evidence"]
    D -->|Changed state| S["Supersede old version"]
    D -->|Correction| U["Create corrected version"]
    D -->|Stale| E["Expire or reduce importance"]
    D -->|Deletion| H["Delete content and index entry"]
    C --> DB[("Versioned memory store")]
    G --> DB
    S --> DB
    U --> DB
    E --> DB
    H --> DB
    DB --> F["Retrieve only active, valid and permitted memories"]
```

Updates preserve provenance. Superseded, expired, deleted or restricted memories are excluded from normal retrieval.

## 8. Agent context assembly

```mermaid
flowchart TD
    S["Safety-policy output"] --> V["Context validation"]
    T["Short-term session state"] --> V
    K["CBT evidence or empty"] --> V
    M["Long-term memories or empty"] --> V
    V --> P["Provenance, lifecycle and sensitivity checks"]
    P --> D["Deduplication and context budget"]
    D --> N["Separate prompt sections"]
    N --> A["Response agent"]
    A --> O["Supportive response"]
    A --> TR["Internal trace of information used"]
```

Professional evidence and personal memory remain separate in the prompt and cannot silently override the current user message.

## 9. Evaluation framework

```mermaid
flowchart TD
    K["Knowledge retrieval"] --> R["Integrated results"]
    M["Memory retrieval"] --> R
    D["End-to-end dialogue"] --> R
    S["Safety evaluation"] --> R
    C1["No cross-session memory"] --> D
    C2["Short-term memory only"] --> D
    C3["Short-term plus Long-term Memory RAG"] --> D
    F["Frozen response model, CBT RAG and safety rules"] --> D
    R --> REP["Development report and later human review"]
```

The main experiment changes only cross-session memory. Knowledge RAG, the response model, safety rules and scenarios remain fixed.
