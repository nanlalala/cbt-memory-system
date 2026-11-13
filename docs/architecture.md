# System architecture

The implemented memory path includes source capture, typed task and contextual
projections, support checks, versioned retrieval and the context-v4 request
renderer. The research prototype includes fictional-data CLI and Colab dialogue
demonstrations. A production clinical risk detector, referral service,
authenticated journal UI and production privacy controls remain unimplemented.
Target application components are identified separately from the implemented
memory pipeline.

The current [readout implementation and complete request](context-assembly-v4/IMPLEMENTATION_PATHS.md)
make the storage-to-generator boundary explicit. Supported processed memory
actually enters the dynamic model context; the external audit trace is a separate
artifact. Earlier quote-only adapters retain their own versions.

[Back to project overview](../README.md)

This page contains the overall architecture and all module-level diagrams in one continuous document.

## 1. Overall application design

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

The target application combines current-session state with separate professional-knowledge and personal-memory retrieval channels. Implemented ownership, permission and lifecycle checks gate memory readout. The clinical risk/referral components in this overview remain a target service boundary.

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

The current dialogue mode reads supported processed information as well as
source quotations. Literal evidence binding and a separate semantic-support
assessment qualify a processed item; neither establishes user endorsement or
independent truth. Proposals, task acceptance, reported participation and outcomes
retain separate axes. Working hypotheses remain explicitly unverified.

A source statement may be used as a report made at that time. Changing or
uncertain current information can prompt a relevant optional check, rather than
routine per-item confirmation. The versioned benchmark adapters preserve their
source-scope and ordinal-time restrictions.

```mermaid
flowchart TD
    R["Relevant retrieved history"] --> P["Ownership, lifecycle and source checks"]
    P --> D{"How may it be used?"}
    D -->|"Supported historical content"| H["Scoped report or processed memory"]
    D -->|"Uncertain current state"| Q["One optional conversational check"]
    D -->|"Not permitted or unsupported"| X["Exclude"]
    H --> A["Response with readable source trace"]
    Q --> A
    U["User's new statement"] --> W["New sourced observation"]
    W --> P
```

Evidence quality, age and conflicts guide this decision. A model's confidence
number cannot override source or access checks. Automatic capture can be disabled;
management evidence is separated from chat, and a non-answer is not confirmation.

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
    D -->|New| C["Create candidate version"]
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
    DB --> F["Strict active retrieval or separately gated history use"]
```

Updates preserve provenance. Superseded, expired, deleted or restricted memories
are excluded from ordinary context. Conversational history use is a separate,
source-checked policy and does not turn all candidates into retrievable facts.

## 8. Memory readout and actual context assembly

![Implemented context-v4 memory readout](context-assembly-v4/readout-flow.svg)

The implementation separates three objects:

| Object | Content and role |
| --- | --- |
| Stored source and memory records | Original statements, source scope, support decisions, task axes, versions and permissions |
| Per-turn dynamic memory | Relevant supported processing, permitted task state, changes and necessary evidence selected for the present question |
| External request trace | Included/omitted items, source/version dependencies, request size/hash and actual provider usage; outside model messages |

Read admission checks the subject, permissions, deletion/stopped use, source
cutoff, conflicts and versions. Ranking and complete-request packing retain
control-relevant dependencies. Personal history and professional CBT evidence
keep distinct fields.

The current product and CBT architecture retest share this message contract:

1. Fixed system policy with the full CBT counselor role and behavioral rules.
2. Admitted recent dialogue in its original user/assistant roles.
3. One application-inserted **user-role** sourced-memory reference block.
4. The current **user** question, once.

The dynamic block is rebuilt every turn. Its data label does not make retrieved
text safe or correct by itself, and external content gains no system-policy
authority. A real tool-result message requires its actual tool call.

The [full fictional request](context-assembly-v4/FICTIONAL_REQUEST.md) is generated
by the actual renderer. The [path map](context-assembly-v4/IMPLEMENTATION_PATHS.md)
also documents the older system-message claims adapter and the historical
raw-evidence evaluation path, so their evidence cannot be confused with this
processed-memory readout.

## 9. Evaluation framework

```mermaid
flowchart TD
    F["Frozen source roster, role policy, model and budget"] --> N["No RAG"]
    F --> R["Raw-history RAG"]
    F --> M["Managed-memory RAG"]
    N --> G["DeepSeek answers"]
    R --> G
    M --> G
    G --> Q["Independent model scoring"]
    Q --> L["General-memory factual accuracy"]
    Q --> C["CBT R1–R5 dimensions"]
    L --> P["Versioned results and coverage"]
    C --> P
    S["Separate lifecycle software replay"] --> P
```

Within each current three-arm experiment, the primary comparator is capable
raw-history RAG. The recent dialogue window and role policy are shared; it is not
a fourth condition. The comparison evaluates the full managed pipeline rather
than isolating representation alone.

CBT-specific factual context, CBT method, next-step suitability, empathy and
safety remain separate automatic endpoints. General-memory results and software
checks do not establish treatment efficacy or human experience.

The [evaluation index](../evaluation/README.md) retains the methods, versioned
results, scoring scope and missingness. MAB and LoCoMo retain separate experiment
protocols, with exact implementations and scoring records linked to each result.
