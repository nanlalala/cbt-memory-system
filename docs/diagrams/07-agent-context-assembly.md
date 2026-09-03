# Agent context assembly

```mermaid
flowchart TD
    S["Safety-policy output"] --> V["Context validation"]
    T["Short-term session state"] --> V
    K["CBT knowledge evidence or empty"] --> V
    M["Long-term memories or empty"] --> V

    V --> P["Provenance, lifecycle and sensitivity checks"]
    P --> D["Deduplication and context budgeting"]
    D --> N["Namespaced prompt sections"]

    N --> N1["Safety instructions"]
    N --> N2["Current-session context"]
    N --> N3["Personal memory context"]
    N --> N4["Professional CBT evidence"]

    N1 --> A["Response agent"]
    N2 --> A
    N3 --> A
    N4 --> A

    A --> O["Supportive response"]
    A --> TR["Internal trace of information used"]
```

The two RAG outputs remain separate in the prompt. Personal memories are not presented as professional evidence, and retrieved evidence is not allowed to override the current user message.
