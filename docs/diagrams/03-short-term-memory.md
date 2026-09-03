# Short-term memory

```mermaid
flowchart TD
    W["Recent conversation window"] --> U["Session-state updater"]
    U --> T["Current topic"]
    U --> E["Current emotion"]
    U --> R["Active thought record"]
    U --> G["Current goal and assignment"]
    U --> N["Unresolved items"]

    T --> S["Structured session state"]
    E --> S
    R --> S
    G --> S
    N --> S

    S --> C["Context builder"]
    S --> SUM["Rolling session summary"]
    SUM --> X["End-of-session memory extraction"]

    UC["User correction in current session"] --> U
```

**Input:** recent turns and current-session corrections.  
**Output:** a compact, structured representation of what is active now.

Short-term memory is session-scoped. Only selected information becomes a candidate for long-term storage.
