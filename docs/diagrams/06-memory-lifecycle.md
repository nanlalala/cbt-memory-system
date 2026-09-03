# Memory lifecycle

```mermaid
flowchart TD
    I["New candidate, correction, time signal or deletion request"] --> M["Match against existing memories"]
    M --> D{"Lifecycle decision"}

    D -->|New supported information| C["Create active version"]
    D -->|Duplicate| G["Merge evidence and preserve provenance"]
    D -->|Conflict or changed state| S["Supersede old version"]
    D -->|User correction| U["Create corrected version"]
    D -->|Stale but potentially useful| E["Expire or reduce importance"]
    D -->|Deletion request| H["Hard delete content and index entry"]

    C --> DB[("Versioned memory store")]
    G --> DB
    S --> DB
    U --> DB
    E --> DB
    H --> DB

    DB --> F["Retrieval filters"]
    F --> O["Only active, valid and permitted memories"]
```

The system preserves history for updates and conflicts, but deleted content must be removed. Superseded, expired or restricted memories cannot enter the response context unless an explicit policy allows them.
