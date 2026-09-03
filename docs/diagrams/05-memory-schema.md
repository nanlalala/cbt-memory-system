# Memory schema

```mermaid
classDiagram
    class BaseMemory {
        +memory_id
        +user_id
        +content
        +source_session_id
        +supporting_span
        +created_at
        +observed_at
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
    CognitivePattern --> EpisodicEvent : supported by
```

Every memory carries provenance, time, confidence, confirmation, lifecycle and sensitivity metadata. Explicit user statements remain distinguishable from model-generated inferences.
