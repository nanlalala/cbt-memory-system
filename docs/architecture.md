# System architecture

The response context is assembled from five independently governed inputs:

```text
Current session summary
+ active assignment and goal state
+ selected long-term memories
+ frozen CBT knowledge-base RAG context
+ deterministic safety rules
→ response agent
```

## Memory layers

### Short-term memory

- recent conversation window;
- current topic and emotion;
- active thought record;
- unresolved items;
- currently valid assignment.

### Long-term memory

- explicit user facts and preferences;
- episodic events;
- goals, assignments and outcomes;
- candidate or confirmed recurring patterns;
- restricted safety-related information.

Every memory must retain provenance, timestamps, confidence, lifecycle status and user-confirmation state. User facts and model inferences must be stored separately.

## Memory manager responsibilities

- extract candidate memories;
- decide short-term versus long-term storage;
- merge duplicates;
- resolve conflicts without erasing provenance;
- expire or reduce obsolete memories;
- support view, confirmation, correction and deletion;
- retrieve by relevance, validity, task state and sensitivity.

## Experimental control

The CBT RAG corpus, retriever, safety rules and response model must be frozen across all memory conditions. Otherwise a response improvement could not be attributed to the memory framework.

## RAG usage policy

RAG content is a compact professional reference rather than the sole basis of the conversation. The response agent should normally receive no more than three highly relevant chunks. It retains its original conversational and empathic capabilities, uses retrieved material only when directly applicable, cites substantive professional claims, and ignores irrelevant examples. Safety routing remains independent of both retrieval and generation.
