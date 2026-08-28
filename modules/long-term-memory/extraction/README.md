# Memory extraction

Convert a session into structured memory candidates without writing them directly to long-term storage.

The extraction stage should:

- separate explicit facts from model inferences;
- attach supporting conversation spans;
- detect goals, assignments, outcomes and corrections;
- assign provisional type, confidence and sensitivity;
- avoid storing transient small talk or unsupported clinical labels.

All candidates pass through validation and lifecycle rules before persistence.
