# Memory extraction

Convert a session into structured memory candidates without writing them directly to long-term storage.

The extraction stage should:

- separate explicit facts from model inferences;
- attach supporting conversation spans;
- detect goals, assignments, outcomes and corrections;
- assign provisional type, confidence and sensitivity;
- avoid storing transient small talk or unsupported clinical labels.

All candidates pass through validation and lifecycle rules before persistence.

## Implemented contract

`extraction_contract.py` defines a provider-independent JSON output contract. A model output is rejected unless every candidate cites existing user turn IDs and contains an exact quote from those turns. Assistant suggestions cannot be used as user evidence. Inferred cognitive patterns remain unconfirmed candidates, and safety signals are automatically restricted.

This module validates extraction output; it does not claim that one particular language model extracts memories accurately. Extraction quality will be measured separately on public benchmarks.
