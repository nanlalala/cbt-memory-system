# Memory Schema v1

This folder contains the first executable contract for long-term user memory.

- `memory_record.schema.json`: language-neutral JSON Schema;
- `memory_models.py`: dependency-free Python model and semantic validation;
- `validate_memory_data.py`: validation command for examples and test cases.

The schema covers explicit facts, preferences, events, emotions, goals, CBT assignments and outcomes, candidate cognitive patterns, corrections and restricted safety signals. Every record includes provenance, timestamps, confidence, confirmation state, lifecycle status and sensitivity.

Two rules are deliberately strict:

1. a model-inferred cognitive pattern remains an unconfirmed candidate until the user confirms it;
2. a correction creates a provenance-preserving record that points to the memory it supersedes.

Run the validation suite from the repository root:

```bash
python -m unittest discover -s tests -v
python modules/long-term-memory/schema/validate_memory_data.py
```
