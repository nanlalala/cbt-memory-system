# Development and regression environment

Use CPython 3.12 from a repository checkout. The CPU regression suite does not
download models, call a paid API, or require CUDA.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-test.lock
.venv/bin/python -m compileall -q modules evaluation app tests
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python app/memory_demo.py --scripted
```

`requirements-test.lock` pins the direct and transitive runtime dependencies used
by this suite. It is a version lock, not a wheel/hash lock; different operating
systems may receive different wheels. CI installs the same versions on Python
3.12. The heavier knowledge-RAG model environment remains separate in
`modules/cbt-knowledge-rag/requirements.txt`; it is not required for regression
tests or the lifecycle demo.

This repository is still checkout-based, not a supported installable wheel.
Legacy modules retain their current directory names and bounded path bootstraps.
A future packaging migration should preserve notebook and Colab compatibility
and verify installation outside the checkout before claiming wheel support.

Do not commit local databases, benchmark inputs, model caches, API credentials,
or generated conversation files. API generation is a separate opt-in step;
dry-run plans and software tests are not dialogue-effectiveness results.
