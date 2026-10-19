# Development and regression environment

The supported CPU regression target is **CPython 3.12, Linux x86_64, Ubuntu
24.04 (glibc 2.39)**, run from a repository checkout. The suite does not download
models, call a paid API, or require CUDA.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip --isolated install --require-hashes --only-binary=:all: --index-url https://pypi.org/simple -r requirements-test.lock
.venv/bin/python -m pip check
.venv/bin/python -m compileall -q modules evaluation app tests
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python app/memory_demo.py --scripted
```

`requirements-test.lock` contains the complete direct/transitive dependency
closure, exact versions and SHA-256 hashes for one compatible wheel per package.
The 12 dependency versions are unchanged from the previous version-only lock.
Installation rejects an unexpected artifact or an unpinned dependency; source
builds are disabled. This is a platform-specific wheel lock, **not** a claim of
macOS, Windows, ARM or other Python-version reproducibility. Create and verify a
separate target lock before supporting those environments.

CI fixes the runner label to `ubuntu-24.04`, Python minor version to 3.12 and
architecture to x64. Checkout and setup-python use full immutable commit SHAs,
with only `contents: read` permission and no persisted checkout credential. GitHub
can still update its hosted runner image and the 3.12 patch release: this is not
a byte-identical operating-system/container image or a lock of Python/pip itself.

The heavier model environment remains separate in
`modules/cbt-knowledge-rag/requirements.txt`; it is not required for these tests
or the lifecycle demo.

## Refreshing the hash lock

`requirements-test.in` is the reviewed list of exact direct and transitive pins.
On the supported Linux x86_64 / CPython 3.12 target, regenerate with:

```bash
python3.12 tools/lock_test_environment.py
python3.12 tools/lock_test_environment.py --check
```

The generator downloads wheels only from the official PyPI index, fetches the
official JSON release metadata for every pinned version, and compares the actual
downloaded file's SHA-256 with PyPI's published digest. It rejects yanked files,
non-wheel archives and Linux wheels newer than Ubuntu 24.04's glibc baseline.
It writes the lock plus [per-wheel provenance](test-environment-sources.json),
including every release-metadata URL, exact wheel filename, download URL and hash.
It does not choose newer dependency versions. Test the result in a **new** venv
using the hash-enforced installation and `pip check` above before committing.

Official sources checked on 2026-10-19:

- [PyPI JSON API](https://docs.pypi.org/api/json/) and the 12 release URLs recorded
  in `test-environment-sources.json`.
- [pip hash-checking mode](https://pip.pypa.io/en/stable/topics/secure-installs/).
- [Official checkout v4 ref](https://api.github.com/repos/actions/checkout/git/ref/tags/v4),
  resolving to [11d5960a326750d5838078e36cf38b85af677262](https://github.com/actions/checkout/tree/11d5960a326750d5838078e36cf38b85af677262).
- [Official setup-python v5 ref](https://api.github.com/repos/actions/setup-python/git/ref/tags/v5),
  resolving to [a26af69be951a213d495a4c3e4e4022e16d87065](https://github.com/actions/setup-python/tree/a26af69be951a213d495a4c3e4e4022e16d87065).

The action refs were verified from official GitHub API responses and `git
ls-remote` against the corresponding `actions/*` repositories. Future action
updates should repeat that check and review the new immutable commit; the
workflow must not return to mutable action tags.

This repository is still checkout-based, not a supported installable wheel.
Legacy modules retain their current directory names and bounded path bootstraps.
A future packaging migration should preserve notebook and Colab compatibility
and verify installation outside the checkout before claiming wheel support.

Do not commit local databases, full benchmark inputs, model caches, API credentials,
or arbitrary generated conversations. The named `pilot_20261019_audit.zip` is an
explicit noncommercial, attributed LoCoMo benchmark-audit release; its manifest,
license, file hashes and offline verifier must travel with it. This exception never
applies to real-user or patient prompts/responses. API generation is a separate
opt-in step; dry-run plans and software tests are not effectiveness results.
