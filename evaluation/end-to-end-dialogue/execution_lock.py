"""Single-writer ownership for local evaluation directories.

Hosted Colab uses a local Linux filesystem. These advisory locks coordinate
cooperating processes and threads on that filesystem; they are not a distributed
lock for copied directories or network filesystems. Never unlink the lock file:
its stable inode is what makes every contender agree on the same lock.
"""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
from pathlib import Path
import fcntl
import hashlib


class EvaluationDirectoryBusy(RuntimeError):
    """Another execution owns the requested output directory."""


def execution_lock_identity() -> dict[str, str]:
    return {"policy": "local_single_writer_flock_nonblocking_v1",
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


@contextmanager
def exclusive_directory(directory: Path):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # Each acquisition opens its own descriptor: Linux flock then also excludes
    # threads and nested calls, rather than sharing one process-global handle.
    with (directory / ".execution.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise EvaluationDirectoryBusy(
                "Evaluation directory is already in use; no request was attempted"
            ) from None
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def exclusive_directories(*directories: Path):
    # Stable ordering prevents opposite reader/judge acquisitions from deadlocking.
    # Identical aliases are acquired once, including a symlink to the same path.
    paths = sorted({Path(path).resolve() for path in directories})
    with ExitStack() as stack:
        for path in paths:
            stack.enter_context(exclusive_directory(path))
        yield
