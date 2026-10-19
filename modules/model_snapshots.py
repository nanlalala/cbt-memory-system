"""Pinned, byte-verified local Hugging Face model snapshots.

Importing this module needs only the standard library and never accesses the
network.  The committed lock records full Hub commits and SHA-256 values of all
files used by the supported PyTorch/safetensors loaders.  Weight hashes were
verified against the official Hub's LFS metadata, not by downloading weights
during the source review; at execution time the actual local bytes are hashed.

``prepare_model_snapshot('dense', local_path=path)`` only reads an already
downloaded snapshot, so ``path`` may be on a read-only filesystem.  Without a
local path, only the locked files at the locked commit are downloaded.  The
caller must load the returned path with ``local_files_only=True`` and
``trust_remote_code=False``.  Neither a missing file nor a hash mismatch permits
falling back to ``main`` or another model.  No absolute path or timestamp enters
the stable identity used by caches and frozen run manifests.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


LOCK_PATH = Path(__file__).with_suffix(".json")
IDENTITY_VERSION = "hf-snapshot-sha256-v1"
RUNTIME_PACKAGES = (
    "sentence-transformers", "transformers", "torch", "huggingface-hub",
    "tokenizers", "safetensors",
)
NON_LOADING_DOCUMENTS = frozenset({
    "README.md", "LICENSE", "LICENSE.md", "LICENSE.txt", ".gitattributes",
})


class ModelSnapshotError(ValueError):
    """A model snapshot cannot satisfy the committed identity policy."""


@dataclass(frozen=True, slots=True)
class VerifiedModelSnapshot:
    path: Path
    identity: dict[str, Any]


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _validate_spec(spec: dict[str, Any]) -> None:
    if not isinstance(spec.get("repo_id"), str) or not spec["repo_id"]:
        raise ModelSnapshotError("Pinned snapshot requires a repository ID")
    if not re.fullmatch(r"[0-9a-f]{40}", str(spec.get("revision", ""))):
        raise ModelSnapshotError("Pinned snapshot revision must be a full immutable commit SHA")
    files = spec.get("files")
    if not isinstance(files, dict) or not files:
        raise ModelSnapshotError("Pinned snapshot requires a nonempty file manifest")
    for name, expected in files.items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ModelSnapshotError("Snapshot manifest contains an unsafe file path")
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected.get("sha256", ""))):
            raise ModelSnapshotError(f"Snapshot file {name} requires a SHA-256 digest")
        if type(expected.get("size")) is not int or expected["size"] < 1:
            raise ModelSnapshotError(f"Snapshot file {name} requires a positive byte size")


def pinned_model_spec(repo_or_role: str = "dense") -> dict[str, Any]:
    """Return an independent JSON-compatible identity spec without downloading.

    Only repository names explicitly present in the reviewed lock are accepted.
    Adding a model or changing a revision requires reviewing and committing a new
    lock rather than resolving a moving branch during an experiment.
    """
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("schema_version") != "1":
        raise ModelSnapshotError("Unsupported model snapshot lock schema")
    for role, raw in lock["models"].items():
        if repo_or_role in {role, raw["repo_id"]}:
            # Verification notes and dates document the review but are not model
            # identity; adding notes must not invalidate a valid model cache.
            spec = {
                "repo_id": raw["repo_id"], "revision": raw["revision"],
                "files": {
                    name: {"sha256": entry["sha256"], "size": entry["size"]}
                    for name, entry in raw["files"].items()
                },
            }
            _validate_spec(spec)
            return copy.deepcopy(spec)
    raise ModelSnapshotError(f"Model is not in the reviewed snapshot lock: {repo_or_role}")


def runtime_environment() -> dict[str, Any]:
    """Record actual installed distributions; missing optional packages are null."""
    versions: dict[str, str | None] = {}
    for name in RUNTIME_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "packages": versions}


def verify_model_snapshot(path: Path | str, spec: dict[str, Any]) -> VerifiedModelSnapshot:
    """Hash actual files and return stable, serializable model/runtime identity.

    This never writes to the supplied directory.  All untracked model files are
    rejected, including added tokenizer tokens, adapter configs and weight
    indices.  Only an explicit root-level documentation allowlist and the Hub's
    ``.cache/huggingface`` metadata are exempt.  A previously downloaded full
    repository containing extra exports must be copied to a clean directory with
    only the locked files, or downloaded with this helper's exact allowlist.
    """
    _validate_spec(spec)
    root = Path(path)
    if not root.is_dir():
        raise ModelSnapshotError(f"Model snapshot directory does not exist: {root}")
    for candidate in root.rglob("*"):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(root).as_posix()
        if relative.startswith(".cache/huggingface/"):
            continue
        if relative not in spec["files"] and relative not in NON_LOADING_DOCUMENTS:
            raise ModelSnapshotError(f"Unexpected model loading file: {relative}")
    actual_files: dict[str, dict[str, Any]] = {}
    for name, expected in sorted(spec["files"].items()):
        file_path = root / name
        if not file_path.is_file():
            raise ModelSnapshotError(f"Pinned model snapshot is missing {name}")
        if file_path.stat().st_size != expected["size"]:
            raise ModelSnapshotError(f"Pinned model snapshot size mismatch: {name}")
        digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        actual = {"sha256": digest.hexdigest(), "size": file_path.stat().st_size}
        if actual != expected:
            raise ModelSnapshotError(f"Pinned model snapshot SHA-256 mismatch: {name}")
        actual_files[name] = actual
    identity: dict[str, Any] = {
        "identity_version": IDENTITY_VERSION,
        "repo_id": spec["repo_id"],
        "resolved_revision": spec["revision"],
        "files": actual_files,
        "runtime": runtime_environment(),
        "loader_policy": {
            "local_files_only": True, "trust_remote_code": False,
            "use_safetensors": True,
        },
    }
    identity["identity_fingerprint"] = _canonical_hash(identity)
    return VerifiedModelSnapshot(root, identity)


def prepare_model_snapshot(
    repo_or_role: str = "dense", *, local_path: Path | str | None = None,
    local_files_only: bool = False, cache_dir: Path | str | None = None,
) -> VerifiedModelSnapshot:
    """Download a locked commit or validate an existing read-only snapshot.

    ``local_files_only`` applies to the Hub cache lookup when ``local_path`` is
    absent.  A supplied local path always suppresses all network/cache writes.
    Unknown names, missing local bytes and mismatches fail closed.
    """
    spec = pinned_model_spec(repo_or_role)
    if local_path is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("Install huggingface-hub to download pinned model snapshots") from exc
        local_path = snapshot_download(
            repo_id=spec["repo_id"], revision=spec["revision"],
            allow_patterns=sorted(spec["files"]),
            local_files_only=local_files_only,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    return verify_model_snapshot(local_path, spec)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="dense", help="Reviewed role or repository ID")
    parser.add_argument("--local-path", type=Path, help="Read-only predownloaded snapshot")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    snapshot = prepare_model_snapshot(
        args.model, local_path=args.local_path,
        local_files_only=args.local_files_only, cache_dir=args.cache_dir,
    )
    print(json.dumps({"path": str(snapshot.path), "identity": snapshot.identity}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
