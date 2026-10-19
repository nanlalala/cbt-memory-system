from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
for relative in (
    "modules", "modules/long-term-memory/schema", "modules/long-term-memory/retrieval",
    "evaluation/memory-retrieval",
):
    sys.path.insert(0, str(ROOT / relative))

from model_snapshots import (  # noqa: E402
    ModelSnapshotError, VerifiedModelSnapshot, pinned_model_spec,
    prepare_model_snapshot, verify_model_snapshot,
)
from memory_retriever import MultilingualE5Encoder  # noqa: E402
from benchmark_adapters import RetrievalExample  # noqa: E402
from run_memory_retrieval_benchmark import run_benchmark  # noqa: E402


def fake_snapshot(root: Path) -> dict:
    """Tiny synthetic bytes exercise verification without downloading a model."""
    files = {
        "config.json": b'{"model_type":"synthetic"}',
        "model.safetensors": b"synthetic test weights, not a real model",
        "tokenizer.json": b'{"synthetic":true}',
    }
    root.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (root / name).write_bytes(content)
    return {
        "repo_id": "synthetic/test-only", "revision": "a" * 40,
        "files": {
            name: {"sha256": hashlib.sha256(content).hexdigest(), "size": len(content)}
            for name, content in files.items()
        },
    }


class ModelSnapshotTests(unittest.TestCase):
    def test_reviewed_specs_pin_full_commits_and_all_loading_files(self) -> None:
        revisions = {
            "dense": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
            "reranker": "1427fd652930e4ba29e8149678df786c240d8825",
        }
        for role, revision in revisions.items():
            spec = pinned_model_spec(role)
            self.assertEqual(spec["revision"], revision)
            self.assertEqual(pinned_model_spec(spec["repo_id"]), spec)
            self.assertTrue({
                "config.json", "model.safetensors", "tokenizer_config.json",
                "tokenizer.json", "special_tokens_map.json", "sentencepiece.bpe.model",
            }.issubset(spec["files"]))
            for record in spec["files"].values():
                self.assertRegex(record["sha256"], r"^[0-9a-f]{64}$")
                self.assertGreater(record["size"], 0)
        self.assertIn("1_Pooling/config.json", pinned_model_spec("dense")["files"])
        with self.assertRaisesRegex(ModelSnapshotError, "reviewed snapshot lock"):
            pinned_model_spec("unreviewed/model")

    def test_specs_are_independent_and_cannot_mutate_the_lock(self) -> None:
        spec = pinned_model_spec("dense")
        spec["revision"] = "main"
        spec["files"]["config.json"]["sha256"] = "0" * 64
        self.assertNotEqual(pinned_model_spec("dense")["revision"], "main")
        self.assertNotEqual(pinned_model_spec("dense")["files"]["config.json"]["sha256"], "0" * 64)

    def test_same_length_file_tampering_and_missing_bytes_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = fake_snapshot(root)
            original = (root / "config.json").read_bytes()
            (root / "config.json").write_bytes(b"x" * len(original))
            with self.assertRaisesRegex(ModelSnapshotError, "SHA-256 mismatch: config.json"):
                verify_model_snapshot(root, spec)
            (root / "config.json").write_bytes(original)
            (root / "tokenizer.json").unlink()
            with self.assertRaisesRegex(ModelSnapshotError, "missing tokenizer.json"):
                verify_model_snapshot(root, spec)

    def test_moving_revision_and_unsafe_manifest_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            spec = fake_snapshot(Path(temporary))
            spec["revision"] = "main"
            with self.assertRaisesRegex(ModelSnapshotError, "immutable commit"):
                verify_model_snapshot(temporary, spec)
            spec["revision"] = "a" * 40
            spec["files"]["../outside"] = spec["files"]["config.json"]
            with self.assertRaisesRegex(ModelSnapshotError, "unsafe file path"):
                verify_model_snapshot(temporary, spec)

    def test_all_unlocked_loader_files_are_rejected_including_added_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = fake_snapshot(root)
            for name in (
                "adapter_config.json", "custom_model.py", "added_tokens.json",
                "tokenizer.model", "model.safetensors.index.json", "pytorch_model.bin",
            ):
                (root / name).write_text("unreviewed", encoding="utf-8")
                with self.subTest(file=name), self.assertRaisesRegex(
                    ModelSnapshotError, "Unexpected model loading file",
                ):
                    verify_model_snapshot(root, spec)
                (root / name).unlink()

    def test_only_explicit_documents_and_hub_metadata_are_exempt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = fake_snapshot(root)
            before = verify_model_snapshot(root, spec).identity
            (root / "README.md").write_text("A model description", encoding="utf-8")
            hub_metadata = root / ".cache/huggingface/download"
            hub_metadata.mkdir(parents=True)
            (hub_metadata / "config.json.metadata").write_text("Hub metadata", encoding="utf-8")
            self.assertEqual(before, verify_model_snapshot(root, spec).identity)
            # A documentation filename elsewhere is not a broad subtree exemption.
            (root / "unreviewed").mkdir()
            (root / "unreviewed/README.md").write_text("Unexpected subtree", encoding="utf-8")
            with self.assertRaisesRegex(ModelSnapshotError, "Unexpected model loading file"):
                verify_model_snapshot(root, spec)

    def test_existing_read_only_snapshot_needs_no_network_or_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = fake_snapshot(root)
            before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir()}
            for path in root.iterdir():
                path.chmod(0o444)
            fake_hub = types.SimpleNamespace(snapshot_download=Mock(side_effect=AssertionError("network called")))
            with patch("model_snapshots.pinned_model_spec", return_value=spec), patch.dict(
                sys.modules, {"huggingface_hub": fake_hub},
            ):
                result = prepare_model_snapshot("dense", local_path=root)
            self.assertEqual(result.path, root)
            fake_hub.snapshot_download.assert_not_called()
            self.assertEqual(before, {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir()})
            for path in root.iterdir():
                path.chmod(0o644)

    def test_download_requests_exact_immutable_snapshot_not_default_branch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            spec = fake_snapshot(root)
            downloader = Mock(return_value=str(root))
            with patch("model_snapshots.pinned_model_spec", return_value=spec), patch.dict(
                sys.modules, {"huggingface_hub": types.SimpleNamespace(snapshot_download=downloader)},
            ):
                result = prepare_model_snapshot("dense", local_files_only=True)
            downloader.assert_called_once_with(
                repo_id=spec["repo_id"], revision=spec["revision"],
                allow_patterns=sorted(spec["files"]), local_files_only=True, cache_dir=None,
            )
            self.assertEqual(result.identity["resolved_revision"], spec["revision"])
            self.assertEqual(result.identity["files"], spec["files"])

    def test_identity_is_path_independent_but_runtime_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            left, right = Path(temporary) / "a", Path(temporary) / "b"
            spec = fake_snapshot(left)
            fake_snapshot(right)
            runtime = {"python": "3.12.0", "packages": {"torch": "2.0"}}
            with patch("model_snapshots.runtime_environment", return_value=runtime):
                first = verify_model_snapshot(left, spec).identity
                other_path = verify_model_snapshot(right, spec).identity
            self.assertEqual(first, other_path)
            self.assertNotIn(str(left), json.dumps(first))
            changed_runtime = {"python": "3.12.0", "packages": {"torch": "2.1"}}
            with patch("model_snapshots.runtime_environment", return_value=changed_runtime):
                second = verify_model_snapshot(left, spec).identity
            self.assertNotEqual(first["identity_fingerprint"], second["identity_fingerprint"])

    def test_e5_encoder_loads_only_verified_local_safetensors(self) -> None:
        identity = {"resolved_revision": "a" * 40, "identity_fingerprint": "b" * 64}
        local_path = Path("/synthetic/read-only-snapshot")
        factory = Mock(return_value=types.SimpleNamespace(encode=Mock(return_value=[[1.0]])))
        fake_st = types.SimpleNamespace(SentenceTransformer=factory)
        with patch.dict(sys.modules, {"sentence_transformers": fake_st}), patch(
            "model_snapshots.prepare_model_snapshot",
            return_value=VerifiedModelSnapshot(local_path, identity),
        ) as prepare:
            encoder = MultilingualE5Encoder(device="cpu", snapshot_path=local_path)
        prepare.assert_called_once_with(
            "intfloat/multilingual-e5-small", local_path=local_path, local_files_only=False,
        )
        factory.assert_called_once_with(
            str(local_path), device="cpu", local_files_only=True,
            trust_remote_code=False, model_kwargs={"use_safetensors": True},
        )
        self.assertEqual(encoder.model_identity, identity)

    def test_retrieval_benchmark_records_verified_model_identity(self) -> None:
        example = RetrievalExample(
            example_id="synthetic", category="single", query="blue notebook", answer="blue",
            evidence_ids=("d1",), sessions=({"session_id": "s1", "turns": [
                {"dia_id": "d1", "speaker": "Alice", "text": "I own a blue notebook."},
            ]},),
        )
        identity = {"resolved_revision": "a" * 40, "files": {"synthetic": "test-only"}}
        encoder = types.SimpleNamespace(
            model_identity=identity,
            encode=lambda texts, **kwargs: [[1.0, 0.0] for _ in texts],
        )
        manifest: dict = {}
        with patch("run_memory_retrieval_benchmark.MultilingualE5Encoder", return_value=encoder):
            metrics, _ = run_benchmark(
                [example], dataset="locomo", backend="dense", run_manifest=manifest,
            )
        self.assertEqual(metrics["examples"], 1)
        self.assertEqual(manifest["model_identity"], identity)
        self.assertEqual(manifest["model_spec"], pinned_model_spec("dense"))
        self.assertTrue(manifest["embedding_preprocessing"]["normalize_embeddings"])


if __name__ == "__main__":
    unittest.main()
