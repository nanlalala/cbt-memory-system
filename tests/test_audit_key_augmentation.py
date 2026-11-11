"""Small invented-fixture assessor checks, no model or public data."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("audit_key_augmentation", ROOT / "evaluation/cbt-memory-effect/audit_key_augmentation.py")
A = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(A)
SUPPORT = importlib.util.spec_from_file_location("audit_key_fixture_support", ROOT / "tests/test_prepare_key_augmentation.py")
F = importlib.util.module_from_spec(SUPPORT); SUPPORT.loader.exec_module(F)


class KeyAssessorTests(unittest.TestCase):
    def fixture(self, root):
        old = F.KeyAugmentationTests().fixture(root, extra_turns=7)
        cases = A.base.read_rows(old / "generation/cases.jsonl")
        target_dir = old / "assessor_only"; target_dir.mkdir()
        targets = []
        for case, position, index in zip(cases, ("early", "middle", "recent"), (0, 2, 9)):
            turn = case["history"][index]
            targets.append({"case_id": case["case_id"], "case_sha256": case["case_sha256"],
                "source_family_id": case["source_family_id"], "position": position,
                "target_message_id": turn["message_id"], "target_quote": turn["content"]})
        target_file = target_dir / "targets.jsonl"
        target_file.write_text("".join(json.dumps(row) + "\n" for row in targets))
        protocol = A.base.read_json(old / "protocol.json")
        protocol["files_sha256"]["assessor_only/targets.jsonl"] = A.base.file_hash(target_file)
        relative = "evaluation/cbt-memory-effect/run_cactus_effect.py"
        protocol["code_sha256"][relative] = A.base.file_hash(ROOT / relative)
        protocol["protocol_sha256"] = A.base.digest({k: v for k, v in protocol.items() if k != "protocol_sha256"})
        (old / "protocol.json").write_text(json.dumps(protocol))
        anchor = old / "cases/probe-0/prepared_contexts.json"
        prepared = A.base.read_json(anchor); prepared["protocol_sha256"] = protocol["protocol_sha256"]
        anchor.write_text(json.dumps(prepared))
        key_dir = root / "keys"
        A.keys.prepare(old, key_dir, product_repo=ROOT, fixture_only=True, encoder_factory=F.FakeEncoder)
        proposal_dir = root / "proposal"; proposal_dir.mkdir()
        proposal = {"version": A.sender.VERSION, "fixture_only": True,
            "old_run": str(old), "product_repo": str(ROOT), "code_sha256": {},
            "selected_cases": [case["case_id"] for case in cases]}
        proposal["proposal_sha256"] = A.sender.sha(proposal)
        (proposal_dir / "proposal.json").write_text(json.dumps(proposal))
        return old, key_dir, proposal_dir

    def test_assessor_reconstructs_nine_requests_without_embedding_or_private_quotes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); old, key_dir, proposal = self.fixture(root)
            before = {str(path): A.base.file_hash(path) for parent in (old, key_dir, proposal) for path in parent.rglob("*") if path.is_file()}
            with patch.object(A.base.TrackedEncoder, "encode", side_effect=AssertionError("assessor must not embed")):
                compact = A.assess(key_dir, proposal, root / "audit", fixture_only=True)
            self.assertEqual(compact["planned_rows"], 9)
            self.assertEqual(compact["actual_provider_calls"], 0)
            self.assertEqual(compact["new_embedding_calls"], 0)
            self.assertIsNone(compact["answer_accuracy"])
            self.assertTrue(all(row["accuracy"] is None for row in compact["cases"]))
            combined = "".join(path.read_text() for path in (root / "audit").iterdir())
            self.assertNotIn("I have not agreed to thought records.", combined)
            self.assertNotIn("I prefer discussing the nervousness.", combined)
            self.assertTrue((root / "audit/per_question.csv").is_file())
            self.assertEqual(before, {str(path): A.base.file_hash(path) for parent in (old, key_dir, proposal) for path in parent.rglob("*") if path.is_file()})

    def test_modified_wire_rejected_before_opening_targets(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); old, key_dir, proposal = self.fixture(root)
            traces = A.base.read_rows(key_dir / "traces.jsonl")
            traces[0]["contexts"]["raw_original"]["messages"][-1]["content"] = "tampered input"
            (key_dir / "traces.jsonl").write_text("".join(json.dumps(row) + "\n" for row in traces))
            real_read = Path.read_bytes
            def guard(path):
                if "assessor_only" in path.parts:
                    raise AssertionError("gold opened before wire check")
                return real_read(path)
            with patch.object(Path, "read_bytes", guard):
                with self.assertRaisesRegex(ValueError, "reconstruction"):
                    A.assess(key_dir, proposal, root / "audit", fixture_only=True)

    def test_unexpected_protocol_or_modified_target_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); old, key_dir, proposal = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "unexpected frozen key protocol"):
                A.assess(key_dir, proposal, root / "audit", fixture_only=True, expected_key_protocol="incorrect")
            (old / "assessor_only/targets.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "target checksum"):
                A.assess(key_dir, proposal, root / "audit", fixture_only=True)


if __name__ == "__main__":
    unittest.main()
