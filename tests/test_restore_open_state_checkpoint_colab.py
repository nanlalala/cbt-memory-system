"""Tiny synthetic archive, metadata and path fixtures; no study data or model IO."""
from io import BytesIO
import importlib.util
import json
from pathlib import Path
import stat
import unittest
from unittest.mock import patch
import zipfile


PATH = Path(__file__).resolve().parents[1] / "evaluation/cbt-memory-effect/restore_open_state_checkpoint_colab.py"
spec = importlib.util.spec_from_file_location("restore_open_state_checkpoint_tested", PATH)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def journal_row(index):
    body = {"event": "fixture", "index": index}
    return (r.canonical({**body, "record_sha256": r.sha(r.canonical(body).encode())}) + "\n").encode()


def fixture_zip(*, name="prepared/protocol.json", data=b"{}", digest=None, special_mode=None,
                duplicate=False, extra=False):
    manifest = {"files": [{"path": name, "bytes": len(data), "sha256": digest or r.sha(data)}]}
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        if special_mode:
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = special_mode << 16
            archive.writestr(info, data)
        else:
            archive.writestr(name, data)
        archive.writestr("manifest.json", json.dumps(manifest))
        if duplicate:
            archive.writestr(name, data)
        if extra:
            archive.writestr("extra.json", b"{}")
    return stream.getvalue()


class RestoreMetadataFixtures(unittest.TestCase):
    def load_fixture(self, data, **overrides):
        args = {"expected_sha": r.sha(data), "expected_bytes": len(data),
                "manifest_name": "manifest.json", "expected_count": 1, **overrides}
        with patch.object(r, "read_regular", return_value=data):
            return r.read_archive("/content/tiny-fixture.zip", **args)

    def test_hosted_guard_precedes_archive_io(self):
        with patch.dict(r.os.environ, {}, clear=True), patch.object(r, "read_archive") as read:
            with self.assertRaisesRegex(RuntimeError, "genuine_hosted_colab"):
                r.restore("/content/base.zip", "/content/pilot.zip", "/content/receipt.json")
            read.assert_not_called()

    def test_tiny_manifest_bytes_are_verified(self):
        manifest, records, payloads, total = self.load_fixture(fixture_zip())
        self.assertEqual(len(manifest["files"]), 1)
        self.assertEqual(records["prepared/protocol.json"]["bytes"], 2)
        self.assertEqual(payloads["prepared/protocol.json"], b"{}")
        self.assertGreater(total, 2)

    def test_archive_pin_is_required(self):
        with self.assertRaisesRegex(ValueError, "pinned_archive"):
            self.load_fixture(fixture_zip(), expected_sha="0" * 64)

    def test_manifest_sha_is_required(self):
        with self.assertRaisesRegex(ValueError, "artifact_size_or_sha256"):
            self.load_fixture(fixture_zip(digest="0" * 64))

    def test_zip_symlink_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "regular_zip"):
            self.load_fixture(fixture_zip(special_mode=stat.S_IFLNK | 0o777))

    def test_duplicates_and_unlisted_entries_are_rejected(self):
        for kwargs in ({"duplicate": True}, {"extra": True}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "entry_count"):
                self.load_fixture(fixture_zip(**kwargs))

    def test_archive_and_destination_traversal_are_rejected(self):
        for value in ("../x.json", "/x.json", "a//x.json", "a/./x.json", "a\\x.json", "a/../x.json",
                      ".git/config.json", "model/model.safetensors", "secrets/x.json"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                r.member_path(value)
        for value in ("/tmp/x", "/content", "/content/../x.json", "/content/a//b", "/content/a/./b"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                r.content_path(value)

    def test_original_and_pilot_paths_are_fixed(self):
        self.assertEqual(str(r.destination_for("prepared_before/protocol.json")),
                         "/content/cbt-open-state-prepared-before-20261112-v2/protocol.json")
        self.assertEqual(str(r.destination_for("tokenization_only_audit.json", pilot=True)),
                         "/content/cbt-constrained-tokenization-audit-20261112-v1.json")
        self.assertEqual(str(r.destination_for("model_metadata/generation_ledger.jsonl", pilot=True)),
                         "/content/cbt-open-reader-20261112-v1/generation_ledger.jsonl")

    def test_receipt_requires_safe_direct_json_filename(self):
        self.assertEqual(str(r.receipt_path("/content/restore-receipt-v1.json")),
                         "/content/restore-receipt-v1.json")
        for value in ("/content/.git/config", "/content/subdir/receipt.json", "/content/model.safetensors",
                      "/content/.hidden.json", "/content/receipt.json/child.json"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                r.receipt_path(value)

    def test_unlisted_model_metadata_and_reports_are_rejected(self):
        for name, pilot in (("model_metadata/config.json", False),
                            ("model_metadata/model_load_receipt.json", True),
                            ("reports/unlisted.json", False), ("unknown/x.json", False)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                r.destination_for(name, pilot=pilot)

    def test_exact_checksum_validated_prefix_extension(self):
        original = journal_row(1)
        extended = original + journal_row(2)
        self.assertEqual(len(r.validate_extension(original, extended, base_count=1, final_count=2)), 2)
        for changed in (journal_row(3) + journal_row(2), extended.replace(b'"index":2', b'"index":3'),
                        extended[:-1]):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                r.validate_extension(original, changed, base_count=1, final_count=2)

    def test_duplicate_keys_and_nonfinite_metadata_are_rejected(self):
        for value in (b'{"x":1,"x":2}', b'{"x":NaN}'):
            with self.assertRaises(ValueError):
                r.parse_json(value)

    def test_existing_mismatch_errors_without_writes(self):
        plan = {"/content/fixture.json": {"data": b"{}"}}
        with patch.object(r, "read_regular", return_value=b"different"), patch.object(r, "write_new") as write:
            with self.assertRaisesRegex(FileExistsError, "mismatch_preserved"):
                r.preflight_existing(plan, "/content/receipt.json")
            write.assert_not_called()

    def test_second_archive_failure_prevents_all_writes(self):
        with patch.object(r, "require_hosted_colab"), patch.object(r, "read_archive", side_effect=[({}, {}, {}, 0), ValueError("pilot rejected")]), patch.object(r, "write_new") as write:
            with self.assertRaisesRegex(ValueError, "pilot rejected"):
                r.restore("/content/base.zip", "/content/pilot.zip", "/content/receipt.json")
            write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
