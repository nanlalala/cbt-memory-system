"""Tiny invented path/conflict fixtures; no models or experiment execution."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'evaluation/cbt-memory-effect/restore_cpu_v6_closure_colab.py'
spec = importlib.util.spec_from_file_location('restore_cpu_v6_tiny_fixture', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RestoreSafetyTests(unittest.TestCase):
    def test_original_content_mapping_is_exact(self):
        self.assertEqual(module.member_target('content/invented/protocol.json'),
                         Path('/content/invented/protocol.json'))

    def test_traversal_absolute_windows_and_normalized_aliases_are_rejected(self):
        for name in ('../secret.json', '/content/a.json', 'content/a/../b.json',
                     'content//a.json', 'content/./a.json', 'content\\a.json',
                     'content/C:/a.json', 'content/a.json\x00suffix'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                module.member_target(name)

    def test_git_credentials_installations_and_weights_are_never_restored(self):
        for name in ('content/repo/.git/config', 'content/setup/site-packages/pkg.py',
                     'content/private/.env', 'content/model/tokenizer.json',
                     'content/model/model.gguf', 'content/model/fragment.partial',
                     'content/.ssh/credentials.json'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                module.member_target(name)

    def test_existing_identical_bytes_skip_while_different_bytes_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'tiny.json'; body = b'{"invented":true}\n'; path.write_bytes(body)
            with patch.object(module, 'CONTENT', root):
                self.assertEqual(module.destination_state(path, len(body), hashlib.sha256(body).hexdigest()),
                                 'identical_existing')
                self.assertEqual(module.destination_state(path, len(body), '0' * 64), 'conflict_different_bytes')
                self.assertEqual(module.destination_state(root / 'missing.json', 1, '0' * 64), 'missing')

    def test_symlink_and_nondirectory_parents_conflict_without_following(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / 'real').mkdir()
            (root / 'link').symlink_to(root / 'real', target_is_directory=True)
            (root / 'file').write_bytes(b'invented')
            with patch.object(module, 'CONTENT', root):
                self.assertEqual(module.destination_state(root / 'link/a.json', 1, '0' * 64),
                                 'conflict_symlink_parent')
                self.assertEqual(module.destination_state(root / 'file/a.json', 1, '0' * 64),
                                 'conflict_nondirectory_parent')

    def test_duplicate_manifest_json_keys_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            module.strict_json('{"files":[],"files":[]}')

    def test_real_restore_requires_hosted_before_any_destination_operation(self):
        with self.assertRaisesRegex(RuntimeError, 'genuine_hosted_colab'):
            module.restore('/tmp/invented.zip', {}, receipt='/content/invented.json')


if __name__ == '__main__':
    unittest.main()
