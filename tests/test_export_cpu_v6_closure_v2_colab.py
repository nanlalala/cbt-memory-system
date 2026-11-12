"""Invented path/catalog fixtures only; no real checkpoints or models."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'evaluation/cbt-memory-effect/export_cpu_v6_closure_v2_colab.py'
spec = importlib.util.spec_from_file_location('cpu_closure_tiny_fixture', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CpuClosureWhitelistTests(unittest.TestCase):
    def test_original_content_path_mapping_is_preserved(self):
        self.assertEqual(module.safe_member('/content/invented-gate/response-01.json'),
                         'content/invented-gate/response-01.json')

    def test_paths_cannot_escape_owned_cloud_content(self):
        for path in ('/tmp/invented.json', '/root/.cache/invented.json', 'relative.json',
                     '/content/../elsewhere/invented.json', '/content'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                module.owned(path)

    def test_weights_and_native_installations_are_never_archive_members(self):
        paths = ['/content/model/weights.gguf', '/content/model/weights.gguf.partial',
                 '/content/model/probe.bin', '/content/model/model.safetensors',
                 '/content/setup/site-packages/native.so', '/content/model/tokenizer.json',
                 '/content/.ssh/secret.json', '/content/setup/credentials.json']
        for path in paths:
            with self.subTest(path=path):
                self.assertIsNotNone(module.excluded_kind(path))
                with self.assertRaises(ValueError):
                    module.safe_member(path)

    def test_only_exact_large_cpu_wheel_gets_required_dependency_exception(self):
        self.assertEqual(module.excluded_kind(module.WHEEL_PATH),
                         'required_official_runtime_wheel_omitted_reacquire_before_restore')
        self.assertIsNone(module.excluded_kind('/content/setup/diskcache-invented.whl'))
        with self.assertRaises(ValueError):
            module.safe_member(module.WHEEL_PATH)

    def test_native_input_token_evidence_is_allowed_not_confused_with_tokenizer(self):
        self.assertEqual(module.safe_member('/content/invented/actual_native_input_preflight.json'),
                         'content/invented/actual_native_input_preflight.json')

    def test_transitive_hash_maps_preserve_original_and_relative_paths(self):
        value = {'code_sha256': {'/content/owned/source.py': 'a' * 64},
            'nested': {'historical_and_setup_reference_sha256': {'/content/old/ledger.jsonl': 'b' * 64},
                       'inherited_identity_sha256': {'load_protocol.json': 'c' * 64}}}
        refs = list(module.hash_references(value, Path('/content/invented/gate_protocol.json')))
        self.assertEqual({str(path) for path, _, _ in refs}, {
            '/content/owned/source.py', '/content/old/ledger.jsonl', '/content/invented/load_protocol.json'})

    def test_arbitrary_model_manifest_paths_are_not_followed(self):
        value = {'files': [{'path': 'weights.gguf', 'sha256': 'a' * 64}],
                 'model_dir': '/root/.cache/imaginary-model', 'wheel_path': '/content/setup/cpu.whl'}
        self.assertEqual(list(module.hash_references(value, Path('/content/invented/model_manifest.json'))), [])

    def test_product_source_files_use_actual_checkout_not_prepared_case_directory(self):
        value = {'code_identity': {'source_files': {'modules/invented.py': 'a' * 64}}}
        refs = list(module.hash_references(value,
            Path('/content/cbt-open-state-prepared-before-20261112-v2/protocol.json')))
        self.assertEqual(str(refs[0][0]), '/content/cbt-open-state-before-20261112-v2/modules/invented.py')
        combined = {'sections': {'after': {'checkout': '/content/cbt-open-state-after-20261112-v2',
                                          'source_code_identity': value['code_identity']}}}
        refs = list(module.hash_references(combined, Path('/content/study/main_freeze.json')))
        self.assertEqual(str(refs[0][0]), '/content/cbt-open-state-after-20261112-v2/modules/invented.py')

    def test_source_snapshot_files_use_original_prepared_root(self):
        value = {'source_prepared': '/content/actual-source-before',
                 'source_files_sha256': {'requests.jsonl': 'b' * 64}}
        refs = list(module.hash_references(value, Path('/content/effective/protocol.json')))
        self.assertEqual(str(refs[0][0]), '/content/actual-source-before/requests.jsonl')

    def test_unresolvable_relative_product_reference_is_blocked_not_dropped(self):
        with self.assertRaisesRegex(ValueError, 'exact.*checkout'):
            list(module.hash_references({'source_files': {'modules/invented.py': 'a' * 64}},
                                        Path('/content/unknown/protocol.json')))

    def test_per_case_code_sha256_app_path_uses_before_git_checkout(self):
        value = {'code_identity': {'code_sha256': {'app/colab_provider.py': 'a' * 64}}}
        origin = Path('/content/cbt-open-state-prepared-before-20261112-v2/cases/controlled-cumulative_repeat-en-1/prepared.json')
        refs = list(module.hash_references(value, origin))
        self.assertEqual(str(refs[0][0]), '/content/cbt-open-state-before-20261112-v2/app/colab_provider.py')
        self.assertNotIn('/cases/', str(refs[0][0]))

    def test_actual_export_refuses_non_colab_before_experiment_file_io(self):
        with self.assertRaisesRegex(RuntimeError, 'genuine_hosted_colab'):
            module.export('a' * 64, 'b' * 64, 'c' * 64)


if __name__ == '__main__':
    unittest.main()
