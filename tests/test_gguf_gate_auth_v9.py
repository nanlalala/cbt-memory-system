"""Tiny invented gate outputs only; no real model, tokenizer, GGUF or dataset."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1] / 'evaluation/cbt-memory-effect'
sys.path.insert(0, str(HERE))
import gguf_gate_auth_v9 as G
from test_run_gguf_reader_gate_v7_colab import GateTests, R


class LightweightGateAuthTests(GateTests):
    # Inherit only fixture helpers, not the gate runner test suite.
    def setUp(self):
        super().setUp()
        self.model = {'model_id': G.native.MODEL_ID, 'model_revision': G.MODEL_REVISION,
                      'quantization': G.native.QUANTIZATION, 'license': 'apache-2.0', 'files': G.FILES}
        self.runtime = {'package_name': 'llama-cpp-python', 'package_version': G.native.PACKAGE_VERSION,
            'wheel_bytes': G.native.WHEEL_BYTES, 'wheel_sha256': G.native.WHEEL_SHA256,
            'site_packages': str(self.root / 'invented-site'),
            'dependencies': [{'name': 'invented-fixture-only', 'version': '1'}]}
        identity = {**self.model, 'files_verified': 2, 'bytes_verified': sum(x['bytes'] for x in G.FILES),
            'manifest_sha256': G.digest(self.model), 'backend_policy_sha256': G.digest(G.native.policy()),
            'control_token_ids': G.native.CONTROL_IDS, 'native_eog_token_ids': [151643, 151645],
            'runtime_identity': {'package_version': G.native.PACKAGE_VERSION,
                'site_packages': self.runtime['site_packages'], 'wheel_bytes': G.native.WHEEL_BYTES,
                'wheel_sha256': G.native.WHEEL_SHA256, 'dependencies': self.runtime['dependencies'],
                'runtime_manifest_sha256': G.digest(self.runtime)}}
        identity['gpu_execution_condition'] = self.backend.identity['gpu_execution_condition']
        self.backend.gpu_execution_proof['constructor_log'] = {'path': str(self.root / 'invented-metadata.log')}
        (self.root / 'invented-metadata.log').write_text('invented fixture log only')
        self.backend.identity = identity
        self.backend.native.identity = identity
        self.setup.update({'model_identity': identity, 'secret_access': False})
        self.replace('model_manifest.json', self.model)
        self.replace('runtime_manifest.json', self.runtime)
        history = self.root / 'invented-historical-reference.json'
        history.write_text('{"fixture_only":true}')
        code_paths = [G.native.__file__, G.native.schema.__file__, R.__file__]
        self.protocol = {'version': G.GATE_VERSION, 'maximum_new_attempts': 8, 'planned_generations': 8,
            'main_generations_authorized_by_this_entry': 0, 'automatic_main_run': False,
            'backend_policy': G.native.policy(), 'new_paid_provider_calls': 0, 'secret_access': False,
            'code_sha256': {str(Path(p).resolve()): G.sha(p) for p in code_paths},
            'historical_and_setup_reference_sha256': {str(history): G.sha(history)},
            'generation_cases_sha256': G.sha(self.root / 'generation_cases.json'),
            'targets_sha256': G.sha(self.root / 'assessor_only/targets.json'),
            'model_manifest_sha256': G.sha(self.root / 'model_manifest.json'),
            'runtime_manifest_sha256': G.sha(self.root / 'runtime_manifest.json'),
            'artifact_verification_sha256': G.sha(self.root / 'pre_generation_artifact_verification.json')}
        self.replace('gate_protocol.json', self.protocol)

    def replace(self, name, value):
        (self.root / name).write_text(R.canonical(value) + '\n')

    def verify(self, **extra):
        with patch.object(G.native, 'verify_gpu_execution_proof', side_effect=self.fake_gpu_verify):
            return G.verify_gate(self.root,
            expected_protocol_sha256=G.sha(self.root / 'gate_protocol.json'),
            expected_report_sha256=G.sha(self.root / 'gate_results.json'),
            expected_authentication_sha256=G.sha(self.root / 'all_eight_authentication_receipt.json'),
            fixture_only=True, **extra)

    def test_lightweight_all_eight_pass_without_loading_backend(self):
        self.execute()
        with patch.object(G.native, 'load_backend') as load:
            receipt = self.verify()
        load.assert_not_called()
        self.assertEqual(receipt['typed_correct'], 8)
        self.assertTrue(receipt['fresh_actual_backend_must_repeat_frozen_runner_authenticate_before_main_admission'])
        self.assertIn('token-to-text not rerun here', receipt['scope'])
        self.assertEqual(receipt['new_model_calls'], 0)

    def test_forged_eight_pass_report_cannot_rescue_six_correct_values(self):
        self.backend.outputs = [5, None, 3, 3, None, 'mixed', None, None]
        self.execute()
        report = R.read(self.root / 'gate_results.json')
        self.assertEqual(report['typed_correct'], 6)
        report['typed_correct'] = 8; report['gate_passed'] = True
        for row in report['per_question']:
            row['typed_correct'] = True
        self.replace('gate_results.json', report)
        with self.assertRaisesRegex(RuntimeError, 'eight_of_eight_exact_typed_native_gate_required'):
            self.verify()

    def test_changed_published_bytes_block_before_target_values_are_parsed(self):
        self.execute()
        answer = self.root / 'response-04.json'
        answer.write_text(answer.read_text() + ' ')
        original_read = G.read
        opened_targets = []
        def checked_read(path):
            if Path(path).name == 'targets.json':
                opened_targets.append(path)
            return original_read(path)
        with patch.object(G, 'read', side_effect=checked_read):
            with self.assertRaisesRegex(ValueError, 'published_bytes_changed'):
                self.verify()
        self.assertEqual(opened_targets, [])

    def test_old_external_receipt_hash_cannot_be_replaced_by_current_bytes(self):
        self.execute()
        expected_report = G.sha(self.root / 'gate_results.json')
        report_path = self.root / 'gate_results.json'
        report_path.write_text(report_path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'externally_observed_gate_artifact_changed'):
            G.verify_gate(self.root, expected_protocol_sha256=G.sha(self.root / 'gate_protocol.json'),
                expected_report_sha256=expected_report,
                expected_authentication_sha256=G.sha(self.root / 'all_eight_authentication_receipt.json'), fixture_only=True)

    def test_actual_hosted_guard_precedes_any_gate_file_access(self):
        with patch.object(G.native.schema, 'require_hosted_colab', side_effect=RuntimeError('hosted_first')):
            with patch.object(G, 'sha') as hash_file:
                with self.assertRaisesRegex(RuntimeError, 'hosted_first'):
                    G.verify_gate('/missing', expected_protocol_sha256='a' * 64,
                        expected_report_sha256='b' * 64, expected_authentication_sha256='c' * 64)
                hash_file.assert_not_called()


# unittest also discovers inherited test methods; hide those here to keep this
# file's five new assertions separately counted from the runner's own suite.
for name in list(vars(GateTests)):
    if name.startswith('test_') and name not in vars(LightweightGateAuthTests):
        setattr(LightweightGateAuthTests, name, None)

del GateTests

if __name__ == '__main__':
    unittest.main()
