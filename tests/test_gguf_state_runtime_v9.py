"""Invented native fixtures only; no real model, tokenizer, GGUF or dataset."""
from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parents[1] / 'evaluation/cbt-memory-effect'
sys.path.insert(0, str(HERE))
import gguf_state_runtime_v9 as R
from test_run_gguf_reader_gate_v7_colab import FakeBackend


class NativeMainRuntimeFixtures(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'invented-new-main'; self.root.mkdir()
        R.reader._write_new(self.root / 'study_protocol.json', {'invented': 'new-v7-only'})
        self.backend = FakeBackend([5])
        identity = {**self.backend.identity, 'model_id': R.native.MODEL_ID,
            'model_revision': 'b' * 40, 'quantization': R.native.QUANTIZATION,
            'backend_policy_sha256': R.native.digest(R.native.policy())}
        self.backend.identity = identity; self.backend.native.identity = identity
        R.reader._write_new(self.root / 'native_setup_before.json', {'model_identity': identity, 'gpu_execution_proof': self.backend.gpu_execution_proof})
        R.reader._write_new(self.root / 'gate_revalidation_before.json', {'native_originals_reproduced': 8, 'typed_correct': 8})
        payload = {'messages': [{'role': 'system', 'content': 'Invented source: agreed target 5.'},
            {'role': 'user', 'content': 'What is my current agreed target count?'}], 'max_tokens': 64}
        self.rows = [{'case_id': 'invented-case-1', 'arm': arm,
            'request_payload': payload, 'request_sha256': R.reader.digest(payload)} for arm in R.ARMS]
        self.gates = SimpleNamespace(identity=lambda: {'invented': 'source-gates'},
            preflight=Mock(return_value={}), begin=Mock(return_value={'replayed': False}),
            finish=Mock(return_value={'publishable': True}))

    def patches(self):
        stack = ExitStack()
        stack.enter_context(patch.object(R, 'study', return_value={}))
        stack.enter_context(patch.object(R, 'code_identity', return_value={'invented': 'frozen-code'}))
        stack.enter_context(patch.object(R.native, 'verify_gpu_execution_proof', return_value=True))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return stack

    def test_shared_generation_retains_native_tokens_and_checks_each_logical_source(self):
        output = self.root / 'answers'
        with self.patches():
            summary = R.generate_rows(self.root, output, self.rows, self.backend, purpose='invented', gates=self.gates, metadata={'side': 'before'})
            answers, _ = R.authenticate(self.root, output, self.rows, self.backend)
        self.assertEqual(self.backend.calls, 1)
        self.assertEqual(self.gates.begin.call_count, 2)
        self.assertEqual(self.gates.finish.call_count, 2)
        self.assertEqual(summary['owned_generation_cost']['unique_admitted_attempts'], 1)
        self.assertEqual(len(answers), 2)
        self.assertEqual(answers[0]['response_text'], '{"answer":5}')
        self.assertEqual(answers[0]['raw_generated_token_ids'][-1], 151645)
        self.assertNotIn('answer_adapter', answers[0])

    def test_native_failure_keeps_observed_partial_and_never_retries(self):
        output = self.root / 'answers'
        self.backend.fail_attempt = 1
        with self.patches():
            with self.assertRaises(R.native.GenerationFailure):
                R.generate_rows(self.root, output, self.rows[:1], self.backend, purpose='invented', gates=self.gates, metadata={'side': 'before'})
            with self.assertRaisesRegex(FileExistsError, 'no_automatic_resume'):
                R.generate_rows(self.root, output, self.rows[:1], self.backend, purpose='invented', gates=self.gates, metadata={'side': 'before'})
        self.assertEqual(self.backend.calls, 1)
        records = [R.reader._checked(r) for r in R.reader.read_rows(self.root / 'generation_ledger.jsonl')]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[-1]['status'], 'discarded_or_failed_or_unknown')
        partial = R.reader.read_json(records[-1]['observed_partial']['path'])
        self.assertEqual(partial['raw_generated_token_ids'], [ord(c) for c in '{"an'])
        self.assertEqual(records[-1]['usage'], partial['usage'])
        self.assertFalse((output / 'answers_unscored.jsonl').exists())
        with self.assertRaisesRegex(RuntimeError, 'no_automatic_retry'):
            R.ledger(self.root)

    def test_withdrawn_source_discards_response_and_forbids_partial_export(self):
        output = self.root / 'answers'
        def revoked(row):
            if self.gates.preflight.call_count >= 4:
                raise PermissionError('invented persistent source withdrawal')
            return {}
        self.gates.preflight.side_effect = revoked
        with self.patches():
            with self.assertRaisesRegex(PermissionError, 'persistent source withdrawal'):
                R.generate_rows(self.root, output, self.rows[:1], self.backend, purpose='invented', gates=self.gates, metadata={'side': 'before'})
        self.assertEqual(self.backend.calls, 1)
        records = [R.reader._checked(r) for r in R.reader.read_rows(self.root / 'generation_ledger.jsonl')]
        self.assertIsNone(records[-1]['observed_partial'])
        self.assertIsNone(records[-1]['response_text'])
        self.assertIsNotNone(records[-1]['usage'])
        self.assertFalse((output / 'failed_partials').exists())
        self.assertFalse((output / 'answers_unscored.jsonl').exists())

    def test_changed_native_stream_is_not_hidden_by_same_whole_json(self):
        output = self.root / 'answers'
        with self.patches():
            R.generate_rows(self.root, output, self.rows[:1], self.backend, purpose='invented', gates=self.gates, metadata={'side': 'before'})
            original = self.backend.engine.detokenize
            def altered_output(ids, prev_tokens=None, special=False):
                if prev_tokens is not None:
                    return b'changed-invented-bytes'
                return original(ids, prev_tokens=prev_tokens, special=special)
            with patch.object(self.backend.engine, 'detokenize', side_effect=altered_output):
                with self.assertRaisesRegex(ValueError, 'native_original_tokens_bytes_schema_or_usage_changed'):
                    R.authenticate(self.root, output, self.rows[:1], self.backend)

    def test_owned_phase_gpu_receipt_is_required_before_admitting_a_request(self):
        receipt_path = self.root / 'native_setup_before.json'
        receipt = R.reader.read_json(receipt_path)
        receipt['model_identity'] = {'invented': 'another-model-condition'}
        receipt_path.write_text(R.reader.canonical(receipt))
        with self.patches():
            with self.assertRaisesRegex(ValueError, 'owned_phase_gpu_execution_proof_required'):
                R.generate_rows(self.root, self.root / 'answers', self.rows[:1], self.backend,
                    purpose='invented', gates=self.gates, metadata={'side': 'before'})
        self.assertEqual(self.backend.calls, 0)
        self.gates.begin.assert_not_called()

    def test_fresh_gpu_load_proof_does_not_change_cache_identity_or_archived_authentication(self):
        output = self.root / 'answers'
        with self.patches():
            R.generate_rows(self.root, output, self.rows[:1], self.backend,
                purpose='invented', gates=self.gates, metadata={'side': 'before'})
            key_before = R.generation_key(self.root, self.backend.identity, R.decoding_identity(), self.rows[0]['request_payload'])
            self.backend.gpu_execution_proof = {**self.backend.gpu_execution_proof, 'invented_new_load': True}
            key_after = R.generation_key(self.root, self.backend.identity, R.decoding_identity(), self.rows[0]['request_payload'])
            self.assertEqual(key_before, key_after)
            answers, _ = R.authenticate(self.root, output, self.rows[:1], self.backend)
        self.assertEqual(len(answers), 1)
        self.assertEqual(self.backend.calls, 1)

    def test_failed_or_missing_gate_blocks_main_creation_before_manifest_copy(self):
        for failure in (RuntimeError('six_of_eight_gate_failed'), FileNotFoundError('gate_report_absent')):
            blocked = self.root / ('blocked-' + type(failure).__name__)
            with patch.object(R, 'guard'), patch.object(R.gate_audit, 'verify_gate', side_effect=failure):
                with patch.object(R.reader, 'read_json') as read:
                    with self.assertRaises(type(failure)):
                        R.initialize(blocked, gate_root=self.root / 'old-gate',
                            expected_gate_protocol_sha256='a' * 64, expected_gate_report_sha256='b' * 64,
                            expected_gate_authentication_sha256='c' * 64)
                    read.assert_not_called()
            self.assertFalse(blocked.exists())

    def test_claimed_six_correct_still_blocks_even_if_auditor_returns(self):
        blocked = self.root / 'blocked-six-return'
        with patch.object(R, 'guard'), patch.object(R.gate_audit, 'verify_gate',
                return_value={'typed_correct': 6, 'schema_valid': 8}):
            with self.assertRaisesRegex(RuntimeError, 'all_eight_native_gate_answers_must_pass'):
                R.initialize(blocked, gate_root=self.root / 'old-gate',
                    expected_gate_protocol_sha256='a' * 64, expected_gate_report_sha256='b' * 64,
                    expected_gate_authentication_sha256='c' * 64)
        self.assertFalse(blocked.exists())

    def test_frozen_validation_time_records_success_and_failure_without_weakening_hashes(self):
        historical = self.root / 'invented-history.json'
        historical.write_text('{"fixture_only":true}')
        protocol = {'version': R.VERSION, 'maximum_new_study_attempts': R.CAP,
            'code_sha256': {'invented': 'frozen-code'}, 'study_root': str(self.root),
            'historical_reference_sha256': {str(historical): R.reader.file_hash(historical)},
            'inherited_identity_sha256': {}}
        (self.root / 'study_protocol.json').write_text(R.reader.canonical(protocol))
        with patch.object(R, 'guard'), patch.object(R, 'code_identity', return_value={'invented': 'frozen-code'}):
            self.assertEqual(R.study(self.root), protocol)
            historical.write_text('{"fixture_only":"changed"}')
            with self.assertRaisesRegex(ValueError, 'old_run_reference_changed_stop'):
                R.study(self.root)
        events = [R.reader._checked(row) for row in R.reader.read_rows(self.root / 'frozen_material_validation_timings.jsonl')]
        self.assertEqual([row['complete'] for row in events], [True, False])
        self.assertTrue(all(row['wall_seconds'] >= 0 and row['new_model_generations'] == 0 for row in events))


if __name__ == '__main__':
    unittest.main()
