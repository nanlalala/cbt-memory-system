"""Tiny invented reading fixtures and fake native outputs; no model/data run."""
import copy
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1] / 'evaluation/cbt-memory-effect'
sys.path.insert(0, str(HERE))
import run_gguf_reader_gate_v6_colab as R
import run_schema_reader_gate_v4_colab as old
from test_gguf_schema_reader_v5_colab import FakeEngine as PreviousFakeEngine, FakeNativeModule

class FakeEngine(PreviousFakeEngine):
    def n_ctx(self):
        return 4352


class FakeBackend:
    def __init__(self, outputs, fail_attempt=None):
        self.native = R.native.Backend(FakeEngine(), FakeNativeModule(), {'invented': True}, fixture_only=True)
        self.identity = self.native.identity
        self.engine = self.native.engine
        self.eog_ids = self.native.eog_ids
        self.outputs = outputs
        self.calls = 0
        self.fail_attempt = fail_attempt

    def input_preflight(self, messages):
        return self.native.input_preflight(messages, fixture_only=True)

    def generate(self, messages, question_schema, max_new_tokens=64):
        self.calls += 1
        self.engine.response = R.canonical({'answer': self.outputs[self.calls - 1]})
        self.engine.fail_after = 4 if self.fail_attempt == self.calls else None
        return self.native.generate(messages, question_schema, max_new_tokens=max_new_tokens, fixture_only=True)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patch_root = patch.object(R, 'ROOT', self.root)
        self.patch_root.start()
        self.rows, self.targets = old.cases()
        (self.root / 'assessor_only').mkdir()
        R.write(self.root / 'generation_cases.json', self.rows)
        R.write(self.root / 'assessor_only/targets.json', self.targets)
        self.protocol = {'targets_sha256': R.sha(self.root / 'assessor_only/targets.json')}
        R.write(self.root / 'gate_protocol.json', self.protocol)
        R.write(self.root / 'model_manifest.json', {'invented': True})
        R.write(self.root / 'runtime_manifest.json', {'site_packages': str(self.root / 'invented-site')})
        R.write(self.root / 'pre_generation_artifact_verification.json', {'freeze_preparation_seconds': 0.5})
        self.outputs = [5, 5, 3, 3, 'mixed', 'mixed', None, None]
        self.backend = FakeBackend(self.outputs)
        self.setup = {'model_identity': self.backend.identity, 'setup_seconds': 2.0,
                      'answer_generations_during_setup': 0, 'new_provider_calls': 0}

    def tearDown(self):
        self.patch_root.stop()
        self.temp.cleanup()

    def execute(self, backend=None):
        backend = backend or self.backend
        setup = {**self.setup, 'model_identity': backend.identity}
        with patch.object(R, 'hosted'), patch.object(R, 'frozen', return_value=(self.protocol, self.rows)), \
                patch.object(R.native, 'load_backend', return_value=(backend, setup)) as load, \
                redirect_stdout(io.StringIO()):
            R.run()
        return load

    def test_exact_old_eight_case_generation_shape_no_answer_labels(self):
        self.assertEqual(len(R.check_generation_cases(self.rows)), 8)
        for row in self.rows:
            self.assertNotIn('answer', row)
            self.assertNotIn('expected_answer', row)
            self.assertNotIn('gold', row)
        wrong = copy.deepcopy(self.rows)
        wrong[0]['messages'][-1]['content'] += ' changed'
        with self.assertRaisesRegex(ValueError, 'generation_case_request'):
            R.check_generation_cases(wrong)

    def test_exact_eight_all_correct_passes_without_main_run(self):
        load = self.execute()
        report = R.read(self.root / 'gate_results.json')
        self.assertEqual((report['planned'], report['schema_valid'], report['typed_correct']), (8, 8, 8))
        self.assertTrue(report['gate_passed'])
        self.assertFalse(report['automatic_main_run'])
        self.assertFalse(report['clinical_or_memory_effect_claim'])
        self.assertEqual(report['main_generations_authorized_by_this_entry'], 0)
        self.assertEqual(self.backend.calls, 8)
        self.assertEqual(load.call_count, 1)
        journal = R.journal_rows(self.root / 'generation_ledger.jsonl')
        self.assertEqual([row['event'] for row in journal], ['generation_intent', 'generation_outcome'] * 8)
        self.assertEqual(len(R.journal_rows(self.root / 'model_setup_journal.jsonl')), 2)

    def test_eight_valid_json_but_six_correct_stays_failed(self):
        self.backend.outputs = [5, None, 3, 3, None, 'mixed', None, None]
        self.execute()
        report = R.read(self.root / 'gate_results.json')
        self.assertEqual((report['schema_valid'], report['typed_correct']), (8, 6))
        self.assertFalse(report['gate_passed'])
        self.assertFalse(report['automatic_main_run'])

    def test_existing_claim_prevents_second_model_load_or_generation(self):
        self.execute()
        with patch.object(R, 'hosted'), patch.object(R, 'frozen', return_value=(self.protocol, self.rows)), \
                patch.object(R.native, 'load_backend') as load:
            with self.assertRaises(FileExistsError):
                R.run()
        load.assert_not_called()
        self.assertEqual(self.backend.calls, 8)

    def test_partial_native_failure_is_counted_and_never_retried(self):
        backend = FakeBackend(self.outputs, fail_attempt=2)
        with self.assertRaises(R.native.GenerationFailure):
            self.execute(backend)
        journal = R.journal_rows(self.root / 'generation_ledger.jsonl')
        self.assertEqual(len(journal), 4)
        self.assertEqual(journal[-1]['status'], 'failed_or_unknown')
        self.assertFalse(journal[-1]['automatic_retry'])
        self.assertEqual(backend.calls, 2)
        self.assertTrue((self.root / 'failed-partial-02.json').is_file())
        self.assertFalse((self.root / 'gate_results.json').exists())
        self.assertFalse((self.root / 'all_eight_authentication_receipt.json').exists())

    def test_partial_authentication_and_scoring_reject_before_target_values(self):
        with patch.object(R, 'read', side_effect=AssertionError('no target read permitted')):
            with self.assertRaisesRegex(ValueError, 'all_eight_authenticated'):
                R.score_after_authentication(self.protocol, self.rows[:-1], [], self.setup)
        with self.assertRaises(FileNotFoundError):
            R.score_after_authentication(self.protocol, self.rows, [{}] * 8, self.setup)

    def test_raw_token_bytes_usage_and_hashes_authenticate(self):
        self.execute()
        plans = [self.backend.input_preflight(row['messages']) for row in self.rows]
        answers = R.authenticate(self.protocol, self.rows, self.backend, plans)
        self.assertEqual(len(answers), 8)
        self.assertEqual(answers[0]['chosen_answer'], 5)
        with (self.root / 'response-01.json').open('w') as stream:
            changed = {**answers[0], 'response_text': '{"answer":17}'}
            stream.write(R.canonical(changed) + '\n')
        with self.assertRaisesRegex(ValueError, 'published_native_answer_bytes_changed'):
            R.authenticate(self.protocol, self.rows, self.backend, plans)

    def test_journal_insertion_or_mutation_is_detected(self):
        path = self.root / 'tiny-journal.jsonl'
        R.append(path, {'event': 'invented', 'value': 1})
        R.append(path, {'event': 'invented', 'value': 2})
        rows = R.journal_rows(path)
        self.assertEqual(len(rows), 2)
        rows[0]['value'] = 3
        path.write_text('\n'.join(R.canonical(row) for row in rows) + '\n')
        with self.assertRaisesRegex(ValueError, 'chained_journal_integrity'):
            R.journal_rows(path)

    def test_hosted_guard_before_artifact_or_setup_work(self):
        with patch.object(R, 'hosted', side_effect=RuntimeError('hosted_first')), \
                patch.object(R, 'read', side_effect=AssertionError('no artifact read')):
            with self.assertRaisesRegex(RuntimeError, 'hosted_first'):
                R.run()
            with self.assertRaisesRegex(RuntimeError, 'hosted_first'):
                R.initialize(expected_download_receipt_sha='a' * 64, expected_python_receipt_sha='b' * 64, expected_diagnostic_receipt_sha='c' * 64)

    def test_no_invalid_external_receipt_hash_default_is_accepted(self):
        for invalid in (None, '', 'a' * 63, 'G' * 64, True):
            with self.assertRaisesRegex(ValueError, 'externally_verified'):
                R._check_expected_sha(invalid)

    def test_changed_recorded_old_history_rejects_before_labels_or_new_freeze(self):
        old_root = self.root / 'invented-old-v4'
        old_root.mkdir()
        history = old_root / 'invented-historical-ledger.jsonl'
        history.write_text('original invented historical bytes\n')
        recorded_sha = R.sha(history)
        protocol = {'historical_reference_sha256': {str(history): recorded_sha},
                    'code_sha256': {'unused-after-earlier-rejection': '0' * 64},
                    'inherited_identity_sha256': {}}
        R.write(old_root / 'gate_protocol.json', protocol)
        expected_old_protocol_sha = R.sha(old_root / 'gate_protocol.json')
        history.write_text('tampered invented historical bytes with unchanged row count\n')
        with patch.object(R, 'OLD_GATE', old_root), \
                patch.object(R, 'OLD_PROTOCOL_SHA', expected_old_protocol_sha):
            with self.assertRaisesRegex(ValueError, 'old_v4_recorded_reference_or_code_identity_changed'):
                R._historical_references()

    def diagnostic_fixture(self):
        failed = self.root / 'invented-failed-v5'
        diagnostic = self.root / 'invented-load-diagnostic'
        failed.mkdir(); diagnostic.mkdir()
        R.write(failed / 'gate_protocol.json', {'code_sha256': {},
            'historical_and_setup_reference_sha256': {}})
        R.write(failed / 'run_claim.json', {'invented': True})
        R.append(failed / 'model_setup_journal.jsonl', {'event': 'native_model_setup_intent'})
        R.append(failed / 'model_setup_journal.jsonl', {'event': 'native_model_setup_failed_or_unknown',
            'failure_type': 'ValueError', 'automatic_retry': False})
        failed_sha = R.sha(failed / 'gate_protocol.json')
        R.write(diagnostic / 'diagnostic_protocol.json', {'maximum_load_attempts': 1,
            'maximum_answer_generations': 0, 'automatic_retry': False,
            'old_gate_protocol_sha256': failed_sha, 'native_backend_sha256': R.FAILED_V5_BACKEND_SHA})
        R.write(diagnostic / 'load_intent.json', {'maximum_answer_generations': 0})
        receipt = {'stage': 'diagnostic_complete_no_answers', 'failure': None,
            'answer_generations': 0, 'failed_v5_gate_unchanged': True, 'automatic_retry': False,
            'actual_details': {'actual_n_ctx': 4352, 'requested_n_ctx': 4160,
                'actual_control_token_ids': {key: [value] for key, value in R.native.CONTROL_IDS.items()},
                'actual_token_eos': 151645,
                'native_eog_flags': {'151643': True, '151644': False, '151645': True},
                'post_load_dll_verification_passed': True}}
        R.write(diagnostic / 'diagnostic_receipt.json', receipt)
        return failed, diagnostic, failed_sha, receipt

    def test_observed_padding_diagnostic_required_and_frozen(self):
        failed, diagnostic, failed_sha, receipt = self.diagnostic_fixture()
        with patch.object(R, 'FAILED_V5', failed), patch.object(R, 'DIAGNOSTIC', diagnostic), \
                patch.object(R, 'FAILED_V5_PROTOCOL_SHA', failed_sha):
            expected = R.sha(diagnostic / 'diagnostic_receipt.json')
            references = R._pre_admission_failure_references(expected)
            self.assertEqual(len(references), 6)
            self.assertEqual(references[str(diagnostic / 'diagnostic_receipt.json')], expected)
            receipt['actual_details']['actual_n_ctx'] = 4160
            (diagnostic / 'diagnostic_receipt.json').write_text(R.canonical(receipt))
            with self.assertRaisesRegex(ValueError, 'observed_zero_answer_context_padding'):
                R._pre_admission_failure_references(R.sha(diagnostic / 'diagnostic_receipt.json'))

    def test_any_v5_admission_blocks_the_new_context_repair_gate(self):
        failed, diagnostic, failed_sha, receipt = self.diagnostic_fixture()
        (failed / 'generation_ledger.jsonl').write_text('')
        with patch.object(R, 'FAILED_V5', failed), patch.object(R, 'DIAGNOSTIC', diagnostic), \
                patch.object(R, 'FAILED_V5_PROTOCOL_SHA', failed_sha):
            with self.assertRaisesRegex(ValueError, 'failed_v5_must_remain_zero_admissions'):
                R._pre_admission_failure_references(R.sha(diagnostic / 'diagnostic_receipt.json'))


if __name__ == '__main__':
    unittest.main()
