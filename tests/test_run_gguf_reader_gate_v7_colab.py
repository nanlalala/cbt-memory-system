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
import run_gguf_reader_gate_v7_colab as R
import run_schema_reader_gate_v4_colab as old
from test_gguf_schema_reader_v5_colab import FakeEngine as PreviousFakeEngine, FakeNativeModule

class FakeEngine(PreviousFakeEngine):
    def n_ctx(self):
        return 4352


class FakeBackend:
    def __init__(self, outputs, fail_attempt=None):
        self.native = R.native.Backend(FakeEngine(), FakeNativeModule(), {'invented': True, 'gpu_execution_condition': {'invented_static_gpu': True}}, fixture_only=True)
        self.identity = self.native.identity
        self.gpu_execution_proof = {'proof_passed': True, 'invented_fixture': True}
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
                      'answer_generations_during_setup': 0, 'new_provider_calls': 0,
                      'gpu_execution_proof': self.backend.gpu_execution_proof}

    def fake_gpu_verify(self, proof, *, expected_condition=None, constructor_log_path=None):
        if proof.get('proof_passed') is not True:
            raise ValueError('invented_gpu_proof_rejected')
        if expected_condition != {'invented_static_gpu': True}:
            raise ValueError('invented_static_gpu_condition_changed')
        return True

    def tearDown(self):
        self.patch_root.stop()
        self.temp.cleanup()

    def execute(self, backend=None):
        backend = backend or self.backend
        setup = {**self.setup, 'model_identity': backend.identity,
                 'gpu_execution_proof': backend.gpu_execution_proof}
        with patch.object(R, 'hosted'), patch.object(R, 'frozen', return_value=(self.protocol, self.rows)), \
                patch.object(R.native, 'load_backend', return_value=(backend, setup)) as load, \
                patch.object(R.native, 'verify_gpu_execution_proof', side_effect=self.fake_gpu_verify, create=True), \
                redirect_stdout(io.StringIO()):
            R.run(constructor_log_path=str(self.root / 'invented-constructor.log'))
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

    def test_cpu_and_gpu_condition_versions_remain_separate(self):
        self.assertEqual(R.CPU_GATE_VERSION,
            'official-qwen7b-gguf-eight-case-readability-gate-v6-context4352')
        self.assertEqual(R.VERSION, 'official-qwen7b-t4-eight-case-readability-gate-v7')
        self.assertNotEqual(R.CPU_GATE_VERSION, R.VERSION)

    def test_unproved_gpu_fallback_blocks_before_first_generation_admission(self):
        self.backend.gpu_execution_proof['proof_passed'] = False
        with self.assertRaisesRegex(ValueError, 'invented_gpu_proof_rejected'):
            self.execute()
        self.assertEqual(self.backend.calls, 0)
        self.assertFalse((self.root / 'generation_ledger.jsonl').exists())
        setup = R.journal_rows(self.root / 'model_setup_journal.jsonl')
        self.assertEqual(setup[-1]['event'], 'native_model_setup_failed_or_unknown')

    def test_dynamic_gpu_proof_cannot_pollute_static_cache_identity(self):
        self.backend.identity['gpu_execution_proof'] = copy.deepcopy(self.backend.gpu_execution_proof)
        with self.assertRaisesRegex(ValueError, 'dynamic_gpu_load_proof'):
            self.execute()
        self.assertEqual(self.backend.calls, 0)
        self.assertFalse((self.root / 'generation_ledger.jsonl').exists())

    def test_setup_proof_must_be_the_same_original_load(self):
        setup = {**self.setup, 'gpu_execution_proof': {'proof_passed': True, 'other_load': True}}
        with patch.object(R.native, 'verify_gpu_execution_proof', create=True) as verify:
            with self.assertRaisesRegex(ValueError, 'same_load_actual_gpu_setup_proof'):
                R.require_gpu_setup_proof(self.backend, setup)
        verify.assert_not_called()

    def test_fresh_decoder_authenticates_both_load_proofs_without_equating_them(self):
        self.execute()
        plans = [self.backend.input_preflight(row['messages']) for row in self.rows]
        self.backend.gpu_execution_proof = {'proof_passed': True, 'invented_fresh_decoder_load': True}
        with patch.object(R.native, 'verify_gpu_execution_proof', side_effect=self.fake_gpu_verify, create=True) as verify:
            answers = R.authenticate(self.protocol, self.rows, self.backend, plans, original_load=False)
        self.assertEqual(len(answers), 8)
        self.assertEqual(verify.call_count, 2)
        self.assertNotEqual(verify.call_args_list[0].args[0], verify.call_args_list[1].args[0])

    def test_fresh_decoder_cannot_bypass_rejected_archived_or_fresh_gpu_proof(self):
        self.execute()
        plans = [self.backend.input_preflight(row['messages']) for row in self.rows]
        self.backend.gpu_execution_proof = {'proof_passed': False, 'invented_fresh_decoder_load': True}
        with patch.object(R.native, 'verify_gpu_execution_proof', side_effect=self.fake_gpu_verify, create=True):
            with self.assertRaisesRegex(ValueError, 'invented_gpu_proof_rejected'):
                R.authenticate(self.protocol, self.rows, self.backend, plans, original_load=False)
        # The archived proof also remains required even with a valid new load.
        original_setup = copy.deepcopy(self.setup)
        original_setup['gpu_execution_proof'] = {'proof_passed': False, 'invented_bad_archive': True}
        self.backend.gpu_execution_proof = {'proof_passed': True, 'invented_fresh_decoder_load': True}
        with patch.object(R.native, 'verify_gpu_execution_proof', side_effect=self.fake_gpu_verify, create=True):
            with self.assertRaisesRegex(ValueError, 'invented_gpu_proof_rejected'):
                R.require_gpu_setup_proof(self.backend, original_setup, original_load=False)

    def test_existing_claim_prevents_second_model_load_or_generation(self):
        self.execute()
        with patch.object(R, 'hosted'), patch.object(R, 'frozen', return_value=(self.protocol, self.rows)), \
                patch.object(R.native, 'load_backend') as load:
            with self.assertRaises(FileExistsError):
                R.run(constructor_log_path=str(self.root / 'invented-constructor.log'))
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
        with patch.object(R.native, 'verify_gpu_execution_proof', side_effect=self.fake_gpu_verify, create=True):
            answers = R.authenticate(self.protocol, self.rows, self.backend, plans)
        self.assertEqual(len(answers), 8)
        self.assertEqual(answers[0]['chosen_answer'], 5)
        with (self.root / 'response-01.json').open('w') as stream:
            changed = {**answers[0], 'response_text': '{"answer":17}'}
            stream.write(R.canonical(changed) + '\n')
        with patch.object(R.native, 'verify_gpu_execution_proof', side_effect=self.fake_gpu_verify, create=True):
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
                R.run(constructor_log_path=str(self.root / 'invented-constructor.log'))
            with self.assertRaisesRegex(RuntimeError, 'hosted_first'):
                R.initialize(expected_download_receipt_sha='a' * 64, expected_python_setup_stopped_receipt_sha='b' * 64,
                             expected_cuda_recovery_receipt_sha='f' * 64,
                             expected_cpu_gate_protocol_sha='c' * 64, expected_cpu_gate_result_sha='d' * 64,
                             expected_cpu_gate_ledger_sha='e' * 64)

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

    def cpu_fixture(self):
        cpu = self.root / 'invented-closed-cpu-v6'
        cpu.mkdir()
        (cpu / 'assessor_only').mkdir()
        frozen = {}
        for name, value in (
            ('generation_cases.json', self.rows), ('assessor_only/targets.json', self.targets),
            ('model_manifest.json', {'invented_model': True}),
            ('runtime_manifest.json', {'invented_runtime': True}),
            ('pre_generation_artifact_verification.json', {'invented_verification': True}),
        ):
            R.write(cpu / name, value)
            frozen[name] = R.sha(cpu / name)
        protocol = {'version': R.CPU_GATE_VERSION, 'maximum_new_attempts': 8,
            'planned_generations': 8, 'main_generations_authorized_by_this_entry': 0,
            'automatic_main_run': False, 'code_sha256': {}, 'historical_and_setup_reference_sha256': {},
            'generation_cases_sha256': frozen['generation_cases.json'],
            'targets_sha256': frozen['assessor_only/targets.json'],
            'model_manifest_sha256': frozen['model_manifest.json'],
            'runtime_manifest_sha256': frozen['runtime_manifest.json'],
            'artifact_verification_sha256': frozen['pre_generation_artifact_verification.json']}
        R.write(cpu / 'gate_protocol.json', protocol)
        protocol_sha = R.sha(cpu / 'gate_protocol.json')
        R.write(cpu / 'gate_results.json', {'never_open_before_new_gpu_authentication': True})
        for index in range(1, 9):
            name = f'response-{index:02}.json'
            R.write(cpu / name, {'invented_cpu_original': index})
            R.append(cpu / 'generation_ledger.jsonl', {'event': 'generation_intent', 'attempt': index,
                'protocol_sha256': protocol_sha})
            R.append(cpu / 'generation_ledger.jsonl', {'event': 'generation_outcome', 'attempt': index,
                'status': 'published', 'artifact': name, 'artifact_sha256': R.sha(cpu / name)})
        for name in ('actual_native_input_preflight.json', 'all_eight_authentication_receipt.json',
                     'model_setup_journal.jsonl', 'model_setup_receipt.json',
                     'preflight_timing_receipt.json', 'run_claim.json'):
            R.write(cpu / name, {'invented': name})
        return cpu, protocol_sha, R.sha(cpu / 'gate_results.json'), R.sha(cpu / 'generation_ledger.jsonl')

    def test_cpu_originals_anchor_without_opening_historical_answer_labels(self):
        cpu, protocol_sha, result_sha, ledger_sha = self.cpu_fixture()
        actual_read = R.read

        def prohibit_historical_labels(path):
            if Path(path).name in ('gate_results.json', 'targets.json'):
                raise AssertionError('historical result labels must stay unopened')
            return actual_read(path)

        with patch.object(R, 'CPU_GATE', cpu), patch.object(R, 'read', side_effect=prohibit_historical_labels):
            references = R._cpu_gate_references(protocol_sha, result_sha, ledger_sha)
        self.assertEqual(references[str(cpu / 'gate_results.json')], result_sha)
        self.assertIn(str(cpu / 'response-08.json'), references)

    def test_changed_cpu_response_rejected_even_with_original_closed_ledger(self):
        cpu, protocol_sha, result_sha, ledger_sha = self.cpu_fixture()
        (cpu / 'response-03.json').write_text('{"altered":true}')
        with patch.object(R, 'CPU_GATE', cpu):
            with self.assertRaisesRegex(ValueError, 'cpu_original_published_bytes'):
                R._cpu_gate_references(protocol_sha, result_sha, ledger_sha)

    def test_cpu_reference_hashes_require_external_full_anchors(self):
        with self.assertRaisesRegex(ValueError, 'externally_verified_complete_receipt'):
            R._cpu_gate_references(None, 'a' * 64, 'b' * 64)

    def recovery_fixture(self):
        model = self.root / 'invented-gpu-model-setup'
        python = self.root / 'invented-gpu-python-setup'
        recovery = self.root / 'invented-driver-validation-v9'
        origin = self.root / 'invented-cuda12-origin-v8'
        metadata = self.root / 'invented-pinned-model-metadata.json'
        release = self.root / 'invented-official-cuda-release.json'
        model.mkdir(); python.mkdir(); recovery.mkdir(); origin.mkdir()
        R.write(metadata, {'invented_model_metadata': True})
        R.write(release, {'invented_cuda_release': True})
        (model / 'official_model_metadata.json').write_bytes(metadata.read_bytes())
        R.write(model / 'download_protocol.json', {'model': {'repo': R.native.MODEL_ID,
            'revision': R.MODEL_REVISION, 'files': list(R.MODEL_FILES)}})
        R.write(model / 'download_journal.jsonl', {'invented': True})
        R.write(model / 'download_complete_receipt.json', {
            'stage': 'both_pinned_shards_downloaded_and_verified',
            'repo': R.native.MODEL_ID, 'revision': R.MODEL_REVISION,
            'protocol_sha256': R.sha(model / 'download_protocol.json'),
            'model_loaded': False, 'new_model_inference_attempts': 0,
            'files': [{**entry, 'verified_bytes': entry['bytes'],
                       'verified_sha256': entry['sha256']} for entry in R.MODEL_FILES]})
        wheels = []
        for filename in ('llama_cpp_python-invented.whl', 'diskcache-invented.whl'):
            path = python / filename
            path.write_bytes(('tiny wheel fixture ' + filename).encode())
            wheels.append({'filename': filename, 'size': path.stat().st_size, 'sha256': R.sha(path)})
        site = python / 'site-packages'
        pp = {'diskcache_version': '5.6.3', 'target': str(site), 'wheels': wheels,
            'official_release_tag': 'v0.3.36-cu124',
            'official_release_metadata_sha256': R.sha(release)}
        R.write(python / 'setup_protocol.json', pp)
        for name in ('setup_journal.jsonl', 'diskcache_pypi_metadata.json', 'import_probe_stderr.log'):
            R.write(python / name, {'invented': True})
        stopped = {'stage': 'isolated_cuda_setup_stopped_failed_or_unknown',
            'phase': 'isolated_import_and_compiled_cuda_probe', 'model_generations': 0,
            'paid_provider_calls': 0, 'secret_access': False,
            'base_environment_modified': False, 'automatic_retry': False}
        R.write(python / 'setup_stopped_receipt.json', stopped)
        nv_wheels = []
        nv_metadata = []
        for index, name in enumerate(('nvidia-cuda-runtime-cu12', 'nvidia-cublas-cu12')):
            wheel = origin / (name + '-invented.whl')
            wheel.write_bytes(('tiny invented dependency wheel ' + name).encode())
            nv_wheels.append({'name': name, 'filename': wheel.name,
                'bytes': wheel.stat().st_size, 'sha256': R.sha(wheel)})
            meta = origin / (name + '-metadata.json')
            R.write(meta, {'invented': name})
            nv_metadata.append({'path': str(meta), 'sha256': R.sha(meta)})
        prior = self.root / 'recover_gguf_cuda_dependencies_v8_colab.py'
        prior.write_text('# invented prior recovery code, never executed\n')
        R.write(origin / 'recovery_protocol.json', {'version': 'invented-stopped-v8'})
        R.write(origin / 'recovery_stopped_receipt.json', {
            'stage': 'isolated_cuda12_recovery_stopped_failed_or_unknown',
            'phase': 'native_linker_diagnosis', 'model_generations': 0,
            'paid_provider_calls': 0, 'secret_access': False, 'automatic_retry': False})
        for name in ('recovery_journal.jsonl', 'resolved_native_linker_report.json', 'install.log'):
            R.write(origin / name, {'invented': name})
        dependency_site = origin / 'site-packages'
        library_path = ':'.join([str(dependency_site / 'nvidia/cuda_runtime/lib'),
            str(dependency_site / 'nvidia/cublas/lib'), str(site / 'llama_cpp/lib'), '/usr/lib64-nvidia'])
        driver = {'library_path': '/usr/lib64-nvidia/libcuda.so.1',
            'resolved_path': '/usr/lib64-nvidia/libcuda.so.invented', 'bytes': 17, 'sha256': 'f' * 64,
            'library_directory': '/usr/lib64-nvidia', 'actual_gpu_query': 'Tesla T4, 580.82.07, 15360 MiB, 0 MiB'}
        rp = {'version': 'isolated-cu124-host-driver-path-validation-v9',
            'prior_v8_source_sha256': R.sha(prior), 'driver_reference': driver,
            'wheel_directory': str(origin),
            'old_failed_setup_references': {str(python / name): R.sha(python / name)
                for name in ('setup_protocol.json', 'setup_stopped_receipt.json', 'setup_journal.jsonl', 'import_probe_stderr.log')},
            'reused_llama_cpp_wheel': {'path': str(python / wheels[0]['filename']),
                'bytes': wheels[0]['size'], 'sha256': wheels[0]['sha256']},
            'dependency_target': str(dependency_site), 'llama_cpp_site_packages': str(site),
            'child_ld_library_path': library_path, 'wheels': nv_wheels, 'nvidia_pypi_metadata': nv_metadata,
            'model_generations': 0, 'maximum_network_requests': 0, 'package_installs': 0,
            'base_environment_modified': False, 'automatic_retry': False}
        rp['old_failed_setup_references'].update({str(origin / name): R.sha(origin / name)
            for name in ('recovery_protocol.json', 'recovery_stopped_receipt.json', 'recovery_journal.jsonl',
                         'resolved_native_linker_report.json', 'install.log')})
        rp['old_failed_setup_references'][str(prior)] = R.sha(prior)
        R.write(recovery / 'recovery_protocol.json', rp)
        manifest = {'version': rp['version'], 'wheels': nv_wheels,
            'installed_library_files': [], 'driver_reference': driver, 'wheel_directory': str(origin),
            'child_ld_library_path': library_path,
            'dependency_site_packages': str(dependency_site), 'llama_cpp_site_packages': str(site)}
        R.write(recovery / 'native_dependency_manifest.json', manifest)
        extension = {'native_dependency_manifest_path': str(recovery / 'native_dependency_manifest.json'),
            'native_dependency_manifest_sha256': R.sha(recovery / 'native_dependency_manifest.json'),
            'child_ld_library_path': library_path, 'dependency_site_packages': str(dependency_site),
            'driver_reference': driver, 'wheel_directory': str(origin)}
        receipt = {'stage': 'isolated_cuda12_driver_path_ready_no_model_loaded',
            'protocol_sha256': R.sha(recovery / 'recovery_protocol.json'),
            'llama_cpp_version': R.native.PACKAGE_VERSION,
            'llama_cpp_path': str(site / 'llama_cpp/__init__.py'),
            'native_library_path': str(site / 'llama_cpp/lib/libllama.so'),
            'compiled_gpu_offload_support': True, 'model_generations': 0,
            'old_failed_setup_preserved': True, 'older_failed_setups_preserved': True,
            'network_requests': 0, 'package_installs': 0, 'base_environment_modified': False,
            'automatic_retry': False, 'paid_provider_calls': 0, 'secret_access': False,
            'native_dependency_manifest_path': extension['native_dependency_manifest_path'],
            'native_dependency_manifest_sha256': extension['native_dependency_manifest_sha256'],
            'runtime_manifest_extension': extension}
        R.write(recovery / 'setup_receipt.json', receipt)
        for name in ('resolved_native_linker_report.json', 'import_probe_stderr.log'):
            R.write(recovery / name, {'invented': True})
        return model, python, recovery, metadata, release, wheels, receipt

    def setup_patches(self, fixture):
        model, python, recovery, metadata, release, wheels, receipt = fixture
        from contextlib import ExitStack
        stack = ExitStack()
        stack.enter_context(patch.multiple(R, MODEL_SETUP=model, PYTHON_SETUP=python,
            CUDA_RECOVERY=recovery, CUDA_DEPENDENCY_ORIGIN=self.root / 'invented-cuda12-origin-v8',
            V8_RECOVERY_SOURCE_SHA=R.sha(self.root / 'recover_gguf_cuda_dependencies_v8_colab.py'),
            MODEL_METADATA=metadata, MODEL_METADATA_SHA=R.sha(metadata),
            GPU_RELEASE_METADATA=release, GPU_RELEASE_METADATA_SHA=R.sha(release)))
        stack.enter_context(patch.object(R.native, 'WHEEL_BYTES', wheels[0]['size']))
        stack.enter_context(patch.object(R.native, 'WHEEL_SHA256', wheels[0]['sha256']))
        stack.enter_context(patch.object(R.importlib.metadata, 'version',
            side_effect={**R.BASE_DEPENDENCIES, 'diskcache': '5.6.3'}.__getitem__))
        return stack

    def setup_args(self, fixture):
        model, python, recovery, *_ = fixture
        return (R.sha(model / 'download_complete_receipt.json'),
                R.sha(python / 'setup_stopped_receipt.json'), R.sha(recovery / 'setup_receipt.json'))

    def test_recovery_contract_preserves_failed_install_and_builds_all_loader_fields(self):
        fixture = self.recovery_fixture()
        original_path = list(sys.path)
        try:
            with self.setup_patches(fixture):
                manifest, runtime, references = R._setup_material(*self.setup_args(fixture))
                self.assertEqual(runtime['backend'], 'cuda')
                self.assertEqual(runtime['cuda_release_tag'], 'v0.3.36-cu124')
                self.assertEqual(runtime['official_release_metadata_sha256'], R.sha(fixture[4]))
                self.assertEqual(manifest['model_revision'], R.MODEL_REVISION)
                self.assertEqual(runtime['cuda_recovery_receipt_sha256'], self.setup_args(fixture)[2])
                self.assertEqual(runtime['native_dependency_manifest_path'],
                    str(fixture[2] / 'native_dependency_manifest.json'))
                self.assertIn(str(fixture[1] / 'setup_stopped_receipt.json'), references)
                self.assertFalse((fixture[1] / 'setup_receipt.json').exists())
        finally:
            sys.path[:] = original_path

    def test_recovery_cannot_relabel_the_original_failed_install_successful(self):
        fixture = self.recovery_fixture()
        R.write(fixture[1] / 'setup_receipt.json', {'fabricated_success': True})
        with self.setup_patches(fixture):
            with self.assertRaisesRegex(ValueError, 'must_not_be_relabelled'):
                R._setup_material(*self.setup_args(fixture))

    def test_recovery_rejects_changed_original_failure_bytes_even_with_new_external_hash(self):
        fixture = self.recovery_fixture()
        path = fixture[1] / 'setup_stopped_receipt.json'
        value = R.read(path)
        value['altered_after_recovery'] = True
        path.write_text(R.canonical(value))
        with self.setup_patches(fixture):
            with self.assertRaisesRegex(ValueError, 'preserved_original_failed_setup_bytes_changed'):
                R._setup_material(*self.setup_args(fixture))

    def test_recovery_compiled_support_is_required_but_not_treated_as_model_offload(self):
        fixture = self.recovery_fixture()
        fixture[-1]['compiled_gpu_offload_support'] = False
        (fixture[2] / 'setup_receipt.json').write_text(R.canonical(fixture[-1]))
        with self.setup_patches(fixture):
            with self.assertRaisesRegex(ValueError, 'separately_verified_cuda12_dependency_recovery'):
                R._setup_material(*self.setup_args(fixture))


if __name__ == '__main__':
    unittest.main()
