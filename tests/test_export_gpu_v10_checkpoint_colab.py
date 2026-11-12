"""Tiny invented metadata only; no hosted session, model or dataset execution."""
import base64
import ast
import hashlib
import importlib.util
import json
from contextlib import contextmanager
import tempfile
import unittest
from pathlib import Path
import sqlite3
from unittest import mock



SOURCE = Path(__file__).parents[1] / 'evaluation/cbt-memory-effect/export_gpu_v10_checkpoint_colab.py'
spec = importlib.util.spec_from_file_location('gpu_checkpoint_v10_fixture', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@contextmanager
def tiny_content():
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        replacements = {}
        for name in ('CONTENT', 'GATE', 'STUDY', 'CODE', 'MODEL', 'PYTHON', 'RECOVERY8', 'RECOVERY9',
                 'READINESS', 'NVIDIA_META', 'PUBLIC_META', 'CPU_CLOSURE', 'CPU_GATE', 'FAILED_V7', 'FAILED_V8'):
            value = tmp_path if name == 'CONTENT' else tmp_path / name.lower()
            if name == 'STUDY': value = tmp_path / 'cbt-qwen7b-gpu-state-study-20261112-v9'
            if name == 'FAILED_V8': value = tmp_path / 'cbt-qwen7b-gpu-state-study-20261112-v8'
            replacements[name] = value
        with mock.patch.multiple(m, **replacements):
            yield tmp_path



def raises(kind, match=None):
    case = unittest.TestCase()
    return case.assertRaisesRegex(kind, match) if match else case.assertRaises(kind)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(m.canonical(value) + '\n')


def record(value):
    return {**value, 'record_sha256': m.digest(value)}


def test_hosted_guard_is_before_all_artifact_io():
    with mock.patch.object(m, 'hosted', side_effect=RuntimeError('not hosted')), \
         mock.patch.object(m, 'sha', side_effect=AssertionError('artifact read')):
        with raises(RuntimeError, match='not hosted'):
            m.export(Path('/content/a.zip'), Path('/content/r.json'), '0' * 64)


def test_owned_paths_and_sensitive_filters(content):
    with raises(ValueError): m.owned(content / '../escape.json')
    other = content / 'real'; other.mkdir()
    link = content / 'link'; link.symlink_to(other, target_is_directory=True)
    with raises(ValueError): m.owned(link / 'a.json')
    assert m.excluded(content / 'site-packages/a.py')
    assert m.excluded(content / '.env')
    assert m.excluded(content / 'libcuda.so.1')
    assert m.excluded(content / 'x.sqlite-wal')


def test_binary_never_read_and_descriptors_enrich_without_losing_identity(content):
    root = content / 'model'; root.mkdir()
    p = root / 'tiny.whl'; p.write_bytes(b'invented binary header')
    with mock.patch.object(m, 'sha', side_effect=AssertionError('binary read')):
        c = m.Collector(content / 'stage', [])
        c.tree(root)
        c.binary_descriptor(p, item={'bytes': p.stat().st_size, 'sha256': 'a' * 64, 'url': 'https://publisher.invalid/fixed'})
        c.binary_descriptor(p, item={'bytes': None, 'url': None})
    assert not c.entries
    d = c.descriptors[str(p)]
    assert d['sha256'] == 'a' * 64 and d['reacquire_original_url'].endswith('/fixed')
    assert d['bytes'] == p.stat().st_size


def test_immutable_text_capture_and_labelled_sqlite_backup(content):
    stage = content / 'stage'; stage.mkdir()
    c = m.Collector(stage, [])
    p = content / 'original.json'; write(p, {'invented': 1})
    c.add(p); entry = c.entries[str(p)]
    p.write_text('changed after capture')
    assert m.sha(entry['capture_source']) == entry['sha256']
    assert entry['representation'] == 'original_bytes'
    db = content / 'case.sqlite3'
    with sqlite3.connect(db) as conn:
        conn.execute('create table fixture (n integer)'); conn.execute('insert into fixture values (1)')
    c.add(db); d = c.entries[str(db)]
    assert d['representation'] == 'sqlite_online_backup_not_original_live_database_bytes'
    assert d['source_sha256'] == m.sha(db)
    with sqlite3.connect(d['capture_source']) as conn: assert conn.execute('select n from fixture').fetchone() == (1,)


def test_original_relative_code_and_source_references(content):
    checkout, prepared = content / 'checkout', content / 'prepared'
    value = {'checkout': str(checkout), 'source_prepared': str(prepared),
        'source_files_sha256': {'requests.jsonl': '1' * 64},
        'code_sha256': {'app/memory.py': '2' * 64}}
    result = list(m.hash_references(value, content / 'manifest.json', {}))
    assert (prepared / 'requests.jsonl', '1' * 64, 'source_files_sha256') in result
    assert (checkout / 'app/memory.py', '2' * 64, 'code_sha256') in result
    unresolved = list(m.hash_references({'code_sha256': {'app/a.py': '3' * 64}}, content / 'm.json', {}))
    assert unresolved[0][0] is None


def test_torn_and_changed_journal_are_explicit(content):
    p = content / 'generation_ledger.jsonl'
    good = record({'event': 'generation_intent', 'canonical_key': 'key'})
    p.write_bytes((m.canonical(good) + '\n{"event":').encode())
    values, integrity = m.rows(p, require_record_sha=True)
    assert len(values) == 1 and integrity['torn_final_line']
    changed = {**good, 'canonical_key': 'changed'}
    p.write_text(m.canonical(changed) + '\n')
    values, integrity = m.rows(p, require_record_sha=True)
    assert values == [] and integrity['record_errors']


def test_raw_byte_binding_and_unstarted_state(content):
    body = b'{"answer":true}'
    answer = {'raw_generated_token_ids': [101, 102],
        'response_bytes_base64': base64.b64encode(body).decode(),
        'raw_generated_bytes_base64': base64.b64encode(body).decode(),
        'response_bytes_sha256': hashlib.sha256(body).hexdigest(),
        'response_sha256': hashlib.sha256(body).hexdigest(),
        'response_text': body.decode(), 'raw_generated_text': body.decode(),
        'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}}
    assert m.validate_answer_bytes(answer)
    with raises(ValueError): m.validate_answer_bytes({**answer, 'response_bytes_sha256': '0' * 64})
    state = m.observed_state([])
    assert state['state'] == 'gpu_gate_not_initialized'
    assert state['logical_answers'] == 0 and not state['main_completion_candidate']
    assert not state['clinical_or_memory_effect_inferred']


def test_gpu_log_range_is_bound_to_original_bytes(content):
    stage = content / 'stage'; stage.mkdir()
    c = m.Collector(stage, [])
    log = content / 'constructor.log'; log.write_bytes(b'prefix\noffloaded fixture\n')
    body = log.read_bytes()[7:]
    proof = {'constructor_log': {'path': str(log), 'start_byte': 7, 'end_byte': log.stat().st_size,
        'bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest()}}
    c.catalogs(proof, content / 'native_setup_before.json')
    assert len(c.log_ranges) == 1 and not c.issues
    c.catalogs({'constructor_log': {**proof['constructor_log'], 'sha256': '0' * 64}}, content / 'setup.json')
    assert c.issues[-1]['kind'] == 'gpu_constructor_log_range_not_verified'


def test_ledger_duplicate_and_unknown_events_are_preserved(content):
    m.STUDY.mkdir()
    values = [record({'event': 'generation_intent', 'canonical_key': 'key'})] * 2
    values.append(record({'event': 'unrecognized', 'canonical_key': 'x'}))
    (m.STUDY / 'generation_ledger.jsonl').write_text(''.join(m.canonical(v) + '\n' for v in values))
    state = m.ledger_state(m.STUDY)
    assert state['event_binding']['duplicate_intents']
    assert state['event_binding']['unknown_events'] == ['unrecognized']
    assert state['unsettled_intents'] == 1


def test_complete_invented_side_bindings_and_cache_tampering(content):
    m.STUDY.mkdir()
    write(m.STUDY / 'study_protocol.json', {'invented_fixture_only': True})
    output, prepared = m.STUDY / 'answers_before', m.STUDY / 'prepared/before'
    output.mkdir(); prepared.mkdir(parents=True)
    payload = {'messages': [{'role': 'user', 'content': 'invented fixture?'}], 'max_tokens': 64}
    request_hash = m.digest(payload)
    requests = [{'case_id': str(n), 'arm': 'fixture', 'request_payload': payload,
                 'request_sha256': request_hash} for n in range(48)]
    (prepared / 'requests.jsonl').write_text(''.join(m.canonical(r) + '\n' for r in requests))
    schema = {'fixture': True}
    pre = {'input_token_ids': [10], 'input_tokens': 1, 'input_token_ids_sha256': m.digest([10]),
        'question_schema': schema, 'question_schema_sha256': m.digest(schema), 'rendered_prompt_sha256': 'a' * 64}
    proof = {'proof_passed': True, 'all_layers_offloaded': True}
    model = {'invented_fixture_only': True}
    setup_path = m.STUDY / 'native_setup_before.json'
    reval_path = m.STUDY / 'gate_revalidation_before.json'
    write(setup_path, {'model_identity': model, 'gpu_execution_proof': proof})
    write(reval_path, {'native_originals_reproduced': 8, 'typed_correct': 8})
    run = {'model': model, 'decoding_identity': {'fixture': True}, 'rows_sha256': m.digest(requests),
        'study_protocol_sha256': m.sha(m.STUDY / 'study_protocol.json'),
        'actual_native_input_preflight': [pre] * 48,
        'actual_gpu_phase_execution_binding': {'phase': 'before', 'setup_receipt_sha256': m.sha(setup_path),
            'gpu_execution_proof_sha256': m.digest(proof), 'gate_revalidation_sha256': m.sha(reval_path)}}
    write(output / 'run_protocol.json', run)
    key = m.digest({'new_study_protocol_sha256': run['study_protocol_sha256'], 'model': model,
        'decoding': run['decoding_identity'], 'payload': payload, 'max_new_tokens': 64, 'do_sample': False})
    body = b'{"answer":true}'
    value = {'canonical_key': key, 'raw_generated_token_ids': [11, 12],
        'response_bytes_base64': base64.b64encode(body).decode(), 'raw_generated_bytes_base64': base64.b64encode(body).decode(),
        'response_bytes_sha256': hashlib.sha256(body).hexdigest(), 'response_sha256': hashlib.sha256(body).hexdigest(),
        'response_text': body.decode(), 'raw_generated_text': body.decode(),
        'usage': {'prompt_tokens': 1, 'completion_tokens': 2, 'total_tokens': 3},
        **{k: pre[k] for k in ('input_token_ids_sha256', 'question_schema_sha256', 'rendered_prompt_sha256')}}
    cache = output / 'canonical' / (key + '.json'); write(cache, value)
    answers = [record({**value, 'case_id': r['case_id'], 'arm': r['arm'], 'request_sha256': request_hash,
        'run_protocol_sha256': m.digest(run), 'source_checks_passed': True,
        'logical_processing_wall_seconds': 0.01}) for r in requests]
    answer_path = output / 'answers_unscored.jsonl'
    answer_path.write_text(''.join(m.canonical(a) + '\n' for a in answers))
    write(output / 'completion_summary.json', {'complete': True, 'logical_completed': 48,
        'answers_sha256': m.sha(answer_path), 'run_protocol_sha256': m.digest(run)})
    ledger = {'records': [{'event': 'generation_intent', 'canonical_key': key, 'output': str(output),
        'request_sha256': request_hash, 'run_protocol_sha256': m.digest(run),
        'input_token_ids_sha256': pre['input_token_ids_sha256'], 'question_schema_sha256': pre['question_schema_sha256']},
        {'event': 'generation_outcome', 'canonical_key': key, 'status': 'published', 'cache_path': str(cache),
         'cache_sha256': m.sha(cache), 'usage': value['usage'], 'response_sha256': value['response_sha256']}]}
    result = m.validate_side_bytes('before', {'fresh_prepared': str(prepared)}, ledger)
    assert result['ordered_request_cache_vector_phase_byte_bindings_verified']
    assert not result['native_token_detokenization_reexecuted']
    write(cache, {**value, 'response_text': 'tampered'})
    with raises(ValueError): m.validate_side_bytes('before', {'fresh_prepared': str(prepared)}, ledger)


def test_new_live_completion_file_cannot_create_archived_completion(content):
    stage = content / 'stage'; stage.mkdir()
    c = m.Collector(stage, [])
    p = content / 'newly_created.json'; write(p, {'fixture_completion': True})
    state = {'main_completion_candidate': False, 'side_original_byte_bindings': {'before': {
        'validated_original_file_sha256': {str(p): m.sha(p)}}}}
    issues = []
    m.completion_snapshot_closure(c, state, issues)
    assert not state['main_completion_candidate']
    assert state['completion_required_snapshot_failures'] == [str(p)]
    assert issues[0]['kind'] == 'live_completion_evidence_not_identical_in_captured_archive'


def test_three_real_cpu_gate_role_aliases_resolve_and_verify(content):
    stage = content / 'stage'; stage.mkdir()
    files = {'protocol': 'gate_protocol.json', 'result': 'gate_results.json', 'ledger': 'generation_ledger.jsonl'}
    refs = {}
    for role, name in files.items():
        p = m.CPU_GATE / name; write(p, {'invented_fixture_role': role}); refs[role] = m.sha(p)
    protocol = m.GATE / 'gate_protocol.json'; write(protocol, {'cpu_gate_reference_sha256': refs})
    c = m.Collector(stage, []); c.add(protocol); c.close_references()
    assert not c.issues
    assert all(str(m.CPU_GATE / name) in c.entries for name in files.values())
    assert not any(str(m.GATE / role) in c.entries for role in files)


def test_unknown_alias_and_real_missing_file_remain_failures(content):
    stage = content / 'stage'; stage.mkdir()
    protocol = m.GATE / 'gate_protocol.json'
    write(protocol, {'cpu_gate_reference_sha256': {'protocol': 'a' * 64, 'invented_alias': 'b' * 64},
                     'reference_hashes': {str(content / 'genuinely_missing.json'): 'c' * 64}})
    c = m.Collector(stage, []); c.add(protocol); c.close_references()
    missing = [i for i in c.issues if i['kind'] == 'required_reference_missing']
    assert {i['path'] for i in missing} == {str(m.CPU_GATE / 'gate_protocol.json'), str(content / 'genuinely_missing.json')}
    unknown = [i for i in c.issues if i['kind'] == 'unresolved_relative_required_reference']
    assert len(unknown) == 1 and 'unknown_or_foreign_alias:invented_alias' in unknown[0]['origin']


def test_foreign_alias_context_does_not_get_known_catalog(content):
    refs = list(m.hash_references({'cpu_gate_reference_sha256': {'protocol': 'a' * 64}},
        content / 'other-protocol.json', {}))
    assert refs[0][0] is None


def test_explicit_identity_does_not_relabel_failed_v7_or_wrong_protocol(content):
    m.verify_explicit_study_identity(m.STUDY, m.STUDY_VERSION)
    with raises(ValueError): m.verify_explicit_study_identity(m.FAILED_V7, m.STUDY_VERSION)
    with raises(ValueError): m.verify_explicit_study_identity(m.STUDY, 'v7')
    write(m.STUDY / 'study_protocol.json', {'version': 'v7', 'study_root': str(m.STUDY)})
    with raises(ValueError): m.verify_explicit_study_identity(m.STUDY, m.STUDY_VERSION)
    write(m.STUDY / 'study_protocol.json', {'version': m.STUDY_VERSION, 'study_root': str(m.STUDY)})
    m.verify_explicit_study_identity(m.STUDY, m.STUDY_VERSION)


def test_registered_v8_v9_pairs_are_explicit_and_cannot_be_crossed(content):
    v8 = content / 'cbt-qwen7b-gpu-state-study-20261112-v8'
    version8 = m.STUDY_IDENTITIES[v8.name]
    assert m.verify_explicit_study_identity(v8, version8) == (v8, version8)
    assert m.verify_explicit_study_identity(m.STUDY, m.STUDY_VERSION) == (m.STUDY, m.STUDY_VERSION)
    with raises(ValueError): m.verify_explicit_study_identity(v8, m.STUDY_VERSION)
    with raises(ValueError): m.verify_explicit_study_identity(m.STUDY, version8)
    with raises(ValueError): m.verify_explicit_study_identity(content / 'cbt-qwen7b-gpu-state-study-20261112-v10', m.STUDY_VERSION)


def test_exact_exporter_cap_branch_accepts_32mib_and_preserves_oversized_partial():
    # Execute the exporter's exact size-check AST against a tiny stat fixture;
    # no ZIP, hosted environment, large allocation or experimental data is used.
    tree = ast.parse(SOURCE.read_text())
    checks = [node for node in ast.walk(tree) if isinstance(node, ast.If)
              and isinstance(node.test, ast.Compare)
              and any(isinstance(child, ast.Name) and child.id == 'MAX_ZIP' for child in ast.walk(node.test))]
    assert len(checks) == 1 and m.MAX_ZIP == 32 * 1024 * 1024
    code = compile(ast.fix_missing_locations(ast.Module(body=[checks[0]], type_ignores=[])), str(SOURCE), 'exec')
    class TinyPartial:
        def __init__(self, size): self.size = size; self.stat_calls = 0
        def stat(self):
            self.stat_calls += 1
            return type('TinyStat', (), {'st_size': self.size})()
    for size in (0, m.MAX_ZIP - 1, m.MAX_ZIP):
        p = TinyPartial(size)
        exec(code, {'partial': p, 'MAX_ZIP': m.MAX_ZIP})
        assert p.stat_calls == 1
    p = TinyPartial(m.MAX_ZIP + 1)
    with raises(ValueError, match='private_checkpoint_exceeds_32MiB_limit_preserve_partial_no_overwrite'):
        exec(code, {'partial': p, 'MAX_ZIP': m.MAX_ZIP})
    assert p.size == m.MAX_ZIP + 1 and p.stat_calls == 1


def test_v10_source_has_only_three_authorized_changes_from_frozen_v9():
    old = SOURCE.with_name('export_gpu_v9_checkpoint_colab.py')
    assert hashlib.sha256(old.read_bytes()).hexdigest() == 'f120f9dfa877dc68f707019078833295c2eca5000d36c28b125655a384d8f7f6'
    expected = old.read_text().replace(
        "VERSION = 'private-gpu-study-byte-checkpoint-v3-explicit-v8-v9-registry'",
        "VERSION = 'private-gpu-study-byte-checkpoint-v4-32MiB-archive-cap'").replace(
        'MAX_ZIP = 10 * 1024 * 1024', 'MAX_ZIP = 32 * 1024 * 1024').replace(
        'private_checkpoint_exceeds_10MiB_limit_preserve_partial_no_overwrite',
        'private_checkpoint_exceeds_32MiB_limit_preserve_partial_no_overwrite')
    assert SOURCE.read_text() == expected


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name, fn in sorted(globals().copy().items()):
        if name.startswith('test_') and callable(fn):
            def run(fn=fn):
                if fn.__code__.co_argcount:
                    with tiny_content() as content: fn(content)
                else: fn()
            suite.addTest(unittest.FunctionTestCase(run, description=name))
    return suite


if __name__ == '__main__':
    unittest.main()
