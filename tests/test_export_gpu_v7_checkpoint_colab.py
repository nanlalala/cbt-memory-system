"""Tiny invented metadata only; no hosted session, model or dataset execution."""
import base64
import hashlib
import importlib.util
import json
from contextlib import contextmanager
import tempfile
import unittest
from pathlib import Path
import sqlite3
from unittest import mock



SOURCE = Path(__file__).parents[1] / 'evaluation/cbt-memory-effect/export_gpu_v7_checkpoint_colab.py'
spec = importlib.util.spec_from_file_location('gpu_checkpoint_fixture', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@contextmanager
def tiny_content():
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        replacements = {}
        for name in ('CONTENT', 'GATE', 'STUDY', 'CODE', 'MODEL', 'PYTHON', 'RECOVERY8', 'RECOVERY9',
                 'READINESS', 'NVIDIA_META', 'PUBLIC_META', 'CPU_CLOSURE'):
            value = tmp_path if name == 'CONTENT' else tmp_path / name.lower()
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
