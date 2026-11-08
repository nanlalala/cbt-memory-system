"""Offline regressions for review F01/F03 and object-version send fences."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import sqlite3
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for relative in ('modules/long-term-memory/schema', 'modules/long-term-memory/storage',
                 'modules/long-term-memory/lifecycle'):
    sys.path.insert(0, str(ROOT / relative))
from durable_memory import DurableMemoryStore, DurableConflict, AttemptUnknown
from memory_models import MemoryRecord, SourceEvidence
from sqlite_store import SQLiteMemoryStore


def source(sid='source', user='u', text='walk twice; record paused'):
    return dict(user_id=user, source_id=sid, input_id='input-' + sid, text=text,
                role='user', source_time='2026-11-08T10:00:00Z',
                permissions={k: True for k in ('save', 'long_term_use', 'response_external', 'extraction_external')})


class DurableReviewFences(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.db = self.root / 'memory.sqlite'
        self.d = DurableMemoryStore.open(self.db, approved_root=self.root)

    def tearDown(self):
        self.d.close()

    def reopen(self):
        self.d.close()
        self.d = DurableMemoryStore.open(self.db, approved_root=self.root)

    def candidate(self, mid='m', sid='source', user='u', start=0, end=None):
        src = self.d.get_source(user, sid)
        end = len(src['text']) if end is None else end
        quote = src['text'][start:end]
        record = MemoryRecord('1.0', mid, user, 'explicit_fact', quote,
            'explicit_user_statement', 'unconfirmed', 'candidate', 'standard', .5, .5,
            src['source_time'], src['source_time'], src['source_time'],
            SourceEvidence('session', [sid], quote), tags=['controlled_v2'])
        with self.d.store.transaction(immediate=True):
            self.d.put_effect(user, 'effect-' + mid, 'memory:' + mid, expected_version=0,
                payload={'record': record.to_dict()}, dependencies=[{'source_id': sid, 'start': start, 'end': end}])
            self.d.store.put(record)
        return record

    def test_source_deletion_atomically_erases_manager_body_and_support_after_reopen(self):
        self.d.submit_source(source(text='PRIVATE controlled quotation'))
        self.candidate()
        receipt = self.d.revoke('u', 'source', 'source')
        self.assertTrue(receipt['primary_payloads_cleared'])
        self.assertIsNone(self.d.store.get('u', 'm'))
        self.assertEqual(self.d.store.list_for_user('u'), [])
        self.assertIsNone(self.d.connection.execute("SELECT record_json FROM memories WHERE memory_id='m'").fetchone())
        self.reopen()
        self.assertEqual(self.d.store.list_for_user('u'), [])
        self.assertNotIn('PRIVATE controlled quotation', '\n'.join(self.d.connection.iterdump()))

    def test_span_deletion_preserves_unrelated_span_and_other_user(self):
        self.d.submit_source(source())
        self.d.submit_source(source(user='other'))
        self.candidate('walk', start=0, end=10)
        self.candidate('record', start=12)
        self.candidate('other-memory', user='other')
        self.d.revoke('u', 'span', 'source', start=12, end=25)
        self.reopen()
        self.assertIsNone(self.d.store.get('u', 'record'))
        self.assertEqual(self.d.store.get('u', 'walk').content, 'walk twice')
        self.assertIsNotNone(self.d.store.get('other', 'other-memory'))

    def test_object_deletion_also_erases_its_manager_copy(self):
        self.d.submit_source(source())
        self.candidate()
        self.d.revoke('u', 'object', 'memory:m')
        self.reopen()
        self.assertIsNone(self.d.store.get('u', 'm'))
        self.assertIsNone(self.d.connection.execute("SELECT content FROM memories WHERE memory_id='m'").fetchone())

    def test_read_fence_hides_preexisting_manager_copy_at_persistent_tombstone(self):
        self.d.submit_source(source())
        record = self.candidate()
        self.d.revoke('u', 'source', 'source')
        # Reproduce an old-version leftover, without using public guarded writes.
        with self.d.store.transaction(immediate=True):
            self.d.connection.execute('INSERT INTO memories VALUES (?,?,?,?,?,?,?,?)',
                ('m', 'u', record.memory_type, record.content, record.lifecycle_status,
                 record.sensitivity, record.updated_at, json.dumps(record.to_dict())))
        self.reopen()
        self.assertIsNone(self.d.store.get('u', 'm'))
        self.assertEqual(self.d.store.list_for_user('u'), [])

    def test_cleanup_failure_rolls_back_control_and_manager_deletion_together(self):
        self.d.submit_source(source())
        self.candidate()
        self.d.connection.execute("CREATE TRIGGER reject_manager_cleanup BEFORE DELETE ON memories BEGIN SELECT RAISE(ABORT,'injected cleanup failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.d.revoke('u', 'source', 'source')
        self.reopen()
        self.assertIsNotNone(self.d.get_source('u', 'source'))
        self.assertIsNotNone(self.d.store.get('u', 'm'))
        self.assertEqual(self.d.control_ledger()['watermark'], 0)

    def test_unknown_source_tombstones_block_future_submit_across_reopen(self):
        for scope in ('source', 'stop_use'):
            sid = 'unknown-' + scope
            self.assertTrue(self.d.revoke('u', scope, sid)['logical_blocked'])
            self.reopen()
            with self.assertRaises(DurableConflict):
                self.d.submit_source(source(sid=sid))
            self.assertIsNone(self.d.get_source('u', sid, require_use=False))
            self.d.submit_source(source(sid=sid, user='other'))
            self.assertIsNotNone(self.d.get_source('other', sid))

    def test_already_open_other_connection_obeys_unknown_source_tombstone(self):
        other = DurableMemoryStore.open(self.db, approved_root=self.root)
        try:
            self.d.revoke('u', 'source', 'later')
            with self.assertRaises(DurableConflict):
                other.submit_source(source(sid='later'))
            self.assertEqual(other.list_sources('u'), [])
            self.assertIsNone(other.claim_job(user_id='u'))
            with self.assertRaises(PermissionError):
                other.begin_attempt('u', 'bad-admission', 'response', ['later'])
        finally:
            other.close()

    def test_stop_use_read_fence_retains_raw_source_but_hides_manager_copy(self):
        self.d.submit_source(source())
        self.candidate()
        result = self.d.revoke('u', 'stop_use', 'source')
        self.assertFalse(result['source_payloads_erased'])
        self.assertIsNone(self.d.get_source('u', 'source'))
        self.assertEqual(self.d.get_source('u', 'source', require_use=False)['text'], 'walk twice; record paused')
        self.assertIsNone(self.d.store.get('u', 'm'))

    def task(self, version=0, key='task-first', quantity=3, object_id='task'):
        return self.d.put_effect('u', key, object_id, expected_version=version,
            payload={'quantity': quantity}, dependencies=[{'source_id': 'source'}])

    def test_task_change_before_send_admission_refuses_stale_object_version(self):
        self.d.submit_source(source()); self.task()
        self.task(version=1, key='correction', quantity=2)
        with self.assertRaises(DurableConflict):
            self.d.begin_attempt('u', 'answer', 'response', ['source'], expected_object_versions={'task': 1})
        self.assertEqual(self.d.connection.execute('SELECT COUNT(*) FROM dm_attempts').fetchone()[0], 0)

    def test_task_change_after_send_discards_body_and_cache_without_recall(self):
        self.d.submit_source(source()); self.task()
        a = self.d.begin_attempt('u', 'answer', 'response', ['source'], expected_object_versions={'task': 1})
        self.reopen()
        self.task(version=1, key='correction', quantity=2)
        result = self.d.finish_attempt(a['attempt_id'], outcome='success', response='old target three', usage={'total_tokens': 7})
        self.assertEqual(result['status'], 'stale')
        self.assertFalse(result['publishable'])
        self.assertIsNone(self.d.get_attempt('u', 'answer')['response'])
        self.assertNotIn('old target three', '\n'.join(self.d.connection.iterdump()))
        self.assertEqual(self.d.connection.execute('SELECT COUNT(*) FROM dm_attempts').fetchone()[0], 1)

    def test_unrelated_object_change_does_not_invalidate_bound_task(self):
        self.d.submit_source(source()); self.task()
        a = self.d.begin_attempt('u', 'answer', 'response', ['source'], expected_object_versions={'task': 1})
        self.task(key='different', object_id='other-task')
        result = self.d.finish_attempt(a['attempt_id'], outcome='success', response='target three')
        self.assertTrue(result['publishable'])

    def test_task_change_after_success_hides_successful_cached_body(self):
        self.d.submit_source(source()); self.task()
        a = self.d.begin_attempt('u', 'response:input-source', 'response', ['source'],
            current_input_id='input-source', expected_object_versions={'task': 1})
        self.d.finish_attempt(a['attempt_id'], outcome='success', response='target three')
        self.assertEqual(self.d.get_attempt('u', 'response:input-source')['response'], 'target three')
        self.task(version=1, key='correction', quantity=2)
        self.assertFalse(self.d.get_attempt('u', 'response:input-source')['publishable'])
        self.assertIsNone(self.d.get_attempt('u', 'response:input-source')['response'])
        self.assertIsNone(self.d.connection.execute('SELECT response_json FROM dm_attempts WHERE attempt_id=?',
                                                    (a['attempt_id'],)).fetchone()[0])

    def test_cache_current_input_exception_requires_original_run_and_session(self):
        s = source(); s['permissions']['long_term_use'] = False
        self.d.submit_source(s, run_id='run', session_id='first')
        a = self.d.begin_attempt('u', 'response:input-source', 'response', ['source'],
            run_id='run', session_id='first', current_source_ids=['source'], current_input_id='input-source')
        self.d.finish_attempt(a['attempt_id'], outcome='success', response={'text': 'session-only echo'})
        self.assertIsNone(self.d.get_attempt('u', 'response:input-source')['response'])
        self.assertEqual(self.d.get_attempt('u', 'response:input-source', run_id='run', session_id='first')['response'],
                         {'text': 'session-only echo'})
        self.assertIsNone(self.d.get_attempt('u', 'response:input-source', run_id='run', session_id='new')['response'])
        self.reopen()
        self.assertIsNone(self.d.get_attempt('u', 'response:input-source', run_id='run', session_id='new')['response'])
        self.assertEqual(self.d.connection.execute('SELECT COUNT(*) FROM dm_attempts').fetchone()[0], 1)

    def test_authenticated_previous_schema_adds_object_fence_without_reinterpreting_attempt(self):
        a = self.d.begin_attempt('u', 'old-unknown', 'response', [])
        with self.d.store.transaction(immediate=True):
            self.d.connection.execute('ALTER TABLE dm_attempts DROP COLUMN object_versions_json')
        self.reopen()
        current = self.d.get_attempt('u', 'old-unknown')
        self.assertEqual(current['attempt_id'], a['attempt_id'])
        self.assertEqual(current['status'], 'dependency_unknown')
        self.assertEqual(current['object_versions'], {})
        self.assertEqual(self.d.connection.execute('SELECT COUNT(*) FROM dm_attempts').fetchone()[0], 1)

    def legacy_checkpoint(self, *, open_current=True):
        """Rehydrate bytes generated by the actual c757 Durable + SQLite classes.

        Path rebinding belongs only to this synthetic fixture construction. No
        production migration/restore operation is bypassed or represented here.
        The fixture is committed so shallow CI clones need no historical git.
        """
        fixture = json.loads((ROOT / 'tests/fixtures/durable_c757_checkpoint.json').read_text())
        self.assertEqual(fixture['generated_from_commit'], 'c757b0d8dca89b39e996fa16c9c40bfc99ec0d1a')
        self.d.close()
        root = self.root / 'legacy';root.mkdir()
        db = root / 'memory.sqlite'
        connection = sqlite3.connect(db)
        connection.executescript(fixture['sql_dump'].replace('__FIXTURE_DATABASE_PATH__', str(db)))
        self.assertNotIn('object_versions_json', {r[1] for r in connection.execute('PRAGMA table_info(dm_attempts)')})
        before = {row[0]: row[1:] for row in connection.execute(
            'SELECT request_key,attempt_id,status,response_json,usage_json FROM dm_attempts')}
        self.assertIn('legacy target three', before['response:input-source'][2])
        connection.close()
        checkpoint = {**fixture['control_checkpoint'], 'database_path': str(db)}
        (Path(str(db) + '.control.json')).write_text(json.dumps(checkpoint))
        anchors = root / '.cbt-memory-control';anchors.mkdir()
        (anchors / (checkpoint['storage_id'] + '.json')).write_text(json.dumps(checkpoint))
        if open_current:
            self.d = DurableMemoryStore.open(db, approved_root=root)
        return before

    def test_real_old_checkpoint_caches_and_inflight_results_fail_closed_without_losing_ledger(self):
        before = self.legacy_checkpoint()
        self.assertEqual(self.d.connection.execute('SELECT COUNT(*) FROM dm_attempts').fetchone()[0], 3)
        self.assertEqual(self.d.get_input('u', 'input-source')['source_status'], 'committed')
        success = self.d.get_attempt('u', 'response:input-source', run_id='default', session_id='default')
        self.assertEqual(success['attempt_id'], before['response:input-source'][0])
        self.assertEqual(success['status'], 'stale')
        self.assertFalse(success['publishable']);self.assertIsNone(success['response'])
        self.assertEqual(success['usage'], {'total_tokens': 11})
        self.task(version=1, key='correction', quantity=2)
        self.assertIsNone(self.d.get_attempt('u', 'response:input-source',
            run_id='default', session_id='default')['response'])
        for key, usage in [('legacy-inflight', {'total_tokens': 13}), ('legacy-unknown', None)]:
            old = self.d.get_attempt('u', key)
            self.assertEqual(old['status'], 'dependency_unknown')
            self.assertEqual(old['attempt_id'], before[key][0])
            with self.assertRaises(AttemptUnknown):
                self.d.begin_attempt('u', key, 'response', ['source'])
            result = self.d.finish_attempt(old['attempt_id'], outcome='success',
                response='legacy late target three', usage=usage)
            self.assertEqual(result['status'], 'stale')
            self.assertFalse(result['publishable']);self.assertIsNone(result['response'])
        self.assertEqual(self.d.get_attempt('u', 'legacy-inflight')['usage'], {'total_tokens': 13})
        self.assertEqual(self.d.get_attempt('u', 'legacy-unknown')['usage'], {'total_tokens': 4})
        self.assertNotIn('legacy target three', '\n'.join(self.d.connection.iterdump()))
        self.assertNotIn('legacy late target three', '\n'.join(self.d.connection.iterdump()))
        self.assertEqual(self.d.connection.execute('SELECT COUNT(*) FROM dm_attempts').fetchone()[0], 3)

    def test_explicit_empty_object_dependencies_on_new_attempt_are_not_legacy_unknown(self):
        self.legacy_checkpoint()
        self.d.submit_source(source(sid='fresh', text='unrelated permitted source'))
        a = self.d.begin_attempt('u', 'response:input-fresh', 'response', ['fresh'],
            current_input_id='input-fresh', expected_object_versions={})
        finished = self.d.finish_attempt(a['attempt_id'], outcome='success', response='fresh permitted answer')
        self.assertTrue(finished['publishable'])
        self.task(version=1, key='correction', quantity=2)
        current = self.d.get_attempt('u', 'response:input-fresh', run_id='default', session_id='default')
        self.assertTrue(current['publishable']);self.assertEqual(current['response'], 'fresh permitted answer')
        self.assertEqual(self.d.connection.execute('SELECT COUNT(*) FROM dm_attempts').fetchone()[0], 4)

    def test_identity_open_waits_for_atomic_database_checkpoint_commit(self):
        anchor_written = threading.Event();release_commit = threading.Event()
        reader_started = threading.Event();reader_done = threading.Event()

        def writer():
            d = DurableMemoryStore.open(self.db, approved_root=self.root)
            original = d._persist_control_anchor
            def hold_checkpoint():
                original()
                anchor_written.set()
                if not release_commit.wait(5):
                    raise AssertionError('test did not release checkpoint commit')
            try:
                with patch.object(d, '_persist_control_anchor', side_effect=hold_checkpoint):
                    return d.submit_source(source(sid='racing'))
            finally:d.close()

        def reader():
            reader_started.set()
            try:
                d = DurableMemoryStore.open(self.db, approved_root=self.root)
                try:return d.get_source('u', 'racing')
                finally:d.close()
            finally:reader_done.set()

        with ThreadPoolExecutor(max_workers=2) as pool:
            writing = pool.submit(writer)
            self.assertTrue(anchor_written.wait(5))
            reading = pool.submit(reader)
            self.assertTrue(reader_started.wait(5))
            try:
                self.assertFalse(reader_done.wait(.1), 'constructor must not compare checkpoint during an uncommitted write')
            finally:release_commit.set()
            self.assertEqual(writing.result(timeout=5)['source_status'], 'committed')
            self.assertEqual(reading.result(timeout=5)['source_id'], 'racing')

    def test_already_open_control_reader_waits_for_checkpoint_commit_without_retry(self):
        reader_ready = threading.Event();start_read = threading.Event()
        anchor_written = threading.Event();release_commit = threading.Event();reader_done = threading.Event()

        def reader():
            d = DurableMemoryStore.open(self.db, approved_root=self.root)
            reader_ready.set()
            try:
                if not start_read.wait(5):raise AssertionError('test did not admit read')
                return d.get_source('u', 'racing')
            finally:
                reader_done.set();d.close()

        def writer():
            d = DurableMemoryStore.open(self.db, approved_root=self.root)
            original = d._persist_control_anchor
            def hold_checkpoint():
                original();anchor_written.set()
                if not release_commit.wait(5):raise AssertionError('test did not release commit')
            try:
                with patch.object(d, '_persist_control_anchor', side_effect=hold_checkpoint):
                    return d.submit_source(source(sid='racing'))
            finally:d.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            reading = pool.submit(reader);self.assertTrue(reader_ready.wait(5))
            writing = pool.submit(writer);self.assertTrue(anchor_written.wait(5));start_read.set()
            try:
                self.assertFalse(reader_done.wait(.1), 'read verification must wait for a consistent checkpoint')
            finally:release_commit.set()
            self.assertEqual(writing.result(timeout=5)['source_status'], 'committed')
            self.assertEqual(reading.result(timeout=5)['source_id'], 'racing')

    def test_concurrent_old_schema_open_performs_exactly_one_cache_fencing_migration(self):
        self.legacy_checkpoint(open_current=False)
        root = self.root / 'legacy';db = root / 'memory.sqlite'
        before_revision = json.loads((Path(str(db) + '.control.json')).read_text())['storage_revision']
        both_old = threading.Barrier(2);local = threading.local()
        original = SQLiteMemoryStore.transaction

        @contextmanager
        def align_prelock_schema_check(store, *, immediate=False, _maintenance=False):
            columns = {r[1] for r in store.connection.execute('PRAGMA table_info(dm_attempts)')}
            if (not _maintenance and store._before_commit is not None and
                    'object_versions_json' not in columns and not getattr(local, 'aligned', False)):
                local.aligned = True
                both_old.wait(timeout=5)
            with original(store, immediate=immediate, _maintenance=_maintenance) as connection:
                yield connection

        def open_old():
            d = DurableMemoryStore.open(db, approved_root=root)
            try:
                return {r['request_key']: (r['status'], r['response_json']) for r in
                        d.connection.execute('SELECT request_key,status,response_json FROM dm_attempts')}
            finally:d.close()

        with patch.object(SQLiteMemoryStore, 'transaction', align_prelock_schema_check):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(open_old) for _ in range(2)]
                results = [f.result(timeout=10) for f in futures]
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0]['response:input-source'], ('stale', None))
        self.assertEqual(results[0]['legacy-inflight'], ('dependency_unknown', None))
        self.d = DurableMemoryStore.open(db, approved_root=root)
        self.assertEqual(self.d.connection.execute("SELECT value FROM dm_meta WHERE key='storage_revision'").fetchone()[0],
                         before_revision + 1)
        self.assertEqual(self.d.connection.execute('SELECT COUNT(*) FROM dm_attempts').fetchone()[0], 3)

    def test_read_checkpoint_verification_does_not_write_revision_or_anchor(self):
        self.d.submit_source(source())
        before_revision = self.d.connection.execute("SELECT value FROM dm_meta WHERE key='storage_revision'").fetchone()[0]
        before_binding = self.d._binding_path.read_bytes();before_anchor = self.d._anchor_path.read_bytes()
        with patch.object(self.d.store, '_before_commit', side_effect=AssertionError('read invoked a write hook')):
            for _ in range(3):
                self.d._verify_control_anchor()
                self.assertIsNotNone(self.d.get_source('u', 'source'))
                self.assertEqual(len(self.d.list_sources('u')), 1)
                self.assertEqual(self.d.store.list_for_user('u'), [])
        self.assertEqual(self.d.connection.execute("SELECT value FROM dm_meta WHERE key='storage_revision'").fetchone()[0], before_revision)
        self.assertEqual(self.d._binding_path.read_bytes(), before_binding)
        self.assertEqual(self.d._anchor_path.read_bytes(), before_anchor)


if __name__ == '__main__':
    unittest.main()
