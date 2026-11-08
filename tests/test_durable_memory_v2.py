from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
for relative in ('modules/long-term-memory/schema','modules/long-term-memory/storage','modules/long-term-memory/lifecycle'):
    sys.path.insert(0,str(ROOT/relative))

from durable_memory import AttemptUnknown, BudgetExhausted, DurableConflict, DurableLimits, DurableMemoryStore, SafetyPaused
from sqlite_store import SQLiteMemoryStore
from memory_manager import MemoryManager
from memory_models import MemoryRecord,SourceEvidence


def source(n='1',**changes):
    result={'user_id':'u','source_id':'s'+n,'input_id':'i'+n,'text':'walk twice; record paused','role':'user','source_time':'2026-11-08T10:00:00Z',
            'permissions':{'save':True,'long_term_use':True,'response_external':True,'extraction_external':True}}
    result.update(changes)
    return result


class DurableMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.db=self.root/'memory.sqlite'
        self.durable=DurableMemoryStore.open(self.db,approved_root=self.root)

    def tearDown(self):
        try:self.durable.close()
        except sqlite3.ProgrammingError:pass
        self.temp.cleanup()

    def effect(self,key='e1',object_id='task',version=0,quantity=3,source_id='s1',**kwargs):
        return self.durable.put_effect('u',key,object_id,expected_version=version,payload={'quantity':quantity},dependencies=[{'source_id':source_id}],**kwargs)

    def test_approved_explicit_file_and_memory_test_only(self):
        memory=SQLiteMemoryStore()
        with self.assertRaises(ValueError):DurableMemoryStore(memory)
        DurableMemoryStore(memory,allow_memory_for_tests=True)
        memory.close()
        with self.assertRaises(PermissionError):DurableMemoryStore.open(self.root.parent/'outside.sqlite',approved_root=self.root)
        self.assertFalse((self.root.parent/'outside.sqlite').exists())

    def test_source_and_job_atomically_rollback_together(self):
        self.durable.connection.execute("CREATE TRIGGER reject_job BEFORE INSERT ON dm_jobs BEGIN SELECT RAISE(ABORT,'fail'); END")
        with self.assertRaises(sqlite3.IntegrityError):self.durable.submit_source(source())
        self.assertEqual(self.durable.connection.execute('SELECT COUNT(*) FROM dm_sources').fetchone()[0],0)
        self.assertEqual(self.durable.connection.execute('SELECT COUNT(*) FROM dm_inputs').fetchone()[0],0)

    def test_no_save_records_only_body_free_input_receipt(self):
        s=source(text='PRIVATE NO SAVE BODY',permissions={'save':False,'response_external':True})
        receipt=self.durable.submit_source(s)
        self.assertEqual(receipt['source_status'],'not_saved')
        self.assertTrue(self.durable.submit_source(s)['replayed'])
        dump='\n'.join(self.durable.connection.iterdump())
        self.assertNotIn('PRIVATE NO SAVE BODY',dump)
        self.assertEqual(self.durable.connection.execute('SELECT COUNT(*) FROM dm_sources').fetchone()[0],0)
        self.assertEqual(self.durable.connection.execute('SELECT COUNT(*) FROM dm_jobs').fetchone()[0],0)
        self.assertIsNone(self.durable.connection.execute('SELECT fingerprint FROM dm_inputs').fetchone()[0])
        with self.assertRaises(ValueError):self.durable.record_input('u','bad',{'source_status':'not_saved'},fingerprint='bodyhash')
        with self.assertRaises(ValueError):self.durable.record_input('u','bad',{'text':'not allowed'})

    def test_unknown_permissions_are_never_an_allow(self):
        self.assertEqual(self.durable.submit_source(source(permissions={}))['source_status'],'not_saved')
        receipt=self.durable.submit_source(source('2',permissions={'save':True}))
        self.assertEqual(receipt['processing_status'],'not_queued_permission')
        self.assertIsNone(self.durable.get_source('u','s2'))
        with self.assertRaises(PermissionError):self.durable.begin_attempt('u','req','response',['s2'])

    def test_frozen_queue_limits_keep_raw_source_unprocessed(self):
        self.durable.configure_run('small',DurableLimits(max_queue_length=1))
        a=self.durable.submit_source(source(),run_id='small')
        b=self.durable.submit_source(source('2'),run_id='small')
        self.assertEqual(a['processing_status'],'queued')
        self.assertEqual(b['processing_status'],'unprocessed_queue_limit')
        self.assertIsNotNone(self.durable.get_source('u','s2'))
        with self.assertRaises(DurableConflict):self.durable.configure_run('small',DurableLimits(max_queue_length=2))

    def test_same_input_and_effect_are_idempotent_changed_identity_rejected(self):
        self.durable.submit_source(source())
        self.assertTrue(self.durable.submit_source(source())['replayed'])
        with self.assertRaises(DurableConflict):self.durable.submit_source(source(text='changed'))
        first=self.effect()
        self.assertEqual(first['version'],1)
        self.assertTrue(self.effect()['replayed'])
        with self.assertRaises(DurableConflict):self.effect(quantity=2)
        self.assertEqual(self.durable.get_object_state('u','task')['payload']['quantity'],3)

    def test_claim_token_persists_and_two_workers_only_one_claim(self):
        receipt=self.durable.submit_source(source())
        def claim(_):
            d=DurableMemoryStore.open(self.db,approved_root=self.root)
            try:return d.claim_job()
            finally:d.close()
        with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(claim,range(2)))
        selected=[r for r in results if r]
        self.assertEqual(len(selected),1)
        job=selected[0]
        self.assertEqual(job['job_id'],receipt['job_id'])
        with self.assertRaises(DurableConflict):self.durable.complete_job(job['job_id'],'wrong')
        self.assertEqual(self.durable.complete_job(job['job_id'],job['claim_token'])['status'],'completed')

    def test_compare_and_swap_two_writers_cannot_both_write_version_zero(self):
        self.durable.submit_source(source())
        def write(i):
            d=DurableMemoryStore.open(self.db,approved_root=self.root)
            try:
                d.put_effect('u','e'+str(i),'task',expected_version=0,payload={'quantity':i},dependencies=[{'source_id':'s1'}]);return True
            except DurableConflict:return False
            finally:d.close()
        with ThreadPoolExecutor(max_workers=2) as pool:result=list(pool.map(write,[1,2]))
        self.assertEqual(sum(result),1)
        self.assertEqual(self.durable.get_object_state('u','task')['version'],1)

    def test_explicit_savepoint_can_retain_independent_valid_group(self):
        self.durable.submit_source(source())
        with self.durable.store.transaction(immediate=True):
            self.effect('first','a')
            try:
                with self.durable.store.savepoint():
                    self.effect('dependent','b')
                    self.durable.put_effect('u','bad','c',expected_version=0,payload={},dependencies=[{'source_id':'missing'}])
            except DurableConflict:pass
            self.effect('last','d')
        self.assertEqual(self.durable.get_object_state('u','a')['status'],'available')
        self.assertEqual(self.durable.get_object_state('u','b')['status'],'unknown')
        self.assertEqual(self.durable.get_object_state('u','d')['status'],'available')

    def test_unknown_external_attempt_never_resends_after_reopen(self):
        self.durable.submit_source(source())
        request=self.durable.begin_attempt('u','canonical','response',['s1'])
        self.durable.close()
        self.durable=DurableMemoryStore.open(self.db,approved_root=self.root)
        with self.assertRaises(AttemptUnknown):self.durable.begin_attempt('u','canonical','response',['s1'])
        self.durable.finish_attempt(request['attempt_id'],outcome='unknown')
        with self.assertRaises(AttemptUnknown):self.durable.begin_attempt('u','canonical','response',['s1'])
        self.assertEqual(self.durable.connection.execute('SELECT COUNT(*) FROM dm_attempts').fetchone()[0],1)

    def test_no_save_response_is_publishable_once_but_not_cached(self):
        request=self.durable.begin_attempt('u','input-response','response',[])
        result=self.durable.finish_attempt(request['attempt_id'],outcome='success',response='SECRET ECHO',usage=None,retain_response=False)
        self.assertTrue(result['publishable'])
        self.assertFalse(result['response_retained'])
        replay=self.durable.begin_attempt('u','input-response','response',[])
        self.assertFalse(replay['publishable'])
        self.assertIsNone(replay['response'])
        self.assertNotIn('SECRET ECHO','\n'.join(self.durable.connection.iterdump()))
        self.assertIsNone(self.durable.connection.execute('SELECT usage_json FROM dm_attempts').fetchone()[0])
        with self.assertRaises(PermissionError):self.durable.begin_attempt('u','bad','extraction',[])

    def test_publish_rechecks_generation_even_without_persisted_source(self):
        request=self.durable.begin_attempt('u','ephemeral','response',[])
        self.durable.revoke('u','object','old-task')
        result=self.durable.finish_attempt(request['attempt_id'],outcome='success',response='STALE BODY',retain_response=False)
        self.assertFalse(result['publishable']);self.assertTrue(result['clear_payload'])
        self.assertNotIn('STALE BODY','\n'.join(self.durable.connection.iterdump()))

    def test_send_admission_refuses_withdrawn_generation_and_fences_late_result(self):
        self.durable.submit_source(source())
        request=self.durable.begin_attempt('u','before','response',['s1'],expected_generation=0)
        self.durable.revoke('u','source','s1')
        with self.assertRaises(DurableConflict):self.durable.begin_attempt('u','late','response',[],expected_generation=0)
        result=self.durable.finish_attempt(request['attempt_id'],outcome='success',response='PRIVATE RESULT',usage={'total_tokens':9})
        self.assertEqual(result['status'],'stale')
        self.assertIsNone(result['response'])
        row=self.durable.connection.execute('SELECT response_json,usage_json FROM dm_attempts').fetchone()
        self.assertIsNone(row[0]);self.assertEqual(json.loads(row[1]),{'total_tokens':9})

    def test_run_and_session_budget_count_failed_and_unknown_attempts(self):
        self.durable.configure_run('limited',DurableLimits(max_run_calls=2,max_session_calls=1))
        a=self.durable.begin_attempt('u','one','response',[],run_id='limited',session_id='a')
        self.durable.finish_attempt(a['attempt_id'],outcome='failed')
        with self.assertRaises(BudgetExhausted):self.durable.begin_attempt('u','two','response',[],run_id='limited',session_id='a')
        self.durable.begin_attempt('u','three','response',[],run_id='limited',session_id='b')
        with self.assertRaises(BudgetExhausted):self.durable.begin_attempt('u','four','response',[],run_id='limited',session_id='c')

    def test_representation_deletion_does_not_claim_content_forgotten(self):
        self.durable.submit_source(source());self.effect()
        self.durable.put_representation('u','summary','task',{'summary':'three'},dependencies=[{'source_id':'s1'}])
        receipt=self.durable.revoke('u','representation','summary')
        self.assertEqual(receipt['scope'],'representation')
        self.assertEqual(self.durable.get_object_state('u','task')['payload']['quantity'],3)
        self.assertIsNotNone(self.durable.get_source('u','s1'))
        with self.assertRaises(DurableConflict):self.durable.put_representation('u','summary','task',{},dependencies=[{'source_id':'s1'}])

    def test_two_spans_one_source_and_span_withdrawal_is_not_whole_source(self):
        self.durable.submit_source(source())
        self.durable.put_effect('u','walk','walk',expected_version=0,payload={'count':2},dependencies=[{'source_id':'s1','start':0,'end':10}])
        self.durable.put_effect('u','pause','record',expected_version=0,payload={'paused':True},dependencies=[{'source_id':'s1','start':12,'end':25}])
        receipt=self.durable.revoke('u','span','s1',start=12,end=25)
        self.assertEqual(receipt['blocked_objects'],['record'])
        self.assertEqual(self.durable.get_object_state('u','walk')['status'],'available')
        self.assertEqual(self.durable.get_object_state('u','record')['status'],'blocked')
        self.assertEqual(self.durable.get_source('u','s1')['text'][:10],'walk twice')
        self.assertNotIn('record paused',self.durable.get_source('u','s1')['text'])

    def test_three_then_two_then_delete_two_never_revives_three(self):
        self.durable.submit_source(source());self.effect(quantity=3)
        self.durable.submit_source(source('2',text='actually twice'))
        self.effect('e2',version=1,quantity=2,source_id='s2')
        self.durable.revoke('u','source','s2')
        state=self.durable.get_object_state('u','task')
        self.assertEqual(state['status'],'blocked');self.assertIsNone(state['payload'])
        with self.assertRaises(DurableConflict):self.effect('revive',version=state['version'],quantity=3)
        self.assertEqual(self.durable.connection.execute("SELECT COUNT(*) FROM dm_object_versions WHERE object_id='task' AND payload_json IS NOT NULL").fetchone()[0],0)

    def test_user_boundaries_cutoff_and_stale_effect_claim(self):
        receipt=self.durable.submit_source(source())
        self.assertEqual(self.durable.list_sources('other'),[])
        self.assertEqual(self.durable.list_sources('u',as_of='2026-11-08T09:00:00Z'),[])
        with self.assertRaises(DurableConflict):self.durable.put_effect('other','e','task',expected_version=0,payload={},dependencies=[{'source_id':'s1'}])
        job=self.durable.claim_job()
        self.durable.revoke('u','object','other-task')
        with self.assertRaises(DurableConflict):self.effect(job_id=job['job_id'],claim_token=job['claim_token'])
        with self.assertRaises(DurableConflict):self.durable.begin_attempt('u','stalejob','extraction',['s1'],job_id=job['job_id'],claim_token=job['claim_token'])

    def test_maintenance_requires_complete_latest_control_ledger(self):
        old=self.durable.control_ledger()
        self.durable.revoke('u','object','task')
        current=self.durable.begin_maintenance()
        with self.assertRaises(DurableConflict):self.durable.submit_source(source())
        with self.assertRaises(DurableConflict):self.durable.finish_maintenance(latest_control_ledger=None)
        with self.assertRaises(DurableConflict):self.durable.finish_maintenance(latest_control_ledger=old)
        self.assertEqual(self.durable.finish_maintenance(latest_control_ledger=current)['status'],'ready')
        self.durable.submit_source(source())

    def test_object_deletion_erases_all_version_spans_but_not_unrelated_same_source(self):
        first='walk twice; record three'
        second='record twice; exercise fine'
        self.durable.submit_source(source(text=first))
        self.durable.put_effect('u','e1','record',expected_version=0,payload={'count':3},dependencies=[{'source_id':'s1','start':12,'end':len(first)}])
        self.durable.submit_source(source('2',text=second))
        self.durable.put_effect('u','e2','record',expected_version=1,payload={'count':2},dependencies=[{'source_id':'s2','start':0,'end':12}])
        self.assertEqual(self.durable.connection.execute("SELECT COUNT(*) FROM dm_dependencies WHERE owner_kind='object' AND owner_id='record'").fetchone()[0],2)
        self.durable.revoke('u','object','record')
        self.assertEqual(self.durable.get_object_state('u','record')['status'],'blocked')
        self.assertEqual(self.durable.get_source('u','s1')['text'][:12],'walk twice; ')
        self.assertNotIn('record three',self.durable.get_source('u','s1')['text'])
        self.assertNotIn('record twice',self.durable.get_source('u','s2')['text'])
        self.assertTrue(self.durable.get_source('u','s2')['text'].endswith('exercise fine'))

    def test_object_stop_use_masks_use_view_without_claiming_source_deletion(self):
        self.durable.submit_source(source())
        self.durable.put_effect('u','e','record',expected_version=0,payload={'paused':True},dependencies=[{'source_id':'s1','start':12,'end':25}])
        receipt=self.durable.stop_use_object('u','record')
        self.assertEqual(receipt['control_status'],'stopped_use')
        self.assertFalse(receipt['source_payloads_erased'])
        self.assertEqual(self.durable.get_source('u','s1',require_use=False)['text'],'walk twice; record paused')
        self.assertNotIn('record paused',self.durable.get_source('u','s1')['text'])
        self.assertEqual(self.durable.get_object_state('u','record')['status'],'blocked')

    def test_metadata_queries_refresh_job_and_attempt_without_no_save_body(self):
        self.assertIsNone(self.durable.get_input('u','missing'))
        self.assertIsNone(self.durable.get_attempt('u','missing'))
        self.durable.submit_source(source())
        job=self.durable.claim_job()
        self.assertEqual(self.durable.get_input('u','i1')['processing_status'],'claimed')
        attempt=self.durable.begin_attempt('u','answer','response',['s1'])
        self.assertEqual(self.durable.get_attempt('u','answer')['status'],'sending')
        self.durable.finish_attempt(attempt['attempt_id'],outcome='success',response='not retained',retain_response=False)
        state=self.durable.get_attempt('u','answer')
        self.assertFalse(state['publishable']);self.assertIsNone(state['response'])

    def test_usage_cannot_be_alternate_no_save_body_storage(self):
        attempt=self.durable.begin_attempt('u','answer','response',[])
        with self.assertRaises(ValueError):self.durable.finish_attempt(attempt['attempt_id'],outcome='success',response='no save',usage={'total_tokens':'SECRET'},retain_response=False)
        self.assertNotIn('SECRET','\n'.join(self.durable.connection.iterdump()))
        self.durable.finish_attempt(attempt['attempt_id'],outcome='success',usage={'total_tokens':None,'completion_tokens_details':{'reasoning_tokens':3}},retain_response=False)

    def test_current_response_permission_is_separate_from_long_term_use(self):
        s=source(permissions={'save':True,'long_term_use':False,'response_external':True,'extraction_external':False})
        self.durable.submit_source(s)
        self.assertIsNone(self.durable.get_source('u','s1'))
        with self.assertRaises(PermissionError):self.durable.begin_attempt('u','historical','response',['s1'])
        attempt=self.durable.begin_attempt('u','response:i1','response',['s1'],current_source_ids=['s1'],current_input_id='i1')
        result=self.durable.finish_attempt(attempt['attempt_id'],outcome='success',response='current answer')
        self.assertTrue(result['publishable'])
        with self.assertRaises(PermissionError):self.durable.begin_attempt('u','extract','extraction',['s1'],current_source_ids=['s1'])

    def test_maintenance_blocks_old_manager_and_atomic_write_connections(self):
        legacy=SQLiteMemoryStore(self.db)
        record=MemoryRecord(schema_version='1.0',memory_id='legacy',user_id='u',memory_type='explicit_fact',content='legacy content',origin='explicit_user_statement',confirmation_state='user_confirmed',lifecycle_status='candidate',sensitivity='standard',confidence=.8,importance=.4,created_at='2026-11-08T10:00:00Z',observed_at='2026-11-08T10:00:00Z',updated_at='2026-11-08T10:00:00Z',source_evidence=SourceEvidence('session',['turn'],'legacy content'))
        ledger=self.durable.begin_maintenance()
        try:
            with self.assertRaises(DurableConflict):MemoryManager(legacy).ingest(record)
            with self.assertRaises(DurableConflict):legacy.atomic_write(records=[record])
            with self.assertRaises(DurableConflict):
                with legacy.savepoint():legacy.connection.execute("DELETE FROM memories")
            reopened=SQLiteMemoryStore(self.db)
            try:
                self.assertEqual(reopened.connection.execute("SELECT value FROM dm_meta WHERE key='maintenance'").fetchone()[0],1)
                with self.assertRaises(DurableConflict):reopened.put(record)
            finally:reopened.close()
        finally:legacy.close()
        self.durable.finish_maintenance(latest_control_ledger=ledger)
        self.durable.store.put(record)

    def test_reopen_during_maintenance_is_readonly_until_latest_ledger_checked(self):
        self.durable.revoke('u','object','task')
        ledger=self.durable.close_for_maintenance()
        self.durable=DurableMemoryStore.open(self.db,approved_root=self.root)
        with self.assertRaises(DurableConflict):self.durable.submit_source(source())
        self.durable.finish_maintenance(latest_control_ledger=ledger)
        self.durable.submit_source(source())

    def test_explicit_restore_refuses_missing_or_stale_external_ledger(self):
        old=self.durable.control_ledger()
        self.durable.revoke('u','object','task')
        current=self.durable.control_ledger()
        with self.assertRaises(DurableConflict):DurableMemoryStore.open_restored(self.db,approved_root=self.root,latest_control_ledger=None)
        with self.assertRaises(DurableConflict):DurableMemoryStore.open_restored(self.db,approved_root=self.root,latest_control_ledger=old)
        restored=DurableMemoryStore.open_restored(self.db,approved_root=self.root,latest_control_ledger=current)
        try:self.assertEqual(restored.get_object_state('u','task')['status'],'blocked')
        finally:restored.close()

    def test_keyboard_interrupt_rolls_back_the_entire_outer_transaction(self):
        with self.assertRaises(KeyboardInterrupt):
            with self.durable.store.transaction(immediate=True):
                self.durable.submit_source(source())
                raise KeyboardInterrupt()
        self.assertFalse(self.durable.connection.in_transaction)
        self.assertIsNone(self.durable.get_input('u','i1'))
        self.assertIsNone(self.durable.get_source('u','s1'))

    def test_job_cannot_switch_run_or_session_to_escape_frozen_budget(self):
        self.durable.submit_source(source(),run_id='frozen',session_id='a')
        job=self.durable.claim_job()
        for run,session in [('other','a'),('frozen','b')]:
            with self.assertRaises(DurableConflict):
                self.durable.begin_attempt('u','attempt-'+run+session,'extraction',['s1'],run_id=run,session_id=session,job_id=job['job_id'],claim_token=job['claim_token'])
        self.assertEqual(self.durable.connection.execute('SELECT COUNT(*) FROM dm_attempts').fetchone()[0],0)

    def test_current_exception_requires_exact_input_run_session_and_no_withdrawal(self):
        self.durable.submit_source(source())
        for kwargs in [dict(),{'current_input_id':'other'}, {'current_input_id':'i1','session_id':'other'}, {'current_input_id':'i1','run_id':'other'}]:
            with self.assertRaises(PermissionError):
                self.durable.begin_attempt('u','bad','response',['s1'],current_source_ids=['s1'],**kwargs)
        self.durable.revoke('u','stop_use','s1')
        with self.assertRaises(PermissionError):
            self.durable.begin_attempt('u','response:i1','response',['s1'],current_source_ids=['s1'],current_input_id='i1')

    def test_source_free_default_cannot_cache_no_save_echo(self):
        self.durable.record_input('u','secret-input',{'source_status':'not_saved','control_status':'none'})
        attempt=self.durable.begin_attempt('u','response:secret-input','response',[],current_input_id='secret-input')
        result=self.durable.finish_attempt(attempt['attempt_id'],outcome='success',response='SECRET UNSAVED ECHO')
        self.assertTrue(result['publishable']);self.assertFalse(result['response_retained'])
        self.assertNotIn('SECRET UNSAVED ECHO','\n'.join(self.durable.connection.iterdump()))

    def test_no_save_current_input_also_blocks_cache_when_history_is_present(self):
        self.durable.submit_source(source())
        self.durable.record_input('u','secret-input',{'source_status':'not_saved','control_status':'none'})
        attempt=self.durable.begin_attempt('u','response:secret-input','response',['s1'],current_input_id='secret-input')
        result=self.durable.finish_attempt(attempt['attempt_id'],outcome='success',response='SECRET WITH HISTORY')
        self.assertFalse(result['response_retained'])
        self.assertNotIn('SECRET WITH HISTORY','\n'.join(self.durable.connection.iterdump()))

    def test_old_database_and_sidecar_cannot_rollback_independent_anchor(self):
        self.durable.submit_source(source(text='OLD PRIVATE BODY'))
        snapshot=self.root/'old.sqlite'
        backup=sqlite3.connect(snapshot)
        self.durable.connection.backup(backup);backup.close()
        binding=Path(str(self.db)+'.control.json')
        old_binding=binding.read_bytes()
        self.durable.revoke('u','source','s1')
        self.durable.close()
        shutil.copyfile(snapshot,self.db)
        binding.write_bytes(old_binding)
        with self.assertRaises(DurableConflict):DurableMemoryStore.open(self.db,approved_root=self.root)
        with self.assertRaises(DurableConflict):DurableMemoryStore(SQLiteMemoryStore(self.db),approved_root=self.root)

    def test_existing_database_cannot_recreate_missing_control_files(self):
        self.durable.submit_source(source())
        self.durable.close()
        Path(str(self.db)+'.control.json').unlink()
        with self.assertRaises(DurableConflict):DurableMemoryStore.open(self.db,approved_root=self.root)

    def test_copied_database_cannot_adopt_new_location_without_latest_identity(self):
        clone=self.root/'clone.sqlite'
        target=sqlite3.connect(clone)
        self.durable.connection.backup(target);target.close()
        shutil.copyfile(Path(str(self.db)+'.control.json'),Path(str(clone)+'.control.json'))
        with self.assertRaises(DurableConflict):DurableMemoryStore.open(clone,approved_root=self.root)

    def test_anchor_write_failure_fences_reads_sends_and_reopen(self):
        self.durable.submit_source(source(text='BODY THAT MUST NOT RETURN AFTER FAILED CONTROL'))
        real=self.durable._atomic_control_file
        def fail_binding(path,payload):
            if path==self.durable._binding_path:raise OSError('simulated binding storage failure')
            real(path,payload)
        with patch.object(self.durable,'_atomic_control_file',side_effect=fail_binding):
            with self.assertRaises(DurableConflict):self.durable.revoke('u','source','s1')
        self.assertEqual(self.durable.connection.execute("SELECT value FROM dm_meta WHERE key='maintenance'").fetchone()[0],1)
        with self.assertRaises(DurableConflict):self.durable.get_source('u','s1')
        with self.assertRaises(DurableConflict):self.durable.begin_attempt('u','blocked','response',['s1'])
        self.durable.close()
        with self.assertRaises(DurableConflict):DurableMemoryStore.open(self.db,approved_root=self.root)

    def test_worker_run_filter_and_persisted_batch_limits(self):
        limits=DurableLimits(max_batch_size=1,max_queue_length=2)
        self.durable.configure_run('one',limits)
        a=self.durable.submit_source(source(),run_id='one')
        b=self.durable.submit_source(source('2'),run_id='two')
        job=self.durable.claim_job(user_id='u',run_id='two')
        self.assertEqual(job['job_id'],b['job_id'])
        self.assertEqual(self.durable.get_input('u','i1')['processing_status'],'queued')
        self.assertEqual(self.durable.limits_for_run('one')['max_batch_size'],1)
        with self.assertRaises(DurableConflict):self.durable.limits_for_run('missing')

    def test_pre_send_backup_cannot_erase_unknown_paid_attempt_without_any_delete(self):
        self.durable.submit_source(source())
        snapshot=self.root/'before-send.sqlite'
        target=sqlite3.connect(snapshot)
        self.durable.connection.backup(target);target.close()
        binding=Path(str(self.db)+'.control.json')
        old_binding=binding.read_bytes()
        self.durable.begin_attempt('u','charged-unknown','response',['s1'])
        self.durable.close()
        shutil.copyfile(snapshot,self.db);binding.write_bytes(old_binding)
        with self.assertRaises(DurableConflict):DurableMemoryStore.open(self.db,approved_root=self.root)

    def test_control_namespace_cannot_symlink_outside_approved_root(self):
        with tempfile.TemporaryDirectory() as approved:
            with tempfile.TemporaryDirectory() as other:
                root=Path(approved);outside=Path(other)
                (root/'.cbt-memory-control').symlink_to(outside,target_is_directory=True)
                with self.assertRaises(PermissionError):DurableMemoryStore.open(root/'new.sqlite',approved_root=root)
                self.assertEqual(list(outside.iterdir()),[])

    def test_registered_store_refuses_legacy_writer_without_watermark_hook(self):
        record=MemoryRecord(schema_version='1.0',memory_id='legacy',user_id='u',memory_type='explicit_fact',content='legacy body',origin='explicit_user_statement',confirmation_state='user_confirmed',lifecycle_status='active',sensitivity='standard',confidence=.8,importance=.4,created_at='2026-11-08T10:00:00Z',observed_at='2026-11-08T10:00:00Z',updated_at='2026-11-08T10:00:00Z',source_evidence=SourceEvidence('session',['turn'],'legacy body'))
        self.durable.store.put(record)
        legacy=SQLiteMemoryStore(self.db)
        try:
            with self.assertRaises(DurableConflict):legacy.hard_delete('u','legacy')
            with self.assertRaises(DurableConflict):legacy.put(record)
            with self.assertRaises(DurableConflict):legacy.get('u','legacy')
            with self.assertRaises(DurableConflict):legacy.list_for_user('u')
            self.assertIsNotNone(self.durable.store.get('u','legacy'))
        finally:legacy.close()
        self.durable.store.hard_delete('u','legacy')
        self.assertIsNone(self.durable.store.get('u','legacy'))

    def test_safety_pause_is_persistent_across_already_open_sessions_and_runs(self):
        self.durable.submit_source(source())
        other=DurableMemoryStore.open(self.db,approved_root=self.root)
        try:
            self.assertFalse(other.is_paused('u'))
            first=self.durable.pause_user('u')
            self.assertTrue(other.is_paused('u'))
            self.assertTrue(other.pause_user('u')['replayed'])
            self.assertEqual(other.current_generation('u'),first['generation'])
            with self.assertRaises(SafetyPaused):other.submit_source(source('2'),run_id='new-run',session_id='new-session')
            with self.assertRaises(SafetyPaused):other.begin_attempt('u','new-response','response',[],run_id='other-run',session_id='other-session')
            with self.assertRaises(SafetyPaused):other.submit_source(source('3',permissions={'save':False}))
            self.assertIsNone(other.claim_job(user_id='u'))
            self.assertIsNone(other.get_source('u','s1'))
            self.assertEqual(other.get_object_state('u','missing')['reason'],'safety_paused')
            self.assertEqual(other.connection.execute('SELECT COUNT(*) FROM dm_sources WHERE user_id=?',('u',)).fetchone()[0],1)
            self.assertEqual(other.connection.execute('SELECT COUNT(*) FROM dm_attempts WHERE user_id=?',('u',)).fetchone()[0],0)
        finally:other.close()
        self.durable.close()
        self.durable=DurableMemoryStore.open(self.db,approved_root=self.root)
        self.assertTrue(self.durable.is_paused('u'))

    def test_safety_pause_fences_already_admitted_body_even_with_no_source(self):
        request=self.durable.begin_attempt('u','in-flight','response',[])
        other=DurableMemoryStore.open(self.db,approved_root=self.root)
        try:
            other.pause_user('u')
            result=self.durable.finish_attempt(request['attempt_id'],outcome='success',response='STALE SAFETY BODY')
            self.assertEqual(result['status'],'stale')
            self.assertFalse(result['publishable']);self.assertTrue(result['clear_payload'])
            self.assertNotIn('STALE SAFETY BODY','\n'.join(self.durable.connection.iterdump()))
        finally:other.close()

    def test_safety_pause_filters_worker_without_blocking_another_user(self):
        self.durable.submit_source(source())
        receipt=self.durable.submit_source(source('2',user_id='other'))
        self.durable.pause_user('u')
        job=self.durable.claim_job()
        self.assertEqual(job['user_id'],'other')
        self.assertEqual(job['job_id'],receipt['job_id'])
        self.assertFalse(self.durable.is_paused('other'))

    def test_safety_pause_keeps_explicit_local_deletion_and_body_free_receipts_available(self):
        self.durable.submit_source(source())
        self.effect()
        self.durable.pause_user('u')
        self.assertIsNotNone(self.durable.get_source('u','s1',require_use=False))
        deletion=self.durable.revoke('u','source','s1')
        self.assertTrue(deletion['logical_blocked'])
        self.assertIsNone(self.durable.get_source('u','s1',require_use=False))
        self.assertTrue(self.durable.is_paused('u'))
        self.durable.record_input('u','local-delete',{'source_status':'not_saved','control_status':'local_deleted'})
        self.assertEqual(self.durable.get_input('u','local-delete')['control_status'],'local_deleted')
        with self.assertRaises(SafetyPaused):self.durable.begin_attempt('u','still-paused','response',[])

    def test_safety_pause_blocks_claimed_worker_send_and_projection(self):
        self.durable.submit_source(source())
        job=self.durable.claim_job(user_id='u')
        self.durable.pause_user('u')
        with self.assertRaises(SafetyPaused):
            self.durable.begin_attempt('u','worker','extraction',['s1'],job_id=job['job_id'],claim_token=job['claim_token'])
        with self.assertRaises(SafetyPaused):self.effect(job_id=job['job_id'],claim_token=job['claim_token'])
        self.assertEqual(self.durable.complete_job(job['job_id'],job['claim_token'])['status'],'safety_paused')


if __name__=='__main__':unittest.main()
