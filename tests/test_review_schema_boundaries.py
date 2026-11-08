"""Review F13/F16 regressions: fail before effects, not after model admission."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
for relative in ('modules/long-term-memory/schema', 'modules/long-term-memory/storage',
                 'modules/long-term-memory/lifecycle', 'modules/agent-integration'):
    sys.path.insert(0, str(ROOT / relative))

from memory_contract import MemoryRecordV2, construct_memory
from memory_models import SourceEvidence
from memory_manager import MemoryManager
from sqlite_store import SQLiteMemoryStore
from natural_memory_policy import moment
from controlled_dialogue import ControlledDialogueSession
from dialogue_service import ModelOutput
from durable_memory import DurableMemoryStore

GOOD = '2026-11-04T10:00:00+00:00'
BAD = ('2026-11-04 10:00:00+00:00', '2026-11-04T10:00+00:00',
       '2026-11-04T10:00:00', '', False, 1)
PERMISSIONS = dict(save=True, long_term_use=True, response_external=True,
                   extraction_external=True)


def record_data(memory_id='first'):
    return dict(schema_version='2.0', memory_id=memory_id, user_id='u',
        memory_type='explicit_fact', content='I did the thought record.',
        origin='explicit_user_statement', confirmation_state='unconfirmed',
        lifecycle_status='candidate', sensitivity='standard', confidence=.5,
        importance=.5, created_at=GOOD, observed_at=GOOD, updated_at=GOOD,
        source_evidence=dict(session_id='session', turn_ids=['source'],
            supporting_text='I did the thought record.'), source_role='user',
        source_time=GOOD, permissions=dict(PERMISSIONS))


class CountingProvider:
    def __init__(self): self.calls = []
    def __call__(self, *, messages, purpose):
        self.calls.append(purpose)
        return ModelOutput('{"candidates":[]}' if purpose=='extraction' else 'Acknowledged.')


class ReviewSchemaBoundaryTests(unittest.TestCase):
    def assert_no_durable_effects(self, store):
        for name in ('dm_sources', 'dm_jobs', 'dm_inputs', 'dm_attempts'):
            self.assertEqual(store.connection.execute(f'SELECT COUNT(*) FROM {name}').fetchone()[0], 0, name)

    def test_policy_rejects_noncanonical_timestamps_before_parsing(self):
        for stamp in BAD:
            with self.subTest(stamp=stamp), self.assertRaises(ValueError):
                moment(stamp)

    def test_standard_seconds_z_fraction_and_offsets_are_supported(self):
        for stamp in (GOOD, '2026-11-04T10:00:00Z', '2026-11-04T10:00:00.125Z',
                      '2026-11-04T18:00:00+08:00', '2026-11-04T05:00:00-05:00'):
            with self.subTest(stamp=stamp):
                self.assertIsNotNone(moment(stamp).tzinfo)
                data=record_data();data.update(source_time=stamp,observed_at=stamp,
                                              created_at=stamp,updated_at=stamp)
                self.assertEqual(construct_memory(data).source_time,stamp)

    def test_invalid_as_of_is_rejected_before_source_job_or_provider(self):
        for stamp in BAD:
            with self.subTest(stamp=stamp), tempfile.TemporaryDirectory() as folder:
                provider=CountingProvider()
                session=ControlledDialogueSession.open(Path(folder)/'memory.sqlite',
                    approved_root=folder,user_id='u',provider=provider,background=False)
                try:
                    with self.assertRaises(ValueError):
                        session.send_message('Thought record now paused.',input_id='input',as_of=stamp)
                    self.assert_no_durable_effects(session.store)
                    self.assertEqual(provider.calls,[])
                finally: session.close()

    def test_invalid_occurrence_is_rejected_before_source_job_or_provider(self):
        for stamp in BAD:
            with self.subTest(stamp=stamp), tempfile.TemporaryDirectory() as folder:
                provider=CountingProvider()
                session=ControlledDialogueSession.open(Path(folder)/'memory.sqlite',
                    approved_root=folder,user_id='u',provider=provider,background=False)
                try:
                    with self.assertRaises(ValueError):
                        session.send_message('I did thought record once.',input_id='input',
                            as_of=GOOD,occurred_at=stamp)
                    self.assert_no_durable_effects(session.store)
                    self.assertEqual(provider.calls,[])
                finally:session.close()

    def test_proposal_ingress_rejects_invalid_times_before_any_source(self):
        for stamp in BAD:
            with self.subTest(stamp=stamp),tempfile.TemporaryDirectory() as folder:
                session=ControlledDialogueSession.open(Path(folder)/'memory.sqlite',
                    approved_root=folder,user_id='u',provider=None,background=False)
                try:
                    with self.assertRaises(ValueError):
                        session.propose_task('thought record',2,as_of=stamp)
                    self.assert_no_durable_effects(session.store)
                finally:session.close()

    def test_standard_timestamp_reaches_successful_source_only_organization(self):
        for stamp in (GOOD,'2026-11-04T10:00:00Z','2026-11-04T18:00:00+08:00'):
            with self.subTest(stamp=stamp),tempfile.TemporaryDirectory() as folder:
                session=ControlledDialogueSession.open(Path(folder)/'memory.sqlite',
                    approved_root=folder,user_id='u',provider=None,background=False)
                try:
                    receipt=session.send_message('An exact fictional report.',input_id='input',as_of=stamp)
                    self.assertEqual(receipt['source_commit']['source_status'],'committed')
                    outcomes=session.run_pending()
                    self.assertTrue(outcomes)
                    self.assertEqual(outcomes[-1]['status'],'completed')
                    records=session.store.list_for_user('u')
                    self.assertEqual(len(records),1)
                    self.assertEqual(records[0].source_time,stamp)
                finally:session.close()

    def test_direct_source_ingress_rejects_invalid_time_and_keeps_unknown(self):
        with tempfile.TemporaryDirectory() as folder:
            durable=DurableMemoryStore.open(Path(folder)/'memory.sqlite',approved_root=folder)
            try:
                source=dict(user_id='u',source_id='source',input_id='input',text='A report.',
                            role='user',source_time=GOOD,permissions=dict(PERMISSIONS))
                for stamp in BAD:
                    with self.subTest(stamp=stamp),self.assertRaises(ValueError):
                        durable.submit_source({**source,'source_time':stamp})
                    self.assert_no_durable_effects(durable.store)
                durable.submit_source({**source,'source_time':None},enqueue=False)
                self.assertIsNone(durable.get_source('u','source')['source_time'])
                self.assertIsNone(durable.get_source('u','source',as_of=GOOD))
                for stamp in BAD:
                    with self.subTest(cutoff=stamp),self.assertRaises(ValueError):
                        durable.get_source('u','source',as_of=stamp)
            finally:durable.close()

    def test_direct_constructor_rejects_dict_before_manager_can_consume_it(self):
        data=record_data()
        with self.assertRaisesRegex(ValueError,'SourceEvidence'):
            MemoryRecordV2(**data)
        canonical=construct_memory(data)
        self.assertIsInstance(canonical.source_evidence,SourceEvidence)
        store=SQLiteMemoryStore()
        try:
            manager=MemoryManager(store)
            self.assertEqual(manager.ingest(canonical).action,'created')
            second=construct_memory(record_data('second'))
            manager.ingest(second)  # The existing-candidate deduplication path must not raise.
            self.assertEqual(len(store.list_for_user('u')),1)
        finally:store.close()

    def test_valid_direct_record_replace_and_mutation_validation(self):
        data=record_data();data['source_evidence']=SourceEvidence(**data['source_evidence'])
        direct=MemoryRecordV2(**data)
        self.assertIsInstance(replace(direct,memory_id='other').source_evidence,SourceEvidence)
        with self.assertRaisesRegex(ValueError,'SourceEvidence'):
            replace(direct,source_evidence=deepcopy(record_data()['source_evidence']))
        direct.source_evidence=record_data()['source_evidence']
        with self.assertRaisesRegex(ValueError,'SourceEvidence'):direct.validate()

    def test_direct_constructor_keeps_strict_permission_and_type_checks(self):
        for updates in ({'confidence':'0.8'}, {'permissions':{**PERMISSIONS,'save':'yes'}},
                        {'source_time':'2026-11-04T10:00+00:00'}):
            data=record_data();data['source_evidence']=SourceEvidence(**data['source_evidence']);data.update(updates)
            with self.subTest(updates=updates),self.assertRaises(ValueError):MemoryRecordV2(**data)


if __name__ == '__main__':unittest.main()
