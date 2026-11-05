"""Strict link-placement compatibility, using only tiny fictional sources."""
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'evaluation/dialogue-benefit'))
import generation_adapter as adapter
from source_replay import SourceReplay


def source(number, *, role='user', storage='allowed', date=None, valid_to=None):
    text = 'Fictional source ' + str(number)
    row = {'source_id': f'A-S{number:02}', 'source_order': number, 'visit': 1,
           'date': date or f'2026-09-01T10:{number:02}:00Z', 'role': role,
           'text': text, 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
           'storage_policy': storage}
    if valid_to:
        row['valid_to'] = valid_to
    return row


def candidate(current, *, typ='user_correction', top=None, payload=None):
    row = {'memory_type': typ, 'content': current['text'],
           'source_turn_ids': [current['source_id']], 'supporting_text': current['text'],
           'structured_payload': dict(payload or {}), 'confidence': .8, 'importance': .5}
    if top is not None:
        row.update(top)
    return json.dumps({'candidates': [row]})


class LinkContractTests(unittest.TestCase):
    def replay(self, previous=None):
        replay = SourceReplay(user_id='fictional-A')
        self.addCleanup(replay.close)
        for item in previous or [source(1)]:
            replay.ingest_source(item)
        return replay

    def semantic_rows(self, replay):
        return [r for r in replay.store.list_for_user(replay.user_id) if r.memory_id.startswith('candidate-')]

    def test_three_correction_placements_have_identical_canonical_record(self):
        seen = []
        current = source(2)
        for top, payload in [({'target_source_id': 'A-S01'}, {}),
                             ({}, {'target_source_id': 'A-S01'}),
                             ({}, {'proposed_update_target_source_id': 'A-S01'}),
                             ({'target_source_id': 'A-S01'}, {'target_source_id': 'A-S01', 'proposed_update_target_source_id': 'A-S01'})]:
            with self.subTest(top=bool(top), keys=sorted(payload)):
                replay = self.replay()
                replay.ingest_source(current, extraction_output=candidate(current, top=top, payload=payload), extraction_context_ids=['A-S01', 'A-S02'])
                row = self.semantic_rows(replay)[0]
                self.assertEqual(row.structured_payload['proposed_update_target_source_id'], 'A-S01')
                self.assertNotIn('target_source_id', row.structured_payload)
                self.assertEqual(row.confirmation_state, 'unconfirmed')
                self.assertEqual(row.lifecycle_status, 'candidate')
                self.assertEqual(replay.semantic_statuses['A-S02'], 'valid_candidates')
                self.assertEqual(replay.events[-1]['writes'][-1]['action'], 'correction_requires_confirmation')
                seen.append(row.to_dict())
        self.assertTrue(all(row == seen[0] for row in seen))

    def test_assignment_top_level_and_payload_are_equivalent(self):
        current = source(2)
        rows = []
        for top, payload in [({'assignment_source_id': 'A-S01'}, {'completion_status': 'partly_done'}),
                             ({}, {'assignment_source_id': 'A-S01', 'completion_status': 'partly_done'})]:
            replay = self.replay()
            replay.ingest_source(current, extraction_output=candidate(current, typ='assignment_outcome', top=top, payload=payload), extraction_context_ids=['A-S01', 'A-S02'])
            row = self.semantic_rows(replay)[0]
            self.assertEqual(row.structured_payload['assignment_memory_id'], replay.quote_id('A-S01'))
            self.assertEqual(replay.record_dependencies[row.memory_id], {'A-S01', 'A-S02'})
            rows.append(row.to_dict())
        self.assertEqual(rows[0], rows[1])

    def test_valid_targets_that_disagree_are_rejected(self):
        current = source(3)
        replay = self.replay([source(1), source(2)])
        replay.ingest_source(current, extraction_output=candidate(current, top={'target_source_id': 'A-S01'}, payload={'target_source_id': 'A-S02'}))
        self.assertEqual(replay.semantic_statuses['A-S03'], 'extraction_invalid')
        self.assertEqual(self.semantic_rows(replay), [])

    def test_invalid_present_value_is_not_rescued_by_valid_alias(self):
        current = source(2)
        for invalid in (None, [], {}, 1, True, '', ' ', ' A-S01 ', 'B-S01', 'A-S99', 'A-S02'):
            with self.subTest(value_type=type(invalid).__name__):
                replay = self.replay()
                replay.ingest_source(current, extraction_output=candidate(current, top={'target_source_id': 'A-S01'}, payload={'target_source_id': invalid}))
                self.assertEqual(replay.semantic_statuses['A-S02'], 'extraction_invalid')
                self.assertEqual(self.semantic_rows(replay), [])

    def test_canonical_payload_target_is_no_longer_an_unvalidated_bypass(self):
        current = source(2)
        for invalid in ('B-S01', 'A-S02', 'A-S03'):
            with self.subTest(invalid=invalid):
                replay = self.replay()
                replay.ingest_source(current, extraction_output=candidate(current, payload={'proposed_update_target_source_id': invalid}))
                self.assertEqual(replay.semantic_statuses['A-S02'], 'extraction_invalid')

    def test_assistant_session_only_future_dated_and_expired_targets_rejected(self):
        current = source(2)
        previous_cases = [source(1, role='assistant'), source(1, storage='session_only'),
                          source(1, date='2027-01-01T00:00:00Z'),
                          source(1, valid_to=current['date'])]
        for previous in previous_cases:
            with self.subTest(role=previous['role'], storage=previous['storage_policy']):
                replay = self.replay([previous])
                replay.ingest_source(current, extraction_output=candidate(current, payload={'target_source_id': 'A-S01'}))
                self.assertEqual(replay.semantic_statuses['A-S02'], 'extraction_invalid')
                self.assertEqual(self.semantic_rows(replay), [])

    def test_deleted_target_cannot_be_reintroduced_by_nested_link(self):
        replay = self.replay()
        current = source(2)
        replay.apply_control({'event_id': 'delete', 'action': 'delete', 'target_source_ids': ['A-S01'],
                              'as_of': current['date'], 'source_order': 2, 'before_task_id': None})
        replay.ingest_source(current, extraction_output=candidate(current, payload={'target_source_id': 'A-S01'}))
        self.assertEqual(replay.semantic_statuses['A-S02'], 'extraction_invalid')
        self.assertNotIn('A-S01', replay.raw_sources)
        self.assertIsNone(replay.store.get(replay.user_id, replay.quote_id('A-S01')))

    def test_backdated_future_order_and_foreign_scope_remain_invalid(self):
        current = source(2)
        future = source(3, date='2026-09-01T10:01:00Z')
        foreign = source(1)
        foreign['source_id'] = 'B-S01'
        for previous in (future, foreign):
            with self.subTest(source_id=previous['source_id']):
                replay = self.replay([previous])
                replay.ingest_source(current, extraction_output=candidate(current, payload={'target_source_id': previous['source_id']}))
                self.assertEqual(replay.semantic_statuses['A-S02'], 'extraction_invalid')
                self.assertEqual(self.semantic_rows(replay), [])

    def test_alias_normalisation_does_not_relax_quote_validation(self):
        current = source(2)
        row = json.loads(candidate(current, payload={'target_source_id': 'A-S01'}))
        row['candidates'][0]['supporting_text'] = 'invented quote'
        replay = self.replay()
        replay.ingest_source(current, extraction_output=json.dumps(row))
        self.assertEqual(replay.semantic_statuses['A-S02'], 'extraction_invalid')
        self.assertEqual(self.semantic_rows(replay), [])

    def test_semantic_sql_rollback_does_not_commit_phantom_dependencies(self):
        current = source(2)
        replay = self.replay()
        parsed = json.loads(candidate(current, typ='explicit_fact'))
        second = dict(parsed['candidates'][0], content='A distinct indexing proposal')
        parsed['candidates'].append(second)
        real_ingest = replay.manager.ingest
        calls = []
        def injected(record, **kwargs):
            calls.append(record.memory_id)
            if len(calls) == 2:
                raise ValueError('synthetic second write failure')
            return real_ingest(record, **kwargs)
        with patch.object(replay.manager, 'ingest', side_effect=injected):
            replay.ingest_source(current, extraction_output=json.dumps(parsed), extraction_context_ids=['A-S01', 'A-S02'])
        self.assertEqual(replay.semantic_statuses['A-S02'], 'extraction_invalid')
        self.assertEqual(self.semantic_rows(replay), [])
        self.assertFalse(any(key.startswith('candidate-') for key in replay.record_dependencies))
        self.assertEqual(set(replay.record_dependencies), {r.memory_id for r in replay.store.list_for_user(replay.user_id)})
        self.assertFalse(replay.extraction_cache)

    def test_rejected_duplicate_does_not_create_phantom_dependency_record(self):
        current = source(2)
        replay = self.replay()
        parsed = json.loads(candidate(current, typ='explicit_fact'))
        parsed['candidates'] *= 2
        replay.ingest_source(current, extraction_output=json.dumps(parsed), extraction_context_ids=['A-S01', 'A-S02'])
        self.assertEqual(len(self.semantic_rows(replay)), 1)
        self.assertEqual(set(replay.record_dependencies), {r.memory_id for r in replay.store.list_for_user(replay.user_id)})
        self.assertEqual(replay.events[-1]['writes'][-1]['action'], 'duplicate_rejected')

    def test_new_contract_is_explicit_and_does_not_change_common_answer_prompt(self):
        current = source(2)
        events = [{'variant': 'A', 'kind': 'source', 'source': item} for item in [source(1), current]]
        request = adapter.source_event_extraction_request(events[-1], events)
        self.assertEqual(json.loads(request[1]['content'])['extraction_contract_version'], 'source-event-extraction-links-v2')
        self.assertIn('AT THE CANDIDATE TOP LEVEL', request[0]['content'])
        self.assertIn('ALL present placements must agree', request[0]['content'])
        self.assertEqual(adapter.VERSION, 'fictional-source-candidate-pipeline-v2')


if __name__ == '__main__':
    unittest.main()
