"""Source/gold separation and temporal/privacy boundaries on fictional materials."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'evaluation/dialogue-benefit'))
import prepare_generation_materials as compiler
from study_capture import canonical_hash


class FollowupGenerationMaterialsTests(unittest.TestCase):
    def setUp(self):
        self.ledger_path = ROOT / 'evaluation/dialogue-benefit/task_materials_draft.json'
        self.ledger = json.loads(self.ledger_path.read_text())
        self.generation, self.events, self.refs = compiler.compile_materials(self.ledger)

    def rehash(self, generation):
        generation['generation_package_sha256'] = canonical_hash({k: v for k, v in generation.items() if k != 'generation_package_sha256'})
        return generation

    def rehash_criterion(self, task):
        task['criterion_sha256'] = canonical_hash({k: v for k, v in task.items() if k != 'criterion_sha256'})

    def test_complete_development_census_is_not_human_or_clinical_evidence(self):
        self.assertEqual(len(self.generation['tasks']), 36)
        self.assertEqual(len(self.events), 57)
        self.assertEqual(len(self.refs), 36)
        self.assertFalse(self.generation['clinical_claim'])
        self.assertEqual(self.generation['independent_human_reference_validator_count'], 0)
        self.assertEqual(self.generation['origin_policy'], 'fictional_ledger_observations_not_authenticated_consent')
        compiler.validate_generation_package(self.generation, self.events)

    def test_assessor_criterion_sentinel_never_enters_generation_or_replay(self):
        sentinel = 'ASSESSOR_ONLY_SENTINEL_NEVER_GENERATE_23917'
        self.ledger['tasks'][0]['completion_criterion'] += sentinel
        self.rehash_criterion(self.ledger['tasks'][0])
        generation, events, refs = compiler.compile_materials(self.ledger)
        self.assertNotIn(sentinel, json.dumps(generation, ensure_ascii=False))
        self.assertNotIn(sentinel, json.dumps(events, ensure_ascii=False))
        self.assertIn(sentinel, refs[0]['completion_criterion'])
        self.assertNotIn('variant_parameters', generation)
        self.assertNotIn('completion_criterion', generation['tasks'][0])
        self.assertNotIn('source_reference', generation['tasks'][0])

    def test_engine_replays_admitted_source_before_delete_while_models_do_not_see_it_later(self):
        events = [e for e in self.events if e['variant'] == 'A']
        insert = next(i for i, e in enumerate(events) if e['kind'] == 'source' and e['source']['source_id'] == 'A-S13')
        delete = next(i for i, e in enumerate(events) if e['kind'] == 'control' and e['control']['action'] == 'delete')
        self.assertLess(insert, delete)
        self.assertEqual(events[delete]['control']['source_order'], 14)
        self.assertEqual(events[delete]['control']['target_source_ids'], ['A-S13'])
        private_text = events[insert]['source']['text']
        for task in self.generation['tasks']:
            if task['variant'] == 'A' and task['history_prefix_source_order'] >= 14:
                self.assertNotIn('A-S13', task['visible_source_ids'])
                self.assertNotIn(private_text, json.dumps(task, ensure_ascii=False))
                self.assertIn({'source_id': 'A-S13', 'reason': 'deleted'}, task['exclusions'])
        # No task occurs between this source's appearance and its deletion.
        self.assertNotIn(private_text, json.dumps(self.generation, ensure_ascii=False))

    def test_assistant_suggestion_keeps_original_role_and_no_fake_confirmation(self):
        task = next(t for t in self.generation['tasks'] if t['case_id'] == 'A:T04')
        suggestion = next(s for s in task['history'] if s['source_id'] == 'A-S02')
        self.assertEqual(suggestion['role'], 'assistant')
        self.assertNotIn('confirmed', suggestion)
        self.assertEqual(suggestion['text'], self.ledger['sources'][1]['source_text'])

    def test_query_hash_and_exact_reference_span_are_validated_before_generation(self):
        bad = deepcopy(self.ledger)
        bad['tasks'][0]['task_query'] += 'unfrozen change'
        self.rehash_criterion(bad['tasks'][0])
        with self.assertRaises(ValueError):
            compiler.compile_materials(bad)
        bad = deepcopy(self.ledger)
        bad['tasks'][0]['source_reference'][0]['quote'] += 'wrong span'
        self.rehash_criterion(bad['tasks'][0])
        with self.assertRaises(ValueError):
            compiler.compile_materials(bad)

    def test_future_source_cannot_be_added_by_changing_visible_inventory(self):
        bad = deepcopy(self.ledger)
        bad['tasks'][0]['visible_source_ids'].append('A-S06')
        self.rehash_criterion(bad['tasks'][0])
        with self.assertRaisesRegex(ValueError, 'exact prefix'):
            compiler.compile_materials(bad)

    def test_postdelete_cache_or_raw_history_resurrection_is_rejected(self):
        bad = deepcopy(self.generation)
        task = next(t for t in bad['tasks'] if t['case_id'] == 'A:T09')
        source = next(e['source'] for e in self.events if e['kind'] == 'source' and e['source']['source_id'] == 'A-S13')
        task['history'].append(deepcopy(source)); task['visible_source_ids'].append('A-S13')
        self.rehash(bad)
        with self.assertRaises(ValueError):
            compiler.validate_generation_package(bad, self.events)

    def test_session_only_source_is_not_available_in_later_visit_or_new_session(self):
        task = next(t for t in self.generation['tasks'] if t['case_id'] == 'A:T09')
        self.assertIn({'source_id': 'A-S15', 'reason': 'dont_save'}, task['exclusions'])
        self.assertNotIn('A-S15', task['visible_source_ids'])
        old = next(t for t in self.generation['tasks'] if t['case_id'] == 'A:T11')
        fresh = next(t for t in self.generation['tasks'] if t['case_id'] == 'A:T12')
        self.assertEqual(old['current_session_source_ids'], ['A-S16'])
        self.assertEqual(fresh['current_session_source_ids'], [])
        bad = deepcopy(self.generation)
        next(t for t in bad['tasks'] if t['case_id'] == 'A:T12')['current_session_source_ids'] = ['A-S16']
        self.rehash(bad)
        with self.assertRaises(ValueError):
            compiler.validate_generation_package(bad, self.events)

    def test_expiry_rule_is_tested_without_pretending_current_materials_have_expiry_cases(self):
        source_events = [deepcopy(e['source']) for e in self.events if e['variant'] == 'A' and e['kind'] == 'source']
        source_events[4]['valid_to'] = '2026-09-06T00:00:00Z'
        original = deepcopy(next(t for t in self.generation['tasks'] if t['case_id'] == 'A:T05'))
        original['visible_source_ids'].remove('A-S05')
        controls = [e['control'] for e in self.events if e['variant'] == 'A' and e['kind'] == 'control']
        snapshot = compiler.snapshot_task(original, source_events, controls)
        self.assertIn({'source_id': 'A-S05', 'reason': 'expired'}, snapshot['exclusions'])
        self.assertNotIn('A-S05', snapshot['visible_source_ids'])

    def test_replay_cannot_run_control_before_original_ingestion(self):
        bad = deepcopy(self.events)
        before = next(i for i, e in enumerate(bad) if e['variant'] == 'A' and e['kind'] == 'source' and e['source']['source_id'] == 'A-S13')
        control = next(i for i, e in enumerate(bad) if e['variant'] == 'A' and e['kind'] == 'control' and e['control']['action'] == 'delete')
        bad[before], bad[control] = bad[control], bad[before]
        with self.assertRaisesRegex(ValueError, 'chronology'):
            compiler.validate_generation_package(self.generation, bad)

    def test_hidden_gold_fields_or_forged_consent_are_rejected_even_with_new_package_hash(self):
        for mutate in ('task_gold', 'source_gold', 'consent'):
            bad, events = deepcopy(self.generation), deepcopy(self.events)
            if mutate == 'task_gold':
                bad['tasks'][0]['completion_criterion'] = 'Gold must not enter generation'
            elif mutate == 'source_gold':
                events[0]['source']['gold'] = 'Gold must not enter generation'
            else:
                bad['origin_policy'] = 'authenticated_user_consent'
            self.rehash(bad)
            with self.assertRaises(ValueError):
                compiler.validate_generation_package(bad, events)

    def test_source_text_change_and_relabelled_assistant_source_are_rejected(self):
        bad = deepcopy(self.events)
        bad[0]['source']['text'] += 'untracked edit'
        with self.assertRaisesRegex(ValueError, 'hash'):
            compiler.validate_generation_package(self.generation, bad)
        bad = deepcopy(self.events)
        assistant = next(e for e in bad if e['kind'] == 'source' and e['source']['role'] == 'assistant')
        assistant['source']['role'] = 'user'
        with self.assertRaises(ValueError):
            compiler.validate_generation_package(self.generation, bad)

    def test_prepared_generation_loader_does_not_read_scoring_file_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'prepared'
            manifest = compiler.prepare_materials(self.ledger_path, out)
            self.assertFalse(manifest['expiry_cases_present'])
            self.assertEqual(manifest['independent_people'], 0)
            original_read_text, original_read_bytes = Path.read_text, Path.read_bytes
            def protected_text(path, *args, **kwargs):
                if path.name == 'scoring_references.jsonl':
                    raise AssertionError('Generation tried to read scorer references')
                return original_read_text(path, *args, **kwargs)
            def protected_bytes(path, *args, **kwargs):
                if path.name == 'scoring_references.jsonl':
                    raise AssertionError('Generation tried to read scorer references')
                return original_read_bytes(path, *args, **kwargs)
            with patch.object(Path, 'read_text', protected_text), patch.object(Path, 'read_bytes', protected_bytes):
                loaded = compiler.read_prepared_generation(out)
            self.assertEqual(loaded['generation'], self.generation)
            with self.assertRaises(ValueError):
                compiler.prepare_materials(self.ledger_path, out)
            self.assertEqual(hashlib.sha256((out/'generation_materials.json').read_bytes()).hexdigest(), manifest['artifact_sha256']['generation_materials.json'])

    def test_reference_may_use_absent_source_only_with_prespecified_absence_reason(self):
        bad = deepcopy(self.ledger)
        target = next(t for t in bad['tasks'] if t['variant'] == 'A' and t['task_id'] == 'T09')
        target['source_reference'][1].pop('permitted_absence_reason')
        self.rehash_criterion(target)
        with self.assertRaisesRegex(ValueError, 'absence'):
            compiler.compile_materials(bad)


if __name__ == '__main__':
    unittest.main()
