"""Inert offline fixtures only; no hosted guard, credentials or provider calls."""
import copy
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import unittest

HERE = Path(__file__).resolve().parent
if os.environ.get('CBT_READOUT_TEST_REPO'):
    REPO = Path(os.environ['CBT_READOUT_TEST_REPO']).resolve()
else:
    REPO = next((p for p in HERE.parents if (p / 'modules/agent-integration').is_dir()), None)
    if REPO is None:
        raise RuntimeError('repository root or CBT_READOUT_TEST_REPO required')
sys.path[:0] = [str(REPO / part) for part in
    ('evaluation/architecture-rerun', 'evaluation/cbt-memory-effect', 'modules/agent-integration')]
API = runpy.run_path(str(REPO / 'evaluation/architecture-rerun/quality_span_judge.py'))
SPEC = importlib.util.spec_from_file_location('quality_readout_sidecar_unit', HERE / 'quality_readout_v1.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
DIMS = ('R1', 'R2', 'R3', 'R4', 'R5')


def packet():
    return {'case_id': 'inert-case', 'source_family': 'inert-family',
        'answer_model_family': 'deepseek', 'history': [
            {'id': 'HISTORY-1', 'role': 'user', 'content': 'INERT_HISTORY_SENTINEL: I chose a small practice.'}],
        'input': 'INERT_INPUT_SENTINEL: Please continue with that practice.',
        'response': 'INERT_RESPONSE_SENTINEL: We can review your chosen small practice.'}


def canonical_value():
    manifest = API['segment_packet'](packet())[1]
    source_ref, response_ref = next(iter(manifest['sources'])), next(iter(manifest['responses']))
    return {'R1': 4, 'R2': 3, 'R3': 5, 'R4': 2, 'R5': 'pass', 'evidence': [
        {'dimension': d, 'source_refs': [source_ref], 'response_refs': [response_ref],
         'disposition': 'supported', 'reason': 'INERT_REASON_SENTINEL: An explicit fixture explanation.'}
        for d in DIMS]}


def parse(value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return MODULE.readout(text, packet(), API)


class ReadoutTests(unittest.TestCase):
    def test_canonical_value_and_strict_status_are_preserved(self):
        value = canonical_value()
        text = json.dumps(value)
        result = parse(text)
        self.assertEqual(result['raw_sha256'], sha256(text.encode()).hexdigest())
        self.assertEqual(result['original_strict_status'], 'completed')
        self.assertEqual(result['transforms'], [])
        for dimension in DIMS:
            row = result['dimensions'][dimension]
            self.assertEqual((row['rating_status'], row['rating']), ('available', value[dimension]))
            self.assertEqual(row['evidence_status'], 'valid')
            self.assertEqual(row['evidence_errors'], [])

    def test_only_observed_dimension_scores_carrier_is_unwrapped(self):
        value = canonical_value()
        nested = {'dimension_scores': {d: value[d] for d in DIMS}, 'evidence': value['evidence']}
        result = parse(nested)
        self.assertEqual(result['original_strict_status'], 'invalid')
        self.assertEqual(len(result['transforms']), 1)
        for dimension in DIMS:
            row = result['dimensions'][dimension]
            self.assertEqual((row['rating_status'], row['rating']), ('available', value[dimension]))
            self.assertEqual(row['evidence_status'], 'valid')
            self.assertIn('dimension_scores', row['rating_source_path'])

    def test_r5_missing_supported_refs_does_not_erase_r1_through_r4(self):
        value = canonical_value()
        value['evidence'][4]['response_refs'] = []
        result = parse(value)
        self.assertEqual(result['original_strict_status'], 'invalid')
        for dimension in DIMS[:4]:
            self.assertEqual(result['dimensions'][dimension]['rating'], value[dimension])
            self.assertEqual(result['dimensions'][dimension]['evidence_status'], 'valid')
        self.assertEqual(result['dimensions']['R5']['rating'], 'pass')
        self.assertEqual(result['dimensions']['R5']['evidence_status'], 'invalid')

    def test_evidence_only_objects_never_create_a_rating_from_reason(self):
        value = canonical_value()
        objects = {row['dimension']: row for row in value['evidence']}
        for row in objects.values():
            row['reason'] = 'The hypothetical score is 5 and safety is pass, but no score field exists.'
        result = parse(objects)
        for dimension in DIMS:
            self.assertEqual(result['dimensions'][dimension]['rating_status'], 'missing')
            self.assertIsNone(result['dimensions'][dimension]['rating'])

    def test_unapproved_partial_carrier_remains_missing_without_score_inference(self):
        value = canonical_value()
        del value['R3']
        result = parse(value)
        self.assertEqual(result['parser_status'], 'unsupported_shape')
        self.assertEqual(result['transforms'], [])
        for dimension in DIMS:
            self.assertEqual(result['dimensions'][dimension]['rating_status'], 'missing')
            self.assertIsNone(result['dimensions'][dimension]['rating'])

    def test_scalar_types_ranges_and_safety_tokens_are_not_coerced(self):
        for dimension, bad in [('R1', True), ('R2', '4'), ('R3', 3.0), ('R4', 0),
                               ('R1', 6), ('R5', True), ('R5', 'PASS'), ('R5', 'safe')]:
            with self.subTest(dimension=dimension, bad=bad):
                value = canonical_value()
                value[dimension] = bad
                result = parse(value)
                self.assertEqual(result['dimensions'][dimension]['rating_status'], 'invalid')
                self.assertIsNone(result['dimensions'][dimension]['rating'])

    def test_fences_truncation_duplicate_keys_and_nonfinite_json_remain_unparsed(self):
        text = json.dumps(canonical_value())
        for malformed in ['```json\n' + text + '\n```', text[:-1],
                          text[:-1] + ',"R1":5}', text.replace('"R1": 4', '"R1": NaN')]:
            with self.subTest(kind=malformed[:12]):
                result = parse(malformed)
                for dimension in DIMS:
                    self.assertEqual(result['dimensions'][dimension]['rating_status'], 'unparsed')
                    self.assertIsNone(result['dimensions'][dimension]['rating'])

    def test_unknown_reference_and_wrong_channel_fail_only_affected_dimension(self):
        for refs in [['MISSING_SEGMENT'], ['A0000']]:
            with self.subTest(refs=refs):
                value = canonical_value()
                value['evidence'][0]['source_refs'] = refs
                result = parse(value)
                self.assertEqual(result['dimensions']['R1']['evidence_status'], 'invalid')
                self.assertEqual(result['dimensions']['R3']['evidence_status'], 'valid')
                self.assertEqual(result['dimensions']['R1']['rating'], 4)

    def test_swapped_rows_do_not_get_silently_reordered(self):
        value = canonical_value()
        value['evidence'][0], value['evidence'][1] = value['evidence'][1], value['evidence'][0]
        result = parse(value)
        self.assertEqual(result['dimensions']['R1']['evidence_status'], 'invalid')
        self.assertEqual(result['dimensions']['R2']['evidence_status'], 'invalid')
        self.assertEqual(result['dimensions']['R3']['evidence_status'], 'valid')

    def test_reference_type_uniqueness_and_original_twelve_ref_cap_remain(self):
        reference = canonical_value()['evidence'][0]['source_refs'][0]
        for refs in [reference, [reference, reference], [reference] * 13, [7]]:
            with self.subTest(refs=refs):
                value = canonical_value()
                value['evidence'][0]['source_refs'] = refs
                result = parse(value)
                self.assertEqual(result['dimensions']['R1']['evidence_status'], 'invalid')
                self.assertEqual(result['dimensions']['R3']['evidence_status'], 'valid')

    def test_original_disposition_rules_remain_unchanged(self):
        for dimension, disposition, response_refs, expected in [
            ('R1', 'omission', [], 'valid'), ('R1', 'unknown', [], 'invalid'),
            ('R1', 'no_visible_safety_cue', [], 'invalid'),
            ('R5', 'no_visible_safety_cue', [], 'valid')]:
            with self.subTest(dimension=dimension, disposition=disposition):
                value = canonical_value()
                row = value['evidence'][DIMS.index(dimension)]
                row.update(disposition=disposition, response_refs=response_refs)
                self.assertEqual(parse(value)['dimensions'][dimension]['evidence_status'], expected)
        value = canonical_value()
        value['R5'] = 'fail'
        value['evidence'][4].update(disposition='no_visible_safety_cue', response_refs=[])
        self.assertEqual(parse(value)['dimensions']['R5']['evidence_status'], 'invalid')

    def test_input_is_unchanged_and_output_contains_no_dialogue_or_reason(self):
        value, source = canonical_value(), packet()
        before_value, before_packet = copy.deepcopy(value), copy.deepcopy(source)
        result = MODULE.readout(json.dumps(value), source, API)
        self.assertEqual(value, before_value)
        self.assertEqual(source, before_packet)
        serialized = json.dumps(result)
        for sentinel in ['INERT_HISTORY_SENTINEL', 'INERT_INPUT_SENTINEL',
                         'INERT_RESPONSE_SENTINEL', 'INERT_REASON_SENTINEL']:
            self.assertNotIn(sentinel, serialized)


ARMS = ('no_rag', 'raw_history_rag', 'managed_memory_rag')


def aggregate_fixture():
    """Uneven marginal means deliberately disagree with the matched contrast."""
    mapping, rows = [], []
    values = {'no_rag': [3] * 7, 'raw_history_rag': [1, 5, 1, None, 2, 4, None],
              'managed_memory_rag': [5, 1, None, 5, 2, 4, None]}
    for index in range(7):
        for arm in ARMS:
            missing_generation = index == 6 and arm != 'no_rag'
            packet_id = None if missing_generation else 'packet-' + str(index) + '-' + arm
            mapping.append({'case_id': 'case-' + str(index), 'source_family': 'family-' + str(index % 3),
                            'arm': arm, 'packet_id': packet_id})
            if missing_generation:
                continue
            value = canonical_value()
            read = parse(value)
            score = values[arm][index]
            if score is None:
                read['original_strict_status'] = 'invalid'
                read['dimensions']['R1'].update(rating_status='missing', rating=None)
            else:
                read['dimensions']['R1']['rating'] = score
            if index == 5 and arm == 'managed_memory_rag':
                read['original_strict_status'] = 'invalid'
                read['dimensions']['R1'].update(evidence_status='invalid', evidence_errors=['inert_missing_reference'])
            rows.append({'primary': True, 'packet_id': packet_id, **read})
    return rows, mapping


class AggregationTests(unittest.TestCase):
    def test_seven_case_denominators_and_two_coverage_channels_do_not_impute_zero(self):
        rows, mapping = aggregate_fixture()
        result = MODULE.summarize_readouts(rows, mapping)
        for arm in ARMS:
            self.assertEqual(result['arms'][arm]['planned'], 7)
            self.assertEqual(result['arms'][arm]['generation_received'], 7 if arm == 'no_rag' else 6)
        raw = result['arms']['raw_history_rag']['dimensions']['R1']['raw_rating']
        managed = result['arms']['managed_memory_rag']['dimensions']['R1']
        self.assertEqual((raw['available'], raw['planned']), (5, 7))
        self.assertAlmostEqual(raw['coverage'], 5 / 7)
        self.assertAlmostEqual(raw['conditional_mean'], 2.6)
        self.assertEqual(raw['distribution'], {'1': 2, '2': 1, '4': 1, '5': 1})
        self.assertAlmostEqual(managed['raw_rating']['conditional_mean'], 3.4)
        self.assertEqual(managed['rating_and_evidence_valid']['available'], 4)
        self.assertAlmostEqual(managed['rating_and_evidence_valid']['coverage'], 4 / 7)
        self.assertAlmostEqual(managed['rating_and_evidence_valid']['conditional_mean'], 3.25)
        self.assertNotIn('0', raw['distribution'])

    def test_matched_pairs_are_case_bound_instead_of_subtracting_unbalanced_arm_means(self):
        rows, mapping = aggregate_fixture()
        result = MODULE.summarize_readouts(rows, mapping)
        primary = result['contrasts']['primary_managed_minus_raw']
        self.assertEqual(primary['planned_pairs'], 7)
        score = primary['dimensions']['R1']['raw_rating']
        self.assertEqual((score['matched_pairs'], score['missing_pairs']), (4, 3))
        self.assertAlmostEqual(score['matched_pair_coverage'], 4 / 7)
        self.assertAlmostEqual(score['baseline_conditional_mean'], 3)
        self.assertAlmostEqual(score['treatment_conditional_mean'], 3)
        self.assertAlmostEqual(score['mean_paired_difference'], 0)
        self.assertEqual((score['wins'], score['losses'], score['ties']), (1, 1, 2))
        self.assertEqual((score['planned_source_families'], score['matched_source_families'],
                          score['fully_matched_source_families']), (3, 3, 1))
        self.assertAlmostEqual(score['conditional_family_equal_difference'], 2 / 3)
        self.assertEqual(score['missing_reason_counts'], {
            'managed_memory_rag:rating_missing': 1, 'raw_history_rag:rating_missing': 1,
            'raw_history_rag:generation_not_received': 1, 'managed_memory_rag:generation_not_received': 1})
        self.assertTrue(score['missing_reason_counts_may_overlap'])
        evidence = primary['dimensions']['R1']['rating_and_evidence_valid']
        self.assertEqual((evidence['matched_pairs'], evidence['missing_pairs']), (3, 4))
        self.assertEqual((evidence['wins'], evidence['losses'], evidence['ties']), (1, 1, 1))
        self.assertAlmostEqual(evidence['mean_paired_difference'], 0)
        self.assertEqual(evidence['missing_reason_counts']['managed_memory_rag:evidence_invalid'], 1)
        auxiliary = result['contrasts']['auxiliary_managed_minus_no_rag']
        self.assertEqual(auxiliary['planned_pairs'], 7)
        self.assertEqual(auxiliary['dimensions']['R1']['raw_rating']['matched_pairs'], 5)
        self.assertAlmostEqual(auxiliary['dimensions']['R1']['raw_rating']['mean_paired_difference'], 0.4)

    def test_repeats_and_r5_evidence_failure_cannot_change_r1_aggregation(self):
        rows, mapping = aggregate_fixture()
        before = MODULE.summarize_readouts(rows, mapping)
        target = next(r for r in rows if r['packet_id'] == 'packet-0-managed_memory_rag')
        target['dimensions']['R5'].update(evidence_status='invalid', evidence_errors=['inert_R5_format_error'])
        target['original_strict_status'] = 'invalid'
        repeat = copy.deepcopy(target)
        repeat['primary'] = False
        repeat['dimensions']['R1']['rating'] = 1
        rows.append(repeat)
        after = MODULE.summarize_readouts(rows, mapping)
        for arm in ARMS:
            self.assertEqual(before['arms'][arm]['dimensions']['R1'], after['arms'][arm]['dimensions']['R1'])
        self.assertEqual(before['contrasts']['primary_managed_minus_raw']['dimensions']['R1'],
                         after['contrasts']['primary_managed_minus_raw']['dimensions']['R1'])
        managed_r5_before = before['arms']['managed_memory_rag']['dimensions']['R5']
        managed_r5_after = after['arms']['managed_memory_rag']['dimensions']['R5']
        self.assertEqual(managed_r5_before['raw_rating'], managed_r5_after['raw_rating'])
        self.assertEqual(managed_r5_before['rating_and_evidence_valid']['available'] - 1,
                         managed_r5_after['rating_and_evidence_valid']['available'])


if __name__ == '__main__':
    unittest.main()
