"""Derive complete-study tables in hosted Colab; no inference or score rewriting.

The frozen scorer and full authenticated archives must agree before any output.
All four related scenario families and both languages are retained in the report.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

HELPER = Path('/content/cbt-prefill-runtime-code-20261112-v2/evaluation/cbt-memory-effect')
ROOT = Path('/content/cbt-unconstrained-state-study-20261112-v3')
PROTOCOL_SHA = '118bccbf288105cafbe4c88e6b582f0742b1b28056c8d06a5cf7860296841d16'
STAGE_SHA = '4a350371c52dc34ac5c5f7c7d7880e2fd70d13ec450698b10c0af5e6d43812bb'
FAMILIES = ('template-target_correction', 'template-cumulative_repeat',
            'template-proposal_not_agreement', 'template-outcome_not_progress')
ARMS = ('raw_history_rag', 'managed_memory_rag')
METRICS = ('raw_strict', 'canonical_pipeline')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grouped(rows, field):
    result = {}
    for value in sorted({r[field] for r in rows}):
        result[value] = {}
        for arm in ARMS:
            subset = [r for r in rows if r[field] == value and r['arm'] == arm]
            result[value][arm] = {'questions': len(subset),
                'adapter_accepted': sum(r['adapter_accepted'] for r in subset),
                **{metric: {'correct': sum(r[metric]['correct'] for r in subset),
                    'error_counts': dict(Counter(r[metric]['reason'] for r in subset
                                                if not r[metric]['correct']))}
                   for metric in METRICS}}
    return result


def fact_metrics(rows, arm, metric):
    """Descriptive answer coverage and precision; no inference about source use."""
    selected = [r for r in rows if r['arm'] == arm]
    known = [r for r in selected if r['expected_answer'] is not None]
    unknown = [r for r in selected if r['expected_answer'] is None]
    valid = [r for r in selected if r[metric]['reason'] not in
             ('malformed_output', 'adapter_rejected')]
    asserted = [r for r in valid if r[metric]['observed_answer'] is not None]
    correct_asserted = sum(r[metric]['correct'] and r['expected_answer'] is not None
                           for r in asserted)
    return {'known_value_questions': len(known),
        'known_values_answered_correctly': sum(r[metric]['correct'] for r in known),
        'known_value_answer_coverage': sum(r[metric]['correct'] for r in known) / len(known)
            if known else None,
        'valid_nonnull_assertions': len(asserted), 'correct_nonnull_assertions': correct_asserted,
        'nonnull_assertion_precision': correct_asserted / len(asserted) if asserted else None,
        'wrong_known_value_assertions': sum(r['expected_answer'] is not None and not r[metric]['correct']
                                           for r in asserted),
        'unsupported_assertions_on_unknown_questions': sum(r['expected_answer'] is None
                                                           for r in asserted),
        'incorrect_abstentions_on_known_questions': sum(r['expected_answer'] is not None
            and r[metric]['observed_answer'] is None for r in valid),
        'reviewed_unknown_questions': len(unknown),
        'correct_unknown_answers': sum(r[metric]['correct'] for r in unknown),
        'unparsed_outputs': len(selected) - len(valid),
        'scope': 'descriptive typed-answer metrics; correct guessing and actual evidence use are not distinguished'}


def derive(output):
    # This file can be syntax checked locally; actual artifact work is hosted only.
    try:
        hosted = (bool(os.environ.get('COLAB_RELEASE_TAG')) and Path('/content').is_dir()
                  and importlib.util.find_spec('google.colab') is not None)
    except (ImportError, ValueError, AttributeError):
        hosted = False
    if not hosted:
        raise RuntimeError('genuine_hosted_colab_required_before_artifact_io')
    sys.path.insert(0, str(HELPER))
    spec = importlib.util.spec_from_file_location('frozen_v3_complete_analysis',
                                                HELPER / 'unconstrained_state_comparison_v3_colab.py')
    if spec is None or spec.loader is None:
        raise RuntimeError('frozen_hosted_stage_required')
    stage = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = stage
    spec.loader.exec_module(stage)
    stage.guard(ROOT, output)
    if output.exists() or output == ROOT or output.is_relative_to(ROOT):
        raise FileExistsError('fresh_external_analysis_directory_required')
    if sha(ROOT / 'study_protocol.json') != PROTOCOL_SHA or sha(stage.__file__) != STAGE_SHA:
        raise ValueError('frozen_v3_identity_changed')
    plan = stage.frozen(ROOT)
    intents, outcomes = stage.ledger(ROOT)
    if len(intents) != len(outcomes) or len(intents) != 57:
        raise RuntimeError('all_57_settled_generations_required_no_partial_analysis')
    authenticated = {}
    for side in ('before', 'after'):
        authenticated[side], _ = stage.authenticate_main(ROOT, side, plan['sections'][side])
        if len(authenticated[side]) != 48:
            raise RuntimeError('both_48_row_archives_required_before_any_labels')

    # Only after both full archives authenticate do we open reports and targets.
    reports, answer_maps, report_maps = {}, {}, {}
    for side in ('before', 'after'):
        report = stage.reader.read_json(ROOT / 'reports' / (side + '_metrics.json'))
        target_path = Path(plan['sections'][side]['fresh_prepared']) / 'assessor_only/targets.json'
        if sha(target_path) != plan['sections'][side]['targets_file_sha256']:
            raise ValueError('frozen_literal_targets_changed')
        targets = {r['case_id']: r for r in stage.reader.read_json(target_path)}
        answers = {(r['case_id'], r['arm']): r for r in authenticated[side]}
        scored = {(r['case_id'], r['arm']): r for r in report['per_question']}
        if len(targets) != 24 or set(answers) != set(scored) or len(scored) != 48:
            raise ValueError('complete_question_coverage_required')
        if (report['answers_sha256'] != sha(ROOT / ('answers_' + side) / 'answers_unscored.jsonl')
                or report['main_freeze_sha256'] != sha(ROOT / 'main_freeze.json')
                or report['scorer_code_sha256'] != STAGE_SHA):
            raise ValueError('frozen_score_report_binding_changed')
        for key, answer in answers.items():
            row, target = scored[key], targets[key[0]]
            if (row['expected_answer'] != target['answer']
                    or type(row['expected_answer']) is not type(target['answer'])
                    or row['request_sha256'] != answer['request_sha256']
                    or row['response_sha256'] != answer['response_sha256']):
                raise ValueError('question_reference_or_response_binding_changed')
            raw = stage.strict_result(answer['response_text'], target['answer'])
            adapted = answer['answer_adapter']
            canonical = (stage.strict_result(adapted['canonical_response_text'], target['answer'])
                if adapted['accepted'] else
                {'correct': False, 'reason': 'adapter_rejected', 'observed_answer': None})
            if raw != row['raw_strict'] or canonical != row['canonical_pipeline']:
                raise ValueError('published_score_differs_from_frozen_metric')
        family_counts = Counter(r['source_family_id'] for r in report['per_question'])
        if set(family_counts) != set(FAMILIES) or set(family_counts.values()) != {12}:
            raise ValueError('all_four_related_families_required')
        reports[side], answer_maps[side], report_maps[side] = report, answers, scored

    pairs = []
    for case_id in sorted({key[0] for key in answer_maps['before']}):
        example = report_maps['before'][(case_id, ARMS[0])]
        row = {key: example[key] for key in ('case_id', 'source_family_id', 'language', 'expected_answer')}
        for side in ('before', 'after'):
            for arm in ARMS:
                answer = answer_maps[side][(case_id, arm)]
                score = report_maps[side][(case_id, arm)]
                prefix = side + '_' + arm
                row[prefix + '_raw_text'] = answer['response_text']
                row[prefix + '_canonical_text'] = answer['answer_adapter']['canonical_response_text']
                row[prefix + '_adapter_accepted'] = answer['answer_adapter']['accepted']
                for metric in METRICS:
                    row[prefix + '_' + metric + '_correct'] = score[metric]['correct']
                    row[prefix + '_' + metric + '_reason'] = score[metric]['reason']
        a, b = (answer_maps[s][(case_id, ARMS[0])] for s in ('before', 'after'))
        if any(a[k] != b[k] for k in ('request_sha256', 'response_sha256', 'response_text', 'canonical_key')):
            raise ValueError('raw_control_changed_between_product_versions')
        pairs.append(row)
    transitions = {}
    for metric in METRICS:
        before_key = 'before_managed_memory_rag_' + metric + '_correct'
        after_key = 'after_managed_memory_rag_' + metric + '_correct'
        transitions[metric] = {'improved': [r['case_id'] for r in pairs if not r[before_key] and r[after_key]],
            'regressed': [r['case_id'] for r in pairs if r[before_key] and not r[after_key]],
            'unchanged': [r['case_id'] for r in pairs if r[before_key] == r[after_key]]}
    resources = stage.reader.read_json(ROOT / 'reports/resources.json')
    saved_cost = dict(resources['new_study_unique_generation_cost'])
    checked_cost = dict(stage.cost(outcomes, intents))
    saved_wall = saved_cost.pop('answer_path_wall_seconds_sum')
    checked_wall = checked_cost.pop('answer_path_wall_seconds_sum')
    # The runtime sums a set of keys: ordering can change a few floating point bits.
    if (saved_cost != checked_cost or saved_wall is None or checked_wall is None
            or not math.isclose(saved_wall, checked_wall, rel_tol=1e-12, abs_tol=1e-8)):
        raise ValueError('unique_generation_cost_changed')
    result = {'version': 'complete-unconstrained-state-v3-derived-analysis-v1',
        'study_protocol_sha256': PROTOCOL_SHA, 'main_freeze_sha256': sha(ROOT / 'main_freeze.json'),
        'scorer_code_sha256': STAGE_SHA, 'analysis_code_sha256': sha(__file__),
        'complete_logical_rows': 96, 'unique_generations': 57,
        'raw_controls_identical_across_versions': True, 'new_model_calls': 0,
        'new_human_reviews': 0, 'clinical_or_natural_CBT_effect_claim': False,
        'material': 'four exposed constructed families; three variants and two languages; not 24 independent cases',
        'by_version': {side: {'arms': report['arms'], 'paired': report['paired_by_metric'],
            'by_family': grouped(report['per_question'], 'source_family_id'),
            'by_language': grouped(report['per_question'], 'language'),
            'descriptive_fact_metrics': {arm: {metric: fact_metrics(report['per_question'], arm, metric)
                                              for metric in METRICS} for arm in ARMS}}
            for side, report in reports.items()},
        'managed_before_after_transitions': transitions, 'resources': resources}
    output.mkdir(mode=0o700, parents=True)
    stage.reader._write_new(output / 'analysis.json', result)
    stage.reader._write_new(output / 'paired_original_answers.json', pairs)
    with (output / 'paired_original_answers.csv').open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader(); writer.writerows(pairs)
    manifest = {'version': result['version'], 'files': [
        {'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(output.iterdir())]}
    stage.reader._write_new(output / 'analysis_manifest.json', manifest)
    print(json.dumps({'analysis_directory': str(output), 'analysis_sha256': sha(output / 'analysis.json'),
        'logical_rows': 96, 'unique_generations': 57, 'new_model_calls': 0}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
        default=Path('/content/cbt-state-v3-paper-analysis-20261112-v1'))
    derive(parser.parse_args().output.resolve())
