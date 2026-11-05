#!/usr/bin/env python3
"""Render a completed objective run using DERIVED metadata only; never run a model.

Inputs come from the controlled Colab exporter. Benchmark histories, questions,
options and reference answers are deliberately excluded. --fixture requires an
explicit fixture_only marker and cannot produce a report labelled actual data.
"""
import argparse
import collections
import csv
import datetime as dt
import hashlib
import html
import json
import math
import re
from pathlib import Path

VERSION = 'derived-objective-report-v1'
ARMS = ('short_term', 'raw_history_rag', 'managed_memory_rag')
ARM_LABEL = dict(zip(ARMS, ('近期上下文', '原始历史检索', '管理记忆检索')))
DATASETS = {'memoryagentbench_conflict': 'MemoryAgentBench conflict', 'personamem_v1': 'PersonaMem-v1 32k'}
EXPECTED = {'memoryagentbench_conflict': (800, 1), 'personamem_v1': (589, 20)}
EXECUTION_COMMIT = '1511706b3a4b0ac71416998b5b4d752a13c4e313'
PLAN_ID = '940a2b3d8fa02aa683f7855f94931ff1d9710c7835b598c70fc201582ae8d568'
AUDIT_CHECKS = {'frozen_execution_commit', 'suite_plan_identity', 'all_4167_planned_rows_retained', 'all_shard_journals_validated_readonly', 'attempt_identity_qualified_by_shard', 'no_cross_shard_deduplication', 'canonical_cost_not_sum_of_logical_arm_rows', 'original_final_score_grid_and_values', 'original_final_costs'}
REQUIRED = ('suite_summary.json', 'suite_plan.json', 'public_metadata.json', 'per_question_rows.jsonl', 'cost_audit.json')
OPTIONAL = ('launcher_manifest.json', 'per_attempt_costs.jsonl', 'per_shard_summary.json')
FORBIDDEN = {'history', 'histories', 'source_history', 'messages', 'prompts', 'query', 'question', 'questions', 'options', 'references', 'accepted_answers', 'correct_option', 'gold', 'api_key', 'access_token', 'refresh_token', 'authorization', 'cookie', 'cookies', 'secret', 'secret_value', 'credentials'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fail(message):
    raise ValueError(message)


def read_json(path):
    def reject(value):
        fail('Non-finite JSON number: ' + value)
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=reject)


def inspect_derived(value, path='root'):
    if isinstance(value, dict):
        for key, item in value.items():
            count_only = key.casefold() in ('question', 'questions') and type(item) is int
            identity_only = key.casefold() == 'questions' and isinstance(item, dict) and set(item) == {'name', 'bytes', 'sha256'}
            if key.casefold() in FORBIDDEN and not (count_only or identity_only):
                fail('Controlled benchmark/secret field is not allowed: ' + path + '.' + key)
            inspect_derived(item, path + '.' + key)
    elif isinstance(value, list):
        for item in value:
            inspect_derived(item, path + '[]')
    elif isinstance(value, float) and not math.isfinite(value):
        fail('Non-finite value: ' + path)


def aggregate(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row['group_id']].append(row['score'])
    n = len(rows)
    if not n:
        fail('Empty aggregate')
    return {
        'questions': n, 'groups': len(groups), 'correct': sum(r['score'] for r in rows),
        'micro': sum(r['score'] for r in rows) / n,
        'group_equal': sum(sum(g) / len(g) for g in groups.values()) / len(groups),
        'strict': sum(r['strict_committed_score'] for r in rows) / n,
        'exact': sum(r['exact_alias_or_option_score'] for r in rows) / n,
        'technical': sum(not r['available'] for r in rows),
        'unresolved': sum(r['available'] and r['score'] == 0 for r in rows),
        'cue': sum(bool(r.get('official_positive_with_abstention_cue')) for r in rows),
        'not_strict': sum(bool(r.get('official_positive_not_strict')) for r in rows),
    }


def validate(data, fixture=False):
    summary, plan, meta, rows = (data[k] for k in REQUIRED[:4])
    if bool(meta.get('fixture_only')) != fixture:
        fail('--fixture and public_metadata.fixture_only must both be true for fixture reports')
    if summary.get('status') != 'complete' or summary.get('final_effectiveness_result') is not True:
        fail('Final report requires a complete frozen suite summary')
    if meta.get('complete') is not True:
        fail('Exporter has not marked this run complete')
    if summary.get('clinical_claim') or summary.get('confirmatory_claim'):
        fail('Unexpected clinical or confirmatory claim')
    if summary.get('suite_plan_sha256') != meta.get('suite_plan_sha256'):
        fail('Suite plan identity mismatch')
    expected_cases = meta['planned_cases']
    if not fixture and (expected_cases != 1389 or meta['planned_arm_rows'] != 4167):
        fail('Production report requires the frozen 1389-case / 4167-arm census')
    if not fixture and (meta.get('execution_commit') != EXECUTION_COMMIT or meta.get('suite_plan_sha256') != PLAN_ID):
        fail('Unexpected production execution/protocol identity')
    if not fixture:
        audit = data['cost_audit.json']
        checks = audit.get('independent_audit_checks', {})
        if set(checks) != AUDIT_CHECKS or not all(value is True for value in checks.values()):
            fail('Independent source/cost audit is incomplete')
        if audit.get('intents_without_outcome') != 0 or audit.get('saved_responses_without_success_outcome') != 0:
            fail('Unresolved provider attempts prohibit a final report')
        if meta.get('schema') != 'capstone-derived-readonly-v1' or meta.get('status') != 'complete' or meta.get('final_scores_available') is not True:
            fail('Unexpected or incomplete exporter schema')
        if summary.get('planned_cases') != meta['planned_cases'] or summary.get('planned_arm_rows') != meta['planned_arm_rows'] or plan.get('all_cases') != meta['planned_cases'] or plan.get('all_arm_rows') != meta['planned_arm_rows']:
            fail('Planned grid differs between source identities')
        plan_hash = sha(json.dumps({k: v for k, v in plan.items() if k != 'suite_plan_sha256'}, sort_keys=True, ensure_ascii=False).encode())
        if plan.get('suite_plan_sha256') != meta['suite_plan_sha256'] or plan_hash != meta['suite_plan_sha256']:
            fail('Frozen plan payload checksum mismatch')
        integer_keys = ('recorded_provider_attempt_intents', 'intents_without_outcome', 'successful_unique_provider_calls', 'failed_provider_attempts', 'failed_attempts_without_token_usage', 'successful_calls_without_token_usage', 'all_recorded_valid_response_count', 'saved_responses_without_success_outcome', 'latency_observation_count', 'successful_calls_without_latency')
        for key in integer_keys:
            if type(audit.get(key)) is not int or audit[key] < 0:
                fail('Missing or invalid independent cost count: ' + key)
        for key in ('sum_successful_api_latency_seconds',):
            if type(audit.get(key)) not in (int, float) or not math.isfinite(audit[key]) or audit[key] < 0:
                fail('Missing or invalid independent latency: ' + key)
        median = audit.get('median_successful_api_latency_seconds', 'missing')
        if audit['latency_observation_count'] == 0:
            if median is not None:
                fail('Missing-latency median must remain unknown')
        elif type(median) not in (int, float) or not math.isfinite(median) or median < 0:
            fail('Missing or invalid independent latency median')
        if audit['recorded_provider_attempt_intents'] != audit['successful_unique_provider_calls'] + audit['failed_provider_attempts'] + audit['intents_without_outcome']:
            fail('Independent attempt accounting does not reconcile')
        if audit['all_recorded_valid_response_count'] != audit['successful_unique_provider_calls']:
            fail('Final valid-response and successful-outcome counts disagree')
        if audit.get('monetary_cost', 'missing') is not None or audit.get('cross_shard_deduplication') is not False:
            fail('Unexpected currency or cross-shard cost claim')
        for token in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            value = audit.get('reported_tokens_including_failed_attempts_when_supplied', {}).get(token)
            if type(value) is not int or value < 0:
                fail('Missing or invalid independent token field')
            for kind in ('successful_calls', 'failed_attempts'):
                value = audit.get('usage_field_missing_counts', {}).get(kind, {}).get(token)
                if type(value) is not int or value < 0:
                    fail('Missing or invalid unknown-token accounting')
        for key, original in summary['cost'].items():
            if key == 'boundary':
                continue
            if key not in audit:
                fail('Independent audit lacks frozen cost field: ' + key)
            actual = audit[key]
            if type(original) in (int, float):
                if type(actual) not in (int, float) or not math.isclose(original, actual, rel_tol=1e-12, abs_tol=1e-8):
                    fail('Independent and frozen costs disagree: ' + key)
            elif original != actual:
                fail('Independent and frozen costs disagree: ' + key)
    if len(rows) != meta['planned_arm_rows']:
        fail('Logical arm row count mismatch')
    cases = collections.defaultdict(dict)
    for row in rows:
        for field in ('case_id', 'dataset_key', 'task', 'group_id', 'arm', 'score', 'available', 'completion_status', 'canonical_request_key', 'strict_committed_score', 'exact_alias_or_option_score', 'pipeline_cause'):
            if field not in row:
                fail('Missing per-row field ' + field)
        arm = row['arm']
        if arm not in ARMS or arm in cases[row['case_id']]:
            fail('Unknown or duplicate case/arm')
        for field in ('score', 'strict_committed_score', 'exact_alias_or_option_score'):
            if type(row[field]) not in (int, float) or row[field] not in (0, 1):
                fail('Invalid binary score: ' + field)
        if row['strict_committed_score'] > row['exact_alias_or_option_score']:
            fail('Strict format score exceeds exact diagnostic score')
        if type(row['available']) is not bool:
            fail('Availability must be a boolean')
        if row['available']:
            answer = row.get('original_hypothesis')
            if row['completion_status'] != 'success' or not isinstance(answer, str) or not answer:
                fail('Available row lacks an original successful model response')
            if sha(answer.encode()) != row.get('response_text_sha256'):
                fail('Original response checksum mismatch')
        elif row['completion_status'] != 'exhausted_failure' or any(row[f] for f in ('score', 'strict_committed_score', 'exact_alias_or_option_score')):
            fail('Technical failures must be terminal and remain zero in every denominator')
        expected_cause = 'provider_technical_failure' if not row['available'] else 'none' if row['score'] else 'answer_incorrect_stage_unresolved'
        if row['pipeline_cause'] != expected_cause:
            fail('Unsupported causal attribution')
        cases[row['case_id']][arm] = row
    if len(cases) != expected_cases:
        fail('Case census mismatch')
    for case, arms in cases.items():
        if set(arms) != set(ARMS):
            fail('Incomplete arm grid: ' + case)
        for field in ('dataset_key', 'task', 'group_id'):
            if len({r[field] for r in arms.values()}) != 1:
                fail('Mismatched case metadata: ' + case)
    datasets = {r['dataset_key'] for r in rows}
    if not fixture and datasets != set(EXPECTED):
        fail('Unexpected production datasets')
    metrics = {}
    for dataset in sorted(datasets):
        selected = [r for r in rows if r['dataset_key'] == dataset]
        if dataset not in summary['datasets']:
            fail('Dataset absent from frozen summary')
        if set(plan['datasets'][dataset]['case_ids']) != {r['case_id'] for r in selected}:
            fail('Dataset case identities differ from frozen plan')
        if not fixture:
            observed = (len({r['case_id'] for r in selected}), len({r['group_id'] for r in selected}))
            if observed != EXPECTED[dataset]:
                fail('Dataset/group census mismatch: ' + dataset)
        for arm in ARMS:
            result = aggregate([r for r in selected if r['arm'] == arm])
            frozen = summary['datasets'][dataset]['conditions'][arm]
            checks = {'questions': result['questions'], 'technical_failures': result['technical'], 'accuracy_micro': result['micro'], 'accuracy_group_equal': result['group_equal']}
            for field, actual in checks.items():
                if not math.isclose(frozen[field], actual, rel_tol=0, abs_tol=1e-12):
                    fail('Derived/frozen score mismatch: ' + dataset + '/' + arm + '/' + field)
            metrics[(dataset, arm)] = result
        pair = summary['datasets'][dataset]['paired_differences']['managed_memory_rag_minus_raw_history_rag']
        deltas, grouped_deltas, shared = [], collections.defaultdict(list), 0
        for case, arms in cases.items():
            if arms[ARMS[0]]['dataset_key'] != dataset:
                continue
            raw, managed = arms['raw_history_rag'], arms['managed_memory_rag']
            delta = managed['score'] - raw['score']
            deltas.append(delta)
            grouped_deltas[raw['group_id']].append(delta)
            if raw['canonical_request_key'] == managed['canonical_request_key']:
                shared += 1
                if (raw['available'], raw.get('original_hypothesis'), raw['score']) != (managed['available'], managed.get('original_hypothesis'), managed['score']):
                    fail('A shared canonical response has contradictory logical rows')
        checks = {'mean_accuracy_delta_micro': sum(deltas) / len(deltas), 'mean_accuracy_delta_group_equal': sum(sum(g) / len(g) for g in grouped_deltas.values()) / len(grouped_deltas), 'wins': sum(d > 0 for d in deltas), 'ties': sum(d == 0 for d in deltas), 'losses': sum(d < 0 for d in deltas), 'identical_prompt_shared_response_cases': shared}
        for field, actual in checks.items():
            if not math.isclose(pair[field], actual, rel_tol=0, abs_tol=1e-12):
                fail('Frozen paired contrast disagrees with rows: ' + dataset + '/' + field)
        if not fixture:
            all_shared = shared == len(deltas)
            if all_shared and (pair.get('result_type') != 'by_construction_control_equivalence' or pair.get('interval_status') != 'blocked_by_identical_inputs_and_shared_responses' or pair.get('group_equal_95_percent_interval') is not None or pair.get('micro_95_percent_interval') is not None):
                fail('Shared-response control cannot carry empirical efficacy intervals')
            if len(grouped_deltas) == 1 and not all_shared and (pair.get('interval_status') != 'not_estimable_fewer_than_two_groups' or pair.get('group_equal_95_percent_interval') is not None or pair.get('micro_95_percent_interval') is not None):
                fail('One independent family cannot carry a group uncertainty interval')
            if dataset == 'personamem_v1' and shared != len(deltas):
                fail('Persona quote-control identity has changed')
            partition = plan['datasets'][dataset]['development_partition']
            partition_ids = {'development_exposed_questions': set(partition['development_case_ids']), 'question_heldout_shared_histories': set(partition['question_heldout_case_ids'])}
            if set.union(*partition_ids.values()) != {r['case_id'] for r in selected} or set.intersection(*partition_ids.values()):
                fail('Exposure partitions do not form a disjoint full census')
            for name, ids in partition_ids.items():
                if not ids:
                    continue
                frozen_part = summary['datasets'][dataset]['question_exposure_partitions'][name]
                for arm in ARMS:
                    result = aggregate([r for r in selected if r['arm'] == arm and r['case_id'] in ids])
                    frozen = frozen_part['conditions'][arm]
                    if frozen['questions'] != result['questions'] or not math.isclose(frozen['accuracy_group_equal'], result['group_equal'], rel_tol=0, abs_tol=1e-12):
                        fail('Exposure partition disagrees with rows')
    return summary, plan, meta, rows, cases, metrics


def pct(value):
    return '未知' if value is None else f'{value * 100:.2f}%'


def pp(value):
    return '未知' if value is None else f'{value * 100:+.2f} 个百分点'


def esc(value):
    return html.escape(str(value), quote=True)


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>' + esc(h) + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + esc(c) + '</td>' for c in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def cost_value(audit, summary, key):
    if key in audit:
        return audit[key]
    for name in ('cost', 'recomputed', 'independent_cost', 'global_cost', 'totals'):
        if key in audit.get(name, {}):
            return audit[name][key]
    return summary.get('cost', {}).get(key)


def render(valid, data, checksums, fixture=False):
    summary, plan, meta, rows, cases, metrics = valid
    audit = data['cost_audit.json']
    title = '合成小样例验证报告（非实验结果）' if fixture else 'Capstone：完整客观对照实验'
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">', '<title>' + title + '</title>', '<style>body{max-width:1220px;margin:36px auto;padding:0 20px;font:16px/1.65 system-ui,sans-serif;color:#172738;background:#f8fafc}h1{font-size:30px}h2{margin-top:36px;font-size:23px}h3{font-size:19px}table{border-collapse:collapse;width:100%;font-size:14px;background:white}th,td{border:1px solid #ccd6df;padding:9px;text-align:left;vertical-align:top}th{background:#edf2f7}code,pre{overflow-wrap:anywhere}pre{white-space:pre-wrap;font:13px/1.5 ui-monospace,monospace}details{margin:8px 0}summary{cursor:pointer}.scroll{overflow-x:auto}.note{background:#eaf2fa;padding:14px;border-left:4px solid #275d88}.warn{background:#fff3da;padding:14px;border-left:4px solid #c78b1c}input,select,button{padding:8px;margin:5px;border:1px solid #a9bac9;border-radius:4px}input{min-width:250px}.muted{color:#526274}.pill{display:inline-block;padding:3px 9px;background:#e4edf5;border-radius:12px}.answer{max-width:400px}.cases td{min-width:105px}.cases td:first-child{min-width:190px}a{color:#175c9b}</style>', '<h1>' + title + '</h1>']
    parts += ['<p class="note">已核验 ' + str(len(cases)) + ' 道题 × 3 臂 = ' + str(len(rows)) + ' 条终态结果。技术失败计入分母；本报告从导出的逐题成绩独立复算官方准确率。'+ ('所有数值均为测试夹具，不得作为效果证据。' if fixture else '全量模型与数据执行位于 hosted Colab；本生成器仅处理脱敏成绩元数据和模型生成的回答。') + '</p>']
    parts += ['<p>主问题：在相同模型、同一历史来源、提问和权限政策及相同字符预算下，管理记忆相对原始历史检索是否改善公开任务回答。字符预算不等同于精确 token 预算。该实验衡量公开问答表现，不能证明 CBT 对话收益或临床疗效。</p>']
    identity_rows = []
    for dataset, metadata in plan.get('datasets', {}).items():
        settings = metadata.get('settings', {})
        source = plan.get('source_identities', {}).get(dataset, {})
        identity_rows.append((DATASETS.get(dataset, dataset), source.get('data_revision', '夹具 / 未提供'), settings.get('model', '夹具 / 未提供'), json.dumps(settings.get('accepted_provider_models', []), ensure_ascii=False), settings.get('backend', '未提供'), settings.get('max_context_chars', '未提供'), f"{settings.get('short_chars', '?')} / {settings.get('long_chars', '?')}", settings.get('max_response_tokens', '未提供')))
    parts += ['<h2>冻结执行口径</h2>', '<p>可执行代码：<code>' + esc(meta.get('execution_commit', '合成夹具')) + '</code><br>套件协议：<code>' + esc(meta.get('suite_plan_sha256')) + '</code></p>', table(('数据集', '数据 revision', '请求模型', '允许的 provider 模型', '检索', '上下文上限：字符', '近期 / 旧历史：字符', '回答上限：token'), identity_rows), '<p class="muted">真实返回模型及其实际调用数：' + esc(json.dumps(cost_value(audit, summary, 'reported_provider_models'), ensure_ascii=False)) + '。各数据集内三臂共享生成参数；不同数据集的输出上限按冻结协议分别规定，不混作单一跨数据集效果。</p>']
    parts += ['<h2>官方逐臂成绩</h2>']
    for dataset in sorted(summary['datasets']):
        if not any(d == dataset for d, a in metrics):
            continue
        results = []
        for arm in ARMS:
            m = metrics[(dataset, arm)]
            results.append((ARM_LABEL[arm], f"{m['correct']}/{m['questions']}", pct(m['micro']), pct(m['group_equal']), pct(m['exact']), pct(m['strict']), m['technical']))
        parts += ['<h3>' + esc(DATASETS.get(dataset, dataset)) + '</h3>', table(('臂', '官方答对 / 全部', '逐题准确率', '组均衡准确率', '精确文本/选项诊断', '严格提交格式诊断', '技术失败'), results)]
        pair = summary['datasets'][dataset].get('paired_differences', {}).get('managed_memory_rag_minus_raw_history_rag', {})
        interval = pair.get('group_equal_95_percent_interval')
        parts += [table(('主对照：管理 − 原始', '逐题差值', '组均衡差值', '组层面 95% 区间', '赢 / 平 / 输', '相同输入共享响应'), [(pair.get('result_type', '未知'), pp(pair.get('mean_accuracy_delta_micro')), pp(pair.get('mean_accuracy_delta_group_equal')), ' 至 '.join(pp(value) for value in interval) if interval is not None else pair.get('interval_status', '不可估计'), f"{pair.get('wins', '?')} / {pair.get('ties', '?')} / {pair.get('losses', '?')}", str(pair.get('identical_prompt_shared_response_cases', '?')))])]
        if dataset == 'memoryagentbench_conflict':
            parts += ['<p class="warn">800 道题包含 fact_sh / fact_mh 和不同长度变体，保守视为 1 个 fact-consolidation family。题量不能作为 800 个独立人；组间不确定性无法从 1 组估计。管理臂执行的是有原文来源的确定性版本槽处理，不包含自然语言语义抽取 API，不能推广为完整语义记忆或 CBT 效果。</p>']
        elif dataset == 'personamem_v1':
            parts += ['<p class="warn">589 道题归入 20 个 persona/connected groups；执行分片数不是独立用户数。本任务管理臂是原话引用控制：与原始历史臂使用相同输入并共享 canonical 回答。两臂差值为零具有构造原因，不能解释为管理有效、统计等效或已验证版本管理。</p>']
    task_rows = []
    for dataset in sorted({r['dataset_key'] for r in rows}):
        for task in sorted({r['task'] for r in rows if r['dataset_key'] == dataset}):
            selected = [r for r in rows if r['dataset_key'] == dataset and r['task'] == task]
            task_rows.append((DATASETS.get(dataset, dataset), task, len(selected) // 3, *(pct(aggregate([r for r in selected if r['arm'] == arm])['micro']) for arm in ARMS)))
    parts += ['<h2>任务与长度分层</h2>', table(('数据集', '发布任务 / 长度', '题数', *(ARM_LABEL[a] for a in ARMS)), task_rows)]
    type_rows = []
    for kind in ('sh', 'mh'):
        selected = [r for r in rows if r['dataset_key'] == 'memoryagentbench_conflict' and re.fullmatch('factconsolidation_' + kind + r'_(6k|32k|64k|262k)', r['task'])]
        if selected:
            type_rows.append(('fact_' + kind, len(selected) // 3, *(pct(aggregate([r for r in selected if r['arm'] == arm])['micro']) for arm in ARMS)))
    if type_rows:
        parts += [table(('MAB 类型汇总（各长度仍相关）', '题数', *(ARM_LABEL[a] for a in ARMS)), type_rows)]
    parts += ['<h2>错误诊断及其边界</h2>', '<p>以下是可观察诊断，不是对失败原因的独立人工标注。回答错误且调用成功时，抽取 → 更新 → 检索 → 上下文裁剪 → 回答的因果阶段一律保留 unresolved。检索/裁剪计数和来源 ID 只能证明路径观察，答案子串命中不能证明错误由某一阶段导致。</p>']
    parts += [table(('阶段', '本次实际覆盖的路径', '未建立的证据'), [
        ('抽取 / 表示', 'MAB：确定性句法解析；Persona：保留原文的引用分块；没有语义抽取 API 调用', '未验证生产 CBT 自然语言目标、作业或个人事实的语义抽取质量'),
        ('更新 / 版本', 'MAB：按 source_order 生成版本视图；Persona：原话引用控制', '未验证 CBT 目标或作业状态的自然语言更新、更正理解与真人确认'),
        ('检索', '共同检索器；按主体范围检索，保留来源 ID 路径观察', '缺少独立证据标注，无法由检索 ID 断定已召回充分答案证据'),
        ('上下文裁剪', '共同字符预算；保留 included / dropped 来源 ID', '字符不等于精确 token；未独立标注裁剪后证据充分性'),
        ('回答', '真实 provider 回答；冻结的官方确定性评分与格式诊断', '未建立独立语义评审、真实 CBT 帮助或临床改善证据'),
    ]), '<p class="muted">该表说明公开任务中实际观察的路径，不表示生产 CBT 系统的五个阶段均已得到效果验证。所有内容性答错的因果阶段仍为 unresolved；技术失败仅标记调用终态，不能由此推断记忆流程的因果失效。</p>']
    error_rows = [(DATASETS.get(d, d), ARM_LABEL[a], m['unresolved'], m['technical'], m['cue'], m['not_strict']) for (d, a), m in sorted(metrics.items())]
    parts += [table(('数据集', '臂', '答错：阶段 unresolved', '技术失败', '官方通过但含弃答线索', '官方通过但非严格提交'), error_rows), '<p class="muted">MemoryAgentBench 官方使用 substring_exact_match；包含弃答措辞或冗长矛盾文本也可能满足子串评分。精确文本/选项和严格提交仅为冻结的次级格式诊断，不是独立语义金标准，也不取代官方指标。弃答措辞检查是词法规则，可能误报。</p>']
    tokens = cost_value(audit, summary, 'reported_tokens_including_failed_attempts_when_supplied') or {}
    def display(value):
        return '未知 / 未提供' if value is None else json.dumps(value, ensure_ascii=False)
    cost_keys = [('记录的 provider 尝试意图', 'recorded_provider_attempt_intents'), ('尚无终态的尝试意图', 'intents_without_outcome'), ('实际成功 canonical 调用', 'successful_unique_provider_calls'), ('失败尝试', 'failed_provider_attempts'), ('成功调用缺 usage', 'successful_calls_without_token_usage'), ('失败尝试缺 usage', 'failed_attempts_without_token_usage'), ('成功 API 延迟合计（秒）', 'sum_successful_api_latency_seconds'), ('成功 API 延迟中位数（秒）', 'median_successful_api_latency_seconds'), ('已核定货币费用', 'monetary_cost')]
    parts += ['<h2>实际调用、token 与负担</h2>', table(('独立成本核算', '数值'), [(label, display(cost_value(audit, summary, key))) for label, key in cost_keys] + [(key, display(tokens.get(key))) for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')]), '<p>成本按 canonical 实际尝试核算，不累加共享响应的逻辑臂 usage。跨分片没有去重，相同请求若跨分片重复发生仍分别记费。缺失 usage 为未知，不补零；未冻结并核验价格时不估算金额。API 延迟不等于端到端耗时，未涵盖所有准备、检索和索引操作。中位数必须由全部独立成功调用计算，不平均分片中位数。</p>', '<p>旧 pilots 的 61 次调用 / 161,445 reported tokens 单独保留，不并入本次成本；旧数据是负向或不充分证据，协议、提示和输出上限不同，不能据新旧分数差宣称优化的因果收益。</p>']
    dataset_cost_rows = []
    for dataset, value in audit.get('per_dataset', {}).items():
        dataset_cost_rows.append((DATASETS.get(dataset, dataset), value.get('recorded_provider_attempt_intents'), value.get('successful_unique_provider_calls'), value.get('failed_provider_attempts'), value.get('reported_tokens_including_failed_attempts_when_supplied', {}).get('total_tokens'), value.get('median_successful_api_latency_seconds')))
    if dataset_cost_rows:
        parts += [table(('数据集实际成本（无臂重复计费）', '尝试意图', '成功调用', '失败尝试', 'reported total tokens', '成功 API 秒数中位数'), dataset_cost_rows)]
    missing_usage = audit.get('usage_field_missing_counts', {})
    if missing_usage:
        parts += [table(('缺失 usage 字段的尝试数', 'prompt_tokens', 'completion_tokens', 'total_tokens'), [(kind, *(missing_usage.get(key, {}).get(token, '未知') for token in ('prompt_tokens', 'completion_tokens', 'total_tokens'))) for kind, key in (('成功调用', 'successful_calls'), ('失败尝试', 'failed_attempts'))])]
    exposure = []
    for dataset, ds in summary['datasets'].items():
        for partition, val in ds.get('question_exposure_partitions', {}).items():
            condition = val.get('conditions', {}).get(ARMS[0], {})
            exposure.append((DATASETS.get(dataset, dataset), partition, condition.get('questions', '未知'), condition.get('independent_groups', '未知'), *(pct(val.get('conditions', {}).get(a, {}).get('accuracy_group_equal')) for a in ARMS)))
    parts += ['<h2>既往暴露与独立性</h2>', table(('数据集', '题目暴露分区', '题数', '组数', *(ARM_LABEL[a] + '：组均衡' for a in ARMS)), exposure), '<p>此前小样例已暴露题目按冻结规则标记；同一规范化问题的长度变体归入同一暴露分区。heldout 指问题层面留出，仍与开发问题共享历史；不能称作独立历史或独立参与者外部验证。主结果同时保留全体与预先定义的暴露分区。</p>']
    parts += ['<h2>CBT 过程与最终效果：当前证据</h2>', table(('层次', '当前证据 / 人数', '可作结论'), [('工程流程', '以原始冻结工件和测试记录为准', '工程正确性与公开任务效果分别报告'), ('公开客观任务', f'{len(cases)} 题；上述独立性限制', '仅支持所测公开任务内的回答比较'), ('独立 / CBT 专业评审', '0 次', '未验证 CBT 目标、作业跟进、纠正质量和治疗适配性'), ('真人四会话试用', '0 位参与者', '未验证实际对话收益、重复追问或打扰负担'), ('自然核对 / 重复核对负担', '未测量（not measured）', '不能将未测量解释为零次打扰或零次重复追问'), ('临床最终效果', '0 项真人症状结局', '不能声称症状改善或临床有效')]), '<p>删除、到期、自然核对次数、重复追问、作业完成和治疗目标推进未由这两个公开任务直接测量；不可用模型评分代替真人体验或专业评审。</p>']
    payload = []
    for case, arms in sorted(cases.items()):
        r = arms[ARMS[0]]
        payload.append({'id': case, 'dataset': r['dataset_key'], 'task': r['task'], 'group': r['group_id'], 'arms': {a: {k: v for k, v in row.items() if k not in ('case_id', 'dataset_key', 'task', 'group_id', 'arm')} for a, row in arms.items()}})
    encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    parts += ['<h2>全部逐题对照</h2><p>表中“题目”使用发布 case ID；受控来源的题干、历史、参考答案未包含在本报告。展开每臂可查看原始模型回答、响应校验值和来源路径观察。技术失败显示为 0，保留在分母。</p><div><input id="search" placeholder="搜索 case ID、任务或模型回答"><select id="dataset"><option value="">全部数据集</option></select><select id="task"><option value="">全部任务 / 长度</option></select><select id="outcome"><option value="">全部结果</option><option value="win">管理赢原始</option><option value="loss">管理输原始</option><option value="tie">管理与原始相同</option><option value="technical">包含技术失败</option><option value="notstrict">包含官方通过但非严格提交</option></select></div><p id="count" class="muted"></p><button id="prev">上一页</button><button id="next">下一页</button><div class="scroll"><table class="cases"><thead><tr><th>case ID / 任务 / 组</th><th>近期上下文</th><th>原始历史检索</th><th>管理记忆检索</th></tr></thead><tbody id="cases"></tbody></table></div>', '<script type="application/json" id="case-data">' + encoded + '</script>', '<script>' + JS + '</script>']
    identity = {'report_builder_version': VERSION, 'exported_at_utc': meta.get('exported_at_utc'), 'execution_commit': meta.get('execution_commit'), 'suite_plan_sha256': meta.get('suite_plan_sha256'), 'source_artifacts_sha256': meta.get('source_artifacts_sha256'), 'reporting_helper_sha256': meta.get('reporting_helper_sha256'), 'input_file_sha256': checksums, 'provenance': meta.get('provenance'), 'limitations': meta.get('limitations')}
    parts += ['<h2>来源与核算凭据</h2><p>身份、失败和成本以原始冻结工件为准。导出元数据的 SHA-256 用于文件完整性核查；单独的校验值不等于独立真实性认证。</p><details><summary>执行身份与导出校验</summary><pre>' + esc(json.dumps(identity, ensure_ascii=False, indent=2)) + '</pre></details><details><summary>独立成本审计</summary><pre>' + esc(json.dumps(audit, ensure_ascii=False, indent=2)) + '</pre></details></html>']
    return ''.join(parts)


JS = r"""
const data=JSON.parse(document.getElementById('case-data').textContent), arms=['short_term','raw_history_rag','managed_memory_rag'];
const $=id=>document.getElementById(id); let page=0; const size=40;
function el(tag,text){let e=document.createElement(tag);if(text!==undefined)e.textContent=String(text);return e;}
for(const id of ['dataset','task'])for(const value of [...new Set(data.map(r=>r[id]))].sort()){let o=el('option',value);o.value=value;$(id).append(o);}
function render(){let q=$('search').value.toLowerCase(),d=$('dataset').value,t=$('task').value,o=$('outcome').value;
let selected=data.filter(r=>{if(d&&r.dataset!==d||t&&r.task!==t)return false;let a=r.arms,delta=a.managed_memory_rag.score-a.raw_history_rag.score;
if(o==='win'&&delta<=0||o==='loss'&&delta>=0||o==='tie'&&delta!==0||o==='technical'&&!arms.some(k=>!a[k].available)||o==='notstrict'&&!arms.some(k=>a[k].official_positive_not_strict))return false;
return !q||[r.id,r.task,r.group,...arms.map(k=>a[k].original_hypothesis||'')].join(' ').toLowerCase().includes(q);});
page=Math.min(page,Math.max(0,Math.ceil(selected.length/size)-1));$('cases').replaceChildren();
for(const r of selected.slice(page*size,(page+1)*size)){let tr=el('tr'),meta=el('td');meta.append(el('strong',r.id),el('p',r.dataset+' / '+r.task),el('p','group: '+r.group));tr.append(meta);
for(const arm of arms){let a=r.arms[arm],td=el('td');td.append(el('strong','官方 '+a.score+' · 严格 '+a.strict_committed_score),el('p',a.completion_status),el('p','阶段: '+a.pipeline_cause));let detail=el('details');detail.append(el('summary','原始模型回答与路径'));let answer=el('pre',a.original_hypothesis===null||a.original_hypothesis===undefined?'无可用回答（技术失败）':a.original_hypothesis);answer.className='answer';detail.append(answer,el('p','模型: '+(a.provider_model||'未知')),el('p','response sha256: '+(a.response_text_sha256||'无')),el('p','canonical: '+a.canonical_request_key),el('pre',JSON.stringify(a.source_path_observations||{},null,2)));td.append(detail);tr.append(td);} $('cases').append(tr);}
$('count').textContent='筛选 '+selected.length+' / '+data.length+' 题；第 '+(page+1)+' / '+Math.max(1,Math.ceil(selected.length/size))+' 页，每页 '+size+' 题';$('prev').disabled=page===0;$('next').disabled=(page+1)*size>=selected.length;}
for(const id of ['search','dataset','task','outcome'])$(id).addEventListener('input',()=>{page=0;render();});$('prev').onclick=()=>{page--;render();};$('next').onclick=()=>{page++;render();};render();
"""


def build(input_dir, output_dir, fixture=False):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if output_dir.exists():
        fail('Refusing to overwrite an existing report directory')
    data, checksums = {}, {}
    for filename in REQUIRED + OPTIONAL:
        path = input_dir / filename
        if not path.exists():
            if filename in REQUIRED:
                fail('Missing derived input: ' + filename)
            continue
        if path.is_symlink():
            fail('Symlink inputs are not accepted')
        raw = path.read_bytes()
        checksums[filename] = sha(raw)
        if filename.endswith('.jsonl'):
            data[filename] = [json.loads(line, parse_constant=lambda v: fail('Non-finite JSON value')) for line in raw.decode('utf-8').splitlines() if line.strip()]
        else:
            data[filename] = read_json(path)
        inspect_derived(data[filename], filename)
    valid = validate(data, fixture)
    document = render(valid, data, checksums, fixture)
    summary, plan, meta, rows, cases, metrics = valid
    output_dir.mkdir(parents=True)
    (output_dir / 'report.html').write_text(document, encoding='utf-8')
    report_summary = {'report_builder_version': VERSION, 'report_builder_sha256': sha(Path(__file__).read_bytes()), 'built_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'fixture_only': fixture, 'execution_commit': meta.get('execution_commit'), 'suite_plan_sha256': meta.get('suite_plan_sha256'), 'verified_cases': len(cases), 'verified_arm_rows': len(rows), 'input_file_sha256': checksums, 'recomputed_metrics': {d: {a: metrics[(d, a)] for a in ARMS} for d in sorted({d for d, a in metrics})}, 'csv_formula_safety_prefix': True, 'raw_benchmark_histories_or_gold_loaded': False, 'model_calls_performed_by_builder': 0, 'human_or_clinical_results_inferred': False}
    (output_dir / 'report_summary.json').write_text(json.dumps(report_summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (output_dir / 'per_question_comparison.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case_id', 'dataset_key', 'task', 'group_id'] + [a + '_' + f for a in ARMS for f in ('official_score', 'strict_score', 'status', 'original_model_response')])
        for case, arms in sorted(cases.items()):
            base = arms[ARMS[0]]
            record = [case, base['dataset_key'], base['task'], base['group_id']]
            for arm in ARMS:
                r = arms[arm]
                answer = r.get('original_hypothesis') or ''
                if answer.startswith(('=', '+', '-', '@', '\t', '\r')):
                    answer = "'" + answer
                record += [r['score'], r['strict_committed_score'], r['completion_status'], answer]
            writer.writerow(record)
    hashes = {p.name: sha(p.read_bytes()) for p in sorted(output_dir.iterdir()) if p.is_file()}
    (output_dir / 'SHA256SUMS.json').write_text(json.dumps(hashes, indent=2) + '\n', encoding='utf-8')
    return report_summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--fixture', action='store_true')
    args = parser.parse_args()
    print(json.dumps(build(args.input_dir, args.output_dir, args.fixture), ensure_ascii=False, indent=2))
