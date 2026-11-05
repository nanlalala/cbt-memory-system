#!/usr/bin/env python3
"""Genuine-Colab-only posthoc report for the bounded v4 retrieval repair.

No provider call, Secret access or original-run mutation. Controlled replies are
processed only in hosted Colab; the derived-only ZIP contains no source/replies.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import zipfile

VERSION = "v4-retrieval-repair-posthoc-report-v1"
HELPER = "5e1f862f615a871a09e498da0199d43b43b09a3b745c827a6c8152aeeae89bdc"
CANDIDATE = "efd75116f985749a96a6070cb80c4b38d5c94dba"
SELECTION = "9d3ee5c84ec55ee970780ffcfab19dd3f693a521adf8b37e2e9f9a38e7902e9f"
ALIAS = "deepseek-v4-1-flash-260910"
ARMS = ("short_term", "raw_history_rag", "managed_memory_rag")
LABEL = dict(zip(ARMS, ("近期上下文", "原始历史检索", "版本管理记忆")))
TOKENS = ("prompt_tokens", "completion_tokens", "total_tokens")
RUN_FILES = ("protocol.json", "requests.jsonl", "logical_changed_pairs.jsonl", "executable_helper.py",
             "prepared_artifacts.json", "attempts.jsonl", "outcomes.jsonl", "public_summary.json",
             "public_per_question_scores.jsonl", "controlled_per_question_answers.jsonl", "SHA256SUMS.json")
PUBLIC_FIELDS = {"case_id", "arm", "dataset_key", "task", "group_id", "score", "old_v3_score", "strict_committed_score",
                 "exact_alias_or_option_score", "available", "completion_status", "provenance", "prompt_changed",
                 "old_prompt_sha256", "prompt_sha256", "new_request_key", "response_text_sha256", "error_cause"}
COST_FIELDS = {"recorded_provider_attempts", "successful_canonical_calls", "failed_attempts", "unresolved_attempts", "reported_tokens",
               "successful_calls_missing_usage_fields", "failed_attempts_missing_usage_fields", "reported_provider_models",
               "median_successful_api_latency_seconds", "sum_successful_api_latency_seconds", "monetary_cost", "boundary"}
SUMMARY_FIELDS = {"schema", "status", "protocol_sha256", "candidate_commit", "planned_cases", "planned_three_arm_rows", "logical_changed_rows",
                  "canonical_new_requests", "new_cost", "development_exposed", "independent_families", "clinical_claim",
                  "causal_stage_attribution", "conditions", "managed_minus_raw", "by_task", "v3_to_v4_by_arm"}


def sha(body):
    return hashlib.sha256(body).hexdigest()


def stable(value):
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def legacy_stable(value):
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode())


def data(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected regular controlled artifact")
    return path.read_bytes()


def json_read(path):
    return json.loads(data(path))


def read_rows(path):
    body = data(path)
    if body and not body.endswith(b"\n"):
        raise ValueError("Incomplete journal tail")
    return [json.loads(line) for line in body.splitlines() if line.strip()]


def require_colab():
    try:
        present = importlib.util.find_spec("google.colab") is not None
    except (ValueError, ModuleNotFoundError):
        present = False
    if not (Path("/content").is_dir() and os.environ.get("COLAB_RELEASE_TAG") and present):
        raise RuntimeError("Genuine hosted Colab required before any controlled input read")


def sealed(row):
    if row.get("record_sha256") != stable({k: v for k, v in row.items() if k != "record_sha256"}):
        raise ValueError("Attempt/outcome integrity differs")


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def counts(values):
    return {"questions": len(values), "old_correct": sum(r["old_v3_score"] for r in values),
            "new_correct": sum(r["score"] for r in values), "strict_correct": sum(r["strict_committed_score"] for r in values),
            "wins": sum(r["score"] > r["old_v3_score"] for r in values),
            "losses": sum(r["score"] < r["old_v3_score"] for r in values),
            "changed_input_rows": sum(r["prompt_changed"] for r in values),
            "reused_old_rows": sum(r["provenance"] == "unchanged_prompt_reused_old_response" for r in values)}


def public_schema(summary, public, controlled):
    if set(summary) != SUMMARY_FIELDS or set(summary["new_cost"]) != COST_FIELDS:
        raise ValueError("Unexpected field in safe summary/cost schema")
    for row in public:
        if set(row) != PUBLIC_FIELDS:
            raise ValueError("Unexpected field in safe per-question schema")
        if type(row["prompt_changed"]) is not bool:
            raise ValueError("Safe changed-input flag must be boolean")
    for row in controlled:
        if set(row) != PUBLIC_FIELDS | {"original_hypothesis", "provider_model"}:
            raise ValueError("Unexpected controlled answer field")
    cost = summary["new_cost"]
    for name in ("reported_tokens", "successful_calls_missing_usage_fields", "failed_attempts_missing_usage_fields"):
        if set(cost[name]) != set(TOKENS) or any(type(v) is not int or v < 0 for v in cost[name].values()):
            raise ValueError("Unexpected safe numeric usage schema")
    if set(cost["reported_provider_models"]) != {ALIAS} or type(cost["reported_provider_models"][ALIAS]) is not int:
        raise ValueError("Unexpected safe model count schema")
    if cost["boundary"] != "Only new canonical attempts; old reused replies are not billed again. Missing usage is unknown, not zero cost. API time is not wall time.":
        raise ValueError("Unexpected safe cost boundary text")
    if (summary["schema"] != "mab-v4-changed-input-development-response-v1" or summary["status"] != "complete"
            or summary["candidate_commit"] != CANDIDATE or summary["development_exposed"] is not True
            or summary["clinical_claim"] is not False or summary["independent_families"] != 1
            or summary["causal_stage_attribution"] != "unresolved_without_independent_evidence_labels"):
        raise ValueError("Unexpected safe summary identity fields")


def checked_module(name, path):
    if name in sys.modules:
        result = sys.modules[name]
        if Path(result.__file__).resolve() != path.resolve():
            raise ValueError("Foreign cached scoring module")
        return result
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def expected_aggregate(public):
    conditions = {}
    by_case = defaultdict(dict)
    for row in public:
        by_case[row["case_id"]][row["arm"]] = row
    for arm in ARMS:
        chosen = [row for row in public if row["arm"] == arm]
        conditions[arm] = {"questions": len(chosen), "official_correct": sum(r["score"] for r in chosen),
                           "official_accuracy": sum(r["score"] for r in chosen) / len(chosen),
                           "strict_format_correct": sum(r["strict_committed_score"] for r in chosen),
                           "new_response_rows": sum(r["provenance"] == "new_v4_response" for r in chosen),
                           "new_failure_rows": sum(r["completion_status"] == "provider_failure" for r in chosen),
                           "reused_old_response_rows": sum(r["provenance"] == "unchanged_prompt_reused_old_response" for r in chosen)}
    deltas = [r["managed_memory_rag"]["score"] - r["raw_history_rag"]["score"] for r in by_case.values()]
    primary = {"accuracy_delta": sum(deltas) / len(deltas), "wins": sum(d > 0 for d in deltas), "ties": sum(d == 0 for d in deltas),
               "losses": sum(d < 0 for d in deltas), "independent_families": 1, "confidence_interval": None,
               "interval_status": "not_estimable_one_family"}
    return {"conditions": conditions, "managed_minus_raw": primary}


def validate(run, repo):
    source_hashes = {name: sha(data(run / name)) for name in RUN_FILES}
    saved = json_read(run / "SHA256SUMS.json")
    for name, expected in saved.items():
        if Path(name).name != name or sha(data(run / name)) != expected:
            raise ValueError("Completed run checksum differs")
    if not set(RUN_FILES) - {"SHA256SUMS.json"} <= set(saved):
        raise ValueError("Completed run checksum manifest is incomplete")
    protocol = json_read(run / "protocol.json")
    if (protocol.get("protocol_sha256") != stable({k: v for k, v in protocol.items() if k != "protocol_sha256"})
            or protocol.get("candidate_commit") != CANDIDATE or protocol.get("source_only_selection_sha256") != SELECTION
            or protocol.get("helper_sha256") != HELPER or protocol.get("logical_changed_rows") != 206
            or protocol.get("max_response_tokens") != 128 or protocol.get("maximum_provider_attempts") != 206
            or protocol.get("temperature") != 0 or protocol.get("thinking_mode") != "disabled"
            or protocol.get("model") != "deepseek-v4.1-flash" or protocol.get("allowed_model") != ALIAS
            or protocol.get("api_base") != "https://tokendance.space/gateway/v1"):
        raise ValueError("Bounded response protocol identity differs")
    if sha(data(run / "executable_helper.py")) != HELPER:
        raise ValueError("Executed helper bytes differ")
    if subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip() != CANDIDATE or subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD"], check=False).returncode:
        raise ValueError("Scoring checkout differs from pinned candidate")
    for name in ("evaluation/objective-memory/score_objective_comparison.py", "evaluation/objective-memory/v3_response_diagnostics.py",
                 "evaluation/end-to-end-dialogue/execution_lock.py"):
        if sha(data(repo / name)) != protocol["code_sha256"][name]:
            raise ValueError("Pinned official/diagnostic scoring code differs")
    scorer = checked_module("score_objective_comparison", repo / "evaluation/objective-memory/score_objective_comparison.py")
    diagnostic = checked_module("v3_response_diagnostics", repo / "evaluation/objective-memory/v3_response_diagnostics.py")
    if Path(sys.modules["execution_lock"].__file__).resolve() != (repo / "evaluation/end-to-end-dialogue/execution_lock.py").resolve():
        raise ValueError("Foreign cached scoring dependency")
    prepared = json_read(run / "prepared_artifacts.json")
    if set(prepared) != {"protocol.json", "requests.jsonl", "logical_changed_pairs.jsonl", "executable_helper.py"} or any(source_hashes[name] != value for name, value in prepared.items()):
        raise ValueError("Prepared generation artifacts changed")
    requests = {}
    for row in read_rows(run / "requests.jsonl"):
        key = row["request_key"]
        if key in requests or stable(row["request"]) != key:
            raise ValueError("New canonical request differs")
        body = row["request"]
        if set(body) != {"messages", "model", "allowed_model", "api_base", "temperature", "thinking_mode", "max_response_tokens"} or {k:v for k,v in body.items() if k != "messages"} != {"model":"deepseek-v4.1-flash","allowed_model":ALIAS,"api_base":"https://tokendance.space/gateway/v1","temperature":0,"thinking_mode":"disabled","max_response_tokens":128}:
            raise ValueError("New frozen request policy differs")
        requests[key] = row["request"]
    if stable(requests) != protocol["canonical_requests_sha256"] or len(requests) != protocol["canonical_new_requests"]:
        raise ValueError("New request grid differs")
    attempts, outcomes = {}, {}
    for collection, name in ((attempts, "attempts.jsonl"), (outcomes, "outcomes.jsonl")):
        for row in read_rows(run / name):
            sealed(row)
            key = row["request_key"]
            if key in collection or key not in requests or row["protocol_sha256"] != protocol["protocol_sha256"]:
                raise ValueError("Actual canonical attempt identity differs")
            if name == "attempts.jsonl" and row.get("attempt_index") != 1:
                raise ValueError("Unexpected new retry")
            collection[key] = row
    if set(attempts) != set(outcomes) or set(outcomes) != set(requests) or len(attempts) > 206:
        raise ValueError("New calls are unresolved or incomplete")
    successes = []
    for row in outcomes.values():
        if row.get("status") != "success":
            raise ValueError("This report identity expects the observed all-success response run")
        response = row["response"]
        if response.get("provider_model") != ALIAS or response.get("finish_reason") != "stop" or not isinstance(response.get("hypothesis"), str) or not response["hypothesis"].strip():
            raise ValueError("Actual response identity differs")
        if type(response.get("latency_seconds")) not in (int, float) or not math.isfinite(response["latency_seconds"]) or response["latency_seconds"] < 0:
            raise ValueError("Actual latency differs")
        if any(type(response.get("usage", {}).get(k)) is not int or response["usage"][k] < 0 for k in TOKENS):
            raise ValueError("This observed complete-usage report has missing/invalid counters")
        successes.append(response)
    summary = json_read(run / "public_summary.json")
    if summary.get("status") != "complete" or summary.get("protocol_sha256") != protocol["protocol_sha256"] or summary.get("planned_three_arm_rows") != 2400 or summary.get("planned_cases") != 800:
        raise ValueError("Full counterfactual summary differs")
    if summary.get("logical_changed_rows") != 206 or summary.get("canonical_new_requests") != len(requests):
        raise ValueError("Saved changed/canonical request count differs")
    cost = summary["new_cost"]
    calculated_tokens = {k: sum(r["usage"][k] for r in successes) for k in TOKENS}
    if (cost.get("recorded_provider_attempts") != len(attempts) or cost.get("successful_canonical_calls") != len(successes)
            or cost.get("failed_attempts") != 0 or cost.get("unresolved_attempts") != 0 or cost.get("reported_tokens") != calculated_tokens
            or cost.get("reported_provider_models") != {ALIAS: len(successes)} or cost.get("monetary_cost") is not None):
        raise ValueError("Actual new-call cost reconciliation differs")
    if any(value != 0 for name in ("successful_calls_missing_usage_fields", "failed_attempts_missing_usage_fields") for value in cost[name].values()):
        raise ValueError("Observed complete-usage counters differ")
    latencies = [r["latency_seconds"] for r in successes]
    if not math.isclose(cost["sum_successful_api_latency_seconds"], sum(latencies), rel_tol=1e-12) or not math.isclose(cost["median_successful_api_latency_seconds"], statistics.median(latencies), rel_tol=1e-12):
        raise ValueError("Actual latency reconciliation differs")
    public = read_rows(run / "public_per_question_scores.jsonl")
    controlled = read_rows(run / "controlled_per_question_answers.jsonl")
    public_schema(summary, public, controlled)
    by_key = {(r["case_id"], r["arm"]): r for r in public}
    replies = {(r["case_id"], r["arm"]): r for r in controlled}
    if len(public) != 2400 or len(controlled) != 2400 or len(by_key) != 2400 or set(by_key) != set(replies):
        raise ValueError("Per-question 800 x 3 grid differs")
    cases = defaultdict(dict)
    pairs = {(r["case_id"], r["arm"]): r for r in read_rows(run / "logical_changed_pairs.jsonl")}
    if len(pairs) != 206 or stable(read_rows(run / "logical_changed_pairs.jsonl")) != protocol["logical_pairs_sha256"]:
        raise ValueError("Changed logical-pair identity differs")
    if len({case for case,arm in pairs}) != 154:
        raise ValueError("Changed case census differs")
    # Bind reused reply text to the actual original journals, not a provenance
    # label alone. Read all original full-source files only inside Colab.
    source_by_parent = defaultdict(dict)
    for text_path, expected in protocol["source_sha256"].items():
        path = Path(text_path)
        if not path.resolve().is_relative_to(Path("/content")) or sha(data(path)) != expected:
            raise ValueError("Original source/diagnostic artifacts changed")
        if path.name in {"prompts.jsonl", "responses.jsonl", "objective_scores.jsonl", "references.jsonl"} and path.parent.name.startswith("mab-"):
            source_by_parent[path.parent][path.name] = path
    old_replies, old_scores, old_prompts, references = {}, {}, {}, {}
    for files in source_by_parent.values():
        if set(files) != {"prompts.jsonl", "responses.jsonl", "objective_scores.jsonl", "references.jsonl"}:
            raise ValueError("Original response provenance files incomplete")
        original = {r["canonical_request_key"]: r for r in read_rows(files["responses.jsonl"])}
        for prompt in read_rows(files["prompts.jsonl"]):
            key = (prompt["case_id"], prompt["arm"])
            if key in old_replies:
                raise ValueError("Duplicate original logical response")
            old_replies[key] = original[prompt["canonical_request_key"]]["hypothesis"]
            old_prompts[key] = prompt
        for reference in read_rows(files["references.jsonl"]):
            if reference["case_id"] in references:
                raise ValueError("Duplicate frozen scoring reference")
            references[reference["case_id"]] = reference
        for row in read_rows(files["objective_scores.jsonl"]):
            old_scores[(row["case_id"], row["arm"])] = row["score"]
    if set(old_replies) != set(by_key) or set(old_scores) != set(by_key):
        raise ValueError("Original provenance grid differs")
    for key, row in by_key.items():
        reply = replies[key]
        if any(reply.get(name) != value for name, value in row.items()):
            raise ValueError("Controlled reply metadata differs from public scores")
        answer = reply.get("original_hypothesis")
        if not isinstance(answer, str) or reply.get("provider_model") != ALIAS or sha(answer.encode()) != row.get("response_text_sha256"):
            raise ValueError("Controlled answer identity differs")
        if any(type(row.get(name)) not in (int, float) or row[name] not in (0., 1.) for name in ("score", "old_v3_score", "strict_committed_score", "exact_alias_or_option_score")):
            raise ValueError("Unexpected score value")
        if row["old_v3_score"] != old_scores[key] or row.get("completion_status") != "success" or row.get("available") is not True:
            raise ValueError("Original/current score status differs")
        old_prompt = old_prompts[key]
        if any(row[name] != old_prompt[name] for name in ("dataset_key", "task", "group_id")) or row["old_prompt_sha256"] != old_prompt["prompt_sha256"]:
            raise ValueError("Displayed source/prompt metadata differs from original")
        reference = references[key[0]]
        if scorer.deterministic_score(old_replies[key], reference)[0] != row["old_v3_score"] or scorer.deterministic_score(answer, reference)[0] != row["score"]:
            raise ValueError("Official score differs from independent frozen-reference recomputation")
        extra = diagnostic.score_supplemental(answer, reference)
        if any(row[name] != extra[name] for name in ("strict_committed_score", "exact_alias_or_option_score")):
            raise ValueError("Supplemental score differs from independent recomputation")
        if key in pairs:
            pair = pairs[key]
            if (not row.get("prompt_changed") or row.get("provenance") != "new_v4_response" or row.get("new_request_key") != pair["request_key"]
                    or row["prompt_sha256"] != pair["prompt_sha256"] or row["old_prompt_sha256"] != pair["old_prompt_sha256"]
                    or row["prompt_sha256"] != legacy_stable(requests[pair["request_key"]]["messages"])
                    or answer != outcomes[pair["request_key"]]["response"]["hypothesis"]):
                raise ValueError("New reply provenance differs")
        elif row.get("prompt_changed") or row.get("provenance") != "unchanged_prompt_reused_old_response" or row.get("new_request_key") is not None or row["prompt_sha256"] != row["old_prompt_sha256"] or answer != old_replies[key]:
            raise ValueError("Unchanged original reply provenance differs")
        expected_error = "none" if row["score"] else "answer_incorrect_stage_unresolved"
        if row.get("error_cause") != expected_error:
            raise ValueError("Unsupported causal error attribution")
        cases[key[0]][key[1]] = reply
    if len(cases) != 800 or any(set(value) != set(ARMS) for value in cases.values()):
        raise ValueError("Full case/arm identities differ")
    aggregate = expected_aggregate(public)
    if summary["conditions"] != aggregate["conditions"] or summary["managed_minus_raw"] != aggregate["managed_minus_raw"]:
        raise ValueError("Safe aggregate schema/values differ")
    expected_tasks = {task: expected_aggregate([r for r in public if r["task"] == task]) for task in sorted({r["task"] for r in public})}
    if summary["by_task"] != expected_tasks:
        raise ValueError("Safe task aggregate schema/values differ")
    if set(summary["v3_to_v4_by_arm"]) != set(ARMS) or any(set(value) != {"old_correct", "new_correct", "wins", "losses"} for value in summary["v3_to_v4_by_arm"].values()):
        raise ValueError("Safe transition schema differs")
    recalculated = {arm: counts([r for r in public if r["arm"] == arm]) for arm in ARMS}
    for arm, metrics in recalculated.items():
        saved_arm = summary["conditions"][arm]
        if saved_arm["official_correct"] != metrics["new_correct"] or saved_arm["strict_format_correct"] != metrics["strict_correct"] or saved_arm["questions"] != 800:
            raise ValueError("Saved condition aggregate differs")
        transition = summary["v3_to_v4_by_arm"][arm]
        if any(transition[name] != metrics[name] for name in ("old_correct", "new_correct", "wins", "losses")):
            raise ValueError("Saved transition aggregate differs")
    deltas = [r["managed_memory_rag"]["score"] - r["raw_history_rag"]["score"] for r in cases.values()]
    primary = summary["managed_minus_raw"]
    if (primary["accuracy_delta"] != sum(deltas) / 800 or primary["wins"] != sum(d > 0 for d in deltas)
            or primary["ties"] != sum(d == 0 for d in deltas) or primary["losses"] != sum(d < 0 for d in deltas)
            or primary["confidence_interval"] is not None or primary["independent_families"] != 1):
        raise ValueError("Saved primary comparison differs")
    return protocol, summary, public, cases, recalculated, source_hashes


def table(headers, values):
    esc = lambda x: html.escape(str(x), quote=True)
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>' + esc(x) + '</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + esc(x) + '</td>' for x in row) + '</tr>' for row in values) + '</tbody></table></div>'


def render(summary, public, cases, metrics):
    body = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MAB v4 检索修复实际复测</title><style>body{font:16px/1.55 system-ui,sans-serif;max-width:1250px;margin:auto;padding:28px;color:#18302a;background:#f5f8f6}table{border-collapse:collapse;width:100%;background:white}th,td{padding:10px;border:1px solid #ccd9d3;vertical-align:top;text-align:left}th{background:#e5f1eb}h1,h2{line-height:1.3}p{max-width:1050px}.scroll{overflow:auto}.muted{color:#52685d}.warn{border-left:4px solid #aa6b28;padding:12px;background:#fff4e2}input,select,button{font:inherit;padding:8px;margin:4px}pre{white-space:pre-wrap;word-break:break-word;max-width:400px}.cases td{min-width:210px}.cases td:first-child{min-width:240px}</style>',
            '<h1>MAB 检索修复：实际有界复测</h1>',
            '<p><b>800 题 × 3 组，共 2400 条对照结果；其中 206 条改变输入产生新回复，2194 条输入未改变，明确复用原回复。</b>这是开发性混合复用对照，不是重新生成全部 2400 条回答，也不是独立新样本验证。</p>',
            '<p>修复对两种检索组共同应用：保留原始来源、直接检索门槛、top-k、字符预算和模型设置，仅避免已由原文精确实体边连接的来源被通用词法门槛再次过滤。所有改变输入在读取旧评分前已选定，没有只重测错题。</p>']
    values = []
    for arm in ARMS:
        m = metrics[arm]
        values.append((LABEL[arm], f'{int(m["old_correct"])}/800', f'{int(m["new_correct"])}/800', f'{m["new_correct"]/8:.3f}%', f'{(m["new_correct"]-m["old_correct"])/8:+.3f} 个百分点', m['wins'], m['losses'], int(m['strict_correct']), m['changed_input_rows'], m['reused_old_rows']))
    body += ['<h2>完整分母结果</h2>', table(('条件','旧 v3 正确','修复后正确','修复后正确率','相对 v3 变化','变对','变错','严格格式正确','新回复行','复用旧回复行'), values)]
    p = summary['managed_minus_raw']
    body += [f'<p>修复后管理组相对原始检索组：<b>{p["accuracy_delta"]*100:+.3f} 个百分点</b>；{p["wins"]} 胜、{p["ties"]} 平、{p["losses"]} 负。仅一个相关 fact-consolidation family，无法估计独立 family 置信区间。</p>',
             '<p class="warn">原始检索整体同分也可能掩盖逐题变好与变坏；不得只展示管理组的改善。温度与返回模型别名相同，不代表底层模型权重已独立验证；复用旧回复也不是同期全部重跑。官方子串评分和严格格式诊断均不等于独立语义标注。</p>']
    tasks = []
    for task in sorted({r['task'] for r in public}):
        for arm in ARMS:
            m = counts([r for r in public if r['task']==task and r['arm']==arm])
            tasks.append((task,LABEL[arm],f'{int(m["old_correct"])}/{m["questions"]}',f'{int(m["new_correct"])}/{m["questions"]}',m['wins'],m['losses']))
    body += ['<h2>全部任务与长度分层</h2>',table(('任务','条件','旧正确','新正确','变对','变错'),tasks)]
    cost = summary['new_cost']
    cost_rows = [(key,cost[key]) for key in ('recorded_provider_attempts','successful_canonical_calls','failed_attempts','unresolved_attempts','median_successful_api_latency_seconds','sum_successful_api_latency_seconds','monetary_cost')]
    cost_rows += [(key,value) for key,value in cost['reported_tokens'].items()]
    body += ['<h2>本次新增成本</h2>',table(('指标','数值'),cost_rows),'<p>只计算本次实际 canonical 调用，不给复用的旧回复重复计费。旧全量实验和 CBT 跟进实验的成本分别保留。API 时间不等于并行墙钟时间；未核实价格，金额未知。</p>',
             '<h2>错误归因与 CBT 证据边界</h2>',
             '<p>逐题保留旧分数、新分数、原始模型回答和来源。内容性错误仍为 <code>answer_incorrect_stage_unresolved</code>；改变检索输入不是找到独立的失败根因。此次复测只评价事实冲突基准，不评价 CBT 作业跟进或真人对话收益，更不能证明临床有效。</p>',
             '<h2>全部 800 题逐题对照</h2><p>使用发布 case ID；此报告不包含题干、完整历史或评分参考答案。每组显示旧分数 → 新分数、新/旧回复来源与原始回答。</p><input id="search" placeholder="搜索 case ID、任务或回答"><select id="outcome"><option value="">全部题目</option><option value="changed">有改变输入</option><option value="improved">任一组变对</option><option value="regressed">任一组变错</option></select><p id="count"></p><button id="prev">上一页</button><button id="next">下一页</button><div class="scroll"><table class="cases"><thead><tr><th>case ID / 任务</th><th>近期上下文</th><th>原始历史检索</th><th>版本管理记忆</th></tr></thead><tbody id="rows"></tbody></table></div>',
             '<script type="application/json" id="data">'+encoded([{'id':key,'arms':value,'task':value[ARMS[0]]['task']} for key,value in sorted(cases.items())])+'</script>',
             '''<script>const data=JSON.parse(document.getElementById('data').textContent),arms=['short_term','raw_history_rag','managed_memory_rag'];let page=0;const $=id=>document.getElementById(id),el=(tag,text)=>{let n=document.createElement(tag);if(text!==undefined)n.textContent=String(text);return n;};function draw(){let q=$('search').value.toLowerCase(),f=$('outcome').value;let selected=data.filter(r=>(!q||[r.id,r.task,...arms.map(a=>r.arms[a].original_hypothesis)].join(' ').toLowerCase().includes(q))&&(!f||f==='changed'&&arms.some(a=>r.arms[a].prompt_changed)||f==='improved'&&arms.some(a=>r.arms[a].score>r.arms[a].old_v3_score)||f==='regressed'&&arms.some(a=>r.arms[a].score<r.arms[a].old_v3_score)));page=Math.max(0,Math.min(page,Math.ceil(selected.length/40)-1));$('rows').textContent='';for(let r of selected.slice(page*40,page*40+40)){let tr=el('tr'),meta=el('td');meta.append(el('strong',r.id),el('p',r.task));tr.append(meta);for(let a of arms){let v=r.arms[a],td=el('td');td.append(el('strong',v.old_v3_score+' → '+v.score),el('p',v.provenance==='new_v4_response'?'本次新回复':'输入未变，复用旧回复'),el('pre',v.original_hypothesis));let details=el('details');details.append(el('summary','身份与归因'),el('pre',JSON.stringify({strict:v.strict_committed_score,error:v.error_cause,prompt_sha:v.prompt_sha256,response_sha:v.response_text_sha256},null,2)));td.append(details);tr.append(td);}$('rows').append(tr);}$('count').textContent='筛选 '+selected.length+' / 800 题；第 '+(page+1)+' 页';$('prev').disabled=page<=0;$('next').disabled=(page+1)*40>=selected.length;}for(let id of ['search','outcome'])$(id).addEventListener('input',()=>{page=0;draw();});$('prev').onclick=()=>{page--;draw();};$('next').onclick=()=>{page++;draw();};draw();</script>''',
             '<h2>复算身份</h2><pre>'+html.escape(json.dumps({'candidate_commit':CANDIDATE,'source_only_selection_sha256':SELECTION,'response_helper_sha256':HELPER,'protocol_sha256':summary['protocol_sha256']},indent=2))+'</pre></html>']
    return ''.join(body)


def csv_safe(value):
    return "'"+value if isinstance(value,str) and value.startswith(('=','+','-','@','\t','\r')) else value


def build(args):
    require_colab()
    original = [Path(value).expanduser() for value in (args.run_dir,args.out,args.candidate_repo)]
    if any(p.is_symlink() for p in original):
        raise ValueError('Symlinked report roots')
    run,out,repo = [p.resolve() for p in original]
    if any(not p.is_relative_to(Path('/content')) for p in (run,out,repo)) or out.exists() or out.is_relative_to(run):
        raise ValueError('Use fresh separate Colab output')
    if (run/'.execution.lock').is_symlink():
        raise ValueError('Symlinked source lock')
    with (run/'.execution.lock').open('rb') as lock:
        fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        protocol,summary,public,cases,metrics,source_hashes=validate(run,repo)
        document=render(summary,public,cases,metrics)
        if any(sha(data(run/name))!=value for name,value in source_hashes.items()):
            raise ValueError('Frozen response run changed during reporting')
        if any(sha(data(Path(path)))!=expected for path,expected in protocol['source_sha256'].items()):
            raise ValueError('Original provenance files changed during reporting')
        out.parent.mkdir(parents=True,exist_ok=True)
        staging=Path(tempfile.mkdtemp(prefix=out.name+'.tmp-',dir=out.parent))
        try:
            (staging/'report.html').write_text(document,encoding='utf-8')
            with (staging/'per_question_comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
                writer=csv.writer(stream)
                writer.writerow(['case_id','task']+[arm+'_'+key for arm in ARMS for key in ('old_score','new_score','strict_score','provenance','original_hypothesis')])
                for case_id,group in sorted(cases.items()):
                    row=[case_id,group[ARMS[0]]['task']]
                    for arm in ARMS:
                        r=group[arm];row += [r['old_v3_score'],r['score'],r['strict_committed_score'],r['provenance'],r['original_hypothesis']]
                    writer.writerow([csv_safe(value) for value in row])
            report_manifest={'schema':VERSION,'created_at_utc':datetime.now(timezone.utc).isoformat(),'candidate_commit':CANDIDATE,
                             'protocol_sha256':protocol['protocol_sha256'],'source_only_selection_sha256':SELECTION,
                             'report_helper_sha256':sha(data(Path(__file__))), 'source_run_sha256':source_hashes,
                             'verified_cases':800,'verified_arm_rows':2400,'new_logical_response_rows':206,'reused_old_response_rows':2194,
                             'recomputed_metrics':metrics,'new_cost':summary['new_cost'],'new_provider_calls_by_reporter':0,
                             'processing_location':'genuine hosted Google Colab','original_source_question_history_gold_exported':False,
                             'html_sha256':sha(data(staging/'report.html')),'clinical_claim':False,
                             'verification_boundary':'SHA and deterministic record reconciliation establish integrity, not independent authenticity or clinical benefit'}
            (staging/'report_manifest.json').write_text(json.dumps(report_manifest,sort_keys=True,indent=2)+'\n')
            shutil.copyfile(Path(__file__),staging/'report_helper.py')
            with zipfile.ZipFile(staging/'safe_metadata.zip','w',zipfile.ZIP_DEFLATED) as archive:
                for name in ('public_summary.json','public_per_question_scores.jsonl'):
                    archive.write(run/name,name)
                archive.write(staging/'report_manifest.json','report_manifest.json')
            with zipfile.ZipFile(staging/'controlled_results_and_audit.zip','w',zipfile.ZIP_DEFLATED) as archive:
                for name in RUN_FILES:
                    archive.write(run/name,'new_response_run/'+name)
                for name in ('report.html','per_question_comparison.csv','report_manifest.json','report_helper.py'):
                    archive.write(staging/name,'report/'+name)
                archive.writestr('README.txt','Controlled audit contains original benchmark source in requests.jsonl. Keep this archive in controlled cloud storage; no public GitHub release. safe_metadata.zip excludes sources and model replies. 206 new logical replies + 2194 old unchanged replies form a development-exposed mixed-reuse 800x3 grid, not a fresh full rerun. No clinical conclusion.\n')
            checksums={p.name:sha(data(p)) for p in staging.iterdir() if p.is_file()}
            (staging/'SHA256SUMS.json').write_text(json.dumps(checksums,sort_keys=True,indent=2)+'\n')
            staging.rename(out)
        except BaseException:
            shutil.rmtree(staging,ignore_errors=True)
            raise
        fcntl.flock(lock,fcntl.LOCK_UN)
    return {'status':'complete','output':str(out),'verified_cases':800,'verified_arm_rows':2400,
            'report_sha256':report_manifest['html_sha256'],'new_provider_calls':0,'recomputed_metrics':metrics}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--candidate-repo',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    try:
        print(json.dumps(build(args),ensure_ascii=False,sort_keys=True))
    except Exception as error:
        print(json.dumps({'status':'safe_error','error_type':type(error).__name__,'raw_source_or_credential_displayed':False}))
        raise SystemExit(1)
