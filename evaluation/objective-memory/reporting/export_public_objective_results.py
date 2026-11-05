#!/usr/bin/env python3
"""Publish-safe objective results from completed derived output, never a model.

This file stays outside the frozen executable repository. It validates the
completed Colab-derived export using the report builder, checks the matching
report-summary identity, and emits only explicitly permitted score metadata.
Benchmark text, reference answers and original model answers are never written.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import tempfile

VERSION = "public-objective-results-v1"
ROW_FIELDS = (
    "case_id", "dataset_key", "task", "group_id", "arm", "score",
    "strict_committed_score", "exact_alias_or_option_score", "available",
    "completion_status", "canonical_request_key", "pipeline_cause",
    "development_question_exposed", "response_text_sha256",
)
PAIR_FIELDS = (
    "result_type", "mean_accuracy_delta_micro", "mean_accuracy_delta_group_equal",
    "wins", "ties", "losses", "identical_prompt_shared_response_cases",
    "interval_status", "group_equal_95_percent_interval", "micro_95_percent_interval",
)
COST_FIELDS = (
    "recorded_provider_attempt_intents", "intents_without_outcome",
    "successful_unique_provider_calls", "failed_provider_attempts",
    "failed_attempts_without_token_usage", "successful_calls_without_token_usage",
    "reported_tokens_including_failed_attempts_when_supplied",
    "usage_field_missing_counts", "all_recorded_valid_response_count",
    "saved_responses_without_success_outcome", "sum_successful_api_latency_seconds",
    "median_successful_api_latency_seconds", "latency_observation_count",
    "successful_calls_without_latency", "monetary_cost",
    "cross_shard_deduplication", "reported_provider_models",
)
ARMS = ("short_term", "raw_history_rag", "managed_memory_rag")
LABELS = dict(zip(ARMS, ("Short-term", "Raw-history RAG", "Managed-memory RAG")))
METRIC_FIELDS = ("questions", "groups", "correct", "micro", "group_equal",
                 "strict", "exact", "technical", "unresolved", "cue", "not_strict")
HEX = re.compile(r"[0-9a-f]{64}")
ID = re.compile(r"[A-Za-z0-9_.:/-]{1,512}")
TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
COST_NESTED = {"reported_tokens_including_failed_attempts_when_supplied", "usage_field_missing_counts", "reported_provider_models"}
COST_SPECIAL = {"sum_successful_api_latency_seconds", "median_successful_api_latency_seconds", "monetary_cost", "cross_shard_deduplication"}

def sha(body):
    return hashlib.sha256(body).hexdigest()

def fail(message):
    raise ValueError(message)

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"),
                      parse_constant=lambda value: fail("Non-finite JSON value"))

def load_builder(path):
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file():
        fail("Expected report builder file")
    spec = importlib.util.spec_from_file_location("capstone_public_report_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path

def load_inputs(source, builder):
    source = Path(source)
    if source.is_symlink() or not source.is_dir():
        fail("Expected regular derived input directory")
    data, hashes = {}, {}
    for name in builder.REQUIRED + builder.OPTIONAL:
        path = source / name
        if not path.exists():
            if name in builder.REQUIRED:
                fail("Missing derived input: " + name)
            continue
        if path.is_symlink() or not path.is_file():
            fail("Expected regular derived input file")
        body = path.read_bytes()
        hashes[name] = sha(body)
        data[name] = ([json.loads(line, parse_constant=lambda value: fail("Non-finite JSON value"))
                       for line in body.decode("utf-8").splitlines() if line.strip()]
                      if name.endswith(".jsonl") else read_json(path))
        builder.inspect_derived(data[name], name)
    return data, hashes

def validate_report(report, valid, input_hashes, builder, builder_path, fixture):
    summary, plan, meta, rows, cases, metrics = valid
    if bool(report.get("fixture_only")) != fixture:
        fail("Report-summary fixture identity mismatch")
    for field in ("execution_commit", "suite_plan_sha256"):
        if report.get(field) != meta.get(field):
            fail("Report-summary execution identity mismatch")
    if report.get("verified_cases") != len(cases) or report.get("verified_arm_rows") != len(rows):
        fail("Report-summary grid differs")
    if report.get("report_builder_version") != builder.VERSION:
        fail("Report-summary builder version mismatch")
    if report.get("report_builder_sha256") != sha(builder_path.read_bytes()):
        fail("Report-summary builder code differs")
    if report.get("input_file_sha256") != input_hashes:
        fail("Report-summary input bytes differ")
    if report.get("raw_benchmark_histories_or_gold_loaded") is not False or report.get("model_calls_performed_by_builder") != 0 or report.get("human_or_clinical_results_inferred") is not False:
        fail("Unexpected report execution/effectiveness boundary")
    expected_metrics = {d: {a: metrics[(d, a)] for a in ARMS}
                        for d in sorted({d for d, _ in metrics})}
    if report.get("recomputed_metrics") != expected_metrics:
        fail("Report-summary metrics differ from independent recomputation")

def public_row(row, plan, fixture=False):
    # All unexpected fields are discarded: no free text from a source,
    # reference, model answer or trace can enter a public row.
    if not fixture:
        for field in ("case_id", "dataset_key", "task", "group_id", "arm"):
            if not isinstance(row.get(field), str) or not ID.fullmatch(row[field]):
                fail("Unexpected public identifier format")
        for field in ("canonical_request_key", "response_text_sha256"):
            value = row.get(field)
            if value is None and field == "response_text_sha256" and not row["available"]:
                continue
            if not isinstance(value, str) or not HEX.fullmatch(value):
                fail("Expected public response/request digest")
        exposed = row["case_id"] in plan["datasets"][row["dataset_key"]]["development_partition"]["development_case_ids"]
        if row.get("development_question_exposed") is not exposed:
            fail("Exposure flag differs from frozen partition")
    return {field: row.get(field) for field in ROW_FIELDS}

def public_cost(cost, fixture=False, accepted_models=()):
    if not fixture and any(field not in cost for field in COST_FIELDS):
        fail("Missing public cost field")
    result = {}
    def count(value):
        if type(value) is not int or value < 0:
            fail("Expected nonnegative public cost count")
        return value
    for field in COST_FIELDS:
        if field not in cost:
            continue
        value = cost[field]
        if field in COST_NESTED:
            if not isinstance(value, dict):
                fail("Expected structured public cost counts")
            if field == "reported_tokens_including_failed_attempts_when_supplied":
                result[field] = {k: count(value[k]) for k in TOKEN_FIELDS if k in value}
                if not fixture and set(result[field]) != set(TOKEN_FIELDS):
                    fail("Missing public token count")
            elif field == "usage_field_missing_counts":
                result[field] = {}
                for kind in ("successful_calls", "failed_attempts"):
                    if kind not in value and fixture:
                        continue
                    chosen = value.get(kind)
                    if not isinstance(chosen, dict):
                        fail("Missing public usage-missingness counts")
                    result[field][kind] = {k: count(chosen[k]) for k in TOKEN_FIELDS if k in chosen}
                    if not fixture and set(result[field][kind]) != set(TOKEN_FIELDS):
                        fail("Missing public usage-missingness token")
            else:
                result[field] = {}
                for model, n in value.items():
                    if not isinstance(model, str) or (not fixture and model not in accepted_models) or (fixture and not ID.fullmatch(model)):
                        fail("Unapproved provider model in public cost")
                    result[field][model] = count(n)
        elif field == "monetary_cost":
            if value is not None:
                fail("No verified public monetary total is permitted")
            result[field] = None
        elif field == "cross_shard_deduplication":
            if value is not False:
                fail("Unexpected cross-shard cost claim")
            result[field] = False
        elif field in COST_SPECIAL:
            if value is None and field == "median_successful_api_latency_seconds":
                result[field] = None
            elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                fail("Invalid public API latency")
            else:
                result[field] = value
        else:
            result[field] = count(value)
    if not fixture and sum(result["reported_provider_models"].values()) != result["successful_unique_provider_calls"]:
        fail("Provider-model success counts do not reconcile")
    return result

def validate_dataset_costs(global_cost, per_dataset, fixture=False):
    if fixture:
        return
    additive = [k for k in COST_FIELDS if k not in COST_NESTED | COST_SPECIAL]
    additive.append("sum_successful_api_latency_seconds")
    for key in additive:
        if not math.isclose(sum(c[key] for c in per_dataset.values()), global_cost[key], rel_tol=1e-12, abs_tol=1e-8):
            fail("Per-dataset cost differs from global cost: " + key)
    for token in TOKEN_FIELDS:
        if sum(c["reported_tokens_including_failed_attempts_when_supplied"][token] for c in per_dataset.values()) != global_cost["reported_tokens_including_failed_attempts_when_supplied"][token]:
            fail("Per-dataset token counts do not reconcile")
        for kind in ("successful_calls", "failed_attempts"):
            if sum(c["usage_field_missing_counts"][kind][token] for c in per_dataset.values()) != global_cost["usage_field_missing_counts"][kind][token]:
                fail("Per-dataset token missingness does not reconcile")
    models = Counter()
    for c in per_dataset.values():
        models.update(c["reported_provider_models"])
    if dict(models) != global_cost["reported_provider_models"]:
        fail("Per-dataset provider models do not reconcile")

def percentage(value):
    return f"{value * 100:.2f}%"

def make_readme(evidence):
    fixture = evidence["fixture_only"]
    lines = [
        "# " + ("SYNTHETIC FIXTURE — NOT EXPERIMENT RESULTS" if fixture else "Full objective memory comparison: 2026-11-05"),
        "",
        ("Every number in this folder is synthetic. Do not cite it as an observed effect."
         if fixture else
         "This folder records the completed, frozen within-v3 comparison on 800 MemoryAgentBench conflict cases and 589 PersonaMem-v1 32k cases. All model calls and full-source scoring ran in genuine hosted Colab. The reporting step used only controlled derived metadata."),
        "",
        f"- Executable: `{evidence['execution_commit']}`.",
        f"- Suite plan: `{evidence['suite_plan_sha256']}`.",
        f"- Retained census: {evidence['verified_cases']} cases × 3 arms = {evidence['verified_arm_rows']} terminal arm rows. Technical failures remain in every denominator.",
        "- Primary contrast: managed-memory RAG minus raw-history RAG. Model, sources, question, access policy and character budget are shared within each dataset. A character budget is not an exact token budget.",
        "",
        "| Dataset | Arm | Official correct / all | Micro accuracy | Equal-group accuracy | Exact-text/option diagnostic | Strict-format diagnostic | Technical failures |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset, result in evidence["datasets"].items():
        for arm in ARMS:
            m = result["conditions"][arm]
            correct = int(m["correct"]) if float(m["correct"]).is_integer() else m["correct"]
            lines.append(f"| {dataset} | {LABELS[arm]} | {correct}/{m['questions']} | {percentage(m['micro'])} | {percentage(m['group_equal'])} | {percentage(m['exact'])} | {percentage(m['strict'])} | {m['technical']} |")
    lines += ["", "| Dataset | Managed − raw micro | Managed − raw equal-group | Win / tie / loss | Identical-input shared answers | Inference boundary |",
              "| --- | ---: | ---: | --- | ---: | --- |"]
    for dataset, result in evidence["datasets"].items():
        p = result["managed_minus_raw"]
        lines.append(f"| {dataset} | {p['mean_accuracy_delta_micro'] * 100:+.2f} pp | {p['mean_accuracy_delta_group_equal'] * 100:+.2f} pp | {p['wins']} / {p['ties']} / {p['losses']} | {p['identical_prompt_shared_response_cases']} | {p.get('interval_status', 'fixture only')} |")
    cost = evidence["cost"]
    tokens = cost.get("reported_tokens_including_failed_attempts_when_supplied", {})
    lines += [
        "",
        "## Interpretation",
        "",
        "- MemoryAgentBench variants are conservatively one fact-consolidation family, not 800 independent people. There is no estimable independent-family uncertainty interval. The managed representation is a deterministic source-order version-slot view, not a general natural-language semantic extractor or a CBT state model.",
        "- PersonaMem has 20 connected persona groups. Its two RAG arms are lossless quotation controls with identical inputs and shared canonical responses. Their zero difference is by construction; it is not evidence of treatment efficacy, statistical equivalence or effective semantic/version management.",
        "- Released official substring/option scoring is primary. Exact-text/option and strict-format diagnostics are descriptive literal rules, not independent semantic gold. Official substring positives may include refusal or contradictory text.",
        "- Source parse, retrieval and clipping traces are observations. Content errors retain `answer_incorrect_stage_unresolved`; no independently annotated causal stage is invented.",
        "- Development-exposed questions and question-held-out cases retain the frozen partition. Held-out questions still share histories; these are not independent unseen users or histories.",
        "- Earlier pilots remain separate negative or insufficient evidence. Changed prompts, budgets and cohorts prevent a causal old-to-new improvement claim.",
        "",
        "## Actual cost and missingness",
        "",
        f"- Recorded provider attempt intents: {cost.get('recorded_provider_attempt_intents', 'fixture unknown')}; successful unique canonical calls: {cost.get('successful_unique_provider_calls', 'fixture unknown')}; failed attempts: {cost.get('failed_provider_attempts', 'fixture unknown')}; unresolved intents: {cost.get('intents_without_outcome', 'fixture unknown')}.",
        f"- Reported tokens: prompt {tokens.get('prompt_tokens', 'unknown')}, completion {tokens.get('completion_tokens', 'unknown')}, total {tokens.get('total_tokens', 'unknown')}. Missing fields remain unknown; no failure is assumed free.",
        "- Costs count actual canonical attempts, never duplicated logical-arm usage. Cross-shard requests are not deduplicated. API latency excludes preparation/retrieval, some failure time and parallel wall-time effects. No monetary total is claimed without verified prices.",
        "",
        "## CBT and user-benefit boundary",
        "",
        "Independent CBT professional ratings: 0. Actual four-visit participants: 0. Clinical symptom/function outcomes: 0. Natural-check and repeated-question burden: not measured. These public tasks do not establish CBT goal progress, homework completion, natural correction quality, deletion handling, user burden or clinical effectiveness.",
        "",
        "## Files and controlled material",
        "",
        "- `aggregate_evidence.json`: recomputed conditions, primary contrast, exposure partitions, uncertainty boundary, actual call/token costs and source hashes.",
        "- `per_question_metadata.jsonl`: all terminal case/arm scores and failure states, with only explicit metadata fields.",
        "- `SHA256SUMS.json`: integrity hashes of the public output files.",
        "",
        "Question text, histories, options, released reference answers, original model replies, private source traces and credentials are excluded from this public folder. Full answers and the independently recomputable controlled audit remain separate cloud artifacts. Hashes establish file identity, not independent authenticity.",
        "",
    ]
    return "\n".join(lines)

def export(source, report_summary, out, builder_path, fixture=False):
    out = Path(out)
    if out.exists() or out.is_symlink():
        fail("Refusing to overwrite an existing public output")
    builder, builder_path = load_builder(builder_path)
    data, input_hashes = load_inputs(source, builder)
    valid = builder.validate(data, fixture=fixture)
    summary, plan, meta, rows, cases, metrics = valid
    report_path = Path(report_summary)
    if report_path.is_symlink() or not report_path.is_file():
        fail("Expected regular report-summary file")
    report = read_json(report_path)
    builder.inspect_derived(report, "report_summary")
    validate_report(report, valid, input_hashes, builder, builder_path, fixture)
    datasets = {}
    for dataset in sorted({d for d, _ in metrics}):
        p = summary["datasets"][dataset]["paired_differences"]["managed_memory_rag_minus_raw_history_rag"]
        if not fixture and (p.get("result_type") not in {"by_construction_control_equivalence", "exploratory_paired_comparison"} or p.get("interval_status") not in {"not_estimable_fewer_than_two_groups", "exploratory_percentile_cluster_bootstrap", "blocked_by_identical_inputs_and_shared_responses"}):
            fail("Unexpected primary-comparison public enum")
        partitions = {}
        if not fixture:
            for name, ids in (
                ("development_exposed_questions", plan["datasets"][dataset]["development_partition"]["development_case_ids"]),
                ("question_heldout_shared_histories", plan["datasets"][dataset]["development_partition"]["question_heldout_case_ids"]),
            ):
                if not ids:
                    continue
                ids = set(ids)
                partitions[name] = {a: builder.aggregate([r for r in rows if r["dataset_key"] == dataset and r["case_id"] in ids and r["arm"] == a]) for a in ARMS}
        datasets[dataset] = {
            "conditions": {a: {field: metrics[(dataset, a)][field] for field in METRIC_FIELDS} for a in ARMS},
            "managed_minus_raw": {field: p[field] for field in PAIR_FIELDS if field in p},
            "exposure_partitions": partitions,
            "incorrect_answer_stage_attribution": "unresolved_without_independent_evidence_annotations",
        }
    audit = data["cost_audit.json"]
    known_datasets = set(datasets)
    original_per_dataset = audit.get("per_dataset", {})
    if not isinstance(original_per_dataset, dict) or (not fixture and set(original_per_dataset) != known_datasets) or set(original_per_dataset) - known_datasets:
        fail("Unexpected per-dataset public cost keys")
    accepted_models = {m for d in plan.get("datasets", {}).values() for m in d.get("settings", {}).get("accepted_provider_models", [])}
    costs = public_cost(audit, fixture, accepted_models)
    per_dataset_costs = {d: public_cost(c, fixture, accepted_models) for d, c in original_per_dataset.items()}
    validate_dataset_costs(costs, per_dataset_costs, fixture)
    helper_digest = meta.get("reporting_helper_sha256")
    if not fixture and (not isinstance(helper_digest, str) or not HEX.fullmatch(helper_digest)):
        fail("Expected public reporting-helper digest")
    audit_checks = {k: audit.get("independent_audit_checks", {})[k] for k in builder.AUDIT_CHECKS if k in audit.get("independent_audit_checks", {})}
    if any(type(v) is not bool for v in audit_checks.values()):
        fail("Expected boolean public audit checks")
    published_rows = [public_row(r, plan, fixture) for r in sorted(rows, key=lambda r: (r["dataset_key"], r["case_id"], ARMS.index(r["arm"])))]
    evidence = {
        "schema": VERSION, "fixture_only": fixture,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_commit": meta.get("execution_commit"),
        "suite_plan_sha256": meta["suite_plan_sha256"],
        "verified_cases": len(cases), "verified_arm_rows": len(rows),
        "datasets": datasets, "cost": costs,
        "per_dataset_cost": per_dataset_costs,
        "independent_audit_checks": audit_checks,
        "provenance": {
            "derived_input_file_sha256": input_hashes,
            "report_summary_sha256": sha(report_path.read_bytes()),
            "report_builder_sha256": report["report_builder_sha256"],
            "reporting_helper_sha256": helper_digest,
            "public_exporter_sha256": sha(Path(__file__).read_bytes()),
        },
        "effectiveness_boundary": {
            "mab_independent_families": 1,
            "personamem_connected_groups": 20,
            "clinical_effectiveness_established": False,
            "cbt_dialogue_benefit_established": False,
            "actual_human_participants": 0,
            "independent_professional_ratings": 0,
            "symptom_or_function_outcome_observations": 0,
            "natural_check_burden": "not_measured",
            "repeated_question_burden": "not_measured",
            "old_to_new_causal_improvement_claim": False,
        },
        "public_per_question_fields": list(ROW_FIELDS),
        "excluded_from_public_export": ["question_text", "source_histories", "released_options", "reference_answers", "original_model_answers", "source_trace_text", "credentials"],
    }
    # No field from input can enter outside its explicit projection.
    builder.inspect_derived(evidence, "public_evidence")
    body = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in published_rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=out.name + ".tmp-", dir=out.parent))
    try:
        (staging / "aggregate_evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (staging / "per_question_metadata.jsonl").write_text(body, encoding="utf-8")
        (staging / "README.md").write_text(make_readme(evidence), encoding="utf-8")
        hashes = {p.name: sha(p.read_bytes()) for p in sorted(staging.iterdir()) if p.is_file()}
        (staging / "SHA256SUMS.json").write_text(json.dumps(hashes, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        staging.rename(out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"output_directory": str(out), "fixture_only": fixture,
            "published_cases": len(cases), "published_arm_rows": len(published_rows),
            "original_model_answers_published": False,
            "benchmark_question_or_history_text_published": False,
            "public_exporter_sha256": sha(Path(__file__).read_bytes())}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--report-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-builder", default=str(Path(__file__).with_name("build_full_objective_report.py")))
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export(args.input_dir, args.report_summary, args.output_dir, args.report_builder, args.fixture), ensure_ascii=False, indent=2))

