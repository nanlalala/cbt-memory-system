#!/usr/bin/env python3
"""One bounded stage at a time for complete-cohort Colab experiments.

No download, Drive mount, model substitution, retry loop or automatic publication.
The existing reader/judge journals remain the authority for charged attempts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import prepare_formal_evaluation as cohorts
import reference_judge as judges
import run_structured_memory as reader
from execution_lock import exclusive_directories, exclusive_directory
from run_three_arm import Settings, freeze_json, stable_digest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
VERSION = "formal-colab-pipeline-v1"
STAGES = ("verify", "prepare", "reader", "judge", "export", "status")
COHORT_FILES = ("cohort_manifest.json", "preregistration_lock.json",
                "generation_cases.jsonl", "scoring_references.jsonl")
READER_FILES = ("protocol.json", "prompts.jsonl", "references.jsonl", "candidate_memories.jsonl",
                "preparation_summary.json", "responses.jsonl", "response_attempts.jsonl", "failures.jsonl")
JUDGE_FILES = ("judge_protocol.json", "judge_requests.jsonl", "judge_mapping.jsonl",
               "judge_call_intents.jsonl", "judge_attempts.jsonl",
               "reference_judge_scores.jsonl", "reference_judge_summary.json")
BOUNDARY = ("Complete released eligible cohorts, reported separately; unconfirmed candidate evidence, "
            "not consented user facts. Adapted model-assisted grading, not official benchmark, "
            "human or clinical evaluation. Partial checkpoints are not final effectiveness results.")


def settings_from_plan(plan: dict, backend: str) -> Settings:
    value = plan["shared_response_configuration"]
    window = value["short_window"]
    return Settings(model=value["model"], accepted_provider_models=tuple(value["accepted_provider_models"]),
                    backend=backend, short_turns=window["maximum_turns"], short_chars=window["maximum_characters"],
                    long_chars=value["long_context_maximum_characters"], max_context_chars=value["total_context_maximum_characters"],
                    top_k=value["retrieval_top_k"], minimum_relevance=value["minimum_relevance"],
                    max_response_tokens=value["max_response_tokens"], thinking_mode=value["thinking_mode"])


def _positive_budget(value: int | None) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError("reader/judge require a positive explicit finite max_calls for this invocation")


def _guard(root: Path, fixture_only: bool) -> None:
    if not fixture_only:
        reader.require_colab()  # Before locks, files, models, datasets or provider calls.
        return
    manifest, _, _ = cohorts.validate_cohort(root / "cohort")
    if manifest.get("fixture_only") is not True:
        raise ValueError("the test seam accepts only bounded synthetic fixture cohorts")
    reader.require_workload_environment(manifest, root / "cohort")


def _configuration(root: Path) -> dict:
    value = json.loads((root / "pipeline.json").read_text(encoding="utf-8"))
    unsigned = {key: item for key, item in value.items() if key != "config_sha256"}
    if value.get("schema_version") != VERSION or value.get("config_sha256") != stable_digest(unsigned):
        raise ValueError("pipeline configuration changed; preserve this run and use a new root")
    plan = cohorts.load_plan(root / "frozen_plan.json")
    if stable_digest(plan) != value["plan_sha256"]:
        raise ValueError("pipeline plan differs from frozen configuration")
    if value.get("api_base") != plan["shared_response_configuration"]["api_base"]:
        raise ValueError("pipeline endpoint differs from frozen plan")
    settings = settings_from_plan(plan, value["settings"]["backend"])
    if asdict(settings) != {**value["settings"], "accepted_provider_models": tuple(value["settings"]["accepted_provider_models"])}:
        raise ValueError("pipeline settings differ from frozen plan")
    return value


def _cohort(root: Path, config: dict, fixture_only: bool) -> dict:
    manifest, _, _ = cohorts.validate_cohort(root / "cohort")
    if (manifest["dataset_key"] != config["dataset_key"] or manifest["plan_sha256"] != config["plan_sha256"]
            or manifest["fixture_only"] != fixture_only or manifest["role"] != config["role"]):
        raise ValueError("cohort differs from frozen pipeline identity")
    reader.require_workload_environment(manifest, root / "cohort")
    return manifest


def _verify_input(path: Path, dataset_key: str, plan: dict) -> dict:
    spec = plan["datasets"][dataset_key]
    if path.is_symlink() or not path.is_file():
        raise ValueError("provide a regular local Colab dataset file; no automatic downloads")
    identity = cohorts.file_identity(path)
    if identity != {"sha256": spec["sha256"], "bytes": spec["bytes"]}:
        raise ValueError("dataset bytes do not match the complete frozen release")
    return {"dataset_key": dataset_key, "source_identity": identity, "status": "verified_bytes_only_no_model_calls"}


def _prepare(root: Path, *, dataset_key: str | None, input_path: Path | None, plan_path: Path,
             backend: str, device: str | None, role: str, fixture_only: bool) -> dict:
    plan = cohorts.load_plan(plan_path)
    if dataset_key not in plan["datasets"]:
        raise ValueError("prepare requires one frozen supported dataset_key")
    settings = settings_from_plan(plan, backend)
    settings.validate()
    config = {"schema_version": VERSION, "dataset_key": dataset_key, "plan_sha256": stable_digest(plan),
              "settings": {**asdict(settings), "accepted_provider_models": list(settings.accepted_provider_models)},
              "api_base": plan["shared_response_configuration"]["api_base"], "role": role,
              "fixture_only": fixture_only, "representation": "verbatim_candidate_baseline_not_semantic_extraction",
              "boundary": BOUNDARY}
    config["config_sha256"] = stable_digest(config)
    freeze_json(root / "pipeline.json", config)
    freeze_json(root / "frozen_plan.json", plan)
    cohort_dir = root / "cohort"
    if (cohort_dir / "cohort_manifest.json").exists():
        manifest = _cohort(root, config, fixture_only)
        if input_path is not None:
            _verify_input(input_path, dataset_key, plan)
    else:
        if fixture_only or input_path is None:
            raise ValueError("prepare needs an existing tiny test cohort or a full local Colab dataset file")
        _verify_input(input_path, dataset_key, plan)
        manifest = cohorts.prepare_dataset(input_path, dataset_key, root / "frozen_plan.json", cohort_dir, role=role)
    run_dir = root / "reader"
    if (run_dir / "protocol.json").exists():
        # Never rebuild embeddings or prompts during a routine resumed invocation.
        with exclusive_directory(run_dir):
            protocol, _, _, _ = reader.load_run(run_dir)
        if protocol["cohort_manifest"] != manifest or protocol["settings"] != config["settings"]:
            raise ValueError("prepared reader differs from frozen pipeline")
        result = {"status": "already_prepared_validated_no_model_calls"}
    else:
        result = reader.prepare_structured_run(cohort_dir, run_dir, settings, config["api_base"], device=device)
    return {**result, "dataset_key": dataset_key, "boundary": BOUNDARY}


def _status_locked(root: Path, config: dict, fixture_only: bool) -> dict:
    """Caller holds reader/judge locks; use lock-aware scorer to avoid reentry."""
    manifest = _cohort(root, config, fixture_only)
    result = {"dataset_key": config["dataset_key"], "examples": manifest["counts"]["examples"],
              "fixture_only": fixture_only, "boundary": BOUNDARY,
              "final": False, "clinical_claim": False, "confirmatory_claim": False,
              "status": "cohort_prepared_no_effectiveness_result"}
    if not (root / "reader/protocol.json").exists():
        return result
    protocol, prompts, _, complete = reader.load_run(root / "reader")
    if protocol["cohort_manifest"] != manifest or protocol["settings"] != config["settings"]:
        raise ValueError("reader is not the frozen pipeline run")
    attempts, failures = reader.attempt_state(root / "reader", protocol, prompts, "response")
    result["reader"] = {"requests": len(prompts), "successful_responses": len(complete),
                        "started_attempts": sum(attempts.values()), "recorded_failures": len(failures),
                        "pending": len(prompts) - len(complete)}
    result["status"] = "reader_checkpoint_only_no_effectiveness_result"
    if not (root / "judge/judge_protocol.json").exists():
        return result
    # This is the existing scorer under its required locks, not a new failure policy.
    try:
        summary = judges._score_reference_judge_locked(root / "reader", root / "judge", write_outputs=False)
    except ValueError as exc:
        if not str(exc).startswith("incomplete judge run: all frozen items need success"):
            raise
        result["status"] = "judge_checkpoint_only_no_effectiveness_result"
    else:
        result.update(final=not fixture_only,
                      status="synthetic_software_smoke_not_effectiveness" if fixture_only else "complete_model_assisted_result_not_clinical",
                      evaluation=summary)
    outcomes = judges.read_jsonl(root / "judge/judge_attempts.jsonl")
    result["judge"] = {"requests": len(judges.read_jsonl(root / "judge/judge_requests.jsonl")),
                       "started_attempts": len(judges.read_jsonl(root / "judge/judge_call_intents.jsonl")),
                       "successful_labels": sum(row.get("status") == "success" for row in outcomes),
                       "recorded_technical_failures": sum(row.get("status") == "technical_failure" for row in outcomes)}
    return result


def _export_locked(root: Path, config: dict, fixture_only: bool, archive_path: Path) -> dict:
    status = _status_locked(root, config, fixture_only)
    members: dict[str, Path | bytes] = {}
    for directory, names in (("", ("pipeline.json", "frozen_plan.json")), ("cohort", COHORT_FILES),
                             ("reader", READER_FILES), ("judge", JUDGE_FILES)):
        for name in names:
            if directory == "judge" and name in {"reference_judge_scores.jsonl", "reference_judge_summary.json"}:
                continue  # Add only a freshly recomputed complete score below.
            relative = f"{directory}/{name}" if directory else name
            path = root / relative
            if path.is_symlink():
                raise ValueError("audit artifacts must be regular files, not symlinks")
            if path.exists():
                if not path.is_file():
                    raise ValueError("audit artifact is not a regular file")
                members[relative] = path
    if "evaluation" in status:
        # Always export scores recomputed from raw journals, never trust a stale summary.
        summary = judges._score_reference_judge_locked(root / "reader", root / "judge", write_outputs=True)
        members["judge/reference_judge_scores.jsonl"] = root / "judge/reference_judge_scores.jsonl"
        members["judge/reference_judge_summary.json"] = root / "judge/reference_judge_summary.json"
        status["evaluation"] = summary
    # Pin enough source code and exact upstream rubric to audit offline. No model snapshots or datasets.
    code = {HERE / name for name in ("run_formal_pipeline.py", "prepare_formal_evaluation.py", "run_three_arm.py",
                                    "run_structured_memory.py", "reference_judge.py", "reference_judge_upstream.json",
                                    "execution_lock.py", "score_structured_memory.py", "score_three_arm.py")}
    if (root / "reader/protocol.json").exists():
        protocol = json.loads((root / "reader/protocol.json").read_text())
        for name, expected in protocol["code_sha256"].items():
            path = REPO / name
            if path.is_symlink() or REPO not in path.resolve().parents or cohorts.file_identity(path)["sha256"] != expected:
                raise ValueError("code differs from frozen reader; cannot export an authentic code snapshot")
            code.add(path)
    for relative in ("modules/long-term-memory", "modules/agent-integration", "evaluation/memory-retrieval"):
        code.update((REPO / relative).rglob("*.py"))
    code.update(REPO / name for name in ("evaluation/memory-retrieval/benchmark_manifest.json",
                                        "modules/model_snapshots.py", "modules/model_snapshots.json"))
    for path in sorted(code):
        if path.is_symlink() or not path.is_file():
            raise ValueError("audit code snapshots must be regular repository files")
        members[f"code/{path.relative_to(REPO)}"] = path
    snapshot = judges.template_snapshot()
    declaration = {"schema_version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
                   "status": status, "plan_sha256": config["plan_sha256"],
                   "distribution": "Research audit archive; contains public benchmark-derived prompts and responses. Review upstream terms before redistribution. Never use this export for real personal or clinical conversations.",
                   "not_included": ["API keys", "environment variables", "full raw benchmark downloads", "model caches", "SQLite databases"],
                   "attribution": {"benchmark_source": manifest_source(root), "judge_upstream_revision": snapshot["upstream_revision"],
                                   "judge_license": snapshot["license_text"]},
                   "recompute": "In Colab, run python code/evaluation/end-to-end-dialogue/reference_judge.py --score --run-dir reader --output-dir judge. The scorer verifies frozen artifacts, identities and complete denominators; this command never generates answers or calls the API."}
    members["AUDIT_DECLARATION.json"] = (json.dumps(declaration, ensure_ascii=False, indent=2) + "\n").encode()
    members["RESEARCH_BOUNDARY.txt"] = (BOUNDARY + "\nPartial archives are recovery checkpoints, not final results.\n").encode()
    if archive_path.is_symlink():
        raise ValueError("archive output must not be a symlink")
    archive_path = archive_path.resolve()
    if archive_path in {path.resolve() for path in members.values() if isinstance(path, Path)}:
        raise ValueError("archive target cannot replace an audit input")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".audit-", suffix=".zip", dir=archive_path.parent)
    os.close(descriptor)
    checksums = []
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, value in sorted(members.items()):
                digest = hashlib.sha256()
                with archive.open(name, "w", force_zip64=True) as target:
                    if isinstance(value, bytes):
                        target.write(value)
                        digest.update(value)
                    else:
                        with value.open("rb") as source:
                            for block in iter(lambda: source.read(1024 * 1024), b""):
                                target.write(block)
                                digest.update(block)
                checksums.append(f"{digest.hexdigest()}  {name}\n")
            archive.writestr("SHA256SUMS", "".join(checksums))
        os.replace(temporary, archive_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {"archive": str(archive_path), "archive_identity": cohorts.file_identity(archive_path),
            "members": len(members) + 1, "final": status["final"], "status": status["status"], "boundary": BOUNDARY}


def manifest_source(root: Path) -> dict:
    return json.loads((root / "cohort/cohort_manifest.json").read_text(encoding="utf-8"))["source"]


def run_stage(stage: str, root: Path, *, dataset_key: str | None = None, input_path: Path | None = None,
              plan_path: Path = cohorts.DEFAULT_PLAN, backend: str = "lexical", device: str | None = None,
              role: str = "exploratory", max_calls: int | None = None, finalize_reader_failures: bool = False,
              archive_path: Path | None = None, _fixture_only: bool = False) -> dict:
    """Run exactly one stage. _fixture_only is a bounded no-network unit-test seam, never CLI."""
    root = Path(root).resolve()
    if stage not in STAGES:
        raise ValueError("unknown pipeline stage")
    if stage in {"reader", "judge"}:
        _positive_budget(max_calls)
    _guard(root, _fixture_only)
    # Separate lock identity: core APIs acquire reader/judge locks themselves.
    # Same-root invocations cannot concurrently prepare, spend calls, or export.
    with exclusive_directory(root / ".pipeline"):
        if stage == "verify" and input_path is not None:
            plan = cohorts.load_plan(plan_path)
            if dataset_key not in plan["datasets"]:
                raise ValueError("verify requires a frozen supported dataset_key")
            return _verify_input(Path(input_path), dataset_key, plan)
        if stage == "prepare":
            return _prepare(root, dataset_key=dataset_key, input_path=Path(input_path) if input_path else None,
                            plan_path=Path(plan_path), backend=backend, device=device, role=role, fixture_only=_fixture_only)
        config = _configuration(root)
        _cohort(root, config, _fixture_only)
        if stage == "reader":
            if (root / "judge/judge_protocol.json").exists():
                raise ValueError("reader is frozen by judge preparation; do not alter its checkpoint")
            return reader.generate_run(root / "reader", max_calls=max_calls)
        if stage == "judge":
            if not (root / "judge/judge_protocol.json").exists():
                judges.prepare_reference_judge(root / "reader", root / "judge", plan_path=root / "frozen_plan.json",
                                               finalize_reader_failures=finalize_reader_failures)
            # Resume validates the existing frozen protocol in execute; never rebuild
            # requests or retroactively change an earlier failure-finalization decision.
            outcome = judges.execute_reference_judge(root / "reader", root / "judge", max_calls=max_calls)
            with exclusive_directories(root / "reader", root / "judge"):
                status = _status_locked(root, config, _fixture_only)
                if "evaluation" in status:
                    judges._score_reference_judge_locked(root / "reader", root / "judge", write_outputs=True)
            return {**outcome, "pipeline_status": status}
        with exclusive_directories(root / "reader", root / "judge"):
            if stage == "export":
                return _export_locked(root, config, _fixture_only, Path(archive_path) if archive_path else root / "audit.zip")
            return _status_locked(root, config, _fixture_only)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset-key", choices=("locomo10", "longmemeval_s_cleaned"))
    parser.add_argument("--input", type=Path, dest="input_path")
    parser.add_argument("--plan", type=Path, default=cohorts.DEFAULT_PLAN, dest="plan_path")
    parser.add_argument("--backend", choices=("lexical", "dense", "hybrid"), default="lexical")
    parser.add_argument("--device")
    parser.add_argument("--role", choices=("exploratory", "locked_external_test"), default="exploratory")
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--finalize-reader-failures", action="store_true")
    parser.add_argument("--archive", type=Path, dest="archive_path")
    arguments = vars(parser.parse_args())
    try:
        result = run_stage(**arguments)
    except Exception as exc:
        # Provider bodies, credentials and arbitrary input text never enter console errors.
        print(json.dumps({"status": "stopped_preserve_checkpoint", "error_type": type(exc).__name__,
                          "message": "Stage failed. Inspect frozen configuration and local journals; no automatic model/settings changes."}))
        raise SystemExit(1) from None
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
