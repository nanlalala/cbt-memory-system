"""Hosted-only three-arm rerun controller. No credential access on import.

Load this pinned-checkout file with runpy.run_path, then call run_all with the
two already constructed transports. This controller never creates or exports a key.
It preserves prior runs and refuses unknown/rejected sends on resume.
"""
from hashlib import sha256
import json
from pathlib import Path
import sys

CONTROLLER_CODE_SHA256=sha256(Path(__file__).resolve().read_bytes()).hexdigest()


JUDGE_IDENTITY = {
    "provider": "https://tokendance.space/gateway/v1",
    "model": "qwen3.5-plus",
    "returned_model": "qwen3.5-plus-2026-04-20",
    "model_family": "qwen",
}
EXPECTED_QUOTE_GENERATION_SHA = "6075b00d85e00f899f031cca788acb7f910b550598d6b21515164b618183ed9b"


def _modules(checkout):
    checkout = Path(checkout).resolve()
    if not checkout.is_relative_to("/content"):
        raise ValueError("genuine hosted /content checkout required")
    if Path(__file__).resolve() != checkout / "evaluation/architecture-rerun/cbt_hosted_controller.py":
        raise ValueError("controller must originate in the explicitly selected pinned checkout")
    if sha256(Path(__file__).resolve().read_bytes()).hexdigest()!=CONTROLLER_CODE_SHA256:
        raise ValueError("controller changed after loading; reimport the pinned checkout in an idle runtime")
    sys.path.insert(0, str(checkout / "modules/agent-integration"))
    from cbt_runtime_guard import assert_repository_runtime
    assert_repository_runtime(checkout)
    sys.path.insert(0, str(checkout / "evaluation/cbt-memory-effect"))
    from architecture_retest import runner, memory_assessment, judge, quotation
    if Path(runner.__file__).resolve() != checkout / "evaluation/cbt-memory-effect/architecture_retest/runner.py":
        raise ValueError("stale cached module from another checkout; use a fresh hosted Python process")
    runner.require_hosted()
    assert_repository_runtime(checkout)
    if runner.ARMS != ("no_rag", "raw_history_rag", "managed_memory_rag"):
        raise ValueError("only the final three-arm design is authorized by this controller")
    return runner, memory_assessment, judge, quotation


def _jsonl(path):
    from architecture_retest.journal import rows
    return rows(Path(path))


def _source_file(root, suffix):
    """Locate an exact archived path without picking an arbitrary duplicate."""
    matches = sorted(p for p in Path(root).rglob(Path(suffix).name)
                     if p.as_posix().endswith(suffix))
    if not matches:
        raise FileNotFoundError(suffix)
    hashes = {sha256(p.read_bytes()).hexdigest() for p in matches}
    if len(hashes) != 1:
        raise ValueError("conflicting restored archive copies: " + suffix)
    return matches[0]


def _remaining(folder):
    """No implicit retries, including after interrupted controller invocations."""
    from architecture_retest.journal import digest, rows, UnknownAttempt
    folder = Path(folder)
    protocol = json.loads((folder / "protocol.json").read_text())
    if protocol["protocol_sha256"] != digest({k: v for k, v in protocol.items() if k != "protocol_sha256"}):
        raise ValueError("protocol checksum differs")
    intents = rows(folder / "intents.jsonl")
    outcomes = rows(folder / "outcomes.jsonl")
    if (len({r["key"] for r in intents}) != len(intents) or
            len({r["key"] for r in outcomes}) != len(outcomes)):
        raise ValueError("duplicate retained journal records")
    if ({r["key"] for r in intents} != {r["key"] for r in outcomes} or
            any(r["status"] != "received" for r in outcomes)):
        raise UnknownAttempt("unreconciled or rejected send preserved; no automatic retry")
    remaining = protocol["maximum_attempts"] - len(intents)
    if remaining < 0:
        raise ValueError("retained attempts exceed frozen maximum")
    return remaining


def _require_finished_invocation(folder):
    path = Path(folder) / "last_invocation.json"
    if path.exists() and json.loads(path.read_text()).get("errors"):
        raise RuntimeError("stage has retained invocation errors; inspect " + str(path))
    _remaining(folder)


def prepare_all(checkout, run_root, *, cactus_json="/content/cbt-v4-sources/cactus.json",
                frozen_root="/content/cbt-v4-frozen-roster",
                original_root="/content/cbt-v4-original-inputs",
                include_controlled_semantic=False, inherited_memory_roots=None):
    """Freeze all known source-only inputs/preflights without any model calls."""
    runner, memory, _, _ = _modules(checkout)
    inherited_memory_roots = dict(inherited_memory_roots or {})
    from architecture_retest.journal import atomic, digest, utc
    run_root = Path(run_root).resolve()
    if not run_root.is_relative_to("/content") or run_root == Path("/content"):
        raise ValueError("new named /content run root required")
    controller_sha256=CONTROLLER_CODE_SHA256
    manifest_path=run_root / "controller_manifest.json"
    if manifest_path.exists():
        prior=json.loads(manifest_path.read_text())
        if prior.get("controller_code_sha256")!=controller_sha256:
            raise ValueError("frozen controller code differs; use a separate new run root")
        if prior.get("checkout")!=str(Path(checkout).resolve()) or prior.get("code_sha256")!=runner.code_hashes():
            raise ValueError("frozen checkout/code differs before source preparation")
    run_root.mkdir(parents=True, exist_ok=True)
    frozen_root = Path(frozen_root)
    old_run = frozen_root / "original/run"
    old_generation = old_run / "generation/cases.jsonl"
    if sha256(old_generation.read_bytes()).hexdigest() != EXPECTED_QUOTE_GENERATION_SHA:
        raise ValueError("historical quotation roster bytes differ")
    quote_generation = _jsonl(old_generation)
    quote_targets = _jsonl(old_run / "assessor_only/targets.jsonl")
    inventory = _jsonl(old_run / "inventory.jsonl")
    metadata = json.loads((frozen_root / "cactus_v2_frozen_roster_metadata.json").read_text())
    families = sorted(metadata["selected_family_ids"])
    if len(families) != 200 or len(set(families)) != 200 or len(inventory) != 600 or len(quote_generation) != 590:
        raise ValueError("exact prior 200-family/600-planned/590-eligible roster required")
    if len(quote_targets) != 590 or {r["case_id"] for r in quote_targets} != {r["case_id"] for r in quote_generation}:
        raise ValueError("historical quotation scoring identities differ")
    if {r["source_family_id"] for r in quote_generation} - set(families):
        raise ValueError("quotation families differ from historical cohort")
    professional_suffixes = [
        "content/cbt-professional-prospective-seven-20261112-v1/batch_01/generation_cases.json",
        "content/cbt-professional-prospective-seven-20261112-v1/batch_02/generation_cases.json",
    ]
    professional_paths = [_source_file(original_root, suffix) for suffix in professional_suffixes]
    professional = sum((json.loads(p.read_text()) for p in professional_paths), [])
    if len(professional) != 7 or len({r["case_id"] for r in professional}) != 7:
        raise ValueError("exact recovered professional seven-case roster required")
    pools = [("professional7", professional, professional_paths)]
    if include_controlled_semantic:
        controlled_path = _source_file(original_root,
            "content/cbt-qwen7b-gpu-state-study-20261112-v9/source_prepared/after/generation_cases.json")
        controlled = json.loads(controlled_path.read_text())
        if len(controlled) != 24 or len({r["case_id"] for r in controlled}) != 24:
            raise ValueError("exact recovered controlled 24-case roster required")
        pools.append(("controlled24_semantic", controlled, [controlled_path]))
    cohorts = []
    for label, cases, paths in pools:
        folder = run_root / label
        identity = {"archived_source_files_sha256": {str(p): sha256(p.read_bytes()).hexdigest() for p in paths}}
        if not folder.exists():
            runner.prepare_case_pool(cases, folder, identity)
        else:
            _, protocol, existing = runner.load(folder)
            if protocol.get("source_identity") != identity or {c["case_id"] for c in existing} != {c["case_id"] for c in cases}:
                raise ValueError("existing historical pool differs")
        cohorts.append(label)
    natural = run_root / "cactus600"
    if not natural.exists():
        runner.prepare(cactus_json, natural, family_limit=200, family_ids=families)
    else:
        _, protocol, _ = runner.load(natural)
        if protocol.get("family_ids") != families:
            raise ValueError("existing natural cohort differs")
    cohorts.append("cactus600")
    if set(inherited_memory_roots)-set(cohorts):
        raise ValueError("inherited receipt mapping names an unknown original cohort")
    caps = {}
    for label in cohorts:
        folder = run_root / label
        _, protocol, cases = runner.load(folder)
        assessment = run_root / (label + "_memory")
        if not assessment.exists():
            memory.prepare(folder, assessment, JUDGE_IDENTITY,
                inherited_folder=inherited_memory_roots.get(label))
        elif json.loads((assessment / "protocol.json").read_text())["cases_sha256"] != digest(cases):
            raise ValueError("existing independent memory roster differs")
        memory_protocol = json.loads((assessment / "protocol.json").read_text())
        caps[label] = {"planned_cases": len(cases), "generation_and_preparation_max": len(cases) * 5,
                       "independent_memory_max": len(cases) * 5,
                       "independent_memory_new_attempt_max": memory_protocol["maximum_attempts"],
                       "independent_memory_inherited_received": memory_protocol.get("inherited_received_attempts",0),
                       "quality_primary_max": len(cases) * 3,
                       "quality_including_repeats_max": (len(cases) * 3 * 11 + 9) // 10}
    manifest = {"created_at_utc": utc(), "checkout": str(Path(checkout).resolve()),
                "controller_code_sha256":controller_sha256,
                "arms": list(runner.ARMS), "code_sha256": runner.code_hashes(),
                "cohorts": cohorts, "caps": caps, "quotation_eligible": 590,
                "quotation_max": 1770, "frozen_root": str(frozen_root),
                "source_only_prepared": True,
                "inherited_memory_roots": {k: str(Path(v).resolve()) for k,v in inherited_memory_roots.items()}}
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        if any(prior.get(k) != manifest[k] for k in ("checkout", "controller_code_sha256", "arms", "code_sha256", "cohorts", "caps", "frozen_root", "inherited_memory_roots")):
            raise ValueError("controller manifest differs; use a separate new run root")
        return prior
    atomic(manifest_path, manifest)
    return manifest


def run_all(checkout, answer_transport, judge_transport, run_root, *, stage="all",
            workers=4, **prepare_kwargs):
    """Run seven first, then the full roster, quality and original quote endpoint.

    stage='professional7' permits an early complete original small cohort;
    stage='cactus600' or 'quotation' resumes only the requested original stage.
    Targets stay assessor-only. Existing receipts are reused; missing cases stay
    in planned denominators. This function performs authorized real hosted calls.
    """
    runner, memory, judge, quotation = _modules(checkout)
    from architecture_retest.journal import atomic, utc
    if answer_transport.identity != runner.IDENTITY or judge_transport.identity != JUDGE_IDENTITY:
        raise ValueError("exact frozen answer/independent judge identities required")
    manifest = prepare_all(checkout, run_root, **prepare_kwargs)
    run_root = Path(run_root).resolve()
    if stage not in {*manifest["cohorts"], "quotation", "all"}:
        raise ValueError("unknown original cohort stage")
    chosen = manifest["cohorts"] if stage == "all" else [stage] if stage != "quotation" else []
    dense = runner.DenseRetriever()
    reports = {}
    for label in chosen:
        folder = run_root / label
        assessment = run_root / (label + "_memory")
        reports[label] = runner.run(folder, answer_transport, max_new_calls=_remaining(folder),
            workers=workers, dense=dense, assessment_folder=assessment,
            assessment_transport=judge_transport, assessment_max_new_calls=_remaining(assessment))
        _require_finished_invocation(folder)
        _remaining(assessment)
        quality_run = run_root / (label + "_quality")
        if not quality_run.exists():
            runner.export_quality(folder, JUDGE_IDENTITY, quality_run,
                maximum_attempts=manifest["caps"][label]["quality_including_repeats_max"])
        reports[label + "_quality"] = judge.run(quality_run, judge_transport,
            max_new_calls=_remaining(quality_run), workers=workers)
        _require_finished_invocation(quality_run)
        reports[label + "_memory"] = json.loads((assessment / "summary.json").read_text())
        atomic(run_root / "controller_progress.json", {"at_utc": utc(), "completed_stages": list(reports),
               "latest_stage": label, "reports": reports})
    if stage in {"all", "quotation"}:
        natural = run_root / "cactus600"
        status = runner.summarize(natural)
        if status["completed_cases"] != status["planned_cases"]:
            raise ValueError("complete natural prefix processing before freezing shared quotation projections")
        _require_finished_invocation(natural)
        quote_run = run_root / "cactus590_quotation"
        old_run = Path(manifest["frozen_root"]) / "original/run"
        if not quote_run.exists():
            quotation.prepare(natural, quote_run,
                frozen_generation=_jsonl(old_run / "generation/cases.jsonl"),
                frozen_gold=_jsonl(old_run / "assessor_only/targets.jsonl"),
                frozen_inventory=_jsonl(old_run / "inventory.jsonl"))
        reports["cactus590_quotation"] = quotation.run(quote_run, answer_transport,
            max_new_calls=_remaining(quote_run), workers=workers, dense=dense)
        _require_finished_invocation(quote_run)
    atomic(run_root / "controller_progress.json", {"at_utc": utc(), "completed_stages": list(reports),
           "latest_stage": stage, "reports": reports})
    return {"run_root": str(run_root), "completed_stages": list(reports),
            "manifest": str(run_root / "controller_manifest.json"),
            "progress": str(run_root / "controller_progress.json")}
