"""Small hosted notebook controller; transports are supplied by the live kernel.

Imports do not download data, access credentials, or send requests. Each cohort
freezes its full original roster and finite protocol cap before execution.
"""
import json
from pathlib import Path
import runpy
import sys

COHORTS=("controlled24","memoryagentbench800","longmemeval500","locomo1540")
ORIGINAL_CONTROLLED="content/cbt-qwen7b-gpu-state-study-20261112-v9/source_prepared/after"
TARGET_SHA256="da8687a273d1ab5a0ba34349e69a0cf242c6e5d346308d26801aea6f5525e55d"


def _entry(repo,name):
    return runpy.run_path(str(Path(repo)/"evaluation/architecture-rerun"/name))


def _guard(repo):
    sys.path.insert(0,str(Path(repo)/"evaluation/cbt-memory-effect"))
    from architecture_retest.transport import require_hosted
    require_hosted()


def prepare_cohort(repo,cohort,*,root="/content/cbt-v4-general",data="/content/cbt-v4-data",
                   originals="/content/cbt-v4-original-inputs"):
    _guard(repo)
    if cohort not in COHORTS:
        raise ValueError("unknown full cohort")
    root,data=Path(root),Path(data)
    root.mkdir(parents=True,exist_ok=True)
    folder=root/cohort
    if cohort=="controlled24":
        entry=_entry(repo,"controlled.py")
        if folder.exists():
            return entry["load"](folder)[1]
        original=Path(originals)/ORIGINAL_CONTROLLED
        targets=original/"assessor_only/targets.json"
        if entry["file_hash"](targets)!=TARGET_SHA256:
            raise ValueError("exact hosted original controlled targets differ")
        return entry["prepare"](original/"generation_cases.json",targets,folder)
    if cohort=="memoryagentbench800":
        entry=_entry(repo,"mab_full.py")
        return entry["load"](folder)[0] if folder.exists() else entry["prepare"](data,folder)
    dataset="longmemeval_s_cleaned" if cohort=="longmemeval500" else "locomo10"
    entry=_entry(repo,"formal_full.py")
    if folder.exists():
        return entry["load"](folder)[1]
    sources=root/(cohort+"-sources")
    if not sources.exists():
        _entry(repo,"formal_sources.py")["prepare"](dataset,data,sources)
    return entry["prepare"](sources,folder)


def run_batch(repo,cohort,transport,*,max_new_calls,root="/content/cbt-v4-general"):
    _guard(repo)
    if cohort not in COHORTS or type(max_new_calls) is not int or max_new_calls<1:
        raise ValueError("full cohort and positive finite invocation cap required")
    folder=Path(root)/cohort
    if cohort=="memoryagentbench800":
        entry=_entry(repo,"mab_full.py")
        entry["generate"](folder,max_new_calls,transport=transport)
        result=entry["score"](folder)
    else:
        entry=_entry(repo,"controlled.py" if cohort=="controlled24" else "formal_full.py")
        result=entry["run"](folder,transport,max_new_calls=max_new_calls)
    (Path(root)/(cohort+".status.json")).write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    return result


def judge_batch(repo,cohort,transport,*,max_new_calls,root="/content/cbt-v4-general"):
    _guard(repo)
    if cohort not in ("longmemeval500","locomo1540"):
        raise ValueError("independent reference judge is only for the formal QA cohorts")
    entry=_entry(repo,"formal_judge.py")
    folder=Path(root)/(cohort+"-judge")
    if not folder.exists():
        entry["prepare"](Path(root)/cohort,folder,transport.identity)
    return entry["run"](folder,transport,max_new_calls=max_new_calls,workers=4)
