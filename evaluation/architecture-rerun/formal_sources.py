"""Recover full pinned LoCoMo and LongMemEval source rosters for the new readout.

All real source download, loading, grouping and transformation require Colab.
Histories and references are written separately. No model or Secret is accessed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation/end-to-end-dialogue"))
sys.path.insert(0, str(ROOT / "evaluation/memory-retrieval"))
import prepare_formal_evaluation as formal
from benchmark_adapters import load_locomo, load_longmemeval
from run_three_arm import parse_date

VERSION = "new-readout-original-formal-sources-v1"
DATASETS = ("locomo10", "longmemeval_s_cleaned")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",",":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def source_file(dataset_key, data_dir, existing=None):
    formal.require_colab_runtime()
    plan = formal.load_plan(formal.DEFAULT_PLAN)
    spec = plan["datasets"][dataset_key]
    if existing is not None:
        path = Path(existing)
    else:
        folder = Path(data_dir)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (dataset_key + ".json")
        if not path.exists():
            temporary = path.with_suffix(".download")
            with urlopen(spec["source"],timeout=180) as response, temporary.open("xb") as stream:
                size = 0
                for block in iter(lambda: response.read(1024*1024),b""):
                    size += len(block)
                    if size > spec["bytes"]:
                        raise ValueError("download exceeded pinned dataset size")
                    stream.write(block)
            if temporary.stat().st_size != spec["bytes"] or hash_file(temporary) != spec["sha256"]:
                raise ValueError("incomplete dataset download retained")
            temporary.replace(path)
    if path.stat().st_size != spec["bytes"] or hash_file(path) != spec["sha256"]:
        raise ValueError("existing source identity differs from the frozen full release")
    return path


def history_at_cutoff(example, dataset_key):
    """Pure tiny-fixture seam preserving original speakers and source dates."""
    if dataset_key not in DATASETS:
        raise ValueError("unknown formal dataset")
    if dataset_key == "longmemeval_s_cleaned" and not example.query_date:
        raise ValueError("LongMemEval requires released question date")
    cutoff = parse_date(example.query_date) if example.query_date else None
    sessions, excluded = [], []
    for position, session in enumerate(example.sessions):
        stamp = parse_date(session["date"])
        if cutoff is not None and stamp > cutoff:
            excluded.append(str(session["session_id"]))
        else:
            sessions.append((stamp,position,session))
    sessions.sort(key=lambda row:(row[0],row[1]))
    history = []
    for _, _, session in sessions:
        for index, turn in enumerate(session["turns"]):
            original_role = str(turn.get("role",turn.get("speaker","unknown")))
            role = original_role if original_role in {"user","assistant"} else "context"
            text = turn.get("content",turn.get("text",""))
            if not isinstance(text,str):
                raise ValueError("source content must retain a string")
            if not text.strip():
                continue
            sid = str(session["session_id"])
            tid = str(turn.get("dia_id",f"turn:{index}"))
            # The date is the released string, not an inferred or schema date.
            source_id = "formal-source:"+digest([sid,tid,original_role,text,session["date"]])[:32]
            history.append({"source_id":source_id,"message_id":source_id,"source_order":len(history),
                "session_id":sid,"role":role,"source_speaker":original_role,
                "content":text,"date":str(session["date"]),"source_kind":"released_benchmark_history"})
    if len({s["source_id"] for s in history}) != len(history):
        raise ValueError("source identity collision; retain session-occurrence normalization")
    return history, excluded


def prepare(dataset_key, data_dir, output_dir, existing=None):
    formal.require_colab_runtime()
    plan = formal.load_plan(formal.DEFAULT_PLAN)
    spec = plan["datasets"][dataset_key]
    source = source_file(dataset_key,data_dir,existing)
    examples = load_locomo(source) if dataset_key=="locomo10" else load_longmemeval(source)
    examples, session_audit = formal.normalize_session_repeats(examples,dataset_key)
    eligible = [e for e in examples if e.category not in spec.get("exclude_categories",[])]
    if len(examples)!=spec["expected_released_examples"] or len(eligible)!=spec["expected_eligible_examples"]:
        raise ValueError("full original source roster differs")
    groups = formal.history_groups(eligible,dataset_key,session_audit)
    aliases = formal.session_original_ids(session_audit)
    output = Path(output_dir)
    output.mkdir(parents=True,exist_ok=False)
    histories, cases, references = {}, [], []
    for example in eligible:
        history, excluded = history_at_cutoff(example,dataset_key)
        key = digest(history)
        histories.setdefault(key,history)
        cases.append({"case_id":example.example_id,"source_family_id":groups[example.example_id],
            "history_key":key,"input":example.query,"query_date":example.query_date,
            "calendar_time_available":True,"source_kind":"released_"+dataset_key,
            "future_session_ids_excluded":excluded,"source_cutoff_order_inclusive":len(history)-1})
        references.append({"case_id":example.example_id,"category":example.category,
            "query":example.query,"reference_answer":example.answer,"should_abstain":example.should_abstain,
            "original_evidence_ids":list(example.evidence_ids),
            "mapped_evidence_ids":formal.mapped_evidence_session_ids(example,aliases.get(example.example_id,{}))
                if dataset_key=="longmemeval_s_cleaned" else list(example.evidence_ids),
            "evidence_granularity":"session" if dataset_key=="longmemeval_s_cleaned" else "turn"})
    artifacts = {
        "generation_cases.jsonl":cases,
        "shared_histories.jsonl":[{"history_key":key,"history":history} for key,history in histories.items()],
        "scoring_references.jsonl":references,
    }
    for name, records in artifacts.items():
        with (output/name).open("x") as stream:
            for record in records:
                stream.write(canonical(record)+"\n")
    manifest = {"version":VERSION,"dataset_key":dataset_key,"questions":len(cases),
        "planned_three_arm_answers":len(cases)*3,"source":spec,
        "source_preparation_code_sha256":hash_file(__file__),
        "files_sha256":{name:hash_file(output/name) for name in artifacts},
        "question_ids_sha256":digest([c["case_id"] for c in cases]),
        "history_groups":len(set(groups.values())),"source_histories":len(histories),
        "session_occurrence_normalization":session_audit,
        "history_boundary":"all released at-or-before-question sessions; full original speakers retained",
        "gold_visibility":"references separate; no labels, answerability or evidence IDs in generation",
        "development_exposure":"retain prior exposure; no new independent holdout claim",
        "model_calls":0,"secret_access":False,"new_readout_generation_status":"pending"}
    manifest["manifest_sha256"] = digest(manifest)
    (output/"source_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    return {key:manifest[key] for key in ("dataset_key","questions","planned_three_arm_answers","source_histories","manifest_sha256")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",choices=DATASETS,required=True)
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--existing-source",type=Path)
    args = parser.parse_args()
    print(canonical(prepare(args.dataset,args.data_dir,args.output_dir,args.existing_source)),flush=True)


if __name__ == "__main__":
    main()
