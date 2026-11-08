"""Frozen, gold-free CACTUS continuation material preparation.

Real download/preparation requires genuine hosted Colab. Pure parsing/grouping
functions are also exercised with tiny invented fixtures in the cloud workspace.
This module never loads a model, reads a Secret, scores semantic correctness or
claims a memory effect. The released metadata is used for grouping only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import unicodedata
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
VERSION = "cactus-memory-continuations-v1"
REVISION = "237cb781cb6cbab609bc11f02a24ea6bc408ea4b"
DATASET = "LangAGI-Lab/cactus"
SOURCE_BYTES = 270309432
SOURCE_SHA256 = "be3421495f9dd76dd47d5fd4abd9fdabed9c97fcb7f7bba34ecfc78fe07d3d18"
DATA_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/cactus.json"
CARD_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/README.md"
MAX_FAMILIES = 200
ARMS = ["short_term", "raw_history_rag", "managed_memory_rag"]
ROW_KEYS = {"thought", "patterns", "intake_form", "cbt_technique", "cbt_plan", "attitude", "dialogue"}
SPEAKER = re.compile(r"(?m)^(Counselor|Client):[ \t]*")
INLINE_SPEAKER = re.compile(r"(?<!\n)(?<!\r)[ \t]+(?:Counselor|Client):")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def require_colab():
    path = ROOT / "app/colab_provider.py"
    spec = importlib.util.spec_from_file_location("cactus_colab_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.require_colab()


def identity_text(value):
    if type(value) is not str or not value.strip():
        raise ValueError("missing_group_identity")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def parse_dialogue(text):
    """Keep literal source spans; reject roles that need guessing.

    The conservative v1 grammar requires line-leading released speaker labels.
    An inline speaker-looking label is ambiguous quoted data, not a role switch.
    Such rows are retained in diagnostics instead of silently repaired.
    """
    if type(text) is not str or not text.strip() or len(text) > 100000:
        raise ValueError("invalid_dialogue_text")
    markers = list(SPEAKER.finditer(text))
    if not markers or text[:markers[0].start()].strip():
        raise ValueError("unlabelled_dialogue_prefix")
    if INLINE_SPEAKER.search(text):
        raise ValueError("ambiguous_inline_speaker_label")
    result = []
    for i, marker in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        start = marker.end()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start == end:
            raise ValueError("empty_dialogue_turn")
        role = "user" if marker.group(1) == "Client" else "assistant"
        if result and result[-1]["role"] == role:
            raise ValueError("nonalternating_dialogue_roles")
        result.append({"message_id": f"turn-{i:03d}", "role": role,
                       "source_label": marker.group(1), "content": text[start:end],
                       "source_start": start, "source_end": end,
                       "source_order": i, "date": None})
    if result[0]["role"] != "assistant" or len(result) < 8:
        raise ValueError("insufficient_or_unexpected_dialogue")
    return result


class UnionFind:
    def __init__(self, length):
        self.parent = list(range(length))

    def root(self, index):
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def join(self, a, b):
        a, b = self.root(a), self.root(b)
        self.parent[b] = a


def group_rows(rows):
    """Connect shared thought OR intake OR exact dialogue transitively.

    Components are source families, not people. Missing grouping identities
    invalidate grouping completeness, including for an otherwise unused row.
    """
    if type(rows) is not list:
        raise ValueError("released_json_must_be_a_list")
    union = UnionFind(len(rows))
    seen = {}
    identities, rejected = {}, []
    for index, row in enumerate(rows):
        try:
            if type(row) is not dict or set(row) != ROW_KEYS:
                raise ValueError("released_row_schema_mismatch")
            if type(row["dialogue"]) is not str or not row["dialogue"].strip():
                raise ValueError("invalid_dialogue_text")
            tokens = ["thought:" + digest(identity_text(row["thought"])),
                      "intake:" + digest(identity_text(row["intake_form"])),
                      "dialogue:" + digest(row["dialogue"])]
            identities[index] = tokens
            for token in tokens:
                if token in seen:
                    union.join(index, seen[token])
                else:
                    seen[token] = index
        except ValueError as error:
            rejected.append({"source_row_index": index, "status": "source_schema_rejected", "reason": str(error)})
    groups = defaultdict(list)
    for index in identities:
        groups[union.root(index)].append(index)
    result = []
    for members in groups.values():
        tokens = sorted({token for index in members for token in identities[index]})
        result.append({"family_id": "cactus-family-" + digest(tokens), "members": members})
    return sorted(result, key=lambda x: digest([VERSION, x["family_id"]])), rejected


def choose_cutoffs(turns, count=3):
    """Fixed within-dialogue User cutoffs, never synthetic session boundaries."""
    if type(count) is not int or count not in (2, 3):
        raise ValueError("cutoff_count_must_be_2_or_3")
    # At least four preceding released turns and a held-out next counselor turn.
    eligible = [i for i, turn in enumerate(turns) if i >= 4 and turn["role"] == "user"
                and i + 1 < len(turns) and turns[i + 1]["role"] == "assistant"]
    if len(eligible) < count:
        raise ValueError("insufficient_eligible_cutoffs")
    positions = [math.floor((len(eligible) - 1) * q / count) for q in range(1, count + 1)]
    selected = [eligible[i] for i in positions]
    if len(set(selected)) != count:
        raise ValueError("cutoff_selection_collision")
    return selected


def compile_rows(rows, *, family_limit=MAX_FAMILIES, cutoff_count=3, fixture_only=False):
    """Pure compiler. Pass only tiny invented rows outside real hosted Colab."""
    if type(family_limit) is not int or not 1 <= family_limit <= MAX_FAMILIES:
        raise ValueError("family_limit_outside_frozen_bound")
    groups, rejected = group_rows(rows)
    generation, references, controller, selected = [], [], [], []
    parsing_errors = []
    for group in groups:
        valid = []
        for index in group["members"]:
            try:
                turns = parse_dialogue(rows[index]["dialogue"])
                cutoffs = choose_cutoffs(turns, cutoff_count)
                valid.append((digest(rows[index]), index, turns, cutoffs))
            except ValueError as error:
                parsing_errors.append({"source_row_index": index, "family_id": group["family_id"],
                                       "status": "dialogue_rejected", "reason": str(error)})
        if not valid or len(selected) >= family_limit:
            continue
        # Deterministic, outcome-blind choice of exactly one released variant.
        record_hash, index, turns, cutoffs = min(valid, key=lambda item: item[0])
        selected.append(group["family_id"])
        dialogue = rows[index]["dialogue"]
        for cutoff in cutoffs:
            history = [{key: t[key] for key in ("message_id", "role", "content", "date", "source_order")}
                       for t in turns[:cutoff]]
            current = turns[cutoff]
            case_id = "cactus-" + digest([group["family_id"], record_hash, cutoff])[:32]
            case = {"case_id": case_id, "source_family_id": group["family_id"],
                    "task_kind": "natural_cbt_continuation", "history": history,
                    "input": current["content"], "input_message_id": current["message_id"],
                    "source_visible_end_exclusive": current["source_end"],
                    "history_cutoff_order_exclusive": cutoff,
                    "calendar_time_available": False,
                    "source_kind": "released_synthetic_cbt_dialogue"}
            case["case_sha256"] = digest(case)
            generation.append(case)
            references.append({"case_id": case_id, "case_sha256": case["case_sha256"],
                               "annotation_status": "pending_independent_source_annotation",
                               "memory_targets": None, "expected_current_state": None,
                               "semantic_gold_available": False, "future_counselor_answer_is_gold": False})
            controller.append({"case_id": case_id, "source_row_index": index,
                               "source_record_sha256": record_hash, "source_family_id": group["family_id"],
                               "source_dialogue_sha256": digest(dialogue), "cutoff": cutoff,
                               "family_member_rows": group["members"],
                               "visible_source_spans": [{key: turn[key] for key in
                                    ("message_id", "source_start", "source_end", "source_label")}
                                                        for turn in turns[:cutoff + 1]]})
    cases = len(generation)
    errors = rejected + parsing_errors
    summary = {"version": VERSION, "fixture_only": fixture_only,
               "dataset": DATASET, "dataset_revision": None if fixture_only else REVISION,
               "source_rows": len(rows), "connected_source_families": len(groups),
               "grouping_complete": not rejected, "source_errors": len(errors),
               "source_error_counts": dict(Counter(e["reason"] for e in errors)),
               "selected_families": len(selected), "requested_families": family_limit,
               "cutoffs_per_selected_family": cutoff_count, "planned_cases": cases,
               "planned_arm_rows": cases * len(ARMS), "arms": ARMS,
               "actual_generation_calls": 0, "actual_model_scores": 0,
               "independently_annotated_cases": 0,
               "semantic_memory_effect_scoring_ready": False,
               "source_preparation_complete": not rejected and len(selected) == family_limit,
               "generation_ready": False,
               "generation_blockers": ["current_product_replay_adapter_fidelity_not_yet_verified",
                                       "new_model_protocol_and_budget_not_yet_frozen"],
               "interpretation": "preflight only; no model effect, human or clinical result",
               "source_family_is_independent_human": False,
               "history_policy": "released dialogue prefix only; no invented calendar dates or sessions",
               "metadata_policy": "thought/intake grouping only; patterns, cbt_plan, intake and future reply absent from generation",
               "memory_scoring_policy": "independent source targets required; substring presence is not semantic correctness"}
    return {"generation": generation, "references": references, "controller": controller,
            "diagnostics": errors, "summary": summary}


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _write_rows(path, values):
    path.write_text("".join(canonical(value) + "\n" for value in values))


def write_package(output, compiled, *, source_identity):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "generation").mkdir()
    (output / "assessor_only").mkdir()
    (output / "controller_only").mkdir()
    _write_rows(output / "generation/cases.jsonl", compiled["generation"])
    _write_rows(output / "assessor_only/pending_annotations.jsonl", compiled["references"])
    _write_rows(output / "controller_only/source_map.jsonl", compiled["controller"])
    _write_rows(output / "source_diagnostics.jsonl", compiled["diagnostics"])
    _write_json(output / "preflight_summary.json", compiled["summary"])
    files = {str(path.relative_to(output)): file_hash(path) for path in output.rglob("*") if path.is_file()}
    manifest = {"version": VERSION, "source_identity": source_identity,
                "prepared_code_sha256": file_hash(Path(__file__)), "files_sha256": files,
                "old_runs_overwritten": False, "paid_calls_made": 0}
    manifest["manifest_sha256"] = digest(manifest)
    _write_json(output / "manifest.json", manifest)
    return compiled["summary"]


def download_cactus(destination):
    require_colab()  # Before dataset or metadata network access.
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError("symlinked_dataset_destination")
    with urlopen(CARD_URL, timeout=30) as response:
        card = response.read(262145)
    if len(card) > 262144 or not re.search(rb"(?m)^license:\s*gpl\s*$", card):
        raise ValueError("fixed_revision_gpl_metadata_unverified")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != SOURCE_BYTES or file_hash(destination) != SOURCE_SHA256:
            raise ValueError("existing_source_identity_mismatch")
    else:
        temporary = destination.with_name(destination.name + ".part")
        if temporary.exists():
            raise ValueError("partial_download_retained_choose_new_destination")
        size = 0
        with urlopen(DATA_URL, timeout=60) as response, temporary.open("xb") as stream:
            while block := response.read(1024 * 1024):
                size += len(block)
                if size > SOURCE_BYTES:
                    raise ValueError("download_exceeds_frozen_file_size")
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
        if size != SOURCE_BYTES or file_hash(temporary) != SOURCE_SHA256:
            raise ValueError("downloaded_source_identity_mismatch")
        temporary.replace(destination)
    return {"dataset": DATASET, "revision": REVISION, "source_bytes": SOURCE_BYTES,
            "source_sha256": SOURCE_SHA256, "license_metadata": "gpl",
            "license_version_specified_by_dataset_card": False,
            "dataset_card_sha256": hashlib.sha256(card).hexdigest()}


def prepare_dataset(source, output, *, family_limit=MAX_FAMILIES, cutoff_count=3):
    require_colab()  # Full released text is never read in the small-fixture cloud path.
    source = Path(source)
    if source.is_symlink() or not source.is_file():
        raise ValueError("unsafe_source_path")
    if source.stat().st_size != SOURCE_BYTES or file_hash(source) != SOURCE_SHA256:
        raise ValueError("source_identity_mismatch")
    # Revalidate current permission metadata at the frozen revision; no inferred licence.
    identity = download_cactus(source)
    with source.open() as stream:
        rows = json.load(stream)
    compiled = compile_rows(rows, family_limit=family_limit, cutoff_count=cutoff_count)
    return write_package(output, compiled, source_identity=identity)


def invented_fixture_rows():
    rows = []
    for family in range(2):
        pairs = []
        for index in range(7):
            pairs.extend([f"Counselor: FICTIONAL fixture question {index}.",
                          f"Client: FICTIONAL fixture report {family}-{index}."])
        pairs.append("Counselor: FICTIONAL final reply excluded from earlier cutoffs.")
        rows.append({"thought": f"FICTIONAL fixture thought {family}", "intake_form": f"FICTIONAL fixture intake {family}",
                     "patterns": ["FICTIONAL metadata never given to model"], "cbt_technique": "FICTIONAL technique",
                     "cbt_plan": "FICTIONAL future plan never given to model", "attitude": "neutral",
                     "dialogue": "\n".join(pairs)})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download")
    download.add_argument("--destination", type=Path, required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--families", type=int, default=MAX_FAMILIES)
    prepare.add_argument("--cutoffs", type=int, choices=(2, 3), default=3)
    fixture = commands.add_parser("fixture")
    fixture.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "download":
        result = download_cactus(args.destination)
    elif args.command == "prepare":
        result = prepare_dataset(args.source, args.output, family_limit=args.families, cutoff_count=args.cutoffs)
    else:
        result = write_package(args.output, compile_rows(invented_fixture_rows(), family_limit=2, fixture_only=True),
                               source_identity={"fixture_only": True, "source": "tiny invented cloud fixture"})
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
