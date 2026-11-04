"""Pinned public-gold dataset adapters; complete data are hosted-Colab only.

Pure ``parse_*`` functions accept tiny fixtures for deterministic unit tests.  All
real file loading/download entry points require hosted Colab.  Generation cases
never contain answer labels or gold evidence.  References are returned separately.
Source ``system`` messages remain untrusted historical data, never instructions.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "dataset_registry.json"
MAB_SOURCE = re.compile(r"^factconsolidation_(sh|mh)_(6k|32k|64k|262k)$")
PERSONA_TASKS = frozenset({
    "recall_user_shared_facts", "suggest_new_ideas",
    "track_full_preference_evolution", "recalling_the_reasons_behind_previous_updates",
    "provide_preference_aligned_recommendations", "generalizing_to_new_scenarios",
    "recalling_facts_mentioned_by_the_user",
})


def stable_hash(value: object) -> str:
    # Complete MAB contexts are shared by many QA rows.  Stream canonical JSON
    # rather than materialising a second full expanded corpus just to hash it.
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    for part in encoder.iterencode(value):
        digest.update(part.encode())
    return digest.hexdigest()


def require_colab_runtime() -> None:
    try:
        module_available = importlib.util.find_spec("google.colab") is not None
    except (ModuleNotFoundError, ValueError):
        module_available = False
    if not (Path("/content").is_dir() and os.environ.get("COLAB_RELEASE_TAG")
            and module_available):
        raise RuntimeError("Complete objective datasets may be downloaded/loaded only in hosted Google Colab; local tests use tiny pure-parser fixtures")


def load_registry(path: str | Path = REGISTRY_PATH) -> dict:
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    if registry.get("schema_version") != "objective-dataset-registry-v1":
        raise ValueError("unsupported objective dataset registry")
    for key, entry in registry.get("datasets", {}).items():
        if entry.get("enabled") is not True:
            continue
        if entry.get("license") != "MIT" or not entry.get("license_evidence"):
            raise ValueError(f"dataset {key} lacks an audited usable release license")
        if not re.fullmatch(r"[0-9a-f]{40}", entry.get("data_revision", "")):
            raise ValueError("data revision must be an immutable commit")
        for spec in entry.get("files", {}).values():
            url = urlparse(spec["url"])
            if (url.scheme != "https" or url.hostname != "huggingface.co"
                    or f"/resolve/{entry['data_revision']}/" not in url.path
                    or not isinstance(spec.get("bytes"), int) or spec["bytes"] <= 0):
                raise ValueError("unfrozen dataset file identity")
            if not (re.fullmatch(r"[0-9a-f]{64}", spec.get("sha256", ""))
                    or re.fullmatch(r"[0-9a-f]{40}", spec.get("git_blob_sha1", ""))):
                raise ValueError("dataset file lacks an immutable content hash")
    return registry


def dataset_spec(dataset_key: str, *, registry: dict | None = None) -> dict:
    entries = (registry or load_registry())["datasets"]
    if dataset_key not in entries or entries[dataset_key].get("enabled") is not True:
        raise ValueError("unknown or disabled objective dataset")
    return entries[dataset_key]


def verify_file(path: str | Path, spec: dict) -> dict:
    """Check released bytes; Git SHA-1 is explicitly *not* a byte SHA-256."""
    path = Path(path)
    size = path.stat().st_size
    if size != spec["bytes"]:
        raise ValueError(f"dataset byte size mismatch: {path.name}")
    sha256 = hashlib.sha256()
    git_hash = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha256.update(block)
            git_hash.update(block)
    if spec.get("sha256") and sha256.hexdigest() != spec["sha256"]:
        raise ValueError(f"dataset SHA-256 mismatch: {path.name}")
    if spec.get("git_blob_sha1") and git_hash.hexdigest() != spec["git_blob_sha1"]:
        raise ValueError(f"dataset Git blob identity mismatch: {path.name}")
    if not spec.get("sha256") and not spec.get("git_blob_sha1"):
        raise ValueError("file has no frozen content identity")
    return {"bytes": size, "sha256": sha256.hexdigest(),
            "git_blob_sha1": git_hash.hexdigest()}


def download_registered_dataset(dataset_key: str, data_dir: str | Path) -> dict[str, Path]:
    require_colab_runtime()
    spec = dataset_spec(dataset_key)
    target = Path(data_dir) / dataset_key
    target.mkdir(parents=True, exist_ok=True)
    paths, identities = {}, {}
    for role, file_spec in spec["files"].items():
        path = target / Path(file_spec["path"]).name
        if not path.exists():
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=target, delete=False) as handle:
                    temporary = Path(handle.name)
                    with urlopen(file_spec["url"], timeout=120) as response:
                        if urlparse(response.geturl()).scheme != "https":
                            raise ValueError("dataset redirect must remain HTTPS")
                        received = 0
                        while block := response.read(1024 * 1024):
                            received += len(block)
                            if received > file_spec["bytes"]:
                                raise ValueError("dataset download exceeded its frozen byte size")
                            handle.write(block)
                verify_file(temporary, file_spec)
                os.replace(temporary, path)
            finally:
                if temporary and temporary.exists():
                    temporary.unlink()
        identities[role] = verify_file(path, file_spec)
        paths[role] = path
    inventory = {"dataset_key": dataset_key, "data_revision": spec["data_revision"],
                 "registry_entry_fingerprint": stable_hash(spec), "files": identities}
    (target / "download_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    return paths


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _identifier(value: object, name: str) -> str:
    if value is None or isinstance(value, (dict, list, bool)) or not str(value).strip():
        raise ValueError(f"{name} must be a nonempty identifier")
    return str(value)


def _turn(scope: str, index: int, role: str, text: str, **extra) -> dict:
    return {"turn_id": f"{scope}:turn:{index}", "session_id": f"{scope}:released-sequence",
            "role": role, "text": text, "order": index, "observed_at": None,
            "sensitivity": "standard", **extra}


def _case(case_id: str, group_id: str, dataset: str, task: str, query: str,
          history: list[dict], options: list[str] | None = None, **extra) -> dict:
    return {"case_id": case_id, "group_id": group_id, "subject_id": group_id,
            "dataset_key": dataset, "task": task, "query": query,
            "options": options or [], "history": history, "events": [], "as_of": None,
            "ordering_kind": "source_order_no_calendar", **extra}


def _finish(cases: list[dict], references: list[dict]) -> tuple[list[dict], list[dict]]:
    ids = [case["case_id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate released question identity")
    if not cases:
        raise ValueError("dataset has no eligible cases")
    if ids != [reference["case_id"] for reference in references]:
        raise ValueError("generation cases and scoring references do not align")
    return cases, references


def parse_mab_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Adapt only released fact_sh/fact_mh. No answer-derived facts/events."""
    cases, references = [], []
    seen_sources = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("metadata"), dict):
            raise ValueError("MemoryAgentBench row requires metadata")
        source = row["metadata"].get("source", "")
        if not isinstance(source, str) or not MAB_SOURCE.fullmatch(source):
            continue
        if source in seen_sources:
            raise ValueError("duplicate MemoryAgentBench source/context variant")
        seen_sources.add(source)
        context = _text(row.get("context"), "context")
        questions, answers = row.get("questions"), row.get("answers")
        if not isinstance(questions, list) or not isinstance(answers, list) or len(questions) != len(answers):
            raise ValueError("questions and answer alias lists must align")
        qa_ids = row["metadata"].get("qa_pair_ids")
        if not isinstance(qa_ids, list) or len(qa_ids) != len(questions):
            raise ValueError("released qa_pair_ids must align with all questions")
        scope = f"mab:{source}:{hashlib.sha256(context.encode()).hexdigest()[:16]}"
        # Exact line bytes and separators are retained; these are benchmark facts,
        # not authenticated real-user statements and not actual dated sessions.
        history = [_turn(scope, i, "context", line, provenance_kind="benchmark_fact_text")
                   for i, line in enumerate(context.splitlines(keepends=True))]
        for index, (query, aliases, qa_id) in enumerate(zip(questions, answers, qa_ids)):
            query = _text(query, "question")
            if isinstance(aliases, str):
                aliases = [aliases]
            if not isinstance(aliases, list) or not aliases:
                raise ValueError("each MAB question must have released answer aliases")
            aliases = [_text(alias, "answer alias") for alias in aliases]
            case_id = f"mab:{_identifier(qa_id, 'qa_pair_id')}"
            group_id = "memoryagentbench:factconsolidation-family"
            cases.append(_case(case_id, group_id, "memoryagentbench_conflict", source,
                               query, history, history_scope_id=scope, source_question_index=index,
                               source_context_id=source, clustering_note="all variants conservatively grouped; exploratory family"))
            references.append({"case_id": case_id, "group_id": group_id,
                               "dataset_key": "memoryagentbench_conflict", "task": source,
                               "metric": "mab_substring_exact_match", "accepted_answers": aliases})
    return _finish(cases, references)


class _Components:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def root(self, node: str) -> str:
        self.parent.setdefault(node, node)
        if self.parent[node] != node:
            self.parent[node] = self.root(self.parent[node])
        return self.parent[node]

    def join(self, left: str, right: str) -> None:
        a, b = sorted((self.root(left), self.root(right)))
        self.parent[b] = a


def parse_personamem_rows(rows: list[dict], contexts: dict[str, list[dict]]) -> tuple[list[dict], list[dict]]:
    """Apply the official exclusive message cutoff *before* any history use."""
    if not isinstance(contexts, dict):
        raise ValueError("PersonaMem contexts must map shared_context_id to messages")
    components = _Components()
    for row in rows:
        persona = _identifier(row.get("persona_id"), "persona_id")
        context_id = _identifier(row.get("shared_context_id"), "shared_context_id")
        components.join(f"persona:{persona}", f"context:{context_id}")
    cases, references = [], []
    for row in rows:
        persona = _identifier(row.get("persona_id"), "persona_id")
        context_id = _identifier(row.get("shared_context_id"), "shared_context_id")
        question_id = _identifier(row.get("question_id"), "question_id")
        task = _text(row.get("question_type"), "question_type")
        if task not in PERSONA_TASKS:
            raise ValueError(f"unknown released PersonaMem question type: {task}")
        full_context = contexts.get(context_id)
        if not isinstance(full_context, list):
            raise ValueError("missing shared context")
        raw_cutoff = row.get("end_index_in_shared_context")
        if isinstance(raw_cutoff, bool) or not re.fullmatch(r"\d+", str(raw_cutoff)):
            raise ValueError("context cutoff must be an exclusive integer message index")
        cutoff = int(raw_cutoff)
        if not 0 < cutoff <= len(full_context):
            raise ValueError("context cutoff is outside the released message sequence")
        options = row.get("all_options")
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except json.JSONDecodeError:
                # The frozen release also serialises some lists with Python
                # single quotes. literal_eval accepts data literals only; never
                # execute a source string with eval or alter option labels.
                try:
                    options = ast.literal_eval(options)
                except (ValueError, SyntaxError) as exc:
                    raise ValueError("invalid released PersonaMem options literal") from exc
        if not isinstance(options, list) or len(options) != 4:
            raise ValueError("PersonaMem must retain its four released options")
        options = [_text(option, "option") for option in options]
        labels = [re.match(r"^\s*\(([a-d])\)", option, re.I) for option in options]
        if [match.group(1).lower() if match else None for match in labels] != list("abcd"):
            raise ValueError("released option labels/order must be a,b,c,d")
        correct = re.fullmatch(r"\s*\(([a-d])\)\s*", _text(row.get("correct_answer"), "correct_answer"), re.I)
        if not correct:
            raise ValueError("invalid released correct option")
        history = []
        for index, message in enumerate(full_context[:cutoff]):
            if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
                raise ValueError("invalid PersonaMem historical message")
            history.append(_turn(f"persona32k:{context_id}", index, message["role"],
                                 _text(message.get("content"), "historical content"),
                                 provenance_kind="synthetic_persona_history_data"))
        group_id = "personamem32k:" + hashlib.sha256(components.root(f"persona:{persona}").encode()).hexdigest()[:20]
        case_id = f"personamem32k:{persona}:{question_id}"
        cases.append(_case(case_id, group_id, "personamem_v1", task,
                           _text(row.get("user_question_or_message"), "query"), history, options,
                           persona_id=persona, source_context_id=context_id,
                           history_scope_id=f"persona32k:{context_id}:prefix:{cutoff}",
                           visible_context_end_exclusive=cutoff, topic=str(row.get("topic", ""))))
        references.append({"case_id": case_id, "group_id": group_id,
                           "dataset_key": "personamem_v1", "task": task,
                           "metric": "personamem_option_exact_match", "correct_option": correct.group(1).lower()})
    return _finish(cases, references)


def load_dataset_cases(dataset_key: str, input_path: str | Path, *,
                       context_path: str | Path | None = None) -> tuple[list[dict], list[dict]]:
    require_colab_runtime()  # Must run before any benchmark file is read.
    spec = dataset_spec(dataset_key)
    if dataset_key == "memoryagentbench_conflict":
        verify_file(input_path, spec["files"]["data"])
        import pyarrow.parquet as pq
        return parse_mab_rows(pq.read_table(input_path).to_pylist())
    if dataset_key == "personamem_v1":
        if context_path is None:
            raise ValueError("PersonaMem requires its pinned shared-context file")
        verify_file(input_path, spec["files"]["questions"])
        verify_file(context_path, spec["files"]["contexts"])
        with Path(input_path).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        contexts = {}
        with Path(context_path).open(encoding="utf-8") as handle:
            for line in handle:
                entry = json.loads(line)
                if not isinstance(entry, dict) or len(entry) != 1:
                    raise ValueError("expected one shared-context mapping per JSONL line")
                for context_id, messages in entry.items():
                    if context_id in contexts:
                        raise ValueError("duplicate shared-context identity")
                    contexts[context_id] = messages
        if len(rows) != spec["released_question_count"]:
            raise ValueError("released PersonaMem question count changed")
        return parse_personamem_rows(rows, contexts)
    raise ValueError("unsupported dataset adapter")


def inventory_cases(cases: list[dict], references: list[dict]) -> dict:
    _finish(cases, references)
    tasks = Counter(case["task"] for case in cases)
    cluster_sizes = Counter(case["group_id"] for case in cases)
    return {"questions": len(cases), "independent_clusters": len({c["group_id"] for c in cases}),
            "source_contexts": len({c["source_context_id"] for c in cases}),
            "personas": len({c["persona_id"] for c in cases if "persona_id" in c}),
            "distinct_visible_prefixes": len({c["history_scope_id"] for c in cases}),
            "questions_per_cluster": dict(sorted(cluster_sizes.items())),
            "tasks": dict(sorted(tasks.items())),
            "minimum_visible_turns": min(len(c["history"]) for c in cases),
            "maximum_visible_turns": max(len(c["history"]) for c in cases),
            "generation_cases_sha256": stable_hash(cases),
            "scoring_references_sha256": stable_hash(references),
            "calendar_timestamps_invented": False,
            "questions_are_independent_users": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect-sources", action="store_true")
    parser.add_argument("--dataset-key", choices=["memoryagentbench_conflict", "personamem_v1"])
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    if args.inspect_sources:
        print(json.dumps(load_registry(), indent=2))
        return
    if not args.dataset_key or not args.data_dir:
        parser.error("--dataset-key and --data-dir are required for Colab inventory")
    paths = download_registered_dataset(args.dataset_key, args.data_dir)
    cases, references = load_dataset_cases(args.dataset_key,
        paths.get("data", paths.get("questions")), context_path=paths.get("contexts"))
    print(json.dumps(inventory_cases(cases, references), indent=2))


if __name__ == "__main__":
    main()
