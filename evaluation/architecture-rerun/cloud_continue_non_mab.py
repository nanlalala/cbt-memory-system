"""One-shot orchestration of retained non-MAB evaluations in genuine Colab.

``run(globals(), repo=..., expected_commit=..., sources=...)`` starts a new,
parent-owned relay and a sequential background controller. The original entry
does not read Secrets. The separate ``run_recovered`` entry validates a sealed
source restoration and lazily reconstructs transports from the existing Colab
Secret, held only in parent memory after all source/budget checks. Neither entry
restores the paused legacy relay, creates a runtime, or retries an invocation.
Each scientific phase runs in a credential-free interpreter. Re-running the
cell reports the existing nonce and trace; it never creates another batch.

The caller must first supply the clean published checkout and original frozen
source folders. LongMemEval is inspected without model calls in this entry.
Its separate, previously pending paid budget is never implied by the CBT run.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import runpy
import stat
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
import uuid
import zipfile

VERSION = "hosted-non-mab-continuation-v1"
REMOTE = "https://github.com/nanlalala/cbt-memory-system.git"
DEFAULT_ROOT = "/content/capstone-non-mab-continuation-20261113-v1"
PROVIDER = "https://tokendance.space/gateway/v1"
ANSWER_IDENTITY = {"provider": PROVIDER, "model": "deepseek-v4.1-flash",
                   "returned_model": "deepseek-v4-1-flash-260910", "model_family": "deepseek"}
JUDGE_IDENTITY = {"provider": PROVIDER, "model": "qwen3.5-plus",
                  "returned_model": "qwen3.5-plus-2026-04-20", "model_family": "qwen"}
CBT_ORIGINAL_LIMITS = {ANSWER_IDENTITY["model"]: 4805, JUDGE_IDENTITY["model"]: 5032}
PROFESSIONAL_QUALITY_REPAIR_CAP = 24
LOCOMO_ORIGINAL_ANSWER_LIMIT = 4958
LOCOMO_NEW_ANSWER_CAP = 1638
LOCOMO_JUDGE_CAP = 5082
LME_PAID_CAPS = {ANSWER_IDENTITY["model"]: 39232, JUDGE_IDENTITY["model"]: 1650}
STATE_NAME = "_capstone_non_mab_continuation_v1"
RESTORED_CONNECTION_VERSION = "hosted-non-mab-restored-connection-v1"
DEFAULT_RECOVERED_ROOT = "/content/capstone-non-mab-restored-continuation-20261113-v1"
RESTORE_ARCHIVE_SHA256 = "f9c4159abc0d7853c1640cfbb7221139a79ababc012dbb960dfc3a227c65a430"
RESTORE_FILE_COUNT = 11633
RESTORE_UNCOMPRESSED_BYTES = 711889502
RESTORE_ROOTS = {
    "content/cbt-v4-isolated-cbt-full/professional7",
    "content/cbt-v4-isolated-cbt-full/professional7_memory",
    "content/cbt-v4-isolated-cbt-full/professional7_quality",
    "content/cbt-v4-isolated-cbt-full/cactus600",
    "content/cbt-v4-isolated-cbt-full/cactus600_memory",
    "content/cbt-v4-repair-general/locomo1540",
}


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _read(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("symlinked orchestration artifact")
    return json.loads(path.read_text())


def _write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def _hosted():
    if not (os.environ.get("COLAB_RELEASE_TAG") and Path("/content").is_dir()
            and importlib.util.find_spec("google.colab") is not None):
        raise RuntimeError("genuine hosted Colab required; zero model calls")


def _content_path(value, *, exists=False):
    path = Path(value)
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("absolute non-symlinked /content path required")
    path = path.resolve()
    if not path.is_relative_to("/content") or path == Path("/content"):
        raise ValueError("named /content path required")
    if exists and not path.is_dir():
        raise FileNotFoundError("retained hosted source folder is unavailable; zero model calls")
    return path


def _checkout(repo, expected_commit):
    if not isinstance(expected_commit, str) or re.fullmatch("[0-9a-f]{40}", expected_commit) is None:
        raise ValueError("explicit published forty-character commit required")
    repo = _content_path(repo, exists=True)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    if (git("rev-parse", "HEAD") != expected_commit or git("status", "--porcelain")
            or git("remote", "get-url", "origin") != REMOTE):
        raise ValueError("published checkout must match the pinned clean revision")
    for relative in ("evaluation/architecture-rerun/hosted_relay.py",
                     "evaluation/architecture-rerun/non_mab_worker.py",
                     "evaluation/architecture-rerun/formal_continuation.py",
                     "evaluation/cbt-memory-effect/architecture_retest/continuation.py"):
        if not (repo / relative).is_file():
            raise ValueError("published continuation code is incomplete")
    return repo


def _source_paths(sources):
    if type(sources) is not dict or set(sources) != {"professional7", "cactus600", "locomo1540", "longmemeval500"}:
        raise ValueError("explicit complete non-MAB source map required")
    result = {}
    for cohort in ("professional7", "cactus600"):
        value = sources[cohort]
        if type(value) is not dict or set(value) != {"reader", "memory", "old_quality"}:
            raise ValueError("CBT reader, memory and original quality paths required")
        result[cohort] = {name: str(_content_path(path, exists=(name != "old_quality"))) for name, path in value.items()}
    value = sources["locomo1540"]
    if type(value) is not dict or set(value) != {"reader", "old_quality"}:
        raise ValueError("original LoCoMo reader and quality path required")
    # Its interrupted judge directory may not yet exist. It remains named in
    # the frozen source map so an undisclosed prior paid run is not ignored.
    result["locomo1540"] = {name: str(_content_path(path, exists=(name == "reader")))
                            for name, path in value.items()}
    value = sources["longmemeval500"]
    if value is None:
        result["longmemeval500"] = None
    elif type(value) is dict and set(value) == {"reader"}:
        result["longmemeval500"] = {"reader": str(_content_path(value["reader"], exists=True))}
    else:
        raise ValueError("LongMemEval must name its frozen source folder or be explicitly unavailable")
    return result


def _file_hash(path):
    checksum = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1_048_576), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _restore_member(name):
    path = PurePosixPath(name)
    if (not name or "\\" in name or path.is_absolute() or ".." in path.parts
            or "." in path.parts or str(path) != name or len(path.parts) < 4
            or "/".join(path.parts[:3]) not in RESTORE_ROOTS
            or path.suffix not in {".json", ".jsonl"}):
        raise ValueError("restoration member is outside the exact frozen non-MAB roots")
    return path


def _regular_file(path):
    path = Path(path)
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("regular non-symlinked restoration file required")
    return path


def _restoration_provenance(archive_path, receipt_path, sources, *, restore_base=Path("/")):
    """Pure, zero-call validation of restored bytes; production always uses /.

    The explicit base permits tiny invented offline fixtures. It does not grant
    hosted execution or construct credentials. Only run_recovered admits work.
    """
    archive_path, receipt_path = _regular_file(archive_path), _regular_file(receipt_path)
    if _file_hash(archive_path) != RESTORE_ARCHIVE_SHA256:
        raise ValueError("original sealed non-MAB restoration archive differs")
    receipt = _read(receipt_path)
    if (receipt.get("version") != "capstone-original-non-mab-source-restore-v1"
            or receipt.get("status") != "restored_original_bytes_zero_model_calls"
            or receipt.get("archive_sha256") != RESTORE_ARCHIVE_SHA256
            or receipt.get("model_calls") != 0 or receipt.get("mab_accessed") is not False
            or receipt.get("files") != RESTORE_FILE_COUNT
            or type(receipt.get("moved_roots")) is not list
            or len(receipt["moved_roots"]) != len(RESTORE_ROOTS)
            or set(receipt["moved_roots"]) != RESTORE_ROOTS):
        raise ValueError("completed original-source restoration receipt required")
    base = Path(restore_base).resolve()
    expected_roots = {base.joinpath(*PurePosixPath(root).parts) for root in RESTORE_ROOTS}
    source_bindings = {
        ("professional7", "reader"): "content/cbt-v4-isolated-cbt-full/professional7",
        ("professional7", "memory"): "content/cbt-v4-isolated-cbt-full/professional7_memory",
        ("professional7", "old_quality"): "content/cbt-v4-isolated-cbt-full/professional7_quality",
        ("cactus600", "reader"): "content/cbt-v4-isolated-cbt-full/cactus600",
        ("cactus600", "memory"): "content/cbt-v4-isolated-cbt-full/cactus600_memory",
        ("cactus600", "old_quality"): "content/cbt-v4-isolated-cbt-full/cactus600_quality",
        ("locomo1540", "reader"): "content/cbt-v4-repair-general/locomo1540",
        ("locomo1540", "old_quality"): "content/cbt-v4-repair-general/locomo1540-judge",
    }
    if any(Path(sources[cohort][kind]).resolve() != base.joinpath(*PurePosixPath(relative).parts)
           for (cohort, kind), relative in source_bindings.items()):
        raise ValueError("restored source mapping differs from the original six roots")
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or "restore_manifest.json" not in names:
            raise ValueError("restoration ZIP has duplicate members or no manifest")
        raw_manifest = archive.read("restore_manifest.json")
        manifest = json.loads(raw_manifest)
        if (manifest.get("version") != "capstone-original-non-mab-source-restore-v1"
                or manifest.get("source_only") is not True or manifest.get("model_calls") != 0
                or manifest.get("mab_accessed") is not False or manifest.get("credentials_included") is not False
                or manifest.get("restore_base") != "/content"):
            raise ValueError("restoration manifest is not the sealed source-only import")
        entries = manifest["entries"]
        by_name = {entry["path"]: entry for entry in entries}
        if (len(entries) != RESTORE_FILE_COUNT or len(by_name) != len(entries)
                or sum(entry["bytes"] for entry in entries) != RESTORE_UNCOMPRESSED_BYTES
                or set(names) != set(by_name) | {"restore_manifest.json"}):
            raise ValueError("complete original restoration roster differs")
        found = set()
        for info in infos:
            if info.filename == "restore_manifest.json":
                continue
            member = _restore_member(info.filename)
            entry = by_name[info.filename]
            if (info.is_dir() or stat.S_ISLNK(info.external_attr >> 16)
                    or type(entry["bytes"]) is not int or entry["bytes"] < 0
                    or info.file_size != entry["bytes"]):
                raise ValueError("restoration member type or length differs")
            actual = _regular_file(base.joinpath(*member.parts))
            if actual.stat().st_size != entry["bytes"] or _file_hash(actual) != entry["sha256"]:
                raise ValueError("restored source bytes differ from their original frozen bytes")
            found.add(actual)
    actual_files = set()
    for root in expected_roots:
        if root.is_symlink() or not root.is_dir():
            raise ValueError("a restored source root is missing or symlinked")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError("symlinked restored source tree")
            if path.is_file() and path.name != ".run.lock":
                actual_files.add(path)
    if actual_files != found:
        raise ValueError("restored tree contains an omitted or extra source artifact")
    return {"version": RESTORED_CONNECTION_VERSION,
        "origin": "reconstructed_from_existing_colab_secret", "secret_name": "Capstone",
        "credential_location": "hosted_Colab_parent_memory_only",
        "source_restoration_receipt": str(receipt_path.resolve()),
        "source_restoration_receipt_sha256": _file_hash(receipt_path),
        "source_restore_archive": str(archive_path.resolve()),
        "source_restore_archive_sha256": RESTORE_ARCHIVE_SHA256,
        "source_manifest_sha256": sha256(raw_manifest).hexdigest(),
        "restored_original_files_verified": len(found),
        "restored_original_bytes_verified": RESTORE_UNCOMPRESSED_BYTES,
        "verification_model_calls": 0, "mab_accessed": False, "verified_at_utc": _utc()}


def _load_recovered_transports(repo):
    """Called only inside the hosted parent after source preparation/budget.

    The existing notebook Secret is read once, never put in a child environment,
    notebook output or manifest. Its value is held only by the parent transport.
    """
    _hosted()
    api = runpy.run_path(str(Path(repo) / "evaluation/cbt-memory-effect/architecture_retest/transport.py"))
    from google.colab import userdata
    key = userdata.get("Capstone")
    try:
        return {identity["model"]: api["HostedGatewayTransport"](identity, key)
                for identity in (ANSWER_IDENTITY, JUDGE_IDENTITY)}
    finally:
        del key


class _RecoveredConnection:
    def __init__(self, repo, provenance):
        if (provenance.get("version") != RESTORED_CONNECTION_VERSION
                or provenance.get("origin") != "reconstructed_from_existing_colab_secret"
                or provenance.get("source_restore_archive_sha256") != RESTORE_ARCHIVE_SHA256
                or provenance.get("restored_original_files_verified") != RESTORE_FILE_COUNT):
            raise ValueError("explicit verified restored-connection provenance required")
        self.repo, self.provenance = Path(repo), dict(provenance)
        self._lock, self._loaded = threading.Lock(), None
        self._load_failed = False
        self.delegates = {identity["model"]: _RecoveredDelegate(self, identity)
                          for identity in (ANSWER_IDENTITY, JUDGE_IDENTITY)}

    def assert_current(self):
        if self._load_failed:
            raise RuntimeError("restored connection initialization failed; no automatic credential retry")
        marker = _regular_file(self.provenance["source_restoration_receipt"])
        if _file_hash(marker) != self.provenance["source_restoration_receipt_sha256"]:
            raise ValueError("restoration receipt changed before provider admission")
        if (Path(DEFAULT_ROOT) / "entry.json").exists():
            raise RuntimeError("original continuation invocation exists; reconcile before restored continuation")

    def send(self, identity, payload):
        self.assert_current()
        with self._lock:
            # Another first send may have failed while this thread waited.
            # Recheck under the shared lock before ever reading the Secret.
            self.assert_current()
            if self._loaded is None:
                try:
                    loaded = _load_recovered_transports(self.repo)
                    if (set(loaded) != set(self.delegates) or any(
                            not callable(loaded[item["model"]]) or loaded[item["model"]].identity != item
                            for item in (ANSWER_IDENTITY, JUDGE_IDENTITY))):
                        raise ValueError("reconstructed gateway identities differ")
                    self._loaded = loaded
                except Exception:
                    self._load_failed = True
                    raise RuntimeError("restored connection initialization failed; credential values withheld") from None
        # A frontend Secret permission wait may have yielded to other work.
        self.assert_current()
        return self._loaded[identity["model"]](payload)


class _RecoveredDelegate:
    def __init__(self, connection, identity):
        self.connection, self.identity = connection, dict(identity)

    def __call__(self, payload):
        return self.connection.send(self.identity, payload)


def _parents(namespace, recovered_connection=None):
    delegates = {}
    if recovered_connection is None:
        for name, identity in (("v5_answer", ANSWER_IDENTITY), ("v5_judge", JUDGE_IDENTITY)):
            parent = namespace.get(name)
            delegate = getattr(parent, "delegate", None)
            if not callable(delegate) or getattr(delegate, "identity", None) != identity:
                raise RuntimeError("original parent-owned delegate unavailable; no Secret read; zero model calls")
            delegates[identity["model"]] = delegate
    else:
        if not isinstance(recovered_connection, _RecoveredConnection):
            raise ValueError("verified explicit restored connection required")
        recovered_connection.assert_current()
        delegates = recovered_connection.delegates
    for name in ("v6_cbt_process", "v5_cbt_process", "v5_general_process"):
        process = namespace.get(name)
        if process is not None and callable(getattr(process, "poll", None)) and process.poll() is None:
            raise RuntimeError("an older study process is active; zero new model calls")
    def active(value):
        if isinstance(value, threading.Thread):
            return value.is_alive()
        if isinstance(value, dict):
            return any(isinstance(item, threading.Thread) and item.is_alive() for item in value.values())
        if isinstance(value, (list, tuple, set)):
            return any(isinstance(item, threading.Thread) and item.is_alive() for item in value)
        return False
    if (any(active(value) for name, value in namespace.items()
            if name.startswith(("v5_", "v6_", "_v5_", "_v6_")) and ("thread" in name or "watcher" in name))
            or any(t.is_alive() and t.name.startswith(("cbt-v5-", "cbt-v6-")) for t in threading.enumerate())):
        raise RuntimeError("an older study thread or grading watcher is active; zero new model calls")
    return delegates


def _prior_recovery(sources):
    # The older top cell may have been clicked after the last Drive read. Its
    # independently authorized 24 calls cannot be silently spent twice.
    legacy = Path("/content/capstone-professional7-regrade-20261113-v1/entry.json")
    quality = Path(sources["professional7"]["reader"]).parent / "professional7_quality_v2"
    if legacy.exists() or _rows(quality / "intents.jsonl"):
        raise RuntimeError("earlier professional quality-v2 recovery exists; reconcile it before spending the repair allowance")


def _rows(path):
    path = Path(path)
    if not path.exists():
        return []
    if path.is_symlink():
        raise ValueError("symlinked request journal")
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("incomplete request journal; reconciliation required")
    return [json.loads(line) for line in raw.split(b"\n") if line]


def _paid_census(folder, identity, *, optional=False):
    folder = Path(folder)
    if not folder.exists() and optional:
        return {"folder": str(folder), "intents": 0, "absent": True}
    protocol = _read(folder / "protocol.json")
    if protocol.get("provider") != identity or protocol.get("fixture_only") is not False:
        raise ValueError("retained journal provider/provenance differs")
    rows = _rows(folder / "intents.jsonl")
    keys = [r["key"] for r in rows]
    if len(set(keys)) != len(keys) or any(r.get("protocol_sha256") != protocol["protocol_sha256"] for r in rows):
        raise ValueError("duplicate or foreign retained paid intent")
    return {"folder": str(folder), "intents": len(rows),
            "protocol_sha256": protocol["protocol_sha256"],
            "intents_sha256": sha256((folder / "intents.jsonl").read_bytes()).hexdigest()}


def _budget(sources, prepared):
    census = {}
    answer_old = judge_old = 0
    for cohort in ("professional7", "cactus600"):
        original = sources[cohort]
        census[cohort] = {}
        for name, identity in (("reader", ANSWER_IDENTITY), ("memory", JUDGE_IDENTITY), ("old_quality", JUDGE_IDENTITY)):
            row = _paid_census(original[name], identity, optional=(name == "old_quality"))
            census[cohort][name] = row
            if identity == ANSWER_IDENTITY:
                answer_old += row["intents"]
            else:
                judge_old += row["intents"]
    remaining = {ANSWER_IDENTITY["model"]: CBT_ORIGINAL_LIMITS[ANSWER_IDENTITY["model"]] - answer_old,
                 JUDGE_IDENTITY["model"]: CBT_ORIGINAL_LIMITS[JUDGE_IDENTITY["model"]] - judge_old
                     + PROFESSIONAL_QUALITY_REPAIR_CAP}
    need = {ANSWER_IDENTITY["model"]: sum(prepared[c]["maximum_new_generation_requests"] for c in ("professional7", "cactus600")),
            JUDGE_IDENTITY["model"]: sum(prepared[c]["maximum_new_assertion_requests"]
                + prepared[c]["quality_after_complete_grid"]["with_independent_10_percent_repeats"]
                for c in ("professional7", "cactus600"))}
    locomo_old = _paid_census(sources["locomo1540"]["reader"], ANSWER_IDENTITY)
    locomo_judge_old = _paid_census(sources["locomo1540"]["old_quality"], JUDGE_IDENTITY, optional=True)
    locomo = prepared["locomo1540"]
    locomo_new = locomo["maximum_attempts"]
    locomo_judge = locomo["maximum_independent_judge_calls"]
    if (any(type(v) is not int or v < 0 or v > remaining[k] for k, v in need.items())
            or not 0 <= locomo_new <= min(LOCOMO_NEW_ANSWER_CAP, LOCOMO_ORIGINAL_ANSWER_LIMIT - locomo_old["intents"])
            or not 0 <= locomo_judge <= LOCOMO_JUDGE_CAP - locomo_judge_old["intents"]):
        raise ValueError("full remaining plan exceeds its retained finite allowance; zero new paid calls")
    return {"version": VERSION, "original_cbt_envelope": CBT_ORIGINAL_LIMITS,
            "cbt_actual_local_paid_intents": {ANSWER_IDENTITY["model"]: answer_old, JUDGE_IDENTITY["model"]: judge_old},
            "separate_professional7_quality_repair_allowance": PROFESSIONAL_QUALITY_REPAIR_CAP,
            "cbt_remaining_after_paid_intents": remaining, "cbt_new_upper_bounds": need,
            "census": census, "locomo_original_answer_envelope": LOCOMO_ORIGINAL_ANSWER_LIMIT,
            "inherited_receipts_outside_current_local_envelope": {cohort: {
                "generation_received": max(0, prepared[cohort].get("inherited_generation_received", 0) - census[cohort]["reader"]["intents"]),
                "assertion_received": max(0, prepared[cohort].get("inherited_assertion_received", 0) - census[cohort]["memory"]["intents"])}
                for cohort in ("professional7", "cactus600")},
            "locomo_original_paid_intents": locomo_old, "locomo_original_judge_paid_intents": locomo_judge_old,
            "locomo_new_upper_bounds": {ANSWER_IDENTITY["model"]: locomo_new, JUDGE_IDENTITY["model"]: locomo_judge},
            "inherited_legacy_receipts": "retained separately by source protocols; not a second send or a second local paid intent",
            "mab_accessed": False, "created_at_utc": _utc()}


class _BoundedDelegate:
    def __init__(self, delegate, semaphore, diagnostics, mutex, before_send=None):
        self.delegate, self.identity = delegate, dict(delegate.identity)
        self._semaphore, self._diagnostics, self._mutex = semaphore, diagnostics, mutex
        self._before_send = before_send

    def __call__(self, payload):
        with self._semaphore:
            try:
                if self._before_send is not None:
                    self._before_send()
                return self.delegate(payload)
            except Exception as exc:
                # Do not stringify exceptions, response bodies, headers or keys.
                row = {"at_utc": _utc(), "model": self.identity["model"], "error_type": type(exc).__name__}
                if isinstance(exc, HTTPError):
                    row["http_status"] = int(exc.code)
                with self._mutex, self._diagnostics.open("a") as stream:
                    stream.write(json.dumps(row) + "\n"); stream.flush(); os.fsync(stream.fileno())
                raise


def _launch(state, phase, action, *, cohort, source=None, limits=None, answer_cap=0, judge_cap=0):
    relay = state["relay"]
    model_limits = limits or {ANSWER_IDENTITY["model"]: 0, JUDGE_IDENTITY["model"]: 0}
    source_only = action.endswith("-source") or action == "lme-source-only"
    if (source_only and any(model_limits.values())) or (not source_only and not any(model_limits.values())):
        raise ValueError("phase call limits differ from source-only/paid action")
    job = state["nonce"] + ":" + phase
    relay.register_job(job, model_limits)
    worker_root = state["root"] / "workers" / phase
    worker_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    scripts = {"non_mab_worker.py": (state["repo"] / "evaluation/architecture-rerun/non_mab_worker.py").read_bytes(),
               "hosted_relay.py": (state["repo"] / "evaluation/architecture-rerun/hosted_relay.py").read_bytes()}
    for name, body in scripts.items():
        (worker_root / name).write_bytes(body)
    manifest = {"version": VERSION, "nonce": state["nonce"], "phase": phase, "action": action,
                "selected_repo": str(state["repo"]), "selected_commit": state["commit"],
                "run_root": str(state["root"] / "studies" / cohort), "cohort": cohort,
                "source": source, "socket": relay.endpoint, "job_id": job,
                "answer_max_new_calls": answer_cap, "judge_max_new_calls": judge_cap,
                "relay_limits": model_limits,
                "helper_sha256": {name: sha256(body).hexdigest() for name, body in scripts.items()},
                "credentials": "only hosted parent-memory transports own provider credentials",
                "connection_origin": state.get("connection_origin", "original_parent_delegate"),
                "created_at_utc": _utc()}
    manifest_path = worker_root / "execution_manifest.json"
    _write(manifest_path, manifest)
    _update(state, phase=phase, status="running", worker_root=str(worker_root))
    with (worker_root / "worker.log").open("ab", buffering=0) as log:
        process = subprocess.Popen([sys.executable, "-B", str(worker_root / "non_mab_worker.py"), "--manifest", str(manifest_path)],
            cwd=worker_root, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            env=state["relay_api"]["sanitized_child_environment"](), start_new_session=True)
    state["process"] = process
    _write(worker_root / "process.json", {"pid": process.pid, "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest()})
    last_notice = time.monotonic()
    while process.poll() is None:
        time.sleep(1)
        if time.monotonic() - last_notice >= 60:
            print(json.dumps({"nonce": state["nonce"], "phase": phase, "status": "running", "pid": process.pid}), flush=True)
            last_notice = time.monotonic()
    state["process"] = None
    if process.returncode != 0 or not (worker_root / "complete.json").is_file():
        raise RuntimeError("isolated phase failed; retained invocation will not be automatically retried")
    result = _read(worker_root / "complete.json")
    # A newly ambiguous send must never cause the next paid phase to start.
    with (state["root"] / "relay" / "events.jsonl").open() as events:
        unknown = any((r := json.loads(line)).get("job_id") == job and r.get("status") == "unknown" for line in events)
    if unknown:
        raise RuntimeError("new ambiguous provider send; stop this cohort without retry")
    _phase_result(action, cohort, result, Path(manifest["run_root"]))
    _update(state, phase=phase, status="phase_complete", result_file=str(worker_root / "complete.json"))
    if action in {"cbt-quality-run", "locomo-judge-run"}:
        print(json.dumps({"CAPSTONE_NON_MAB_AGGREGATE": aggregate_results(state["root"])}, ensure_ascii=False, allow_nan=False), flush=True)
    export_checkpoint(state["root"], complete=False)
    return result


def _phase_result(action, cohort, result, output):
    """Separate terminated processing from a valid/effective final score."""
    paid_folders = {"cbt-run": output / "reader", "cbt-quality-run": output / "quality_v2",
                    "locomo-run": output, "locomo-judge-run": output / "formal_judge"}
    if action not in paid_folders:
        return
    invocation = _read(paid_folders[action] / "last_invocation.json")
    if invocation.get("errors"):
        raise RuntimeError("scientific phase retained item errors; no automatic continuation")
    if action == "cbt-run":
        expected = 7 if cohort == "professional7" else 600
        if (result.get("planned_cases") != expected or result.get("completed_cases") != expected
                or result.get("unknown_receipts") != 0 or result.get("unbound_answer_files")):
            raise RuntimeError("original CBT grid is incomplete or has unbound receipts")
        if cohort == "professional7" and any(result.get("arms", {}).get(arm, {}).get("received") != 7
                for arm in ("no_rag", "raw_history_rag", "managed_memory_rag")):
            raise RuntimeError("professional seven must have all 21 bound answers before full-grid grading")
    elif result.get("complete") is not True or result.get("unknown_receipts", 0) != 0:
        raise RuntimeError("scientific phase processing is incomplete; no final-score claim")


def _update(state, **fields):
    state["status"].update(fields, updated_at_utc=_utc())
    state["status"]["relay"] = state["relay"].snapshot()
    _write(state["root"] / "status.json", state["status"])
    print(json.dumps({"nonce": state["nonce"], **fields}), flush=True)


def _controller(state):
    try:
        prepared = {}
        for cohort in ("professional7", "cactus600"):
            prepared[cohort] = _launch(state, cohort + "-source", "cbt-source", cohort=cohort, source=state["sources"][cohort])
        prepared["locomo1540"] = _launch(state, "locomo1540-source", "locomo-source", cohort="locomo1540", source=state["sources"]["locomo1540"])
        lme = state["sources"]["longmemeval500"]
        if lme is not None:
            prepared["longmemeval500"] = _launch(state, "longmemeval500-source", "lme-source-only", cohort="longmemeval500", source=lme)
        else:
            prepared["longmemeval500"] = {"status": "retained_source_unavailable", "model_calls": 0, "paid_execution_authorized": False}
        _write(state["root"] / "source_preparation.json", prepared)
        _prior_recovery(state["sources"])
        _parents(state["namespace"], state.get("recovered_connection"))
        plan = _budget(state["sources"], prepared)
        _write(state["root"] / "finite_budget_plan.json", plan)
        state["status"]["cohorts"] = {}
        for cohort in ("professional7", "cactus600", "locomo1540"):
            try:
                _parents(state["namespace"], state.get("recovered_connection"))
                budget = prepared[cohort]
                if cohort != "locomo1540":
                    answer_cap, assertion_cap = budget["maximum_new_generation_requests"], budget["maximum_new_assertion_requests"]
                    _launch(state, cohort + "-continuation", "cbt-run", cohort=cohort,
                        limits={ANSWER_IDENTITY["model"]: answer_cap, JUDGE_IDENTITY["model"]: assertion_cap},
                        answer_cap=answer_cap, judge_cap=assertion_cap)
                    quality = _launch(state, cohort + "-quality-source", "cbt-quality-source", cohort=cohort)
                    cap = quality["maximum_attempts"]
                    if cap > budget["quality_after_complete_grid"]["with_independent_10_percent_repeats"] or quality.get("ready") is not True:
                        raise ValueError("quality plan exceeds frozen full-grid allowance or request preflight")
                    _launch(state, cohort + "-quality", "cbt-quality-run", cohort=cohort,
                        limits={JUDGE_IDENTITY["model"]: cap}, judge_cap=cap)
                else:
                    cap = budget["maximum_attempts"]
                    _launch(state, cohort + "-continuation", "locomo-run", cohort=cohort,
                        limits={ANSWER_IDENTITY["model"]: cap}, answer_cap=cap)
                    quality = _launch(state, cohort + "-judge-source", "locomo-judge-source", cohort=cohort)
                    cap = quality["maximum_attempts"]
                    if cap > budget["maximum_independent_judge_calls"]:
                        raise ValueError("LoCoMo judging exceeds its original independent judge allowance")
                    _launch(state, cohort + "-judge", "locomo-judge-run", cohort=cohort,
                        limits={JUDGE_IDENTITY["model"]: cap}, judge_cap=cap)
                state["status"]["cohorts"][cohort] = {"status": "processing_complete", "effect_scores": "see original-method scientific summaries and valid-score coverage"}
            except Exception as exc:
                state["status"]["cohorts"][cohort] = {"status": "stopped_without_retry", "error_type": type(exc).__name__}
                _update(state, phase=cohort, status="cohort_stopped_without_retry", error_type=type(exc).__name__)
        state["status"]["cohorts"]["longmemeval500"] = {"status": "source_only_paid_execution_pending", "model_calls": 0,
            "explicit_paid_authorization_recorded": state["allow_longmemeval_paid"], "paid_execution_supported_in_this_entry": False}
        _update(state, phase="all", status="finished", result_root=str(state["root"] / "studies"))
    except Exception as exc:
        _update(state, phase="preflight_or_budget", status="stopped_without_retry", error_type=type(exc).__name__)
    finally:
        # No child remains active here. Never close or modify the old relay.
        if state.get("process") is None:
            state["relay"].close()
            aggregate = aggregate_results(state["root"])
            print(json.dumps({"CAPSTONE_NON_MAB_AGGREGATE": aggregate}, ensure_ascii=False, allow_nan=False), flush=True)
            export_checkpoint(state["root"], complete=True)


def export_checkpoint(root=DEFAULT_ROOT, *, complete=False):
    """Write a private downloadable checkpoint from this invocation only.

    The small archive is refreshed after each completed phase. The final full
    archive includes immutable sources and exact requests/receipts, but neither
    this process nor its children ever owns a provider key on disk. No original
    cohort, MAB folder, browser state or notebook is traversed.
    """
    root = _content_path(root, exists=True)
    export = root / "exports"
    export.mkdir(exist_ok=True, mode=0o700)
    filename = "non_mab_full_audit_PRIVATE.zip" if complete else "non_mab_progress_PRIVATE.zip"
    target, temp = export / filename, export / (filename + ".tmp")
    top = {"entry.json", "status.json", "source_preparation.json", "finite_budget_plan.json", "provider_diagnostics.jsonl", "aggregate_results.json"}
    summary_names = {"protocol.json", "budget_plan.json", "complete.json", "progress.json", "summary.json",
                     "report.json", "last_invocation.json", "error.json", "process.json", "execution_manifest.json"}
    manifest = []
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if relative.parts[0] == "exports" or path.is_symlink() or not path.is_file():
                continue
            allowed = relative.parts[0] in top or relative.parts[0] in {"studies", "workers", "relay"}
            if not allowed or (not complete and path.name not in top | summary_names):
                continue
            checksum = sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1_048_576), b""):
                    checksum.update(block)
            archive.write(path, str(relative))
            manifest.append({"path": str(relative), "bytes": path.stat().st_size, "sha256": checksum.hexdigest()})
        archive.writestr("export_manifest.json", json.dumps({"version": VERSION, "complete_audit": complete,
            "created_at_utc": _utc(), "mab_accessed": False, "files": manifest}, indent=2))
    temp.replace(target)
    receipt = {"path": str(target), "bytes": target.stat().st_size, "sha256": sha256(target.read_bytes()).hexdigest(),
               "complete_audit": complete, "created_at_utc": _utc()}
    _write(export / (filename + ".json"), receipt)
    print(json.dumps({"checkpoint": receipt}), flush=True)
    return receipt


def status(root=DEFAULT_ROOT):
    """Read the existing invocation only; never creates a relay or job."""
    root = _content_path(root)
    result = _read(root / "status.json") if (root / "status.json").is_file() else _read(root / "entry.json")
    return {**result, "repeat_run_action": "report_existing_invocation_only"}


def poll(root=DEFAULT_ROOT):
    """Small notebook-visible state suitable for occasional progress polling."""
    saved = status(root)
    return {key: saved[key] for key in ("nonce", "phase", "status", "updated_at_utc", "cohorts", "worker_root", "result_root") if key in saved}


def aggregate_results(root=DEFAULT_ROOT):
    """Return only bounded aggregate scientific results, never case messages."""
    root = _content_path(root, exists=True)
    result = {"version": VERSION, "status": poll(root), "mab_accessed": False,
              "interpretation": "independent automatic evaluation; missing and invalid scores are reported, not clinical efficacy",
              "longmemeval_paid_model_calls": 0, "cohorts": {}}
    for cohort in ("professional7", "cactus600", "locomo1540"):
        report = {}
        expected = {"generation": root / "workers" / (cohort + "-continuation") / "complete.json"}
        if cohort == "locomo1540":
            expected["reference_judge"] = root / "studies" / cohort / "formal_judge" / "summary.json"
        else:
            expected["fact_memory"] = root / "studies" / cohort / "memory" / "summary.json"
            expected["five_dimensions"] = root / "studies" / cohort / "quality_v2" / "summary.json"
        for label, path in expected.items():
            if path.is_file():
                value = _read(path)
                if label == "fact_memory":
                    excluded = {"target_preparation", "target_unavailable_case_ids", "inherited_frozen_unavailable_keys"}
                    value = {k: v for k, v in value.items() if k not in excluded}
                    value["paired"] = {comparison: {metric: {k: v for k, v in details.items() if k != "per_question"}
                        for metric, details in metrics.items()} for comparison, metrics in value.get("paired", {}).items()}
                if label == "generation" and "unbound_answer_files" in value:
                    value["unbound_answer_file_count"] = len(value.pop("unbound_answer_files"))
                report[label] = value
        result["cohorts"][cohort] = report
    if (root / "source_preparation.json").is_file():
        result["longmemeval_source_only"] = _read(root / "source_preparation.json").get("longmemeval500")
    encoded = json.dumps(result, ensure_ascii=False, allow_nan=False).encode()
    if len(encoded) > 512_000:
        raise ValueError("aggregate result exceeds notebook-safe output allowance; use private checkpoint files")
    _write(root / "aggregate_results.json", result)
    return result


def run(namespace, *, repo, expected_commit, sources, root=DEFAULT_ROOT,
        allow_longmemeval_paid=False, longmemeval_authorization=None, background=True,
        _recovered_connection=None):
    """Start once; default to the original parent-owned hosted delegates.

    ``sources`` has professional7/cactus600 {reader,memory,old_quality},
    locomo1540 {reader,old_quality}, longmemeval500 {reader} or None. Every
    path refers to an existing original hosted source, never a new test roster.
    The optional LME authorization is recorded for a separately reviewed entry;
    this entry still performs zero LongMemEval model calls.
    ``_recovered_connection`` is supplied by the distinct validated restoration
    entry; it never manufactures original v5 variables or changes old journals.
    """
    _hosted()
    root = _content_path(root)
    if (root / "entry.json").exists():
        return status(root)
    if type(allow_longmemeval_paid) is not bool or type(background) is not bool:
        raise ValueError("explicit Boolean execution flags required")
    if allow_longmemeval_paid:
        expected = {"authorized_by": "user", "cohort": "longmemeval500", "planned_cases": 500, "maximum_calls": LME_PAID_CAPS}
        if (type(longmemeval_authorization) is not dict or
                any(longmemeval_authorization.get(k) != v for k, v in expected.items()) or
                not isinstance(longmemeval_authorization.get("authorization_reference"), str) or
                not longmemeval_authorization["authorization_reference"].strip()):
            raise ValueError("explicit separate LongMemEval paid authorization and fixed upper bounds required")
    elif longmemeval_authorization is not None:
        raise ValueError("LongMemEval paid authorization cannot be implicit")
    repo = _checkout(repo, expected_commit)
    sources = _source_paths(sources)
    delegates = _parents(namespace, _recovered_connection)
    _prior_recovery(sources)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    connection_origin = ("original_parent_delegate" if _recovered_connection is None
                         else "reconstructed_from_existing_colab_secret")
    entry = {"version": VERSION if _recovered_connection is None else RESTORED_CONNECTION_VERSION,
             "nonce": str(uuid.uuid4()), "status": "reserved_zero_calls",
             "root": str(root), "selected_commit": expected_commit, "sources": sources,
             "allow_longmemeval_paid": allow_longmemeval_paid,
             "longmemeval_authorization": longmemeval_authorization,
             "longmemeval_execution": "source_only_zero_model_calls", "mab_accessed": False,
             "connection_origin": connection_origin, "created_at_utc": _utc()}
    if _recovered_connection is not None:
        entry["source_restoration"] = dict(_recovered_connection.provenance)
    try:
        with (root / "entry.json").open("x", encoding="utf-8") as stream:
            json.dump(entry, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    except FileExistsError:
        return status(root)
    relay_api = runpy.run_path(str(repo / "evaluation/architecture-rerun/hosted_relay.py"))
    semaphore, mutex = threading.BoundedSemaphore(4), threading.Lock()
    def before_send():
        _prior_recovery(sources)
        if _recovered_connection is not None:
            _recovered_connection.assert_current()
    transports = {model: _BoundedDelegate(delegate, semaphore, root / "provider_diagnostics.jsonl", mutex,
                                        before_send=before_send)
                  for model, delegate in delegates.items()}
    relay = relay_api["start_relay"](root / "relay", transports)
    state = {"root": root, "repo": repo, "commit": expected_commit, "sources": sources,
             "nonce": entry["nonce"], "relay": relay, "relay_api": relay_api, "status": dict(entry),
             "allow_longmemeval_paid": allow_longmemeval_paid, "process": None, "namespace": namespace,
             "connection_origin": connection_origin, "recovered_connection": _recovered_connection}
    namespace[STATE_NAME] = state  # retain parent delegate/relay/thread for the notebook kernel lifetime
    _update(state, phase="source_preparation", status="started_zero_paid_calls")
    if background:
        thread = threading.Thread(target=_controller, args=(state,), name="capstone-non-mab-" + entry["nonce"], daemon=False)
        state["thread"] = thread
        thread.start()
    else:
        _controller(state)
    return status(root)


def run_recovered(namespace, *, repo, expected_commit, sources, source_restoration_receipt,
                  source_restore_archive, root=DEFAULT_RECOVERED_ROOT, background=True):
    """Continue from verified restored originals in a fresh genuine Colab.

    This separate entry never creates v5 globals or claims a reused original
    delegate. It is permanently source-only for LongMemEval. The existing
    Capstone Secret is loaded in the parent only after every zero-call source
    preparation and finite-budget check succeeds, at the first paid send.
    """
    _hosted()
    root = _content_path(root)
    if root != Path(DEFAULT_RECOVERED_ROOT):
        raise ValueError("restored continuation must use its one fixed invocation root")
    if (root / "entry.json").exists():
        return status(root)
    if (Path(DEFAULT_ROOT) / "entry.json").exists():
        raise RuntimeError("original continuation exists; inspect it before any restored run")
    repo = _checkout(repo, expected_commit)
    sources = _source_paths(sources)
    archive = _content_path(source_restore_archive)
    receipt = _content_path(source_restoration_receipt)
    provenance = _restoration_provenance(archive, receipt, sources)
    connection = _RecoveredConnection(repo, provenance)
    return run(namespace, repo=repo, expected_commit=expected_commit, sources=sources,
               root=root, background=background, allow_longmemeval_paid=False,
               _recovered_connection=connection)
