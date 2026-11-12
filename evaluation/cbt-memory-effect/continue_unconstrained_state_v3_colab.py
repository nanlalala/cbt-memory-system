"""One-shot external continuation of an already running, frozen hosted study.

This launcher never retries or reruns the before phase. It waits only for the
observed owned PID, authenticates complete archives in fresh subprocesses, and
then invokes the unchanged after and score CLIs once each. It does not import a
model, read a secret, alter a study protocol, increase a cap or kill a process.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime, timezone


VERSION = "one-shot-owned-unconstrained-v3-continuation-v1"
CONTENT = Path("/content")
HELPER = CONTENT / "cbt-prefill-runtime-code-20261112-v2"
MAIN = HELPER / "evaluation/cbt-memory-effect/unconstrained_state_comparison_v3_colab.py"
ROOT = CONTENT / "cbt-unconstrained-state-study-20261112-v3"
EXPECTED_HEAD = "8d1b0f7e6688dc480ad4ef282a80b8c1fd4916fb"
PROTOCOL_SHA = "118bccbf288105cafbe4c88e6b582f0742b1b28056c8d06a5cf7860296841d16"
BEFORE_PID = 25306
BEFORE_START_TICKS = "631820"
BEFORE_ARGV = ("python3", "-u", str(MAIN), "run-side", "--side", "before")
PREFIX = CONTENT / "cbt-unconstrained-v3-orchestrator-20261112-v1"
MARKER = Path(str(PREFIX) + ".marker.json")
JOURNAL = Path(str(PREFIX) + ".journal.jsonl")
LOGS = {phase: CONTENT / ("cbt-unconstrained-v3-" + phase + "-20261112.log")
        for phase in ("verify-before", "after", "verify-after", "score")}
WAIT_SECONDS = 120 * 60
POLL_SECONDS = 5


class StopContinuation(RuntimeError):
    pass


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def require_hosted():
    try:
        valid = (bool(os.environ.get("COLAB_RELEASE_TAG")) and CONTENT.is_dir()
                 and not CONTENT.is_symlink() and importlib.util.find_spec("google.colab") is not None)
    except (ImportError, ValueError, AttributeError):
        valid = False
    if not valid:
        raise StopContinuation("genuine_hosted_colab_required_before_any_io")


def check_identity(expected_source_sha256):
    require_hosted()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_source_sha256):
        raise StopContinuation("reviewed_launcher_source_sha256_required")
    if file_sha(__file__) != expected_source_sha256:
        raise StopContinuation("launcher_source_changed")
    actual_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HELPER, text=True).strip()
    if actual_head != EXPECTED_HEAD:
        raise StopContinuation("frozen_helper_git_head_changed")
    protocol = ROOT / "study_protocol.json"
    if protocol.is_symlink() or file_sha(protocol) != PROTOCOL_SHA:
        raise StopContinuation("frozen_study_protocol_changed")


def parse_proc_stat(text):
    close = text.rfind(")")
    if close < 0:
        raise StopContinuation("unreadable_owned_process_stat")
    rest = text[close + 1:].split()
    if len(rest) < 20 or not rest[19].isdigit():
        raise StopContinuation("unreadable_owned_process_start_ticks")
    return {"state": rest[0], "start_ticks": rest[19]}


def owned_before_status(proc_root=Path("/proc")):
    """Return running/exited for the exact observed PID; reject reuse/changes."""
    directory = Path(proc_root) / str(BEFORE_PID)
    try:
        first = parse_proc_stat((directory / "stat").read_text())
    except FileNotFoundError:
        return {"state": "exited", "exit_code": None, "evidence": "owned_pid_absent"}
    if first["start_ticks"] != BEFORE_START_TICKS:
        raise StopContinuation("owned_before_pid_reused_no_action")
    if first["state"] in ("Z", "X", "x"):
        return {"state": "exited", "exit_code": None, "evidence": "owned_process_terminal_state"}
    try:
        raw = (directory / "cmdline").read_bytes()
        argv = tuple(part.decode("utf-8") for part in raw.rstrip(b"\x00").split(b"\x00"))
        executable = (directory / "exe").resolve(strict=True)
        second = parse_proc_stat((directory / "stat").read_text())
    except FileNotFoundError:
        # A missing executable link alone is not evidence of process exit.
        # Recheck the process identity before allowing archive authentication.
        try:
            terminal = parse_proc_stat((directory / "stat").read_text())
        except FileNotFoundError:
            return {"state": "exited", "exit_code": None, "evidence": "owned_pid_exited_during_observation"}
        if terminal["start_ticks"] != BEFORE_START_TICKS:
            raise StopContinuation("owned_before_pid_reused_during_observation")
        if terminal["state"] in ("Z", "X", "x"):
            return {"state": "exited", "exit_code": None, "evidence": "owned_process_terminal_state"}
        raise StopContinuation("live_owned_process_identity_became_unreadable_no_action")
    if second["start_ticks"] != BEFORE_START_TICKS:
        raise StopContinuation("owned_before_pid_reused_during_observation")
    if second["state"] in ("Z", "X", "x"):
        return {"state": "exited", "exit_code": None, "evidence": "owned_process_terminal_state"}
    if argv != BEFORE_ARGV or executable != Path(sys.executable).resolve(strict=True):
        raise StopContinuation("owned_before_command_or_executable_changed_no_action")
    return {"state": "running", "pid": BEFORE_PID, "start_ticks": BEFORE_START_TICKS}


class Journal:
    def __init__(self, path):
        self.stream = Path(path).open("x", encoding="utf-8")
        os.fchmod(self.stream.fileno(), 0o600)
        self.sequence = 0
        self.previous = None

    def record(self, event, **fields):
        self.sequence += 1
        body = {"sequence": self.sequence, "event": event,
                "at_utc": datetime.now(timezone.utc).isoformat(),
                "previous_record_sha256": self.previous, **fields}
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self.previous = digest_bytes(encoded)
        self.stream.write(json.dumps({**body, "record_sha256": self.previous}, sort_keys=True) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self):
        self.stream.close()


def wait_for_before(journal, expected_source_sha256):
    deadline = time.monotonic() + WAIT_SECONDS
    last_heartbeat = 0.0
    while True:
        status = owned_before_status()
        if status["state"] == "exited":
            journal.record("owned_before_exited_authentication_still_required", **status)
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise StopContinuation("owned_before_wait_120_minutes_expired_no_kill_no_retry")
        if time.monotonic() - last_heartbeat >= 60:
            check_identity(expected_source_sha256)
            journal.record("waiting_for_owned_before", **status)
            last_heartbeat = time.monotonic()
        time.sleep(min(POLL_SECONDS, remaining))


def _phase_argv(phase, expected_source_sha256):
    if phase in ("verify-before", "verify-after"):
        return [sys.executable, "-u", str(Path(__file__).resolve()), phase,
                "--expected-source-sha256", expected_source_sha256]
    if phase == "after":
        return [sys.executable, "-u", str(MAIN), "run-side", "--side", "after"]
    if phase == "score":
        return [sys.executable, "-u", str(MAIN), "score"]
    raise ValueError("unknown_continuation_phase")


def run_phase(phase, journal, expected_source_sha256):
    check_identity(expected_source_sha256)
    if phase == "after" and (ROOT / "answers_after").exists():
        raise StopContinuation("existing_after_output_no_automatic_resume")
    if phase == "score" and (ROOT / "reports").exists():
        raise StopContinuation("existing_score_output_no_automatic_resume")
    path = LOGS[phase]
    if path.exists() or path.is_symlink():
        raise StopContinuation("existing_phase_log_no_automatic_resume")
    argv = _phase_argv(phase, expected_source_sha256)
    journal.record("child_start_intent", phase=phase, argv=argv, stdout_log=str(path),
                   unknown_child_start_must_never_be_automatically_retried=True)
    try:
        with path.open("x") as stream:
            os.fchmod(stream.fileno(), 0o600)
            child = subprocess.Popen(argv, cwd=CONTENT, stdout=stream, stderr=subprocess.STDOUT)
            journal.record("child_started", phase=phase, pid=child.pid)
            deadline = time.monotonic() + WAIT_SECONDS
            last_heartbeat = time.monotonic()
            while child.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    journal.record("child_wait_expired_may_still_be_running", phase=phase, pid=child.pid,
                                   kill_attempted=False, automatic_retry=False)
                    raise StopContinuation("child_wait_120_minutes_expired_no_kill_no_retry")
                if time.monotonic() - last_heartbeat >= 60:
                    journal.record("waiting_for_owned_child", phase=phase, pid=child.pid)
                    last_heartbeat = time.monotonic()
                time.sleep(min(POLL_SECONDS, remaining))
            journal.record("child_exited", phase=phase, pid=child.pid, returncode=child.returncode)
            if child.returncode != 0:
                raise StopContinuation("child_phase_failed_no_automatic_retry:" + phase)
    except StopContinuation:
        raise
    except BaseException as error:
        journal.record("child_start_or_wait_failed_or_unknown", phase=phase,
                       failure_type=type(error).__name__, automatic_retry=False)
        raise
    check_identity(expected_source_sha256)


def verify_side(side, expected_source_sha256):
    """Fresh process, original frozen verifier; no main target is decoded."""
    check_identity(expected_source_sha256)
    sys.path.insert(0, str(MAIN.parent))
    spec = importlib.util.spec_from_file_location("continuation_frozen_v3_verifier", MAIN)
    if spec is None or spec.loader is None:
        raise StopContinuation("frozen_main_verifier_unavailable")
    stage = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = stage
    spec.loader.exec_module(stage)
    plan = stage.frozen(ROOT)
    intents, outcomes = stage.ledger(ROOT)
    expected = 48 if side == "before" else 57
    if len(intents) != expected or len(outcomes) != expected:
        raise StopContinuation("complete_expected_settled_generation_count_required")
    counts = {}
    for selected in (("before",) if side == "before" else ("before", "after")):
        answers, _ = stage.authenticate_main(ROOT, selected, plan["sections"][selected])
        if len(answers) != 48:
            raise StopContinuation("complete_48_row_authenticated_side_required")
        counts[selected] = len(answers)
    check_identity(expected_source_sha256)
    print(json.dumps({"version": VERSION, "stage": "complete_source_bound_archive_authenticated",
        "side": side, "authenticated_logical_rows": counts, "settled_unique_generations": expected,
        "main_gold_read": False, "new_model_calls_by_verifier": 0}, sort_keys=True), flush=True)


def run(expected_source_sha256):
    check_identity(expected_source_sha256)
    fresh = (MARKER, JOURNAL, *LOGS.values(), ROOT / "answers_after", ROOT / "reports")
    if any(path.exists() or path.is_symlink() for path in fresh):
        raise StopContinuation("existing_continuation_or_after_artifact_no_automatic_resume")
    # An exact running PID or an already exited PID is allowed. Completion is
    # never inferred from process exit; fresh authentication is mandatory.
    initial_process = owned_before_status()
    marker = {"version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "launcher_source_sha256": expected_source_sha256, "launcher_pid": os.getpid(),
        "helper_head": EXPECTED_HEAD, "study_protocol_sha256": PROTOCOL_SHA,
        "owned_before_pid": BEFORE_PID, "owned_before_start_ticks": BEFORE_START_TICKS,
        "owned_before_argv": list(BEFORE_ARGV), "initial_owned_process": initial_process,
        "new_authorization_or_budget": False, "automatic_retry": False}
    with MARKER.open("x") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(json.dumps(marker, sort_keys=True, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    journal = Journal(JOURNAL)
    try:
        journal.record("one_shot_continuation_started", **marker)
        wait_for_before(journal, expected_source_sha256)
        for phase in ("verify-before", "after", "verify-after", "score"):
            run_phase(phase, journal, expected_source_sha256)
        summary_path = ROOT / "reports/summary.json"
        summary = json.loads(summary_path.read_text())
        if (summary.get("version") != "unconstrained-state-development-study-v3"
                or summary.get("main_logical_rows") != 96
                or summary.get("new_study_unique_generations") != 57
                or summary.get("study_cap") != 57):
            raise StopContinuation("final_score_summary_identity_mismatch")
        journal.record("completed", authenticated_main_logical_rows=96, settled_unique_generations=57,
            final_summary_sha256=file_sha(summary_path), new_budget=0, automatic_retry=False)
        print(json.dumps({"version": VERSION, "stage": "completed", "main_logical_rows": 96,
            "settled_unique_generations": 57, "journal": str(JOURNAL)}, sort_keys=True), flush=True)
    except BaseException as error:
        journal.record("stopped_no_kill_no_automatic_retry", failure_type=type(error).__name__,
                       reason=str(error) if isinstance(error, StopContinuation) else "inspect preserved child logs and journal",
                       processes_killed=0, new_budget=0)
        raise
    finally:
        journal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify-before", "verify-after"))
    parser.add_argument("--expected-source-sha256", required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args.expected_source_sha256)
    else:
        verify_side(args.command.split("-", 1)[1], args.expected_source_sha256)


if __name__ == "__main__":
    main()
