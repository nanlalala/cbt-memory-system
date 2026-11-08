"""Bounded fictional replay through the actual file-backed v2 prototype.

This command uses no model, network or external dataset. Its optional provider
is a local test double. Successful assertions demonstrate software contracts,
not better dialogue, reduced participant burden, CBT benefit or clinical effect.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
for relative in (
    "modules/long-term-memory/schema", "modules/long-term-memory/storage",
    "modules/long-term-memory/lifecycle", "modules/long-term-memory/extraction",
    "modules/long-term-memory/retrieval", "modules/agent-integration",
):
    module_path = str(ROOT / relative)
    if module_path not in sys.path:
        sys.path.insert(0, module_path)

from controlled_dialogue import ControlledDialogueSession, PrototypeLimits  # noqa: E402
from dialogue_service import ModelOutput  # noqa: E402
from durable_memory import DurableLimits  # noqa: E402


DEMO_VERSION = "controlled-memory-scripted-v2.1"
STAMPS = (
    "2026-11-03T09:00:00Z", "2026-11-04T09:00:00Z",
    "2026-11-05T09:00:00Z", "2026-11-06T09:00:00Z",
    "2026-11-06T10:00:00Z",
)
SCENARIO_NAMES = ("multi_axis_and_worker", "correction_and_deletion", "no_save_pause", "answer_unknown",
                  "history_source_integrity")
MAX_INPUTS_BY_SCENARIO = dict(zip(SCENARIO_NAMES, (4, 4, 3, 2, 2)))


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class DemoContractFailure(RuntimeError):
    pass


class LocalMockProvider:
    """A local function test double; it has no network transport or model."""
    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = []

    def __call__(self, *, messages, purpose):
        self.calls.append({"messages": deepcopy(messages), "purpose": purpose})
        if self.fail:
            raise TimeoutError("fictional local transport-unknown fixture")
        if purpose == "extraction":
            material = json.loads(messages[-1]["content"])
            turn = material["turns"][0]
            return ModelOutput(json.dumps({"candidates": [{
                "memory_type": "explicit_fact", "content": turn["text"],
                "source_turn_ids": [turn["turn_id"]], "supporting_text": turn["text"],
            }]}, ensure_ascii=False), provider_model="local_test_double_not_a_model")
        # Echoing also tests that a no-save answer cannot smuggle the private
        # utterance back into a durable response cache or the next context.
        current = next(message["content"] for message in reversed(messages) if message["role"] == "user")
        return ModelOutput("本地虚构回声：" + current, provider_model="local_test_double_not_a_model")


def _history_claims(messages):
    """Read the final provider-bound packet, rather than a pre-packaging view."""
    packets = [message["content"].split("\n", 1)[1] for message in messages
               if message["role"] == "system" and message["content"].startswith("SOURCED MEMORY DATA ONLY.")]
    if len(packets) > 1:
        raise DemoContractFailure("more than one sourced-memory packet")
    return json.loads(packets[0])["claims"] if packets else []


def _exact_history_quotes(claims, sources):
    """Verify source IDs, full Unicode offsets, body, role and time together."""
    for claim in claims:
        if claim.get("field") != "history_quote" or claim.get("support_status") != "source_quote":
            return False
        evidence = claim.get("evidence")
        if not isinstance(evidence, list) or len(evidence) != 1:
            return False
        span = evidence[0]
        source = sources.get(span.get("source_id"))
        if source is None or any((
            span.get("text") != source["text"], claim.get("value") != source["text"],
            span.get("start") != 0, span.get("end") != len(source["text"]),
            span.get("role") != source["role"], span.get("source_time") != source["source_time"],
        )):
            return False
    return True


def run_scripted(database, *, approved_root, run_id):
    """Run exactly five independent fictional-user software scenarios once."""
    path, approved = Path(database), Path(approved_root).resolve(strict=True)
    if not path.is_absolute() or path.is_symlink() or not path.resolve().is_relative_to(approved):
        raise ValueError("explicit database path must be under the approved root")
    if path.exists() or Path(str(path) + ".control.json").exists():
        raise ValueError("a fresh demonstration database is required; existing runs are never overwritten")
    if type(run_id) is not str or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", run_id):
        raise ValueError("run_id must be an explicit short identifier")
    limits = PrototypeLimits(max_outbound_chars=18000, max_response_tokens=512,
                             max_worker_jobs_per_turn=4, max_session_turns=12)
    durable_limits = DurableLimits(max_run_calls=8, max_session_calls=8,
                                   max_queue_length=12, max_batch_size=4)
    results, mocks = [], []
    input_counts = {name: 0 for name in SCENARIO_NAMES}

    def session(name, provider=None):
        return ControlledDialogueSession.open(path, approved_root=approved,
            user_id=f"fictional-demo:{run_id}:{name}", run_id=f"{run_id}:{name}",
            session_id=f"fictional-session:{name}", provider=provider,
            limits=limits, durable_limits=durable_limits, background=False,
            timezone_name="UTC")

    def check(stage, label, condition, observed=None):
        stage["assertions"].append({"assertion": label, "passed": bool(condition), "observed": observed})
        if not condition:
            raise DemoContractFailure(f"{stage['scenario']}: {label}")

    def send(s, name, text, number, **kwargs):
        input_counts[name] += 1
        if input_counts[name] > MAX_INPUTS_BY_SCENARIO[name]:
            raise DemoContractFailure("scripted input limit exceeded")
        return s.send_message(text, input_id=f"fixture-input-{number}", as_of=STAMPS[number], **kwargs)

    name = SCENARIO_NAMES[0]
    s = session(name)
    try:
        stage = {"scenario": name, "protocol_hash": s.protocol["protocol_hash"], "assertions": []}
        first = send(s, name, "我同意每周做两次想法记录。", 0)
        task_id = first["task_state"]["task_instance_id"]
        send(s, name, "想法记录累计做了一次。", 1)
        paused = send(s, name, "这周暂停", 2)["task_state"]
        observed = {"acceptance": paused["acceptance"], "reported_count": paused["progress"]["reported_count"],
                    "target_count": paused["progress"]["target_count"], "participation": paused["participation"],
                    "outcome": paused["outcome"]}
        check(stage, "acceptance_progress_pause_and_unknown_coexist",
              observed == {"acceptance": "agreed", "reported_count": 1, "target_count": 2,
                           "participation": "paused", "outcome": "unknown"}, observed)
        check(stage, "task_identity_survives_progress_and_pause", paused["task_instance_id"] == task_id)
        work = s.run_pending()
        records = s.store.list_for_user(s.user_id)
        check(stage, "actual_bounded_worker_and_manager_saved_source_only_candidates",
              len(records) == 3 and len(work) == 3 and
              all(x.get("semantic_status") == "source_only_no_model" for x in work),
              {"candidate_count": len(records), "worker_outcomes": work})
        check(stage, "canonical_schema_two_candidates_remain_unconfirmed",
              all(r.schema_version == "2.0" and r.lifecycle_status == "candidate" and
                  r.confirmation_state == "unconfirmed" for r in records))
        followup = send(s, name, "想法记录现在怎么安排？", 3)
        check(stage, "task_projection_preserves_current_multi_axis_state",
              followup["task_state"]["participation"] == "paused" and
              followup["task_state"]["progress"]["reported_count"] == 1,
              {"included_claim_count": len(followup.get("memory_ids", [])),
               "history_quote_claim_count": len(followup.get("history_memory_ids", [])),
               "answer_status": followup["answer_status"]})
        results.append(stage)
    finally:
        s.close()

    name = SCENARIO_NAMES[1]
    s = session(name)
    try:
        stage = {"scenario": name, "protocol_hash": s.protocol["protocol_hash"], "assertions": []}
        initial = send(s, name, "我同意做三次想法记录。", 0)["task_state"]
        revised = send(s, name, "这周改成两次", 1)["task_state"]
        check(stage, "correction_keeps_task_identity_and_replaces_current_target",
              initial["task_instance_id"] == revised["task_instance_id"] and
              revised["progress"]["target_count"] == 2)
        s.run_pending()
        object_id = "task:" + revised["task_instance_id"] + ":" + revised["period_id"]
        deletion = s.delete_content("object", object_id)
        blocked = s.durable.get_object_state(s.user_id, object_id)
        check(stage, "content_deletion_blocks_current_and_clears_payload",
              deletion["logical_blocked"] is True and blocked["status"] == "blocked" and blocked["payload"] is None,
              {"status": blocked["status"], "replica_cleanup": deletion["replica_cleanup"]})
        after = send(s, name, "想法记录现在安排几次？", 2)
        check(stage, "deleted_two_does_not_restore_old_three",
              after.get("task_state") is None and s.task_states() == [])
        check(stage, "ordinary_source_view_no_longer_contains_deleted_targets",
              all("三次" not in src["text"] and "两次" not in src["text"]
                  for src in s.durable.list_sources(s.user_id)))
    finally:
        s.close()
    s = session(name)
    try:
        check(stage, "reopen_obeys_the_same_persistent_deletion_barrier",
              s.durable.get_object_state(s.user_id, object_id)["status"] == "blocked" and s.task_states() == [])
        results.append(stage)
    finally:
        s.close()

    name = SCENARIO_NAMES[2]
    provider = LocalMockProvider(); mocks.append(provider)
    s = session(name, provider)
    try:
        stage = {"scenario": name, "protocol_hash": s.protocol["protocol_hash"], "assertions": []}
        initial = send(s, name, "我同意做两次想法记录。", 0)["task_state"]
        private = "这周暂停，这条别保存"
        current = send(s, name, private, 1)
        check(stage, "no_save_pause_applies_only_to_current_turn",
              current["source_commit"]["source_status"] == "not_saved" and
              current["organizing_status"] == "not_queued" and
              current["task_state"]["participation"] == "paused" and
              current["task_state"]["not_retained"] is True)
        dump = "\n".join(s.store.connection.iterdump())
        check(stage, "private_input_not_retained_in_database_snapshot_or_hash",
              private not in dump and "这周暂停" not in json.dumps(s.snapshot(), ensure_ascii=False) and
              hashlib.sha256(private.encode()).hexdigest() not in dump)
        next_reply = send(s, name, "想法记录现在怎么安排？", 2)
        outbound = json.dumps(provider.calls[-1], ensure_ascii=False)
        check(stage, "private_input_not_reused_in_next_outbound_or_extraction_background",
              private not in outbound and "这周暂停" not in outbound and
              all(call["purpose"] == "response" for call in provider.calls))
        object_id = "task:" + initial["task_instance_id"] + ":" + initial["period_id"]
        check(stage, "unsaved_current_change_does_not_reactivate_old_current_value",
              s.durable.get_object_state(s.user_id, object_id)["status"] == "blocked" and
              next_reply.get("task_state") is None)
        stage["mock_provider_calls"] = len(provider.calls)
        results.append(stage)
    finally:
        s.close()

    name = SCENARIO_NAMES[3]
    provider = LocalMockProvider(fail=True); mocks.append(provider)
    s = session(name, provider)
    try:
        stage = {"scenario": name, "protocol_hash": s.protocol["protocol_hash"], "assertions": []}
        quote = "我同意做两次想法记录。"
        failed = send(s, name, quote, 0)
        state_before = deepcopy(s.task_states()[0])
        check(stage, "mock_answer_failure_does_not_rollback_source_or_current_control",
              failed["source_commit"]["source_status"] == "committed" and
              failed["answer_status"] == "failed_or_unknown" and
              state_before["acceptance"] == "agreed")
        retry = send(s, name, quote, 0)
        check(stage, "same_input_retry_does_not_repeat_model_source_or_state_update",
              retry["replayed"] is True and retry["answer_status"] == "attempt_unknown" and
              len(provider.calls) == 1 and len(s.durable.list_sources(s.user_id)) == 1 and
              s.task_states()[0] == state_before)
        attempt = s.durable.get_attempt(s.user_id, "response:fixture-input-0")
        check(stage, "unknown_attempt_and_unknown_usage_are_distinct_from_zero_cost",
              attempt["status"] == "attempt_unknown" and attempt["usage"] is None,
              {"attempt_status": attempt["status"], "usage": attempt["usage"]})
        stage["mock_provider_calls"] = len(provider.calls)
        results.append(stage)
    finally:
        s.close()

    # An answer to these queries can only use admitted historical source text;
    # no task object, user question, semantic note or expected answer contains
    # the fixture bodies. Seven equally relevant sources exercise top-k=6;
    # the eighth is deliberately unrelated. The second query has zero overlap.
    name = SCENARIO_NAMES[4]
    provider = LocalMockProvider(); mocks.append(provider)
    s = session(name, provider)
    try:
        stage = {"scenario": name, "protocol_hash": s.protocol["protocol_hash"], "assertions": []}
        sources = {}
        for index in range(8):
            sid = "fixture-history-" + str(index)
            source = {"user_id": s.user_id, "source_id": sid, "input_id": "fixture-seed-" + str(index),
                      "text": "archivekey marker" + str(index) if index < 7 else "zebra garden brick",
                      "role": "user", "source_time": STAMPS[0], "permissions": dict(s.default_permissions)}
            s.durable.submit_source(source, enqueue=False, run_id=s.run_id, session_id=s.session_id)
            sources[sid] = source
        positive = send(s, name, "archivekey", 3)
        claims = _history_claims(provider.calls[-1]["messages"])
        history_ids = {e["source_id"] for claim in claims for e in claim.get("evidence", [])}
        expected = {"fixture-history-" + str(index) for index in range(6)}
        check(stage, "history_query_retrieves_expected_sources_with_top_k_six",
              positive["answer_status"] == "succeeded" and len(claims) == 6 and history_ids == expected,
              {"selected_source_ids": sorted(history_ids), "history_quote_claim_count": len(claims)})
        check(stage, "final_outbound_quotes_match_full_source_body_offsets_role_and_time",
              _exact_history_quotes(claims, sources) and len(claims) == 6)
        check(stage, "unrelated_source_and_seventh_rank_are_absent_from_outbound",
              "fixture-history-6" not in history_ids and "fixture-history-7" not in history_ids)
        absent = send(s, name, "qzxnomatch", 4)
        absent_claims = _history_claims(provider.calls[-1]["messages"])
        check(stage, "zero_hit_query_has_no_historical_claims_or_body_in_final_messages",
              absent["answer_status"] == "succeeded" and absent_claims == [] and
              absent.get("history_memory_ids") == [] and
              all(source["text"] not in json.dumps(provider.calls[-1]["messages"], ensure_ascii=False)
                  for source in sources.values()))
        check(stage, "history_fixture_uses_only_two_bounded_response_mocks",
              len(provider.calls) == 2 and all(call["purpose"] == "response" for call in provider.calls),
              {"mock_response_calls": len(provider.calls)})
        stage["mock_provider_calls"] = len(provider.calls)
        results.append(stage)
    finally:
        s.close()

    protocol = {
        "version": DEMO_VERSION, "run_id": run_id,
        "kind": "fictional_software_contract_replay_not_effectiveness_experiment",
        "scenario_names": list(SCENARIO_NAMES), "maximum_inputs": MAX_INPUTS_BY_SCENARIO,
        "fixture_source_times": list(STAMPS), "synthetic_clock": True,
        "scenario_protocol_hashes": {stage["scenario"]: stage["protocol_hash"] for stage in results},
        "maximum_mock_provider_attempts_per_scenario": durable_limits.max_run_calls,
        "maximum_mock_provider_attempts_total": durable_limits.max_run_calls * len(SCENARIO_NAMES),
        "worker_batch_limit": durable_limits.max_batch_size, "worker_queue_limit": durable_limits.max_queue_length,
        "background_worker": False, "automatic_retries": False,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "real_api_calls_permitted": 0,
    }
    return {
        "status": "passed", "protocol": protocol, "protocol_hash": _hash(protocol),
        "scenarios": results, "assertion_count": sum(len(stage["assertions"]) for stage in results),
        "input_counts": input_counts, "api_calls": 0,
        "mock_provider_calls": sum(len(provider.calls) for provider in mocks),
        "live_model": False, "live_model_or_network_used": False, "effectiveness_claim": False,
        "execution_host_claim": "software_only_no_colab_guard_bypass",
        "persistence_boundary": "file_and_independent_control_anchor_depend_on_the_approved_storage_location",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", required=True, help="fresh absolute SQLite file under approved root")
    parser.add_argument("--approved-root", required=True, help="existing approved independent-cloud directory")
    parser.add_argument("--run-id", required=True, help="explicit frozen software replay identifier")
    parser.add_argument("--scripted", action="store_true", help="run exactly the supplied fictional scenarios")
    args = parser.parse_args(argv)
    if not args.scripted:
        parser.error("this command only accepts --scripted fictional software replay")
    try:
        report = run_scripted(args.database, approved_root=args.approved_root, run_id=args.run_id)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "reason": str(exc),
                          "api_calls": 0, "live_model": False, "live_model_or_network_used": False,
                          "effectiveness_claim": False}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
