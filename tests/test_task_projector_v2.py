from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules/long-term-memory/lifecycle"))

from task_projector import project_task, supporting_spans_for_event, validate_bound_choice, validate_task_event, verify_claim_binding  # noqa: E402
from task_input_rules import resolve_event_date_reference  # noqa: E402


T0 = "2026-11-02T09:00:00Z"
T1 = "2026-11-03T09:00:00Z"
T2 = "2026-11-04T09:00:00Z"
T3 = "2026-11-05T09:00:00Z"
T4 = "2026-11-06T09:00:00Z"
T5 = "2026-11-07T09:00:00Z"
LABEL = "想法记录"


def binding(text, kind, value, *, field=None, excerpt=None, unit=None):
    excerpt = text if excerpt is None else excerpt
    start = text.rfind(excerpt)
    if start < 0:
        raise AssertionError(excerpt)
    claim = {"kind": kind, "value": value}
    if field is not None:
        claim["field"] = field
    if unit is not None:
        claim["unit"] = unit
    return {"start": start, "end": start + len(excerpt), "claim": claim}


class TaskProjectorTests(unittest.TestCase):
    def setUp(self):
        self.sources = {}
        self.events = []

    def event(self, ident, kind, text, payload=None, *, time=T1, occurred=None,
              period="week-1", task="task-1", origin="explicit_user_statement",
              bindings=None, append=True):
        role = "assistant" if origin == "assistant_proposal" else "user"
        self.sources[ident] = {"user_id": "u", "text": text, "role": role, "source_time": time}
        obj = {
            "event_id": ident, "source_id": ident, "user_id": "u", "goal_id": "g",
            "task_instance_id": task, "period_id": period, "event_type": kind,
            "occurred_at": occurred or time, "reported_at": time, "origin": origin,
            "payload": {"object_label": LABEL, **(payload or {})},
            "claim_bindings": bindings if bindings is not None else [binding(text, "object", LABEL)],
        }
        if append:
            self.events.append(obj)
        return obj

    def agree(self, ident="agree", count=2, time=T0, due=None):
        text = f"我同意每周做{count}次{LABEL}。"
        payload = {"target_count": count}
        if due:
            payload["due_at"] = due
        bindings = [binding(text, "object", LABEL), binding(text, "choice", "agreed"),
                    binding(text, "quantity", count, field="target_count", unit="次")]
        if due:
            text += f"截止时间{due}。"
            bindings.append(binding(text, "time", due, excerpt=due))
        return self.event(ident, "agreed", text, payload, time=time, bindings=bindings)

    def attempt(self, ident="wed", attempt="attempt-wed", time=T2, occurred=T2, count=1):
        text = f"我做了{count}次{LABEL}。"
        return self.event(ident, "progress_report", text,
                          {"report_mode": "attempt", "count": count, "attempt_id": attempt},
                          time=time, occurred=occurred,
                          bindings=[binding(text, "object", LABEL), binding(text, "quantity", count, field="count"),
                                    binding(text, "choice", "attempt")])

    def total(self, ident="total", count=2, time=T3, occurred=None):
        occurred = occurred or time
        text = f"{LABEL}累计做了{count}次。"
        return self.event(ident, "progress_report", text,
                          {"report_mode": "cumulative", "count": count, "covers_through": occurred},
                          time=time, occurred=occurred,
                          bindings=[binding(text, "object", LABEL), binding(text, "quantity", count, field="count"),
                                    binding(text, "choice", "cumulative")])

    def project(self, **kwargs):
        return project_task(self.events, user_id="u", task_instance_id="task-1",
                            period_id="week-1", sources=self.sources, **kwargs)

    def test_negation_outside_selected_span_still_blocks_agreement(self):
        source = "我没答应每天做想法记录。"
        start = source.index("答应")
        self.assertFalse(verify_claim_binding(source, start, len(source), {"kind": "choice", "value": "agreed"}))
        source = "我不是不愿意做想法记录。"
        self.assertFalse(verify_claim_binding(source, 0, len(source), {"kind": "choice", "value": "agreed"}))

    def test_negative_quantity_is_not_positive_or_zero(self):
        source = "我没做2次想法记录。"
        for value in [0, 2]:
            self.assertFalse(verify_claim_binding(source, 3, len(source), {"kind": "quantity", "value": value}))

    def test_bilingual_explicit_rules_and_exact_unicode_bounds(self):
        text = "I agreed to try 想法记录 2 times."
        self.assertTrue(verify_claim_binding(text, 0, len(text), {"kind": "choice", "value": "agreed"}))
        self.assertTrue(verify_claim_binding(text, 0, len(text), {"kind": "quantity", "value": 2, "unit": "times"}))
        chinese = "我同意做两次想法记录。"
        self.assertTrue(verify_claim_binding(chinese, 0, len(chinese), {"kind": "quantity", "value": 2}))
        for start, end in [(True, 4), (-1, 2), (1, 100), (3, 3)]:
            self.assertFalse(verify_claim_binding(chinese, start, end, {"kind": "object", "value": LABEL}))

    def test_acceptance_progress_pause_and_unknown_outcome_coexist(self):
        self.agree()
        self.attempt()
        text = "想法记录这周先暂停。"
        self.event("pause", "participation_changed", text, {"participation": "paused"}, time=T3,
                   bindings=[binding(text, "object", LABEL), binding(text, "choice", "paused")])
        result = self.project()
        self.assertEqual(result["acceptance"], "agreed")
        self.assertEqual(result["participation"], "paused")
        self.assertEqual(result["outcome"], "unknown")
        self.assertEqual(result["progress"]["reported_count"], 1)
        self.assertEqual(result["progress"]["target_count"], 2)
        self.assertEqual(result["progress"]["completion"], "partial")

    def test_same_attempt_re_report_adds_source_not_count(self):
        self.attempt()
        self.attempt("fri-report", time=T4, occurred=T2)
        result = self.project()
        self.assertEqual(result["progress"]["reported_count"], 1)
        self.assertEqual(result["progress"]["attempt_ids"], ["attempt-wed"])
        self.assertEqual(len(result["source_ids"]), 2)

    def test_event_replay_is_idempotent_and_conflicting_id_rejected(self):
        event = self.attempt()
        self.events.append(deepcopy(event))
        self.assertEqual(self.project()["state_version"], 1)
        changed = deepcopy(event)
        changed["reported_at"] = T4
        self.events.append(changed)
        with self.assertRaisesRegex(ValueError, "event_id"):
            self.project()

    def test_cumulative_reports_are_not_added(self):
        self.total("first-total", 2, time=T3)
        self.total("repeat-total", 2, time=T4)
        self.assertEqual(self.project()["progress"]["reported_count"], 2)

    def test_late_covered_attempt_not_added_but_later_attempt_is(self):
        self.total(count=2, time=T3, occurred=T3)
        self.attempt("late-wed", time=T5, occurred=T2)
        self.attempt("fri-new", attempt="attempt-fri", time=T4, occurred=T4)
        result = self.project()
        self.assertEqual(result["progress"]["reported_count"], 3)
        self.assertEqual(result["progress"]["known_attempt_count"], 2)

    def test_unexplained_decreasing_total_is_unresolved(self):
        self.total("three", 3, time=T2)
        self.total("two", 2, time=T3)
        self.assertIsNone(self.project()["progress"]["reported_count"])
        self.assertTrue(self.project()["progress"]["count_conflict"])

    def test_correction_appends_and_does_not_rewrite_attempt_sources(self):
        original = self.total("three", 3, time=T2)
        before = deepcopy(original)
        text = "想法记录说错了，不是3次，而是2次。"
        self.event("correction", "corrected", text,
                   {"target_event_id": "three", "changes": {"count": 2}}, time=T4,
                   bindings=[binding(text, "object", LABEL), binding(text, "quantity", 2, field="count", excerpt="2次")])
        result = self.project()
        self.assertEqual(result["progress"]["reported_count"], 2)
        self.assertEqual(original, before)
        self.assertEqual(self.sources["three"]["text"], "想法记录累计做了3次。")
        self.assertEqual(result["state_version"], 2)

    def test_late_correction_does_not_restore_old_acceptance_or_participation(self):
        self.agree(count=3)
        text = "我拒绝想法记录。"
        self.event("decline", "declined", text, time=T2,
                   bindings=[binding(text, "object", LABEL), binding(text, "choice", "declined")])
        text = "想法记录这周先暂停。"
        self.event("pause", "participation_changed", text, {"participation": "paused"}, time=T3,
                   bindings=[binding(text, "object", LABEL), binding(text, "choice", "paused")])
        text = "想法记录更正为2次。"
        self.event("correct-count", "corrected", text,
                   {"target_event_id": "agree", "changes": {"target_count": 2}}, time=T4,
                   bindings=[binding(text, "object", LABEL), binding(text, "quantity", 2, field="target_count")])
        result = self.project()
        self.assertEqual(result["acceptance"], "declined")
        self.assertEqual(result["participation"], "paused")
        self.assertEqual(result["progress"]["target_count"], 2)
        self.assertEqual(result["field_support_source_ids"]["target_count"], "correct-count")

    def test_historical_period_and_new_instance_do_not_pollute_current(self):
        self.agree()
        event = self.attempt("late-previous", time=T4, occurred=T1)
        event["period_id"] = "previous-week"
        event = self.attempt("new-week", time=T4, occurred=T4)
        event["task_instance_id"] = "task-next-week"
        self.assertIsNone(self.project()["progress"]["reported_count"])

    def test_due_review_does_not_clear_progress_or_generate_outcome(self):
        self.agree(due=T3)
        self.attempt()
        result = self.project(as_of=T4)
        self.assertTrue(result["due_review"])
        self.assertEqual(result["progress"]["reported_count"], 1)
        self.assertEqual(result["outcome"], "unknown")
        self.assertEqual(result["acceptance"], "agreed")

    def test_blocked_object_never_falls_back_to_three_after_two_deleted(self):
        self.total("three", 3, time=T2)
        text = "想法记录更正为2次。"
        self.event("two", "corrected", text, {"target_event_id": "three", "changes": {"count": 2}}, time=T3,
                   bindings=[binding(text, "object", LABEL), binding(text, "quantity", 2, field="count")])
        result = self.project(blocked=True)
        self.assertTrue(result["blocked"])
        self.assertIsNone(result["progress"]["reported_count"])
        self.assertEqual(result["eligibility"], "blocked")
        self.assertEqual(result["source_ids"], [])

    def test_missing_correction_source_fails_instead_of_reviving_ancestor(self):
        self.total("three", 3, time=T2)
        text = "想法记录更正为2次。"
        self.event("two", "corrected", text, {"target_event_id": "three", "changes": {"count": 2}}, time=T3,
                   bindings=[binding(text, "object", LABEL), binding(text, "quantity", 2, field="count")])
        del self.sources["two"]
        with self.assertRaisesRegex(ValueError, "source"):
            self.project()

    def test_late_event_time_does_not_bypass_source_cutoff(self):
        self.attempt(time=T5, occurred=T1)
        self.assertIsNone(self.project(source_cutoff=T3)["progress"]["reported_count"])

    def test_strict_types_unknown_fields_and_inference_rejected(self):
        event = self.agree()
        for key, value in [("event_id", 123), ("oracle_supported", True), ("origin", "agent_inference")]:
            bad = deepcopy(event);bad[key] = value
            with self.assertRaises(ValueError):
                validate_task_event(bad, sources=self.sources)
        bad = deepcopy(event);bad["payload"]["target_count"] = "2"
        with self.assertRaises(ValueError):
            validate_task_event(bad, sources=self.sources)
        bad = deepcopy(event);bad["claim_bindings"] = [event["claim_bindings"][-1]]
        with self.assertRaisesRegex(ValueError, "object"):
            validate_task_event(bad, sources=self.sources)

    def bound_choice(self):
        text = "愿意试试想法记录每周2次吗？"
        self.sources["assistant-proposal"] = {"user_id": "u", "role": "assistant", "text": text, "source_time": T0}
        proposal = {"proposal_id": "p", "user_id": "u", "goal_id": "g", "task_instance_id": "task-1",
                    "period_id": "week-1", "source_id": "assistant-proposal", "created_at": T0,
                    "expires_at": T3, "session_id": "session-1", "action_count": 1,
                    "target_count": 2, "object_label": LABEL,
                    "claim_bindings": [binding(text, "object", LABEL), binding(text, "quantity", 2)]}
        event = self.event("yes", "agreed", "yes", {"proposal_id": "p", "target_count": 2},
                           bindings=[{"start": 0, "end": 3, "claim": {"kind": "choice", "value": "agreed", "proposal_id": "p"}}])
        event["session_id"] = "session-1"
        return event, {"p": proposal}

    def test_unique_live_single_action_yes_is_task_choice_not_confirmation(self):
        event, proposals = self.bound_choice()
        self.assertTrue(validate_bound_choice(event, sources=self.sources, proposals=proposals))
        result = self.project(proposals=proposals)
        self.assertEqual(result["acceptance"], "agreed")
        self.assertEqual(result["progress"]["target_count"], 2)
        self.assertNotIn("confirmation_state", result)
        self.assertFalse(verify_claim_binding("yes", 0, 3, {"kind": "choice", "value": "agreed"}))

    def test_unbound_expired_compound_and_multi_proposal_yes_fail(self):
        event, proposals = self.bound_choice()
        self.assertFalse(validate_bound_choice(event, sources=self.sources, proposals={}))
        for field, value in [("action_count", 2), ("session_id", "different"), ("expires_at", T0)]:
            altered = deepcopy(proposals);altered["p"][field] = value
            self.assertFalse(validate_bound_choice(event, sources=self.sources, proposals=altered))
        multi = deepcopy(proposals);multi["another"] = {**deepcopy(multi["p"]), "proposal_id": "other"}
        self.assertFalse(validate_bound_choice(event, sources=self.sources, proposals=multi))
        self.sources["assistant-proposal"]["text"] += "以及散步2次？"
        self.assertFalse(validate_bound_choice(event, sources=self.sources, proposals=proposals))

    def test_source_owner_role_and_datetime_order_are_enforced(self):
        event = self.agree()
        for field, value in [("role", "assistant"), ("user_id", "other"), ("source_time", T5)]:
            sources = deepcopy(self.sources);sources["agree"][field] = value
            with self.assertRaises(ValueError):
                validate_task_event(event, sources=sources)
        bad = deepcopy(event);bad["occurred_at"] = T5
        with self.assertRaisesRegex(ValueError, "occurrence"):
            validate_task_event(bad, sources=self.sources)

    def context_object(self):
        self.sources["bound-object"] = {"user_id": "u", "role": "user", "text": LABEL, "source_time": T0}
        return {"task-1": {"object_label": LABEL, "goal_id": "g", "period_id": "week-1",
                           "source_id": "bound-object", "start": 0, "end": len(LABEL)}}

    def test_bound_current_object_supports_short_pause_without_inventing_source(self):
        objects = self.context_object()
        text = "这周暂停"
        event = self.event("short-pause", "participation_changed", text, {"participation": "paused"},
                           bindings=[binding(text, "choice", "paused")])
        with self.assertRaisesRegex(ValueError, "object"):
            validate_task_event(event, sources=self.sources)
        result = self.project(context_objects=objects)
        self.assertEqual(result["participation"], "paused")
        self.assertEqual(self.sources["short-pause"]["text"], "这周暂停")
        self.assertEqual(result["source_ids"], ["bound-object", "short-pause"])
        self.assertEqual(result["object_support_source_ids"], ["bound-object"])

    def test_context_object_cannot_swallow_different_activity_or_wrong_period(self):
        objects = self.context_object()
        text = "散步这周暂停"
        event = self.event("other-activity", "participation_changed", text, {"participation": "paused"},
                           bindings=[binding(text, "choice", "paused")])
        with self.assertRaisesRegex(ValueError, "object"):
            validate_task_event(event, sources=self.sources, context_objects=objects)
        event["source_id"] = "short-pause"
        self.sources["short-pause"] = {"user_id": "u", "role": "user", "text": "这周暂停", "source_time": T1}
        event["claim_bindings"] = [binding("这周暂停", "choice", "paused")]
        wrong = deepcopy(objects);wrong["task-1"]["period_id"] = "other-week"
        with self.assertRaisesRegex(ValueError, "object"):
            validate_task_event(event, sources=self.sources, context_objects=wrong)
        del self.sources["bound-object"]
        with self.assertRaisesRegex(ValueError, "object"):
            validate_task_event(event, sources=self.sources, context_objects=objects)

    def test_different_clause_quantity_cannot_be_attached_to_task(self):
        text = "散步做了2次；想法记录累计做了1次。"
        event = self.event("cross-clause", "progress_report", text,
                           {"report_mode": "cumulative", "count": 2},
                           bindings=[binding(text, "object", LABEL), binding(text, "quantity", 2, field="count", excerpt="2次"),
                                     binding(text, "choice", "cumulative", excerpt="想法记录累计做了1次")])
        with self.assertRaisesRegex(ValueError, "quantity or report mode"):
            validate_task_event(event, sources=self.sources)

    def test_plan_quantity_cannot_be_reported_as_completed_count(self):
        text = "想法记录累计计划做2次，我做了1次。"
        event = self.event("plan-count", "progress_report", text,
                           {"report_mode": "cumulative", "count": 2},
                           bindings=[binding(text, "object", LABEL), binding(text, "quantity", 2, field="count", excerpt="2次"),
                                     binding(text, "choice", "cumulative", excerpt="想法记录累计计划做2次")])
        with self.assertRaisesRegex(ValueError, "quantity or report mode"):
            validate_task_event(event, sources=self.sources)

    def test_explicit_no_progress_short_form_keeps_known_zero(self):
        objects = self.context_object()
        text = "这周还没做"
        self.event("none-yet", "progress_report", text,
                   {"report_mode": "attempt", "count": 0, "attempt_id": "attempt-current"},
                   bindings=[binding(text, "quantity", 0, field="count"), binding(text, "choice", "not_done")])
        self.assertEqual(self.project(context_objects=objects)["progress"]["reported_count"], 0)


    def test_correction_of_old_total_cannot_roll_back_later_explicit_total(self):
        self.total("old-total", 3, time=T1)
        self.total("new-total", 4, time=T3)
        text = "想法记录更正为2次。"
        self.event("old-correction", "corrected", text,
                   {"target_event_id": "old-total", "changes": {"count": 2}}, time=T4,
                   bindings=[binding(text, "object", LABEL), binding(text, "quantity", 2, field="count")])
        with self.assertRaisesRegex(ValueError, "superseded current field"):
            self.project()

    def test_guessed_deadline_is_not_product_eligibility(self):
        event = self.agree()
        event["payload"]["due_at"] = T3
        with self.assertRaisesRegex(ValueError, "deadline"):
            validate_task_event(event, sources=self.sources)


    def test_hypothetical_and_reported_third_person_are_not_current_agreement(self):
        for text in ["比如我同意做2次想法记录。", "假如我同意做2次想法记录。",
                     "妈妈同意做2次想法记录。", "For example, I agreed to 2 times 想法记录.",
                     "She agreed to 2 times 想法记录."]:
            self.assertFalse(verify_claim_binding(text, 0, len(text), {"kind": "choice", "value": "agreed"}))


    def test_bound_current_object_supports_explicit_quantity_correction(self):
        self.agree(count=3)
        objects = self.context_object()
        text = "这周改成2次"
        self.event("short-correction", "corrected", text,
                   {"target_event_id": "agree", "changes": {"target_count": 2}}, time=T3,
                   bindings=[binding(text, "quantity", 2, field="target_count")])
        self.assertEqual(self.project(context_objects=objects)["progress"]["target_count"], 2)


    def test_bound_pause_with_terminal_no_save_control_uses_original_offsets(self):
        objects = self.context_object()
        text = "这周暂停，这条别保存"
        event = self.event("ephemeral-pause", "participation_changed", text, {"participation": "paused"},
                           bindings=[binding(text, "choice", "paused", excerpt="这周暂停")])
        result = self.project(context_objects=objects)
        self.assertEqual(result["participation"], "paused")
        self.assertEqual(self.sources["ephemeral-pause"]["text"], text)
        self.assertEqual(event["claim_bindings"][0]["end"], 4)
        with self.assertRaisesRegex(ValueError, "object"):
            validate_task_event(event, sources=self.sources)
        changed = "散步这周暂停，这条别保存"
        self.sources["ephemeral-pause"]["text"] = changed
        event["claim_bindings"] = [binding(changed, "choice", "paused", excerpt="散步这周暂停")]
        with self.assertRaisesRegex(ValueError, "object"):
            validate_task_event(event, sources=self.sources, context_objects=objects)

    def attach_day(self, event):
        source = self.sources[event["source_id"]]
        reference = resolve_event_date_reference(source["text"], source_time=source["source_time"],
            task_instance_id=event["task_instance_id"], period_id=event["period_id"])
        for key in ["occurred_at", "occurred_before", "occurred_precision", "event_date", "event_timezone"]:
            event[key] = reference[key]
        return event

    def dated_attempt(self, source_time=T2):
        text = "周三做了1次想法记录。"
        event = self.event("dated-wed", "progress_report", text,
            {"report_mode": "attempt", "count": 1, "attempt_id": "wed-attempt"}, time=source_time,
            bindings=[binding(text, "object", LABEL), binding(text, "quantity", 1, field="count"),
                      binding(text, "choice", "attempt")])
        return self.attach_day(event)

    def test_daily_attempt_overlapping_cumulative_cutoff_is_unresolved_not_added(self):
        self.agree()
        self.dated_attempt()
        self.total(count=2, time="2026-11-04T12:00:00Z")
        result = self.project()
        self.assertIsNone(result["progress"]["reported_count"])
        self.assertEqual(result["progress"]["reported_total"], 2)
        self.assertTrue(result["progress"]["count_conflict"])
        self.assertEqual(result["progress"]["coverage_conflict_attempt_ids"], ["wed-attempt"])
        self.assertEqual(result["acceptance"], "agreed")

    def test_day_interval_before_or_after_cutoff_is_classified_without_guessing_time(self):
        self.dated_attempt(source_time=T4)
        self.total(count=2, time=T3)
        self.assertEqual(self.project()["progress"]["reported_count"], 2)
        self.events = [e for e in self.events if e["event_id"] != "total"]
        self.total("earlier-total", count=2, time=T1)
        self.assertEqual(self.project()["progress"]["reported_count"], 3)

    def test_day_precision_total_does_not_claim_an_exact_cumulative_snapshot(self):
        self.attempt(occurred="2026-11-04T10:00:00Z", time=T4)
        text = "周三想法记录累计做了2次。"
        event = self.event("dated-total", "progress_report", text,
            {"report_mode": "cumulative", "count": 2, "covers_through": "2026-11-04T00:00:00Z"}, time=T4,
            bindings=[binding(text, "object", LABEL), binding(text, "quantity", 2, field="count"),
                      binding(text, "choice", "cumulative")])
        self.attach_day(event)
        self.assertIsNone(self.project()["progress"]["reported_count"])
        self.assertTrue(self.project()["progress"]["count_conflict"])

    def test_day_metadata_must_match_actual_source_date_rule(self):
        event = self.dated_attempt()
        event["event_date"] = "2026-11-03"
        with self.assertRaisesRegex(ValueError, "day interval"):
            validate_task_event(event, sources=self.sources)

    def test_typed_assistant_proposal_cannot_overwrite_user_agreement(self):
        self.agree(count=2)
        text = "试试每周做4次想法记录？"
        self.event("new-proposal", "proposed", text, {"target_count": 4}, time=T3,
                   origin="assistant_proposal",
                   bindings=[binding(text, "object", LABEL), binding(text, "quantity", 4, field="target_count")])
        result = self.project()
        self.assertEqual(result["acceptance"], "agreed")
        self.assertEqual(result["progress"]["target_count"], 2)
        self.assertEqual(result["pending_proposals"][0]["target_count"], 4)

    def test_proposal_alone_is_not_user_agreed_target(self):
        text = "试试每周做4次想法记录？"
        self.event("proposal-only", "proposed", text, {"target_count": 4}, origin="assistant_proposal",
                   bindings=[binding(text, "object", LABEL), binding(text, "quantity", 4, field="target_count")])
        result = self.project()
        self.assertEqual(result["acceptance"], "proposed")
        self.assertIsNone(result["progress"]["target_count"])
        self.assertEqual(result["eligibility"], "proposal_only")

    def test_yes_at_expiry_boundary_or_wrong_question_count_is_rejected(self):
        event, proposals = self.bound_choice()
        self.sources[event["source_id"]]["source_time"] = T3
        event["reported_at"] = T3
        self.assertFalse(validate_bound_choice(event, sources=self.sources, proposals=proposals))
        self.sources[event["source_id"]]["source_time"] = T1
        event["reported_at"] = T1
        for value in [0, 2, True, "1"]:
            changed = deepcopy(proposals);changed["p"]["question_count"] = value
            self.assertFalse(validate_bound_choice(event, sources=self.sources, proposals=changed))


    def test_consumed_proposal_allows_exact_accepted_event_replay_only(self):
        event, proposals = self.bound_choice()
        proposals["p"].update(status="consumed", consumed_at=T1, accepted_event_id=event["event_id"])
        self.assertTrue(validate_bound_choice(event, sources=self.sources, proposals=proposals))
        other = deepcopy(event);other["event_id"] = "another-yes-same-clock"
        self.assertFalse(validate_bound_choice(other, sources=self.sources, proposals=proposals))
        self.sources[event["source_id"]]["source_time"] = T2
        event["reported_at"] = T2
        self.assertFalse(validate_bound_choice(event, sources=self.sources, proposals=proposals))

    def test_consumed_view_replays_a_reply_from_before_consumption(self):
        event, proposals = self.bound_choice()
        proposals["p"].update(status="consumed", consumed_at=T2, accepted_event_id="later-accept")
        self.assertTrue(validate_bound_choice(event, sources=self.sources, proposals=proposals))

    def test_other_consumed_proposal_does_not_create_multiple_live_choices(self):
        event, proposals = self.bound_choice()
        earlier = {**deepcopy(proposals["p"]), "proposal_id": "old", "status": "consumed",
                   "consumed_at": T0, "accepted_event_id": "old-accept"}
        proposals["old"] = earlier
        self.assertTrue(validate_bound_choice(event, sources=self.sources, proposals=proposals))
        proposals["old"]["consumed_at"] = T2
        self.assertFalse(validate_bound_choice(event, sources=self.sources, proposals=proposals))

    def test_supporting_spans_select_correct_object_clause_not_first_state_word(self):
        text = "散步继续，想法记录暂停。"
        event = self.event("correct-clause", "participation_changed", text, {"participation": "paused"},
                           bindings=[binding(text, "object", LABEL), binding(text, "choice", "paused")])
        spans = supporting_spans_for_event(event, self.sources)
        self.assertEqual([text[x["start"]:x["end"]] for x in spans], ["想法记录暂停"])
        self.assertNotIn("散步继续", "".join(text[x["start"]:x["end"]] for x in spans))

    def test_supporting_spans_keep_separate_required_clauses_in_one_source(self):
        text = "想法记录我同意做2次；截止2026-11-05T09:00:00Z。"
        event = self.event("two-clauses", "agreed", text, {"target_count": 2, "due_at": T3},
                           bindings=[binding(text, "object", LABEL), binding(text, "choice", "agreed"),
                                     binding(text, "quantity", 2, field="target_count"),
                                     binding(text, "time", T3, excerpt=T3)])
        spans = supporting_spans_for_event(event, self.sources)
        self.assertEqual(len(spans), 2)
        self.assertEqual({s["source_id"] for s in spans}, {"two-clauses"})
        self.assertIn("截止2026-11-05T09:00:00Z", [text[s["start"]:s["end"]] for s in spans])

    def test_supporting_spans_preserve_negation_and_context_object_source(self):
        text = "我不同意做想法记录。"
        event = self.event("negated", "declined", text,
                           bindings=[binding(text, "object", LABEL), binding(text, "choice", "declined")])
        spans = supporting_spans_for_event(event, self.sources)
        self.assertEqual(text[spans[0]["start"]:spans[0]["end"]], "我不同意做想法记录")
        objects = self.context_object()
        text = "这周暂停"
        event = self.event("implicit-support", "participation_changed", text, {"participation": "paused"},
                           bindings=[binding(text, "choice", "paused")])
        spans = supporting_spans_for_event(event, self.sources, context_objects=objects)
        self.assertEqual({s["source_id"] for s in spans}, {"implicit-support", "bound-object"})

    def test_yes_spans_include_full_assistant_proposal_and_original_yes(self):
        event, proposals = self.bound_choice()
        spans = supporting_spans_for_event(event, self.sources, proposals=proposals)
        self.assertEqual({s["source_id"] for s in spans}, {"yes", "assistant-proposal"})
        for span in spans:
            self.assertEqual((span["start"], span["end"]), (0, len(self.sources[span["source_id"]]["text"])))

    def test_day_and_correction_support_fallback_preserves_all_temporal_and_negative_text(self):
        dated = self.dated_attempt(source_time=T4)
        spans = supporting_spans_for_event(dated, self.sources)
        self.assertEqual(spans, [{"source_id": "dated-wed", "start": 0, "end": len(self.sources["dated-wed"]["text"])}])
        text = "想法记录说错了，不是3次，而是2次。"
        event = self.event("full-correction", "corrected", text,
                           {"target_event_id": "older", "changes": {"count": 2}},
                           bindings=[binding(text, "object", LABEL), binding(text, "quantity", 2, field="count", excerpt="2次")])
        spans = supporting_spans_for_event(event, self.sources)
        self.assertEqual(text[spans[0]["start"]:spans[0]["end"]], text)

    def test_unsupported_event_cannot_receive_a_rule_verified_evidence_span(self):
        text = "我没答应每天做想法记录。"
        event = self.event("unsupported", "agreed", text,
                           bindings=[binding(text, "object", LABEL), binding(text, "choice", "agreed")])
        with self.assertRaises(ValueError):
            supporting_spans_for_event(event, self.sources)

    def test_reported_count_support_excludes_obsolete_long_cumulative_sources(self):
        for index, stamp in enumerate([T1, T2, T3]):
            text = "背景。" * 300 + "想法记录累计做了1次。"
            self.event("old-total-" + str(index), "progress_report", text,
                {"report_mode": "cumulative", "count": 1}, time=stamp,
                bindings=[binding(text, "object", LABEL), binding(text, "quantity", 1, field="count"),
                          binding(text, "choice", "cumulative")])
        self.total("short-current-total", 2, time=T4)
        result = self.project()
        self.assertEqual(result["progress"]["reported_count"], 2)
        self.assertEqual(result["field_support_event_ids"]["reported_count"], ["short-current-total"])

    def test_reported_count_support_includes_only_total_plus_later_independent_attempt(self):
        self.attempt("covered", time=T2, occurred=T2)
        self.total("base-total", 2, time=T3)
        self.attempt("later", attempt="fri", time=T4, occurred=T4)
        result = self.project()
        self.assertEqual(result["progress"]["reported_count"], 3)
        self.assertEqual(set(result["field_support_event_ids"]["reported_count"]), {"base-total", "later"})

    def test_current_corrected_total_support_uses_correction_not_obsolete_value(self):
        self.total("old-three", 3, time=T2)
        text = "想法记录累计更正为2次。"
        self.event("new-two", "corrected", text, {"target_event_id": "old-three", "changes": {"count": 2}}, time=T3,
                   bindings=[binding(text, "object", LABEL), binding(text, "quantity", 2, field="count")])
        result = self.project()
        self.assertEqual(result["progress"]["reported_count"], 2)
        self.assertEqual(result["field_support_event_ids"]["reported_count"], ["new-two"])

    def test_unanchored_historical_and_quoted_reports_do_not_become_current(self):
        for text in ["去年我同意每周做2次想法记录。", "2025年我同意每周做2次想法记录。",
                     "2025-11-03，我同意每周做2次想法记录。", "Last year I agreed to 2 times 想法记录.",
                     "老师说：“我同意做2次想法记录。”", "某位路人表示：我同意做2次想法记录。", 'Someone said: "I agreed to 2 times 想法记录."']:
            event = self.event("not-current", "agreed", text, {"target_count": 2}, append=False,
                bindings=[binding(text, "object", LABEL), binding(text, "choice", "agreed"),
                          binding(text, "quantity", 2, field="target_count")])
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "current self-report|historical qualifier"):
                validate_task_event(event, sources=self.sources)


if __name__ == "__main__":
    unittest.main()
