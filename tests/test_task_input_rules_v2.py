from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules/long-term-memory/lifecycle"))

from task_input_rules import parse_current_correction, resolve_attempt_reference, resolve_event_date_reference  # noqa: E402


class TaskInputDateRuleTests(unittest.TestCase):
    def resolve(self, text, *, source="2026-11-06T09:00:00Z", period="2026-11-02", tz="UTC", task="task-1"):
        return resolve_attempt_reference(text, source_time=source, task_instance_id=task,
                                         period_id=period, timezone_name=tz)

    def test_wednesday_reference_and_re_report_share_same_attempt_identity(self):
        first = self.resolve("本周三做了一次想法记录。", source="2026-11-04T20:00:00Z")
        later = self.resolve("周三那次我做过一次想法记录。")
        dated = self.resolve("2026-11-04那次做了一次想法记录。")
        self.assertEqual(first["attempt_id"], later["attempt_id"])
        self.assertEqual(later["attempt_id"], dated["attempt_id"])
        self.assertEqual(later["event_date"], "2026-11-04")
        self.assertEqual(later["occurred_precision"], "day")
        self.assertEqual(later["occurred_at"], "2026-11-04T00:00:00Z")
        self.assertEqual(later["occurred_before"], "2026-11-05T00:00:00Z")
        self.assertTrue(later["requires_existing_date_uniqueness_check"])

    def test_today_then_yesterday_does_not_create_a_new_physical_attempt(self):
        first = self.resolve("今天做了一次想法记录。", source="2026-11-04T09:00:00Z")
        next_day = self.resolve("昨天那次做过一次想法记录。", source="2026-11-05T09:00:00Z")
        self.assertEqual(first["attempt_id"], next_day["attempt_id"])
        self.assertIsNone(self.resolve("今天又做了一次想法记录。"))
        self.assertIsNone(self.resolve("I tried another one today."))

    def test_source_timezone_controls_today_not_host_date(self):
        local = self.resolve("今天做了一次想法记录。", source="2026-11-04T16:30:00Z", tz="Asia/Shanghai")
        utc = self.resolve("今天做了一次想法记录。", source="2026-11-04T16:30:00Z")
        self.assertEqual(local["event_date"], "2026-11-05")
        self.assertEqual(local["occurred_at"], "2026-11-04T16:00:00Z")
        self.assertEqual(utc["event_date"], "2026-11-04")

    def test_cross_year_yesterday_uses_source_year(self):
        reference = self.resolve("昨天做了一次想法记录。", source="2027-01-01T10:00:00Z", period="2026-12-28")
        self.assertEqual(reference["event_date"], "2026-12-31")
        self.assertEqual(reference["period_id"], "2026-12-28")

    def test_late_report_returns_historical_period_not_reporting_period(self):
        reference = self.resolve("上周三那次做过一次想法记录。", source="2026-11-13T10:00:00Z", period="2026-11-09")
        self.assertEqual(reference["event_date"], "2026-11-04")
        self.assertEqual(reference["period_id"], "2026-11-02")
        self.assertIsNone(self.resolve("上周三那次做过一次想法记录。", source="2026-11-13T10:00:00Z", period="opaque-current-period"))

    def test_unqualified_past_weekday_uses_most_recent_visible_day(self):
        reference = self.resolve("周三那次做过一次想法记录。", source="2026-11-09T10:00:00Z", period="2026-11-09")
        self.assertEqual(reference["event_date"], "2026-11-04")
        self.assertIsNone(self.resolve("本周三做一次想法记录。", source="2026-11-09T10:00:00Z", period="2026-11-09"))

    def test_explicit_ordinal_is_identity_not_completed_count(self):
        first = self.resolve("今天做了第一次想法记录。")
        second = self.resolve("今天做了第二次想法记录。")
        self.assertEqual(second["attempt_ordinal"], 2)
        self.assertTrue(second["explicit_ordinal"])
        self.assertNotEqual(first["attempt_id"], second["attempt_id"])
        self.assertNotIn("count", second)
        self.assertEqual(second["attempt_id"], self.resolve("今天又做了第二次想法记录。")["attempt_id"])

    def test_unknown_conflicting_dates_plans_and_aggregate_do_not_get_attempt_ids(self):
        for text in ["做了一次想法记录。", "周三和周五都做了一次想法记录。",
                     "今天计划做一次想法记录。", "今天总共做了两次想法记录。",
                     "今天做了两次想法记录。", "2026-11-10那次做了一次想法记录。",
                     "2026-02-30那次做了一次想法记录。", "今天很焦虑。"]:
            with self.subTest(text=text):
                self.assertIsNone(self.resolve(text))

    def test_task_and_period_are_identity_scopes(self):
        first = self.resolve("今天做了一次想法记录。")
        other = self.resolve("今天做了一次想法记录。", task="different-task")
        self.assertNotEqual(first["attempt_id"], other["attempt_id"])

    def test_daily_interval_accounts_for_dst_not_fixed_24_hours(self):
        reference = self.resolve("2026-11-01那次做了一次想法记录。", source="2026-11-02T15:00:00Z",
                                 period="2026-11-02", tz="America/New_York")
        self.assertEqual(reference["occurred_at"], "2026-11-01T04:00:00Z")
        self.assertEqual(reference["occurred_before"], "2026-11-02T05:00:00Z")

    def test_english_day_reference_is_source_anchored(self):
        reference = self.resolve("I did one record last Wednesday.", source="2026-11-13T10:00:00Z", period="2026-11-09")
        self.assertEqual(reference["event_date"], "2026-11-04")

    def test_general_date_reference_can_anchor_cumulative_without_new_attempt(self):
        text = "周三累计做了2次想法记录。"
        self.assertIsNone(self.resolve(text))
        reference = resolve_event_date_reference(text, source_time="2026-11-06T09:00:00Z",
                    task_instance_id="task-1", period_id="2026-11-02")
        self.assertEqual(reference["event_date"], "2026-11-04")
        self.assertNotIn("attempt_id", reference)

    def test_bad_clock_timezone_and_period_metadata_are_not_guessed(self):
        for args in [{"source": "2026-11-06T09:00:00"}, {"tz": "invalid/timezone"}, {"period": "2026-11-03"}]:
            with self.assertRaises(ValueError):
                self.resolve("今天做了一次想法记录。", **args)

    def task_payload(self):
        return {"task_instance_id": "t", "goal_id": "g", "period_id": "2026-11-02",
                "projection": {"progress": {"target_count": 3, "reported_count": 1}},
                "events": [
                    {"event_id": "agree", "event_type": "agreed", "task_instance_id": "t", "goal_id": "g",
                     "period_id": "2026-11-02", "reported_at": "2026-11-02T09:00:00Z", "payload": {"target_count": 3}},
                    {"event_id": "total", "event_type": "progress_report", "task_instance_id": "t", "goal_id": "g",
                     "period_id": "2026-11-02", "reported_at": "2026-11-04T09:00:00Z", "payload": {"report_mode": "cumulative", "count": 1}},
                ]}

    def test_explicit_plan_correction_selects_target_axis_and_exact_span(self):
        text = "想法记录每周改成2次，这条别保存"
        result = parse_current_correction(text, self.task_payload())
        self.assertEqual(result["target_event_id"], "agree")
        self.assertEqual(result["changes"], {"target_count": 2})
        span = result["claim_bindings"][0]
        self.assertEqual(text[span["start"]:span["end"]], "2次")

    def test_explicit_total_correction_never_changes_plan(self):
        result = parse_current_correction("想法记录累计更正为2次", self.task_payload())
        self.assertEqual(result["target_event_id"], "total")
        self.assertEqual(result["changes"], {"count": 2})

    def test_ambiguous_plain_correction_or_oracle_only_payload_is_not_used(self):
        self.assertIsNone(parse_current_correction("想法记录改成2次", self.task_payload()))
        self.assertIsNone(parse_current_correction("想法记录每周改成2次", {"oracle_target_count": 3}))
        self.assertIsNone(parse_current_correction("比如想法记录每周改成2次", self.task_payload()))
        self.assertIsNone(parse_current_correction("想法记录每周改成2次再改成1次", self.task_payload()))


if __name__ == "__main__":
    unittest.main()
