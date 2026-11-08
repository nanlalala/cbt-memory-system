"""Fictional product-path checks; no provider, model, dataset or network."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules/agent-integration"))

from controlled_dialogue import ControlledDialogueSession  # noqa: E402


class ControlledDateIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.session = ControlledDialogueSession.open(
            str(Path(self.folder.name) / "memory.sqlite"), approved_root=self.folder.name,
            user_id="fictional-date-user", provider=None, background=False, timezone_name="UTC")
        self.addCleanup(self.session.close)
        self.serial = 0

    def send(self, text, stamp):
        self.serial += 1
        return self.session.send_message(text, input_id=f"date-input-{self.serial}", as_of=stamp)

    def agree(self, count="两", stamp="2026-11-02T09:00:00Z"):
        return self.send(f"我同意每周做{count}次想法记录。", stamp)

    def test_wednesday_attempt_and_friday_same_attempt_report_count_once(self):
        self.agree()
        first = self.send("今天做了一次想法记录。", "2026-11-04T10:00:00Z")["task_state"]
        again = self.send("周三那次做过一次想法记录。", "2026-11-06T10:00:00Z")["task_state"]
        self.assertEqual(first["progress"]["reported_count"], 1)
        self.assertEqual(again["progress"]["reported_count"], 1)
        self.assertEqual(first["progress"]["attempt_ids"], again["progress"]["attempt_ids"])
        self.assertEqual(len(again["progress"]["attempt_ids"]), 1)
        task = self.session._objects("task")[0]["payload"]
        reports = [e for e in task["events"] if e["event_type"] == "progress_report"]
        self.assertEqual(len(reports), 2)
        self.assertTrue(all(e["occurred_precision"] == "day" for e in reports))
        self.assertTrue(all(e["event_date"] == "2026-11-04" for e in reports))

    def test_late_last_week_report_changes_historical_instance_only(self):
        old = self.agree()["task_state"]
        self.send("今天做了一次想法记录。", "2026-11-04T10:00:00Z")
        new = self.agree("三", "2026-11-09T09:00:00Z")["task_state"]
        late = self.send("上周三那次做过一次想法记录。", "2026-11-13T09:00:00Z")["task_state"]
        self.assertNotEqual(old["task_instance_id"], new["task_instance_id"])
        self.assertEqual(late["task_instance_id"], old["task_instance_id"])
        self.assertEqual(late["period_id"], "2026-11-02")
        self.assertEqual(late["progress"]["reported_count"], 1)
        current = next(s for s in self.session.task_states() if s["task_instance_id"] == new["task_instance_id"])
        self.assertIsNone(current["progress"]["reported_count"])
        self.assertEqual(current["progress"]["target_count"], 3)

    def test_explicit_old_date_does_not_attach_to_reporting_weeks_instance(self):
        old = self.agree()["task_state"]
        self.send("今天做了一次想法记录。", "2026-11-04T10:00:00Z")
        new = self.agree("三", "2026-11-09T09:00:00Z")["task_state"]
        late = self.send("2026-11-04那次做过一次想法记录。", "2026-11-13T09:00:00Z")["task_state"]
        self.assertEqual(late["task_instance_id"], old["task_instance_id"])
        self.assertEqual(late["progress"]["reported_count"], 1)
        current = next(s for s in self.session.task_states() if s["task_instance_id"] == new["task_instance_id"])
        self.assertIsNone(current["progress"]["reported_count"])

    def test_explicit_target_correction_preserves_progress_and_task_identity(self):
        before = self.agree("三")["task_state"]
        self.send("想法记录累计做了一次。", "2026-11-03T09:00:00Z")
        result = self.send("想法记录每周改成两次。", "2026-11-04T09:00:00Z")["task_state"]
        self.assertEqual(result["task_instance_id"], before["task_instance_id"])
        self.assertEqual(result["progress"]["target_count"], 2)
        self.assertEqual(result["progress"]["reported_count"], 1)
        task = self.session._objects("task")[0]["payload"]
        self.assertEqual([e["event_type"] for e in task["events"]], ["agreed", "progress_report", "corrected"])
        self.assertEqual(task["events"][0]["payload"]["target_count"], 3)

    def test_explicit_total_correction_is_not_also_a_new_progress_report(self):
        self.agree("三")
        self.send("想法记录累计做了一次。", "2026-11-03T09:00:00Z")
        result = self.send("想法记录累计更正为两次。", "2026-11-04T09:00:00Z")["task_state"]
        self.assertEqual(result["progress"]["reported_count"], 2)
        self.assertEqual(result["progress"]["target_count"], 3)
        task = self.session._objects("task")[0]["payload"]
        self.assertEqual(sum(e["event_type"] == "progress_report" for e in task["events"]), 1)
        self.assertEqual(task["events"][-1]["event_type"], "corrected")

    def test_plain_change_to_two_with_both_axes_known_does_not_guess_field(self):
        self.agree("三")
        self.send("想法记录累计做了一次。", "2026-11-03T09:00:00Z")
        before = len(self.session._objects("task")[0]["payload"]["events"])
        result = self.send("这周改成两次。", "2026-11-04T09:00:00Z")["task_state"]
        self.assertEqual(result["progress"]["reported_count"], 1)
        self.assertEqual(result["progress"]["target_count"], 3)
        self.assertEqual(len(self.session._objects("task")[0]["payload"]["events"]), before)

    def test_second_ordinal_is_one_new_attempt_not_count_two(self):
        self.agree("三")
        self.send("今天做了第一次想法记录。", "2026-11-04T10:00:00Z")
        second = self.send("今天做了第二次想法记录。", "2026-11-04T16:00:00Z")["task_state"]
        self.assertEqual(second["progress"]["reported_count"], 2)
        self.assertEqual(len(second["progress"]["attempt_ids"]), 2)
        task = self.session._objects("task")[0]["payload"]
        reports = [e for e in task["events"] if e["event_type"] == "progress_report"]
        self.assertEqual([e["payload"]["count"] for e in reports], [1, 1])
        self.assertTrue(any(b["claim"]["kind"] == "attempt_ordinal" for b in reports[-1]["claim_bindings"]))

    def test_undated_physical_attempt_does_not_borrow_report_date(self):
        self.agree()
        result = self.send("做了一次想法记录。", "2026-11-04T09:00:00Z")["task_state"]
        self.assertIsNone(result["progress"]["reported_count"])
        self.assertEqual(result["progress"]["attempt_ids"], [])


if __name__ == "__main__":
    unittest.main()
