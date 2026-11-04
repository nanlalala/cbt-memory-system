"""Unsaved editor text survives actual notebook refresh/event paths."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_dialogue_ui as support


class NotebookDraftPersistenceTests(unittest.TestCase):
    def launch(self, session):
        modules = {"ipywidgets": support.fake_widgets(),
                   "IPython.display": SimpleNamespace(display=lambda _: None)}
        with patch.object(support.ui, "_require_colab"), patch.object(
                support.ui.importlib, "import_module", side_effect=lambda name: modules[name]):
            root = support.ui.launch_notebook_demo(session)
        next(w for w in support.walk(root) if w.kind == "Checkbox").change(True)
        return root

    def card(self, root, memory_id):
        return next(w for w in support.walk(root) if w.kind == "VBox" and any(
            c.kind == "HTML" and "记忆 ID" in str(c.value) and memory_id in str(c.value)
            for c in w.children))

    def editor(self, root, memory_id):
        return next(w for w in self.card(root, memory_id).children if w.kind == "Textarea")

    def click(self, root, description, memory_id=None):
        area = self.card(root, memory_id) if memory_id else root
        next(w for w in support.walk(area) if w.description == description).click()

    def rendered(self, root):
        return "\n".join(str(w.value) for w in support.walk(root) if w.kind == "HTML")

    def test_edit_a_operate_b_then_return_a_preserves_and_can_save_draft(self):
        session = support.SessionStub()
        root = self.launch(session)
        text = "A draft that must survive another card's confirmation."
        # Deliberately assign .value directly: a refresh must also capture a
        # frontend value even if no observer callback has run yet.
        self.editor(root, "fixture-goal").value = text
        self.click(root, "确认这条记忆", "second-candidate")
        self.assertEqual(self.editor(root, "fixture-goal").value, text)
        self.assertEqual(session.data["records"][0]["content"], "Walk after lunch.")
        self.click(root, "保存修订，保持待确认", "fixture-goal")
        self.assertEqual(session.calls[-1], ("revise", "fixture-goal", text))
        self.assertEqual(session.data["records"][0]["content"], text)
        self.assertEqual(session.data["records"][0]["lifecycle_status"], "candidate")

    def test_edit_a_send_message_then_confirm_a_still_blocks_unsaved_text(self):
        session = support.SessionStub()
        root = self.launch(session)
        text = "My unsaved longer-term goal is different."
        self.editor(root, "fixture-goal").value = text
        next(w for w in support.walk(root) if w.kind == "Textarea").value = "A fictional follow-up."
        self.click(root, "发送消息")
        self.assertEqual(self.editor(root, "fixture-goal").value, text)
        self.click(root, "确认这条记忆", "fixture-goal")
        self.assertEqual(session.calls, [("send", "A fictional follow-up.")])
        self.assertIn("文字尚未保存", self.rendered(root))
        self.assertEqual(session.data["records"][0]["lifecycle_status"], "candidate")

    def test_changed_source_version_preserves_draft_and_refuses_stale_write(self):
        session = support.SessionStub()
        root = self.launch(session)
        text = "Unsaved draft based on the earlier source."
        self.editor(root, "fixture-goal").value = text
        record = session.data["records"][0]
        record["content"] = "A saved replacement from another operation."
        record["source_evidence"]["supporting_text"] = record["content"]
        record["updated_at"] = "2026-11-05T12:00:00Z"
        self.click(root, "整理当前窗口的记忆候选")
        self.assertEqual(self.editor(root, "fixture-goal").value, text)
        self.assertIn("旧草稿保留", self.rendered(root))
        before = list(session.calls)
        self.click(root, "保存修订，保持待确认", "fixture-goal")
        self.click(root, "确认这条记忆", "fixture-goal")
        self.assertEqual(session.calls, before)
        self.assertEqual(record["content"], "A saved replacement from another operation.")
        self.click(root, "放弃这份未保存草稿", "fixture-goal")
        self.assertEqual(self.editor(root, "fixture-goal").value, record["content"])
        self.assertEqual(session.calls, before)  # Discard is UI-only.

    def test_dirty_draft_survives_external_status_change_but_cannot_be_submitted(self):
        session = support.SessionStub()
        root = self.launch(session)
        text = "Still unsaved when the record becomes active elsewhere."
        self.editor(root, "fixture-goal").value = text
        session.data["records"][0]["lifecycle_status"] = "active"
        self.click(root, "整理当前窗口的记忆候选")
        card = self.card(root, "fixture-goal")
        self.assertIn(text, [w.value for w in card.children if w.kind == "Textarea"])
        self.assertIn("未保存草稿仍保留", self.rendered(root))
        self.assertFalse(any(w.description == "保存修订，保持待确认" for w in support.walk(card)))

    def test_confirm_unchanged_card_cannot_silently_confirm_draft_on_another_card(self):
        session = support.SessionStub()
        root = self.launch(session)
        self.editor(root, "fixture-goal").value = "Not yet saved."
        self.click(root, "确认这条记忆", "second-candidate")
        self.click(root, "确认这条记忆", "fixture-goal")
        self.assertEqual([c for c in session.calls if c[0] == "confirm"],
                         [("confirm", "second-candidate", "CONFIRM second-candidate")])

    def test_active_correction_draft_survives_message_and_detects_changed_source(self):
        session = support.SessionStub()
        record = session.data["records"][0]
        record["lifecycle_status"] = "active"
        root = self.launch(session)
        draft = "A correction still being edited."
        self.editor(root, "fixture-goal").value = draft
        next(w for w in support.walk(root) if w.kind == "Textarea").value = "A new fictional message."
        self.click(root, "发送消息")
        self.assertEqual(self.editor(root, "fixture-goal").value, draft)
        record["source_evidence"]["turn_ids"] = ["different-source-turn"]
        before = list(session.calls)
        self.click(root, "提交更正候选", "fixture-goal")
        self.assertEqual(session.calls, before)
        self.assertIn("草稿已保留", self.rendered(root))
        self.click(root, "整理当前窗口的记忆候选")
        self.assertEqual(self.editor(root, "fixture-goal").value, draft)
        self.click(root, "放弃这份未保存草稿", "fixture-goal")
        self.assertEqual(self.editor(root, "fixture-goal").value, "")

    def test_real_offline_session_preserves_a_draft_while_sending_and_after_save(self):
        session = support.DialogueSession()
        self.addCleanup(session.close)
        session.send_message("我想试试午饭后散步。")
        record = session.extract_current_session()[0]
        memory_id = record["memory_id"]
        root = self.launch(session)
        draft = "我只是想试试散步，还没有决定次数。"
        self.editor(root, memory_id).value = draft
        next(w for w in support.walk(root) if w.kind == "Textarea").value = "今天有点累。"
        self.click(root, "发送消息")
        self.assertEqual(self.editor(root, memory_id).value, draft)
        self.click(root, "确认这条记忆", memory_id)
        self.assertEqual(session.snapshot()["records"][0]["lifecycle_status"], "candidate")
        self.click(root, "保存修订，保持待确认", memory_id)
        self.assertEqual(session.snapshot()["records"][0]["content"], draft)
        self.assertEqual(self.editor(root, memory_id).value, draft)
        self.click(root, "确认这条记忆", memory_id)
        self.assertEqual(session.snapshot()["records"][0]["lifecycle_status"], "active")


if __name__ == "__main__":
    unittest.main()
