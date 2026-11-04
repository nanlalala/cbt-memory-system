"""User-data controls and revision-bound low-burden check regressions.

Tiny fictional local fixtures only; these are software behavior checks, not an
extraction-quality, participant burden or clinical outcome study.
"""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules/agent-integration"))
from dialogue_service import DialogueSession, ModelOutput


class FixtureProvider:
    def __init__(self):
        self.calls = []

    def __call__(self, *, messages, purpose):
        self.calls.append((purpose, messages))
        if purpose == "response":
            return ModelOutput("我们接着看这次的安排。", "offline-fixture", {}, False)
        payload = json.loads(messages[-1]["content"])
        source = next(t for t in payload["turns"] if t["turn_id"] in payload["eligible_source_turn_ids"])
        return ModelOutput(json.dumps({"candidates": [{"memory_type": "explicit_fact", "content": source["text"],
            "supporting_text": source["text"], "source_turn_ids": [source["turn_id"]]}]}, ensure_ascii=False))


class DialogueReviewRegressions(unittest.TestCase):
    def session(self, **kwargs):
        session = DialogueSession(memory_mode="conversational", **kwargs)
        self.addCleanup(session.close)
        return session

    def stage(self, session, quote="我觉得写思维记录这项作业有点麻烦。"):
        session.send_message(quote)
        record = session.snapshot()["records"][-1]
        session.new_session()
        return record

    def test_common_explicit_storage_opt_outs_are_never_automatically_or_manually_reimported(self):
        for text in ("我喜欢游泳，但不要记下这条。", "我喜欢游泳，但别记下这条。",
                     "我喜欢游泳，但不要留下这条记录。", "我喜欢游泳，这条不用保存。",
                     "I prefer swimming, but please don’t save this.",
                     "I prefer swimming — don’t save this.",
                     "I prefer swimming; please don‘t record this."):
            with self.subTest(text=text):
                provider = FixtureProvider()
                session = self.session(provider=provider)
                reply = session.send_message(text)
                self.assertEqual(reply["memory_processing"]["reason_code"], "explicit_do_not_remember")
                self.assertEqual(session.snapshot()["records"], [])
                self.assertEqual(session.extract_current_session(), [])
                self.assertEqual([kind for kind, _ in provider.calls], ["response"])
                # The current user message is legitimate response context; its
                # storage instruction is not long-term consent.
                self.assertEqual(provider.calls[0][1][-1]["content"], text)
                session.send_message("我想尝试午饭后散步这个小练习。")
                self.assertEqual([r["content"] for r in session.snapshot()["records"]], ["我想尝试午饭后散步这个小练习。"])

    def test_manual_extraction_rechecks_old_pending_sources_even_without_exclusion_flag(self):
        provider = FixtureProvider()
        session = DialogueSession(provider=provider, memory_mode="review")
        self.addCleanup(session.close)
        private = "我喜欢游泳，但不要记下这条。"
        session.send_message(private)
        # Simulate a turn left pending by a previous/incorrect matcher.
        session._excluded_user_turn_ids.clear()
        session._excluded_user_turn_count = 0
        allowed = "我想尝试一项允许整理的虚构陶艺练习。"
        session.send_message(allowed)
        records = session.extract_current_session()
        self.assertEqual([r["content"] for r in records], [allowed])
        payload = provider.calls[-1][1][-1]["content"]
        self.assertNotIn(private, payload)
        self.assertEqual(session.extract_current_session(), [])

    def test_storage_opt_out_does_not_confuse_reporting_forgetfulness(self):
        for text in ("I don’t remember my homework goal.", "After swimming, I don't remember the plan.",
                     "I don't save money.", "Please don't save money on healthy food.", "我不想留下遗憾，不要留下遗憾。"):
            self.assertFalse(DialogueSession._do_not_remember(text))

    def test_explicit_quantity_update_works_for_running_and_unlisted_activities_without_reasking(self):
        for activity in ("跑步", "陶艺", "练琴"):
            with self.subTest(activity=activity):
                provider = FixtureProvider()
                session = self.session(provider=provider)
                target = self.stage(session, f"我的目标是每天{activity}十分钟。")
                update = "改成五分钟，十分钟不适合我。"
                reply = session.send_message(update)
                self.assertEqual(reply["check_memory_ids"], [])
                self.assertNotIn(target["memory_id"], reply["memory_ids"])
                new = next(r for r in session.snapshot()["records"] if r["content"] == update)
                link = new["structured_payload"]["explicit_update_response"]
                self.assertEqual(link["responds_to_memory_id"], target["memory_id"])
                self.assertNotIn("check_turn_id", link)
                self.assertEqual(new["confirmation_state"], "unconfirmed")
                session.new_session()
                next_reply = session.send_message(f"这周{activity}怎么安排？")
                self.assertEqual(next_reply["check_memory_ids"], [])
                self.assertIn(update, [s["content"] for s in next_reply["history_sources"]])
                self.assertTrue(all(r["confirmation_state"] == "unconfirmed" for r in session.snapshot()["records"]))

    def test_quantity_only_update_does_not_choose_between_two_plausible_goals(self):
        session = self.session()
        session.send_message("我的目标是每天跑步10分钟。")
        session.send_message("我的目标是每天陶艺10分钟。")
        session.new_session()
        update = "改成5分钟，10分钟不适合我。"
        session.send_message(update)
        newest = next(r for r in session.snapshot()["records"] if r["content"] == update)
        self.assertNotIn("explicit_update_response", newest["structured_payload"])

    def test_quantity_update_with_a_different_explicit_topic_never_binds_to_unique_old_goal(self):
        for old, update in (("我的目标是每天跑步十分钟。", "我的午休安排改成五分钟，十分钟不适合我。"),
                            ("My goal is running for 10 minutes daily.",
                             "My pottery time changed to 5 minutes; 10 minutes does not suit me.")):
            with self.subTest(update=update):
                session = self.session()
                self.stage(session, old)
                session.send_message(update)
                new = next(r for r in session.snapshot()["records"] if r["content"] == update)
                self.assertNotIn("explicit_update_response", new["structured_payload"])
                self.assertNotIn("clarification_response", new["structured_payload"])

    def test_transitive_explicit_updates_reach_latest_quote_without_reasking_or_activation(self):
        session = self.session()
        old = self.stage(session, "我的目标是每天跑步十分钟。")
        five = "改成五分钟，十分钟不适合我。"
        three = "改成三分钟，五分钟不适合我。"
        session.send_message(five)
        session.new_session()
        session.send_message(three)
        session.new_session()
        reply = session.send_message("这周跑步怎么安排？")
        contents = [source["content"] for source in reply["history_sources"]]
        self.assertIn(three, contents)
        self.assertEqual(contents[0], three)
        self.assertEqual(reply["check_memory_ids"], [])
        self.assertTrue(all(r["confirmation_state"] == "unconfirmed" for r in session.snapshot()["records"]))
        # An inactive or changed parent cannot be used as an implicit bridge.
        parent = session.store.get(session.user_id, old["memory_id"])
        parent.valid_to = "2020-01-01T00:00:00Z"
        session.store.put(parent)
        session.new_session()
        self.assertEqual(session.send_message("这周跑步怎么安排？")["history_sources"], [])

    def test_check_has_source_revision_session_turn_and_expiry_binding(self):
        session = self.session()
        old = self.stage(session)
        session.send_message("这周思维记录作业怎么安排？")
        binding = session._pending_memory_check
        self.assertEqual(binding["memory_id"], old["memory_id"])
        self.assertEqual(binding["session_id"], session.session_id)
        self.assertEqual(binding["check_turn_id"], session.snapshot()["turns"][-1]["turn_id"])
        self.assertIn("record_binding_sha256", binding)
        self.assertIn("asked_at", binding)
        self.assertIn("expires_at", binding)

    def test_revision_invalidates_old_question_even_if_next_reply_matches_new_topic(self):
        session = self.session()
        old = self.stage(session)
        session.send_message("这周思维记录作业怎么安排？")
        old_check = dict(session._pending_memory_check)
        session.revise_candidate(old["memory_id"], "我觉得游泳作业很麻烦。")
        self.assertIsNone(session._pending_memory_check)
        text = "不是，最近游泳作业很有帮助。"
        session.send_message(text)
        new = next(r for r in session.snapshot()["records"] if r["content"] == text)
        self.assertNotIn("clarification_response", new["structured_payload"])
        self.assertNotIn(old_check["check_turn_id"], json.dumps(new))

    def test_same_id_direct_content_change_cannot_reuse_check_binding(self):
        session = self.session()
        old = self.stage(session)
        session.send_message("这周思维记录作业怎么安排？")
        stored = session.store.get(session.user_id, old["memory_id"])
        stored.content = stored.source_evidence.supporting_text = "游泳作业是我的计划。"
        session.store.put(stored)
        text = "最近游泳作业有帮助。"
        session.send_message(text)
        new = next(r for r in session.snapshot()["records"] if r["content"] == text)
        self.assertNotIn("clarification_response", new["structured_payload"])

    def test_month_late_reply_never_binds_to_old_question(self):
        session = self.session()
        with patch("dialogue_service._now", return_value="2026-01-01T09:00:00Z"):
            self.stage(session)
            session.send_message("这周思维记录作业怎么安排？")
        old_check = dict(session._pending_memory_check)
        with patch("dialogue_service._now", return_value="2026-02-01T09:00:00Z"):
            text = "不是，最近写记录有帮助。"
            session.send_message(text)
        new = next(r for r in session.snapshot()["records"] if r["content"] == text)
        self.assertNotIn("clarification_response", new["structured_payload"])
        self.assertNotIn(old_check["check_turn_id"], json.dumps(new))

    def test_unrelated_affirmation_and_multitopic_reply_are_not_question_confirmations(self):
        for text in ("是的。", "是的，今天我想聊学校经历。", "最近思维记录和游泳都很有帮助。"):
            with self.subTest(text=text):
                session = self.session()
                session.send_message("我觉得游泳这项练习有点麻烦。")
                old = self.stage(session)
                session.send_message("这周思维记录作业怎么安排？")
                old_check = dict(session._pending_memory_check)
                session.send_message(text)
                new = [r for r in session.snapshot()["records"] if r["content"] == text]
                for record in new:
                    self.assertNotIn("clarification_response", record["structured_payload"])
                    self.assertNotIn(old_check["check_turn_id"], json.dumps(record))
                self.assertTrue(all(r["confirmation_state"] == "unconfirmed" for r in session.snapshot()["records"]))


if __name__ == "__main__":
    unittest.main()
