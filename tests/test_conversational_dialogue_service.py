from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules/agent-integration"))
from dialogue_service import DialogueSession, ModelOutput  # noqa: E402


QUOTE = "我觉得写思维记录这项作业有点麻烦。"
CURRENT = "这周思维记录作业怎么安排？"
HISTORY = "我上次提到思维记录作业时说了什么？"


class Provider:
    def __init__(self):
        self.calls = []
        self.extraction = None

    def __call__(self, *, messages, purpose):
        self.calls.append((purpose, messages))
        if purpose == "extraction":
            request = json.loads(messages[-1]["content"])
            turns = request["turns"]
            turn = next(turn for turn in turns if turn["turn_id"] in request["eligible_source_turn_ids"])
            text = self.extraction or turn["text"]
            return ModelOutput(json.dumps({"candidates": [{"memory_type": "explicit_fact",
                "content": text, "source_turn_ids": [turn["turn_id"]], "supporting_text": turn["text"]}]}),
                "offline-test", {}, False)
        return ModelOutput("我们可以看看这次的具体安排。", "offline-test", {}, False)


class ConversationalDialogueTests(unittest.TestCase):
    def session(self, **kwargs):
        session = DialogueSession(memory_mode="conversational", **kwargs)
        self.addCleanup(session.close)
        return session

    def stage(self, session):
        session.send_message(QUOTE)
        record = session.snapshot()["records"][0]
        session.new_session()
        return record

    def test_automatic_quote_requires_no_extra_model_call_or_confirmation(self):
        provider = Provider()
        session = self.session(provider=provider)
        session.send_message(QUOTE)
        data = session.snapshot()
        self.assertEqual([purpose for purpose, _ in provider.calls], ["response"])
        self.assertEqual(data["memory_processing"]["status"], "saved")
        self.assertEqual(data["extraction_scope"]["pending_user_turn_count"], 0)
        self.assertEqual(data["records"][0]["content"], QUOTE)
        self.assertEqual(data["records"][0]["lifecycle_status"], "candidate")
        self.assertEqual(data["records"][0]["confirmation_state"], "unconfirmed")
        self.assertEqual(data["records"][0]["memory_use_decision"]["action"], "use_as_history")
        self.assertNotIn("user_confirmed", [event["action"] for event in data["audit"]])

    def test_history_question_supplies_dated_quote_without_reconfirmation(self):
        provider = Provider()
        session = self.session(provider=provider)
        record = self.stage(session)
        reply = session.send_message(HISTORY)
        self.assertEqual(reply["history_memory_ids"], [record["memory_id"]])
        self.assertEqual(reply["check_memory_ids"], [])
        prompt = provider.calls[-1][1][0]["content"]
        self.assertIn(QUOTE, prompt)
        self.assertIn("not proof of current state", prompt)
        self.assertEqual(reply["history_sources"][0]["content"], QUOTE)

    def test_current_use_asks_one_optional_check_without_supplying_old_fact(self):
        provider = Provider()
        session = self.session(provider=provider)
        record = self.stage(session)
        reply = session.send_message(CURRENT)
        self.assertEqual(reply["memory_ids"], [])
        self.assertEqual(reply["check_memory_ids"], [record["memory_id"]])
        self.assertEqual(len(reply["memory_checks"]), 1)
        self.assertIn(QUOTE, reply["text"])
        self.assertNotIn(QUOTE, provider.calls[-1][1][0]["content"])
        self.assertNotIn("别的阻碍", reply["text"])
        second = session.send_message(CURRENT)
        self.assertEqual(second["check_memory_ids"], [])

    def test_check_question_does_not_turn_a_feeling_report_into_an_agreed_plan(self):
        source = "这周我对自己的状态有些担心。"
        question = DialogueSession._check_question(source, "我这周的目标可以怎么安排？")
        self.assertIn(source, question)
        self.assertNotIn("沿用这个安排", question)
        self.assertIn("变化", question)

        exercise = DialogueSession._check_question(QUOTE, CURRENT)
        self.assertIn(QUOTE, exercise)
        self.assertNotIn("沿用这个安排", exercise)
        self.assertNotIn("别的阻碍", exercise)

    def test_irrelevant_turn_does_not_trigger_a_memory_check(self):
        session = self.session()
        self.stage(session)
        reply = session.send_message("Explain Python recursion with a short example?")
        self.assertEqual(reply["check_memory_ids"], [])
        self.assertEqual(reply["memory_ids"], [])

    def test_same_window_quote_is_not_immediately_asked_again(self):
        session = self.session()
        session.send_message(QUOTE)
        self.assertEqual(session.send_message(CURRENT)["check_memory_ids"], [])

    def test_auto_disabled_and_opt_out_are_not_backfilled_on_extract_or_enable(self):
        provider = Provider()
        session = self.session(provider=provider)
        session.set_automatic_memory_enabled(False)
        session.send_message("关闭期间这条虚构散步信息不要进入长期整理。")
        session.set_automatic_memory_enabled(True)
        session.send_message("请不要记住这条：PRIVATE-FICTIONAL-SOURCE。")
        self.assertEqual(session.extract_current_session(), [])
        session.send_message("我想尝试午饭后散步这个小练习。")
        contents = [record["content"] for record in session.snapshot()["records"]]
        self.assertEqual(contents, ["我想尝试午饭后散步这个小练习。"])
        self.assertEqual(session.snapshot()["extraction_scope"]["pending_user_turn_count"], 0)
        self.assertEqual([purpose for purpose, _ in provider.calls], ["response"] * 3)

    def test_excluded_body_is_not_visible_in_later_model_extraction_context(self):
        provider = Provider()
        session = DialogueSession(provider=provider)
        self.addCleanup(session.close)
        session.send_message("不要记住这一条：PRIVATE-FICTIONAL-SOURCE。")
        session.send_message("这是一条允许整理的虚构散步原话。")
        records = session.extract_current_session()
        extraction_prompt = provider.calls[-1][1][-1]["content"]
        self.assertNotIn("PRIVATE-FICTIONAL-SOURCE", extraction_prompt)
        self.assertNotIn("PRIVATE-FICTIONAL-SOURCE", json.dumps(records))

    def test_successful_reply_survives_automatic_write_failure_and_can_retry(self):
        provider = Provider()
        session = self.session(provider=provider)
        with patch.object(session.manager, "ingest", side_effect=RuntimeError("private database error")):
            reply = session.send_message(QUOTE)
        self.assertEqual(reply["text"], "我们可以看看这次的具体安排。")
        self.assertEqual(session.snapshot()["memory_processing"]["status"], "failed")
        self.assertNotIn("private database error", json.dumps(session.snapshot()))
        self.assertEqual(session.snapshot()["records"], [])
        self.assertEqual(len(session.snapshot()["turns"]), 2)
        session.send_message("谢谢。")
        self.assertEqual([record["content"] for record in session.snapshot()["records"]], [QUOTE])

    def test_failed_provider_leaves_no_automatic_memory_or_successful_turn(self):
        def fail(**kwargs):
            raise TimeoutError("fixture")
        session = self.session(provider=fail)
        with self.assertRaises(TimeoutError):
            session.send_message(QUOTE)
        self.assertEqual(session.snapshot()["turns"], [])
        self.assertEqual(session.snapshot()["records"], [])

    def test_direct_revision_current_quote_binding_rejects_tampering(self):
        for change in ("content", "observed_at"):
            with self.subTest(change=change):
                session = self.session()
                record = self.stage(session)
                session.revise_candidate(record["memory_id"], "我上周只是想试试思维记录作业。")
                stored = session.store.get(session.user_id, record["memory_id"])
                if change == "content":
                    stored.content = "I live in Paris."
                    stored.source_evidence.supporting_text = stored.content
                else:
                    stored.observed_at = "2020-01-01T00:00:00Z"
                session.store.put(stored)
                view = session.snapshot()["records"][0]
                self.assertEqual(view["memory_use_decision"]["action"], "exclude")

    def test_model_confidence_or_fake_metadata_cannot_authorize_a_candidate(self):
        provider = Provider()
        # Explicit manual extraction of an eligible review turn, then switch to
        # conversational reads; an unrelated summary remains only a question.
        session = DialogueSession(provider=provider)
        self.addCleanup(session.close)
        session.send_message("I like walking after lunch.")
        provider.extraction = "The user lives in Paris."
        record = session.extract_current_session()[0]
        stored = session.store.get(session.user_id, record["memory_id"])
        stored.confidence = 1.0
        stored.structured_payload["verified"] = True
        session.store.put(stored)
        session.memory_mode = "conversational"
        session.new_session()
        reply = session.send_message("What did I previously mention about walking?")
        self.assertNotIn("Paris", provider.calls[-1][1][0]["content"])
        self.assertEqual(reply["history_sources"][0]["content"], "I like walking after lunch.")

    def test_unrelated_next_statement_does_not_claim_to_answer_a_check(self):
        session = self.session()
        self.stage(session)
        session.send_message(CURRENT)
        text = "今天我想聊一段完全不同的虚构学校经历。"
        session.send_message(text)
        newest = next(record for record in session.snapshot()["records"] if record["content"] == text)
        self.assertNotIn("clarification_response", newest["structured_payload"])

    def test_explicit_related_update_is_linked_as_dated_history_not_confirmation(self):
        session = self.session()
        old = self.stage(session)
        session.send_message(CURRENT)
        answer = "不是，最近写记录反而有帮助。"
        session.send_message(answer)
        newest = next(record for record in session.snapshot()["records"] if record["content"] == answer)
        self.assertEqual(newest["structured_payload"]["clarification_response"]["responds_to_memory_id"], old["memory_id"])
        self.assertEqual(newest["confirmation_state"], "unconfirmed")
        session.new_session()
        reply = session.send_message(CURRENT)
        self.assertEqual(reply["check_memory_ids"], [])
        self.assertIn(answer, [source["content"] for source in reply["history_sources"]])

    def test_short_yes_does_not_confirm_or_create_a_stable_fact(self):
        session = self.session()
        self.stage(session)
        session.send_message(CURRENT)
        before = len(session.snapshot()["records"])
        session.send_message("是的。")
        self.assertEqual(len(session.snapshot()["records"]), before)
        self.assertTrue(all(record["confirmation_state"] == "unconfirmed" for record in session.snapshot()["records"]))

    def test_short_chinese_self_reports_are_saved_but_unpunctuated_questions_are_not(self):
        session = self.session()
        for text in ("我喜欢游泳", "我想散步", "我的散步目标是什么", "是的"):
            session.send_message(text)
        self.assertEqual({record["content"] for record in session.snapshot()["records"]},
                         {"我喜欢游泳", "我想散步"})

    def test_english_forgetfulness_question_is_not_an_opt_out_command(self):
        session = self.session()
        self.assertFalse(session._do_not_remember("I don't remember my goal. What did I say?"))
        self.assertTrue(session._do_not_remember("Please don't remember this fictional detail."))

    def test_declining_checks_does_not_promote_unchecked_memory_to_current_fact(self):
        provider = Provider()
        session = self.session(provider=provider)
        self.stage(session)
        reply = session.send_message(CURRENT + "不用再问我过去的感受。")
        self.assertEqual(reply["check_memory_ids"], [])
        self.assertEqual(reply["memory_ids"], [])
        self.assertNotIn(QUOTE, provider.calls[-1][1][0]["content"])
        self.assertTrue(session.snapshot()["memory_checks_paused"])
        self.assertEqual(session.send_message(CURRENT)["check_memory_ids"], [])

    def test_deletion_expiry_restricted_and_other_user_are_excluded(self):
        session = self.session()
        record = self.stage(session)
        stored = session.store.get(session.user_id, record["memory_id"])
        stored.sensitivity = "restricted"
        session.store.put(stored)
        self.assertEqual(session.send_message(CURRENT)["check_memory_ids"], [])
        stored.sensitivity = "standard"
        stored.valid_to = "2020-01-01T00:00:00Z"
        session.store.put(stored)
        session.new_session()
        self.assertEqual(session.send_message(HISTORY)["memory_ids"], [])
        other = self.session(store=session.store, user_id="different-operator")
        self.assertEqual(other.send_message(HISTORY)["memory_ids"], [])
        session.delete(record["memory_id"], "DELETE " + record["memory_id"])
        self.assertEqual(session.send_message(HISTORY)["memory_ids"], [])
        self.assertEqual(session.snapshot()["records"], [])


if __name__ == "__main__":
    unittest.main()
