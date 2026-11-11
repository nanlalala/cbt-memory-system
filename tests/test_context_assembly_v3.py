"""Pure invented fixtures; no model calls or corpus execution."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("context_assembly_v3", Path(__file__).resolve().parents[1] /
    "evaluation/cbt-memory-effect/context_assembly_v3.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def source(sid, order, text=None, role="user", **overrides):
    return {"source_id": sid, "source_order": order, "role": role,
            "text": text or "Original text for " + sid, "user_id": "u",
            "permissions": {"save": True, "long_term_use": True, "response_external": True}, **overrides}


class ContextAssemblyTests(unittest.TestCase):
    def options(self, **overrides):
        return {"sources": [source("old", 0, "I did not agree to the practice."),
                            source("candidate-only", 1, "The proposed plan remains undecided."),
                            source("recent-u", 2, "I would like to discuss this."),
                            source("recent-a", 3, "Which aspect would be useful?", role="assistant")],
                "user_id": "u", "cutoff_order_exclusive": 4, "current_input": "That last question.",
                "current_source_id": "current", "system_prompt": "Respond using the supplied evidence.",
                "recent_source_ids": ["recent-u", "recent-a"], "raw_source_ids": ["old", "recent-u"],
                "candidate_source_ids": ["candidate-only", "old"],
                "request_metadata": {"model": "fixture-only", "max_tokens": 128},
                "max_outbound_chars": 10000, **overrides}

    def test_rrf_deduplicates_within_channel_and_allows_joint_rank_to_change_order(self):
        result = M.fuse_source_ranks(["raw-first", "shared", "shared"], ["candidate-first", "shared"])
        self.assertEqual(result[0]["source_id"], "shared")
        self.assertAlmostEqual(result[0]["score"], 2 / 62)
        self.assertEqual(len(result), 3)
        self.assertEqual(M.fuse_source_ranks(["a", "a", "b"], []), M.fuse_source_ranks(["a", "b"], []))

    def test_index_projection_returns_sources_without_candidate_inference(self):
        hits = [{"source_ids": ["s2", "s1"], "content": "AI inference MUST NOT ENTER CONTEXT"},
                {"source_ids": ["s1", "s3"]}]
        self.assertEqual(M.project_candidate_sources(hits), ["s2", "s1", "s3"])
        with self.assertRaises(ValueError):
            M.project_candidate_sources([{"content": "unsupported"}])

    def test_all_arms_share_direct_recent_order_and_current_once(self):
        result = M.assemble_contexts(**self.options())
        for context in result["contexts"].values():
            direct = [message for message in context["messages"] if message["role"] != "system"]
            self.assertEqual(direct, [{"role": "user", "content": "I would like to discuss this."},
                {"role": "assistant", "content": "Which aspect would be useful?"},
                {"role": "user", "content": "That last question."}])
            self.assertEqual(sum(m["content"] == "That last question." for m in context["messages"]), 1)
            self.assertEqual(context["recent_source_ids"], ["recent-u", "recent-a"])
        self.assertEqual(result["contexts"]["short_term"]["retrieved_source_ids"], [])

    def test_compact_packet_stores_each_complete_source_once_and_keeps_dependencies(self):
        result = M.assemble_contexts(**self.options())
        managed = result["contexts"]["managed_memory_rag"]
        packet_message = next(m["content"] for m in managed["messages"] if m["content"].startswith(M.PACKET_HEADER))
        packet = json.loads(packet_message[len(M.PACKET_HEADER):])
        self.assertEqual({r["source_id"] for r in packet["sources"]}, {"old", "candidate-only"})
        self.assertTrue(all(set(r) == {"source_id", "role", "text", "source_order"} for r in packet["sources"]))
        self.assertEqual(packet_message.count("I did not agree to the practice."), 1)
        self.assertNotIn("I would like to discuss this.", packet_message)
        self.assertEqual(set(managed["source_dependencies"]), {"old", "candidate-only", "recent-u", "recent-a"})
        self.assertEqual(set(managed["source_snapshot_sha256"]), set(managed["source_dependencies"]))

    def test_permission_foreign_subject_and_future_excluded_before_fusion(self):
        options = self.options()
        options["sources"] += [source("future", 4), source("foreign", 0, user_id="other"),
            source("blocked", 0, permissions={"save": True, "long_term_use": True, "response_external": False})]
        options["candidate_source_ids"] = ["future", "foreign", "blocked", "candidate-only", "old"]
        result = M.assemble_contexts(**options)
        baseline = M.assemble_contexts(**self.options())
        self.assertEqual(result, baseline)

    def test_full_escaped_request_count_and_whole_quote_budget(self):
        options = self.options(max_recent_messages=0)
        options["sources"][0]["text"] = "原话里有否定，不能截断。" * 50
        options["max_outbound_chars"] = 1200
        result = M.assemble_contexts(**options)
        for context in result["contexts"].values():
            self.assertEqual(context["outbound_chars"], len(context["serialized_request"]))
            self.assertEqual(json.loads(context["serialized_request"]), context["request_payload"])
            self.assertLessEqual(context["outbound_chars"], 1200)
            self.assertNotIn("old", context["retrieved_source_ids"])
            self.assertNotIn("原话里有否定", json.dumps(context["messages"], ensure_ascii=False))
        self.assertEqual(result["contexts"]["raw_history_rag"]["dropped_sources"]["old"],
                         "complete_source_exceeds_outbound_budget")

    def test_long_recent_exchange_trimmed_identically_across_arms(self):
        options = self.options(max_outbound_chars=600)
        options["sources"][2]["text"] = "This old recent message is very long. " * 100
        result = M.assemble_contexts(**options)
        for context in result["contexts"].values():
            self.assertEqual(context["recent_source_ids"], [])
            self.assertEqual(context["messages"][-1]["content"], "That last question.")
            self.assertLessEqual(context["outbound_chars"], 600)

    def test_recent_order_conflicting_versions_and_metadata_override_rejected(self):
        with self.assertRaises(ValueError):
            M.assemble_contexts(**self.options(recent_source_ids=["recent-a", "recent-u"]))
        options = self.options(); options["sources"].append(source("old", 0, "different source version"))
        with self.assertRaises(ValueError):
            M.assemble_contexts(**options)
        with self.assertRaises(ValueError):
            M.assemble_contexts(**self.options(request_metadata={"messages": []}))
        with self.assertRaises(M.ContextBudgetError):
            M.assemble_contexts(**self.options(max_outbound_chars=10))

    def test_inputs_unchanged_and_development_status_explicit(self):
        options = self.options(); before = deepcopy(options)
        result = M.assemble_contexts(**options)
        self.assertEqual(options, before)
        self.assertFalse(result["ready_for_external_generation"])
        self.assertEqual(result["integration_status"], M.INTEGRATION_STATUS)
        self.assertIn("durable send and publication admission for every source dependency", result["remaining_requirements"])

    def test_current_source_id_collision_with_history_rejected(self):
        options = self.options()
        options["sources"].append(source("current", 1, options["current_input"]))
        options["recent_source_ids"].append("current")
        options["raw_source_ids"].insert(0, "current")
        with self.assertRaisesRegex(ValueError, "conflicts with eligible historical source"):
            M.assemble_contexts(**options)
        with self.assertRaisesRegex(ValueError, "conflicts with eligible historical source"):
            M.assemble_contexts(**self.options(current_source_id="old"))

    def test_unique_current_input_appears_once_with_repeated_rank_ids(self):
        options = self.options()
        options["recent_source_ids"].append("current")
        options["raw_source_ids"].insert(0, "current")
        result = M.assemble_contexts(**options)
        for context in result["contexts"].values():
            self.assertNotIn("current", context["source_dependencies"])
            self.assertEqual(sum(m["content"] == options["current_input"] for m in context["messages"]), 1)

    def test_recent_window_does_not_begin_with_orphan_assistant(self):
        result = M.assemble_contexts(**self.options(max_recent_messages=1))
        for context in result["contexts"].values():
            self.assertEqual(context["recent_source_ids"], [])
            self.assertFalse(any(m["role"] == "assistant" for m in context["messages"]))

    def test_filtered_recent_user_does_not_leave_orphan_assistant(self):
        options = self.options()
        options["sources"][2]["permissions"]["response_external"] = False
        result = M.assemble_contexts(**options)
        for context in result["contexts"].values():
            self.assertEqual(context["recent_source_ids"], [])
            self.assertFalse(any(m["role"] == "assistant" for m in context["messages"]))


if __name__ == "__main__":
    unittest.main()
