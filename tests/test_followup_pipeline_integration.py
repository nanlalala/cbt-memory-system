"""Explicit small fictional replay fixture; not live models or human outcomes."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation/dialogue-benefit"))
import prepare_generation_materials as compiler
import run_followup_pipeline as runner


class FollowupIntegrationTests(unittest.TestCase):
    def test_partial_resume_event_extraction_privacy_and_shared_assessment(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            run = Path(directory) / "run"
            compiler.prepare_materials(ROOT / "evaluation/dialogue-benefit/task_materials_draft.json", source)
            runner.prepare(source, run, fixture=True)
            events = runner.rows(source / "replay_events.jsonl")
            private = {e["variant"]: e["source"]["text"] for e in events if e["kind"] == "source" and e["source"]["source_id"].endswith("S13")}
            seen_sources = []; purposes = []
            def fake(messages, purpose):
                purposes.append(purpose)
                if purpose == "extraction":
                    data = json.loads(messages[-1]["content"])
                    sid = data["current_source_id"]
                    self.assertNotIn("completion_criterion", data)
                    self.assertNotIn("query", data)
                    if sid.endswith("S14"):
                        self.assertIn(sid[:1] + "-S13", seen_sources)
                        self.assertNotIn(private[sid[:1]], json.dumps(data, ensure_ascii=False))
                    self.assertFalse(sid.endswith("S15"))
                    seen_sources.append(sid)
                    current = next(s for s in data["sources"] if s["source_id"] == sid)
                    answer = json.dumps({"candidates": [{"memory_type": "explicit_fact", "content": current["text"],
                        "source_turn_ids": [sid], "supporting_text": current["text"], "structured_payload": {}}]}, ensure_ascii=False)
                elif purpose == "response":
                    answer = json.dumps({"answer": "Explicit offline fixture only.", "used_source_ids": [], "clarification": None})
                else:
                    data = json.loads(messages[-1]["content"])
                    self.assertNotIn("arm", data)
                    self.assertTrue(all(s.get("permitted_absence_reason") is None for s in data["scoring_source_references"]))
                    answer = json.dumps({"criterion_result": "unresolved", "source_quotes": [], "response_quotes": [], "reason": "Fixture, not actual model assessment"})
                return {"hypothesis": answer, "provider_model": runner.ALIAS, "finish_reason": "stop", "latency_seconds": .01,
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
            first = runner.generate(run, fixture=True, transport=fake, max_new_calls=40, workers=4)
            self.assertEqual(first["status"], "partial")
            self.assertFalse((run / "generation_receipt.json").exists())
            self.assertEqual(purposes.count("extraction"), 39)
            final = runner.generate(run, fixture=True, transport=fake, max_new_calls=160, workers=4)
            self.assertEqual(final["terminal_arm_rows"], 108)
            self.assertEqual(purposes.count("extraction"), 39, "resume must not regenerate earlier source extractions")
            generated = runner.rows(run / "generation_rows.jsonl")
            self.assertEqual(len(generated), 108)
            self.assertTrue(all(row["extraction_available"] for row in generated))
            reviewed = runner.review(run, fixture=True, transport=fake)
            self.assertTrue(reviewed["fixture_only"])
            self.assertEqual(reviewed["status"], "complete")
            self.assertEqual(reviewed["human_participants"], 0)
            self.assertFalse(reviewed["final_effectiveness_result"])
            self.assertEqual(purposes.count("judge"), 27, "identical answers share one same-case assessment; privacy references withheld")
            judged = runner.rows(run / "reviewed_rows.jsonl")
            self.assertEqual(sum(r["judgment"]["reason"] == "privacy_reference_withheld_from_model_review" for r in judged), 27)


if __name__ == "__main__": unittest.main()
