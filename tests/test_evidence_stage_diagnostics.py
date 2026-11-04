"""Real frozen-run regressions for scope, source versions and read-only diagnosis."""
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation/objective-memory"))
import run_objective_comparison as runner
import v3_run_objective_comparison as v3
import diagnose_evidence_stages as diagnostics


def turn(turn_id, text, order=0, *, observed_at=None, session="s"):
    return {"turn_id": turn_id, "session_id": session, "role": "user", "order": order,
            "text": text, "observed_at": observed_at, "sensitivity": "standard"}


def case(cid="q", history=None, *, as_of=None):
    return {"case_id": cid, "subject_id": "u", "group_id": "g", "dataset_key": "synthetic",
            "task": "recall", "query": "What colour was recorded?", "options": [], "events": [],
            "as_of": as_of, "history": history if history is not None else [turn("t0", "blue"), turn("t1", "Hello", 1)]}


def prepare(path, cases=None, *, module=runner):
    cases = cases or [case()]
    refs = [{"case_id": c["case_id"], "group_id": "g", "dataset_key": c["dataset_key"], "task": c["task"],
             "metric": "mab_substring_exact_match", "accepted_answers": ["blue"]} for c in cases]
    settings = module.Settings(model="fixture", backend="lexical", short_turns=1, top_k=3, minimum_relevance=0,
                               short_chars=1200, long_chars=1600, max_context_chars=6000)
    module.prepare_run(cases, refs, path, settings, "https://example.invalid/v1", {}, fixture_only=True)
    return module.read_run_state_readonly(path)


def annotation(state, cid="q", turn_id="t0", *, span=None):
    c = next(c for c in state["frozen_rows"]["cases.jsonl"] if c["case_id"] == cid)
    source = next(s for s in state["frozen_rows"]["inventory.jsonl"] if s["source"]["turn_id"] == turn_id)
    text = source["source"]["text"]
    start, end = span if span is not None else (0, len(text))
    return {"case_id": cid, "protocol_sha256": state["protocol"]["protocol_sha256"],
            "history_inventory_sha256": c["history_inventory_sha256"], "reviewer_id": "fixture-author",
            "annotation_origin": "developer_fixture", "coverage": "complete_sufficient_alternatives",
            "alternatives": [[{"document_id": source["document_id"], "start": start, "end": end,
                               "source_text_sha256": hashlib.sha256(text.encode()).hexdigest()}]]}


def file_snapshot(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest() for p in path.rglob("*") if p.is_file()}


class EvidenceDiagnosticsTests(unittest.TestCase):
    def test_without_labels_no_stage_blame_and_unknown_user_effort(self):
        with tempfile.TemporaryDirectory() as d:
            prepare(Path(d))
            summary, rows = diagnostics.diagnose(Path(d))
            self.assertEqual(summary["arm_row_counts"], {"pending": 3})
            self.assertEqual(summary["unique_canonical_requests"], 2)
            self.assertFalse(summary["final_effectiveness_result"])
            self.assertTrue(all(r["causal_attribution"] == "unresolved" for r in rows))
            self.assertTrue(all(r["user_effort_seconds"] is None for r in rows))

    def test_real_quote_context_positive_but_recent_gap_not_retrieval_failure(self):
        with tempfile.TemporaryDirectory() as d:
            state = prepare(Path(d))
            _, rows = diagnostics.diagnose(Path(d), [annotation(state)])
            by_arm = {r["arm"]: r for r in rows}
            self.assertEqual(by_arm["short_term"]["evidence_observation"], "annotated_evidence_outside_recent_context")
            self.assertEqual(by_arm["short_term"]["stages"]["retrieval"], "not_applicable_recent_only")
            for arm in ("raw_history_rag", "managed_memory_rag"):
                self.assertTrue(by_arm[arm]["stages"]["context"])
                self.assertEqual(by_arm[arm]["evidence_observation"], "sufficient_annotated_source_spans_in_actual_context")
                self.assertEqual(by_arm[arm]["causal_attribution"], "unresolved")

    def test_same_subject_future_annotation_rejected_by_real_diagnose(self):
        early = [turn("t0", "blue", observed_at="2026-10-01T00:00:00Z"),
                 turn("t1", "Hello", 1, observed_at="2026-10-01T01:00:00Z")]
        late = early + [turn("future", "red", 2, observed_at="2026-10-03T00:00:00Z")]
        with tempfile.TemporaryDirectory() as d:
            state = prepare(Path(d), [case("early", early, as_of="2026-10-02T00:00:00Z"),
                                      case("late", late, as_of="2026-10-04T00:00:00Z")])
            label = annotation(state, "early", "future")
            with self.assertRaisesRegex(ValueError, "outside current case-visible"):
                diagnostics.diagnose(Path(d), [label])

    def test_same_subject_released_prefix_annotation_cannot_see_later_turn(self):
        def persona(cid, n):
            c = case(cid, [turn(f"t{i}", text, i, session="persona32k:ctx:released-sequence")
                           for i, text in enumerate(["blue", "Hello", "red"][:n])])
            c.update(dataset_key="personamem_v1", source_context_id="ctx", history_scope_id=f"persona32k:ctx:prefix:{n}",
                     visible_context_end_exclusive=n, options=["(a) blue", "(b) red", "(c) green", "(d) black"])
            return c
        with tempfile.TemporaryDirectory() as d:
            state = prepare(Path(d), [persona("early", 2), persona("late", 3)])
            with self.assertRaisesRegex(ValueError, "outside current case-visible"):
                diagnostics.diagnose(Path(d), [annotation(state, "early", "t2")])

    def test_repeated_id_different_source_version_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            state = prepare(Path(d), [case("early"), case("late", [turn("t0", "red"), turn("t1", "Hello", 1)])])
            label = annotation(state, "early")
            label["alternatives"][0][0]["source_text_sha256"] = hashlib.sha256(b"red").hexdigest()
            with self.assertRaisesRegex(ValueError, "conflicting source versions"):
                diagnostics.diagnose(Path(d), [label])

    def test_wrong_source_hash_and_wrong_case_history_binding_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            state = prepare(Path(d))
            for which in ("source", "history"):
                label = annotation(state)
                if which == "source":label["alternatives"][0][0]["source_text_sha256"] = hashlib.sha256(b"red").hexdigest()
                else:label["history_inventory_sha256"] = "wrong-scope"
                with self.assertRaisesRegex(ValueError, "checksum/version|case-visible history"):
                    diagnostics.diagnose(Path(d), [label])

    def test_actual_prompt_body_version_checked_not_trace_ids_only(self):
        original = runner._prepare_prompt
        def corrupt(*args, **kwargs):
            p = original(*args, **kwargs)
            if p["arm"] != "short_term":
                p["messages"][1]["content"] = p["messages"][1]["content"].replace(">blue</turn>", ">red</turn>")
                p["prompt_sha256"] = p["canonical_request_key"] = runner.stable_digest(p["messages"])
            return p
        with tempfile.TemporaryDirectory() as d:
            with patch.object(runner, "_prepare_prompt", side_effect=corrupt):state = prepare(Path(d))
            raw = next(p for p in state["prompts"] if p["arm"] == "raw_history_rag")
            self.assertIn(">red</turn>", raw["messages"][1]["content"])
            with self.assertRaisesRegex(ValueError, "quotation differs from source version"):
                diagnostics.diagnose(Path(d), [annotation(state)])

    def test_missing_or_reordered_actual_context_trace_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            state = prepare(Path(d))
            view = diagnostics.visible_source_views(state["frozen_rows"]["cases.jsonl"], state["prompts"], state["frozen_rows"]["inventory.jsonl"])["q"]
            p = deepcopy(next(p for p in state["prompts"] if p["arm"] == "raw_history_rag"))
            p["trace"]["included_long_ids"] = []
            with self.assertRaisesRegex(ValueError, "differ from trace"):
                diagnostics.observe(p, annotation(state), view=view)

    def test_partial_annotations_rejected_and_unknown_encoding_stays_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            state = prepare(Path(d));label = annotation(state)
            label["coverage"] = "partial"
            with self.assertRaisesRegex(ValueError, "partial"):
                diagnostics.diagnose(Path(d), [label])
            label["coverage"] = "complete_sufficient_alternatives"
            view = diagnostics.visible_source_views(state["frozen_rows"]["cases.jsonl"], state["prompts"], state["frozen_rows"]["inventory.jsonl"])["q"]
            p = deepcopy(state["prompts"][0]);p["trace"]["encoding"] = "unproven_summary"
            row = diagnostics.observe(p, label, view=view)
            self.assertIsNone(row["context_coverage"])
            self.assertEqual(row["evidence_observation"], "unresolved_unsupported_context_encoding")

    def test_joint_evidence_and_alternatives_are_span_based(self):
        with tempfile.TemporaryDirectory() as d:
            state = prepare(Path(d));label = annotation(state)
            second = annotation(state, turn_id="t1")["alternatives"][0][0]
            label["alternatives"][0].append(second)
            _, rows = diagnostics.diagnose(Path(d), [label])
            self.assertTrue(next(r for r in rows if r["arm"] == "raw_history_rag")["stages"]["context"])
            self.assertFalse(next(r for r in rows if r["arm"] == "short_term")["stages"]["context"])

    def test_v3_exact_span_context_does_not_claim_whole_document_coverage(self):
        source = "lunch " * 600 + "I prefer swimming at sunset. " + "lunch " * 600
        c = case(history=[turn("t0", source, session="persona32k:ctx:released-sequence"),
                          turn("t1", "Hello", 1, session="persona32k:ctx:released-sequence")])
        c.update(dataset_key="personamem_v1", task="preference", query="What should I choose?", source_context_id="ctx",
                 history_scope_id="persona32k:ctx:prefix:2", visible_context_end_exclusive=2,
                 options=["(a) swimming", "(b) walking", "(c) coding", "(d) sailing"])
        with tempfile.TemporaryDirectory() as d:
            state = prepare(Path(d), [c], module=v3)
            start = source.index("I prefer swimming")
            label = annotation(state, span=(start, start + len("I prefer swimming at sunset.")))
            _, rows = diagnostics.diagnose(Path(d), [label])
            raw = next(r for r in rows if r["arm"] == "raw_history_rag")
            self.assertTrue(raw["stages"]["context"])
            _, rows = diagnostics.diagnose(Path(d), [annotation(state)])
            raw = next(r for r in rows if r["arm"] == "raw_history_rag")
            self.assertFalse(raw["stages"]["context"])
            self.assertIsNone(raw["stages"]["retrieval"])

    def test_readonly_diagnose_missing_mirror_never_creates_lock_or_file(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d);live = base / "live"
            state = prepare(live, [case(history=[turn("t0", "blue")])])
            prompt = state["prompts"][0]
            intent = {"request_key": prompt["canonical_request_key"], "canonical_request_key": prompt["canonical_request_key"],
                      "protocol_sha256": state["protocol"]["protocol_sha256"], "prompt_sha256": prompt["prompt_sha256"],
                      "attempt_index": 1, "started_at_unix": 1}
            runner._append(live / "response_attempts.jsonl", intent)
            runner._append(live / "response_outcomes.jsonl", {**intent, "status": "technical_failure", "retryable": False})
            offline = base / "offline"
            shutil.copytree(live, offline, ignore=shutil.ignore_patterns(".execution.lock"))
            before = file_snapshot(offline)
            with patch.object(runner, "_replace_rows", side_effect=AssertionError("readonly diagnosis attempted a write")):
                summary, _ = diagnostics.diagnose(offline)
            self.assertEqual(summary["arm_row_counts"], {"exhausted_failure": 3})
            self.assertEqual(summary["canonical_request_counts"], {"exhausted_failure": 1})
            self.assertEqual(summary["recorded_provider_attempt_intents"], 1)
            self.assertEqual(before, file_snapshot(offline))
            self.assertFalse((offline / "failures.jsonl").exists())
            self.assertFalse((offline / ".execution.lock").exists())

    def test_valid_readonly_copy_has_identical_file_list_and_hashes_after_diagnose(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d);state = prepare(path);before = file_snapshot(path)
            original = Path.open
            def readonly_open(p, mode="r", *args, **kwargs):
                if any(c in mode for c in "wa+"):raise AssertionError("write mode requested by readonly diagnosis")
                return original(p, mode, *args, **kwargs)
            with patch.object(Path, "open", readonly_open):
                diagnostics.diagnose(path, [annotation(state)])
            self.assertEqual(before, file_snapshot(path))


if __name__ == "__main__":
    unittest.main()
