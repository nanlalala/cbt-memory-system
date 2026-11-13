"""Budget and quiescence checks; no hosted runtime or provider is fabricated."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "evaluation/architecture-rerun/cloud_quality_followup.py"
spec = importlib.util.spec_from_file_location("quality_followup", SOURCE)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


def plan(old=915, prior=0):
    return {"original_cbt_envelope": {api.MODEL: 5032},
        "separate_professional7_quality_repair_allowance": 24,
        "cbt_actual_local_paid_intents": {api.MODEL: old},
        "previous_authorized_recovery_attempts_counted": {api.MODEL: prior},
        "cbt_remaining_after_paid_intents": {api.MODEL: 5056-old-prior}}


def pair(request="one", cohort="professional7-quality", status="received", model=api.MODEL):
    common = {"request_id": request, "job_id": "nonce:" + cohort,
        "model": model, "wire_sha256": "a" * 64}
    return [{"event": "intent", **common}, {"event": "outcome", "status": status, **common}]


class BudgetTests(unittest.TestCase):
    def test_counts_all_cbt_new_sends_and_preserves_old_census(self):
        rows = pair() + pair("two", "cactus600-continuation")
        result = api.reconciled_budget(plan(prior=2), iter(rows), "nonce")
        self.assertEqual(result["v4_cbt_qwen_admitted"], 2)
        self.assertEqual(result["remaining_qwen"], 4137)

    def test_locomo_and_deepseek_never_create_cbt_qwen_allowance(self):
        rows = pair("one", "locomo1540-judge") + pair("two", model="deepseek-v4.1-flash")
        result = api.reconciled_budget(plan(), rows, "nonce")
        self.assertEqual(result["remaining_qwen"], 4141)
        self.assertFalse(result["locomo_allowance_borrowed"])

    def test_inflight_and_unmatched_outcomes_block(self):
        for rows in (pair()[:1], pair()[1:]):
            with self.subTest(rows=rows), self.assertRaises(RuntimeError):
                api.reconciled_budget(plan(), rows, "nonce")

    def test_unknown_outcome_blocks_instead_of_retrying(self):
        with self.assertRaises(RuntimeError):
            api.reconciled_budget(plan(), pair(status="unknown"), "nonce")

    def test_duplicate_foreign_or_mismatched_receipts_block(self):
        bad = pair()
        bad[1]["wire_sha256"] = "b" * 64
        variants = (pair()+pair(), [{**r, "job_id": "other:professional7-quality"} for r in pair()], bad)
        for rows in variants:
            with self.subTest(rows=rows), self.assertRaises((RuntimeError, ValueError)):
                api.reconciled_budget(plan(), rows, "nonce")

    def test_original_cap_and_paid_census_cannot_be_enlarged(self):
        variants = []
        for field in ("original_cbt_envelope", "cbt_remaining_after_paid_intents"):
            value = deepcopy(plan());value[field][api.MODEL] += 1;variants.append(value)
        value = deepcopy(plan());value["separate_professional7_quality_repair_allowance"] += 1;variants.append(value)
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError):
                api.reconciled_budget(value, [], "nonce")

    def test_negative_remaining_does_not_open_a_new_cap(self):
        with self.assertRaises(ValueError):
            api.reconciled_budget(plan(old=5056), pair(), "nonce")

    def test_physical_lf_reader_preserves_unicode_separators(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"events.jsonl"
            rows = pair();rows[0]["payload"] = {"text": "one\u2028two\u2029three"}
            path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows))
            self.assertEqual(api.reconciled_budget(plan(), api._event_rows(path), "nonce")["v4_cbt_qwen_admitted"], 1)
            path.write_text(json.dumps(rows[0]))
            with self.assertRaises(ValueError):
                list(api._event_rows(path))


class QuiescenceTests(unittest.TestCase):
    def test_each_original_invocation_has_exactly_one_followup_root(self):
        nonce = "b12ea223-f4c7-476c-83e1-a91233443f78"
        expected = Path("/content") / ("capstone-quality-v3-" + nonce)
        self.assertEqual(api.followup_root(nonce), expected)
        for bad in ("different", nonce.upper(), "../" + nonce):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                api.followup_root(bad)

    def test_active_thread_or_worker_blocks_followup(self):
        for additions in ({"thread": SimpleNamespace(is_alive=lambda: True)},
                          {"process": SimpleNamespace(poll=lambda: None)}):
            with self.subTest(additions=additions), self.assertRaises(RuntimeError):
                api._terminal({**additions})

    def test_closed_original_invocation_required(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            (base/"entry.json").write_text(json.dumps({"nonce": "nonce"}))
            (base/"status.json").write_text(json.dumps({"status": "finished"}))
            endpoint = base/"api.sock"
            prior = {"root": base, "nonce": "nonce", "status": {"status": "finished"},
                "relay": SimpleNamespace(endpoint=endpoint)}
            api._terminal(prior)
            endpoint.touch()
            with self.assertRaises(RuntimeError):api._terminal(prior)
            endpoint.unlink()
            (base/"entry.json").write_text(json.dumps({"nonce": "different"}))
            with self.assertRaises(ValueError):api._terminal(prior)


if __name__ == "__main__":
    unittest.main()
