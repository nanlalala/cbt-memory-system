"""Fixture end-to-end run of the union layers on top of a frozen prior run."""
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation/cbt-memory-effect"))
from architecture_retest import runner
from architecture_retest.journal import atomic
from architecture_union import run as union_run
from architecture_union.layers import UnionConfig
import cactus_materials


class FakeDense:
    def rank(self, query, documents):
        return list(range(len(documents)))


class Transport:
    fixture_only = True

    def __init__(self, identity):
        self.identity, self.sent = identity, []

    def __call__(self, body):
        self.sent.append(body)
        system = body["messages"][0]["content"]
        data = json.loads(body["messages"][-1]["content"]) if body["messages"][-1]["content"].startswith("{") else {}
        if system.startswith("Extract only reusable"):
            s = next(s for s in data["history"] if s["role"] == "user")
            value = {"items": [{"item_id": "m0", "section": "case_state", "value": "Processed fixture summary.",
                                "origin": "explicit_user_statement",
                                "evidence": [{"source_id": s["source_id"], "quote": s["content"]}]}]}
        elif system.startswith("Independently assess whether"):
            value = {"verdicts": [{"item_id": i["item_id"], "supported": True, "reason": "fixture"} for i in data["items"]]}
        else:
            value = None
        return dict(text=json.dumps(value) if value is not None else "A fixture response.",
                    provider_model=self.identity["returned_model"], finish_reason="stop", usage={"total_tokens": 3},
                    wire_sha256=sha256(json.dumps(body).encode()).hexdigest())


LADDER = {"L1": UnionConfig(), "L4": UnionConfig(rerank=True, fact_key_sources=3, pin_profile=True, renderer="v5")}


class UnionRun(unittest.TestCase):
    def prior(self, root):
        source = root / "invented.json"
        source.write_text(json.dumps(cactus_materials.invented_fixture_rows()))
        runner.prepare(source, root / "prior", family_limit=1, fixture_only=True)
        runner.run(root / "prior", Transport(runner.IDENTITY), max_new_calls=15, workers=1,
                   dense=FakeDense(), fixture_only=True)
        return root / "prior"

    def test_prepare_run_resume_export_and_compare(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            prior = self.prior(root)
            protocol = union_run.prepare(prior, root / "union", dev_families=1, ladder=LADDER, fixture_only=True)
            self.assertEqual(protocol["case_count"], 3)
            self.assertEqual(protocol["maximum_attempts"], 6)
            transport = Transport(runner.IDENTITY)
            summary = union_run.run(root / "union", transport, max_new_calls=6, workers=1,
                                    dense=FakeDense(), fixture_only=True)
            self.assertEqual(summary["completed_cases"], 3)
            self.assertEqual({k: v["received"] for k, v in summary["configs"].items()}, {"L1": 3, "L4": 3})
            # Only managed answers are sent; nothing is re-extracted.
            self.assertTrue(all(not b["messages"][0]["content"].startswith("Extract") for b in transport.sent))
            for request in (root / "union/cases").glob("*/*.request.json"):
                audit = json.loads(request.read_text())["selection_audit"]
                self.assertTrue(audit["raw_selection_is_subset_of_managed"])
            again = Transport(runner.IDENTITY)
            union_run.run(root / "union", again, max_new_calls=0, workers=1, dense=FakeDense(), fixture_only=True)
            self.assertEqual(again.sent, [])

            judge_identity = {"provider": "fixture", "model": "judge", "returned_model": "judge-r", "model_family": "qwen"}
            union_run.export_quality(root / "union", judge_identity, root / "quality", maximum_attempts=50,
                                     fixture_only=True)
            mapping = json.loads((root / "quality/controller_arm_mapping.json").read_text())
            self.assertEqual({r["arm"] for r in mapping}, {"raw_history_rag", "managed_memory_rag", "L1", "L4"})
            self.assertTrue(all(r["status"] == "received" for r in mapping))
            reviews = [{"item_id": "r" + str(n), "case_id": r["packet_id"], "source_family": r["source_family"],
                        "repeat_of": None, "review": {"status": "completed", "assessment": {
                            "R1": 4 if r["arm"] == "L4" else 3, "R2": 3, "R3": 3, "R4": 3, "R5": "pass"}}}
                       for n, r in enumerate(mapping)]
            (root / "quality/reviews.jsonl").write_text("".join(json.dumps(r) + "\n" for r in reviews))
            report = union_run.compare(root / "quality", draws=100)
            self.assertEqual(report["comparisons"]["L4"]["R1"]["wins"], 3)
            self.assertEqual(report["comparisons"]["L1"]["R1"]["ties"], 3)
            self.assertEqual(report["comparisons"]["L4"]["R1"]["family_cluster_difference"]["mean_difference"], 1)

    def test_changed_code_or_prior_blocks_the_run(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            prior = self.prior(root)
            union_run.prepare(prior, root / "union", dev_families=1, ladder=LADDER, fixture_only=True)
            protocol = json.loads((root / "union/protocol.json").read_text())
            protocol["code_sha256"]["x.py"] = "0"
            atomic(root / "union/protocol.json", protocol)
            with self.assertRaises(ValueError):
                union_run.load(root / "union", fixture_only=True)

    def test_split_is_deterministic_and_disjoint(self):
        ids = ["family-%d" % n for n in range(10)]
        a, b = union_run.split_families(ids, 4), union_run.split_families(list(reversed(ids)), 4)
        self.assertEqual(a, b)
        self.assertEqual(len(a["dev"]), 4)
        self.assertFalse(set(a["dev"]) & set(a["test"]))
        with self.assertRaises(ValueError):
            union_run.split_families(ids, 0)

    def test_bad_ladder_names_are_refused(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            prior = self.prior(root)
            for bad in ({"raw_history_rag": UnionConfig()}, {"a/b": UnionConfig()}, {}):
                with self.assertRaises(ValueError):
                    union_run.prepare(prior, root / ("u" + str(len(bad))), dev_families=1, ladder=bad,
                                      fixture_only=True)


if __name__ == "__main__":
    unittest.main()
