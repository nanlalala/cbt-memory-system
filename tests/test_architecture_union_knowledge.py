"""CBT knowledge (textbook RAG) channel in the union run: same cards in every arm."""
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation/cbt-memory-effect"))
sys.path.insert(0, str(ROOT / "tests"))
from architecture_retest import runner
from architecture_union import union_requests as u, knowledge as k, run as union_run
from architecture_union.run import DEFAULT_LADDER
from architecture_union.layers import UnionConfig
from test_architecture_union import fixture, long_fixture, item_on, user_source, FakeDense
from test_architecture_union_run import Transport, LADDER


def card(n, text="Fixture CBT note: review the agreed practice and its barriers before setting a new task."):
    return {"source_id": "fixture_cbt_notes", "chunk_id": "fx-%d" % n, "citation": "Fixture notes, p. %d" % n,
            "page_start": n, "page_end": n, "text": text, "chunk_sha256": "a" * 64, "source_sha256": "b" * 64,
            "evidence_status": "fixture_not_real_k1"}


def block(payload):
    message = next(m for m in payload["messages"] if m["content"].startswith("SOURCED MEMORY DATA ONLY."))
    return json.loads(message["content"].split("\n", 1)[1])


class FakeSelection:
    def __init__(self, items):
        self.items, self.status, self.reason = tuple(items), "retrieved" if items else "no_hit", "fixture"


class FakeAdapter:
    mode, policy_hash, fixture_only = "K1", "p" * 64, True

    class retriever:
        corpus_fingerprint = "c" * 64

    def __init__(self):
        self.queries = []

    def retrieve(self, query, *, task_kind, external):
        self.queries.append((query, task_kind, external))
        return FakeSelection([card(1), card(2)])


class KnowledgeRequests(unittest.TestCase):
    def test_without_knowledge_requests_are_byte_identical_to_before_the_channel(self):
        out = {}
        for fname, f in (("fixture", fixture), ("long", long_fixture)):
            case, sources = f()
            users = [s for s in sources[:-6] if s["role"] == "user"]
            items = [item_on(users[0], "a", "Plans walking practice."),
                     dict(item_on(users[-1], "b", "Background."), section="relevant_background")]
            for name, cfg in DEFAULT_LADDER.items():
                plans = u.prepare_case_requests_union(case, sources, items, FakeDense(), threading.Lock(), cfg)
                out[fname + "/" + name] = sha256(json.dumps({a: v["payload"] for a, v in sorted(plans.items())},
                                                            sort_keys=True).encode()).hexdigest()
        self.assertEqual(sha256(json.dumps(out, sort_keys=True).encode()).hexdigest(),
                         "f8ab9699980a23ac238c3f282c9fe7426602082df4c55fbc6ebad6c40d42ba04")

    def test_every_arm_gets_the_same_cards_and_memory_still_differs(self):
        case, sources = long_fixture()
        cards = [card(1), card(2)]
        plans = u.prepare_case_requests_union(case, sources, [item_on(user_source(sources))], FakeDense(),
                                              threading.Lock(), DEFAULT_LADDER["L4_plus_pinned_profile"], knowledge=cards)
        for arm in runner.ARMS:
            self.assertEqual(block(plans[arm]["payload"])["cbt_knowledge"], cards)
            self.assertEqual(plans[arm]["selection_audit"]["knowledge_card_ids"], ["fx-1", "fx-2"])
        self.assertTrue(plans["managed_memory_rag"]["selection_audit"]["raw_selection_is_subset_of_managed"])
        frozen = runner.prepare_case_requests(case, sources, [], FakeDense(), threading.Lock())
        self.assertEqual(plans["raw_history_rag"]["selection_audit"]["raw_selected_source_ids"],
                         frozen["raw_history_rag"]["selection_audit"]["raw_selected_source_ids"])

    def test_without_cards_base_arm_equals_frozen_runner(self):
        case, sources = long_fixture()
        frozen = runner.assemble(case, sources, [], [], "raw")
        self.assertEqual(u.assemble_base(case, sources, [], ()).request_payload, frozen.request_payload)

    def test_fit_drops_cards_from_the_end_until_every_arm_fits(self):
        case, sources = long_fixture()
        raw_ids = runner.prepare_case_requests(case, sources, [], FakeDense(), threading.Lock())[
            "raw_history_rag"]["selection_audit"]["raw_selected_source_ids"]
        big = "x" * (runner.SETTINGS["max_answer_chars"] // 3)
        cards = [card(1), card(2, big), card(3, big), card(4, big)]
        fitted = u.fit_knowledge(case, sources, raw_ids, cards, list(DEFAULT_LADDER.values()))
        self.assertTrue(0 < len(fitted) < len(cards))
        self.assertEqual(fitted, cards[:len(fitted)])
        self.assertEqual(u.fit_knowledge(case, sources, raw_ids, [card(1, "x" * 30000)], [UnionConfig()]), [])


class KnowledgePackets(unittest.TestCase):
    def test_build_and_load_packets(self):
        with tempfile.TemporaryDirectory() as d:
            cases = [{"case_id": "c1", "input": "How do I review homework?"}, {"case_id": "c2", "input": "Hi"}]
            adapter = FakeAdapter()
            meta = k.build_packets(cases, adapter, Path(d) / "p.json")
            self.assertEqual(meta["status_counts"], {"retrieved": 2})
            self.assertEqual([q[0] for q in adapter.queries], ["How do I review homework?", "Hi"])
            self.assertTrue(all(q[2] is True for q in adapter.queries))
            body = k.load_packets(Path(d) / "p.json", ["c1", "c2"], fixture_only=True)
            self.assertEqual(body["packets"]["c1"]["items"], [card(1), card(2)])
            self.assertEqual(list(k.load_packets(Path(d) / "p.json", ["c1"], fixture_only=True)["packets"]), ["c1"])
            with self.assertRaises(ValueError):
                k.load_packets(Path(d) / "p.json", ["c1", "c3"], fixture_only=True)
            with self.assertRaises(ValueError):
                k.load_packets(Path(d) / "p.json", ["c1", "c2"], fixture_only=False)
            with self.assertRaises(ValueError):
                k.build_packets(cases, adapter, Path(d) / "p.json")

    def test_k0_adapter_and_all_empty_packets_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            off = FakeAdapter()
            off.mode = "K0"
            with self.assertRaises(ValueError):
                k.build_packets([{"case_id": "c", "input": "x"}], off, Path(d) / "a.json")
            empty = FakeAdapter()
            empty.retrieve = lambda q, **kw: FakeSelection([])
            k.build_packets([{"case_id": "c", "input": "x"}], empty, Path(d) / "b.json")
            with self.assertRaises(ValueError):
                k.load_packets(Path(d) / "b.json", ["c"], fixture_only=True)

    def test_the_real_registry_gives_no_cards_today(self):
        sys.path.insert(0, str(ROOT / "modules/agent-integration"))
        from knowledge_adapter_v2 import KnowledgeAdapter
        selection = KnowledgeAdapter(mode="K1").retrieve("How should I review CBT homework?", task_kind="general")
        self.assertEqual(selection.items, ())
        self.assertEqual(selection.status, "unavailable")


class KnowledgeRun(unittest.TestCase):
    def test_fixture_run_sends_cards_to_all_arms_and_exports_them(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            source = root / "invented.json"
            import cactus_materials
            source.write_text(json.dumps(cactus_materials.invented_fixture_rows()))
            runner.prepare(source, root / "prior", family_limit=1, fixture_only=True)
            runner.run(root / "prior", Transport(runner.IDENTITY), max_new_calls=15, workers=1,
                       dense=FakeDense(), fixture_only=True)
            cases = json.loads((root / "prior/cases.json").read_text())
            k.build_packets(cases, FakeAdapter(), root / "packets.json")
            protocol = union_run.prepare(root / "prior", root / "union", dev_families=1, ladder=LADDER,
                                         fixture_only=True, knowledge_packets=root / "packets.json")
            self.assertEqual(protocol["maximum_attempts"], 3 * (len(LADDER) + 2))
            self.assertEqual(protocol["reused_from_prior"], ["extracted_and_checked_items"])
            self.assertTrue(protocol["knowledge"]["shared_by_all_arms"])
            transport = Transport(runner.IDENTITY)
            summary = union_run.run(root / "union", transport, max_new_calls=12, workers=1,
                                    dense=FakeDense(), fixture_only=True)
            self.assertEqual(summary["completed_cases"], 3)
            self.assertEqual({n: v["received"] for n, v in summary["configs"].items()},
                             {"no_rag": 3, "raw_history_rag": 3, "L1": 3, "L4": 3})
            self.assertEqual(len(transport.sent), 12)
            for body in transport.sent:
                self.assertEqual([c["chunk_id"] for c in block(body)["cbt_knowledge"]], ["fx-1", "fx-2"])
            packets, mapping, _ = union_run.quality_packets(root / "union", fixture_only=True)
            self.assertEqual(sorted({m["arm"] for m in mapping}), ["L1", "L4", "raw_history_rag"])
            self.assertTrue(all(m["status"] == "received" for m in mapping))

    def test_old_knowledge_free_runs_stay_readable(self):
        protocol = {"code_sha256": dict(union_run.code_hashes()), "knowledge": None}
        key = "evaluation/cbt-memory-effect/architecture_union/union_requests.py"
        protocol["code_sha256"][key] = next(iter(union_run.KNOWLEDGE_FREE_EQUIVALENT[key]))
        self.assertTrue(union_run.generation_compatible(protocol))
        protocol["knowledge"] = {"packets_sha256": "x"}
        self.assertFalse(union_run.generation_compatible(protocol))
        protocol = {"code_sha256": {**union_run.code_hashes(), key: "0" * 64}}
        self.assertFalse(union_run.generation_compatible(protocol))


if __name__ == "__main__":
    unittest.main()
