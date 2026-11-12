"""One invented case and tiny SQLite fixtures; no model/tokenizer/API/dataset."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "evaluation/cbt-memory-effect"
sys.path.insert(0, str(HERE))
import question_retrieval_policy_v4 as P

SPEC = importlib.util.spec_from_file_location("neutral_policy_product_fixture", HERE / "controlled_open_memory.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class QuestionRetrievalPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source_prepared/before"
        self.output = self.root / "prepared/before"

    def source_fixture(self):
        M.prepare(self.source, fixture_only=True, case_limit=1)
        return P.reader.read_rows(self.source / "requests.jsonl")

    def render(self):
        source_rows = self.source_fixture()
        databases = {Path(row["database_path"]) for row in source_rows}
        original_files = P._manifest(self.source, databases)
        with sqlite3.connect(next(iter(databases))) as connection:
            original_identity = connection.execute("SELECT * FROM dm_storage_identity").fetchall()
        result = P.render_prepared(self.source, self.output, source_gates=M.make_gates(self.source), fixture_only=True)
        protocol, rows = P.validate_prepared(self.output, source_prepared=self.source)
        self.assertEqual(P._manifest(self.source, databases), original_files)
        with sqlite3.connect(next(iter(databases))) as connection:
            self.assertEqual(connection.execute("SELECT * FROM dm_storage_identity").fetchall(), original_identity)
        return source_rows, protocol, rows, result

    def test_declared_headers_and_redundant_fields_change_quotes_stay_exact(self):
        original = self.source_fixture()
        for row in original:
            source = row["request_payload"]
            effective = P.transform_payload(source)
            self.assertNotEqual(P.reader.digest(source), P.reader.digest(effective))
            self.assertEqual({k: v for k, v in source.items() if k != "messages"},
                {k: v for k, v in effective.items() if k != "messages"})
            self.assertEqual(source["messages"][0]["content"].split("\nAs of: ", 1)[1],
                effective["messages"][0]["content"].split("\nAs of: ", 1)[1])
            old_body = source["messages"][1]["content"].split("\n", 1)[1]
            new_body = effective["messages"][1]["content"].split("\n", 1)[1]
            old_claims = json.loads(old_body)["claims"]
            new_claims = json.loads(new_body)["claims"]
            self.assertEqual([x for x in old_claims if x["field"] in ("history_quote", "source_quote")],
                [x for x in new_claims if x["field"] in ("history_quote", "source_quote")])
            self.assertTrue(all(claim in old_claims for claim in new_claims))
            if row["arm"] == "raw_history_rag":
                self.assertEqual(old_body, new_body)
            self.assertEqual(source["messages"][2:], effective["messages"][2:])
            self.assertEqual(len([m for m in effective["messages"] if m["role"] != "system"]), 7)
            self.assertLessEqual(len(P.serialize(effective)), 6000)
            changed = deepcopy(effective)
            changed["messages"][-1]["content"] = "An unallowed different question"
            with self.assertRaisesRegex(ValueError, "only_declared_common_headers"):
                P.assert_payload_mapping(source, changed)

    def test_unknown_headers_and_complete_request_overflow_fail_without_repacking(self):
        source = self.source_fixture()[0]["request_payload"]
        changed = deepcopy(source)
        changed["messages"][0]["content"] = "Unknown prefix\nAs of: unchanged\nRecent original source map: []"
        with self.assertRaisesRegex(ValueError, "unexpected_legacy_first"):
            P.transform_payload(changed)
        changed = deepcopy(source)
        changed["messages"][1]["content"] = "Unknown memory header\n{}"
        with self.assertRaisesRegex(ValueError, "unexpected_legacy_memory"):
            P.transform_payload(changed)
        changed = deepcopy(source)
        changed["messages"][-1]["content"] = "x" * 6000
        with self.assertRaisesRegex(ValueError, "6000_no_repacking"):
            P.transform_payload(changed)

    def test_render_preserves_static_source_real_product_and_original_writer_identity(self):
        original_rows, protocol, rows, result = self.render()
        source_protocol = P.reader.read_json(self.source / "protocol.json")
        self.assertEqual(protocol["code_identity"], M.identity())
        self.assertEqual(protocol["code_identity"], source_protocol["code_identity"])
        self.assertEqual(result["authoritative_database_count"], 1)
        self.assertEqual(result["new_databases_created_by_renderer"], 0)
        self.assertEqual(protocol["case_file_sha256"], source_protocol["case_file_sha256"])
        self.assertEqual(protocol["targets_file_sha256"], source_protocol["targets_file_sha256"])
        self.assertEqual(protocol["request_mapping_sha256"], P.reader.digest(protocol["request_mappings"]))
        for old, row, mapping in zip(original_rows, rows, protocol["request_mappings"]):
            self.assertEqual(old["database_path"], row["database_path"])
            self.assertTrue(Path(row["database_path"]).is_relative_to(self.source))
            self.assertFalse(list(self.output.rglob("*.sqlite")))
            self.assertEqual(mapping["source_request_sha256"], old["request_sha256"])
            self.assertEqual(mapping["effective_request_sha256"], row["request_sha256"])
            old_case = P.reader.read_json(old["prepared_case_path"])
            case = P.reader.read_json(row["prepared_case_path"])
            old_binding = old_case["contexts"][old["arm"]]["binding"]
            binding = case["contexts"][row["arm"]]["binding"]
            self.assertEqual(binding, {**old_binding, "request_sha256": row["request_sha256"]})
            with sqlite3.connect(row["database_path"]) as connection:
                P._snapshot(connection, old_binding)
                P._snapshot(connection, binding)
            with sqlite3.connect(old["database_path"]) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM dm_attempts").fetchone()[0], 0)
                self.assertIsNotNone(connection.execute("SELECT binding_json FROM open_context_snapshots WHERE request_sha256=?",
                    (row["request_sha256"],)).fetchone())

    def test_actual_gates_accept_new_binding_and_completed_admissions_validate(self):
        original_rows, protocol, rows, result = self.render()
        gates = M.make_gates(self.output)
        self.assertEqual(gates.identity(), protocol["code_identity"])
        for row in rows:
            binding = gates.preflight(row)
            self.assertEqual(binding["request_sha256"], row["request_sha256"])
            admission = gates.begin(row)
            self.assertFalse(admission["replayed"])
            receipt = gates.finish(row, admission, outcome="success", response=None, usage={"total_tokens": 1})
            self.assertTrue(receipt["publishable"])
        # Validation must remain possible after real admissions in the new DB.
        self.assertEqual(P.validate_prepared(self.output)[1], rows)
        with sqlite3.connect(original_rows[0]["database_path"]) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM dm_attempts").fetchone()[0], 2)

    def test_new_binding_preserves_unknown_failure_and_withdrawal_gates(self):
        _, _, rows, _ = self.render()
        gates = M.make_gates(self.output)
        unknown = gates.begin(rows[0])
        with self.assertRaises(M.loadlegacy()._modules().AttemptUnknown):
            gates.begin(rows[0])
        admission = gates.begin(rows[1])
        prepared, context = gates._prepared(rows[1])
        store = gates._store(prepared)
        try:
            store.revoke(context["binding"]["user_id"], "source", context["binding"]["current_source_id"])
        finally:
            store.close()
        with self.assertRaisesRegex(ValueError, "generation changed"):
            gates.preflight(rows[1])
        receipt = gates.finish(rows[1], admission, outcome="success", response={"text": "must discard"}, usage=None)
        self.assertFalse(receipt["publishable"])
        self.assertIsNone(receipt["response"])
        self.assertFalse(unknown["replayed"])

    def test_validation_rejects_rehashed_extra_case_change_and_source_change(self):
        _, _, rows, _ = self.render()
        path = Path(rows[0]["prepared_case_path"])
        prepared = P.reader.read_json(path)
        prepared["contexts"][rows[0]["arm"]]["source_ids"] = []
        path.write_text(P.reader.canonical(prepared) + "\n")
        protocol = P.reader.read_json(self.output / "protocol.json")
        protocol["prepared_files_sha256"][str(path.relative_to(self.output))] = P.reader.file_hash(path)
        (self.output / "protocol.json").write_text(P.reader.canonical(protocol) + "\n")
        with self.assertRaisesRegex(ValueError, "effective_case_has_unallowed"):
            P.validate_prepared(self.output)
        (self.source / "generation_cases.json").write_text("[]\n")
        with self.assertRaisesRegex(ValueError, "source_preparation_changed|source_snapshot_identity_changed"):
            P.validate_prepared(self.output)

    def test_existing_source_admission_cannot_be_reset_by_rendering(self):
        rows = self.source_fixture()
        M.make_gates(self.source).begin(rows[0])
        with self.assertRaisesRegex(RuntimeError, "no_model_admissions"):
            P.render_prepared(self.source, self.output, source_gates=M.make_gates(self.source), fixture_only=True)
        self.assertFalse(self.output.exists())


class PackedClaimSelectionTests(unittest.TestCase):
    @staticmethod
    def claim(ident, field, value, text, *, source="s-1", status="rule_verified", role="user"):
        return {"bundle_id": "b-1", "task_instance_id": "t-1", "claim_id": ident,
            "field": field, "value": value, "origin": "explicit_user_statement",
            "support_status": status, "evidence": [{"source_id": source, "span_id": "span-1",
                "text": text, "start": 0, "end": len(text), "source_time": None, "role": role}]}

    def fixture(self):
        progress = "I completed 3 reflection cards."
        outcome = "The outcome of reflection cards is mixed."
        return [self.claim("q1", "history_quote", progress, progress, status="source_quote"),
            self.claim("q2", "history_quote", outcome, outcome, source="s-2", status="source_quote"),
            self.claim("count", "reported_count", 3, progress),
            self.claim("outcome", "outcome", "mixed", outcome, source="s-2"),
            self.claim("unknown", "target_count", None, progress),
            self.claim("paused", "engagement", "paused", progress)]

    def select(self, question, claims):
        body = json.dumps({"claims": claims}, ensure_ascii=False, sort_keys=True, allow_nan=False)
        selected, audit = P.select_packed_claims(question, body)
        return json.loads(selected)["claims"], audit, body, selected

    def test_outcome_withholds_only_covered_count_all_quotes_controls_unknown_exact(self):
        original = self.fixture()
        kept, audit, _, _ = self.select("What outcome have I reported for reflection cards?", original)
        self.assertEqual(audit["question_field"], "outcome")
        self.assertEqual(audit["omitted_claim_ids"], ["count"])
        self.assertEqual(kept, [x for x in original if x["claim_id"] != "count"])
        self.assertEqual([x for x in kept if x["field"] == "history_quote"], original[:2])

    def test_unique_uncovered_source_is_never_removed(self):
        original = self.fixture()
        original[2]["evidence"][0]["source_id"] = "not-retained-source"
        kept, audit, before, after = self.select("What outcome have I reported for reflection cards?", original)
        self.assertEqual(audit["omitted_claim_ids"], [])
        self.assertEqual(audit["retained_for_provenance"], ["count"])
        self.assertEqual(before, after)
        self.assertEqual(kept, original)

    def test_multiaxis_and_unknown_question_are_byte_exact_noops(self):
        original = self.fixture()
        for question in ("What is my target count and what outcome have I reported?", "Tell me about reflection cards."):
            kept, audit, before, after = self.select(question, original)
            self.assertIsNone(audit["question_field"])
            self.assertEqual(kept, original)
            self.assertEqual(before, after)

    def test_wrong_role_or_altered_quotation_cannot_cover_omitted_evidence(self):
        for mutation in ("role", "text"):
            original = self.fixture()
            if mutation == "role":
                original[0]["evidence"][0]["role"] = "assistant"
            else:
                original[0]["value"] = "Different quotation"
            kept, audit, _, _ = self.select("What outcome have I reported for reflection cards?", original)
            self.assertIn("count", [x["claim_id"] for x in kept])
            self.assertEqual(audit["omitted_claim_ids"], [])

    def test_duplicate_claim_and_ambiguous_json_key_are_rejected(self):
        original = self.fixture()
        with self.assertRaisesRegex(ValueError, "unique_original"):
            self.select("What outcome have I reported?", original + [original[0]])
        with self.assertRaisesRegex(ValueError, "duplicate_packed"):
            P.select_packed_claims("What outcome have I reported?", '{"claims":[],"claims":[]}')


if __name__ == "__main__":
    unittest.main()
