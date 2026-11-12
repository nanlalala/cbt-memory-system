"""Synthetic metadata fixtures only: no real study files, DBs, models or export."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


PATH = Path(__file__).resolve().parents[1] / "evaluation/cbt-memory-effect/export_unconstrained_state_checkpoint_v3_colab.py"
spec = importlib.util.spec_from_file_location("export_unconstrained_v3_tested", PATH)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def enc(value):
    return (r.canonical(value) + "\n").encode()


def journal(rows):
    return b"".join(enc({**row, "record_sha256": r.digest(row)}) for row in rows)


def synthetic():
    files = {name: enc({"synthetic_identity": name}) for name in
             ("load_protocol.json", "model_files.json", "model_load_receipt.json")}
    adapter_policy = {"domain": {"null_allowed": True, "integer_minimum": 0, "integer_maximum": 9999,
                                 "string_enum": ["unchanged"]}}
    adapter = {"code_sha256": r.FROZEN_CODE_SHA256["common_answer_adapter_v3.py"],
               "policy": adapter_policy, "policy_sha256": r.digest(adapter_policy)}
    protocol = {"version": r.VERSION, "study_root": str(r.ROOT), "maximum_new_study_attempts": 57,
        "planned_pilot_attempts": 0, "planned_main_unique_attempts": 57, "planned_main_logical_rows": 96,
        "failed_v2_gate_unchanged_and_closed": True, "decoding_policy": r.DECODING,
        "output_adapter_identity": adapter,
        "code_sha256": {"/content/fake-helper/evaluation/cbt-memory-effect/" + name: digest
                        for name, digest in r.FROZEN_CODE_SHA256.items()},
        "inherited_identity_sha256": {name: r.sha(data) for name, data in files.items()}}
    files["study_protocol.json"] = enc(protocol)
    identity = {key: r.sha(files[name]) for name, key in (("load_protocol.json", "load_protocol_sha256"),
        ("model_files.json", "model_files_sha256"), ("model_load_receipt.json", "load_receipt_sha256"))}
    sections, planned_by_side = {}, {}
    for side in ("before", "after"):
        source, fresh = "source_prepared/" + side, "prepared/" + side
        rows, source_rows = [], []
        for index in range(48):
            payload_index = index if side == "before" or index not in range(31, 48, 2) else 48 + (index - 31) // 2
            payload = {"messages": [{"role": "user", "content": "synthetic " + str(payload_index)}], "max_tokens": 64}
            row = {"case_id": "case-" + str(index // 2),
                   "arm": "raw_history_rag" if index % 2 == 0 else "managed_memory_rag",
                   "request_payload": payload, "request_sha256": r.digest(payload),
                   "database_path": str(r.ROOT / source / "cases" / str(index // 2) / "state.sqlite")}
            rows.append(row)
            source_rows.append({**row, "request_payload": {"synthetic_source": payload_index}})
            files[source + "/cases/" + str(index // 2) + "/state.sqlite"] = b"synthetic snapshot bytes; not SQLite"
            files[source + "/cases/" + str(index // 2) + "/state.sqlite.control.json"] = enc({"live": True})
        for row in source_rows:
            row["request_sha256"] = r.digest(row["request_payload"])
        planned_by_side[side] = rows
        files[source + "/requests.jsonl"] = b"".join(enc(row) for row in source_rows)
        files[fresh + "/requests.jsonl"] = b"".join(enc(row) for row in rows)
        files[fresh + "/generation_cases.json"] = files[source + "/generation_cases.json"] = enc({"synthetic_cases": 24})
        files[fresh + "/assessor_only/targets.json"] = files[source + "/assessor_only/targets.json"] = enc({"synthetic_labels": True})
        files[source + "/protocol.json"] = enc({"synthetic_source_protocol": side})
        files[source + "/cases/case.json"] = enc({"static": side})
        files[fresh + "/cases/case.json"] = enc({"effective_static": side})
        mappings = [{"case_id": row["case_id"], "arm": row["arm"], "synthetic_mapping": True} for row in rows]
        common_policy = {"synthetic_common_policy": True}
        prepared = {"renderer_code_sha256": r.FROZEN_CODE_SHA256["common_retrieval_policy_v1.py"],
            "common_policy": common_policy, "common_policy_sha256": r.digest(common_policy),
            "request_mappings": mappings, "request_mapping_sha256": r.digest(mappings),
            "source_files_sha256": {name.removeprefix(source + "/"): r.sha(data) for name, data in files.items()
                                    if name.startswith(source + "/") and not name.endswith((".sqlite", ".sqlite.control.json"))},
            "prepared_files_sha256": {"cases/case.json": r.sha(files[fresh + "/cases/case.json"])}}
        prepared["source_snapshot_sha256"] = r.digest(prepared["source_files_sha256"])
        files[fresh + "/protocol.json"] = enc(prepared)
        section = {"checkout": "/content/synthetic-checkout-" + side, "fresh_prepared": str(r.ROOT / fresh),
            "source_prepared": str(r.ROOT / source), "new_database_count": 24,
            "source_code_identity": {"synthetic_source": side},
            **{key: prepared[key] for key in ("renderer_code_sha256", "common_policy_sha256", "request_mapping_sha256")}}
        for name, key in ((source + "/protocol.json", "source_protocol_sha256"),
                          (source + "/requests.jsonl", "source_requests_sha256"),
                          (fresh + "/protocol.json", "fresh_protocol_sha256"),
                          (fresh + "/requests.jsonl", "requests_sha256"),
                          (fresh + "/generation_cases.json", "case_file_sha256"),
                          (fresh + "/assessor_only/targets.json", "targets_file_sha256")):
            section[key] = r.sha(files[name])
        audit = {"version": r.VERSION + "-source-gate-audit", "side": side, **section,
            "study_protocol_sha256": r.sha(files["study_protocol.json"]),
            "main_stage_code_sha256": r.FROZEN_CODE_SHA256["unconstrained_state_comparison_v3_colab.py"],
            "all_preflight_passed": True, "checked_row_count": 48, "new_model_calls": 0,
            "generation_admissions": 0, "automatic_retry": False,
            "ordered_checks": [{**{k: row[k] for k in ("case_id", "arm", "request_sha256")},
                                "source_binding_sha256": "a" * 64} for row in rows]}
        files["source_gate_audits/" + side + ".json"] = enc(audit)
        section.update(source_gate_audit_path=str(r.ROOT / "source_gate_audits" / (side + ".json")),
                       source_gate_audit_sha256=r.sha(files["source_gate_audits/" + side + ".json"]))
        sections[side] = section
    freeze = {"version": r.VERSION, "study_protocol_sha256": r.sha(files["study_protocol.json"]),
        "main_stage_code_sha256": r.FROZEN_CODE_SHA256["unconstrained_state_comparison_v3_colab.py"],
        "output_adapter_identity": adapter, "decoding_identity": r.DECODING,
        "main_logical_rows": 96, "main_unique_payloads": 57, "total_study_cap": 57,
        "no_new_pilot_claim": True, "source_databases_are_live_mutable_authorities": True,
        "live_database_relocation": False, "sections": sections}
    files["main_freeze.json"] = enc(freeze)
    events, intents, outcomes, caches, logical = [], {}, {}, {}, []
    for side in ("before", "after"):
        events.extend([{"event": "model_reload_intent", "setup_id": side},
                       {"event": "model_reload_completed", "setup_id": side, "setup_seconds": 1.0, "model_identity": identity}])
        section, directory, rows = sections[side], "answers_" + side, planned_by_side[side]
        metadata = {"side": side, "main_freeze_sha256": r.sha(files["main_freeze.json"]),
            "fresh_prepared_protocol_sha256": section["fresh_protocol_sha256"],
            **{key: section[key] for key in ("renderer_code_sha256", "common_policy_sha256", "request_mapping_sha256")},
            "all_cases_exposed_development_only": True}
        run = {"rows_sha256": r.digest(rows), "study_protocol_sha256": r.sha(files["study_protocol.json"]),
            "code_sha256": protocol["code_sha256"], "decoding_identity": r.DECODING,
            "output_adapter_identity": adapter, "metadata": metadata, "model": identity,
            "gate_identity": section["source_code_identity"],
            "actual_tensor_preflight": [{"input_tensor_sha256": "synthetic-input", "input_tokens": 10}] * 48}
        files[directory + "/run_protocol.json"] = enc(run)
        answers = []
        for row in rows:
            key = r.digest({"new_study_protocol_sha256": r.sha(files["study_protocol.json"]),
                "model": identity, "decoding": r.DECODING, "payload": row["request_payload"],
                "max_new_tokens": 64, "do_sample": False})
            if key not in caches:
                cache = {"canonical_key": key, "response_text": "11", "response_sha256": r.sha(b"11"),
                         "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                         "finish_reason": "eos", "input_tensor_sha256": "synthetic-input"}
                name = directory + "/canonical/" + key + ".json"
                files[name] = enc(cache)
                intents[key] = {"event": "generation_intent", "canonical_key": key, "attempt_number": len(intents) + 1,
                    "request_sha256": row["request_sha256"], "run_protocol_sha256": r.digest(run), "output": str(r.ROOT / directory)}
                outcomes[key] = {"event": "generation_outcome", "canonical_key": key, "status": "published",
                    "cache_path": str(r.ROOT / name), "cache_sha256": r.sha(files[name]),
                    **{k: cache[k] for k in ("response_sha256", "usage", "finish_reason")}, "generation_wall_seconds": 0.01}
                caches[key] = cache
            adapted = {"accepted": True, "raw_response_text": "11", "raw_response_sha256": r.sha(b"11"),
                       "canonical_response_text": '{"answer":11}', "canonical_response_sha256": r.sha(b'{"answer":11}'),
                       "rule": "bare_integer", "reason": "accepted"}
            answers.append({**caches[key], "case_id": row["case_id"], "arm": row["arm"],
                "request_sha256": row["request_sha256"], "run_protocol_sha256": r.digest(run),
                "status": "completed", "source_checks_passed": True,
                "answer_adapter": adapted, "output_adaptation_wall_seconds": 0.001})
        files[directory + "/answers_unscored.jsonl"] = journal(answers)
        files[directory + "/completion_summary.json"] = enc({"complete": True, "logical_completed": 48,
            "logical_planned": 48, "answers_sha256": r.sha(files[directory + "/answers_unscored.jsonl"]),
            "run_protocol_sha256": r.digest(run)})
        files["reports/" + side + "_metrics.json"] = enc({"answers_sha256": r.sha(files[directory + "/answers_unscored.jsonl"]),
            "main_freeze_sha256": r.sha(files["main_freeze.json"]), "output_adapter_identity": adapter})
        logical.extend(answers)
    files["generation_ledger.jsonl"] = journal([row for key in intents for row in (intents[key], outcomes[key])])
    files["model_load_events.jsonl"] = journal(events)
    files["reports/summary.json"] = enc({"version": r.VERSION, "main_logical_rows": 96, "main_unique_generations": 57,
        "pilot_generations": 0, "new_study_unique_generations": 57, "study_cap": 57})
    files["reports/resources.json"] = enc({"new_study_unique_generation_cost": {
        "observed_tokens": {k: sum(o["usage"][k] for o in outcomes.values()) for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
        "unique_admitted_attempts": 57, "answer_path_wall_seconds_sum": sum(o["generation_wall_seconds"] for o in outcomes.values())},
        "output_adaptation_wall_seconds_sum": sum(a["output_adaptation_wall_seconds"] for a in logical),
        "actual_new_model_setup_events": r.checked_rows(files["model_load_events.jsonl"])})
    return files


class ExportV3Fixtures(unittest.TestCase):
    def inspect(self, files):
        with patch.object(r, "STUDY_PROTOCOL_SHA256", r.sha(files["study_protocol.json"])):
            return r.inspect_state(files)

    def test_complete57_and96_authenticated_answers(self):
        state = self.inspect(synthetic())
        self.assertEqual(state["issues"], [])
        self.assertEqual(state["stage"], "completed_study_settled")
        self.assertTrue(state["terminal"])
        self.assertEqual(state["new_attempts"], 57)
        self.assertEqual(state["logical_answers"], 96)
        self.assertEqual(state["observed_tokens"], {"prompt_tokens": 570, "completion_tokens": 114, "total_tokens": 684})

    def test_hosted_guard_before_any_io(self):
        with patch.dict(r.os.environ, {}, clear=True), patch.object(r, "discover") as discover:
            with self.assertRaisesRegex(RuntimeError, "genuine_hosted_colab"):
                r.export("/content/fixture.zip", "/content/fixture.json")
            discover.assert_not_called()

    def test_no_pilot_or_unknown_root_files(self):
        self.assertNotIn("pilot_prepared", r.SUBTREES)
        self.assertNotIn("pilot_answers", r.SUBTREES)
        self.assertNotIn("unknown.log", r.ROOT_FILES)
        self.assertTrue({"render_before.log", "render_after.log"}.issubset(r.ROOT_FILES))

    def test_optional_worker_artifacts_are_exactly_named(self):
        self.assertEqual(len(r.OPTIONAL_WORKER_ARTIFACTS), 9)
        names = {path.name for path in r.OPTIONAL_WORKER_ARTIFACTS}
        self.assertIn("cbt-unconstrained-v3-orchestrator-20261112-v1.marker.json", names)
        self.assertIn("cbt-unconstrained-v3-orchestrator-20261112-v1.journal.jsonl", names)
        self.assertIn("cbt-unconstrained-v3-orchestrator-20261112-v1.log", names)
        self.assertNotIn("credentials.json", names)
        self.assertNotIn("unknown.log", names)

    def test_traversal_weights_and_secret_paths_rejected(self):
        for value in ("../x.json", "/x.json", "a//x.json", "a\\x.json", ".git/config.json", "secrets/x.json", "model.safetensors"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                r.safe_member(value)
        for value in ("/content/.git/archive.zip", "/content/nested/archive.zip", "/tmp/archive.zip"):
            with self.assertRaises(ValueError):
                r.output_path(value, ".zip")

    def test_frozen_protocol_sha_is_mandatory(self):
        self.assertEqual(r.inspect_state(synthetic())["stage"], "live_partial")

    def test_failed_or_unmatched_journal_is_partial(self):
        for failure in (True, False):
            files = synthetic(); rows = r.checked_rows(files["generation_ledger.jsonl"])
            if failure: rows[-1]["status"] = "generation_failed_or_unknown"
            else: rows.pop()
            files["generation_ledger.jsonl"] = journal([{k:v for k,v in row.items() if k != "record_sha256"} for row in rows])
            self.assertFalse(self.inspect(files)["terminal"])

    def test_raw_cache_tampering_is_partial(self):
        files = synthetic(); name = next(n for n in files if "/canonical/" in n)
        files[name] += b" "
        self.assertIn("checksum", self.inspect(files)["issues"][0])

    def test_adapter_rejection_never_becomes_null(self):
        policy = {"policy": {"domain": {"null_allowed": True, "integer_minimum": 0, "integer_maximum": 9999, "string_enum": []}}}
        rejected = {"accepted": False, "raw_response_text": "prose", "raw_response_sha256": r.sha(b"prose"),
                    "canonical_response_text": None, "canonical_response_sha256": None, "rule": "adapter_rejected", "reason": "prose"}
        r.adapter_binding("prose", rejected, policy)
        rejected["canonical_response_text"] = '{"answer":null}'
        with self.assertRaises(ValueError):
            r.adapter_binding("prose", rejected, policy)

    def test_output_only_adapter_acceptance_and_value_are_independent(self):
        domain = {"null_allowed": True, "integer_minimum": 0, "integer_maximum": 9999, "string_enum": ["unchanged"]}
        for raw, text, rule in (("11", '{"answer":11}', "bare_integer"),
                                ("answer: null", '{"answer":null}', "answer_prefix_null"),
                                ('"unchanged"', '{"answer":"unchanged"}', "json_quoted_status"),
                                (' {"answer":11} ', '{"answer":11}', "single_answer_json")):
            self.assertEqual(r.adapter_expected(raw, domain), (True, text, rule))
        for raw in ('"null"', '01', '-0', '11.0', 'true', 'answer: {"answer":11}',
                    '{"answer":1,"answer":2}', 'answer: answer: 11', 'prose'):
            self.assertEqual(r.adapter_expected(raw, domain), (False, None, "adapter_rejected"))

    def test_static_hash_excludes_live_db_but_snapshots_still_required(self):
        files = synthetic(); db = next(n for n in files if n.endswith(".sqlite"))
        files[db] = b"different consistent synthetic live snapshot"
        self.assertTrue(self.inspect(files)["terminal"])
        del files[db]
        self.assertFalse(self.inspect(files)["terminal"])

    def test_source_audit_and_report_tampering_are_partial(self):
        for name in ("source_gate_audits/before.json", "reports/before_metrics.json", "reports/resources.json"):
            files = synthetic(); value = r.parse(files[name]); value["tampered"] = True
            if name.startswith("reports/"):
                value["answers_sha256"] = "0" * 64
                if name.endswith("resources.json"): value["new_study_unique_generation_cost"]["observed_tokens"]["total_tokens"] += 1
            files[name] = enc(value)
            self.assertFalse(self.inspect(files)["terminal"])

    def test_cumulative_bound_checked_before_full_capture(self):
        with patch.object(r, "MAX_BYTES", 10):
            self.assertEqual(r.bounded_size(4, 6), 10)
            with self.assertRaisesRegex(ValueError, "cumulative"):
                r.bounded_size(10, 1)

    def test_missing_historical_identity_cannot_be_verified(self):
        with self.assertRaisesRegex(ValueError, "thirteen"):
            r.reference_identity(synthetic())

    def test_writer_or_mutation_can_never_be_called_terminal(self):
        settled = {"stage": "completed_study_settled", "terminal": True, "issues": []}
        locks = {name: "held_shared_no_generation_or_setup_writer" for name in r.LOCKS}
        self.assertTrue(r.capture_observations(settled, locks, [], False)["terminal"])
        for current, changed, membership in (({**locks, ".generation.lock": "active_writer_live_partial"}, [], False),
                                              ({**locks, ".model_setup.lock": "absent"}, [], False),
                                              (locks, ["fixture.json"], False), (locks, [], True), ({}, [], False)):
            self.assertFalse(r.capture_observations(settled, current, changed, membership)["terminal"])
        self.assertTrue(settled["terminal"])

    def test_exact_five_code_files_and_thirteen_history_reads(self):
        files = synthetic(); protocol = r.parse(files["study_protocol.json"])
        source_bytes = {name: ("synthetic code " + name).encode() for name in r.FROZEN_CODE_SHA256}
        pins = {name: r.sha(data) for name, data in source_bytes.items()}
        protocol["code_sha256"] = {"/content/fake/evaluation/cbt-memory-effect/" + name: pins[name] for name in pins}
        names = {str(r.OLD_ROOT / name) for name in {"load_protocol.json", "model_files.json", "model_load_receipt.json", "generation_ledger.jsonl"}}
        names |= {str(r.FAILED_V2_ROOT / name) for name in r.V2_REFERENCES}
        protocol["historical_reference_sha256"] = {name: r.sha(b"historical fixture") for name in names}
        files["study_protocol.json"] = enc(protocol)
        def read(path):
            name = str(path)
            return b"historical fixture" if name in names else source_bytes[Path(name).name]
        with patch.object(r, "FROZEN_CODE_SHA256", pins), patch.object(r, "read_bytes", side_effect=read) as reader:
            evidence, code = r.reference_identity(files)
            self.assertEqual(reader.call_count, 18)
            self.assertEqual(len(evidence), 18)
            self.assertEqual(code, source_bytes)
        bad = dict(source_bytes); bad["common_answer_adapter_v3.py"] = b"changed"
        with patch.object(r, "FROZEN_CODE_SHA256", pins), patch.object(r, "read_bytes", side_effect=lambda path: b"historical fixture" if str(path) in names else bad[Path(path).name]):
            with self.assertRaisesRegex(ValueError, "frozen_v3_code_changed"):
                r.reference_identity(files)


if __name__ == "__main__":
    unittest.main()
