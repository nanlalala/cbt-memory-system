"""Synthetic metadata-only fixtures; no models, weights, study data or exports."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch


PATH = Path(__file__).resolve().parents[1] / "evaluation/cbt-memory-effect/export_constrained_study_checkpoint_colab.py"
spec = importlib.util.spec_from_file_location("export_constrained_checkpoint_tested", PATH)
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def encoded(value):
    return (r.canonical(value) + "\n").encode()


def journal(rows):
    return b"".join(encoded({**row, "record_sha256": r.sha(r.canonical(row).encode())}) for row in rows)


def fixture(*, pilot_correct=2, main=False, fail=False, open_intent=False):
    protocol = {"version": r.VERSION, "study_root": str(r.ROOT), "maximum_new_study_attempts": 61,
                "planned_pilot_attempts": 4, "planned_main_unique_attempts": 57}
    files = {"study_protocol.json": encoded(protocol), "pilot_prepared/protocol.json": b"{}\n"}
    setups = [{"event": "model_reload_intent", "setup_id": "fixture"},
              {"event": "model_reload_completed", "setup_id": "fixture"}]
    files["model_load_events.jsonl"] = journal(setups)
    ledger, caches, requests, answers_by_section = [], {}, {}, {}
    sections = [("pilot_answers", list(range(4)))]
    if main:
        sections += [("answers_before", list(range(4, 52))),
                     ("answers_after", list(range(52, 61)) + list(range(4, 43)))]
    next_attempt = 1
    for section, indexes in sections:
        planned = []
        for index in indexes:
            payload = {"messages": [{"role": "user", "content": "synthetic fixture " + str(index)}]}
            planned.append({"case_id": "fixture-" + str(index), "arm": "fixture_arm", "decode_mode": "prefill_v2",
                            "request_payload": payload, "request_sha256": r.sha(r.canonical(payload).encode())})
        requests[section] = planned
        run = {"study_protocol_sha256": r.sha(files["study_protocol.json"]),
               "rows_sha256": r.sha(r.canonical(planned).encode()),
               "actual_tensor_preflight": [{"input_tensor_sha256": "synthetic", "input_tokens": 1}] * len(planned)}
        run_sha = r.sha(r.canonical(run).encode())
        logical = []
        for index, wanted in zip(indexes, planned):
            key = "fixture-key-" + str(index)
            if key not in caches:
                text = '{"answer": null}'
                cache = {"canonical_key": key, "response_text": text, "response_sha256": r.sha(text.encode()),
                         "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                         "finish_reason": "eos", "input_tensor_sha256": "synthetic"}
                cache_name = section + "/canonical/" + key + ".json"
                files[cache_name] = encoded(cache)
                ledger.append({"event": "generation_intent", "canonical_key": key,
                               "attempt_number": next_attempt, "request_sha256": wanted["request_sha256"]})
                next_attempt += 1
                ledger.append({"event": "generation_outcome", "canonical_key": key,
                    "status": "generation_failed_or_unknown" if fail and index == 3 else "published",
                    "cache_path": str(r.ROOT / cache_name), "cache_sha256": r.sha(files[cache_name]),
                    **{k: cache[k] for k in ("response_sha256", "usage", "finish_reason")}})
                caches[key] = cache
            logical.append({**caches[key], "case_id": wanted["case_id"], "arm": wanted["arm"],
                            "decode_mode": wanted["decode_mode"], "request_sha256": wanted["request_sha256"],
                            "run_protocol_sha256": run_sha, "status": "completed"})
        answers_by_section[section] = logical
        files[section + "/run_protocol.json"] = encoded(run)
        files[section + "/answers_unscored.jsonl"] = journal(logical)
        files[section + "/completion_summary.json"] = encoded({"complete": True,
            "logical_completed": len(planned), "answers_sha256": r.sha(files[section + "/answers_unscored.jsonl"]),
            "run_protocol_sha256": run_sha})
        if section == "pilot_answers":
            files["pilot_prepared/requests.json"] = encoded(planned)
        else:
            files["prepared/" + section.removeprefix("answers_") + "/requests.jsonl"] = b"".join(encoded(row) for row in planned)
    files["pilot_answers/pilot_score.json"] = encoded({"prefill_schema_valid": 2,
        "prefill_basic_readability_correct": pilot_correct, "prefill_denominator": 2,
        "ready_for_main": pilot_correct == 2,
        "prepared_protocol_sha256": r.sha(files["pilot_prepared/protocol.json"]),
        "rows": [{k: row[k] for k in ("case_id", "decode_mode", "response_sha256")} for row in answers_by_section["pilot_answers"]]})
    if main:
        files["main_freeze.json"] = encoded({"study_protocol_sha256": r.sha(files["study_protocol.json"])})
        for name in ("before_strict", "after_strict", "resources", "summary"):
            files["reports/" + name + ".json"] = encoded({"future_report_schema": "preserved raw"})
    if open_intent:
        ledger.pop()
    files["generation_ledger.jsonl"] = journal(ledger)
    return files


class ExportMetadataFixtures(unittest.TestCase):
    def test_hosted_guard_precedes_io(self):
        with patch.dict(r.os.environ, {}, clear=True), patch.object(r, "discover") as discover:
            with self.assertRaisesRegex(RuntimeError, "genuine_hosted_colab"):
                r.export("/content/fixture.zip", "/content/fixture.json", model_reacquisition_sha256="0" * 64)
            discover.assert_not_called()

    def test_fixed_root_allowlist_and_safe_outputs(self):
        for value in ("../x.json", "/x.json", "a//x.json", "a/./x.json", "a\\x.json", ".git/config.json", "x.safetensors"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                r.safe_member(value)
        for value in ("/content/.git/config.zip", "/content/nested/archive.zip", "/tmp/archive.zip", "/content/.hidden.zip"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                r.output_path(value, ".zip")
        self.assertEqual(r.output_path("/content/checkpoint-v1.zip", ".zip").name, "checkpoint-v1.zip")

    def test_bootstrap_python_is_only_recovery_exception(self):
        with self.assertRaises(ValueError):
            r.safe_member("study/bootstrap.py")
        self.assertEqual(str(r.safe_member("recovery/bootstrap.py", recovery=True)), "recovery/bootstrap.py")

    def test_optional_v3_transport_evidence_is_explicitly_allowlisted(self):
        paths = {str(path) for path in r.OPTIONAL_RECOVERY}
        self.assertTrue({
            "/content/cbt-prefill-recovery-bootstrap-20261112-v3.py",
            "/content/cbt-prefill-recovery-bootstrap-20261112-v3.log",
            "/content/cbt-model-download-failure-20261112-v1.json",
        }.issubset(paths))
        self.assertEqual(r.BOOTSTRAP_SHA, "626c7d7877283897ba0c0d9e8b327171405616830b52f8ac5b8af18bc94ca25c")
        self.assertEqual(r.OPTIONAL_RECOVERY_SOURCE_SHA256[
            "/content/cbt-prefill-recovery-bootstrap-20261112-v3.py"],
            "4a643e1bb050c06a6d6ca41e2374f057981c98a795198ceab05044849b631808")

    def test_optional_source_is_pinned_but_worker_log_hash_is_observed(self):
        source = "/content/cbt-prefill-recovery-bootstrap-20261112-v3.py"
        with patch.dict(r.OPTIONAL_RECOVERY_SOURCE_SHA256, {source: r.sha(b"synthetic source")}, clear=True):
            r.check_optional_recovery_source(source, b"synthetic source")
            with self.assertRaisesRegex(ValueError, "pinned_optional"):
                r.check_optional_recovery_source(source, b"changed synthetic source")
        r.check_optional_recovery_source("/content/cbt-prefill-recovery-bootstrap-20261112-v3.log", b"still growing log")
        r.check_optional_recovery_source("/content/cbt-model-download-failure-20261112-v1.json", b"{}")

    def test_four_passed_pilots_are_settled_but_not_final(self):
        state = r.inspect_state(fixture())
        self.assertEqual(state["stage"], "settled_incomplete_study")
        self.assertTrue(state["all_intents_published"])
        self.assertFalse(state["terminal"])
        self.assertEqual(state["new_attempts"], 4)

    def test_readability_gate_rejection_is_terminal_without_main(self):
        state = r.inspect_state(fixture(pilot_correct=1))
        self.assertEqual(state["stage"], "pilot_gate_rejected_settled")
        self.assertTrue(state["terminal"])
        self.assertFalse(state["pilot_gate"]["passed"])

    def test_failed_and_unknown_calls_never_become_final(self):
        for flags in ({"fail": True}, {"open_intent": True}):
            with self.subTest(flags=flags):
                state = r.inspect_state(fixture(**flags))
                self.assertEqual(state["stage"], "live_partial")
                self.assertFalse(state["terminal"])
                self.assertTrue(state["issues"])

    def test_complete_61_calls_and_96_rows_are_terminal(self):
        state = r.inspect_state(fixture(main=True))
        self.assertEqual(state["stage"], "completed_study_settled")
        self.assertTrue(state["terminal"])
        self.assertEqual(state["new_attempts"], 61)

    def test_reports_remain_schema_independent_raw_evidence(self):
        files = fixture(main=True)
        files["reports/summary.json"] = b'{"a_future_schema":true}\n'
        self.assertEqual(r.inspect_state(files)["stage"], "completed_study_settled")

    def test_main_activity_after_gate_rejection_is_partial(self):
        state = r.inspect_state(fixture(pilot_correct=1, main=True))
        self.assertEqual(state["stage"], "live_partial")
        self.assertFalse(state["terminal"])

    def test_raw_terminal_cache_tampering_is_detected(self):
        files = fixture()
        name = next(name for name in files if "/canonical/" in name)
        files[name] += b" "
        state = r.inspect_state(files)
        self.assertEqual(state["stage"], "live_partial")
        self.assertIn("hash_changed", state["issues"][0])

    def test_unfinished_setup_is_partial(self):
        files = fixture()
        files["model_load_events.jsonl"] = journal([{"event": "model_reload_intent", "setup_id": "open"}])
        self.assertEqual(r.inspect_state(files)["stage"], "live_partial")

    def test_pilot_report_must_bind_preparation(self):
        files = fixture(pilot_correct=1)
        files["pilot_prepared/protocol.json"] = b'{"changed":true}\n'
        self.assertEqual(r.inspect_state(files)["stage"], "live_partial")

    def test_empty_inherited_identity_cannot_be_verified(self):
        with self.assertRaisesRegex(ValueError, "exact_three"):
            r.reference_identity(fixture())

    def test_duplicate_json_keys_and_broken_journal_checksums_fail(self):
        with self.assertRaises(ValueError):
            r.parse(b'{"x":1,"x":2}')
        with self.assertRaisesRegex(ValueError, "checksum"):
            r.checked_rows(journal([{"event": "fixture"}]).replace(b'"fixture"', b'"changed"'))


if __name__ == "__main__":
    unittest.main()
