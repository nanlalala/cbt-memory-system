"""Restored-connection guards using inert fixtures; no credentials or network."""
import importlib.util
from contextlib import ExitStack, redirect_stdout
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "evaluation/architecture-rerun/cloud_continue_non_mab.py"
SPEC = importlib.util.spec_from_file_location("non_mab_recovered_driver_unit", SOURCE)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)

FIXTURE_ROOTS = {
    "content/cbt-v4-isolated-cbt-full/professional7",
    "content/cbt-v4-isolated-cbt-full/professional7_memory",
    "content/cbt-v4-isolated-cbt-full/professional7_quality",
    "content/cbt-v4-isolated-cbt-full/cactus600",
    "content/cbt-v4-isolated-cbt-full/cactus600_memory",
    "content/cbt-v4-repair-general/locomo1540",
}


def restoration_fixture(root):
    """Six invented inert files; does not create /content or set Colab flags."""
    base = root / "restored"
    entries = []
    for number, relative in enumerate(sorted(FIXTURE_ROOTS)):
        path = base / relative / "audit.json"
        path.parent.mkdir(parents=True)
        data = json.dumps({"invented_fixture_number":number}).encode()
        path.write_bytes(data)
        entries.append({"path":relative + "/audit.json", "bytes":len(data), "sha256":sha256(data).hexdigest()})
    manifest = {"version":"capstone-original-non-mab-source-restore-v1", "source_only":True,
        "model_calls":0, "mab_accessed":False, "credentials_included":False, "restore_base":"/content",
        "entries":entries, "absent_optional_roots":["content/cbt-v4-isolated-cbt-full/cactus600_quality",
        "content/cbt-v4-repair-general/locomo1540-judge"]}
    archive = root / "fixture.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for row in entries:
            output.writestr(row["path"], (base / row["path"]).read_bytes())
        output.writestr("restore_manifest.json", json.dumps(manifest))
    archive_sha = sha256(archive.read_bytes()).hexdigest()
    marker = root / "restoration.json"
    receipt = {"version":manifest["version"], "archive_sha256":archive_sha,
        "status":"restored_original_bytes_zero_model_calls", "moved_roots":sorted(FIXTURE_ROOTS),
        "files":len(entries), "model_calls":0, "mab_accessed":False}
    marker.write_text(json.dumps(receipt))
    content = base / "content"
    cbt, general = content / "cbt-v4-isolated-cbt-full", content / "cbt-v4-repair-general"
    sources = {"professional7":{"reader":str(cbt/"professional7"), "memory":str(cbt/"professional7_memory"),
        "old_quality":str(cbt/"professional7_quality")}, "cactus600":{"reader":str(cbt/"cactus600"),
        "memory":str(cbt/"cactus600_memory"), "old_quality":str(cbt/"cactus600_quality")},
        "locomo1540":{"reader":str(general/"locomo1540"), "old_quality":str(general/"locomo1540-judge")},
        "longmemeval500":None}
    return {"base":base, "archive":archive, "marker":marker, "receipt":receipt,
        "manifest":manifest, "sources":sources, "sha":archive_sha,
        "count":len(entries), "uncompressed":sum(row["bytes"] for row in entries)}


def fixture_constants(fixture):
    stack = ExitStack()
    stack.enter_context(patch.object(driver, "RESTORE_ARCHIVE_SHA256", fixture["sha"]))
    stack.enter_context(patch.object(driver, "RESTORE_FILE_COUNT", fixture["count"]))
    stack.enter_context(patch.object(driver, "RESTORE_UNCOMPRESSED_BYTES", fixture["uncompressed"]))
    stack.enter_context(patch.object(driver, "DEFAULT_ROOT", str(fixture["base"].parent / "old-continuation")))
    return stack


def provenance(fixture):
    return driver._restoration_provenance(fixture["archive"], fixture["marker"],
        fixture["sources"], restore_base=fixture["base"])


class RecoveredAdmissionTests(unittest.TestCase):
    def test_unhosted_recovery_stops_before_code_source_or_secret_access(self):
        with patch.object(driver, "_hosted", side_effect=RuntimeError("unit unhosted")), \
                patch.object(driver, "_checkout") as checkout, \
                patch.object(driver, "_restoration_provenance") as restore, \
                patch.object(driver, "_load_recovered_transports") as load:
            with self.assertRaises(RuntimeError):
                driver.run_recovered({}, repo="unused", expected_commit="unused", sources={},
                    source_restoration_receipt="unused", source_restore_archive="unused")
            checkout.assert_not_called()
            restore.assert_not_called()
            load.assert_not_called()

    def test_complete_inert_source_fixture_has_explicit_reconstructed_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = restoration_fixture(Path(tmp))
            with fixture_constants(fixture), patch.object(driver, "_load_recovered_transports") as load:
                result = provenance(fixture)
                self.assertEqual(result["origin"], "reconstructed_from_existing_colab_secret")
                self.assertEqual(result["restored_original_files_verified"], 6)
                self.assertEqual(result["verification_model_calls"], 0)
                self.assertFalse(result["mab_accessed"])
                load.assert_not_called()

    def test_restored_file_tamper_or_extra_file_is_rejected_before_secret(self):
        for extra in (False, True):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as tmp:
                fixture = restoration_fixture(Path(tmp))
                path = fixture["base"] / fixture["manifest"]["entries"][0]["path"]
                if extra:
                    path.with_name("extra.json").write_text('{"unexpected":true}')
                else:
                    path.write_text('{"modified_source":true}')
                with fixture_constants(fixture), patch.object(driver, "_load_recovered_transports") as load:
                    with self.assertRaises(ValueError): provenance(fixture)
                    load.assert_not_called()

    def test_source_field_mapping_cannot_be_swapped_between_reader_and_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = restoration_fixture(Path(tmp))
            fields = fixture["sources"]["professional7"]
            fields["reader"], fields["memory"] = fields["memory"], fields["reader"]
            with fixture_constants(fixture), patch.object(driver, "_load_recovered_transports") as load:
                with self.assertRaises(ValueError): provenance(fixture)
                load.assert_not_called()

    def test_archive_and_unfinished_restoration_receipt_are_rejected(self):
        for changed in ("archive", "receipt"):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as tmp:
                fixture = restoration_fixture(Path(tmp))
                if changed == "archive":
                    with fixture["archive"].open("ab") as stream: stream.write(b"changed archive")
                else:
                    marker = dict(fixture["receipt"], status="reserved_zero_model_calls")
                    fixture["marker"].write_text(json.dumps(marker))
                with fixture_constants(fixture), patch.object(driver, "_load_recovered_transports") as load:
                    with self.assertRaises(ValueError): provenance(fixture)
                    load.assert_not_called()

    def test_reconstructed_delegates_do_not_create_original_v5_globals_or_load_secret(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = restoration_fixture(Path(tmp))
            with fixture_constants(fixture), patch.object(driver, "_load_recovered_transports") as load:
                connection = driver._RecoveredConnection(REPO, provenance(fixture))
                namespace = {}
                delegates = driver._parents(namespace, connection)
                self.assertEqual(namespace, {})
                self.assertEqual(set(delegates), {driver.ANSWER_IDENTITY["model"], driver.JUDGE_IDENTITY["model"]})
                self.assertEqual(delegates[driver.ANSWER_IDENTITY["model"]].identity, driver.ANSWER_IDENTITY)
                load.assert_not_called()

    def test_inert_transport_factory_loads_once_only_at_first_send(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = restoration_fixture(Path(tmp))
            answer = Mock(identity=driver.ANSWER_IDENTITY, return_value={"fixture":"answer"})
            judge = Mock(identity=driver.JUDGE_IDENTITY, return_value={"fixture":"judge"})
            loaded = {driver.ANSWER_IDENTITY["model"]:answer, driver.JUDGE_IDENTITY["model"]:judge}
            with fixture_constants(fixture), patch.object(driver, "_load_recovered_transports", return_value=loaded) as load:
                connection = driver._RecoveredConnection(REPO, provenance(fixture))
                delegates = driver._parents({}, connection)
                load.assert_not_called()
                self.assertEqual(delegates[driver.ANSWER_IDENTITY["model"]]({"inert":"first"}), {"fixture":"answer"})
                self.assertEqual(delegates[driver.JUDGE_IDENTITY["model"]]({"inert":"second"}), {"fixture":"judge"})
                load.assert_called_once_with(REPO)
                answer.assert_called_once_with({"inert":"first"})
                judge.assert_called_once_with({"inert":"second"})

    def test_changed_receipt_blocks_first_send_without_loading_secret(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = restoration_fixture(Path(tmp))
            with fixture_constants(fixture), patch.object(driver, "_load_recovered_transports") as load:
                connection = driver._RecoveredConnection(REPO, provenance(fixture))
                fixture["marker"].write_text(json.dumps(dict(fixture["receipt"], model_calls=1)))
                with self.assertRaises(ValueError):
                    connection.delegates[driver.ANSWER_IDENTITY["model"]]({"inert":True})
                load.assert_not_called()

    def test_factory_failure_is_sanitized_and_never_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = restoration_fixture(Path(tmp))
            with fixture_constants(fixture), patch.object(driver, "_load_recovered_transports",
                    side_effect=RuntimeError("fixture-private-error-value")) as load:
                connection = driver._RecoveredConnection(REPO, provenance(fixture))
                for identity in (driver.ANSWER_IDENTITY, driver.JUDGE_IDENTITY):
                    with self.assertRaises(RuntimeError) as error:
                        connection.delegates[identity["model"]]({"inert":True})
                    self.assertNotIn("fixture-private-error-value", str(error.exception))
                load.assert_called_once_with(REPO)

    def test_concurrent_failed_initialization_does_not_retry_secret_factory(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = restoration_fixture(Path(tmp))
            factory_entered, second_initial_guard = threading.Event(), threading.Event()
            failures = []
            def failed_factory(repo):
                factory_entered.set()
                if not second_initial_guard.wait(timeout=2):
                    raise AssertionError("second fixture caller did not reach its initial guard")
                raise RuntimeError("fixture-private-auth-error")
            with fixture_constants(fixture), patch.object(driver, "_load_recovered_transports",
                    side_effect=failed_factory) as load:
                connection = driver._RecoveredConnection(REPO, provenance(fixture))
                original_guard = connection.assert_current
                def observed_guard():
                    original_guard()
                    if threading.current_thread().name == "unit-recovered-judge":
                        second_initial_guard.set()
                def send(identity):
                    try:
                        connection.delegates[identity["model"]]({"inert":True})
                    except Exception as error:
                        failures.append(error)
                first = threading.Thread(target=send, args=(driver.ANSWER_IDENTITY,), name="unit-recovered-answer")
                second = threading.Thread(target=send, args=(driver.JUDGE_IDENTITY,), name="unit-recovered-judge")
                with patch.object(connection, "assert_current", side_effect=observed_guard):
                    first.start()
                    self.assertTrue(factory_entered.wait(timeout=2))
                    second.start()
                    first.join(timeout=2)
                    second.join(timeout=2)
                self.assertFalse(first.is_alive())
                self.assertFalse(second.is_alive())
                self.assertTrue(second_initial_guard.is_set())
                self.assertEqual(len(failures), 2)
                self.assertTrue(all(isinstance(error, RuntimeError) for error in failures))
                self.assertTrue(all("fixture-private-auth-error" not in str(error) for error in failures))
                load.assert_called_once_with(REPO)

    def test_receipt_change_during_initialization_blocks_actual_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = restoration_fixture(Path(tmp))
            answer = Mock(identity=driver.ANSWER_IDENTITY)
            judge = Mock(identity=driver.JUDGE_IDENTITY)
            def changed_while_loading(repo):
                fixture["marker"].write_text(json.dumps(dict(fixture["receipt"], model_calls=1)))
                return {driver.ANSWER_IDENTITY["model"]:answer, driver.JUDGE_IDENTITY["model"]:judge}
            with fixture_constants(fixture), patch.object(driver, "_load_recovered_transports",
                    side_effect=changed_while_loading) as load:
                connection = driver._RecoveredConnection(REPO, provenance(fixture))
                with self.assertRaises(ValueError):
                    connection.delegates[driver.ANSWER_IDENTITY["model"]]({"inert":True})
                load.assert_called_once_with(REPO)
                answer.assert_not_called()
                judge.assert_not_called()

    def test_existing_original_invocation_or_live_legacy_worker_blocks_recovered_send(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = restoration_fixture(Path(tmp))
            with fixture_constants(fixture), patch.object(driver, "_load_recovered_transports") as load:
                connection = driver._RecoveredConnection(REPO, provenance(fixture))
                with self.assertRaises(RuntimeError):
                    driver._parents({"v6_cbt_process":SimpleNamespace(poll=lambda:None)}, connection)
                thread = threading.Thread()
                with patch.object(thread, "is_alive", return_value=True):
                    with self.assertRaises(RuntimeError):
                        driver._parents({"v5_threads":{"legacy":thread}}, connection)
                legacy = Path(driver.DEFAULT_ROOT)
                legacy.mkdir()
                (legacy / "entry.json").write_text('{"nonce":"earlier-original-attempt"}')
                with self.assertRaises(RuntimeError):
                    connection.delegates[driver.ANSWER_IDENTITY["model"]]({"inert":True})
                load.assert_not_called()

    def test_source_preparation_then_budget_failure_never_initializes_credentials(self):
        # Exercise the orchestration up to its real fail-closed budget boundary.
        # Only the Colab/checkout boundary and workers are inert unit seams; no
        # production Colab flags, browser, key, network or executable child exists.
        with tempfile.TemporaryDirectory() as tmp:
            fixture = restoration_fixture(Path(tmp))
            root = Path(tmp) / "recovered-invocation"
            phases = []
            relay = SimpleNamespace(snapshot=lambda:{}, close=Mock())
            def source_worker(state, phase, action, **kwargs):
                phases.append(action)
                self.assertIn(action, {"cbt-source", "locomo-source"})
                return {"inert_source_only":True}
            with fixture_constants(fixture):
                verified = provenance(fixture)
                with patch.object(driver, "_hosted"), \
                        patch.object(driver, "DEFAULT_RECOVERED_ROOT", str(root)), \
                        patch.object(driver, "_content_path", side_effect=lambda p, **kw: Path(p)), \
                        patch.object(driver, "_checkout", return_value=REPO), \
                        patch.object(driver, "_source_paths", return_value=fixture["sources"]), \
                        patch.object(driver, "_restoration_provenance", return_value=verified), \
                        patch.object(driver.runpy, "run_path", return_value={"start_relay":Mock(return_value=relay)}), \
                        patch.object(driver, "_launch", side_effect=source_worker), \
                        patch.object(driver, "_budget", side_effect=ValueError("unit original budget exhausted")), \
                        patch.object(driver, "aggregate_results", return_value={"inert_summary":True}), \
                        patch.object(driver, "export_checkpoint"), \
                        patch.object(driver, "_load_recovered_transports") as load, redirect_stdout(io.StringIO()):
                    namespace = {}
                    result = driver.run_recovered(namespace, repo=REPO, expected_commit="unit pinned commit",
                        sources=fixture["sources"], root=root, source_restoration_receipt=fixture["marker"],
                        source_restore_archive=fixture["archive"], background=False)
                self.assertEqual(phases, ["cbt-source", "cbt-source", "locomo-source"])
                self.assertEqual(result["status"], "stopped_without_retry")
                self.assertEqual(result["phase"], "preflight_or_budget")
                self.assertNotIn("v5_answer", namespace)
                self.assertNotIn("v5_judge", namespace)
                entry = json.loads((root / "entry.json").read_text())
                self.assertEqual(entry["connection_origin"], "reconstructed_from_existing_colab_secret")
                self.assertEqual(entry["version"], driver.RESTORED_CONNECTION_VERSION)
                self.assertFalse(entry["allow_longmemeval_paid"])
                self.assertFalse(entry["mab_accessed"])
                self.assertEqual(entry["source_restoration"]["verification_model_calls"], 0)
                load.assert_not_called()
                relay.close.assert_called_once()

    def test_original_entry_still_requires_original_parents_without_secret(self):
        with patch.object(driver, "_load_recovered_transports") as load:
            with self.assertRaises(RuntimeError):
                driver._parents({})
            load.assert_not_called()

    def test_repeated_recovered_entry_returns_same_nonce_without_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "entry.json").write_text(json.dumps({"nonce":"retained-restored-run", "status":"stopped_without_retry"}))
            with patch.object(driver, "_hosted"), \
                    patch.object(driver, "DEFAULT_RECOVERED_ROOT", str(root)), \
                    patch.object(driver, "_content_path", side_effect=lambda p, **kw: Path(p)), \
                    patch.object(driver, "_checkout") as checkout, \
                    patch.object(driver, "_restoration_provenance") as restore, \
                    patch.object(driver, "_load_recovered_transports") as load:
                result = driver.run_recovered({}, root=root, repo="unused", expected_commit="unused", sources={},
                    source_restoration_receipt="unused", source_restore_archive="unused")
            self.assertEqual(result["nonce"], "retained-restored-run")
            self.assertEqual(result["repeat_run_action"], "report_existing_invocation_only")
            checkout.assert_not_called()
            restore.assert_not_called()
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
