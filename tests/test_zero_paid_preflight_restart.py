"""Zero-call restart proof fixtures; no runtime flags, transport or credentials."""
import importlib.util
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "evaluation/architecture-rerun/cloud_continue_non_mab.py"
SPEC = importlib.util.spec_from_file_location("non_mab_zero_restart_unit", SOURCE)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)
NONCE = "4ef01c9e-9cf4-4c41-8f47-be9985b526bf"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def zero_fixture(base):
    """Invent the exact inactive source-only control pattern in a temporary tree."""
    previous = base / "previous-v1"
    source_base = base / "sources"
    sources = {c:{"reader":str(source_base/c), "memory":str(source_base/(c+"_memory")),
                  "old_quality":str(source_base/(c+"_quality"))} for c in ("professional7", "cactus600")}
    sources["locomo1540"] = {"reader":str(source_base/"locomo1540"), "old_quality":str(source_base/"locomo1540-judge")}
    sources["longmemeval500"] = None
    marker = base / "restoration.json"
    write_json(marker, {"status":"restored_original_bytes_zero_model_calls", "model_calls":0})
    restoration = {"version":"hosted-non-mab-restored-connection-v1",
        "origin":"reconstructed_from_existing_colab_secret", "source_restore_archive_sha256":driver.RESTORE_ARCHIVE_SHA256,
        "source_restoration_receipt":str(marker), "source_restoration_receipt_sha256":sha256(marker.read_bytes()).hexdigest(),
        "source_manifest_sha256":"1"*64, "restored_original_files_verified":driver.RESTORE_FILE_COUNT,
        "restored_original_bytes_verified":driver.RESTORE_UNCOMPRESSED_BYTES,
        "verification_model_calls":0, "mab_accessed":False}
    entry = {"version":"hosted-non-mab-restored-connection-v1", "nonce":NONCE,
        "status":"reserved_zero_calls", "root":str(previous), "sources":sources,
        "selected_commit":driver.PREVIOUS_PREFLIGHT_COMMIT, "connection_origin":"reconstructed_from_existing_colab_secret",
        "source_restoration":restoration, "allow_longmemeval_paid":False, "mab_accessed":False}
    write_json(previous/"entry.json", entry)
    saved = {**entry, "status":"stopped_without_retry", "phase":"preflight_or_budget",
        "worker_root":str(previous/"workers/locomo1540-source"), "error_type":"RuntimeError"}
    events = []
    workers = []
    for number, cohort in enumerate(("professional7", "cactus600", "locomo1540")):
        phase = cohort + "-source"
        action = "locomo-source" if cohort=="locomo1540" else "cbt-source"
        folder = previous / "workers" / phase
        manifest = {"version":"hosted-non-mab-continuation-v1", "nonce":NONCE, "phase":phase,
            "action":action, "cohort":cohort, "selected_repo":str(REPO), "selected_commit":driver.PREVIOUS_PREFLIGHT_COMMIT,
            "run_root":str(previous/"studies"/cohort), "source":sources[cohort],
            "socket":str(previous/"relay/api.sock"), "job_id":NONCE+":"+phase,
            "answer_max_new_calls":0, "judge_max_new_calls":0,
            "relay_limits":{driver.ANSWER_IDENTITY["model"]:0, driver.JUDGE_IDENTITY["model"]:0}}
        path = folder / "execution_manifest.json"
        write_json(path, manifest)
        write_json(folder/"process.json", {"pid":2_000_000_000+number,
            "manifest_sha256":sha256(path.read_bytes()).hexdigest()})
        if cohort=="locomo1540":
            write_json(folder/"error.json", {"status":"failed_without_retry", "error_type":"ValueError"})
        else:
            write_json(folder/"complete.json", {"maximum_new_generation_requests":8 if cohort=="professional7" else 2111})
            for leaf in ("reader", "memory"):
                target = previous / "studies" / cohort / leaf
                target.mkdir(parents=True)
                (target/"intents.jsonl").touch()
                (target/"outcomes.jsonl").touch()
        workers.append(folder)
        events.append({"event":"register", "fixture_only":False, "job_id":NONCE+":"+phase,
            "limits":dict(manifest["relay_limits"])})
    event_file = previous / "relay/events.jsonl"
    event_file.parent.mkdir(parents=True)
    event_file.write_text("".join(json.dumps(event)+"\n" for event in events))
    saved["relay"] = {"jobs":{row["job_id"]:{model:{"maximum":0, "admitted":0} for model in row["limits"]} for row in events}}
    write_json(previous/"status.json", saved)
    return {"previous":previous, "sources":sources, "restoration":restoration,
            "workers":workers, "events":events, "event_file":event_file}


def proof(fixture, namespace=None):
    return driver._zero_call_preflight_proof(fixture["previous"], {} if namespace is None else namespace,
        expected_nonce=NONCE, sources=fixture["sources"], restoration=fixture["restoration"])


def bytes_snapshot(folder):
    return {str(path.relative_to(folder)):sha256(path.read_bytes()).hexdigest()
        for path in folder.rglob("*") if path.is_file()}


class ZeroPaidRestartProofTests(unittest.TestCase):
    def test_exact_zero_source_only_failure_is_proven_without_writing_old_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = zero_fixture(Path(tmp))
            before = bytes_snapshot(fixture["previous"])
            with patch.object(driver, "_load_recovered_transports") as load:
                result = proof(fixture)
            self.assertEqual(bytes_snapshot(fixture["previous"]), before)
            self.assertEqual(result["previous_nonce"], NONCE)
            self.assertEqual(result["provider_intents"], 0)
            self.assertEqual(result["provider_outcomes"], 0)
            self.assertEqual(result["registered_source_jobs"], 3)
            self.assertTrue(result["old_root_preserved"])
            self.assertIn("relay/events.jsonl", result["previous_files_sha256"])
            load.assert_not_called()

    def test_any_possible_relay_send_nonzero_cap_or_fixture_provenance_is_rejected(self):
        for change in ("intent", "outcome", "nonzero", "boolean", "fixture"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                fixture = zero_fixture(Path(tmp))
                if change in {"intent", "outcome"}:
                    fixture["events"].append({"event":change, "fixture_only":False, "request_id":"invented"})
                elif change in {"nonzero", "boolean"}:
                    fixture["events"][0]["limits"][driver.ANSWER_IDENTITY["model"]] = 1 if change=="nonzero" else False
                else:
                    fixture["events"][0]["fixture_only"] = True
                fixture["event_file"].write_text("".join(json.dumps(row)+"\n" for row in fixture["events"]))
                with self.assertRaises(ValueError): proof(fixture)

    def test_nonzero_or_mismatched_worker_cannot_be_disguised_as_source_only(self):
        for change in ("cap", "boolean", "action", "nonce", "cohort"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                fixture = zero_fixture(Path(tmp))
                path = fixture["workers"][0]/"execution_manifest.json"
                manifest = json.loads(path.read_text())
                if change in {"cap", "boolean"}:
                    manifest["answer_max_new_calls"] = 1 if change=="cap" else False
                elif change=="action": manifest["action"] = "cbt-run"
                elif change=="nonce": manifest["nonce"] = "another-invocation"
                else: manifest["cohort"] = "cactus600"
                write_json(path, manifest)
                process = json.loads((path.parent/"process.json").read_text())
                process["manifest_sha256"] = sha256(path.read_bytes()).hexdigest()
                write_json(path.parent/"process.json", process)
                with self.assertRaises(ValueError): proof(fixture)

    def test_every_old_reader_and_assessor_journal_must_be_physically_empty(self):
        for name, body in (("intents.jsonl",'{"key":"invented"}\n'), ("outcomes.jsonl",'{"status":"unknown"}\n'),
                           ("intents.jsonl"," ")):
            with self.subTest(name=name, body=body), tempfile.TemporaryDirectory() as tmp:
                fixture = zero_fixture(Path(tmp))
                journal = fixture["previous"]/"studies/professional7/memory"/name
                journal.write_text(body)
                with self.assertRaises(ValueError): proof(fixture)

    def test_successor_must_retain_same_archive_marker_and_source_map(self):
        for change in ("archive", "marker", "sources"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                fixture = zero_fixture(Path(tmp))
                if change=="archive": fixture["restoration"]["source_restore_archive_sha256"] = "2"*64
                elif change=="marker": fixture["restoration"]["source_restoration_receipt_sha256"] = "2"*64
                else: fixture["sources"]["professional7"]["reader"] += "-different"
                with self.assertRaises(ValueError): proof(fixture)

    def test_live_previous_controller_or_matching_worker_prevents_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = zero_fixture(Path(tmp))
            thread = threading.Thread()
            with patch.object(thread, "is_alive", return_value=True):
                with self.assertRaises(RuntimeError):
                    proof(fixture, {driver.STATE_NAME:{"nonce":NONCE, "thread":thread}})
            with self.assertRaises(RuntimeError):
                proof(fixture, {driver.STATE_NAME:{"nonce":NONCE, "process":SimpleNamespace(poll=lambda:None)}})
            with patch.object(driver, "_worker_is_running", return_value=True) as active:
                with self.assertRaises(RuntimeError): proof(fixture)
                active.assert_called()

    def test_missing_or_wrong_explicit_observed_nonce_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = zero_fixture(Path(tmp))
            for nonce in (None, "invented-nonce"):
                with self.subTest(nonce=nonce), self.assertRaises(ValueError):
                    driver._zero_call_preflight_proof(fixture["previous"], {}, expected_nonce=nonce,
                        sources=fixture["sources"], restoration=fixture["restoration"])

    def test_changed_terminal_state_or_admission_snapshot_is_rejected(self):
        for change in ("phase", "status", "admitted", "boolean"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                fixture = zero_fixture(Path(tmp))
                path = fixture["previous"]/"status.json"
                value = json.loads(path.read_text())
                if change=="phase": value["phase"] = "professional7-continuation"
                elif change=="status": value["status"] = "running"
                else:
                    row = next(iter(value["relay"]["jobs"].values()))
                    row[driver.ANSWER_IDENTITY["model"]]["admitted"] = 1 if change=="admitted" else False
                write_json(path, value)
                with self.assertRaises(ValueError): proof(fixture)

    def test_changed_proof_control_or_reappearing_old_relay_blocks_secret_access(self):
        for change in ("control", "socket"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                fixture = zero_fixture(Path(tmp))
                verified = proof(fixture)
                restored = dict(fixture["restoration"], version=driver.RESTORED_CONNECTION_VERSION)
                with patch.object(driver, "DEFAULT_ROOT", str(Path(tmp)/"unrelated-original")), \
                        patch.object(driver, "PREVIOUS_RECOVERED_ROOT", str(fixture["previous"])), \
                        patch.object(driver, "_load_recovered_transports") as load:
                    connection = driver._RecoveredConnection(REPO, restored, prior_preflight=verified)
                    connection.assert_current()
                    if change=="control":
                        with fixture["event_file"].open("a") as stream: stream.write('{"event":"intent"}\n')
                    else:
                        (fixture["previous"]/"relay/api.sock").touch()
                    with self.assertRaises((ValueError, RuntimeError)):
                        connection.delegates[driver.ANSWER_IDENTITY["model"]]({"inert":True})
                    load.assert_not_called()

    def test_v2_restart_requires_nonce_before_checkout_or_credential_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = zero_fixture(Path(tmp))
            new_root = Path(tmp)/"successor-v2"
            with patch.object(driver, "_hosted"), \
                    patch.object(driver, "DEFAULT_ROOT", str(Path(tmp)/"unrelated-original")), \
                    patch.object(driver, "DEFAULT_RECOVERED_ROOT", str(new_root)), \
                    patch.object(driver, "PREVIOUS_RECOVERED_ROOT", str(fixture["previous"])), \
                    patch.object(driver, "_content_path", side_effect=lambda path, **kw:Path(path)), \
                    patch.object(driver, "_checkout") as checkout, \
                    patch.object(driver, "_load_recovered_transports") as load:
                with self.assertRaises(ValueError):
                    driver.run_recovered({}, repo="unused", expected_commit="unused", sources=fixture["sources"],
                        root=new_root, source_restoration_receipt="unused", source_restore_archive="unused")
                checkout.assert_not_called()
                load.assert_not_called()

    def test_repeated_v2_run_reports_same_nonce_without_reexamining_or_relaunching(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"successor-v2"
            write_json(root/"entry.json", {"nonce":"retained-v2", "status":"running"})
            with patch.object(driver, "_hosted"), patch.object(driver, "DEFAULT_RECOVERED_ROOT", str(root)), \
                    patch.object(driver, "_content_path", side_effect=lambda path, **kw:Path(path)), \
                    patch.object(driver, "_zero_call_preflight_proof") as previous, \
                    patch.object(driver, "_checkout") as checkout, \
                    patch.object(driver, "_load_recovered_transports") as load:
                value = driver.run_recovered({}, repo="unused", expected_commit="unused", sources={}, root=root,
                    source_restoration_receipt="unused", source_restore_archive="unused")
                self.assertEqual(value["nonce"], "retained-v2")
                self.assertEqual(value["repeat_run_action"], "report_existing_invocation_only")
                previous.assert_not_called(); checkout.assert_not_called(); load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
