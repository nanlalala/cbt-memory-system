"""Authorized-attempt admission tests with inert records; no runtime or provider."""
import copy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
import uuid
from unittest.mock import Mock, patch
from contextlib import ExitStack
from contextlib import redirect_stdout
import io
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "evaluation/architecture-rerun/authorized_attempt_recovery.py"
SPEC = importlib.util.spec_from_file_location("authorized_recovery_unit", SOURCE)
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)
DRIVER_SPEC = importlib.util.spec_from_file_location("authorized_attempt_driver_unit",
    REPO/"evaluation/architecture-rerun/cloud_continue_non_mab.py")
driver = importlib.util.module_from_spec(DRIVER_SPEC)
DRIVER_SPEC.loader.exec_module(driver)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def inert_attempt(number, model, identity):
    payload = {"model":model, "messages":[{"role":"system", "content":"Inert unit policy"},
        {"role":"user", "content":json.dumps({"invented_case":number})}], "temperature":0,
        "thinking":{"type":"disabled"}, "max_tokens":2048}
    wire = json.dumps(payload, allow_nan=False)
    purpose, case = "support_check", "invented-case-"+str(number)
    return {"case_id":case, "purpose":purpose, "key":digest({"purpose":purpose, "payload":payload}),
            "wire_sha256":sha256(wire.encode()).hexdigest(), "serialized_request":wire,
            "payload":payload, "model":model, "identity":dict(identity),
            "request_id":str(uuid.uuid5(uuid.NAMESPACE_OID, "prior-authorized-"+str(number))),
            "protocol_sha256":"1"*64}


def snapshot(folder):
    return {str(path.relative_to(folder)):sha256(path.read_bytes()).hexdigest()
            for path in folder.rglob("*") if path.is_file()}


class InertRelay:
    """Only captures middleware journal records; opens no socket or connection."""
    def __init__(self, root, transports, **kwargs):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.events = self.root / "events.jsonl"
        self.transports = dict(transports)
        self.identities = {model:dict(transport.identity) for model, transport in transports.items()}
        self._mutex = threading.RLock()
        self._jobs = {}
        self.recorded = []

    def _append(self, event):
        saved = copy.deepcopy(event)
        self.recorded.append(saved)
        with self.events.open("a") as stream:
            stream.write(json.dumps(saved)+"\n")

    def _validate_limits(self, job_id, limits):
        if not isinstance(job_id, str) or not job_id or type(limits) is not dict:
            raise ValueError("inert finite registration required")
        if any(model not in self.transports or type(cap) is not int or cap < 0 for model, cap in limits.items()):
            raise ValueError("inert known model/finite cap required")

    def register_job(self, job_id, limits):
        self._validate_limits(job_id, limits)
        self._append({"event":"register", "job_id":job_id, "limits":dict(limits)})
        self._jobs[job_id] = dict(limits)

    def snapshot(self):
        return {"jobs":copy.deepcopy(self._jobs)}


def relay_fixture(base):
    identity = {"provider":"https://tokendance.space/gateway/v1", "model":recovery.MODEL,
        "returned_model":"deepseek-v4-1-flash-260910", "model_family":"deepseek"}
    attempts = [inert_attempt(number, recovery.MODEL, identity) for number in range(3)]
    for number, row in enumerate(attempts):
        row["cohort"] = "professional7" if number < 2 else "cactus600"
    permits = [{key:row[key] for key in ("cohort","case_id","purpose","key","wire_sha256","request_id","protocol_sha256")}
               for row in attempts]
    previous = base/"previous"
    write_json(previous/"evidence.json", {"inert_frozen_control":True})
    proof = {"version":recovery.VERSION, "previous_run_root":str(previous), "previous_nonce":recovery.PREVIOUS_NONCE,
        "retry_allowlist":permits, "prior_paid_attempts":{recovery.MODEL:3, recovery.JUDGE:0},
        "previous_files_sha256":{"evidence.json":sha256((previous/"evidence.json").read_bytes()).hexdigest()}}
    authorization = base/"private_authorization.json"
    write_json(authorization, {"inert_operator_authorization":True})
    proof["user_authorization"] = {"private_manifest_path":str(authorization),
        "private_manifest_sha256":sha256(authorization.read_bytes()).hexdigest()}
    current = base/"current"
    nonce = str(uuid.uuid5(uuid.NAMESPACE_OID, "current-inert-authorized-run"))
    write_json(current/"entry.json", {"nonce":nonce, "version":"hosted-non-mab-restored-connection-v4"})
    transport = Mock(identity=identity)
    return {"base":base, "root":current/"relay", "current_nonce":nonce, "attempts":attempts,
        "permits":permits, "proof":proof, "transports":{recovery.MODEL:transport}, "transport":transport}


def new_intent(fixture, number=0, *, nonce=None):
    row = fixture["attempts"][number]
    return {"event":"intent", "request_id":str(uuid.uuid4()),
        "job_id":(nonce or fixture["current_nonce"])+":"+row["cohort"]+"-continuation",
        "model":recovery.MODEL, "identity":row["identity"], "payload":copy.deepcopy(row["payload"]),
        "serialized_request":row["serialized_request"], "wire_sha256":row["wire_sha256"]}


def validation_fixture(base):
    fixture = relay_fixture(base)
    previous = Path(fixture["proof"]["previous_run_root"])
    repo = base/"inert-repository"
    write_json(repo/"science.json", {"inert_frozen_scientific_source":True})
    sources = {name:{"reader":str(base/"sources"/name)} for name in ("professional7", "cactus600", "locomo1540")}
    sources["longmemeval500"] = None
    restoration = {"source_restore_archive_sha256":"1"*64, "source_manifest_sha256":"2"*64,
        "source_restoration_receipt":str(base/"source-restoration.json"), "source_restoration_receipt_sha256":"3"*64,
        "restored_original_files_verified":11633, "restored_original_bytes_verified":711889502}
    prior = {"previous_nonce":"inert-two-predecessor-chain", "provider_intents":0, "provider_outcomes":0}
    write_json(previous/"entry.json", {"nonce":recovery.PREVIOUS_NONCE, "version":"hosted-non-mab-restored-connection-v3",
        "selected_commit":recovery.PREVIOUS_COMMIT, "sources":sources, "mab_accessed":False,
        "allow_longmemeval_paid":False, "source_restoration":restoration, "preflight_predecessor":prior})
    write_json(previous/"status.json", {"nonce":recovery.PREVIOUS_NONCE, "status":"finished"})
    for filename in ("operator_pause.json", "operator_quiescence.json", "source_preparation.json", "finite_budget_plan.json"):
        write_json(previous/filename, {"inert_control":filename})
    records = []
    for cohort in ("professional7", "cactus600", "locomo1540"):
        folder = previous/"studies"/cohort
        if cohort!="locomo1540": folder /= "reader"
        protocol = {"cohort":cohort, "code_sha256":{"science.json":sha256((repo/"science.json").read_bytes()).hexdigest()}}
        protocol["protocol_sha256"] = digest(protocol)
        write_json(folder/"protocol.json", protocol)
        matching = [row for row in fixture["attempts"] if row["cohort"]==cohort]
        for row in matching: row["protocol_sha256"] = protocol["protocol_sha256"]
        folder.mkdir(parents=True, exist_ok=True)
        intents, outcomes = [], []
        for row in matching:
            intents.append({key:copy.deepcopy(row[key]) for key in ("case_id","purpose","key","wire_sha256",
                "protocol_sha256","payload","serialized_request")})
            outcomes.append({key:row[key] for key in ("case_id","purpose","key","protocol_sha256")}|{"status":"unknown"})
            job = recovery.PREVIOUS_NONCE+":"+cohort+"-continuation"
            records.append({"event":"intent", "fixture_only":False, "request_id":row["request_id"], "job_id":job,
                "model":recovery.MODEL, "identity":row["identity"], "payload":row["payload"],
                "serialized_request":row["serialized_request"], "wire_sha256":row["wire_sha256"]})
            records.append({"event":"outcome", "fixture_only":False, "status":"unknown", "request_id":row["request_id"],
                "job_id":job, "model":recovery.MODEL, "wire_sha256":row["wire_sha256"]})
        for filename, values in (("intents.jsonl",intents), ("outcomes.jsonl",outcomes)):
            (folder/filename).write_text("".join(json.dumps(row)+"\n" for row in values))
        worker = previous/"workers"/(cohort+"-continuation")
        write_json(worker/"execution_manifest.json", {"nonce":recovery.PREVIOUS_NONCE, "selected_commit":recovery.PREVIOUS_COMMIT})
        write_json(worker/"process.json", {"pid":2_000_000_000,
            "manifest_sha256":sha256((worker/"execution_manifest.json").read_bytes()).hexdigest()})
    memory = previous/"studies/cactus600/memory"
    memory_protocol = {"cohort":"cactus600-memory",
        "code_sha256":{"science.json":sha256((repo/"science.json").read_bytes()).hexdigest()}}
    memory_protocol["protocol_sha256"] = digest(memory_protocol)
    write_json(memory/"protocol.json", memory_protocol)
    local_identity = {"provider":"https://tokendance.space/gateway/v1", "model":recovery.JUDGE,
        "returned_model":"qwen3.5-plus-2026-04-20", "model_family":"qwen"}
    local = inert_attempt(99, recovery.JUDGE, local_identity)
    local.update(cohort="cactus600", purpose="fact_targets", protocol_sha256=memory_protocol["protocol_sha256"])
    local["key"] = digest({"purpose":local["purpose"],"payload":local["payload"]})
    local_permit = {key:local[key] for key in ("cohort","case_id","purpose","model","key","protocol_sha256","wire_sha256")}
    local_permit["error_type"] = "FileNotFoundError"
    local_intent = {key:copy.deepcopy(local[key]) for key in ("case_id","purpose","key","protocol_sha256",
        "wire_sha256","serialized_request","payload")}
    local_outcome = {key:local[key] for key in ("case_id","purpose","key","protocol_sha256")}
    local_outcome.update(status="unknown",error_type="FileNotFoundError",provider_model=None,text=None,usage={})
    (memory/"intents.jsonl").write_text(json.dumps(local_intent)+"\n")
    (memory/"outcomes.jsonl").write_text(json.dumps(local_outcome)+"\n")
    event_file = previous/"relay/events.jsonl"
    event_file.parent.mkdir(parents=True)
    event_file.write_text("".join(json.dumps(row)+"\n" for row in records))
    permits = [{key:row[key] for key in ("cohort","case_id","purpose","key","wire_sha256","request_id","protocol_sha256")}
               for row in fixture["attempts"]]
    authorization_text = "fixture operator allows one new attempt"
    authorization = base/"private_authorization.json"
    write_json(authorization, {"user_instruction":authorization_text, "previous_nonce":recovery.PREVIOUS_NONCE,
                             "authorized_action":"one_new_attempt_per_exact_listed_wire"})
    archive = base/"inert_error_archive.zip"
    archive.write_bytes(b"inert preserved error audit; contains no real requests")
    fixture.update(previous=previous, repo=repo, sources=sources, restoration=restoration, prior=prior,
        records=records, event_file=event_file, permits=permits, authorization=authorization,
        authorization_sha=sha256(authorization_text.encode()).hexdigest(), archive=archive, local_permit=local_permit)
    return fixture


def fixture_validation_constants(fixture):
    scope = ExitStack()
    scope.enter_context(patch.object(recovery, "ALLOWLIST", tuple(fixture["permits"])))
    scope.enter_context(patch.object(recovery, "LOCAL_UNADMITTED", fixture["local_permit"]))
    scope.enter_context(patch.object(recovery, "AUTHORIZATION_TEXT_SHA256", fixture["authorization_sha"]))
    scope.enter_context(patch.object(recovery, "ERROR_ARCHIVE", str(fixture["archive"])))
    scope.enter_context(patch.object(recovery, "ERROR_ARCHIVE_SHA256", sha256(fixture["archive"].read_bytes()).hexdigest()))
    return scope


def validate_fixture(fixture, *, namespace=None, zero_validator=None):
    return recovery.validate(fixture["previous"], {} if namespace is None else namespace,
        sources=fixture["sources"], restoration=fixture["restoration"], repo=fixture["repo"],
        zero_proof_validator=zero_validator or (lambda:fixture["prior"]), user_authorization_path=fixture["authorization"])


class FrozenAttemptProofTests(unittest.TestCase):
    def test_exact_local_unadmitted_unknown_is_preserved_without_charge_or_retry_grant(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = validation_fixture(Path(tmp))
            before = snapshot(fixture["previous"])
            with fixture_validation_constants(fixture): result = validate_fixture(fixture)
            self.assertEqual(snapshot(fixture["previous"]), before)
            local = result["preserved_local_unadmitted"]
            self.assertEqual(len(local), 1)
            self.assertEqual(local[0]["previous_status"], "unknown")
            self.assertIs(local[0]["parent_admitted"], False)
            self.assertEqual(local[0]["provider_attempts_counted"], 0)
            self.assertEqual(result["prior_paid_attempts"], {recovery.MODEL:3,recovery.JUDGE:0})
            self.assertEqual(len(result["retry_allowlist"]), 3)
            self.assertNotIn(local[0]["key"], {row["key"] for row in result["retry_allowlist"]})

    def test_local_unadmitted_record_cannot_hide_wrong_error_wire_missing_or_extra_attempt(self):
        for change in ("error","wire","missing","extra"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                fixture = validation_fixture(Path(tmp))
                memory = fixture["previous"]/"studies/cactus600/memory"
                if change in {"error","missing"}:
                    path = memory/"outcomes.jsonl"
                    row = json.loads(path.read_text())
                    if change=="error": row["error_type"] = "RuntimeError"
                    path.write_text("" if change=="missing" else json.dumps(row)+"\n")
                else:
                    path = memory/"intents.jsonl"
                    row = json.loads(path.read_text())
                    if change=="wire":
                        row["wire_sha256"] = "4"*64
                        path.write_text(json.dumps(row)+"\n")
                    else:
                        row["key"] = "5"*64
                        with path.open("a") as stream: stream.write(json.dumps(row)+"\n")
                with fixture_validation_constants(fixture), self.assertRaises(ValueError): validate_fixture(fixture)

    def test_exact_three_unknowns_stay_unknown_and_are_debited_without_rewriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = validation_fixture(Path(tmp))
            before = snapshot(fixture["previous"])
            zero = Mock(return_value=fixture["prior"])
            with fixture_validation_constants(fixture):
                result = validate_fixture(fixture, zero_validator=zero)
            self.assertEqual(snapshot(fixture["previous"]), before)
            self.assertEqual(result["prior_paid_attempts"], {recovery.MODEL:3, recovery.JUDGE:0})
            self.assertEqual(result["previous_status_preserved"], "unknown")
            self.assertEqual(result["previous_cost_status"], "unknown")
            self.assertIsNone(result["provider_http_status_claim"])
            self.assertEqual(len(result["retry_allowlist"]), 3)
            zero.assert_called_once()

    def test_changed_payload_protocol_binding_or_duplicate_parent_intent_is_rejected(self):
        for change in ("payload", "child-binding", "duplicate"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                fixture = validation_fixture(Path(tmp))
                if change=="payload":
                    fixture["records"][0]["payload"]["messages"][0]["content"] = "changed context"
                elif change=="duplicate":
                    fixture["records"][2] = copy.deepcopy(fixture["records"][0])
                else:
                    path = fixture["previous"]/"studies/professional7/reader/intents.jsonl"
                    rows = [json.loads(row) for row in path.read_text().splitlines()]
                    rows[0]["protocol_sha256"] = "9"*64
                    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
                fixture["event_file"].write_text("".join(json.dumps(row)+"\n" for row in fixture["records"]))
                with fixture_validation_constants(fixture), self.assertRaises(ValueError): validate_fixture(fixture)

    def test_received_outcome_or_missing_specific_private_authorization_is_rejected(self):
        for change in ("received", "authorization"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                fixture = validation_fixture(Path(tmp))
                if change=="received":
                    fixture["records"][1]["status"] = "received"
                    fixture["event_file"].write_text("".join(json.dumps(row)+"\n" for row in fixture["records"]))
                else:
                    write_json(fixture["authorization"], {"user_instruction":"different fixture authorization"})
                with fixture_validation_constants(fixture), self.assertRaises(ValueError): validate_fixture(fixture)

    def test_restoration_science_or_zero_predecessor_tamper_is_rejected(self):
        for change in ("restoration", "science", "predecessor"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                fixture = validation_fixture(Path(tmp))
                if change=="restoration": fixture["restoration"]["source_manifest_sha256"] = "8"*64
                elif change=="science": (fixture["repo"]/"science.json").write_text('{"changed":true}')
                else: fixture["prior"]["provider_intents"] = 1
                with fixture_validation_constants(fixture), self.assertRaises(ValueError): validate_fixture(fixture)

    def test_exact_permits_cannot_be_expanded_to_any_of_the_six_older_unknowns(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = validation_fixture(Path(tmp))
            fixture["permits"][0]["wire_sha256"] = next(iter(recovery.OLD_FORBIDDEN_WIRES))
            with fixture_validation_constants(fixture), self.assertRaises(ValueError): validate_fixture(fixture)

    def test_recomputed_protocol_metadata_cannot_break_its_cohort_attempt_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = validation_fixture(Path(tmp))
            path = fixture["previous"]/"studies/professional7/reader/protocol.json"
            protocol = json.loads(path.read_text())
            protocol["changed_metadata"] = True
            protocol["protocol_sha256"] = digest({key:value for key,value in protocol.items() if key!="protocol_sha256"})
            write_json(path, protocol)
            with fixture_validation_constants(fixture), self.assertRaises(ValueError): validate_fixture(fixture)


def budget_fixture():
    sources = {cohort:{"reader":cohort+"-reader", "memory":cohort+"-memory", "old_quality":cohort+"-quality"}
        for cohort in ("professional7","cactus600")}
    sources["locomo1540"] = {"reader":"locomo-reader", "old_quality":"locomo-quality"}
    sources["longmemeval500"] = None
    counts = {"professional7-reader":27,"professional7-memory":24,"professional7-quality":19,
        "cactus600-reader":889,"cactus600-memory":872,"cactus600-quality":0,
        "locomo-reader":3319,"locomo-quality":0}
    prepared = {"professional7":{"maximum_new_generation_requests":8,"maximum_new_assertion_requests":4,
        "quality_after_complete_grid":{"with_independent_10_percent_repeats":24}},
        "cactus600":{"maximum_new_generation_requests":2111,"maximum_new_assertion_requests":2128,
        "quality_after_complete_grid":{"with_independent_10_percent_repeats":1980}},
        "locomo1540":{"maximum_attempts":1638,"maximum_independent_judge_calls":5082}}
    return sources,counts,prepared


class AuthorizedDriverTests(unittest.TestCase):
    def test_three_previous_unknown_attempts_reduce_allowance_without_changing_cohort_caps(self):
        sources,counts,prepared = budget_fixture()
        prior = {recovery.MODEL:3, recovery.JUDGE:0}
        with patch.object(driver, "_paid_census", side_effect=lambda folder,*args,**kw:{"intents":counts[folder]}):
            original = driver._budget(sources, prepared)
            revised = driver._budget(sources, prepared, prior_paid_attempts=prior)
        self.assertEqual(revised["cbt_remaining_after_paid_intents"][recovery.MODEL],
            original["cbt_remaining_after_paid_intents"][recovery.MODEL]-3)
        self.assertEqual(revised["cbt_remaining_after_paid_intents"][recovery.JUDGE],
            original["cbt_remaining_after_paid_intents"][recovery.JUDGE])
        self.assertEqual(revised["previous_authorized_recovery_attempts_counted"], prior)
        self.assertEqual(revised["cbt_new_upper_bounds"], original["cbt_new_upper_bounds"])
        self.assertEqual(revised["locomo_new_upper_bounds"], original["locomo_new_upper_bounds"])

    def test_repeat_fixed_invocation_does_not_reload_credentials_or_relaunch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"authorized-invocation"
            write_json(root/"entry.json", {"nonce":"retained-invocation", "status":"running"})
            with patch.object(driver, "_hosted"), patch.object(driver, "AUTHORIZED_ATTEMPT_ROOT", str(root)), \
                    patch.object(driver, "_content_path", side_effect=lambda path,**kw:Path(path)), \
                    patch.object(driver, "_checkout") as checkout, \
                    patch.object(driver, "_load_recovered_transports") as load:
                result = driver.run_authorized_recovery({}, root=root, repo="unused", expected_commit="unused", sources={},
                    source_restoration_receipt="unused", source_restore_archive="unused", user_authorization_path="unused")
                self.assertEqual(result["nonce"], "retained-invocation")
                checkout.assert_not_called(); load.assert_not_called()

    def test_missing_initialized_prior_connection_stops_without_secret_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/"authorized-invocation"
            with patch.object(driver, "_hosted"), patch.object(driver, "AUTHORIZED_ATTEMPT_ROOT", str(root)), \
                    patch.object(driver, "_content_path", side_effect=lambda path,**kw:Path(path)), \
                    patch.object(driver, "_checkout", return_value=REPO), \
                    patch.object(driver, "_source_paths", return_value={}), \
                    patch.object(driver.runpy, "run_path", return_value={"PREVIOUS_NONCE":recovery.PREVIOUS_NONCE}), \
                    patch.object(driver, "_restoration_provenance") as restoration, \
                    patch.object(driver, "_load_recovered_transports") as load:
                with self.assertRaises(RuntimeError):
                    driver.run_authorized_recovery({}, root=root, repo=REPO, expected_commit="inert", sources={},
                        source_restoration_receipt="unused", source_restore_archive="unused", user_authorization_path="unused")
                restoration.assert_not_called(); load.assert_not_called()

    def test_new_unknown_stops_controller_before_any_later_cohort_is_admitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources,counts,prepared = budget_fixture()
            stopped, actions = [False], []
            relay = SimpleNamespace(snapshot=lambda:{"admission_stopped":stopped[0]}, close=Mock())
            state = {"root":root,"nonce":"inert-current-controller","sources":sources,"namespace":{},
                "status":{},"relay":relay,"process":None,"allow_longmemeval_paid":False,
                "authorized_attempt":{"proof":{"prior_paid_attempts":{recovery.MODEL:3,recovery.JUDGE:0}}}}
            def phase(state, name, action, **kwargs):
                actions.append((kwargs["cohort"],action))
                if action in {"cbt-source","locomo-source"}: return prepared[kwargs["cohort"]]
                self.assertEqual((kwargs["cohort"],action), ("professional7","cbt-run"))
                stopped[0] = True
                raise RuntimeError("inert newly unknown attempt")
            with patch.object(driver, "_launch", side_effect=phase), patch.object(driver, "_prior_recovery"), \
                    patch.object(driver, "_parents"), patch.object(driver, "_budget", return_value={}) as budget, \
                    patch.object(driver, "aggregate_results", return_value={}), patch.object(driver, "export_checkpoint"), \
                    patch.object(driver, "_load_recovered_transports") as load, redirect_stdout(io.StringIO()):
                driver._controller(state)
            self.assertEqual(actions, [("professional7","cbt-source"),("cactus600","cbt-source"),
                ("locomo1540","locomo-source"),("professional7","cbt-run")])
            self.assertEqual(state["status"]["status"], "stopped_on_new_unknown")
            self.assertEqual(state["status"]["phase"], "global_unknown")
            self.assertEqual(budget.call_args.kwargs["prior_paid_attempts"], {recovery.MODEL:3,recovery.JUDGE:0})
            relay.close.assert_called_once(); load.assert_not_called()


class ExactAttemptRelayTests(unittest.TestCase):
    def test_one_new_attempt_retains_unknown_origin_and_never_rewrites_old_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = relay_fixture(Path(tmp))
            before = snapshot(Path(fixture["proof"]["previous_run_root"]))
            relay = recovery.make_relay(InertRelay, fixture["root"], fixture["transports"], fixture["proof"], nonce=fixture["current_nonce"])
            intent = new_intent(fixture)
            relay._append(intent)
            relay._append({"event":"outcome", "status":"received", "request_id":intent["request_id"],
                "job_id":intent["job_id"], "wire_sha256":intent["wire_sha256"]})
            self.assertEqual(relay.recorded[0]["authorized_retry_of"]["request_id"], fixture["permits"][0]["request_id"])
            self.assertEqual(relay.recorded[0]["authorized_retry_of"]["attempt_kind"], "new_authorized_attempt")
            self.assertEqual(relay.recorded[1]["authorized_retry_of"], relay.recorded[0]["authorized_retry_of"])
            self.assertEqual(snapshot(Path(fixture["proof"]["previous_run_root"])), before)
            with self.assertRaises(ValueError): relay._append(new_intent(fixture))
            self.assertEqual(relay.snapshot()["new_authorized_retry_count"], 1)
            fixture["transport"].assert_not_called()

    def test_old_job_or_wrong_cohort_cannot_claim_a_new_authorized_attempt(self):
        for change in ("old-job", "cohort"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                fixture = relay_fixture(Path(tmp))
                relay = recovery.make_relay(InertRelay, fixture["root"], fixture["transports"], fixture["proof"], nonce=fixture["current_nonce"])
                event = new_intent(fixture, nonce=recovery.PREVIOUS_NONCE if change=="old-job" else None)
                if change=="cohort": event["job_id"] = fixture["current_nonce"]+":cactus600-continuation"
                with self.assertRaises(ValueError): relay._append(event)
                self.assertEqual(relay.recorded, [])

    def test_all_six_historical_unknowns_remain_forbidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = relay_fixture(Path(tmp))
            forbidden = [inert_attempt(number+100, recovery.MODEL, fixture["attempts"][0]["identity"]) for number in range(6)]
            with patch.object(recovery, "OLD_FORBIDDEN_WIRES", {row["wire_sha256"] for row in forbidden}):
                relay = recovery.make_relay(InertRelay, fixture["root"], fixture["transports"], fixture["proof"], nonce=fixture["current_nonce"])
                for row in forbidden:
                    event = new_intent(fixture)
                    event.update(payload=row["payload"], serialized_request=row["serialized_request"], wire_sha256=row["wire_sha256"])
                    with self.assertRaises(ValueError): relay._append(event)
                self.assertEqual(relay.recorded, [])

    def test_first_new_unknown_blocks_further_intents_and_all_later_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = relay_fixture(Path(tmp))
            relay = recovery.make_relay(InertRelay, fixture["root"], fixture["transports"], fixture["proof"], nonce=fixture["current_nonce"])
            intent = new_intent(fixture)
            relay._append(intent)
            relay._append({"event":"outcome", "status":"unknown", "request_id":intent["request_id"],
                "job_id":intent["job_id"], "wire_sha256":intent["wire_sha256"]})
            self.assertTrue(relay.snapshot()["admission_stopped"])
            with self.assertRaises(RuntimeError): relay._append(new_intent(fixture, 1))
            with self.assertRaises(RuntimeError): relay.register_job("later-job", {recovery.MODEL:0})
            self.assertEqual(len(list(Path(tmp).rglob("admission_halted.json"))), 1)
            fixture["transport"].assert_not_called()

    def test_received_outcome_for_already_admitted_call_is_retained_after_global_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = relay_fixture(Path(tmp))
            relay = recovery.make_relay(InertRelay, fixture["root"], fixture["transports"], fixture["proof"], nonce=fixture["current_nonce"])
            first, second = new_intent(fixture), new_intent(fixture, 1)
            relay._append(first); relay._append(second)
            relay._append({"event":"outcome", "status":"unknown", "request_id":first["request_id"],
                "job_id":first["job_id"], "wire_sha256":first["wire_sha256"]})
            relay._append({"event":"outcome", "status":"received", "request_id":second["request_id"],
                "job_id":second["job_id"], "wire_sha256":second["wire_sha256"]})
            self.assertEqual(relay.recorded[-1]["status"], "received")
            self.assertEqual(relay.recorded[-1]["authorized_retry_of"]["key"], fixture["permits"][1]["key"])

    def test_existing_relay_audit_cannot_reset_once_only_permission(self):
        with tempfile.TemporaryDirectory() as tmp:
            fixture = relay_fixture(Path(tmp))
            relay = recovery.make_relay(InertRelay, fixture["root"], fixture["transports"], fixture["proof"], nonce=fixture["current_nonce"])
            relay._append(new_intent(fixture))
            with self.assertRaises((ValueError, RuntimeError)):
                recovery.make_relay(InertRelay, fixture["root"], fixture["transports"], fixture["proof"], nonce=fixture["current_nonce"])


if __name__ == "__main__":
    unittest.main()
