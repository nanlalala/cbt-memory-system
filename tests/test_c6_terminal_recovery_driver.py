"""Pure budget/drain fixtures; no fabricated hosted runtime or model calls."""
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"evaluation/architecture-rerun/cloud_c6_terminal_recovery.py"
spec=importlib.util.spec_from_file_location("c6_recovery_driver",SOURCE)
api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
NONCE="b12ea223-f4c7-476c-83e1-a91233443f78"


def budget():
    return {"original_cbt_envelope":{api.ANSWER:4805,api.JUDGE:5032},
        "separate_professional7_quality_repair_allowance":24,
        "cbt_actual_local_paid_intents":{api.ANSWER:916,api.JUDGE:915},
        "previous_authorized_recovery_attempts_counted":{api.ANSWER:3,api.JUDGE:0},
        "cbt_remaining_after_paid_intents":{api.ANSWER:3886,api.JUDGE:4141}}


def pair(key="r1",model=api.ANSWER,cohort="cactus600-continuation",status="received"):
    common={"request_id":key,"job_id":NONCE+":"+cohort,"model":model,"wire_sha256":"a"*64}
    return [{"event":"intent",**common},{"event":"outcome","status":status,**common}]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+"\n")


def source_fixture(root):
    write(root/"entry.json",{"nonce":NONCE})
    worker=root/"workers/cactus600-continuation"
    manifest=worker/"execution_manifest.json"
    write(manifest,{"nonce":NONCE,"phase":"cactus600-continuation","cohort":"cactus600",
        "action":"cbt-run","run_root":str(root/"studies/cactus600")})
    write(worker/"process.json",{"pid":2_147_483_647,"manifest_sha256":sha256(manifest.read_bytes()).hexdigest()})
    write(worker/"complete.json",{"planned_cases":600,"completed_cases":273})
    write(root/"studies/cactus600/reader/last_invocation.json",{"new_calls":485,
        "errors":[{"case_id":"invented-"+str(i),"status":"UnknownAttempt"} for i in range(327)]})
    for part in ("reader","memory"):
        for name in ("protocol.json","cases.json","intents.jsonl","outcomes.jsonl"):
            write(root/"studies/cactus600"/part/name,{"invented_fixture":True})
    return {"root":root,"nonce":NONCE,"status":{"cohorts":{"cactus600":{"status":"stopped_without_retry"}}},
        "process":SimpleNamespace(poll=lambda:None)}


class BudgetTests(unittest.TestCase):
    def test_counts_both_cbt_models_and_debits_prior_three_once(self):
        result=api.remaining_cbt_budget(budget(),pair()+pair("r2",api.JUDGE,"professional7-quality"),NONCE)
        self.assertEqual(result["available"],{api.ANSWER:3885,api.JUDGE:4140})
        self.assertEqual(result["prior_recovery_attempts_already_debited"][api.ANSWER],3)

    def test_later_locomo_can_remain_inflight_without_borrowing_its_allowance(self):
        rows=pair()+pair("locomo",api.JUDGE,"locomo1540-judge")[:1]
        result=api.remaining_cbt_budget(budget(),rows,NONCE)
        self.assertEqual(result["available"][api.JUDGE],4141)
        self.assertFalse(result["locomo_allowance_borrowed"])

    def test_length_response_is_counted_as_a_paid_parent_admission(self):
        rows=pair();rows[-1]["receipt"]={"finish_reason":"length","text":"invented truncation"}
        result=api.remaining_cbt_budget(budget(),rows,NONCE)
        self.assertEqual(result["v4_cbt_parent_admissions"][api.ANSWER],1)

    def test_cbt_unknown_or_inflight_blocks(self):
        for rows in (pair(status="unknown"),pair()[:1],pair()[1:]):
            with self.subTest(rows=rows),self.assertRaises(RuntimeError):
                api.remaining_cbt_budget(budget(),rows,NONCE)

    def test_duplicates_mismatched_wires_and_foreign_model_block(self):
        bad=pair();bad[1]["wire_sha256"]="b"*64
        for rows in (pair()+pair(),bad,pair(model="invented-other-provider")):
            with self.subTest(rows=rows),self.assertRaises((RuntimeError,ValueError)):
                api.remaining_cbt_budget(budget(),rows,NONCE)

    def test_negative_historical_counts_cannot_enlarge_original_cap(self):
        for field in ("cbt_actual_local_paid_intents","previous_authorized_recovery_attempts_counted"):
            value=deepcopy(budget());value[field][api.ANSWER]=-1
            value["cbt_remaining_after_paid_intents"][api.ANSWER]=4805-value["cbt_actual_local_paid_intents"][api.ANSWER]-value["previous_authorized_recovery_attempts_counted"][api.ANSWER]
            with self.subTest(field=field),self.assertRaises(ValueError):
                api.remaining_cbt_budget(value,[],NONCE)

    def test_cap_or_arithmetic_change_blocks(self):
        for field in ("original_cbt_envelope","cbt_remaining_after_paid_intents"):
            value=deepcopy(budget());value[field][api.JUDGE]+=1
            with self.subTest(field=field),self.assertRaises(ValueError):
                api.remaining_cbt_budget(value,[],NONCE)

    def test_event_stream_uses_physical_lf_and_tolerates_unrelated_partial_tail(self):
        with tempfile.TemporaryDirectory() as raw:
            path=Path(raw)/"events.jsonl";rows=pair()
            rows[0]["payload"]={"text":"invented\u2028source\u2029text"}
            path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows)+'{"event":"intent"')
            result=api.remaining_cbt_budget(budget(),api._events(path),NONCE)
            self.assertEqual(result["v4_cbt_parent_admissions"][api.ANSWER],1)


class DrainTests(unittest.TestCase):
    def test_drained_cbt_allows_the_old_controller_to_keep_running_locomo(self):
        with tempfile.TemporaryDirectory() as raw:
            prior=source_fixture(Path(raw));proof=api.drained_c6(prior)
            self.assertEqual(proof["old_source_complete_cases"],273)
            self.assertEqual(len(proof["source_file_sha256"]),8)
            api._source_unchanged({"source_proof":proof})
            (Path(raw)/"studies/cactus600/reader/outcomes.jsonl").write_text("changed\n")
            with self.assertRaises(ValueError):api._source_unchanged({"source_proof":proof})

    def test_c6_must_have_reached_its_recorded_stop(self):
        with tempfile.TemporaryDirectory() as raw:
            prior=source_fixture(Path(raw));prior["status"]["cohorts"]["cactus600"]["status"]="running"
            with self.assertRaises(RuntimeError):api.drained_c6(prior)

    def test_active_exact_c6_process_blocks_even_if_status_was_written(self):
        with tempfile.TemporaryDirectory() as raw:
            prior=source_fixture(Path(raw));manifest=Path(raw)/"workers/cactus600-continuation/execution_manifest.json"
            cmd=Path("/proc/2147483647/cmdline")
            original_exists,original_read=Path.exists,Path.read_bytes
            def exists(p):return True if p==cmd else original_exists(p)
            def read(p):return str(manifest).encode() if p==cmd else original_read(p)
            with patch.object(Path,"exists",exists),patch.object(Path,"read_bytes",read),self.assertRaises(RuntimeError):
                api.drained_c6(prior)

    def test_other_failure_totals_cannot_masquerade_as_the_observed_checkpoint(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw);prior=source_fixture(root)
            path=root/"studies/cactus600/reader/last_invocation.json"
            value=json.loads(path.read_text());value["new_calls"]=484;write(path,value)
            with self.assertRaises(ValueError):api.drained_c6(prior)


if __name__=="__main__":unittest.main()
