"""Tiny fictional two-arm fixtures; no corpus, model, API, credentials or guard bypass."""
from copy import deepcopy
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import zipfile

HERE = Path(__file__).resolve().parents[1] / "evaluation/objective-memory"
sys.path.insert(0, str(HERE))
import persona_management_scoring as scoring


def fixture():
    cases, refs, rows, prepared = [], [], [], []
    for n, group in ((1, "fixture-g1"), (2, "fixture-g1"), (3, "fixture-g2")):
        cid = f"fixture:{n}"
        source = {"turn_id": "fixture-turn", "session_id": "fixture-session", "role": "user",
                  "text": "Fictional source zebra", "order": 0}
        case = {"case_id": cid, "group_id": group, "subject_id": group,
                "dataset_key": "personamem_v1", "task": "recall_user_shared_facts", "history": [source]}
        cases.append(case)
        refs.append({"case_id": cid, "group_id": group, "dataset_key": "personamem_v1",
                     "task": case["task"], "metric": "personamem_option_exact_match", "correct_option": "a"})
        did = scoring._official().stable_digest([group, source["session_id"], source["turn_id"]])
        source_sha = scoring.sha_bytes(source["text"].encode())
        trace = {"source_catalog": [{"source_id": did, "source_text_sha256": source_sha,
                                    "role": "user", "source_order": 0}],
                 "included_source_spans": [{"source_id": did, "start": 0, "end": 9, "role": "user",
                      "source_text_sha256": source_sha, "quoted_text_sha256": scoring.sha_bytes(b"Fictional"),
                      "complete": True, "offset_unit": "unicode_characters"}],
                 "included_source_ids": [did], "pipeline_valid": n != 3,
                 "semantic_pipeline_status": "valid" if n != 3 else "extraction_invalid",
                 "source_text": "NEVER_PUBLIC_SOURCE_SENTINEL"}
        prep = {}
        for arm in scoring.ARMS:
            messages = [{"role": "user", "content": f"Fictional prompt {cid} {arm}"}]
            prompt_sha = scoring.stable(messages)
            raw = "(a)" if arm == scoring.ARMS[0] or n != 1 else "(b)"
            if n == 3 and arm == scoring.ARMS[0]: raw = None
            prep[arm] = {"messages": messages, "prompt_sha256": prompt_sha, "trace": deepcopy(trace)}
            rows.append({"case_id": cid, "group_id": group, "arm": arm, "request_key": prompt_sha,
                         "completion_status": "success" if raw is not None else "provider_failure",
                         "raw_response": raw, "response_sha256": scoring.stable(raw) if raw is not None else None,
                         "prompt_sha256": prompt_sha, "trace": deepcopy(trace)})
        prepared.append({"case_id": cid, "group_id": group, "prepared_by_arm": prep})
    def cost(attempts, success, usage, unknown):
        return {"provider_attempts": attempts, "successful_calls": success, "failed_calls": attempts-success,
                "reported_usage": dict(zip(("prompt_tokens", "completion_tokens", "total_tokens"), usage)),
                "usage_missing_or_partial_calls": unknown, "billable_usage_unknown_calls": unknown,
                "unresolved_inflight_calls": 0}
    costs = {**cost(8, 7, (80, 14, 94), 1), "by_purpose": {
        "extraction": cost(2, 2, (20, 4, 24), 0), "response": cost(6, 5, (60, 10, 70), 1)}}
    protocol = {"fixture_only": True, "arms": list(scoring.ARMS), "case_ids": [c["case_id"] for c in cases],
                "protocol_sha256": "fictional-fixture-protocol", "evaluation_policy": {
                    "bootstrap_seed": 42, "bootstrap_samples": 200}}
    return protocol, cases, refs, rows, costs, prepared


def score(data):
    return scoring.score_loaded_run(*data[:5], prepared_rows=data[5], bootstrap_samples=200, fixture=True)


class PersonaManagementScoringTests(unittest.TestCase):
    def test_all_rows_failure_and_invalid_pipeline_remain_in_primary_denominator(self):
        summary, rows = score(fixture())
        self.assertEqual(summary["status"], "complete")
        self.assertFalse(summary["final_benchmark_result"])
        self.assertEqual(len(rows), 6)
        raw, managed = (summary["conditions"][arm] for arm in scoring.ARMS)
        self.assertEqual(raw["planned_questions"], 3)
        self.assertEqual(raw["technical_failures"], 1)
        self.assertAlmostEqual(raw["accuracy_micro"], 2/3)
        self.assertAlmostEqual(raw["accuracy_group_equal"], .5)
        self.assertAlmostEqual(managed["accuracy_group_equal"], .75)
        self.assertEqual(managed["pipeline_invalid_questions"], 1)
        pair = summary["primary_paired_comparison"]
        self.assertEqual((pair["wins"], pair["ties"], pair["losses"]), (1, 1, 1))
        self.assertAlmostEqual(pair["mean_accuracy_delta_group_equal"], .25)
        self.assertAlmostEqual(pair["mean_accuracy_delta_micro"], 0)
        self.assertEqual(pair["connected_source_groups"], 2)
        self.assertEqual(pair["interval_status"], "exploratory_percentile_cluster_bootstrap")
        self.assertFalse(summary["clinical_or_user_benefit_claim"])

    def test_partial_never_reads_or_requires_gold_or_produces_comparison(self):
        data = fixture(); data[2].clear(); data[3].pop()
        summary, rows = score(data)
        self.assertEqual(summary["status"], "partial")
        self.assertEqual(rows, [])
        self.assertNotIn("conditions", summary)
        self.assertFalse(summary["comparative_scores_available"])
        self.assertEqual(summary["remaining_arm_rows"], 1)

    def test_unknown_status_duplicates_and_missing_references_rejected(self):
        for mutate in (lambda d:d[3].append(deepcopy(d[3][0])),
                       lambda d:d[3][0].update(completion_status="invented_success"),
                       lambda d:d[2].pop(),
                       lambda d:d[2][0].update(group_id="wrong-group")):
            data=fixture(); mutate(data)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): score(data)

    def test_original_parser_is_primary_even_when_secondary_disagrees(self):
        data=fixture(); row=data[3][0]
        row["raw_response"]="(a) earlier <final_answer>b</final_answer>"
        row["response_sha256"]=scoring.stable(row["raw_response"])
        _,rows=score(data)
        target=rows[0]
        self.assertEqual(target["official_score"],1)
        self.assertEqual(target["strict_committed_score"],0)
        self.assertTrue(target["official_positive_not_strict"])

    def test_answer_response_hash_and_prepared_trace_binding_rejected_on_tamper(self):
        for mutate in (lambda d:d[3][0].update(raw_response="(d)"),
                       lambda d:d[3][0].update(prompt_sha256="different"),
                       lambda d:d[3][0]["trace"].update(pipeline_valid=False)):
            data=fixture(); mutate(data)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): score(data)

    def test_exact_source_role_offset_and_sha_are_independently_bound(self):
        for key,value in (("start",1),("end",10),("role","assistant"),
                          ("quoted_text_sha256","0"*64),("source_text_sha256","0"*64),
                          ("source_id","unknown-source")):
            data=fixture();span=data[3][0]["trace"]["included_source_spans"][0];span[key]=value
            data[5][0]["prepared_by_arm"][scoring.ARMS[0]]["trace"]=deepcopy(data[3][0]["trace"])
            with self.subTest(key=key),self.assertRaises(ValueError):score(data)

    def test_shared_canonical_answers_cannot_diverge(self):
        data=fixture();data[3][1]["request_key"]=data[3][0]["request_key"]
        with self.assertRaises(ValueError):score(data)
        data=fixture()
        for n in range(0,6,2):
            raw,managed=data[3][n:n+2]
            for field in ("request_key","raw_response","response_sha256","completion_status","prompt_sha256"):
                managed[field]=raw[field]
            data[5][n//2]["prepared_by_arm"][scoring.ARMS[1]]["prompt_sha256"]=raw["prompt_sha256"]
        summary,_=score(data)
        comparison=summary["primary_paired_comparison"]
        self.assertEqual(comparison["shared_canonical_response_cases"],3)
        self.assertFalse(comparison["management_input_contrast_present"])
        self.assertIsNone(comparison["group_equal_95_percent_interval"])

    def test_public_projection_omits_query_gold_response_and_unknown_trace_text(self):
        summary,rows=score(fixture());safe=scoring.public_rows(rows)
        encoded=json.dumps(safe)
        self.assertNotIn("parsed_prediction",encoded)
        self.assertNotIn("raw_response",encoded)
        self.assertNotIn("correct_option",encoded)
        self.assertNotIn("NEVER_PUBLIC_SOURCE_SENTINEL",encoded)
        self.assertEqual(safe[0]["trace"]["included_span_bindings_verified"],1)
        self.assertFalse(safe[0]["trace"]["semantic_sufficiency_established"])

    def test_actual_cost_counts_extraction_failures_and_unknown_usage(self):
        data=fixture();cost=score(data)[0]["cost"]
        self.assertEqual(cost["provider_attempts"],8)
        self.assertEqual(cost["by_purpose"]["extraction"]["provider_attempts"],2)
        self.assertEqual(cost["failed_calls"],1)
        self.assertEqual(cost["billable_usage_unknown_calls"],1)
        self.assertIsNone(cost["monetary_cost"])
        self.assertFalse(cost["total_compute_matched"])
        data[4]["provider_attempts"]+=1
        with self.assertRaises(ValueError):score(data)

    def test_guards_precede_controlled_reads_and_fixture_cannot_be_promoted(self):
        with self.assertRaises(RuntimeError):scoring.score_run('/no-run','/no-suite')
        with self.assertRaises(RuntimeError):scoring.export_run('/no-run','/no-output')
        data=fixture()
        with self.assertRaises(RuntimeError):scoring.score_loaded_run(*data[:5],prepared_rows=data[5])
        data=fixture();data[1][0]['history'][0]['text']='x'*12001
        with self.assertRaisesRegex(ValueError,'tiny invented'):score(data)

    def test_actual_adapter_long_source_spans_bind_source_not_each_chunk_catalog_row(self):
        from tests.test_persona_management_adapter import case, settings, empty_outputs
        from persona_management_adapter import prepare_scope_plan, prepare_arms
        example=case(prefix=2,rows=[('user','  '+'fictional swimming preference '*40+'  '),('user','hello')])
        plan=prepare_scope_plan([example],chunk_chars=128)
        prepared=prepare_arms(example,plan=plan,extraction_outputs=empty_outputs(plan),settings=settings())
        for arm in scoring.ARMS:
            trace=prepared['prepared_by_arm'][arm]['trace']
            self.assertGreater(len(trace['included_source_spans']),1)
            result=scoring.verify_source_bindings(example,trace,fixture=True)
            self.assertEqual(result['source_catalog_bindings_verified'],2)
            projected=scoring.safe_trace(trace)['included_source_spans'][0]
            self.assertEqual(projected['normalization'],'legacy_document_strip')
            self.assertIn('original_start',projected)
            self.assertIn('original_source_text_sha256',projected)
        trace=deepcopy(prepared['prepared_by_arm'][scoring.ARMS[0]]['trace'])
        trace['included_source_spans'][0]['original_start']+=1
        with self.assertRaises(ValueError):scoring.verify_source_bindings(example,trace,fixture=True)

    def test_deleted_or_future_source_cannot_be_reported_as_eligible_evidence(self):
        data=fixture();case=data[1][0];trace=data[3][0]['trace']
        case['history'][0]['lifecycle_status']='deleted'
        with self.assertRaises(ValueError):scoring.verify_source_bindings(case,trace,fixture=True)
        case['history'][0].pop('lifecycle_status');case['visible_context_end_exclusive']=0
        with self.assertRaises(ValueError):scoring.verify_source_bindings(case,trace,fixture=True)

    def test_reference_shard_hash_and_case_binding_can_be_verified_on_tiny_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'references.jsonl'
            path.write_text(json.dumps(fixture()[2][0])+'\n')
            protocol={'original_suite':str(root),'reference_files':[{'relative_path':'references.jsonl',
                      'sha256':scoring.sha_bytes(path.read_bytes()),'case_ids':['fixture:1']}]}
            self.assertEqual(len(scoring._load_references(protocol,None)),1)
            protocol['reference_files'][0]['case_ids']=['other']
            with self.assertRaises(ValueError):scoring._load_references(protocol,None)
            with self.assertRaises(ValueError):scoring._load_references(protocol,root/'another')

    def test_zip_publish_refuses_overwrite_and_checksums_can_be_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'controlled.zip'
            scoring._zip_publish(path,{'safe/fixture.json':b'{}'})
            with zipfile.ZipFile(path) as archive:
                self.assertEqual(archive.namelist(),['safe/fixture.json'])
                self.assertEqual(archive.read('safe/fixture.json'),b'{}')
            with self.assertRaises(FileExistsError):scoring._zip_publish(path,{'other':b'no'})

    def test_baseline_gate_covers_all_cases_and_binds_actual_raw_prompt(self):
        data=fixture();ids=data[0]['case_ids']
        gate={'schema':'persona-full-manager-quote-control-gate-v1','case_count':3,
              'new_raw_equals_frozen_old_raw_cases':3,'empty_semantic_manager_equals_raw_cases':3,
              'new_provider_calls':0,'case_rows':[{'case_id':p['case_id'],
                  'prompt_sha256':p['prepared_by_arm'][scoring.ARMS[0]]['prompt_sha256'],
                  'new_raw_equals_frozen_old_raw':True,'empty_semantic_manager_equals_raw':True}
                  for p in data[5]]}
        self.assertEqual(scoring.verify_quote_control_gate(gate,ids,data[5])['case_count'],3)
        gate['case_rows'][0]['prompt_sha256']='different'
        with self.assertRaises(ValueError):scoring.verify_quote_control_gate(gate,ids,data[5])

    def test_restored_archive_empty_locks_allow_provider_free_real_lock_and_scoring_replay(self):
        # Use the actual runner's rb shared-lock path, not a fake lock or a
        # monkeypatched Colab guard. All contents remain explicit tiny fixtures.
        from persona_management_runner import lock, retain_original_locks
        data=fixture()
        expected_summary,expected_rows=score(data)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);original_run=root/'original-run';suite=root/'original-suite'
            original_run.mkdir();suite.mkdir()
            (original_run/'.execution.lock').write_bytes(b'')
            (suite/'.execution.lock').write_bytes(b'')
            descriptors=[];source_members={};restore={}
            for index,subset in enumerate((data[2][:1],data[2][1:2],data[2][2:],[])):
                relative=f'shards/persona-{index:02d}/references.jsonl'
                path=suite/relative;path.parent.mkdir(parents=True)
                body=''.join(json.dumps(ref)+'\n' for ref in subset).encode();path.write_bytes(body)
                (path.parent/'.execution.lock').write_bytes(b'')
                descriptors.append({'relative_path':relative,'sha256':scoring.sha_bytes(body),
                                    'case_ids':[ref['case_id'] for ref in subset]})
                source_members['source_suite/'+relative]=body
                restore['source_suite/'+relative]=str(path)
            data[0].update(original_suite=str(suite),reference_files=descriptors)
            locks,lock_restore=scoring.replay_lock_members(original_run,data[0])
            self.assertEqual(len(locks),6)
            self.assertTrue(all(body==b'' for body in locks.values()))
            restore.update(lock_restore)
            members={**source_members,**locks}
            for name,value in (('protocol.json',data[0]),('cases.json',data[1]),
                               ('generation.json',data[3]),('cost.json',data[4]),('prepared.json',data[5])):
                members['run/'+name]=json.dumps(value).encode()
            archive_path=root/'controlled.zip';scoring._zip_publish(archive_path,members)
            shutil.rmtree(original_run);shutil.rmtree(suite)
            restored_run=root/'restored-run';restored_run.mkdir()
            with zipfile.ZipFile(archive_path) as archive:
                for name in archive.namelist():
                    target=restored_run/name.removeprefix('run/') if name.startswith('run/') else Path(restore[name])
                    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.read(name))
            self.assertFalse(original_run.exists())
            # The suite and all four shard locks were absent after original deletion;
            # the copied locks now make the actual read-only entry API usable.
            restored_protocol=json.loads((restored_run/'protocol.json').read_text())
            last_lock=(suite/descriptors[-1]['relative_path']).parent/'.execution.lock'
            last_lock.unlink()
            with self.assertRaises(FileNotFoundError),ExitStack() as stack:
                stack.enter_context(lock(restored_run));retain_original_locks(restored_protocol,stack)
            last_lock.write_bytes(b'')
            with ExitStack() as stack:
                stack.enter_context(lock(restored_run));retain_original_locks(restored_protocol,stack)
                refs=scoring._load_references(restored_protocol,None)
                summary,rows=scoring.score_loaded_run(restored_protocol,
                    json.loads((restored_run/'cases.json').read_text()),refs,
                    json.loads((restored_run/'generation.json').read_text()),
                    json.loads((restored_run/'cost.json').read_text()),
                    prepared_rows=json.loads((restored_run/'prepared.json').read_text()),
                    bootstrap_samples=200,fixture=True)
            self.assertEqual(summary['conditions'],expected_summary['conditions'])
            self.assertEqual(summary['primary_paired_comparison'],expected_summary['primary_paired_comparison'])
            self.assertEqual(rows,expected_rows)
            (suite/'.execution.lock').write_bytes(b'NONEMPTY_FIXTURE_NOT_EXPORTED')
            with self.assertRaises(ValueError):scoring.replay_lock_members(restored_run,restored_protocol)


if __name__ == '__main__':unittest.main()
