"""Hosted-only, one-shot continuation after both existing readers finish.

Never starts/resumes either reader. Existing phase artifacts cause a stop, not
automatic resending. Optional calibration is the already frozen five, not R1–R5.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import open_reader_colab as reader

C = Path('/content')
MODEL = C/'cbt-open-reader-20261112-v1'
LABELS = ('before', 'after')
PREP = {x:C/f'cbt-open-state-prepared-{x}-20261112-v2' for x in LABELS}
ANS = {x:C/f'cbt-open-state-answers-{x}-20261112-v2' for x in LABELS}
OWN = {x:C/f'cbt-open-state-{x}-20261112-v2/evaluation/cbt-memory-effect/controlled_open_memory.py' for x in LABELS}
SCORES = {x:C/f'cbt-open-state-primary-score-{x}-20261112-v2.json' for x in LABELS}
FORMATS = {x:C/f'cbt-open-state-secondary-format-{x}-20261112-v2.json' for x in LABELS}
CAL_PREP = C/'therapyjudge-five-calibration-prepared-v1'
CAL_OUT = C/'therapyjudge-five-calibration-results-v1'
CAL_SHA = 'e2b8f0b82e8d4175f757291693667bced676b220f8d7182c8cf0fe0569baeb19'
TOKENS = {'Conversation 73':2047, 'Conversation 72':1817, 'Conversation 79':2230,
          'Conversation 78':2031, 'Conversation 83':1844}
MANIFEST = C/'therapyjudgebench-v2-source-freeze-20261112/source_bank_manifest.json'
SCHEMA = MANIFEST.parent/'assessor_only/schema_audit.json'
SOURCE_HASHES = {MANIFEST:'c85f1a454169fcd818a3d3ad2937c205eae651458c0dd91c40a0854ad2be7a04',
                 SCHEMA:'5d883c2ba82907a4322d884ffa1f557f7f97381971039ea7b40562a87a8f5e0e'}


def files(paths):
    return {str(p):reader.file_hash(p) for p in paths if Path(p).is_file()}


def finished(folder, count, event):
    if reader.read_rows(folder/'failures.jsonl'):
        raise RuntimeError('recorded_failure_no_automatic_retry')
    summary = folder/'completion_summary.json'
    if not summary.exists(): return False
    value = reader.read_json(summary)
    if value.get('complete') is not True or value.get('logical_completed') != count:
        raise RuntimeError('settled_incomplete_run_no_automatic_retry')
    if value.get('answers_sha256') != reader.file_hash(folder/'answers_unscored.jsonl'):
        raise ValueError('completed_answers_changed')
    events = [reader._checked(r) for r in reader.read_rows(folder/'execution_events.jsonl')]
    return any(r.get('event') == event and r.get('complete') is True for r in events)


def settled(expected):
    intents, outcomes = reader._global_state(MODEL)
    if len(intents) != expected or len(outcomes) != expected:
        raise RuntimeError('unexpected_global_generation_count_no_retry')
    return intents, outcomes


def wait_readers(timeout=7200, poll=5):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        # Do not take exclusive setup locks in the gap between the two reader
        # processes: their loaders intentionally refuse concurrent ownership.
        if not all((ANS[x]/'completion_summary.json').exists() for x in LABELS):
            if any((ANS[x]/'failures.jsonl').exists() and (ANS[x]/'failures.jsonl').stat().st_size for x in LABELS):
                raise RuntimeError('recorded_failure_no_automatic_retry')
            time.sleep(min(poll,max(0,deadline-time.monotonic())))
            continue
        try:
            with ExitStack() as stack:
                for lock in ('.model_setup.lock', '.generation.lock'):
                    stack.enter_context(reader._locked(MODEL/lock))
                intents, outcomes = reader._global_state(MODEL)
                done = all(finished(ANS[x],48,'fresh_process_reader_finished') for x in LABELS)
                if done:
                    if len(intents) != 57: raise RuntimeError('expected_exactly_57_published_before_calibration')
                    return
        except BlockingIOError:
            # Inspect the journal only after its live writer releases both
            # locks. A partially appended line is not a settled failure.
            pass
        time.sleep(min(poll,max(0,deadline-time.monotonic())))
    raise TimeoutError('readers_not_finished_within_two_hours_no_generation_started')


def phase(journal, name, command, outputs, inputs=()):
    if any(Path(p).exists() for p in outputs):
        raise FileExistsError('existing_phase_output_inspect_do_not_resend: '+name)
    record = {'phase':name,'command_sha256':reader.digest(command), 'inputs_sha256':files(inputs),
              'script_sha256':reader.file_hash(command[1]),'at_utc':reader.stamp()}
    reader._append(journal/'phases.jsonl',{'event':'phase_intent',**record})
    log = journal/(name+'.log')
    with log.open('x') as stream:
        result = subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=False)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    outcome = {'event':'phase_outcome','phase':name,'returncode':result.returncode,
               'outputs_sha256':files(outputs),'log_sha256':reader.file_hash(log),'at_utc':reader.stamp()}
    reader._append(journal/'phases.jsonl',outcome)
    print(json.dumps(outcome,sort_keys=True),flush=True)
    if result.returncode or any(not Path(p).exists() for p in outputs):
        raise RuntimeError('phase_failed_or_missing_output_no_automatic_retry: '+name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal-dir',type=Path,default=C/'cbt-open-state-continuation-20261112-v1')
    parser.add_argument('--run-calibration',action='store_true')
    parser.add_argument('--final-checkpoint',action='store_true')
    parser.add_argument('--include-v1-resources',action='store_true',help='Also audit old v1 preparation metadata; old run allocation is not inferred.')
    a = parser.parse_args(argv)
    reader.require_colab()  # Before study paths, locks, journals or subprocesses.
    if a.final_checkpoint and not a.run_calibration: raise ValueError('final_requires_requested_calibration')
    if not a.journal_dir.resolve().is_relative_to(C) or a.journal_dir.resolve()==C:
        raise ValueError('named_hosted_output_required')
    if a.journal_dir.exists(): raise FileExistsError('existing_orchestration_inspect_no_automatic_resume')
    planned = list(SCORES.values())+list(FORMATS.values())+[C/'cbt-open-state-resource-audit-57-20261112-v2.json',
        C/'CBT_Open_State_Complete_Paper_Evidence_20261112.json.gz']
    if a.run_calibration: planned += [CAL_OUT,C/'cbt-open-state-resource-audit-62-20261112-v2.json']
    if a.final_checkpoint: planned += [C/'cbt-open-state-final-checkpoint-20261112-v2',C/'cbt-open-state-final-checkpoint-20261112-v2.zip']+[C/f'cbt-open-state-final-source-fence-{x}-20261112-v2.json' for x in LABELS]
    if any(p.exists() for p in planned):
        print(json.dumps({'existing_phase_artifacts':files(planned),'automatic_resume':False},sort_keys=True))
        raise FileExistsError('existing_phase_artifacts_inspect_do_not_resend')
    a.journal_dir.mkdir(mode=0o700,parents=True)
    reader._write_new(a.journal_dir/'orchestration.json',{'version':'open-state-sequential-continuation-v1',
        'code_sha256':reader.file_hash(__file__),'run_calibration':a.run_calibration,
        'automatic_retry':False,'maximum_global_generation_attempts':64,'started_at_utc':reader.stamp()})
    wait_readers()
    py = sys.executable
    format_commands = {}
    for label in LABELS:
        phase(a.journal_dir,'score_'+label,[py,str(OWN[label]),'score','--prepared-dir',str(PREP[label]),
            '--answers',str(ANS[label]/'answers_unscored.jsonl'),'--output',str(SCORES[label]),'--model-root',str(MODEL)],
            [SCORES[label]],[PREP[label]/'protocol.json',ANS[label]/'answers_unscored.jsonl'])
        format_commands[label] = [py,str(HERE/'audit_open_reader_response_format.py'),
            '--prepared-dir',str(PREP[label]),'--answers',str(ANS[label]/'answers_unscored.jsonl'),
            '--primary-score',str(SCORES[label]),'--expected-score-sha256',reader.file_hash(SCORES[label]),
            '--planner-path',str(OWN[label]),'--model-root',str(MODEL),'--output',str(FORMATS[label])]
        phase(a.journal_dir,'format_'+label,format_commands[label],
            [FORMATS[label]],[SCORES[label]])
    def resource(suffix):
        out = C/f'cbt-open-state-resource-audit-{suffix}-20261112-v2.json'
        command = [py,str(HERE/'audit_open_state_resources.py'),'--model-root',str(MODEL),'--output',str(out)]
        for label in LABELS:
            command += ['--prepared-dir',str(PREP[label]),'--reader-output',str(ANS[label])]
            if a.include_v1_resources:
                command += ['--prepared-dir',str(C/f'cbt-open-state-prepared-{label}-20261112-v1')]
        phase(a.journal_dir,'resource_'+suffix,command,[out]); return out
    resource57 = resource('57')
    for path, expected in SOURCE_HASHES.items():
        if reader.file_hash(path)!=expected: raise ValueError('clinical_source_identity_changed')
    export = C/'CBT_Open_State_Complete_Paper_Evidence_20261112.json.gz'
    command = [py,str(HERE.parents[1]/'tools/revise_capstone_from_controlled_results.py'),'--export-evidence',str(export)]
    for label in LABELS:
        command += [f'--{label}-report',str(SCORES[label]),f'--{label}-report-sha256',reader.file_hash(SCORES[label]),
            f'--{label}-protocol',str(PREP[label]/'protocol.json'),f'--{label}-reader-run',str(ANS[label]/'run_protocol.json'),
            f'--{label}-reader-completion',str(ANS[label]/'completion_summary.json'),f'--{label}-answers',str(ANS[label]/'answers_unscored.jsonl'),
            f'--{label}-format',str(FORMATS[label]),f'--{label}-format-sha256',reader.file_hash(FORMATS[label])]
    command += ['--calibration-manifest',str(MANIFEST),'--calibration-manifest-sha256',SOURCE_HASHES[MANIFEST],
        '--calibration-audit',str(SCHEMA),'--calibration-audit-sha256',SOURCE_HASHES[SCHEMA],
        '--evidence-extra','reader_cost_summary='+str(resource57),'--evidence-extra','reader_global_ledger='+str(MODEL/'generation_ledger.jsonl')]
    phase(a.journal_dir,'export_complete_evidence',command,[export],list(SCORES.values())+list(FORMATS.values()))
    if a.run_calibration:
        with ExitStack() as stack:
            for lock in ('.model_setup.lock','.generation.lock'): stack.enter_context(reader._locked(MODEL/lock))
            settled(57)
        if reader.file_hash(CAL_PREP/'protocol.json')!=CAL_SHA: raise ValueError('frozen_five_protocol_changed')
        preflight = reader.read_json(C/'therapyjudge-five-calibration-token-preflight-v2.json')
        if (preflight.get('stage')!='clinical_judge_actual_tensor_token_preflight' or preflight.get('all_five_fit_4096') is not True
            or len(preflight.get('cases',[]))!=5 or {r['case_id']:r['input_tokens'] for r in preflight.get('cases',[])}!=TOKENS
            or preflight.get('answer_generations')!=0 or preflight.get('model_weights_loaded') is not False or preflight.get('paid_api_calls')!=0
            or preflight.get('requests_sha256')!='2b42c037efe4244a4f1283c212be65c41f0856f9df9bbe46e0af1da0c9368171'):
            raise ValueError('actual_tensor_v2_preflight_required')
        if reader.file_hash(CAL_PREP/'requests.jsonl')!=preflight['requests_sha256']: raise ValueError('preflight_requests_changed')
        if CAL_OUT.exists(): raise FileExistsError('existing_calibration_do_not_resume_or_resend')
        phase(a.journal_dir,'calibration_run',[py,str(HERE/'therapyjudge_calibration_colab.py'),'run',
            '--prepared-dir',str(CAL_PREP),'--model-root',str(MODEL),'--output',str(CAL_OUT),'--max-new-generations','5'],
            [CAL_OUT/x for x in ('run_protocol.json','completion_summary.json','answers_unscored.jsonl')],
            [CAL_PREP/'protocol.json',C/'therapyjudge-five-calibration-token-preflight-v2.json'])
        settled(62)
        if not finished(CAL_OUT,5,'calibration_fresh_process_finished'): raise RuntimeError('five_calibration_not_finished')
        phase(a.journal_dir,'calibration_score',[py,str(HERE/'therapyjudge_calibration_colab.py'),'score',
            '--prepared-dir',str(CAL_PREP),'--model-root',str(MODEL),'--output',str(CAL_OUT)],
            [CAL_OUT/'calibration_score.json'],[CAL_OUT/'completion_summary.json'])
        resource('62')
    if a.final_checkpoint:
        settled(62)
        for path, expected in SOURCE_HASHES.items():
            if reader.file_hash(path)!=expected: raise ValueError('clinical_source_changed_before_final_checkpoint')
        # Recheck every current source gate in its own frozen product process
        # after calibration. Reuse scoring bytes; never generate answers again.
        for label in LABELS:
            output = C/f'cbt-open-state-final-source-fence-{label}-20261112-v2.json'
            command = format_commands[label][:-1]+[str(output)]
            phase(a.journal_dir,'final_source_fence_'+label,command,[output],[SCORES[label]])
        phase(a.journal_dir,'final_checkpoint',[py,str(HERE/'checkpoint_open_state_colab.py'),
            '--output',str(C/'cbt-open-state-final-checkpoint-20261112-v2'),
            '--archive',str(C/'cbt-open-state-final-checkpoint-20261112-v2.zip'),'--final'],
            [C/'cbt-open-state-final-checkpoint-20261112-v2.zip'])
    print(json.dumps({'stage':'continuation_complete','journal_dir':str(a.journal_dir),'evidence_sha256':reader.file_hash(export),
        'calibration_executed':a.run_calibration,'new_provider_calls':0,'automatic_retry':False},sort_keys=True))


if __name__=='__main__': main()
