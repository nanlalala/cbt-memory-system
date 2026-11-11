"""Invented sequencing only: never actual checkpoints, hosted data or models."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import nullcontext

P=Path(__file__).resolve().parents[1]/'evaluation/cbt-memory-effect/continue_open_state_colab.py'
S=importlib.util.spec_from_file_location('continue_open_state_colab',P)
M=importlib.util.module_from_spec(S);S.loader.exec_module(M)

class ContinueTests(unittest.TestCase):
    def test_busy_writer_is_not_read_as_a_partial_journal_failure(self):
        with patch.object(M.reader,'_locked',side_effect=BlockingIOError('busy')),patch.object(M.reader,'read_rows',side_effect=ValueError('partial live line')) as read,patch.object(M.reader,'_global_state') as state,patch.object(M.time,'monotonic',side_effect=[0,0,0,2]),patch.object(M.time,'sleep'):
            with self.assertRaises(TimeoutError):M.wait_readers(timeout=1)
            read.assert_not_called();state.assert_not_called()
    def test_wait_needs_both_finished_and_exact_57(self):
        state=({str(i):{} for i in range(57)},{str(i):{} for i in range(57)})
        with patch.object(M.reader,'_locked',side_effect=lambda p:nullcontext()),patch.object(M.reader,'_global_state',return_value=state),patch.object(M,'finished',return_value=True) as done:
            M.wait_readers(timeout=1)
            self.assertEqual(done.call_count,2)
        with patch.object(M.reader,'_locked',side_effect=lambda p:nullcontext()),patch.object(M.reader,'_global_state',return_value=({},{})),patch.object(M,'finished',return_value=True):
            with self.assertRaisesRegex(RuntimeError,'57'):M.wait_readers(timeout=1)
    def test_failure_stops_without_retry_or_generation(self):
        with patch.object(M.reader,'_locked',side_effect=lambda p:nullcontext()),patch.object(M.reader,'_global_state',side_effect=RuntimeError('failed_no_retry')),patch.object(M.time,'sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError,'failed_no_retry'):M.wait_readers(timeout=1)
            sleep.assert_not_called()
    def test_existing_output_never_starts_subprocess(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'existing';out.write_text('preserved')
            with patch.object(M.subprocess,'run') as run:
                with self.assertRaises(FileExistsError):M.phase(root,'fixture',['python','fake.py'],[out])
                run.assert_not_called()
            self.assertEqual(out.read_text(),'preserved')
    def test_live_guard_before_any_study_file_or_subprocess(self):
        with patch.object(M.reader,'require_colab',side_effect=RuntimeError('hosted_required')),patch.object(M.subprocess,'run') as run:
            with self.assertRaisesRegex(RuntimeError,'hosted_required'):M.main([])
            run.assert_not_called()

if __name__=='__main__':unittest.main()
