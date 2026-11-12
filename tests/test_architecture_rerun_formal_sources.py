"""Source boundary regressions use tiny invented records only."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("formal_architecture_sources",ROOT/"evaluation/architecture-rerun/formal_sources.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
from benchmark_adapters import RetrievalExample


class FormalSourceBoundaryTest(unittest.TestCase):
    def test_peer_speakers_stay_context_and_real_dates_are_preserved(self):
        record = RetrievalExample("fixture:qa:1","1","What happened?","scoring only",(),(
            {"session_id":"fictional-session","date":"2026-11-01T09:00:00Z","turns":[
                {"dia_id":"t1","speaker":"Peer Alice","text":"I went walking."},
                {"dia_id":"t2","speaker":"Peer Bob","text":"How was it?"}]},))
        history, excluded = module.history_at_cutoff(record,"locomo10")
        self.assertEqual(excluded,[])
        self.assertEqual([r['role'] for r in history],['context','context'])
        self.assertEqual([r['source_speaker'] for r in history],['Peer Alice','Peer Bob'])
        self.assertTrue(all(r['date']=='2026-11-01T09:00:00Z' for r in history))
        self.assertNotIn('scoring only',module.canonical(history))

    def test_question_cutoff_excludes_future_and_order_sorts_released_dates(self):
        def session(identity,date):
            return {'session_id':identity,'date':date,'turns':[{'role':'user','content':identity}]}
        record = RetrievalExample('fixture','single-session-user','Question','gold',(),(
            session('later','2026-11-04T09:00:00Z'),session('first','2026-11-01T09:00:00Z'),
            session('future','2026-11-08T09:00:00Z')),query_date='2026-11-05T09:00:00Z')
        history, excluded=module.history_at_cutoff(record,'longmemeval_s_cleaned')
        self.assertEqual([r['content'] for r in history],['first','later'])
        self.assertEqual(excluded,['future'])
        self.assertEqual([r['source_order'] for r in history],[0,1])
