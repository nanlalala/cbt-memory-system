"""Fictional, no-network checks for tentative wording cues."""
from pathlib import Path
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules/agent-integration"))
import cbt_projection_contract as contract
from cbt_wording_cues import wording_cues

SOURCES = [
    {"source_id": "u0", "role": "user", "content": "I did it once, but a colleague interrupted me. Maybe this just isn't for me."},
    {"source_id": "a0", "role": "assistant", "content": "It is not for me to say it is too much; shall we make it smaller?"},
    {"source_id": "u1", "role": "user", "content": "Honestly I'm dreading the meeting."},
    {"source_id": "u2", "role": "user", "content": "I completed the task and it went fine."},
]


class WordingCues(unittest.TestCase):
    def test_items_bind_under_frozen_contract(self):
        items = wording_cues(SOURCES)
        bound = contract.bind_candidates(contract.canonical({"items": items}), SOURCES)
        self.assertEqual(len(bound), len(items))
        for i in bound:
            self.assertEqual((i["section"], i["origin"]), ("working_formulation", "agent_inference"))

    def test_user_sources_only_and_exact_quotes(self):
        items = wording_cues(SOURCES)
        self.assertEqual({e["source_id"] for i in items for e in i["evidence"]}, {"u0", "u1"})
        self.assertEqual([i["value"]["cue"] for i in items], ["possibly discouraged", "possibly anxious about it"])

    def test_no_cue_without_wording(self):
        self.assertEqual(wording_cues([SOURCES[3]]), [])

    def test_never_rates_or_diagnoses(self):
        text = json.dumps(wording_cues(SOURCES)).lower()
        for banned in ("depress", "intensity", "severity", "diagnos", "score"):
            self.assertNotIn(banned, text)


if __name__ == "__main__":
    unittest.main()
