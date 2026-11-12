"""Large invented source JSONL must never use model-output or Unicode line rules."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"evaluation/cbt-memory-effect"))
from architecture_retest.journal import document_json, rows


class PersistedDocumentTest(unittest.TestCase):
    def test_megabyte_history_with_unicode_separators_remains_one_complete_source_row(self):
        text="invented "*120000+"\u2028not a JSONL boundary\u2029still the same source\u0085end"
        source={"history_key":"invented-large-history","history":[{"source_id":"invented-source","content":text}]}
        wire=(json.dumps(source,ensure_ascii=False)+"\n").encode("utf-8")
        self.assertGreater(len(wire),1_000_000)
        self.assertGreater(len(wire.decode().splitlines()),1)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"shared_histories.jsonl"
            path.write_bytes(wire)
            self.assertEqual(rows(path),[source])
            self.assertEqual(path.read_bytes(),wire)

    def test_physical_newlines_and_incomplete_source_are_checked_without_repair(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"source.jsonl"
            path.write_text('{"id":1}\n{"id":2}\n')
            self.assertEqual(rows(path),[{"id":1},{"id":2}])
            path.write_text('{"id":1}\n{"id":2}')
            with self.assertRaisesRegex(ValueError,"incomplete journal"):
                rows(path)
            path.write_text('{"content":"unfinished\n')
            with self.assertRaises(ValueError):
                rows(path)

    def test_document_parser_rejects_duplicate_keys_and_nonfinite_numbers(self):
        for value in ('{"id":1,"id":2}','{"value":NaN}','{"value":Infinity}','{"value":1e9999}'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                document_json(value)


if __name__=="__main__":
    unittest.main()
