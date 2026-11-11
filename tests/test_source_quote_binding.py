"""Invented quotation fixtures only; no corpus, gold, provider or I/O."""
import importlib.util
import json
from pathlib import Path
import unittest

SPEC=importlib.util.spec_from_file_location("source_quote_binding",Path(__file__).resolve().parents[1]/
    "modules/agent-integration/source_quote_binding.py")
M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)


class SourceQuoteBindingTests(unittest.TestCase):
    def catalog(self):
        return [{"source_id":"source-A","role":"user","text":"I have not agreed to continue."},
                {"source_id":"source-B","role":"user","text":"I completed one practice."}]
    def answer(self,quote="I completed one practice.",source_id="source-B",unknown=False):
        return json.dumps({"quote":quote,"source_id":source_id,"unknown":unknown})

    def test_coherent_binding_retains_original_bytes(self):
        original='  '+self.answer()+'\n'
        result=M.repair_source_quote_binding(original,self.catalog())
        self.assertEqual(result["output_text"],original)
        self.assertFalse(result["changed"])
        self.assertEqual(result["action"],"passed_through")

    def test_unique_complete_quote_repairs_id_only_with_whitespace_normalization(self):
        original=self.answer(quote="I  completed\none practice.",source_id="wrong")
        result=M.repair_source_quote_binding(original,self.catalog())
        self.assertEqual(json.loads(result["output_text"]),
                         {"quote":"I  completed\none practice.","source_id":"source-B","unknown":False})
        self.assertEqual(result["action"],"repaired_source_id")

    def test_ambiguous_quote_does_not_guess_but_valid_declared_binding_is_retained(self):
        catalog=self.catalog()+[{"source_id":"source-C","role":"user","text":"I completed one practice."}]
        bad=M.repair_source_quote_binding(self.answer(source_id="wrong"),catalog)
        self.assertTrue(json.loads(bad["output_text"])["unknown"])
        self.assertEqual(bad["reason"],"ambiguous_visible_full_quote")
        original=self.answer(source_id="source-C")
        self.assertEqual(M.repair_source_quote_binding(original,catalog)["output_text"],original)

    def test_original_unknown_never_becomes_known(self):
        original=self.answer(quote="",source_id="",unknown=True)
        result=M.repair_source_quote_binding(original,self.catalog())
        self.assertEqual(result["output_text"],original)
        incoherent=M.repair_source_quote_binding(self.answer(unknown=True),self.catalog())
        self.assertTrue(json.loads(incoherent["output_text"])["unknown"])
        self.assertEqual(incoherent["action"],"rejected_to_unknown")

    def test_partial_or_punctuation_changed_quotes_cannot_be_repaired(self):
        for quote in ("completed one practice", "I completed one practice!", "I have agreed to continue."):
            result=M.repair_source_quote_binding(self.answer(quote=quote,source_id="wrong"),self.catalog())
            self.assertTrue(json.loads(result["output_text"])["unknown"])
            self.assertEqual(result["reason"],"full_quote_not_in_visible_catalog")

    def test_strict_schema_duplicate_fields_and_extra_text_are_unknown(self):
        for original in ('{"quote":"x","quote":"y","source_id":"z","unknown":false}',
                         self.answer()+" trailing", self.answer()[:-1]+',"extra":1}',
                         '{"quote":"x","source_id":"y","unknown":0}', 'NaN'):
            result=M.repair_source_quote_binding(original,self.catalog())
            self.assertEqual(result["action"],"rejected_to_unknown")
            self.assertTrue(json.loads(result["output_text"])["unknown"])

    def test_catalog_must_be_visible_user_sources_and_identifiers_unambiguous(self):
        for catalog in ([{"source_id":"a","role":"assistant","text":"I completed one practice."}],
                        self.catalog()+[{"source_id":"source-A","role":"user","text":"Different text"}]):
            with self.assertRaises(ValueError):
                M.repair_source_quote_binding(self.answer(),catalog)

    def test_empty_visible_catalog_never_recovers_hidden_quote(self):
        result=M.repair_source_quote_binding(self.answer(),[])
        self.assertTrue(json.loads(result["output_text"])["unknown"])
        self.assertEqual(result["matched_visible_source_count"],0)


if __name__=="__main__": unittest.main()
