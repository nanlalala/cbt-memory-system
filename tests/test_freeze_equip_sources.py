"""Invented byte fixtures only. No EQUIP download, source data or model calls."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "freeze_equip_sources", ROOT / "evaluation/cbt-memory-effect/freeze_equip_sources.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class FreezeEquipSourceTests(unittest.TestCase):
    def test_terms_evidence_points_to_exact_original_fixture_bytes(self):
        html = (b'<p>INVENTED TERMS: CC BY-NC-SA 3.0 IGO '
                b'<a href="https://creativecommons.org/licenses/by-nc-sa/3.0/igo/">'
                b'INVENTED LINK</a></p>')
        result = M.validate_terms(html)
        self.assertEqual(result["license"], "CC-BY-NC-SA-3.0-IGO")
        self.assertEqual(result["source_sha256"], M.sha256(html))
        for span in result["original_html_byte_ranges"]:
            exact = html[span["start_byte"]:span["end_byte_exclusive"]]
            self.assertEqual(exact.decode(), span["exact_utf8"])
            self.assertEqual(M.sha256(exact), span["span_sha256"])

    def test_terms_fail_closed_without_visible_identifier_and_license_link(self):
        for html in (b'<p>INVENTED LICENSE MIT</p>',
                     b'<p>CC BY-NC-SA 3.0 IGO</p>',
                     b'<script>CC BY-NC-SA 3.0 IGO</script>'
                     b'<a href="https://creativecommons.org/licenses/by-nc-sa/3.0/igo">x</a>'):
            with self.assertRaisesRegex(ValueError, "license"):
                M.validate_terms(html)

    def test_only_pinned_source_url_can_be_followed(self):
        source = M.SOURCES[1]
        M._check_source_url(source["url"], source)
        M._check_source_url(source["url"].replace("%20", " "), source)
        for url in ("https://example.org/arbitrary.pdf",
                    "https://equipcompetency.org/en-gb/user/login",
                    source["url"] + "?login=1",
                    source["url"].replace("https://", "http://")):
            with self.assertRaises(ValueError):
                M._check_source_url(url, source)

    def test_pdf_signature_rejects_html_and_truncated_bytes(self):
        M.validate_pdf(b"%PDF-1.7\nINVENTED FIXTURE, NOT A VALID DOCUMENT\n%%EOF\n")
        for data in (b"<html>INVENTED ACCESS PAGE</html>", b"%PDF-1.7\nTRUNCATED"):
            with self.assertRaisesRegex(ValueError, "complete_pdf"):
                M.validate_pdf(data)

    def test_all_sources_assessor_only_and_pm_plus_quarantined_contract(self):
        self.assertEqual(len(M.SOURCES), 7)
        self.assertTrue(M.SEPARATION_CONTRACT["complete_source_files_assessor_only"])
        self.assertFalse(M.SEPARATION_CONTRACT["generation_inputs_created"])
        self.assertFalse(M.SEPARATION_CONTRACT["published_independent_memory_gold_exists"])
        self.assertNotIn("actor_hidden", M.SEPARATION_CONTRACT["generation_allowed_classes"])
        self.assertNotIn("future_prompt", M.SEPARATION_CONTRACT["generation_allowed_classes"])
        pm = next(source for source in M.SOURCES if source["id"] == "pm_plus_roleplay")
        self.assertIn("quarantined", pm["role"])

    def test_hosted_guard_precedes_network_and_output_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "must-not-exist"
            with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(M, "_download") as network:
                with self.assertRaisesRegex(RuntimeError, "genuine_hosted_colab"):
                    M.freeze(output)
                self.assertFalse(output.exists())
                network.assert_not_called()

    def test_archive_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.bin"
            M._write(path, b"INVENTED FIRST BYTES")
            with self.assertRaises(FileExistsError):
                M._write(path, b"REPLACEMENT")
            self.assertEqual(path.read_bytes(), b"INVENTED FIRST BYTES")


if __name__ == "__main__":
    unittest.main()
