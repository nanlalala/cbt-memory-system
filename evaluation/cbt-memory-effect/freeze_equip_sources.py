"""Freeze official EQUIP professional sources in genuine hosted Colab only.

Source-bank acquisition, not a memory benchmark or a source of new gold labels.
All complete PDFs remain assessor-only: helper-visible instructions, hidden
actor information, future prompts and scoring guidance have not been separated.
The internally conflicting PM+ script is additionally quarantined. No PDF text
extraction, adaptation, model call, credential access or automatic retry occurs.

Example in hosted Colab (a fresh /content output directory is mandatory)::

    !python evaluation/cbt-memory-effect/freeze_equip_sources.py \\
        --output /content/equip-source-freeze-20261112-v1

The platform URLs are mutable. This collector establishes an exact-byte local
freeze; it does not pretend that an unobserved SHA256 was known in advance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import importlib
import importlib.util
import json
import os
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


VERSION = "equip-professional-source-bank-v1"
LICENSE = "CC-BY-NC-SA-3.0-IGO"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/3.0/igo"
TERMS_URL = "https://equipcompetency.org/en-gb/policies/terms-of-use"
MAX_SOURCE_BYTES = 12_000_000
SOURCES = (
    {"id": "terms", "filename": "terms-of-use.html", "kind": "html",
     "url": TERMS_URL, "role": "license_evidence"},
    {"id": "enact_roleplay", "filename": "enact_roleplay.pdf", "kind": "pdf",
     "url": "https://equipcompetency.org/sites/default/files/downloads/2022-08/ENACT%20Role%20Play%20Instructions%20&%20Scripts_platform_220830.pdf",
     "role": "professional_roleplay_source_pending_span_audit"},
    {"id": "cognitive_prompts", "filename": "cognitive_prompts.pdf", "kind": "pdf",
     "url": "https://equipcompetency.org/sites/default/files/downloads/2022-08/Cog%20competency%20descrip_single%20role%20plays_v1.0_220320.pdf",
     "role": "professional_cognitive_criteria_and_roleplay_prompts"},
    {"id": "ba_assessment", "filename": "ba_assessment.pdf", "kind": "pdf",
     "url": "https://whoequipremote.org/sites/default/files/downloads/2026-04/BA_EN_260417.pdf",
     "role": "professional_behavioral_activation_criteria"},
    {"id": "pm_plus_roleplay", "filename": "pm_plus_roleplay.pdf", "kind": "pdf",
     "url": "https://equipcompetency.org/sites/default/files/downloads/2022-08/EQUIP%20NS%20NYC%20PM%2B%20Role%20Play%20Scripts_v21Jan_Clean220307.pdf",
     "role": "quarantined_conflicting_sam_reese_contexts"},
    {"id": "ba_prompts", "filename": "ba_prompts.pdf", "kind": "pdf",
     "url": "https://equipcompetency.org/sites/default/files/downloads/2022-08/BA%20competency%20descrip_single%20role%20plays_v1.0_220320.pdf",
     "role": "professional_behavioral_activation_roleplay_prompts"},
    {"id": "problem_solving_prompts", "filename": "problem_solving_prompts.pdf", "kind": "pdf",
     "url": "https://whoequipremote.org/sites/default/files/downloads/2022-08/Problem%20Solving%20competency%20descrip_single%20role%20plays_v1.0__220320.pdf",
     "role": "professional_problem_solving_roleplay_prompts"},
)

# No source is a ready-made generation input. A later, separately versioned
# adaptation must establish these boundaries from exact archived source spans.
SEPARATION_CONTRACT = {
    "complete_source_files_assessor_only": True,
    "generation_inputs_created": False,
    "required_span_classes": ["helper_visible", "actor_disclosed_before_cutoff",
                              "actor_hidden", "future_prompt", "assessor_guidance"],
    "generation_allowed_classes": ["helper_visible", "actor_disclosed_before_cutoff"],
    "unknown_or_unclassified_span_policy": "exclude_from_generation",
    "each_allowed_span_requires": ["source_sha256", "page", "exact_source_span",
                                   "disclosure_event", "cutoff_order"],
    "forbidden_in_generation": ["undisclosed_actor_background", "future_prompts",
                                "reference_responses", "rating_guidance"],
    "no_inferred_dates_sessions_or_client_agreement": True,
    "conditional_scripts_are_chronological_transcripts": False,
    "derived_messages_require_explicit_adaptation_and_source_map": True,
    "all_arms_require_equal_history_recent_buffer_and_budget": True,
    "published_independent_memory_gold_exists": False,
    "new_human_or_clinical_review_exists": False,
    "pm_plus_policy": "quarantine_pending_authoritative_resolution_or_labelled_adaptation",
    "pm_plus_observed_issue": "helper_prompts_Sam_vs_adjacent_hidden_background_Reese",
    "manual_exposure": {
        "pm_plus_roleplay": "development_exposed_contexts",
        "cognitive_prompts": "development_exposed_examples",
        "enact_roleplay": "metadata_and_structure_seen_not_verified_holdout",
        "other_sources": "prior_exposure_must_be_checked_not_verified_holdout"},
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def require_colab():
    """Runtime guard, not cryptographic attestation; before any I/O in freeze()."""
    available = False
    try:
        available = (bool(os.environ.get("COLAB_RELEASE_TAG"))
                     and Path("/content").is_dir()
                     and importlib.util.find_spec("google.colab") is not None)
        if available:
            importlib.import_module("google.colab")
    except (ImportError, ModuleNotFoundError, ValueError, AttributeError):
        available = False
    if not available:
        raise RuntimeError("genuine_hosted_colab_required_before_any_request")


def _url_key(url):
    parts = urlsplit(url)
    if (parts.scheme != "https" or parts.username or parts.password
            or parts.port not in (None, 443) or parts.query or parts.fragment):
        raise ValueError("unexpected_source_url")
    return parts.hostname, unquote(parts.path).rstrip("/")


def _check_source_url(url, source):
    # Only the exact prespecified resource, not arbitrary same-host URLs or a
    # login/error page. Equivalent percent encoding and trailing slash are OK.
    if _url_key(url) != _url_key(source["url"]):
        raise ValueError("unexpected_source_redirect")


class _PinnedRedirect(HTTPRedirectHandler):
    def __init__(self, source):
        super().__init__()
        self.source = source

    def redirect_request(self, request, fp, code, message, headers, newurl):
        _check_source_url(urljoin(request.full_url, newurl), self.source)
        return super().redirect_request(request, fp, code, message, headers, newurl)


def _download(source):
    _check_source_url(source["url"], source)
    request = Request(source["url"], headers={
        "User-Agent": "Capstone-EQUIP-Source-Freeze/1",
        "Accept-Encoding": "identity"})
    # No cookies, authorization headers, API tokens or per-user credential files.
    opener = build_opener(_PinnedRedirect(source))
    with opener.open(request, timeout=45) as response:
        _check_source_url(response.geturl(), source)
        declared = response.headers.get("Content-Length")
        if declared is not None and not 0 < int(declared) <= MAX_SOURCE_BYTES:
            raise ValueError("declared_source_size_outside_bound")
        encoding = response.headers.get("Content-Encoding", "identity").lower()
        if encoding not in ("", "identity"):
            raise ValueError("encoded_transfer_not_exact_source_bytes")
        data = response.read(MAX_SOURCE_BYTES + 1)
        if not data or len(data) > MAX_SOURCE_BYTES:
            raise ValueError("source_size_outside_bound_or_empty")
        if declared is not None and len(data) != int(declared):
            raise ValueError("incomplete_source_transfer")
        metadata = {"final_url": response.geturl(), "retrieved_at_utc": stamp(),
                    "content_type": response.headers.get("Content-Type"),
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified")}
    return data, metadata


class _TermsText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.text, self.hrefs = [], []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skip += 1
        if tag == "a" and not self.skip:
            self.hrefs.extend(value for key, value in attrs if key == "href" and value)

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.text.append(data)


def validate_terms(data):
    """Require the declared licence in visible HTML and retain exact byte ranges.

    The complete unmodified HTML is archived separately; evidence byte ranges
    point into it. No parser output replaces the original licence terms.
    """
    parser = _TermsText()
    parser.feed(data.decode("utf-8-sig", errors="strict"))
    visible = " ".join(" ".join(parser.text).split())
    if "CC BY-NC-SA 3.0 IGO" not in visible:
        raise ValueError("official_terms_license_identifier_missing")
    links = [urljoin(TERMS_URL, href).rstrip("/") for href in parser.hrefs]
    if LICENSE_URL not in links:
        raise ValueError("official_terms_license_link_missing")
    needles = [b"CC BY-NC-SA 3.0 IGO", LICENSE_URL.encode()]
    evidence = []
    for needle in needles:
        offset = data.find(needle)
        if offset < 0:
            raise ValueError("exact_html_license_evidence_missing")
        evidence.append({"start_byte": offset, "end_byte_exclusive": offset + len(needle),
                         "exact_utf8": needle.decode(), "span_sha256": sha256(needle)})
    return {"license": LICENSE, "license_url": LICENSE_URL,
            "source_url": TERMS_URL, "source_sha256": sha256(data),
            "original_html_byte_ranges": evidence,
            "license_basis": "official_platform_terms_declared_license",
            "individual_pdf_notices_require_review_before_adaptation": True}


def validate_pdf(data):
    # Avoid silently accepting an HTML access/error page as a frozen PDF.
    # Full structural/text/page review remains a separate, explicit step.
    if not data.startswith(b"%PDF-") or b"%%EOF" not in data[-2048:]:
        raise ValueError("download_is_not_a_complete_pdf_signature")


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _json(path, value):
    _write(path, (canonical(value) + "\n").encode())


def _event(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(canonical({"at_utc": stamp(), **value}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def freeze(output):
    require_colab()
    output = Path(output).resolve()
    if not output.is_relative_to(Path("/content")):
        raise ValueError("output_must_be_inside_hosted_content")
    output.mkdir(parents=True, exist_ok=False)
    _json(output / "acquisition_protocol.json", {
        "version": VERSION, "sources": SOURCES, "declared_license": LICENSE,
        "purpose": "noncommercial_academic_source_bank_freeze",
        "max_bytes_per_source": MAX_SOURCE_BYTES, "automatic_retry": False,
        "new_model_calls": 0, "secret_access": False,
        "code_sha256": sha256(Path(__file__).read_bytes()),
        "separation_contract": SEPARATION_CONTRACT})
    receipts, evidence, stage = [], None, "download"
    try:
        for source in SOURCES:
            stage = "download_" + source["id"]
            _event(output / "acquisition_journal.jsonl", {
                "event": "download_intent", "id": source["id"],
                "url": source["url"], "automatic_retry": False})
            data, metadata = _download(source)
            folder = "quarantine" if source["id"] == "pm_plus_roleplay" else "source_archive"
            relative = "assessor_only/" + folder + "/" + source["filename"]
            _write(output / relative, data)
            receipt = {**source, **metadata, "archive_path": relative,
                       "bytes": len(data), "sha256": sha256(data),
                       "assessor_only": True, "generation_eligible": False,
                       "quarantined": folder == "quarantine"}
            receipts.append(receipt)
            _event(output / "acquisition_journal.jsonl", {
                "event": "bytes_archived_before_validation", **receipt})
            stage = "validate_" + source["id"]
            if source["kind"] == "html":
                evidence = validate_terms(data)
                evidence["archive_path"] = relative
                _json(output / "license_evidence.json", evidence)
            else:
                validate_pdf(data)
            _event(output / "acquisition_journal.jsonl", {
                "event": "source_validated", "id": source["id"],
                "sha256": receipt["sha256"]})
        # Read every archived byte back before declaring a successful freeze.
        stage = "archive_integrity"
        for receipt in receipts:
            data = (output / receipt["archive_path"]).read_bytes()
            if len(data) != receipt["bytes"] or sha256(data) != receipt["sha256"]:
                raise ValueError("archived_source_integrity_mismatch")
        _json(output / "source_manifest.json", {
            "version": VERSION, "frozen_at_utc": stamp(), "sources": receipts,
            "declared_license": LICENSE, "license_evidence": evidence,
            "separation_contract": SEPARATION_CONTRACT,
            "attribution": "WHO & UNICEF. Ensuring Quality in Psychological Support (EQUIP). World Health Organization; 2021.",
            "per_tool_citation_and_notices_pending_review": True,
            "derived_material_license_must_remain_same_or_equivalent": True,
            "who_endorsement_implied": False, "who_logo_reuse_permitted": False})
        summary = {"version": VERSION, "status": "source_bank_frozen_not_benchmark_ready",
                   "files": len(receipts), "pdfs": len(receipts) - 1,
                   "bytes": sum(row["bytes"] for row in receipts),
                   "quarantined_source_ids": [row["id"] for row in receipts if row["quarantined"]],
                   "source_manifest_sha256": sha256((output / "source_manifest.json").read_bytes()),
                   "license_evidence_sha256": sha256((output / "license_evidence.json").read_bytes()),
                   "generation_inputs_created": False, "new_gold_labels": 0,
                   "new_model_calls": 0, "new_human_reviews": 0,
                   "memory_effect_scores": None, "secret_access": False}
        _json(output / "summary.json", summary)
        return summary
    except BaseException as error:
        failure = {"version": VERSION, "status": "failed_preserve_partial_archive_no_retry",
                   "stage": stage, "failure_type": type(error).__name__,
                   "archived_files": len(receipts), "automatic_retry": False,
                   "new_model_calls": 0, "secret_access": False}
        _json(output / "failure.json", failure)
        # Do not print source text, server bodies or arbitrary exception content.
        raise RuntimeError(canonical(failure)) from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True,
                        help="Fresh /content directory; no overwrite or automatic resume")
    args = parser.parse_args()
    print(canonical(freeze(args.output)), flush=True)


if __name__ == "__main__":
    main()
