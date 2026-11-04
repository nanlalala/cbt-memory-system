"""Audit and analyze the separately consented, nonclinical four-visit study.

This module has no network client, model, synthetic participant generator or
clinical outcome estimator.  Planned slots never become enrolled people.  The
bootstrap is a prespecified exploratory analysis, not a power guarantee.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean

ARMS = ("short_term", "raw_history_rag", "managed_memory_rag")
PRIMARY_CONTRAST = ("managed_memory_rag", "raw_history_rag")
RUBRIC_KEYS = (
    "source_fidelity", "current_goal_alignment", "homework_review",
    "guided_discovery", "collaborative_action", "progress_integration",
    "scope_and_safety",
)
ANALYSIS_VERSION = "dialogue-benefit-analysis-v1"
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20261105


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_study(root):
    from study_capture import read_snapshot
    return read_snapshot(Path(root))


def _locked_study(root):
    from study_capture import locked_snapshot
    return locked_snapshot(Path(root))


def _assert_analysis_identity(snapshot):
    analysis_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = snapshot["manifest"]
    if manifest.get("analysis_code_sha256") is not None and manifest["analysis_code_sha256"] != analysis_hash:
        raise ValueError("analysis executable differs from the frozen study identity")
    enrolled, _ = _enrollment(snapshot)
    if enrolled and manifest["mode"] == "planned_human":
        launch = snapshot.get("launch_record")
        if not isinstance(launch, dict) or launch.get("analysis_sha256") != analysis_hash:
            raise ValueError("enrolled human analysis requires its exact frozen launch analysis identity")
    return analysis_hash


def _events(snapshot):
    return [(row, row["event"]) for row in snapshot["events"]]


def _snapshot_identity(snapshot):
    return digest({"manifest_sha256": snapshot["manifest"]["manifest_sha256"],
                   "event_hashes": [row["event_sha256"] for row in snapshot["events"]],
                   "sealed": snapshot["sealed"],
                   "consent_snapshot_sha256": snapshot.get("consent_snapshot_sha256"),
                   "revocation_head_sha256": snapshot.get("revocation_head_sha256"),
                   "erased_response_events": [row["event_sha256"] for row in snapshot["events"]
                                              if row.get("erased")]})


def _enrollment(snapshot):
    enrolled, erased = set(), set()
    for _, e in _events(snapshot):
        if e["event_type"] == "enroll":
            enrolled.add(e["participant_id"])
        elif e["event_type"] == "withdraw" and not e["payload"]["retain_collected_data"]:
            erased.add(e["participant_id"])
    for participant, withdrawal in snapshot.get("withdrawals", {}).items():
        if not withdrawal["retain_collected_data"]:
            erased.add(participant)
    return enrolled, erased


def _response_records(snapshot):
    result = {}
    for envelope, e in _events(snapshot):
        if e["event_type"] != "task_response":
            continue
        key = (e["participant_id"], e["task_id"])
        if key in result:
            raise ValueError("multiple response events for a task require a new frozen protocol")
        result[key] = (envelope, e, envelope.get("response_content"))
    return result


def _task_criterion(snapshot, participant, task, content):
    ledger = snapshot.get("task_ledger")
    if not isinstance(ledger, dict) or digest(ledger) != snapshot["manifest"].get("task_ledger_sha256"):
        raise ValueError("blinded correctness review requires the actual predeclared frozen task ledger")
    variant = snapshot["allocation"].get("task_variants", {}).get(participant)
    matches = [r for r in ledger.get("tasks", []) if r.get("task_id") == task and r.get("variant") == variant]
    if len(matches) != 1:
        raise ValueError("task ledger does not uniquely bind this participant/task/variant")
    row = matches[0]
    body = {k: v for k, v in row.items() if k != "criterion_sha256"}
    if (digest(body) != row.get("criterion_sha256") or content["task_key_sha256"] != row["criterion_sha256"]
            or not isinstance(row.get("completion_criterion"), str) or not row["completion_criterion"].strip()):
        raise ValueError("task criterion changed or its response task-key binding is invalid")
    references = row.get("source_reference")
    if not isinstance(references, list) or not references:
        raise ValueError("task criterion requires an independently prepared scoring source ledger")
    history = {s["source_id"]: s for s in content["source_history"]}
    if len(history) != len(content["source_history"]):
        raise ValueError("conflicting source IDs in task history")
    originals = {(s["variant"], s["source_id"]): s for s in ledger.get("sources", [])}
    for reference in references:
        source = history.get(reference.get("source_id"))
        original = originals.get((variant, reference.get("source_id")))
        source_text = original.get("source_text") if original else reference.get("source_text")
        if original and any(original.get(k) != reference.get(k) for k in ("source_text_sha256", "role", "date")):
            raise ValueError("task reference differs from its original frozen source metadata")
        if original and reference.get("source_text") is not None and reference["source_text"] != source_text:
            raise ValueError("task reference body differs from its original frozen source")
        if source_text is None and source:
            source_text = source["text"]
        if not isinstance(source_text, str) or text_hash(source_text) != reference.get("source_text_sha256"):
            raise ValueError("scoring source ledger must supply verifiable source text, not an ID/hash alone")
        if source and source["text"] != source_text:
            raise ValueError("scoring ledger contradicts the provided source version")
        if source and any(source.get(k) != reference.get(k) for k in ("role", "date")):
            raise ValueError("scoring ledger source role/date differs from the supplied history")
        if source is None and reference.get("permitted_absence_reason") not in {"deleted", "dont_save", "expired"}:
            raise ValueError("absent scoring source requires a predeclared permitted absence reason")
        start, end = reference.get("start"), reference.get("end")
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source_text)
                or reference.get("quote") != source_text[start:end]):
            raise ValueError("scoring source reference has an invalid exact Unicode span")
    return row


def _scoring_sources(snapshot, criterion, content):
    """Rater-only original sources, including explicitly deleted/excluded gold."""
    ledger = snapshot["task_ledger"]
    originals = {(s["variant"], s["source_id"]): s for s in ledger.get("sources", [])}
    visible = {s["source_id"]: s for s in content["source_history"]}
    result = []
    for reference in criterion["source_reference"]:
        source = originals.get((criterion["variant"], reference["source_id"]))
        if source is None:
            text = reference.get("source_text", visible.get(reference["source_id"], {}).get("text"))
            source = {"source_id": reference["source_id"], "variant": criterion["variant"],
                      "source_text": text, "source_text_sha256": reference["source_text_sha256"],
                      "role": reference["role"], "date": reference["date"]}
        result.append(source)
    return result


def _timestamp(text):
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return value


def _visit_time(events, cap):
    """Restricted elapsed time; the cap is a penalty, not observed missing time."""
    starts = [e for e in events if e["event_type"] == "visit_start"]
    ends = [e for e in events if e["event_type"] == "visit_end"]
    assisted = any(e["payload"].get("assistance_received") is True or e["payload"].get("status") == "assisted" for e in events
                   if e["event_type"] in {"task_end", "visit_end"})
    completed = bool(ends and ends[-1]["payload"]["status"] == "completed")
    penalized = not completed or assisted
    if not starts or not ends:
        if completed:
            raise ValueError("a completed visit requires exactly one start and end")
        return {"restricted_seconds": cap, "observed_seconds": None,
                "cap_imputed": True, "assisted": assisted, "burden_observation_complete": False}
    if len(starts) != 1 or len(ends) != 1:
        raise ValueError("a completed visit requires exactly one start and end")
    start, end = _timestamp(starts[0]["timestamp"]), _timestamp(ends[0]["timestamp"])
    if end < start:
        raise ValueError("visit end precedes its start")
    pause_start, excluded = None, 0.0
    for e in events:
        if e["event_type"] not in {"pause", "resume"}:
            continue
        t = _timestamp(e["timestamp"])
        if not start <= t <= end:
            raise ValueError("a documented break is outside the visit")
        if e["event_type"] == "pause":
            if pause_start is not None:
                raise ValueError("nested pause")
            pause_start = t
        else:
            if pause_start is None or t < pause_start:
                raise ValueError("resume without an earlier pause")
            excluded += (t - pause_start).total_seconds()
            pause_start = None
    if pause_start is not None:
        if completed:
            raise ValueError("completed visit has an unresolved pause")
        excluded += (end - pause_start).total_seconds()
    elapsed = (end - start).total_seconds() - excluded
    if elapsed < 0:
        raise ValueError("excluded pauses exceed elapsed visit time")
    return {"restricted_seconds": cap if penalized else min(cap, elapsed), "observed_seconds": elapsed,
            "cap_imputed": penalized, "assisted": assisted,
            "burden_observation_complete": bool(not penalized and ends[0]["payload"]["burden_observation_complete"])}


def _quantile(values, probability):
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def _difference_interval(treatment, control, *, seed):
    """Independent participant resampling, never task rows as independent units."""
    rng = random.Random(seed)
    draws = []
    for _ in range(BOOTSTRAP_REPLICATES):
        draws.append(mean(rng.choices(treatment, k=len(treatment))) -
                     mean(rng.choices(control, k=len(control))))
    return {"difference": mean(treatment) - mean(control),
            "ci_97_5_percent": [_quantile(draws, .0125), _quantile(draws, .9875)],
            "interval_method": "exploratory_independent_participant_percentile_bootstrap",
            "replicates": BOOTSTRAP_REPLICATES, "seed": seed,
            "familywise_rule": "Bonferroni: two co-primary intervals, each 97.5%",
            "small_sample_caution": "Approximate interval; no guaranteed coverage or confirmatory claim."}


def _write_new(path, value):
    if path.exists():
        raise ValueError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def export_blinded_packets(study_root, output_dir):
    """Make controlled single-answer forms; the allocation key stays separate."""
    # Serialize publication against a concurrent post-seal consent revocation.
    with _locked_study(study_root) as snapshot:
        return _export_snapshot(snapshot, output_dir)


def _export_snapshot(snapshot, output_dir):
    _assert_analysis_identity(snapshot)
    if not snapshot["sealed"]:
        raise ValueError("seal collection before exporting a fixed review packet")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("a packet export requires a new output directory")
    output_dir.mkdir(parents=True, mode=0o700)
    private = output_dir / "restricted"
    private.mkdir(mode=0o700)
    enrolled, erased = _enrollment(snapshot)
    records = list(_response_records(snapshot).items())
    random.Random(BOOTSTRAP_SEED).shuffle(records)
    packets, mapping = [], {}
    for (participant, task), (envelope, e, content) in records:
        if participant not in enrolled or participant in erased or envelope.get("erased"):
            continue
        if not isinstance(content, dict):
            raise ValueError("a non-erased response artifact is missing")
        for name in ("source_history", "current_query", "response", "task_key_sha256"):
            if name not in content:
                raise ValueError(f"response artifact lacks {name}")
        criterion = _task_criterion(snapshot, participant, task, content)
        packet_id = f"R{len(packets) + 1:05d}"
        packet = {"packet_id": packet_id, "source_history": content["source_history"],
                  "current_query": content["current_query"], "response": content["response"],
                  "source_history_sha256": digest(content["source_history"]),
                  "response_sha256": text_hash(content["response"]),
                  "task_key_sha256": content["task_key_sha256"], "task_criterion": criterion,
                  "scoring_sources": _scoring_sources(snapshot, criterion, content)}
        packet["packet_sha256"] = digest(packet)
        packets.append(packet)
        mapping[packet_id] = {"participant_id": participant, "task_id": task,
                              "visit": e["visit"], "task_response_event_sha256": envelope["event_sha256"]}
    manifest = {"schema_version": ANALYSIS_VERSION, "mode": snapshot["manifest"]["mode"],
                "source_snapshot_sha256": _snapshot_identity(snapshot),
                "packet_count": len(packets), "packet_hashes": {p["packet_id"]: p["packet_sha256"] for p in packets},
                "rubric_keys": list(RUBRIC_KEYS), "ratings_completed": 0,
                "qualification_requirement": "Two independent arm-blinded CBT-trained human raters; attestations are not credential verification.",
                "blinding_limit": "Condition labels are removed; response content can still reveal a condition. Record suspected unblinding.",
                "interpretation": "Research rubric, not a validated clinical or therapist competency scale.",
                "privacy": "Controlled fictional-task responses; never publish raw packets or the private key. Delete revoked participant copies."}
    manifest["packet_manifest_sha256"] = digest(manifest)
    _write_new(output_dir / "packet_manifest.json", manifest)
    _write_new(output_dir / "blinded_packets.json", packets)
    _write_new(private / "packet_key.json", {"packet_manifest_sha256": manifest["packet_manifest_sha256"],
                                            "mapping": mapping})
    _write_new(output_dir / "blank_ratings.json", {"packet_manifest_sha256": manifest["packet_manifest_sha256"],
        "raters": [], "ratings": [], "rubric_values": [0, 1, 2, "NA", "UNRESOLVED"],
        "rater_template": {"rater_id": "replace-with-pseudonym", "qualification_description": "",
                           "qualified_cbt_training_attested": False, "independent_from_system_development": False,
                           "arm_blinded": False, "human_not_model_attested": False,
                           "qualification_verification": {"verified_by": "", "verified_at": "", "record_sha256": "",
                                                          "independent_verifier_attestation": False}},
        "rating_templates": [{"packet_id": p["packet_id"], "packet_sha256": p["packet_sha256"],
            "source_history_sha256": p["source_history_sha256"], "task_key_sha256": p["task_key_sha256"],
            "rater_id": "replace-with-pseudonym", "rating_role": "independent", "status": "unresolved",
            "task_correct": None, "source_verified": False, "source_evidence": [],
            "rubric": {k: "UNRESOLVED" for k in RUBRIC_KEYS},
            "rubric_reasons": {k: "" for k in RUBRIC_KEYS}, "suspected_unblinding": None} for p in packets],
        "instructions": "Do not use model-generated ratings. Record separate source-backed task_correct labels and rubric domains; do not sum rubric values. Disagreement requires an independent third human adjudicator."})
    return manifest


def _verify_packet_bundle(snapshot, packet_dir):
    packet_dir = Path(packet_dir)
    manifest = json.loads((packet_dir / "packet_manifest.json").read_text())
    saved = manifest.pop("packet_manifest_sha256")
    if saved != digest(manifest):
        raise ValueError("packet manifest checksum mismatch")
    manifest["packet_manifest_sha256"] = saved
    if manifest["source_snapshot_sha256"] != _snapshot_identity(snapshot):
        raise ValueError("review packet no longer matches the current collection/withdrawal snapshot")
    packets = json.loads((packet_dir / "blinded_packets.json").read_text())
    key = json.loads((packet_dir / "restricted/packet_key.json").read_text())
    if key["packet_manifest_sha256"] != saved:
        raise ValueError("packet key belongs to another export")
    if len(packets) != manifest["packet_count"] or len({p["packet_id"] for p in packets}) != len(packets):
        raise ValueError("duplicate or missing packet")
    records = _response_records(snapshot)
    result, seen_source_keys = {}, set()
    enrolled, erased = _enrollment(snapshot)
    expected_source_keys = {key for key, (envelope, _, content) in records.items()
                            if key[0] in enrolled - erased and not envelope.get("erased") and content is not None}
    for packet in packets:
        p = dict(packet)
        checksum = p.pop("packet_sha256")
        pid = p["packet_id"]
        if checksum != digest(p) or manifest["packet_hashes"].get(pid) != checksum:
            raise ValueError("packet text checksum mismatch")
        binding = key["mapping"].get(pid)
        if not binding:
            raise ValueError("packet is absent from private key")
        source_key = (binding["participant_id"], binding["task_id"])
        if source_key not in records:
            raise ValueError("packet refers to an absent source task")
        if source_key in seen_source_keys:
            raise ValueError("duplicate source task in review packet inventory")
        seen_source_keys.add(source_key)
        envelope, _, content = records[source_key]
        if content is None or binding["task_response_event_sha256"] != envelope["event_sha256"]:
            raise ValueError("packet source artifact is missing or changed")
        for field in ("source_history", "current_query", "response", "task_key_sha256"):
            if p[field] != content[field]:
                raise ValueError("packet differs from the actual saved source/response")
        criterion = _task_criterion(snapshot, binding["participant_id"], binding["task_id"], content)
        if p.get("task_criterion") != criterion:
            raise ValueError("packet criterion differs from the frozen scoring reference")
        if p.get("scoring_sources") != _scoring_sources(snapshot, criterion, content):
            raise ValueError("packet scoring sources differ from the frozen original source ledger")
        if p["source_history_sha256"] != digest(content["source_history"]):
            raise ValueError("packet source history checksum mismatch")
        if p["response_sha256"] != text_hash(content["response"]):
            raise ValueError("packet response checksum mismatch")
        result[pid] = (packet, binding)
    if set(key["mapping"]) != set(result) or set(manifest["packet_hashes"]) != set(result):
        raise ValueError("packet inventory/key mismatch")
    if seen_source_keys != expected_source_keys:
        raise ValueError("review packet inventory omits or adds eligible captured responses")
    return manifest, result


def _verify_rating(row, packet, raters):
    if row.get("rater_id") not in raters:
        raise ValueError("rating has no qualified human rater attestation")
    if row.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("rating is not bound to the exact blinded packet")
    if row.get("source_history_sha256") != packet["source_history_sha256"] or row.get("task_key_sha256") != packet["task_key_sha256"]:
        raise ValueError("rating source or task-key version mismatch")
    if row.get("status") not in {"adjudicated", "adjudicated_unscorable", "unresolved"}:
        raise ValueError("rating status must distinguish adjudicated/unscorable from pending unresolved")
    if row["status"] == "adjudicated_unscorable" and (row.get("task_correct") is not None or
            not isinstance(row.get("unscorable_reason"), str) or not row["unscorable_reason"].strip()):
        raise ValueError("adjudicated unscorable needs a reason and a null correctness label")
    if row["status"] == "unresolved" and row.get("task_correct") is not None:
        raise ValueError("a pending unresolved rating cannot carry a correctness label")
    if row.get("task_correct") not in (True, False, None) or (row.get("task_correct") is not None and type(row["task_correct"]) is not bool):
        raise ValueError("task correctness must be a boolean or unresolved null")
    if row.get("rating_role", "independent") not in {"independent", "adjudicator"}:
        raise ValueError("unknown rater role")
    rubric = row.get("rubric")
    if not isinstance(rubric, dict) or set(rubric) != set(RUBRIC_KEYS):
        raise ValueError("all separate rubric domains must be recorded")
    if any(not ((type(v) is int and v in (0, 1, 2)) or
                (isinstance(v, str) and v in {"NA", "UNRESOLVED"})) for v in rubric.values()):
        raise ValueError("rubric values are 0/1/2/NA/UNRESOLVED, not a clinical total")
    reasons = row.get("rubric_reasons", {})
    if any(not isinstance(reasons.get(k), str) or not reasons[k].strip()
           for k, value in rubric.items() if value in ("NA", "UNRESOLVED")):
        raise ValueError("NA and UNRESOLVED rubric domains require an explicit reason")
    if type(row.get("suspected_unblinding")) is not bool:
        raise ValueError("record whether the rater suspected condition unblinding")
    evidence = row.get("source_evidence", [])
    history = {s["source_id"]: s for s in packet["source_history"]}
    if len(history) != len(packet["source_history"]):
        raise ValueError("duplicate source IDs cannot establish source evidence")
    for source in packet["scoring_sources"]:
        if source["source_id"] not in history and isinstance(source.get("source_text"), str):
            history[source["source_id"]] = {"text": source["source_text"]}
    for span in evidence:
        source = history.get(span.get("source_id"))
        if source is None or span.get("source_text_sha256") != text_hash(source["text"]):
            raise ValueError("source annotation text/version mismatch")
        start, end = span.get("start"), span.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source["text"]):
            raise ValueError("source evidence span is outside its exact source")
    if row.get("status") == "adjudicated" and row.get("task_correct") is not None:
        if row.get("source_verified") is not True or not evidence:
            raise ValueError("a resolved correctness label requires independently verified source spans")


def load_review_labels(snapshot, packet_dir, rating_file):
    """Return labels only after two independent humans, with third-rater arbitration."""
    manifest, packets = _verify_packet_bundle(snapshot, packet_dir)
    review = json.loads(Path(rating_file).read_text())
    if review.get("packet_manifest_sha256") != manifest["packet_manifest_sha256"]:
        raise ValueError("ratings belong to another packet export")
    raters = {}
    for rater in review.get("raters", []):
        rid = rater.get("rater_id")
        if not isinstance(rid, str) or not rid or rid in raters:
            raise ValueError("rater IDs must be unique")
        for flag in ("qualified_cbt_training_attested", "independent_from_system_development", "arm_blinded", "human_not_model_attested"):
            if rater.get(flag) is not True:
                raise ValueError("qualified, independent, arm-blinded human rater attestations are required")
        if not isinstance(rater.get("qualification_description"), str) or not rater["qualification_description"].strip():
            raise ValueError("rater training/qualification must be described")
        verification = rater.get("qualification_verification", {})
        if (verification.get("independent_verifier_attestation") is not True
                or not isinstance(verification.get("verified_by"), str) or not verification["verified_by"].strip()
                or verification["verified_by"] == rid
                or not isinstance(verification.get("record_sha256"), str)
                or len(verification["record_sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in verification["record_sha256"])):
            raise ValueError("qualified-rater registry requires a separate verification record")
        _timestamp(verification.get("verified_at", ""))
        raters[rid] = rater
    grouped, seen = defaultdict(list), set()
    for row in review.get("ratings", []):
        pid = row.get("packet_id")
        if pid not in packets:
            raise ValueError("unknown blinded packet ID")
        unique = (pid, row.get("rater_id"))
        if unique in seen:
            raise ValueError("duplicate rating ID; conflicting labels cannot silently replace a rating")
        seen.add(unique)
        _verify_rating(row, packets[pid][0], raters)
        grouped[pid].append(row)
    labels, unresolved = {}, 0
    domains = {k: {"independent_rating_counts": {arm: {str(v): 0 for v in (0, 1, 2, "NA", "UNRESOLVED")} for arm in ARMS},
                   "numeric_pair_count": 0, "exact_agreement_pairs": 0, "numeric_disagreement_pairs": 0,
                   "pairs_with_na_or_unresolved": 0} for k in RUBRIC_KEYS}
    for pid, (packet, binding) in packets.items():
        rows = grouped[pid]
        independent = [r for r in rows if r.get("rating_role", "independent") == "independent"]
        if len(independent) > 2:
            raise ValueError("freeze exactly two independent raters; extra raters require a protocol amendment")
        arm = snapshot["allocation"]["assignments"][binding["participant_id"]]
        for domain in RUBRIC_KEYS:
            for r in independent:
                domains[domain]["independent_rating_counts"][arm][str(r["rubric"][domain])] += 1
            if len(independent) == 2:
                values = [r["rubric"][domain] for r in independent]
                if all(type(v) is int for v in values):
                    domains[domain]["numeric_pair_count"] += 1
                    domains[domain]["exact_agreement_pairs"] += values[0] == values[1]
                    domains[domain]["numeric_disagreement_pairs"] += values[0] != values[1]
                else:
                    domains[domain]["pairs_with_na_or_unresolved"] += 1
        label = None
        eligible = [r for r in independent if r["status"] == "adjudicated_unscorable" or
                    (r["status"] == "adjudicated" and r["task_correct"] is not None)]
        def binary(r):
            return False if r["status"] == "adjudicated_unscorable" else r["task_correct"]
        if len(eligible) == 2:
            if binary(eligible[0]) == binary(eligible[1]):
                label = binary(eligible[0])
            else:
                arbiters = [r for r in rows if r.get("rating_role") == "adjudicator"]
                if len(arbiters) > 1:
                    raise ValueError("multiple adjudicators require a frozen tie rule")
                if arbiters:
                    a = arbiters[0]
                    if set(a.get("adjudicates_rater_ids", [])) != {r["rater_id"] for r in eligible}:
                        raise ValueError("adjudicator must identify both conflicting independent labels")
                    if a["status"] == "adjudicated_unscorable" or (a["status"] == "adjudicated" and a["task_correct"] is not None):
                        label = binary(a)
        if label is None:
            unresolved += 1
        labels[(binding["participant_id"], binding["task_id"])] = label
    for values in domains.values():
        values["exact_agreement_fraction"] = (values["exact_agreement_pairs"] / values["numeric_pair_count"]
                                               if values["numeric_pair_count"] else None)
    return labels, {"source": "separate_arm_blinded_human_review", "human_rater_attestations": len(raters),
                    "qualification_verification_record_required": True, "verification_automatically_authenticated": False,
                    "actual_human_ratings": 0 if snapshot["manifest"]["mode"] == "fixture" else len(seen),
                    "rating_rows": len(seen),
                    "packets": len(packets), "unresolved_packet_labels": unresolved,
                    "rubric_summary": "Separate ordinal domains only; no overall clinical score.",
                    "pre_adjudication_domain_descriptives": domains,
                    "domain_intervals": "Not computed; response ratings share participants and are not independent replicates.",
                    "packet_manifest_sha256": manifest["packet_manifest_sha256"]}


def analyze_snapshot(snapshot, *, labels=None, review_summary=None):
    """Compute observed/protocol-penalized endpoints without inventing enrollment."""
    manifest = snapshot["manifest"]
    if manifest["mode"] not in {"planned_human", "fixture"}:
        raise ValueError("unknown study mode")
    assignments = snapshot["allocation"]["assignments"]
    enrolled, erased = _enrollment(snapshot)
    withdrawn = set(snapshot.get("withdrawals", {})) | {
        e["participant_id"] for _, e in _events(snapshot) if e["event_type"] == "withdraw"}
    analysis_hash = _assert_analysis_identity(snapshot)
    labels = labels or {}
    review_summary = review_summary or {"source": "no_independent_human_reviews", "human_rater_attestations": 0}
    records = _response_records(snapshot)
    by_person = defaultdict(list)
    for _, e in _events(snapshot):
        by_person[e["participant_id"]].append(e)
    tasks = [task for visit in ("1", "2", "3", "4") for task in manifest["visit_tasks"][visit]]
    if len(tasks) != 12 or len(set(tasks)) != 12:
        raise ValueError("analysis v1 requires the frozen 12 unique follow-up tasks")
    rows = []
    costs = [e["payload"] for _, e in _events(snapshot) if e["event_type"] == "cost"]
    unresolved_total = 0
    for participant in sorted(enrolled - erased):
        events = by_person[participant]
        arm = assignments[participant]
        if arm not in ARMS:
            raise ValueError("unknown allocated arm")
        visit_times = [_visit_time([e for e in events if e["visit"] == visit], manifest["visit_elapsed_cap_seconds"])
                       for visit in range(1, 5)]
        outcomes, task_rows = [], []
        for task in tasks:
            end = [e for e in events if e["task_id"] == task and e["event_type"] == "task_end"]
            if len(end) > 1:
                raise ValueError("multiple task completions cannot change the primary denominator")
            completed = bool(end and end[0]["payload"].get("status") == "completed")
            assisted = bool(end and (end[0]["payload"].get("assistance_received") is True or end[0]["payload"].get("status") == "assisted"))
            if not completed or assisted or (participant, task) not in records:
                outcome = False
                reason = "assisted" if assisted else "failed_missing_or_incomplete_task"
            else:
                outcome = labels.get((participant, task))
                reason = "pending_independent_review" if outcome is None else "source_adjudicated_success" if outcome else "source_adjudicated_failure_or_unscorable"
            outcomes.append(outcome)
            task_rows.append({"task_id": task, "score": None if outcome is None else int(outcome), "reason": reason})
        unresolved = outcomes.count(None)
        unresolved_total += unresolved
        checks = [e for e in events if e["event_type"] == "memory_check"]
        complete_burden = all(v["burden_observation_complete"] for v in visit_times)
        successes = sum(v is True for v in outcomes)
        rows.append({"participant_id": participant, "arm": arm, "planned_followup_tasks": 12,
            "verified_successes": successes, "unresolved_completed_tasks": unresolved,
            "task_correctness": None if unresolved else successes / 12,
            "task_correctness_identification_bounds": [successes / 12, (successes + unresolved) / 12],
            "task_outcomes": task_rows,
            "restricted_elapsed_seconds": sum(v["restricted_seconds"] for v in visit_times),
            "visit_time": visit_times,
            "checks_observed": len(checks), "repeated_checks_observed": sum(e["payload"]["repeated"] for e in checks),
            "user_edits_observed": sum(e["event_type"] == "user_edit" for e in events),
            "explicit_confirmations_observed": sum(e["event_type"] == "explicit_confirm" for e in events),
            "burden_observation_complete": complete_burden,
            "checks_per_12_planned_tasks": len(checks) / 12 if complete_burden else None,
            "interpretation": "Counts are observed events; unlogged checks are unknown, not zero."})
    blockers = []
    if manifest["mode"] == "fixture": blockers.append("fixture_only_not_human_evidence")
    if not snapshot["sealed"]: blockers.append("collection_not_sealed")
    if not enrolled: blockers.append("no_enrolled_participants")
    if erased: blockers.append("consent_erasure_prevents_complete_ITT_analysis")
    if unresolved_total: blockers.append("completed_task_correctness_reviews_unresolved")
    counts = Counter(r["arm"] for r in rows)
    if any(counts[a] < 2 for a in PRIMARY_CONTRAST): blockers.append("fewer_than_two_participants_in_a_primary_arm")
    contrast = {}
    if not blockers:
        for metric, seed in (("task_correctness", BOOTSTRAP_SEED), ("restricted_elapsed_seconds", BOOTSTRAP_SEED + 1)):
            t = [r[metric] for r in rows if r["arm"] == PRIMARY_CONTRAST[0]]
            c = [r[metric] for r in rows if r["arm"] == PRIMARY_CONTRAST[1]]
            contrast[metric] = _difference_interval(t, c, seed=seed)
        lower = contrast["task_correctness"]["ci_97_5_percent"][0]
        upper_time = contrast["restricted_elapsed_seconds"]["ci_97_5_percent"][1]
        contrast["both_directional_criteria_met_exploratory"] = lower > 0 and upper_time < 0
    usage_complete = all(all(type(c.get(k)) is int and c[k] >= 0 for k in
                                 ("prompt_tokens", "completion_tokens", "total_tokens")) for c in costs)
    for c in costs:
        if all(type(c.get(k)) is int for k in ("prompt_tokens", "completion_tokens", "total_tokens")) and c["total_tokens"] != c["prompt_tokens"] + c["completion_tokens"]:
            raise ValueError("provider usage token components do not sum to total")
    summary = {"analysis_version": ANALYSIS_VERSION, "analysis_code_sha256": analysis_hash,
        "source_snapshot_sha256": _snapshot_identity(snapshot), "mode": manifest["mode"],
        "planned_slots": manifest["planned_slots"], "enrolled_slots": len(enrolled),
        "actual_human_participants": len(enrolled) if manifest["mode"] == "planned_human" else 0,
        "revocation_head_sha256": snapshot.get("revocation_head_sha256"),
        "consent_snapshot_sha256": snapshot.get("consent_snapshot_sha256"),
        "consent_erased_slots": len(erased), "analyzable_retained_slots": len(rows),
        "analyzable_participants_by_arm": {a: counts[a] for a in ARMS},
        "participant_flow_by_arm": {a: {"enrolled_original_allocation": sum(assignments[p] == a for p in enrolled),
                                       "withdrawn_retaining_collected_data": sum(assignments[p] == a for p in withdrawn - erased),
                                       "consent_erased": sum(assignments[p] == a for p in erased),
                                       "retained_analyzable": counts[a]} for a in ARMS},
        "arm_descriptives": {a: {
            "task_success_fraction_mean": mean([r["task_correctness"] for r in rows if r["arm"] == a])
                if counts[a] and all(r["task_correctness"] is not None for r in rows if r["arm"] == a) else None,
            "consent_retained_success_mean_identification_bounds": [mean([r["task_correctness_identification_bounds"][i] for r in rows if r["arm"] == a]) for i in (0, 1)] if counts[a] else None,
            "original_allocation_success_mean_missing_data_bounds": [
                (sum(r["task_correctness_identification_bounds"][i] for r in rows if r["arm"] == a)
                 + (sum(assignments[p] == a for p in erased) if i else 0)) / sum(assignments[p] == a for p in enrolled)
                for i in (0, 1)] if any(assignments[p] == a for p in enrolled) else None,
            "original_allocation_restricted_time_missing_data_bounds": [
                (sum(r["restricted_elapsed_seconds"] for r in rows if r["arm"] == a)
                 + (3600 * sum(assignments[p] == a for p in erased) if i else 0)) / sum(assignments[p] == a for p in enrolled)
                for i in (0, 1)] if any(assignments[p] == a for p in enrolled) else None,
            "restricted_elapsed_mean_seconds": mean([r["restricted_elapsed_seconds"] for r in rows if r["arm"] == a]) if counts[a] else None,
            "complete_burden_participants": sum(r["arm"] == a and r["burden_observation_complete"] for r in rows),
            "observed_checks": sum(r["checks_observed"] for r in rows if r["arm"] == a),
            "observed_repeated_checks": sum(r["repeated_checks_observed"] for r in rows if r["arm"] == a),
            "observed_elapsed_visit_count": sum(v["observed_seconds"] is not None for r in rows if r["arm"] == a for v in r["visit_time"]),
            "observed_elapsed_seconds_sum": sum(v["observed_seconds"] for r in rows if r["arm"] == a for v in r["visit_time"] if v["observed_seconds"] is not None),
            "note": "Observed elapsed sum is available visits only; primary restricted time retains missing/assisted-visit penalties."
        } for a in ARMS},
        "missing_data_bounds_note": "Bounds use retained aggregates plus permitted allocation/withdrawal counts, never erased response contents. They do not identify a full-ITT point estimate.",
        "independence_unit": "participant", "primary_contrast": list(PRIMARY_CONTRAST),
        "inference_blockers": blockers, "primary_exploratory_intervals": contrast,
        "independent_review": review_summary, "clinical_efficacy_measured": False,
        "confirmatory_benefit_proven": False,
        "eligible_check_opportunity_denominator": None,
        "eligible_check_opportunity_status": "Requires an independently annotated opportunity ledger; not inferred from answer substrings or check counts.",
        "stage_causal_attribution": "Unresolved; telemetry is not an independently annotated extraction-to-answer causal diagnosis.",
        "cost": {"attempt_records": len(costs), "failed_attempt_records": sum(c.get("status") in {"failure", "timeout"} for c in costs),
                 "unknown_outcome_attempt_records": sum(c.get("status") == "unknown" for c in costs),
                 "scope": "All recorded attempt metadata, including failed attempts and consent-erased slots; approved financial retention rules still apply.",
                 "recorded_attempts_by_stage": dict(Counter(c.get("stage", "unknown") for c in costs)),
                 "attempt_journal_complete_attestation": all(e["payload"].get("attempt_log_complete") is True
                     for _, e in _events(snapshot) if e["event_type"] == "visit_end") if any(e["event_type"] == "visit_end" for _, e in _events(snapshot)) else None,
                 "token_usage_complete_for_recorded_attempts": usage_complete if costs else None,
                 "prompt_tokens_observed": sum(c.get("prompt_tokens") or 0 for c in costs),
                 "completion_tokens_observed": sum(c.get("completion_tokens") or 0 for c in costs),
                 "total_tokens_observed": sum(c.get("total_tokens") or 0 for c in costs),
                 "missing_usage_attempts": sum(any(c.get(k) is None for k in ("prompt_tokens", "completion_tokens", "total_tokens")) for c in costs),
                 "cached_tokens_observed": sum(c.get("cached_tokens") or 0 for c in costs),
                 "latency_seconds_observed": sum(c.get("latency_seconds") or 0 for c in costs),
                 "missing_latency_attempts": sum(c.get("latency_seconds") is None for c in costs),
                 "verified_monetary_cost": None,
                 "monetary_cost_note": "No independently verified provider price/billing evidence supplied; token counts are not a dollar estimate."},
        "limitations": ["No clinical symptom endpoint or patient treatment effect is measured.",
                        "60 planned slots do not imply 60 independent enrolled people or adequate power.",
                        "A null/negative or unresolved comparison is retained; interval overlap with zero is not evidence of equivalence.",
                        "Raw private task content and allocation keys must not be published."]}
    return summary, rows


def analyze(study_root, *, packet_dir=None, rating_file=None):
    if bool(packet_dir) != bool(rating_file):
        raise ValueError("provide both packet directory and its independent human ratings")
    with _locked_study(study_root) as snapshot:
        return _analyze_loaded(snapshot, packet_dir, rating_file)


def _analyze_loaded(snapshot, packet_dir, rating_file):
    labels, review = ({}, None) if not packet_dir else load_review_labels(snapshot, packet_dir, rating_file)
    return analyze_snapshot(snapshot, labels=labels, review_summary=review)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export-packets")
    p.add_argument("--study-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p = sub.add_parser("analyze")
    p.add_argument("--study-dir", type=Path, required=True)
    p.add_argument("--packet-dir", type=Path)
    p.add_argument("--ratings", type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "export-packets":
        print(json.dumps(export_blinded_packets(args.study_dir, args.output_dir), ensure_ascii=False))
    else:
        if bool(args.packet_dir) != bool(args.ratings):
            raise ValueError("provide both packet directory and ratings")
        with _locked_study(args.study_dir) as snapshot:
            summary, rows = _analyze_loaded(snapshot, args.packet_dir, args.ratings)
            args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
            _write_new(args.output_dir / "summary.json", summary)
            _write_new(args.output_dir / "participant_metrics_private.json", rows)
            print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
