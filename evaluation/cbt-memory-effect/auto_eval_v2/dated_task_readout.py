"""Prepare matched requests from a frozen, dated controlled-product replay.

The input is a replay database at its cutoff; no provider is called here.
CACTUS ordinal-only histories do not supply dated product task snapshots.
"""
from .context_assembly import ARMS, assemble
from .retrieval import canonical

VERIFIER = "controlled-product-rule-verified-task-fields-v1"


def prepare_task_requests(session, *, current_input, as_of, clock_mode, selected_ids,
                          recent_ids, request_options, max_request_chars):
    if clock_mode != "dated-sources-v1" or not as_of:
        raise ValueError("task readout needs the dated replay's explicit source cutoff")
    if session.provider is not None:
        raise ValueError("input must be a provider-free frozen replay")
    if len(set(recent_ids)) != len(recent_ids) or set(selected_ids) & set(recent_ids):
        raise ValueError("recent and older selections must be disjoint")
    from evidence_bundles import _supported_options
    from controlled_dialogue import moment

    uid = session.user_id
    generation = session.durable.current_generation(uid)
    sources = []
    paused, _ = session._paused_source_ids(as_of)
    for i, row in enumerate(session.durable.list_sources(uid)):
        if not row["source_time"] or moment(row["source_time"]) > moment(as_of):
            raise ValueError("future or undated input in dated replay snapshot")
        flags = row["permissions"]
        sources.append(dict(source_id=row["source_id"], source_order=i, content=row["text"],
                            subject_id=uid, role=row["role"], retained=flags["save"],
                            use_allowed=flags["long_term_use"] and row["source_id"] not in paused,
                            external_allowed=flags["response_external"]))
    by_id = {s["source_id"]: s for s in sources}
    recent = [by_id[sid] for sid in recent_ids]
    if [r["source_order"] for r in recent] != sorted(r["source_order"] for r in recent):
        raise ValueError("recent messages out of order")
    task = session._task_for_input(current_input, as_of)
    task = task if task and task["status"] == "available" else None
    bundles, _, _ = session._bundles(current_input, as_of=as_of, task=task)
    memories, omissions = [], {}
    for bundle in bundles:
        if not task or bundle.task_instance_id != task["payload"]["task_instance_id"]:
            continue
        for claim in bundle.claims:
            if claim.support_status != "rule_verified":
                omissions[claim.claim_id] = "no_current_rule_verified_value"
                continue
            options = [spans for spans in _supported_options(claim)
                       if all(span.source_id in selected_ids and span.source_id in by_id and
                              by_id[span.source_id]["role"] == span.role for span in spans)]
            if not options:
                omissions[claim.claim_id] = "no_supported_option_on_selected_sources"
                continue
            spans = min(options, key=lambda option: (sum(len(s.text) for s in option),
                        tuple((s.source_id, s.start, s.end) for s in option)))
            evidence = [dict(source_id=s.source_id, start=s.start, end=s.end, text=s.text) for s in spans]
            memories.append(dict(memory_id=claim.claim_id, version=task["version"], subject_id=uid,
                active=True, deleted=False, valid_at_cutoff=True, superseded_by=None,
                support="verified_projection", verifier_revision=VERIFIER,
                epistemic_status="assistant_proposal" if claim.origin=="assistant_proposal" else "supported_state",
                summary=f"{claim.field} = {canonical(claim.value)}",
                source_ids=list(dict.fromkeys(s.source_id for s in spans)), evidence=evidence))
    packages = {arm:assemble(sources=sources, selected_ids=selected_ids, memories=memories, recent=recent,
        current_input=current_input, subject_id=uid, cutoff_order=max(0,len(sources)-1),
        verifier_revision=VERIFIER, arm=arm, request_options=request_options,
        max_request_chars=max_request_chars) for arm in ARMS}
    for packed in packages.values():
        packed["trace"].update(product_as_of=as_of, product_generation=generation,
                               product_claim_omissions=omissions)
    if (session.durable.current_generation(uid) != generation or
        (task and session.durable.get_object_state(uid, task["object_id"]) !=
         {k:v for k,v in task.items() if k != "object_id"})):
        raise ValueError("product state changed during preparation")
    return packages
