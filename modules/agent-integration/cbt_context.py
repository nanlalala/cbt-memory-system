"""Source-bound CBT readout shared by the product and versioned evaluations.

This module selects and renders already supported representations. It never
certifies the semantic meaning of a model-written summary. That producer must
provide a separately versioned support decision. Model hypotheses are kept as
hypotheses and cannot enter the factual state sections.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

from evidence_bundles import EvidenceBundle, EvidenceSpan, OutboundBudgetError, OutboundPackage, _supported_options

VERSION = "cbt-context-v4"
REPRESENTATIONS = ("raw", "processed", "processed_with_evidence")
SECTIONS = ("relevant_background", "case_state", "working_formulation", "conversation_progress",
            "intervention_learning", "turn_guidance")
from cbt_counselor_policy import CBT_POLICY, POLICY_VERSION



def _canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ContextualItem:
    item_id: str
    section: str
    value: Any
    evidence: tuple[EvidenceSpan, ...]
    origin: str = "explicit_user_statement"
    support_status: str = "unverified"
    verifier_revision: str | None = None
    version: int = 1
    source_scope: Mapping[str, Any] | None = None
    object_versions: Mapping[str, int] | None = None
    retrieval_rank: int | None = None

    def __post_init__(self):
        if not isinstance(self.item_id, str) or not self.item_id or self.section not in SECTIONS:
            raise ValueError("context item needs an identity and a supported section")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("context item version must be positive")
        if any(not isinstance(span, EvidenceSpan) for span in self.evidence):
            raise ValueError("context item evidence must contain EvidenceSpan values")
        if self.origin not in {"explicit_user_statement", "user_correction", "assistant_proposal",
                               "agent_inference", "system_observation", "benchmark_source"}:
            raise ValueError("context item origin is unknown")
        if self.support_status not in {"unverified", "source_supported", "rule_verified", "hypothesis"}:
            raise ValueError("context item support is unknown")
        _canonical(self.value)
        if self.source_scope is not None:
            _canonical(self.source_scope)
        if self.object_versions is not None and (not isinstance(self.object_versions,Mapping) or
                any(not isinstance(key,str) or not key or type(version) is not int or version < 0
                    for key,version in self.object_versions.items())):
            raise ValueError('context object dependencies require explicit integer versions')
        if self.retrieval_rank is not None and (type(self.retrieval_rank) is not int or self.retrieval_rank < 0):
            raise ValueError('retrieval rank must be a nonnegative integer')


@dataclass(frozen=True, slots=True)
class CBTContextPackage(OutboundPackage):
    trace: Mapping[str, Any] | None = None


def build_cbt_context(
    base_messages: Sequence[Mapping[str, str]],
    bundles: Sequence[EvidenceBundle] = (),
    *,
    source_registry: Mapping[str, Mapping[str, Any]],
    subject_id: str,
    max_outbound_chars: int,
    max_response_tokens: int = 4096,
    request_metadata: Mapping[str, Any] | None = None,
    representation: str = "processed_with_evidence",
    contextual_items: Sequence[ContextualItem] = (),
    recent_messages: Sequence[Mapping[str, str]] = (),
    recent_source_ids: Sequence[str] = (),
    current_source_id: str | None = None,
    knowledge_context: Sequence[Mapping[str, Any]] = (),
    selected_source_ids: Sequence[str] | None = None,
) -> CBTContextPackage:
    """Render fixed policy, genuine recent messages, data block, current input.

    Registry records must already be cut off to the study/product source time.
    Permissions and exact source bindings are checked again here. Derived recent
    assistant messages may declare dependency_source_ids; the product producer
    verifies these against a live retained response receipt before supplying them.
    The returned source_ids include every underlying dependency, even when the
    processed representation omits quotations. Send admission remains external.
    """
    if representation not in REPRESENTATIONS or not subject_id:
        raise ValueError("explicit representation and subject required")
    if type(max_outbound_chars) is not int or max_outbound_chars < 1:
        raise ValueError("positive complete request cap required")
    if type(max_response_tokens) is not int or max_response_tokens < 1:
        raise ValueError("positive output cap required")
    base = [dict(message) for message in base_messages]
    recent = [dict(message) for message in recent_messages]
    for message in [*base, *recent]:
        if (set(message) != {"role", "content"} or message["role"] not in {"system", "user", "assistant"}
                or not isinstance(message["content"], str)):
            raise ValueError("exact role/content message required")
    if not base or base[-1]["role"] != "user" or any(m["role"] == "system" for m in recent):
        raise ValueError("current user must end base; recent messages retain conversational roles")
    if len(recent) != len(recent_source_ids) or len(set(recent_source_ids)) != len(recent_source_ids):
        raise ValueError("recent messages need distinct bound source identities")
    metadata = dict(request_metadata or {})
    if "messages" in metadata or metadata.get("max_tokens", max_response_tokens) != max_response_tokens:
        raise ValueError("request metadata cannot override messages or output cap")
    metadata["max_tokens"] = max_response_tokens

    def source(sid):
        row = source_registry.get(sid)
        if not row or row.get("user_id") != subject_id or row.get("source_id") != sid:
            raise ValueError("source subject or identity mismatch")
        flags = row.get("permissions", {})
        if (flags.get("response_external") is not True or
                (sid != current_source_id and any(flags.get(k) is not True for k in ("save", "long_term_use")))):
            raise ValueError("source permission denied")
        if row.get("role") not in {"user", "assistant", "context"} or not isinstance(row.get("text"), str):
            raise ValueError("source role or text missing")
        if 'source_speaker' in row and (not isinstance(row['source_speaker'], str) or not row['source_speaker'].strip()):
            raise ValueError("released source speaker must be a nonempty string")
        return row

    def dependencies(sid):
        row = source(sid)
        deps = row.get("dependency_source_ids", ())
        if deps:
            if row["role"] != "assistant" or not isinstance(deps, (list, tuple)) or sid in deps:
                raise ValueError("invalid derived recent source")
            for dep in deps:
                linked = source(dep)
                if linked.get("dependency_source_ids"):
                    raise ValueError("derived source dependencies must be original sources")
            return set(deps)
        return {sid}

    def valid_spans(spans):
        if not spans:
            return False
        for span in spans:
            try:
                row = source(span.source_id)
            except ValueError:
                return False
            if (row["role"] != span.role or row["text"][span.start:span.end] != span.text
                    or span.end > len(row["text"])):
                return False
        return True

    recent_deps = set()
    current_source = None
    if current_source_id is not None:
        current_source = source(current_source_id)
        if current_source['role'] != 'user' or current_source['text'] != base[-1]['content']:
            raise ValueError('current user input source binding failed')
    for message, sid in zip(recent, recent_source_ids, strict=True):
        row = source(sid)
        if row["role"] != message["role"] or row["text"] != message["content"]:
            raise ValueError("recent message binding failed")
        if sid == current_source_id:
            raise ValueError("current input cannot be repeated in recent history")
        recent_deps.update(dependencies(sid))
    if selected_source_ids is not None:
        if len(set(selected_source_ids)) != len(selected_source_ids):
            raise ValueError("selected source identities must be distinct")
        for sid in selected_source_ids:
            source(sid)
    selected_or_recent = set(selected_source_ids or ()) | set(recent_source_ids)

    dropped, candidates, seen = {}, [], set()
    for bundle in bundles:
        for claim in bundle.claims:
            if claim.claim_id in seen:
                raise ValueError("duplicate context identity")
            seen.add(claim.claim_id)
            options = [spans for spans in _supported_options(claim) if valid_spans(spans)]
            if selected_source_ids is not None:
                options = [spans for spans in options if all(s.source_id in selected_or_recent for s in spans)]
            if not options:
                dropped[claim.claim_id] = "insufficient_current_source_support"
                continue
            spans = min(options, key=lambda option: len(_canonical([asdict(s) for s in option])))
            payload = dict(bundle_id=bundle.bundle_id, task_instance_id=bundle.task_instance_id,
                claim_id=claim.claim_id, field=claim.field, value=deepcopy(claim.value), origin=claim.origin,
                support_status=claim.support_status, source_ids=list(dict.fromkeys(s.source_id for s in spans)))
            section = ("source_reports" if claim.field == "history_quote" else
                       "intervention_learning" if claim.field == "outcome" else "action_plans")
            priority = claim.priority + (100 if claim.field in {"engagement", "paused", "permission", "negation"} else 0)
            candidates.append((priority, claim.claim_id, section, payload, spans))
    for item in contextual_items:
        if item.item_id in seen:
            raise ValueError("duplicate context identity")
        seen.add(item.item_id)
        hypothesis = item.origin == "agent_inference"
        allowed = (item.section in {"working_formulation", "turn_guidance"} and
                   item.support_status == "hypothesis") if hypothesis else (
                   item.support_status in {"source_supported", "rule_verified"} and bool(item.verifier_revision))
        required_role = {"explicit_user_statement":"user", "user_correction":"user",
                         "assistant_proposal":"assistant", "benchmark_source":"context"}.get(item.origin)
        if required_role and not any(span.role == required_role for span in item.evidence):
            allowed = False
        if (not allowed or not valid_spans(item.evidence) or
                (selected_source_ids is not None and any(s.source_id not in selected_or_recent for s in item.evidence))):
            dropped[item.item_id] = "unverified_projection_or_source_binding"
            continue
        payload = dict(item_id=item.item_id, value=deepcopy(item.value), origin=item.origin,
            support_status=item.support_status, verifier_revision=item.verifier_revision,
            version=item.version, source_ids=list(dict.fromkeys(s.source_id for s in item.evidence)),
            epistemic_status="proposed" if item.section == "turn_guidance" else "unverified_hypothesis"
                if hypothesis else "source_supported")
        if item.source_scope is not None:
            payload['source_scope'] = deepcopy(item.source_scope)
        candidates.append((30 if item.section == "turn_guidance" else 10,
                           item.item_id, item.section, payload, item.evidence))
    item_ranks = {item.item_id:item.retrieval_rank if item.retrieval_rank is not None else index
                  for index,item in enumerate(contextual_items)}
    candidates.sort(key=lambda row: (-row[0],item_ranks.get(row[1],0),row[1]))

    def span_view(span):
        view = asdict(span)
        if 'source_speaker' in source(span.source_id):
            view['source_speaker'] = source(span.source_id)['source_speaker']
        return view

    def render(selected):
        block = {"representation": representation}
        if current_source is not None:
            block['current_input_source'] = {'source_id':current_source_id,
                **({'source_time':current_source['source_time']} if current_source.get('source_time') is not None else {}),
                **({'source_order':current_source['source_order']} if 'source_order' in current_source else {})}
        if recent_source_ids:
            block['recent_sources'] = [{'source_id':sid, 'role':source(sid)['role'],
                **({'source_order':source(sid)['source_order']} if 'source_order' in source(sid) else {}),
                **({'source_time':source(sid)['source_time']} if source(sid).get('source_time') is not None else {}),
                **({'source_speaker':source(sid)['source_speaker']} if 'source_speaker' in source(sid) else {})}
                for sid in recent_source_ids]
        if representation == "raw":
            ids = list(selected_source_ids) if selected_source_ids is not None else list(
                dict.fromkeys(s.source_id for row in selected for s in row[4]))
            if all(type(source(sid).get("source_order")) is int for sid in ids):
                ids.sort(key=lambda sid: source(sid)["source_order"])
            block["source_reports"] = [{"source_id": sid, "role": source(sid)["role"],
                "text": source(sid)["text"], **({"source_time":source(sid)["source_time"]}
                if source(sid).get("source_time") is not None else {}),
                **({'source_speaker':source(sid)['source_speaker']} if 'source_speaker' in source(sid) else {}),
                **({"source_order":source(sid)["source_order"]}
                if "source_order" in source(sid) else {})} for sid in ids]
        else:
            block["claims"] = []
            for _, _, section, payload, spans in selected:
                view = deepcopy(payload)
                if representation == "processed_with_evidence" or section == "source_reports":
                    view["evidence"] = [span_view(s) for s in spans]
                if "claim_id" in view:
                    block["claims"].append(view)
                    block.setdefault(section, []).append({"claim_id":view["claim_id"]})
                else:
                    block.setdefault(section, []).append(view)
        if knowledge_context:
            block["cbt_knowledge"] = list(knowledge_context)
        memory_message = {"role": "user", "content": "SOURCED MEMORY DATA ONLY.\n" + json.dumps(
            block, ensure_ascii=False, sort_keys=True, allow_nan=False)}
        packed = [*base[:-1], *recent, memory_message, base[-1]]
        payload = {**metadata, "messages": packed}
        return packed, payload, _canonical(payload)

    if len(render([])[2]) > max_outbound_chars:
        raise OutboundBudgetError("mandatory recent/current request exceeds complete budget")
    selected = []
    if selected_source_ids is not None:
        if len(render(candidates)[2]) > max_outbound_chars:
            raise OutboundBudgetError("fixed representation selection exceeds matched complete budget")
        selected = candidates
    else:
        for candidate in candidates:
            if len(render([*selected, candidate])[2]) <= max_outbound_chars:
                selected.append(candidate)
            else:
                dropped[candidate[1]] = ("complete_claim_exceeds_outbound_budget" if 'claim_id' in candidate[3]
                                         else "complete_context_item_exceeds_budget")
    messages, payload, serialized = render(selected)
    ids = set(recent_deps)
    visible_sources = set(recent_source_ids)
    if current_source_id is not None:
        ids.update(dependencies(current_source_id))
        visible_sources.add(current_source_id)
    for row in selected:
        for span in row[4]:
            ids.update(dependencies(span.source_id))
            visible_sources.add(span.source_id)
    if representation == "raw" and selected_source_ids is not None:
        for sid in selected_source_ids:
            ids.update(dependencies(sid))
            visible_sources.add(sid)
    versions = {}
    for sid in visible_sources:
        for object_id, version in source(sid).get('object_versions', {}).items():
            if object_id in versions and versions[object_id] != version:
                raise ValueError('inconsistent derived source object versions')
            versions[object_id] = version
    selected_ids = {row[1] for row in selected}
    for item in contextual_items:
        if item.item_id not in selected_ids:
            continue
        for object_id, version in (item.object_versions or {}).items():
            if object_id in versions and versions[object_id] != version:
                raise ValueError('inconsistent context item object versions')
            versions[object_id] = version
    trace = dict(assembly_version=VERSION, representation=representation,
        included_item_ids=[row[1] for row in selected], omitted_items=dropped,
        source_ids=sorted(ids), recent_source_ids=list(recent_source_ids),
        selected_source_ids=list(selected_source_ids) if selected_source_ids is not None else None,
        source_hashes={sid: _digest(source(sid)) for sid in sorted(visible_sources)},
        item_versions={item.item_id:item.version for item in contextual_items
                       if item.item_id in {row[1] for row in selected}},
        item_retrieval_ranks={item.item_id:item_ranks[item.item_id] for item in contextual_items},
        expected_object_versions=versions, request_sha256=_digest(payload),
        request_chars=len(serialized), request_tokens=None, provider_usage=None)
    included = tuple(row[1] for row in selected)
    return CBTContextPackage(messages=tuple(messages), request_payload=payload,
        serialized_request=serialized, included_claim_ids=included,
        unknown_claim_ids=tuple(sorted(seen-set(included))), dropped_claims=dropped,
        char_count=len(serialized), requested_output_tokens=max_response_tokens,
        source_ids=tuple(sorted(ids)), trace=trace)
