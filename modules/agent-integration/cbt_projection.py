"""Production wrappers for the shared CBT extraction/support contract.

The evaluator and product use the same prompt and semantic verdict parser.
A supported verdict remains a model assessment, with source scope, provenance and
user endorsement kept separate. Persistent send admission belongs to the worker.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib

from cbt_projection_contract import (EXTRACT, VERIFY, VERIFIER_VERSION, bind_candidates,
                                     canonical, checked_items, fingerprint, request, strict_json)
from cbt_projection_contract import BINDING_VERSION, bind_candidates_with_status

VERSION = 'cbt-source-projection-v1'
VERIFIER_REVISION = VERIFIER_VERSION
EXTRACTION_DOMAIN_GUIDE = EXTRACT
SUPPORT_GUIDE = VERIFY
FACT_SECTIONS = frozenset({'case_state', 'relevant_background', 'conversation_progress', 'intervention_learning'})
SECTION_TAGS = {'cbt:' + section: section for section in FACT_SECTIONS}
MEMORY_SECTIONS = {'preference':'relevant_background', 'goal':'relevant_background',
                   'cbt_assignment':'intervention_learning', 'assignment_outcome':'intervention_learning'}


def visible_sources(sources):
    return [{'source_id':s['source_id'], 'role':s['role'], 'content':s['text'],
             'source_time':s.get('source_time')} for s in sources]


def extraction_messages(sources):
    return request(EXTRACT, {'history':visible_sources(sources)})


def projection_candidates(raw, sources):
    """Bind shared extraction output to real sources without declaring it true."""
    bound = bind_candidates(raw, visible_sources(sources))
    by_id = {s['source_id']:s for s in sources}
    result = []
    for item in bound:
        value = deepcopy(item)
        value['source_ids'] = list(dict.fromkeys(e['source_id'] for e in item['evidence']))
        value['source_scope'] = {'kind':'historical_source_reports',
            'sources':[{'source_id':sid, 'source_time':by_id[sid].get('source_time')} for sid in value['source_ids']]}
        value['evidence'] = [{**e, 'role':by_id[e['source_id']]['role'],
                             'source_time':by_id[e['source_id']].get('source_time')} for e in item['evidence']]
        value['projection_version'] = VERSION
        result.append(value)
    return result


def projection_candidates_with_status(raw, sources):
    """Explicit V2 binding; retained candidates still need support checking."""
    bound, statuses = bind_candidates_with_status(raw, visible_sources(sources))
    # The representation/enrichment contract is unchanged; its separate binding
    # revision lives in the audit. Valid unchanged batches retain exact payloads.
    return _enrich_candidates(bound, sources, VERSION), statuses


def _enrich_candidates(bound, sources, version):
    by_id = {s['source_id']:s for s in sources}
    result = []
    for item in bound:
        value = deepcopy(item)
        value['source_ids'] = list(dict.fromkeys(e['source_id'] for e in item['evidence']))
        value['source_scope'] = {'kind':'historical_source_reports',
            'sources':[{'source_id':sid, 'source_time':by_id[sid].get('source_time')} for sid in value['source_ids']]}
        value['evidence'] = [{**e, 'role':by_id[e['source_id']]['role'],
                             'source_time':by_id[e['source_id']].get('source_time')} for e in item['evidence']]
        value['projection_version'] = version
        result.append(value)
    return result


def propose_projections(records, source):
    """Explicit compatibility for retained legacy candidate extraction results."""
    items = []
    for record in records:
        row = record.to_dict() if hasattr(record, 'to_dict') else record
        if row.get('memory_type') in {'cognitive_pattern', 'safety_signal'}:
            continue
        evidence = row.get('source_evidence') or {}
        if (evidence.get('turn_ids') != [source['source_id']] or
            not evidence.get('supporting_text') or evidence['supporting_text'] not in source['text']):
            raise ValueError('legacy candidate must bind to the worker source')
        sections = {section for tag, section in SECTION_TAGS.items() if tag in set(row.get('tags') or [])}
        if len(sections) > 1:
            continue
        section = next(iter(sections)) if sections else MEMORY_SECTIONS.get(row.get('memory_type'), 'case_state')
        value = {'summary':row['content'], 'reported_details':deepcopy(row.get('structured_payload') or {})}
        item = {'item_id':'projection-' + fingerprint([source['source_id'], section, value])[:24],
                'section':section, 'value':value, 'origin':'explicit_user_statement',
                'evidence':[{'source_id':source['source_id'], 'quote':source['text']}]}
        if item not in items and len(items) < 16:
            items.append(item)
    return projection_candidates(canonical({'items':items}), [source])


def verification_messages(proposals, sources):
    if not proposals:
        raise ValueError('nonempty projection batch required')
    return request(VERIFY, {'sources':visible_sources(sources), 'items':proposals})


def supported_projections(raw, proposals, sources, *, verifier_model=None):
    accepted, reasons = checked_items(raw, proposals)
    by_id = {s['source_id']:s for s in sources}
    result = []
    for item in accepted:
        for evidence in item['evidence']:
            source = by_id[evidence['source_id']]
            if (source['role'] != evidence['role'] or
                source['text'][evidence['start']:evidence['end']] != evidence['text']):
                raise ValueError('projection evidence changed before publication')
        result.append({**deepcopy(item), 'kind':'supported_projection',
            'support_status':'hypothesis' if item['origin']=='agent_inference' else 'source_supported',
            'verifier_revision':VERIFIER_REVISION,
            'verification':{'decision':'supported', 'reason':reasons[item['item_id']],
                'kind':'model_support_check', 'provider_model':verifier_model,
                'response_sha256':hashlib.sha256(raw.encode()).hexdigest()}})
    return result
