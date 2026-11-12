"""Rehydrate saved assistant replies without promoting them to user sources.

A reply is a derived source: every original send dependency and object version
must still be live. Operational commit/completion times prove dialogue order;
they are never a clinical or user-reported date. Legacy rows with no completion
metadata remain available through their original sources, not guessed aliases.
"""
from __future__ import annotations

from datetime import datetime
import json


def _host_time(value):
    if not isinstance(value, str):
        return None
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return None
    return result if result.tzinfo is not None else None


def operational_dialogue_key(source):
    """Order known host events; keep legacy original insertion order first.

    Unknown legacy source commit times predate the new chronology metadata.
    They are usable original records, but cannot establish a guessed assistant
    completion event. Equal known times put inputs before assistant replies.
    """
    stamp = _host_time(source.get('host_completed_at') or source.get('host_committed_at'))
    rowid = source.get('source_rowid', source.get('bound_source_rowid', 0))
    if stamp is None:
        return (0, rowid, 0, source['source_id'])
    return (1, stamp, int('host_completed_at' in source), rowid, source['source_id'])


def retained_assistant_sources(durable, user_id, *, cutoff_source_id=None,
                               as_of=None, purpose='response', excluded_source_ids=()):
    """Return virtual assistant sources admitted for this concrete readout.

    For a queued projection/readout, ``cutoff_source_id`` bounds both original
    input order and actual completion time. A reply finishing after that input
    was committed is excluded even when its original input is earlier. The
    caller expands ``dependency_source_ids`` for persistent send/effect fences.
    """
    if purpose not in {'response', 'projection', 'verification', 'extraction'}:
        raise ValueError('unknown retained-dialogue purpose')
    permission = 'response_external' if purpose == 'response' else 'extraction_external'
    excluded = set(excluded_source_ids)
    cutoff_rowid = cutoff_time = None
    if cutoff_source_id is not None:
        cutoff = durable.connection.execute(
            'SELECT rowid,committed_at FROM dm_sources WHERE user_id=? AND source_id=?',
            (user_id, cutoff_source_id)).fetchone()
        if cutoff is None or (cutoff_time := _host_time(cutoff['committed_at'])) is None:
            return {}
        cutoff_rowid = cutoff['rowid']
    result = {}
    rows = durable.connection.execute(
        "SELECT a.*,s.rowid AS bound_rowid,s.source_id AS bound_source_id "
        "FROM dm_attempts a JOIN dm_sources s ON s.user_id=a.user_id AND s.input_id=a.current_input_id "
        "WHERE a.user_id=? AND a.purpose='response' AND a.status='succeeded' "
        "AND a.response_json IS NOT NULL AND a.can_retain_response=1 "
        "ORDER BY s.rowid,a.completed_at,a.attempt_id", (user_id,)).fetchall()
    for row in rows:
        completed = _host_time(row['completed_at'])
        if completed is None or (cutoff_rowid is not None and
                (row['bound_rowid'] >= cutoff_rowid or completed >= cutoff_time)):
            continue
        alias = 'assistant:' + row['current_input_id']
        if alias in excluded or row['bound_source_id'] in excluded:
            continue
        # A virtual identity must never shadow a separately admitted original.
        if durable.connection.execute('SELECT 1 FROM dm_sources WHERE user_id=? AND source_id=?',
                                      (user_id,alias)).fetchone():
            continue
        receipt = durable.get_attempt(user_id, row['request_key'],
                                      run_id=row['run_id'], session_id=row['session_id'])
        response = (receipt or {}).get('response')
        if (not receipt or not receipt['publishable'] or type(response) is not dict or
                type(response.get('text')) is not str or not response['text'].strip()):
            continue
        dependency_ids = json.loads(row['source_ids_json'])
        if not dependency_ids or any(sid in excluded for sid in dependency_ids):
            continue
        sources = [durable.get_source(user_id, sid, as_of=as_of) for sid in dependency_ids]
        if any(source is None or any(source['permissions'].get(k) is not True
                for k in ('save', 'long_term_use', permission)) for source in sources):
            continue
        result[alias] = {
            'user_id':user_id, 'source_id':alias, 'role':'assistant', 'text':response['text'],
            'source_time':None, 'permissions':{k:all(source['permissions'].get(k) is True for source in sources)
                for k in ('save','long_term_use','response_external','extraction_external')},
            'dependency_source_ids':dependency_ids, 'object_versions':receipt['object_versions'],
            'response_attempt_id':row['attempt_id'], 'bound_source_id':row['bound_source_id'],
            'bound_source_rowid':row['bound_rowid'],
            'host_completed_at':row['completed_at'],
        }
    return result


def expanded_source_dependencies(sources):
    """Flatten virtual aliases to distinct original source dependencies."""
    return sorted({sid for source in sources for sid in
                   source.get('dependency_source_ids', [source['source_id']])})


def source_object_versions(sources):
    """Merge the exact object versions used by retained derived replies."""
    expected = {}
    for source in sources:
        for object_id, version in source.get('object_versions', {}).items():
            if object_id in expected and expected[object_id] != version:
                raise ValueError('inconsistent retained reply object dependencies')
            expected[object_id] = version
    return expected
