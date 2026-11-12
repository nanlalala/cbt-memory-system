"""Pure checks that loaded project modules belong to one explicit checkout.

The check never imports project code, changes ``sys.modules`` or ``sys.path``,
accesses credentials, or starts work. Equal source bytes in another checkout
are insufficient: cached module globals may still resolve paths from that root.
Restart an idle runtime after changing executable files in the same checkout.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys


class RuntimeOriginError(RuntimeError):
    """A loaded project module belongs to a different repository checkout."""


def repository_root(path: str | Path) -> Path | None:
    """Find the nearest ancestor with this project's two directory markers."""
    candidate = Path(path).expanduser().resolve()
    for root in (candidate, *candidate.parents):
        if (root / 'modules').is_dir() and (root / 'evaluation').is_dir():
            return root
    return None


def assert_repository_runtime(expected_root: str | Path, *, loaded_modules: Mapping | None = None) -> dict:
    """Reject foreign repository origins before admitting an external request.

    File/spec origins and package search paths are inspected. ``loaded_modules``
    is a read-only test seam; normal callers inspect a snapshot of sys.modules.
    Modules outside any marked repository are unrelated dependencies and ignored.
    This is an origin check, not proof that in-place source edits were reloaded.
    """
    expected = Path(expected_root).expanduser().resolve()
    if repository_root(expected) != expected:
        raise ValueError('expected runtime root must contain modules and evaluation directories')
    snapshot = dict(sys.modules if loaded_modules is None else loaded_modules)
    checked, mismatches = {}, []
    for name, module in sorted(snapshot.items()):
        if module is None:
            continue
        try:
            attributes = vars(module)
        except TypeError:
            continue
        spec = attributes.get('__spec__')
        origins = {origin for origin in (attributes.get('__file__'), getattr(spec, 'origin', None))
                   if isinstance(origin, str) and origin and origin not in {'built-in', 'frozen'}}
        package_paths = attributes.get('__path__', ())
        if not isinstance(package_paths, (str, bytes)):
            origins.update(path for path in package_paths if isinstance(path, str) and path)
        owned = []
        for origin in sorted(origins):
            resolved = Path(origin).expanduser().resolve()
            owner = repository_root(resolved)
            if owner is None:
                continue
            owned.append(str(resolved))
            if owner != expected:
                mismatches.append({'module': name, 'origin': str(resolved), 'repository_root': str(owner)})
        if owned:
            checked[name] = owned
    if mismatches:
        details = '; '.join(row['module'] + '=' + row['origin'] for row in mismatches[:12])
        raise RuntimeOriginError('foreign repository modules are loaded; expected ' + str(expected) +
                                 '. Restart an idle runtime or clear all repository modules before retrying: ' + details)
    return {'repository_root': str(expected), 'checked_module_count': len(checked), 'module_origins': checked}
