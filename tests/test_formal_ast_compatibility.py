"""Frozen V1 AST serialization across the 3.12 / 3.13 dump option boundary."""
import ast
import importlib.util
import inspect
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules/agent-integration"))
sys.path.insert(0, str(ROOT / "evaluation/cbt-memory-effect"))
sys.path.insert(0, str(ROOT / "evaluation/architecture-rerun"))
SPEC = importlib.util.spec_from_file_location("formal_ast_option_unit", ROOT / "evaluation/architecture-rerun/formal_continuation.py")
continuation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(continuation)
ORIGINAL_DUMP = ast.dump


def full_field_dump(node, **options):
    if "show_empty" in inspect.signature(ORIGINAL_DUMP).parameters:
        options["show_empty"] = True
    return ORIGINAL_DUMP(node, **options)


class AstOptionCompatibilityTests(unittest.TestCase):
    def test_legacy_dump_does_not_receive_an_unsupported_option(self):
        calls = []
        def old_dump(node, annotate_fields=True, include_attributes=False, *, indent=None):
            calls.append(include_attributes)
            return full_field_dump(node, annotate_fields=annotate_fields, include_attributes=include_attributes, indent=indent)
        tree = ast.parse("def unchanged():\n    pass\n")
        expected = full_field_dump(tree, include_attributes=False)
        with patch.object(continuation.ast, "dump", old_dump):
            self.assertEqual(continuation._v1_ast_dump(tree), expected)
        self.assertEqual(calls, [False])

    def test_new_dump_option_is_explicit_and_preserves_both_frozen_hashes(self):
        # The new default is deliberately different. Returning the actual old
        # full-field dump only when requested catches the original regression.
        calls = []
        def new_dump(node, annotate_fields=True, include_attributes=False, *, indent=None, show_empty=False):
            calls.append(show_empty)
            if not show_empty:
                return "optional empty AST lists omitted by the new default"
            return full_field_dump(node, annotate_fields=annotate_fields, include_attributes=include_attributes, indent=indent)
        with patch.object(continuation.ast, "dump", new_dump):
            for name, (original_source_hash, frozen_ast_hash) in continuation.V1_AST.items():
                self.assertEqual(continuation._v1_ast_compatibility(name, original_source_hash), frozen_ast_hash)
        self.assertEqual(calls, [True, True])

    def test_preserving_empty_fields_never_allows_changed_v1_code(self):
        def new_dump(node, annotate_fields=True, include_attributes=False, *, indent=None, show_empty=False):
            if not show_empty:
                return "optional empty lists omitted"
            return full_field_dump(node, annotate_fields=annotate_fields, include_attributes=include_attributes, indent=indent)
        with tempfile.TemporaryDirectory() as tmp:
            frozen_root = Path(tmp)
            for name, (original_source_hash, _) in continuation.V1_AST.items():
                target = frozen_root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text((ROOT / name).read_text() + "\nALTERED_V1_BEHAVIOR = True\n")
                with patch.object(continuation.full, "ROOT", frozen_root), patch.object(continuation.ast, "dump", new_dump):
                    with self.assertRaisesRegex(ValueError, "frozen V1 projection behavior changed"):
                        continuation._v1_ast_compatibility(name, original_source_hash)


if __name__ == "__main__":
    unittest.main()
