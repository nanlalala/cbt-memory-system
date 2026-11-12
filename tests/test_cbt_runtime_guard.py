"""Pure origin fixtures and rejection before send admission; no live runtime."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'modules/agent-integration'))
from cbt_runtime_guard import RuntimeOriginError, assert_repository_runtime, repository_root


def marked_repository(path):
    (path / 'modules/agent-integration').mkdir(parents=True)
    (path / 'evaluation/cbt-memory-effect').mkdir(parents=True)
    source = path / 'modules' / 'same.py'
    source.write_text('IDENTICAL = True\n')
    return source


def module_at(path):
    module = ModuleType('fixture')
    module.__file__ = str(path)
    module.__spec__ = SimpleNamespace(origin=str(path))
    return module


class RuntimeOriginTests(unittest.TestCase):
    def test_third_party_generic_modules_and_evaluation_directories_are_not_project_markers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            current = marked_repository(root / 'current')
            dependency = root / 'site-packages/sentence_transformers/base'
            (dependency / 'modules').mkdir(parents=True)
            (dependency / 'evaluation').mkdir()
            foreign_looking_source = dependency / 'modules/fixture.py'
            foreign_looking_source.write_text('THIRD_PARTY = True\n')
            namespace = ModuleType('dependency_namespace')
            namespace.__path__ = [str(dependency / 'evaluation')]
            result = assert_repository_runtime(root / 'current', loaded_modules={
                'project': module_at(current),
                'sentence_transformers.base.modules.fixture': module_at(foreign_looking_source),
                'sentence_transformers.base.evaluation': namespace,
            })
            self.assertIsNone(repository_root(foreign_looking_source))
            self.assertEqual(result['checked_module_count'], 1)

    def test_current_checkout_and_unrelated_dependencies_are_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = marked_repository(root / 'current')
            unrelated = root / 'dependency.py'
            unrelated.write_text('EXTERNAL = True\n')
            loaded = {'project': module_at(source), 'dependency': module_at(unrelated), 'none': None}
            result = assert_repository_runtime(root / 'current', loaded_modules=loaded)
            self.assertEqual(result['checked_module_count'], 1)
            self.assertEqual(repository_root(source), root / 'current')
            self.assertIsNone(repository_root(unrelated))
            self.assertEqual(set(loaded), {'project', 'dependency', 'none'})

    def test_identical_bytes_and_generic_module_name_do_not_allow_foreign_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            current = marked_repository(root / 'new')
            foreign = marked_repository(root / 'old-without-cbt-prefix')
            self.assertEqual(current.read_bytes(), foreign.read_bytes())
            with self.assertRaisesRegex(RuntimeOriginError, 'durable_memory'):
                assert_repository_runtime(root / 'new', loaded_modules={'durable_memory': module_at(foreign)})

    def test_spec_origin_and_namespace_package_paths_are_also_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            current = marked_repository(root / 'current')
            foreign = marked_repository(root / 'foreign')
            module = module_at(current)
            module.__spec__.origin = str(foreign)
            with self.assertRaises(RuntimeOriginError):
                assert_repository_runtime(root / 'current', loaded_modules={'mixed': module})
            namespace = ModuleType('namespace')
            namespace.__path__ = [str(foreign.parent)]
            with self.assertRaises(RuntimeOriginError):
                assert_repository_runtime(root / 'current', loaded_modules={'namespace': namespace})

    def test_mab_rejects_foreign_module_before_intent_file_or_transport(self):
        spec = importlib.util.spec_from_file_location('runtime_guard_mab_fixture',
            ROOT / 'evaluation/architecture-rerun/mab_full.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            foreign = marked_repository(directory / 'other-checkout')
            run = directory / 'run'
            run.mkdir()
            calls = []
            with patch.dict(sys.modules, {'foreign_generic_fixture': module_at(foreign)}), \
                    patch.object(runner.old, 'require_colab'), \
                    patch.object(runner, 'load', side_effect=AssertionError('must reject before reading prepared requests')):
                with self.assertRaises(RuntimeOriginError):
                    runner.generate(run, 1, transport=lambda payload: calls.append(payload))
            self.assertEqual(calls, [])
            self.assertFalse((run / 'journal.jsonl').exists())
            self.assertFalse((run / '.send.lock').exists())

    def test_journal_hosted_guard_precedes_intent_read_or_admission(self):
        sys.path.insert(0, str(ROOT / 'evaluation/cbt-memory-effect'))
        from architecture_retest import journal
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            foreign = marked_repository(directory / 'other-checkout')
            calls = []
            # Inject only the runtime predicate to exercise the admission gate.
            # No Colab environment, credential or model execution is simulated.
            with patch.dict(sys.modules, {'foreign_generic_fixture': module_at(foreign)}), \
                    patch.object(journal, '_hosted_runtime_present', return_value=True), \
                    patch.object(journal, 'rows', side_effect=AssertionError('must guard before reading intents')):
                with self.assertRaises(RuntimeOriginError):
                    journal.Journal(directory, {}, lambda payload: calls.append(payload), 1)
            self.assertEqual(calls, [])
            self.assertFalse((directory / 'intents.jsonl').exists())
            self.assertFalse((directory / 'outcomes.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
