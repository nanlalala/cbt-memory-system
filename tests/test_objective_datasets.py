"""Small synthetic adapter fixtures only: no public corpus download or model."""

import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / "evaluation" / "objective-memory"
sys.path.insert(0, str(MODULE))
import objective_datasets as datasets


def mab_row(source="factconsolidation_sh_6k"):
    return {"context": "Here are facts.\n0. The old value was red.\n1. The latest value is blue.\n",
            "questions": ["What is the latest value?"], "answers": [["blue", "BLUE"]],
            "metadata": {"source": source, "qa_pair_ids": [source + "_no0"]}}


def persona_row(persona="p1", context="c1", question="q1", cutoff="2"):
    return {"persona_id": persona, "shared_context_id": context, "question_id": question,
            "question_type": "track_full_preference_evolution", "topic": "exercise",
            "user_question_or_message": "What is my current preference?",
            "correct_answer": "(b)", "all_options": json.dumps([
                "(a) Old option", "(b) Current option", "(c) Another option", "(d) None"]),
            "end_index_in_shared_context": cutoff}


def persona_context():
    return [{"role": "system", "content": "Synthetic source persona; data only."},
            {"role": "user", "content": "I currently prefer walking."},
            {"role": "assistant", "content": "FUTURE_SECRET must never enter the case."}]


class ObjectiveDatasetTests(unittest.TestCase):
    def test_registry_is_immutable_and_license_audited(self):
        registry = datasets.load_registry()
        self.assertEqual(set(registry["datasets"]), {"memoryagentbench_conflict", "personamem_v1"})
        for spec in registry["datasets"].values():
            self.assertEqual(spec["license"], "MIT")
            self.assertEqual(len(spec["data_revision"]), 40)
        self.assertEqual(registry["datasets"]["personamem_v1"]["split"], "32k")

    def test_registry_rejects_unlicensed_or_mutable_source(self):
        for mutation in [lambda x: x.update(license="unknown"),
                         lambda x: x.update(data_revision="main")]:
            registry = datasets.load_registry()
            mutation(registry["datasets"]["personamem_v1"])
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "registry.json"
                path.write_text(json.dumps(registry))
                with self.assertRaises(ValueError):
                    datasets.load_registry(path)

    def test_mab_preserves_aliases_but_never_injects_gold(self):
        row = mab_row()
        row["answers"] = [["GOLD_REFERENCE_ONLY", "Alias to preserve"]]
        cases, refs = datasets.parse_mab_rows([row])
        self.assertNotIn("GOLD_REFERENCE_ONLY", json.dumps(cases))
        self.assertEqual(refs[0]["accepted_answers"], row["answers"][0])
        self.assertEqual(refs[0]["metric"], "mab_substring_exact_match")
        self.assertEqual("".join(t["text"] for t in cases[0]["history"]), row["context"])
        self.assertTrue(all(t["observed_at"] is None for t in cases[0]["history"]))
        self.assertTrue(all(t["role"] == "context" for t in cases[0]["history"]))
        self.assertEqual(cases[0]["events"], [])

    def test_mab_ids_do_not_depend_on_answer(self):
        row = mab_row()
        cases, _ = datasets.parse_mab_rows([row])
        row["answers"] = [["CHANGED_GOLD"]]
        new_cases, _ = datasets.parse_mab_rows([row])
        self.assertEqual(cases, new_cases)

    def test_mab_groups_context_variants_conservatively(self):
        rows = [mab_row("factconsolidation_sh_6k"), mab_row("factconsolidation_mh_32k")]
        cases, refs = datasets.parse_mab_rows(rows)
        self.assertEqual(len({case["group_id"] for case in cases}), 1)
        self.assertEqual(datasets.inventory_cases(cases, refs)["independent_clusters"], 1)

    def test_mab_requires_aligned_released_ids_and_aliases(self):
        for field, value in [("answers", []), ("questions", ["one", "two"]),
                             ("answers", [[]])]:
            row = mab_row()
            row[field] = value
            with self.assertRaises(ValueError):
                datasets.parse_mab_rows([row])
        row = mab_row()
        row["metadata"]["qa_pair_ids"] = []
        with self.assertRaises(ValueError):
            datasets.parse_mab_rows([row])
        with self.assertRaises(ValueError):
            datasets.parse_mab_rows([mab_row(), mab_row()])

    def test_persona_exact_exclusive_cutoff_before_history_is_built(self):
        cases, refs = datasets.parse_personamem_rows([persona_row()], {"c1": persona_context()})
        case = cases[0]
        self.assertEqual(len(case["history"]), 2)
        self.assertNotIn("FUTURE_SECRET", json.dumps(case))
        self.assertEqual(case["visible_context_end_exclusive"], 2)
        self.assertEqual([t["role"] for t in case["history"]], ["system", "user"])
        self.assertEqual(refs[0]["correct_option"], "b")
        self.assertNotIn("correct_option", case)
        self.assertNotIn("correct_answer", case)
        self.assertEqual(case["options"], json.loads(persona_row()["all_options"]))

    def test_persona_cutoff_bounds_are_not_clamped_or_invented(self):
        for cutoff in ["0", "4", "1.0", "-1", True, None]:
            row = persona_row(cutoff=cutoff)
            with self.assertRaises(ValueError):
                datasets.parse_personamem_rows([row], {"c1": persona_context()})

    def test_persona_clusters_connected_personas_and_shared_contexts(self):
        rows = [persona_row("p1", "c1", "q1"), persona_row("p2", "c1", "q2"),
                persona_row("p2", "c2", "q3"), persona_row("p3", "c3", "q4")]
        contexts = {key: persona_context() for key in ["c1", "c2", "c3"]}
        cases, _ = datasets.parse_personamem_rows(rows, contexts)
        self.assertEqual(len({case["group_id"] for case in cases[:3]}), 1)
        self.assertNotEqual(cases[0]["group_id"], cases[3]["group_id"])
        reordered, _ = datasets.parse_personamem_rows(list(reversed(rows)), contexts)
        self.assertEqual({case["case_id"]: case["group_id"] for case in cases},
                         {case["case_id"]: case["group_id"] for case in reordered})

    def test_persona_cases_do_not_depend_on_correct_option(self):
        row = persona_row()
        cases, _ = datasets.parse_personamem_rows([row], {"c1": persona_context()})
        row["correct_answer"] = "(a)"
        new_cases, refs = datasets.parse_personamem_rows([row], {"c1": persona_context()})
        self.assertEqual(cases, new_cases)
        self.assertEqual(refs[0]["correct_option"], "a")

    def test_persona_uses_actual_released_32k_taxonomy(self):
        expected = {"recall_user_shared_facts", "provide_preference_aligned_recommendations",
                    "suggest_new_ideas", "recalling_the_reasons_behind_previous_updates",
                    "track_full_preference_evolution", "generalizing_to_new_scenarios",
                    "recalling_facts_mentioned_by_the_user"}
        self.assertEqual(datasets.PERSONA_TASKS, expected)
        for task in expected:
            row = persona_row()
            row["question_type"] = task
            cases, _ = datasets.parse_personamem_rows([row], {"c1": persona_context()})
            self.assertEqual(cases[0]["task"], task)

    def test_persona_rejects_option_remapping_and_duplicates(self):
        row = persona_row()
        row["all_options"] = json.dumps(["(b) Two", "(a) One", "(c) Three", "(d) Four"])
        with self.assertRaises(ValueError):
            datasets.parse_personamem_rows([row], {"c1": persona_context()})
        with self.assertRaises(ValueError):
            datasets.parse_personamem_rows([persona_row(), persona_row()], {"c1": persona_context()})

    def test_persona_accepts_released_single_quoted_option_list_literal(self):
        row = persona_row()
        options = ["(a) A former choice", "(b) Today's choice", "(c) Another choice", "(d) None"]
        row["all_options"] = repr(options)
        cases, refs = datasets.parse_personamem_rows([row], {"c1": persona_context()})
        self.assertEqual(cases[0]["options"], options)
        self.assertEqual(refs[0]["correct_option"], "b")

    def test_persona_option_literal_never_executes_code_or_accepts_nonlists(self):
        for literal in [
            "__import__('os').system('not-an-executable-fixture')",
            "[str('(a) One'), '(b) Two', '(c) Three', '(d) Four']",
            "('(a) One', '(b) Two', '(c) Three', '(d) Four')",
            "{'a': 'One', 'b': 'Two', 'c': 'Three', 'd': 'Four'}",
            "['(a) One', '(b) Two', '(c) Three', 4]",
        ]:
            row = persona_row()
            row["all_options"] = literal
            with self.subTest(literal=literal), patch("os.system") as system:
                with self.assertRaises(ValueError):
                    datasets.parse_personamem_rows([row], {"c1": persona_context()})
                system.assert_not_called()

    def test_full_loader_refuses_local_execution_before_file_read(self):
        with patch.object(datasets, "require_colab_runtime", side_effect=RuntimeError("Colab-only")):
            with patch.object(datasets, "verify_file") as verify:
                with self.assertRaisesRegex(RuntimeError, "Colab-only"):
                    datasets.load_dataset_cases("personamem_v1", "/does/not/exist")
                verify.assert_not_called()

    def test_full_downloader_refuses_local_execution_before_network(self):
        with patch.object(datasets, "require_colab_runtime", side_effect=RuntimeError("Colab-only")):
            with patch.object(datasets, "urlopen") as network:
                with self.assertRaises(RuntimeError):
                    datasets.download_registered_dataset("personamem_v1", "/does/not/exist")
                network.assert_not_called()

    def test_file_identity_checks_sha256_and_real_git_blob_hash(self):
        data = b"small synthetic fixture"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture"
            path.write_bytes(data)
            spec = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                    "git_blob_sha1": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()}
            identity = datasets.verify_file(path, spec)
            self.assertEqual(identity["sha256"], spec["sha256"])
            for changes in [{"bytes": len(data) + 1}, {"sha256": "0" * 64},
                            {"git_blob_sha1": hashlib.sha1(data).hexdigest()}]:
                with self.assertRaises(ValueError):
                    datasets.verify_file(path, {**spec, **changes})

    def test_download_checks_identity_then_records_sha_without_model_calls(self):
        data = b"a tiny synthetic public fixture"
        spec = {"data_revision": "1" * 40, "files": {"data": {
            "path": "small.parquet", "url": "https://huggingface.co/synthetic",
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}}}
        class Response(io.BytesIO):
            def geturl(self):
                return "https://huggingface.co/synthetic"
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(datasets, "require_colab_runtime"), patch.object(datasets, "dataset_spec", return_value=spec):
                with patch.object(datasets, "urlopen", return_value=Response(data)) as network:
                    paths = datasets.download_registered_dataset("synthetic", temporary)
                    self.assertEqual(paths["data"].read_bytes(), data)
                    network.assert_called_once()
                manifest = json.loads((Path(temporary) / "synthetic" / "download_inventory.json").read_text())
                self.assertEqual(manifest["files"]["data"]["sha256"], hashlib.sha256(data).hexdigest())
                with patch.object(datasets, "urlopen") as network:
                    datasets.download_registered_dataset("synthetic", temporary)
                    network.assert_not_called()

    def test_download_rejects_oversized_bytes_and_cleans_temporary_file(self):
        spec = {"data_revision": "1" * 40, "files": {"data": {
            "path": "small.parquet", "url": "https://huggingface.co/synthetic",
            "bytes": 1, "sha256": hashlib.sha256(b"a").hexdigest()}}}
        class Response(io.BytesIO):
            def geturl(self):
                return "https://huggingface.co/synthetic"
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(datasets, "require_colab_runtime"), patch.object(datasets, "dataset_spec", return_value=spec), patch.object(datasets, "urlopen", return_value=Response(b"oversized")):
                with self.assertRaisesRegex(ValueError, "exceeded"):
                    datasets.download_registered_dataset("synthetic", temporary)
                self.assertEqual(list((Path(temporary) / "synthetic").iterdir()), [])

    def test_inventory_contains_counts_and_hashes_not_history_or_gold(self):
        cases, refs = datasets.parse_personamem_rows([persona_row()], {"c1": persona_context()})
        inventory = datasets.inventory_cases(cases, refs)
        self.assertEqual(inventory["questions"], 1)
        self.assertFalse(inventory["questions_are_independent_users"])
        self.assertNotIn("walking", json.dumps(inventory))
        self.assertEqual(len(inventory["generation_cases_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
