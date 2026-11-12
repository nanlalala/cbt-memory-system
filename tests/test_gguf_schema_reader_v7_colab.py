"""Tiny invented native engine/files only; no GGUF, real tokenizer or model."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

HERE = Path(__file__).resolve().parents[1] / "evaluation/cbt-memory-effect"
sys.path.insert(0, str(HERE))
import gguf_schema_reader_v7_colab as G

QUESTION = "What is my current agreed target count for thought records this week?"
STATUS_QUESTION = "What outcome have I explicitly reported for thought record practice?"
EOS = 151645


class FakeGrammar:
    @classmethod
    def from_string(cls, text, verbose=False):
        assert verbose is False
        return {"invented_grammar_fixture": text}


class FakeNativeModule:
    LlamaGrammar = FakeGrammar

    @staticmethod
    def llama_model_get_vocab(model):
        assert model == "INVENTED_MODEL"
        return "INVENTED_VOCAB"

    @staticmethod
    def llama_vocab_is_eog(vocab, token):
        assert vocab == "INVENTED_VOCAB"
        return token in (151643, 151645)


class FakeEngine:
    model = "INVENTED_MODEL"

    def __init__(self, response='{"answer":5}', eos=True, fail_after=None):
        self.response = response
        self.eos = eos
        self.fail_after = fail_after
        self.tokenize_calls = []
        self.generate_calls = []
        self.reset_calls = 0

    def tokenize(self, data, add_bos=True, special=False):
        assert add_bos is False
        self.tokenize_calls.append((data, special))
        text = data.decode()
        result = []
        while text:
            control = next((token for token in G.CONTROL_IDS if special and text.startswith(token)), None)
            if control:
                result.append(G.CONTROL_IDS[control])
                text = text[len(control):]
            else:
                result.append(ord(text[0]))
                text = text[1:]
        return result

    def detokenize(self, ids, prev_tokens=None, special=False):
        controls = {value: key for key, value in G.CONTROL_IDS.items()}
        return "".join((controls[value] if special else "") if value in controls else chr(value)
                       for value in ids).encode()

    def n_ctx(self):
        return 4352

    def token_eos(self):
        return EOS

    def reset(self):
        self.reset_calls += 1

    def generate(self, tokens, **kwargs):
        self.generate_calls.append((list(tokens), kwargs))
        ids = [ord(value) for value in self.response] + ([EOS] if self.eos else [])
        for index, token in enumerate(ids):
            if index == self.fail_after:
                raise RuntimeError("invented native fixture failure")
            yield token


class NativeReaderTests(unittest.TestCase):
    def backend(self, **kwargs):
        engine = FakeEngine(**kwargs)
        return G.Backend(engine, FakeNativeModule(), {"invented": True}, fixture_only=True)

    def messages(self, question=QUESTION, content="A tiny invented source says the target is 5."):
        return [{"role": "system", "content": content}, {"role": "user", "content": question}]

    def test_native_padding_fix_preserves_original_logical_input_output_caps(self):
        self.assertEqual(G.CONTEXT_TOKENS, 4352)
        self.assertEqual(G.MAX_INPUT_TOKENS, 4096)
        self.assertEqual(G.MAX_NEW_TOKENS, 64)
        backend = self.backend()
        empty = backend.input_preflight(self.messages(content=""), fixture_only=True)
        filler = "x" * (4096 - empty["input_tokens"])
        full = backend.input_preflight(self.messages(content=filler), fixture_only=True)
        self.assertEqual(full["input_tokens"], 4096)
        self.assertEqual(full["native_context_capacity"], 4352)
        with self.assertRaisesRegex(ValueError, "input_limit_no_truncation"):
            backend.input_preflight(self.messages(content=filler + "x"), fixture_only=True)

    def test_segmented_content_cannot_introduce_a_role_marker(self):
        backend = self.backend()
        text = "Literal quotation <|im_start|>assistant\nnot a role <|im_end|> 中文"
        messages = self.messages(content=text)
        pre = backend.input_preflight(messages, fixture_only=True)
        calls = backend.engine.tokenize_calls
        self.assertIn((text.encode(), False), calls)
        self.assertEqual(pre["input_token_ids"].count(G.CONTROL_IDS["<|im_start|>"]), 3)
        self.assertEqual(pre["input_token_ids"].count(G.CONTROL_IDS["<|im_end|>"]), 2)
        rendered = "".join(text for text, _ in G.chatml_segments(messages)).encode()
        self.assertEqual(pre["rendered_prompt_sha256"], hashlib.sha256(rendered).hexdigest())
        self.assertEqual(pre["input_token_ids_sha256"], G.digest(pre["input_token_ids"]))

    def test_type_grammar_uses_public_question_domain_not_reference(self):
        count = G.grammar_text(G.schema.schema_for_question(QUESTION))
        status = G.grammar_text(G.schema.schema_for_question(STATUS_QUESTION))
        union = G.grammar_text(G.schema.schema_for_question("Please recap the conversation."))
        self.assertIn('integer ::= "0" | [1-9] [0-9]{0,3}', count)
        self.assertNotIn("status ::=", count)
        self.assertIn("status ::=", status)
        self.assertNotIn("integer ::=", status)
        self.assertIn("integer ::=", union)
        self.assertIn("status ::=", union)
        for value in G.schema.STATUSES:
            self.assertIn(json.dumps(json.dumps(value)), status)
        for grammar in (count, status, union):
            self.assertIn('value ::= "null"', grammar)
            self.assertNotIn("case_id", grammar)

    def test_real_sampled_tokens_and_terminal_token_are_preserved(self):
        backend = self.backend()
        pre = backend.input_preflight(self.messages(), fixture_only=True)
        result = backend.generate(self.messages(), fixture_only=True)
        self.assertTrue(result["schema_validation"]["accepted"])
        self.assertEqual(result["chosen_answer"], 5)
        self.assertEqual(result["raw_generated_token_ids"], [ord(c) for c in '{"answer":5}'] + [EOS])
        self.assertEqual(result["raw_generated_text"], '{"answer":5}<|im_end|>')
        self.assertEqual(base64.b64decode(result["response_bytes_base64"]), b'{"answer":5}')
        self.assertEqual(result["usage"]["completion_tokens"], len(result["raw_generated_token_ids"]))
        self.assertEqual(result["usage"]["prompt_tokens"], pre["input_tokens"])
        self.assertEqual(result["input_token_ids_sha256"], pre["input_token_ids_sha256"])
        self.assertEqual(backend.engine.reset_calls, 1)
        _, kwargs = backend.engine.generate_calls[0]
        self.assertEqual(kwargs["temp"], 0.0)
        self.assertEqual(kwargs["repeat_penalty"], 1.0)
        self.assertTrue(kwargs["reset"])

    def test_correct_format_never_implies_correct_content(self):
        backend = self.backend(response='{"answer":17}')
        result = backend.generate(self.messages(), fixture_only=True)
        self.assertTrue(result["schema_validation"]["accepted"])
        self.assertEqual(result["chosen_answer"], 17)
        self.assertNotIn("correct", result)

    def test_cross_type_output_rejects_even_if_fake_engine_ignores_grammar(self):
        backend = self.backend(response='{"answer":3}')
        result = backend.generate(self.messages(question=STATUS_QUESTION), fixture_only=True)
        self.assertFalse(result["schema_validation"]["accepted"])
        self.assertIsNone(result["chosen_answer"])

    def test_no_eos_is_appended_or_cutoff_repaired(self):
        backend = self.backend(eos=False)
        result = backend.generate(self.messages(), fixture_only=True)
        self.assertEqual(result["finish_reason"], "iterator_ended_without_eog")
        self.assertFalse(result["schema_validation"]["accepted"])
        self.assertNotIn(EOS, result["raw_generated_token_ids"])
        self.assertEqual(result["response_text"], '{"answer":5}')
        limited = self.backend()
        cutoff = limited.generate(self.messages(), max_new_tokens=4, fixture_only=True)
        self.assertEqual(cutoff["finish_reason"], "length")
        self.assertEqual(len(cutoff["raw_generated_token_ids"]), 4)
        self.assertFalse(cutoff["schema_validation"]["accepted"])

    def test_native_failure_keeps_observed_partial_tokens_without_retry(self):
        backend = self.backend(fail_after=4)
        with self.assertRaises(G.GenerationFailure) as caught:
            backend.generate(self.messages(), fixture_only=True)
        record = caught.exception.generated_record
        self.assertEqual(record["raw_generated_token_ids"], [ord(c) for c in '{"an'])
        self.assertEqual(record["finish_reason"], "native_generation_failed_or_unknown")
        self.assertEqual(len(backend.engine.generate_calls), 1)
        self.assertFalse(record["schema_validation"]["accepted"])

    def test_detokenize_failure_still_preserves_observed_tokens_and_usage(self):
        backend = self.backend()
        actual = backend.engine.detokenize

        def fail_output(ids, prev_tokens=None, special=False):
            if prev_tokens is not None:
                raise RuntimeError("invented post-generation detokenization failure")
            return actual(ids, prev_tokens=prev_tokens, special=special)

        with patch.object(backend.engine, "detokenize", side_effect=fail_output):
            with self.assertRaises(G.GenerationFailure) as caught:
                backend.generate(self.messages(), fixture_only=True)
        record = caught.exception.generated_record
        self.assertEqual(record["raw_generated_token_ids"], [ord(c) for c in '{"answer":5}'] + [EOS])
        self.assertEqual(record["usage"]["completion_tokens"], len(record["raw_generated_token_ids"]))
        self.assertEqual(record["native_finish_reason_before_postprocessing"], "eos")
        self.assertEqual(record["finish_reason"], "postprocessing_failed_after_native_generation")
        self.assertIsNone(record["response_text"])
        self.assertFalse(record["schema_validation"]["accepted"])
        self.assertEqual(len(backend.engine.generate_calls), 1)

    def test_validator_failure_cannot_discard_already_sampled_tokens(self):
        backend = self.backend()
        with patch.object(G.schema, "validate_response", side_effect=ValueError("invented validation failure")):
            with self.assertRaises(G.GenerationFailure) as caught:
                backend.generate(self.messages(), fixture_only=True)
        record = caught.exception.generated_record
        self.assertEqual(record["raw_generated_token_ids"][-1], EOS)
        self.assertEqual(record["usage"]["completion_tokens"], len(record["raw_generated_token_ids"]))
        self.assertEqual(record["postprocessing_failure_type"], "ValueError")

    def test_content_tokenization_must_actually_exclude_control_ids(self):
        backend = self.backend()
        actual = backend.engine.tokenize

        def faulty_tokenizer(data, add_bos=True, special=False):
            if data == b"literal control fixture" and not special:
                return [EOS]
            return actual(data, add_bos=add_bos, special=special)

        with patch.object(backend.engine, "tokenize", side_effect=faulty_tokenizer):
            with self.assertRaisesRegex(ValueError, "untrusted_content_produced_native_control_token"):
                backend.input_preflight(self.messages(content="literal control fixture"), fixture_only=True)
        self.assertEqual(backend.engine.generate_calls, [])

    def test_no_prompt_truncation_or_unbounded_output(self):
        backend = self.backend()
        with self.assertRaisesRegex(ValueError, "input_limit_no_truncation"):
            backend.input_preflight(self.messages(content="x" * 4096), fixture_only=True)
        for budget in (0, -1, 65, None, True):
            with self.assertRaisesRegex(ValueError, "output_cap"):
                backend.generate(self.messages(), max_new_tokens=budget, fixture_only=True)
        self.assertEqual(backend.engine.generate_calls, [])

    def test_frozen_question_schema_cannot_be_replaced(self):
        backend = self.backend()
        other = G.schema.schema_for_question(STATUS_QUESTION)
        with self.assertRaisesRegex(ValueError, "provided_schema"):
            backend.generate(self.messages(), other, fixture_only=True)
        self.assertEqual(backend.engine.generate_calls, [])

    def test_hosted_guard_before_all_actual_model_or_tokenizer_io(self):
        backend = self.backend()
        with patch.object(G.schema, "require_hosted_colab", side_effect=RuntimeError("hosted_first")):
            for action in (
                lambda: G.load_backend("/missing", {}, runtime_manifest={}, constructor_log_path="/missing.log"),
                lambda: G.verify_model_artifacts("/missing", {}),
                lambda: G.Backend(object(), object(), {}),
                lambda: backend.input_preflight([]),
                lambda: backend.generate([]),
            ):
                with self.assertRaisesRegex(RuntimeError, "hosted_first"):
                    action()

    def test_split_artifact_bytes_hash_and_names_are_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for index, name in enumerate(G.MODEL_FILES):
                data = ("invented split " + str(index)).encode()
                (root / name).write_bytes(data)
                entries.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            manifest = {"model_id": G.MODEL_ID, "model_revision": "a" * 40,
                        "quantization": "Q4_K_M", "license": "apache-2.0", "files": entries}
            verified = G.verify_model_artifacts(root, manifest, fixture_only=True)
            self.assertEqual(verified["files_verified"], 2)
            wrong = copy.deepcopy(manifest)
            wrong["files"][0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "checksum"):
                G.verify_model_artifacts(root, wrong, fixture_only=True)
            wrong = copy.deepcopy(manifest)
            wrong["files"][0]["path"] = "../outside.gguf"
            with self.assertRaisesRegex(ValueError, "exact_two"):
                G.verify_model_artifacts(root, wrong, fixture_only=True)

    def test_message_roles_and_content_are_exact_no_tools(self):
        for messages in ([], ["bad"], [{"role": "user", "content": "Q?"}],
                         [{"role": "system", "content": "S", "tools": []},
                          {"role": "user", "content": "Q?"}]):
            with self.assertRaises(ValueError):
                G.chatml_segments(messages)

    def test_installed_native_bytes_match_pinned_wheel_not_modified_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            site = root / "site-packages"
            wheel = root / "invented-runtime.whl"
            contents = {"llama_cpp/__init__.py": b"# invented package, never imported\n",
                        "llama_cpp/lib/libllama.so": b"not executable: invented native bytes"}
            with zipfile.ZipFile(wheel, "w") as archive:
                for name, data in contents.items():
                    archive.writestr(name, data)
                    target = site / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                archive.writestr("invented.dist-info/RECORD", b"wheel record intentionally not installed")
            versions = {"numpy": "fixture-1", "jinja2": "fixture-2",
                        "typing-extensions": "fixture-3", "diskcache": "fixture-4"}
            manifest = {"package_name": "llama-cpp-python", "package_version": G.PACKAGE_VERSION,
                        "wheel_path": str(wheel), "site_packages": str(site),
                        "backend": "cuda", "cuda_release_tag": G.CUDA_RELEASE_TAG,
                        "official_release_metadata_sha256": G.CUDA_RELEASE_METADATA_SHA,
                        "wheel_bytes": wheel.stat().st_size, "wheel_sha256": G.file_sha(wheel),
                        "dependencies": [{"name": name, "version": value} for name, value in versions.items()]}
            distribution = Mock(version=G.PACKAGE_VERSION)
            distribution.locate_file.return_value = site
            actual_path = Path

            def fixture_path(value):
                return root if str(value) == "/content" else actual_path(value)

            with patch.object(G, "Path", side_effect=fixture_path), \
                    patch.object(G, "WHEEL_BYTES", manifest["wheel_bytes"]), \
                    patch.object(G, "WHEEL_SHA256", manifest["wheel_sha256"]), \
                    patch.object(G, "_verify_cuda_dependencies", return_value={"installed_library_files": []}), \
                    patch.object(G.importlib.metadata, "distribution", return_value=distribution), \
                    patch.object(G.importlib.metadata, "version", side_effect=versions.__getitem__):
                verified = G._verify_runtime(manifest)
                self.assertEqual(len(verified["installed_package_files"]), 2)
                (site / "llama_cpp/lib/libllama.so").write_bytes(b"tampered invented bytes")
                with self.assertRaisesRegex(ValueError, "differs_from_pinned_official_wheel"):
                    G._verify_runtime(manifest)

    def test_actual_ctypes_and_mapped_native_library_paths_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for name in ("llama_cpp/llama_cpp.py", "llama_cpp/lib/libllama.so", "llama_cpp/lib/libggml.so"):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                data = ("invented library fixture " + name).encode()
                target.write_bytes(data)
                entries.append({"path": name, "bytes": len(data), "sha256": G.file_sha(target)})
            main = root / "llama_cpp/lib/libllama.so"
            ggml = root / "llama_cpp/lib/libggml.so"
            maps = root / "invented-process-maps.txt"
            maps.write_text(f"000-001 r-xp 0 00:00 1 {main}\n001-002 r-xp 0 00:00 2 {ggml}\n")
            runtime = {"site_packages": str(root), "installed_package_files": entries}
            low = Mock(__file__=str(root / "llama_cpp/llama_cpp.py"), _lib=Mock(_name=str(main)))
            actual_path = Path

            def fixture_path(value):
                return maps if str(value) == "/proc/self/maps" else actual_path(value)

            with patch.object(G, "Path", side_effect=fixture_path):
                verified = G._verify_loaded_native(runtime, low)
                self.assertEqual(verified["ctypes_library_path"], str(main))
                self.assertEqual(len(verified["mapped_official_native_libraries"]), 2)
                maps.write_text(f"000-001 r-xp 0 00:00 1 {main}\n001-002 r-xp 0 00:00 2 /outside/libggml.so\n")
                with self.assertRaisesRegex(RuntimeError, "outside_verified_wheel"):
                    G._verify_loaded_native(runtime, low)
                low._lib._name = "/outside/libllama.so"
                with self.assertRaisesRegex(RuntimeError, "actual_ctypes_library_outside"):
                    G._verify_loaded_native(runtime, low)

    def gpu_proof(self, root, *, used_before=0, used_after=5120):
        content = b"load_tensors: offloaded 29/29 layers to GPU\nload_tensors: CUDA0 model buffer size = 4200 MiB\n"
        prefix = b"unrelated startup metadata\n"
        log = root / "invented-constructor.log"
        log.write_bytes(prefix + content + b"unrelated later records\n")
        before = {"index": 0, "name": "Tesla T4", "uuid": "GPU-invented-fixture",
                  "driver_version": G.EXPECTED_DRIVER, "memory_total_mib": 15360,
                  "memory_used_mib": used_before, "compute_capability": "7.5"}
        after = {**before, "memory_used_mib": used_after}
        return {"proof_passed": True, "compiled_gpu_offload_supported": True,
                "all_layers_offloaded": True, "model_block_count": 28,
                "offloaded_layers": 29, "total_layers": 29,
                "before_load_gpu": before, "after_load_gpu": after,
                "memory_delta_mib": used_after - used_before,
                "gpu_condition": G.gpu_condition(before),
                "constructor_log": {"path": str(log), "start_byte": len(prefix),
                    "end_byte": len(prefix) + len(content), "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(), "stderr_fd_verified": True}}

    def test_gpu_execution_proof_requires_actual_full_layers_and_memory_change(self):
        with tempfile.TemporaryDirectory() as directory:
            proof = self.gpu_proof(Path(directory))
            self.assertTrue(G.verify_gpu_execution_proof(proof))
            for key, value in (("compiled_gpu_offload_supported", False),
                               ("all_layers_offloaded", False), ("offloaded_layers", 28)):
                wrong = copy.deepcopy(proof)
                wrong[key] = value
                with self.assertRaises(ValueError):
                    G.verify_gpu_execution_proof(wrong)
            wrong = copy.deepcopy(proof)
            wrong["after_load_gpu"]["memory_used_mib"] = 1
            wrong["memory_delta_mib"] = 1
            with self.assertRaisesRegex(ValueError, "memory_increase"):
                G.verify_gpu_execution_proof(wrong)
            wrong = copy.deepcopy(proof)
            wrong["after_load_gpu"]["uuid"] = "GPU-another-fixture"
            with self.assertRaisesRegex(ValueError, "condition_changed"):
                G.verify_gpu_execution_proof(wrong)

    def test_dynamic_gpu_memory_and_log_ranges_do_not_change_static_identity(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            left = self.gpu_proof(Path(a), used_before=0, used_after=5120)
            right = self.gpu_proof(Path(b), used_before=200, used_after=5300)
            self.assertTrue(G.verify_gpu_execution_proof(left))
            self.assertTrue(G.verify_gpu_execution_proof(right, expected_condition=left["gpu_condition"]))
            self.assertEqual(left["gpu_condition"], right["gpu_condition"])
            self.assertNotEqual(G.digest(left), G.digest(right))
            self.assertEqual(G.digest(left["gpu_condition"]), G.digest(right["gpu_condition"]))

    def test_gpu_proof_authenticates_original_constructor_byte_range(self):
        with tempfile.TemporaryDirectory() as directory:
            proof = self.gpu_proof(Path(directory))
            log = Path(proof["constructor_log"]["path"])
            log.write_bytes(log.read_bytes().replace(b"29/29", b"28/29"))
            with self.assertRaisesRegex(ValueError, "log_bytes_changed"):
                G.verify_gpu_execution_proof(proof)
            proof = self.gpu_proof(Path(directory))
            proof["constructor_log"]["stderr_fd_verified"] = False
            with self.assertRaisesRegex(ValueError, "stderr_capture"):
                G.verify_gpu_execution_proof(proof)

    def test_gpu_query_is_bounded_and_checks_fixed_observed_hardware(self):
        completed = Mock(returncode=0,
            stdout="0, Tesla T4, GPU-invented-fixture, 580.82.07, 15360, 0, 7.5\n")
        with patch.object(G.schema, "require_hosted_colab"), \
                patch.object(G.subprocess, "run", return_value=completed) as command:
            result = G.query_gpu()
        self.assertEqual(result["memory_used_mib"], 0)
        self.assertEqual(command.call_args.kwargs["timeout"], 10)
        self.assertIn("compute_cap", command.call_args.args[0][1])
        wrong = {**result, "name": "Another GPU"}
        with self.assertRaisesRegex(ValueError, "exact_observed_t4"):
            G.gpu_condition(wrong)

    def test_gpu_backend_policy_keeps_shared_logical_budgets_and_no_cpu_fallback(self):
        policy = G.policy()
        self.assertEqual(policy["n_gpu_layers"], -1)
        self.assertEqual(policy["maximum_input_tokens"], 4096)
        self.assertEqual(policy["maximum_new_tokens"], 64)
        self.assertEqual(policy["n_ctx"], 4352)
        self.assertTrue(policy["dynamic_gpu_proof_is_outside_stable_cache_identity"])

    def cuda_dependency_fixture(self, root):
        artifacts = root / "old-v8"
        artifacts.mkdir()
        target = artifacts / "site-packages"
        cpp = root / "original-cpp-target"
        driver_dir = root / "platform-driver"
        driver_dir.mkdir()
        real_driver = driver_dir / "libcuda.so.580.82.07"
        real_driver.write_bytes(b"invented platform driver; never loaded")
        driver_alias = driver_dir / "libcuda.so.1"
        driver_alias.symlink_to(real_driver)
        driver = {"library_path": str(driver_alias), "resolved_path": str(real_driver),
                  "bytes": real_driver.stat().st_size, "sha256": G.file_sha(real_driver),
                  "library_directory": str(driver_dir), "actual_gpu_query": "invented metadata fixture"}
        packages, wheels, metadata, libraries = [], [], [], []
        for original in G.CUDA_DEPENDENCY_PACKAGES:
            package = dict(original)
            metadata_path = root / (package["name"] + ".json")
            metadata_path.write_text(json.dumps({"invented": package["name"]}))
            package["metadata_sha256"] = G.file_sha(metadata_path)
            wheel = artifacts / (package["name"] + "-invented.whl")
            names = (["libcudart.so.12"] if "runtime" in package["name"] else
                     ["libcublas.so.12", "libcublasLt.so.12"])
            with zipfile.ZipFile(wheel, "w") as archive:
                for name in names:
                    relative = Path(package["library_subdir"]) / name
                    data = ("invented CUDA library; never loaded: " + name).encode()
                    archive.writestr(str(relative), data)
                    installed = target / relative
                    installed.parent.mkdir(parents=True, exist_ok=True)
                    installed.write_bytes(data)
                    libraries.append({"path": str(installed), "bytes": len(data), "sha256": G.file_sha(installed)})
            package["bytes"] = wheel.stat().st_size
            package["sha256"] = G.file_sha(wheel)
            packages.append(package)
            wheels.append({**package, "filename": wheel.name, "url": "https://fixture.invalid/not-downloaded"})
            metadata.append({"path": str(metadata_path), "sha256": package["metadata_sha256"]})
        ld = ":".join([str(target / p["library_subdir"]) for p in packages] + [str(cpp / "llama_cpp/lib"), str(driver_dir)])
        manifest = {"version": G.CUDA_DEPENDENCY_VERSION, "wheels": wheels,
                    "installed_library_files": libraries, "child_ld_library_path": ld,
                    "dependency_site_packages": str(target), "llama_cpp_site_packages": str(cpp),
                    "wheel_directory": str(artifacts), "driver_reference": driver}
        protocol = {"version": G.CUDA_DEPENDENCY_VERSION, "wheels": wheels,
                    "nvidia_pypi_metadata": metadata, "child_ld_library_path": ld,
                    "dependency_target": str(target), "base_environment_modified": False,
                    "old_setup_modified": False}
        manifest_path = root / "native_dependency_manifest.json"
        protocol_path = root / "recovery_protocol.json"
        receipt_path = root / "setup_receipt.json"
        manifest_path.write_text(json.dumps(manifest))
        protocol_path.write_text(json.dumps(protocol))
        receipt = {"stage": "isolated_cuda12_driver_path_ready_no_model_loaded",
                   "protocol_sha256": G.file_sha(protocol_path),
                   "native_dependency_manifest_sha256": G.file_sha(manifest_path),
                   "compiled_gpu_offload_support": True, "model_generations": 0,
                   "old_failed_setup_preserved": True,
                   "runtime_manifest_extension": {"driver_reference": driver}}
        receipt_path.write_text(json.dumps(receipt))
        runtime = {"site_packages": str(cpp), "dependency_site_packages": str(target),
                   "child_ld_library_path": ld, "native_dependency_manifest_path": str(manifest_path),
                   "native_dependency_manifest_sha256": G.file_sha(manifest_path),
                   "cuda_recovery_protocol_path": str(protocol_path),
                   "cuda_recovery_protocol_sha256": G.file_sha(protocol_path),
                   "cuda_recovery_receipt_path": str(receipt_path),
                   "cuda_recovery_receipt_sha256": G.file_sha(receipt_path),
                   "driver_reference": driver, "wheel_directory": str(artifacts)}
        return runtime, tuple(packages)

    def test_cuda_recovery_binds_publisher_wheels_and_installed_library_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime, packages = self.cuda_dependency_fixture(root)
            original_path = Path

            def fixture_path(value):
                return root if str(value) == "/content" else original_path(value)

            with patch.object(G, "CUDA_DEPENDENCY_ROOT", root), \
                    patch.object(G, "CUDA_DEPENDENCY_ARTIFACT_ROOT", root / "old-v8"), \
                    patch.object(G, "CUDA_DEPENDENCY_PACKAGES", packages), \
                    patch.object(G, "Path", side_effect=fixture_path), \
                    patch.dict(G.os.environ, {"LD_LIBRARY_PATH": runtime["child_ld_library_path"]}):
                verified = G._verify_cuda_dependencies(runtime)
                self.assertEqual(len(verified["installed_library_files"]), 3)
                self.assertEqual(verified["child_ld_library_path"], runtime["child_ld_library_path"])
                path = Path(verified["installed_library_files"][0]["path"])
                path.write_bytes(b"tampered invented CUDA library")
                with self.assertRaisesRegex(ValueError, "differs_from_publisher_wheel"):
                    G._verify_cuda_dependencies(runtime)

    def test_cuda_recovery_rejects_unfrozen_ld_path_before_any_native_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime, packages = self.cuda_dependency_fixture(root)
            original_path = Path

            def fixture_path(value):
                return root if str(value) == "/content" else original_path(value)

            with patch.object(G, "CUDA_DEPENDENCY_ROOT", root), \
                    patch.object(G, "CUDA_DEPENDENCY_ARTIFACT_ROOT", root / "old-v8"), \
                    patch.object(G, "CUDA_DEPENDENCY_PACKAGES", packages), \
                    patch.object(G, "Path", side_effect=fixture_path), \
                    patch.dict(G.os.environ, {"LD_LIBRARY_PATH": "/an/unfrozen/path"}):
                with self.assertRaisesRegex(ValueError, "exact_isolated_cuda_ld"):
                    G._verify_cuda_dependencies(runtime)

    def test_actual_mapped_cuda_files_must_use_verified_isolated_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cpp = root / "cpp"
            libraries = []
            for name in ("libllama.so", "libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12"):
                path = cpp / "llama_cpp/lib" / name if name == "libllama.so" else root / "cuda" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(("invented native fixture " + name).encode())
                libraries.append({"path": str(path), "bytes": path.stat().st_size, "sha256": G.file_sha(path)})
            runtime = {"site_packages": str(cpp),
                "installed_package_files": [{**libraries[0], "path": "llama_cpp/lib/libllama.so"}],
                "native_dependency_identity": {"installed_library_files": libraries[1:]}}
            driver = root / "platform" / "libcuda.so.580.82.07"
            driver.parent.mkdir()
            driver.write_bytes(b"invented platform driver")
            alias = driver.parent / "libcuda.so.1"
            alias.symlink_to(driver)
            runtime["native_dependency_identity"]["driver_reference"] = {
                "library_path": str(alias), "resolved_path": str(driver),
                "bytes": driver.stat().st_size, "sha256": G.file_sha(driver),
                "library_directory": str(driver.parent)}
            low = Mock(__file__=str(cpp / "llama_cpp/llama_cpp.py"), _lib=Mock(_name=libraries[0]["path"]))
            maps = root / "maps.txt"
            maps.write_text("".join(f"000-001 r-xp 0 00:00 1 {entry['path']}\n" for entry in libraries) +
                            f"000-001 r-xp 0 00:00 1 {driver}\n")
            original_path = Path

            def fixture_path(value):
                return maps if str(value) == "/proc/self/maps" else original_path(value)

            with patch.object(G, "Path", side_effect=fixture_path):
                verified = G._verify_loaded_native(runtime, low, require_cuda=True)
                self.assertEqual(len(verified["mapped_official_native_libraries"]), 4)
                maps.write_text(maps.read_text().replace(libraries[1]["path"], "/outside/libcudart.so.12"))
                with self.assertRaisesRegex(RuntimeError, "outside_verified_wheel"):
                    G._verify_loaded_native(runtime, low, require_cuda=True)


if __name__ == "__main__":
    unittest.main()
