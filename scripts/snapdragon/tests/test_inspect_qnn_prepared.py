"""CPU metadata/interface regressions for an explicitly requested static prefill window."""
import copy
import json
import tempfile
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import onnx


path = Path(__file__).resolve().parents[1] / "inspect_qnn_prepared.py"
spec = importlib.util.spec_from_file_location("inspect_qnn_prepared", path)
inspector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inspector)


def value(name, dtype, shape):
    # Exercise the real ONNX value-info representation, rather than copying inspector descriptors.
    return onnx.helper.make_tensor_value_info(name, dtype, shape)


def fixture(window, hidden_size=896, layers=2, head_size=64):
    T = onnx.TensorProto
    dynamic = ["batch_size", "sequence_length"]
    embedding = "embedding_activation_q"
    hidden = "final_hidden_activation_q"
    components = {
        "embeddings": {"inputs": [value("input_ids", T.INT64, dynamic)],
                       "outputs": [value(embedding, T.UINT16, dynamic + [hidden_size])]},
        "lm_head": {"inputs": [value(hidden, T.UINT16, dynamic + [hidden_size])],
                    "outputs": [value("logits", T.FLOAT, dynamic + [151936])]}}
    for name, length in (("context", window), ("iterator", 1)):
        inputs = [value(embedding, T.UINT16, [1, length, hidden_size]),
                  value("past_seq_len", T.INT32, [1, 1]), value("total_seq_len", T.INT32, [])]
        outputs = [value(hidden, T.UINT16, [1, length, hidden_size])]
        for layer in range(layers):
            for kind in ("key", "value"):
                inputs.append(value(f"past_key_values.{layer}.{kind}", T.FLOAT, [1, 2, "past_sequence_length", head_size]))
                outputs.append(value(f"present.{layer}.{kind}", T.FLOAT, [1, 2, "past_sequence_length", head_size]))
        components[name] = {"inputs": inputs, "outputs": outputs}
    genai = {"model": {"vocab_size": 151936, "decoder": {
        "num_hidden_layers": layers, "num_key_value_heads": 2, "head_size": head_size, "hidden_size": hidden_size,
        "sliding_window": {"window_size": window},
        "inputs": {"past_sequence_length": "past_seq_len", "total_sequence_length": "total_seq_len",
                   "past_key_names": "past_key_values.%d.key", "past_value_names": "past_key_values.%d.value"},
        "outputs": {"present_key_names": "present.%d.key", "present_value_names": "present.%d.value"}}}}
    return components, genai


def describe_components(components):
    return {name: {side: [inspector.describe(v) for v in rows] for side, rows in component.items()}
            for name, component in components.items()}


class ExpectedWindowTests(unittest.TestCase):
    def test_accepts_both_windows_only_when_requested_metadata_and_graph_agree(self):
        for window in (32, 64):
            with self.subTest(window=window):
                components, genai = fixture(window)
                self.assertEqual(inspector.validate_expected_window(genai, window), window)
                self.assertEqual(inspector.validate_interfaces(describe_components(components), genai, window), [])

    def test_default_remains64_and_does_not_auto_accept32_artifact(self):
        components, genai = fixture(64)
        self.assertEqual(inspector.validate_interfaces(describe_components(components), genai), [])
        _, changed = fixture(32)
        with self.assertRaisesRegex(ValueError, "window mismatch"):
            inspector.validate_expected_window(changed)

    def test_metadata_of_another_supported_window_is_not_automatically_accepted(self):
        for requested, artifact in ((32, 64), (64, 32)):
            with self.subTest(requested=requested, artifact=artifact):
                components, genai = fixture(artifact)
                with self.assertRaisesRegex(ValueError, "window mismatch"):
                    inspector.validate_interfaces(describe_components(components), genai, requested)

    def test_rejects_unsupported_or_non_integer_window(self):
        _, genai = fixture(64)
        for requested in (0, 16, 128, True, 32.0):
            with self.subTest(requested=requested), self.assertRaises(ValueError):
                inspector.validate_expected_window(genai, requested)
        for actual in (True, 32.0, "32", None):
            changed = copy.deepcopy(genai)
            changed["model"]["decoder"]["sliding_window"]["window_size"] = actual
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                inspector.validate_expected_window(changed, 32)

    def test_stale_context_input_shape_is_rejected_even_if_metadata_was_changed(self):
        components, genai = fixture(64)
        genai["model"]["decoder"]["sliding_window"]["window_size"] = 32
        failures = inspector.validate_interfaces(describe_components(components), genai, 32)
        self.assertIn("context embeddings boundary mismatch", failures)
        self.assertIn("context head boundary mismatch", failures)

    def test_stale_context_output_shape_cannot_hide_behind_a_correct_input(self):
        components, genai = fixture(32)
        components["context"]["outputs"][0].type.tensor_type.shape.dim[1].dim_value = 64
        failures = inspector.validate_interfaces(describe_components(components), genai, 32)
        self.assertNotIn("context embeddings boundary mismatch", failures)
        self.assertIn("context head boundary mismatch", failures)

    def test_iterator_stays_single_token_with32_window(self):
        components, genai = fixture(32)
        components["iterator"]["inputs"][0].type.tensor_type.shape.dim[1].dim_value = 32
        failures = inspector.validate_interfaces(describe_components(components), genai, 32)
        self.assertIn("iterator embeddings boundary mismatch", failures)

    def test_window32_does_not_change_kv_head_dimension64(self):
        components, genai = fixture(32)
        components["context"]["inputs"][3].type.tensor_type.shape.dim[3].dim_value = 32
        failures = inspector.validate_interfaces(describe_components(components), genai, 32)
        self.assertTrue(any("invalid KV" in failure for failure in failures))

    def test_cli_rejects_unsupported_window_before_reading_artifacts(self):
        args = ["--artifact-dir", "missing", "--source", "missing", "--samples", "missing", "--out", "unused-report.json", "--expected-window", "128"]
        with patch.object(inspector, "inspect") as inspect:
            with self.assertRaises(SystemExit) as error:
                inspector.main(args)
            self.assertEqual(error.exception.code, 2)
            inspect.assert_not_called()


class SmallHeadSizeTests(unittest.TestCase):
    def test_small28_layers_and1536_hidden_accept128_heads_for_both_windows(self):
        for window in (32, 64):
            with self.subTest(window=window):
                components, genai = fixture(window, hidden_size=1536, layers=28, head_size=128)
                self.assertEqual(len(components["context"]["inputs"]), 59)
                self.assertEqual(len(components["context"]["outputs"]), 57)
                self.assertEqual(inspector.validate_interfaces(describe_components(components), genai, window, 128), [])

    def test_nano_default_does_not_auto_accept_small_head_metadata(self):
        components, genai = fixture(64, hidden_size=1536, layers=28, head_size=128)
        with self.assertRaisesRegex(ValueError, "head size mismatch"):
            inspector.validate_interfaces(describe_components(components), genai)

    def test_wrong_head_metadata_is_rejected_even_when_graph_uses128(self):
        components, genai = fixture(32, hidden_size=1536, layers=28, head_size=128)
        genai["model"]["decoder"]["head_size"] = 64
        with self.assertRaisesRegex(ValueError, "head size mismatch"):
            inspector.validate_interfaces(describe_components(components), genai, 32, 128)

    def test_last_layer_wrong_kv_shape_is_rejected_with_valid128_metadata(self):
        components, genai = fixture(32, hidden_size=1536, layers=28, head_size=128)
        target = next(v for v in components["iterator"]["outputs"] if v.name == "present.27.value")
        target.type.tensor_type.shape.dim[3].dim_value = 64
        failures = inspector.validate_interfaces(describe_components(components), genai, 32, 128)
        self.assertTrue(any("iterator invalid KV: present.27.value" in failure for failure in failures))

    def test_missing_final_small_layer_is_rejected(self):
        components, genai = fixture(32, hidden_size=1536, layers=28, head_size=128)
        components["context"]["inputs"] = [v for v in components["context"]["inputs"] if v.name != "past_key_values.27.key"]
        failures = inspector.validate_interfaces(describe_components(components), genai, 32, 128)
        self.assertIn("context incomplete or unexpected KV names", failures)

    def test_hidden_dimension_is_independent_of_head_dimension(self):
        components, genai = fixture(32, hidden_size=1536, layers=28, head_size=128)
        components["context"]["inputs"][0].type.tensor_type.shape.dim[2].dim_value = 896
        failures = inspector.validate_interfaces(describe_components(components), genai, 32, 128)
        self.assertIn("context embeddings boundary mismatch", failures)

    def test_invalid_head_sizes_fail_before_inspecting_interfaces(self):
        _, genai = fixture(32, hidden_size=1536, layers=28, head_size=128)
        for expected in (0, 32, 256, True, 128.0):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                inspector.validate_expected_head_size(genai, expected)

    def test_cli_forwards_small_head_size_without_changing_default_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = ["--artifact-dir", str(root / "artifact"), "--source", str(root / "source"),
                    "--samples", str(root / "samples"), "--out", str(root / "report.json"), "--expected-head-size", "128"]
            result = {"passed": True, "transfer_ready": True, "components": {}, "failures": []}
            with patch.object(inspector, "inspect", return_value=result) as call:
                self.assertEqual(inspector.main(args), 0)
                call.assert_called_once_with(root / "artifact", root / "source", root / "samples", 64, False, 128)


class TokenizerByteParityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.artifact = self.root / "source", self.root / "artifact"
        self.source.mkdir(); self.artifact.mkdir()
        for name in ("tokenizer.json", "tokenizer_config.json"):
            for directory in (self.source, self.artifact):
                (directory / name).write_bytes(b'{"version":"1.0"}\n')

    def test_default_does_not_require_byte_identity_or_new_auxiliary_files(self):
        (self.artifact / "tokenizer.json").write_bytes(b'{"version":"2.0"}\n')
        (self.artifact / "tokenizer_config.json").unlink()
        result = inspector.tokenizer_byte_parity(self.artifact, self.source)
        self.assertTrue(result["passed"])
        self.assertFalse(result["required"])
        self.assertFalse(result["checked"])

    def test_strict_compares_required_and_all_auxiliaries_that_exist_in_source(self):
        auxiliaries = ("vocab.json", "merges.txt", "generation_config.json", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja")
        for name in auxiliaries:
            for directory in (self.source, self.artifact):
                (directory / name).write_bytes(("original-" + name).encode())
        result = inspector.tokenizer_byte_parity(self.artifact, self.source, required=True)
        self.assertTrue(result["passed"])
        self.assertTrue(result["required"])
        self.assertTrue(result["checked"])
        self.assertEqual(len(result["files"]), 8)
        self.assertTrue(all(entry["source_sha256"] == entry["artifact_sha256"] for entry in result["files"]))

    def test_single_changed_byte_fails_even_when_sizes_stay_identical(self):
        for name in ("tokenizer.json", "tokenizer_config.json"):
            with self.subTest(name=name):
                original = (self.source / name).read_bytes()
                (self.artifact / name).write_bytes(original.replace(b"1.0", b"2.0"))
                result = inspector.tokenizer_byte_parity(self.artifact, self.source, required=True)
                entry = next(entry for entry in result["files"] if entry["path"] == name)
                self.assertFalse(result["passed"])
                self.assertFalse(entry["matches"])
                self.assertEqual(entry["source_bytes"], entry["artifact_bytes"])
                self.assertNotEqual(entry["source_sha256"], entry["artifact_sha256"])
                (self.artifact / name).write_bytes(original)

    def test_missing_auxiliary_present_in_source_fails(self):
        for name in ("vocab.json", "merges.txt", "generation_config.json", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja"):
            with self.subTest(name=name):
                (self.source / name).write_bytes(b"original auxiliary")
                result = inspector.tokenizer_byte_parity(self.artifact, self.source, required=True)
                entry = next(entry for entry in result["files"] if entry["path"] == name)
                self.assertFalse(result["passed"])
                self.assertIsNone(entry["artifact_sha256"])
                self.assertIn("artifact_error", entry)
                (self.source / name).unlink()

    def test_generation_config_one_changed_byte_fails_without_modifying_artifact(self):
        original = b'{"eos_token_id":151643}\n'
        changed = b'{"eos_token_id":151644}\n'
        (self.source / "generation_config.json").write_bytes(original)
        artifact_file = self.artifact / "generation_config.json"
        artifact_file.write_bytes(changed)
        result = inspector.tokenizer_byte_parity(self.artifact, self.source, required=True)
        entry = next(entry for entry in result["files"] if entry["path"] == "generation_config.json")
        self.assertFalse(result["passed"])
        self.assertFalse(entry["matches"])
        self.assertEqual(entry["source_bytes"], entry["artifact_bytes"])
        self.assertNotEqual(entry["source_sha256"], entry["artifact_sha256"])
        self.assertEqual(artifact_file.read_bytes(), changed)

    def test_generation_config_is_optional_when_absent_from_source(self):
        (self.artifact / "generation_config.json").write_bytes(b'{"derived":true}\n')
        result = inspector.tokenizer_byte_parity(self.artifact, self.source, required=True)
        self.assertTrue(result["passed"])
        self.assertIn("generation_config.json", result["optional_not_present_in_source"])
        self.assertNotIn("generation_config.json", [entry["path"] for entry in result["files"]])

    def test_each_mandatory_file_is_required_in_both_source_and_artifact(self):
        for directory in (self.source, self.artifact):
            for name in ("tokenizer.json", "tokenizer_config.json"):
                with self.subTest(directory=directory.name, name=name):
                    original = (directory / name).read_bytes()
                    (directory / name).unlink()
                    self.assertFalse(inspector.tokenizer_byte_parity(self.artifact, self.source, required=True)["passed"])
                    (directory / name).write_bytes(original)

    def test_derived_model_configuration_is_not_compared_or_modified(self):
        (self.source / "config.json").write_text('{"max_position_embeddings":32768}', encoding="utf-8")
        derived = '{"max_position_embeddings":2048,"quantization":"derived"}'
        (self.artifact / "config.json").write_text(derived, encoding="utf-8")
        result = inspector.tokenizer_byte_parity(self.artifact, self.source, required=True)
        self.assertTrue(result["passed"])
        self.assertNotIn("config.json", [entry["path"] for entry in result["files"]])
        self.assertEqual((self.artifact / "config.json").read_text(encoding="utf-8"), derived)

    def test_cli_keeps_old64_default_and_forwards_strict32_flag(self):
        base = ["--artifact-dir", str(self.artifact), "--source", str(self.source), "--samples", str(self.root / "samples"), "--out", str(self.root / "report.json")]
        result = {"passed": True, "transfer_ready": True, "components": {}, "failures": []}
        for flags, window, strict in (([], 64, False), (["--expected-window", "32", "--require-original-tokenizer"], 32, True)):
            with self.subTest(flags=flags), patch.object(inspector, "inspect", return_value=result) as call:
                self.assertEqual(inspector.main(base + flags), 0)
                call.assert_called_once_with(self.artifact, self.source, self.root / "samples", window, strict, 64)
                self.assertTrue(json.loads((self.root / "report.json").read_text())["passed"])


if __name__ == "__main__":
    unittest.main()
