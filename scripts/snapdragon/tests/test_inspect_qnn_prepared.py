"""CPU metadata/interface regressions for an explicitly requested static prefill window."""
import copy
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


def fixture(window):
    T = onnx.TensorProto
    dynamic = ["batch_size", "sequence_length"]
    embedding = "embedding_activation_q"
    hidden = "final_hidden_activation_q"
    components = {
        "embeddings": {"inputs": [value("input_ids", T.INT64, dynamic)],
                       "outputs": [value(embedding, T.UINT16, dynamic + [896])]},
        "lm_head": {"inputs": [value(hidden, T.UINT16, dynamic + [896])],
                    "outputs": [value("logits", T.FLOAT, dynamic + [151936])]}}
    for name, length in (("context", window), ("iterator", 1)):
        inputs = [value(embedding, T.UINT16, [1, length, 896]),
                  value("past_seq_len", T.INT32, [1, 1]), value("total_seq_len", T.INT32, [])]
        outputs = [value(hidden, T.UINT16, [1, length, 896])]
        for layer in range(2):
            for kind in ("key", "value"):
                inputs.append(value(f"past_key_values.{layer}.{kind}", T.FLOAT, [1, 2, "past_sequence_length", 64]))
                outputs.append(value(f"present.{layer}.{kind}", T.FLOAT, [1, 2, "past_sequence_length", 64]))
        components[name] = {"inputs": inputs, "outputs": outputs}
    genai = {"model": {"vocab_size": 151936, "decoder": {
        "num_hidden_layers": 2, "num_key_value_heads": 2, "head_size": 64, "hidden_size": 896,
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


if __name__ == "__main__":
    unittest.main()
