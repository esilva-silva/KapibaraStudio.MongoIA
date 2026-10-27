import importlib.util
import unittest
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

spec = importlib.util.spec_from_file_location('decompose', Path(__file__).resolve().parents[1] / 'decompose_rmsnorm.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def make_model(skip=False):
    rng = np.random.default_rng(123)
    scale = numpy_helper.from_array(rng.uniform(.5, 1.5, 896).astype(np.float16), 'scale')
    node = helper.make_node('SkipSimplifiedLayerNormalization' if skip else 'SimplifiedLayerNormalization',
        ['x', 'scale'], ['y'], domain='com.microsoft' if skip else '', axis=-1, epsilon=1e-6, stash_type=1)
    graph = helper.make_graph([node], 'test', [helper.make_tensor_value_info('x', TensorProto.FLOAT16, [2, 896])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT16, [2, 896])], [scale])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid('', 21), helper.make_opsetid('com.microsoft', 1)], ir_version=10)


class TestDecompose(unittest.TestCase):
    def test_cpu_matches_fused_fp16(self):
        model = make_model()
        original = model.SerializeToString()
        self.assertEqual(mod.transform_model(model), 1)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        rng = np.random.default_rng(42)
        inputs = {'x': rng.normal(size=(2, 896)).astype(np.float16)}
        expected = ort.InferenceSession(original, sess_options=options, providers=['CPUExecutionProvider']).run(None, inputs)[0]
        actual = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=['CPUExecutionProvider']).run(None, inputs)[0]
        np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), rtol=0.002, atol=0.002)
        print('max_abs_error:', np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32))))

    def test_skip_cpu_matches_output_and_residual(self):
        rng = np.random.default_rng(123)
        scale = numpy_helper.from_array(rng.uniform(.5, 1.5, 896).astype(np.float16), 'scale')
        node = helper.make_node('SkipSimplifiedLayerNormalization', ['x', 'skip', 'scale'],
            ['y', '', '', 'residual'], domain='com.microsoft', epsilon=1e-6)
        graph = helper.make_graph([node], 'test',
            [helper.make_tensor_value_info(k, TensorProto.FLOAT16, [2, 3, 896]) for k in ('x', 'skip')],
            [helper.make_tensor_value_info(k, TensorProto.FLOAT16, [2, 3, 896]) for k in ('y', 'residual')], [scale])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 21), helper.make_opsetid('com.microsoft', 1)], ir_version=10)
        original = model.SerializeToString()
        self.assertEqual(mod.transform_model(model), 1)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        inputs = {k: rng.normal(size=(2, 3, 896)).astype(np.float16) for k in ('x', 'skip')}
        expected = ort.InferenceSession(original, sess_options=options, providers=['CPUExecutionProvider']).run(None, inputs)
        actual = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=['CPUExecutionProvider']).run(None, inputs)
        for a, e in zip(actual, expected):
            np.testing.assert_allclose(a.astype(np.float32), e.astype(np.float32), rtol=0.002, atol=0.002)
        print('skip_max_abs_errors:', [float(np.max(np.abs(a.astype(np.float32)-e.astype(np.float32)))) for a,e in zip(actual,expected)])

    def test_skip_y_only_cpu_matches(self):
        rng = np.random.default_rng(789)
        scale = numpy_helper.from_array(rng.uniform(.5, 1.5, 896).astype(np.float16), 'scale')
        node = helper.make_node('SkipSimplifiedLayerNormalization', ['x', 'skip', 'scale'], ['y'],
            domain='com.microsoft', epsilon=1e-6)
        graph = helper.make_graph([node], 'skip_y',
            [helper.make_tensor_value_info(k, TensorProto.FLOAT16, [2, 3, 896]) for k in ('x', 'skip')],
            [helper.make_tensor_value_info('y', TensorProto.FLOAT16, [2, 3, 896])], [scale])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 21), helper.make_opsetid('com.microsoft', 1)], ir_version=10)
        original = model.SerializeToString()
        self.assertEqual(mod.transform_model(model), 1)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        inputs = {k: rng.normal(size=(2, 3, 896)).astype(np.float16) for k in ('x', 'skip')}
        expected = ort.InferenceSession(original, sess_options=options, providers=['CPUExecutionProvider']).run(None, inputs)[0]
        actual = ort.InferenceSession(model.SerializeToString(), sess_options=options, providers=['CPUExecutionProvider']).run(None, inputs)[0]
        np.testing.assert_allclose(actual.astype(np.float32), expected.astype(np.float32), rtol=0.002, atol=0.002)
        print('skip_y_only_max_abs_error:', float(np.max(np.abs(actual.astype(np.float32)-expected.astype(np.float32)))))

    def test_skip_unknown_outputs_rejected(self):
        model = make_model(skip=True)
        with self.assertRaises(ValueError):
            mod.transform_model(model)

    def test_package_is_independent_and_manifest_hashes_match(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.mkdir()
            onnx.save_model(make_model(), str(source / 'model.onnx'))
            (source / 'model.onnx.data').write_bytes(b'unchanged external weights')
            original_genai = {'model': {'decoder': {'filename': 'model.onnx', 'pipeline': [{'filename': 'part.onnx'}]}}}
            (source / 'genai_config.json').write_text(json.dumps(original_genai), encoding='utf-8')
            (source / 'kapicoder_manifest.json').write_text(json.dumps({'files': [{'path': 'model.onnx', 'sha256': 'old', 'bytes': 0}]}), encoding='utf-8')
            (source / 'export_manifest.json').write_text(json.dumps({'source': {'revision': 'abc', 'model_sha256': 'original'}}), encoding='utf-8')
            parent_hash = mod.sha256(source / 'export_manifest.json')
            before = mod.sha256(source / 'model.onnx')
            manifest = mod.build(source, root / 'output', 'model.onnx')
            self.assertEqual(mod.sha256(source / 'model.onnx'), before)
            self.assertEqual(mod.sha256(source / 'model.onnx.data'), mod.sha256(root / 'output/model.onnx.data'))
            self.assertEqual(manifest['source_model_sha256'], before)
            self.assertEqual(manifest['parent_export_manifest_sha256'], parent_hash)
            self.assertEqual(manifest['parent_source_provenance']['revision'], 'abc')
            kapi = json.loads((root / 'output/kapicoder_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(kapi['files'][0]['sha256'], mod.sha256(root / 'output/model.onnx'))
            self.assertEqual(kapi['graph_decomposition']['parent_export_manifest_sha256'], parent_hash)
            for entry in manifest['files']:
                self.assertEqual(entry['sha256'], mod.sha256(root / 'output' / entry['path']))
            output_config = json.loads((root / 'output/genai_config.json').read_text())
            self.assertEqual(output_config['model']['decoder']['session_options']['graph_optimization_level'], 'ORT_DISABLE_ALL')
            self.assertEqual(output_config['model']['decoder']['pipeline'][0]['session_options']['graph_optimization_level'], 'ORT_DISABLE_ALL')
            self.assertEqual(json.loads((source / 'genai_config.json').read_text()), original_genai)
            self.assertEqual(manifest['graph_optimization_level'], 'ORT_DISABLE_ALL')
            self.assertTrue((root / 'output/RMSNORM_TRANSFORM.md').exists())

    def test_optional_output_is_rejected(self):
        model = make_model()
        model.graph.node[0].output.append('inv')
        with self.assertRaises(ValueError):
            mod.transform_model(model)


if __name__ == '__main__':
    unittest.main()
