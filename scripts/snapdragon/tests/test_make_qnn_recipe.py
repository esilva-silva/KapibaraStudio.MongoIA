import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('make_qnn_recipe', SCRIPTS / 'make_qnn_recipe.py')
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)


class RecipeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=SCRIPTS)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        entries = []
        for name, contents in [('config.json', '{"hidden_size":896,"num_attention_heads":14}'), ('tokenizer.json', '{}'), ('model.safetensors', 'weights')]:
            path = self.source / name
            path.write_text(contents, encoding='utf-8')
            entries.append({'path': name, 'size': path.stat().st_size, 'sha256': recipe.sha256(path)})
        (self.source / 'snapshot_manifest.json').write_text(json.dumps({'revision': 'a' * 40, 'files': entries}), encoding='utf-8')
        self.cal = self.root / 'samples.jsonl'
        self.cal.write_text('{"prompt":"db.test.find()"}\n{"prompt_text":"db.x.countDocuments()"}\n', encoding='utf-8')

    def update_source_config(self, metadata):
        path = self.source / 'config.json'
        path.write_text(json.dumps(metadata), encoding='utf-8')
        manifest_path = self.source / 'snapshot_manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        entry = next(x for x in manifest['files'] if x['path'] == 'config.json')
        entry.update(size=path.stat().st_size, sha256=recipe.sha256(path))
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def test_prepare_has_local_calibration_cpu_and_no_compiler(self):
        config = recipe.generate(self.source, self.cal, self.root, self.root / 'out', 'prepare')
        self.assertNotIn('cb', config['passes'])
        self.assertEqual(config['target'], 'qnn_system')
        self.assertEqual(config['systems']['qnn_system']['accelerators'][0]['execution_providers'], ['QNNExecutionProvider'])
        self.assertNotIn('task', config['data_configs'][0])
        self.assertEqual(config['data_configs'][0]['pre_process_data_config']['type'], 'text_generation_huggingface_pre_process')
        self.assertEqual(config['data_configs'][0]['pre_process_data_config']['max_samples'], 128)
        self.assertEqual(config['data_configs'][1]['pre_process_data_config']['max_samples'], 256)
        self.assertEqual(config['passes']['sq']['calibration_providers'], ['CPUExecutionProvider'])
        self.assertEqual(config['passes']['g']['bits'], 4)
        self.assertEqual(config['passes']['st']['context_length'], 64)
        self.assertEqual(config['data_configs'][0]['pre_process_data_config']['model_name'], str(self.source))
        self.assertEqual(json.loads((self.root / 'out/calibration.jsonl').read_text(encoding='utf-8').splitlines()[0]), {'text': 'db.test.find()'})

    def test_kv_head_axis_is_fixed_without_freezing_cache_sequence(self):
        config = recipe.generate(self.source, self.cal, self.root, self.root / 'out', 'prepare')
        fix = config['passes']['kv_shape']
        self.assertEqual(fix['dim_param'], ['kv_cache_dim'])
        self.assertEqual(fix['dim_value'], [64])
        self.assertLess(list(config['passes']).index('kv_shape'), list(config['passes']).index('sq'))

    def test_explicit_head_dim_overrides_hidden_ratio(self):
        self.update_source_config({'hidden_size': 1536, 'num_attention_heads': 12, 'head_dim': 128})
        config = recipe.generate(self.source, self.cal, self.root, self.root / 'out', 'prepare')
        self.assertEqual(config['passes']['kv_shape']['dim_value'], [128])

    def test_missing_or_invalid_head_metadata_is_not_silently_guessed(self):
        for metadata in ({}, {'head_dim': 0}, {'head_dim': True}, {'hidden_size': 10, 'num_attention_heads': 3}):
            with self.subTest(metadata=metadata):
                self.update_source_config(metadata)
                with self.assertRaisesRegex(ValueError, 'KV head dimension'):
                    recipe.generate(self.source, self.cal, self.root, self.root / 'out', 'prepare')

    def test_prepare_python_and_olive_qnn_selector(self):
        import zipfile
        config = recipe.generate(self.source, self.cal, self.root / 'fallback', self.root / 'out', 'prepare',
                                 prepare_python=self.root / 'qnn-env')
        self.assertEqual(config['systems']['qnn_system']['python_environment_path'], str((self.root / 'qnn-env').resolve()))
        wheel_dir = SCRIPTS.parents[1] / 'tmp/olive-source'
        wheels = list(wheel_dir.glob('olive_ai-0.13.0-*.whl'))
        if not wheels:
            self.skipTest('Olive 0.13 wheel unavailable')
        with zipfile.ZipFile(wheels[0]) as wheel:
            source = wheel.read('olive/passes/onnx/quantization.py').decode()
        self.assertIn('self.accelerator_spec.execution_provider == ExecutionProvider.QNNExecutionProvider', source)
        self.assertIn('from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config as get_qdq_config', source)

    def composite(self, origin='/build/output'):
        artifact = self.root / 'artifact'
        artifact.mkdir(exist_ok=True)
        for name in ('embeddings.onnx', 'context.onnx', 'iterator.onnx', 'lm_head.onnx', 'genai_config.json'):
            (artifact / name).write_text('fixture', encoding='utf-8')
        names = ['embeddings', 'context', 'iterator', 'lm_head']
        return artifact, {'type': 'CompositeModel', 'config': {'model_path': origin,
            'model_component_names': names,
            'model_components': [{'type': 'ONNXModel', 'config': {'model_path': origin,
                'onnx_file_name': name + '.onnx'}} for name in names],
            'model_attributes': {'additional_files': [origin + '/genai_config.json'],
                'llm_pipeline': {'embeddings': 'embeddings', 'context': ['context'],
                    'iterator': ['iterator'], 'lm_head': 'lm_head'}}}}

    def test_compile_rebases_composite_and_disables_fallback(self):
        artifact, data = self.composite()
        model = self.root / 'model.json'
        model.write_text(json.dumps(data), encoding='utf-8')
        config = recipe.generate(self.source, self.cal, self.root, self.root / 'out', 'compile', model,
                                 artifact_root=artifact, original_artifact_root='/build/output')
        self.assertEqual(set(config['passes']), {'cb', 'cp'})
        self.assertTrue(config['passes']['cb']['disable_cpu_fallback'])
        self.assertNotIn('soc_model', config['passes']['cb']['provider_options'])
        actual = config['input_model']['config']
        self.assertEqual(actual['model_components'][0]['config']['model_path'], str(artifact.resolve()))
        self.assertEqual(actual['model_attributes']['llm_pipeline'], data['config']['model_attributes']['llm_pipeline'])

    def test_windows_origin_rebases(self):
        artifact, data = self.composite('C:/build/output')
        rebased = recipe.rebase_composite(data, artifact, 'C:/build/output')
        self.assertEqual(rebased['config']['model_path'], str(artifact.resolve()))

    def test_rejects_outside_or_missing_resources(self):
        artifact, data = self.composite()
        data['config']['model_components'][0]['config']['model_path'] = '/elsewhere'
        with self.assertRaises(ValueError):
            recipe.rebase_composite(data, artifact, '/build/output')
        data['config']['model_components'][0]['config']['model_path'] = '/build/output'
        data['config']['model_components'][0]['config']['onnx_file_name'] = 'missing.onnx'
        with self.assertRaises(ValueError):
            recipe.rebase_composite(data, artifact, '/build/output')

    def test_rejects_parent_traversal_and_single_model(self):
        artifact, data = self.composite()
        data['config']['model_components'][0]['config']['onnx_file_name'] = '../escape.onnx'
        with self.assertRaises(ValueError):
            recipe.rebase_composite(data, artifact, '/build/output')
        with self.assertRaises(ValueError):
            recipe.rebase_composite({'type': 'ONNXModel', 'config': {}}, artifact)

    def test_tampered_weights_rejected(self):
        (self.source / 'model.safetensors').write_text('tampered', encoding='utf-8')
        with self.assertRaises(ValueError):
            recipe.generate(self.source, self.cal, self.root, self.root / 'out', 'prepare')


if __name__ == '__main__':
    unittest.main()
