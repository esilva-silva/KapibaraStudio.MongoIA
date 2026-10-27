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
        for name, contents in [('config.json', '{}'), ('tokenizer.json', '{}'), ('model.safetensors', 'weights')]:
            path = self.source / name
            path.write_text(contents, encoding='utf-8')
            entries.append({'path': name, 'size': path.stat().st_size, 'sha256': recipe.sha256(path)})
        (self.source / 'snapshot_manifest.json').write_text(json.dumps({'revision': 'a' * 40, 'files': entries}), encoding='utf-8')
        self.cal = self.root / 'samples.jsonl'
        self.cal.write_text('{"prompt":"db.test.find()"}\n{"prompt_text":"db.x.countDocuments()"}\n', encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def test_prepare_has_local_calibration_cpu_and_no_compiler(self):
        config = recipe.generate(self.source, self.cal, self.root, self.root / 'out', 'prepare')
        self.assertNotIn('cb', config['passes'])
        self.assertEqual(config['passes']['sq']['calibration_providers'], ['CPUExecutionProvider'])
        self.assertEqual(config['passes']['g']['bits'], 4)
        self.assertEqual(config['passes']['st']['context_length'], 64)
        self.assertEqual(config['data_configs'][0]['pre_process_data_config']['model_name'], str(self.source))
        self.assertEqual(json.loads((self.root / 'out/calibration.jsonl').read_text(encoding='utf-8').splitlines()[0]), {'text': 'db.test.find()'})

    def test_compile_requires_model_and_does_not_guess_soc(self):
        with self.assertRaises(ValueError):
            recipe.generate(self.source, self.cal, self.root, self.root / 'bad', 'compile')
        model = self.root / 'model.json'
        model.write_text(json.dumps({'type': 'ONNXModel', 'config': {'model_path': 'model.onnx'}}), encoding='utf-8')
        config = recipe.generate(self.source, self.cal, self.root, self.root / 'out', 'compile', model)
        self.assertEqual(set(config['passes']), {'cb', 'cp'})
        self.assertNotIn('soc_model', config['passes']['cb']['provider_options'])

    def test_tampered_weights_rejected(self):
        (self.source / 'model.safetensors').write_text('tampered', encoding='utf-8')
        with self.assertRaises(ValueError):
            recipe.generate(self.source, self.cal, self.root, self.root / 'out', 'prepare')


if __name__ == '__main__':
    unittest.main()
