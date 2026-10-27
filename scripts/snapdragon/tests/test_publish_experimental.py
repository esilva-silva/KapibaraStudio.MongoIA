import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

path = Path(__file__).resolve().parents[1] / 'publish_experimental.py'
spec = importlib.util.spec_from_file_location('publish_experimental', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class TestInventory(unittest.TestCase):
    def test_hash_and_unexpected_files(self):
        with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
            root = Path(tmp)
            (root / 'model.onnx').write_bytes(b'graph')
            manifest = root / 'export_manifest.json'
            manifest.write_text(json.dumps({'files': [{'path': 'model.onnx', 'sha256': mod.sha(root / 'model.onnx')}]}))
            plan = {'prepared_package': str(root), 'manifest_sha256': mod.sha(manifest)}
            _, expected = mod.inventory(plan)
            self.assertEqual(expected['model.onnx'], mod.sha(root / 'model.onnx'))
            (root / 'extra').write_text('unexpected')
            with self.assertRaises(ValueError):
                mod.inventory(plan)
            (root / 'extra').unlink()
            (root / 'model.onnx').write_bytes(b'tampered')
            with self.assertRaises(ValueError):
                mod.inventory(plan)

    def test_escape_rejected(self):
        with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
            root = Path(tmp); manifest = root / 'export_manifest.json'
            manifest.write_text(json.dumps({'files': [{'path': '../escape', 'sha256': 'x'}]}))
            with self.assertRaises(ValueError):
                mod.inventory({'prepared_package': str(root), 'manifest_sha256': mod.sha(manifest)})

if __name__ == '__main__':
    unittest.main()
