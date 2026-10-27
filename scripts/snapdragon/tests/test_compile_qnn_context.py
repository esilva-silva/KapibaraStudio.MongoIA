import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import onnx
from onnx import helper

scripts = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('compile_qnn_context', scripts / 'compile_qnn_context.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

class ContextTest(unittest.TestCase):
    def test_context_reference_hash_and_missing(self):
        with tempfile.TemporaryDirectory(dir=scripts) as tmp:
            root=Path(tmp); (root/'ctx.bin').write_bytes(b'context')
            node=helper.make_node('EPContext', [], [], domain='com.microsoft', ep_cache_context='ctx.bin')
            onnx.save(helper.make_model(helper.make_graph([node], 'g', [], [])), str(root/'ctx.onnx'))
            self.assertEqual(mod.context_files(root/'ctx.onnx'), [(root/'ctx.bin').resolve()])
            (root/'ctx.bin').unlink()
            with self.assertRaises(FileNotFoundError): mod.context_files(root/'ctx.onnx')

    def test_manifest_checks_input_and_size(self):
        with tempfile.TemporaryDirectory(dir=scripts) as tmp:
            root=Path(tmp); model=root/'model.onnx'
            onnx.save(helper.make_model(helper.make_graph([], 'g', [], [])), str(model))
            manifest=root/'manifest.json'
            manifest.write_text(json.dumps({'files':[{'path':'model.onnx','bytes':model.stat().st_size,'sha256':mod.sha(model)}]}))
            mod.verify_manifest(manifest, [model])
            model.write_bytes(b'tampered')
            with self.assertRaises(ValueError): mod.verify_manifest(manifest, [model])

    def test_context_path_escape_rejected(self):
        with tempfile.TemporaryDirectory(dir=scripts) as tmp:
            root=Path(tmp)
            node=helper.make_node('EPContext', [], [], domain='com.microsoft', ep_cache_context='../escape.bin')
            onnx.save(helper.make_model(helper.make_graph([node], 'g', [], [])), str(root/'ctx.onnx'))
            with self.assertRaises(ValueError): mod.context_files(root/'ctx.onnx')

if __name__ == '__main__': unittest.main()
