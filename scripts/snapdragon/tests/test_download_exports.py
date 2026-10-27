import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

scripts = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('download_exports', scripts / 'download_exports.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

class DownloadTest(unittest.TestCase):
    def test_source_provenance_formats_and_package_integrity(self):
        with tempfile.TemporaryDirectory(dir=scripts) as tmp:
            root=Path(tmp); source=root/'source'; source.mkdir(); package=root/'package'; package.mkdir()
            hashes={}
            for name in ('model.safetensors','config.json','tokenizer.json'):
                path=source/name; path.write_bytes(name.encode()); hashes[name]=mod.file_hash(path)
            (package/'model.onnx').write_bytes(b'graph')
            entry={'path':'model.onnx','bytes':5,'sha256':mod.file_hash(package/'model.onnx')}
            for key in ('source','parent_source_provenance'):
                manifest={'files':[entry],key:{'files_sha256':hashes}}
                (package/'export_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
                self.assertTrue(mod.verify_export(package, source)['passed'])
            (source/'model.safetensors').write_bytes(b'tampered')
            with self.assertRaises(RuntimeError): mod.verify_export(package, source)

    def test_rejects_unsafe_subfolders(self):
        for value in ('../evil','/absolute','a/../b','C:/bad','a\\b','a//b','.'):
            with self.subTest(value=value), self.assertRaises(ValueError): mod.safe_subfolder(value)
        self.assertEqual(mod.safe_subfolder('experimental/model'), 'experimental/model')

if __name__ == '__main__': unittest.main()
