import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
scripts=Path(__file__).resolve().parents[1];sys.path.insert(0,str(scripts))
spec=importlib.util.spec_from_file_location('private_transfer',scripts/'publish_private_transfer.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class TransferTests(unittest.TestCase):
    def test_public_repo_rejected(self):
        api=SimpleNamespace(repo_info=lambda *a,**kw:SimpleNamespace(private=False))
        with self.assertRaises(ValueError):m.require_private(api,'me/repo')
    def test_manifest_and_tampering(self):
        with tempfile.TemporaryDirectory(dir=scripts) as tmp:
            root=Path(tmp);file=root/'model.onnx';file.write_bytes(b'graph')
            manifest=root/'export_manifest.json';manifest.write_text(json.dumps({'files':[{'path':'model.onnx','bytes':5,'sha256':m.sha256(file)}]}))
            digest=m.sha256(manifest);_,entries=m.validate_package(root,digest)
            self.assertEqual(len(entries),2)
            file.write_bytes(b'other')
            with self.assertRaises(ValueError):m.validate_package(root,digest)
    def test_unsafe_subfolder(self):
        for value in ('../outside','/absolute','C:/bad','a/../b'):
            with self.subTest(value=value),self.assertRaises(ValueError):m.safe_subfolder(value)
if __name__=='__main__':unittest.main()
