import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

path = Path(__file__).resolve().parents[1] / 'prepare_experimental_publication.py'
spec = importlib.util.spec_from_file_location('prep_publish', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

class PublicationTest(unittest.TestCase):
    def test_preserves_graph_and_updates_manifest_and_report(self):
        with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
            root = Path(tmp); source = root / 'source'; source.mkdir()
            (source / 'model.onnx').write_bytes(b'graph')
            (source / 'model.onnx.data').write_bytes(b'weights')
            (source / 'RMSNORM_TRANSFORM.md').write_text('pending', encoding='utf-8')
            files = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': mod.sha(p)} for p in source.iterdir()]
            (source / 'kapicoder_manifest.json').write_text(json.dumps({'files': files}), encoding='utf-8')
            files.append({'path': 'kapicoder_manifest.json', 'sha256': mod.sha(source / 'kapicoder_manifest.json')})
            (source / 'export_manifest.json').write_text(json.dumps({'files': files, 'source': {'revision': 'abc'}}), encoding='utf-8')
            comparison = root / 'comparison.json'
            comparison.write_text(json.dumps({'qualified': False, 'proxy': True, 'baseline': {'n': 200, 'trimmed_exact_match': .4}, 'candidate': {'n': 200, 'trimmed_exact_match': .395, 'provider_evidence': {'requested': 'DmlExecutionProvider', 'verified': True}, 'execution_protocol': {'single_session': False, 'batch_size': 25}}, 'limitations': ['proxy']}), encoding='utf-8')
            before = mod.sha(source / 'model.onnx')
            plan = mod.prepare(source, root / 'out', comparison, 'esilva/test', 'experimental/rmsnorm', '913ef5fc78e70000e118d4384cadb401fcac4f4f', 'dml-fp16-split')
            self.assertEqual(plan['base_commit'], '913ef5fc78e70000e118d4384cadb401fcac4f4f')
            self.assertEqual(plan['path_in_repo'], 'dml-fp16-split')
            self.assertEqual(plan['status'], 'not_published')
            self.assertEqual(mod.sha(root / 'out/model.onnx'), before)
            self.assertEqual((source / 'RMSNORM_TRANSFORM.md').read_text(), 'pending')
            manifest = json.loads((root / 'out/export_manifest.json').read_text())
            self.assertFalse(manifest['validation']['qualified'])
            self.assertEqual(manifest['validation']['execution_protocol']['batch_size'], 25)
            self.assertIn('fresh process every 25 requests', (root / 'out/RMSNORM_TRANSFORM.md').read_text())
            for item in manifest['files']:
                self.assertEqual(item['sha256'], mod.sha(root / 'out' / item['path']))
            self.assertEqual(manifest['validation_parent_export_manifest_sha256'], mod.sha(source / 'export_manifest.json'))
            self.assertEqual(mod.sha(root / 'out/model.onnx.data'), mod.sha(source / 'model.onnx.data'))

    def test_rejects_unsafe_subfolder(self):
        with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
            root = Path(tmp); (root / 'source').mkdir(); (root / 'comp').write_text('{}')
            with self.assertRaises(ValueError):
                mod.prepare(root / 'source', root / 'out', root / 'comp', 'esilva/test', 'experimental/test', subfolder='../escape')

    def test_refuses_main(self):
        with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
            root = Path(tmp); (root / 'source').mkdir(); (root / 'comp').write_text('{}')
            with self.assertRaises(ValueError):
                mod.prepare(root / 'source', root / 'out', root / 'comp', 'esilva/test', 'main')

if __name__ == '__main__':
    unittest.main()
