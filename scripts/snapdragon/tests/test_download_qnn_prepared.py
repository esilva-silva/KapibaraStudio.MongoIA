import json
from pathlib import Path
import sys
import tempfile
import unittest

scripts = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(scripts))
import download_qnn_prepared as mod


class PreparedDownloadTests(unittest.TestCase):
    def test_immutable_pin_and_safe_paths_before_network(self):
        for revision in ('main', 'a' * 39, '../' + 'a' * 40):
            with self.assertRaises(ValueError):
                mod.download('owner/repo', revision, 'qnn-prepared', '.', '.', 'b' * 64)
        for name in ('../escape', '/absolute', 'a\\b', 'a//b', 'C:/escape'):
            with self.assertRaises(ValueError):
                mod.file_inventory([{'path': name, 'bytes': 1, 'sha256': 'b' * 64}])

    def test_manifest_source_hashes_inventory_and_tokenizer(self):
        with tempfile.TemporaryDirectory(dir=scripts) as temporary:
            root = Path(temporary)
            source = root / 'source'; source.mkdir()
            package = root / 'package'; package.mkdir()
            source_entries = []
            for name in ('model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'):
                path = source / name; path.write_text(name, encoding='utf-8')
                source_entries.append({'path': name, 'size': path.stat().st_size, 'sha256': mod.sha256(path)})
            provenance = {'repo_id': 'owner/raw', 'revision': 'a' * 40, 'files': source_entries}
            (source / 'source_manifest.json').write_text(json.dumps(provenance), encoding='utf-8')
            entries = []
            for name in ('model.onnx', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'):
                path = package / name; path.write_text(name, encoding='utf-8')
                entries.append({'path': name, 'bytes': path.stat().st_size, 'sha256': mod.sha256(path)})
            manifest = package / 'export_manifest.json'
            data = {'source': provenance, 'files': entries}
            manifest.write_text(json.dumps(data), encoding='utf-8')
            expected = mod.sha256(manifest)
            self.assertTrue(mod.verify_prepared(package, source, expected, True)['passed'])
            for alias in ('parent_source_provenance', 'source_provenance'):
                aliased = {alias: data['source'], 'files': entries}
                manifest.write_text(json.dumps(aliased), encoding='utf-8')
                self.assertTrue(mod.verify_prepared(package, source, mod.sha256(manifest), True)['passed'])
                aliased[alias] = {**data['source'], 'revision': 'b' * 40}
                manifest.write_text(json.dumps(aliased), encoding='utf-8')
                with self.assertRaises(ValueError): mod.verify_prepared(package, source, mod.sha256(manifest))
            data['source']['files'] = list(reversed(source_entries))
            data['source']['files'] = [{**entry, 'sha256': entry['sha256'].upper()} for entry in data['source']['files']]
            manifest.write_text(json.dumps(data), encoding='utf-8')
            expected = mod.sha256(manifest)
            self.assertTrue(mod.verify_prepared(package, source, expected, True)['passed'])
            self.assertEqual(mod.file_inventory([{'path': 'a', 'size': 1, 'sha256': 'b' * 64}]),
                             mod.file_inventory([{'path': 'a', 'bytes': 1, 'sha256': 'b' * 64}]))
            with self.assertRaises(ValueError): mod.verify_prepared(package, source, '0' * 64)
            (package / 'extra').write_text('extra')
            with self.assertRaises(ValueError): mod.verify_prepared(package, source, expected)
            (package / 'extra').unlink()
            data['source']['revision'] = 'b' * 40
            manifest.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaises(ValueError): mod.verify_prepared(package, source, mod.sha256(manifest))
            data['source']['revision'] = 'a' * 40
            tokenizer = package / 'tokenizer.json'
            tokenizer.write_text('derived tokenizer')
            token_entry = next(entry for entry in entries if entry['path'] == 'tokenizer.json')
            token_entry.update(bytes=tokenizer.stat().st_size, sha256=mod.sha256(tokenizer))
            manifest.write_text(json.dumps(data), encoding='utf-8')
            self.assertTrue(mod.verify_prepared(package, source, mod.sha256(manifest))['passed'])
            with self.assertRaises(ValueError): mod.verify_prepared(package, source, mod.sha256(manifest), True)
            manifest.write_text(json.dumps(data), encoding='utf-8')
            (source / 'model.safetensors').write_text('tampered')
            with self.assertRaises(ValueError): mod.verify_prepared(package, source, mod.sha256(manifest))


if __name__ == '__main__':
    unittest.main()
