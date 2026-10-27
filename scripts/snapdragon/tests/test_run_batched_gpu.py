import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
scripts = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location('batch', scripts / 'run_batched_gpu.py'); batch = importlib.util.module_from_spec(spec); spec.loader.exec_module(batch)

class BatchTests(unittest.TestCase):
    def test_child_validation_rejects_unverified_and_wrong_ids(self):
        report = {'status':'completed','corpus_sha256':'c','runner_binary_sha256':'b','items':[{'id':'a'}],
                  'provider_evidence':{'requested':'DmlExecutionProvider','verified':True}}
        self.assertEqual(len(batch.validate_child(report, ['a'], 'c', 'b', 0)), 1)
        with self.assertRaises(ValueError): batch.validate_child(report, ['b'], 'c', 'b', 0)
        report['provider_evidence']['verified'] = False
        with self.assertRaises(ValueError): batch.validate_child(report, ['a'], 'c', 'b', 0)

    def test_failed_child_stops_aggregate_and_preserves_batch(self):
        with tempfile.TemporaryDirectory(dir=scripts) as d:
            root=Path(d); model=root/'model'; model.mkdir(); (model/'genai_config.json').write_text('{}')
            (model/'export_manifest.json').write_text('{}'); runner=root/'runner.exe';runner.write_bytes(b'launcher');runner.with_suffix('.dll').write_bytes(b'runner')
            corpus=root/'corpus.jsonl'; corpus.write_text(''.join(json.dumps({'id':str(i)})+'\n' for i in range(3)))
            def child(cmd, **kwargs):
                out=Path(cmd[cmd.index('--out')+1]);out.write_text(json.dumps({'status':'failed','items':[]}))
                return SimpleNamespace(returncode=1, stdout=['failed child\n'], wait=lambda: 1)
            with patch.object(batch, 'verify', return_value={'passed':True,'manifest':'export_manifest.json'}), patch.object(batch.subprocess, 'Popen', side_effect=child) as call:
                code=batch.run(runner,model,corpus,root/'out.json',root/'work',2)
            report=json.loads((root/'out.json').read_text())
            self.assertEqual(code,1);self.assertEqual(report['status'],'failed');self.assertEqual(call.call_count,1)
            self.assertTrue((root/'work/batch-000.jsonl').exists());self.assertEqual(report['items'],[])

    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory(dir=scripts) as d:
            path=Path(d)/'data';path.write_text('{"id":"a"}\n{"id":"a"}\n')
            with self.assertRaises(ValueError): batch.parse_corpus(path)

if __name__ == '__main__': unittest.main()
