import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import sys

HERE = Path(__file__).resolve().parents[1]
tempfile.tempdir = str(HERE / 'tests')

def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

runner = load('run_sample')
exporter = load('export_gpu')
with patch.dict(sys.modules, {'run_sample': runner}):
    remote = load('remote_prepare')

class RunnerTests(unittest.TestCase):
    def test_qnn_overlay_preserves_two_stages_and_disables_fallback(self):
        original = {'model': {'decoder': {'session_options': {'existing': 1}, 'pipeline': [
            {'session_options': {'provider_options': [{'qnn': {'enable_htp_weight_sharing': '1'}}]}},
            {'session_options': {'custom': 'keep'}}]}}}
        overlay = runner.genai_session_overlay(original, 'profile', 'qnn')['model']['decoder']
        self.assertTrue(overlay['session_options']['disable_cpu_ep_fallback'])
        self.assertEqual(overlay['session_options']['existing'], 1)
        for index, stage in enumerate(overlay['pipeline']):
            self.assertTrue(stage['session_options']['disable_cpu_ep_fallback'])
            self.assertEqual(stage['session_options']['enable_profiling'], f'profile-{index}')
        self.assertEqual(overlay['pipeline'][0]['session_options']['provider_options'][0]['qnn']['enable_htp_weight_sharing'], '1')
        self.assertNotIn('disable_cpu_ep_fallback', original['model']['decoder']['pipeline'][0]['session_options'])
        cpu = runner.genai_session_overlay(original, None, 'cpu')['model']['decoder']
        self.assertNotIn('disable_cpu_ep_fallback', cpu['session_options'])

    def test_qnn_rejects_non_htp_backend(self):
        with self.assertRaises(ValueError): runner.validate_qnn_options({'backend_type': 'cpu'}, 'QnnHtp.dll')
        with self.assertRaises(ValueError): runner.validate_qnn_options({'backend_path': 'QnnGpu.dll'}, 'QnnHtp.dll')
        runner.validate_qnn_options({'backend_path': 'QnnHtp.dll'}, 'QnnHtp.dll')

    def test_single_item_records_phases_and_new_convention(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            corpus = root / 'one.jsonl'
            corpus.write_text(json.dumps({'id': 'one', 'prompt_text': 'x'}))
            fake = SimpleNamespace(evidence={'effective': 'cpu'}, generate=lambda item: {'generated_text': 'x'})
            with patch.object(runner, 'HFRunner', return_value=fake):
                self.assertEqual(runner.run(root, corpus, 'hf', root / 'result.json'), 0)
            report = json.loads((root / 'result.json').read_text())
            progress = json.loads((root / 'result.json.progress.json').read_text())
            self.assertEqual(report['runner_version'], 2)
            self.assertEqual(progress['phase'], 'completed')
            self.assertIn('model_load_elapsed_s', report)
    def test_gpu_selection_rejects_igpu_and_uses_real_index(self):
        devices = [{'index': 0, 'name': 'AMD Radeon Graphics', 'vram_bytes': 512},
                   {'index': 1, 'name': 'AMD Radeon RX 7800 XT', 'vram_bytes': 16384}]
        self.assertEqual(remote.select_gpu(devices)['index'], 1)
        with self.assertRaises(ValueError):
            remote.select_gpu(devices[:1])
        fake = SimpleNamespace(device=lambda _: SimpleNamespace(index=0),
                               cuda=SimpleNamespace(get_device_name=lambda _: 'AMD Radeon Graphics'))
        with self.assertRaisesRegex(ValueError, 'GPU mismatch'):
            runner.verify_gpu(fake, 'cuda:0', 'RX 7800 XT')

    def test_budget_preserves_generation_room(self):
        runner.check_ids([1] * 2016)
        with self.assertRaises(ValueError):
            runner.check_ids([1] * 2017)

    def test_timing_excludes_first_token_from_decode_rate(self):
        r = runner.measurement([1, 2, 3], 'x', 10, 11, 13, 'eos', 5)
        self.assertEqual(r['ttft_ms'], 1000)
        self.assertEqual(r['decode_tokens_per_second'], 1)
        self.assertTrue(r['eos'])

    def test_profile_evidence_requires_requested_provider_events(self):
        with tempfile.TemporaryDirectory() as d:
            obj = runner.GenAIRunner.__new__(runner.GenAIRunner)
            obj.profile_prefix = Path(d) / 'profile'
            obj.backend = 'dml'
            obj.evidence = {}
            (Path(d) / 'profile_1.json').write_text(json.dumps([{'args': {'provider': 'CPUExecutionProvider'}}]))
            obj.close()
            self.assertFalse(obj.evidence['verified'])
            self.assertEqual(obj.evidence['cpu_kernel_events'], 1)
            (Path(d) / 'profile_2.json').write_text(json.dumps([{'args': {'provider': 'DmlExecutionProvider'}}]))
            obj.close()
            self.assertTrue(obj.evidence['verified'])

    def test_suffix_is_removed_from_text(self):
        row = runner.trim_suffix({'generated_text': 'answerSUFFIXtail'}, {'suffix': 'SUFFIX'})
        self.assertEqual(row['generated_text'], 'answer')
        self.assertEqual(row['stop_reason'], 'suffix')

    def test_qnn_fails_with_written_report(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / 'items.jsonl').write_text(json.dumps({'id': 'x', 'prompt_text': 'a'}), encoding='utf-8')
            self.assertEqual(runner.run(p, p / 'items.jsonl', 'qnn', p / 'report.json'), 1)
            report = json.loads((p / 'report.json').read_text())
            self.assertEqual(report['status'], 'failed')

class ExportTests(unittest.TestCase):
    def test_requires_commit(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                exporter.source_provenance(d)

    def test_original_tokenizer_and_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            source, out = Path(d) / 'src', Path(d) / 'out'
            source.mkdir()
            (source / 'config.json').write_text('{}')
            (source / 'tokenizer.json').write_bytes(b'{"original": true}\n')
            source_files = [{'path': p.name, 'size': p.stat().st_size, 'sha256': exporter.sha256(p)} for p in source.iterdir()]
            (source / 'source_manifest.json').write_text(json.dumps({'revision': 'a' * 40, 'repo_id': 'test', 'files': source_files}))
            def build(cmd, **kwargs):
                self.assertEqual(cmd[cmd.index('-p') + 1], 'fp16')
                self.assertEqual(cmd[cmd.index('-e') + 1], 'dml')
                self.assertEqual(kwargs['env']['HF_HUB_OFFLINE'], '1')
                (out / 'genai_config.json').write_text('{"model":{},"search":{}}')
                (out / 'model.onnx').write_bytes(b'fake')
                (out / 'tokenizer.json').write_text('changed')
            with patch.object(exporter.subprocess, 'run', side_effect=build):
                manifest = exporter.export(source, out)
            self.assertEqual((out / 'tokenizer.json').read_bytes(), (source / 'tokenizer.json').read_bytes())
            self.assertEqual(json.loads((out / 'genai_config.json').read_text())['search']['max_length'], 2048)
            for entry in manifest['files']:
                self.assertEqual(entry['sha256'], exporter.sha256(out / entry['path']))
            with self.assertRaises(ValueError):
                exporter.export(source, out)
            (source / 'tokenizer.json').write_text('tampered')
            with self.assertRaisesRegex(ValueError, 'integrity mismatch'):
                exporter.source_provenance(source)

if __name__ == '__main__':
    unittest.main()
