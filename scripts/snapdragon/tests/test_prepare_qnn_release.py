from copy import deepcopy
from pathlib import Path
import json
import sys
import tempfile
import unittest

scripts = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(scripts))
import prepare_qnn_release as mod


class ReleaseValidationTests(unittest.TestCase):
    def test_compile_hash_and_runtime_binary_mismatch(self):
        with tempfile.TemporaryDirectory(dir=scripts) as temp:
            package = Path(temp).resolve()
            binary = package / 'context.bin'; binary.write_bytes(b'compiled')
            report = package / 'compile.json'
            report.write_text(json.dumps({'status': 'compiled_not_inference_validated', 'cpu_fallback_disabled': True,
                'device_type': 'NPU', 'versions': {'onnxruntime': '1', 'onnxruntime-qnn': '2', 'onnx': '3'},
                'artifacts': [{'path': binary.name, 'bytes': binary.stat().st_size, 'sha256': mod.sha256(binary)}]}))
            expected = mod.sha256(report)
            mod.validate_compile_report(report, package, expected)
            with self.assertRaises(ValueError): mod.validate_compile_report(report, package, '0' * 64)
            binary.write_bytes(b'tampered')
            with self.assertRaises(ValueError): mod.validate_compile_report(report, package, expected)

    def test_model_path_and_generation_contract(self):
        package = Path('package').resolve()
        report = {'model': str(package), 'status': 'completed', 'backend': 'qnn', 'greedy': True,
                  'context_tokens': 2048, 'max_new_tokens': 32, 'total': 1,
                  'items': [{'generated_tokens': 1, 'generated_ids': [7]}],
                  'provider_evidence': {'verified': True, 'requested': 'QNNExecutionProvider',
                                        'kernel_events_by_provider': {'QNNExecutionProvider': 1}}}
        mod.validate_run(report, package, 1)
        for key, bad in (('model', str(package / 'other')), ('greedy', False), ('status', 'failed')):
            changed = deepcopy(report); changed[key] = bad
            with self.assertRaises(ValueError): mod.validate_run(changed, package, 1)
        report['items'][0]['generated_tokens'] = 33
        with self.assertRaises(ValueError): mod.validate_run(report, package, 1)

    def test_stage_hash_and_transformer_cpu_rejected(self):
        with tempfile.TemporaryDirectory(dir=scripts) as temp:
            run = Path(temp) / 'run.json'; run.write_text('{}')
            summary = {'report': str(run), 'report_sha256': mod.sha256(run), 'status': 'verified', 'errors': [],
                       'stages': [{'name': name, 'valid': True, 'intended_providers': [provider],
                                   'kernel_events_by_provider': {ep: 1}}
                                  for name, provider, ep in [('embeddings', 'cpu', 'CPUExecutionProvider'),
                                      ('context', 'qnn', 'QNNExecutionProvider'), ('iterator', 'qnn', 'QNNExecutionProvider'),
                                      ('lm_head', 'cpu', 'CPUExecutionProvider')]]}
            mod.validate_stages(summary, run)
            invalid = deepcopy(summary); invalid['report_sha256'] = '0' * 64
            with self.assertRaises(ValueError): mod.validate_stages(invalid, run)
            invalid = deepcopy(summary); invalid['stages'][1]['kernel_events_by_provider']['CPUExecutionProvider'] = 1
            with self.assertRaises(ValueError): mod.validate_stages(invalid, run)


if __name__ == '__main__':
    unittest.main()
