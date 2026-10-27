import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

path = Path(__file__).resolve().parents[1] / 'probe_gptq.py'
spec = importlib.util.spec_from_file_location('probe_gptq', path)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class FakeTensor:
    def __init__(self, n):
        self.shape = (1, n)


class FakeTokenizer:
    def __call__(self, prompt, **kwargs):
        assert kwargs['truncation'] and kwargs['max_length'] == 64
        n = min(len(prompt), kwargs['max_length'])
        return {'input_ids': FakeTensor(n), 'attention_mask': FakeTensor(n)}


class TestProbe(unittest.TestCase):
    def test_read_and_tokenize_limited(self):
        with tempfile.TemporaryDirectory(dir=path.parent) as temp:
            data = Path(temp) / 'data.jsonl'
            data.write_text(''.join(json.dumps({'prompt': 'a' * 100}) + '\n' for _ in range(10)), encoding='utf-8')
            samples = probe.tokenize_prompts(FakeTokenizer(), probe.read_prompts(data))
        self.assertEqual(len(samples), 4)
        self.assertTrue(all(s['input_ids'].shape == (1, 64) for s in samples))

    def test_empty_dataset_fails(self):
        with tempfile.TemporaryDirectory(dir=path.parent) as temp:
            data = Path(temp) / 'data.jsonl'
            data.write_text('', encoding='utf-8')
            with self.assertRaises(ValueError):
                probe.read_prompts(data)


if __name__ == '__main__':
    unittest.main()

class TestGpuSelection(unittest.TestCase):
    def test_refuses_integrated_gpu_before_quantization(self):
        from types import SimpleNamespace
        cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
            get_device_name=lambda i: ['AMD Radeon Graphics', 'AMD Radeon RX 7800 XT'][i],
            get_device_properties=lambda i: SimpleNamespace(total_memory=[1024, 16384][i]))
        torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(hip='6', cuda=None))
        facts = probe.inspect_gpu(torch, 'RX 7800 XT', False)
        self.assertEqual(facts['devices'][1]['vram_bytes'], 16384)
        with self.assertRaisesRegex(ValueError, 'HIP_VISIBLE_DEVICES'):
            probe.inspect_gpu(torch, 'RX 7800 XT', True)

    def test_accepts_normalized_discrete_gpu(self):
        from types import SimpleNamespace
        cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
            get_device_name=lambda i: 'AMD Radeon RX7800XT',
            get_device_properties=lambda i: SimpleNamespace(total_memory=16384))
        torch = SimpleNamespace(cuda=cuda, version=SimpleNamespace(hip='6', cuda=None))
        self.assertEqual(probe.inspect_gpu(torch, 'RX 7800 XT', True)['current_name'], 'AMD Radeon RX7800XT')
