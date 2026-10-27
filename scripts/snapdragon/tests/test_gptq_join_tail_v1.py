import importlib.util
from pathlib import Path
import unittest

script = Path(__file__).resolve().parents[1] / 'gptq_join_tail_v1.py'
spec = importlib.util.spec_from_file_location('tail_policy', script)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


class Encoding:
    def __init__(self, ids): self.ids = ids


class Tokenizer:
    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return Encoding([271] if text == '\n\n' else [int(text)] * 238)


class JoinTailTests(unittest.TestCase):
    def test_all_256_ids_and_tail_preserved_in_order_without_padding(self):
        rows = [{'id': str(i), 'prompt_text': str(i), 'prompt_ids': [i] * 238} for i in range(256)]
        blocks, report = mod.build_blocks(rows, Tokenizer())
        expected = [token for i in range(256) for token in [i] * 238 + [271]]
        self.assertEqual([token for block in blocks for token in block], expected)
        self.assertEqual(report['selected_rows'], 256)
        self.assertEqual(len(blocks), 30)
        self.assertEqual(len(blocks[-1]), 1792)
        self.assertEqual([entry['id'] for entry in report['prompts']], [str(i) for i in range(256)])
        self.assertTrue(all(entry['omitted_prompt_tokens'] == 0 for entry in report['prompts']))
        self.assertTrue(all(batch['padding_tokens'] == 0 for batch in report['batches']))
        self.assertEqual(report['stream_token_ids_sha256'], mod.token_sha(expected))

    def test_prompt_identity_and_duplicate_ids_fail(self):
        rows = [{'id': str(i), 'prompt_text': str(i), 'prompt_ids': [i] * 238} for i in range(256)]
        rows[-1]['prompt_ids'] = [1]
        with self.assertRaises(ValueError): mod.build_blocks(rows, Tokenizer())
        rows[-1]['id'] = rows[0]['id']
        with self.assertRaises(ValueError): mod.build_blocks(rows, Tokenizer())


if __name__ == '__main__': unittest.main()
