import importlib.util
import json
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
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
    def test_shared_olive_parameters_do_not_change_the_token_stream(self):
        rows = [{'id': str(i), 'prompt_text': str(i), 'prompt_ids': [i] * 238} for i in range(256)]
        expected, ledger = mod.build_blocks(rows, Tokenizer())
        tokenizer = Tokenizer()
        tokenizer.no_truncation = lambda: None
        tokenizer.no_padding = lambda: None
        fake_tokenizers = SimpleNamespace(Tokenizer=SimpleNamespace(from_file=lambda _: tokenizer))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory); (root / 'tokenizer.json').write_text('{}')
            with patch.dict(sys.modules, {'tokenizers': fake_tokenizers}):
                actual = mod.kapi_gptq_join_tail_v1(rows, root, root / 'audit.json', ledger['stream_token_ids_sha256'],
                    task='text-generation', text_cols=['prompt'], arbitrary_shared='not logged')
            self.assertEqual([item[0]['input_ids'].tolist() for item in actual], expected)
            audit = json.loads((root / 'audit.json').read_text())
            self.assertEqual(audit['stream_token_ids_sha256'], ledger['stream_token_ids_sha256'])
            self.assertEqual(audit['forwarded_parameter_names'], ['arbitrary_shared', 'task', 'text_cols'])
            self.assertNotIn('not logged', (root / 'audit.json').read_text())

    def test_projection_ignores_heterogeneous_facts_and_preserves_calibration(self):
        rows = [{'id': str(i), 'prompt': str(i), 'prompt_text': str(i), 'prompt_ids': [i] * 238,
                 'facts': {'keys': {'PatientID': i if i % 2 else str(i)}}} for i in range(256)]
        projected = mod.project_rows(rows)
        self.assertEqual(set(projected[0]), {'id', 'prompt', 'prompt_text', 'prompt_ids'})
        self.assertEqual([tuple(row[key] for key in ('id', 'prompt', 'prompt_text', 'prompt_ids')) for row in rows],
                         [tuple(row[key] for key in ('id', 'prompt', 'prompt_text', 'prompt_ids')) for row in projected])
        original_blocks, original_report = mod.build_blocks(rows, Tokenizer())
        projected_blocks, projected_report = mod.build_blocks(projected, Tokenizer())
        self.assertEqual(original_blocks, projected_blocks)
        self.assertEqual(original_report['stream_token_ids_sha256'], projected_report['stream_token_ids_sha256'])

    @unittest.skipUnless(importlib.util.find_spec('pyarrow'), 'PyArrow not installed')
    def test_arrow_loads_projected_rows_with_mixed_metadata(self):
        import pyarrow as pa
        import pyarrow.json as pajson
        rows = [{'id': str(i), 'prompt': str(i), 'prompt_text': str(i), 'prompt_ids': [i],
                 'facts': {'keys': {'PatientID': i if i % 2 else str(i)}}} for i in range(4)]
        payload = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in mod.project_rows(rows)).encode('utf-8')
        table = pajson.read_json(pa.BufferReader(payload))
        self.assertEqual(table.to_pylist(), mod.project_rows(rows))

    def test_projection_rejects_invalid_identity_text_and_tokens(self):
        valid = {'id':'x', 'prompt':'x', 'prompt_text':'x', 'prompt_ids':[1, 2]}
        for changed in ({'id': 1}, {'id': ''}, {'prompt': 'other'}, {'prompt_text': None},
                        {'prompt_ids': [1, '2']}, {'prompt_ids': [True]}):
            with self.subTest(changed=changed), self.assertRaises(ValueError): mod.project_rows([{**valid, **changed}])
        with self.assertRaises(ValueError): mod.project_rows([valid, valid])

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
