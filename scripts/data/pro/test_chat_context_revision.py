"""A revisão do contexto não pode alterar o pedido nem a consulta de referência."""
import json
from itertools import zip_longest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_revised_chats_preserve_queries_and_parameters():
    source = ROOT / 'data/generated/pro-context-v1b/chat-train.jsonl'
    revised = ROOT / 'data/generated/pro-chat-context-v2/chat-train.jsonl'
    count = 0
    with source.open(encoding='utf8') as a, revised.open(encoding='utf8') as b:
        for old_line, new_line in zip_longest(a, b):
            assert old_line and new_line
            old, new = json.loads(old_line), json.loads(new_line)
            assert new['instruction'] == old['instruction']
            assert new['output'] == old['output']
            assert new['split'] == old['split'] == 'train'
            assert {k: v for k, v in new['context'].items() if k != 'editor'} == {
                k: v for k, v in old['context'].items() if k != 'editor'}
            i = int(old['id'].split('-')[-1])
            n = i // 6
            editor = new['context']['editor']
            assert all(name in editor for name in ('projetoId', 'sistema', 'situacao', 'prioridade'))
            assert f'PRJ-{n:05d}' in editor and f'SIS-TRAIN-{n % 71:02d}' in editor
            assert ('ativo', 'pausado', 'concluido')[n % 3] in editor
            assert f'Prioridade: {n % 5 + 1}' in editor
            count += 1
    assert count == 6000
