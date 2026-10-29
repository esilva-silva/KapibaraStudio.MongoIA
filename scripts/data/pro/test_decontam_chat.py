"""Regressão: chats sintéticos sem snippet_id participam da descontaminação."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import decontaminate as dc


def test_chat_without_snippet_id_has_features():
    rec = {'type': 'chat', 'instruction': 'Busque o projeto no sistema',
           'context': {'editor': 'Esquema sintético: projetoId, sistema'},
           'output': 'db.projetos.find({projetoId: "PRJ-1", sistema: "SIS-1"})'}
    features = dc.features(rec)
    assert dc.record_kind(rec) == 'chat'
    assert features['exact'] and features['tokens'] and features['grams']
    assert features['exact'] == dc.features({**rec, 'id': 'outro'})['exact']


def test_existing_legacy_chat_remains_legacy():
    assert dc.record_kind({'input': 'pedido', 'output': 'codigo'}) == 'legacy_chat'
