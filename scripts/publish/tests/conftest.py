"""Fixtures dos testes de publicacao: tudo local; qualquer tentativa de rede falha o teste."""
from __future__ import annotations

import socket
import sys
from pathlib import Path

PUBLISH_DIR = Path(__file__).resolve().parents[1]
if str(PUBLISH_DIR) not in sys.path:
    sys.path.insert(0, str(PUBLISH_DIR))

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("tentativa de acesso a rede nos testes de publicacao")

    monkeypatch.setattr(socket.socket, "connect", boom)
    monkeypatch.setattr(socket, "getaddrinfo", boom)
    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.delenv("HF_TOKEN", raising=False)
