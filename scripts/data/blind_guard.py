#!/usr/bin/env python3
"""Guarda do conjunto cego (Fase 03 §4.6 / G3.4).

Geradores (`scripts/data/gen_*.py`) e o decontaminador importam daqui:

    from blind_guard import check_path, check_record, check_text
    check_path(path)          # levanta BlindAccessError se path estiver em data/eval/blind
    check_record(obj)         # idem se o hash canonico do item coincide com um item selado
    check_text(prompt)        # idem se o hash do prompt normalizado coincide com um selado

Diretorio do blind: `data/eval/blind` da raiz do repo, sobrescrevel por $KAPI_BLIND_DIR (testes).
Somente `blind_tool.py` (e o gate final de avaliacao) le o blind; todo o resto passa por aqui.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BLIND_DIR = ROOT / "data" / "eval" / "blind"


class BlindAccessError(PermissionError):
    """Tentativa de ler/usar o conjunto cego fora do caminho autorizado."""


def blind_dir(override=None) -> Path:
    if override is not None:
        return Path(override)
    env = os.environ.get("KAPI_BLIND_DIR")
    return Path(env) if env else DEFAULT_BLIND_DIR


def _norm(p) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(p))))


def is_blind_path(path, blind=None) -> bool:
    """True se `path` (existente ou nao; segue symlinks/junctions) esta dentro do blind."""
    b = _norm(blind_dir(blind))
    p = _norm(path)
    return p == b or p.startswith(b + os.sep)


def check_path(path, blind=None):
    """Levanta BlindAccessError se `path` estiver dentro de data/eval/blind. Retorna `path`."""
    if is_blind_path(path, blind):
        raise BlindAccessError(
            f"acesso recusado ao conjunto cego: {path!s} (use scripts/data/blind_tool.py read --reason ...)")
    return path


def check_paths(paths: Iterable, blind=None):
    for p in paths:
        check_path(p, blind)


# ---- hashes (compartilhados com blind_tool) --------------------------------------------------

def canonical_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def item_sha256(item: dict) -> str:
    return hashlib.sha256(canonical_json(item).encode("utf-8")).hexdigest()


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s).strip().lower()


def prompt_text(item: dict) -> str:
    if item.get("kind") == "autocomplete":
        return (item.get("prefix") or "") + "\x00" + (item.get("suffix") or "")
    return (item.get("instruction") or "") + "\x00" + (item.get("context") or "")


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def _manifest(blind=None) -> dict:
    mf = blind_dir(blind) / "sealed" / "manifest.json"
    try:
        return json.loads(mf.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def check_record(record: dict, blind=None):
    """Recusa registro cujo hash canonico coincide com um item selado."""
    m = _manifest(blind)
    if item_sha256(record) in set((m.get("item_sha256") or {}).values()):
        raise BlindAccessError("registro identico a item do conjunto cego (hash)")
    return record


def check_text(text: str, blind=None):
    """Recusa texto cujo hash normalizado coincide com o prompt de um item selado
    (para autocomplete passe prefix + '\x00' + suffix; para chat instruction + '\x00' + context)."""
    m = _manifest(blind)
    if prompt_sha256(text) in set(m.get("prompt_sha256") or []):
        raise BlindAccessError("prompt identico a item do conjunto cego (hash)")
    return text
