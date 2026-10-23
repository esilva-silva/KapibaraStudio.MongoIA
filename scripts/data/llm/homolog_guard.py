#!/usr/bin/env python3
"""Guarda do conjunto de HOMOLOGACAO (data/homolog; mesma ideia do blind_guard, mas para o conjunto LLM `hom`).

Geradores de treino NAO podem ler data/homolog nem produzir exemplos cujo hash coincida com um item selado:

    import homolog_guard as hg
    hg.enforce()               # instala o bloqueio de leitura (open/io.open/os.open/os.listdir/os.scandir) sobre data/homolog
    hg.check_path(path)        # levanta HomologAccessError se `path` esta dentro de data/homolog
    hg.check_record(rec)       # idem se o hash canonico coincide com um item selado
    hg.check_text(prefix + "\\x00" + suffix)   # idem se o hash do prompt normalizado coincide com um selado

So `homolog_tool.py` (e o gate final de avaliacao) le o diretorio, dentro de `with authorized(reason, who):` (grava access.log.jsonl).
`run_guarded.py script.py args...` executa qualquer gerador/pipeline existente com `enforce()` ativo SEM editar o gerador.
Diretorio: data/homolog da raiz do repo, sobrescrevel por $KAPI_HOMOLOG_DIR (testes).
"""
from __future__ import annotations

import builtins
import hashlib
import io
import json
import os
import re
import threading
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DIR = ROOT / "data" / "homolog"
_local = threading.local()
_installed = False
_orig: dict = {}


class HomologAccessError(PermissionError):
    """Tentativa de ler/usar o conjunto de homologacao fora do caminho autorizado."""


def homolog_dir(override=None) -> Path:
    if override is not None:
        return Path(override)
    env = os.environ.get("KAPI_HOMOLOG_DIR")
    return Path(env) if env else DEFAULT_DIR


def _norm(p) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(p))))


def is_homolog_path(path, hdir=None) -> bool:
    if isinstance(path, int):
        return False
    b = _norm(homolog_dir(hdir))
    p = _norm(path)
    return p == b or p.startswith(b + os.sep)


def authorized_now() -> bool:
    return bool(getattr(_local, "depth", 0))


class authorized:
    """Contexto de leitura autorizada (homolog_tool). Registra quem leu e por que."""

    def __init__(self, reason: str, who: str = "homolog_tool", hdir=None, detail: dict | None = None):
        if not reason:
            raise HomologAccessError("leitura do conjunto de homologacao exige --reason")
        self.reason, self.who, self.hdir, self.detail = reason, who, hdir, detail or {}

    def __enter__(self):
        _local.depth = getattr(_local, "depth", 0) + 1
        log = homolog_dir(self.hdir) / "v1" / "access.log.jsonl"
        try:
            log.parent.mkdir(parents=True, exist_ok=True)
            with _orig.get("open", builtins.open)(log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "who": self.who, "reason": self.reason,
                                    "pid": os.getpid(), **self.detail}, ensure_ascii=False) + "\n")
        except OSError:
            pass
        return self

    def __exit__(self, *a):
        _local.depth = max(0, getattr(_local, "depth", 1) - 1)


def check_path(path, hdir=None):
    if not authorized_now() and is_homolog_path(path, hdir):
        raise HomologAccessError(f"acesso recusado ao conjunto de homologacao: {path!s} (use scripts/data/llm/homolog_tool.py read --reason ...)")
    return path


def enforce(hdir=None) -> None:
    """Bloqueia leitura/listagem de data/homolog neste processo (idempotente)."""
    global _installed
    if _installed:
        return
    _orig.update(open=builtins.open, io_open=io.open, os_open=os.open, listdir=os.listdir, scandir=os.scandir)

    def g_open(file, *a, **k):
        if not isinstance(file, int):
            check_path(file, hdir)
        return _orig["open"](file, *a, **k)

    def g_io_open(file, *a, **k):
        if not isinstance(file, int):
            check_path(file, hdir)
        return _orig["io_open"](file, *a, **k)

    def g_os_open(path, *a, **k):
        check_path(path, hdir)
        return _orig["os_open"](path, *a, **k)

    def g_listdir(path=".", *a, **k):
        check_path(path, hdir)
        return _orig["listdir"](path, *a, **k)

    def g_scandir(path=".", *a, **k):
        check_path(path, hdir)
        return _orig["scandir"](path, *a, **k)

    builtins.open, io.open, os.open, os.listdir, os.scandir = g_open, g_io_open, g_os_open, g_listdir, g_scandir
    _installed = True


def uninstall() -> None:
    global _installed
    if _installed:
        builtins.open, io.open, os.open = _orig["open"], _orig["io_open"], _orig["os_open"]
        os.listdir, os.scandir = _orig["listdir"], _orig["scandir"]
        _installed = False


# ---- hashes (compartilham as regras do blind_guard) --------------------------------------------------------------------
def canonical_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def item_sha256(item: dict) -> str:
    return hashlib.sha256(canonical_json(item).encode("utf-8")).hexdigest()


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s).strip().lower()


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def manifest(hdir=None) -> dict:
    mf = homolog_dir(hdir) / "v1" / "sealed" / "manifest.json"
    try:
        with _orig.get("open", builtins.open)(mf, encoding="utf-8") as f:     # leitura do manifesto e permitida (so hashes)
            return json.load(f)
    except (OSError, ValueError):
        return {}


def check_record(record: dict, hdir=None):
    m = manifest(hdir)
    if item_sha256(record) in set((m.get("item_sha256") or {}).values()):
        raise HomologAccessError("registro identico a item do conjunto de homologacao (hash)")
    return record


def check_text(text: str, hdir=None):
    m = manifest(hdir)
    if prompt_sha256(text) in set(m.get("prompt_sha256") or []):
        raise HomologAccessError("prompt identico a item do conjunto de homologacao (hash)")
    return text
