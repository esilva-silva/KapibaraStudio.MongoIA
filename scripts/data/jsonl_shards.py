#!/usr/bin/env python3
"""Utilitarios de jsonl em shards (streaming) usados pelos geradores e pelo pipeline do dataset v2.

  resolve_inputs(path)        arquivo unico `x.jsonl` OU shards `x-00000.jsonl ...` (mesmo diretorio) OU diretorio
  iter_jsonl(paths)           gera dicts, linha a linha (nunca carrega o arquivo inteiro)
  ShardWriter(dir, stem, n)   grava `stem-00000.jsonl`, ... com `n` linhas por shard (n=0: arquivo unico `stem.jsonl`);
                              `close()` devolve [{path, sha256, records}] e o sha256 do fluxo completo
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def resolve_inputs(path) -> list[Path]:
    p = Path(path)
    if p.is_dir():
        return sorted(p.glob("*.jsonl"))
    if p.exists():
        return [p]
    shards = sorted(p.parent.glob(p.stem + "-[0-9][0-9][0-9][0-9][0-9]" + p.suffix))
    return shards


def iter_jsonl(paths):
    if isinstance(paths, (str, Path)):
        paths = resolve_inputs(paths)
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def count_lines(paths) -> int:
    n = 0
    for p in paths:
        with open(p, "rb") as f:
            for line in f:
                if line.strip():
                    n += 1
    return n


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


class ShardWriter:
    def __init__(self, out_dir, stem: str, shard_size: int = 0, suffix: str = ".jsonl"):
        self.dir, self.stem, self.size, self.suffix = Path(out_dir), stem, int(shard_size or 0), suffix
        self.dir.mkdir(parents=True, exist_ok=True)
        for old in list(self.dir.glob(f"{stem}-[0-9][0-9][0-9][0-9][0-9]{suffix}")) + [self.dir / f"{stem}{suffix}"]:
            if old.exists():
                old.unlink()                      # nao deixa shards velhos misturados
        self.files: list[dict] = []
        self._f = None
        self._cur: dict | None = None
        self._h = hashlib.sha256()
        self._hs = hashlib.sha256()
        self.total = 0

    def _open(self):
        idx = len(self.files)
        name = f"{self.stem}{self.suffix}" if self.size <= 0 else f"{self.stem}-{idx:05d}{self.suffix}"
        self._cur = {"path": str(self.dir / name), "records": 0}
        self.files.append(self._cur)
        self._f = open(self.dir / name, "w", encoding="utf-8", newline="\n")
        self._hs = hashlib.sha256()

    def _close_cur(self):
        if self._f is not None:
            self._f.close()
            self._cur["sha256"] = self._hs.hexdigest()
            self._f = None

    def write(self, obj_or_line) -> None:
        line = obj_or_line if isinstance(obj_or_line, str) else json.dumps(obj_or_line, ensure_ascii=False, separators=(",", ":"))
        if self._f is None or (self.size > 0 and self._cur["records"] >= self.size):
            self._close_cur()
            self._open()
        data = (line + "\n").encode("utf-8")
        self._f.write(line + "\n")
        self._hs.update(data)
        self._h.update(line.encode("utf-8"))
        self._cur["records"] += 1
        self.total += 1

    def close(self) -> dict:
        self._close_cur()
        return {"files": self.files, "records": self.total, "stream_sha256": self._h.hexdigest()}


class HashReservoir:
    """Amostra uniforme e DETERMINISTICA de no maximo `max_n` itens de um fluxo (os de menor hash de `salt:key`).
    Memoria O(max_n); o resultado nao depende da ordem de chegada (so do conjunto de chaves)."""

    def __init__(self, max_n: int, salt: str = ""):
        self.max_n, self.salt = int(max_n), salt
        self._buf: list = []
        self.seen = 0

    def add(self, key: str, item) -> None:
        h = int.from_bytes(hashlib.blake2b(f"{self.salt}:{key}".encode("utf-8"), digest_size=8).digest(), "big")
        self.seen += 1
        self._buf.append((h, key, item))
        if self.max_n > 0 and len(self._buf) > 2 * self.max_n:
            self._buf.sort(key=lambda t: (t[0], t[1]))
            del self._buf[self.max_n:]

    def items(self) -> list:
        self._buf.sort(key=lambda t: (t[0], t[1]))
        if self.max_n > 0:
            del self._buf[self.max_n:]
        return [t[2] for t in self._buf]
