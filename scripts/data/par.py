#!/usr/bin/env python3
"""Mapa ordenado em processos (spawn) com janela limitada: saida DETERMINISTICA independente do numero de workers.

  ordered_map(fn, chunks, workers, initializer, initargs, window)

`chunks` e um iteravel de blocos (listas); `fn(bloco) -> resultado` roda em workers (cada um com o estado criado por
`initializer(*initargs)`); os resultados saem na ordem dos blocos. Com `workers <= 1` roda no proprio processo (mesmo codigo).
A janela limita blocos em voo (memoria constante, nada de materializar o fluxo inteiro).
"""
from __future__ import annotations

import multiprocessing
import os
from collections import deque


def default_workers(cap: int = 10, reserve: int = 2) -> int:
    return max(1, min(cap, (os.cpu_count() or 4) - reserve))


def chunked(it, size: int):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def ordered_map(fn, chunks, workers: int = 1, initializer=None, initargs=(), window: int | None = None):
    if workers <= 1:
        if initializer is not None:
            initializer(*initargs)
        for c in chunks:
            yield fn(c)
        return
    window = window or workers * 3
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(workers, initializer, initargs) as pool:
        q: deque = deque()
        for c in chunks:
            q.append(pool.apply_async(fn, (c,)))
            if len(q) >= window:
                yield q.popleft().get()
        while q:
            yield q.popleft().get()
