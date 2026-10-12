"""Métricas de autocomplete: EM, prefix match, edit distance, accepted prefix tokens, TAB e ruído conversacional."""
from __future__ import annotations

import re
from typing import Sequence

from .tab import chunks

CONVERSATIONAL = re.compile(
    r"""^\s*(?:
        ```|~~~|\#{1,6}\s+[A-Za-zÀ-ÿ]|\*\*[A-Za-zÀ-ÿ]|>\s+[A-Za-zÀ-ÿ]{2,}\s|
        (?:claro|aqui\ est[áa]|segue|certo|com\ certeza|voc[êe]\ pode|para\ (?:isso|fazer)|este\ c[óo]digo|explica[çc][ãa]o|observa[çc][ãa]o|nota:|resposta:)\b|
        (?:sure|here(?:'s|\ is|\ are)|certainly|of\ course|to\ (?:do|achieve|query|find)|this\ (?:code|query|will)|the\ following|explanation|note:|answer:|solution:|i\ (?:will|can|would))\b|
        <\|(?:user|assistant)\|>|\bassistant:|\buser:
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def exact_match(pred: str, ref: str) -> bool:
    return pred.strip() == ref.strip()


def prefix_match(pred: str, ref: str) -> bool:
    """Sugestão não vazia que é prefixo da referência (parcial correta) ou a contém por inteiro no início."""
    p, r = pred.strip(), ref.strip()
    return bool(p) and bool(r) and (r.startswith(p) or p.startswith(r))


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def edit_similarity(pred: str, ref: str, cap: int = 1024) -> tuple[int, float]:
    p, r = pred.strip()[:cap], ref.strip()[:cap]
    distance = levenshtein(p, r)
    longest = max(len(p), len(r))
    return distance, (1.0 if longest == 0 else 1.0 - distance / longest)


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def accepted_prefix_tokens(pred: str, ref: str, ref_token_ends: Sequence[int]) -> int:
    """Tokens iniciais da referência integralmente cobertos pelo prefixo comum (texto cru, sem strip).

    Usa offsets da tokenização da referência para não depender de a sugestão ter sido tokenizada
    da mesma forma (BPE pode segmentar o mesmo texto de modo diferente).
    """
    shared = common_prefix_len(pred, ref)
    return sum(1 for end in ref_token_ends if end <= shared)


def tab_acceptance(pred: str, ref: str) -> tuple[int, int]:
    """Simula TABs incrementais da IDE: aceita blocos enquanto coincidem com a referência.

    Retorna (blocos aceitos, caracteres aceitos).
    """
    position = 0
    accepted = 0
    for chunk in chunks(pred):
        if ref.startswith(chunk, position):
            position += len(chunk)
            accepted += 1
        else:
            break
    return accepted, position


def is_conversational(pred: str) -> bool:
    return bool(CONVERSATIONAL.match(pred))


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)
