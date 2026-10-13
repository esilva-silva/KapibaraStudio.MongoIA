"""Estatistica da avaliacao (Fase 06 §7). Somente biblioteca padrao (sem scipy/numpy).

- mcnemar_exact: teste exato bicaudal sobre pares discordantes (teste do sinal).
- paired_bootstrap: IC95% do delta medio pareado, 10.000 reamostragens, seed fixa.
- p95_bootstrap: p95 (nearest-rank) com IC por bootstrap.
- holm_bonferroni: p ajustados (monotonos).
- is_superior / is_noninferior: regras de decisao (§7).
"""
from __future__ import annotations

import math
import random
from fractions import Fraction
from typing import Iterable, Sequence

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20261009
ALPHA = 0.05


def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict:
    """A = referencia/base, B = candidato. better = B acerta e A erra."""
    if len(a_correct) != len(b_correct):
        raise ValueError("vetores de tamanhos diferentes (alinhe pelos mesmos exemplos)")
    better = worse = equal = 0
    for a, b in zip(a_correct, b_correct):
        a, b = bool(a), bool(b)
        if a == b:
            equal += 1
        elif b:
            better += 1
        else:
            worse += 1
    return {"better": better, "worse": worse, "equal": equal, "n": better + worse + equal,
            "discordant": better + worse, "p": mcnemar_p(better, worse)}


def mcnemar_p(better: int, worse: int) -> float:
    """p bicaudal exato: 2*P(X <= min(b,w)), X~Binomial(b+w, 0.5), limitado a 1."""
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    p = Fraction(2 * tail, 2 ** n)
    return float(min(p, Fraction(1)))


def percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Percentil com interpolacao linear (q em 0..100); entrada ja ordenada."""
    if not sorted_vals:
        raise ValueError("vazio")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo))


def nearest_rank(values: Iterable[float], q: float) -> float:
    """Percentil nearest-rank (mesma convencao do KapiLab)."""
    s = sorted(values)
    if not s:
        raise ValueError("vazio")
    rank = max(1, math.ceil(q / 100.0 * len(s)))
    return float(s[rank - 1])


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_resamples: int = DEFAULT_RESAMPLES,
                     seed: int = DEFAULT_SEED, ci: float = 0.95) -> dict:
    """Delta medio pareado (B - A) com IC percentil por bootstrap dos pares."""
    if len(a) != len(b):
        raise ValueError("vetores de tamanhos diferentes")
    if not a:
        raise ValueError("vazio")
    diffs = [float(y) - float(x) for x, y in zip(a, b)]
    n = len(diffs)
    mean = sum(diffs) / n
    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        means.append(sum(rng.choices(diffs, k=n)) / n)
    means.sort()
    lo_q = (1 - ci) / 2 * 100
    return {"n": n, "mean_a": sum(map(float, a)) / n, "mean_b": sum(map(float, b)) / n, "delta": mean,
            "ci_low": percentile(means, lo_q), "ci_high": percentile(means, 100 - lo_q),
            "resamples": n_resamples, "seed": seed, "ci": ci}


def p95_bootstrap(values: Sequence[float], n_resamples: int = DEFAULT_RESAMPLES,
                  seed: int = DEFAULT_SEED, ci: float = 0.95, q: float = 95.0) -> dict:
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("vazio")
    n = len(vals)
    rng = random.Random(seed)
    k = max(1, math.ceil(q / 100.0 * n)) - 1
    est = []
    for _ in range(n_resamples):
        est.append(sorted(rng.choices(vals, k=n))[k])
    est.sort()
    lo_q = (1 - ci) / 2 * 100
    return {"n": n, "p95": nearest_rank(vals, q), "ci_low": percentile(est, lo_q),
            "ci_high": percentile(est, 100 - lo_q), "resamples": n_resamples, "seed": seed}


def holm_bonferroni(pvalues: Sequence[float]) -> list[float]:
    """p ajustados de Holm-Bonferroni, na ordem de entrada."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adj[i] = running
    return adj


def is_superior(p: float | None = None, ci_low: float | None = None, alpha: float = ALPHA) -> bool:
    """§7: p < alpha (McNemar) OU IC95% do delta inteiramente > 0."""
    return (p is not None and p < alpha) or (ci_low is not None and ci_low > 0)


def is_noninferior(ci_low: float | None, margin: float | None) -> bool:
    """§7: limite inferior do IC95% do delta > -margem (estrito)."""
    if ci_low is None or margin is None:
        return False
    return ci_low > -margin
