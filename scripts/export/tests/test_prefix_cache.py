"""Testes do calculo de prefixo comum, das estatisticas de TTFT/mistura e da equivalencia greedy do prototipo de cache de prefixo.

Puros (sem onnxruntime-genai / tokenizers): importam so as funcoes do topo de `prefix_cache_proto.py`. Um teste opcional (marcado) confere o
corte estrutural do cabecalho no golden do contrato quando `tokenizers` esta disponivel.
  env/.venv/Scripts/python.exe -m pytest scripts/export/tests/test_prefix_cache.py -q -p no:cacheprovider
"""
from __future__ import annotations

import random

import pytest

import prefix_cache_proto as pc


def brute_nn(seqs):
    out = []
    for i, a in enumerate(seqs):
        out.append(max((pc.common_prefix_len(a, b) for j, b in enumerate(seqs) if j != i), default=0))
    return out


def test_common_prefix_len_basic():
    assert pc.common_prefix_len([], []) == 0
    assert pc.common_prefix_len([1, 2, 3], [1, 2, 3]) == 3
    assert pc.common_prefix_len([1, 2, 3], [1, 2]) == 2
    assert pc.common_prefix_len([1, 2, 3], [9, 2, 3]) == 0
    assert pc.common_prefix_len((5, 6, 7, 8), [5, 6, 0, 8]) == 2


def test_group_lcp():
    assert pc.group_lcp([]) == 0
    assert pc.group_lcp([[1, 2, 3]]) == 3
    assert pc.group_lcp([[1, 2, 3, 4], [1, 2, 9], [1, 2, 3]]) == 2
    assert pc.group_lcp([[1, 2], [3, 4]]) == 0


def test_nn_lcp_matches_bruteforce_and_keeps_order():
    rng = random.Random(7)
    for _ in range(50):
        seqs = [[rng.randint(0, 3) for _ in range(rng.randint(1, 8))] for _ in range(rng.randint(1, 12))]
        assert pc.nn_lcp(seqs) == brute_nn(seqs)


def test_nn_lcp_edge_cases():
    assert pc.nn_lcp([]) == []
    assert pc.nn_lcp([[1, 2, 3]]) == [0]
    assert pc.nn_lcp([[1, 2, 3], [1, 2, 3]]) == [3, 3]
    assert pc.nn_lcp([[1, 2], [7, 8], [1, 2, 3]]) == [2, 0, 2]


def test_quantile_and_stats():
    assert pc.quantile([1, 2, 3, 4, 5], 0.5) == 3
    assert pc.quantile(list(range(101)), 0.95) == 95
    s = pc.ttft_stats([100.0, 200.0, 300.0])
    assert s["mean"] == 200.0 and s["median"] == 200.0 and s["n"] == 3


def test_mixture_stats_limits():
    hit, miss = [100.0] * 50, [400.0] * 50
    assert pc.mixture_stats(hit, miss, 1.0)["mean"] == 100.0
    assert pc.mixture_stats(hit, miss, 0.0)["mean"] == 400.0
    m = pc.mixture_stats(hit, miss, 0.9)
    assert m["mean"] == pytest.approx(130.0, abs=0.1)
    assert m["p95"] == 400.0          # 10% de erros > 5%: o p95 e do erro
    assert pc.mixture_stats(hit, miss, 0.97)["p95"] == 100.0   # so 3% de erros: o p95 continua no acerto


def test_weighted_quantile():
    assert pc.weighted_quantile([(1.0, 1), (2.0, 1), (3.0, 2)], 0.5) == 2.0
    assert pc.weighted_quantile([(1.0, 1), (2.0, 1), (3.0, 2)], 0.9) == 3.0


def test_greedy_equivalence():
    base = [[1, 2, 3], [4, 5], [6], [7, 8]]
    other = [[1, 2, 3], [4, 9], [6], [7, 8, 1]]
    e = pc.greedy_equivalence(base, other)
    assert e["n"] == 4 and e["identical"] == 2 and e["identical_pct"] == 50.0
    assert [d["index"] for d in e["diffs"]] == [1, 3]
    assert e["diffs"][0]["first_diff"] == 1 and e["diffs"][1]["first_diff"] == 2


def test_greedy_equivalence_all_identical():
    e = pc.greedy_equivalence([[1], [2, 3]], [[1], [2, 3]])
    assert e["identical_pct"] == 100.0 and e["diffs"] == []


def test_cache_cut_always_leaves_a_token():
    assert pc.cache_cut(60, 270) == 60
    assert pc.cache_cut(300, 270) == 269              # nunca todo o prompt: o logit da ultima posicao exige >= 1 token novo
    assert pc.cache_cut(60, 270, available=40) == 40    # KV valido menor que o corte (miss parcial)
    assert pc.cache_cut(60, 1) == 0
    assert pc.cache_cut(0, 10) == 0


def test_prefix_cache_reuse_model():
    """Modelo do cache: o KV valido apos um pedido e o prompt dele; o proximo reaproveita min(corte, LCP, len-1)."""
    kv = [1, 2, 3, 4, 5, 6]
    new = [1, 2, 3, 4, 9, 9, 9]
    assert pc.cache_cut(10 ** 6, len(new), pc.common_prefix_len(kv, new)) == 4        # LCP completo (politica C)
    assert pc.cache_cut(2, len(new), pc.common_prefix_len(kv, new)) == 2              # politica com teto (A/B)
    assert pc.cache_cut(5, len(kv), pc.common_prefix_len(kv, kv)) == 5                # prompt repetido: ainda 1 token novo


def test_trunc_row_limits_lists_only():
    row = {"prefix": "x", "suffix": "y", "editor": {"known_names": list("abcdef"), "result_fields": list("12345"), "recent_commands": ["r1", "r2", "r3"],
                                                     "input_panel": "p", "target": {"collection": "c"}}}
    t = pc.trunc_row(row, {"recent_commands": 1, "result_fields": 2, "known_names": 4})
    assert t["editor"]["recent_commands"] == ["r1"] and t["editor"]["result_fields"] == ["1", "2"] and t["editor"]["known_names"] == list("abcd")
    assert t["editor"]["input_panel"] == "p" and row["editor"]["recent_commands"] == ["r1", "r2", "r3"]   # original intacto
    assert pc.trunc_row({"prefix": "x"}, {"recent_commands": 0}) == {"prefix": "x"}


def test_header_bounds_on_contract_text():
    rc = pytest.importorskip("render_contract", reason="scripts/data fora do sys.path")
    ctx = rc.build_context("Mongo Console JavaScript", ["a", "b"], ["f1", "f2"], "", ["db.a.find({})"])
    before = rc.model_prefix("db.a.find(", ctx)
    b = pc.header_bounds(before, rc)
    assert before[: b["lang"]].endswith("KNOWN NAMES:")
    assert before[b["lang"]: b["names"]].startswith(" a, b")
    assert before[b["names"]: b["fields"]].startswith("RESULT FIELDS: f1, f2")
    assert before[b["fields"]: b["ctx"]].startswith("RECENT COMMAND: db.a.find({})")
    assert before[b["ctx"]: b["tail"]].endswith("*/\n")
    assert before[b["tail"]:] == "db.a.find("
    assert b["global"] < b["lang"] < b["names"] <= b["fields"] <= b["ctx"] < b["tail"]
    assert pc.header_bounds("db.a.find(", rc) is None          # sem cabecalho (context_kind none): nada a cachear
