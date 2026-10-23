#!/usr/bin/env python3
"""Metricas de diversidade/estrutura de textos de consultas (mesma funcao para o corpus de gramatica v2c e para as consultas LLM)."""
from __future__ import annotations

import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path

import jsscan
from common import ROOT, shape_key

OP_RE = re.compile(r"\$[A-Za-z][A-Za-z0-9]*")


def norm_tokens(text: str) -> list[str]:
    """tokens do codigo com literais -> N/S (numeros, strings que nao sao chaves/campos), comentarios removidos."""
    toks, _ = jsscan.tokenize(text)
    out = []
    for i, (k, a, b) in enumerate(toks):
        s = text[a:b]
        if k == "num":
            out.append("N")
        elif k == "str":
            inner = s[1:-1]
            nxt = text[toks[i + 1][1]:toks[i + 1][2]] if i + 1 < len(toks) else ""
            out.append(inner if (nxt == ":" or inner.startswith("$")) else "S")
        elif k in ("regex", "tpl"):
            out.append("S")
        else:
            out.append(s)
    return out


def depth_of(masked: str) -> int:
    d = m = 0
    for ch in masked:
        if ch in "([{":
            d += 1
            m = max(m, d)
        elif ch in ")]}":
            d -= 1
    return m


def metrics(texts: list[str]) -> dict:
    n = len(texts)
    ops_all, stages_all, opsets = Counter(), Counter(), set()
    g3, g4, t3, t4 = set(), set(), 0, 0
    lens, lines, depths, nstages, per_op = [], [], [], [], []
    flags = Counter({k: 0 for k in ("multiline", "comments", "multi_statement", "variables", "functions", "sibling_or_getcollection", "aggregate", "find_family", "bson_ctors", "crlf")})
    shapes = set()
    for t in texts:
        sc = jsscan.scan(t)
        masked = sc["masked"]
        ops = set(OP_RE.findall(masked)) | set(OP_RE.findall(t))   # operadores como chave (inclusive entre aspas)
        ops_all.update(ops)
        opsets.add(frozenset(ops))
        st = [t[s:e] for s, e, r, _ in sc["spans"] if r == "stage"]
        sn = [re.match(r"\{\s*[\"']?(\$\w+)", x).group(1) for x in st if re.match(r"\{\s*[\"']?(\$\w+)", x)]
        stages_all.update(set(sn))
        nstages.append(len(sn))
        per_op.append(len(ops))
        tk = norm_tokens(t)
        for i in range(len(tk) - 2):
            g3.add(tuple(tk[i:i + 3]))
            t3 += 1
        for i in range(len(tk) - 3):
            g4.add(tuple(tk[i:i + 4]))
            t4 += 1
        lens.append(len(t))
        nl = t.count("\n") + 1
        lines.append(nl)
        depths.append(depth_of(masked))
        flags["multiline"] += nl > 1
        flags["comments"] += bool(sc["comments"])
        flags["multi_statement"] += sum(1 for s in sc["spans"] if s[2] == "stmt") > 1
        flags["variables"] += bool(re.search(r"\b(const|let|var)\s+\w+\s*=", masked))
        flags["functions"] += bool(re.search(r"=>|\bfunction\b", masked))
        flags["sibling_or_getcollection"] += bool(re.search(r"getSiblingDB|getCollection|db\s*\[|ConnectionPool", t))
        flags["aggregate"] += "aggregate" in masked
        flags["find_family"] += bool(re.search(r"\b(find|findOne)\b", masked))
        flags["bson_ctors"] += bool(re.search(r"\b(ObjectId|ISODate|NumberDecimal|NumberLong|UUID|new Date)\b", masked))
        flags["crlf"] += "\r\n" in t
        shapes.add(shape_key(t))
    q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * (len(xs) - 1)))] if xs else 0  # noqa: E731
    return {
        "n": n,
        "distinct_operators": len(ops_all), "operators_per_query_mean": round(statistics.mean(per_op), 2) if per_op else 0,
        "distinct_stages": len(stages_all), "stages_per_pipeline_mean": round(statistics.mean([x for x in nstages if x] or [0]), 2),
        "distinct_operator_sets_ratio": round(len(opsets) / max(1, n), 3),
        "distinct_3gram": len(g3), "distinct_4gram": len(g4), "distinct_4gram_ratio": round(len(g4) / max(1, t4), 3),
        "distinct_3gram_ratio": round(len(g3) / max(1, t3), 3),
        "distinct_shape_ratio": round(len(shapes) / max(1, n), 3),
        "chars_mean": round(statistics.mean(lens), 1) if lens else 0, "chars_p50": q(lens, .5), "chars_p90": q(lens, .9),
        "lines_mean": round(statistics.mean(lines), 2) if lines else 0, "depth_mean": round(statistics.mean(depths), 2) if depths else 0,
        "depth_p90": q(depths, .9), "depth_max": max(depths) if depths else 0,
        **{f"frac_{k}": round(v / max(1, n), 3) for k, v in sorted(flags.items())},
        "top_operators": [o for o, _ in ops_all.most_common(25)], "stage_set": sorted(stages_all),
    }


def corpus_texts(n: int, seed: int = 7, categories=None, max_lines: int = 400000) -> list[str]:
    """Textos de snippets UNICOS do corpus v2c (split train) lidos dos shards do FIM validado (do ultimo para o primeiro, ate juntar 6*n
    candidatos); amostra deterministica de `n`."""
    seen, rows = set(), []
    shards = sorted((ROOT / "data" / "validated" / "v2c").glob("fim-*.jsonl"), reverse=True)
    lines_read = 0
    for p in shards:
        with open(p, encoding="utf-8") as f:
            for line in f:
                lines_read += 1
                if lines_read > max_lines or len(rows) >= 6 * n:
                    break
                if '"split":"train"' not in line:
                    continue
                r = json.loads(line)
                if r["snippet_id"] in seen:
                    continue
                seen.add(r["snippet_id"])
                if categories and r.get("category") not in categories:
                    continue
                rows.append((r["snippet_id"], r["prefix"] + r["middle"] + r["suffix"]))
        if len(rows) >= 6 * n or lines_read > max_lines:
            break
    rng = random.Random(seed)
    rng.shuffle(rows)
    return [t for _, t in rows[:n]]
