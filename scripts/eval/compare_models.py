"""Compara dois predictions.jsonl nos MESMOS exemplos (licao v1; Fase 06 §7).

Uso:
  python compare_models.py --a base/predictions.jsonl --b cand/predictions.jsonl
      --out-json cmp.json --out-md cmp.md [--group-by bench_group] [--min-group-n 50]
      [--margin exact_match=0.05] [--metrics-key metrics]

Metricas: booleanas em record[metrics-key] -> McNemar exato; numericas -> bootstrap pareado (IC95 do delta);
campos numericos de topo (ttft_ms, tokens_per_second) -> bootstrap (+ p95 para ttft_ms). Delta = B - A.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats  # noqa: E402

TOP_LEVEL_NUMERIC = ("ttft_ms", "tokens_per_second", "generated_tokens", "prompt_tokens")
LATENCY_FIELDS = ("ttft_ms",)
LOWER_BETTER = {"conversational", "empty", "ide_rejected", "edit_distance", "ttft_ms",
                "generated_tokens", "prompt_tokens"}


def load(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "id" not in rec:
                raise SystemExit(f"{path}:{ln}: registro sem 'id'")
            if rec["id"] in out:
                raise SystemExit(f"{path}:{ln}: id duplicado {rec['id']!r}")
            out[rec["id"]] = rec
    return out


def _isnum(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def extract(rec: dict, metrics_key: str) -> dict:
    vals = {}
    for k, v in (rec.get(metrics_key) or {}).items():
        if isinstance(v, bool) or _isnum(v):
            vals[k] = v
    for k in TOP_LEVEL_NUMERIC:
        if _isnum(rec.get(k)):
            vals[k] = rec[k]
    return vals


def compare_metric(name, pairs, n_resamples, seed, margin=None):
    """pairs: [(a, b)] ja filtrados (sem nulos)."""
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    lower = name in LOWER_BETTER
    if all(isinstance(x, bool) for x in a + b):
        mc = stats.mcnemar_exact(a, b)
        better, worse = mc["better"], mc["worse"]
        if lower:  # menor e melhor: inverte rotulos melhores/piores
            better, worse = worse, better
        n = mc["n"]
        res = {"metric": name, "kind": "binary", "n": n, "lower_is_better": lower,
               "rate_a": sum(a) / n, "rate_b": sum(b) / n, "delta": (sum(b) - sum(a)) / n,
               "better": better, "worse": worse, "equal": mc["equal"], "p": mc["p"]}
        bs = stats.paired_bootstrap([float(x) for x in a], [float(x) for x in b], n_resamples, seed)
        res["ci_low"], res["ci_high"] = bs["ci_low"], bs["ci_high"]
    else:
        bs = stats.paired_bootstrap(a, b, n_resamples, seed)
        res = {"metric": name, "kind": "continuous", "n": bs["n"], "lower_is_better": lower,
               "mean_a": bs["mean_a"], "mean_b": bs["mean_b"], "delta": bs["delta"],
               "ci_low": bs["ci_low"], "ci_high": bs["ci_high"]}
        if name in LATENCY_FIELDS:
            res["p95_a"] = stats.p95_bootstrap(a, n_resamples, seed)
            res["p95_b"] = stats.p95_bootstrap(b, n_resamples, seed)
    eff_low = -res["ci_high"] if lower else res["ci_low"]  # IC do ganho (positivo = B melhor)
    res["superior"] = stats.is_superior(ci_low=eff_low)
    if margin is not None:
        res["margin"] = margin
        res["noninferior"] = stats.is_noninferior(eff_low, margin)
    return res


def run(a_path, b_path, group_by="bench_group", min_group_n=50, metrics_key="metrics",
        margins=None, n_resamples=stats.DEFAULT_RESAMPLES, seed=stats.DEFAULT_SEED):
    margins = margins or {}
    A, B = load(Path(a_path)), load(Path(b_path))
    ids = [i for i in A if i in B]  # so a interseccao (mesmos exemplos)
    if not ids:
        raise SystemExit("nenhum id em comum entre os dois arquivos")
    ex = {i: (extract(A[i], metrics_key), extract(B[i], metrics_key)) for i in ids}
    names = sorted({k for pa, pb in ex.values() for k in pa if k in pb})

    def block(sel_ids):
        out = []
        for m in names:
            pairs = [(ex[i][0][m], ex[i][1][m]) for i in sel_ids if m in ex[i][0] and m in ex[i][1]]
            if pairs:
                out.append(compare_metric(m, pairs, n_resamples, seed, margins.get(m)))
        return out

    groups: dict[str, list[str]] = {}
    for i in ids:
        groups.setdefault(str(A[i].get(group_by)), []).append(i)
    result = {
        "schema": "kapi-compare-v1",
        "a": str(a_path), "b": str(b_path), "n_aligned": len(ids),
        "only_in_a": len(A) - len(ids), "only_in_b": len(B) - len(ids),
        "group_by": group_by, "min_group_n": min_group_n, "metrics_key": metrics_key,
        "bootstrap": {"resamples": n_resamples, "seed": seed, "ci": 0.95},
        "delta_definition": "B - A (positivo = B maior; veja lower_is_better)",
        "overall": block(ids), "groups": {}, "groups_skipped": {},
    }
    for g, gids in sorted(groups.items()):
        if len(gids) >= min_group_n:
            result["groups"][g] = {"n": len(gids), "metrics": block(gids)}
        else:
            result["groups_skipped"][g] = len(gids)
    tests = [r for r in result["overall"] if r["kind"] == "binary"]
    for g in result["groups"].values():
        tests += [r for r in g["metrics"] if r["kind"] == "binary"]
    for r, adj in zip(tests, stats.holm_bonferroni([r["p"] for r in tests])):
        r["p_holm"] = adj
    return result


def _fmt(r):
    if r["kind"] == "binary":
        core = f"{r['rate_a']:.4f} | {r['rate_b']:.4f}"
        extra = f"{r['better']}/{r['worse']}/{r['equal']} | {r['p']:.4g} | {r.get('p_holm', float('nan')):.4g}"
    else:
        core = f"{r['mean_a']:.4g} | {r['mean_b']:.4g}"
        extra = "- | - | -"
    return (f"| {r['metric']} | {r['n']} | {core} | {r['delta']:+.4f} | "
            f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {extra} |")


def to_markdown(res):
    head = ("| metrica | n | A | B | delta | IC95 | melhor/pior/igual | p | p Holm |\n"
            "|---|---|---|---|---|---|---|---|---|")
    L = ["# Comparacao A x B", "",
         f"- A: `{res['a']}`", f"- B: `{res['b']}`",
         f"- exemplos alinhados: {res['n_aligned']} (so em A: {res['only_in_a']}, so em B: {res['only_in_b']})",
         f"- delta = B - A; bootstrap {res['bootstrap']['resamples']} reamostragens, seed {res['bootstrap']['seed']}",
         "", "## Geral", "", head]
    L += [_fmt(r) for r in res["overall"]]
    for g, blk in res["groups"].items():
        L += ["", f"## {res['group_by']} = {g} (n={blk['n']})", "", head] + [_fmt(r) for r in blk["metrics"]]
    if res["groups_skipped"]:
        L += ["", f"Grupos com n < {res['min_group_n']} omitidos: " +
              ", ".join(f"{g} (n={n})" for g, n in res["groups_skipped"].items())]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="predictions.jsonl base/referencia")
    ap.add_argument("--b", required=True, help="predictions.jsonl candidato")
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    ap.add_argument("--group-by", default="bench_group")
    ap.add_argument("--min-group-n", type=int, default=50)
    ap.add_argument("--metrics-key", default="metrics")
    ap.add_argument("--margin", action="append", default=[], metavar="METRICA=MARGEM")
    ap.add_argument("--resamples", type=int, default=stats.DEFAULT_RESAMPLES)
    ap.add_argument("--seed", type=int, default=stats.DEFAULT_SEED)
    args = ap.parse_args(argv)
    margins = {k: float(v) for k, v in (m.split("=", 1) for m in args.margin)}
    res = run(args.a, args.b, args.group_by, args.min_group_n, args.metrics_key, margins,
              args.resamples, args.seed)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    md = to_markdown(res)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(md, encoding="utf-8")
    if not args.out_json and not args.out_md:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
