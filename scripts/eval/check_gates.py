"""Avalia os gates de um tier (configs/gates.yaml) e grava reports/eval/<tier>/<run>/gates.json.

Uso:
  python check_gates.py --tier T0 --gates configs/gates.yaml --measures python.json
      [--measures-kapilab kapilab.json] [--out reports/eval/T0/<run>/gates.json] [--owners python,kapilab]

Formato de --measures (python) / --measures-kapilab:
  {"run_id": "...", "env": {"used_fallback": false, "allow_gpu_parallel": false, "machine_load_pct": 1.2,
                            "genai": "0.15.2", "package_min_genai": "0.15.2"},
   "metrics": {"<id do gate> | <metric>": <numero|bool|string|{value|delta, ci_low, ci_high, p}>}}
  O arquivo do KapiLab tambem pode ser o proprio gates.json dele (lista "gates" com id/value) ou
  usar "metrics". Busca primeiro por id do gate; por `metric` so se o nome for unico no tier.

Exit: 0 tudo ok | 6 gate bloqueante reprovado | 7 sem reprovacao, mas ha bloqueante nao avaliado/nao qualificado
      | 2 erro de uso.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats  # noqa: E402

EXIT_OK, EXIT_FAIL, EXIT_UNEVAL, EXIT_USAGE = 0, 6, 7, 2
IDLE_LOAD_MAX_PCT = 5.0  # kapilab-console.md: carga de CPU/GPU antes da rodada > 5% => nao qualificada
OPS = (">=", "<=", "==", "noninferior", "superior")


# ---------------------------------------------------------------- carga

def load_gates(path: Path, tier: str) -> list[dict]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    tiers = cfg.get("tiers") or {}
    if tier not in tiers:
        raise SystemExit(f"tier {tier!r} nao existe em {path} (tiers: {', '.join(tiers)})")
    out = []
    t = tiers[tier]
    for src in (t.get("gates") or [], (cfg.get("common") or {}).get("gates") or [],
                (cfg.get("common") or {}).get("parity") or []):
        out.extend(src)
    stat = cfg.get("statistics") or {}
    alpha = ((stat.get("superiority") or {}).get("alpha")) or stats.ALPHA
    for g in out:
        g.setdefault("_alpha", alpha)
    return out


def load_measures(path: Path | None) -> dict | None:
    if path is None:
        return None
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    metrics = dict(raw.get("metrics") or {})
    for g in raw.get("gates") or []:  # gates.json do KapiLab
        if "id" in g and "value" in g:
            metrics.setdefault(g["id"], g["value"])
    env = dict(raw.get("env") or {})
    for k in ("used_fallback", "allow_gpu_parallel", "genai", "package_min_genai", "machine_load_pct",
              "machine_idle", "requested_provider", "effective_provider"):
        if k in raw and k not in env:
            env[k] = raw[k]
    if raw.get("qualified") is False:
        env.setdefault("qualified_false_reported", True)
    return {"run_id": raw.get("run_id"), "env": env, "metrics": metrics, "raw": raw,
            "kapilab_gate_entries": {g["id"]: g for g in raw.get("gates") or [] if "id" in g}}


# ---------------------------------------------------------------- qualificacao

def qualification(*envs: dict) -> list[str]:
    reasons = []
    for env in envs:
        if env.get("used_fallback") is True:
            reasons.append("used_fallback")
        if env.get("allow_gpu_parallel") is True:
            reasons.append("allow_gpu_parallel")
        if env.get("machine_idle") is False:
            reasons.append("machine_not_idle")
        load = env.get("machine_load_pct")
        if isinstance(load, (int, float)) and load > IDLE_LOAD_MAX_PCT:
            reasons.append(f"machine_not_idle(load={load}%>{IDLE_LOAD_MAX_PCT}%)")
        g, want = env.get("genai"), env.get("package_min_genai") or env.get("expected_genai")
        if g is not None and want is not None and str(g) != str(want):
            reasons.append(f"genai_mismatch({g}!={want})")
        if env.get("qualified_false_reported"):
            reasons.append("reported_unqualified_by_source")
    return sorted(set(reasons))


# ---------------------------------------------------------------- avaliacao

def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not (isinstance(x, float) and math.isnan(x))


def _scalar(v):
    if isinstance(v, dict):
        for k in ("value", "delta", "mean", "estimate"):
            if k in v:
                return v[k]
        return None
    return v


def _p_of(gate, v):
    if isinstance(v, dict):
        for k in ("p", "p_value"):
            if _num(v.get(k)):
                return float(v[k])
        return None
    if gate.get("unit") == "p" and _num(v):
        return float(v)
    return None


def _ci_low(v):
    if isinstance(v, dict):
        for k in ("ci_low", "ci95_low", "lower"):
            if _num(v.get(k)):
                return float(v[k])
    return None


def _result(gate, status, value=None, reason=None, **extra):
    r = {"id": gate["id"], "metric": gate.get("metric"), "owner": gate.get("owner", "python"),
         "op": gate.get("op"), "threshold": gate.get("threshold"), "value": value,
         "passed": {"passed": True, "failed": False}.get(status), "status": status,
         "blocking": bool(gate.get("blocking", False))}
    if gate.get("margin") is not None:
        r["margin"] = gate["margin"]
    if gate.get("group"):
        r["group"] = gate["group"]
        r["group_mode"] = gate.get("group_mode", "all")
    if reason:
        r["reason"] = reason
    r.update(extra)
    return r


def evaluate(gate, v, env, p_holm=None):
    """Retorna dict de resultado para um gate com medida `v` (ou None)."""
    op, thr = gate.get("op"), gate.get("threshold")
    if gate.get("todo"):
        return _result(gate, "unevaluated", _scalar(v), "todo: a doc nao fornece limiar numerico")
    if op not in OPS:
        return _result(gate, "unevaluated", None, f"op desconhecido: {op!r}")
    if v is None:
        return _result(gate, "unevaluated", None, "sem medida")
    if op == "noninferior":
        margin = gate.get("margin")
        if margin is None:
            return _result(gate, "unevaluated", _scalar(v), "margem nao definida")
        lo = _ci_low(v)
        if lo is None:
            return _result(gate, "unevaluated", _scalar(v), "requer ci_low do IC95% do delta")
        ok = stats.is_noninferior(lo, margin)
        return _result(gate, "passed" if ok else "failed", _scalar(v), ci_low=lo,
                       rule=f"ci_low > -{margin}")
    if thr is None:
        return _result(gate, "unevaluated", _scalar(v), "threshold nulo")
    if op == "superior":
        p = _p_of(gate, v)
        lo = _ci_low(v)
        alpha = float(thr) if gate.get("unit") == "p" else float(gate.get("alpha", gate["_alpha"]))
        if p is None and lo is None:
            return _result(gate, "unevaluated", _scalar(v), "requer p ou ci_low")
        adj = p_holm if p_holm is not None else p
        sig = stats.is_superior(p=adj, ci_low=lo, alpha=alpha)
        extra = {"p": p, "p_holm": adj, "ci_low": lo}
        if gate.get("unit") != "p":  # threshold e o ganho minimo (delta) alem da significancia
            d = _scalar(v)
            if not _num(d):
                return _result(gate, "unevaluated", None, "requer delta", **extra)
            sig = sig and d >= float(thr)
            return _result(gate, "passed" if sig else "failed", d, **extra)
        return _result(gate, "passed" if sig else "failed", p if p is not None else _scalar(v), **extra)
    # ops escalares
    val = _scalar(v)
    if thr == "requested":
        req = v.get("requested") if isinstance(v, dict) else env.get("requested_provider")
        eff = v.get("effective") if isinstance(v, dict) else val
        if req is None or eff is None:
            return _result(gate, "unevaluated", val, "provider solicitado/efetivo ausente")
        ok = str(req).lower() == str(eff).lower()
        return _result(gate, "passed" if ok else "failed", eff, requested=req)
    if isinstance(thr, bool) or isinstance(val, bool) or isinstance(val, str):
        ok = val == thr if op == "==" else None
        if ok is None:
            return _result(gate, "unevaluated", val, "op numerico com valor nao numerico")
        return _result(gate, "passed" if ok else "failed", val)
    if not _num(val):
        return _result(gate, "unevaluated", val, "valor nao numerico")
    t = float(thr)
    ok = {">=": val >= t, "<=": val <= t, "==": math.isclose(val, t, abs_tol=1e-9)}[op]
    return _result(gate, "passed" if ok else "failed", val)


def _flatten(gates):
    """Sub-gates viram entradas proprias (nao bloqueantes), pai agrega."""
    for g in gates:
        yield g
        for sg in g.get("sub_gates") or []:
            s = dict(sg)
            s.update(owner=g.get("owner", "python"), blocking=False, _parent=g["id"], _alpha=g["_alpha"])
            yield s


def run_check(tier, gates_cfg, py, kl, owners=("python", "kapilab")):
    gates = load_gates(gates_cfg, tier)
    py_m = (py or {}).get("metrics", {})
    kl_m = (kl or {}).get("metrics", {})
    active = [g for g in gates if not g.get("inactive")]
    metric_count: dict[str, int] = {}
    for g in active:
        metric_count[g.get("metric")] = metric_count.get(g.get("metric"), 0) + 1
    # prefixos de cenario com alguma medida kapilab/paridade medida
    def prefix(gid):
        return gid.rsplit(".", 1)[0]
    measured_prefix = set()

    def lookup(g):
        src = kl_m if g.get("owner") == "kapilab" else py_m
        if g["id"] in src:
            return src[g["id"]]
        if g["id"].startswith("common.parity."):
            return None
        if metric_count.get(g.get("metric"), 0) == 1 and g.get("metric") in src:
            return src[g["metric"]]
        return None

    for g in active:
        scoped = (g.get("owner") == "kapilab" and kl is not None and not g["id"].startswith("common.")) \
            or g["id"].startswith("common.parity.")
        if scoped and lookup(g) is not None:
            measured_prefix.add(prefix(g["id"]))

    env_py, env_kl = (py or {}).get("env", {}), (kl or {}).get("env", {})
    disq = qualification(env_py, env_kl)
    qualified = not disq

    # passe 1: p das superioridades para Holm
    fam = {}
    for g in _flatten(active):
        if g.get("op") == "superior" and not g.get("todo") and g.get("owner") in owners:
            p = _p_of(g, lookup(g))
            if p is not None:
                fam[g["id"]] = p
    adj = dict(zip(fam, stats.holm_bonferroni(list(fam.values())))) if fam else {}

    results = []
    for g in gates:
        if g.get("inactive"):
            results.append(_result(g, "inactive", reason="gate inativo (ignorado)"))
            continue
        owner = g.get("owner", "python")
        if owner not in owners:
            results.append(_result(g, "skipped", reason=f"owner {owner} fora de --owners"))
            continue
        if owner == "kapilab" and kl is None and not g.get("todo"):
            results.append(_result(g, "unevaluated", reason="--measures-kapilab nao fornecido"))
            continue
        scoped = (owner == "kapilab" and not g["id"].startswith("common.")) or g["id"].startswith("common.parity.")
        if scoped and prefix(g["id"]) not in measured_prefix and lookup(g) is None and not g.get("todo"):
            results.append(_result(g, "not_applicable", reason="cenario/variante sem nenhuma medida neste run"))
            continue
        subs = g.get("sub_gates") or []
        if subs:
            sub_res = []
            for sg in [s for s in _flatten([g]) if s.get("_parent")]:
                r = evaluate(sg, lookup(sg), env_kl if sg["owner"] == "kapilab" else env_py, adj.get(sg["id"]))
                r["parent"] = g["id"]
                sub_res.append(r)
            sts = [r["status"] for r in sub_res]
            st = "failed" if "failed" in sts else ("unevaluated" if "unevaluated" in sts else "passed")
            results.append(_result(g, st, None, "agregado de sub_gates"))
            results.extend(sub_res)
            continue
        r = evaluate(g, lookup(g), env_kl if owner == "kapilab" else env_py, adj.get(g["id"]))
        if owner == "kapilab" and not qualified and r["status"] in ("passed", "failed"):
            r["raw_result"] = r["status"]
            r["status"], r["passed"] = "unqualified", None
            r["reason"] = "rodada nao qualificada: " + ", ".join(disq)
        results.append(r)

    # grupos
    groups = {}
    for r in results:
        if r.get("group") and not r.get("parent"):
            groups.setdefault(r["group"], []).append(r)
    group_out = []
    in_group_ok = set()
    for gid, members in groups.items():
        live = [m for m in members if m["status"] not in ("inactive", "skipped", "not_applicable")]
        if not live:
            continue
        mode = live[0].get("group_mode", "all")
        sts = [m["status"] for m in live]
        if mode == "any":
            st = "passed" if "passed" in sts else ("failed" if all(s == "failed" for s in sts) else "unevaluated")
        else:
            st = "failed" if "failed" in sts else ("passed" if all(s == "passed" for s in sts) else "unevaluated")
        if st == "unevaluated" and "unqualified" in sts:
            st = "unqualified"
        blocking = any(m["blocking"] for m in live)
        group_out.append({"group": gid, "mode": mode, "status": st, "blocking": blocking,
                          "members": [m["id"] for m in live]})
        for m in live:
            m["group_status"] = st
    gstat = {g["group"]: g for g in group_out}

    failed_block, uneval_block = [], []
    for r in results:
        if r.get("parent"):
            continue
        if r.get("group") and r["group"] in gstat:
            continue
        if r["blocking"]:
            if r["status"] == "failed":
                failed_block.append(r["id"])
            elif r["status"] in ("unevaluated", "unqualified"):
                uneval_block.append(r["id"])
    for g in group_out:
        if g["blocking"]:
            if g["status"] == "failed":
                failed_block.append(f"group:{g['group']}")
            elif g["status"] in ("unevaluated", "unqualified"):
                uneval_block.append(f"group:{g['group']}")

    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    exit_code = EXIT_FAIL if failed_block else (EXIT_UNEVAL if uneval_block else EXIT_OK)
    run_id = (py or {}).get("run_id") or (kl or {}).get("run_id")
    doc = {"schema": "kapi-gates-v1", "tier": tier, "run_id": run_id, "owner": "python+kapilab",
           "qualified": qualified, "qualified_reasons": disq,
           "gates_file": str(gates_cfg),
           "approved": exit_code == EXIT_OK,
           "exit_code": exit_code,
           "blocking_failed": failed_block, "blocking_unevaluated": uneval_block,
           "gates": results, "groups": group_out, "summary": counts}
    return doc


def print_table(doc):
    rows = [("id", "owner", "value", "op thr", "status", "blk")]
    for r in doc["gates"]:
        if r["status"] in ("inactive", "skipped", "not_applicable"):
            continue
        v = r["value"]
        v = "-" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v))
        thr = r.get("threshold") if r.get("op") != "noninferior" else f"-{r.get('margin')}"
        rows.append((r["id"], r["owner"], v, f"{r['op']} {thr}", r["status"], "Y" if r["blocking"] else "n"))
    w = [max(len(str(x[i])) for x in rows) for i in range(6)]
    for i, row in enumerate(rows):
        print("  ".join(str(c).ljust(w[j]) for j, c in enumerate(row)))
        if i == 0:
            print("  ".join("-" * x for x in w))
    for g in doc["groups"]:
        print(f"[grupo {g['group']} ({g['mode']})] {g['status']}")
    print(f"tier={doc['tier']} qualified={doc['qualified']} {doc['qualified_reasons'] or ''} "
          f"summary={doc['summary']} exit={doc['exit_code']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", required=True)
    ap.add_argument("--gates", default="configs/gates.yaml")
    ap.add_argument("--measures", required=True, help="JSON de medidas do Python")
    ap.add_argument("--measures-kapilab")
    ap.add_argument("--owners", default="python,kapilab")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    py = load_measures(Path(args.measures))
    kl = load_measures(Path(args.measures_kapilab)) if args.measures_kapilab else None
    owners = tuple(x.strip() for x in args.owners.split(",") if x.strip())
    doc = run_check(args.tier, Path(args.gates), py, kl, owners)
    out = args.out
    if not out:
        out = f"reports/eval/{args.tier}/{doc['run_id'] or 'run'}/gates.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print_table(doc)
    print(f"-> {out}")
    return doc["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
