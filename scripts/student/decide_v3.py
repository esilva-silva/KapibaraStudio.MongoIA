#!/usr/bin/env python3
"""Decisao do retreino v3 (T0 Nano / T1 Small): aplica os criterios do README_job a partir das medidas da homologacao v1 e do val v2c.

  python -I scripts/student/decide_v3.py --tier t0        # le reports/datasets/homolog-v3-eval-t0.json + Repos/.../reports/eval/{sft_distill_v1,sft_v3}/summary.json
  -> reports/student/v3_decision_t0.{json,md}; se o veredito for GO cria reports/student/v3_gate_t0.go (pre-requisito do J21: sem ele a fila NAO inicia o T1).

Criterios (todos sobre os MESMOS itens; "atual" = sft_distill_v1 do mesmo tier):
  C1 meta       campos inexistentes em `full` (homologacao): v3 < atual (T0 15,35%; T1 9,19%); "ideal" = <= base + 1 pp (~7-8%).
  C2 confirmada Delta pareado de `no_nonexistent_field` no grupo `full` (IC95 bootstrap pareado) com limite inferior > 0.
  C3 guarda     FIM acc (normalized_match) e exec Ã· teto na homologacao: perda <= 1,0 pp contra o atual (estimativa pontual; o IC e registrado).
  C4 val v2c    (mesmos 2.000 ids) FIM acc e exec Ã· teto: perda <= 2,0 pp (informativo-duro: acima disso e REJEICAO); Generic-JS: queda <= 3 pp vs zero-shot (G2.5).
Veredito: GO (C1+C2+C3+C4) | INCONCLUSIVO (C1+C3+C4 sem C2) | REJEITADO (C3 ou C4 violados) | SEM_MELHORA (C1 falha).
Regra: so GO libera a GPU para o proximo passo (T1) e troca o modelo publicado; qualquer outro veredito = NAO gastar mais GPU neste retreino, registrar e manter o atual.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPOS = {"t0": "KapiCoder-Mongo-Nano", "t1": "KapiCoder-Mongo-Small"}
BASE_KEY = {"t0": "base_0.5b", "t1": "base_1.5b"}
BASE_RUN = {"t0": "base-qwen2.5-coder-0.5b", "t1": "base-qwen2.5-coder-1.5b"}
CURRENT_FULL = {"t0": 0.1535, "t1": 0.0919}          # referencias do relatorio v1 (so informativo; a comparacao usa o JSON medido)
HOM_LOSS_PP, VAL_LOSS_PP, GENERIC_DROP_PP, IDEAL_MARGIN_PP = 1.0, 2.0, 3.0, 1.0


def _pp(x) -> float | None:
    return None if x is None else 100.0 * x


def _metric(blocks: list[dict], name: str) -> dict | None:
    return next((m for m in blocks if m.get("metric") == name), None)


def decide(tier: str, hom: dict, val: dict, generic: dict) -> dict:
    """hom = homolog-v3-eval-<tier>.json; val = {"current": summary, "new": summary}; generic = {"new": summary, "base": summary|None}."""
    cur, new, base = f"{tier}_distill", f"{tier}_v3", BASE_KEY[tier]
    ms = hom["models"]
    for k in (cur, new):
        if k not in ms:
            raise SystemExit(f"homologacao sem o modelo {k} (rode eval_homolog run/report com --names {base} {cur} {new})")
    def required(block, key, *, positive=False):
        value = block.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (positive and value == 0) or (key != "exec_over_ceiling" and value > 1):
            raise ValueError(f"metrica obrigatoria ausente/invalida: {key}")
        return value

    for name in (cur, new, base):
        if name not in ms:
            raise ValueError(f"modelo obrigatorio ausente: {name}")
        required(ms[name].get("overall", {}), "normalized_match")
        required(ms[name].get("overall", {}), "exec_over_ceiling")
        required(ms[name].get("by_context_kind", {}).get("full", {}), "nonexistent_fields_rate")
    for key in ("current", "new"):
        required((val.get(key) or {}).get("headline", {}), "normalized_match")
        required((val.get(key) or {}).get("headline", {}), "exec_valid")
        required((val.get(key) or {}).get("reference_ceiling", {}), "exec_valid", positive=True)
    for key in ("base", "new"):
        required((generic.get(key) or {}).get("headline", {}), "normalized_match")
    f_cur = ms[cur]["by_context_kind"]["full"]["nonexistent_fields_rate"]
    f_new = ms[new]["by_context_kind"]["full"]["nonexistent_fields_rate"]
    f_base = ms[base]["by_context_kind"]["full"]["nonexistent_fields_rate"] if base in ms else None
    pair = (hom.get("pairs") or {}).get(f"{new}_vs_{cur}") or {}
    full_blk = (((pair.get("context_kind") or {}).get("groups") or {}).get("full") or {}).get("metrics") or []
    nn = _metric(full_blk, "no_nonexistent_field")
    ov = (pair.get("context_kind") or {}).get("overall") or []
    d_fim, d_exec = _metric(ov, "normalized_match"), _metric(ov, "exec_valid")
    fim_loss = _pp(ms[cur]["overall"]["normalized_match"]) - _pp(ms[new]["overall"]["normalized_match"])
    eoc_cur, eoc_new = ms[cur]["overall"].get("exec_over_ceiling"), ms[new]["overall"].get("exec_over_ceiling")
    eoc_loss = None if eoc_cur is None or eoc_new is None else _pp(eoc_cur) - _pp(eoc_new)
    c1 = f_new < f_cur
    ideal = f_base is not None and f_new <= f_base + IDEAL_MARGIN_PP / 100.0
    c2 = bool(nn and all(isinstance(nn.get(key), (int, float)) and not isinstance(nn[key], bool) and math.isfinite(nn[key]) for key in ("delta", "ci_low", "ci_high"))
              and 0 < nn["ci_low"] <= nn["delta"] <= nn["ci_high"])
    c3 = fim_loss <= HOM_LOSS_PP + 1e-9 and (eoc_loss is not None and eoc_loss <= HOM_LOSS_PP + 1e-9)
    vh_cur, vh_new = val["current"]["headline"], val["new"]["headline"]
    ceil_c, ceil_n = (val["current"].get("reference_ceiling") or {}).get("exec_valid"), (val["new"].get("reference_ceiling") or {}).get("exec_valid")
    v_fim_loss = _pp(vh_cur["normalized_match"]) - _pp(vh_new["normalized_match"])
    v_eoc_loss = None
    if ceil_c and ceil_n and vh_cur.get("exec_valid") is not None and vh_new.get("exec_valid") is not None:
        v_eoc_loss = _pp(vh_cur["exec_valid"] / ceil_c) - _pp(vh_new["exec_valid"] / ceil_n)
    gen_drop = None
    if generic.get("new") and generic.get("base"):
        gen_drop = _pp(generic["base"]["headline"]["normalized_match"]) - _pp(generic["new"]["headline"]["normalized_match"])
    gen_vs_cur = None
    if generic.get("new") and generic.get("current"):
        gen_vs_cur = _pp(generic["current"]["headline"]["normalized_match"]) - _pp(generic["new"]["headline"]["normalized_match"])
    gen_ok = gen_drop is not None and gen_drop <= GENERIC_DROP_PP + 1e-9
    c4 = v_fim_loss <= VAL_LOSS_PP + 1e-9 and (v_eoc_loss is not None and v_eoc_loss <= VAL_LOSS_PP + 1e-9) and gen_ok
    if not (c3 and c4):
        verdict = "REJEITADO"
    elif not c1:
        verdict = "SEM_MELHORA"
    elif not c2:
        verdict = "INCONCLUSIVO"
    else:
        verdict = "GO"
    return {"tier": tier, "verdict": verdict, "go": verdict == "GO",
            "checks": {"C1_full_fields_lower": c1, "C2_improvement_confirmed_ci95": c2, "C3_hom_guardrail_loss_le_1pp": c3, "C4_val_and_generic_guard": c4},
            "ideal_met": ideal,
            "numbers": {"comparisons": {
                        "hom_fim_acc": {"current": ms[cur]["overall"]["normalized_match"], "v3": ms[new]["overall"]["normalized_match"], "delta_pp": round(-fim_loss, 3)},
                        "hom_exec_over_ceiling": {"current": eoc_cur, "v3": eoc_new, "delta_pp": round(-eoc_loss, 3)},
                        "generic_js": {"base": generic["base"]["headline"]["normalized_match"], "v3": generic["new"]["headline"]["normalized_match"], "delta_vs_base_pp": round(-gen_drop, 3)}},
                        "full_nonexistent_pct": {"current": _pp(f_cur), "v3": _pp(f_new), "base": _pp(f_base), "reference_v1_report": _pp(CURRENT_FULL[tier])},
                        "paired_no_nonexistent_field_full": nn and {k: nn.get(k) for k in ("delta", "ci_low", "ci_high", "p_holm", "p")},
                        "hom_fim_acc_loss_pp": round(fim_loss, 3), "hom_exec_over_ceiling_loss_pp": None if eoc_loss is None else round(eoc_loss, 3),
                        "hom_paired_delta_fim": d_fim and {k: d_fim.get(k) for k in ("delta", "ci_low", "ci_high")},
                        "hom_paired_delta_exec": d_exec and {k: d_exec.get(k) for k in ("delta", "ci_low", "ci_high")},
                        "val_fim_acc_loss_pp": round(v_fim_loss, 3), "val_exec_over_ceiling_loss_pp": None if v_eoc_loss is None else round(v_eoc_loss, 3),
                        "generic_js_drop_vs_zero_shot_pp": None if gen_drop is None else round(gen_drop, 3),
                        "generic_js_loss_vs_current_pp": None if gen_vs_cur is None else round(gen_vs_cur, 3)},
            "thresholds": {"hom_loss_pp": HOM_LOSS_PP, "val_loss_pp": VAL_LOSS_PP, "generic_drop_pp": GENERIC_DROP_PP, "ideal_margin_pp": IDEAL_MARGIN_PP},
            "action": {"GO": "promover o modelo v3 (merged/sft_v3); liberar o passo seguinte (J21 no T0->T1)",
                       "INCONCLUSIVO": "NAO gastar mais GPU; registrar; manter o atual (a melhora nao e distinguivel de ruido)",
                       "SEM_MELHORA": "NAO gastar mais GPU neste retreino; registrar o resultado negativo; manter o atual",
                       "REJEITADO": "NAO promover (perda acima da tolerancia); NAO gastar mais GPU; registrar; manter o atual"}[verdict]}


def render_md(d: dict) -> str:
    n = d["numbers"]
    f = n["full_nonexistent_pct"]
    L = [f"# Decisao do retreino v3 - {d['tier'].upper()}", "", f"**Veredito: {d['verdict']}** - {d['action']}.", "",
         f"- Campos inexistentes em `full` (homologacao): atual {f['current']:.2f}% -> v3 {f['v3']:.2f}% (base {f['base']:.2f}%; relatorio v1: {f['reference_v1_report']:.2f}%); "
         f"meta ideal (<= base + 1 pp): {'atingida' if d['ideal_met'] else 'nao atingida'}.",
         f"- Delta pareado `no_nonexistent_field` (full): {n['paired_no_nonexistent_field_full']}.",
         f"- Guarda na homologacao: perda de FIM acc {n['hom_fim_acc_loss_pp']} pp, de exec Ã· teto {n['hom_exec_over_ceiling_loss_pp']} pp (tolerancia {d['thresholds']['hom_loss_pp']} pp).",
         f"- val v2c (mesmos 2.000 ids): perda de FIM acc {n['val_fim_acc_loss_pp']} pp, de exec Ã· teto {n['val_exec_over_ceiling_loss_pp']} pp; Generic-JS vs zero-shot {n['generic_js_drop_vs_zero_shot_pp']} pp.",
         "", "| criterio | resultado |", "|---|---|"]
    L += [f"| {k} | {'ok' if v else 'FALHA'} |" for k, v in d["checks"].items()]
    L += ["", "## Comparacoes sem ambiguidade de sinal", "", "Delta = novo - referencia; positivo significa ganho nas metricas abaixo.", ""]
    for name, row in n["comparisons"].items():
        ref = row.get("current", row.get("base"))
        delta = row.get("delta_pp", row.get("delta_vs_base_pp"))
        L.append(f"- {name}: {ref:.4f} -> {row['v3']:.4f}; delta {delta:+.3f} pp.")
    L += ["", "Os campos legados *_loss_pp usam referencia - novo: valor negativo significa ganho, nao regressao."]
    return "\n".join(L) + "\n"


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=["t0", "t1"], required=True)
    ap.add_argument("--hom-json")
    ap.add_argument("--out-dir", default=str(ROOT / "reports" / "student"))
    a = ap.parse_args(argv)
    tier = a.tier
    ev = ROOT / "Repos" / REPOS[tier] / "reports" / "eval"
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"v3_gate_{tier}.go").unlink(missing_ok=True)
    hom = load(Path(a.hom_json) if a.hom_json else ROOT / "reports" / "datasets" / f"homolog-v3-eval-{tier}.json")
    val = {"current": load(ev / "sft_distill_v1" / "summary.json"), "new": load(ev / "sft_v3" / "summary.json")}
    gen = {}
    for key, name in (("new", "sft_v3"), ("current", "sft_distill_v1"), ("base", BASE_RUN[tier])):
        p = ev / f"{name}-generic-js" / "summary.json"
        gen[key] = load(p) if p.exists() else None
    d = decide(tier, hom, val, gen)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"v3_decision_{tier}.json").write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / f"v3_decision_{tier}.md").write_text(render_md(d), encoding="utf-8")
    gate = out / f"v3_gate_{tier}.go"
    if d["go"]:
        gate.write_text(json.dumps({"tier": tier, "verdict": d["verdict"]}) + "\n", encoding="utf-8")
    elif gate.exists():
        gate.unlink()
    print(render_md(d))
    return 0


if __name__ == "__main__":
    sys.exit(main())
