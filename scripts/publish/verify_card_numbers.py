#!/usr/bin/env python3
"""Confere que os numeros dos model cards do Hugging Face batem com as fontes do projeto (sem rede, somente leitura).

  python scripts/publish/verify_card_numbers.py                  # confere nano e small (exit 1 se algo divergir)
  python scripts/publish/verify_card_numbers.py --tier nano
  python scripts/publish/verify_card_numbers.py --write          # reescreve as tabelas entre os marcadores do card a partir dos JSON
  python scripts/publish/verify_card_numbers.py --write-sources  # regenera hf/SOURCES.md (apendice interno afirmacao <-> fonte)

O que e conferido (tudo contra arquivos do repositorio; nenhuma metrica e digitada a mao no script):
  1. tabelas de metricas do card == renderizacao dos `measures.json` (e `summary.json` para a coerencia cruzada);
  2. `model-index` do front matter == as mesmas medidas, com `verified: false`;
  3. fatos de treino (exemplos, tokens, passos, horas, revisao da base) == `model.lock.json` / `run_manifest.json`;
  4. afirmacoes de prosa: cada par (trecho do card, trecho da fonte) precisa existir literalmente nos dois arquivos;
  5. frases obrigatorias (rotulo "avaliacao interna sobre dados sinteticos", rotulo da homologacao sintetica, PREVIEW);
  6. homologacao v1 (fora da distribuicao): tabelas e blocos de leitura do card == renderizacao de
     `reports/datasets/homolog-v1-eval.json` (falha se qualquer numero divergir); `model-index` da homologacao == o mesmo JSON;
     o JSON bate com `homolog-v1-manifest.json` (sha do conjunto selado, n, 28 schemas) e com as medidas do val v2c do card.
     Nunca le itens de `data/homolog`: so os agregados publicados em `reports/datasets/`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BEGIN = "<!-- BEGIN:{} -->"
END = "<!-- END:{} -->"

REQUIRED_EN = "Internal evaluation on synthetic data; no blind human-written set."
REQUIRED_PT = "avaliação interna sobre dados sintéticos; sem conjunto cego humano"
REQUIRED_HOM_EN = "Internal evaluation, synthetic out-of-distribution homologation"
REQUIRED_HOM_PT = "avaliação interna, homologação sintética fora da distribuição"
REQUIRED_PHRASES = (
    REQUIRED_EN,
    REQUIRED_PT,
    REQUIRED_HOM_EN,
    REQUIRED_HOM_PT,
    "PREVIEW / experimental",
    "### Out-of-distribution validation",
    "Do not trust field names unless the schema is in the context",
    "never executes",  # "the model never executes anything"
    "Modification notice",
    "Apache License 2.0",
)

# coluna da tabela de acuracia: (cabecalho, chave em measures.metrics, formato)
ACC_COLUMNS = (
    ("FIM acc", "fim_acc", "frac"),
    ("Exec-valid", "exec_valid_autocomplete", "frac"),
    ("Exec ÷ ceiling", "exec_valid_vs_ceiling", "frac"),
    ("Syntax-valid", "syntax_valid", "frac"),
    ("Generic-JS FIM acc (n=500)", "generic_js_fim_acc", "frac"),
)
FIELD_COLUMNS = (
    ("`full` (n={n_context_full})", "nonexistent_fields_full_rate_pct"),
    ("`irrelevant` (n={n_context_irrelevant})", "nonexistent_fields_irrelevant_rate_pct"),
    ("`none` (n={n_context_none})", "nonexistent_fields_none_rate_pct"),
)

# chave do model-index -> (chave em measures.metrics)
MODEL_INDEX_MAP = {
    "fim_accuracy": "fim_acc",
    "exec_valid": "exec_valid_autocomplete",
    "exec_valid_vs_ceiling": "exec_valid_vs_ceiling",
    "syntax_valid": "syntax_valid",
    "nonexistent_fields_full_pct": "nonexistent_fields_full_rate_pct",
}


@dataclass
class Tier:
    key: str
    name: str
    ws: str                      # workspace relativo a ROOT
    base_name: str               # run zero-shot
    run: str                     # run do modelo publicado
    rows: tuple                  # (rotulo, run) na ordem da tabela
    claims: list = field(default_factory=list)   # (trecho do card, arquivo fonte, trecho da fonte)

    @property
    def card(self) -> Path:
        return ROOT / self.ws / "hf" / "README.md"

    @property
    def eval_dir(self) -> Path:
        return ROOT / self.ws / "reports" / "eval"


RESUMO = "docs/resumo-executivo.md"
ONNX = "reports/export/onnx_export_t0_t1.md"
ADOPT = "reports/teacher_adoption.json"
CORPUS = "reports/student/corpus_report.md"
TEACHER = "Repos/KapiTeacher-Mongo-7B/reports/eval/sft_v2d-final/summary.json"

COMMON_CLAIMS = [
    ("Holm-adjusted p=0.019", RESUMO, "Holm p=0,019"),
    ("difference -0.85 percentage points", ADOPT, '"fim_acc_delta_pp": -0.85'),
    ("failed (-0.15 points)", ADOPT, '"value_pp": -0.15'),
    ("The 7B teacher", TEACHER, '"normalized_match": 0.41'),
    ("110 schemas in 14 domains", RESUMO, "110 schemas sintéticos em 14 domínios"),
    ("12.12%", CORPUS, "12.12%"),
    ("(MIT, Apache-2.0, BSD, ISC)", "data/replay/v1/LICENSES.md", "permissivas (MIT, Apache-2.0, BSD-2/3-Clause, ISC)"),
    ("0.6255", RESUMO, "O teto do validador é 0,6255"),
    ("Hui et al.", "BaseRepos/Qwen2.5-Coder-0.5B/README.md", "arXiv:2409.12186"),
    ("`kapilab-kapi-model-v1`", ONNX, "kapilab-kapi-model-v1"),
    ("does not append that token", "scripts/export/README.md", "o acrescenta a sequencia"),
    ("without a workaround", "scripts/export/README.md", "os pacotes entregues NAO foram alterados"),
    ("0 write executions", "{ws}/reports/eval/{run}/measures.json", '"write_or_destructive_executions": 0'),
    ("2 s limit", "{ws}/reports/eval/{run}/summary.json", '"max_time_ms": 2000'),
    ("seed 20261008", "{ws}/reports/eval/{run}/summary.json", '"seed": 20261008'),
    ("AMD Radeon RX 7800 XT", "{ws}/reports/eval/{run}/summary.json", '"device_name": "AMD Radeon RX 7800 XT"'),
    ("One AMD Radeon RX 7800 XT", "{ws}/reports/{run}/run_manifest.json", "7800 XT"),
    ("transformers 4.57.6", "{ws}/reports/{run}/run_manifest.json", '"transformers": "4.57.6"'),
    ("up to 96 new tokens", "{ws}/reports/eval/{run}/summary.json", '"max_new_tokens": 96'),
    ("2048-token context", "{ws}/reports/eval/{run}/summary.json", '"context_tokens": 2048'),
    ("(the IDE caps it at 32 tokens)", "{ws}/exports/{pkg}-ONNX-int4/kapibarastudio-model.json", '"maxTokens": 32'),
    ("`Mongo Console JavaScript` and `json`", "reports/datasets/v2c-validation-report.md", "dialetos {'console': 889205, 'json': 111410}"),
]

NANO_CLAIMS = [
    ("(151,180 MongoDB examples + 8,685", CORPUS, "(151,180 Mongo + 8,685 replay)"),
    ("54.27%", CORPUS, "54.27%"),
    ("| `onnx/int4` | 392 MB", ONNX, "`int4` 392 MB"),
    ("| `onnx/dml-fp16` | 959 MB", ONNX, "`dml-fp16` 959 MB"),
    ("identical in 79.0% of prompts", ONNX, "| int4 | 392 | 79,0 |"),
    ("identical in 97.5% of prompts", ONNX, "| dml-fp16 | 959 | 97,5 |"),
    ("within 0.8 points of the bf16 weights, none significant (p >= 0.34)", ONNX, "|Δ| ≤ 0,8 pp, todos p ≥ 0,34"),
    ("285 ms on average and 480 ms at p95", ONNX, "| 285/480 |"),
    ("109/124 ms", ONNX, "**109/124**"),
    ("internal target of 150/250 ms", ONNX, "T0 150/250 ms"),
    ("exec-valid is 1.7 points higher", RESUMO, "exec-valid +1,7 p.p."),
    ("FIM accuracy 0.4115 against 0.403", "Repos/KapiCoder-Mongo-Nano/reports/eval/compare/sft_distill_v1_vs_sft_gold_v1.md",
     "| exact_match | 2000 | 0.4115 | 0.4030 |"),
]

SMALL_CLAIMS = [
    ("(217,306 MongoDB examples + 12,528", CORPUS, "(217,306 Mongo + 12,528 replay)"),
    ("54.11%", CORPUS, "54.11%"),
    ("alpha=128, dropout 0.05", "Repos/KapiCoder-Mongo-Small/configs/sft_distill_v1.yaml", "alpha: 128"),
    ("peak LR 1.5e-4", "Repos/KapiCoder-Mongo-Small/configs/sft_distill_v1.yaml", "lr: 1.5e-4"),
    ("dropout 0.05", "Repos/KapiCoder-Mongo-Small/configs/sft_distill_v1.yaml", "dropout: 0.05"),
    ("3,422 steps", "Repos/KapiCoder-Mongo-Small/README_job.md", "`--total-steps 3422` (1,5 epoca)"),
    ("| `onnx/dml-fp16` | 2,964 MB", ONNX, "`dml-fp16` 2.964 MB"),
    ("| `onnx/int8` | 2,578 MB", ONNX, "`int8` 2.578 MB"),
    ("identical in 96.5% of prompts", ONNX, "| dml-fp16 | 2964 | 96,5 |"),
    ("identical in 94.5% of prompts", ONNX, "| int8 | 2578 | 94,5 |"),
    ("0.4 points below the bf16 weights (not significant, p=0.50)", ONNX, "0,408 (−0,4; 0,50)"),
    ("(1,090 MB and 952 MB for CPU, 1,293 MB for DirectML)", ONNX, "`int4` 1.090 MB, `int4-rtn` 952 MB"),
    ("fell by 1.4 to 1.8 points (p between 0.004 and 0.04)", ONNX, "**−1,8; 0,004**"),
    ("lost 1.4 to 1.8 FIM-accuracy points", ONNX, "**−1,4; 0,039**"),
    ("148 ms on average and 170 ms at p95", ONNX, "**148/170**"),
    ("(104 tokens/s)", ONNX, "| 104 |"),
    ("697 ms on average and 1192 ms at p95", ONNX, "697/1192"),
    ("internal target of 500/650 ms", ONNX, "T1 500/650"),
    ("FIM accuracy is 0.75 points higher (not significant)", RESUMO, "FIM acc +0,75 p.p., **não significativo**"),
    ("exec-valid 1.3 points higher (p=0.0016)", RESUMO, "exec-valid +1,3 p.p. (p=0,0016)"),
    ("(1.58% against 2.06%)", "Repos/KapiCoder-Mongo-Nano/reports/eval/sft_distill_v1/measures.json", '"nonexistent_fields_full_rate_pct": 2.06'),
    ("No control trained on the original reference labels", ADOPT, "o controle gold do T1 (J10) fica fora"),
    ("FIM accuracy 0.403 against 0.4115", "Repos/KapiCoder-Mongo-Nano/reports/eval/compare/sft_distill_v1_vs_sft_gold_v1.md",
     "| exact_match | 2000 | 0.4115 | 0.4030 |"),
]

TIERS = {
    "nano": Tier("nano", "KapiCoder-Mongo-Nano", "Repos/KapiCoder-Mongo-Nano", "base-qwen2.5-coder-0.5b", "sft_distill_v1",
                 (("Qwen2.5-Coder-0.5B, zero-shot base", "base-qwen2.5-coder-0.5b"),
                  ("**KapiCoder-Mongo-Nano** (this model)", "sft_distill_v1"),
                  ("0.5B trained on original labels (control, not released)", "sft_gold_v1")),
                 NANO_CLAIMS),
    "small": Tier("small", "KapiCoder-Mongo-Small", "Repos/KapiCoder-Mongo-Small", "base-qwen2.5-coder-1.5b", "sft_distill_v1",
                  (("Qwen2.5-Coder-1.5B, zero-shot base", "base-qwen2.5-coder-1.5b"),
                   ("**KapiCoder-Mongo-Small** (this model)", "sft_distill_v1")),
                  SMALL_CLAIMS),
}


# ------------------------------------------------------------------ homologacao v1 (fora da distribuicao)
HOM_JSON = "reports/datasets/homolog-v1-eval.json"
HOM_MD = "reports/datasets/homolog-v1-eval.md"
HOM_MANIFEST = "reports/datasets/homolog-v1-manifest.json"
TEACHER_LABEL = "7B teacher (`KapiTeacher-Mongo-7B`, not released)"

# tier -> (base, modelo publicado, controle gold ou None) como chaves de `models` no JSON da homologacao
HOM_KEYS = {"nano": ("base_0.5b", "t0_distill", "t0_gold"), "small": ("base_1.5b", "t1_distill", None)}
HOM_CTX = ("full", "irrelevant", "none")

HOM_CLAIMS = [
    ("28 schemas", HOM_MD, "28 schemas `hom`"),
    ("Qwen3-14B", HOM_MD, "Autor Qwen3-14B / gramatica"),
    ("`aca0aae5`", HOM_MANIFEST, '"set_sha256": "aca0aae5'),
    ("telecom and adtech", HOM_MD, "telecom/adtech"),
    ("418 items", HOM_MD, "418 itens"),
    ("smallest `cut_kind` has n=269", HOM_MD, "menor `cut_kind` tem n=269"),
    ("single greedy run", HOM_MD, "Greedy unico"),
    ("NF4", HOM_MD, "NF4+LoRA"),
    ("McNemar", HOM_MD, "McNemar exato"),
    ("95% CI", HOM_MD, "Wilson (FIM) e bootstrap"),
    ("no write command was forwarded", HOM_MD, "nenhum comando de escrita encaminhado"),
    ("RX 7800 XT", HOM_MD, "Todas as geracoes na RX 7800 XT"),
    ("greedy decoding, up to 96 new tokens", HOM_MD, "greedy, 96 tokens, contexto 2048"),
]

_HOM_CACHE: dict = {}


def hom() -> dict:
    if "d" not in _HOM_CACHE:
        _HOM_CACHE["d"] = json.loads((ROOT / HOM_JSON).read_text(encoding="utf-8"))
    return _HOM_CACHE["d"]


def hp(x: float) -> str:
    """fracao -> percentual com 2 casas (exato ao JSON, que guarda 4 casas); sem arredondar para 1 casa."""
    return f"{x * 100:.2f}"


def hci(a) -> str:
    return f"[{hp(a[0])}, {hp(a[1])}]"


def hd(x: float, nd: int = 1) -> str:
    return f"{x * 100:+.{nd}f}"


def hnum(n: int) -> str:
    return f"{n:,}"


def to_pt(text: str) -> str:
    """'1,234.56' (en) -> '1.234,56' (pt-BR)."""
    return text.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def hom_rows(tier: "Tier", with_teacher: bool = True) -> list[tuple[str, str]]:
    base, mdl, gold = HOM_KEYS[tier.key]
    rows = [(tier.rows[0][0], base), (tier.rows[1][0], mdl)]
    if gold:
        rows.append((tier.rows[2][0], gold))
    if with_teacher:
        rows.append((TEACHER_LABEL, "teacher_v2d"))
    return rows


def hpair(b: str, a: str, metric: str, dim: str | None = None, group: str | None = None) -> dict:
    """Linha B-A de `pairs[<b>_vs_<a>]` (delta = B - A); geral ou de um grupo de `dim`."""
    node = hom()["pairs"][f"{b}_vs_{a}"]
    rows = node["context_kind"]["overall"] if group is None else node[dim]["groups"][group]["metrics"]
    for r in rows:
        if r["metric"] == metric:
            return r
    raise KeyError((b, a, metric, dim, group))


def hdelta(r: dict) -> str:
    return f"{hd(r['delta'])} pp [{hd(r['ci_low'])}, {hd(r['ci_high'])}]"


def hpval(r: dict) -> str:
    p = r.get("p_holm", r.get("p"))
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def render_hom_table(tier: "Tier") -> str:
    H = hom()["models"]
    lines = ["| Model | FIM acc (%) [95% CI] | Exec-valid (%) | Exec ÷ ceiling (%) [95% CI] | Syntax-valid (%) |",
             "|---|---|---:|---|---:|"]
    for label, k in hom_rows(tier):
        o = H[k]["overall"]
        lines.append(f"| {label} | {hp(o['normalized_match'])} {hci(o['normalized_match_ci95'])} | {hp(o['exec_valid'])} | "
                     f"{hp(o['exec_over_ceiling'])} {hci(o['exec_over_ceiling_ci95'])} | {hp(o['syntax_valid'])} |")
    o = H[HOM_KEYS[tier.key][1]]["overall"]
    lines += ["", f"Internal evaluation, synthetic out-of-distribution homologation, n={hnum(o['n'])}. Validator ceiling for exec-valid: {hp(o['ceiling_exec_valid'])}%."]
    return "\n".join(lines)


def render_hom_fields_table(tier: "Tier") -> str:
    H = hom()["models"]
    ref = H[HOM_KEYS[tier.key][1]]["by_context_kind"]
    head = "| Nonexistent fields (%) by `context_kind`, homologation | " + " | ".join(f"`{c}` (n={hnum(ref[c]['n'])})" for c in HOM_CTX) + " |"
    lines = [head, "|---|" + "---:|" * len(HOM_CTX)]
    for label, k in hom_rows(tier):
        lines.append("| " + label + " | " + " | ".join(hp(H[k]["by_context_kind"][c]["nonexistent_fields_rate"]) for c in HOM_CTX) + " |")
    return "\n".join(lines)


def hom_gap(k: str) -> dict:
    d = hom()
    h, v = d["models"][k]["overall"], d["val_v2c"][k]
    return {"fim": (v["normalized_match"], h["normalized_match"]),
            "exec": (v["exec_over_ceiling"], h["exec_over_ceiling"]),
            "full": (v["nonexistent_fields_full"], d["models"][k]["by_context_kind"]["full"]["nonexistent_fields_rate"])}


def render_hom_gap_table(tier: "Tier") -> str:
    lines = ["| Model | FIM acc (%): validation → homologation (Δ pp) | Exec ÷ ceiling (%): validation → homologation (Δ pp) | Nonexistent fields `full` (%): validation → homologation (Δ pp) |",
             "|---|---|---|---|"]
    for label, k in hom_rows(tier):
        g = hom_gap(k)
        cells = [f"{hp(a)} → {hp(b)} ({(b - a) * 100:+.2f})" for a, b in (g["fim"], g["exec"], g["full"])]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    mdl = HOM_KEYS[tier.key][1]
    vc, hc = hom()["val_v2c"][mdl]["ceiling_exec_valid"], hom()["models"][mdl]["overall"]["ceiling_exec_valid"]
    lines += ["", f"The exec ÷ ceiling column compares two validators (validation: ceiling {hp(vc)}%, homologation: ceiling {hp(hc)}%); read trends, not decimals."]
    return "\n".join(lines)


def hom_vals(tier: "Tier") -> dict:
    """Todos os numeros usados nos blocos de leitura, tirados do JSON."""
    H = hom()["models"]
    base, mdl, gold = HOM_KEYS[tier.key]
    o = lambda k: H[k]["overall"]  # noqa: E731
    f = lambda k, c: H[k]["by_context_kind"][c]["nonexistent_fields_rate"]  # noqa: E731
    return {"base": base, "mdl": mdl, "gold": gold,
            "gap": hom_gap(mdl), "gap_base": hom_gap(base),
            "o_m": o(mdl), "o_b": o(base),
            "f_m": {c: f(mdl, c) for c in HOM_CTX}, "f_b": {c: f(base, c) for c in HOM_CTX},
            "val_full_m": hom()["val_v2c"][mdl]["nonexistent_fields_full"],
            "au_m": {a: H[mdl]["by_author"][a] for a in ("grammar", "llm")},
            "ck_stage": H[mdl]["by_cut_kind"]["stage"]["normalized_match"],
            "ck_multiline": H[mdl]["by_cut_kind"]["multiline"]["normalized_match"],
            "n_total": o(mdl)["n"],
            "vs_base_author": {a: hpair(mdl, base, "normalized_match", "author", a) for a in ("grammar", "llm")},
            "vs_teacher_fim": hpair(mdl, "teacher_v2d", "normalized_match"),
            "vs_teacher_exec": hpair(mdl, "teacher_v2d", "exec_valid"),
            "t1_t0_fim": hpair("t1_distill", "t0_distill", "normalized_match"),
            "t1_t0_exec": hpair("t1_distill", "t0_distill", "exec_valid"),
            "dg_fim": hpair("t0_distill", "t0_gold", "normalized_match"),
            "dg_exec": hpair("t0_distill", "t0_gold", "exec_valid"),
            "dg_exec_grammar": hpair("t0_distill", "t0_gold", "exec_valid", "author", "grammar"),
            "dg_exec_llm": hpair("t0_distill", "t0_gold", "exec_valid", "author", "llm"),
            "nx_distill": o("t0_distill")["nonexistent_fields_rate"], "nx_gold": o("t0_gold")["nonexistent_fields_rate"],
            "nx_t1": o("t1_distill")["nonexistent_fields_rate"], "nx_teacher": o("teacher_v2d")["nonexistent_fields_rate"],
            "nx_t0": o("t0_distill")["nonexistent_fields_rate"]}


def render_hom_reading(tier: "Tier") -> str:
    v = hom_vals(tier)
    g, gb = v["gap"], v["gap_base"]
    full, irr, none = (v["f_m"][c] for c in HOM_CTX)
    bfull, birr, bnone = (v["f_b"][c] for c in HOM_CTX)
    A, B = v["au_m"]["grammar"], v["au_m"]["llm"]
    ga, lb = v["vs_base_author"]["grammar"], v["vs_base_author"]["llm"]
    L = ["Reading the homologation tables:", ""]
    L.append(f"- **Generalisation gap.** From the synthetic validation set to unseen schemas, FIM accuracy goes {hp(g['fim'][0])}% → {hp(g['fim'][1])}% "
             f"({(g['fim'][1] - g['fim'][0]) * 100:+.2f} pp) and exec ÷ ceiling {hp(g['exec'][0])}% → {hp(g['exec'][1])}% ({(g['exec'][1] - g['exec'][0]) * 100:+.2f} pp). "
             f"The zero-shot base model goes the other way ({hp(gb['fim'][0])}% → {hp(gb['fim'][1])}% FIM accuracy), so the homologation queries are "
             "easier for a model that was not trained here, and the gap of the trained model may be understated (interpretation, not tested).")
    L.append("- **Field names are the main risk. Do not trust field names unless the schema is in the context, and validate a suggestion before running it.** "
             f"On schemas never seen in training this model invents field names more often than its base: {hp(full)}% of suggestions with the schema visible (`full`) "
             f"against {hp(bfull)}% for the zero-shot base (on the synthetic validation set it was {hp(v['val_full_m'])}%). "
             f"Without the schema the rates are {hp(irr)}% (`irrelevant`) and {hp(none)}% (`none`), against {hp(birr)}% and {hp(bnone)}% for the base. "
             f"Having the schema in the header did not lower the rate here ({hp(full)}% with it, {hp(none)}% without); why is a hypothesis that was not tested. "
             f"The rate is counted only on suggestions whose command reached the server, and the base model more often fails to parse (syntax-valid {hp(v['o_b']['syntax_valid'])}% "
             f"against {hp(v['o_m']['syntax_valid'])}%), so the base rates cover fewer suggestions and are not directly comparable.")
    if v["gold"]:
        L.append(f"- **Distillation against the gold-label control is a small effect.** Same 6,030 items, paired: FIM accuracy {hdelta(v['dg_fim'])} for the distilled model "
                 f"(McNemar-Holm p={hpval(v['dg_fim'])}, no evidence of a difference in FIM accuracy); exec-valid {hdelta(v['dg_exec'])} "
                 f"({hdelta(v['dg_exec_grammar'])} on `grammar`, {hdelta(v['dg_exec_llm'])} on `llm`); nonexistent fields {hp(v['nx_distill'])}% against {hp(v['nx_gold'])}% "
                 "(all `context_kind` together). On the synthetic validation set the distilled model was also higher in exec-valid and not significantly different in FIM accuracy (see above); the size is small here too.")
        L.append(f"- **Against the siblings.** KapiCoder-Mongo-Small is ahead on this set: FIM accuracy {hdelta(v['t1_t0_fim'])} and exec-valid {hdelta(v['t1_t0_exec'])} over this model "
                 f"(95% CIs above zero). The 7B teacher is ahead of this model by {hd(-v['vs_teacher_fim']['delta'])} pp in FIM accuracy and {hd(-v['vs_teacher_exec']['delta'])} pp in exec-valid.")
    else:
        L.append("- **The effect of distillation at this size was not measured** (no gold-label control exists for this model). In the 0.5B sibling, on this same set, distilled against gold control: "
                 f"FIM accuracy {hdelta(v['dg_fim'])} (McNemar-Holm p={hpval(v['dg_fim'])}, no evidence of a difference) and exec-valid {hdelta(v['dg_exec'])}; small effect.")
        L.append(f"- **Against the 0.5B sibling and the teacher.** This model is ahead of KapiCoder-Mongo-Nano: FIM accuracy {hdelta(v['t1_t0_fim'])}, exec-valid {hdelta(v['t1_t0_exec'])} "
                 f"(95% CIs above zero), nonexistent fields {hp(v['nx_t1'])}% against {hp(v['nx_t0'])}%. Against the 7B teacher its FIM accuracy is tied: {hdelta(v['vs_teacher_fim'])} "
                 f"(McNemar-Holm p={hpval(v['vs_teacher_fim'])}); exec-valid is {hdelta(v['vs_teacher_exec'])} and nonexistent fields {hp(v['nx_t1'])}% against {hp(v['nx_teacher'])}% "
                 "(differences small; the 7B NF4 model does not beat this one here).")
    L.append(f"- **`grammar` against `llm`.** The gain over the zero-shot base in FIM accuracy is larger on `grammar` ({hdelta(ga)}) than on `llm` ({hdelta(lb)}), and exec ÷ ceiling is higher on "
             f"`grammar` ({hp(A['exec_over_ceiling'])}%) than on `llm` ({hp(B['exec_over_ceiling'])}%). This fits learning a known structure: `grammar` queries come from templates of the same family as the "
             "training data, with different schemas, so they are not an independent test of style; `llm` is the new-style test, and it is where the trained model is weakest.")
    L.append(f"- **Long cuts.** FIM accuracy on `stage` and `multiline` cuts is {hp(v['ck_stage'])}% and {hp(v['ck_multiline'])}% (exact match against a long target within 96 new tokens); "
             "do not read the overall number as evenly spread over cut kinds.")
    return "\n".join(L)


def render_hom_pt(tier: "Tier") -> str:
    v = hom_vals(tier)
    g = v["gap"]
    full, irr, none = (v["f_m"][c] for c in HOM_CTX)
    bfull = v["f_b"]["full"]

    def t(x: float) -> str:
        return to_pt(hp(x))

    return (f"- **Homologação ({REQUIRED_HOM_PT}):** {to_pt(hnum(v['n_total']))} cortes de 28 schemas de telecom e adtech nunca vistos no treino, escritos pelo Qwen3-14B e por gramática, **não por humanos** "
            f"(mesma família Qwen dos modelos). FIM acc {t(g['fim'][1])}% (validação sintética: {t(g['fim'][0])}%; base: {t(v['o_b']['normalized_match'])}%); "
            f"exec ÷ teto {t(g['exec'][1])}% (validação: {t(g['exec'][0])}%, com validadores diferentes). "
            f"**Alerta:** com o schema visível (`full`), {t(full)}% das sugestões usam campos inexistentes, contra {t(bfull)}% da base zero-shot (validação sintética: {t(v['val_full_m'])}%); "
            f"sem schema, {t(irr)}% (`irrelevant`) e {t(none)}% (`none`). Não confie nos nomes de campo sem o schema no contexto; valide antes de executar. "
            "Executar não é acertar a intenção.")


HOM_BLOCKS = {"homolog-table": render_hom_table, "homolog-fields-table": render_hom_fields_table,
              "homolog-gap-table": render_hom_gap_table, "homolog-reading": render_hom_reading, "homolog-pt": render_hom_pt}


def check_homolog_sources() -> list[str]:
    """Consistencia interna das fontes da homologacao (JSON x manifesto selado)."""
    errs: list[str] = []
    d = hom()
    man = json.loads((ROOT / HOM_MANIFEST).read_text(encoding="utf-8"))
    if d.get("set_sha256") != man.get("set_sha256"):
        errs.append("homolog: set_sha256 do eval.json difere do manifesto selado")
    if d.get("n") != man.get("n_items"):
        errs.append(f"homolog: n={d.get('n')} difere do manifesto ({man.get('n_items')})")
    if len(man.get("by_schema", {})) != 28:
        errs.append(f"homolog: manifesto tem {len(man.get('by_schema', {}))} schemas, o card diz 28")
    if min(man.get("by_cut_kind", {}).values()) != 269:
        errs.append("homolog: menor cut_kind do manifesto nao e 269")
    for k, m in d["models"].items():
        if m["overall"]["n"] != d["n"]:
            errs.append(f"homolog: modelo {k} com n={m['overall']['n']} != {d['n']}")
    for c, n in man.get("by_context_kind", {}).items():
        if d["models"]["t0_distill"]["by_context_kind"][c]["n"] != n:
            errs.append(f"homolog: n de context_kind={c} difere do manifesto")
    return errs


# ------------------------------------------------------------------ utilidades
def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def frac(x: float) -> str:
    s = f"{x:.4f}"
    return s[:-1] if s.endswith("0") else s


def pct(x: float) -> str:
    return f"{x:.2f}"


def en_int(n: int) -> str:
    return f"{n:,}"


def pt_int(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def measures(tier: Tier, run: str) -> dict:
    return read_json(tier.eval_dir / run / "measures.json")["metrics"]


def summary(tier: Tier, run: str) -> dict:
    return read_json(tier.eval_dir / run / "summary.json")


def resolve(template: str, tier: Tier) -> Path:
    pkg = tier.name
    return ROOT / template.format(ws=tier.ws, run=tier.run, pkg=pkg)


# ------------------------------------------------------------------ tabelas
def render_accuracy_table(tier: Tier) -> str:
    head = "| Model | " + " | ".join(c[0] for c in ACC_COLUMNS) + " |"
    sep = "|---|" + "---:|" * len(ACC_COLUMNS)
    lines = [head, sep]
    for label, run in tier.rows:
        m = measures(tier, run)
        lines.append("| " + label + " | " + " | ".join(frac(m[key]) for _h, key, _f in ACC_COLUMNS) + " |")
    ref = measures(tier, tier.run)["reference_ceiling_exec_valid"]
    lines.append("")
    lines.append(f"Internal evaluation, synthetic data, n=2,000 (Generic-JS: n=500). Validator ceiling for exec-valid: {frac(ref)}.")
    return "\n".join(lines)


def render_fields_table(tier: Tier) -> str:
    ref = measures(tier, tier.run)
    head = "| Nonexistent fields (%) by `context_kind` | " + " | ".join(h.format(**ref) for h, _k in FIELD_COLUMNS) + " |"
    sep = "|---|" + "---:|" * len(FIELD_COLUMNS)
    lines = [head, sep]
    for label, run in tier.rows:
        m = measures(tier, run)
        lines.append("| " + label + " | " + " | ".join(pct(m[k]) for _h, k in FIELD_COLUMNS) + " |")
    return "\n".join(lines)


TABLES = {"accuracy-table": render_accuracy_table, "fields-table": render_fields_table}


def block(text: str, name: str) -> str | None:
    b, e = BEGIN.format(name), END.format(name)
    i, j = text.find(b), text.find(e)
    if i < 0 or j < 0 or j < i:
        return None
    return text[i + len(b):j].strip("\n")


def write_tables(tier: Tier) -> None:
    text = read_text(tier.card)
    for name, fn in {**TABLES, **HOM_BLOCKS}.items():
        b, e = BEGIN.format(name), END.format(name)
        i, j = text.find(b), text.find(e)
        if i < 0 or j < 0:
            raise SystemExit(f"{tier.card}: marcadores {name} ausentes")
        text = text[:i + len(b)] + "\n" + fn(tier) + "\n" + text[j:]
    tier.card.write_text(text, encoding="utf-8", newline="\n")


# ------------------------------------------------------------------ front matter
def front_matter(text: str) -> dict:
    import yaml
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    if not m:
        raise ValueError("front matter YAML ausente")
    data = yaml.safe_load(m.group(1))
    if not isinstance(data, dict):
        raise ValueError("front matter nao e um mapa")
    return data


def facts(tier: Tier) -> list[tuple[str, list[str]]]:
    """(rotulo, [trechos que precisam existir no card]) derivados dos arquivos de fonte."""
    lock = read_json(ROOT / tier.ws / "model.lock.json")
    rm = read_json(ROOT / tier.ws / "reports" / tier.run / "run_manifest.json")
    corpus = lock["arms"]["distill"]["corpus"]
    s = summary(tier, tier.run)
    out = [
        ("base revision", [lock["base"]["revision"]]),
        ("base repo", [lock["base"]["repo_id"]]),
        ("examples", [en_int(corpus["examples"]), pt_int(corpus["examples"])]),
        ("tokens per epoch", [en_int(corpus["tokens"]), pt_int(corpus["tokens"])]),
        ("tokens seen", [en_int(rm["tokens_seen"]), pt_int(rm["tokens_seen"])]),
        ("steps", [en_int(rm["total_steps"])]),
        ("hours", [f"{float(rm['elapsed_train_h']):.2f}", f"{float(rm['elapsed_train_h']):.2f}".replace(".", ",")]),
        ("eval n", [en_int(s["n"])]),
        ("eval domains", [f"{len(s['dataset']['domain'])} domains"]),
        ("context_kind n", [str(measures(tier, tier.run)[k]) for k in ("n_context_full", "n_context_irrelevant", "n_context_none")]),
        ("trainable params", [en_int(rm["plan"]["trainable_params"])] if rm["plan"]["method"] == "lora" else []),
    ]
    if rm["plan"]["method"] == "full":
        out.append(("method", ["Full fine-tuning", "peak LR 2e-5"]))
    else:
        out.append(("method", ["LoRA (r=64"]))
    return out


# ------------------------------------------------------------------ verificacao
def check_tier(key: str) -> list[str]:
    tier = TIERS[key]
    errors: list[str] = []
    if not tier.card.exists():
        return [f"{tier.card}: nao existe"]
    text = read_text(tier.card)

    # 1. tabelas
    for name, fn in TABLES.items():
        got = block(text, name)
        if got is None:
            errors.append(f"[{key}] marcadores {name} ausentes")
        elif got.strip() != fn(tier).strip():
            errors.append(f"[{key}] tabela {name} diverge das fontes (rode --write para ver a diferenca esperada)")

    for name, fn in HOM_BLOCKS.items():
        got = block(text, name)
        if got is None:
            errors.append(f"[{key}] marcadores {name} ausentes")
        elif got.strip() != fn(tier).strip():
            errors.append(f"[{key}] bloco {name} diverge de {HOM_JSON} (rode --write para ver a diferenca esperada)")
    errors += [f"[{key}] {e}" for e in check_homolog_sources()]
    # o val v2c dentro do JSON da homologacao precisa ser o mesmo que o card traz na tabela sintetica
    hm = hom()["val_v2c"]
    hom_of_run = {tier.rows[0][1]: HOM_KEYS[key][0], tier.rows[1][1]: HOM_KEYS[key][1]}
    if HOM_KEYS[key][2]:
        hom_of_run[tier.rows[2][1]] = HOM_KEYS[key][2]
    for run, hk in hom_of_run.items():
        m = measures(tier, run)
        for hkey, mkey, scale in (("normalized_match", "fim_acc", 1), ("exec_over_ceiling", "exec_valid_vs_ceiling", 1),
                                  ("nonexistent_fields_full", "nonexistent_fields_full_rate_pct", 0.01)):
            if abs(hm[hk][hkey] - m[mkey] * scale) > 1e-4:
                errors.append(f"[{key}] {run}: val_v2c.{hk}.{hkey}={hm[hk][hkey]} != measures.{mkey}={m[mkey]}")

    # coerencia cruzada summary x measures
    for _label, run in tier.rows:
        s = summary(tier, run)["headline"]
        m = measures(tier, run)
        pairs = (("normalized_match", "fim_acc"), ("exec_valid", "exec_valid_autocomplete"), ("syntax_valid", "syntax_valid"))
        for sk, mk in pairs:
            if abs(s[sk] - m[mk]) > 1e-9:
                errors.append(f"[{key}] {run}: summary.{sk}={s[sk]} != measures.{mk}={m[mk]}")
        gj = summary(tier, run + "-generic-js")["headline"]["normalized_match"]
        if abs(gj - m["generic_js_fim_acc"]) > 1e-9:
            errors.append(f"[{key}] {run}: generic-js summary {gj} != measures {m['generic_js_fim_acc']}")

    # 2. front matter / model-index
    try:
        fm = front_matter(text)
    except Exception as exc:  # noqa: BLE001
        return errors + [f"[{key}] front matter invalido: {exc}"]
    lock = read_json(ROOT / tier.ws / "model.lock.json")
    expect = {"license": "apache-2.0", "library_name": "transformers", "pipeline_tag": "text-generation",
              "base_model": lock["base"]["repo_id"]}
    for k, v in expect.items():
        if fm.get(k) != v:
            errors.append(f"[{key}] front matter {k}={fm.get(k)!r}, esperado {v!r}")
    if fm.get("language") != ["en", "pt"]:
        errors.append(f"[{key}] front matter language={fm.get('language')!r}")
    missing_tags = {"mongodb", "code", "fim", "autocomplete", "onnx"} - set(fm.get("tags") or [])
    if missing_tags:
        errors.append(f"[{key}] tags ausentes: {sorted(missing_tags)}")
    mi = fm.get("model-index")
    if not mi:
        errors.append(f"[{key}] model-index ausente")
    else:
        m = measures(tier, tier.run)
        seen = set()
        for res in mi[0]["results"]:
            ds = res["dataset"]["name"].lower()
            if "internal" not in ds or ("synthetic" not in ds and "npm" not in ds):
                errors.append(f"[{key}] dataset do model-index sem rotulo interno/sintetico: {res['dataset']['name']!r}")
            is_hom = "homolog" in res["dataset"].get("type", "")
            if is_hom and ("out-of-distribution" not in ds or "synthetic" not in ds):
                errors.append(f"[{key}] model-index da homologacao sem o rotulo 'internal ... synthetic out-of-distribution homologation'")
            for met in res["metrics"]:
                if met.get("verified") is not False:
                    errors.append(f"[{key}] model-index metrica {met['type']} sem `verified: false`")
                if is_hom:
                    ho = hom()["models"][HOM_KEYS[key][1]]
                    hmap = {"fim_accuracy": ho["overall"]["normalized_match"], "exec_valid": ho["overall"]["exec_valid"],
                            "exec_valid_vs_ceiling": ho["overall"]["exec_over_ceiling"],
                            "nonexistent_fields_full_pct": ho["by_context_kind"]["full"]["nonexistent_fields_rate"] * 100}
                    if met["type"] not in hmap:
                        errors.append(f"[{key}] model-index homologacao: metrica desconhecida {met['type']}")
                        continue
                    seen.add("hom:" + met["type"])
                    if abs(float(met["value"]) - hmap[met["type"]]) > 1e-9:
                        errors.append(f"[{key}] model-index homologacao {met['type']}={met['value']} != {HOM_JSON} ({hmap[met['type']]})")
                    continue
                if "Generic JavaScript" in res["task"]["name"]:
                    want = m["generic_js_fim_acc"]
                    seen.add("generic")
                else:
                    src = MODEL_INDEX_MAP.get(met["type"])
                    if src is None:
                        errors.append(f"[{key}] model-index metrica desconhecida {met['type']}")
                        continue
                    want = m[src]
                    seen.add(met["type"])
                if abs(float(met["value"]) - float(want)) > 1e-9:
                    errors.append(f"[{key}] model-index {met['type']}={met['value']} != fonte {want}")
        for k in list(MODEL_INDEX_MAP) + ["generic"]:
            if k not in seen:
                errors.append(f"[{key}] model-index sem {k}")
        # a homologacao so entra no model-index rotulada; se entrar, entra completa
        hom_seen = {x for x in seen if x.startswith("hom:")}
        if hom_seen and len(hom_seen) != 4:
            errors.append(f"[{key}] model-index da homologacao incompleto: {sorted(hom_seen)}")

    # 3. fatos
    for label, needles in facts(tier):
        for n in needles:
            if n not in text:
                errors.append(f"[{key}] fato '{label}': trecho '{n}' ausente do card")

    # 4. afirmacoes de prosa
    for snippet, src, src_snippet in COMMON_CLAIMS + HOM_CLAIMS + tier.claims:
        path = resolve(src, tier)
        if snippet not in text:
            errors.append(f"[{key}] trecho do card ausente: {snippet!r}")
        if not path.exists():
            errors.append(f"[{key}] fonte inexistente: {path}")
        elif src_snippet not in read_text(path):
            errors.append(f"[{key}] fonte {src} nao contem {src_snippet!r}")

    # 5. frases obrigatorias e proibidas
    for p in REQUIRED_PHRASES:
        if p not in text:
            errors.append(f"[{key}] frase obrigatoria ausente: {p!r}")
    if "**TBD.**" in text:
        errors.append(f"[{key}] o TBD da homologacao deveria ter sido substituido pelos resultados")
    for bad in ("state-of-the-art", "best-in-class", "outperforms", "beats GPT", "production-ready", "blazing"):
        if bad.lower() in text.lower():
            errors.append(f"[{key}] linguagem nao sobria: {bad!r}")
    return errors


# ------------------------------------------------------------------ SOURCES.md
def render_sources(tier: Tier) -> str:
    lines = [f"# SOURCES - {tier.name} (apendice interno, NAO sobe para o Hugging Face)", "",
             "Cada numero ou afirmacao do `README.md` deste diretorio e conferido por `scripts/publish/verify_card_numbers.py`; esta tabela e gerada por ele",
             "(`--write-sources`). Caminhos relativos a raiz do workspace.", "",
             "## Tabelas de metricas (celula = chave do JSON)", "",
             "| Linha | Arquivo (measures.json) | Chaves |", "|---|---|---|"]
    keys = ", ".join(f"`{k}`" for _h, k, _f in ACC_COLUMNS) + ", " + ", ".join(f"`{k}`" for _h, k in FIELD_COLUMNS)
    for label, run in tier.rows:
        lines.append(f"| {label} | `{tier.ws}/reports/eval/{run}/measures.json` | {keys} |")
    lines += ["", "Coerencia cruzada: `summary.json` (`headline.normalized_match`, `exec_valid`, `syntax_valid`) e `<run>-generic-js/summary.json` batem com `measures.json`.", "",
              "## Fatos de treino", "", "| Fato | Fonte |", "|---|---|",
              f"| base, revisao, exemplos, tokens por epoca | `{tier.ws}/model.lock.json` |",
              f"| passos, tokens vistos, horas, metodo, parametros treinaveis, bibliotecas | `{tier.ws}/reports/{tier.run}/run_manifest.json` |",
              f"| hiperparametros | `{tier.ws}/configs/sft_distill_v1.yaml` |",
              f"| pesos publicados (sha256) | `{tier.ws}/merged/{tier.run}/export_info.json` |",
              "| composicao do corpus, replay 12%, rotulo difere do gold | `reports/student/corpus_report.md` |",
              "| dataset v2c (funil, dialetos, decontaminacao) | `reports/datasets/v2c-validation-report.md` |",
              "| decisoes e limites (cego descartado, teto 0,6255, sem schema 11-14%) | `docs/resumo-executivo.md` |",
              "| professor, G2.4, papel de filtro | `reports/teacher_adoption.json`, `Repos/KapiTeacher-Mongo-7B/README_job.md` |",
              "| atribuicao do replay | `data/replay/v1/LICENSES.md` -> `REPLAY_ATTRIBUTION.md` |",
              "| base: licenca e revisao | `BaseRepos/Qwen2.5-Coder-*/README.md`, `BaseRepos/base-repos.lock.json` |",
              "| contrato `editor-context-v1` | `data/contracts/editor-context-v1.golden`, `scripts/data/render_contract.py` (exemplo do card testado contra o golden) |",
              "| ONNX: variantes, tamanhos, paridade, latencia proxy, bug do EOS | `reports/export/onnx_export_t0_t1.md`, `scripts/export/README.md`, `kapibarastudio-model.json` / `kapicoder_manifest.json` de cada pacote |",
              "", "## Homologacao v1 (fora da distribuicao; autor Qwen3-14B/gramatica, nao humano)", "",
              "Tabelas e blocos de leitura do card (`homolog-table`, `homolog-fields-table`, `homolog-gap-table`, `homolog-reading`, `homolog-pt`) e o `model-index` da homologacao sao renderizados/conferidos por `verify_card_numbers.py` a partir de `reports/datasets/homolog-v1-eval.json` (agregados; nenhum item de `data/homolog` e lido). Percentuais com 2 casas = o valor de 4 casas do JSON (o `.md` da fonte arredonda para 1 casa).", "",
              "| Celula do card | Chave em `reports/datasets/homolog-v1-eval.json` |", "|---|---|",
              "| FIM acc [IC95] | `models.<m>.overall.normalized_match`, `normalized_match_ci95` |",
              "| Exec-valid, sintaxe | `models.<m>.overall.exec_valid`, `syntax_valid` |",
              "| Exec ÷ teto [IC95]; teto | `models.<m>.overall.exec_over_ceiling`, `exec_over_ceiling_ci95`, `ceiling_exec_valid` |",
              "| Campos inexistentes por `context_kind` | `models.<m>.by_context_kind.<c>.nonexistent_fields_rate` |",
              "| Lacuna val -> hom | `val_v2c.<m>.{normalized_match, exec_over_ceiling, nonexistent_fields_full}` vs `models.<m>.overall` / `by_context_kind.full` |",
              "| Deltas pareados, IC95, McNemar-Holm | `pairs.<b>_vs_<a>.context_kind.overall[]` e `pairs.<b>_vs_<a>.author.groups.<g>.metrics[]` (`delta`, `ci_low`, `ci_high`, `p_holm`) |",
              "| `grammar` x `llm` | `models.<m>.by_author.<g>.exec_over_ceiling`; pares `author.groups` |",
              "| Conjunto selado (28 schemas, n=6.030, sha `aca0aae5...`, menor corte 269) | `reports/datasets/homolog-v1-manifest.json` (conferido contra o eval.json) |",
              "| Protocolo e limites (autor, executar != acertar, teto 93,1% com 418 itens, n por celula, greedy unico) | `reports/datasets/homolog-v1-eval.md` |",
              "", "Os `v1-*-report.md` em `reports/datasets/` descrevem o dataset v1 e NAO foram usados para nenhum numero do card.", "",
              "## Afirmacoes de prosa (trecho do card <-> trecho da fonte)", "", "| Trecho no card | Fonte | Trecho na fonte |", "|---|---|---|"]
    for snippet, src, src_snippet in COMMON_CLAIMS + HOM_CLAIMS + tier.claims:
        path = src.format(ws=tier.ws, run=tier.run, pkg=tier.name)
        lines.append(f"| {snippet.replace('|', '\\|')} | `{path}` | {src_snippet.replace('|', '\\|')} |")
    lines += ["", "## Sem fonte numerica (descritivo; revisar)", "",
              "- Homologacao: \"the gap of the trained model may be understated\" e \"why is a hypothesis that was not tested\" sao interpretacao do relatorio da fonte (`homolog-v1-eval.md`, secoes Interpretacao e Contexto), marcadas como hipotese no card; nao ha experimento que as teste.",
              "- Homologacao: \"the base more often fails to parse, so base rates cover fewer suggestions\" decorre da definicao da fonte (campo inexistente so conta quando o comando chega ao servidor) e das taxas de sintaxe; o efeito sobre a taxa nao foi quantificado.",
              "- \"No personal data by construction\": README dos workspaces (\"dados sinteticos ... nao contem dados pessoais\") e rejeicao de 17 registros com padroes proibidos na etapa sanitize do v2c. Nao ha revisao humana (G3.7 nao feita); o card diz isso.",
              "- \"Behaviour around destructive operations was neither trained nor tested\": ausencia de qualquer teste desse tipo nos relatorios; so ha `write_or_destructive_executions: 0` (guarda de leitura).",
              "- \"Not evaluated: bias or toxicity, long-context beyond 2048, multi-turn, adversarial\": ausencia nos relatorios.",
              "- Nome do repositorio (`esilva/...`), contato e visibilidade: decisao do usuario (ver `MODEL_CARD_REVIEW.md`).",
              "- O exemplo do card foi executado localmente em CPU contra os pesos `merged` (sem rede) e a funcao `build_prompt` e testada contra os SHA do golden.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=sorted(TIERS), action="append")
    ap.add_argument("--write", action="store_true", help="reescreve as tabelas do card a partir das fontes")
    ap.add_argument("--write-sources", action="store_true", help="regenera hf/SOURCES.md")
    args = ap.parse_args(argv)
    keys = args.tier or sorted(TIERS)
    if args.write:
        for k in keys:
            write_tables(TIERS[k])
            print(f"[{k}] tabelas reescritas em {TIERS[k].card}")
    if args.write_sources:
        for k in keys:
            out = TIERS[k].card.parent / "SOURCES.md"
            out.write_text(render_sources(TIERS[k]), encoding="utf-8", newline="\n")
            print(f"[{k}] {out}")
    bad = 0
    for k in keys:
        errs = check_tier(k)
        if errs:
            bad += len(errs)
            print(f"[{k}] FALHOU ({len(errs)}):")
            for e in errs:
                print("  -", e)
        else:
            print(f"[{k}] OK: tabelas, model-index, fatos e afirmacoes batem com as fontes")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
