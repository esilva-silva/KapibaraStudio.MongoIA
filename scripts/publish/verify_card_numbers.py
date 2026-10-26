#!/usr/bin/env python3
"""Confere que os numeros dos model cards (pesos v3) do Hugging Face batem com as fontes do projeto (sem rede, somente leitura).

  python scripts/publish/verify_card_numbers.py                  # confere nano e small (exit 1 se algo divergir)
  python scripts/publish/verify_card_numbers.py --tier nano
  python scripts/publish/verify_card_numbers.py --write          # reescreve as tabelas entre os marcadores do card a partir dos JSON
  python scripts/publish/verify_card_numbers.py --write-sources  # regenera hf/SOURCES.md (apendice interno afirmacao <-> fonte)

O que e conferido (tudo contra arquivos do repositorio; nenhuma metrica e digitada a mao no script):
  1. tabelas de metricas do card (val v2c sintetico: base, checkpoint interno v1 nao publicado, v3) == renderizacao dos `measures.json`;
  2. `model-index` do front matter == as mesmas medidas, com `verified: false`;
  3. fatos de treino v3 (exemplos, tokens, passos, horas, revisao da base) == `run_manifest.json` do `sft_v3` / `model.lock.json`;
  4. afirmacoes de prosa: cada par (trecho do card, trecho da fonte) precisa existir literalmente nos dois arquivos;
  5. frases obrigatorias (rotulo "avaliacao interna sobre dados sinteticos", homologacao nao independente, uma semente, PREVIEW privado);
  6. homologacao v1 (fora da distribuicao) com os pesos v3: tabelas e blocos de leitura do card == renderizacao de
     `reports/datasets/homolog-v3-eval-t{0,1}.json` (falha se qualquer numero divergir); `model-index` da homologacao == o mesmo JSON;
     o JSON bate com `homolog-v1-manifest.json` (sha do conjunto selado, n, 28 schemas), com as medidas do val v2c do card
     e com `reports/student/v3_decision_t{0,1}.json` (campos inexistentes em `full`).
     Nunca le itens de `data/homolog`: so os agregados publicados em `reports/`.
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
    "public preview",
    "### Out-of-distribution validation",
    "Do not trust field names unless the schema is in the context",
    "Validate field names before running",
    "is not fully independent of the v3 training data",
    "probably overstates",
    "not written by people",
    "single training run",
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

PREV_LABEL = "Previous internal checkpoint, v1 (not released)"
HOM_CTX = ("full", "irrelevant", "none")


@dataclass
class Tier:
    key: str
    name: str
    ws: str                      # workspace relativo a ROOT
    run: str                     # run do modelo publicado (sft_v3)
    rows: tuple                  # (rotulo, run medido, chave no JSON da homologacao) na ordem das tabelas
    hom_json: str                # reports/datasets/homolog-v3-eval-tN.json
    decision_json: str           # reports/student/v3_decision_tN.json
    claims: list = field(default_factory=list)   # (trecho do card, arquivo fonte, trecho da fonte)

    @property
    def card(self) -> Path:
        return ROOT / self.ws / "hf" / "README.md"

    @property
    def eval_dir(self) -> Path:
        return ROOT / self.ws / "reports" / "eval"

    @property
    def base_hom(self) -> str:
        return self.rows[0][2]

    @property
    def prev_hom(self) -> str:
        return self.rows[1][2]

    @property
    def mdl_hom(self) -> str:
        return self.rows[2][2]


RESUMO = "docs/resumo-executivo.md"
CORPUS3 = "reports/student/corpus_v3_report.md"
HOM_MD = "reports/datasets/homolog-v1-eval.md"
HOM_MANIFEST = "reports/datasets/homolog-v1-manifest.json"
TEACHER = "Repos/KapiTeacher-Mongo-7B/reports/eval/sft_v2d-final/summary.json"
ONNX_KAPI = "{ws}/exports/{pkg}-ONNX-int4/kapibarastudio-model.json"

COMMON_CLAIMS = [
    ("110 schemas in 14 domains", RESUMO, "110 schemas sintéticos em 14 domínios"),
    ("61 synthetic v3 training schemas", CORPUS3, "gramatica dos 61 schemas v3 `train`"),
    ("(MIT, Apache-2.0, BSD, ISC)", "data/replay/v1/LICENSES.md", "permissivas (MIT, Apache-2.0, BSD-2/3-Clause, ISC)"),
    ("0.6255", RESUMO, "O teto do validador é 0,6255"),
    ("Hui et al.", "BaseRepos/Qwen2.5-Coder-0.5B/README.md", "arXiv:2409.12186"),
    ("The 7B teacher", TEACHER, '"normalized_match": 0.41'),
    ("LLM + grammar = 42.0%", CORPUS3, "LLM + gramatica v3 = **42.0%**"),
    ("58.0%", CORPUS3, "| accepted-v2c-teacher | teacher_seq (fim_raw: gold) |"),
    ("26.0%", CORPUS3, "| llm-qwen3-14b | gold |"),
    ("16.0%", CORPUS3, "| grammar-v3 | gold |"),
    ("12.0% of the tokens", CORPUS3, "= **12.0%** dos tokens"),
    ("first training run decontaminated against the sealed homologation set", CORPUS3, "o v3 e o primeiro treino decontaminado contra a homologacao"),
    ("may be slightly optimistic", CORPUS3, "a homologacao v1 deles pode estar levemente otimista"),
    ("hyper-parameters are those of the previous run", "{ws}/configs/sft_v3.yaml", "Hiperparametros IDENTICOS ao sft_distill_v1.yaml"),
    ("prints `'total'`", "reports/publish/example_fim_run.txt", "saida: 'total'"),
    ("same SHA-256, see `provenance/model_lock.json`", "BaseRepos/base-repos.lock.json", "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"),
    ("seed 20261018", "{ws}/configs/sft_v3.yaml", "seed: 20261018"),
    ("600 examples of this validation split", "{ws}/configs/sft_v3.yaml", "600 FIM do val v2c"),
    ("0 write executions", "{ws}/reports/eval/{run}/measures.json", '"write_or_destructive_executions": 0'),
    ("2 s limit", "{ws}/reports/eval/{run}/summary.json", '"max_time_ms": 2000'),
    ("seed 20261008", "{ws}/reports/eval/{run}/summary.json", '"seed": 20261008'),
    ("AMD Radeon RX 7800 XT", "{ws}/reports/eval/{run}/summary.json", '"device_name": "AMD Radeon RX 7800 XT"'),
    ("One AMD Radeon RX 7800 XT", "{ws}/reports/{run}/run_manifest.json", "7800 XT"),
    ("transformers 4.57.6", "{ws}/reports/{run}/run_manifest.json", '"transformers": "4.57.6"'),
    ("up to 96 new tokens", "{ws}/reports/eval/{run}/summary.json", '"max_new_tokens": 96'),
    ("2048-token context", "{ws}/reports/eval/{run}/summary.json", '"context_tokens": 2048'),
    ("(the IDE caps it at 32 tokens)", ONNX_KAPI, '"maxTokens": 32'),
    ("`Mongo Console JavaScript` and `json`", "reports/datasets/v2c-validation-report.md", "dialetos {'console': 889205, 'json': 111410}"),
]

NANO_CLAIMS = [
    ("159,865 examples", "Repos/KapiCoder-Mongo-Nano/model.lock.json", '"examples": 159865'),
    ("225,835 MongoDB examples + 12,618", CORPUS3, "(225,835 cortes Mongo + 12,618 replay)"),
    ("39.1% of the MongoDB cuts", CORPUS3, "**none+irrelevant = 39.1%**"),
    ("130,984", CORPUS3, "| accepted-v2c-teacher | teacher_seq (fim_raw: gold) | 130,984 |"),
    ("58,808", CORPUS3, "| llm-qwen3-14b | gold | 58,808 |"),
    ("36,043", CORPUS3, "| grammar-v3 | gold | 36,043 |"),
    ("full fine-tuning", "{ws}/configs/sft_v3.yaml", "method: full"),
    ("peak LR 2e-5", "{ws}/configs/sft_v3.yaml", "lr: 2.0e-5"),
    ("4.38 hours", "{ws}/reports/{run}/run_manifest.json", '"elapsed_train_h": 4.381'),
]

SMALL_CLAIMS = [
    ("229,834 examples", "Repos/KapiCoder-Mongo-Small/model.lock.json", '"examples": 229834'),
    ("292,812 MongoDB examples + 16,759", CORPUS3, "(292,812 cortes Mongo + 16,759 replay)"),
    ("34.8% of the MongoDB cuts", CORPUS3, "**none+irrelevant = 34.8%**"),
    ("169,831", CORPUS3, "| accepted-v2c-teacher | teacher_seq (fim_raw: gold) | 169,831 |"),
    ("76,248", CORPUS3, "| llm-qwen3-14b | gold | 76,248 |"),
    ("46,733", CORPUS3, "| grammar-v3 | gold | 46,733 |"),
    ("LoRA (r=64", "{ws}/configs/sft_v3.yaml", "r: 64"),
    ("alpha=128, dropout 0.05", "{ws}/configs/sft_v3.yaml", "alpha: 128"),
    ("peak LR 1.5e-4", "{ws}/configs/sft_v3.yaml", "lr: 1.5e-4"),
    ("dropout 0.05", "{ws}/configs/sft_v3.yaml", "dropout: 0.05"),
    ("15.63 hours", "{ws}/reports/{run}/run_manifest.json", '"elapsed_train_h": 15.63'),
    ("cut to 1.5 epochs", "{ws}/configs/sft_v3.yaml", "1,5 epoca"),
]

TIERS = {
    "nano": Tier("nano", "KapiCoder-Mongo-Nano", "Repos/KapiCoder-Mongo-Nano", "sft_v3",
                 (("Qwen2.5-Coder-0.5B, zero-shot base", "base-qwen2.5-coder-0.5b", "base_0.5b"),
                  (PREV_LABEL, "sft_distill_v1", "t0_distill"),
                  ("**KapiCoder-Mongo-Nano** (this model, v3)", "sft_v3", "t0_v3")),
                 "reports/datasets/homolog-v3-eval-t0.json", "reports/student/v3_decision_t0.json", NANO_CLAIMS),
    "small": Tier("small", "KapiCoder-Mongo-Small", "Repos/KapiCoder-Mongo-Small", "sft_v3",
                  (("Qwen2.5-Coder-1.5B, zero-shot base", "base-qwen2.5-coder-1.5b", "base_1.5b"),
                   (PREV_LABEL, "sft_distill_v1", "t1_distill"),
                   ("**KapiCoder-Mongo-Small** (this model, v3)", "sft_v3", "t1_v3")),
                  "reports/datasets/homolog-v3-eval-t1.json", "reports/student/v3_decision_t1.json", SMALL_CLAIMS),
}
SIBLING = {"nano": "Small", "small": "Nano"}


# ------------------------------------------------------------------ homologacao v1 com os pesos v3
HOM_CLAIMS = [
    ("28 schemas", HOM_MD, "28 schemas `hom`"),
    ("Qwen3-14B", HOM_MD, "Autor Qwen3-14B / gramatica"),
    ("`aca0aae5`", HOM_MANIFEST, '"set_sha256": "aca0aae5'),
    ("telecom and adtech", HOM_MD, "telecom/adtech"),
    ("418 items", HOM_MD, "418 itens"),
    ("smallest `cut_kind` has n=269", HOM_MD, "menor `cut_kind` tem n=269"),
    ("single greedy run", HOM_MD, "Greedy unico"),
    ("McNemar", HOM_MD, "McNemar exato"),
    ("95% CI", HOM_MD, "Wilson (FIM) e bootstrap"),
    ("no write command was forwarded", HOM_MD, "nenhum comando de escrita encaminhado"),
    ("RX 7800 XT", HOM_MD, "Todas as geracoes na RX 7800 XT"),
    ("greedy decoding, up to 96 new tokens", HOM_MD, "greedy, 96 tokens, contexto 2048"),
]

_CACHE: dict = {}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def hom(tier: "Tier") -> dict:
    key = ("hom", tier.hom_json)
    if key not in _CACHE:
        _CACHE[key] = read_json(ROOT / tier.hom_json)
    return _CACHE[key]


def hp(x: float) -> str:
    """fracao -> percentual com 2 casas (exato ao JSON, que guarda 4 casas)."""
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


def hom_rows(tier: "Tier") -> list[tuple[str, str]]:
    return [(label, hk) for label, _run, hk in tier.rows]


def hpair(tier: "Tier", b: str, a: str, metric: str, dim: str | None = None, group: str | None = None) -> dict:
    """Linha B-A de `pairs[<b>_vs_<a>]` (delta = B - A); geral ou de um grupo de `dim`."""
    node = hom(tier)["pairs"][f"{b}_vs_{a}"]
    rows = node["context_kind"]["overall"] if group is None else node[dim]["groups"][group]["metrics"]
    for r in rows:
        if r["metric"] == metric:
            return r
    raise KeyError((b, a, metric, dim, group))


def hdelta(r: dict) -> str:
    return f"{hd(r['delta'])} pp [{hd(r['ci_low'])}, {hd(r['ci_high'])}]"


def hpval(r: dict) -> str:
    p = r.get("p_holm", r.get("p"))
    return "<0.001" if p < 0.001 else f"={p:.3f}"


def render_hom_table(tier: "Tier") -> str:
    H = hom(tier)["models"]
    lines = ["| Model | FIM acc (%) [95% CI] | Exec-valid (%) | Exec ÷ ceiling (%) [95% CI] | Syntax-valid (%) |",
             "|---|---|---:|---|---:|"]
    for label, k in hom_rows(tier):
        o = H[k]["overall"]
        lines.append(f"| {label} | {hp(o['normalized_match'])} {hci(o['normalized_match_ci95'])} | {hp(o['exec_valid'])} | "
                     f"{hp(o['exec_over_ceiling'])} {hci(o['exec_over_ceiling_ci95'])} | {hp(o['syntax_valid'])} |")
    o = H[tier.mdl_hom]["overall"]
    lines += ["", f"Internal evaluation, synthetic out-of-distribution homologation, n={hnum(o['n'])}. Validator ceiling for exec-valid: {hp(o['ceiling_exec_valid'])}%."]
    return "\n".join(lines)


def render_hom_fields_table(tier: "Tier") -> str:
    H = hom(tier)["models"]
    ref = H[tier.mdl_hom]["by_context_kind"]
    head = "| Nonexistent fields (%) by `context_kind`, homologation | " + " | ".join(f"`{c}` (n={hnum(ref[c]['n'])})" for c in HOM_CTX) + " |"
    lines = [head, "|---|" + "---:|" * len(HOM_CTX)]
    for label, k in hom_rows(tier):
        lines.append("| " + label + " | " + " | ".join(hp(H[k]["by_context_kind"][c]["nonexistent_fields_rate"]) for c in HOM_CTX) + " |")
    return "\n".join(lines)


def hom_gap(tier: "Tier", k: str) -> dict:
    d = hom(tier)
    h, v = d["models"][k]["overall"], d["val_v2c"][k]
    return {"fim": (v["normalized_match"], h["normalized_match"]),
            "exec": (v["exec_over_ceiling"], h["exec_over_ceiling"]),
            "full": (v["nonexistent_fields_full"], d["models"][k]["by_context_kind"]["full"]["nonexistent_fields_rate"])}


def render_hom_gap_table(tier: "Tier") -> str:
    lines = ["| Model | FIM acc (%): validation → homologation (Δ pp) | Exec ÷ ceiling (%): validation → homologation (Δ pp) | Nonexistent fields `full` (%): validation → homologation (Δ pp) |",
             "|---|---|---|---|"]
    for label, k in hom_rows(tier):
        g = hom_gap(tier, k)
        cells = [f"{hp(a)} → {hp(b)} ({(b - a) * 100:+.2f})" for a, b in (g["fim"], g["exec"], g["full"])]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    vc, hc = hom(tier)["val_v2c"][tier.mdl_hom]["ceiling_exec_valid"], hom(tier)["models"][tier.mdl_hom]["overall"]["ceiling_exec_valid"]
    lines += ["", f"The exec ÷ ceiling column compares two validators (validation: ceiling {hp(vc)}%, homologation: ceiling {hp(hc)}%); read trends, not decimals."]
    return "\n".join(lines)


def hom_vals(tier: "Tier") -> dict:
    """Todos os numeros usados nos blocos de leitura, tirados do JSON."""
    H = hom(tier)["models"]
    base, prev, mdl = tier.base_hom, tier.prev_hom, tier.mdl_hom
    o = lambda k: H[k]["overall"]  # noqa: E731
    f = lambda k, c: H[k]["by_context_kind"][c]["nonexistent_fields_rate"]  # noqa: E731
    return {"base": base, "prev": prev, "mdl": mdl,
            "gap": hom_gap(tier, mdl), "gap_prev": hom_gap(tier, prev), "gap_base": hom_gap(tier, base),
            "o_m": o(mdl), "o_b": o(base), "o_p": o(prev),
            "f_m": {c: f(mdl, c) for c in HOM_CTX}, "f_b": {c: f(base, c) for c in HOM_CTX}, "f_p": {c: f(prev, c) for c in HOM_CTX},
            "val_full_m": hom(tier)["val_v2c"][mdl]["nonexistent_fields_full"],
            "au_m": {a: H[mdl]["by_author"][a] for a in ("grammar", "llm")},
            "au_p": {a: H[prev]["by_author"][a] for a in ("grammar", "llm")},
            "au_b": {a: H[base]["by_author"][a] for a in ("grammar", "llm")},
            "ck_stage": H[mdl]["by_cut_kind"]["stage"]["normalized_match"],
            "ck_multiline": H[mdl]["by_cut_kind"]["multiline"]["normalized_match"],
            "n_total": o(mdl)["n"],
            "vb_fim": hpair(tier, mdl, base, "normalized_match"), "vb_exec": hpair(tier, mdl, base, "exec_valid"),
            "vb_clean_full": hpair(tier, mdl, base, "no_nonexistent_field", "context_kind", "full"),
            "vp_fim": hpair(tier, mdl, prev, "normalized_match"), "vp_exec": hpair(tier, mdl, prev, "exec_valid"),
            "vp_clean_full": hpair(tier, mdl, prev, "no_nonexistent_field", "context_kind", "full"),
            "vb_author": {a: hpair(tier, mdl, base, "normalized_match", "author", a) for a in ("grammar", "llm")},
            "vp_author": {a: hpair(tier, mdl, prev, "normalized_match", "author", a) for a in ("grammar", "llm")}}


def vs_base_phrase(v: dict) -> tuple[str, str]:
    """(curta, longa): posicao do modelo v3 frente a base em campos inexistentes (`full`), decidida pelo IC95 pareado."""
    cf = v["vb_clean_full"]
    stat = f"paired difference in items without a nonexistent field, `full`: {hdelta(cf)}, Holm p{hpval(cf)}"
    # `no_nonexistent_field` = taxa de itens SEM campo inexistente: delta negativo = mais campos inventados que a base
    if cf["ci_high"] < 0:
        return ("still above the base", f"above its base ({stat}; negative means more invented fields than the base)")
    if cf["ci_low"] > 0:
        return ("below the base", f"below its base ({stat})")
    return ("close to the base", f"close to its base ({stat}; the interval includes zero, and no equivalence test was run)")


def render_caveats(tier: "Tier") -> str:
    v = hom_vals(tier)
    full = v["f_m"]["full"]
    L = ["**Read this first**", ""]
    L.append("- **Internal evaluation, synthetic data, no human-written queries.** Every number in this card comes from the authors' own evaluation on synthetic data "
             "(a validation split of the training generator and a homologation set written by an LLM and a grammar); nothing was measured on queries written by people.")
    L.append(f"- **Invented field names on unseen schemas.** With the schema visible (`full`), {hp(full)}% of this model's suggestions on the homologation set use a field that does not exist, "
             f"against {hp(v['f_b']['full'])}% for the zero-shot base and {hp(v['f_p']['full'])}% for the previous internal checkpoint (not released): {vs_base_phrase(v)[0]}. "
             "Validate field names before running a suggestion.")
    L.append("- **The homologation set is not fully independent of the v3 training data.** The v3 corpus contains queries from the same LLM generator (Qwen3-14B) and grammar templates over other schemas "
             "(training cuts that shared long n-grams with the set were removed by decontamination, and its 28 schemas were never in training), and the set is LLM- and grammar-written, not written by people: "
             "the gain on LLM-written queries probably overstates the gain on queries written by people.")
    L.append("- **A single training run per model** (one seed) and a single greedy evaluation run: there is no estimate of variance between runs.")
    return "\n".join(L)


def render_hom_reading(tier: "Tier") -> str:
    v = hom_vals(tier)
    g, gp, gb = v["gap"], v["gap_prev"], v["gap_base"]
    full, irr, none = (v["f_m"][c] for c in HOM_CTX)
    bfull, birr, bnone = (v["f_b"][c] for c in HOM_CTX)
    pfull = v["f_p"]["full"]
    A, B = v["au_m"]["grammar"], v["au_m"]["llm"]
    ga, lb = v["vb_author"]["grammar"], v["vb_author"]["llm"]
    pa, pl = v["vp_author"]["grammar"], v["vp_author"]["llm"]
    vs_base = vs_base_phrase(v)[1]
    up = g["fim"][1] >= g["fim"][0]
    L = ["Reading the homologation tables:", ""]
    L.append(f"- **Generalisation gap.** From the synthetic validation set to unseen schemas, FIM accuracy goes {hp(g['fim'][0])}% → {hp(g['fim'][1])}% "
             f"({(g['fim'][1] - g['fim'][0]) * 100:+.2f} pp) and exec ÷ ceiling {hp(g['exec'][0])}% → {hp(g['exec'][1])}% ({(g['exec'][1] - g['exec'][0]) * 100:+.2f} pp). "
             f"For the previous internal checkpoint (v1) the same step was {hp(gp['fim'][0])}% → {hp(gp['fim'][1])}% in FIM accuracy and "
             f"{hp(gp['exec'][0])}% → {hp(gp['exec'][1])}% in exec ÷ ceiling. "
             + ("FIM accuracy is higher here than on the validation set, which is not evidence of better generalisation: see the next sections: the homologation is not fully independent of the v3 training data. "
                if up else "")
             + f"The zero-shot base model goes {hp(gb['fim'][0])}% → {hp(gb['fim'][1])}%, so the homologation queries are easier for a model that was not trained here (interpretation, not tested).")
    L.append("- **Field names are the main risk. Do not trust field names unless the schema is in the context, and validate a suggestion before running it.** "
             f"On schemas never seen in training this model still invents field names at a rate {vs_base}: {hp(full)}% of suggestions with the schema visible (`full`) "
             f"against {hp(bfull)}% for the zero-shot base and {hp(pfull)}% for the previous internal checkpoint (on the synthetic validation set it was {hp(v['val_full_m'])}%). "
             f"Without the schema the rates are {hp(irr)}% (`irrelevant`) and {hp(none)}% (`none`), against {hp(birr)}% and {hp(bnone)}% for the base. "
             f"Having the schema in the header did not lower the rate here ({hp(full)}% with it, {hp(none)}% without); why is a hypothesis that was not tested. "
             f"The rate is counted only on suggestions whose command reached the server, and the base model more often fails to parse (syntax-valid {hp(v['o_b']['syntax_valid'])}% "
             f"against {hp(v['o_m']['syntax_valid'])}%), so the base rates cover fewer suggestions and are not directly comparable.")
    L.append(f"- **Against the base and the previous checkpoint** (same {hnum(v['n_total'])} items, paired): over the zero-shot base, FIM accuracy {hdelta(v['vb_fim'])} and exec-valid {hdelta(v['vb_exec'])}; "
             f"over the previous internal checkpoint, FIM accuracy {hdelta(v['vp_fim'])}, exec-valid {hdelta(v['vp_exec'])} and items without a nonexistent field in `full` {hdelta(v['vp_clean_full'])} "
             "(95% CIs, Holm-adjusted McNemar tests). The previous checkpoint is not released.")
    L.append(f"- **`grammar` against `llm`.** The gain over the zero-shot base in FIM accuracy is {hdelta(ga)} on `grammar` and {hdelta(lb)} on `llm`; exec ÷ ceiling is "
             f"{hp(A['exec_over_ceiling'])}% on `grammar` and {hp(B['exec_over_ceiling'])}% on `llm` (previous checkpoint: {hp(v['au_p']['grammar']['exec_over_ceiling'])}% and {hp(v['au_p']['llm']['exec_over_ceiling'])}%). "
             f"The gain of the v3 model over the previous checkpoint in FIM accuracy is {hdelta(pa)} on `grammar` and {hdelta(pl)} on `llm`. "
             "Both parts of the set resemble what the v3 corpus was built from (queries from the same LLM and grammar templates over other schemas), so these are not independent tests of style; "
             "the gain on LLM-written queries probably overstates the gain on queries written by people.")
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
            f"(mesma família Qwen dos modelos). **A homologação não é totalmente independente do treino v3** (o corpus inclui consultas do mesmo gerador LLM e templates de gramática de outros schemas), "
            f"então o ganho em consultas escritas pelo LLM provavelmente superestima o ganho em consultas humanas. FIM acc {t(g['fim'][1])}% (validação sintética: {t(g['fim'][0])}%; base: {t(v['o_b']['normalized_match'])}%; "
            f"checkpoint interno anterior, não publicado: {t(v['o_p']['normalized_match'])}%); exec ÷ teto {t(g['exec'][1])}% (validação: {t(g['exec'][0])}%, com validadores diferentes). "
            f"**Alerta:** com o schema visível (`full`), {t(full)}% das sugestões usam campos inexistentes, contra {t(bfull)}% da base zero-shot (validação sintética: {t(v['val_full_m'])}%); "
            f"sem schema, {t(irr)}% (`irrelevant`) e {t(none)}% (`none`). Valide os nomes de campo antes de executar e não confie neles sem o schema no contexto. "
            "Executar não é acertar a intenção.")


HOM_BLOCKS = {"homolog-table": render_hom_table, "homolog-fields-table": render_hom_fields_table,
              "homolog-gap-table": render_hom_gap_table, "key-caveats": render_caveats, "homolog-reading": render_hom_reading, "homolog-pt": render_hom_pt}


def check_homolog_sources(tier: "Tier") -> list[str]:
    """Consistencia interna das fontes da homologacao (JSON x manifesto selado x decisao v3)."""
    errs: list[str] = []
    d = hom(tier)
    man = read_json(ROOT / HOM_MANIFEST)
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
        if d["models"][tier.mdl_hom]["by_context_kind"][c]["n"] != n:
            errs.append(f"homolog: n de context_kind={c} difere do manifesto")
    dec = read_json(ROOT / tier.decision_json)["numbers"]["full_nonexistent_pct"]
    d_ = d["models"]
    for label, got, want in (("v3 full", d_[tier.mdl_hom]["by_context_kind"]["full"]["nonexistent_fields_rate"] * 100, dec["v3"]),
                             ("base full", d_[tier.base_hom]["by_context_kind"]["full"]["nonexistent_fields_rate"] * 100, dec["base"]),
                             ("v1 full", d_[tier.prev_hom]["by_context_kind"]["full"]["nonexistent_fields_rate"] * 100, dec["current"])):
        if abs(got - float(want)) > 0.006:
            errs.append(f"homolog: {label}={got:.2f}% difere de {tier.decision_json} ({want}%)")
    return errs


# ------------------------------------------------------------------ utilidades
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
    return ROOT / template.format(ws=tier.ws, run=tier.run, pkg=tier.name)


# ------------------------------------------------------------------ tabelas
def render_accuracy_table(tier: Tier) -> str:
    head = "| Model | " + " | ".join(c[0] for c in ACC_COLUMNS) + " |"
    sep = "|---|" + "---:|" * len(ACC_COLUMNS)
    lines = [head, sep]
    for label, run, _hk in tier.rows:
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
    for label, run, _hk in tier.rows:
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
        text = text[:i] + b + "\n" + fn(tier) + "\n" + text[j:]
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


def run_manifest(tier: Tier) -> dict:
    return read_json(ROOT / tier.ws / "reports" / tier.run / "run_manifest.json")


def facts(tier: Tier) -> list[tuple[str, list[str]]]:
    """(rotulo, [trechos que precisam existir no card]) derivados dos arquivos de fonte."""
    lock = read_json(ROOT / tier.ws / "model.lock.json")
    rm = run_manifest(tier)
    ds = rm["dataset"]
    s = summary(tier, tier.run)
    out = [
        ("base revision", [lock["base"]["revision"], rm["base_model"]["revision"]]),
        ("base repo", [lock["base"]["repo_id"]]),
        ("examples", [en_int(ds["examples"]), pt_int(ds["examples"])]),
        ("replay examples", [en_int(ds["replay"]["examples"]), pt_int(ds["replay"]["examples"])]),
        ("tokens per epoch", [en_int(ds["tokens_in_corpus"]), pt_int(ds["tokens_in_corpus"])]),
        ("tokens seen", [en_int(rm["tokens_seen"]), pt_int(rm["tokens_seen"])]),
        ("steps", [en_int(rm["total_steps"]), pt_int(rm["total_steps"])]),
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
            errors.append(f"[{key}] bloco {name} diverge de {tier.hom_json} (rode --write para ver a diferenca esperada)")
    errors += [f"[{key}] {e}" for e in check_homolog_sources(tier)]
    # o val v2c dentro do JSON da homologacao precisa ser o mesmo que o card traz na tabela sintetica
    hm = hom(tier)["val_v2c"]
    for _label, run, hk in tier.rows:
        m = measures(tier, run)
        for hkey, mkey, scale in (("normalized_match", "fim_acc", 1), ("exec_over_ceiling", "exec_valid_vs_ceiling", 1),
                                  ("nonexistent_fields_full", "nonexistent_fields_full_rate_pct", 0.01)):
            if abs(hm[hk][hkey] - m[mkey] * scale) > 1e-4:
                errors.append(f"[{key}] {run}: val_v2c.{hk}.{hkey}={hm[hk][hkey]} != measures.{mkey}={m[mkey]}")

    # coerencia cruzada summary x measures
    for _label, run, _hk in tier.rows:
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
    missing_tags = {"mongodb", "code", "fim", "autocomplete"} - set(fm.get("tags") or [])
    if missing_tags:
        errors.append(f"[{key}] tags ausentes: {sorted(missing_tags)}")
    if "onnx" in (fm.get("tags") or []):
        errors.append(f"[{key}] tag `onnx` no repositorio principal (so safetensors): o ONNX vai em repositorio separado")
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
                    ho = hom(tier)["models"][tier.mdl_hom]
                    hmap = {"fim_accuracy": ho["overall"]["normalized_match"], "exec_valid": ho["overall"]["exec_valid"],
                            "exec_valid_vs_ceiling": ho["overall"]["exec_over_ceiling"],
                            "nonexistent_fields_full_pct": ho["by_context_kind"]["full"]["nonexistent_fields_rate"] * 100}
                    if met["type"] not in hmap:
                        errors.append(f"[{key}] model-index homologacao: metrica desconhecida {met['type']}")
                        continue
                    seen.add("hom:" + met["type"])
                    if abs(float(met["value"]) - hmap[met["type"]]) > 1e-9:
                        errors.append(f"[{key}] model-index homologacao {met['type']}={met['value']} != {tier.hom_json} ({hmap[met['type']]})")
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
    for bad in ("state-of-the-art", "best-in-class", "outperforms", "beats GPT", "production-ready", "blazing", "superior", "excellent"):
        if bad.lower() in text.lower():
            errors.append(f"[{key}] linguagem nao sobria: {bad!r}")
    return errors


# ------------------------------------------------------------------ SOURCES.md
def render_sources(tier: Tier) -> str:
    lines = [f"# SOURCES - {tier.name} (apendice interno, NAO sobe para o Hugging Face)", "",
             "Cada numero ou afirmacao do `README.md` deste diretorio e conferido por `scripts/publish/verify_card_numbers.py`; esta tabela e gerada por ele",
             "(`--write-sources`). Caminhos relativos a raiz do workspace. Pesos publicados: `merged/sft_v3` (v3); o checkpoint v1 (`sft_distill_v1`) aparece so como linha comparativa, nao publicado.", "",
             "## Tabelas de metricas (celula = chave do JSON)", "",
             "| Linha | Arquivo (measures.json) | Chaves |", "|---|---|---|"]
    keys = ", ".join(f"`{k}`" for _h, k, _f in ACC_COLUMNS) + ", " + ", ".join(f"`{k}`" for _h, k in FIELD_COLUMNS)
    for label, run, _hk in tier.rows:
        lines.append(f"| {label} | `{tier.ws}/reports/eval/{run}/measures.json` | {keys} |")
    lines += ["", "Coerencia cruzada: `summary.json` (`headline.normalized_match`, `exec_valid`, `syntax_valid`) e `<run>-generic-js/summary.json` batem com `measures.json`.", "",
              "## Fatos de treino (v3)", "", "| Fato | Fonte |", "|---|---|",
              f"| base, revisao | `{tier.ws}/model.lock.json`, `{tier.ws}/reports/{tier.run}/run_manifest.json` |",
              f"| exemplos, replay, tokens por epoca, passos, tokens vistos, horas, metodo, parametros treinaveis, bibliotecas | `{tier.ws}/reports/{tier.run}/run_manifest.json` |",
              f"| hiperparametros, semente (20261018) | `{tier.ws}/configs/sft_v3.yaml` |",
              f"| pesos publicados (sha256) | `{tier.ws}/merged/{tier.run}/export_info.json` |",
              "| composicao do corpus v3 (58% destilado do professor, 26% Qwen3-14B gold, 16% gramatica v3 gold, replay 12%, none+irrelevant), decontaminacao contra a homologacao | `reports/student/corpus_v3_report.md` |",
              f"| decisao GO do retreino (campos inexistentes em `full`: antes -> v3 -> base) | `reports/student/v3_decision_t{0 if tier.key == 'nano' else 1}.{{md,json}}` |",
              "| dataset v2c (funil, dialetos, decontaminacao) | `reports/datasets/v2c-validation-report.md` |",
              "| decisoes e limites (cego descartado, teto 0,6255, 110 schemas) | `docs/resumo-executivo.md` |",
              "| professor (rotulos do corpus destilado) | `Repos/KapiTeacher-Mongo-7B/reports/eval/sft_v2d-final/summary.json` |",
              "| atribuicao do replay | `data/replay/v1/LICENSES.md` -> `REPLAY_ATTRIBUTION.md` |",
              "| base: licenca e revisao | `BaseRepos/Qwen2.5-Coder-*/README.md`, `BaseRepos/base-repos.lock.json` |",
              "| contrato `editor-context-v1` | `data/contracts/editor-context-v1.golden`, `scripts/data/render_contract.py` (exemplo do card testado contra o golden) |",
              "", "## Homologacao v1 com os pesos v3 (fora da distribuicao; autor Qwen3-14B/gramatica, nao humano)", "",
              f"Tabelas e blocos de leitura do card (`homolog-table`, `homolog-fields-table`, `homolog-gap-table`, `homolog-reading`, `homolog-pt`) e o `model-index` da homologacao sao renderizados/conferidos por `verify_card_numbers.py` a partir de `{tier.hom_json}` (agregados; nenhum item de `data/homolog` e lido). Percentuais com 2 casas = o valor de 4 casas do JSON.", "",
              f"| Celula do card | Chave em `{tier.hom_json}` |", "|---|---|",
              "| FIM acc [IC95] | `models.<m>.overall.normalized_match`, `normalized_match_ci95` |",
              "| Exec-valid, sintaxe | `models.<m>.overall.exec_valid`, `syntax_valid` |",
              "| Exec ÷ teto [IC95]; teto | `models.<m>.overall.exec_over_ceiling`, `exec_over_ceiling_ci95`, `ceiling_exec_valid` |",
              "| Campos inexistentes por `context_kind` | `models.<m>.by_context_kind.<c>.nonexistent_fields_rate` (conferido contra `reports/student/v3_decision_tN.json`) |",
              "| Lacuna val -> hom | `val_v2c.<m>.{normalized_match, exec_over_ceiling, nonexistent_fields_full}` vs `models.<m>.overall` / `by_context_kind.full` |",
              "| Deltas pareados, IC95, McNemar-Holm | `pairs.<b>_vs_<a>.context_kind.overall[]` e `pairs.<b>_vs_<a>.{context_kind,author}.groups.<g>.metrics[]` (`delta`, `ci_low`, `ci_high`, `p_holm`) |",
              "| `grammar` x `llm` | `models.<m>.by_author.<g>.exec_over_ceiling`; pares `author.groups` |",
              "| Conjunto selado (28 schemas, n=6.030, sha `aca0aae5...`, menor corte 269) | `reports/datasets/homolog-v1-manifest.json` (conferido contra o eval.json) |",
              "| Protocolo e limites (autor, executar != acertar, teto 93,1% com 418 itens, n por celula, greedy unico) | `reports/datasets/homolog-v1-eval.md` (protocolo da v1; o v3 reaproveita o mesmo conjunto e o mesmo harness, JSON `format: homolog-v1-eval-1`) |",
              "", "Os `v1-*-report.md` em `reports/datasets/` descrevem o dataset v1 e NAO foram usados para nenhum numero do card.", "",
              "## Afirmacoes de prosa (trecho do card <-> trecho da fonte)", "", "| Trecho no card | Fonte | Trecho na fonte |", "|---|---|---|"]
    for snippet, src, src_snippet in COMMON_CLAIMS + HOM_CLAIMS + tier.claims:
        path = src.format(ws=tier.ws, run=tier.run, pkg=tier.name)
        lines.append(f"| {snippet.replace('|', '\\|')} | `{path}` | {src_snippet.replace('|', '\\|')} |")
    lines += ["", "## Sem fonte numerica (descritivo; revisar)", "",
              "- **Homologacao nao totalmente independente do treino v3** (mensagem obrigatoria): decorre da composicao do corpus v3 (26% de consultas do Qwen3-14B, mesmo gerador do conjunto de homologacao, e 16% da gramatica, familia de templates da parte `grammar`), documentada em `corpus_v3_report.md`; os cortes de treino que compartilham n-gramas longos com a homologacao foram removidos por decontaminacao e os 28 schemas dela nunca entraram no treino. O quanto isso infla o resultado nao foi medido.",
              "- \"The gain on LLM-written queries probably overstates the gain on queries written by people\": inferencia; nao ha medida com consultas humanas.",
              "- Homologacao: \"interpretation, not tested\" (consultas mais faceis para quem nao treinou) e \"why is a hypothesis that was not tested\" (schema visivel nao reduziu a taxa) sao hipoteses marcadas como tal no card.",
              "- Homologacao: \"the base more often fails to parse, so base rates cover fewer suggestions\" decorre da definicao da fonte (campo inexistente so conta quando o comando chega ao servidor) e das taxas de sintaxe; o efeito sobre a taxa nao foi quantificado.",
              "- \"Single training run\" (uma semente por tier): a semente 20261018 esta no yaml; que nao houve repeticao e a ausencia de qualquer outro run v3 nos relatorios.",
              "- \"No personal data by construction\": README dos workspaces (\"dados sinteticos ... nao contem dados pessoais\") e rejeicao de 17 registros com padroes proibidos na etapa sanitize do v2c. Nao ha revisao humana (G3.7 nao feita); o card diz isso.",
              "- \"Behaviour around destructive operations was neither trained nor tested\": ausencia de qualquer teste desse tipo nos relatorios; so ha `write_or_destructive_executions: 0` (guarda de leitura).",
              "- \"Not evaluated: bias or toxicity, long-context beyond 2048, multi-turn, adversarial\": ausencia nos relatorios.",
              "- REPLAY_ATTRIBUTION.md lista os 555 pacotes de `data/replay/v1/LICENSES.md`; o replay do corpus v3 vem do mesmo `data/replay/v1` (`configs/student_corpus.yaml`, `replay.source_dir`), mas nao verifiquei pacote a pacote que o v3 usou todos (a lista pode ser um superconjunto do realmente usado).",
              "- Qual checkpoint do run (final ou `best`) foi mesclado em `merged/sft_v3` nao foi conferido; os hashes dos pesos publicados vem de `export_info.json`.",
              "- Nome do repositorio (`esilva/...`), contato e visibilidade: decisao do usuario (ver `MODEL_CARD_REVIEW.md`).",
              "- O exemplo do card foi executado localmente em CPU contra os pesos do staging (sem rede) e a funcao `build_prompt` e testada contra os SHA do golden.", ""]
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
