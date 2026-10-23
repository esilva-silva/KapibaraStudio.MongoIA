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
  5. frases obrigatorias (rotulo "avaliacao interna sobre dados sinteticos", TBD da homologacao, PREVIEW).
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
REQUIRED_PHRASES = (
    REQUIRED_EN,
    REQUIRED_PT,
    "PREVIEW / experimental",
    "### Out-of-distribution validation",
    "**TBD.**",
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
    for name, fn in TABLES.items():
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
            for met in res["metrics"]:
                if met.get("verified") is not False:
                    errors.append(f"[{key}] model-index metrica {met['type']} sem `verified: false`")
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

    # 3. fatos
    for label, needles in facts(tier):
        for n in needles:
            if n not in text:
                errors.append(f"[{key}] fato '{label}': trecho '{n}' ausente do card")

    # 4. afirmacoes de prosa
    for snippet, src, src_snippet in COMMON_CLAIMS + tier.claims:
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
              "", "Os `v1-*-report.md` em `reports/datasets/` descrevem o dataset v1 e NAO foram usados para nenhum numero do card.", "",
              "## Afirmacoes de prosa (trecho do card <-> trecho da fonte)", "", "| Trecho no card | Fonte | Trecho na fonte |", "|---|---|---|"]
    for snippet, src, src_snippet in COMMON_CLAIMS + tier.claims:
        path = src.format(ws=tier.ws, run=tier.run, pkg=tier.name)
        lines.append(f"| {snippet.replace('|', '\\|')} | `{path}` | {src_snippet.replace('|', '\\|')} |")
    lines += ["", "## Sem fonte numerica (descritivo; revisar)", "",
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
