#!/usr/bin/env python3
"""Relatorio da gramatica v3 (reports/datasets/v3-grammar-report.md): funil de 9 etapas, taxa exec-valid, composicao e isolamento.

  env/.venv/Scripts/python.exe scripts/data/report_grammar_v3.py
Le data/generated/v3/grammar/{train,eval}.funnel.json e data/validated/v3g[/eval]. Nao le data/homolog.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from jsonl_shards import iter_jsonl  # noqa: E402

GEN = ROOT / "data" / "generated" / "v3" / "grammar"
STAGES = ["sanitize", "syntax", "api", "fields", "exec", "semantic", "budget", "dedup", "decontam"]


def pct(x, n) -> str:
    return f"{100 * x / max(1, n):.1f}%"


def dist(recs, key):
    c = Counter(r[key] for r in recs)
    n = sum(c.values())
    return ", ".join(f"{k} {100 * v / n:.1f}%" for k, v in sorted(c.items(), key=lambda kv: -kv[1]))


def main() -> int:
    L = ["# Gramatica dos schemas v3 (cortes FIM de treino e de avaliacao)", "",
         "Gerado por `scripts/data/report_grammar_v3.py` a partir de `scripts/data/build_grammar_v3.py` (rodado via `scripts/data/llm/run_guarded.py`; sem `--include-hom`). "
         "Seeds fixas: treino 20261024, avaliacao 20261025. Saidas: `data/generated/v3/grammar/` e `data/validated/v3g[/eval]/` (gitignored); ids em `data/splits/v3g[_eval]/`.", "",
         "Pipeline: schemas v3 por PAPEL (train: 61; eval: 15; hom: 28, nunca lidos) -> snippets (`gen_snippets`, `--seeded-only`) -> execucao somente-leitura dos snippets "
         "(`validate_snippets`) -> prefiltro de decontaminacao oficial (`ngram_min_content: 2`) contra val/test/bench v2c, v3-eval e a homologacao selada -> cortes FIM "
         "(`gen_fim`, 9 cut_kinds, ate 96 tokens Qwen, categorias com mais atlas_search/json_ejson/dsl/indexes) -> contexto `editor-context-v1` com `context_kind` "
         "CONSERVADOR (>= 25% none/irrelevant, so onde o rotulo usa apenas nomes visiveis) -> `validate_pipeline` (9 etapas, etapa 9 tambem contra as mesmas referencias).", ""]
    for role in ("train", "eval"):
        p = GEN / f"{role}.funnel.json"
        if not p.exists():
            continue
        f = json.loads(p.read_text(encoding="utf-8"))
        vdir = ROOT / "data" / "validated" / "v3g" / ("eval" if role == "eval" else "")
        recs = [r for q in sorted(vdir.glob("fim*.jsonl")) for r in iter_jsonl(q)] if vdir.exists() else []
        ex = f["snippet_exec"]
        reads = ex.get("ok", 0) + ex.get("error", 0)
        pf = f["prefilter"]
        mf = f["pipeline"]
        L += [f"## {'Treino (schemas `train`)' if role == 'train' else 'Avaliacao (schemas `eval`: so val/test/bench, NUNCA treino)'}", "",
              f"- Schemas: **{f['schemas']}**; snippets brutos: **{f['snippets_raw']:,}**.",
              f"- **Taxa exec-valid dos snippets** (leituras executadas no MongoDB, somente-leitura): **{100 * ex.get('ok', 0) / max(1, reads):.2f}%** "
              f"({ex.get('ok', 0):,} ok / {ex.get('error', 0):,} erro); estaticos (atlas) {ex.get('skipped_static', 0):,}; escritas nao executadas {ex.get('skipped_write', 0):,}.",
              f"- Prefiltro: entram {pf['in']:,}, ficam **{pf['kept']:,}**; descartes {pf['dropped']}; documentos de referencia {pf['eval_reference_docs']:,} {pf['by_source']}.",
              f"- Cortes FIM pedidos/gerados: {f['fim_cuts']:,}; contexto: {f['context']['by_context_kind']} "
              f"(none+irrelevant {100 * f['context']['none_or_irrelevant_share']:.1f}%, alvo conservador).", "",
              "Funil da pipeline de 9 etapas (registros rejeitados por etapa):", "", "| etapa | rejeitados |", "|---|---:|"]
        L += [f"| {s} | {mf['rejections'].get(s, 0):,} |" for s in STAGES]
        n_in = f["fim_cuts"]
        n_out = sum(mf["counts"].get(k, 0) for k in ("train", "val", "test", "bench"))
        L += ["", f"Entrada {n_in:,} -> **validados {n_out:,}** ({pct(n_out, n_in)}); por split {{train {mf['counts']['train']:,}, val {mf['counts']['val']:,}, "
              f"test {mf['counts']['test']:,}, bench {mf['counts']['bench']:,}}}; modo de execucao da etapa 5: `{mf['exec_mode']}`.",
              f"- Decontaminacao etapa 9: removidos {mf['decontam']['removed']:,}; por regra {mf['decontam']['by_rule']}.",
              f"- Gates (padrao da pipeline v2c; G3.3 de volume/pisos nao se aplica a um lote parcial): {mf['gates']}.", ""]
        if recs:
            L += [f"Composicao validada ({len(recs):,} cortes): cut_kind {dist(recs, 'cut_kind')}.", "",
                  f"categoria: {dist(recs, 'category')}.", "", f"context_kind: {dist(recs, 'context_kind')}; dominios: {dist(recs, 'domain')}.", "",
                  f"Schemas distintos: {len({r['schema_id'] for r in recs})}; snippets distintos: {len({r['snippet_id'] for r in recs}):,}; "
                  f"hom na saida: {sum(1 for r in recs if r['domain'] in ('telecom', 'adtech'))}.", ""]
    L += ["## Isolamento e determinismo", "",
          "- G2.7: treino so de schemas `train` (split `train`); avaliacao so de schemas `eval` (val/test/bench); nenhum schema nos dois lados. G2.7-hom: schemas `hom` nunca copiados ao stage "
          "(`assert` + `hom_guard` por registro) e `validate_pipeline` aborta se vir um.",
          "- Decontaminacao oficial (`ngram_min_content: 2`) do treino contra val/test/bench v2c (cache `data/generated/v3/_eval_docs_v2c.jsonl`), v3-eval e a homologacao selada. "
          "A leitura da homologacao usa `scripts/data/homolog_refs.py` (API autorizada, log em `access.log.jsonl`); o `llm_to_fim.homolog_docs()` procura o arquivo no caminho errado "
          "(`data/homolog/v1/items` em vez de `data/homolog/v1/v1/items`) e devolve lista vazia sem erro - ver `reports/student/corpus_v3_report.md`.",
          "- Reprodutibilidade: seeds fixas, `par.ordered_map` (saida identica para qualquer numero de workers); teste `scripts/data/tests/test_grammar_v3.py`.", ""]
    out = ROOT / "reports" / "datasets" / "v3-grammar-report.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
