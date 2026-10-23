#!/usr/bin/env python3
"""Piloto da geracao por LLM: gera ~300 consultas (4 por chamada, persona/tarefa/estilo variados), valida por execucao do texto, converte em FIM,
mede diversidade contra o corpus de gramatica v2c e escreve reports/datasets/llm-gen-pilot.md.

  env/.venv/Scripts/python.exe scripts/data/llm/pilot.py --target 300 --hom-target 100 --out data/generated/v3/pilot [--phase gen|validate|fim|hom|report|all]

Fases independentes e retomaveis (artefatos em <out>/ e <out>_hom/). Nao inicia geracao em volume: `--target` e o piloto.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from common import ROOT, load_cfg, load_schema_infos, read_jsonl, rel, sha, write_jsonl
import metrics as M


def phase_gen(a, cfg, out: Path, role: str, target: int, extra=()):
    import gen_llm_queries as G
    argv = ["--role", role, "--target", str(target), "--out", rel(out), "--resume"] + list(extra)
    rc = G.run(G.argparse.Namespace(config=None, role=role, target=target, out=rel(out), seed=None, schemas=None, limit_schemas=None, tasks=None,
                                    per_call=None, server_url=a.server_url, resume=True))
    if rc:
        raise SystemExit(rc)


def phase_validate(a, cfg, out: Path):
    import validate_text as V
    t0 = time.time()
    summ = V.validate(out / "queries.raw.jsonl", out, cfg, workers=a.workers)
    summ["wall_seconds"] = round(time.time() - t0, 1)
    (out / "validation.summary.json").write_text(json.dumps(summ, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


def phase_fim(a, cfg, out: Path):
    import llm_to_fim as L
    import homolog_guard as hg
    hg.enforce()
    t0 = time.time()
    st = L.build(read_jsonl(out / "queries.validated.jsonl"), "train", out, cfg)
    st["wall_seconds"] = round(time.time() - t0, 1)
    (out / "fim_ctx.manifest.json").write_text(json.dumps(st, indent=1, ensure_ascii=False), encoding="utf-8")
    hg.uninstall()


def phase_hom(a, cfg, hom_out: Path):
    """Ensaio do conjunto de homologacao com as consultas hom do piloto (sem selar, fora de data/homolog)."""
    import build_homolog as B
    ns = B.argparse.Namespace(config=None, queries=[rel(hom_out / "queries.validated.jsonl")], grammar=a.hom_grammar, n=a.hom_n,
                              cuts_per_snippet=30, max_per_query=14, seed=None, dry_run=True, train_schemas=False)
    rc = B.run(ns)
    if rc:
        print("[aviso] ensaio da homologacao nao rodou (rc=%s)" % rc)


# ----------------------------------------------------------------------------------------------------------------------------
def pct(a, b):
    return f"{100.0 * a / b:.1f}%" if b else "n/d"


def trunc(t: str, n=900) -> str:
    t = t.replace("\r\n", "\n")
    return t if len(t) <= n else t[:n] + "\n// ... (truncado)"


def fence(t: str) -> str:
    return "```js\n" + trunc(t) + "\n```"


def pick_stratified(rows, key, n, seed=11):
    by = defaultdict(list)
    for r in sorted(rows, key=lambda r: sha(f"{seed}:{r['id']}")):
        by[r.get(key)].append(r)
    out, i = [], 0
    keys = sorted(by, key=lambda k: sha(f"{seed}:{k}"))
    while len(out) < n and any(by.values()):
        k = keys[i % len(keys)]
        if by[k]:
            out.append(by[k].pop(0))
        i += 1
        if i > 10000:
            break
    return out


def phase_report(a, cfg, out: Path, hom_out: Path):
    gen = json.loads((out / "gen_summary.json").read_text(encoding="utf-8"))
    val = json.loads((out / "validation.summary.json").read_text(encoding="utf-8"))
    fim = json.loads((out / "fim_ctx.manifest.json").read_text(encoding="utf-8"))
    raw = read_jsonl(out / "queries.raw.jsonl")
    acc = read_jsonl(out / "queries.validated.jsonl")
    rej = read_jsonl(out / "queries.rejected.jsonl")
    calls = read_jsonl(out / "calls.jsonl")
    fn = val["funnel"]
    n_raw = fn["raw"]
    ex_cfg = cfg["execution"]
    # ---- diversidade
    llm_m = M.metrics([r["text"] for r in acc])
    n_ref = len(acc)          # mesmo n nos dois lados (as contagens de tipos distintos dependem de n)
    all_c = M.metrics(M.corpus_texts(n_ref))
    sub_c = M.metrics(M.corpus_texts(n_ref, categories={"aggregation", "crud", "operators", "dsl"}))
    # ---- por tarefa / persona
    acc_ids = {r["id"] for r in acc}
    by_task = defaultdict(lambda: [0, 0])
    for r in raw:
        by_task[r["task"]][0] += 1
        by_task[r["task"]][1] += r["id"] in acc_ids
    by_persona = defaultdict(lambda: [0, 0])
    for r in raw:
        by_persona[r["persona"]][0] += 1
        by_persona[r["persona"]][1] += r["id"] in acc_ids
    reasons = Counter()
    for r in rej:
        reasons[(r["stage"], r["reason"].split(":")[0][:60] if r["stage"] in ("static", "fields") else r["reason"][:70])] += 1
    # ---- estimativas
    qph = gen["queries_per_hour"]
    acc_rate = len(acc) / max(1, n_raw)
    cuts_per_acc = fim["cuts_final"] / max(1, len(acc))
    exec_rate = (val["funnel"].get("after_dup", 0)) / max(1.0, float(val.get("wall_seconds") or val.get("exec_seconds_sum") or 1))
    est = {}
    for target in (60000,):
        raw_needed = target / max(acc_rate, 1e-6)
        est[target] = {"raw": raw_needed, "gen_h": raw_needed / max(qph, 1), "val_h": raw_needed / max(exec_rate, 1e-6) / 3600,
                       "cuts": target * cuts_per_acc}
    hom = None
    hp = ROOT / "tmp" / "homolog_dryrun" / "build_stats.json"
    if hp.exists():
        hom = json.loads(hp.read_text(encoding="utf-8"))
    hom_val = json.loads((hom_out / "validation.summary.json").read_text(encoding="utf-8")) if (hom_out / "validation.summary.json").exists() else None
    hom_gen = json.loads((hom_out / "gen_summary.json").read_text(encoding="utf-8")) if (hom_out / "gen_summary.json").exists() else None
    L = []
    w = L.append
    gpu = gen.get("gpu", {})
    conf = gpu.get("confirm", {})
    w("# Piloto da geração de consultas por LLM local (Qwen3-14B Q4_K_M)\n")
    w(f"Gerado em {time.strftime('%Y-%m-%d %H:%M')} por `scripts/data/llm/pilot.py`. Artefatos: `{rel(out)}` (não versionado). "
      "Nada em volume foi gerado; este é o piloto que a decisão do usuário precisa para liberar J18/J19.\n")
    w("## 1. Resumo\n")
    w(f"- **Vazão:** {qph:,} consultas/h geradas ({gen['completion_tok_per_s']} tok/s de saída agregados, {gen['slots']} slots, {cfg['queries_per_call']} consultas por chamada); "
      f"validação por execução do texto: {exec_rate * 3600:,.0f} consultas/h em {ex_cfg['workers']} lotes paralelos (CPU, em paralelo com a GPU).")
    w(f"- **Aceitação:** {len(acc)}/{n_raw} = {100 * acc_rate:.1f}% (funil na §3); vazio mantido em {val['empty_in_accepted']} das aceitas "
      f"({pct(val['empty_in_accepted'], len(acc))}, limite 25%).")
    w(f"- **Diversidade:** {llm_m['distinct_operators']} operadores distintos e {llm_m['distinct_stages']} estágios distintos em {len(acc)} consultas "
      f"(corpus de gramática, mesma amostra: {sub_c['distinct_operators']} / {sub_c['distinct_stages']}); 4-gramas distintos por token "
      f"{llm_m['distinct_4gram_ratio']} vs {sub_c['distinct_4gram_ratio']}; consultas multi-instrução {pct(llm_m['frac_multi_statement'] * len(acc), len(acc))} vs "
      f"{pct(sub_c['frac_multi_statement'] * sub_c['n'], sub_c['n'])}.")
    w(f"- **Estimativa 60k aceitas:** {est[60000]['raw']:,.0f} consultas brutas ≈ **{est[60000]['gen_h']:.0f} h de GPU** de geração "
      f"(+ validação {est[60000]['val_h']:.0f} h de CPU em paralelo), ≈ {est[60000]['cuts']:,.0f} cortes FIM ({cuts_per_acc:.1f} por consulta, após decontaminação).\n")
    w("## 2. Configuração e dispositivo\n")
    w(f"- Modelo: `{gen['model']}` (Apache-2.0, sem download). `llama-server` b11693, `-fa on`, KV q8_0, {gen['slots']} slots × {cfg['server']['ctx_per_slot']} tokens, "
      "`--jinja`, thinking desligado (`chat_template_kwargs.enable_thinking=false`, `/no_think`), saída restrita por `json_schema`.")
    w(f"- **Dispositivo escolhido por NOME:** `{gpu.get('device_id')}` = {gpu.get('device_name')} ({gpu.get('total_mib')} MiB). Guarda de velocidade: "
      f"prompt {gpu.get('sanity', {}).get('prompt_per_second')} t/s (mínimo {cfg['server']['min_prompt_tps']}), decode {gpu.get('sanity', {}).get('predicted_per_second')} t/s.")
    if conf:
        big = conf.get("big_adapter") or {}
        w(f"- **Confirmação independente (contadores do Windows, durante a geração):** `\\GPU Adapter Memory(*)\\Dedicated Usage` = {big.get('gb')} GB no adaptador "
          f"`{big.get('n')}` (os outros ≈ 0 GB); `\\GPU Engine(*)` do pid do servidor: {conf.get('engine_util_on_big_adapter')}% nesse adaptador e "
          f"{conf.get('engine_util_elsewhere')}% nos demais. Veredito: **{conf.get('verdict')}**.")
    w("")
    w("## 3. Funil de aceitação\n")
    w("| etapa | restam | % do bruto | rejeitadas na etapa |\n|---|---:|---:|---:|")
    steps = [("bruto (gerado)", n_raw, None), ("static (escrita/ddl/eval…)", fn["after_static"], fn.get("rej_static", 0)),
             ("syntax (node vm.Script)", fn["after_syntax"], fn.get("rej_syntax", 0)), ("dup (hash normalizado)", fn["after_dup"], fn.get("rej_dup", 0)),
             ("exec (mongosh, texto real)", fn["after_exec"], fn.get("rej_exec", 0)), ("fields (plano real × schema)", fn["after_fields"], fn.get("rej_fields", 0)),
             ("shape (≤2 por forma/coleção)", fn["after_shape"], fn.get("rej_shape", 0)), ("vazio ≤ 25%", fn["after_empty"], fn.get("rej_empty", 0))]
    for name, n, r in steps:
        w(f"| {name} | {n} | {pct(n, n_raw)} | {r if r is not None else ''} |")
    w("")
    w("Principais motivos de rejeição:\n")
    w("| etapa | motivo | n |\n|---|---|---:|")
    for (st, why), n in reasons.most_common(14):
        w(f"| {st} | {why} | {n} |")
    w("")
    rp = val["readonly_proof"]
    w("## 4. Prova de leitura apenas\n")
    w(f"- **Estática:** `common.static_check` recusa insert/update/delete/replace/drop/createIndex/bulkWrite/findAndModify/runCommand/eval/load/require/"
      f"`$out`/`$merge`/`$function`/`$where`/`$accumulator`, `getSiblingDB` fora do schema, `maxTimeMS` e loops infinitos (mascarando strings, comentários e regex).")
    w("- **Proxy de rede somente-leitura (`ro_proxy.py`)** entre o mongosh e o mongod: allowlist de comandos (find, aggregate, count, distinct, getMore, killCursors, "
      "explain de leitura, listCollections/listIndexes, handshake); qualquer outro é respondido com `{ok:0, code:13}` sem chegar ao servidor; `aggregate` com "
      "`$out`/`$merge`/`$function` é recusado; bancos fora da lista são recusados; `maxTimeMS=3000` é injetado em todo comando de dados.")
    w(f"- Comandos que **chegaram** ao servidor: `{json.dumps(rp['proxy_allowed_commands'])}`; escritas encaminhadas: `{rp['proxy_write_commands_forwarded']}`; "
      f"recusados pelo proxy: `{json.dumps(rp['proxy_blocked_commands'])}`.")
    w(f"- **`dbHash` de cada banco tocado antes = depois:** {rp['dbHash_before_equals_after']} ({rp['dbs_checked']} bancos). "
      f"(`opcounters` do servidor no período, só informativo — o outro agente semeia bancos em paralelo: `{rp['server_opcounters_delta_during_run']}`.)")
    w("")
    w("## 5. Diversidade × corpus de gramática (v2c)\n")
    w(f"Mesma função de métrica (`metrics.py`) nas {len(acc)} consultas LLM aceitas e em {all_c['n']} snippets únicos de treino do v2c (amostra determinística dos shards finais do FIM validado). "
      "“Corpus-sub” = só categorias aggregation/crud/operators/dsl (as comparáveis).\n")
    keys = [("distinct_operators", "operadores `$` distintos"), ("operators_per_query_mean", "operadores por consulta"), ("distinct_stages", "estágios distintos"),
            ("stages_per_pipeline_mean", "estágios por pipeline"), ("distinct_operator_sets_ratio", "conjuntos de operadores distintos / consulta"),
            ("distinct_3gram", "3-gramas distintos (tokens normalizados)"), ("distinct_4gram", "4-gramas distintos"), ("distinct_4gram_ratio", "4-gramas distintos / total"),
            ("distinct_shape_ratio", "formas estruturais distintas / consulta"), ("chars_mean", "caracteres (média)"), ("chars_p90", "caracteres (p90)"),
            ("lines_mean", "linhas (média)"), ("depth_mean", "aninhamento máx. (média)"), ("depth_p90", "aninhamento máx. (p90)"), ("frac_multiline", "multiline"),
            ("frac_comments", "com comentários"), ("frac_multi_statement", "multi-instrução"), ("frac_variables", "com const/let/var"), ("frac_functions", "com função/arrow"),
            ("frac_sibling_or_getcollection", "getSiblingDB/getCollection/colchetes"), ("frac_bson_ctors", "ObjectId/ISODate/NumberDecimal/…"),
            ("frac_aggregate", "aggregate"), ("frac_find_family", "find/findOne"), ("frac_crlf", "CRLF")]
    w("| métrica | LLM (aceitas) | corpus-sub | corpus (todas) |\n|---|---:|---:|---:|")
    for k, label in keys:
        w(f"| {label} | {llm_m[k]} | {sub_c[k]} | {all_c[k]} |")
    w_note = ("O CRLF do texto LLM é aplicado depois (50% das consultas, determinístico por hash, em `llm_to_fim.py`), por isso a linha CRLF acima mede 0 no texto bruto.")
    only_llm = sorted(set(llm_m["stage_set"]) - set(all_c["stage_set"]))
    only_c = sorted(set(all_c["stage_set"]) - set(llm_m["stage_set"]))
    w(f"\nEstágios só no LLM: {', '.join(only_llm) or '—'}; só no corpus: {', '.join(only_c) or '—'}. Operadores mais usados no LLM: {', '.join(llm_m['top_operators'][:18])}.\n")
    w("## 6. Aceitação por tarefa e persona\n")
    w("| tarefa | geradas | aceitas | taxa |\n|---|---:|---:|---:|")
    for t, (n, k) in sorted(by_task.items(), key=lambda x: -x[1][0]):
        w(f"| {t} | {n} | {k} | {pct(k, n)} |")
    w("\n| persona | geradas | aceitas | taxa |\n|---|---:|---:|---:|")
    for t, (n, k) in sorted(by_persona.items(), key=lambda x: -x[1][0]):
        w(f"| {t} | {n} | {k} | {pct(k, n)} |")
    w("")
    w("## 7. Cortes FIM e contexto\n")
    d = fim.get("decontam") or {}
    w(f"- {fim['queries']} consultas → {fim['snippets']} snippets → {fim['cuts_generated']} cortes → **{fim['cuts_final']}** após decontaminação oficial (`ngram_min_content: 2`) "
      f"contra val/test/bench do v2c ({d.get('eval_docs')} documentos) e contra a homologação ({d.get('matched_against')}); removidos: {d.get('removed_docs')} documentos.")
    w(f"- `context_kind`: {fim['by_context_kind']} → **sem schema visível (none+irrelevant): {100 * fim['none_or_irrelevant_share']:.1f}%** (meta ≥ 25%), com alvo conservador "
      f"(`field_need ≠ 2`; {fim['context_assignment']['blocked_full_need2']} cortes exigiriam campo invisível e ficaram só em `full`).")
    w(f"- `cut_kind`: {fim['by_cut_kind']}; categoria: {fim['by_category']}.")
    w("- Registro: `source: llm-qwen3-14b`, `author: llm`, `split_role: train`, `group_id`/`split_key` por consulta; `prompt` renderizado com o contrato `editor-context-v1` e orçamento de 2048 tokens "
      "(tokenizer Qwen2.5-Coder); G2.7: `blind_guard` em cada registro/prompt e `homolog_guard` no treino.\n")
    if hom or hom_val:
        w("## 8. Ensaio da homologação (schemas `hom`; fora de `data/homolog`, sem selar)\n")
        if hom_gen and hom_val:
            w(f"- Geração hom: {hom_gen['queries']} consultas a {hom_gen['queries_per_hour']:,}/h; aceitas {hom_val['accepted']} ({100 * hom_val['acceptance_rate']:.1f}%).")
        if hom:
            w(f"- Ensaio `build_homolog --dry-run`: {hom['n_items']} itens a partir de {hom['queries_llm']} consultas LLM (+{hom['snippets_grammar']} snippets de gramática); "
              f"menor célula (cut_kind × context_kind) = {hom['min_cell']} (alvo ≥ {hom['min_per_group_target']}); none+irrelevant {100 * hom['none_or_irrelevant_share']:.1f}%; "
              f"células em falta: {len(hom['shortfalls'])}.")
            w(f"  - por cut_kind: {hom['by_cut_kind']}; por context_kind: {hom['by_context_kind']}.")
        w("")
    w("## 9. Amostra: 10 consultas aceitas\n")
    for i, r in enumerate(pick_stratified(acc, "task", 10), 1):
        w(f"**A{i}** — `{r['schema_id']}.{r['collection']}` · {r['task']} · {r['persona']} · {r['layout']} · {r['validation']['n_docs']} docs · intenção: {r.get('intent', '')[:140]}\n")
        w(fence(r["text"]) + "\n")
    w("## 10. Amostra: 10 consultas rejeitadas (com motivo)\n")
    for i, r in enumerate(pick_stratified([x for x in rej if x["stage"] not in ("dup",)], "stage", 10), 1):
        w(f"**R{i}** — etapa `{r['stage']}` — motivo: {r['reason'][:200]} — `{r['schema_id']}.{r['collection']}` · {r.get('task')}\n")
        w(fence(r["text"]) + "\n")
    w("## 11. Estimativas de horas e volume\n")
    w("Base: vazão e taxas medidas neste piloto (vale para 8 slots; ver §12 para sensibilidade).\n")
    w("| alvo | consultas brutas | geração (GPU) | validação (CPU, paralela) | cortes FIM |\n|---|---:|---:|---:|---:|")
    for t, e in est.items():
        w(f"| {t:,} aceitas (J18, treino) | {e['raw']:,.0f} | {e['gen_h']:.0f} h | {e['val_h']:.0f} h | {e['cuts']:,.0f} |")
    for nm, t in (("J18 mínimo útil", 20000), ("J18 meta spec (300-400k cortes)", 60000)):
        pass
    if hom:
        # consultas hom necessárias: itens/consulta do ensaio
        per_q = hom["by_source"].get("llm-qwen3-14b", 0) / max(1, hom["queries_llm"])
        need_q = max(1000, int(5600 / 14 * 2.5))
        hom_acc = (hom_val["acceptance_rate"] if hom_val else acc_rate)
        hom_qph = hom_gen["queries_per_hour"] if hom_gen else qph
        raw_h = need_q / max(hom_acc, 1e-6)
        w(f"| homologação (J19, ≥ 5.000 cortes) | {raw_h:,.0f} (≈ {need_q:,} aceitas; regra: ≥ 1.000 consultas para ≥ 5.000 cortes com teto de 14 cortes/consulta e ≥ 50 por célula cut_kind × context_kind; o J19 pede 2.000 aceitas por folga) | {raw_h / hom_qph:.1f} h | {raw_h / max(exec_rate, 1e-6) / 3600:.1f} h | ≥ 5.000 |")
    w("")
    w("## 12. Limitações, riscos e decisões\n")
    w("- **Viés Qwen:** autor e alunos são da mesma família; a homologação mede estilo/estrutura novos, não consultas humanas. Estilos/idiomas “típicos de Qwen” aparecem nos dois lados.")
    w("- **Execução ≠ correção semântica:** uma consulta que executa e devolve dados pode não fazer o que a tarefa pedida diz; o filtro é de executabilidade e de campos, não de intenção.")
    w("- **Vazio:** consultas com valores inventados tendem a retornar vazio; mantido ≤ 25% e medido por tarefa.")
    w("- **Campos:** o checador usa o plano real enviado ao servidor (variáveis/funções já resolvidas) e o fluxo de campos por estágio; é conservador (pode recusar consultas válidas após `$replaceRoot`/`$unionWith`, que viram permissivas).")
    w("- **Cobertura da execução:** consultas que nunca iteram o cursor (`no_exec`) e scripts que dependem de estado entre blocos são recusados; `$search`/Atlas e `$text` ficam fora.")
    w("- **Decisões do usuário:** (1) liberar J18 (volume do treino) e J19 (homologação); (2) alvo de aceitas (60k) vs. horas de GPU; (3) usar `Qwen3-14B` também no filtro/professor? (4) ativar `run_guarded.py` nos jobs de treino existentes (guard sem editar os geradores).")
    (ROOT / "reports" / "datasets" / "llm-gen-pilot.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("relatorio:", ROOT / "reports" / "datasets" / "llm-gen-pilot.md")
    (out / "pilot_estimates.json").write_text(json.dumps({"est": {str(k): v for k, v in est.items()}, "acc_rate": acc_rate, "qph": qph,
                                                         "exec_rate_per_h": exec_rate * 3600, "cuts_per_acc": cuts_per_acc}, indent=1), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/generated/v3/pilot")
    ap.add_argument("--target", type=int, default=300)
    ap.add_argument("--hom-target", type=int, default=100)
    ap.add_argument("--hom-grammar", type=int, default=600)
    ap.add_argument("--hom-n", type=int, default=800)
    ap.add_argument("--phase", default="all", choices=["gen", "validate", "fim", "hom", "report", "all"])
    ap.add_argument("--workers", type=int)
    ap.add_argument("--server-url")
    a = ap.parse_args(argv)
    cfg = load_cfg()
    out = ROOT / a.out
    hom_out = ROOT / (a.out + "_hom")
    ph = a.phase
    if ph in ("gen", "all"):
        phase_gen(a, cfg, out, "train", a.target)
        if a.hom_target:
            phase_gen(a, cfg, hom_out, "hom", a.hom_target)
    if ph in ("validate", "all"):
        phase_validate(a, cfg, out)
        if a.hom_target and (hom_out / "queries.raw.jsonl").exists():
            phase_validate(a, cfg, hom_out)
    if ph in ("fim", "all"):
        phase_fim(a, cfg, out)
    if ph in ("hom", "all") and a.hom_target:
        phase_hom(a, cfg, hom_out)
    if ph in ("report", "all"):
        phase_report(a, cfg, out, hom_out)


if __name__ == "__main__":
    main()
