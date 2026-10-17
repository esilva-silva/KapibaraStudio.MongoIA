#!/usr/bin/env python3
"""Pool de PROMPTS para o professor (destilacao sequence-level, autocomplete FIM `editor-context-v1`).

  env\\.venv\\Scripts\\python.exe -I scripts\\distill\\build_prompt_pool.py [--target N --base N --rerender N] [--out-dir DIR]

Entrada : data/validated/v2c/fim-*.jsonl (SOMENTE `split == train` vira prompt; val/test/bench so alimentam o indice de isolamento).
Saida   : <out-dir>/pool/pool-NNNNN.jsonl + pool.manifest.json; <eval-cache>/eval_index.json + eval_docs.jsonl (indice G2.7 e
          documentos de avaliacao para a decontaminacao do filtro).

Cada linha do pool e um prompt NAO ROTULADO: prefixo/sufixo/`editor` (cabecalho do contrato), metadados e `gold` = `middle` original
(guardado so para MEDIR a concordancia professor x gold; NUNCA e usado como rotulo do aluno). Composicao:
  * base      : cortes distintos de treino, ordenados por hash(id), com o contexto NATURAL (mesma mistura de context_kind do treino);
  * rerender  : cortes distintos seguintes na ordem do hash, com o contexto RE-RENDERIZADO por gen_editor_context.make_context
                (`force_kind` sorteado por hash segundo pool.rerender_kinds; recent_commands/SimilarStatement so de snippets de treino).
Isolamento G2.7: nenhum id, grupo, schema, snippet, prompt (sha256) ou par prefixo/sufixo de val/test/bench no pool; dominios holdout
nunca; blind via blind_guard (manifesto carregado uma vez). Violacao => registro descartado e contada no manifesto (esperado: 0).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _distill_common import (CONFIG_PATH, ROOT, BlindChecker, git_hash, guard_inputs, lite_chat, lite_fim, load_config, rel,  # noqa: E402
                             resolve, sha256_file, unit_hash, write_json)

KEEP = ("id", "type", "kind", "dialect", "prefix", "suffix", "domain", "schema_id", "collection", "cut_kind", "cursor_role", "category",
        "fields_used", "group_id", "template_id", "split", "holdout", "snippet_id", "multiline", "mid_token", "crlf", "split_role",
        "read_only", "language", "context_kind", "editor", "prompt_tokens", "prompt_truncated", "prompt_sha256", "contrato")
EVAL = ("val", "test", "bench")
DIST_KEYS = ("cut_kind", "domain", "cursor_role", "category", "context_kind")


def compact(rec: dict) -> dict:
    """Registro validado -> linha do pool: sem `middle`/`target`/facts/prompt (a resposta vira `gold`, so para medicao)."""
    out = {k: rec[k] for k in KEEP if k in rec}
    out["gold"] = rec["middle"]
    out["gold_exec"] = rec.get("exec")                 # ok | skipped_static | skipped_write (o gold e executavel quando `ok`)
    out["gold_tokens"] = rec.get("middle_tokens")
    out["origin"] = "base"
    return out


def ps_sha(prefix: str, suffix: str) -> str:
    return hashlib.sha256((prefix + "\x00" + suffix).encode("utf-8")).hexdigest()


# -- varredura de um shard (worker) -----------------------------------------------------------------------------------------------
def scan_shard(args: tuple) -> dict:
    path, threshold, salt, keep_split = args
    train, eval_docs = [], []
    idx = {"ids": [], "groups": set(), "schemas": set(), "snippets": set(), "prompt_sha": set(), "ps_sha": set()}
    n_train = 0
    dist = {k: Counter() for k in DIST_KEYS}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("type", "fim") != "fim":
                continue
            if rec.get("split") == keep_split:
                n_train += 1
                for k in DIST_KEYS:
                    dist[k][rec.get(k)] += 1
                h = unit_hash(salt, rec["id"])
                if h < threshold:
                    train.append((h, compact(rec)))
            elif rec.get("split") in EVAL:
                eval_docs.append(lite_fim(rec))
                idx["ids"].append(rec["id"])
                idx["groups"].add(rec.get("group_id"))
                idx["schemas"].add(rec.get("schema_id"))
                idx["snippets"].add(rec.get("snippet_id"))
                if rec.get("prompt_sha256"):
                    idx["prompt_sha"].add(rec["prompt_sha256"])
                idx["ps_sha"].add(ps_sha(rec["prefix"], rec["suffix"]))
    return {"train": train, "eval_docs": eval_docs, "idx": idx, "n_train": n_train, "dist": dist}


def _scan_block(block):
    return scan_shard(block[0])


def scan_chat_eval(path: Path) -> tuple[list[dict], dict]:
    """Chat de avaliacao: so para o indice de decontaminacao (o chat NUNCA vira prompt desta rodada)."""
    docs, groups, schemas = [], set(), set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("split") in EVAL:
                docs.append(lite_chat(rec))
                groups.add(rec.get("group_id"))
                schemas.add(rec.get("schema_id"))
    return docs, {"groups": groups, "schemas": schemas}


def build_eval_index(idx_parts: list[dict], chat_idx: dict, holdout_domains: list[str]) -> dict:
    ids, groups, schemas, snippets, psha, pssha = set(), set(), set(), set(), set(), set()
    for p in idx_parts:
        ids.update(p["ids"])
        groups |= p["groups"]
        schemas |= p["schemas"]
        snippets |= p["snippets"]
        psha |= p["prompt_sha"]
        pssha |= p["ps_sha"]
    return {"ids": sorted(ids), "group_ids": sorted(g for g in groups if g), "schema_ids": sorted(s for s in schemas if s),
            "snippet_ids": sorted(s for s in snippets if s), "prompt_sha256": sorted(psha), "prefix_suffix_sha256": sorted(pssha),
            "chat_group_ids": sorted(g for g in chat_idx.get("groups", ()) if g), "holdout_domains": list(holdout_domains)}


# -- isolamento G2.7 ----------------------------------------------------------------------------------------------------------------
class Isolation:
    def __init__(self, index: dict, blind: BlindChecker | None = None) -> None:
        self.ids = set(index["ids"])
        self.groups = set(index["group_ids"]) | set(index.get("chat_group_ids", ()))
        self.schemas = set(index["schema_ids"])
        self.snippets = set(index["snippet_ids"])
        self.prompt_sha = set(index["prompt_sha256"])
        self.ps_sha = set(index["prefix_suffix_sha256"])
        self.holdout = set(index.get("holdout_domains", ()))
        self.blind = blind or BlindChecker()

    def violations(self, rec: dict) -> list[str]:
        """Motivos pelos quais o registro NAO pode ser prompt de treino (lista vazia = isolado)."""
        why = []
        if rec.get("split") != "train":
            why.append("split_not_train")
        if rec.get("holdout") or rec.get("domain") in self.holdout:
            why.append("holdout_domain")
        if (rec.get("split_role") or "train") != "train":
            why.append("schema_role_eval")
        parent = rec.get("parent_id") or rec["id"]
        if rec["id"] in self.ids or parent in self.ids:
            why.append("eval_id")
        if rec.get("group_id") in self.groups:
            why.append("eval_group")
        if rec.get("schema_id") in self.schemas:
            why.append("eval_schema")
        if rec.get("snippet_id") in self.snippets:
            why.append("eval_snippet")
        if rec.get("prompt_sha256") in self.prompt_sha:
            why.append("eval_prompt_sha")
        if ps_sha(rec["prefix"], rec["suffix"]) in self.ps_sha:
            why.append("eval_prefix_suffix")
        if self.blind.hit_text(rec["prefix"], rec["suffix"]):
            why.append("blind")
        return why


# -- re-renderizacao do contexto ----------------------------------------------------------------------------------------------------
def load_train_snippets(path: Path):
    """Snippets de TREINO em streaming (Pools usa so (schema, colecao, split=train) como fonte de recent_commands)."""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if '"split":"train"' in line:
                s = json.loads(line)
                if s.get("split") == "train":
                    yield s


def pick_kind(kinds: dict, seed, ident: str) -> str:
    names = list(kinds)
    pick = unit_hash(f"{seed}:rk", ident)
    acc = 0.0
    for n in names:
        acc += kinds[n]
        if pick < acc:
            return n
    return names[-1]


def rerender(records: list[dict], dcfg: dict, seed: int, kinds: dict, snippets_path: Path, log=print) -> list[dict]:
    """Novo contexto de editor por registro: `gen_editor_context.make_context(..., force_kind)`; id novo `<id>~r1`."""
    import gen_editor_context as gec
    import render_contract as rc
    from gen_snippets import load_schemas

    if not records:
        return []
    schemas = load_schemas(ROOT / dcfg.get("schemas_dir", "data/schemas"))
    log(f"  carregando snippets de treino para recent_commands ({rel(snippets_path)}) ...")
    t0 = time.time()
    pools = gec.Pools(load_train_snippets(snippets_path), schemas, int(dcfg["editor_context"].get("pool_max_per_key", 0) or 0))
    log(f"  pools prontos em {time.time() - t0:.0f}s")
    tok = rc.Tokenizer(ROOT / dcfg["editor_context"]["tokenizer"])
    out: list[dict] = []
    chunk: list[dict] = []

    def flush() -> None:
        gec._finish_chunk(chunk, dcfg, tok)  # noqa: SLF001  (orcamento de 2048 tokens, prompt_sha256 e guarda do blind)
        for r in chunk:
            c = {k: r[k] for k in KEEP if k in r}
            for k in ("gold", "gold_exec", "gold_tokens"):
                c[k] = r[k]
            c.update(id=r["id"], parent_id=r["parent_id"], origin="rerender", rerender_kind=r["context_kind"])
            out.append(c)
        chunk.clear()

    for rec in records:
        kind = pick_kind(kinds, seed, rec["id"])
        work = {**rec, "middle": rec["gold"], "id": rec["id"] + "~r1"}
        ctx = gec.make_context(work, dcfg, pools, seed, kind)
        full = gec.build_record(work, ctx, dcfg)
        full.update(parent_id=rec["id"], gold=rec["gold"], gold_exec=rec.get("gold_exec"), gold_tokens=rec.get("gold_tokens"))
        chunk.append(full)
        if len(chunk) >= 2000:
            flush()
            log(f"  re-renderizados {len(out)}/{len(records)}")
    if chunk:
        flush()
    return out


def write_pool_report(path: Path, m: dict, cfg: dict) -> None:
    iso, ni = m["isolation_G2_7"], m["none_irrelevant"]
    L = [f"# Pool de prompts da destilacao - {m['records']} prompts (autocomplete FIM, editor-context-v1)", "",
         f"Gerado {m['created']} (git {m['git']}), seed {m['seed']}, `{m['config']}`. Chat fora de escopo (Q1 pendente).", "",
         "## Composicao", "",
         f"- base (contexto natural do v2c): **{m['base']}**; re-renderizados (`make_context(force_kind)`): **{m['rerender']}**; treino visto: {m['train_seen']}.",
         f"- none + irrelevant: **{ni['n']} ({ni['share'] * 100:.1f}%)** (none {ni['none']}, irrelevant {ni['irrelevant']}); piso exigido {ni['required']}.",
         f"- origem: {m['composition']['origin']}; re-render por tipo: {m['composition']['rerender_kind']}.",
         f"- gold executavel: {m['composition']['gold_executable']}.", "",
         "## Isolamento G2.7", "",
         f"- Indice de avaliacao: {iso['checked_against']}.",
         f"- Apenas `{iso['only_split']}`; dominios holdout {iso['holdout_domains']} nunca; blind: **{iso['blind']}** (via `blind_guard`, manifesto lido uma vez).",
         f"- Descartados na amostragem: {iso['dropped_by_reason']}; **violacoes no pool final: {iso['violations_in_pool']}**.",
         f"- Regra de decontaminacao do filtro (mesma de `configs/dataset.yaml`): `ngram_min_content: {m['ngram_min_content']}`.", "",
         f"## Cobertura vs corpus de treino (tolerancia +-{cfg['pool'].get('coverage_tolerance_pp')} p.p. em cut_kind/dominio/cursor_role; pior desvio {m['coverage_worst_abs_delta_pp']} p.p.; ok={m['coverage_within_tolerance']})", ""]
    for k, rows in m["coverage_vs_train"].items():
        L += [f"### {k}", "", "| valor | treino % | pool % | delta pp |", "|---|---:|---:|---:|"]
        for v, r in sorted(rows.items(), key=lambda kv: -kv[1]["train_pct"]):
            L.append(f"| {v} | {r['train_pct']} | {r['pool_pct']} | {r['delta_pp']:+} |")
        L.append("")
    L += ["## Hashes", ""] + [f"- {k}: `{v}`" for k, v in m["hashes"].items()]
    L += [f"- source_manifest_sha256: `{m['source_manifest_sha256']}`"] + [f"- {x['path']}: {x['records']} registros, `{x['sha256']}`" for x in m["shards"]]
    L += ["", f"Tempo de construcao: {m['seconds']} s."]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


# -- principal ----------------------------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--config")
    ap.add_argument("--source-dir")
    ap.add_argument("--out-dir", help="padrao: <out_dir do yaml>; o pool vai em <out-dir>/pool")
    ap.add_argument("--eval-cache", help="onde gravar eval_index.json e eval_docs.jsonl (padrao: <out_dir do yaml>)")
    ap.add_argument("--target", type=int)
    ap.add_argument("--base", type=int)
    ap.add_argument("--rerender", type=int)
    ap.add_argument("--no-rerender", action="store_true", help="so contexto natural (nao carrega snippets)")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--shard-size", type=int)
    ap.add_argument("--report", help="grava o relatorio markdown (ex.: reports/distill/pool-v2c-300k.md)")
    ap.add_argument("--max-shards", type=int, help="teste: varre so os N primeiros shards (o indice de isolamento fica PARCIAL)")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    pc = cfg["pool"]
    if a.target and a.base is None and a.rerender is None:      # escala proporcional ao yaml
        base_n = int(round(a.target * pc["base"] / pc["target"]))
        rr_n = a.target - base_n
    else:
        base_n = a.base if a.base is not None else pc["base"]
        rr_n = a.rerender if a.rerender is not None else pc["rerender"]
    if a.no_rerender:
        base_n, rr_n = (a.target or a.base or base_n + rr_n), 0
    target = base_n + rr_n
    src = resolve(a.source_dir or cfg["source_dir"])
    out_dir = resolve(a.out_dir or cfg["out_dir"])
    cache_dir = resolve(a.eval_cache or cfg["out_dir"])
    workers = a.workers or pc["workers"]
    shard_size = a.shard_size or pc["shard_size"]
    salt = f"{cfg['seed']}:pool"
    t0 = time.time()
    files = sorted(src.glob("fim-[0-9][0-9][0-9][0-9][0-9].jsonl"))
    if a.max_shards:
        files = files[: a.max_shards]
    if not files:
        raise SystemExit(f"sem shards fim-*.jsonl em {src}")
    guard_inputs(files)
    import yaml

    dcfg_text = resolve(cfg["dataset_config"]).read_text(encoding="utf-8")
    dcfg = yaml.safe_load(dcfg_text)
    manifest_in = src / "manifest.json"
    n_train_total = 1_000_615
    if manifest_in.exists():
        n_train_total = int(json.loads(manifest_in.read_text(encoding="utf-8"))["counts"]["by_type_split"].get("fim.train", n_train_total))
    threshold = 1.0 if a.max_shards else min(1.0, target * pc["oversample_margin"] / max(1, n_train_total))
    print(f"[pool] alvo {target} (base {base_n} + rerender {rr_n}); limiar de hash {threshold:.4f}; {len(files)} shards; {workers} workers", flush=True)

    import par

    results = []
    for res in par.ordered_map(_scan_block, [[(str(f), threshold, salt, pc["keep_split"])] for f in files], workers):
        results.append(res)
        print(f"  shard {len(results)}/{len(files)} varrido ({time.time() - t0:.0f}s)", flush=True)
    candidates = sorted(((h, rec["id"], rec) for r in results for h, rec in r["train"]), key=lambda t: (t[0], t[1]))
    n_train_seen = sum(r["n_train"] for r in results)
    eval_docs = [d for r in results for d in r["eval_docs"]]
    chat_idx = {"groups": set(), "schemas": set()}
    chat_path = src / "chat.jsonl"
    if chat_path.exists():
        guard_inputs([chat_path])
        cdocs, chat_idx = scan_chat_eval(chat_path)
        eval_docs += cdocs
    index = build_eval_index([r["idx"] for r in results], chat_idx, dcfg.get("holdout_domains", []))
    train_dist = {k: Counter() for k in DIST_KEYS}
    for r in results:
        for k in DIST_KEYS:
            train_dist[k].update(r["dist"][k])
    del results
    print(f"[pool] treino visto {n_train_seen}; candidatos {len(candidates)}; docs de avaliacao {len(eval_docs)}", flush=True)

    iso = Isolation(index)
    dropped: Counter = Counter()
    chosen: list[dict] = []
    for _, _, rec in candidates:
        why = iso.violations(rec)
        if why:
            dropped.update(why)
            continue
        chosen.append(rec)
        if len(chosen) >= target:
            break
    if len(chosen) < target:
        print(f"AVISO: so {len(chosen)} candidatos isolados (< {target}); reduza o alvo ou aumente oversample_margin")
        base_n = min(base_n, len(chosen))
    base = chosen[:base_n]
    src_render = base if pc["rerender_source"] == "base" else chosen[base_n: base_n + rr_n]
    rendered: list[dict] = []
    if rr_n and src_render:
        rendered = rerender(src_render[:rr_n], dcfg, cfg["seed"], pc["rerender_kinds"], resolve(cfg["snippets"]))
        keep = []
        for r in rendered:
            why = iso.violations(r)
            if why:
                dropped.update(f"rerender:{w}" for w in why)
                continue
            keep.append(r)
        rendered = keep
    pool = base + rendered

    pool_dir = out_dir / "pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    for old in pool_dir.glob("pool-*.jsonl"):
        old.unlink()
    shards = []
    for i in range(0, len(pool), shard_size):
        p = pool_dir / f"pool-{i // shard_size:05d}.jsonl"
        part = pool[i: i + shard_size]
        with open(p, "w", encoding="utf-8", newline="\n") as handle:
            for rec in part:
                handle.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
        shards.append({"path": p.name, "records": len(part), "sha256": sha256_file(p)})
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_json(cache_dir / "eval_index.json", index)
    with open(cache_dir / "eval_docs.jsonl", "w", encoding="utf-8", newline="\n") as handle:
        for d in eval_docs:
            handle.write(json.dumps(d, ensure_ascii=False, separators=(",", ":")) + "\n")

    def dist(key: str) -> dict:
        return dict(sorted(Counter(r.get(key) for r in pool).items(), key=lambda kv: -kv[1]))

    violations_final = sum(1 for r in pool if iso.violations(r))   # reverificacao do pool final
    manifest = {
        "kind": "distill-prompt-pool/1", "version": cfg["version"], "created": time.strftime("%FT%T"), "git": git_hash(),
        "scope": "autocomplete FIM (editor-context-v1); chat fora ate a decisao Q1", "seed": cfg["seed"], "pool_salt": salt,
        "source": rel(src), "source_shards": [f.name for f in files],
        "source_manifest_sha256": sha256_file(manifest_in) if manifest_in.exists() else None,
        "config": rel(resolve(a.config) if a.config else CONFIG_PATH),
        "records": len(pool), "base": len(base), "rerender": len(rendered), "train_seen": n_train_seen, "hash_threshold": threshold,
        "shards": shards, "shard_size": shard_size,
        "composition": {"origin": dist("origin"), "context_kind": dist("context_kind"), "cut_kind": dist("cut_kind"),
                        "category": dist("category"), "domain": dist("domain"), "dialect": dist("dialect"),
                        "rerender_kind": dict(Counter(r.get("rerender_kind") for r in rendered)),
                        "gold_executable": dict(Counter(r.get("gold_exec") for r in pool))},
        "isolation_G2_7": {"checked_against": {"eval_ids": len(index["ids"]), "groups": len(index["group_ids"]),
                                               "schemas": len(index["schema_ids"]), "snippets": len(index["snippet_ids"]),
                                               "prompts": len(index["prompt_sha256"]), "prefix_suffix": len(index["prefix_suffix_sha256"])},
                           "only_split": pc["keep_split"], "dropped_by_reason": dict(dropped), "violations_in_pool": violations_final,
                           "blind": "sealed" if iso.blind.sealed else "not_sealed", "holdout_domains": index["holdout_domains"]},
        "eval_cache": {"index": rel(cache_dir / "eval_index.json"), "docs": rel(cache_dir / "eval_docs.jsonl"), "docs_n": len(eval_docs)},
        "partial_scan": bool(a.max_shards), "seconds": round(time.time() - t0, 1),
    }
    coverage = {}
    worst = 0.0
    pn = len(pool)
    for k in DIST_KEYS:
        tn = sum(train_dist[k].values())
        pc_count = Counter(c.get(k) for c in pool)
        rows = {}
        for v in sorted(set(train_dist[k]) | set(pc_count), key=str):
            tp = 100 * train_dist[k][v] / tn if tn else 0.0
            pp = 100 * pc_count[v] / pn if pn else 0.0
            rows[str(v)] = {"train_pct": round(tp, 2), "pool_pct": round(pp, 2), "delta_pp": round(pp - tp, 2)}
            if k in ("cut_kind", "domain", "cursor_role"):
                worst = max(worst, abs(pp - tp))
        coverage[k] = rows
    ni = sum(1 for c in pool if c.get("context_kind") in ("none", "irrelevant"))
    manifest["coverage_vs_train"] = coverage
    manifest["coverage_worst_abs_delta_pp"] = round(worst, 2)
    manifest["coverage_within_tolerance"] = worst <= pc.get("coverage_tolerance_pp", 2.0)
    manifest["none_irrelevant"] = {"n": ni, "share": round(ni / max(1, pn), 4), "required": pc.get("min_none_irrelevant"),
                                   "none": sum(1 for c in pool if c.get("context_kind") == "none"),
                                   "irrelevant": sum(1 for c in pool if c.get("context_kind") == "irrelevant")}
    manifest["hashes"] = {"config_sha256": sha256_file(resolve(a.config) if a.config else CONFIG_PATH),
                          "dataset_yaml_sha256": sha256_file(resolve(cfg["dataset_config"])),
                          "eval_index_sha256": sha256_file(cache_dir / "eval_index.json"),
                          "eval_docs_sha256": sha256_file(cache_dir / "eval_docs.jsonl"),
                          "pool_stream_sha256": hashlib.sha256("".join(x["sha256"] for x in shards).encode()).hexdigest()}
    manifest["ngram_min_content"] = dcfg.get("decontam", {}).get("ngram_min_content")
    write_json(pool_dir / "pool.manifest.json", manifest)
    if a.report:
        write_pool_report(resolve(a.report), manifest, cfg)
    gated = not a.max_shards and rr_n > 0 and pn >= 10000      # pisos de cobertura/none+irrelevant so valem para o pool completo
    ok = violations_final == 0 and (not gated or (manifest["coverage_within_tolerance"] and ni / max(1, pn) >= (pc.get("min_none_irrelevant") or 0)))
    if not ok:
        print("FALHA: isolamento/cobertura/piso none+irrelevant nao atendidos", json.dumps({k: manifest[k] for k in ("coverage_within_tolerance", "none_irrelevant")}))
    print(json.dumps({"records": len(pool), "base": len(base), "rerender": len(rendered), "seconds": manifest["seconds"],
                      "context_kind": manifest["composition"]["context_kind"], "dropped": dict(dropped)}, ensure_ascii=False))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
