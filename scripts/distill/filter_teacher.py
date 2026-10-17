#!/usr/bin/env python3
"""Filtro das candidatas do professor pelo VALIDADOR do dataset (destilacao sequence-level; plano B da Fase 02 §3.5 / G2.4).

  env\\.venv\\Scripts\\python.exe -I scripts\\distill\\filter_teacher.py --pool <pool> --gen <gen> --out <dir>

O professor nunca rotula as cegas: cada candidata (greedy k=0, amostra k>=1) vira o documento `prefix + pred + suffix` e passa por
  1 sanitize (PII/segredos)  2 sintaxe (node vm.Script + parser do Console)  3 API (allowlist do Console, operadores, update ops)
  4 campos (existem no schema da colecao, tipos, `$` faltando)  5 execucao SOMENTE LEITURA em kapi_fx_* (escrita/DDL/Atlas so estaticos,
  como o gold no pipeline; o guard do driver registra qualquer comando fora da allowlist e o total vai no relatorio)
  6 semantica (resposta completa: EOS/eco do sufixo, nao vazia)  7 orcamento (<= 96 tokens, prompt <= 2048)  8 dedup (exato + MinHash,
  `validate_pipeline.stage_dedup`)  9 decontaminacao (MESMA regra oficial `ngram_min_content: 2` de dataset.yaml contra val/test/bench
  + legacy-v1; blind via blind_guard).
Entre candidatas validas do mesmo prompt vence a melhor por rank (config `filter.rank`: igual ao gold, plano igual ao gold, logprob).
Saidas: <out>/accepted.jsonl, <out>/rejected.jsonl (uma linha por candidata nao aceita, com etapa e motivo), <out>/filter_report.json,
reports/distill/filter-report-<nome>.md. Concordancia professor x gold e medida onde o gold e executavel (`gold_exec == ok`); onde o
professor diverge do gold mas e valido, a predicao do professor e mantida com `agrees_with_gold: false` (peso 0,85) para auditoria.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _distill_common import (ROOT, BlindChecker, git_hash, iter_jsonl, iter_pool, lite_fim, load_config, norm_ws, pool_shards, rel,  # noqa: E402
                             resolve, sha256_file, write_json)

STAGE_ORDER = ["sanitize", "syntax", "api", "fields", "exec", "semantic", "budget", "dedup", "decontam", "select"]
VALIDATION_STAGES = ["semantic", "sanitize", "budget", "syntax", "api", "fields", "exec"]   # ordem real das checagens por candidata
IDE_BAD = ("<|", "<｜", "```", "\0")


# ----------------------------------------------------------------------------------------------------------------------------------
# validacao de UMA candidata (etapas 1-7)
# ----------------------------------------------------------------------------------------------------------------------------------
class Validator:
    def __init__(self, dataset_cfg: dict, fcfg: dict, *, executor="auto", node: bool = True, tokenizer=True, schemas=None) -> None:
        import chat_validate as cv
        import validate_pipeline as vp
        from gen_snippets import load_schemas

        self.cv, self.vp, self.fcfg = cv, vp, fcfg
        schemas = schemas if schemas is not None else load_schemas(ROOT / dataset_cfg.get("schemas_dir", "data/schemas"))
        self.by_schema = {s.full_id: s for s in schemas}
        self.node = cv.NodeChecker() if node else None
        self.executor = None
        if executor == "auto":
            self.executor = cv.Executor(dataset_cfg)
            if not self.executor.existing:
                raise SystemExit("MongoDB sem bancos kapi_fx_* (suba o servico no WSL: wsl -d Ubuntu -u root -e systemctl start mongod)")
        elif executor not in (None, "off"):
            self.executor = executor                      # injetado (testes)
        self.tok = None
        if tokenizer:
            import render_contract as rc

            self.tok = rc.Tokenizer(ROOT / dataset_cfg["editor_context"]["tokenizer"])
        self.exec_cache: dict = {}
        self.doc_cache: dict = {}
        self.exec_calls: list[tuple] = []                 # (db, colecao, op) de cada execucao pedida ao driver (auditoria: so leituras)

    def close(self) -> None:
        if self.node:
            self.node.close()

    def coll(self, schema_id: str, name: str):
        s = self.by_schema.get(schema_id)
        if s is None:
            return None, None
        return s, next((c for c in s.colls if c.name == name or c.key == name), None)

    def _gold_plan(self, rec: dict, sch, coll):
        key = ("gold", rec["id"])
        if key not in self.doc_cache:
            try:
                plan, _ = self.vp.parse_doc(rec["prefix"] + rec["gold"] + rec["suffix"], rec.get("dialect") or "console")
            except Exception:  # noqa: BLE001
                plan = None
            if len(self.doc_cache) > 50000:
                self.doc_cache.clear()
            self.doc_cache[key] = plan
        return self.doc_cache[key]

    def validate(self, rec: dict, cand: dict) -> dict:
        """{ok, stage, reason, exec_kind, plan_equal_gold, gold_exact, gold_norm, agrees_with_gold, fields_used, pred_tokens}"""
        vp, cv = self.vp, self.cv
        pred = cand["pred"]
        gold = rec["gold"]
        out = {"ok": False, "stage": None, "reason": None, "exec_kind": None, "plan_equal_gold": None,
               "gold_exact": pred.strip() == gold.strip(), "gold_norm": norm_ws(pred) == norm_ws(gold), "agrees_with_gold": False,
               "pred_tokens": None}

        def bad(stage: str, why: str) -> dict:
            out.update(stage=stage, reason=why, stages_passed=VALIDATION_STAGES.index(stage) if stage in VALIDATION_STAGES else 0)
            return out

        # 6 (parte barata) completude/IDE: antes de gastar node/Mongo
        if self.fcfg.get("require_complete", True) and cand["stop"] == "max":
            return bad("semantic", "resposta incompleta: 96 tokens sem EOS nem eco do sufixo")
        if not pred.strip():
            return bad("semantic", "resposta vazia")
        if len(pred) > 8192 or any(x in pred for x in IDE_BAD):
            return bad("semantic", "descartada pelos criterios da IDE (marcador/cerca Markdown/NUL/tamanho)")
        # 1 sanitize
        why = vp.sanitize_text(pred)
        if why:
            return bad("sanitize", why)
        # 7 orcamento (barato, tokenizer em memoria)
        if self.tok is not None:
            n = len(self.tok.encode(pred))
            out["pred_tokens"] = n
            if n > self.fcfg.get("max_response_tokens", 96):
                return bad("budget", f"resposta com {n} tokens > {self.fcfg.get('max_response_tokens', 96)}")
        if (rec.get("prompt_tokens") or 0) > self.fcfg.get("prompt_max_tokens", 2048):
            return bad("budget", f"prompt com {rec.get('prompt_tokens')} tokens > {self.fcfg.get('prompt_max_tokens', 2048)}")
        sch, coll = self.coll(rec.get("schema_id"), rec.get("collection"))
        if sch is None or coll is None:
            return bad("fields", f"schema/colecao desconhecidos: {rec.get('schema_id')}/{rec.get('collection')}")
        doc = rec["prefix"] + pred + rec["suffix"]
        dialect = rec.get("dialect") or "console"
        # 2-4 sintaxe, API, campos
        node_msg = self.node.check(doc) if self.node else None
        an = vp.analyze_text(doc, sch, coll, node_msg, allow_write=True, expect_plan=None, dialect=dialect)
        for stage in ("syntax", "api", "fields"):
            if an[stage][0] == "fail":
                return bad(stage, an[stage][1])
        plan = an["plan"]
        out["fields_used"] = an.get("fields_used")
        # 5 execucao somente leitura
        is_write = bool(plan and plan["op"] in vp.WRITE_OPS)
        is_static = bool(plan and plan["op"] == "aggregate" and any(
            isinstance(st, dict) and (set(st) & cv.ATLAS_STAGES) for st in plan.get("pipeline") or []))
        if is_write:
            out["exec_kind"] = "skipped_write"
        elif is_static:
            out["exec_kind"] = "skipped_static"
        elif self.executor is None:
            out["exec_kind"] = "exec_off"
        else:
            ck = (sch.db_name, rec["collection"], json.dumps(plan, sort_keys=True, default=str))
            if ck not in self.exec_cache:
                self.exec_calls.append((sch.db_name, rec["collection"], plan["op"]))
                if len(self.exec_cache) > 100000:
                    self.exec_cache.clear()
                self.exec_cache[ck] = self.executor.run(sch.db_name, rec["collection"], plan)
            ex = self.exec_cache[ck]
            if ex["exec"] != "ok":
                return bad("exec", str(ex.get("error") or ex.get("reason") or ex["exec"]))
            out["exec_kind"] = "live"
        if out["exec_kind"] in ("skipped_write", "skipped_static") and self.fcfg.get("require_exec_for_static", False):
            return bad("exec", "exige execucao e a operacao so tem validacao estatica")
        # concordancia com o gold (so medicao; nunca entra na decisao de validade)
        gplan = self._gold_plan(rec, sch, coll)
        out["plan_equal_gold"] = bool(gplan is not None and plan is not None and cv.plans_equal(plan, gplan))
        out["agrees_with_gold"] = bool(out["gold_norm"] or out["plan_equal_gold"])
        out["stages_passed"] = len(VALIDATION_STAGES)
        out["ok"] = True
        return out


# ----------------------------------------------------------------------------------------------------------------------------------
# paralelismo
# ----------------------------------------------------------------------------------------------------------------------------------
_W: dict = {}


def _w_init(dataset_cfg: dict, fcfg: dict, exec_mode: str) -> None:
    import os

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _W["v"] = Validator(dataset_cfg, fcfg, executor="auto" if exec_mode == "live" else "off")


def _w_run(chunk: list) -> dict:
    v: Validator = _W["v"]
    out = []
    for rec, cands in chunk:
        out.append([(c, v.validate(rec, c)) for c in cands])
    viol = list(v.executor.violations) if v.executor is not None else []
    return {"results": out, "violations": viol, "exec_calls": Counter(op for _, _, op in v.exec_calls)}


def merged_candidates(pool_dir: Path, gen_dir, limit: int | None = None):
    """Gera (pool_rec, [candidatas]) por prompt, na ordem do pool; candidatas deduplicadas por (idx, k) (a primeira vence).
    `gen_dir` pode ser uma lista: a 1a pasta define os prompts; as demais (2a passada) so acrescentam candidatas (k>=1) aos mesmos idx."""
    dirs = [Path(d) for d in ([gen_dir] if isinstance(gen_dir, (str, Path)) else gen_dir) if Path(d).exists()]
    extras: dict[int, dict] = {}
    for d in dirs[1:]:
        for sh in sorted(d.glob("gen-[0-9][0-9][0-9][0-9][0-9].jsonl")):
            for r in iter_jsonl(sh):
                extras.setdefault(r["idx"], {}).setdefault(r["k"], r)
    shards = sorted(dirs[0].glob("gen-[0-9][0-9][0-9][0-9][0-9].jsonl"))
    pool_iter = iter(iter_pool(pool_dir))
    state = {"p": next(pool_iter, (None, None))}

    def pool_rec(idx: int) -> dict:
        while state["p"][0] is not None and state["p"][0] < idx:
            state["p"] = next(pool_iter, (None, None))
        if state["p"][0] != idx:
            raise SystemExit(f"idx {idx} do gen nao existe no pool {pool_dir}")
        return state["p"][1]

    cur_idx, cur, n = None, {}, 0
    for shard in shards:
        for r in iter_jsonl(shard):
            if cur_idx is not None and r["idx"] != cur_idx:
                for k_, r_ in extras.get(cur_idx, {}).items():
                    cur.setdefault(k_, r_)
                yield pool_rec(cur_idx), [cur[k] for k in sorted(cur)]
                n += 1
                if limit and n >= limit:
                    return
                cur = {}
            cur_idx = r["idx"]
            cur.setdefault(r["k"], r)
    if cur_idx is not None:
        for k_, r_ in extras.get(cur_idx, {}).items():
            cur.setdefault(k_, r_)
        yield pool_rec(cur_idx), [cur[k] for k in sorted(cur)]


def rank_key(rank: list[str], v: dict, cand: dict):
    parts = []
    for name in rank:
        if name == "stages_passed":                  # criterio independente do gold: etapas do validador vencidas
            parts.append(v.get("stages_passed", 0))
        elif name == "logprob_mean":
            parts.append(cand["logprob_mean"])
        else:
            raise SystemExit(f"criterio de rank desconhecido/dependente do gold: {name}")
    parts.append(-cand["k"])                         # empate: a greedy vence
    return tuple(parts)


def norm_reason(reason: str) -> str:
    return re.sub(r"[0-9a-f]{24}|\d+(\.\d+)?", "#", reason or "")[:110]


def accepted_record(rec: dict, cand: dict, v: dict, weights: dict, teacher: dict) -> dict:
    out = {k: rec[k] for k in rec if k not in ("gold",)}
    out.update(
        type="fim", middle=cand["pred"], target=cand["pred"], gold=rec["gold"], label_source="teacher_seq",
        agrees_with_gold=v["agrees_with_gold"], gold_exact=v["gold_exact"], plan_equal_gold=v["plan_equal_gold"],
        gold_executable=rec.get("gold_exec") == "ok", weight=weights["agrees_with_gold"] if v["agrees_with_gold"] else weights["disagrees_with_gold"],
        teacher={"mode": cand["mode"], "k": cand["k"], "seed": cand.get("seed"), "stop": cand["stop"], "gen_tokens": cand["gen_tokens"],
                 "logprob_mean": cand["logprob_mean"], "logprob_min": cand["logprob_min"], "topk_pos": cand.get("topk_pos"),
                 "pool_idx": cand["idx"], **teacher},
        exec_kind=v["exec_kind"], response_tokens=v["pred_tokens"], pool_id=rec["id"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--pool")
    ap.add_argument("--gen", nargs="+", help="pastas de geracao: greedy primeiro; opcionalmente a 2a passada (gen-retry)")
    ap.add_argument("--out", help="pasta de saida (padrao: <out_dir do yaml>)")
    ap.add_argument("--eval-cache", help="pasta com eval_docs.jsonl (padrao: <out_dir do yaml>)")
    ap.add_argument("--name", help="nome do relatorio (padrao: nome da pasta de saida)")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--exec", choices=["live", "off"], default="live")
    ap.add_argument("--limit-prompts", type=int)
    ap.add_argument("--chunk", type=int, default=200)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    fcfg = cfg["filter"]
    import par
    import yaml

    dataset_cfg = yaml.safe_load(resolve(cfg["dataset_config"]).read_text(encoding="utf-8"))
    out_root = resolve(cfg["out_dir"])
    pool_dir = resolve(a.pool) if a.pool else out_root / "pool"
    gen_dirs = [resolve(g) for g in a.gen] if a.gen else [out_root / "gen"]
    gen_dir = gen_dirs[0]
    out = resolve(a.out) if a.out else out_root
    cache = resolve(a.eval_cache) if a.eval_cache else out_root
    name = a.name or out.name
    workers = a.workers or fcfg["workers"]
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    gen_summary = json.loads((gen_dir / "gen_summary.json").read_text(encoding="utf-8")) if (gen_dir / "gen_summary.json").exists() else {}
    teacher_meta = {"adapter_sha256": ((gen_summary.get("meta") or {}).get("adapter") or {}).get("sha256"),
                    "model_fingerprint": ((gen_summary.get("meta") or {}).get("model") or {}).get("fingerprint")}

    # ---- fase 1: etapas 1-7 por candidata (paralelo, ordem preservada)
    prompts = 0
    results_by_prompt: list[tuple[dict, list[tuple[dict, dict]]]] = []
    violations: list[str] = []
    exec_ops: Counter = Counter()
    pool_recs: list[dict] = []

    def tee(gen):
        for rec, cands in gen:
            pool_recs.append(rec)
            yield rec, cands

    chunks = par.chunked(tee(merged_candidates(pool_dir, gen_dirs, a.limit_prompts)), a.chunk)
    for res in par.ordered_map(_w_run, chunks, workers, _w_init, (dataset_cfg, fcfg, a.exec)):
        violations += res["violations"]
        exec_ops.update(res["exec_calls"])
        for pairs in res["results"]:
            results_by_prompt.append(pairs)
        prompts = len(results_by_prompt)
        if prompts % 20000 < a.chunk:
            print(f"  validadas {prompts} ({time.time() - t0:.0f}s)", flush=True)
    assert len(pool_recs) == len(results_by_prompt)
    print(f"[filter] {prompts} prompts / {sum(len(x) for x in results_by_prompt)} candidatas validadas em {time.time() - t0:.0f}s", flush=True)

    # ---- selecao entre candidatas validas do mesmo prompt
    max_acc = int(fcfg.get("max_accept_per_prompt", 1))
    selected: list[tuple[dict, dict, dict]] = []          # (rec, cand, verdict)
    rejected: list[dict] = []
    stat_cand = Counter()
    for rec, pairs in zip(pool_recs, results_by_prompt):
        valid = [(c, v) for c, v in pairs if v["ok"]]
        valid.sort(key=lambda cv_: rank_key(fcfg["rank"], cv_[1], cv_[0]), reverse=True)
        chosen_ids = {id(c) for c, _ in valid[:max_acc]}
        for c, v in pairs:
            stat_cand[(c["mode"], "valid" if v["ok"] else "invalid")] += 1
            if v["ok"] and id(c) in chosen_ids:
                continue
            rejected.append(rejection(rec, c, v, v["stage"] or "select", v["reason"] or "valida, mas outra candidata do mesmo prompt venceu o rank"))
        for c, v in valid[:max_acc]:
            selected.append((rec, c, v))

    # ---- etapas 8 (dedup) e 9 (decontam) sobre as escolhidas
    import decontaminate as dc
    import validate_pipeline as vp

    lite = {}
    entries = []
    for rec, c, v in selected:
        lr = lite_fim(rec, middle=c["pred"])
        lr["split"] = "train"
        lite[rec["id"]] = lr
        entries.append(vp.Entry(lr, "fim"))
    seed = cfg["seed"]
    ded = vp.stage_dedup(entries, lite, seed, fcfg["dedup_jaccard"])
    dedup_fail = {e.id: e.failed for e in entries if e.failed}
    # decontaminacao por REGISTRO (a predicao muda o documento): snippet_id=None => doc_key = id
    survivors = [(rec, c, v) for rec, c, v in selected if rec["id"] not in dedup_fail]
    eval_items = []
    docs = cache / "eval_docs.jsonl"
    if not docs.exists():
        raise SystemExit(f"falta {rel(docs)} (gerado pelo build_prompt_pool.py)")
    for d in iter_jsonl(docs):
        eval_items.append((d["split"], d))
    legacy = []
    for p in fcfg["decontam"].get("legacy", []):
        pp = resolve(p)
        if pp.exists():
            legacy += [("legacy-v1", r) for r in vp.read_jsonl(pp)]
    min_content = int(dataset_cfg.get("decontam", {}).get("ngram_min_content", dc.DEFAULT_NGRAM_MIN_CONTENT))
    assert min_content == int(fcfg["decontam"]["ngram_min_content"]), "regra de decontaminacao diverge de configs/dataset.yaml"
    train_lite = []
    for rec, c, v in survivors:
        lr = dict(lite[rec["id"]])
        lr["snippet_id"] = None
        train_lite.append(lr)
    t1 = time.time()
    dec = dc.run(train_lite, eval_items + legacy, seed=seed, min_content=min_content)
    removed = dec["removed"]
    kept = [(r_, c, v) for r_, c, v in survivors if r_["id"] not in removed]
    leak = dc.run([dict(lite[r_["id"]], snippet_id=None) for r_, _, _ in kept], eval_items + legacy, seed=seed, min_content=min_content)
    print(f"[filter] dedup {ded}; decontam removeu {len(removed)}; 2a passada {leak['train_removed']} ({time.time() - t1:.0f}s)", flush=True)

    # blind (defesa em profundidade)
    blind = BlindChecker()
    final = []
    for rec, c, v in kept:
        if blind.hit_text(rec["prefix"], rec["suffix"]):
            rejected.append(rejection(rec, c, v, "decontam", "blind"))
            continue
        final.append((rec, c, v))
    for rec, c, v in selected:
        if rec["id"] in dedup_fail:
            rejected.append(rejection(rec, c, v, "dedup", dedup_fail[rec["id"]][1]))
        elif rec["id"] in removed:
            rejected.append(rejection(rec, c, v, "decontam", "coincide com avaliacao: " + ",".join(removed[rec["id"]])))

    # ---- saidas
    with open(out / "accepted.jsonl", "w", encoding="utf-8", newline="\n") as handle:
        for rec, c, v in final:
            handle.write(json.dumps(accepted_record(rec, c, v, fcfg["weights"], teacher_meta), ensure_ascii=False, separators=(",", ":")) + "\n")
    with open(out / "rejected.jsonl", "w", encoding="utf-8", newline="\n") as handle:
        for r in rejected:
            handle.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")

    report = build_report(pool_recs, results_by_prompt, final, rejected, violations, exec_ops, dec, leak, ded, cfg, name, gen_summary,
                          stat_cand, time.time() - t0, blind)
    report["files"] = {"accepted": {"path": rel(out / "accepted.jsonl"), "sha256": sha256_file(out / "accepted.jsonl"), "records": len(final)},
                       "rejected": {"path": rel(out / "rejected.jsonl"), "sha256": sha256_file(out / "rejected.jsonl"), "records": len(rejected)}}
    write_json(out / "filter_report.json", report)
    md = ROOT / "reports" / "distill" / f"filter-report-{name}.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(render_md(report, name), encoding="utf-8", newline="\n")
    print(json.dumps({k: report[k] for k in ("prompts", "accepted", "accept_rate_prompts", "agreement", "write_commands_seen_by_guard")}, ensure_ascii=False, indent=1))
    return 0


def rejection(rec: dict, cand: dict, v: dict, stage: str, reason: str) -> dict:
    return {"id": rec["id"], "pool_idx": cand["idx"], "k": cand["k"], "mode": cand["mode"], "stage": stage, "reason": reason,
            "context_kind": rec.get("context_kind"), "cut_kind": rec.get("cut_kind"), "category": rec.get("category"),
            "domain": rec.get("domain"), "pred": cand["pred"][:600], "gold": rec["gold"][:600],
            "agrees_with_gold": v.get("agrees_with_gold"), "gold_exec": rec.get("gold_exec")}


# ----------------------------------------------------------------------------------------------------------------------------------
# relatorio
# ----------------------------------------------------------------------------------------------------------------------------------
def rate(a: int, b: int):
    return round(a / b, 4) if b else None


def build_report(pool_recs, results, final, rejected, violations, exec_ops, dec, leak, ded, cfg, name, gen_summary, stat_cand, seconds, blind) -> dict:
    acc_ids = {r["id"] for r, _, _ in final}
    groups = {k: defaultdict(lambda: Counter()) for k in ("context_kind", "cut_kind", "domain", "category", "origin", "rerender_kind")}
    agree = Counter()
    greedy_fields = defaultdict(Counter)
    for rec, pairs in zip(pool_recs, results):
        ok_prompt = rec["id"] in acc_ids
        for key, g in groups.items():
            val = rec.get(key)
            g[val]["prompts"] += 1
            g[val]["accepted"] += ok_prompt
        g0 = next(((c, v) for c, v in pairs if c["k"] == 0), None)
        if g0:
            c, v = g0
            ck = rec.get("context_kind")
            greedy_fields[ck]["n"] += 1
            greedy_fields[ck]["valid"] += v["ok"]
            greedy_fields[ck]["field_hallucination"] += (v["stage"] == "fields" and str(v["reason"]).startswith("campo/estrutura invalido"))
            if rec.get("gold_exec") == "ok":
                agree["gold_exec_n"] += 1
                agree["exact"] += v["gold_exact"]
                agree["normalized"] += v["gold_norm"]
                agree["plan_equal"] += bool(v["plan_equal_gold"])
                agree["agrees"] += v["agrees_with_gold"]
                agree["valid"] += v["ok"]
                agree["valid_and_agrees"] += v["ok"] and v["agrees_with_gold"]
            agree["all_n"] += 1
            agree["all_agrees"] += v["agrees_with_gold"]
            agree["all_valid"] += v["ok"]
            agree["all_exact"] += v["gold_exact"]
    reasons = Counter((r["stage"], norm_reason(r["reason"])) for r in rejected if r["k"] == 0)
    accepted_disagree = sum(1 for r, c, v in final if not v["agrees_with_gold"])
    n = len(pool_recs)
    gn = agree["gold_exec_n"]

    def table(g):
        return {str(k): {"prompts": v["prompts"], "accepted": v["accepted"], "rate": rate(v["accepted"], v["prompts"])} for k, v in sorted(g.items(), key=lambda kv: str(kv[0]))}

    return {
        "name": name, "kind": "distill-filter-report/1", "created": time.strftime("%FT%T"), "git": git_hash(), "prompts": n,
        "candidates": sum(len(x) for x in results), "accepted": len(final), "accept_rate_prompts": rate(len(final), n),
        "greedy_valid_rate": rate(sum(1 for _, p in zip(pool_recs, results) for c, v in p if c["k"] == 0 and v["ok"]), n),
        "candidate_validity": {f"{m}:{s}": c for (m, s), c in sorted(stat_cand.items())},
        "by": {k: table(g) for k, g in groups.items()},
        "greedy_by_context_kind": {str(k): {"n": v["n"], "valid_rate": rate(v["valid"], v["n"]),
                                            "field_hallucination_rate": rate(v["field_hallucination"], v["n"])} for k, v in greedy_fields.items()},
        "agreement": {"gold_executable_prompts": gn, "greedy_exact": rate(agree["exact"], gn), "greedy_normalized": rate(agree["normalized"], gn),
                      "greedy_plan_equal": rate(agree["plan_equal"], gn), "greedy_agrees": rate(agree["agrees"], gn),
                      "greedy_valid_among_gold_exec": rate(agree["valid"], gn), "greedy_valid_and_agrees": rate(agree["valid_and_agrees"], gn),
                      "all_prompts": {"n": agree["all_n"], "exact": rate(agree["all_exact"], agree["all_n"]),
                                      "agrees": rate(agree["all_agrees"], agree["all_n"]), "valid": rate(agree["all_valid"], agree["all_n"])}},
        "accepted_disagree_with_gold": {"n": accepted_disagree, "share": rate(accepted_disagree, len(final))},
        "accepted_exec_kind": dict(Counter(v["exec_kind"] for _, _, v in final)),
        "accepted_mode": dict(Counter(c["mode"] for _, c, _ in final)),
        "rejection_reasons_greedy": [{"stage": s, "reason": r, "n": c} for (s, r), c in reasons.most_common(25)],
        "rejection_by_stage": dict(Counter(r["stage"] for r in rejected)),
        "dedup": ded, "decontam": {"blind": dec["blind"], "removed": dec["train_removed"], "by_rule": dec["by_rule"], "second_pass_removed": leak["train_removed"],
                                   "params": dec["params"]},
        "write_commands_seen_by_guard": violations, "exec_ops_requested": dict(exec_ops),
        "volume": {"t0_target": cfg["targets"]["t0"], "t1_target": cfg["targets"]["t1"],
                   "projection_for_pool_600k": int(round(rate(len(final), n) * cfg["pool"]["target"])) if n else None},
        "gen_throughput": {k: gen_summary.get(k) for k in ("ex_per_h", "steady_ex_per_h", "gen_tok_s", "prompt_tok_s", "topk_gb_per_1k_examples")},
        "seconds": round(seconds, 1),
    }


def render_md(r: dict, name: str) -> str:
    def pct(x):
        return "-" if x is None else f"{x * 100:.1f}%"

    L = [f"# Filtro da destilacao - `{name}`", "",
         f"Prompts {r['prompts']} - candidatas {r['candidates']} - **aceitos {r['accepted']} ({pct(r['accept_rate_prompts'])} dos prompts)** - "
         f"greedy valida em {pct(r['greedy_valid_rate'])}.", "",
         f"Projecao para o pool de 600k: ~{r['volume']['projection_for_pool_600k']} exemplos (alvo T1 {r['volume']['t1_target']}, T0 {r['volume']['t0_target']}).", "",
         "## Concordancia professor x gold (greedy)", "",
         f"- Prompts com gold executavel: {r['agreement']['gold_executable_prompts']}; igual ao gold {pct(r['agreement']['greedy_exact'])}, "
         f"normalizado {pct(r['agreement']['greedy_normalized'])}, plano igual {pct(r['agreement']['greedy_plan_equal'])}, "
         f"concorda {pct(r['agreement']['greedy_agrees'])}, valida {pct(r['agreement']['greedy_valid_among_gold_exec'])}.",
         f"- Todos os prompts: igual {pct(r['agreement']['all_prompts']['exact'])}, concorda {pct(r['agreement']['all_prompts']['agrees'])}, valida {pct(r['agreement']['all_prompts']['valid'])}.",
         f"- Aceitas que divergem do gold: {r['accepted_disagree_with_gold']['n']} ({pct(r['accepted_disagree_with_gold']['share'])}) - `agrees_with_gold:false`, peso 0,85.", ""]
    for key, title in (("context_kind", "context_kind"), ("cut_kind", "cut_kind"), ("domain", "dominio"), ("category", "categoria"), ("origin", "origem do prompt"),
                       ("rerender_kind", "contexto re-renderizado")):
        L += [f"## Aceitacao por {title}", "", "| grupo | prompts | aceitos | taxa |", "|---|---:|---:|---:|"]
        for k, v in r["by"][key].items():
            L.append(f"| {k} | {v['prompts']} | {v['accepted']} | {pct(v['rate'])} |")
        L.append("")
    L += ["## Greedy por context_kind (campo inexistente = G2.2)", "", "| context_kind | n | valida | campo inexistente |", "|---|---:|---:|---:|"]
    for k, v in r["greedy_by_context_kind"].items():
        L.append(f"| {k} | {v['n']} | {pct(v['valid_rate'])} | {pct(v['field_hallucination_rate'])} |")
    L += ["", "## Principais motivos de rejeicao (greedy)", "", "| etapa | motivo | n |", "|---|---|---:|"]
    for x in r["rejection_reasons_greedy"]:
        L.append(f"| {x['stage']} | {x['reason']} | {x['n']} |")
    L += ["", f"Dedup {r['dedup']}; decontaminacao (regra oficial k={r['decontam']['params']['ngram_min_content']}, blind {r['decontam']['blind']}): "
          f"removidos {r['decontam']['removed']}, 2a passada {r['decontam']['second_pass_removed']}.",
          f"Escritas vistas pelo guard do driver: **{len(r['write_commands_seen_by_guard'])}**; operacoes enviadas ao driver: {r['exec_ops_requested']}.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
