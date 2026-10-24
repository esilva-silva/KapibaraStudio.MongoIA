#!/usr/bin/env python3
"""Corpus v3 dos alunos T0 Nano / T1 Small: destilado atual + cortes do LLM (gold) + gramatica dos schemas v3 train (gold) + replay.

  C:/SlopStudio.MongoAI-artifacts/.venv/Scripts/python.exe -I scripts/data/llm/run_guarded.py scripts/student/build_student_corpus_v3.py \
      --config configs/student_corpus_v3.yaml [--tiers t1,t0] [--workers 10] [--report reports/student/corpus_v3_report.md]
  ... --verify-only      # relê os .meta.jsonl e refaz as checagens de isolamento/proveniencia (sem renderizar)

Estende (sem alterar) build_student_corpus.py: reaproveita Isolation, allocate, render_example, replay, empacotamento, shards e val.
Diferencas: UM unico braco (`v3`); fontes com rotulos distintos (`source`, `label_source` gravados por exemplo); selecao por ORCAMENTO DE TOKENS por
tier; tres fontes de cortes com cotas fixas (new_share) e alvo de context_kind; filtro de fidelidade ao schema; G2.7 / G2.7-hom e a
decontaminacao oficial (ngram_min_content 2) reverificadas por registro, inclusive contra a homologacao selada (API autorizada).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "scripts" / "teacher", ROOT / "scripts" / "distill", ROOT / "scripts" / "data", ROOT / "scripts" / "data" / "llm", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

import homolog_guard as hg  # noqa: E402

hg.enforce()          # leitura de data/homolog so por `hg.authorized` (homolog_docs); qualquer outra e recusada

import blind_guard  # noqa: E402
import build_student_corpus as bsc  # noqa: E402
import hom_guard  # noqa: E402
import homolog_refs  # noqa: E402
import render_contract as rc  # noqa: E402
from _distill_common import BlindChecker  # noqa: E402
from build_prompt_pool import Isolation  # noqa: E402

ARM = "v3"
SOURCES = ("accepted", "llm", "grammar")
KEEP = tuple(bsc.KEEP) + ("collection", "fields_used")
NAME_TOKEN = re.compile(r"(?<![\w$])[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_0-9][A-Za-z0-9_]*)*")
STRICT_CTX = ("none", "irrelevant")
# Violacoes SOFT: o corte e descartado e contado (`soft_dropped:<motivo>`): o par (prefixo, sufixo)/conteudo/prompt coincide com um item de avaliacao ou da
# homologacao selada (acontece com prefixos curtos e genericos). Qualquer outra violacao e HARD: indica fonte misturada e ABORTA o build.
SOFT_REASONS = frozenset({"eval_prefix_suffix", "eval_prompt_sha", "content_in_eval", "homolog_hash", "blind"})


# -- fidelidade ao schema ---------------------------------------------------------------------------------------------------------------
def label_need(label: str, visible: str, names, ignore=frozenset()) -> int:
    """0 = o rotulo nao usa nomes do schema; 1 = usa, mas todos aparecem em `visible`; 2 = algum nome de campo/colecao/banco do schema NAO
    aparece em `visible` (o modelo teria de adivinha-lo). Mesma regra de llm_to_fim.field_need, ignorando `ignore` (ex.: `_id`)."""
    names = names - ignore if ignore else names
    need = 0
    for token in NAME_TOKEN.findall(label):
        segs = token.split(".")
        hit = [token] if token in names else [x for x in segs if x in names]
        if not hit:
            continue
        need = 1
        if re.search(r"(?<![\w$])" + re.escape(token) + r"(?![\w])", visible):
            continue
        for seg in hit:
            if not re.search(r"(?<![\w$])" + re.escape(seg) + r"(?![\w])", visible):
                return 2
    return need


def header_text(rec: dict) -> str:
    """Texto do cabecalho `editor-context-v1` que o modelo ve (nomes conhecidos, campos do resultado, comandos recentes, painel de entrada)."""
    ed = rec.get("editor")
    if rec.get("context_kind") == "none" or not isinstance(ed, dict):
        return ""
    parts = list(ed.get("known_names") or []) + list(ed.get("result_fields") or []) + list(ed.get("recent_commands") or [])
    parts.append(ed.get("input_panel") or "")
    return "\n".join(str(x) for x in parts)


def schema_name_index() -> tuple[dict, set, dict]:
    """({(schema_full_id, colecao): frozenset(nomes)}, {schema_full_id de papel train}, {schema_full_id: 'v3'|'base'}). Sem schemas hom."""
    import gen_snippets as gs
    names, trainable, gen = {}, set(), {}
    for s in gs.load_schemas(ROOT / "data" / "schemas"):
        gen[s.full_id] = "v3" if s.schema_id.startswith("v3_") else "base"
        if s.raw_role == "train":
            trainable.add(s.full_id)
        for c in s.colls:
            ns = {c.name, s.schema_id}
            for path in c.fields:
                ns.add(path)
                ns.update(path.split("."))
            names[(s.full_id, c.name)] = frozenset(ns)
    return names, trainable, gen


# -- util ---------------------------------------------------------------------------------------------------------------------------
unit = bsc.unit
sha256_file = bsc.sha256_file


def prio(seed: int, salt: str, ident: str, schema_gen: str, v3_priority: float) -> float:
    return unit(seed, salt, ident) * (v3_priority if schema_gen == "v3" else 1.0)


def merge_cfg(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    base = yaml.safe_load((ROOT / cfg["base_config"]).read_text(encoding="utf-8"))
    out = dict(base)
    out.update({k: v for k, v in cfg.items() if k != "base_config"})
    out["_base_config"] = cfg["base_config"]
    return out


def jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sealed_sets() -> tuple[set, set]:
    """Hashes do manifesto selado da homologacao (so hashes; leitura do manifesto e permitida pelo guard)."""
    m = hg.manifest()
    return set((m.get("item_sha256") or {}).values()), set(m.get("prompt_sha256") or [])


# -- carga + G2.7 por registro ------------------------------------------------------------------------------------------------------
class Loader:
    def __init__(self, cfg: dict, limit: int | None = None):
        self.cfg, self.limit = cfg, limit
        self.seed = int(cfg["seed"])
        comp = cfg["composition"]
        self.ignore = frozenset(comp.get("ignore_name_tokens") or [])
        self.keep_unobs = float(comp["unobservable_full_keep"])
        dcfg = cfg["distilled"]
        self.splits_dir = ROOT / cfg["splits_dir"]
        index = json.loads((ROOT / dcfg["eval_index"]).read_text(encoding="utf-8"))
        self.blind = BlindChecker()
        self.iso = Isolation(index, self.blind)
        self.train_ids = bsc.read_ids(self.splits_dir / "train.ids.txt")
        self.other_ids = {n: bsc.read_ids(self.splits_dir / f"{n}.ids.txt") for n in ("val", "test", "bench")}
        self.eval_content: set[int] = set()
        with (ROOT / dcfg["eval_docs"]).open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d.get("type", "fim") == "fim":
                    self.eval_content.add(bsc.content_hash(d["prefix"], d["suffix"], d["middle"]))
        # v3-eval (gramatica): ids/grupos/schemas/snippets/conteudo que o treino nunca pode usar
        self.v3_eval_ids: set[str] = set()
        ev = ROOT / cfg["sources"]["grammar"]["eval_docs"]
        self.v3_eval_snips: set[str] = set()
        if ev.exists():
            for d in jsonl(ev):
                self.v3_eval_ids.add(d["id"])
                self.v3_eval_snips.add(d["snippet_id"])
                self.eval_content.add(bsc.content_hash(d["prefix"], d["suffix"], d["middle"]))
        self.sealed_items, self.sealed_prompts = sealed_sets()
        self.names, self.trainable, self.schema_gen = schema_name_index()
        self.stats: dict[str, Counter] = {s: Counter() for s in SOURCES}
        self.seen_content: set[int] = set()

    # --- um registro
    def _common_checks(self, r: dict, src: str, why: list[str], item_hash: str | None = None) -> None:
        hom_guard.assert_no_hom([r], f"fonte {src} do corpus v3")          # aborta: hom nunca vira corpus
        why += self.iso.violations(r)
        if r.get("schema_id") not in self.trainable:
            why.append("schema_not_train_role")
        if r.get("snippet_id") in self.v3_eval_snips or r["id"] in self.v3_eval_ids:
            why.append("v3_eval_overlap")
        labels = {r["gold"]} | ({r["target"]} if src == "accepted" else set())
        for lab in labels:
            if bsc.content_hash(r["prefix"], r["suffix"], lab) in self.eval_content:
                why.append("content_in_eval")
                break
        if hg.item_sha256(r) in self.sealed_items or hg.prompt_sha256(r["prefix"] + "\x00" + r["suffix"]) in self.sealed_prompts:
            why.append("homolog_hash")

    def _fidelity(self, r: dict, label_train: str) -> tuple[int, int, str | None]:
        """(need_prompt, need_text, motivo de descarte|None)."""
        names = self.names.get((r["schema_id"], r["collection"]))
        if names is None:
            return 0, 0, "unknown_collection"
        text = r["prefix"] + "\n" + r["suffix"]
        need_text = label_need(label_train, text, names, self.ignore)
        need_prompt = label_need(label_train, text + "\n" + header_text(r), names, self.ignore)
        if r["context_kind"] in STRICT_CTX:
            return need_prompt, need_text, ("unobservable_blind" if need_text == 2 else None)
        if need_prompt == 2 and unit(self.seed, "unobs-keep", r["id"]) >= self.keep_unobs:
            return need_prompt, need_text, "unobservable_full"
        return need_prompt, need_text, None

    def _admit(self, r: dict, src: str, label_source: str, source_tag: str, label_train_fn, item_hash: str | None = None) -> dict | None:
        st = self.stats[src]
        st["read"] += 1
        why: list[str] = []
        self._common_checks(r, src, why, item_hash)
        hard = [w for w in why if w not in SOFT_REASONS]
        if hard:
            st["isolation_dropped"] += 1
            for w in set(why):
                st[f"violation:{w}"] += 1
            return None
        if why:
            st["soft_dropped"] += 1
            for w in set(why):
                st[f"soft_dropped:{w}"] += 1
            return None
        ch = bsc.content_hash(r["prefix"], r["suffix"], r["gold"])
        if ch in self.seen_content:
            st["duplicate_content"] += 1
            return None
        need_p, need_t, drop = self._fidelity(r, label_train_fn(r))
        if drop:
            st[f"fidelity_dropped:{drop}"] += 1
            st["fidelity_dropped"] += 1
            return None
        self.seen_content.add(ch)
        gen = self.schema_gen.get(r["schema_id"], "base")
        out = {k: r.get(k) for k in KEEP}
        out.update({"source": source_tag, "label_source": label_source, "need_prompt": need_p, "need_text": need_t, "schema_gen": gen})
        st["kept"] += 1
        return out

    # --- fontes
    def accepted(self) -> list[dict]:
        sc = self.cfg["sources"]["accepted"]
        out = []
        with (ROOT / sc["path"]).open(encoding="utf-8") as f:
            for n, line in enumerate(f):
                if self.limit and n >= self.limit:
                    break
                r = json.loads(line)
                if r.get("label_source") != "teacher_seq":
                    self.stats["accepted"]["bad_label_source"] += 1
                    continue
                base_id = r.get("parent_id") or r["id"]
                bad = []
                if base_id not in self.train_ids:
                    bad.append("id_not_in_train_ids")
                for name, s in self.other_ids.items():
                    if base_id in s or r["id"] in s:
                        bad.append(f"id_in_{name}")
                if bad:
                    self.stats["accepted"]["isolation_dropped"] += 1
                    for w in set(bad):
                        self.stats["accepted"][f"violation:{w}"] += 1
                    self.stats["accepted"]["read"] += 1
                    continue

                def lab(rr):          # rotulo usado no treino: professor em fim_ctx/completion; gold so em fim_raw (5%, hash por id)
                    return rr["target"]

                a = self._admit(r, "accepted", "teacher_seq", sc["source"], lab)
                if a is not None:
                    a["origin"] = r.get("origin") or "base"
                    out.append(a)
        return out

    def _new(self, src: str, files: list[Path]) -> list[dict]:
        sc = self.cfg["sources"][src]
        out = []
        n = 0
        for p in files:
            blind_guard.check_path(p)
            for r in jsonl(p):
                n += 1
                if self.limit and n > self.limit:
                    return out
                if r.get("split") != "train":
                    self.stats[src]["read"] += 1
                    self.stats[src]["isolation_dropped"] += 1
                    self.stats[src]["violation:split_not_train"] += 1
                    continue
                ih = hg.item_sha256(r)          # hash do registro ORIGINAL (formato dos itens selados) antes de qualquer campo derivado
                r["gold"] = r["target"] = r["middle"]
                r["agrees_with_gold"] = True
                r["origin"] = src
                a = self._admit(r, src, sc["label_source"], sc["source"], lambda rr: rr["gold"], ih)
                if a is not None:
                    out.append(a)
        return out

    def llm(self) -> list[dict]:
        return self._new("llm", [ROOT / self.cfg["sources"]["llm"]["path"]])

    def grammar(self) -> list[dict]:
        d = ROOT / self.cfg["sources"]["grammar"]["dir"]
        files = sorted(d.glob("fim*.jsonl"))
        if not files:
            raise SystemExit(f"sem cortes de gramatica validados em {d}")
        return self._new("grammar", files)


# -- decontaminacao oficial (reverificacao) -------------------------------------------------------------------------------------------
def decontam_check(recs: list[dict], seed: int, min_content: int = 2, refs=None) -> dict:
    """Roda a decontaminacao OFICIAL (13-grama com >= min_content tokens de conteudo, chave aproximada, MinHash) de `recs` (dicts com id, prefix, gold,
    suffix, collection, snippet_id) contra val/test/bench v2c, v3-eval e a homologacao selada (API autorizada). -> {removed:{id:regras}, ...}."""
    import decontaminate as dc
    import llm_to_fim as l2f
    evals = refs if refs is not None else (l2f.eval_docs_v2c() + homolog_refs.homolog_docs())
    docs = [{"id": r["id"], "type": "fim", "prefix": r["prefix"], "middle": r["gold"], "suffix": r["suffix"], "collection": r.get("collection"),
             "snippet_id": r.get("snippet_id"), "cut_kind": r.get("cut_kind")} for r in recs]
    res = dc.run(docs, evals, seed=seed, min_content=min_content)
    return {"removed": res["removed"], "train_in": res["train_in"], "by_rule": res["by_rule"], "matched_against": res["matched_against"],
            "eval_docs": res["eval_docs"]}


# -- selecao -------------------------------------------------------------------------------------------------------------------------
def stratum(c: dict) -> tuple:
    return (c["kind"], c["category"], c["ctx"])


def source_targets(cfg: dict, pool_dims: dict, avail_cats: dict[str, set], adj: dict | None = None) -> dict[str, dict]:
    """Alvo de estratos (cut_kind, categoria, context_kind) por fonte = cut_kind do pool v2c x categoria da fonte x context_kind do v3 (produto),
    multiplicado pelos fatores de rebalanceamento `adj` ({"cut": {k: f}, "cat": {c: f}}; ver `rebalance`)."""
    comp = cfg["composition"]
    adj = adj or {}
    f_cut, f_cat = adj.get("cut") or {}, adj.get("cat") or {}
    cut_m = {k: v / sum(pool_dims["cut_kind"].values()) for k, v in pool_dims["cut_kind"].items()}
    cats_pool = {k: v / sum(pool_dims["category"].values()) for k, v in pool_dims["category"].items()}
    out = {}
    for src in SOURCES:
        ctx_m = (comp.get("context_kind_by_source") or {}).get(src) or comp["context_kind"]
        if src == "grammar":
            cats = dict(comp["grammar_category"])
        else:
            cats = {k: v for k, v in cats_pool.items() if src == "accepted" or k in avail_cats.get(src, set())}
        z = sum(cats.values())
        cats = {k: v / z for k, v in cats.items()}
        raw = {(k, c, x): cut_m[k] * f_cut.get(k, 1.0) * cats[c] * f_cat.get(c, 1.0) * ctx_m[x] for k in cut_m for c in cats for x in ctx_m}
        zz = sum(raw.values())
        out[src] = {k: v / zz for k, v in raw.items()}
    return out


def pool_marginals(pool_dims: dict) -> dict[str, dict]:
    return {d: {k: v / sum(pool_dims[d].values()) for k, v in pool_dims[d].items()} for d in ("cut_kind", "category")}


def selection_marginals(chosen: dict[str, list], cands: dict) -> dict[str, dict]:
    n = max(1, sum(len(v) for v in chosen.values()))
    out = {"cut_kind": Counter(), "category": Counter()}
    for s in SOURCES:
        for i in chosen[s]:
            out["cut_kind"][cands[i]["stratum"][0]] += 1
            out["category"][cands[i]["stratum"][1]] += 1
    return {d: {k: v / n for k, v in c.items()} for d, c in out.items()}


def rebalance(adj: dict, target: dict, got: dict, power: float = 0.8) -> tuple[dict, float]:
    """Atualiza os fatores multiplicativos para que as marginais GLOBAIS (cut_kind, categoria) da selecao se aproximem das do pool v2c. Fontes limitadas por oferta
    nao respondem (teto), as demais compensam. Retorna (novos fatores, desvio maximo em p.p.)."""
    new = {"cut": dict(adj.get("cut") or {}), "cat": dict(adj.get("cat") or {})}
    dev = 0.0
    for dim, key in (("cut_kind", "cut"), ("category", "cat")):
        for k, t in target[dim].items():
            g = got[dim].get(k, 0.0)
            dev = max(dev, abs(t - g) * 100)
            ratio = (t / g) ** power if g > 0 else 1.6
            new[key][k] = min(4.0, max(0.25, new[key].get(k, 1.0) * min(1.6, max(0.6, ratio))))
    return new, dev


def split_quota(n: int, weights: dict[str, float], caps: dict[str, int]) -> dict[str, int]:
    return bsc.allocate(n, weights, caps)


def select(by_src: dict[str, dict], n_total: int, comp: dict, targets: dict, caps: dict | None) -> dict[str, list]:
    """-> {fonte: [ids escolhidos]}. `by_src[src][stratum]` = ids ordenados por prioridade; T0 usa `caps[src][stratum]` (teto = T1)."""
    avail_src = {s: sum(len(v) for v in by_src[s].values()) for s in SOURCES}
    if caps is not None:
        avail_src = {s: sum(min(len(v), caps[s].get(k, 0)) for k, v in by_src[s].items()) for s in SOURCES}
    w_new = comp["new_share"]
    weights = {"accepted": 1 - w_new, "llm": w_new * comp["llm_share_of_new"], "grammar": w_new * (1 - comp["llm_share_of_new"])}
    quota = split_quota(n_total, weights, avail_src)
    chosen: dict[str, list] = {}
    for s in SOURCES:
        avail = {k: len(v) for k, v in by_src[s].items()}
        if caps is not None:
            avail = {k: min(n, caps[s].get(k, 0)) for k, n in avail.items()}
        share = {k: targets[s].get(k, 0.0) or 1e-9 for k in avail}
        alloc = bsc.allocate(quota[s], share, avail)
        chosen[s] = [i for k, n in sorted(alloc.items(), key=lambda kv: str(kv[0])) for i in by_src[s][k][:n]]
    return chosen


def calibrate(by_src, token_of: dict, budget: int, comp, targets, caps) -> tuple[int, dict]:
    """Menor N de exemplos de cortes cujos tokens (prompt+alvo) somados atingem `budget` (busca binaria; select e monotono em N)."""
    hi = sum(len(v) for s in SOURCES for v in by_src[s].values())
    lo = 1
    while lo < hi:
        mid = (lo + hi) // 2
        ch = select(by_src, mid, comp, targets, caps)
        tok = sum(token_of[i] for s in SOURCES for i in ch[s])
        if tok < budget:
            lo = mid + 1
        else:
            hi = mid
    ch = select(by_src, lo, comp, targets, caps)
    return lo, ch


# -- principal ---------------------------------------------------------------------------------------------------------------------
def build(args: argparse.Namespace) -> int:
    t0 = time.time()
    cfg_path = Path(args.config)
    cfg = merge_cfg(cfg_path)
    seed = int(cfg["seed"])
    out_root = Path(args.out_root) if args.out_root else ROOT / cfg["out_root"]
    blind_guard.check_path(out_root)
    if args.verify_only:
        res = {t: verify_v3(out_root, t, cfg) for t in args.tiers.split(",") if (out_root / t / "manifest.json").exists()}
        print(json.dumps(res, indent=1)[:4000])
        return 0 if all(v["n_issues"] == 0 for v in res.values()) else 3
    tok_path = ROOT / cfg["tokenizer"]
    tok = rc.Tokenizer(tok_path)
    tok_sha = sha256_file(tok_path)
    for text, want in bsc.FIM_IDS.items():
        if tok.encode(text) != [want]:
            raise SystemExit(f"tokenizer: {text} -> {tok.encode(text)}, esperado [{want}]")
    comp, ccfg, mix = cfg["composition"], cfg["completion"], cfg["mix"]

    print("[1/8] isolamento + carga das fontes (G2.7 / G2.7-hom por registro, fidelidade ao schema)", flush=True)
    ld = Loader(cfg, args.limit)
    recs = {"accepted": ld.accepted(), "llm": ld.llm(), "grammar": ld.grammar()}
    stats = {s: dict(ld.stats[s]) for s in SOURCES}
    for s in SOURCES:
        print(f"      {s}: {stats[s]}", flush=True)
        left = {k: v for k, v in stats[s].items() if k.startswith("violation:")}
        if left:
            raise SystemExit(f"G2.7: {s} tem violacoes de isolamento ({left}); revisar a fonte")
    print("[2/8] decontaminacao oficial (k=2) reverificada: v2c val/test/bench + v3-eval + homologacao selada", flush=True)
    dec: dict = {}
    if args.no_decontam_check:
        dec = {"skipped": True}
    else:
        import decontaminate as dc
        import llm_to_fim as l2f
        refs = l2f.eval_docs_v2c() + homolog_refs.homolog_docs()
        evp = ROOT / cfg["sources"]["grammar"]["eval_docs"]
        if evp.exists():
            refs += [(d["split"], d) for d in jsonl(evp)]
        flat = [r for s in SOURCES for r in recs[s]]
        r = decontam_check(flat, seed, 2, refs)
        bad_keys = {dc.doc_key({"type": "fim", "prefix": "", "middle": "", "snippet_id": x["snippet_id"], "id": x["id"]})
                    for x in flat if x["id"] in r["removed"]}
        before = {s: len(recs[s]) for s in SOURCES}
        for s in SOURCES:
            recs[s] = [x for x in recs[s] if dc.doc_key({"type": "fim", "prefix": "", "middle": "", "snippet_id": x["snippet_id"], "id": x["id"]}) not in bad_keys]
        dec = {"train_in": r["train_in"], "removed_cuts_direct": len(r["removed"]), "removed_docs": len(bad_keys), "by_rule": r["by_rule"],
               "matched_against": r["matched_against"], "eval_docs": r["eval_docs"], "min_content": 2,
               "dropped_by_source": {s: before[s] - len(recs[s]) for s in SOURCES}, "kept_by_source": {s: len(recs[s]) for s in SOURCES}}
    print(f"      {dec}", flush=True)

    # componente por id (mesma regra do corpus v2c: completion so se elegivel, depois fim_raw, senao fim_ctx)
    all_recs = [r for s in SOURCES for r in recs[s]]
    for r in all_recs:
        r["_elig"] = (len(r["target"]) + len(r["suffix"]) <= ccfg["max_chars"] and len(r["gold"]) + len(r["suffix"]) <= ccfg["max_chars"])
    elig_frac = sum(r["_elig"] for r in all_recs) / max(1, len(all_recs))
    q_completion = min(1.0, mix["completion"] / max(1e-9, elig_frac))
    raw_given_rest = mix["fim_raw"] / max(1e-9, 1.0 - mix["completion"])
    for r in all_recs:
        r["_comp"] = bsc.component_of(r["id"], r["_elig"], seed, q_completion, raw_given_rest)

    print(f"[3/8] renderizando/tokenizando {len(all_recs)} candidatos ({args.workers} workers)", flush=True)
    items = [(r, r["_comp"]) for r in all_recs]
    chunks = [(items[a: a + 2000], ccfg) for a in range(0, len(items), 2000)]
    results: list[dict] = []
    with mp.get_context("spawn").Pool(args.workers, initializer=bsc._init_worker, initargs=(str(tok_path), cfg)) as pool:  # noqa: SLF001
        for res in pool.imap(bsc._render_chunk, chunks):  # noqa: SLF001
            results += res
    by_id = {r["id"]: r for r in all_recs}
    dropped = Counter(r["drop"] for r in results if "drop" in r)
    cands: dict[str, dict] = {}
    for res in results:
        if "drop" in res:
            continue
        r = by_id[res["id"]]
        t = res["t_distill"]
        cands[r["id"]] = {"p": res["p"], "t": t, "id": r["id"], "rec": r}
    del results, items, chunks
    pool_shares, pool_info = bsc.target_shares(ROOT / cfg["distilled"]["pool_dir"], args.limit and args.limit * 2)
    for c in cands.values():
        r = c["rec"]
        c["src"] = ("accepted" if r["source"] == cfg["sources"]["accepted"]["source"] else "llm" if r["source"] == cfg["sources"]["llm"]["source"] else "grammar")
        c["stratum"] = (r["cut_kind"], r["category"], r["context_kind"])
        c["len"] = len(c["p"]) + len(c["t"])
        c["prio"] = prio(seed, cfg["priority_salt"], r["id"], r["schema_gen"], float(comp["v3_schema_priority"]))
    by_src: dict[str, dict] = {s: defaultdict(list) for s in SOURCES}
    for c in sorted(cands.values(), key=lambda c: (c["prio"], c["id"])):
        by_src[c["src"]][c["stratum"]].append(c["id"])
    avail_cats = {s: {k[1] for k in by_src[s]} for s in SOURCES}
    adj: dict = {}
    targets = source_targets(cfg, pool_info["dims"], avail_cats, adj)
    token_of = {i: c["len"] for i, c in cands.items()}
    avail_report = {s: {"candidates": sum(len(v) for v in by_src[s].values()), "tokens": sum(token_of[i] for v in by_src[s].values() for i in v)} for s in SOURCES}
    print(f"      candidatos: {avail_report}; descartes de renderizacao: {dict(dropped)}", flush=True)

    # replay
    print(f"[4/8] replay ({cfg['replay']['token_share']:.0%} dos tokens)", flush=True)
    share_replay = float(cfg["replay"]["token_share"]) if cfg["replay"]["enabled"] else 0.0
    replay = bsc.load_replay(cfg, tok._tok, seed, cfg["seq_len"], ld.eval_content) if share_replay else None  # noqa: SLF001
    src_hashes = {"accepted_sha256": sha256_file(ROOT / cfg["sources"]["accepted"]["path"]),
                  "llm_manifest_sha256": sha256_file(ROOT / cfg["sources"]["llm"]["manifest"]),
                  "llm_fim_ctx_sha256": sha256_file(ROOT / cfg["sources"]["llm"]["path"]),
                  "grammar_manifest_sha256": sha256_file(ROOT / cfg["sources"]["grammar"]["dir"] / "manifest.json"),
                  "eval_index_sha256": sha256_file(ROOT / cfg["distilled"]["eval_index"]),
                  "eval_docs_sha256": sha256_file(ROOT / cfg["distilled"]["eval_docs"]), "train_ids_sha256": sha256_file(ld.splits_dir / "train.ids.txt")}
    homolog_manifest = ROOT / "reports" / "datasets" / "homolog-v1-manifest.json"
    if homolog_manifest.exists():
        src_hashes["homolog_manifest_sha256"] = sha256_file(homolog_manifest)

    tiers = [t for t in args.tiers.split(",") if t]
    order = sorted(tiers, key=lambda t: -cfg["tiers"][t]["tokens_per_epoch"])
    caps_prev: dict | None = None
    summary: dict = {}
    for tier in order:
        tcfg = cfg["tiers"][tier]
        budget_total = int(tcfg["tokens_per_epoch"])
        budget_mongo = int(round(budget_total * (1 - share_replay)))
        caps = caps_prev if (tcfg.get("subset_of") and caps_prev is not None) else None
        tgt_marg = pool_marginals(pool_info["dims"])
        history = []
        for it in range(int(comp.get("rebalance_iterations", 8)) + 1):
            targets = source_targets(cfg, pool_info["dims"], avail_cats, adj)
            n_mongo, chosen = calibrate(by_src, token_of, budget_mongo, comp, targets, caps)
            got = selection_marginals(chosen, cands)
            adj_new, dev = rebalance(adj, tgt_marg, got)
            history.append(round(dev, 3))
            if dev <= float(comp.get("rebalance_tolerance_pp", 0.5)) or it == int(comp.get("rebalance_iterations", 8)):
                break
            adj = adj_new
        print(f"      {tier}: rebalanceamento cut_kind/categoria: desvio maximo vs pool (p.p.) por iteracao {history}", flush=True)
        if tcfg.get("subset_of") is None:
            caps_prev = {s: dict(Counter(cands[i]["stratum"] for i in chosen[s])) for s in SOURCES}
        print(f"[5/8] {tier}: N cortes={n_mongo} (orcamento {budget_mongo:,} tokens): " +
              str({s: len(chosen[s]) for s in SOURCES}), flush=True)
        examples = []
        for s in SOURCES:
            for i in chosen[s]:
                c = cands[i]
                r = c["rec"]
                diff = bool(r["source"] == cfg["sources"]["accepted"]["source"] and r["_comp"] in ("fim_ctx", "completion") and r["target"] != r["gold"])
                label_source = r["label_source"] if r["_comp"] != "fim_raw" else "gold"
                meta = {"id": i, "component": r["_comp"], "kind": r["cut_kind"], "domain": r["domain"], "category": r["category"],
                        "ctx": r["context_kind"], "role": r["cursor_role"], "origin": r["origin"], "n_prompt": int(len(c["p"])),
                        "agrees_with_gold": bool(r["agrees_with_gold"]), "label_differs": diff, "src": s, "source": r["source"],
                        "label_source": label_source, "schema_id": r["schema_id"], "schema_gen": r["schema_gen"], "split_role": "train",
                        "need_prompt": r["need_prompt"], "need_text": r["need_text"]}
                meta["label_hash_distill"] = meta["label_hash_gold"] = hashlib.sha1(c["t"].tobytes()).hexdigest()[:12]
                examples.append({"p": c["p"], "t_distill": c["t"], "id": i, "meta": meta})
        nonreplay_tokens = sum(len(e["p"]) + len(e["t_distill"]) for e in examples)
        picked, rep_stats = ([], {"examples": 0, "tokens": 0, "isolation": None})
        if replay:
            picked, rep_stats = bsc.pick_replay(replay, int(round(nonreplay_tokens * share_replay / (1 - share_replay))))
        for c in picked:
            t = np.asarray(c["t_ids"], dtype=np.uint32)
            meta = {"id": c["id"], "component": f"replay_{c['kind']}", "kind": c["kind"], "domain": c["doc"]["package"],
                    "language": "ts" if c["doc"]["ext"] in (".ts", ".tsx") else "js", "category": c["doc"]["license"], "n_prompt": len(c["p_ids"]),
                    "agrees_with_gold": True, "label_differs": False, "src": "replay", "source": "replay-js", "label_source": "replay"}
            meta["label_hash_distill"] = meta["label_hash_gold"] = hashlib.sha1(t.tobytes()).hexdigest()[:12]
            examples.append({"p": np.asarray(c["p_ids"], dtype=np.uint32), "t_distill": t, "id": c["id"], "meta": meta})
        examples.sort(key=lambda e: e["id"])
        pack_len = [len(e["p"]) + len(e["t_distill"]) for e in examples]
        rows = bsc.pack_rows(pack_len, cfg["micro_tokens"], cfg["max_examples_per_row"], seed, tier)
        print(f"[6/8] {tier}: {len(examples)} exemplos ({len(picked)} replay), {len(rows)} linhas; gravando", flush=True)
        sel_dir = out_root / "selection"
        sel_dir.mkdir(parents=True, exist_ok=True)
        with (sel_dir / f"{tier}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for e in examples:
                m = e["meta"]
                handle.write(json.dumps({"id": e["id"], "component": m["component"], "src": m["src"], "source": m["source"],
                                         "label_source": m["label_source"],
                                         "stratum": [m["kind"], m["category"], m.get("ctx")] if m["src"] != "replay" else m["kind"]}, ensure_ascii=False) + "\n")
        d = out_root / tier
        d.mkdir(parents=True, exist_ok=True)
        for old in list(d.glob("train-*")) + list((d / "val").glob("*") if (d / "val").exists() else []):
            old.unlink()
        for e in examples:
            e["meta"]["label_hash"] = e["meta"]["label_hash_distill"]
        shards = bsc.write_shards(d, rows, examples, "distill", cfg["micro_tokens"], int(cfg["shard_examples"]))
        mongo = [e for e in examples if e["meta"]["src"] != "replay"]
        val = bsc.build_val(cfg, d, {e["id"] for e in mongo}, cfg["micro_tokens"], cfg["max_examples_per_row"], seed)
        tokens = sum(s["tokens"] for s in shards)
        rows_n = sum(s["rows"] for s in shards)
        by_comp: dict = defaultdict(lambda: {"examples": 0, "tokens": 0, "target_tokens": 0})
        by_source: dict = defaultdict(lambda: {"examples": 0, "tokens": 0})
        for e in examples:
            c = by_comp[e["meta"]["component"]]
            c["examples"] += 1
            c["tokens"] += len(e["p"]) + len(e["t_distill"])
            c["target_tokens"] += len(e["t_distill"])
            b = by_source[e["meta"]["source"]]
            b["examples"] += 1
            b["tokens"] += len(e["p"]) + len(e["t_distill"])
        steps_epoch = rows_n // 2
        epochs = float(tcfg["train_epochs"])
        total_steps = int(steps_epoch * epochs)
        tok_per_step = tokens / max(1, steps_epoch)
        trained_tokens = int(tokens * epochs)
        man = {
            "schema": "student-corpus-v3/1", "arm": ARM, "tier": tier, "tier_name": tcfg["name"], "base": tcfg["base"],
            "built_utc": time.strftime("%FT%TZ", time.gmtime()), "git_hash": bsc.git_hash(), "seed": seed,
            "config": {"path": bsc.rel(cfg_path), "sha256": sha256_file(cfg_path), "base": cfg["_base_config"],
                       "base_sha256": sha256_file(ROOT / cfg["_base_config"])},
            "builder_sha256": sha256_file(Path(__file__)),
            "tokenizer": {"path": cfg["tokenizer"], "sha256": tok_sha, "special_ids": bsc.FIM_IDS},
            "label_source": "mixed: teacher_seq (accepted fim_ctx/completion) | gold (llm, grammar, fim_raw) | replay",
            "label_policy": "sem destilacao nova: o rotulo dos dados novos e gold (exato: execucao do texto real / gerador de gramatica)",
            "source": {**{k: cfg["sources"][k] for k in SOURCES}, **src_hashes, "limit": args.limit},
            "selection": {"n_mongo": len(mongo), "n_mongo_calibrated": n_mongo, "budget_total_tokens_per_epoch": budget_total,
                          "budget_mongo_tokens": budget_mongo, "by_source": {s: len(chosen[s]) for s in SOURCES}, "candidates": avail_report,
                          "subset_of": tcfg.get("subset_of"), "composition": comp, "dropped_in_render": dict(dropped),
                          "completion_eligible_fraction": round(elig_frac, 4), "q_completion": round(q_completion, 4),
                          "rebalance": {"max_dev_pp_by_iteration": history, "factors": {k: {a: round(b, 3) for a, b in v.items()} for k, v in adj.items()},
                                        "pool_marginals": tgt_marg}},
            "loader_stats": stats,
            "mix": {"target": mix, "achieved": {k: round(v["examples"] / max(1, len(examples)), 4) for k, v in sorted(by_comp.items())}},
            "replay": {"status": "enabled" if replay else "none", "token_share_target": share_replay,
                       "token_share_achieved": round(rep_stats["tokens"] / max(1, tokens), 4), **rep_stats},
            "counts": {"examples": len(examples), "mongo_examples": len(mongo), "tokens": tokens,
                       "target_tokens": sum(s["target_tokens"] for s in shards), "padded_tokens": sum(s["padded_tokens"] for s in shards),
                       "rows": rows_n, "mean_tokens_per_example": round(tokens / max(1, len(examples)), 1), "by_component": dict(by_comp),
                       "by_source": dict(by_source)},
            "schedule": {"steps_per_epoch": steps_epoch, "train_epochs": epochs, "total_steps": total_steps, "tokens_per_step_real": round(tok_per_step),
                         "trained_tokens": trained_tokens, "tok_s": tcfg["tok_s"], "tok_s_conservative": tcfg["tok_s_conservative"],
                         "hours": round(trained_tokens / tcfg["tok_s"] / 3600, 2),
                         "hours_conservative": round(trained_tokens / tcfg["tok_s_conservative"] / 3600, 2)},
            "micro_tokens": cfg["micro_tokens"], "seq_len": cfg["seq_len"], "val": val, "shards": shards,
            "isolation": {"G2_7": "so train por registro (Isolation: split, holdout, papel, ids, grupos, schemas, snippets, prompt_sha, prefixo+sufixo, cego) + "
                                  "train.ids.txt (destilado) + conteudo fora de eval_docs v2c e v3-eval; G2.7-hom: hom_guard por registro, hashes do selo "
                                  "da homologacao e decontaminacao oficial k=2 reverificada contra v2c + v3-eval + homologacao",
                          "accepted_load": {"read": stats["accepted"].get("read", 0), "isolation_dropped": 0},
                          "loaders": stats, "decontam_check": dec, "blind_sealed": ld.blind.sealed,
                          "homolog_access_log": "data/homolog/v1/access.log.jsonl (leituras autorizadas: decontaminacao)"},
        }
        (d / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        ver = verify_v3(out_root, tier, cfg)
        man["verification"] = ver
        (d / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        summary[tier] = {"manifest": man, "verification": ver, "balance": balance(mongo)}
        print(f"      {tier}: verificacao n_issues={ver['n_issues']}", flush=True)
        if ver["n_issues"]:
            print(json.dumps(ver, indent=1), file=sys.stderr)
            return 3
    print("[8/8] relatorio", flush=True)
    write_report(Path(args.report), cfg, summary, pool_info, time.time() - t0)
    print(f"OK em {time.time() - t0:.0f}s")
    return 0


def balance(mongo: list[dict]) -> dict:
    n = max(1, len(mongo))
    out = {}
    for key, name in (("kind", "cut_kind"), ("category", "category"), ("ctx", "context_kind"), ("src", "src"), ("schema_gen", "schema_gen"),
                      ("source", "source"), ("label_source", "label_source"), ("component", "component")):
        c = Counter(e["meta"].get(key) for e in mongo)
        out[name] = {str(k): round(v / n, 5) for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))}
    out["need_prompt"] = {str(k): round(v / n, 5) for k, v in sorted(Counter(e["meta"].get("need_prompt") for e in mongo).items())}
    by_src: dict = defaultdict(Counter)
    for e in mongo:
        by_src[e["meta"]["src"]][e["meta"]["ctx"]] += 1
    out["ctx_by_src"] = {s: {k: round(v / sum(c.values()), 4) for k, v in sorted(c.items())} for s, c in by_src.items()}
    return out


# -- verificacao independente -------------------------------------------------------------------------------------------------------
def verify_v3(root: Path, tier: str, cfg: dict) -> dict:
    """Relê os meta.jsonl de data/student/v3/<tier>: proveniencia completa, ids unicos, sem hom, schemas de papel train, fora de val/test/bench/v3-eval,
    label_source coerente com a fonte, e nenhum exemplo fora de `train`."""
    splits_dir = ROOT / cfg["splits_dir"]
    train, val, test, bench = (bsc.read_ids(splits_dir / f"{n}.ids.txt") for n in ("train", "val", "test", "bench"))
    d = root / tier
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    metas, row_sizes = [], []
    for sh in man["shards"]:
        metas += list(jsonl(d / sh["meta_file"]))
        with np.load(d / sh["file"]) as data:
            row_sizes += np.diff(data["row_ptr"]).tolist()
    ev_ids: set[str] = set()
    evp = ROOT / cfg["sources"]["grammar"]["eval_docs"]
    if evp.exists():
        ev_ids = {x["id"] for x in jsonl(evp)}
    ok_label = {"accepted-v2c-teacher": {"teacher_seq", "gold"}, "llm-qwen3-14b": {"gold"}, "grammar-v3": {"gold"}, "replay-js": {"replay"}}
    issues: list[str] = []
    seen: set[str] = set()
    n_by_source: Counter = Counter()
    for m in metas:
        i = m["id"]
        n_by_source[m.get("source")] += 1
        if i in seen:
            issues.append(f"{i}: duplicado")
        seen.add(i)
        if not m.get("source") or not m.get("label_source"):
            issues.append(f"{i}: sem proveniencia (source/label_source)")
            continue
        if m["label_source"] not in ok_label.get(m["source"], set()):
            issues.append(f"{i}: label_source {m['label_source']} incoerente com {m['source']}")
        if m["src"] == "replay":
            if not i.startswith("rpl-") or i in train or i in val or i in test or i in bench:
                issues.append(f"replay id suspeito {i}")
            continue
        if m.get("split_role") != "train":
            issues.append(f"{i}: split_role {m.get('split_role')}")
        if hom_guard.is_hom_schema(m.get("schema_id"), m.get("domain"), None):
            issues.append(f"{i}: schema hom {m.get('schema_id')}")
        base = i.split("~r1")[0]
        if m["src"] == "accepted" and base not in train:
            issues.append(f"{i}: destilado fora de train.ids.txt")
        for name, s in (("val", val), ("test", test), ("bench", bench)):
            if base in s or i in s:
                issues.append(f"{i}: presente em {name}.ids.txt")
        if i in ev_ids:
            issues.append(f"{i}: id de v3-eval")
    return {"tier": tier, "examples": len(metas), "mongo_examples": len(seen) - n_by_source.get("replay-js", 0), "by_source": dict(n_by_source),
            "n_issues": len(issues), "issues": issues[:20], "rows": len(row_sizes), "ids_sha256": bsc.sha256_text("\n".join(m["id"] for m in metas)),
            "rows_sha256": bsc.sha256_text(",".join(map(str, row_sizes)))}


# -- relatorio -----------------------------------------------------------------------------------------------------------------------
def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def write_report(path: Path, cfg: dict, summary: dict, pool_info: dict, seconds: float) -> None:
    comp = cfg["composition"]
    blind_txt = ", ".join(f"{t.upper()} {pct(v['balance']['context_kind'].get('none', 0) + v['balance']['context_kind'].get('irrelevant', 0))}" for t, v in sorted(summary.items(), reverse=True))
    L = ["# Corpus v3 dos alunos T0 Nano / T1 Small", "",
         f"Gerado {time.strftime('%FT%T')} por `scripts/student/build_student_corpus_v3.py` (config `configs/student_corpus_v3.yaml` sobre "
         f"`{cfg['_base_config']}`, seed {cfg['seed']}, git {bsc.git_hash()[:12]}, {seconds:.0f} s). Dados em `data/student/v3/{{t0,t1}}/` (gitignored).", "",
         "## Por que e o que muda", "",
         "Homologacao v1 (`reports/datasets/homolog-v1-eval.md`): em schemas NOVOS os alunos inventam campos (contexto `full`: T0 15,35%, T1 9,19%, bases ~7%). "
         "O corpus v3 ataca o ponto com tres alavancas, todas medidas aqui: (1) **mais schemas novos**: cortes do LLM (Qwen3-14B, rotulo gold validado por "
         "execucao do texto real) e da gramatica dos 61 schemas v3 `train` (rotulo gold), somando "
         f"{pct(comp['new_share'])} dos exemplos de cortes; (2) **mais none/irrelevant com alvo conservador** (o rotulo so usa nomes visiveis no texto): "
         f"{blind_txt} dos cortes contra ~29% no v2c (teto = oferta de cortes conservadores de cada fonte); (3) **filtro de fidelidade ao schema**: exemplo `full` cujo rotulo usa campo/colecao ausente do prompt inteiro "
         f"so entra com p={comp['unobservable_full_keep']} (o modelo aprenderia a adivinhar nome), e none/irrelevant exige `field_need` <= 1.", "",
         "**Sem destilacao nova**: o rotulo dos dados novos e gold; o efeito da destilacao (J14 x J10b) foi pequeno e o gold e exato nesses dados (o professor so "
         "poderia piorar o rotulo). O destilado atual segue com `teacher_seq`. Cada exemplo grava `source` e `label_source` no `.meta.jsonl`.", ""]
    for tier, s in sorted(summary.items(), reverse=True):
        m, b, v = s["manifest"], s["balance"], s["verification"]
        c, sc = m["counts"], m["schedule"]
        L += [f"## {tier.upper()} {m['tier_name']} (`{m['base']}`)", "",
              f"- **{c['examples']:,} exemplos** ({c['mongo_examples']:,} cortes Mongo + {m['replay']['examples']:,} replay), **{c['tokens']:,} tokens/epoca** "
              f"(media {c['mean_tokens_per_example']}/ex.; alvo com perda {pct(c['target_tokens'] / c['tokens'])}), {c['rows']:,} linhas <= {m['micro_tokens']} tokens "
              f"(pad {pct(1 - c['tokens'] / max(1, c['padded_tokens']))}), **{sc['steps_per_epoch']:,} passos/epoca**.",
              f"- Treino: **{sc['train_epochs']} epocas = {sc['total_steps']:,} passos (`--total-steps {sc['total_steps']}` se < {int(2 * sc['steps_per_epoch']):,}), "
              f"{sc['trained_tokens']:,} tokens** -> **{sc['hours']} h** a {sc['tok_s']} tok/s medidos ({sc['hours_conservative']} h a {sc['tok_s_conservative']} tok/s).",
              f"- Replay: {m['replay']['examples']:,} trechos, {m['replay']['tokens']:,} tokens = **{pct(m['replay']['token_share_achieved'])}** dos tokens.", "",
              "| fonte | `label_source` | exemplos | tokens | % dos cortes Mongo |", "|---|---|---:|---:|---:|"]
        for src, label in (("accepted-v2c-teacher", "teacher_seq (fim_raw: gold)"), ("llm-qwen3-14b", "gold"), ("grammar-v3", "gold"), ("replay-js", "replay")):
            v2 = c["by_source"].get(src, {"examples": 0, "tokens": 0})
            share = "" if src == "replay-js" else pct(v2["examples"] / max(1, c["mongo_examples"]))
            L.append(f"| {src} | {label} | {v2['examples']:,} | {v2['tokens']:,} | {share} |")
        new_share = b["src"].get("llm", 0) + b["src"].get("grammar", 0)
        smoke = None
        sp = ROOT / "reports" / "student" / "smoke_v3_metrics.json"
        if sp.exists():
            smoke = json.loads(sp.read_text(encoding="utf-8")).get(tier)
        if smoke:
            L += ["", f"- **Smoke GPU** (20 passos + retomada do passo 12, RX 7800 XT confirmada por contadores do Windows; `reports/student/smoke_v3_metrics.json`): "
                  f"**{smoke['tok_s_real_mean_steps2_20']:.0f} tok/s reais** (mediana {smoke['tok_s_median']:.0f}), {smoke['step_s_mean']} s/passo, pico de VRAM alocada "
                  f"**{smoke['vram_peak_gb']} GB** (reservada {smoke['vram_reserved_peak_gb']} GB), {smoke['wall_seconds_incl_load_ckpt_eval']} s de GPU no total -> treino completo a essa "
                  f"velocidade: **{sc['trained_tokens'] / smoke['tok_s_real_mean_steps2_20'] / 3600:.1f} h**."]
        L += ["", f"**Composicao dos cortes Mongo**: LLM + gramatica v3 = **{pct(new_share)}** (alvo {pct(comp['new_share'])}); em schemas `v3_*` (qualquer fonte) "
              f"{pct(b['schema_gen'].get('v3', 0))}; **none+irrelevant = {pct(b['context_kind'].get('none', 0) + b['context_kind'].get('irrelevant', 0))}** "
              f"(full {pct(b['context_kind'].get('full', 0))}); cortes cujo rotulo usa nome do schema ausente do prompt (`need_prompt`=2): {pct(b['need_prompt'].get('2', 0))}.", "",
              f"context_kind por fonte: {b.get('ctx_by_src')} (teto = oferta de cortes conservadores de cada fonte).", "",
              "| cut_kind | % | categoria | % | componente | % |", "|---|---:|---|---:|---|---:|"]
        ks, cs, ps = list(b["cut_kind"].items()), list(b["category"].items()), list(b["component"].items())
        for i in range(max(len(ks), len(cs), len(ps))):
            row = []
            for lst in (ks, cs, ps):
                row += [lst[i][0], pct(lst[i][1])] if i < len(lst) else ["", ""]
            L.append("| " + " | ".join(row) + " |")
        L += ["", "Funil das fontes (carga + G2.7 + fidelidade):", "", "| fonte | lidos | mantidos | violacoes G2.7 | duplicados | descartes de fidelidade |", "|---|---:|---:|---:|---:|---|"]
        for src in SOURCES:
            st = m["loader_stats"][src]
            fid = {k.split(":", 1)[1]: v for k, v in st.items() if k.startswith("fidelity_dropped:")}
            L.append(f"| {src} | {st.get('read', 0):,} | {st.get('kept', 0):,} | {st.get('isolation_dropped', 0)} | {st.get('duplicate_content', 0)} | {fid or 'nenhum'} |")
        L += ["", f"Decontaminacao oficial (k=2) reverificada: {json.dumps(m['isolation']['decontam_check'], ensure_ascii=False)}", "",
              f"Verificacao independente: problemas **{v['n_issues']}**, fontes {v['by_source']}, `ids_sha256` {v['ids_sha256'][:16]}...", ""]
    dec0 = next(iter(summary.values()))["manifest"]["isolation"]["decontam_check"]
    L += ["## Achado: decontaminacao contra a homologacao (importante)", "",
          "`llm_to_fim.homolog_docs()` (usado no J18) le `data/homolog/v1/items/items.jsonl`, mas o conjunto selado esta em `data/homolog/v1/v1/items/items.jsonl`; com o caminho "
          "errado devolve lista vazia SEM erro, e a decontaminacao n-grama do corpus LLM contra a homologacao nao ocorreu (so o hash exato do manifesto). Aqui a leitura vem de "
          "`scripts/data/homolog_refs.py` (`homolog_guard.set_dir`, falha alto se o selo existe e os itens nao). Efeito medido pela regra OFICIAL (ngram_min_content 2): "
          f"{dec0.get('dropped_by_source')} cortes removidos por fonte; `matched_against` {dec0.get('matched_against')}. Ou seja, **~24% dos cortes do LLM e ~4% do corpus destilado "
          "compartilhavam 13-gramas com >= 2 tokens de conteudo com a homologacao** (nomes de campo genericos como status/createdAt). Os modelos atuais (sft_distill_v1) foram treinados com o "
          "destilado sem esse corte: a homologacao v1 deles pode estar levemente otimista; o v3 e o primeiro treino decontaminado contra a homologacao. O codigo de `scripts/data/llm` nao "
          "foi alterado (fora do escopo); correcao sugerida: `items_path` = `hg.set_dir() / 'items' / 'items.jsonl'`.", "",
          "## Isolamento", "",
          "G2.7 por registro (aborta em qualquer violacao): `Isolation` (split train, holdout, papel do schema, ids, grupos, schemas, snippets, prompt_sha256, "
          "prefixo+sufixo, cego), hash de conteudo fora de `eval_docs` (v2c) e do v3-eval, schema em papel `train`. G2.7-hom: `hom_guard.assert_no_hom` em cada registro, "
          "hashes do selo da homologacao e a decontaminacao oficial (ngram_min_content 2) reexecutada contra v2c + v3-eval + homologacao selada (leitura SO pela API autorizada "
          "`homolog_guard.authorized`, linha em `data/homolog/v1/access.log.jsonl`). Replay: pacotes/arquivos/cortes fora do holdout Generic-JS e do conteudo Mongo de avaliacao. "
          "Testes: `scripts/student/tests/test_student_corpus_v3.py`.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "student_corpus_v3.yaml"))
    ap.add_argument("--tiers", default="t1,t0")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out-root")
    ap.add_argument("--report", default=str(ROOT / "reports" / "student" / "corpus_v3_report.md"))
    ap.add_argument("--limit", type=int, help="teste: primeiras N linhas de cada fonte")
    ap.add_argument("--no-decontam-check", action="store_true", help="testes: pula a reverificacao n-grama")
    ap.add_argument("--verify-only", action="store_true")
    return build(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
