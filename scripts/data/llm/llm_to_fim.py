#!/usr/bin/env python3
"""Consultas LLM validadas -> cortes FIM com contexto de editor (`editor-context-v1`), reaproveitando (por import, sem editar) gen_fim,
gen_editor_context, render_contract, decontaminate, blind_guard.

  env/.venv/Scripts/python.exe scripts/data/llm/llm_to_fim.py --in data/generated/v3/pilot/queries.validated.jsonl --out data/generated/v3/pilot

  * spans sintaticos vem de jsscan.py (o gerador de gramatica emite spans na renderizacao; aqui o texto e do LLM) e alimentam
    `gen_fim.candidates/pick_end/make_record` (mesmos 9 cut_kinds, mesmo limite de 96 tokens medido com o tokenizer Qwen);
  * `context_kind` full|irrelevant|none via `gen_editor_context.make_context(force_kind)`; >= 25% sem schema visivel (none/irrelevant) com alvo
    CONSERVADOR (como na v2d: o corte so entra em none/irrelevant se `field_need` != 2, i.e. o middle nao exige nome de campo/colecao invisivel);
  * `source: llm-qwen3-14b`, `author: llm`, `split_role` do schema (train -> split train; hom -> split hom), group/split_key, isolamento G2.7
    (blind_guard dentro do _finish_chunk; homolog_guard para o treino) e decontaminacao OFICIAL (ngram_min_content: 2) contra val/test/bench do
    v2c e contra o conjunto de homologacao.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from common import (ROOT, append_jsonl, load_cfg, load_schema_infos, query_key, read_jsonl, rel, sha, write_jsonl)
import jsscan
import gen_fim as gf
import gen_editor_context as gec
import render_contract as rc
import decontaminate as dc
import blind_guard
import homolog_guard as hg
from gen_snippets import load_config as load_dataset_cfg, split_key as snip_split_key

os.environ.setdefault("KAPI_TOKENIZERS_SITE", gf.DEFAULT_TOK_SITE)
_NAME_TOKEN = re.compile(r"(?<![\w$])[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_0-9][A-Za-z0-9_]*)*")
CUT_KINDS = ["key", "value", "operator", "stage", "member", "partial", "rest", "multiline", "stmt_start"]


# ----------------------------------------------------------------------------------------------------------------------------
# necessidade de nome invisivel (mesma regra de scripts/teacher/build_teacher_corpus.py::field_need, copiada para nao importar numpy/replay)
# ----------------------------------------------------------------------------------------------------------------------------
def field_need(target: str, visible: str, names) -> int:
    """0 = alvo sem nomes do schema; 1 = tem, mas todos visiveis no texto; 2 = algum nome de campo/colecao/banco ausente do texto visivel."""
    need = 0
    for token in _NAME_TOKEN.findall(target):
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


def schema_names(info, coll) -> frozenset:
    names = {coll.name, info.schema_id}
    for path in coll.fields:
        names.add(path)
        names.update(path.split("."))
    return frozenset(names)


class LlmPools(gec.Pools):
    """Pools de recent_commands/SimilarStatement so com consultas LLM do MESMO papel; distratores (irrelevant) so de schemas permitidos."""

    def __init__(self, snippets, schemas, distractors):
        super().__init__(snippets, schemas, max_per_key=400)
        self._dist = sorted(distractors, key=lambda s: s.full_id)

    def distractor_schemas(self, rec):
        return [s for s in self._dist if s.full_id != rec["schema_id"]]


# ----------------------------------------------------------------------------------------------------------------------------
# consulta -> snippet (formato do gen_fim)
# ----------------------------------------------------------------------------------------------------------------------------
def snippet_from_query(q: dict, info, seed: int, crlf_ratio: float = 0.5) -> dict | None:
    text = q["text"].replace("\r\n", "\n").strip("\n")
    u = int(hashlib.sha256(f"{seed}:crlf:{q['id']}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    crlf = u < crlf_ratio
    if crlf:
        text = text.replace("\n", "\r\n")
    sc = jsscan.scan(text)
    spans = sc["spans"]
    coll = info.coll(q["collection"])
    used = []
    for s, e, role, _u in spans:
        if role in ("key", "fieldref"):
            p = ".".join(x for x in text[s:e].lstrip("$").split(".") if not x.isdigit())
            if p in coll.fields and p not in used:
                used.append(p)
    cmds = (q.get("validation") or {}).get("commands") or []
    first = (cmds[0] if cmds else "find").lower()
    op = {"find": "find", "aggregate": "aggregate", "count": "countDocuments", "distinct": "distinct"}.get(first, "find")
    if "aggregate" in [c.lower() for c in cmds]:
        op = "aggregate"
    if re.search(r"getSiblingDB", text):
        category = "dsl"
    elif op == "aggregate":
        category = "aggregation"
    else:
        category = "crud"
    root_form = "var" if re.search(r"\b(?:const|let|var)\s+\w+\s*=\s*db\b", text) else "dot"
    task = q.get("task") or "free"
    group_id = f"{info.full_id}::llm.{task}.{sha(q['qkey'], 8)}"
    return {
        "id": "snip-" + q["id"], "op": op, "kind": "read", "text": text, "spans": spans, "domain": info.domain, "schema_id": info.full_id,
        "collection": q["collection"], "fields_used": used, "group_id": group_id, "template_id": f"llm.{task}", "split_key": snip_split_key(group_id),
        "split": "train" if info.role == "train" else "hom", "holdout": False, "dialect": "console", "split_role": info.role,
        "category": category, "style": {"root_form": root_form}, "crlf": crlf, "llm_query_id": q["id"], "intent": q.get("intent"),
        "persona": q.get("persona"), "task": task,
    }


def cuts_for_snippet(snip: dict, dscfg: dict, tc, seed: int, n: int, weights: dict | None = None, per_kind_cap: int | None = None) -> list[dict]:
    """Ate `n` cortes distintos do snippet, kinds sorteados por `weights` (padrao fim.cut_targets), sem repetir (prefixo, middle)."""
    rng = random.Random(f"{seed}:cuts:{snip['id']}")
    cand = gf.candidates(snip)
    w = weights or dscfg["fim"]["cut_targets"]
    out, seen, tries = [], set(), 0
    avail = {k: v for k, v in cand.items() if v}
    k_counts = Counter()
    while avail and len(out) < n and tries < n * 6:
        tries += 1
        kinds = list(avail)
        kind = rng.choices(kinds, weights=[w.get(k, 0.05) / (1 + k_counts[k]) for k in kinds])[0]
        c = rng.choice(avail[kind])
        e = gf.pick_end(snip, kind, c, dscfg, rng, tc)
        if e is None:
            continue
        key = (c["start"], e)
        if key in seen:
            continue
        seen.add(key)
        k_counts[kind] += 1
        if per_kind_cap and k_counts[kind] >= per_kind_cap:
            avail.pop(kind, None)
        out.append(gf.make_record(snip, kind, c, e, dscfg, len(out), tc))
    return out


def finish_fim_record(r: dict, snip: dict, q: dict | None, cfg: dict) -> dict:
    r["source"] = cfg["source_tag"]
    r["author"] = "llm"
    r["llm_query_id"] = snip.get("llm_query_id")
    r["task"] = snip.get("task")
    r["persona"] = snip.get("persona")
    r["read_only"] = True
    r["exec"] = "ok"
    r["validation"] = {"exec": "ok", "validator": "mongosh-text+ro-proxy"}
    return r


# ----------------------------------------------------------------------------------------------------------------------------
# contexto de editor
# ----------------------------------------------------------------------------------------------------------------------------
def assign_context_kinds(cuts: list[dict], names_by_key: dict, share: float, seed: int, ratios: dict | None = None) -> dict:
    """Escolhe context_kind por corte. `share` = fracao alvo de none+irrelevant sobre o TOTAL; so cortes com field_need <= 1 sao elegiveis
    (conservador, v2d). Retorna {id: kind} e estatisticas."""
    for c in cuts:
        c["_need"] = field_need(c["middle"], c["prefix"] + c["suffix"], names_by_key[(c["schema_id"], c["collection"])])
    elig = sorted([c for c in cuts if c["_need"] <= 1], key=lambda c: sha(f"{seed}:ctx:{c['id']}"))
    n_blind = min(len(elig), int(round(share * len(cuts) + 0.4999)))
    kinds = {c["id"]: "full" for c in cuts}
    ratio_none = (ratios or {}).get("none", 0.5) / max(1e-9, (ratios or {}).get("none", 0.5) + (ratios or {}).get("irrelevant", 0.5))
    for c in elig[:n_blind]:
        u = int(sha(f"{seed}:nk:{c['id']}", 8), 16) / 0xFFFFFFFF
        kinds[c["id"]] = "none" if u < ratio_none else "irrelevant"
    st = {"cuts": len(cuts), "eligible_for_blind": len(elig), "blind_assigned": n_blind, "blocked_full_need2": sum(1 for c in cuts if c["_need"] == 2),
          "blind_share": round(n_blind / max(1, len(cuts)), 4)}
    return kinds, st


def contextualize(cuts: list[dict], kinds: dict, pools, dscfg: dict, tokenizer, seed: int) -> list[dict]:
    out = []
    for r in cuts:
        r = dict(r)
        r.pop("_need", None)
        ctx = gec.make_context(r, dscfg, pools, seed, kinds[r["id"]])
        rec = gec.build_record(r, ctx, dscfg)
        blind_guard.check_record(rec)
        blind_guard.check_text(blind_guard.prompt_text({"kind": "autocomplete", "prefix": rec["prefix"], "suffix": rec["suffix"]}))
        out.append(rec)
    for i in range(0, len(out), 2000):
        gec._finish_chunk(out[i:i + 2000], dscfg, tokenizer)    # noqa: SLF001  (orcamento 2048 tokens + guarda do blind)
    return out


def make_tokenizer(dscfg):
    try:
        return rc.Tokenizer(ROOT / dscfg["editor_context"]["tokenizer"])
    except Exception as e:  # noqa: BLE001
        print(f"[aviso] sem tokenizer ({e}); prompts sem orcamento de tokens", file=sys.stderr)
        return None


# ----------------------------------------------------------------------------------------------------------------------------
# decontaminacao (regra OFICIAL ngram_min_content: 2) contra val/test/bench v2c e homologacao
# ----------------------------------------------------------------------------------------------------------------------------
EVAL_CACHE = ROOT / "data" / "generated" / "v3" / "_eval_docs_v2c.jsonl"


def eval_docs_v2c(validated_dir: Path = ROOT / "data" / "validated" / "v2c", cache: Path = EVAL_CACHE) -> list[tuple[str, dict]]:
    """Um registro leve por snippet (cortes FIM do mesmo snippet compartilham o documento) dos splits val/test/bench do corpus v2c."""
    if not cache.exists():
        seen, rows = set(), []
        markers = ('"split":"val"', '"split":"test"', '"split":"bench"')
        for p in sorted(validated_dir.glob("fim-*.jsonl")):
            blind_guard.check_path(p)
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if not any(m in line for m in markers):
                        continue
                    r = json.loads(line)
                    if r.get("split") not in ("val", "test", "bench") or r["snippet_id"] in seen:
                        continue
                    seen.add(r["snippet_id"])
                    rows.append({k: r.get(k) for k in ("id", "type", "kind", "prefix", "middle", "suffix", "snippet_id", "collection", "split", "cut_kind")})
        write_jsonl(cache, rows)
    return [(r["split"], r) for r in read_jsonl(cache)]


def homolog_docs(hdir: Path | None = None) -> list[tuple[str, dict]]:
    p = (hdir or hg.homolog_dir() / "v1") / "items" / "items.jsonl"
    if not p.exists():
        return []
    out = []
    with hg.authorized("decontaminacao do treino LLM contra a homologacao (so features/ngrams; nada e copiado)", "llm_to_fim"):
        with open(p, encoding="utf-8") as f:
            seen = set()
            for line in f:
                r = json.loads(line)
                if r["snippet_id"] in seen:
                    continue
                seen.add(r["snippet_id"])
                out.append(("homolog", {k: r.get(k) for k in ("id", "type", "kind", "prefix", "middle", "suffix", "snippet_id", "collection")}))
    return out


def decontaminate(recs: list[dict], evals: list[tuple[str, dict]], min_content: int = 2, seed: int = 20261008) -> tuple[list[dict], dict]:
    res = dc.run(recs, evals, seed=seed, min_content=min_content)
    removed = res["removed"]
    # remove o DOCUMENTO inteiro (todos os cortes do mesmo snippet)
    bad_docs = {dc.doc_key(r) for r in recs if r["id"] in removed}
    kept = [r for r in recs if dc.doc_key(r) not in bad_docs]
    rep = {"train_in": res["train_in"], "removed_cuts_direct": res["train_removed"], "removed_docs": len(bad_docs), "kept": len(kept),
           "by_rule": res["by_rule"], "matched_against": res["matched_against"], "eval_docs": res["eval_docs"], "min_content": min_content,
           "removed_by_cut_kind": None}
    return kept, rep


# ----------------------------------------------------------------------------------------------------------------------------
def build(validated: list[dict], role: str, out_dir: Path, cfg: dict, *, n_cuts: int | None = None, seed: int | None = None,
          do_decontam: bool = True, hom_eval=None) -> dict:
    t0 = time.time()
    seed = int(seed if seed is not None else cfg["seed"])
    dscfg = load_dataset_cfg()
    infos = {i.full_id: i for i in load_schema_infos(roles=(role,))}
    all_train = load_schema_infos(roles=("train",))
    all_hom = load_schema_infos(roles=("hom",))
    tc = gf.middle_tokens_for(dscfg)
    tok = make_tokenizer(dscfg)
    snips, sk = [], Counter()
    for q in validated:
        info = infos.get(q["schema_id"])
        if info is None or info.role != role:
            sk["schema_fora_do_papel"] += 1
            continue
        s = snippet_from_query(q, info, seed)
        snips.append(s)
    n = n_cuts or int(cfg["fim"]["per_query_cuts"])
    cuts = []
    for s in snips:
        cs = cuts_for_snippet(s, dscfg, tc, seed, n)
        for c in cs:
            finish_fim_record(c, s, None, cfg)
        cuts += cs
    names = {}
    for info in infos.values():
        for coll in info.schema.colls:
            names[(info.full_id, coll.name)] = schema_names(info, coll)
    share = float(cfg["fim"]["min_no_schema_visible"]) + 0.02
    ratios = {"none": cfg["fim"]["context_ratios"]["none"], "irrelevant": cfg["fim"]["context_ratios"]["irrelevant"]}
    kinds, kst = assign_context_kinds(cuts, names, share, seed, ratios)
    schemas_for_pools = [i.schema for i in infos.values()]
    dist = [i.schema for i in (all_train if role == "train" else all_hom + all_train)]
    pools = LlmPools([{"id": s["id"], "op": s["op"], "style": s["style"], "text": s["text"], "schema_id": s["schema_id"],
                       "collection": s["collection"], "split": s["split"], "group_id": s["group_id"]} for s in snips],
                     list({s.full_id: s for s in schemas_for_pools + dist}.values()), dist)
    recs = contextualize(cuts, kinds, pools, dscfg, tok, seed)
    # guarda de isolamento
    hom_dropped = 0
    if role == "train":
        # prompt/id identico a item selado da homologacao NAO e fatal: prefixos curtos e genericos (p.ex. contexto none)
        # coincidem entre schemas diferentes; o candidato e descartado e contado, e a decontaminacao n-grama roda em seguida
        kept = []
        for r in recs:
            try:
                hg.check_record(r)
                hg.check_text(r["prefix"] + "\x00" + r["suffix"])
            except hg.HomologAccessError:
                hom_dropped += 1
                continue
            kept.append(r)
        recs = kept
    rep = None
    if do_decontam:
        evals = eval_docs_v2c()
        evals += hom_eval if hom_eval is not None else (homolog_docs() if role == "train" else [])
        recs, rep = decontaminate(recs, evals, int(cfg["fim"]["decontam"]["ngram_min_content"]), seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    nrec = write_jsonl(out_dir / "fim_ctx.jsonl", recs)
    st = {"role": role, "queries": len(validated), "snippets": len(snips), "skipped": dict(sk), "cuts_generated": len(cuts), "cuts_final": nrec,
          "context_assignment": kst, "homolog_exact_dropped": hom_dropped, "decontam": rep, "seconds": round(time.time() - t0, 1),
          "by_cut_kind": dict(Counter(r["cut_kind"] for r in recs)), "by_context_kind": dict(Counter(r["context_kind"] for r in recs)),
          "by_category": dict(Counter(r["category"] for r in recs)), "by_domain": dict(Counter(r["domain"] for r in recs)),
          "middle_tokens_method": dict(Counter(r["middle_tokens_method"] for r in recs)),
          "none_or_irrelevant_share": round(sum(1 for r in recs if r["context_kind"] != "full") / max(1, len(recs)), 4)}
    (out_dir / "fim_ctx.manifest.json").write_text(json.dumps(st, indent=1, ensure_ascii=False), encoding="utf-8")
    return st


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--role", choices=["train", "hom"], default="train")
    ap.add_argument("--cuts", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--no-decontam", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_cfg(a.config)
    if a.role == "train":
        hg.enforce()          # o treino nunca le data/homolog (salvo a leitura autorizada e registrada da decontaminacao)
    st = build(read_jsonl(ROOT / a.inp), a.role, ROOT / a.out, cfg, n_cuts=a.cuts, seed=a.seed, do_decontam=not a.no_decontam)
    print(json.dumps(st, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
