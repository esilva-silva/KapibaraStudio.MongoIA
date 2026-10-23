#!/usr/bin/env python3
"""Corpus dos alunos T0 Nano / T1 Small: bracos DISTILADO (rotulo do professor) e CONTROLE/J10 (rotulo gold), mesmos prompts e mesma ordem.

  C:/SlopStudio.MongoAI-artifacts/.venv/Scripts/python.exe -I scripts/student/build_student_corpus.py \
      --config configs/student_corpus.yaml [--tiers t1,t0] [--workers 10] [--report reports/student/corpus_report.md]
  ... --verify-only          # relê os .meta.jsonl e refaz as checagens de isolamento/pareamento (sem renderizar)

Entrada : data/distilled/v2c-final/accepted.jsonl (prompts FIM `editor-context-v1` + alvo do professor + `gold`), pool de 300k prompts (so a
          composicao, para o balanco), data/replay/v1 (replay de codigo generico), val do corpus do professor (data/teacher/v2d/val).
Saida   : data/student/{distill,gold}/{t0,t1}/{train-NNNN.npz,train-NNNN.meta.jsonl,val/,manifest.json}  (formato lido por train_teacher.Corpus)
          data/student/selection/{t0,t1}.jsonl (id, estrato, componente) e reports/student/corpus_report.md.
Pareamento: as duas pastas de um tier tem os MESMOS ids, componentes, ordem e linhas (comprimento de empacotamento = prompt + max(alvo
          distilado, alvo gold)); so o rotulo de fim_ctx/completion muda. O gold nunca decide a escolha entre candidatas.
G2.7    : so `train`; `Isolation` (ids/grupos/schemas/snippets/prompt_sha/prefixo+sufixo/dominios holdout/cego), ids em train.ids.txt e fora de
          val/test/bench, conteudo (prefixo+sufixo+middle) fora de eval_docs, replay x generic_js_fim_eval/monitor. Violacao => aborta.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "scripts" / "distill", ROOT / "scripts" / "teacher", ROOT / "scripts" / "data"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

import blind_guard  # noqa: E402
import hom_guard  # noqa: E402
import build_teacher_corpus as btc  # noqa: E402  (chunk_replay_doc, render_replay_chunk, make_rows, _TokCounter)
import render_contract as rc  # noqa: E402
from _distill_common import BlindChecker, iter_pool  # noqa: E402
from build_prompt_pool import Isolation, ps_sha  # noqa: E402

EOS_ID = 151643
FIM_IDS = {"<|fim_prefix|>": 151659, "<|fim_middle|>": 151660, "<|fim_suffix|>": 151661, "<|endoftext|>": 151643}
STRATA_KEYS = ("cut_kind", "category", "context_kind")
REPORT_DIMS = ("cut_kind", "category", "context_kind", "domain", "cursor_role", "origin")
# campos do registro aceito que o construtor mantem (o resto - teacher, facts - nao e necessario e pesaria na memoria)
KEEP = ("id", "parent_id", "origin", "prefix", "suffix", "editor", "language", "dialect", "cut_kind", "category", "context_kind", "domain",
        "cursor_role", "group_id", "schema_id", "snippet_id", "split", "holdout", "split_role", "prompt_sha256", "prompt_tokens", "gold",
        "target", "agrees_with_gold", "label_source")


# -- utilidades ---------------------------------------------------------------------------------------------------------------------
def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def unit(seed: int, salt: str, ident: str) -> float:
    """Hash uniforme em [0,1) de seed:salt:id (blake2b 64 bits): amostragem determinista independente da ordem e do rotulo."""
    return int.from_bytes(hashlib.blake2b(f"{seed}:{salt}:{ident}".encode("utf-8"), digest_size=8).digest(), "big") / 2.0 ** 64


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def content_hash(prefix: str, suffix: str, middle: str) -> int:
    """Mesma formula de build_teacher_corpus.content_hash (registros FIM)."""
    return int.from_bytes(hashlib.sha1(f"{prefix}\x00{suffix}\x00{middle}".encode("utf-8", "surrogatepass")).digest()[:8], "big")


def read_ids(path: Path) -> set[str]:
    with Path(path).open(encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def git_hash() -> str:
    try:
        import subprocess
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def stratum_of(rec: dict) -> tuple:
    return tuple(rec[k] for k in STRATA_KEYS)


# -- selecao estratificada ----------------------------------------------------------------------------------------------------------
def allocate(total: int, target_share: dict, avail: dict) -> dict:
    """Alocacao proporcional a `target_share` com teto `avail` (water-filling): estratos que estouram o teto ficam no teto e o resto e
    redistribuido entre os demais. Inteiros por maiores restos. Determinista. Soma = min(total, sum(avail))."""
    keys = sorted(avail, key=str)
    cap = {k: int(avail[k]) for k in keys}
    out: dict = {k: 0 for k in keys}
    active = [k for k in keys if cap[k] > 0 and target_share.get(k, 0) > 0]
    remaining = min(int(total), sum(cap.values()))
    while active and remaining > 0:
        wsum = sum(target_share[k] for k in active)
        want = {k: remaining * target_share[k] / wsum for k in active}
        capped = [k for k in active if want[k] >= cap[k] - out[k]]
        if not capped:
            base = {k: int(math.floor(want[k])) for k in active}
            left = remaining - sum(base.values())
            for k in sorted(active, key=lambda k: (-(want[k] - base[k]), str(k)))[:left]:
                base[k] += 1
            for k in active:
                out[k] += min(base[k], cap[k] - out[k])
            break
        for k in capped:
            take = cap[k] - out[k]
            out[k] += take
            remaining -= take
        active = [k for k in active if k not in capped]
    # sobras de arredondamento (teto de algum estrato impediu o +1): entrega aos estratos com folga, por prioridade determinista
    short = min(int(total), sum(cap.values())) - sum(out.values())
    if short > 0:
        for k in sorted((k for k in keys if cap[k] - out[k] > 0 and target_share.get(k, 0) > 0), key=lambda k: (-target_share[k], str(k))):
            take = min(short, cap[k] - out[k])
            out[k] += take
            short -= take
            if short <= 0:
                break
    return out


def select_ids(records: list[dict], total: int, target_share: dict, seed: int, salt: str, caps: dict | None = None) -> dict[str, tuple]:
    """-> {id: estrato}. `caps` (estrato -> n) limita a alocacao (T0 dentro da selecao do T1). Prioridade = menor hash(seed, salt, id)."""
    by_stratum: dict = defaultdict(list)
    for r in records:
        by_stratum[stratum_of(r)].append(r["id"])
    avail = {s: len(v) for s, v in by_stratum.items()}
    if caps is not None:
        avail = {s: min(avail.get(s, 0), caps.get(s, 0)) for s in set(avail) | set(caps)}
    alloc = allocate(total, target_share, avail)
    chosen: dict[str, tuple] = {}
    for s, n in alloc.items():
        if n <= 0:
            continue
        for ident in heapq.nsmallest(n, by_stratum[s], key=lambda i: (unit(seed, salt, i), i)):
            chosen[ident] = s
    return chosen


def component_of(ident: str, eligible: bool, seed: int, q_completion: float, raw_given_rest: float) -> str:
    """Componente por id (hash determinista; aninhado entre tiers; NAO depende do rotulo): completion so se elegivel, depois fim_raw, senao fim_ctx."""
    if eligible and unit(seed, "comp-c", ident) < q_completion:
        return "completion"
    return "fim_raw" if unit(seed, "comp-r", ident) < raw_given_rest else "fim_ctx"


# -- renderizacao (workers) ---------------------------------------------------------------------------------------------------------
_TOK = None
_CFG = None


def _init_worker(tok_path: str, cfg: dict) -> None:
    global _TOK, _CFG
    _TOK = rc.Tokenizer(tok_path)
    _CFG = cfg


def render_example(rec: dict, component: str, tok, ccfg: dict) -> dict | None:
    """-> {p, t_distill, t_gold, ...} (listas de ids) ou None (descartado, motivo em `drop`). O prompt e o MESMO ids que o professor viu."""
    m_prefix, m_suffix, m_middle = tok.marker_ids  # (151659, 151661, 151660)
    prefix_w, suffix_w = rc.bound_window(rec["prefix"], rec["suffix"])
    if component == "fim_raw":
        before = prefix_w
    else:
        before = rc.render_model_prefix(prefix_w, rec.get("editor"), rec.get("language") or rc.editor_language(rec))
    ids, info = rc.build_ids(tok, before, suffix_w)
    if component != "fim_raw" and len(ids) != rec["prompt_tokens"]:
        return {"drop": "prompt_tokens_mismatch"}
    if ids.count(m_prefix) != 1 or ids.count(m_suffix) != 1 or ids.count(m_middle) != 1:
        return {"drop": "marker_in_text"}
    if component == "completion":
        head = ids[: ids.index(m_suffix)]
        prompt = head + [m_suffix, m_middle]
        t_d = tok.encode(rec["target"] + rec["suffix"]) + [EOS_ID]
        t_g = tok.encode(rec["gold"] + rec["suffix"]) + [EOS_ID]
        if max(len(t_d), len(t_g)) - 1 > ccfg["max_target_tokens"]:
            return {"drop": "completion_gt_96"}
    else:
        prompt = ids
        t_g = tok.encode(rec["gold"]) + [EOS_ID]
        t_d = (tok.encode(rec["target"]) + [EOS_ID]) if component == "fim_ctx" else t_g
    return {"p": np.asarray(prompt, dtype=np.uint32), "t_distill": np.asarray(t_d, dtype=np.uint32), "t_gold": np.asarray(t_g, dtype=np.uint32),
            "truncated": bool(info["truncated"])}


def _render_chunk(args: tuple) -> list[dict]:
    items, ccfg = args
    out = []
    for rec, component in items:
        res = render_example(rec, component, _TOK, ccfg)
        out.append({"id": rec["id"], "component": component, **(res or {"drop": "none"})})
    return out


# -- replay -------------------------------------------------------------------------------------------------------------------------
def load_replay(cfg: dict, tok_lib, seed: int, seq_len: int, eval_content_hashes: set[int]) -> dict:
    """Trechos de replay prontos (ids), mais o isolamento contra generic_js_fim_eval/monitor e contra o conteudo do Mongo val/test/bench."""
    rcfg = cfg["replay"]
    rdir = ROOT / rcfg["source_dir"]
    blind_guard.check_path(rdir)
    docs = [json.loads(line) for line in (rdir / rcfg["train_file"]).open(encoding="utf-8")]
    eval_rows = [json.loads(line) for line in (rdir / rcfg["eval_file"]).open(encoding="utf-8")]
    monitor_rows = [json.loads(line) for line in (rdir / rcfg["monitor_file"]).open(encoding="utf-8")]
    eval_pkgs = {r["package"] for r in eval_rows} | {r["package"] for r in monitor_rows}
    eval_shas = {r["file_sha256"] for r in eval_rows} | {r["file_sha256"] for r in monitor_rows}
    manifest = json.loads((rdir / "manifest.json").read_text(encoding="utf-8"))
    holdout_pkgs = {p["package"] for p in manifest["packages"] if p["role"] == "eval_holdout"}
    bad = [d["id"] for d in docs if d["package"] in eval_pkgs or d["package"] in holdout_pkgs or d["sha256"] in eval_shas]
    if bad:
        raise SystemExit(f"G2.7/replay: {len(bad)} documentos de replay em pacote/arquivo do holdout generic_js ({bad[:3]})")
    tcount = btc._TokCounter(tok_lib)  # noqa: SLF001
    chunks = []
    variants = max(1, int(rcfg.get("max_variants", 1)))
    for d in docs:
        parts = btc.chunk_replay_doc(d["text"], int(rcfg.get("window_chars", 4000)))
        for n, text in enumerate(parts):
            for rep in range(variants):   # rep>0: MESMO trecho com outro sorteio (tipo FIM/causal e ponto de corte) - id `...~v<rep>`
                ident = f"{d['id']}-{n:03d}" + (f"~v{rep}" if rep else "")
                chunks.append({"id": ident, "text": text, "last": n == len(parts) - 1, "doc": d, "rep": rep})
    rcfg_view = {"replay": rcfg}
    for c in chunks:
        c["kind"], c["prompt"], c["target"] = btc.render_replay_chunk(c["text"], c["id"], rcfg_view, seed, tcount)
    enc_p = tok_lib.encode_batch([c["prompt"] for c in chunks], add_special_tokens=False)
    enc_t = tok_lib.encode_batch([c["target"] for c in chunks], add_special_tokens=False)
    for c, ep, et in zip(chunks, enc_p, enc_t):
        c["p_ids"] = ep.ids if c["prompt"] else []
        c["t_ids"] = et.ids + ([EOS_ID] if (c["kind"] == "fim" or c["last"]) else [])
        c["tokens"] = len(c["p_ids"]) + len(c["t_ids"])
    needles = [r["prefix"][-80:] + r["middle"] + r["suffix"][:80] for r in eval_rows + monitor_rows]
    contaminated = {c["id"] for c in chunks if any(n in c["text"] for n in needles)}
    chunks = [c for c in chunks if c["id"] not in contaminated and c["tokens"] <= seq_len]
    chunks.sort(key=lambda c: (c["rep"], unit(seed, "replay-pick", c["id"]), c["id"]))   # rep 0 de todos antes de qualquer rep 1
    return {"chunks": chunks, "contaminated": len(contaminated), "needles": needles, "eval_pkgs": eval_pkgs | holdout_pkgs, "eval_shas": eval_shas,
            "manifest": manifest, "dir": rdir, "eval_content": eval_content_hashes,
            "mean_tokens": sum(c["tokens"] for c in chunks) / max(1, len(chunks))}


def pick_replay(replay: dict, budget_tokens: int) -> tuple[list[dict], dict]:
    """Prefixo da ordem por hash que cabe no orcamento (para ao primeiro estouro => aninhado entre tiers) + prova de isolamento."""
    picked, used = [], 0
    for c in replay["chunks"]:
        if used + c["tokens"] > budget_tokens:
            break
        picked.append(c)
        used += c["tokens"]
    needles = replay["needles"]
    sub_hits = sum(1 for c in picked if any(n in c["text"] for n in needles))
    mongo_hits = sum(1 for c in picked if int.from_bytes(hashlib.sha1(c["text"].encode("utf-8", "surrogatepass")).digest()[:8], "big")
                     in replay["eval_content"])
    pk_overlap = sorted({c["doc"]["package"] for c in picked} & replay["eval_pkgs"])
    sha_overlap = len({c["doc"]["sha256"] for c in picked} & replay["eval_shas"])
    iso = {"package_overlap_with_generic_eval": pk_overlap, "file_sha256_overlap_with_generic_eval": sha_overlap,
           "chunks_containing_eval_cut": sub_hits, "chunks_matching_mongo_eval_content": mongo_hits,
           "chunks_dropped_containing_eval_cut": replay["contaminated"], "source_manifest_isolation": replay["manifest"].get("isolation")}
    if pk_overlap or sha_overlap or sub_hits or mongo_hits:
        raise SystemExit(f"G2.7/replay: vazamento replay x generic_js_fim_eval/val/test/bench: {iso}")
    stats = {"examples": len(picked), "tokens": used, "budget": budget_tokens, "by_kind": dict(Counter(c["kind"] for c in picked)),
             "docs": len({c["doc"]["id"] for c in picked}), "packages": len({c["doc"]["package"] for c in picked}),
             "tokens_by_license": dict(Counter({lic: sum(c["tokens"] for c in picked if c["doc"]["license"] == lic)
                                                for lic in {c["doc"]["license"] for c in picked}})),
             "variants_used": dict(Counter(c["rep"] for c in picked)), "available_chunks": len(replay["chunks"]), "available_tokens": sum(c["tokens"] for c in replay["chunks"]), "isolation": iso}
    return picked, stats


# -- gravacao -----------------------------------------------------------------------------------------------------------------------
def write_shards(out_dir: Path, rows: list[list[int]], examples: list[dict], arm: str, micro_tokens: int, shard_examples: int) -> list[dict]:
    """Grava as linhas (ja empacotadas, iguais nos dois bracos) em shards .npz + .meta.jsonl. `examples[i]` tem p (uint32), t_<arm>, meta."""
    shards, buf_rows, count = [], [], 0
    groups: list[list[list[int]]] = []
    for row in rows:
        buf_rows.append(row)
        count += len(row)
        if count >= shard_examples:
            groups.append(buf_rows)
            buf_rows, count = [], 0
    if buf_rows:
        groups.append(buf_rows)
    for n, group in enumerate(groups):
        ordered = [i for row in group for i in row]
        row_ptr = np.zeros(len(group) + 1, dtype=np.int32)
        row_ptr[1:] = np.cumsum([len(r) for r in group])
        seqs = [np.concatenate([examples[i]["p"], examples[i][f"t_{arm}"]]) for i in ordered]
        ex_offsets = np.zeros(len(ordered) + 1, dtype=np.int64)
        ex_offsets[1:] = np.cumsum([len(s) for s in seqs])
        ids = np.concatenate(seqs).astype(np.uint32) if seqs else np.zeros(0, dtype=np.uint32)
        loss_start = np.array([len(examples[i]["p"]) for i in ordered], dtype=np.int32)
        path = out_dir / f"train-{n:04d}.npz"
        np.savez(path, ids=ids, ex_offsets=ex_offsets, loss_start=loss_start, row_ptr=row_ptr)
        meta_path = path.with_suffix(".meta.jsonl")
        with meta_path.open("w", encoding="utf-8", newline="\n") as handle:
            for i in ordered:
                m = {k: v for k, v in examples[i]["meta"].items() if k not in ("label_hash_distill", "label_hash_gold")}
                m["n_target"] = int(len(examples[i][f"t_{arm}"]))
                handle.write(json.dumps(m, ensure_ascii=False) + "\n")
        padded = sum(max(len(seqs[k]) for k in range(a, b)) * (b - a)
                     for a, b in zip(row_ptr[:-1].tolist(), row_ptr[1:].tolist()) if b > a)
        shards.append({"file": path.name, "sha256": sha256_file(path), "meta_file": meta_path.name, "meta_sha256": sha256_file(meta_path),
                       "examples": len(ordered), "rows": len(group), "tokens": int(ex_offsets[-1]),
                       "target_tokens": int(sum(len(examples[i][f"t_{arm}"]) for i in ordered)), "padded_tokens": int(padded),
                       "pad_fraction": round(1 - ex_offsets[-1] / padded, 4) if padded else 0.0})
    return shards


def pack_rows(pack_lengths: list[int], micro_tokens: int, max_per_row: int, seed: int, tag: str) -> list[list[int]]:
    """Mesmo empacotamento do corpus do professor (ordena por comprimento, linha = n x Lmax <= micro_tokens, linhas embaralhadas)."""
    return btc.make_rows(pack_lengths, micro_tokens, max_per_row, random.Random(f"{seed}-rows-{tag}"))


def build_val(cfg: dict, out_dir: Path, train_ids: set[str], micro_tokens: int, max_per_row: int, seed: int) -> dict:
    """val do professor (v2d) so com FIM: val_loss (fim_ctx + replay_fim do monitor), val_gen e val_gen_generic copiados."""
    vdir = ROOT / cfg["teacher_val_dir"]
    val_out = out_dir / "val"
    val_out.mkdir(parents=True, exist_ok=True)
    meta = [json.loads(line) for line in (vdir / "val_loss.meta.jsonl").read_text(encoding="utf-8").splitlines() if line]
    with np.load(vdir / "val_loss.npz") as data:
        ids, off, ls = data["ids"], data["ex_offsets"], data["loss_start"]
    keep = [i for i, m in enumerate(meta) if m["component"] in ("fim_ctx", "replay_fim")]
    leak = [meta[i]["id"] for i in keep if not meta[i]["id"].startswith(("rpl-",)) and meta[i]["id"] in train_ids]
    if leak:
        raise SystemExit(f"G2.7: ids de val dentro do treino dos alunos: {leak[:3]}")
    examples = [{"p": ids[off[i]: off[i] + ls[i]], "t_gold": ids[off[i] + ls[i]: off[i + 1]], "meta": meta[i]} for i in keep]
    lengths = [len(e["p"]) + len(e["t_gold"]) for e in examples]
    rows = btc.make_rows(lengths, micro_tokens, max_per_row, random.Random(f"{seed}-val-rows"))
    shards = write_shards(val_out, rows, examples, "gold", micro_tokens, 10 ** 9)
    for name in ("train-0000.npz", "train-0000.meta.jsonl"):
        os.replace(val_out / name, val_out / name.replace("train-0000", "val_loss"))
    for name in ("val_gen.jsonl", "val_gen_generic.jsonl"):
        (val_out / name).write_bytes((vdir / name).read_bytes())
    gen_ids = [json.loads(line)["id"] for line in (val_out / "val_gen.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if any(i in train_ids for i in gen_ids):
        raise SystemExit("G2.7: ids de val_gen dentro do treino dos alunos")
    return {"loss_file": "val/val_loss.npz", "loss_sha256": sha256_file(val_out / "val_loss.npz"), "examples": len(examples),
            "tokens": int(sum(lengths)), "gen_file": "val/val_gen.jsonl", "gen_sha256": sha256_file(val_out / "val_gen.jsonl"),
            "gen_examples": len(gen_ids), "gen_generic_file": "val/val_gen_generic.jsonl",
            "gen_generic_sha256": sha256_file(val_out / "val_gen_generic.jsonl"), "source": rel(vdir),
            "source_components": dict(Counter(meta[i]["component"] for i in keep)), "shard": shards[0]["file"]}


# -- verificacao independente (G2.7 + pareamento) -----------------------------------------------------------------------------------
def verify_pair(root: Path, tier: str, splits_dir: Path) -> dict:
    """Relê os meta.jsonl de data/student/{distill,gold}/<tier>: ids em train.ids.txt, fora de val/test/bench, e IDENTICOS (ids, ordem,
    componentes, linhas) entre os bracos; diferenca de rotulo so em fim_ctx/completion."""
    train, val, test, bench = (read_ids(splits_dir / f"{n}.ids.txt") for n in ("train", "val", "test", "bench"))
    arms = {}
    for arm in ("distill", "gold"):
        d = root / arm / tier
        man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        metas, row_sizes = [], []
        for sh in man["shards"]:
            with (d / sh["meta_file"]).open(encoding="utf-8") as handle:
                metas += [json.loads(line) for line in handle]
            with np.load(d / sh["file"]) as data:
                row_sizes += np.diff(data["row_ptr"]).tolist()
        arms[arm] = (man, metas, row_sizes)
    issues: list[str] = []
    seen: set[str] = set()
    for m in arms["distill"][1]:
        i = m["id"]
        if m["component"].startswith("replay"):
            if not i.startswith("rpl-") or i in train or i in val or i in test or i in bench:
                issues.append(f"replay id suspeito {i}")
            continue
        base_id = i.split("~r1")[0]
        if base_id not in train:
            issues.append(f"{i}: nao esta em train.ids.txt")
        for name, s in (("val", val), ("test", test), ("bench", bench)):
            if base_id in s or i in s:
                issues.append(f"{i}: presente em {name}.ids.txt")
        if i in seen:
            issues.append(f"{i}: duplicado")
        seen.add(i)
    d_ids = [(m["id"], m["component"]) for m in arms["distill"][1]]
    g_ids = [(m["id"], m["component"]) for m in arms["gold"][1]]
    same_order = d_ids == g_ids
    same_rows = arms["distill"][2] == arms["gold"][2]
    differ = defaultdict(lambda: [0, 0])
    for md, mg in zip(arms["distill"][1], arms["gold"][1]):
        differ[md["component"]][0] += 1
        differ[md["component"]][1] += int(md["label_hash"] != mg["label_hash"])
    bad_diff = {c: v for c, v in differ.items() if c not in ("fim_ctx", "completion") and v[1]}
    if not same_order:
        issues.append("distill e gold NAO tem os mesmos ids/componentes na mesma ordem")
    if not same_rows:
        issues.append("distill e gold NAO tem as mesmas linhas (micro-batches)")
    if bad_diff:
        issues.append(f"rotulos diferem em componentes sem professor: {bad_diff}")
    return {"tier": tier, "examples": len(d_ids), "mongo_examples": len(seen), "same_ids_same_order": same_order, "same_rows": same_rows,
            "label_differs": {c: {"n": v[0], "differ": v[1]} for c, v in differ.items()}, "issues": issues[:20], "n_issues": len(issues),
            "in_train_ids": len({i.split("~r1")[0] for i in seen} & train), "in_val_ids": len(seen & val), "in_test_ids": len(seen & test),
            "in_bench_ids": len(seen & bench), "ids_sha256": sha256_text("\n".join(i for i, _ in d_ids)),
            "rows_sha256": sha256_text(",".join(map(str, arms["distill"][2])))}


# -- principal ----------------------------------------------------------------------------------------------------------------------
def load_accepted(path: Path, limit: int | None, iso: Isolation, train_ids: set[str], other_ids: dict[str, set[str]],
                  eval_content: set[int], blind: BlindChecker) -> tuple[list[dict], dict]:
    """Le o aceito, aplica G2.7 por registro e descarta o que nao for isolado. Devolve (registros, contagens)."""
    recs, stats = [], Counter()
    with Path(path).open(encoding="utf-8") as handle:
        for n, line in enumerate(handle):
            if limit and n >= limit:
                break
            r = json.loads(line)
            stats["read"] += 1
            hom_guard.assert_no_hom([r], f"aceito do professor {path}")   # abort: hom nunca vira corpus do aluno
            why = iso.violations(r)
            base_id = r.get("parent_id") or r["id"]
            if base_id not in train_ids:
                why.append("id_not_in_train_ids")
            for name, s in other_ids.items():
                if base_id in s or r["id"] in s:
                    why.append(f"id_in_{name}")
            if r.get("label_source") != "teacher_seq":
                why.append("label_source")
            for lab in (r["target"], r["gold"]):
                if content_hash(r["prefix"], r["suffix"], lab) in eval_content:
                    why.append("content_in_eval")
                    break
            if why:
                stats["isolation_dropped"] += 1
                for w in set(why):
                    stats[f"violation:{w}"] += 1
                continue
            recs.append({k: r.get(k) for k in KEEP})
    if blind.sealed:
        stats["blind_sealed"] = 1
    return recs, dict(stats)


def target_shares(pool_dir: Path, limit: int | None) -> tuple[dict, dict]:
    """Composicao dos estratos no POOL de prompts (antes da geracao do professor): alvo do balanco; nao depende do rotulo nem do gold."""
    cnt: Counter = Counter()
    dims = {k: Counter() for k in REPORT_DIMS}
    total = 0
    for _, r in iter_pool(pool_dir, 0, limit):
        cnt[stratum_of(r)] += 1
        total += 1
        for k in REPORT_DIMS:
            dims[k][r.get(k)] += 1
    return {s: n / total for s, n in cnt.items()}, {"total": total, "dims": {k: dict(v) for k, v in dims.items()}}


def dist(records: list[dict], key: str) -> dict:
    c = Counter(r[key] for r in records)
    n = max(1, len(records))
    return {str(k): round(v / n, 5) for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))}


def balance_report(sel: list[dict], pool_info: dict) -> dict:
    out = {}
    for k in REPORT_DIMS:
        s = dist(sel, k)
        p = {str(a): b / pool_info["total"] for a, b in pool_info["dims"][k].items()}
        keys = sorted(set(s) | set(p))
        out[k] = {"selected": s, "pool": {a: round(p.get(a, 0), 5) for a in keys},
                  "max_abs_dev_pp": round(100 * max(abs(s.get(a, 0) - p.get(a, 0)) for a in keys), 3)}
    return out


def run(args: argparse.Namespace) -> int:
    t0 = time.time()
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    seed = int(cfg["seed"])
    out_root = Path(args.out_root) if args.out_root else ROOT / "data" / "student"
    splits_dir = ROOT / cfg["splits_dir"]
    blind_guard.check_path(out_root)
    if args.verify_only:
        res = {t: verify_pair(out_root, t, splits_dir) for t in args.tiers.split(",") if (out_root / "distill" / t / "manifest.json").exists()}
        print(json.dumps(res, indent=1))
        return 0 if all(v["n_issues"] == 0 for v in res.values()) else 3

    tok_path = ROOT / cfg["tokenizer"]
    tok = rc.Tokenizer(tok_path)
    tok_sha = sha256_file(tok_path)
    for other in cfg.get("same_tokenizer_as", []):
        if sha256_file(ROOT / other) != tok_sha:
            raise SystemExit(f"tokenizer de {other} difere do de {cfg['tokenizer']}")
    for text, want in FIM_IDS.items():
        if tok.encode(text) != [want]:
            raise SystemExit(f"tokenizer: {text} -> {tok.encode(text)}, esperado [{want}]")

    dcfg = cfg["distilled"]
    print("[1/7] isolamento: indice de avaliacao e splits", flush=True)
    index = json.loads((ROOT / dcfg["eval_index"]).read_text(encoding="utf-8"))
    blind = BlindChecker()
    iso = Isolation(index, blind)
    train_ids = read_ids(splits_dir / "train.ids.txt")
    other_ids = {n: read_ids(splits_dir / f"{n}.ids.txt") for n in ("val", "test", "bench")}
    eval_content: set[int] = set()
    with (ROOT / dcfg["eval_docs"]).open(encoding="utf-8") as handle:
        for line in handle:
            d = json.loads(line)
            if d.get("type", "fim") == "fim":
                eval_content.add(content_hash(d["prefix"], d["suffix"], d["middle"]))

    print("[2/7] lendo o aceito do professor + G2.7 por registro", flush=True)
    records, load_stats = load_accepted(ROOT / dcfg["accepted"], args.limit, iso, train_ids, other_ids, eval_content, blind)
    print(f"      {load_stats}", flush=True)
    if load_stats.get("isolation_dropped"):
        raise SystemExit(f"G2.7: {load_stats['isolation_dropped']} registros do aceito violam o isolamento ({load_stats}); revisar o filtro")
    shares, pool_info = target_shares(ROOT / dcfg["pool_dir"], args.limit and args.limit * 2)
    ccfg = cfg["completion"]
    mix = cfg["mix"]
    by_id = {r["id"]: r for r in records}
    elig = {r["id"]: (len(r["target"]) + len(r["suffix"]) <= ccfg["max_chars"] and len(r["gold"]) + len(r["suffix"]) <= ccfg["max_chars"])
            for r in records}
    elig_frac = sum(elig.values()) / max(1, len(records))
    # q_c: probabilidade (entre elegiveis) de virar completion, tal que a fracao global de completion = mix.completion
    q_completion = min(1.0, mix["completion"] / max(1e-9, elig_frac))
    # fim_raw = mix.fim_raw do total (a fracao de completion e decidida primeiro; o resto sorteia raw/ctx proporcionalmente)
    raw_given_rest = mix["fim_raw"] / max(1e-9, 1.0 - mix["completion"])

    def comp_of(ident: str) -> str:
        return component_of(ident, elig[ident], seed, q_completion, raw_given_rest)

    print(f"[3/7] replay (budget {cfg['replay']['token_share']:.0%} dos tokens)", flush=True)
    replay = load_replay(cfg, tok._tok, seed, cfg["seq_len"], eval_content) if cfg["replay"]["enabled"] else None  # noqa: SLF001

    src_hashes = {"accepted_sha256": sha256_file(ROOT / dcfg["accepted"]), "filter_report_sha256": sha256_file(ROOT / dcfg["filter_report"]),
                  "pool_manifest_sha256": sha256_file(ROOT / dcfg["pool_dir"] / "pool.manifest.json"),
                  "eval_index_sha256": sha256_file(ROOT / dcfg["eval_index"]), "eval_docs_sha256": sha256_file(ROOT / dcfg["eval_docs"]),
                  "train_ids_sha256": sha256_file(splits_dir / "train.ids.txt")}
    tiers = [t for t in args.tiers.split(",") if t]
    order = sorted(tiers, key=lambda t: -cfg["tiers"][t]["total_examples"])    # t1 antes de t0 (t0 usa o teto de t1)
    caps_prev: dict | None = None
    summary: dict = {}
    for tier in order:
        tcfg = cfg["tiers"][tier]
        total = int(tcfg["total_examples"])
        share_replay = float(cfg["replay"]["token_share"]) if replay else 0.0
        # N_mongo tal que N_mongo + N_replay ~= total: estima tokens/ex. com a prioridade global (piloto de ate 6000 exemplos)
        pilot_ids = heapq.nsmallest(6000, [r["id"] for r in records], key=lambda i: (unit(seed, cfg["priority_salt"], i), i))
        pilot, pilot_drop = _render_serial([(by_id[i], comp_of(i)) for i in pilot_ids], tok, ccfg)
        t_m = float(np.mean([len(e["p"]) + max(len(e["t_distill"]), len(e["t_gold"])) for e in pilot if "p" in e]))
        t_r = replay["mean_tokens"] if replay else 1.0
        n_mongo = int(round(total / (1.0 + (share_replay / (1 - share_replay)) * (t_m / t_r)))) if replay else total
        n_mongo = int(round(n_mongo / (1.0 - pilot_drop / len(pilot_ids))))      # compensa os descartes da renderizacao (completion > 96 tokens)
        caps = caps_prev if (tcfg.get("subset_of") and caps_prev is not None) else None
        print(f"[4/7] {tier}: selecao estratificada n_mongo={n_mongo} (piloto: {t_m:.0f} tok/ex., replay {t_r:.0f} tok/trecho)", flush=True)
        chosen = select_ids(records, n_mongo, shares, seed, cfg["priority_salt"], caps)
        caps_new = Counter(chosen.values())
        if tcfg.get("subset_of") is None:
            caps_prev = dict(caps_new)
        sel_recs = [by_id[i] for i in chosen]
        items = [(by_id[i], comp_of(i)) for i in sorted(chosen)]
        print(f"[5/7] {tier}: renderizando/tokenizando {len(items)} exemplos ({args.workers} workers)", flush=True)
        chunks = [(items[a: a + 2000], ccfg) for a in range(0, len(items), 2000)]
        results: list[dict] = []
        with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker, initargs=(str(tok_path), cfg)) as pool:
            for res in pool.imap(_render_chunk, chunks):
                results += res
        dropped = Counter(r["drop"] for r in results if "drop" in r)
        good = [r for r in results if "drop" not in r]
        # monta exemplos (ordenados por id: a ordem final vem do empacotamento embaralhado por seed)
        examples = []
        for r in good:
            rec = by_id[r["id"]]
            diff = int(not np.array_equal(r["t_distill"], r["t_gold"]))
            meta = {"id": r["id"], "component": r["component"], "kind": rec["cut_kind"], "domain": rec["domain"], "category": rec["category"],
                    "ctx": rec["context_kind"], "role": rec["cursor_role"], "origin": rec["origin"], "n_prompt": int(len(r["p"])),
                    "agrees_with_gold": bool(rec["agrees_with_gold"]), "label_differs": bool(diff) and r["component"] in ("fim_ctx", "completion"),
                    "src": "accepted"}
            examples.append({"p": r["p"], "t_distill": r["t_distill"], "t_gold": r["t_gold"], "meta": meta, "id": r["id"]})
        nonreplay_tokens = sum(len(e["p"]) + max(len(e["t_distill"]), len(e["t_gold"])) for e in examples)
        picked, rep_stats = ([], {"examples": 0, "tokens": 0, "isolation": None})
        if replay:
            budget = int(round(nonreplay_tokens * share_replay / (1 - share_replay)))
            picked, rep_stats = pick_replay(replay, budget)
        for c in picked:
            t = np.asarray(c["t_ids"], dtype=np.uint32)
            examples.append({"p": np.asarray(c["p_ids"], dtype=np.uint32), "t_distill": t, "t_gold": t, "id": c["id"],
                             "meta": {"id": c["id"], "component": f"replay_{c['kind']}", "kind": c["kind"], "domain": c["doc"]["package"],
                                      "language": "ts" if c["doc"]["ext"] in (".ts", ".tsx") else "js", "category": c["doc"]["license"],
                                      "n_prompt": len(c["p_ids"]), "agrees_with_gold": True, "label_differs": False, "src": "replay"}})
        # hash do rotulo por exemplo/bracos (verificacao de pareamento) e ordem determinista
        for e in examples:
            e["meta"]["label_hash_distill"] = hashlib.sha1(e["t_distill"].tobytes()).hexdigest()[:12]
            e["meta"]["label_hash_gold"] = hashlib.sha1(e["t_gold"].tobytes()).hexdigest()[:12]
        examples.sort(key=lambda e: e["id"])
        pack_len = [len(e["p"]) + max(len(e["t_distill"]), len(e["t_gold"])) for e in examples]
        rows = pack_rows(pack_len, cfg["micro_tokens"], cfg["max_examples_per_row"], seed, tier)
        print(f"[6/7] {tier}: {len(examples)} exemplos ({len(picked)} replay), {len(rows)} linhas; gravando bracos", flush=True)
        sel_dir = out_root / "selection"
        sel_dir.mkdir(parents=True, exist_ok=True)
        with (sel_dir / f"{tier}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for e in examples:
                handle.write(json.dumps({"id": e["id"], "component": e["meta"]["component"], "stratum": e["meta"].get("kind") if e["meta"]["src"] == "replay"
                                         else list(stratum_of(by_id[e["id"]]))}, ensure_ascii=False) + "\n")
        mongo_examples = [e for e in examples if e["meta"]["src"] == "accepted"]
        arm_manifests = {}
        for arm in ("distill", "gold"):
            d = out_root / arm / tier
            d.mkdir(parents=True, exist_ok=True)
            for old in list(d.glob("train-*")) + list((d / "val").glob("*") if (d / "val").exists() else []):
                old.unlink()
            for e in examples:
                e["meta"]["label_hash"] = e["meta"][f"label_hash_{arm}"]
            shards = write_shards(d, rows, examples, arm, cfg["micro_tokens"], int(cfg["shard_examples"]))
            val = build_val(cfg, d, {e["id"] for e in mongo_examples}, cfg["micro_tokens"], cfg["max_examples_per_row"], seed)
            tokens = sum(s["tokens"] for s in shards)
            by_comp: dict = defaultdict(lambda: {"examples": 0, "tokens": 0, "target_tokens": 0, "label_differs": 0})
            for e in examples:
                c = by_comp[e["meta"]["component"]]
                c["examples"] += 1
                c["tokens"] += len(e["p"]) + len(e[f"t_{arm}"])
                c["target_tokens"] += len(e[f"t_{arm}"])
                c["label_differs"] += int(e["meta"]["label_differs"])
            arm_manifests[arm] = {
                "schema": "student-corpus/1", "arm": arm, "tier": tier, "tier_name": tcfg["name"], "base": tcfg["base"],
                "built_utc": time.strftime("%FT%TZ", time.gmtime()), "git_hash": git_hash(), "seed": seed,
                "config": {"path": rel(cfg_path), "sha256": sha256_file(cfg_path)}, "builder_sha256": sha256_file(Path(__file__)),
                "tokenizer": {"path": cfg["tokenizer"], "sha256": tok_sha, "special_ids": FIM_IDS},
                "label_source": "teacher_seq (fim_ctx, completion)" if arm == "distill" else "gold (middle original do dataset v2c)",
                "teacher_labeled_components": cfg["teacher_labeled_components"],
                "source": {"accepted": dcfg["accepted"], **src_hashes, "limit": args.limit},
                "selection": {"n_mongo_target": n_mongo, "n_mongo": len(mongo_examples), "strata": len(set(chosen.values())),
                              "allocation": "proporcional ao pool com teto de disponibilidade (water-filling), prioridade=hash(seed,id)",
                              "subset_of": tcfg.get("subset_of"), "completion_eligible_fraction": round(elig_frac, 4),
                              "q_completion": round(q_completion, 4), "dropped_in_render": dict(dropped),
                              "balance_vs_pool": balance_report([by_id[e["id"]] for e in mongo_examples], pool_info),
                              "accepted_balance": balance_report(records, pool_info)},
                "mix": {"target": mix, "achieved": {k: round(v["examples"] / max(1, len(examples)), 4) for k, v in sorted(by_comp.items())},
                        "achieved_of_mongo": {k: round(v["examples"] / max(1, len(mongo_examples)), 4) for k, v in sorted(by_comp.items())
                                              if not k.startswith("replay")}},
                "replay": {"status": "enabled" if replay else "none", "token_share_target": share_replay,
                           "token_share_achieved": round(rep_stats["tokens"] / max(1, tokens), 4), **rep_stats},
                "counts": {"examples": len(examples), "mongo_examples": len(mongo_examples), "tokens": tokens,
                           "target_tokens": sum(s["target_tokens"] for s in shards), "padded_tokens": sum(s["padded_tokens"] for s in shards),
                           "rows": sum(s["rows"] for s in shards), "mean_tokens_per_example": round(tokens / max(1, len(examples)), 1),
                           "by_component": dict(by_comp), "label_differs_mongo": sum(1 for e in mongo_examples if e["meta"]["label_differs"]),
                           "agrees_with_gold_mongo": sum(1 for e in mongo_examples if e["meta"]["agrees_with_gold"])},
                "micro_tokens": cfg["micro_tokens"], "seq_len": cfg["seq_len"], "val": val, "shards": shards,
                "isolation": {"G2_7": "so train: Isolation(ids, grupos, schemas, snippets, prompt_sha, prefixo+sufixo, holdout, cego) + train.ids.txt + "
                                      "conteudo vs eval_docs, por registro (violacoes=0, aborta caso contrario)",
                              "accepted_load": load_stats, "blind_sealed": blind.sealed,
                              "pool_lines_for_balance": pool_info["total"]},
            }
        for arm in ("distill", "gold"):
            (out_root / arm / tier / "manifest.json").write_text(json.dumps(arm_manifests[arm], indent=1, ensure_ascii=False) + "\n",
                                                                 encoding="utf-8", newline="\n")
        ver = verify_pair(out_root, tier, splits_dir)
        for arm in ("distill", "gold"):
            m = json.loads((out_root / arm / tier / "manifest.json").read_text(encoding="utf-8"))
            m["verification"] = ver
            (out_root / arm / tier / "manifest.json").write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        summary[tier] = {"manifests": arm_manifests, "verification": ver}
        print(f"      {tier}: verificacao n_issues={ver['n_issues']} mesma ordem={ver['same_ids_same_order']} mesmas linhas={ver['same_rows']}", flush=True)
        if ver["n_issues"]:
            print(json.dumps(ver, indent=1), file=sys.stderr)
            return 3
    print("[7/7] relatorio", flush=True)
    write_report(Path(args.report), cfg, summary, load_stats, pool_info, time.time() - t0)
    print(f"OK em {time.time() - t0:.0f}s")
    return 0


def _render_serial(items: list[tuple], tok, ccfg: dict) -> tuple[list[dict], int]:
    out, dropped = [], 0
    for rec, comp in items:
        res = render_example(rec, comp, tok, ccfg)
        if res and "drop" not in res:
            out.append(res)
        else:
            dropped += 1
    return out, dropped


# -- relatorio ----------------------------------------------------------------------------------------------------------------------
def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def write_report(path: Path, cfg: dict, summary: dict, load_stats: dict, pool_info: dict, seconds: float) -> None:
    accum = 2  # tokens_per_step 32768 / micro_tokens 16384 (train_student.py: optim.tokens_per_step)
    L = ["# Corpus dos alunos T0 Nano / T1 Small (autocomplete FIM, editor-context-v1)", "",
         f"Gerado {time.strftime('%FT%T')} por `scripts/student/build_student_corpus.py` (config `configs/student_corpus.yaml`, seed {cfg['seed']}, "
         f"git {git_hash()[:12]}, {seconds:.0f} s). Dados em `data/student/{{distill,gold}}/{{t0,t1}}/` (gitignored).", "",
         "## Bracos e pareamento", "",
         "- **distill**: `fim_ctx` e `completion` rotulados pelo professor (alvo `teacher_seq` de `data/distilled/v2c-final/accepted.jsonl`, sem usar o "
         "gold para escolher candidata; `agrees_with_gold` so nos metadados).",
         "- **gold (controle/J10)**: os MESMOS ids, prompts, componentes, ordem e linhas (micro-batches), com o `middle` original como rotulo.",
         "- `fim_raw` (gold) e `replay` (codigo generico) sao IDENTICOS nos dois bracos. Linhas empacotadas por `prompt + max(alvo distilado, alvo gold)` "
         "-> mesma composicao de lote nos dois bracos.", ""]
    for tier, s in sorted(summary.items(), reverse=True):
        d, g = s["manifests"]["distill"], s["manifests"]["gold"]
        c, ver = d["counts"], s["verification"]
        steps_epoch = c["rows"] // accum
        L += [f"## {tier.upper()} {d['tier_name']} (`{d['base']}`)", "",
              f"- **{c['examples']:,} exemplos** ({c['mongo_examples']:,} Mongo + {d['replay']['examples']:,} replay), **{c['tokens']:,} tokens/epoca** "
              f"(media {c['mean_tokens_per_example']}/ex.; alvo com perda {c['target_tokens']:,} = {pct(c['target_tokens'] / c['tokens'])}), "
              f"{c['rows']:,} linhas de <= {d['micro_tokens']} tokens, **{steps_epoch:,} passos/epoca** (accum {accum}, ~32,8k tokens/passo) -> "
              f"{2 * steps_epoch:,} passos em 2 epocas ({2 * c['tokens']:,} tokens).",
              f"- Gold: {g['counts']['tokens']:,} tokens/epoca (a diferenca vem so do comprimento dos alvos do professor).",
              f"- Exemplos Mongo cujo rotulo do professor difere do gold: **{c['label_differs_mongo']:,} ({pct(c['label_differs_mongo'] / c['mongo_examples'])})**; "
              f"concordam com o gold (informativo): {pct(c['agrees_with_gold_mongo'] / c['mongo_examples'])}.",
              f"- Replay: {d['replay']['examples']:,} trechos, {d['replay']['tokens']:,} tokens = **{pct(d['replay']['token_share_achieved'])}** dos tokens "
              f"(alvo {pct(d['replay']['token_share_target'])}); {d['replay'].get('docs')} arquivos de {d['replay'].get('packages')} pacotes; "
              f"por tipo {d['replay'].get('by_kind')}.",
              "", "| componente | exemplos | % exemplos | tokens (distill) | alvo c/ perda (distill) | rotulo difere do gold |", "|---|---:|---:|---:|---:|---:|"]
        for name, v in sorted(c["by_component"].items()):
            L.append(f"| {name} | {v['examples']:,} | {pct(v['examples'] / c['examples'])} | {v['tokens']:,} | {v['target_tokens']:,} | {v['label_differs']:,} |")
        L += ["", f"Mistura alvo (sobre o Mongo): fim_ctx {d['mix']['target']['fim_ctx']}, completion {d['mix']['target']['completion']}, "
              f"fim_raw {d['mix']['target']['fim_raw']}; obtido: {d['mix']['achieved_of_mongo']}.", ""]
        bal = d["selection"]["balance_vs_pool"]
        acc_bal = d["selection"]["accepted_balance"]
        L += ["Balanco da selecao Mongo (estratos cut_kind x categoria x context_kind) contra o POOL de 300k (alvo; ele proprio a <= 0,3 p.p. do treino "
              "em cut_kind/categoria/dominio/papel do cursor, context_kind rebalanceado para ~29% none+irrelevant):", "",
              "| dimensao | desvio max. selecao x pool (p.p.) | desvio max. aceito bruto x pool (p.p.) |", "|---|---:|---:|"]
        for k in REPORT_DIMS:
            L.append(f"| {k} | {bal[k]['max_abs_dev_pp']} | {acc_bal[k]['max_abs_dev_pp']} |")
        L += ["", "context_kind: " + ", ".join(f"{a} {pct(b)}" for a, b in bal["context_kind"]["selected"].items()) +
              "  |  cut_kind: " + ", ".join(f"{a} {pct(b)}" for a, b in bal["cut_kind"]["selected"].items()), "",
              f"Descartes na renderizacao: {d['selection']['dropped_in_render'] or 'nenhum'}; fracao elegivel a completion "
              f"{d['selection']['completion_eligible_fraction']}.", "",
              f"### Verificacao independente ({tier.upper()})", "",
              f"- ids/componentes/ordem iguais nos dois bracos: **{ver['same_ids_same_order']}**; linhas iguais: **{ver['same_rows']}**; "
              f"problemas: **{ver['n_issues']}**; `ids_sha256` {ver['ids_sha256'][:16]}...",
              f"- Mongo: {ver['mongo_examples']:,} ids; em train.ids.txt (base): {ver['in_train_ids']:,}; em val/test/bench: "
              f"{ver['in_val_ids']}/{ver['in_test_ids']}/{ver['in_bench_ids']}.",
              f"- Replay isolamento: {d['replay']['isolation']}", ""]
    L += ["## Notas de dimensionamento", "",
          "- Volume pela receita da Fase 05 §6.2 em EXEMPLOS (T0 ~160k, T1 ~230k); em tokens ficou abaixo dos ~62 M / ~100 M estimados na spec porque o prompt "
          "`editor-context-v1` medido tem ~270 tokens (nao ~370) e o alvo ~13: ~277 tokens/ex. em media. A GPU custa por token, nao por exemplo: ver "
          "`configs/gpu_queue.yaml` (horas medidas no smoke).",
          "- Mistura: a spec previa 5-23% de chat/correcao por tier; com o chat fora de escopo (Q1) esse percentual foi para FIM/completion, e o replay sobe de "
          "3-7% para ~12% dos tokens (faixa 10-15% pedida). O replay de origem tem so ~7,2k trechos (4,45 M tokens): para chegar a 12% o mesmo trecho entra "
          "ate 3x com outro sorteio FIM/causal e corte (`max_variants`); as variantes 0 de todos os trechos vem antes de qualquer variante 1.",
          "- `fim_raw` (5%) usa o gold nos dois bracos e `completion` usa rotulo + sufixo original: o efeito do professor fica isolado em fim_ctx + completion "
          "(~90% dos exemplos Mongo). Onde o professor concorda com o gold (~44%), os dois bracos sao identicos token a token.",
          "- T0 e subconjunto de T1 (mesmo teto por estrato, mesma ordem de prioridade); o replay do T0 e prefixo do replay do T1.", "",
          "## Isolamento G2.7 (por registro, aborta em qualquer violacao)", "",
          f"Registros lidos do aceito: {load_stats.get('read')}; descartados por isolamento: {load_stats.get('isolation_dropped', 0)} (esperado 0). "
          "Checagens: `Isolation` (split, holdout, papel do schema, id/parent_id, grupo, schema, snippet, prompt_sha256, prefixo+sufixo, cego), id em "
          "`train.ids.txt` e fora de `val/test/bench.ids.txt`, hash de conteudo (prefixo+sufixo+middle/alvo) fora de `eval_docs.jsonl` para os DOIS rotulos, "
          "val do treino (copiado de `data/teacher/v2d/val`) sem ids do treino dos alunos, replay x `generic_js_fim_eval`/monitor (pacote, sha de arquivo, "
          "corte, conteudo). Testes: `scripts/student/tests/`.", "",
          f"Pool de balanco: {pool_info['total']:,} prompts. Nenhum arquivo de `data/eval/blind` e lido (so hashes do manifesto via `blind_guard`).", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "student_corpus.yaml"))
    ap.add_argument("--tiers", default="t1,t0")
    ap.add_argument("--workers", type=int, default=max(1, min(10, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--out-root", help="padrao data/student")
    ap.add_argument("--report", default=str(ROOT / "reports" / "student" / "corpus_report.md"))
    ap.add_argument("--limit", type=int, help="teste: usa so as N primeiras linhas do aceito (e 2N do pool)")
    ap.add_argument("--verify-only", action="store_true")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
