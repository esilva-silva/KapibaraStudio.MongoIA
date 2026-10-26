#!/usr/bin/env python3
"""Prototipo de cache de prefixo (KV) para o autocomplete `editor-context-v1` em CPU, com onnxruntime-genai 0.15.2 (= IDE).

Hipotese: o prompt tem um PREFIXO estavel (cabecalho LANGUAGE / AVAILABLE COMMANDS / KNOWN NAMES / RESULT FIELDS ...) e so o final varia por
pedido. A IDE (C# 0.15.2) tem `Generator.AppendTokens` + `RewindTo`; a API Python espelha (`append_tokens` / `rewind_to`). Este script mede:

  anatomy : anatomia do prompt em 2.000 prompts do val v2c (tokens por segmento, prefixo comum por grupo, fracao cacheavel, politicas A e B)
  bench   : TTFT (media/p95) e equivalencia greedy: sem cache (um Generator por pedido = IDE hoje) x Generator persistente com rewind
            (politica A: cabecalho fixo por linguagem; politica B: cabecalho + fatos do contexto por conexao/banco/colecao) x prefill em blocos
  levers  : outras alavancas de TTFT em CPU (threads, past_present_share_buffer, num_beams, blocos, int4 x int8)
  trunc   : trade-off qualidade x tokens ao truncar recent_commands / result_fields / known_names (so como DADO; contrato oficial inalterado)

Somente CPU: nenhum provider alem do CPU, `torch` nunca e importado, HIP/CUDA ocultos; `--gpu-check` confirma 0 MB/0% de GPU no processo (gpu_probe.ps1).
venv: C:/SlopStudio.MongoAI-artifacts/.venv/Scripts/python.exe -I -X utf8 scripts/export/prefix_cache_proto.py <comando> ...
Funcoes puras (prefixo comum, estatisticas, equivalencia) ficam no topo e sao testadas por scripts/export/tests/test_prefix_cache.py.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "export"))
from build_variants import WORK, package_dir  # noqa: E402  (KAPI_EXPORT_GEN=v3 => pacotes `-v3-ONNX-` e work_v3)
ROWS_PATH = ROOT / "reports" / "export" / "work" / "v2c_val2000_rows.jsonl"   # rows do val v2c: compartilhado entre geracoes
SAMPLE_IDS = ROOT / "Repos/KapiTeacher-Mongo-7B/reports/eval/base-v2c/sample_ids.json"
V2C_DIR = ROOT / "data" / "validated" / "v2c"
class _Pkg(dict):
    def __getitem__(self, key):
        return str(package_dir(*key).relative_to(ROOT))


PKG = _Pkg()   # (tier, variante) -> caminho relativo do pacote (respeita KAPI_EXPORT_GEN)
GATES = {"t0": (150.0, 250.0), "t1": (500.0, 650.0)}   # configs/gates.yaml: T0.cpu.ttft_{mean,p95}_ms ; T1.cpu_int8.ttft_{mean,p95}_ms
CTX_TOKENS = 2048
MAX_NEW = 96
SEED = 20261020


# =====================================================================================================================================
# Funcoes puras (testadas)
# =====================================================================================================================================
def common_prefix_len(a, b) -> int:
    """Comprimento do prefixo comum de duas sequencias (listas/tuplas de ids)."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def group_lcp(seqs) -> int:
    """Prefixo comum mais longo de TODAS as sequencias do grupo (0 se vazio)."""
    seqs = list(seqs)
    if not seqs:
        return 0
    first = seqs[0]
    n = len(first)
    for s in seqs[1:]:
        n = min(n, common_prefix_len(first, s))
        if n == 0:
            break
    return n


def nn_lcp(seqs) -> list[int]:
    """Para cada sequencia, o maior prefixo comum com QUALQUER outra do grupo (vizinho mais proximo). Ordena e compara adjacentes: O(n log n).
    Grupo de 1 elemento -> 0. Mantem a ordem de entrada."""
    seqs = [tuple(s) for s in seqs]
    n = len(seqs)
    if n < 2:
        return [0] * n
    order = sorted(range(n), key=lambda i: seqs[i])
    out = [0] * n
    adj = [common_prefix_len(seqs[order[k]], seqs[order[k + 1]]) for k in range(n - 1)]
    for k, i in enumerate(order):
        left = adj[k - 1] if k > 0 else 0
        right = adj[k] if k < n - 1 else 0
        out[i] = max(left, right)
    return out


def quantile(values, q: float) -> float:
    """Quantil por posicao (mesma convencao dos outros scripts: sorted[round(q*(n-1))])."""
    v = sorted(values)
    if not v:
        return float("nan")
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def weighted_quantile(samples: list[tuple[float, float]], q: float) -> float:
    """samples = [(valor, peso)]; quantil q da distribuicao ponderada (para misturar acertos e erros de cache)."""
    s = sorted(samples)
    total = sum(w for _, w in s)
    if total <= 0:
        return float("nan")
    acc = 0.0
    for v, w in s:
        acc += w
        if acc / total >= q - 1e-12:
            return v
    return s[-1][0]


def ttft_stats(values_ms) -> dict:
    v = list(values_ms)
    return {"n": len(v), "mean": round(st.mean(v), 1), "median": round(st.median(v), 1), "p95": round(quantile(v, 0.95), 1)} if v else {"n": 0}


def mixture_stats(hit_ms, miss_ms, hit_rate: float) -> dict:
    """Media e p95 de uma mistura: com probabilidade `hit_rate` o pedido acerta o cache (amostras `hit_ms`), senao usa `miss_ms`."""
    hit_ms, miss_ms = list(hit_ms), list(miss_ms)
    samples = [(x, hit_rate / len(hit_ms)) for x in hit_ms] + [(x, (1 - hit_rate) / len(miss_ms)) for x in miss_ms] if hit_rate < 1 else \
        [(x, 1.0 / len(hit_ms)) for x in hit_ms]
    mean = sum(v * w for v, w in samples) / sum(w for _, w in samples)
    return {"mean": round(mean, 1), "p95": round(weighted_quantile(samples, 0.95), 1)}


def greedy_equivalence(base: list[list[int]], other: list[list[int]]) -> dict:
    """Compara saidas greedy token a token. identical = listas iguais; first_diff = indice do 1o token diferente (ou None)."""
    assert len(base) == len(other)
    same, diffs = 0, []
    for k, (a, b) in enumerate(zip(base, other)):
        if a == b:
            same += 1
        else:
            diffs.append({"index": k, "first_diff": common_prefix_len(a, b), "len_base": len(a), "len_other": len(b)})
    n = len(base)
    return {"n": n, "identical": same, "identical_pct": round(100.0 * same / n, 2) if n else None, "diffs": diffs}


def cache_cut(prefix_cut: int, n_prompt: int, available: int | None = None) -> int:
    """Quantos tokens do prompt podem vir do KV: no maximo `prefix_cut`, e sempre deixando >= 1 token novo (o logit da ultima posicao so existe
    depois de processar ao menos 1 token). `available` = tamanho valido do KV (None = sem limite)."""
    n = min(prefix_cut, n_prompt - 1)
    if available is not None:
        n = min(n, available)
    return max(0, n)


# =====================================================================================================================================
# Dados / prompts
# =====================================================================================================================================
def load_rows() -> list[dict]:
    if not ROWS_PATH.exists():
        ids = set(json.loads(SAMPLE_IDS.read_text(encoding="utf-8"))["ids"])
        out = []
        for shard in sorted(V2C_DIR.glob("fim-*.jsonl")):
            with shard.open(encoding="utf-8") as h:
                for line in h:
                    if '"split":"val"' in line:
                        r = json.loads(line)
                        if r.get("id") in ids:
                            out.append(r)
        ROWS_PATH.parent.mkdir(parents=True, exist_ok=True)
        ROWS_PATH.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out), encoding="utf-8")
    return [json.loads(x) for x in ROWS_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]


def _rc():
    sys.path.insert(0, str(ROOT / "scripts" / "data"))
    import render_contract as rc
    return rc


def _tokenizer():
    return _rc().Tokenizer()


def header_bounds(before: str, rc=None) -> dict | None:
    """Posicoes (em caracteres de `before`, o `ModelPrefix`) dos limites de segmento do contexto; None se nao houver cabecalho."""
    rc = rc or _rc()
    head = rc.WRAP_HEAD
    if not before.startswith(head):
        return None
    c0 = len(head)
    kn = before.find("KNOWN NAMES:", c0)
    tail = before.find(rc.WRAP_TAIL, c0)
    if kn < 0 or tail < 0:
        return None
    lang = before.find("LANGUAGE:", c0)
    eol = before.find("\r\n", kn)
    names_end = (eol + 2) if 0 <= eol < tail else tail      # fim da linha KNOWN NAMES (CRLF incluido)
    rf = before.find("RESULT FIELDS:", names_end)
    rf = rf if 0 <= rf < tail else -1
    nxt = [p for p in (before.find("INPUT PANEL:", names_end), before.find("RECENT COMMAND:", names_end)) if 0 <= p < tail]
    fields_end = (min(nxt) if nxt else tail) if rf >= 0 else names_end
    return {"global": lang + len("LANGUAGE:"), "lang": kn + len("KNOWN NAMES:"), "names": names_end, "fields": fields_end, "ctx": tail,
            "tail": tail + len(rc.WRAP_TAIL)}


def build_prompt(tok, rc, row: dict) -> dict:
    """ids do prompt (identicos ao `eval_teacher.build_ids` / IDE) + indices de corte em TOKENS do prompt final para cada limite de segmento.
    Os cortes so contam tokens que terminam ANTES do limite (token que atravessa o limite fica de fora). cuts = None se o inicio do prefixo
    foi truncado pelo orcamento (o cabecalho cai primeiro) ou se o registro nao tem cabecalho."""
    prefix, suffix = rc.bound_window(row["prefix"], row["suffix"])
    lang = row.get("language") or rc.editor_language(row)
    before = rc.render_model_prefix(prefix, row.get("editor"), lang)
    enc = tok._tok.encode(before, add_special_tokens=False) if before else None
    b = enc.ids if enc else []
    a = tok.encode(suffix)
    ids, info = rc.build_ids(tok, before, suffix, CTX_TOKENS, enc=(b, a))
    bounds = header_bounds(before, rc)
    cuts = None
    dropped = len(b) - info["kept_prefix"]
    if bounds and dropped == 0:
        ends = [e for _, e in enc.offsets]
        cuts = {k: 1 + sum(1 for e in ends if e <= pos) for k, pos in bounds.items()}   # 1 = <|fim_prefix|>
        cuts["code"] = cuts["tail"]
    return {"ids": ids, "cuts": cuts, "info": info, "lang": lang, "before_tokens": len(b), "suffix_tokens": info["kept_suffix"],
            "n": len(ids), "has_header": bounds is not None, "text_before": before}


def row_key(row: dict) -> tuple:
    t = (row.get("editor") or {}).get("target") or {}
    return (t.get("connection"), t.get("database"), t.get("collection"))


# =====================================================================================================================================
# anatomy
# =====================================================================================================================================
def dist(values) -> dict:
    v = list(values)
    if not v:
        return {"n": 0}
    return {"n": len(v), "mean": round(st.mean(v), 1), "p50": quantile(v, 0.5), "p90": quantile(v, 0.9), "min": min(v), "max": max(v)}


def group_report(prompts: list[dict], keyfn, label: str) -> dict:
    """Prefixo comum por grupo: (a) LCP de todo o grupo, (b) LCP com o melhor vizinho (por prompt), em tokens e em fracao do prompt."""
    groups: dict = {}
    for p in prompts:
        groups.setdefault(keyfn(p), []).append(p)
    nn_tok, nn_frac, glcp_tok, glcp_frac, with_peer = [], [], [], [], 0
    for members in groups.values():
        ids = [m["ids"] for m in members]
        nn = nn_lcp(ids)
        g = group_lcp(ids) if len(ids) > 1 else 0
        for m, v in zip(members, nn):
            nn_tok.append(v)
            nn_frac.append(v / m["n"])
            glcp_tok.append(g)
            glcp_frac.append(g / m["n"])
            with_peer += len(members) > 1
    return {"label": label, "groups": len(groups), "groups_ge2": sum(len(m) > 1 for m in groups.values()),
            "prompts_with_peer_pct": round(100 * with_peer / max(1, len(prompts)), 1),
            "nn_lcp_tokens": dist(nn_tok), "nn_lcp_frac": {**dist(nn_frac)},
            "group_lcp_tokens": dist(glcp_tok), "group_lcp_frac": dist(glcp_frac)}


def anatomy(n_rows: int | None = None) -> dict:
    rc = _rc()
    tok = _tokenizer()
    rows = load_rows()[:n_rows]
    prompts = []
    for r in rows:
        p = build_prompt(tok, rc, r)
        p.update(row=r, key=row_key(r), kind=r.get("context_kind"), schema=r.get("schema_id"), coll=(row_key(r))[2], conn=(row_key(r))[0],
                 db=(row_key(r))[1], snippet=r.get("snippet_id"))
        prompts.append(p)
    # confere com os ids do proxy (t0_prompts.json = eval_teacher.build_ids)
    ref = {it["id"]: it["prompt_ids"] for it in json.loads((WORK / "t0_prompts.json").read_text(encoding="utf-8"))["items"]}
    checked = [(p["row"]["id"], p["ids"] == ref[p["row"]["id"]]) for p in prompts if p["row"]["id"] in ref]
    out: dict = {"n_rows": len(rows), "ids_identical_to_eval_build_ids": {"checked": len(checked), "identical": sum(ok for _, ok in checked)}}
    out["context_kind"] = {k: sum(p["kind"] == k for p in prompts) for k in sorted({p["kind"] for p in prompts})}
    ide = [p for p in prompts if p["has_header"]]          # a IDE SEMPRE emite o cabecalho; context_kind=none (treino) nao ocorre na IDE
    usable = [p for p in ide if p["cuts"]]
    out["ide_prompts"] = len(ide)
    out["header_truncated_by_budget"] = len(ide) - len(usable)
    out["total_tokens"] = {"all": dist(p["n"] for p in prompts), "ide": dist(p["n"] for p in ide)}
    out["by_language"] = {}
    for lang in sorted({p["lang"] for p in usable}):
        out["by_language"][lang] = dist(p["n"] for p in usable if p["lang"] == lang)
    seg = {}
    for name, f in (("fixed_global", lambda p: p["cuts"]["global"]),
                    ("fixed_lang(A)", lambda p: p["cuts"]["lang"]),
                    ("names", lambda p: p["cuts"]["names"] - p["cuts"]["lang"]),
                    ("result_fields", lambda p: p["cuts"]["fields"] - p["cuts"]["names"]),
                    ("input+recent", lambda p: p["cuts"]["ctx"] - p["cuts"]["fields"]),
                    ("tail_wrapper", lambda p: p["cuts"]["tail"] - p["cuts"]["ctx"]),
                    ("code_prefix", lambda p: p["before_tokens"] + 1 - p["cuts"]["tail"]),
                    ("suffix+markers", lambda p: p["n"] - 1 - p["before_tokens"])):
        seg[name] = dist(f(p) for p in usable)
    out["segment_tokens"] = seg
    fr = {}
    for name, key in (("A0_global_fixed", "global"), ("A_fixed_header_per_lang", "lang"), ("A+names", "names"), ("B_fields(header+names+fields)", "fields"),
                      ("B_ctx(header..recent_commands)", "ctx"), ("B_full(ctx+tail wrapper)", "tail")):
        fr[name] = {"tokens": dist(p["cuts"][key] for p in usable), "fraction_of_prompt": dist(p["cuts"][key] / p["n"] for p in usable)}
    out["cacheable_structural"] = fr
    # prefixo comum empirico (ids reais) por grupo
    hdr = [p for p in usable]
    out["empirical_groups"] = [
        group_report(hdr, lambda p: "all", "todos os prompts (cache global)"),
        group_report(hdr, lambda p: p["lang"], "mesma linguagem"),
        group_report(hdr, lambda p: (p["lang"], p["kind"]), "mesma linguagem + context_kind"),
        group_report(hdr, lambda p: p["conn"], "mesma conexao"),
        group_report(hdr, lambda p: (p["conn"], p["db"]), "mesma conexao + banco"),
        group_report(hdr, lambda p: p["schema"], "mesmo schema_id"),
        group_report(hdr, lambda p: p["coll"], "mesma colecao (nome)"),
        group_report(hdr, lambda p: p["key"], "mesma (conexao, banco, colecao) = politica B"),
        group_report(hdr, lambda p: p["snippet"], "mesmo snippet (sessao de digitacao: mesmo documento, outro cursor)"),
    ]
    # validade do corte estrutural: dentro da mesma chave, os ids ate o corte B batem?
    by_key: dict = {}
    for p in usable:
        by_key.setdefault(p["key"], []).append(p)
    ok = tot = 0
    for members in by_key.values():
        for m in members[1:]:
            f = members[0]
            if f["cuts"]["fields"] == m["cuts"]["fields"] and f["ids"][:f["cuts"]["fields"]] == m["ids"][:m["cuts"]["fields"]]:
                ok += 1
            tot += 1
    out["same_key_pairs_equal_up_to_fields"] = {"pairs": tot, "equal": ok}
    # token que atravessa o limite A: ids[:lang] iguais em TODOS os prompts da mesma linguagem?
    lcp_lang = {}
    for lang in sorted({p["lang"] for p in usable}):
        ms = [p for p in usable if p["lang"] == lang]
        cut = min(p["cuts"]["lang"] for p in ms)
        lcp_lang[lang] = {"n": len(ms), "structural_cut": cut, "group_lcp": group_lcp(p["ids"] for p in ms),
                          "all_equal_to_cut": all(p["ids"][:cut] == ms[0]["ids"][:cut] for p in ms)}
    out["policy_A_by_language"] = lcp_lang
    # custo de prefill estimado ~ proporcional a tokens: fracao economizada por politica (hit steady-state)
    out["policy_summary"] = {
        "A_per_language": {"cacheable_tokens": dist(p["cuts"]["lang"] for p in usable), "fraction": dist(p["cuts"]["lang"] / p["n"] for p in usable)},
        "A_plus_names": {"cacheable_tokens": dist(p["cuts"]["names"] for p in usable), "fraction": dist(p["cuts"]["names"] / p["n"] for p in usable)},
        "B_ctx_through_fields": {"cacheable_tokens": dist(p["cuts"]["fields"] for p in usable), "fraction": dist(p["cuts"]["fields"] / p["n"] for p in usable)},
        "B_full_header": {"cacheable_tokens": dist(p["cuts"]["tail"] for p in usable), "fraction": dist(p["cuts"]["tail"] / p["n"] for p in usable)}}
    return out


# =====================================================================================================================================
# Runtime (onnxruntime-genai) — somente CPU
# =====================================================================================================================================
def _cpu_only_env() -> None:
    for k in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        os.environ[k] = "-1" if k.startswith("HIP") else ""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def open_model(pkg: str, threads: int, overlay: dict | None = None):
    import onnxruntime_genai as og
    cfg = og.Config(pkg)
    cfg.clear_providers()                     # CPU (como a IDE: ClearProviders + AppendProvider so para GPU/NPU)
    ov: dict = {}
    if threads:
        ov = {"model": {"decoder": {"session_options": {"intra_op_num_threads": threads}}}}
    if overlay:
        ov = _deep_merge(ov, overlay)
    if ov:
        cfg.overlay(json.dumps(ov))
    return og, og.Model(cfg)


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class Session:
    """Generator persistente. `kv_ids` = ids cujo KV esta valido (prompt + o que foi processado). Politica de cache = `cap` por pedido."""

    def __init__(self, og, model, search: dict | None = None):
        self.og, self.model, self.search = og, model, search or {}
        p = og.GeneratorParams(model)
        p.set_search_options(do_sample=False, max_length=CTX_TOKENS, batch_size=1, **self.search)
        self.g = og.Generator(model, p)
        self.kv: list[int] = []
        self.cold_prefill_ms = 0.0

    def valid_len(self) -> int:
        return len(self.kv)

    def rewind(self, n: int) -> None:
        self.g.rewind_to(n)
        del self.kv[n:]

    def prime(self, ids: list[int]) -> None:
        """Garante que o KV contenha exatamente `ids`, reaproveitando o prefixo comum com o que ja esta la. NAO e cronometrado: e o 'pedido anterior'."""
        import numpy as np
        n = common_prefix_len(self.kv, ids)
        self.g.rewind_to(n)
        rest = ids[n:]
        if rest:
            self.g.append_tokens(np.array(rest, dtype=np.int32))      # em 0.15.2 append_tokens ja executa o prefill
        self.kv = list(ids)


def _stop_loop(g, first: int, limit: int, suffix: str, stops: set, dec, force: bool = False):
    out = [first]
    reason = "eos" if (not force and first in stops) else "max"
    while reason == "max" and len(out) < limit and not g.is_done():
        if not force and len(suffix) >= 2 and suffix in dec(out):
            reason = "suffix"
            break
        g.generate_next_token()
        out.append(int(g.get_next_tokens()[0]))
        if not force and out[-1] in stops:
            reason = "eos"
    return out, reason


def req_baseline(og, model, ids, limit, suffix, stops, dec, search=None):
    """O que a IDE faz hoje: um Generator novo por pedido; TTFT = append_tokens + 1o generate_next_token (como onnx_run.py)."""
    import numpy as np
    p = og.GeneratorParams(model)
    p.set_search_options(do_sample=False, max_length=len(ids) + limit, batch_size=1, **(search or {}))
    g = og.Generator(model, p)
    arr = np.array(ids, dtype=np.int32)
    t = time.perf_counter()
    g.append_tokens(arr)
    g.generate_next_token()
    ttft = time.perf_counter() - t
    first = int(g.get_next_tokens()[0])
    out, reason = _stop_loop(g, first, limit, suffix, stops, dec)
    return {"tokens": out, "stop": reason, "ttft_ms": ttft * 1000}


def req_chunked(og, model, ids, limit, suffix, stops, dec, chunk, search=None):
    """Prefill em blocos de `chunk` tokens com um unico Generator: append_tokens(bloco) para todos, so o ULTIMO bloco e seguido de
    generate_next_token (em 0.15.2 append_tokens ja processa o bloco quando a sequencia ja tem tokens; ver relatorio)."""
    import numpy as np
    p = og.GeneratorParams(model)
    p.set_search_options(do_sample=False, max_length=len(ids) + limit, batch_size=1, **(search or {}))
    g = og.Generator(model, p)
    arr = np.array(ids, dtype=np.int32)
    t = time.perf_counter()
    for s in range(0, len(ids), chunk):
        g.append_tokens(arr[s:s + chunk])
    g.generate_next_token()
    ttft = time.perf_counter() - t
    first = int(g.get_next_tokens()[0])
    out, reason = _stop_loop(g, first, limit, suffix, stops, dec)
    return {"tokens": out, "stop": reason, "ttft_ms": ttft * 1000}


def req_cached(sess: Session, ids, cut, limit, suffix, stops, dec):
    """Pedido com Generator persistente: rewind_to(n) + append_tokens(ids[n:]) onde n = min(cut, prefixo comum com o KV valido, len-1).
    TTFT inclui o rewind. Se o KV nao cobre `cut` (miss), n menor -> prefill maior (no limite, n=0 = baseline sem a criacao do Generator)."""
    import numpy as np
    g = sess.g
    n = cache_cut(cut, len(ids), common_prefix_len(sess.kv, ids))
    t = time.perf_counter()
    g.rewind_to(n)
    g.append_tokens(np.array(ids[n:], dtype=np.int32))
    g.generate_next_token()
    ttft = time.perf_counter() - t
    first = int(g.get_next_tokens()[0])
    out, reason = _stop_loop(g, first, limit, suffix, stops, dec)
    sess.kv = list(ids)            # KV valido: o prompt inteiro (os tokens gerados nao entram na comparacao)
    return {"tokens": out, "stop": reason, "ttft_ms": ttft * 1000, "reused": n}


# =====================================================================================================================================
# bench
# =====================================================================================================================================
def load_items(tier: str):
    P = json.loads((WORK / f"{tier}_prompts.json").read_text(encoding="utf-8"))
    return P


def prepare(tier: str):
    """Itens do proxy (mesmos 500 ids/prompts do export) + cortes de cache de cada politica."""
    rc = _rc()
    tok = _tokenizer()
    P = load_items(tier)
    rows = {r["id"]: r for r in load_rows()}
    items = []
    for it in P["items"]:
        bp = build_prompt(tok, rc, rows[it["id"]])
        assert bp["ids"] == it["prompt_ids"], "ids divergem do eval_teacher.build_ids"
        c = bp["cuts"] or {}
        row = rows[it["id"]]
        nxt = None
        if row.get("middle"):   # proxima 'tecla': o usuario digita o 1o caractere do trecho esperado (prefixo + 1 char; sufixo igual)
            nb = build_prompt(tok, rc, {**row, "prefix": row["prefix"] + row["middle"][:1]})
            nxt = nb["ids"] if nb["cuts"] else None
        items.append({"id": it["id"], "ids": it["prompt_ids"], "suffix": it["suffix"], "lang": bp["lang"], "key": row_key(row),
                      "cuts": c, "n": bp["n"], "has_cache": bool(c), "next_ids": nxt})
    return P, items


def policy_cut(item: dict, policy: str) -> int:
    c = item["cuts"]
    if not c:
        return 0
    return {"A": c["lang"], "A+names": c["names"], "B": c["tail"], "Bfields": c["fields"]}[policy]


def run_bench(a) -> dict:
    _cpu_only_env()
    import psutil
    from tokenizers import Tokenizer as HfTok
    tier, variant = a.tier, a.variant
    P, items = prepare(tier)
    tkz = HfTok.from_file(str(_rc().DEFAULT_TOKENIZER))
    dec = lambda ids: tkz.decode(list(ids), skip_special_tokens=False)   # noqa: E731  (so para a parada por sufixo)
    stops = set(P["stops"])
    pkg = str(ROOT / PKG[(tier, variant)])
    overlay = json.loads(a.overlay) if a.overlay else None
    search = json.loads(a.search) if a.search else None
    og, model = open_model(pkg, a.threads, overlay)
    res: dict = {"tier": tier, "variant": variant, "threads": a.threads, "genai": og.__version__, "overlay": overlay, "search": search,
                 "load_pct_before": psutil.cpu_percent(interval=2.0), "pid": os.getpid()}
    usable = [it for it in items if it["has_cache"]]
    warm, n = a.warm, a.n
    sel = usable[: n + warm]
    modes = a.modes.split(",")
    rng = random.Random(SEED)
    pool: dict = {}                 # sessoes por linguagem (politica A)
    s_b, s_bf, s_c = Session(og, model, search), Session(og, model, search), Session(og, model, search)
    LIM = 16

    def step(mode, it):
        """Um pedido no modo `mode`. TTFT cronometrado dentro de req_*; o 'pedido anterior' (prime) fica fora do cronometro."""
        if mode == "base":
            return req_baseline(og, model, it["ids"], LIM, "", stops, dec, search)
        if mode.startswith("chunk"):
            return req_chunked(og, model, it["ids"], LIM, "", stops, dec, int(mode[5:]), search)
        if mode == "A":
            s = pool.get(it["lang"]) or pool.setdefault(it["lang"], Session(og, model, search))
            cut = policy_cut(it, "A")
            if common_prefix_len(s.kv, it["ids"]) < cut:      # miss (1o pedido da linguagem): prefill do cabecalho fora do cronometro
                s.prime(it["ids"][:cut])
            return req_cached(s, it["ids"], cut, LIM, "", stops, dec)
        if mode in ("B", "Bfields"):
            s = s_b if mode == "B" else s_bf                  # acerto estavel: o Generator ja tem o cabecalho DESTE pedido (pedido anterior da aba)
            cut = policy_cut(it, mode)
            s.prime(it["ids"][:cut])
            return req_cached(s, it["ids"], cut, LIM, "", stops, dec)
        if mode == "C":                                       # digitacao: KV do pedido anterior (prompt do registro) reaproveitado pelo LCP inteiro
            s_c.prime(it["ids"])
            r = req_cached(s_c, it["next_ids"], len(it["next_ids"]), LIM, "", stops, dec)
            r["n"] = len(it["next_ids"])
            return r
        if mode == "baseC":
            r = req_baseline(og, model, it["next_ids"], LIM, "", stops, dec, search)
            r["n"] = len(it["next_ids"])
            return r
        raise SystemExit(f"modo desconhecido {mode}")

    data: dict = {m: [] for m in modes}
    t_all = time.perf_counter()
    for k, it in enumerate(sel):
        order = modes[:]
        rng.shuffle(order)       # ordem aleatoria por item: nenhum modo herda sistematicamente o cache/frequencia do anterior
        for m in order:
            if m in ("C", "baseC") and not it["next_ids"]:
                continue
            r = step(m, it)
            if k >= warm:
                data[m].append({"id": it["id"], "n": r.get("n", it["n"]), "ttft_ms": r["ttft_ms"], "reused": r.get("reused", 0)})
    out_modes = {}
    for m in modes:
        rows_ = data[m]
        out_modes[m] = {"stats": ttft_stats([r["ttft_ms"] for r in rows_]), "rows": rows_,
                        "mean_prompt_tokens": round(st.mean(r["n"] for r in rows_), 1), "mean_reused_tokens": round(st.mean(r["reused"] for r in rows_), 1)}
        print(f"{tier}/{variant} thr={a.threads} {m}: {out_modes[m]['stats']} prompt~{out_modes[m]['mean_prompt_tokens']} reused~{out_modes[m]['mean_reused_tokens']}", flush=True)
    res["wall_s"] = round(time.perf_counter() - t_all, 1)
    res["modes"] = out_modes
    mi = psutil.Process().memory_info()
    res["peak_wset_gb"] = round(getattr(mi, "peak_wset", mi.rss) / 2**30, 3)
    res["load_pct_after"] = psutil.cpu_percent(interval=1.0)
    if a.gpu_check:
        res["gpu_probe"] = gpu_probe()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res), encoding="utf-8")
    return res


def gpu_probe() -> dict:
    import subprocess
    r = subprocess.run(["powershell", "-NoProfile", "-File", str(ROOT / "scripts/export/gpu_probe.ps1"), "-ProcId", str(os.getpid()), "-Samples", "1"],
                       capture_output=True, text=True, timeout=120)
    try:
        d = json.loads(r.stdout.strip().splitlines()[-1])
        return {"proc_dedicated_mb": d.get("proc_dedicated_mb"), "proc_shared_mb": d.get("proc_shared_mb"), "proc_engine_util_pct": d.get("proc_engine_util_pct")}
    except Exception:  # noqa: BLE001
        return {"error": (r.stdout + r.stderr)[-300:]}


# =====================================================================================================================================
# equivalencia (n=500, greedy ate 96 tokens, mesmas paradas do onnx_run quality)
# =====================================================================================================================================
def run_equiv(a) -> dict:
    _cpu_only_env()
    import numpy as np
    from tokenizers import Tokenizer as HfTok
    tier, variant = a.tier, a.variant
    P, items = prepare(tier)
    tkz = HfTok.from_file(str(_rc().DEFAULT_TOKENIZER))
    dec = lambda ids: tkz.decode(list(ids), skip_special_tokens=False)   # noqa: E731
    stops = set(P["stops"])
    og, model = open_model(str(ROOT / PKG[(tier, variant)]), a.threads)
    sel = items[: a.n]
    res: dict = {"tier": tier, "variant": variant, "threads": a.threads, "n": len(sel), "genai": og.__version__}

    def run_all(name, fn):
        t = time.perf_counter()
        outs = []
        for it in sel:
            r = fn(it)
            outs.append({"id": it["id"], "tokens": r["tokens"], "stop": r["stop"], "ttft_ms": round(r["ttft_ms"], 1), "reused": r.get("reused", 0)})
        print(name, f"{time.perf_counter() - t:.0f}s", flush=True)
        return outs

    runs = {}
    runs["base"] = run_all("base", lambda it: req_baseline(og, model, it["ids"], MAX_NEW, it["suffix"], stops, dec))
    nrep = min(len(sel), a.repeat_n)
    sel_full, sel = sel, sel[:nrep]
    runs["base_repeat"] = run_all("base_repeat", lambda it: req_baseline(og, model, it["ids"], MAX_NEW, it["suffix"], stops, dec))
    sel = sel_full
    s_b = Session(og, model)
    runs["B"] = run_all("B", lambda it: (s_b.prime(it["ids"][:policy_cut(it, "B")]),
                                          req_cached(s_b, it["ids"], policy_cut(it, "B"), MAX_NEW, it["suffix"], stops, dec))[1])
    pool: dict = {}

    def fa(it):
        s = pool.setdefault(it["lang"], Session(og, model))
        cut = policy_cut(it, "A")
        if common_prefix_len(s.kv, it["ids"]) < cut:
            s.prime(it["ids"][:cut])
        return req_cached(s, it["ids"], cut, MAX_NEW, it["suffix"], stops, dec)
    runs["A"] = run_all("A", fa)
    # chunked (iii): prefill em blocos de 128 (append_tokens sucessivos)
    sel_full = sel
    sel = sel[:a.chunk_n]
    runs["chunk128"] = run_all("chunk128", lambda it: req_chunked(og, model, it["ids"], MAX_NEW, it["suffix"], stops, dec, 128))
    sel = sel_full
    # C: sessao de digitacao (pedido anterior = prompt do registro; novo pedido = +1 caractere do trecho esperado) x baseline do novo prompt
    selc = [it for it in sel if it["next_ids"]]
    sel = selc
    runs["base_next"] = run_all("base_next", lambda it: req_baseline(og, model, it["next_ids"], MAX_NEW, it["suffix"], stops, dec))
    s_c = Session(og, model)
    runs["C"] = run_all("C", lambda it: (s_c.prime(it["ids"]), req_cached(s_c, it["next_ids"], len(it["next_ids"]), MAX_NEW, it["suffix"], stops, dec))[1])
    sel = sel_full
    toks = {k: [r["tokens"] for r in v] for k, v in runs.items()}
    eq = {}
    for k in toks:
        if k == "base":
            continue
        ref = "base_next" if k == "C" else "base"
        if k == "base_next":
            continue
        n_ = len(toks[k])
        eq[k] = greedy_equivalence(toks[ref][:n_], toks[k]) if k != "C" else greedy_equivalence(toks["base_next"], toks["C"])
        eq[k]["ids"] = [r["id"] for r in runs[k]]
    sel_by_run = {k: [r["id"] for r in runs[k]] for k in runs}
    # diagnostico das divergencias: margem top1-top2 do baseline no 1o token divergente
    by_id = {it["id"]: it for it in items}
    for k, e in eq.items():
        for d in e["diffs"][:40]:
            it = by_id[e["ids"][d["index"]]]
            ids_ = it["next_ids"] if k == "C" else it["ids"]
            ref_toks = toks["base_next"][d["index"]] if k == "C" else toks["base"][d["index"]]
            d["id"] = it["id"]
            d["margin"] = divergence_margin(og, model, ids_, ref_toks, d["first_diff"])
        e.pop("ids")
    res["equivalence"] = eq
    res["runs"] = runs
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res), encoding="utf-8")
    print(json.dumps({k: {"identical_pct": v["identical_pct"], "diffs": len(v["diffs"])} for k, v in eq.items()}))
    return res


def divergence_margin(og, model, ids, base_tokens, k):
    """top1-top2 dos logits do baseline na posicao k (1o token em que cache e baseline divergiram)."""
    import numpy as np
    try:
        p = og.GeneratorParams(model)
        p.set_search_options(do_sample=False, max_length=len(ids) + k + 2, batch_size=1)
        g = og.Generator(model, p)
        g.append_tokens(np.array(list(ids) + list(base_tokens[:k]), dtype=np.int32))
        g.generate_next_token()
        lg = np.asarray(g.get_logits()).reshape(-1)
        top = np.partition(lg, -2)[-2:]
        return round(float(top[1] - top[0]), 4)
    except Exception as e:  # noqa: BLE001
        return f"erro: {e}"


# =====================================================================================================================================
# trunc: qualidade x tokens de prompt (so dado; contrato oficial inalterado)
# =====================================================================================================================================
TRUNC_VARIANTS = {
    "full": {},
    "recent0": {"recent_commands": 0},
    "recent1": {"recent_commands": 1},
    "fields32": {"result_fields": 32},
    "fields16": {"result_fields": 16},
    "fields0": {"result_fields": 0},
    "names32": {"known_names": 32},
    "names16": {"known_names": 16},
    "lean": {"recent_commands": 0, "result_fields": 16, "known_names": 16},
    "lean0": {"recent_commands": 0, "result_fields": 0, "known_names": 16},
}


def trunc_row(row: dict, spec: dict) -> dict:
    r = dict(row)
    ed = dict(r.get("editor") or {})
    if not ed:
        return r
    for field, key in (("recent_commands", "recent_commands"), ("result_fields", "result_fields"), ("known_names", "known_names")):
        if field in spec:
            ed[key] = list(ed.get(key, []))[: spec[field]]
    r["editor"] = ed
    return r


def run_trunc(a) -> dict:
    """Gera (greedy, <= 96 tokens) com os prompts truncados e grava o JSON no formato de `onnx_run.py --mode quality` para o score_onnx.py."""
    _cpu_only_env()
    from tokenizers import Tokenizer as HfTok
    rc = _rc()
    tok = _tokenizer()
    tier, variant = a.tier, a.variant
    P = load_items(tier)
    rows = {r["id"]: r for r in load_rows()}
    tkz = HfTok.from_file(str(rc.DEFAULT_TOKENIZER))
    dec = lambda ids: tkz.decode(list(ids), skip_special_tokens=False)   # noqa: E731
    stops = set(P["stops"])
    og, model = open_model(str(ROOT / PKG[(tier, variant)]), a.threads)
    summary = {}
    for name in a.variants.split(","):
        spec = TRUNC_VARIANTS[name]
        out, lens, ttft = [], [], []
        t = time.perf_counter()
        for it in P["items"][: a.n]:
            row = trunc_row(rows[it["id"]], spec)
            bp = build_prompt(tok, rc, row)
            r = req_baseline(og, model, bp["ids"], MAX_NEW, it["suffix"], stops, dec)
            out.append({"id": it["id"], "tokens": r["tokens"], "stop": r["stop"], "ttft_ms": round(r["ttft_ms"], 1)})
            lens.append(bp["n"])
            ttft.append(r["ttft_ms"])
        dest = WORK / f"{tier}_{variant}-trunc-{name}_quality.json"
        dest.write_text(json.dumps({"pkg": str(ROOT / PKG[(tier, variant)]), "provider": "cpu", "mode": "quality", "genai": og.__version__, "items": out,
                                    "seconds": round(time.perf_counter() - t, 1), "variant_spec": spec}), encoding="utf-8")
        summary[name] = {"prompt_tokens": dist(lens), "ttft": ttft_stats(ttft), "run": str(dest)}
        print(name, summary[name]["prompt_tokens"], summary[name]["ttft"], flush=True)
    Path(a.out).write_text(json.dumps(summary), encoding="utf-8")
    return summary


# =====================================================================================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("anatomy")
    s.add_argument("--out", default=str(WORK / "prefix_anatomy.json"))
    for name in ("bench", "equiv", "trunc"):
        s = sub.add_parser(name)
        s.add_argument("--tier", choices=["t0", "t1"], required=True)
        s.add_argument("--variant", default=None)
        s.add_argument("--threads", type=int, default=0, help="intra_op_num_threads (0 = padrao da IDE)")
        s.add_argument("--n", type=int, default=200 if name == "bench" else 500)
        s.add_argument("--out", required=True)
        if name == "equiv":
            s.add_argument("--repeat-n", dest="repeat_n", type=int, default=100)
            s.add_argument("--chunk-n", dest="chunk_n", type=int, default=200)
        if name == "bench":
            s.add_argument("--warm", type=int, default=5)
            s.add_argument("--modes", default="base,A,B,C,baseC")
            s.add_argument("--overlay", default=None, help="JSON de overlay do genai_config (ex.: threads/ORT)")
            s.add_argument("--search", default=None, help="JSON de search options (ex.: {\"past_present_share_buffer\": false})")
            s.add_argument("--gpu-check", action="store_true")
        if name == "trunc":
            s.add_argument("--variants", default=",".join(TRUNC_VARIANTS))
    a = ap.parse_args()
    if a.cmd == "anatomy":
        r = anatomy()
        Path(a.out).write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(r["policy_summary"], indent=1))
        return
    a.variant = a.variant or ("int4" if a.tier == "t0" else "int8")
    {"bench": run_bench, "equiv": run_equiv, "trunc": run_trunc}[a.cmd](a)


if __name__ == "__main__":
    main()
