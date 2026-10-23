#!/usr/bin/env python3
"""Cortes FIM sobre os snippets (spec fase-03 §4.3 `gen_fim.py`, tabela de pontos de corte).

  python scripts/data/gen_fim.py --seed 20261008 --count 6000 --ver v0 [--validation validation.jsonl] [--emit-qwen]

Sem parser JS: os cortes usam os *spans* sintaticos emitidos pelo renderizador do gerador
(chave, valor, operador, referencia "$campo", estagio, membro, metodo, statement), equivalentes
aos tokens de AST da spec. Saida: data/generated/<ver>/fim.jsonl (+ fim.manifest.json).

Registro: id, kind, prefix, suffix, middle, domain, schema_id, collection, cut_kind, fields_used,
group, split_key (+ group_id, split, template_id, snippet_id, cursor_role, multiline, mid_token,
category, split_role, middle_tokens, middle_tokens_method, exec...).
Formato Qwen (--emit-qwen): <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|> -> middle.

Ciclo 2:
  * `fim.max_tokens` e medido com o tokenizer Qwen (mesma contagem da etapa 7 do validate_pipeline); sem o
    pacote/arquivo cai em chars/3 e o registro marca `middle_tokens_method: approx`.
  * Amostragem em duas dimensoes: categoria (`fim.category_targets`, piso `fim.category_floor`) x cut_kind
    (`fim.cut_targets` corrigido por `fim.decontam_survival`: o alvo vale POS-decontaminacao).
  * Snippets com `exec_mode` skipped_static/skipped_write (atlas_search, indexes) entram sem execucao.

Ciclo 3 (volume v2):
  * Streaming e shards: `--shard-size N` grava `fim-00000.jsonl`, ... (ver jsonl_shards.py); o manifesto lista sha256 por shard.
  * `fim.eval_counts` (val/test/bench) x `fim.eval_margin`: contagens FIXAS de avaliacao (spec §4.6: 10.000/5.000/3.000);
    train = `count` - avaliacao. Cada split e amostrado separadamente (so snippets do split) com as mesmas marginais.
    Sem `eval_counts`: amostragem unica (v1).
  * `fim.category_survival` (alem de `decontam_survival`): sobrevivencia por categoria na decontaminacao -> reamostragem.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_snippets import CONFIG_PATH, ROOT, load_config  # noqa: E402
import blind_guard  # noqa: E402
import hom_guard  # noqa: E402
from jsonl_shards import HashReservoir, ShardWriter, iter_jsonl  # noqa: E402

ROLE_TO_CURSOR = {
    "key": "PropertyKey", "value": "PropertyValue", "op": "OperatorName", "fieldref": "FieldReferenceString",
    "stage": "ArrayElement", "member": "MemberAccess", "method": "MemberAccess", "strarg": "StringArgument",
    "chain": "RestOfStatement", "obj": "ObjectBody", "arr": "ObjectBody", "stmt": "StatementStart", "ident": "PropertyValue",
}
PARTIAL_ROLES = {"key", "op", "method", "member", "fieldref", "strarg"}
QUOTES = "\"'"
DEFAULT_TOK_SITE = "C:\\SlopStudio.MongoAI-artifacts\\.venv\\Lib\\site-packages"   # venv ROCm do bake-off (tem `tokenizers`)
STATIC_EXEC = ("skipped_static", "skipped_write")
_TOK_CACHE: dict = {}


class MiddleTokens:
    """Conta tokens do `middle` com o tokenizer Qwen2.5-Coder (mesma contagem da etapa 7 do validate_pipeline:
    encode sem tokens especiais). Sem o pacote `tokenizers`/arquivo => fallback chars/cpt marcado `method="approx"`."""

    def __init__(self, path: Path | None, cpt: float = 3.0, site: str | None = None, force_approx: bool = False):
        self.cpt = float(cpt)
        self.method = "approx"
        self.path = None
        self._tok = None
        if force_approx or path is None or not Path(path).exists():
            return
        tk = None
        for attempt in range(2):
            try:
                from tokenizers import Tokenizer as tk  # noqa: N813
                break
            except ImportError:
                site_dir = os.environ.get("KAPI_TOKENIZERS_SITE") or site or DEFAULT_TOK_SITE
                if attempt == 0 and site_dir and Path(site_dir).exists() and site_dir not in sys.path:
                    sys.path.append(site_dir)
                    continue
                break
        if tk is None:
            return
        try:
            self._tok = tk.from_file(str(path))
            self.method, self.path = "qwen", str(path)
        except Exception:
            self._tok = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._tok is not None:
            return len(self._tok.encode(text, add_special_tokens=False).ids)
        return math.ceil(len(text) / self.cpt)


def middle_tokens_for(cfg: dict, force_approx: bool = False) -> MiddleTokens:
    fc = cfg.get("fim", {})
    rel = fc.get("tokenizer") or (cfg.get("editor_context") or {}).get("tokenizer") or "BaseRepos/Qwen2.5-Coder-0.5B/tokenizer.json"
    force = bool(force_approx or fc.get("tokenizer_mode") == "approx")
    cpt = fc.get("chars_per_token", 3.0)
    key = (str(rel), force, cpt, fc.get("tokenizer_site") or "")
    if key not in _TOK_CACHE:
        _TOK_CACHE[key] = MiddleTokens(ROOT / rel, cpt, fc.get("tokenizer_site") or None, force)
    return _TOK_CACHE[key]


def snippet_category(r: dict) -> str:
    """`category` gravada pelo gen_snippets; fallback por `op` para snippets antigos (v0/v1)."""
    c = r.get("category")
    if c:
        return c
    return "aggregation" if r.get("op") == "aggregate" else "crud"


def qwen_prompt(prefix: str, suffix: str) -> str:
    return f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"


def line_end(text: str, pos: int) -> int:
    i = text.find("\n", pos)
    if i < 0:
        return len(text)
    return i - 1 if i > 0 and text[i - 1] == "\r" else i


def stmt_bounds(spans, pos):
    best = None
    for s, e, role, _u in spans:
        if role == "stmt" and s <= pos < e:
            if best is None or (e - s) < (best[1] - best[0]):
                best = (s, e)
    return best or (0, 10 ** 9)


def candidates(rec: dict) -> dict[str, list[dict]]:
    """Mapeia cut_kind -> candidatos {start, tok_end, unit_end, role, mid_token}."""
    text, spans = rec["text"], rec["spans"]
    out: dict[str, list[dict]] = {k: [] for k in
                                  ("key", "value", "operator", "stage", "member", "partial", "rest", "multiline", "stmt_start")}
    for s, e, role, u in spans:
        if e <= s:
            continue
        tok_end = e + 1 if e < len(text) and text[e] in QUOTES and role in ("key", "op", "fieldref", "member", "strarg") else e
        base = {"start": s, "tok_end": tok_end, "unit_end": max(u, tok_end), "role": role, "mid_token": False}
        if role == "key":
            out["key"].append(base)
            t = text[s:e]
            for i, ch in enumerate(t):
                if ch == "." and i + 1 < len(t):
                    out["key"].append({**base, "start": s + i + 1, "mid_token": True})
        elif role == "value" or role == "ident":
            out["value"].append(base)
        elif role in ("op", "fieldref"):
            out["operator"].append(base)
        elif role == "stage":
            out["stage"].append(base)
        elif role in ("member", "method", "strarg"):
            out["member"].append(base)
        elif role == "chain":
            out["rest"].append({**base, "tok_end": e, "start": s})
        elif role in ("obj", "arr") and "\n" in text[s:e]:
            out["multiline"].append({**base, "tok_end": e})
        elif role == "stmt" and s > 0:
            out["stmt_start"].append({**base, "tok_end": e})
        if role in PARTIAL_ROLES and e - s >= 3:
            for k in range(1, e - s):
                out["partial"].append({**base, "start": s + k, "mid_token": True})
    # rest-of-line apos virgulas de propriedade
    for i, ch in enumerate(text):
        if ch == "," and i + 1 < len(text) and text[i + 1] == " ":
            out["rest"].append({"start": i + 2, "tok_end": line_end(text, i + 2), "unit_end": line_end(text, i + 2),
                                "role": "chain", "mid_token": False})
    return out


def pick_end(rec, kind, c, cfg, rng, tc: MiddleTokens):
    """Escolhe o fim do middle em fronteira natural; `middle` <= fim.max_tokens tokens (tokenizer Qwen, ou approx)."""
    max_tokens = int(cfg["fim"]["max_tokens"])
    hard_chars = max_tokens * 12   # pre-filtro barato (digitos/pontuacao ~1 char/token)
    text = rec["text"]
    s = c["start"]
    sb = stmt_bounds(rec["spans"], s)
    stmt_end = min(sb[1], len(text))
    opts: list[tuple[int, float]] = []
    if kind in ("key", "value", "operator", "member"):
        opts = [(c["tok_end"], 0.5), (c["unit_end"], 0.25), (line_end(text, s), 0.15), (stmt_end, 0.10)]
    elif kind == "partial":
        opts = [(c["tok_end"], 0.45), (c["unit_end"], 0.2), (line_end(text, s), 0.2), (stmt_end, 0.15)]
    elif kind == "stage":
        e = c["tok_end"]
        comma = e + 1 if e < len(text) and text[e] == "," else e
        opts = [(e, 0.4), (comma, 0.3), (line_end(text, e), 0.1), (stmt_end, 0.2)]
        # "resto do pipeline": do estagio ate o fim do array
    elif kind == "multiline":
        opts = [(c["tok_end"], 1.0)]
    elif kind == "rest":
        opts = [(c["tok_end"], 0.35), (line_end(text, s), 0.25), (stmt_end, 0.4)]
    elif kind == "stmt_start":
        chains = [(a, b) for a, b, r, _ in rec["spans"] if r == "chain" and sb[0] <= a < sb[1]]
        head = min(chains)[1] if chains else None
        opts = []
        if head and head < stmt_end:
            opts.append((head, 0.5))
        le = line_end(text, s)
        if le < stmt_end:
            opts.append((le, 0.3))
        opts.append((stmt_end, 0.2))
    cands = [(e, w) for e, w in opts if e > s]
    rng.shuffle(cands)
    # tenta por peso decrescente sorteado
    pool = list(cands)
    while pool:
        total = sum(w for _, w in pool)
        r = rng.random() * total
        acc = 0.0
        for idx, (e, w) in enumerate(pool):
            acc += w
            if r <= acc:
                break
        e, _ = pool.pop(idx)
        if (e - s <= hard_chars and text[:s].strip() and text[e:].strip() and text[s:e].strip()
                and tc.count(text[s:e]) <= max_tokens):
            return e
    return None


def make_record(rec, kind, c, e, cfg, n, tc: MiddleTokens | None = None):
    text = rec["text"]
    s = c["start"]
    prefix, middle, suffix = text[:s], text[s:e], text[e:]
    cpt = cfg["fim"].get("chars_per_token", 3.0)
    cut_id = f"{kind}"
    suffix_id = rec["id"].split("snip-", 1)[-1]
    return {
        "id": f"fim-{suffix_id}-{cut_id}-{n:06d}",
        "type": "fim",
        "kind": rec["op"],
        "task": "autocomplete",
        "dialect": rec.get("dialect", "console"),
        "prefix": prefix,
        "suffix": suffix,
        "middle": middle,
        "domain": rec["domain"],
        "schema_id": rec["schema_id"],
        "collection": rec["collection"],
        "cut_kind": kind,
        "cursor_role": ROLE_TO_CURSOR.get(c["role"], "Unknown"),
        "fields_used": rec["fields_used"],
        "group": rec["group_id"],
        "group_id": rec["group_id"],
        "template_id": rec["template_id"],
        "split_key": rec["split_key"],
        "split": rec["split"],
        "holdout": rec["holdout"],
        "snippet_id": rec["id"],
        "multiline": "\n" in middle,
        "mid_token": bool(c["mid_token"]),
        "crlf": "\r\n" in text,
        "approx_middle_tokens": math.ceil(len(middle) / cpt),
        "middle_tokens": tc.count(middle) if tc else math.ceil(len(middle) / cpt),
        "middle_tokens_method": tc.method if tc else "approx",
        "category": snippet_category(rec),
        "split_role": rec.get("split_role"),
        "read_only": rec["kind"] == "read",
    }


def _normalize(d: dict) -> dict:
    t = sum(d.values()) or 1.0
    return {k: v / t for k, v in d.items()}


def category_weights(present: list[str], fc: dict) -> dict[str, float]:
    """Alvo por categoria presente, com piso `category_floor` para as pequenas (spec §4.3, "Como o volume cai")."""
    tg = fc.get("category_targets") or {}
    floor = float(fc.get("category_floor", 0.04))
    t = {c: float(tg.get(c, 0.0)) for c in present}
    if not any(t.values()):
        t = {c: 1.0 for c in present}
    t = _normalize(t)
    small = [c for c in t if t[c] < floor]
    if small and len(small) < len(t):
        rest = 1.0 - floor * len(small)
        big = sum(v for c, v in t.items() if c not in small) or 1.0
        t = {c: (floor if c in small else v / big * rest) for c, v in t.items()}
    surv = fc.get("category_survival") or {}
    if fc.get("resample_for_decontam", False) and surv:
        # o alvo (com piso) vale POS-decontaminacao: sobe o peso bruto das categorias que perdem mais
        t = _normalize({c: v / max(0.05, float(surv.get(c, 1.0))) for c, v in t.items()})
    return t


def kind_weights(present: list[str], fc: dict) -> dict[str, float]:
    """Pesos de amostragem por cut_kind: alvo pos-decontaminacao / taxa de sobrevivencia (reamostragem)."""
    tg = {k: float(fc["cut_targets"][k]) for k in present}
    if fc.get("resample_for_decontam", False):
        surv = fc.get("decontam_survival") or {}
        tg = {k: v / max(0.05, float(surv.get(k, 1.0))) for k, v in tg.items()}
    return _normalize(tg)


def expected_post_decontam(raw_counts: dict, fc: dict) -> dict:
    surv = fc.get("decontam_survival") or {}
    kept = {k: v * float(surv.get(k, 1.0)) for k, v in raw_counts.items()}
    return {k: round(v, 4) for k, v in sorted(_normalize(kept).items())} if sum(kept.values()) else {}


def joint_weights(kinds: list[str], cats: list[str], support: set, kw: dict, cw: dict, iters: int = 300) -> dict:
    """Distribuicao conjunta (cut_kind x categoria) com as marginais `kw` e `cw` (IPF sobre o suporte disponivel:
    nem toda categoria tem todos os cut_kinds, p.ex. `indexes` nao tem `stage`)."""
    p = {(k, c): (1.0 if (k, c) in support else 0.0) for k in kinds for c in cats}
    for _ in range(iters):
        for k in kinds:
            s = sum(p[k, c] for c in cats)
            if s > 0:
                for c in cats:
                    p[k, c] *= kw[k] / s
        for c in cats:
            s = sum(p[k, c] for k in kinds)
            if s > 0:
                for k in kinds:
                    p[k, c] *= cw[c] / s
    return p


def snippet_exec_status(r: dict, validation: dict | None) -> str | None:
    """Status de execucao do snippet: exec_mode estatico do gerador tem precedencia; senao o validation.jsonl."""
    em = r.get("exec_mode")
    if em in STATIC_EXEC:
        return em
    if validation is None:
        return None
    return validation.get(r["id"], {}).get("exec")


def eval_counts_for(cfg: dict, count: int) -> dict | None:
    """Contagens FIXAS de val/test/bench (spec §4.6: 10.000/5.000/3.000 + `fim.eval_margin` para as perdas do pipeline);
    train recebe o resto. None => amostragem unica sobre todos os splits (comportamento v1)."""
    fc = cfg.get("fim", {})
    ec = fc.get("eval_counts")
    if not ec:
        return None
    margin = float(fc.get("eval_margin", 1.0))
    out = {s: int(math.ceil(int(ec.get(s, 0)) * margin)) for s in ("val", "test", "bench")}
    cap = int(count * float(fc.get("eval_max_fraction", 0.2)))   # volumes pequenos (testes/pilotos): avaliacao <= 20% do total
    tot = sum(out.values())
    if tot > cap:
        out = {s: int(v * cap / tot) for s, v in out.items()}
    out["train"] = max(0, int(count) - sum(out.values()))
    return out


def _slim(r: dict) -> dict:
    """So o que o amostrador precisa (economiza memoria com ~1 M de snippets)."""
    return {k: r[k] for k in ("id", "text", "spans", "op", "domain", "schema_id", "collection", "group_id", "template_id", "split_key",
                              "split", "holdout", "dialect", "kind", "fields_used", "category", "split_role", "exec_mode")
            if k in r}


def waterfill(total: int, weights: dict, capacity: dict) -> dict:
    """Cotas inteiras por par: proporcionais a `weights`, limitadas por `capacity`; o excedente e redistribuido (agua-enchendo).
    Soma <= total (menor se a capacidade total nao basta)."""
    quota = {k: 0 for k in weights}
    active = {k for k, w in weights.items() if w > 0 and capacity.get(k, 0) > 0}
    left = int(total)
    while left > 0 and active:
        sw = sum(weights[k] for k in active)
        prop = {k: left * weights[k] / sw for k in active}
        over = sorted(k for k in active if quota[k] + prop[k] >= capacity[k])
        if over:
            for k in over:
                left -= capacity[k] - quota[k]
                quota[k] = capacity[k]
                active.discard(k)
            continue
        fl = {k: int(prop[k]) for k in active}
        rem = left - sum(fl.values())
        for k in sorted(active):
            quota[k] += fl[k]
        for k in sorted(active, key=lambda k: (-(prop[k] - fl[k]), k))[:rem]:
            quota[k] += 1
        left = 0
    return quota


class _PoolSampler:
    """Amostra cortes (kind x categoria) de um conjunto de snippets: marginais `fim.cut_targets` x `fim.category_targets`.

    Cotas por par (kind, categoria) por agua-enchendo: se a oferta de um par nao basta (p.ex. poucos snippets de uma categoria
    sobrevivem a decontaminacao), ele fica no maximo da oferta e o excedente vai para os demais (o volume cai menos que a
    distribuicao desvia). `max_cuts` limita cortes por snippet (a oferta e o numero de snippets x teto)."""

    def __init__(self, snips: list[dict], cfg: dict, tc: MiddleTokens, max_cuts: int = 0):
        self.snips, self.cfg, self.tc = snips, cfg, tc
        fc = cfg["fim"]
        self.max_cuts = int(max_cuts or 0)
        self.cat_of = [snippet_category(r) for r in snips]
        by_kc: dict[tuple[str, str], list[int]] = {}
        cap: dict[tuple[str, str], int] = {}
        for i, r in enumerate(snips):
            for k, lst in candidates(r).items():       # so a presenca: os candidatos sao recalculados no sorteio
                if lst and k in fc["cut_targets"]:
                    pr = (k, self.cat_of[i])
                    by_kc.setdefault(pr, []).append(i)
                    cap[pr] = cap.get(pr, 0) + (min(len(lst), self.max_cuts) if self.max_cuts else len(lst) * 4)
        self.by_kc = by_kc
        self.cap = cap
        self.kinds = [k for k in fc["cut_targets"] if any(kk == k for kk, _ in by_kc)]
        self.cats = sorted({c for _, c in by_kc})
        self.pairs = sorted(by_kc)
        if self.kinds and self.cats:
            kw = kind_weights(self.kinds, fc)
            cw = category_weights(self.cats, fc)
            jw = joint_weights(self.kinds, self.cats, set(self.pairs), kw, cw)
            self.pw = {pr: jw[pr] for pr in self.pairs}
        else:
            self.pw = {}
        self.shortfall: dict = {}

    def capacity_total(self, count: int) -> int:
        return sum(waterfill(count, self.pw, self.cap).values()) if self.pw else 0

    def run(self, count: int, rng: random.Random, validation, emit_qwen: bool, seq_start: int = 0):
        cfg, tc, snips = self.cfg, self.tc, self.snips
        if not self.pw or count <= 0:
            return
        pairs = self.pairs
        seen: set = set()
        per_snip: Counter = Counter()
        used: Counter = Counter()
        exhausted: set = set()
        n_out = 0
        self.shortfall = {}
        while n_out < count:
            cap_left = {pr: self.cap[pr] - used[pr] for pr in pairs if pr not in exhausted and self.cap[pr] - used[pr] > 0}
            quota = waterfill(count - n_out, {pr: self.pw[pr] for pr in cap_left}, cap_left)
            left = {pr: q for pr, q in quota.items() if q > 0}
            if not left:
                break
            dead = 0
            while left:
                live = sorted(left)
                pair = rng.choices(live, weights=[left[pr] for pr in live])[0]
                kind, cat = pair
                pool = self.by_kc[pair]
                got = None
                for _ in range(40):  # varias tentativas DENTRO do mesmo (kind, categoria) para nao distorcer a distribuicao
                    si = rng.choice(pool)
                    if self.max_cuts and per_snip[si] >= self.max_cuts:
                        continue
                    rec = snips[si]
                    c = rng.choice(candidates(rec)[kind])
                    e = pick_end(rec, kind, c, cfg, rng, tc)
                    if e is None or (rec["id"], c["start"], e) in seen:
                        continue
                    got = (si, rec, c, e)
                    break
                if got is None:
                    dead += 1
                    if dead > 3:                  # par esgotado (oferta menor que a estimada): o excedente e redistribuido
                        exhausted.add(pair)
                        self.shortfall[pair] = self.shortfall.get(pair, 0) + left.pop(pair)
                        dead = 0
                    continue
                dead = 0
                si, rec, c, e = got
                r = make_record(rec, kind, c, e, cfg, seq_start + n_out, tc)
                try:  # guarda do conjunto cego: so a API do guard (hash de registro e de prefix+NUL+suffix)
                    blind_guard.check_record(r)
                    blind_guard.check_text(blind_guard.prompt_text({"kind": "autocomplete", "prefix": r["prefix"],
                                                                    "suffix": r["suffix"]}))
                except blind_guard.BlindAccessError:
                    continue
                seen.add((rec["id"], c["start"], e))
                per_snip[si] += 1
                used[pair] += 1
                st = snippet_exec_status(rec, validation)
                if validation is not None or st in STATIC_EXEC:
                    r["exec"] = st or "unvalidated"
                if emit_qwen:
                    r["qwen_prompt"] = qwen_prompt(r["prefix"], r["suffix"])
                    r["qwen_target"] = r["middle"]
                left[pair] -= 1
                if left[pair] <= 0:
                    del left[pair]
                n_out += 1
                yield r
        # o que ficou sem oferta (nao redistribuido) vira `shortfall` por par


# --------------------------------------------------------------------------- pre-filtro de decontaminacao (a regra LITERAL do pipeline)
class EvalIndex:
    """Indice de decontaminacao (o MESMO `decontaminate`, regra literal: exact, 13-grama, approx_key, MinHash >= 0,7) construido
    com os documentos de avaliacao escolhidos (+ legacy-v1). `hit(text, collection)` diz se um snippet de treino seria removido
    na etapa 9 do pipeline; assim o gerador so emite cortes de documentos que sobrevivem (nao ha desperdicio a jusante)."""

    def __init__(self, eval_docs: list[tuple[str, dict]], seed: int, min_content: int | None = None):
        import decontaminate as dc
        self.dc = dc
        # MESMA regra oficial do pipeline: 13-grama comum so conta com >= min_content tokens de conteudo (padrao 2)
        self.min_content = dc.DEFAULT_NGRAM_MIN_CONTENT if min_content is None else int(min_content)
        self.mh = dc.MinHasher(dc.NUM_PERM, seed)
        self.idx = dc.build_index(eval_docs, self.mh, dc.DECONTAM_JACCARD)
        self.n = self.idx["n"]
        self.by_source = dict(self.idx["by_source"])

    def hit(self, text: str, collection: str | None = None) -> bool:
        dc, idx = self.dc, self.idx
        toks = dc.code_tokens(text)
        grams = dc.ngram_map(toks)
        mc = self.min_content
        for g, n_content in grams.items():
            if n_content >= mc and g in idx["grams"]:
                return True
        rec_toks = toks
        if idx["exact"] and hashlib.sha256(" ".join(rec_toks).encode("utf-8")).hexdigest() in idx["exact"]:
            return True
        ak = dc.approx_key(text, collection if isinstance(collection, str) else None)
        if ak and ak in idx["approx"]:
            return True
        sig = self.mh.signature(rec_toks)
        if sig is not None and idx["lsh"].query(sig, limit=1):
            return True
        return False


_HW: dict = {}


def _hit_init(eval_docs, seed, min_content):
    _HW["idx"] = EvalIndex(eval_docs, seed, min_content)


def _hit_run(chunk):
    idx = _HW["idx"]
    return [idx.hit(t, c) for t, c in chunk]


def fim_eval_doc(r: dict) -> dict:
    """Snippet -> registro no formato que a decontaminacao indexa (documento = prefixo+meio+sufixo = texto do snippet)."""
    return {"id": r["id"], "snippet_id": r["id"], "type": "fim", "prefix": r["text"], "middle": "", "suffix": "",
            "collection": r.get("collection")}


def legacy_docs(cfg: dict) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for rel in (cfg.get("fim", {}).get("decontam_legacy") or ["bakeoff/data/v1_benchmark.jsonl", "bakeoff/data/v1_validation.jsonl"]):
        p = ROOT / rel
        if p.exists():
            out += [("legacy-v1", r) for r in iter_jsonl(p)]
    return out


def _valid_snip(r: dict, validation: dict | None) -> bool:
    if validation is not None:
        st = snippet_exec_status(r, validation)
        if not (st == "ok" or (st or "").startswith("skipped")):
            return False
    return "<|" not in r["text"] and "```" not in r["text"]      # nunca sai texto com tokens especiais


def iter_fim(snips, cfg: dict, count: int, validation: dict | None = None, emit_qwen=False,
             tokenizer: MiddleTokens | None = None, report: dict | None = None, workers: int = 1):
    """Gera cortes FIM em streaming. `snips`: lista de snippets OU funcao sem argumentos que devolve um iterador novo
    (necessario para o pre-filtro de decontaminacao, que percorre os snippets duas vezes).

    Sem `fim.eval_counts`: um unico amostrador sobre todos os splits (v1). Com `fim.eval_counts`: cada split e amostrado
    separadamente (val/test/bench com contagens fixas = tamanhos da spec §4.6; train = restante), com as MESMAS marginais de
    categoria e cut_kind; so snippets do respectivo split entram no respectivo pool.

    `fim.decontam_prefilter: true` (so com eval_counts): o treino so usa snippets que SOBREVIVEM a decontaminacao literal contra
    os documentos de avaliacao escolhidos (+ legacy-v1). `fim.max_cuts_per_snippet` limita cortes por snippet."""
    seed = cfg.get("seed", 20261008)
    fc = cfg.get("fim", {})
    tc = tokenizer or middle_tokens_for(cfg)
    open_snips = snips if callable(snips) else (lambda: iter(snips))
    ec = eval_counts_for(cfg, count)
    max_cuts = int(fc.get("max_cuts_per_snippet", 0) or 0)
    prefilter = bool(fc.get("decontam_prefilter", False)) and ec is not None
    seq = 0
    rep = report if report is not None else {}

    if ec is None:
        keep = [_slim(r) for r in open_snips() if _valid_snip(r, validation)]
        sampler = _PoolSampler(keep, cfg, tc, max_cuts)
        rng = random.Random(f"{seed}:fim")
        got = 0
        for r in sampler.run(count, rng, validation, emit_qwen, 0):
            got += 1
            yield r
        rep["all"] = {"requested": count, "got": got, "snippets": len(keep)}
        return

    # ---- avaliacao primeiro (val, test, bench): pools por split
    # pools de avaliacao com teto `fim.eval_pool_max` por split (amostra uniforme por hash: memoria constante com ~3 M de snippets)
    pool_max = int(fc.get("eval_pool_max", 80000) or 0)
    res = {s: HashReservoir(pool_max, f"{seed}:evalpool:{s}") for s in ("val", "test", "bench")}
    for r in open_snips():
        if r["split"] in res and _valid_snip(r, validation):
            res[r["split"]].add(r["id"], _slim(r))
    evals: dict[str, list[dict]] = {s: v.items() for s, v in res.items()}
    del res
    eval_docs: list[tuple[str, dict]] = []
    for name in ("val", "test", "bench"):
        pool = evals[name]
        n_target = ec[name]
        if not pool:
            rep[name] = {"requested": n_target, "got": 0, "snippets": 0}
            continue
        sampler = _PoolSampler(pool, cfg, tc, max_cuts)
        rng = random.Random(f"{seed}:fim:{name}")
        used: set = set()
        got = 0
        for r in sampler.run(n_target, rng, validation, emit_qwen, seq):
            got += 1
            used.add(r["snippet_id"])
            yield r
        seq += got
        rep[name] = {"requested": n_target, "got": got, "snippets": len(pool), "snippets_used": len(used),
                     "shortfall": {f"{k}:{c}": v for (k, c), v in sampler.shortfall.items()}}
        eval_docs += [(name, fim_eval_doc(s)) for s in pool if s["id"] in used]
        del sampler
    evals.clear()

    # ---- treino (opcionalmente so snippets que sobrevivem a decontaminacao literal)
    stats = {"train_snippets_in": 0, "train_snippets_dropped": 0}
    mc = (cfg.get("decontam") or {}).get("ngram_min_content")
    all_docs = eval_docs + legacy_docs(cfg) if prefilter else []
    del eval_docs
    if prefilter:
        import par
        index = EvalIndex(all_docs, seed, mc)               # no processo principal (contagens para o relatorio)
        rep["decontam_prefilter"] = {"eval_docs": index.n, "by_source": index.by_source, "min_content": index.min_content}
        del index

        def train_stream():
            for r in open_snips():
                if r["split"] == "train" and _valid_snip(r, validation):
                    yield r

        # `hit` por snippet e independente: blocos em processos (cada worker com o seu indice); ordem preservada
        train = []
        pending: list = []

        def flush(chunk_recs, flags):
            for rr, h in zip(chunk_recs, flags):
                stats["train_snippets_in"] += 1
                if h:
                    stats["train_snippets_dropped"] += 1
                else:
                    train.append(_slim(rr))

        def blocks():
            for ch in par.chunked(train_stream(), 2000):
                pending.append(ch)
                yield [(x["text"], x.get("collection")) for x in ch]

        for flags in par.ordered_map(_hit_run, blocks(), workers, _hit_init, (all_docs, seed, mc), window=max(2, workers * 2)):
            flush(pending.pop(0), flags)
    else:
        train = []
        for r in open_snips():
            if r["split"] != "train" or not _valid_snip(r, validation):
                continue
            stats["train_snippets_in"] += 1
            train.append(_slim(r))
    del all_docs
    if prefilter:
        by_cat: Counter = Counter(snippet_category(r) for r in train)
        rep["decontam_prefilter"].update(stats)
        rep["decontam_prefilter"]["survivors_by_category"] = dict(by_cat)
        rep["_survivor_ids"] = [r["id"] for r in train]      # ids de snippets de treino que sobrevivem (usado pelo gen_chat)
    n_train = ec["train"]
    sampler = _PoolSampler(train, cfg, tc, max_cuts)
    rng = random.Random(f"{seed}:fim:train")
    got = 0
    for r in sampler.run(n_train, rng, validation, emit_qwen, seq):
        got += 1
        yield r
    rep["train"] = {"requested": n_train, "got": got, "snippets": len(train),
                    "shortfall": {f"{k}:{c}": v for (k, c), v in sampler.shortfall.items()}}


def generate_fim(snips: list[dict], cfg: dict, count: int, validation: dict | None = None, emit_qwen=False,
                 tokenizer: MiddleTokens | None = None) -> list[dict]:
    return list(iter_fim(snips, cfg, count, validation, emit_qwen, tokenizer))


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--count", type=int)
    ap.add_argument("--ver")
    ap.add_argument("--snippets", help="snippets.jsonl (default: data/generated/<ver>/snippets.jsonl)")
    ap.add_argument("--validation", help="validation.jsonl: descarta snippets com exec=error")
    ap.add_argument("--include-hom", action="store_true",
                    help="SO para scripts de homologacao (outro diretorio, split `homolog`); nunca em jobs de treino")
    ap.add_argument("--emit-qwen", action="store_true")
    ap.add_argument("--approx-tokens", action="store_true", help="forca o fallback chars/3 (sem tokenizer Qwen)")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, help="processos do pre-filtro (padrao: nucleos - 2, max 10); saida identica para qualquer N")
    ap.add_argument("--shard-size", type=int, default=0, help="linhas por shard (fim-00000.jsonl, ...); 0 = arquivo unico")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.seed is not None:
        cfg["seed"] = a.seed
    ver = a.ver or cfg.get("version", "v0")
    gdir = ROOT / cfg.get("out_dir", "data/generated") / ver
    snips_path = Path(a.snippets) if a.snippets else gdir / "snippets.jsonl"
    val = None
    if a.validation:
        # so o status de execucao (`exec`) importa aqui; evita guardar o registro inteiro de ~1 M de linhas
        val = {}
        shared: dict = {}
        for r in iter_jsonl(Path(a.validation)):
            ex = r.get("exec")
            val[r["id"]] = shared.setdefault(ex, {"exec": ex})
    count = a.count or cfg["fim"]["count"]
    tc = middle_tokens_for(cfg, a.approx_tokens)
    out = Path(a.out) if a.out else gdir / "fim.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.stem
    w = ShardWriter(out.parent, stem, a.shard_size, out.suffix)
    dist: Counter = Counter()
    cdist: Counter = Counter()
    splits: Counter = Counter()
    joint: Counter = Counter()
    max_mid = 0
    n = 0
    report: dict = {}
    import par
    workers = a.workers if a.workers is not None else par.default_workers()
    for r in iter_fim(lambda: hom_guard.drop_hom(iter_jsonl(snips_path), a.include_hom), cfg, count, val, a.emit_qwen, tc, report, workers):
        w.write(r)
        n += 1
        dist[r["cut_kind"]] += 1
        cdist[r["category"]] += 1
        splits[r["split"]] += 1
        joint[(r["split"], r["category"], r["cut_kind"])] += 1
        max_mid = max(max_mid, r["middle_tokens"])
    wr = w.close()
    surv_ids = report.pop("_survivor_ids", None)
    if surv_ids is not None:
        (out.parent / f"{stem}.survivors.txt").write_text("".join(i + chr(10) for i in sorted(surv_ids)), encoding="utf-8", newline=chr(10))
    nn = max(1, n)
    man = {"seed": cfg["seed"], "count": n, "requested": count, "sha256": wr["stream_sha256"], "shards": wr["files"],
           "cut_distribution": {k: round(v / nn, 4) for k, v in sorted(dist.items())},
           "cut_distribution_expected_post_decontam": expected_post_decontam(dict(dist), cfg["fim"]),
           "category_distribution": {k: round(v / nn, 4) for k, v in sorted(cdist.items())},
           "target": cfg["fim"]["cut_targets"],
           "category_target": cfg["fim"].get("category_targets"),
           "tokenizer": {"method": tc.method, "path": tc.path, "max_tokens": cfg["fim"]["max_tokens"], "max_middle_tokens": max_mid},
           "splits": {s: splits.get(s, 0) for s in ("train", "val", "test", "bench")},
           "per_split_sampling": report,
           "train_cut_distribution": {k: round(c / max(1, splits.get("train", 0)), 4) for k, c in sorted(
               Counter({ck: sum(v for (sp, _c, ck2), v in joint.items() if sp == "train" and ck2 == ck)
                        for ck in dist}).items())}}
    (out.parent / f"{stem}.manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in man.items() if k != "shards"}))


if __name__ == "__main__":
    main()
