#!/usr/bin/env python3
"""Cortes FIM sobre os snippets (spec fase-03 §4.3 `gen_fim.py`, tabela de pontos de corte).

  python scripts/data/gen_fim.py --seed 20261008 --count 6000 --ver v0 [--validation validation.jsonl] [--emit-qwen]

Sem parser JS: os cortes usam os *spans* sintaticos emitidos pelo renderizador do gerador
(chave, valor, operador, referencia "$campo", estagio, membro, metodo, statement), equivalentes
aos tokens de AST da spec. Saida: data/generated/<ver>/fim.jsonl (+ fim.manifest.json).

Registro: id, kind, prefix, suffix, middle, domain, schema_id, collection, cut_kind, fields_used,
group, split_key (+ group_id, split, template_id, snippet_id, cursor_role, multiline, mid_token...).
Formato Qwen (--emit-qwen): <|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|> -> middle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_snippets import CONFIG_PATH, ROOT, load_config  # noqa: E402
import blind_guard  # noqa: E402

ROLE_TO_CURSOR = {
    "key": "PropertyKey", "value": "PropertyValue", "op": "OperatorName", "fieldref": "FieldReferenceString",
    "stage": "ArrayElement", "member": "MemberAccess", "method": "MemberAccess", "strarg": "StringArgument",
    "chain": "RestOfStatement", "obj": "ObjectBody", "arr": "ObjectBody", "stmt": "StatementStart", "ident": "PropertyValue",
}
PARTIAL_ROLES = {"key", "op", "method", "member", "fieldref", "strarg"}
QUOTES = "\"'"


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


def pick_end(rec, kind, c, cfg, rng, max_chars):
    """Escolhe o fim do middle em fronteira natural, respeitando o limite de tokens."""
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
        if e - s <= max_chars and text[:s].strip() and text[e:].strip() and text[s:e].strip():
            return e
    return None


def make_record(rec, kind, c, e, cfg, n):
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
        "read_only": rec["kind"] == "read",
    }


def generate_fim(snips: list[dict], cfg: dict, count: int, validation: dict | None = None, emit_qwen=False) -> list[dict]:
    seed = cfg.get("seed", 20261008)
    fc = cfg["fim"]
    max_chars = int(fc["max_tokens"] * fc.get("chars_per_token", 3.0))
    if validation is not None:
        snips = [r for r in snips if validation.get(r["id"], {}).get("exec") in ("ok", "skipped")]
    # nunca sai texto com tokens especiais
    snips = [r for r in snips if "<|" not in r["text"] and "```" not in r["text"]]
    cands = [candidates(r) for r in snips]
    by_kind: dict[str, list[int]] = {k: [] for k in fc["cut_targets"]}
    for i, c in enumerate(cands):
        for k, lst in c.items():
            if lst and k in by_kind:
                by_kind[k].append(i)
    kinds = [k for k in fc["cut_targets"] if by_kind[k]]
    weights = [fc["cut_targets"][k] for k in kinds]
    rng = random.Random(f"{seed}:fim")
    seen: set = set()
    out: list[dict] = []
    fails = 0
    while len(out) < count and fails < count * 20:
        kind = rng.choices(kinds, weights=weights)[0]
        # tenta varias vezes DENTRO do mesmo cut_kind para nao distorcer a distribuicao alvo
        got = None
        for _ in range(40):
            si = rng.choice(by_kind[kind])
            rec = snips[si]
            c = rng.choice(cands[si][kind])
            e = pick_end(rec, kind, c, cfg, rng, max_chars)
            if e is None or (rec["id"], c["start"], e) in seen:
                fails += 1
                continue
            got = (rec, c, e)
            break
        if got is None:
            continue
        rec, c, e = got
        key = (rec["id"], c["start"], e)
        seen.add(key)
        r = make_record(rec, kind, c, e, cfg, len(out))
        try:  # guarda do conjunto cego: so a API do guard (hash de registro e de prefix+NUL+suffix)
            blind_guard.check_record(r)
            blind_guard.check_text(blind_guard.prompt_text({"kind": "autocomplete", "prefix": r["prefix"], "suffix": r["suffix"]}))
        except blind_guard.BlindAccessError:
            fails += 1
            continue
        if validation is not None:
            v = validation.get(rec["id"], {})
            r["exec"] = v.get("exec", "unvalidated")
        if emit_qwen:
            r["qwen_prompt"] = qwen_prompt(r["prefix"], r["suffix"])
            r["qwen_target"] = r["middle"]
        out.append(r)
    return out


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
    ap.add_argument("--emit-qwen", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.seed is not None:
        cfg["seed"] = a.seed
    ver = a.ver or cfg.get("version", "v0")
    gdir = ROOT / cfg.get("out_dir", "data/generated") / ver
    snips = load_jsonl(Path(a.snippets) if a.snippets else gdir / "snippets.jsonl")
    val = None
    if a.validation:
        val = {r["id"]: r for r in load_jsonl(Path(a.validation))}
    count = a.count or cfg["fim"]["count"]
    recs = generate_fim(snips, cfg, count, val, a.emit_qwen)
    out = Path(a.out) if a.out else gdir / "fim.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            line = json.dumps(r, ensure_ascii=False, separators=(",", ":"))
            f.write(line + "\n")
            h.update(line.encode("utf-8"))
    dist: dict = {}
    for r in recs:
        dist[r["cut_kind"]] = dist.get(r["cut_kind"], 0) + 1
    n = max(1, len(recs))
    man = {"seed": cfg["seed"], "count": len(recs), "requested": count, "sha256": h.hexdigest(),
           "cut_distribution": {k: round(v / n, 4) for k, v in sorted(dist.items())},
           "target": cfg["fim"]["cut_targets"],
           "splits": {s: sum(1 for r in recs if r["split"] == s) for s in ("train", "val", "test", "bench")}}
    (out.parent / "fim.manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print(json.dumps(man))


if __name__ == "__main__":
    main()
