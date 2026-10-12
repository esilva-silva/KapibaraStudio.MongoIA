#!/usr/bin/env python3
"""Ferramenta do conjunto cego (Fase 03 §4.6, Fase 06 §9). Unico caminho autorizado de leitura.

  blind_tool.py init
  blind_tool.py add   --file itens.jsonl [--new-round --reason "..."]
  blind_tool.py seal  [--allow-drafts]            # sha256 por item e do conjunto -> sealed/manifest.json
  blind_tool.py verify                            # exit 1 se divergir
  blind_tool.py read  --reason "..." [--who X] [--model M] [--id ID] [--kind K] [--domain D] [--language L] [--limit N]
  blind_tool.py stats [--json] [--strict]

Layout: data/eval/blind/{items/items.jsonl, sealed/manifest.json, access.log.jsonl, README.md}
Sem dependencias externas (validador de JSON-schema embutido).
"""
from __future__ import annotations

import argparse
import datetime as dt
import getpass
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blind_guard as G  # noqa: E402

ROOT = G.ROOT
ITEM_SCHEMA = ROOT / "data" / "eval" / "blind_item.schema.json"
SCHEMAS_DIR = ROOT / "data" / "schemas"
TARGET_TOTAL = 400
HOLDOUT_MIN = 0.40
LANG_BAND = (0.45, 0.55)


# ---- mini JSON-schema ---------------------------------------------------------------------

_TYPES = {"string": str, "object": dict, "array": list, "boolean": bool, "null": type(None),
          "integer": int, "number": (int, float)}


def _is_type(v, t):
    if t in ("integer", "number") and isinstance(v, bool):
        return False
    return isinstance(v, _TYPES[t])


def js_validate(inst, sch, path="$") -> list[str]:
    errs = []
    t = sch.get("type")
    if t is not None:
        ts = t if isinstance(t, list) else [t]
        if not any(_is_type(inst, x) for x in ts):
            return [f"{path}: tipo {type(inst).__name__} != {'|'.join(ts)}"]
    if "const" in sch and inst != sch["const"]:
        errs.append(f"{path}: deve ser {sch['const']!r}")
    if "enum" in sch and inst not in sch["enum"]:
        errs.append(f"{path}: {inst!r} fora de {sch['enum']}")
    if isinstance(inst, str):
        if len(inst) < sch.get("minLength", 0):
            errs.append(f"{path}: string curta demais (min {sch['minLength']})")
        if "pattern" in sch and not re.search(sch["pattern"], inst):
            errs.append(f"{path}: nao casa /{sch['pattern']}/")
    if isinstance(inst, list):
        if len(inst) < sch.get("minItems", 0):
            errs.append(f"{path}: min {sch['minItems']} itens")
        if "items" in sch:
            for i, x in enumerate(inst):
                errs += js_validate(x, sch["items"], f"{path}[{i}]")
    if isinstance(inst, dict):
        for r in sch.get("required", []):
            if r not in inst:
                errs.append(f"{path}: campo obrigatorio ausente: {r}")
        props = sch.get("properties", {})
        if sch.get("additionalProperties") is False:
            for k in inst:
                if k not in props:
                    errs.append(f"{path}: campo desconhecido: {k}")
        for k, sub in props.items():
            if k in inst:
                errs += js_validate(inst[k], sub, f"{path}.{k}")
    for sub in sch.get("allOf", []):
        if "if" in sub:
            cond_ok = not js_validate(inst, sub["if"], path)
            branch = sub.get("then") if cond_ok else sub.get("else")
            if branch:
                errs += js_validate(inst, branch, path)
        else:
            errs += js_validate(inst, sub, path)
    return errs


_item_schema_cache = None


def item_schema() -> dict:
    global _item_schema_cache
    if _item_schema_cache is None:
        _item_schema_cache = json.loads(ITEM_SCHEMA.read_text(encoding="utf-8"))
    return _item_schema_cache


_schemas_cache = None


def load_data_schemas() -> dict:
    """schema_id -> {domain, holdout, collections:set}"""
    global _schemas_cache
    if _schemas_cache is None:
        out = {}
        for f in sorted(SCHEMAS_DIR.glob("*/*.json")):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except ValueError:
                continue
            out[d["schema_id"]] = {"domain": d.get("domain"), "holdout": bool(d.get("holdout")),
                                   "collections": {c["name"] for c in d.get("collections", [])}}
        _schemas_cache = out
    return _schemas_cache


def validate_item(item) -> tuple[list[str], list[str]]:
    """(erros, avisos)."""
    if not isinstance(item, dict):
        return ["item nao e objeto JSON"], []
    errs = js_validate(item, item_schema())
    warns = []
    if errs:
        return errs, warns
    sch = load_data_schemas().get(item["schema_id"])
    if sch is None:
        warns.append(f"schema_id {item['schema_id']!r} nao encontrado em data/schemas (sem checagem estrutural)")
    else:
        if item["collection"] not in sch["collections"]:
            errs.append(f"collection {item['collection']!r} nao existe em {item['schema_id']} ({sorted(sch['collections'])})")
        if sch["domain"] != item["domain"]:
            errs.append(f"domain {item['domain']!r} != domain do schema {sch['domain']!r}")
    if item["kind"] == "autocomplete" and not item["prefix"].strip():
        errs.append("autocomplete com prefix vazio")
    return errs, warns


# ---- armazenamento ------------------------------------------------------------------------

def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_who() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USERNAME", "unknown")


def die(msg, code=1):
    print(f"ERRO: {msg}", file=sys.stderr)
    sys.exit(code)


class Store:
    def __init__(self, base=None, who=None):
        self.base = G.blind_dir(base)
        self.who = who
        self.items_dir = self.base / "items"
        self.items_file = self.items_dir / "items.jsonl"
        self.sealed_dir = self.base / "sealed"
        self.manifest = self.sealed_dir / "manifest.json"
        self.history = self.sealed_dir / "history"
        self.log = self.base / "access.log.jsonl"

    def require_init(self):
        if not (self.items_dir.is_dir() and self.sealed_dir.is_dir()):
            die(f"nao inicializado: rode `blind_tool.py init` ({self.base})", 2)

    def load_items(self) -> list[dict]:
        out = []
        for f in sorted(self.items_dir.glob("*.jsonl")):
            for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except ValueError as e:
                        die(f"{f.name}:{n}: JSON invalido: {e}", 2)
        return out

    def sealed(self):
        if self.manifest.exists():
            return json.loads(self.manifest.read_text(encoding="utf-8"))
        return None

    def log_access(self, action, reason=None, **extra):
        rec = {"ts": now(), "who": self.who or default_who(), "action": action, "reason": reason, **extra}
        with open(self.log, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def set_hash(item_hashes: dict) -> str:
    h = hashlib.sha256()
    for i in sorted(item_hashes):
        h.update(f"{i}:{item_hashes[i]}\n".encode())
    return h.hexdigest()


def compute_hashes(items):
    ih = {it["id"]: G.item_sha256(it) for it in items}
    return ih, set_hash(ih)


# ---- subcomandos --------------------------------------------------------------------------

def cmd_init(a, st: Store):
    st.items_dir.mkdir(parents=True, exist_ok=True)
    st.sealed_dir.mkdir(parents=True, exist_ok=True)
    if not st.items_file.exists():
        st.items_file.write_text("", encoding="utf-8")
    if not st.log.exists():
        st.log.write_text("", encoding="utf-8")
    st.log_access("init", "init")
    print(f"ok: {st.base}")


def cmd_add(a, st: Store):
    st.require_init()
    m = st.sealed()
    if m:
        if not a.new_round:
            die(f"conjunto selado ({m['version']}); `add` recusado sem --new-round "
                f"(reescreva os itens para a nova rodada)", 3)
        if not (a.reason and a.reason.strip()):
            die("--new-round exige --reason", 2)
    text = sys.stdin.read() if a.file == "-" else Path(a.file).read_text(encoding="utf-8")
    existing = st.load_items()
    ids = {i["id"] for i in existing}
    prompts = {G.prompt_sha256(G.prompt_text(i)) for i in existing}
    new, bad = [], 0
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            it = json.loads(line)
        except ValueError as e:
            print(f"linha {n}: JSON invalido: {e}", file=sys.stderr)
            bad += 1
            continue
        errs, warns = validate_item(it)
        label = it.get("id", f"linha {n}") if isinstance(it, dict) else f"linha {n}"
        if not errs:
            if it["id"] in ids:
                errs.append("id duplicado")
            ph = G.prompt_sha256(G.prompt_text(it))
            if ph in prompts:
                errs.append("prompt duplicado (mesmo prefix/suffix ou instruction/context)")
            ids.add(it["id"])
            prompts.add(ph)
        for e in errs:
            print(f"{label}: {e}", file=sys.stderr)
        for w in warns:
            print(f"{label}: aviso: {w}", file=sys.stderr)
        bad += bool(errs)
        if not errs:
            new.append(it)
    if bad:
        die(f"{bad} item(ns) invalido(s); nada foi gravado", 1)
    if m:  # nova rodada: arquiva o manifesto so depois de validar tudo
        st.history.mkdir(exist_ok=True)
        dest = st.history / f"manifest.round{m['round']}.json"
        os.replace(st.manifest, dest)
        st.log_access("new-round", a.reason, archived=dest.name, previous=m["version"])
        print(f"rodada {m['round']} arquivada em {dest.name}; selar novamente apos adicionar")
    with open(st.items_file, "a", encoding="utf-8", newline="\n") as fh:
        for it in new:
            fh.write(json.dumps(it, ensure_ascii=False, sort_keys=True) + "\n")
    st.log_access("add", a.reason or "add", n=len(new), file=str(a.file))
    print(f"ok: {len(new)} item(ns) adicionados (total {len(existing) + len(new)})")


def cmd_seal(a, st: Store):
    st.require_init()
    if st.sealed():
        die("ja selado; use `add --new-round` para abrir nova rodada", 3)
    items = st.load_items()
    if not items:
        die("sem itens para selar", 2)
    problems = []
    for it in items:
        errs, _ = validate_item(it)
        problems += [f"{it.get('id')}: {e}" for e in errs]
        if not a.allow_drafts:
            if it.get("status") != "reviewed" or not it.get("reviewed_by"):
                problems.append(f"{it.get('id')}: nao revisado por humano (status={it.get('status')})")
            if it.get("author") != "human":
                problems.append(f"{it.get('id')}: author={it.get('author')} "
                                f"(reescrever/assumir autoria humana antes de selar)")
    if problems:
        for p in problems[:50]:
            print(p, file=sys.stderr)
        die(f"{len(problems)} problema(s); selagem recusada (--allow-drafts so para testes)", 1)
    history = sorted(st.history.glob("manifest.round*.json")) if st.history.exists() else []
    rnd = len(history) + 1
    ih, sh = compute_hashes(items)
    mf = {
        "format_version": 1,
        "version": f"blind-v{rnd}",
        "round": rnd,
        "sealed_at": now(),
        "sealed_by": st.who or default_who(),
        "count": len(items),
        "set_sha256": sh,
        "composition": {"kind": dict(Counter(i["kind"] for i in items)),
                        "language": dict(Counter(i["language"] for i in items))},
        "files_sha256": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                         for f in sorted(st.items_dir.glob("*.jsonl"))},
        "item_sha256": ih,
        "prompt_sha256": sorted({G.prompt_sha256(G.prompt_text(i)) for i in items}),
    }
    st.manifest.write_text(json.dumps(mf, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    st.log_access("seal", "seal", version=mf["version"], count=mf["count"], set_sha256=sh)
    print(f"selado {mf['version']}: {mf['count']} itens, set_sha256={sh}")


def do_verify(st: Store) -> tuple[int, list[str]]:
    m = st.sealed()
    if not m:
        return 2, ["nao selado (sealed/manifest.json ausente)"]
    items = st.load_items()
    ih, sh = compute_hashes(items)
    probs = []
    exp = m["item_sha256"]
    for i in sorted(set(exp) - set(ih)):
        probs.append(f"item removido: {i}")
    for i in sorted(set(ih) - set(exp)):
        probs.append(f"item nao selado (adicionado): {i}")
    for i in sorted(set(ih) & set(exp)):
        if ih[i] != exp[i]:
            probs.append(f"item alterado: {i}")
    if len(items) != len(ih):
        probs.append("ids duplicados no arquivo de itens")
    if len(items) != m["count"]:
        probs.append(f"contagem {len(items)} != manifesto {m['count']}")
    if sh != m["set_sha256"] and not probs:
        probs.append("hash do conjunto diverge")
    return (1 if probs else 0), probs


def cmd_verify(a, st: Store):
    st.require_init()
    code, probs = do_verify(st)
    st.log_access("verify", "verify", ok=(code == 0), problems=len(probs))
    for p in probs:
        print(p, file=sys.stderr)
    if code == 0:
        m = st.sealed()
        print(f"OK {m['version']}: {m['count']} itens, set_sha256={m['set_sha256']}")
    else:
        print("FALHOU" if code == 1 else "NAO SELADO", file=sys.stderr)
    return code


def cmd_read(a, st: Store):
    st.require_init()
    if not (a.reason and a.reason.strip()):
        die("--reason obrigatorio (motivo do acesso fica no access.log)", 2)
    items = st.load_items()
    flt = {k: v for k, v in (("id", a.id), ("kind", a.kind), ("domain", a.domain), ("language", a.language)) if v}
    sel = [i for i in items if all(i.get(k) == v for k, v in flt.items())]
    if a.limit:
        sel = sel[:a.limit]
    m = st.sealed()
    vcode = do_verify(st)[0] if m else None
    st.log_access("read", a.reason, model=a.model, filter=flt, n_items=len(sel),
                  sealed=(m or {}).get("version"), verify_ok=(vcode == 0) if m else None,
                  ids=[i["id"] for i in sel])
    if m and vcode != 0:
        print("AVISO: conjunto selado diverge do manifesto (ver `verify`)", file=sys.stderr)
    for it in sel:
        print(json.dumps(it, ensure_ascii=False, sort_keys=True))


def compute_stats(items):
    sch = load_data_schemas()
    dom_holdout = {}
    for s in sch.values():
        dom_holdout[s["domain"]] = dom_holdout.get(s["domain"], False) or s["holdout"]

    def is_holdout(i):
        s = sch.get(i["schema_id"])
        return s["holdout"] if s else dom_holdout.get(i["domain"], False)

    n = len(items)
    st = {
        "count": n,
        "by_domain": dict(sorted(Counter(i["domain"] for i in items).items())),
        "by_kind": dict(sorted(Counter(i["kind"] for i in items).items())),
        "by_language": dict(sorted(Counter(i["language"] for i in items).items())),
        "by_group": dict(sorted(Counter(i.get("group") or f"{i['schema_id']}::{i['collection']}"
                                        for i in items).items())),
        "by_status": dict(sorted(Counter(i["status"] for i in items).items())),
        "by_author": dict(sorted(Counter(i["author"] for i in items).items())),
        "by_kind_language": {k: dict(Counter(i["language"] for i in items if i["kind"] == k))
                             for k in sorted({i["kind"] for i in items})},
    }
    ho = sum(1 for i in items if is_holdout(i))
    pt = st["by_language"].get("pt-BR", 0)
    en = st["by_language"].get("en", 0)
    checks = []

    def chk(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    chk("holdout>=40%", n > 0 and ho / n >= HOLDOUT_MIN, f"{ho}/{n} = {ho / n:.1%}" if n else "0/0")
    frac = pt / (pt + en) if pt + en else 0
    chk("pt-BR x en balanceado (45-55%)", LANG_BAND[0] <= frac <= LANG_BAND[1],
        f"pt-BR {pt} / en {en} ({frac:.1%} pt-BR)")
    unrev = sum(1 for i in items if i["status"] != "reviewed" or i["author"] != "human")
    chk("todos revisados e escritos por humano", unrev == 0, f"{unrev} itens draft/draft-agent")
    chk(f"tamanho {TARGET_TOTAL}", n >= TARGET_TOTAL, f"{n}/{TARGET_TOTAL} (meta: 200 autocomplete + 200 chat)")
    st["holdout_items"] = ho
    st["checks"] = checks
    return st


def cmd_stats(a, st: Store):
    st.require_init()
    s = compute_stats(st.load_items())
    st.log_access("stats", "stats", count=s["count"])
    if a.json:
        print(json.dumps(s, ensure_ascii=False, indent=1))
    else:
        print(f"itens: {s['count']}  (holdout: {s['holdout_items']})")
        for k in ("by_kind", "by_language", "by_domain", "by_group", "by_status", "by_author", "by_kind_language"):
            print(f"{k[3:]}:")
            for kk, v in s[k].items():
                print(f"  {kk}: {v}")
        for c in s["checks"]:
            print(f"[{'PASS' if c['ok'] else 'FAIL'}] {c['check']}: {c['detail']}")
    return 1 if a.strict and not all(c["ok"] for c in s["checks"]) else 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--blind-dir", help="sobrescreve data/eval/blind (testes)")
    p.add_argument("--who", help="identidade registrada no access.log (padrao: usuario do SO)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    s = sub.add_parser("add")
    s.add_argument("--file", required=True, help="JSONL de itens ('-' = stdin)")
    s.add_argument("--new-round", action="store_true")
    s.add_argument("--reason")
    s = sub.add_parser("seal")
    s.add_argument("--allow-drafts", action="store_true", help="so para testes; nunca no blind final")
    sub.add_parser("verify")
    s = sub.add_parser("read")
    s.add_argument("--reason")
    s.add_argument("--model")
    s.add_argument("--id")
    s.add_argument("--kind")
    s.add_argument("--domain")
    s.add_argument("--language")
    s.add_argument("--limit", type=int)
    s = sub.add_parser("stats")
    s.add_argument("--json", action="store_true")
    s.add_argument("--strict", action="store_true", help="exit 1 se alguma meta falhar")
    return p


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    a = build_parser().parse_args(argv)
    st = Store(a.blind_dir, a.who)
    fn = {"init": cmd_init, "add": cmd_add, "seal": cmd_seal, "verify": cmd_verify,
          "read": cmd_read, "stats": cmd_stats}[a.cmd]
    return fn(a, st) or 0


if __name__ == "__main__":
    sys.exit(main())
