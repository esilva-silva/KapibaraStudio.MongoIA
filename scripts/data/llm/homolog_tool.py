#!/usr/bin/env python3
"""Ferramenta do conjunto de HOMOLOGACAO LLM (data/homolog/v1). Unico caminho autorizado de leitura.

  homolog_tool.py seal   [--dir data/homolog/v1]    # sha256 por item e do conjunto -> sealed/manifest.json (+ copia versionavel em reports/datasets)
  homolog_tool.py verify [--dir ...]                # exit 1 se divergir (item alterado, removido, adicionado)
  homolog_tool.py read   --reason "..." [--who X] [--model M] [--cut-kind K] [--context-kind C] [--domain D] [--limit N]
  homolog_tool.py stats  [--json]

Layout: data/homolog/v1/{items/items.jsonl, sealed/manifest.json, access.log.jsonl, README.md}. `author: llm` (NAO humano): o conjunto mede
generalizacao para estilo/estrutura novos, nao para consultas humanas reais (mesma familia Qwen dos alunos). data/homolog NAO e versionado.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import homolog_guard as hg  # noqa: E402

ROOT = hg.ROOT
VERSIONED_MANIFEST = ROOT / "reports" / "datasets" / "homolog-v1-manifest.json"


def _dir(a) -> Path:
    return Path(a.dir) if a.dir else hg.homolog_dir() / "v1"


def _items(d: Path):
    p = d / "items" / "items.jsonl"
    with hg.authorized("leitura interna do homolog_tool", "homolog_tool", detail={"cmd": "internal"}):
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def set_hash(item_hashes: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(item_hashes):
        h.update(f"{k}:{item_hashes[k]}\n".encode())
    return h.hexdigest()


def seal(d: Path, quiet=False) -> dict:
    items = list(_items(d))
    if not items:
        raise SystemExit("sem itens")
    ih, ph, groups = {}, [], Counter()
    for it in items:
        # build_homolog marca consultas do LLM como author=llm e cortes dos geradores de gramatica como author=grammar;
        # nenhum dos dois e humano (o conjunto nunca se apresenta como revisado por pessoas)
        if it.get("author") not in ("llm", "grammar") or it.get("split") != "hom" or it.get("split_role") != "hom":
            raise SystemExit(f"item fora do contrato (author in llm|grammar, split=hom, split_role=hom): {it.get('id')} "
                             f"author={it.get('author')} split={it.get('split')} split_role={it.get('split_role')}")
        if it["id"] in ih:
            raise SystemExit(f"id repetido: {it['id']}")
        ih[it["id"]] = hg.item_sha256(it)
        ph.append(hg.prompt_sha256((it.get("prefix") or "") + "\x00" + (it.get("suffix") or "")))
        groups[(it.get("cut_kind"), it.get("context_kind"))] += 1
    man = {"format": "homolog-sealed-v1", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "author": "llm", "n_items": len(items),
           "set_sha256": set_hash(ih), "item_sha256": ih, "prompt_sha256": sorted(set(ph)),
           "limitation": "gerado por LLM da mesma familia (Qwen) dos alunos: mede generalizacao para estilo/estrutura novos, nao consultas humanas reais"}
    (d / "sealed").mkdir(parents=True, exist_ok=True)
    (d / "sealed" / "manifest.json").write_text(json.dumps(man, indent=0, ensure_ascii=False), encoding="utf-8")
    ip = d / "items" / "items.jsonl"
    os.chmod(ip, stat.S_IREAD)
    pub = {k: v for k, v in man.items() if k not in ("item_sha256", "prompt_sha256")}
    pub["by_cut_kind"] = dict(Counter(it["cut_kind"] for it in items))
    pub["by_context_kind"] = dict(Counter(it["context_kind"] for it in items))
    pub["by_domain"] = dict(Counter(it["domain"] for it in items))
    pub["by_schema"] = dict(Counter(it["schema_id"] for it in items))
    pub["items_file_sha256"] = hashlib.sha256(Path(ip).read_bytes()).hexdigest()
    VERSIONED_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    VERSIONED_MANIFEST.write_text(json.dumps(pub, indent=1, ensure_ascii=False), encoding="utf-8")
    if not quiet:
        print(json.dumps({"sealed": len(items), "set_sha256": man["set_sha256"], "versioned_manifest": str(VERSIONED_MANIFEST)}))
    return man


def verify(d: Path) -> bool:
    mp = d / "sealed" / "manifest.json"
    if not mp.exists():
        print("sem manifesto")
        return False
    man = json.loads(mp.read_text(encoding="utf-8"))
    cur = {it["id"]: hg.item_sha256(it) for it in _items(d)}
    bad = [k for k in man["item_sha256"] if cur.get(k) != man["item_sha256"][k]]
    extra = [k for k in cur if k not in man["item_sha256"]]
    ok = not bad and not extra and set_hash(cur) == man["set_sha256"]
    print(json.dumps({"ok": ok, "n": len(cur), "alterados_ou_ausentes": len(bad), "extras": len(extra), "set_sha256": set_hash(cur)}))
    return ok


def read(d: Path, a):
    who = a.who or "?"
    with hg.authorized(a.reason, who, detail={"cmd": "read", "model": a.model, "filters": {"cut_kind": a.cut_kind, "context_kind": a.context_kind,
                                                                                           "domain": a.domain, "limit": a.limit}}):
        n = 0
        with open(d / "items" / "items.jsonl", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                it = json.loads(line)
                if a.cut_kind and it.get("cut_kind") != a.cut_kind:
                    continue
                if a.context_kind and it.get("context_kind") != a.context_kind:
                    continue
                if a.domain and it.get("domain") != a.domain:
                    continue
                print(line.rstrip("\n"))
                n += 1
                if a.limit and n >= a.limit:
                    break


def stats(d: Path, as_json: bool):
    items = list(_items(d))
    c = lambda k: dict(Counter(it.get(k) for it in items))  # noqa: E731
    out = {"n": len(items), "cut_kind": c("cut_kind"), "context_kind": c("context_kind"), "domain": c("domain"), "schema_id": c("schema_id"),
           "source": c("source"), "author": c("author")}
    print(json.dumps(out, indent=None if as_json else 1, ensure_ascii=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("seal", "verify", "read", "stats"):
        p = sp.add_parser(name)
        p.add_argument("--dir")
        if name == "read":
            p.add_argument("--reason", required=True)
            p.add_argument("--who")
            p.add_argument("--model")
            p.add_argument("--cut-kind")
            p.add_argument("--context-kind")
            p.add_argument("--domain")
            p.add_argument("--limit", type=int)
        if name == "stats":
            p.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    d = _dir(a)
    if a.cmd == "seal":
        seal(d)
    elif a.cmd == "verify":
        sys.exit(0 if verify(d) else 1)
    elif a.cmd == "read":
        read(d, a)
    else:
        stats(d, a.json)


if __name__ == "__main__":
    main()
