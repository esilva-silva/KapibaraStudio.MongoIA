"""Documentos de referencia da HOMOLOGACAO selada para a decontaminacao do treino (corpus v3), lidos SO pela API autorizada de homolog_guard.

  import homolog_refs
  docs = homolog_refs.homolog_docs()      # [("homolog", {id, type, kind, prefix, middle, suffix, snippet_id, collection})] um por snippet

Por que existe: `llm_to_fim.homolog_docs()` procura `<data/homolog>/v1/items/items.jsonl`, mas o conjunto selado fica em `<data/homolog>/v1/v1/items/items.jsonl`
(ver `homolog_guard.set_dir`); com o caminho errado ele devolve [] SEM erro e a decontaminacao n-grama contra a homologacao nao acontece (so o hash exato
do manifesto). Aqui o caminho vem de `set_dir()` e a ausencia dos itens com manifesto selado e ERRO, nunca lista vazia silenciosa.
Cada leitura grava uma linha em access.log.jsonl (`homolog_guard.authorized`); nada e copiado para o corpus.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_LLM = HERE / "llm"
if str(_LLM) not in sys.path:
    sys.path.insert(0, str(_LLM))

import homolog_guard as hg  # noqa: E402

FIELDS = ("id", "type", "kind", "prefix", "middle", "suffix", "snippet_id", "collection")


def items_path(hdir=None) -> Path:
    return hg.set_dir(hdir) / "items" / "items.jsonl"


def homolog_docs(hdir=None, who: str = "homolog_refs") -> list[tuple[str, dict]]:
    p = items_path(hdir)
    sealed = bool(hg.manifest(hdir).get("prompt_sha256"))
    with hg.authorized("decontaminacao do treino (corpus v3) contra a homologacao selada (so features/ngrams; nada e copiado)", who, hdir):
        if not p.exists():
            if sealed:
                raise SystemExit(f"[homolog_refs] manifesto selado existe mas {p} nao: a decontaminacao contra a homologacao nao pode ser pulada")
            return []
        out, seen = [], set()
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["snippet_id"] in seen:
                    continue
                seen.add(r["snippet_id"])
                out.append(("homolog", {k: r.get(k) for k in FIELDS}))
    return out
