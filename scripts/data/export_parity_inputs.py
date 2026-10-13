#!/usr/bin/env python3
"""Exporta as entradas da prova de paridade Python x IDE (`kapilab contract render/tokenize/diff`, PR 1 da IDE).

  python scripts/data/export_parity_inputs.py [--ver v1] [--n 2000] [--seed 20261008]
                                             [--src data/generated/v1/fim_ctx.jsonl] [--out-dir data/contracts]

Grava
  data/contracts/parity_inputs.jsonl    exatamente o que o KapiLab precisa receber por registro: snapshot do editor
                                        (text = prefix+suffix, caret em unidades UTF-16 = len16(prefix), language, input_panel,
                                        result_fields, known_names, recent_commands, opcoes padrao), subconjunto de schema
                                        (facts FieldSchema/Operator/Indexes) e orcamento (context_tokens, tokenizer);
  data/contracts/parity_expected.jsonl  o esperado do renderer Python: sha256 do contexto, do ModelPrefix, do prompt FIM Qwen,
                                        n. de tokens e sha256 dos ids (CSV decimal) do `QwenFimPromptBuilder`.
Selecao deterministica (ordem por sha256(seed:id)), estratificada por context_kind. Nunca le data/eval/blind.
Registros `context_kind=none` levam `raw_context_empty=true` (a IDE equivalente e um AutocompleteRequest com Context vazio).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blind_guard  # noqa: E402
import render_contract as rc  # noqa: E402
from jsonl_shards import iter_jsonl, resolve_inputs  # noqa: E402

ROOT = rc.ROOT


def ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest()


def select(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Primeiros n por hash, round-robin entre context_kind para manter none/irrelevant representados."""
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest()):
        by_kind[r["context_kind"]].append(r)
    total = len(rows)
    out: list[dict] = []
    for kind in sorted(by_kind):
        out += by_kind[kind][: max(1, round(n * len(by_kind[kind]) / total))]
    out = sorted(out, key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest())[:n]
    return sorted(out, key=lambda r: r["id"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ver", default="v1")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--src")
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "contracts"))
    ap.add_argument("--tokenizer", default=str(rc.DEFAULT_TOKENIZER))
    a = ap.parse_args(argv)
    src = Path(a.src) if a.src else ROOT / "data" / "generated" / a.ver / "fim_ctx.jsonl"
    blind_guard.check_path(src)
    files = resolve_inputs(src)          # arquivo unico ou shards fim_ctx-00000.jsonl (v2: ~1,5 M de registros -> 2 passadas)
    light = [{"id": r["id"], "context_kind": r["context_kind"]} for r in iter_jsonl(files)]
    want = {r["id"] for r in select(light, a.n, a.seed)}
    del light
    chosen = sorted((r for r in iter_jsonl(files) if r["id"] in want), key=lambda r: r["id"])
    tok = rc.Tokenizer(a.tokenizer)
    try:
        tok_label = Path(a.tokenizer).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        tok_label = str(a.tokenizer)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_trunc = 0
    with open(out_dir / "parity_inputs.jsonl", "w", encoding="utf-8", newline="\n") as fi, \
            open(out_dir / "parity_expected.jsonl", "w", encoding="utf-8", newline="\n") as fe:
        for r in chosen:
            ed = r.get("editor")
            language = rc.editor_language(r)
            prefix, suffix = rc.bound_window(r["prefix"], r["suffix"])
            context = rc.editor_context(ed, language)
            before = rc.model_prefix(prefix, context)
            ids, info = rc.build_ids(tok, before, suffix)
            n_trunc += info["truncated"]
            prompt = rc.qwen_fim_text(before, suffix) if not info["truncated"] else tok.decode(ids)
            inp = {
                "id": r["id"], "contract": rc.CONTRACT_ID, "split": r["split"], "group_id": r["group_id"],
                "snapshot": {"text": r["prefix"] + r["suffix"], "caret": rc.u16_len(r["prefix"]), "language": language,
                             "input_panel": (ed or {}).get("input_panel", ""), "result_fields": (ed or {}).get("result_fields", []),
                             "known_names": (ed or {}).get("known_names", []), "recent_commands": (ed or {}).get("recent_commands", [])},
                "options": {"use_editor_context": True, "use_result_panel": True, "use_input_panel": True},
                "raw_context_empty": ed is None,
                "schema_subset": [f for f in r["facts"] if f["kind"] in ("FieldSchema", "Operator", "Indexes")],
                "target": (ed or {}).get("target"), "context_tokens": rc.CONTEXT_TOKENS, "model_family": "qwen2.5-coder-fim",
                "tokenizer": tok_label,
            }
            exp = {
                "id": r["id"], "context_sha256": None if context is None else rc.prompt_sha256(context),
                "model_prefix_sha256": rc.prompt_sha256(before), "prompt_sha256": rc.prompt_sha256(prompt),
                "prompt_tokens": len(ids), "ids_sha256": ids_sha256(ids), "truncated": info["truncated"],
                "prefix_len_utf16": rc.u16_len(prefix), "suffix_len_utf16": rc.u16_len(suffix),
            }
            if "prompt_sha256" in r and not info["truncated"]:
                assert r["prompt_sha256"] == exp["prompt_sha256"], r["id"]  # fim_ctx.jsonl consistente com o renderer
            fi.write(json.dumps(inp, ensure_ascii=False, separators=(",", ":")) + "\n")
            fe.write(json.dumps(exp, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(json.dumps({"records": len(chosen), "truncated": n_trunc, "out_dir": str(out_dir)}))


if __name__ == "__main__":
    main()
