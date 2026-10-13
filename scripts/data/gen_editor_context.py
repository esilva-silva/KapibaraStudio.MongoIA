#!/usr/bin/env python3
"""Gerador de contexto de editor sobre o FIM (spec fase-03 §4.3 `gen_editor_context.py`, contrato `editor-context-v1`).

  python scripts/data/gen_editor_context.py --ver v1 [--seed 20261008] [--limit N] [--no-tokenizer]

Le   data/generated/<ver>/fim.jsonl + snippets.jsonl + data/schemas/**.json (nada de data/eval/blind).
Grava data/generated/<ver>/fim_ctx.jsonl (+ fim_ctx.manifest.json, reports/datasets/<ver>-editor-context-report.md).

Registro: todos os campos do FIM de origem (split, group_id, holdout, cut_kind, ... preservados) mais
  contrato, language, editor (target, known_names, result_fields, recent_commands, input_panel, newline) ou null (8%),
  known_names/result_fields/recent_commands (copias de editor, [] sem contexto), facts, context_kind (full|irrelevant|none),
  editor_text (prefix + cursor + suffix), target (= middle), prompt (FIM Qwen completo, pos-orcamento), prompt_tokens,
  prompt_truncated, prompt_sha256.

Simulacao (so a partir do schema da colecao e de snippets do MESMO schema/colecao/split):
  * known_names: conexoes, banco (= schema_id, como nos snippets ConnectionPool.local.<schema>.<colecao>), colecoes do schema;
  * result_fields: caminhos do schema (todos, projecao de 3-8, ou nenhum) -> nunca campo inexistente (exceto `irrelevant`, que
    de proposito usa campos de OUTRA colecao);
  * recent_commands: snippets da mesma colecao e split, de outro grupo (nao ha vazamento train->test);
  * facts: Target, FieldSchema (subconjunto), Operator, Indexes, LocalVariable, SimilarStatement — todos verdadeiros para a colecao.
O texto do prompt e o do renderer `render_contract.py` (cabecalho com CRLF fixo; `*/` -> `* /`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blind_guard  # noqa: E402
import render_contract as rc  # noqa: E402
from gen_snippets import CONFIG_PATH, ROOT, load_config, load_schemas  # noqa: E402

SCHEMA_VERSION = "kapi-ds-record-v2"
OP_RE = re.compile(r"\$[A-Za-z][A-Za-z0-9]*")
DECL_RE = re.compile(r"(?:^|[;\n\r])\s*(const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(.{0,60})")

QUERY_OPS = {
    "number": ["$gt", "$gte", "$lt", "$lte", "$eq", "$ne", "$in", "$nin"],
    "date": ["$gt", "$gte", "$lt", "$lte", "$eq", "$ne"],
    "string": ["$regex", "$eq", "$ne", "$in", "$nin", "$exists"],
    "array": ["$in", "$all", "$elemMatch", "$size", "$exists"],
    "bool": ["$eq", "$ne", "$exists"],
    "id": ["$in", "$eq", "$ne", "$nin", "$exists"],
    "any": ["$exists", "$type", "$and", "$or", "$nor", "$not", "$expr"],
}
STAGE_OPS = ["$match", "$group", "$project", "$sort", "$limit", "$skip", "$lookup", "$unwind", "$addFields", "$set", "$unset",
             "$count", "$sum", "$avg", "$min", "$max", "$first", "$last", "$push", "$addToSet"]
UPDATE_OPS = ["$set", "$unset", "$inc", "$push", "$pull", "$addToSet", "$mul", "$min", "$max", "$currentDate", "$rename"]
TYPE_FAMILY = {"int32": "number", "int64": "number", "double": "number", "decimal128": "number", "date": "date", "string": "string",
               "array": "array", "bool": "bool", "objectid": "id", "uuid": "id"}


def _rng(seed, rid: str, salt: str = "ctx") -> random.Random:
    return random.Random(f"{seed}:{salt}:{rid}")


def _weighted(rng: random.Random, weights: dict):
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# -- indices auxiliares -------------------------------------------------------------------------------------------------------------
class Pools:
    """Snippets por (schema_id, colecao, split), para recent_commands/SimilarStatement sem vazamento entre splits."""

    def __init__(self, snippets: list[dict], schemas):
        self.by_key: dict[tuple, list[tuple[str, str, str]]] = defaultdict(list)
        self.snippet_by_id: dict[str, dict] = {}
        for s in snippets:
            self.snippet_by_id[s["id"]] = s
            if len(s["text"]) <= rc.RECENT_MAX_CHARS and rc.is_safe(s["text"]):
                self.by_key[(s["schema_id"], s["collection"], s["split"])].append((s["id"], s["text"], s["group_id"]))
        for v in self.by_key.values():
            v.sort()
        self.schemas = {s.full_id: s for s in schemas}
        self.sorted_schemas = sorted(schemas, key=lambda s: s.full_id)
        self.public = [s for s in self.sorted_schemas if not s.holdout]

    def candidates(self, rec: dict, exclude_snippet: str) -> list[tuple[str, str, str]]:
        return [c for c in self.by_key.get((rec["schema_id"], rec["collection"], rec["split"]), [])
                if c[2] != rec["group_id"] and c[0] != exclude_snippet]

    def distractor_schemas(self, rec: dict):
        """Treino/val nunca recebem vocabulario dos dominios holdout."""
        pool = self.sorted_schemas if rec["split"] in ("test", "bench") else self.public
        return [s for s in pool if s.full_id != rec["schema_id"]]


def schema_paths(coll) -> list[str]:
    return list(coll.fields)


def family(f) -> str:
    return TYPE_FAMILY.get(f.t, "any")


# -- uma linha -----------------------------------------------------------------------------------------------------------------------
def make_context(rec: dict, cfg: dict, pools: Pools, seed) -> dict:
    ec = cfg["editor_context"]
    rid = rec["id"]
    rng = _rng(seed, rid)
    sch = pools.schemas[rec["schema_id"]]
    coll = next(c for c in sch.colls if c.name == rec["collection"])
    snippet = pools.snippet_by_id.get(rec["snippet_id"], {})
    language = rc.DIALECT_LANGUAGE.get(rec.get("dialect", "console"), rc.LANGUAGE_CONSOLE)
    text_visible = rec["prefix"] + rec["suffix"]

    roll = rng.random()
    if roll < ec["none_ratio"]:
        kind = "none"
    elif roll < ec["none_ratio"] + ec["irrelevant_ratio"]:
        kind = "irrelevant"
    else:
        kind = "full"

    root_form = (snippet.get("style") or {}).get("root_form", "dot")
    conn = "local" if root_form == "pool" else rng.choice(ec["connections"])
    db = sch.schema_id
    target = {"connection": conn, "database": db, "collection": coll.name,
              "confidence": "Inferred" if root_form == "var" else "Explicit"}

    facts: list[dict] = []
    editor = None
    if kind != "none":
        sibling = [c.name for c in sch.colls]
        others = [c for c in ec["connections"] if c != conn]
        rng.shuffle(others)
        conns = [conn] + others[: rng.choice([0, 1, 1, 2])]
        rng.shuffle(conns)
        dbs = [db]
        if kind == "full":
            dbs += [s.schema_id for s in rng.sample(pools.distractor_schemas(rec), 2)] if rng.random() < 0.3 else []
            names = conns + dbs + sibling[:12]
            if rng.random() < ec["repeat_target_names"]:
                names += [conn, db, coll.name]
            result_pool = schema_paths(coll)
        else:  # irrelevant: outra conexao/banco/colecoes e campos de OUTRA colecao (de outro schema)
            other = rng.choice(pools.distractor_schemas(rec))
            other_coll = rng.choice(other.colls)
            conns = [c for c in conns if c != conn] or [rng.choice([c for c in ec["connections"] if c != conn])]
            names = conns + [other.schema_id] + [c.name for c in other.colls][:12]
            names = [n for n in names if n not in (db, coll.name)]
            result_pool = schema_paths(other_coll)
        names = names[: ec["max_names"]]

        mode = _weighted(rng, ec["result_fields"])
        if mode == "all":
            result_fields = result_pool[: ec["max_result_fields"]]
        elif mode == "projection":
            k = min(len(result_pool), rng.randint(3, 8))
            idx = sorted(rng.sample(range(len(result_pool)), k))
            result_fields = [result_pool[i] for i in idx][: ec["max_result_fields"]]
        else:
            result_fields = []

        recent: list[str] = []
        n_recent = _weighted(rng, {int(k): v for k, v in ec["recent_commands"].items()})
        cands = pools.candidates(rec, rec["snippet_id"]) if kind == "full" else []
        if n_recent and cands:
            recent = [c[1] for c in rng.sample(cands, min(n_recent, len(cands)))][: ec["max_recent_commands"]]

        input_panel = ""
        if kind == "full" and rng.random() < ec["input_panel_ratio"]:
            scalars = [p for p, f in coll.fields.items() if f.t in ("string", "int32", "int64", "double", "bool") and not f.via_array]
            if scalars:
                pick = rng.sample(scalars, min(len(scalars), rng.randint(1, 2)))
                body = []
                for p in pick:
                    t = coll.fields[p].t
                    v = {"string": '"x"', "bool": "true"}.get(t, "1")
                    body.append(json.dumps(p) + ": " + v)
                input_panel = "{ " + ", ".join(body) + " }"

        editor = {"target": target, "known_names": names, "result_fields": result_fields, "recent_commands": recent,
                  "input_panel": input_panel, "newline": rc.CONTRACT_NEWLINE}

        # ---- facts (sempre verdadeiros para a colecao-alvo)
        facts.append({"kind": "Target", **target})
        sub = ec["schema_subset"]
        hidden = set(rec.get("fields_used") or [])
        chosen: list[str] = []
        for p in schema_paths(coll):
            leaf = p.rsplit(".", 1)[-1]
            if p in hidden or leaf in hidden:
                visible = (p in text_visible) or (leaf in text_visible)
                if visible or rng.random() < sub["include_hidden_target"]:
                    chosen.append(p)
        want = rng.randint(sub["min"], sub["max"])
        rest = [p for p in schema_paths(coll) if p not in chosen]
        rng.shuffle(rest)
        chosen += rest[: max(0, want - len(chosen))]
        for p in [p for p in schema_paths(coll) if p in set(chosen)]:
            f = coll.fields[p]
            fact = {"kind": "FieldSchema", "path": p, "type": f.t, "presence": round(float(f.presence or 1.0), 2)}
            if f.t == "array" and f.item_t:
                fact["item"] = f.item_t
            facts.append(fact)
        ops = [o for o in OP_RE.findall(text_visible)]
        op_kind = rec.get("kind") or snippet.get("op")
        base = STAGE_OPS if op_kind == "aggregate" else UPDATE_OPS if op_kind in ("updateOne", "updateMany") else []
        if not base or op_kind != "aggregate":
            fams = {family(f) for f in coll.fields.values()} | {"any"}
            base = base + [o for fam in sorted(fams) for o in QUERY_OPS.get(fam, [])]
        base = sorted(set(base))
        rng.shuffle(base)
        op_names = sorted(set(ops) | set(base[: rng.randint(3, 8)]))
        facts += [{"kind": "Operator", "name": o} for o in op_names]
        idx_keys = []
        for ix in coll.indexes or []:
            keys = ix.get("keys") if isinstance(ix, dict) else None
            if keys:
                idx_keys.append({k: d for k, d in keys})
        if idx_keys:
            facts.append({"kind": "Indexes", "keys": idx_keys})
        for m in DECL_RE.finditer(rec["prefix"]):
            facts.append({"kind": "LocalVariable", "name": m.group(2), "declaration": m.group(1), "value_head": m.group(3).strip()})
        if kind == "full" and rng.random() < ec["similar_statement_ratio"]:
            cands = pools.candidates(rec, rec["snippet_id"])
            if cands:
                facts.append({"kind": "SimilarStatement", "text": rng.choice(cands)[1]})

    return {"context_kind": kind, "editor": editor, "facts": facts, "language": language}


def build_record(rec: dict, ctx: dict, cfg: dict) -> dict:
    ec = cfg["editor_context"]
    ed = ctx["editor"]
    out = dict(rec)
    out.update({
        "schema_version": SCHEMA_VERSION, "contrato": ec["contract"], "language": ctx["language"],
        "context_kind": ctx["context_kind"], "editor": ed, "facts": ctx["facts"],
        "known_names": ed["known_names"] if ed else [], "result_fields": ed["result_fields"] if ed else [],
        "recent_commands": ed["recent_commands"] if ed else [],
        "editor_text": rec["prefix"] + ec["cursor_marker"] + rec["suffix"],
        "target": rec["middle"],
    })
    return out


def percentile(sorted_vals: list[int], q: float) -> int:
    return sorted_vals[min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1))))]


def generate(fim: list[dict], snippets: list[dict], schemas, cfg: dict, tokenizer=None, seed=None, progress=False) -> list[dict]:
    ec = cfg["editor_context"]
    seed = cfg.get("seed", 20261008) if seed is None else seed
    pools = Pools(snippets, schemas)
    out: list[dict] = []
    for i, rec in enumerate(fim):
        r = build_record(rec, make_context(rec, cfg, pools, seed), cfg)
        # guarda do conjunto cego (so a API do guard)
        blind_guard.check_record(r)
        blind_guard.check_text(blind_guard.prompt_text({"kind": "autocomplete", "prefix": r["prefix"], "suffix": r["suffix"]}))
        out.append(r)
    # prompt renderizado + orcamento de tokens (lotes: tokenizacao e o custo dominante)
    rendered = []
    for r in out:
        prefix, suffix = rc.bound_window(r["prefix"], r["suffix"])
        before = rc.render_model_prefix(prefix, r["editor"], r["language"])
        rendered.append((before, suffix))
    if tokenizer is not None:
        step = 4000
        for s in range(0, len(out), step):
            chunk = rendered[s:s + step]
            bs = tokenizer.encode_batch([b for b, _ in chunk])
            as_ = tokenizer.encode_batch([a for _, a in chunk])
            for r, (before, after), b, a in zip(out[s:s + step], chunk, bs, as_):
                ids, info = rc.build_ids(tokenizer, before, after, ec["context_tokens"], (b, a))
                r["prompt_tokens"] = len(ids)
                r["prompt_truncated"] = info["truncated"]
                r["prompt_tokens_raw"] = info["prefix_tokens"] + info["suffix_tokens"] + 3
                r["prompt"] = tokenizer.decode(ids) if info["truncated"] else rc.qwen_fim_text(before, after)
            if progress:
                print(f"  tokens {min(s + step, len(out))}/{len(out)}", flush=True)
    else:
        for r, (before, after) in zip(out, rendered):
            r["prompt_tokens"] = None
            r["prompt_truncated"] = None
            r["prompt_tokens_raw"] = None
            r["prompt"] = rc.qwen_fim_text(before, after)
    for r in out:
        r["prompt_sha256"] = rc.prompt_sha256(r["prompt"])
        blind_guard.check_text(r["prompt"])
    return out


def stats(recs: list[dict]) -> dict:
    toks = sorted(r["prompt_tokens"] for r in recs if r.get("prompt_tokens") is not None)
    raw = sorted(r["prompt_tokens_raw"] for r in recs if r.get("prompt_tokens_raw") is not None)
    d: dict = {"count": len(recs), "context_kind": dict(Counter(r["context_kind"] for r in recs)),
               "splits": dict(Counter(r["split"] for r in recs))}
    if toks:
        d["prompt_tokens"] = {"p50": percentile(toks, .5), "p90": percentile(toks, .9), "p95": percentile(toks, .95),
                              "p99": percentile(toks, .99), "max": toks[-1], "mean": round(statistics.mean(toks), 1)}
        d["prompt_tokens_before_budget"] = {"p50": percentile(raw, .5), "p95": percentile(raw, .95), "max": raw[-1]}
        d["truncated"] = sum(1 for r in recs if r["prompt_truncated"])
        by = defaultdict(list)
        for r in recs:
            by[r["context_kind"]].append(r["prompt_tokens"])
        d["tokens_by_kind"] = {k: {"p50": percentile(sorted(v), .5), "p95": percentile(sorted(v), .95), "max": max(v)}
                               for k, v in by.items()}
    fact_kinds = Counter(f["kind"] for r in recs for f in r["facts"])
    d["facts"] = dict(fact_kinds)
    d["with_result_fields"] = sum(1 for r in recs if r["result_fields"])
    d["with_recent_commands"] = sum(1 for r in recs if r["recent_commands"])
    d["with_input_panel"] = sum(1 for r in recs if r["editor"] and r["editor"]["input_panel"])
    d["recent_commands_dist"] = dict(sorted(Counter(len(r["recent_commands"]) for r in recs).items()))
    d["cursor_marker_collisions"] = sum(1 for r in recs if r["editor_text"].count("\u258C") != 1)
    return d


def write_report(path: Path, man: dict, parity: dict | None = None, golden: dict | None = None) -> None:
    s = man["stats"]
    L = ["# Dataset v1 - contexto de editor (`editor-context-v1`)", "",
         f"Gerado por `scripts/data/gen_editor_context.py --ver {man['ver']}` (seed {man['seed']}), sobre `fim.jsonl` "
         f"(SHA-256 `{man['source_fim_sha256'][:16]}...`). Saida: `data/generated/{man['ver']}/fim_ctx.jsonl` "
         f"(SHA-256 `{man['sha256'][:16]}...`).", "",
         "## Contrato e renderer",
         "- Renderer Python: `scripts/data/render_contract.py` (porta de `AutocompleteContextBuilder.Build/ModelPrefix` + `QwenFimPromptBuilder`).",
         "- Cabecalho com CRLF fixo; embrulho do `ModelPrefix` com `\\n` literal; `*/` -> `* /`; `AVAILABLE COMMANDS` por linguagem; janelas 4096/1024 em unidades UTF-16.",
         "- Golden congelado da IDE copiado para `data/contracts/editor-context-v1.golden` (20 casos x context/modelPrefix.false/modelPrefix.true).", ""]
    if golden:
        L += [f"- Golden: {golden['cases']} casos, {golden['artifact_checks']} conferencias (texto, origem H/L dos terminadores, SHA-256 CRLF/LF, "
              f"prefix/suffix/dictionary); falhas: {len(golden['failures'])}.", ""]
    if parity:
        L += ["## Paridade com a v1 (`bakeoff/data/v1_*.jsonl`, renderer `compact/contracts.py` intocado)", "",
              "| conjunto | registros com contexto | texto identico (CRLF fixo) | identico se `newline` da v1 honrado | ids de token identicos |",
              "|---|---:|---:|---:|---:|"]
        for name, p in parity.items():
            L.append(f"| {name} | {p['with_context']} | {p['same_fixed_crlf']} | {p['same_honor_newline']} | {p.get('same_ids', '—')} |")
        L += ["", "A v1 gravou `newline: \"\\n\"` em parte dos registros (artefato do gerador antigo); o contrato congelado da IDE e CRLF fixo, "
              "entao esses registros divergem so nos separadores do cabecalho.", ""]
    L += ["## Distribuicao", "", f"- Registros: {s['count']} · splits {s['splits']}",
          f"- `context_kind`: {s['context_kind']}",
          f"- com result_fields: {s['with_result_fields']} · com recent_commands: {s['with_recent_commands']} "
          f"(por quantidade {s['recent_commands_dist']}) · com input_panel: {s['with_input_panel']}",
          f"- fatos: {s['facts']}", ""]
    if "prompt_tokens" in s:
        t, b = s["prompt_tokens"], s["prompt_tokens_before_budget"]
        L += ["## Orcamento de tokens (tokenizer Qwen2.5-Coder, contexto 2048)", "",
              "| medida | p50 | p95 | max |", "|---|---:|---:|---:|",
              f"| tokens do prompt (pos-orcamento) | {t['p50']} | {t['p95']} | {t['max']} |",
              f"| tokens antes do orcamento | {b['p50']} | {b['p95']} | {b['max']} |", "",
              f"- truncados: {s['truncated']} de {s['count']}; por tipo de contexto: {s['tokens_by_kind']}", ""]
    L += ["## Limites",
          "- `result_fields`/`recent_commands` sao simulados a partir do schema e dos snippets (nao ha execucao real do explorer da IDE).",
          "- Os fixtures (dumps `mongodump`) nao sao lidos aqui: o resultado simulado usa os caminhos do schema (uniao dos campos opcionais).",
          "- `facts` so alimentam contratos v2 (`kapi-context-v2`); o prompt v1 usa so `editor`.",
          "- Os snippets sao curtos (prefixo <= ~950 caracteres): o orcamento de 2048 tokens nunca e atingido aqui; a IDE real pode enviar ate 4096 "
          "caracteres de prefixo. O truncamento esta coberto por testes sinteticos, mas a distribuicao de documentos longos nao existe no dataset.",
          "- Paridade byte a byte contra a IDE real depende do `kapilab contract render/tokenize/diff` (PR 1): entradas em `data/contracts/parity_inputs.jsonl`.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8", newline="\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(CONFIG_PATH))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--ver")
    ap.add_argument("--fim")
    ap.add_argument("--snippets")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-tokenizer", action="store_true")
    ap.add_argument("--report", action="store_true", help="grava reports/datasets/<ver>-editor-context-report.md")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.seed is not None:
        cfg["seed"] = a.seed
    ver = a.ver or cfg.get("version", "v1")
    gdir = ROOT / cfg.get("out_dir", "data/generated") / ver
    fim_path = Path(a.fim) if a.fim else gdir / "fim.jsonl"
    blind_guard.check_path(fim_path)
    fim = read_jsonl(fim_path)
    if a.limit:
        fim = fim[: a.limit]
    snippets = read_jsonl(Path(a.snippets) if a.snippets else gdir / "snippets.jsonl")
    schemas = load_schemas(ROOT / cfg.get("schemas_dir", "data/schemas"))
    tok = None
    if not a.no_tokenizer:
        tok = rc.Tokenizer(ROOT / cfg["editor_context"]["tokenizer"])
    t0 = time.time()
    recs = generate(fim, snippets, schemas, cfg, tok, progress=True)
    out = Path(a.out) if a.out else gdir / "fim_ctx.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            line = json.dumps(r, ensure_ascii=False, separators=(",", ":"))
            f.write(line + "\n")
            h.update(line.encode("utf-8"))
    src_sha = hashlib.sha256(fim_path.read_bytes()).hexdigest()
    man = {"ver": ver, "seed": cfg["seed"], "contract": cfg["editor_context"]["contract"], "source_fim": str(fim_path.relative_to(ROOT)),
           "source_fim_sha256": src_sha, "sha256": h.hexdigest(), "tokenizer": None if tok is None else str(tok.path.relative_to(ROOT)),
           "config": cfg["editor_context"], "seconds": round(time.time() - t0, 1), "stats": stats(recs)}
    (out.parent / "fim_ctx.manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    if a.report:
        parity = golden = None
        pp = ROOT / "data" / "contracts" / "parity_report.json"
        if pp.exists():
            pr = json.loads(pp.read_text(encoding="utf-8"))
            parity, golden = pr.get("v1"), pr.get("golden")
        write_report(ROOT / "reports" / "datasets" / f"{ver}-editor-context-report.md", man, parity, golden)
    print(json.dumps(man["stats"], ensure_ascii=False))


if __name__ == "__main__":
    main()
