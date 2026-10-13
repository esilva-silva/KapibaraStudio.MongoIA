#!/usr/bin/env python3
"""Baseline zero-shot FIM no `val` do dataset v1 (fase-02-professor.md §3.2) — somente FIM/autocomplete (sem chat).

  python -I scripts/teacher/eval_teacher.py --model BaseRepos/Qwen2.5-Coder-7B --split val --contract fim-raw --n 1000 [--load auto|bf16|nf4]

Fases:
  gen    (venv ROCm do bake-off, GPU)  : gera greedy, metricas de texto, sintaxe (node) e mongo_valid estatico (mongo_validate da v1).
  score  (qualquer python com pymongo) : exec-valid + campos inexistentes + tipos, via scripts/data/chat_validate (Executor somente
                                         leitura em kapi_fx_*, maxTimeMS) sobre prefix+predicao+suffix; calibra com a referencia.
  all    : gen e depois score num subprocesso com env/.venv (que tem pymongo; o venv ROCm nao tem).

Contratos: `fim-raw` (padrao; prompt = QwenFimPromptBuilder sobre prefix/suffix do registro, sem cabecalho de contexto, pois o
fim.jsonl da v1 nao traz `context`) e `editor-context-v1`: le `data/generated/v1/fim_ctx.jsonl` (gen_editor_context.py; campo
`editor` de cada linha) e renderiza o cabecalho real da IDE com `scripts/data/render_contract.py` (CRLF fixo, ModelPrefix, janelas);
linhas `context_kind=none` ficam sem cabecalho (8%). Sem fim_ctx.jsonl cai na aproximacao antiga (LANGUAGE + KNOWN NAMES=colecao).
`--ids-from <sample_ids.json>` restringe a amostra aos ids de outra execucao (comparacao pareada; `--n` sorteia dentre eles).
kapi-context-v2/local-agent-fim-v0 nao se aplicam a esta etapa (chat/SFT).
Saida: Repos/KapiTeacher-Mongo-7B/reports/baseline_base/<model>/{summary.json,predictions.jsonl,report.md,gen_meta.json,sample_ids.json}
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (ENV_VENV_PY, ROOT, TEACHER_REPO, auto_load, gpu_lock, library_versions, model_fingerprint, read_jsonl,  # noqa: E402
                     sha256_file, sha256_text, write_json)

DATASET = ROOT / "data" / "generated" / "v1" / "fim.jsonl"
DATASET_MANIFEST = ROOT / "data" / "generated" / "v1" / "fim.manifest.json"
DATASET_CTX = ROOT / "data" / "generated" / "v1" / "fim_ctx.jsonl"
DATASET_CTX_MANIFEST = ROOT / "data" / "generated" / "v1" / "fim_ctx.manifest.json"
sys.path.insert(0, str(ROOT / "scripts" / "data"))
MAX_NEW_TOKENS = 96      # §3.5: greedy ate 96 tokens (autocomplete)
IDE_MAX_NEW_TOKENS = 32  # AutocompleteSettings.MaximumCompletionTokens (metricas @32 como no bake-off)
CONTEXT_TOKENS = 2048
CONTRACTS = ("fim-raw", "editor-context-v1")
LANGUAGE_BY_DIALECT = {"console": "Mongo Console JavaScript", "mongosh": "JavaScript (mongosh)", "json": "json"}


def dataset_paths(contract_name: str) -> tuple[Path, Path]:
    """(jsonl, manifest) do contrato: editor-context-v1 usa fim_ctx.jsonl quando existe."""
    if contract_name == "editor-context-v1" and DATASET_CTX.exists():
        return DATASET_CTX, DATASET_CTX_MANIFEST
    return DATASET, DATASET_MANIFEST


def read_rows(contract_name: str) -> list[dict]:
    path, _ = dataset_paths(contract_name)
    rows = read_jsonl(path)
    for r in rows:  # fim_ctx.jsonl grava a resposta em `target` (= middle)
        r.setdefault("middle", r.get("target"))
    return rows


# -- amostragem -------------------------------------------------------------------------------------------------------------------
def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """Alocacao proporcional por (cut_kind, domain) com maiores restos; dentro do estrato, sorteio com seed. Retorna ate n linhas."""
    if n >= len(rows):
        return sorted(rows, key=lambda r: r["id"])
    strata: dict[tuple, list[dict]] = collections.defaultdict(list)
    for row in rows:
        strata[(row["cut_kind"], row["domain"])].append(row)
    keys = sorted(strata)
    quota = {k: n * len(strata[k]) / len(rows) for k in keys}
    take = {k: int(math.floor(quota[k])) for k in keys}
    for k in sorted(keys, key=lambda k: (-(quota[k] - take[k]), k))[: n - sum(take.values())]:
        take[k] += 1
    rng = random.Random(seed)
    chosen: list[dict] = []
    for k in keys:
        pool = sorted(strata[k], key=lambda r: r["id"])
        rng.shuffle(pool)
        chosen += pool[: take[k]]
    return sorted(chosen, key=lambda r: r["id"])


def load_sample(split: str, n: int, seed: int, contract_name: str = "fim-raw", ids_from: str | None = None) -> tuple[list[dict], dict]:
    rows = [r for r in read_rows(contract_name) if r["split"] == split]
    if ids_from:
        wanted = set(json.loads(Path(ids_from).read_text(encoding="utf-8"))["ids"])
        rows = [r for r in rows if r["id"] in wanted]
    sample = stratified_sample(rows, n, seed)
    info = {"split": split, "split_size": len(rows), "n": len(sample), "seed": seed, "ids_from": ids_from, "stratify_by": ["cut_kind", "domain"],
            "cut_kind": dict(collections.Counter(r["cut_kind"] for r in sample)),
            "domain": dict(collections.Counter(r["domain"] for r in sample)),
            "sample_ids_sha256": sha256_text("\n".join(r["id"] for r in sample))}
    return sample, info


# -- fase gen ---------------------------------------------------------------------------------------------------------------------
def build_ids(contract, row: dict, contract_name: str) -> list[int]:
    from compact.contracts import PREFIX_WINDOW, SUFFIX_WINDOW, model_prefix

    prefix, suffix = row["prefix"][-PREFIX_WINDOW:], row["suffix"][:SUFFIX_WINDOW]
    if contract_name == "editor-context-v1" and "editor" in row:
        import render_contract as rc  # scripts/data: renderer do contrato (cabecalho real; None = sem contexto)

        prefix, suffix = rc.bound_window(row["prefix"], row["suffix"])
        prefix = rc.render_model_prefix(prefix, row["editor"], row.get("language") or rc.editor_language(row))
    elif contract_name == "editor-context-v1":
        context = {"language": LANGUAGE_BY_DIALECT.get(row["dialect"]), "newline": "\r\n" if row.get("crlf") else "\n",
                   "known_names": [row["collection"]]}
        prefix = model_prefix(prefix, context, contract.deepseek_context)
    return contract.build(prefix, suffix, CONTEXT_TOKENS)


def norm_ws(text: str) -> str:
    import re

    return re.sub(r"\s+", " ", text.replace("\r\n", "\n")).strip()


def score_text(pred: str, ref: str, tokenizer) -> dict:
    from compact import metrics

    ends = [end for _, end in tokenizer(ref, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]]
    _, similarity = metrics.edit_similarity(pred, ref)
    _, tab_chars = metrics.tab_acceptance(pred, ref)
    return {"exact_match": metrics.exact_match(pred, ref), "normalized_match": norm_ws(pred) == norm_ws(ref),
            "prefix_match": metrics.prefix_match(pred, ref), "edit_similarity": round(similarity, 4),
            "chars_accepted": metrics.common_prefix_len(pred, ref), "tab_chars": tab_chars,
            "accepted_prefix_tokens": metrics.accepted_prefix_tokens(pred, ref, ends), "reference_tokens": len(ends),
            "reference_chars": len(ref), "empty": not pred.strip()}


def phase_gen(args, out_dir: Path) -> None:
    from compact import syntax
    from compact.contracts import contract_for, ide_rejects, postprocess
    from compact.mongo_validate import validate

    from _common import Generator, load_hf

    model_dir = Path(args.model).resolve()
    sample, sample_info = load_sample(args.split, args.n, args.seed, args.contract, args.ids_from)
    write_json(out_dir / "sample_ids.json", {**sample_info, "ids": [r["id"] for r in sample]})
    predictions = out_dir / "predictions.jsonl"
    done: dict[str, dict] = {}
    if args.resume and predictions.exists():
        done = {r["id"]: r for r in read_jsonl(predictions)}
    load = args.load if args.load != "auto" else auto_load(model_dir)
    wall0 = time.time()
    with gpu_lock(f"eval_teacher:{model_dir.name}"):
        model, tokenizer, device, gpu, load_seconds = load_hf(model_dir, load)
        import torch

        contract = contract_for(tokenizer)
        gen = Generator(model, tokenizer, device)
        for row in sample[:3]:  # aquecimento (igual ao bake-off)
            gen.generate(build_ids(contract, row, args.contract), 4, contract.stops, "")
        gen_started = time.time()
        with predictions.open("a", encoding="utf-8", newline="\n") as handle:
            for index, row in enumerate(sample):
                if row["id"] in done:
                    continue
                ids = build_ids(contract, row, args.contract)
                result = gen.generate(ids, MAX_NEW_TOKENS, contract.stops, row["suffix"][:1024])
                tokens = [t for t in result["tokens"] if t not in contract.stops]
                raw = gen.decode(tokens)
                pred = postprocess(raw, row)
                pred32 = postprocess(gen.decode(tokens[:IDE_MAX_NEW_TOKENS]), row)
                reference = row["middle"]
                record = {"id": row["id"], "split": row["split"], "cut_kind": row["cut_kind"], "domain": row["domain"],
                          "dialect": row["dialect"], "schema_id": row["schema_id"], "collection": row["collection"],
                          "cursor_role": row.get("cursor_role"), "prompt_tokens": len(ids), "generated_tokens": len(result["tokens"]),
                          "stop": result["stop"], "stop_token": result["tokens"][-1], "ttft_ms": round(result["ttft"] * 1000, 1),
                          "raw": raw, "prediction": pred, "prediction_ide32": pred32, "reference": reference,
                          "ide_rejected": ide_rejects(pred, result["stop"] in ("eos", "suffix")),
                          "metrics": score_text(pred, reference, tokenizer), "metrics_ide32": score_text(pred32, reference, tokenizer)}
                done[row["id"]] = record
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                if (index + 1) % 50 == 0:
                    sub = list(done.values())
                    print(f"[{index + 1}/{len(sample)}] {time.time() - gen_started:.0f}s "
                          f"EM={sum(r['metrics']['exact_match'] for r in sub) / len(sub):.3f} "
                          f"chars={sum(r['metrics']['chars_accepted'] for r in sub) / len(sub):.2f}", flush=True)
        gen_seconds = time.time() - gen_started
        memory = {"vram_max_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
                  "vram_max_reserved_gb": round(torch.cuda.max_memory_reserved(device) / 2**30, 2)}
        gpu_report = {k: gpu.get(k) for k in ("device_name", "gcn_arch", "total_vram_gb", "torch", "hip", "vram_limit")}
        stops = sorted(contract.stops)
        contract_name = contract.name
        del model
    rows = [done[r["id"]] for r in sample if r["id"] in done]
    # sintaxe e mongo_valid estaticos do documento reconstruido (node + mongo_validate), como attach_validity do bake-off
    by_id = {r["id"]: r for r in sample}
    jobs = []
    for r in rows:
        src = by_id[r["id"]]
        for key, text in (("pred", r["prediction"]), ("pred32", r["prediction_ide32"]), ("ref", r["reference"])):
            jobs.append((f"{r['id']}|{key}", (src["prefix"] + text + src["suffix"]).replace("\r\n", "\n"), syntax.kind_for(src["dialect"])))
    syntax_results = syntax.check_many(jobs)
    texts = {j[0]: j[1] for j in jobs}
    for r in rows:
        for key, target in (("pred", "metrics"), ("pred32", "metrics_ide32")):
            job = f"{r['id']}|{key}"
            ok = syntax_results[job][0]
            check = validate(texts[job], r["dialect"]) if ok else None
            r[target]["syntax_valid"] = ok
            r[target]["mongo_valid"] = check.valid if check and check.validatable else (False if not ok else None)
        job = f"{r['id']}|ref"
        ok = syntax_results[job][0]
        check = validate(texts[job], r["dialect"]) if ok else None
        r["reference_checks"] = {"syntax_valid": ok, "mongo_valid": check.valid if check and check.validatable else (False if not ok else None)}
    with predictions.open("w", encoding="utf-8", newline="\n") as handle:
        for r in rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_json(out_dir / "gen_meta.json", {
        "model_path": str(model_dir), "model": model_fingerprint(model_dir), "load": load, "load_seconds": load_seconds,
        "contract": args.contract, "model_contract": contract_name, "stop_ids": stops, "max_new_tokens": MAX_NEW_TOKENS,
        "ide_max_new_tokens": IDE_MAX_NEW_TOKENS, "context_tokens": CONTEXT_TOKENS, "decoding": "greedy (argmax), padding do vocab mascarado",
        "seed": args.seed, "dataset": {"path": str(dataset_paths(args.contract)[0].relative_to(ROOT)),
                                       "sha256": sha256_file(dataset_paths(args.contract)[0]),
                                       "manifest_sha256": sha256_file(dataset_paths(args.contract)[1]), **sample_info},
        "libraries": library_versions(), "gpu": gpu_report, "memory": memory,
        "gpu_seconds": round(time.time() - wall0, 1), "generation_seconds": round(gen_seconds, 1),
        "examples_generated": len(rows), "finished": time.strftime("%FT%T"),
    })


# -- fase score (exec) -------------------------------------------------------------------------------------------------------------
def phase_score(args, out_dir: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts" / "data"))
    import chat_validate as cv  # noqa: E402
    from gen_snippets import CONFIG_PATH, load_config, load_schemas  # noqa: E402

    cfg = load_config(CONFIG_PATH)
    schemas = {s.full_id: s for s in load_schemas(ROOT / cfg.get("schemas_dir", "data/schemas"))}
    executor = cv.Executor(cfg)
    node = cv.NodeChecker()
    sample = {r["id"]: r for r in read_rows(args.contract) if r["split"] == args.split}
    rows = read_jsonl(out_dir / "predictions.jsonl")
    started = time.time()

    def check(src: dict, text: str) -> dict:
        sch = schemas.get(src["schema_id"])
        if sch is None:
            return {"stage": "schema", "error": "schema desconhecido"}
        coll = next((c for c in sch.colls if c.name == src["collection"]), None)
        if coll is None:
            return {"stage": "schema", "error": "colecao desconhecida"}
        doc = (src["prefix"] + text + src["suffix"]).replace("\r\n", "\n")
        res = cv.validate_text(doc, sch, coll, executor, node.check(doc), db=sch.db_name)
        return {"stage": res["stage"], "error": res.get("error"), "n_docs": res.get("n_docs"), "ms": res.get("ms")}

    try:
        for r in rows:
            src = sample[r["id"]]
            for key, text in (("pred", r["prediction"]), ("pred32", r["prediction_ide32"]), ("ref", r["reference"])):
                v = check(src, text)
                v["exec_valid"] = v["stage"] == "ok"
                v["field_hallucination"] = v["stage"] == "fields" and str(v.get("error", "")).startswith("campo/estrutura invalido")
                target = {"pred": "exec", "pred32": "exec_ide32", "ref": "exec_reference"}[key]
                r[target] = v
    finally:
        node.close()
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for r in rows:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_json(out_dir / "score_meta.json", {"exec_seconds": round(time.time() - started, 1), "mongo_uri": "mongodb://127.0.0.1:27017",
                                             "write_commands_seen_by_guard": executor.violations, "max_time_ms": executor.max_ms,
                                             "result_cap": executor.cap, "libraries": library_versions(),
                                             "validator": "scripts/data/chat_validate.validate_text (syntax>api>fields>type>exec)"})
    print(f"score ok: {len(rows)} linhas em {time.time() - started:.0f}s, violacoes do guard: {executor.violations}")


# -- agregacao / relatorio --------------------------------------------------------------------------------------------------------
def mean(values) -> float | None:
    values = [float(v) for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def aggregate(rows: list[dict], key: str = "metrics", exec_key: str = "exec") -> dict:
    ex = [r.get(exec_key) for r in rows if r.get(exec_key)]
    # exec_valid estrito: documento reconstruido executa E a predicao nao e vazia (vazio e "valido" so por omitir o trecho)
    strict = [bool(r[exec_key]["exec_valid"]) and not r[key]["empty"] for r in rows if r.get(exec_key)]
    return {
        "n": len(rows),
        "exact_match": mean(r[key]["exact_match"] for r in rows), "normalized_match": mean(r[key]["normalized_match"] for r in rows),
        "prefix_match": mean(r[key]["prefix_match"] for r in rows), "edit_similarity": mean(r[key]["edit_similarity"] for r in rows),
        "chars_accepted": mean(r[key]["chars_accepted"] for r in rows), "tab_chars": mean(r[key]["tab_chars"] for r in rows),
        "accepted_prefix_tokens": mean(r[key]["accepted_prefix_tokens"] for r in rows),
        "syntax_valid": mean(r[key].get("syntax_valid") for r in rows), "mongo_valid": mean(r[key].get("mongo_valid") for r in rows),
        "exec_valid": mean(strict) if strict else None,
        "exec_valid_incl_empty": mean(e["exec_valid"] for e in ex) if ex else None,
        "nonexistent_fields_rate": mean(e["field_hallucination"] for e in ex) if ex else None,
        "empty_rate": mean(r[key]["empty"] for r in rows),
        "eos_or_suffix_stop_rate": mean(r["stop"] != "max" for r in rows) if key == "metrics" else None,
        "mean_generated_tokens": mean(r["generated_tokens"] for r in rows) if key == "metrics" else None,
    }


def group_by(rows: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        groups[str(r.get(key))].append(r)
    return {k: aggregate(v) for k, v in sorted(groups.items())}


def pct(value) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def phase_report(args, out_dir: Path) -> dict:
    rows = read_jsonl(out_dir / "predictions.jsonl")
    meta = json.loads((out_dir / "gen_meta.json").read_text(encoding="utf-8"))
    score_meta = json.loads((out_dir / "score_meta.json").read_text(encoding="utf-8")) if (out_dir / "score_meta.json").exists() else None
    ref_rows = [{"metrics": {**r["reference_checks"], "exact_match": 1, "normalized_match": 1, "prefix_match": 1, "edit_similarity": 1,
                             "chars_accepted": 0, "tab_chars": 0, "accepted_prefix_tokens": 0, "empty": 0}, "exec_reference": r.get("exec_reference"),
                 "stop": "eos", "generated_tokens": 0} for r in rows]
    reference = {"syntax_valid": mean(r["metrics"]["syntax_valid"] for r in ref_rows), "mongo_valid": mean(r["metrics"]["mongo_valid"] for r in ref_rows)}
    if all(r.get("exec_reference") for r in rows):
        reference["exec_valid"] = mean(r["exec_reference"]["exec_valid"] for r in rows)
        reference["nonexistent_fields_rate"] = mean(r["exec_reference"]["field_hallucination"] for r in rows)
    stages = collections.Counter(r["exec"]["stage"] for r in rows if r.get("exec"))
    overall = aggregate(rows)
    latency = [r["ttft_ms"] for r in rows]
    summary = {
        "run_name": out_dir.name, "kind": "baseline_zero_shot_fim", "spec": "docs/fase-02-professor.md §3.2", "split": args.split,
        "contract": meta["contract"], "n": len(rows), "model": {k: meta["model"][k] for k in ("model_sha256", "repo_id", "revision", "source")},
        "model_path": meta["model_path"], "load": meta["load"], "dataset": meta["dataset"], "seed": meta["seed"],
        "decoding": meta["decoding"], "max_new_tokens": meta["max_new_tokens"], "context_tokens": meta["context_tokens"],
        "headline": overall, "headline_ide32": aggregate(rows, "metrics_ide32", "exec_ide32"), "reference_ceiling": reference,
        "exec_stage_distribution": dict(stages), "by_cut_kind": group_by(rows, "cut_kind"), "by_domain": group_by(rows, "domain"),
        "stop_reasons": dict(collections.Counter(r["stop"] for r in rows)),
        "ide_rejected_rate": mean(r["ide_rejected"] for r in rows), "ttft_ms_mean": mean(latency),
        "gpu_seconds": meta["gpu_seconds"], "generation_seconds": meta["generation_seconds"], "memory": meta["memory"], "gpu": meta["gpu"],
        "libraries": meta["libraries"], "exec_score": score_meta, "generated_at": time.strftime("%FT%T"),
        "metric_definitions": {
            "exact_match": "pred.strip() == middle.strip()", "normalized_match": "igualdade com espacos normalizados (FIM accuracy do bake-off)",
            "chars_accepted": "tamanho do prefixo comum (caracteres) entre predicao e middle",
            "accepted_prefix_tokens": "tokens iniciais da referencia cobertos pelo prefixo comum (APT)",
            "syntax_valid": "node vm.Script sobre prefix+pred+suffix", "mongo_valid": "mongo_validate (AST, estatico) sobre o documento reconstruido",
            "exec_valid": "chat_validate.validate_text stage=ok (sintaxe, API do Console, campos, tipos, execucao somente-leitura em kapi_fx_*) E predicao nao vazia; exec_valid_incl_empty conta vazio como valido",
            "nonexistent_fields_rate": "stage=fields com 'campo/estrutura invalido' (campo ausente do schema) sobre todos os exemplos",
            "reference_ceiling": "mesmas checagens aplicadas ao middle de referencia (teto/calibracao do validador)"},
    }
    write_json(out_dir / "summary.json", summary)
    h, h32 = summary["headline"], summary["headline_ide32"]
    lines = [f"# Baseline zero-shot FIM `{out_dir.name}`", "",
             f"- Spec: `docs/fase-02-professor.md` §3.2 · split `{args.split}` · n={len(rows)} (de {meta['dataset']['split_size']}), estratificado por cut_kind x dominio, seed {meta['seed']}",
             f"- Modelo: `{meta['model_path']}` · carga `{meta['load']}` · SHA-256 do modelo `{meta['model']['model_sha256'][:16]}…` (revisao `{meta['model'].get('revision')}`)",
             f"- Dataset: `{meta['dataset']['path']}` SHA-256 `{meta['dataset']['sha256'][:16]}…` · amostra `{meta['dataset']['sample_ids_sha256'][:16]}…`",
             f"- Contrato `{meta['contract']}` · greedy · max_new_tokens {meta['max_new_tokens']} · contexto {meta['context_tokens']} tokens",
             f"- GPU: {meta['gpu'].get('device_name')} · {meta['gpu_seconds']} s totais ({meta['generation_seconds']} s de geracao) · VRAM pico {meta['memory']['vram_max_allocated_gb']} GB",
             f"- Libs: torch {meta['libraries'].get('torch')} · transformers {meta['libraries'].get('transformers')} · bitsandbytes {meta['libraries'].get('bitsandbytes')} · pymongo {(score_meta or {}).get('libraries', {}).get('pymongo', '—')}",
             "", "## Principais", "", "| metrica | 96 tokens | @32 (IDE) | teto da referencia |", "|---|---:|---:|---:|"]
    for key, label in (("exact_match", "Exact match"), ("normalized_match", "Normalized match (FIM acc.)"), ("prefix_match", "Prefix match"),
                       ("syntax_valid", "Syntax valid"), ("mongo_valid", "Mongo valid (estatico)"), ("exec_valid", "Exec-valid"),
                       ("nonexistent_fields_rate", "Campos inexistentes")):
        lines.append(f"| {label} | {pct(h[key])} | {pct(h32.get(key))} | {pct(reference.get(key)) if key in reference else '—'} |")
    lines.append(f"| Chars aceitos (media) | {h['chars_accepted']} | {h32['chars_accepted']} | — |")
    lines.append(f"| APT (tokens) | {h['accepted_prefix_tokens']} | {h32['accepted_prefix_tokens']} | — |")
    lines += ["", f"Distribuicao do estagio de validacao (pred): {dict(stages)}", f"Paradas: {summary['stop_reasons']}", ""]
    for title, key in (("Por cut_kind", "by_cut_kind"), ("Por dominio", "by_domain")):
        lines += [f"## {title}", "", "| grupo | n | EM | FIM acc | chars | syntax | mongo | exec | campos inex. |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, a in summary[key].items():
            lines.append(f"| {name} | {a['n']} | {pct(a['exact_match'])} | {pct(a['normalized_match'])} | {a['chars_accepted']} | {pct(a['syntax_valid'])} | "
                         f"{pct(a['mongo_valid'])} | {pct(a['exec_valid'])} | {pct(a['nonexistent_fields_rate'])} |")
        lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--contract", choices=CONTRACTS, default="fim-raw")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--load", choices=["auto", "bf16", "nf4"], default="auto")
    parser.add_argument("--phase", choices=["gen", "score", "report", "all"], default="all")
    parser.add_argument("--name", help="nome da pasta de saida (padrao: nome do modelo)")
    parser.add_argument("--out-root", default=str(TEACHER_REPO / "reports" / "baseline_base"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ids-from", help="sample_ids.json de outra execucao: amostra so dentre esses ids")
    args = parser.parse_args()
    if args.split == "test":
        print("AVISO: split test usado so para leitura final; confirme que e intencional.", file=sys.stderr)
    out_dir = Path(args.out_root) / (args.name or Path(args.model).name)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.phase in ("gen", "all"):
        phase_gen(args, out_dir)
    if args.phase == "all":
        command = [str(ENV_VENV_PY), "-I", str(Path(__file__).resolve()), "--phase", "score", "--model", args.model, "--split", args.split,
                   "--contract", args.contract, "--n", str(args.n), "--seed", str(args.seed), "--out-root", args.out_root]
        if args.ids_from:
            command += ["--ids-from", args.ids_from]
        if args.name:
            command += ["--name", args.name]
        subprocess.run(command, check=True)
    if args.phase == "score":
        phase_score(args, out_dir)
    if args.phase in ("report", "all"):
        summary = phase_report(args, out_dir)
        print(json.dumps(summary["headline"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
