#!/usr/bin/env python3
"""Prepare bounded, reproducible FIM samples; no model inference or MongoDB required."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from collections import Counter

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "data"))
import gen_snippets as gs
import gen_fim as gf
import gen_editor_context as gec
import render_contract as rc


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


class MiddleCounter:
    method = "qwen"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.path = str(tokenizer.path)

    def count(self, text):
        return len(self.tokenizer.encode(text))


def generate_role(cfg, schemas, tokenizer, role, count, seed, initial=512, maximum=8192):
    """Double a bounded snippet pool until enough unique cuts survive."""
    cfg = copy.deepcopy(cfg)
    cfg["seed"] = seed
    cfg["fim"].pop("eval_counts", None)
    cfg["fim"]["decontam_prefilter"] = False
    pool = [s for s in schemas if s.split_role == role and s.raw_role != "hom"]
    if role == "train":
        pool = [s for s in pool if not s.holdout and s.domain not in cfg.get("holdout_domains", [])]
    if not pool:
        raise ValueError(f"No schemas with split_role={role}")
    size = initial
    while True:
        snippets = gs.generate(cfg, pool, size)
        snippets = [s for s in snippets if (s["split"] == "train") == (role == "train")]
        fim = gf.generate_fim(snippets, cfg, count * 2, tokenizer=MiddleCounter(tokenizer))
        rows = gec.generate(fim, snippets, pool, cfg, tokenizer)
        unique = {}
        for row in sorted(rows, key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest()):
            rendered = rc.render_prompt(row, tokenizer, cfg["editor_context"]["context_tokens"])
            ids, _ = rc.build_ids(tokenizer, rendered["before"], rendered["after"], cfg["editor_context"]["context_tokens"])
            row["prompt_ids"] = ids
            row["prompt"] = rendered["prompt"]
            row["prompt_text"] = row["prompt"]
            row["reference"] = row["target"]
            row["prompt_sha256"] = rc.prompt_sha256(row["prompt"])
            row["prompt_tokens"] = len(ids)
            row["validation"] = {"contract": "ok", "token_budget": "ok", "exec": "not_measured", "syntax": "not_measured"}
            unique.setdefault(row["prompt_sha256"], row)
        if len(unique) >= count:
            return list(unique.values())[:count], {"snippet_budget": size, "generated_snippets": len(snippets), "unique_prompts": len(unique)}
        if size >= maximum:
            raise ValueError(f"Insufficient unique {role} prompts: {len(unique)}/{count}; bounded pool exhausted ({maximum})")
        size = min(size * 2, maximum)


def validate(calibration, evaluation, max_tokens):
    for key in ("group_id", "split_key", "schema_id", "prompt_sha256"):
        if {r[key] for r in calibration} & {r[key] for r in evaluation}:
            raise ValueError(f"Calibration/evaluation overlap: {key}")
    for role, rows in (("train", calibration), ("eval", evaluation)):
        for r in rows:
            if r["split_role"] != role or r["contrato"] != rc.CONTRACT_ID:
                raise ValueError("Invalid schema role or context contract")
            if not 3 <= len(r["prompt_ids"]) <= max_tokens:
                raise ValueError("Prompt token budget exceeded")
            if r["target"] != r["middle"] or not r["target"].strip():
                raise ValueError("Invalid FIM target")
            if rc.prompt_sha256(r["prompt"]) != r["prompt_sha256"]:
                raise ValueError("Prompt hash mismatch")


def prepare(config_path, tokenizer_path, out):
    config_path, tokenizer_path, out = Path(config_path), Path(tokenizer_path), Path(out)
    if tokenizer_path.is_dir():
        tokenizer_path /= "tokenizer.json"
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    opts = settings.get("samples", {})
    dataset_path = resolve(settings.get("dataset_config", "configs/dataset.yaml"))
    cfg = gs.load_config(dataset_path)
    seed = int(opts.get("seed", 20261026))
    ncal, neval = int(opts.get("calibration_count", 256)), int(opts.get("evaluation_count", 200))
    initial, maximum = int(opts.get("initial_snippets", 512)), int(opts.get("max_snippets", 8192))
    max_tokens = int(opts.get("max_prompt_tokens", 1024))
    if not (1 <= ncal <= 4096 and 1 <= neval <= 4096 and 1 <= initial <= maximum <= 32768 and 3 <= max_tokens <= 1024):
        raise ValueError("Sample counts, bounded snippet pool, or token budget outside supported limits")
    cfg["editor_context"]["context_tokens"] = max_tokens
    cfg["editor_context"]["contract"] = rc.CONTRACT_ID
    tokenizer = rc.Tokenizer(tokenizer_path)
    schemas_dir = resolve(settings.get("schemas_dir", cfg["schemas_dir"]))
    schemas = gs.load_schemas(schemas_dir)
    calibration, cal_info = generate_role(cfg, schemas, tokenizer, "train", ncal, seed, initial, maximum)
    evaluation, eval_info = generate_role(cfg, schemas, tokenizer, "eval", neval, seed + 1, initial, maximum)
    validate(calibration, evaluation, max_tokens)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in (("calibration", calibration), ("evaluation", evaluation)):
        path = out / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8", newline="\n")
        files[name] = {"file": path.name, "sha256": sha(path), "count": len(rows), "context_kinds": dict(Counter(r["context_kind"] for r in rows)), "cut_kinds": dict(Counter(r["cut_kind"] for r in rows))}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    git_status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    manifest = {"schema": "snapdragon-fim-samples-v1", "contract": rc.CONTRACT_ID, "seed": seed, "evaluation_seed": seed + 1,
                "max_prompt_tokens": max_tokens, "max_new_tokens": 32, "generation": "greedy", "files": files,
                "config_sha256": sha(config_path), "dataset_config_sha256": sha(dataset_path), "tokenizer_sha256": sha(tokenizer_path),
                "generator_commit": git.stdout.strip() if git.returncode == 0 else None,
                "generator_dirty": bool(git_status.stdout.strip()) if git_status.returncode == 0 else None,
                "generator_sha256": {p.name: sha(p) for p in (Path(__file__), Path(gs.__file__), Path(gf.__file__), Path(gec.__file__), Path(rc.__file__))},
                "schemas_sha256": {p.relative_to(schemas_dir).as_posix(): sha(p) for p in sorted(schemas_dir.rglob("*.json"))},
                "models": settings.get("models", {}), "generation_info": {"calibration": cal_info, "evaluation": eval_info},
                "validation": {"disjoint_groups": True, "disjoint_schemas": True, "tokenizer": "exact", "exec": "not_measured", "syntax": "not_measured", "integration": "not_measured"}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True, help="HF snapshot directory or tokenizer.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    manifest = prepare(args.config, args.tokenizer, args.out)
    print(json.dumps({"files": manifest["files"], "contract": manifest["contract"]}))


if __name__ == "__main__":
    main()
