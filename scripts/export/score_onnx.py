#!/usr/bin/env python3
"""Converte a saida do runtime ONNX (onnx_run.py --mode quality) em predictions.jsonl no formato do eval_teacher e pontua com o MESMO scorer
(metricas de texto, sintaxe node, mongo_valid, exec-valid somente leitura em kapi_fx_*, campos inexistentes) -> summary.json.

  python -I scripts/export/score_onnx.py --tier t0 --variant dml-fp16 --run reports/export/work/t0_dml-fp16_quality.json

Saida: reports/export/eval/<tier>/<variante>/{predictions.jsonl,summary.json,report.md,...}. Venv ROCm (texto/sintaxe) + env/.venv (exec; pymongo; Mongo no WSL).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "teacher"))
import eval_teacher as et  # noqa: E402
from _common import ENV_VENV_PY, library_versions, model_fingerprint, read_jsonl, sha256_file, write_json  # noqa: E402

SAMPLE = ROOT / "Repos/KapiTeacher-Mongo-7B/reports/eval/base-v2c/sample_ids.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["t0", "t1"], required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--skip-exec", action="store_true")
    a = ap.parse_args()
    et.DATA_VERSION = "v2c"
    from compact import syntax
    from compact.contracts import contract_for, ide_rejects, postprocess
    from compact.mongo_validate import validate
    from transformers import AutoTokenizer

    prompts = json.loads((ROOT / "reports/export/work" / f"{a.tier}_prompts.json").read_text(encoding="utf-8"))
    run = json.loads(Path(a.run).read_text(encoding="utf-8"))
    model_dir = Path(prompts["model"])
    tok = AutoTokenizer.from_pretrained(model_dir)
    contract = contract_for(tok)
    stops = contract.stops
    rows_src = {r["id"]: r for r in et.read_rows("editor-context-v1", "val")}
    out_dir = ROOT / "reports" / "export" / "eval" / a.tier / a.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    by_prompt = {p["id"]: p for p in prompts["items"]}
    rows = []
    for item in run["items"]:
        src = rows_src[item["id"]]
        toks = item["tokens"]
        kept = [t for t in toks if t not in stops]
        raw = tok.decode(kept, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        pred = postprocess(raw, src)
        pred32 = postprocess(tok.decode(kept[: et.IDE_MAX_NEW_TOKENS], skip_special_tokens=False, clean_up_tokenization_spaces=False), src)
        ref = src["middle"]
        rows.append({"id": src["id"], "split": src["split"], "cut_kind": src["cut_kind"], "domain": src["domain"], "dialect": src["dialect"],
                     "schema_id": src.get("schema_id"), "collection": src.get("collection"), "cursor_role": src.get("cursor_role"),
                     "context_kind": src.get("context_kind"), "prompt_tokens": len(by_prompt[src["id"]]["prompt_ids"]),
                     "generated_tokens": len(toks), "stop": item["stop"], "stop_token": toks[-1], "ttft_ms": item["ttft_ms"],
                     "raw": raw, "prediction": pred, "prediction_ide32": pred32, "reference": ref,
                     "ide_rejected": ide_rejects(pred, item["stop"] in ("eos", "suffix")),
                     "metrics": et.score_text(pred, ref, tok), "metrics_ide32": et.score_text(pred32, ref, tok)})
    jobs = []
    by_id = {r["id"]: rows_src[r["id"]] for r in rows}
    for r in rows:
        s = by_id[r["id"]]
        for key, text in (("pred", r["prediction"]), ("pred32", r["prediction_ide32"]), ("ref", r["reference"])):
            jobs.append((f"{r['id']}|{key}", (s["prefix"] + text + s["suffix"]).replace("\r\n", "\n"), syntax.kind_for(s["dialect"])))
    sres = syntax.check_many(jobs)
    texts = {j[0]: j[1] for j in jobs}
    for r in rows:
        for key, target in (("pred", "metrics"), ("pred32", "metrics_ide32")):
            job = f"{r['id']}|{key}"
            ok = sres[job][0]
            chk = validate(texts[job], r["dialect"]) if ok else None
            r[target]["syntax_valid"] = ok
            r[target]["mongo_valid"] = chk.valid if chk and chk.validatable else (False if not ok else None)
        job = f"{r['id']}|ref"
        ok = sres[job][0]
        chk = validate(texts[job], r["dialect"]) if ok else None
        r["reference_checks"] = {"syntax_valid": ok, "mongo_valid": chk.valid if chk and chk.validatable else (False if not ok else None)}
    pred_path = out_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8", newline="\n") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    ds_path, ds_manifest = et.dataset_paths("editor-context-v1")
    ids = [r["id"] for r in rows]
    write_json(out_dir / "gen_meta.json", {
        "model_path": run["pkg"], "model": {**model_fingerprint(model_dir), "source": "merged (HF bf16 de origem)"}, "load": f"onnx:{run['provider']}",
        "data_version": "v2c", "contract": "editor-context-v1", "model_contract": contract.name, "stop_ids": sorted(stops),
        "max_new_tokens": et.MAX_NEW_TOKENS, "ide_max_new_tokens": et.IDE_MAX_NEW_TOKENS, "context_tokens": et.CONTEXT_TOKENS,
        "decoding": "greedy (onnxruntime-genai 0.15.2, argmax)", "seed": prompts["seed"],
        "dataset": {"path": str(ds_path.relative_to(ROOT)), "sha256": sha256_file(ds_path), "manifest_sha256": sha256_file(ds_manifest),
                    "split": "val", "split_size": len(rows_src), "n": len(rows), "seed": prompts["seed"], "sample_ids_sha256": et.sha256_text("\n".join(ids))},
        "libraries": {**library_versions(), "onnxruntime_genai": run.get("genai")}, "gpu": {"device_name": run["provider"]},
        "memory": {"vram_max_allocated_gb": 0, "vram_max_reserved_gb": 0}, "gpu_seconds": run.get("seconds", 0), "generation_seconds": run.get("seconds", 0),
        "examples_generated": len(rows), "finished": time.strftime("%FT%T")})
    write_json(out_dir / "sample_ids.json", {"ids": ids, "n": len(ids)})
    if not a.skip_exec:
        cmd = [str(ENV_VENV_PY), "-I", str(ROOT / "scripts/teacher/eval_teacher.py"), "--phase", "score", "--model", str(model_dir), "--split", "val",
               "--contract", "editor-context-v1", "--n", str(len(rows)), "--out-root", str(out_dir.parent), "--name", a.variant,
               "--data-version", "v2c", "--ids-from", str(out_dir / "sample_ids.json")]
        subprocess.run(cmd, check=True, cwd=ROOT)
    ns = argparse.Namespace(split="val")
    summary = et.phase_report(ns, out_dir)
    print(json.dumps({k: summary["headline"][k] for k in ("normalized_match", "exec_valid", "nonexistent_fields_rate", "syntax_valid")}))


if __name__ == "__main__":
    main()
