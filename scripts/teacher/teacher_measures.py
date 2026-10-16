#!/usr/bin/env python3
"""Monta o JSON de medidas dos gates do professor (configs/gates.teacher.yaml) a partir dos summary.json do eval_teacher.py.

  python -I scripts/teacher/teacher_measures.py \
      --adapter-summary Repos/KapiTeacher-Mongo-7B/reports/eval/sft_v2c-final/summary.json \
      --base-summary    Repos/KapiTeacher-Mongo-7B/reports/eval/base-v2c/summary.json \
      [--student-summary <summary.json do melhor student sem professor>] \
      [--generic-base <summary.json> --generic-adapter <summary.json>] \
      --out Repos/KapiTeacher-Mongo-7B/reports/eval/sft_v2c-final/measures.json
  python -I scripts/eval/check_gates.py --tier TEACHER --gates configs/gates.teacher.yaml --measures <out>

Os summaries base e adapter precisam ser do MESMO split/amostra (use --ids-from do sample_ids.json do primeiro). G2.7 vem de
data/teacher/v2c/manifest.json (verificacao do corpus). Metricas sem fonte ficam ausentes (-> unevaluated, nunca aprovadas).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_ID_RE = re.compile(r'^\{"id":\s*"([^"]+)"')
CONTEXT_KINDS = ("full", "irrelevant", "none")


def context_kinds_by_id(ids: set[str], validated_dir: Path) -> dict[str, str]:
    """`context_kind` de cada id lendo data/validated/<versao>/fim-*.jsonl (so o prefixo da linha e filtrado antes do parse)."""
    found: dict[str, str] = {}
    for shard in sorted(validated_dir.glob("fim-*.jsonl")):
        with shard.open(encoding="utf-8") as handle:
            for line in handle:
                m = _ID_RE.match(line)
                if m and m.group(1) in ids:
                    found[m.group(1)] = json.loads(line).get("context_kind")
        if len(found) == len(ids):
            break
    return found


def nonexistent_by_context(pred_rows: list[dict], validated_dir: Path | None) -> dict:
    """Taxa de campos inexistentes (stage=fields `campo/estrutura invalido`) por context_kind, sobre as predicoes do adapter."""
    ids = {r["id"] for r in pred_rows if not r.get("context_kind")}
    joined = context_kinds_by_id(ids, validated_dir) if ids and validated_dir else {}
    groups: dict[str, list[bool]] = defaultdict(list)
    for r in pred_rows:
        kind = r.get("context_kind") or joined.get(r["id"])
        if kind and r.get("exec"):
            groups[kind].append(bool(r["exec"]["field_hallucination"]))
    return {k: {"n": len(v), "nonexistent": sum(v), "rate_pct": round(100 * sum(v) / len(v), 2)} for k, v in groups.items() if v}


def load(path: str | None) -> dict | None:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter-summary", required=True)
    ap.add_argument("--base-summary", required=True)
    ap.add_argument("--student-summary")
    ap.add_argument("--generic-base")
    ap.add_argument("--generic-adapter")
    ap.add_argument("--predictions", help="predictions.jsonl do adapter (padrao: ao lado do --adapter-summary)")
    ap.add_argument("--validated-dir", help="data/validated/<versao> para juntar context_kind (padrao: deduzido do dataset do summary)")
    ap.add_argument("--corpus-manifest", default=str(ROOT / "data" / "teacher" / "v2c" / "manifest.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ad, base = load(a.adapter_summary), load(a.base_summary)
    metrics: dict = {}
    notes: list[str] = []
    if ad["dataset"]["sample_ids_sha256"] != base["dataset"]["sample_ids_sha256"]:
        notes.append("AVISO: amostras de adapter e base diferem (use --ids-from); comparacao pareada invalida")
    h, hb = ad["headline"], base["headline"]
    metrics["fim_acc_gain_vs_base_pp"] = round((h["normalized_match"] - hb["normalized_match"]) * 100, 2)
    metrics["exec_valid_autocomplete"] = h["exec_valid"]
    # recalibracao do G2.2 (2026-10-15): o validador executa so um subconjunto do Console; o proprio `middle` de referencia
    # tem exec-valid < 1 (teto). O gate passa a ser relativo ao teto medido no mesmo conjunto/validador.
    ceil = (ad.get("reference_ceiling") or {}).get("exec_valid")
    if ceil:
        metrics["reference_ceiling_exec_valid"] = ceil
        metrics["exec_valid_vs_ceiling"] = round(h["exec_valid"] / ceil, 4)
    if h.get("nonexistent_fields_rate") is not None:
        metrics["nonexistent_fields_rate_pct"] = round(h["nonexistent_fields_rate"] * 100, 2)
    # G2.2 (decisao de 2026-10-15): o gate bloqueante e a taxa em context_kind=full; geral/none/irrelevant sao informativos
    pred_path = Path(a.predictions) if a.predictions else Path(a.adapter_summary).parent / "predictions.jsonl"
    if pred_path.exists():
        vdir = Path(a.validated_dir) if a.validated_dir else None
        if vdir is None:
            m = re.search(r"validated[\/]+([^\/]+)", ad["dataset"].get("path", ""))
            vdir = ROOT / "data" / "validated" / (m.group(1) if m else "v2c")
        pred_rows = [json.loads(line) for line in pred_path.open(encoding="utf-8")]
        by_ctx = nonexistent_by_context(pred_rows, vdir)
        for kind in CONTEXT_KINDS:
            if kind in by_ctx:
                metrics[f"nonexistent_fields_{kind}_rate_pct"] = by_ctx[kind]["rate_pct"]
                metrics[f"n_context_{kind}"] = by_ctx[kind]["n"]
        if "full" in by_ctx:   # nome da metrica do gate G2.2.nonexistent_fields_full
            metrics["nonexistent_fields_full_rate_pct"] = by_ctx["full"]["rate_pct"]
    else:
        notes.append(f"AVISO: {pred_path} ausente; nonexistent_fields_full_rate_pct nao calculado (gate fica unevaluated)")
    if a.student_summary:
        st = load(a.student_summary)["headline"]
        metrics["fim_acc_vs_best_student_pp"] = round((h["normalized_match"] - st["normalized_match"]) * 100, 2)
    if a.generic_base and a.generic_adapter:
        # G2.5: conjunto generic_js_fim_eval (500 cortes, eval_teacher.py --data-version generic-js --split eval), MESMA amostra;
        # delta = adapter - base em FIM acc (normalized match) e em exact match; o gate usa o normalized (como o G2.2)
        sb, sa = load(a.generic_base), load(a.generic_adapter)
        gb, ga = sb["headline"], sa["headline"]
        if sb["dataset"]["sample_ids_sha256"] != sa["dataset"]["sample_ids_sha256"]:
            notes.append("AVISO: amostras generic-js de base e adapter diferem; G2.5 invalido")
        else:
            metrics["generic_js_fim_delta_pp"] = round((ga["normalized_match"] - gb["normalized_match"]) * 100, 2)
            metrics["generic_js_fim_drop_pp"] = round((gb["normalized_match"] - ga["normalized_match"]) * 100, 2)  # informativo (= -delta)
            metrics["generic_js_exact_match_delta_pp"] = round((ga["exact_match"] - gb["exact_match"]) * 100, 2)
            metrics["generic_js_syntax_valid_delta_pp"] = round(((ga["syntax_valid"] or 0) - (gb["syntax_valid"] or 0)) * 100, 2)
    corpus = load(a.corpus_manifest)
    ver, iso = corpus["verification"], corpus["isolation"]
    leaks = (ver["in_val_ids"] + ver["in_test_ids"] + ver["in_bench_ids"] + ver["n_issues"] + sum(iso["group_overlap_train_x_eval"].values())
             + sum(iso["schema_overlap_train_x_eval"].values()) + iso["train_records_with_wrong_role_or_holdout"])
    rep = (corpus.get("replay") or {}).get("isolation")   # replay x generic_js_fim_eval / val-test-bench do Mongo
    if rep:
        leaks += (len(rep["package_overlap_with_generic_eval"]) + rep["file_sha256_overlap_with_generic_eval"]
                  + rep["chunks_containing_eval_cut"] + rep["chunks_matching_mongo_eval_content"])
    metrics["train_eval_leak_count"] = leaks
    out = {"run_id": Path(a.adapter_summary).parent.name, "env": {}, "metrics": metrics,
           "sources": {"adapter_summary": a.adapter_summary, "base_summary": a.base_summary, "corpus_manifest_sha256": corpus.get("manifest_sha256")},
           "notes": notes}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=1), *notes, sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
