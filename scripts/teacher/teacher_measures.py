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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(path: str | None) -> dict | None:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter-summary", required=True)
    ap.add_argument("--base-summary", required=True)
    ap.add_argument("--student-summary")
    ap.add_argument("--generic-base")
    ap.add_argument("--generic-adapter")
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
    if h.get("nonexistent_fields_rate") is not None:
        metrics["nonexistent_fields_rate_pct"] = round(h["nonexistent_fields_rate"] * 100, 2)
    if a.student_summary:
        st = load(a.student_summary)["headline"]
        metrics["fim_acc_vs_best_student_pp"] = round((h["normalized_match"] - st["normalized_match"]) * 100, 2)
    if a.generic_base and a.generic_adapter:
        gb, ga = load(a.generic_base)["headline"], load(a.generic_adapter)["headline"]
        metrics["generic_js_fim_drop_pp"] = round((gb["normalized_match"] - ga["normalized_match"]) * 100, 2)
    corpus = load(a.corpus_manifest)
    ver, iso = corpus["verification"], corpus["isolation"]
    leaks = (ver["in_val_ids"] + ver["in_test_ids"] + ver["in_bench_ids"] + ver["n_issues"] + sum(iso["group_overlap_train_x_eval"].values())
             + sum(iso["schema_overlap_train_x_eval"].values()) + iso["train_records_with_wrong_role_or_holdout"])
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
