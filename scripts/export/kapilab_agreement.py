#!/usr/bin/env python3
"""Concordancia de texto IDE (kapilab model run autocomplete) x Python (ONNX, mesmo pacote, greedy 96 tokens -> ghost). Gate common.harness.python_agreement
(>= 99,5%) e do KapiLab; aqui e so informativo (o pos-processamento da IDE e o CompletionOutputProcessor, o do Python e compact.contracts.postprocess;
a IDE limita a 32 tokens e os registros `context_kind=none` usam cabecalho diferente).
  python -I scripts/export/kapilab_agreement.py --tier t0 --variant int4 --run tmp/kapilab_ws/tmp/run_t0_int4.jsonl"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--tier", required=True); ap.add_argument("--variant", required=True); ap.add_argument("--run", required=True)
a = ap.parse_args()
pred = {}
for l in (ROOT / "reports/export/eval" / a.tier / a.variant / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(l)
    pred[r["id"]] = r
rows = [json.loads(x) for x in Path(a.run).read_text(encoding="utf-8-sig").splitlines() if x.startswith('{"schema":"kapilab-autocomplete-run')]
res = {"n": len(rows), "by_context_kind": {}}
for r in rows:
    p = pred[r["id"]]
    k = p.get("context_kind")
    ide = r.get("suggestion") or ""
    same32 = ide == p["prediction_ide32"]
    same96 = ide == p["prediction"]
    d = res["by_context_kind"].setdefault(k, {"n": 0, "same_ide32": 0, "same_96": 0, "outcomes": {}})
    d["n"] += 1; d["same_ide32"] += same32; d["same_96"] += same96
    d["outcomes"][r["outcome"]] = d["outcomes"].get(r["outcome"], 0) + 1
tot = sum(d["n"] for d in res["by_context_kind"].values())
res["agreement_ide32_pct_all"] = round(100 * sum(d["same_ide32"] for d in res["by_context_kind"].values()) / max(1, tot), 1)
ok = [d for k, d in res["by_context_kind"].items() if k != "none"]
res["agreement_ide32_pct_full_irrelevant"] = round(100 * sum(d["same_ide32"] for d in ok) / max(1, sum(d["n"] for d in ok)), 1)
print(json.dumps(res))
out = ROOT / "reports" / "export" / f"kapilab_agreement_{a.tier}_{a.variant}.json"
out.write_text(json.dumps(res, indent=2), encoding="utf-8")
