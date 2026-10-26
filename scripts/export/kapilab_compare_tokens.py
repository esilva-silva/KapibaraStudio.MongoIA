#!/usr/bin/env python3
"""Compara os tokenIds do `kapilab contract tokenize` (IDE real: renderer + tokenizer GenAI do adapter) com os ids do Python (editor-context-v1,
reports/export/work/<tier>_prompts.json): % de prompts com sequencia identica."""
import argparse, json, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "teacher"))
sys.path.insert(0, str(ROOT / "scripts" / "export"))
from build_variants import WORK  # noqa: E402
import eval_teacher as et  # noqa: E402
ap = argparse.ArgumentParser(); ap.add_argument("--tier", default="t0"); ap.add_argument("--tokens", required=True); ap.add_argument("--out")
a = ap.parse_args()
py = {p["id"]: p["prompt_ids"] for p in json.loads((WORK / f"{a.tier}_prompts.json").read_text(encoding="utf-8"))["items"]}
rows = [json.loads(l) for l in Path(a.tokens).read_text(encoding="utf-8").splitlines() if l.strip()]
rows = [r for r in rows if r.get("id") in py]
same = [r["id"] for r in rows if r["tokenIds"] == py[r["id"]]]
bad = [r for r in rows if r["tokenIds"] != py[r["id"]]]
et.DATA_VERSION = "v2c"
kind = {r["id"]: r.get("context_kind") for r in et.read_rows("editor-context-v1", "val")}
by = Counter((kind[r["id"]], r["tokenIds"] == py[r["id"]]) for r in rows)
res = {"by_context_kind": {f"{k}:{'identical' if ok else 'different'}": n for (k, ok), n in sorted(by.items(), key=str)}, "n": len(rows), "identical": len(same), "identical_pct": round(100 * len(same) / max(1, len(rows)), 2),
       "mismatch_examples": [{"id": r["id"], "kapilab_len": len(r["tokenIds"]), "python_len": len(py[r["id"]])} for r in bad[:5]]}
print(json.dumps(res))
if a.out: Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
