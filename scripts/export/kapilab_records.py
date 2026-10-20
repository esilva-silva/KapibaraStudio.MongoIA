#!/usr/bin/env python3
"""Gera records.jsonl (formato de entrada de `kapilab contract render`: id,type=fim,prefix,suffix,language,editor{...}) com os N primeiros ids da
amostra de paridade (reports/export/work/<tier>_prompts.json), a partir do val v2c (fim_ctx.jsonl). Somente leitura dos dados; saida no workspace do KapiLab."""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "teacher"))
import eval_teacher as et  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tier", default="t0"); ap.add_argument("--n", type=int, default=100); ap.add_argument("--out", required=True)
a = ap.parse_args()
et.DATA_VERSION = "v2c"
prompts = json.loads((ROOT / "reports/export/work" / f"{a.tier}_prompts.json").read_text(encoding="utf-8"))
rows = {r["id"]: r for r in et.read_rows("editor-context-v1", "val")}
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w", encoding="utf-8", newline="\n") as f:
    for p in prompts["items"][: a.n]:
        r = rows[p["id"]]
        f.write(json.dumps({"id": r["id"], "type": "fim", "prefix": r["prefix"], "suffix": r["suffix"], "language": r.get("language") or "Mongo Console JavaScript",
                            "editor": r.get("editor") or {}}, ensure_ascii=False) + "\n")
print("ok", a.n)
