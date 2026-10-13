#!/usr/bin/env python3
"""Compara baseline `fim-raw` x `editor-context-v1` (mesmos ids) a partir dos predictions.jsonl de eval_teacher.py.

  python scripts/teacher/compare_ctx.py --models Qwen2.5-Coder-1.5B Qwen2.5-Coder-7B [--root <baseline_base>]
Metricas do baseline restritas aos ids da execucao com contexto (`<modelo>-ctx`). Usa aggregate() de eval_teacher.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import TEACHER_REPO, read_jsonl  # noqa: E402
import eval_teacher as et  # noqa: E402

KEYS = ("exact_match", "normalized_match", "edit_similarity", "chars_accepted", "accepted_prefix_tokens", "syntax_valid", "mongo_valid",
        "exec_valid", "nonexistent_fields_rate", "empty_rate")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--root", default=str(TEACHER_REPO / "reports" / "baseline_base"))
    a = ap.parse_args()
    root = Path(a.root)
    lines = []
    for m in a.models:
        ctx = read_jsonl(root / f"{m}-ctx" / "predictions.jsonl")
        base_all = {r["id"]: r for r in read_jsonl(root / m / "predictions.jsonl")}
        ids = [r["id"] for r in ctx if r["id"] in base_all]
        base = [base_all[i] for i in ids]
        ctx = [r for r in ctx if r["id"] in base_all]
        ha, hb = et.aggregate(base), et.aggregate(ctx)
        lines += [f"### {m} (n={len(ids)}, ids pareados)", "", "| metrica | fim-raw | editor-context-v1 | delta (p.p.) |", "|---|---:|---:|---:|"]
        for k in KEYS:
            x, y = ha.get(k), hb.get(k)
            if x is None or y is None:
                continue
            scale = 1 if k in ("chars_accepted", "accepted_prefix_tokens") else 100
            d = (y - x) * scale
            fmt = (lambda v: f"{v:.2f}") if scale == 1 else (lambda v: f"{v * 100:.1f}%")
            lines.append(f"| {k} | {fmt(x)} | {fmt(y)} | {d:+.1f} |")
        pt = sorted(r["prompt_tokens"] for r in ctx)
        pb = sorted(r["prompt_tokens"] for r in base)
        lines += ["", f"Tokens do prompt p50/p95/max: fim-raw {pb[len(pb) // 2]}/{pb[int(len(pb) * .95)]}/{pb[-1]} ; ctx {pt[len(pt) // 2]}/{pt[int(len(pt) * .95)]}/{pt[-1]}", ""]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
