#!/usr/bin/env python3
"""Avaliacao completa de um aluno do retreino v3 (J22 = T0, J23 = T1) e decisao.

  C:/SlopStudio.MongoAI-artifacts/.venv/Scripts/python.exe -I scripts/student/eval_v3.py --tier t0 [--dry-run]

Passos (retomaveis; cada um pula o que ja existe):
  1  val v2c, MESMA amostra de 2.000 ids + Generic-JS (500): eval_student.py run --name sft_v3 (venv ROCm, GPU 7800 XT; Mongo no WSL para a execucao)
  2  homologacao v1: eval_homolog.py run --name <tier>_v3 --resume (gen no venv ROCm; score no env/.venv por execucao do TEXTO atras do proxy somente-leitura).
     O diretorio data/homolog so e lido dentro de homolog_guard.authorized (linha em access.log.jsonl); este driver nunca abre data/homolog.
  3  relatorio da homologacao (base x atual x v3 + comparacao pareada): eval_homolog.py report --names ... --out-name homolog-v3-eval-<tier>
  4  val v2c pareado (atual x v3, base x v3): compare_models sobre os predictions.jsonl -> reports/student/v3_val_compare_<tier>.{json,md}
  5  decide_v3.py: aplica os criterios do README_job -> reports/student/v3_decision_<tier>.{json,md} e (so se GO) reports/student/v3_gate_<tier>.go
Antes: GPU livre (tmp/gpu.lock), `wsl -d Ubuntu -u root -e systemctl start mongod` e `wsl -d Ubuntu -e sleep infinity` abertos, Node no PATH.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV_PY = ROOT / "env" / ".venv" / "Scripts" / "python.exe"
TIERS = {
    "t0": {"repo": "KapiCoder-Mongo-Nano", "base_name": "base-qwen2.5-coder-0.5b", "hom": ("base_0.5b", "t0_distill", "t0_v3")},
    "t1": {"repo": "KapiCoder-Mongo-Small", "base_name": "base-qwen2.5-coder-1.5b", "hom": ("base_1.5b", "t1_distill", "t1_v3")},
}


def sh(cmd: list[str], dry: bool) -> int:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    return 0 if dry else subprocess.run([str(c) for c in cmd], cwd=ROOT).returncode


def val_pairs(tier: str, dry: bool) -> None:
    t = TIERS[tier]
    ev = ROOT / "Repos" / t["repo"] / "reports" / "eval"
    print(f"[val-pairs] {ev}", flush=True)
    if dry:
        return
    sys.path[:0] = [str(ROOT / "scripts" / "eval"), str(ROOT / "scripts" / "student")]
    import compare_models
    import eval_student as es
    cmp_dir = ev / "compare_v3"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in ("sft_distill_v1", "sft_v3", t["base_name"]):
        p = cmp_dir / f"{name}.cmp.jsonl"
        with p.open("w", encoding="utf-8", newline="\n") as f:
            for r in es.to_cmp(es.read_jsonl(ev / name / "predictions.jsonl")):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        files[name] = p
    out = {}
    for a, b in (("sft_distill_v1", "sft_v3"), (t["base_name"], "sft_v3")):
        res = compare_models.run(files[a], files[b], group_by="context_kind", min_group_n=es.MIN_GROUP_N,
                                 margins={"normalized_match": 0.02, "exec_valid": 0.02})
        out[f"{b}_vs_{a}"] = res
        (cmp_dir / f"{b}_vs_{a}.md").write_text(compare_models.to_markdown(res), encoding="utf-8")
    dst = ROOT / "reports" / "student"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / f"v3_val_compare_{tier}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=list(TIERS), required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip", nargs="*", default=[], choices=["val", "hom", "report", "pairs", "decide"])
    a = ap.parse_args(argv)
    t = TIERS[a.tier]
    model = f"Repos/{t['repo']}/merged/sft_v3"
    py = sys.executable
    base_k, cur_k, new_k = t["hom"]
    if "val" not in a.skip and sh([py, "-I", "scripts/student/eval_student.py", "run", "--tier", a.tier, "--model", model, "--name", "sft_v3",
                                   "--base-name", t["base_name"]], a.dry_run):
        return 1
    if "hom" not in a.skip and sh([py, "-I", "scripts/eval/eval_homolog.py", "run", "--model", model, "--name", new_k, "--resume"], a.dry_run):
        return 1
    if "report" not in a.skip and sh([ENV_PY, "-I", "scripts/eval/eval_homolog.py", "report", "--names", base_k, cur_k, new_k,
                                      "--out-name", f"homolog-v3-eval-{a.tier}"], a.dry_run):
        return 1
    if "pairs" not in a.skip:
        val_pairs(a.tier, a.dry_run)
    if "decide" not in a.skip:
        return sh([py, "-I", "scripts/student/decide_v3.py", "--tier", a.tier], a.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
