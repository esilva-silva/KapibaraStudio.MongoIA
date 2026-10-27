#!/usr/bin/env python3
"""Escreve/atualiza Repos/KapiCoder-Mongo-{Nano,Small}/model.lock.json (Fase 00): base (BaseRepos + revisao + sha), corpus dos bracos,
professor, hashes de config/gates e commit dos scripts; apos o treino, acrescenta os modelos treinados (merged/<run>/export_info.json).

  python -I scripts/student/write_model_lock.py [--tiers t0,t1]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "teacher"))
from _common import model_fingerprint  # noqa: E402

TIERS = {"t0": ("KapiCoder-Mongo-Nano", "Qwen2.5-Coder-0.5B", "T0"), "t1": ("KapiCoder-Mongo-Small", "Qwen2.5-Coder-1.5B", "T1")}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 24):
            h.update(block)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()


def scripts_dirty() -> bool:
    # Training configs live under Repos as well as the workspace-level configs directory.
    return bool(git("status", "--porcelain", "--untracked-files=all", "--", "scripts", "configs", ":(glob)Repos/*/configs/**"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", default="t0,t1")
    ap.add_argument("--require-clean", action="store_true", help="Recusa lock de scripts/configs com alteracoes locais; executar depois dos commits dos scripts.")
    args = ap.parse_args()
    if args.require_clean and scripts_dirty():
        raise SystemExit("scripts/configs sujos: nao declarar scripts_dirty=false; revisar e commitar antes")
    base_lock = json.loads((ROOT / "BaseRepos" / "base-repos.lock.json").read_text(encoding="utf-8"))["repos"]
    teacher = json.loads((ROOT / "Repos" / "KapiTeacher-Mongo-7B" / "model.lock.json").read_text(encoding="utf-8"))
    for t in args.tiers.split(","):
        repo, base, gate_tier = TIERS[t]
        base_dir = ROOT / "BaseRepos" / base
        fp = model_fingerprint(base_dir)
        lock = {"name": repo, "tier": gate_tier, "updated_utc": time.strftime("%FT%TZ", time.gmtime()),
                "base": {"path": f"BaseRepos/{base}", "repo_id": base_lock[base]["repo_id"], "revision": base_lock[base]["revision"],
                         "model_sha256": fp["model_sha256"], "tokenizer_sha256_identical_to_teacher": True},
                "dataset_version": "v2c", "scope": "autocomplete FIM editor-context-v1 (chat fora ate a decisao Q1); KD por logits adiado",
                "teacher": {"workspace": "Repos/KapiTeacher-Mongo-7B", "adapter_sha256": teacher["adapter"]["adapter_model_sha256"],
                            "gguf_q8_0_sha256": teacher["gguf"]["files"]["Q8_0"]["sha256"],
                            "distilled_dataset": "data/distilled/v2c-final/accepted.jsonl", "label_source": "teacher_seq"},
                "arms": {}, "scripts_commit": git("rev-parse", "HEAD"), "scripts_dirty": scripts_dirty(),
                "student_corpus_config_sha256": sha256(ROOT / "configs" / "student_corpus.yaml"), "gates_yaml_sha256": sha256(ROOT / "configs" / "gates.yaml"),
                "rounds_yaml_sha256": sha256(ROOT / "configs" / "rounds.yaml") if (ROOT / "configs" / "rounds.yaml").exists() else None}
        for arm, run in (("distill", "sft_distill_v1"), ("gold", "sft_gold_v1"), ("v3", "sft_v3")):
            man_path = ROOT / "data" / "student" / arm / t / "manifest.json"
            arm_info: dict = {"run": run, "config": f"Repos/{repo}/configs/{run}.yaml",
                              "config_sha256": sha256(ROOT / "Repos" / repo / "configs" / f"{run}.yaml")}
            if arm == "v3":
                arm_info["corpus"] = {"path": f"data/student/v3/{t}/manifest.json", "manifest_sha256": sha256(man_path)} if man_path.exists() else None
                training = ROOT / "Repos" / repo / "reports" / run / "run_manifest.json"
                if training.exists():
                    arm_info["training_manifest"] = {"path": f"Repos/{repo}/reports/{run}/run_manifest.json", "sha256": sha256(training)}
            elif man_path.exists():
                man = json.loads(man_path.read_text(encoding="utf-8"))
                arm_info["corpus"] = {"path": f"data/student/{arm}/{t}/manifest.json", "manifest_sha256": sha256(man_path),
                                      "examples": man["counts"]["examples"], "tokens": man["counts"]["tokens"],
                                      "ids_sha256": (man.get("verification") or {}).get("ids_sha256"), "accepted_sha256": man["source"]["accepted_sha256"]}
            export = ROOT / "Repos" / repo / "merged" / run / "export_info.json"
            if export.exists():
                arm_info["merged"] = json.loads(export.read_text(encoding="utf-8"))
            lock["arms"][arm] = arm_info
        v3_config = ROOT / "configs" / "student_corpus_v3.yaml"
        if v3_config.exists():
            lock["student_corpus_v3_config_sha256"] = sha256(v3_config)
        out = ROOT / "Repos" / repo / "model.lock.json"
        out.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print("->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
