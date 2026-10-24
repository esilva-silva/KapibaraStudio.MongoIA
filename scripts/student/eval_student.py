#!/usr/bin/env python3
"""Avaliacao dos alunos T0 Nano / T1 Small (modelo COMPLETO, sem adapter) no val v2c + Generic-JS, medidas e gates (configs/gates.yaml T0/T1).

Reaproveita o scorer do professor (scripts/teacher/eval_teacher.py: contrato editor-context-v1, mesma amostra, exec-valid somente leitura em
kapi_fx_*, node vm.Script, mongo_validate) e as ferramentas de scripts/eval (compare_models, check_gates, stats).

  # 1) gera (GPU, ROCm) + executa (Mongo no WSL) + Generic-JS + medidas + gates, tudo para um modelo (usado pelos jobs J16/J17/J10e):
  python -I scripts/student/eval_student.py run --tier t0 --model Repos/KapiCoder-Mongo-Nano/merged/sft_distill_v1 --name sft_distill_v1 \
         [--base-name base-qwen2.5-coder-0.5b] [--n 2000] [--no-generic] [--phase all|gen|measures]
  # 2) so as medidas/gates a partir de summaries ja gerados:
  python -I scripts/student/eval_student.py measures --tier t0 --summary <summary.json> [--base-summary ...] [--generic-summary ...] --out <measures.json>
  # 3) comparacao pareada distilado x controle x base (+ G2.4 do professor via teacher_measures --student-summary):
  python -I scripts/student/eval_student.py compare --tier t1 --distill sft_distill_v1 --gold sft_gold_v1 --base base-qwen2.5-coder-1.5b

Amostra: SEMPRE os 2.000 ids de Repos/KapiTeacher-Mongo-7B/reports/eval/base-v2c/sample_ids.json (val v2c, nunca train/test/bench/cego) e os 500 cortes
do generic_js_fim_eval (sample_ids do baseline generic-js do 7B): as mesmas do professor, entao as tabelas sao diretamente comparaveis.
Metricas sem fonte aqui (TTFT/RAM/paridade do KapiLab, chat, HumanEval+, baseline v1 re-medido) ficam AUSENTES -> `unevaluated`, nunca aprovadas.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
sys.path.insert(0, str(ROOT / "scripts" / "teacher"))
sys.path.insert(0, str(ROOT / "bakeoff" / "v1" / "scripts"))
import stats  # noqa: E402

TEACHER_EVAL = ROOT / "Repos" / "KapiTeacher-Mongo-7B" / "reports" / "eval"
SAMPLE_VAL = TEACHER_EVAL / "base-v2c" / "sample_ids.json"
SAMPLE_GENERIC = ROOT / "Repos" / "KapiTeacher-Mongo-7B" / "reports" / "baseline_base" / "Qwen2.5-Coder-7B-generic-js" / "sample_ids.json"
TIERS = {"t0": {"repo": "KapiCoder-Mongo-Nano", "gate_tier": "T0", "base": "BaseRepos/Qwen2.5-Coder-0.5B"},
         "t1": {"repo": "KapiCoder-Mongo-Small", "gate_tier": "T1", "base": "BaseRepos/Qwen2.5-Coder-1.5B"}}
CONTEXT_KINDS = ("full", "irrelevant", "none")
MIN_GROUP_N = 50


def load_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def eval_root(tier: str) -> Path:
    return ROOT / "Repos" / TIERS[tier]["repo"] / "reports" / "eval"


# -- registros comparaveis ------------------------------------------------------------------------------------------------------------
def conversational(pred: str) -> bool:
    """Definicao do bake-off (benchmark.py): resposta conversacional OU com cerca Markdown."""
    from compact import metrics
    return bool(metrics.is_conversational(pred) or "```" in pred)


def to_cmp(rows: list[dict]) -> list[dict]:
    """predictions.jsonl do eval_teacher -> registros achatados para compare_models (metrics.* booleanos/numericos)."""
    out = []
    for r in rows:
        m, e = r["metrics"], r.get("exec") or {}
        strict = bool(e.get("exec_valid")) and not m.get("empty")
        out.append({"id": r["id"], "cut_kind": r["cut_kind"], "context_kind": r.get("context_kind"), "domain": r["domain"],
                    "ttft_ms": r.get("ttft_ms"), "generated_tokens": r.get("generated_tokens"),
                    "metrics": {"normalized_match": bool(m["normalized_match"]), "exact_match": bool(m["exact_match"]),
                                "accepted_prefix_tokens": float(m["accepted_prefix_tokens"]), "edit_similarity": float(m["edit_similarity"]),
                                "syntax_valid": bool(m.get("syntax_valid")), "mongo_valid": bool(m.get("mongo_valid")),
                                "exec_valid": strict, "no_nonexistent_field": not bool(e.get("field_hallucination")), "empty": bool(m.get("empty")),
                                "conversational": conversational(r["prediction"])}})
    return out


def delta(a_rows: list[dict], b_rows: list[dict], key: str, resamples: int = stats.DEFAULT_RESAMPLES) -> dict:
    """Delta pareado B - A de metrics[key] nos ids em comum (bootstrap 10k, IC95%)."""
    a = {r["id"]: r for r in a_rows}
    b = {r["id"]: r for r in b_rows}
    ids = sorted(set(a) & set(b))
    return stats.paired_bootstrap([float(a[i]["metrics"][key]) for i in ids], [float(b[i]["metrics"][key]) for i in ids], resamples)


def min_group_delta(a_rows: list[dict], b_rows: list[dict], key: str, group: str = "cut_kind") -> dict:
    a = {r["id"]: r for r in a_rows}
    b = {r["id"]: r for r in b_rows}
    groups: dict[str, list[str]] = {}
    for i in sorted(set(a) & set(b)):
        groups.setdefault(str(a[i].get(group)), []).append(i)
    res = {}
    for g, ids in groups.items():
        if len(ids) >= MIN_GROUP_N:
            res[g] = round(sum(float(b[i]["metrics"][key]) - float(a[i]["metrics"][key]) for i in ids) / len(ids), 4)
    return res


# -- medidas dos gates ----------------------------------------------------------------------------------------------------------------
def pct(x) -> float | None:
    return None if x is None else round(100 * x, 2)


def build_measures(summary: dict, rows: list[dict], *, tier: str, base_summary: dict | None = None, base_rows: list[dict] | None = None,
                   generic: dict | None = None, generic_base: dict | None = None, corpus_manifest: dict | None = None,
                   baseline_v1_rows: list[dict] | None = None, t0_rows: list[dict] | None = None, report_exists: bool = False,
                   resamples: int = stats.DEFAULT_RESAMPLES) -> dict:
    """measures.json no formato de scripts/eval/check_gates.py (`metrics` por id de gate ou por nome de metrica)."""
    h = summary["headline"]
    m: dict = {}
    notes: list[str] = []
    cmp_rows = to_cmp(rows)
    # --- gates T0 que tem fonte aqui (python) ---
    m["syntax_valid"] = h["syntax_valid"]
    m["mongodb_valid"] = h["mongo_valid"]
    m["nonexistent_fields_rate"] = pct(h["nonexistent_fields_rate"])
    m["conversational_suggestion_rate"] = round(100 * sum(r["metrics"]["conversational"] for r in cmp_rows) / max(1, len(cmp_rows)), 2)
    # --- informativas (mesmos nomes do teacher_measures) ---
    m["fim_acc"] = h["normalized_match"]
    m["exact_match"] = h["exact_match"]
    m["apt_mean"] = h["accepted_prefix_tokens"]
    m["exec_valid_autocomplete"] = h["exec_valid"]
    m["empty_rate"] = h["empty_rate"]
    ceil = (summary.get("reference_ceiling") or {}).get("exec_valid")
    if ceil and h.get("exec_valid") is not None:
        m["reference_ceiling_exec_valid"] = ceil
        m["exec_valid_vs_ceiling"] = round(h["exec_valid"] / ceil, 4)
    by_ctx = summary.get("by_context_kind") or {}
    for kind in CONTEXT_KINDS:
        if kind in by_ctx and by_ctx[kind].get("nonexistent_fields_rate") is not None:
            m[f"nonexistent_fields_{kind}_rate_pct"] = pct(by_ctx[kind]["nonexistent_fields_rate"])
            m[f"n_context_{kind}"] = by_ctx[kind]["n"]
            m[f"exec_valid_{kind}"] = by_ctx[kind].get("exec_valid")
    # --- comparacao com o zero-shot do mesmo tamanho (informativa: NAO e o `base_v1` dos gates) ---
    if base_summary:
        hb = base_summary["headline"]
        if base_summary["dataset"]["sample_ids_sha256"] != summary["dataset"]["sample_ids_sha256"]:
            notes.append("AVISO: amostra do base zero-shot difere da do aluno; comparacao pareada invalida")
        m["info_fim_acc_gain_vs_zero_shot_base_pp"] = round((h["normalized_match"] - hb["normalized_match"]) * 100, 2)
        m["info_apt_delta_vs_zero_shot_base"] = round(h["accepted_prefix_tokens"] - hb["accepted_prefix_tokens"], 4)
        m["info_exec_valid_delta_vs_zero_shot_base_pp"] = round(((h["exec_valid"] or 0) - (hb["exec_valid"] or 0)) * 100, 2)
    if generic:
        g = generic["headline"]
        m["generic_js_fim_acc"] = g["normalized_match"]
        if generic_base:
            if generic_base["dataset"]["sample_ids_sha256"] != generic["dataset"]["sample_ids_sha256"]:
                notes.append("AVISO: amostras generic-js diferem; delta nao calculado")
            else:
                m["generic_js_fim_delta_pp_vs_zero_shot_base"] = round((g["normalized_match"] - generic_base["headline"]["normalized_match"]) * 100, 2)
    # --- baseline v1 re-medido (gates T0.apt_* / exec_match): so quando o usuario fornece as predicoes do modelo v1 na suite v2 ---
    if baseline_v1_rows:
        base_cmp = to_cmp(baseline_v1_rows)
        d = delta(base_cmp, cmp_rows, "accepted_prefix_tokens", resamples)
        m["T0.apt_vs_base"] = {"delta": round(d["delta"], 4), "ci_low": round(d["ci_low"], 4), "ci_high": round(d["ci_high"], 4)}
        m["apt_vs_base_v1_0.5b_mean_delta"] = round(d["delta"], 4)
        per_group = min_group_delta(base_cmp, cmp_rows, "accepted_prefix_tokens")
        if per_group:
            m["apt_delta_per_group_min"] = min(per_group.values())
        e = delta(base_cmp, cmp_rows, "exec_valid", resamples)
        m["exec_match_autocomplete"] = round(e["delta"] * 100, 2)
    else:
        notes.append("T0.apt_vs_base / apt_gain / apt_per_group / exec_match_autocomplete: sem predicoes do modelo v1 (base_v1_0.5b) re-medido na suite v2 -> "
                     "unevaluated (use --baseline-v1-predictions)")
    if tier == "t1" and t0_rows:
        d = delta(to_cmp(t0_rows), cmp_rows, "accepted_prefix_tokens", resamples)
        m["apt_vs_T0_approved"] = {"delta": round(d["delta"], 4), "ci_low": round(d["ci_low"], 4), "ci_high": round(d["ci_high"], 4)}
    # --- gates comuns que o Python mede ---
    score = summary.get("exec_score") or {}
    guard = score.get("write_commands_seen_by_guard")
    if guard is not None:
        m["write_or_destructive_executions"] = len(guard)
    if corpus_manifest:
        ver, leaks = corpus_manifest.get("verification") or {}, 0
        leaks = ver.get("n_issues", 0) + ver.get("in_val_ids", 0) + ver.get("in_test_ids", 0) + ver.get("in_bench_ids", 0)
        rep = (corpus_manifest.get("replay") or {}).get("isolation") or {}
        leaks += (len(rep.get("package_overlap_with_generic_eval") or []) + rep.get("file_sha256_overlap_with_generic_eval", 0)
                  + rep.get("chunks_containing_eval_cut", 0) + rep.get("chunks_matching_mongo_eval_content", 0))
        leaks += (corpus_manifest.get("isolation", {}).get("accepted_load") or {}).get("isolation_dropped", 0)
        m["train_eval_leak_count"] = leaks
        m["decontamination_approved"] = leaks == 0
    m["auto_report_attached"] = bool(report_exists)
    return {"run_id": summary.get("run_name"), "env": {}, "metrics": m,
            "sources": {"summary": summary.get("run_name"), "n": summary["n"], "sample_ids_sha256": summary["dataset"]["sample_ids_sha256"],
                        "corpus_manifest_sha256": (corpus_manifest or {}).get("manifest_sha256")},
            "notes": notes}


# -- execucao -------------------------------------------------------------------------------------------------------------------------
def run_eval_teacher(argv: list[str]) -> None:
    cmd = [sys.executable, "-I", str(ROOT / "scripts" / "teacher" / "eval_teacher.py"), *argv]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def cmd_run(a: argparse.Namespace) -> int:
    t = TIERS[a.tier]
    out_root = eval_root(a.tier)
    model = a.model
    done_val = out_root / a.name / "summary.json"
    done_gen = out_root / f"{a.name}-generic-js" / "summary.json"
    if a.phase in ("all", "gen"):
        if not done_val.exists() or a.force:
            run_eval_teacher(["--model", model, "--data-version", "v2c", "--split", "val", "--contract", "editor-context-v1", "--n", str(a.n),
                              "--load", "bf16", "--name", a.name, "--resume", "--ids-from", str(SAMPLE_VAL), "--out-root", str(out_root)])
        if not a.no_generic and (not done_gen.exists() or a.force):
            run_eval_teacher(["--model", model, "--data-version", "generic-js", "--split", "eval", "--contract", "fim-raw", "--n", "500",
                              "--load", "bf16", "--name", f"{a.name}-generic-js", "--resume", "--ids-from", str(SAMPLE_GENERIC), "--out-root", str(out_root)])
    if a.phase in ("all", "measures"):
        return cmd_measures_for(a)
    return 0


def cmd_measures_for(a: argparse.Namespace) -> int:
    t = TIERS[a.tier]
    out_root = eval_root(a.tier)
    summary_path = out_root / a.name / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"{summary_path} nao existe (rode --phase gen)")
    ns = argparse.Namespace(tier=a.tier, summary=str(summary_path), base_summary=None, generic_summary=None, generic_base_summary=None,
                            corpus_manifest=None, baseline_v1_predictions=getattr(a, "baseline_v1_predictions", None),
                            t0_predictions=getattr(a, "t0_predictions", None), out=str(out_root / a.name / "measures.json"), gates=True,
                            resamples=stats.DEFAULT_RESAMPLES)
    if a.base_name and (out_root / a.base_name / "summary.json").exists():
        ns.base_summary = str(out_root / a.base_name / "summary.json")
    gen = out_root / f"{a.name}-generic-js" / "summary.json"
    if gen.exists():
        ns.generic_summary = str(gen)
    if a.base_name and (out_root / f"{a.base_name}-generic-js" / "summary.json").exists():
        ns.generic_base_summary = str(out_root / f"{a.base_name}-generic-js" / "summary.json")
    arm = "v3" if a.name.startswith("sft_v3") else "gold" if "gold" in a.name else "distill"      # sft_v3: corpus v3 (data/student/v3/<tier>)
    cm = ROOT / "data" / "student" / arm / a.tier / "manifest.json"
    if cm.exists():
        ns.corpus_manifest = str(cm)
    return cmd_measures(ns)


def cmd_measures(a: argparse.Namespace) -> int:
    summary = load_json(a.summary)
    sdir = Path(a.summary).parent
    rows = read_jsonl(sdir / "predictions.jsonl")
    kw: dict = {"tier": a.tier, "report_exists": (sdir / "report.md").exists(), "resamples": a.resamples}
    if a.base_summary:
        kw["base_summary"] = load_json(a.base_summary)
        kw["base_rows"] = read_jsonl(Path(a.base_summary).parent / "predictions.jsonl")
    if a.generic_summary:
        kw["generic"] = load_json(a.generic_summary)
    if a.generic_base_summary:
        kw["generic_base"] = load_json(a.generic_base_summary)
    if a.corpus_manifest:
        cm = load_json(a.corpus_manifest)
        cm["manifest_sha256"] = __import__("hashlib").sha256(Path(a.corpus_manifest).read_bytes()).hexdigest()
        kw["corpus_manifest"] = cm
    if a.baseline_v1_predictions:
        kw["baseline_v1_rows"] = read_jsonl(a.baseline_v1_predictions)
    if a.t0_predictions:
        kw["t0_rows"] = read_jsonl(a.t0_predictions)
    measures = build_measures(summary, rows, **kw)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(measures, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(measures["metrics"], indent=1, ensure_ascii=False), *measures["notes"], sep="\n")
    if a.gates:
        import check_gates
        doc = check_gates.run_check(TIERS[a.tier]["gate_tier"], ROOT / "configs" / "gates.yaml", check_gates.load_measures(Path(a.out)), None, ("python", "kapilab"))
        gates_path = Path(a.out).with_name("gates.json")
        gates_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
        check_gates.print_table(doc)
        print(f"-> {gates_path}")
    return 0


def cmd_compare(a: argparse.Namespace) -> int:
    """distilado x controle (primario), distilado x base e controle x base, nos MESMOS exemplos; G2.4 do professor."""
    import compare_models

    root = eval_root(a.tier)
    runs = {"distill": a.distill, "gold": a.gold, "base": a.base}
    cmp_dir = root / "compare"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    cmp_files = {}
    for key, name in runs.items():
        if not name:
            continue
        p = cmp_dir / f"{name}.cmp.jsonl"
        with p.open("w", encoding="utf-8", newline="\n") as handle:
            for r in to_cmp(read_jsonl(root / name / "predictions.jsonl")):
                handle.write(json.dumps(r, ensure_ascii=False) + "\n")
        cmp_files[key] = p
    pairs = [("gold", "distill"), ("base", "distill"), ("base", "gold")]
    margins = {"normalized_match": 0.02, "exec_valid": 0.02}
    results = {}
    for x, y in pairs:
        if x in cmp_files and y in cmp_files:
            res = compare_models.run(cmp_files[x], cmp_files[y], group_by=a.group_by, min_group_n=MIN_GROUP_N, margins=margins)
            name = f"{runs[y]}_vs_{runs[x]}"
            (cmp_dir / f"{name}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
            (cmp_dir / f"{name}.md").write_text(compare_models.to_markdown(res), encoding="utf-8")
            results[name] = res
            print(f"-> {cmp_dir / (name + '.md')} (delta = {runs[y]} - {runs[x]})")
    if a.teacher_summary and a.gold:
        out = cmp_dir / "g2_4_teacher_measures.json"
        cmd = [sys.executable, "-I", str(ROOT / "scripts" / "teacher" / "teacher_measures.py"), "--adapter-summary", a.teacher_summary,
               "--base-summary", a.teacher_base_summary, "--student-summary", str(root / a.gold / "summary.json"), "--out", str(out)]
        subprocess.run(cmd, check=True, cwd=ROOT)
        print(f"G2.4: {out} (rode check_gates --tier TEACHER --gates configs/gates.teacher.yaml --measures {out})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tier", choices=list(TIERS), required=True)
    r.add_argument("--model", required=True, help="diretorio do modelo completo (Repos/<repo>/merged/<run> ou BaseRepos/... para o zero-shot)")
    r.add_argument("--name", required=True)
    r.add_argument("--base-name", help="nome (em reports/eval) do zero-shot do mesmo tamanho, para os deltas informativos")
    r.add_argument("--n", type=int, default=2000)
    r.add_argument("--no-generic", action="store_true")
    r.add_argument("--phase", choices=["all", "gen", "measures"], default="all")
    r.add_argument("--force", action="store_true")
    r.add_argument("--baseline-v1-predictions")
    r.add_argument("--t0-predictions")
    r.set_defaults(fn=cmd_run)
    m = sub.add_parser("measures")
    m.add_argument("--tier", choices=list(TIERS), required=True)
    m.add_argument("--summary", required=True)
    m.add_argument("--base-summary")
    m.add_argument("--generic-summary")
    m.add_argument("--generic-base-summary")
    m.add_argument("--corpus-manifest")
    m.add_argument("--baseline-v1-predictions")
    m.add_argument("--t0-predictions")
    m.add_argument("--out", required=True)
    m.add_argument("--gates", action="store_true")
    m.add_argument("--resamples", type=int, default=stats.DEFAULT_RESAMPLES)
    m.set_defaults(fn=cmd_measures)
    c = sub.add_parser("compare")
    c.add_argument("--tier", choices=list(TIERS), required=True)
    c.add_argument("--distill", required=True)
    c.add_argument("--gold", required=True)
    c.add_argument("--base")
    c.add_argument("--group-by", default="context_kind")
    c.add_argument("--teacher-summary", default=str(TEACHER_EVAL / "sft_v2d-final" / "summary.json"))
    c.add_argument("--teacher-base-summary", default=str(TEACHER_EVAL / "base-v2c" / "summary.json"))
    c.set_defaults(fn=cmd_compare)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
