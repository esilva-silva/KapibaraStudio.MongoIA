#!/usr/bin/env python3
"""Pipeline de volume (J18 treino / J19 homologacao): geracao LLM em rodadas + validacao por execucao do texto em paralelo (CPU/WSL) + FIM.

  env/.venv/Scripts/python.exe scripts/data/llm/run_pipeline.py --role train --accepted 60000 --out data/generated/v3/train --resume
  env/.venv/Scripts/python.exe scripts/data/llm/run_pipeline.py --role hom   --accepted 9000  --out data/homolog/_work/v1 --resume

Cada rodada (`--round-size` consultas brutas, padrao 4000) gera com um UNICO llama-server (dispositivo por nome + guarda de velocidade +
confirmacao por contadores do Windows na 1a rodada), e a rodada anterior e validada em thread enquanto a GPU gera a seguinte. Estado em
<out>/pipeline_state.json (retomavel; rodadas completas nao sao refeitas) e heartbeat em <out>/heartbeat.json (watchdog da fila).
Dedup/forma sao GLOBAIS entre rodadas. Ao atingir `--accepted`: concatena `queries.validated.jsonl` e, para `train`, roda llm_to_fim (FIM + contexto +
decontaminacao). Para `hom`, o fechamento e `build_homolog.py` (selo), chamado aqui com --finalize-hom.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import Counter
from pathlib import Path

from common import ROOT, load_cfg, load_schema_infos, query_key, read_jsonl, rel, write_jsonl

STATE = "pipeline_state.json"


def load_state(out: Path) -> dict:
    p = out / STATE
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"rounds_gen": [], "rounds_val": [], "accepted": 0, "raw": 0, "started": time.time()}


def save_state(out: Path, st: dict):
    (out / STATE).write_text(json.dumps(st, indent=1), encoding="utf-8")


_HB_LOCK = threading.Lock()


def heartbeat(out: Path, **kw):
    """Escrita atomica (geracao e validacao escrevem de threads diferentes; o watchdog le o arquivo)."""
    with _HB_LOCK:
        tmp = out / "heartbeat.json.tmp"
        tmp.write_text(json.dumps({"ts": time.time(), **kw}), encoding="utf-8")
        for _ in range(5):
            try:
                tmp.replace(out / "heartbeat.json")
                return
            except OSError:
                time.sleep(0.2)
        (out / "heartbeat.json").write_text(json.dumps({"ts": time.time(), **kw}), encoding="utf-8")


def run(a) -> int:
    import gen_llm_queries as G
    import validate_text as V
    import homolog_guard as hg
    from llm_client import LlmServer
    if a.role == "train":
        hg.enforce()          # o treino nunca le data/homolog (so leitura autorizada e registrada na decontaminacao)
    cfg = load_cfg(a.config)
    seed0 = int(a.seed if a.seed is not None else cfg["seed"])
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    st = load_state(out) if a.resume else {"rounds_gen": [], "rounds_val": [], "accepted": 0, "raw": 0, "started": time.time()}
    infos = load_schema_infos(roles=(a.role,))
    if not infos:
        print(f"[erro] sem schemas {a.role} semeados", file=sys.stderr)
        return 2
    # estado global de dedup/forma a partir das rodadas ja validadas
    seen, shapes = set(), Counter()
    from common import shape_key
    for r in st["rounds_val"]:
        for q in read_jsonl(out / "rounds" / r / "queries.validated.jsonl"):
            seen.add(q["qkey"])
            shapes[(q["schema_id"], q["collection"], shape_key(q["text"]))] += 1
    srv = LlmServer(cfg, url=a.server_url)
    val_thread: threading.Thread | None = None
    val_err: list = []

    def validate_round(name: str):
        try:
            d = out / "rounds" / name
            def prog(p, _n=name):
                # heartbeat por lote validado (o watchdog enxerga progresso); `accepted` = aceitas consolidadas + aprovadas na execucao nesta rodada (provisorio)
                heartbeat(out, phase="validate", round=_n, accepted=st["accepted"], accepted_provisional=st["accepted"] + p.get("exec_ok", 0), **{
                    k: v for k, v in p.items() if k in ("stage", "batches_done", "batches_total", "exec_ok", "exec_resumed", "queries")})

            heartbeat(out, phase="validate", round=name, accepted=st["accepted"], stage="start")
            summ = V.validate(d / "queries.raw.jsonl", d, cfg, workers=a.workers, seen_keys=seen, infos=infos, shape_seen=shapes, progress=prog)
            for q in read_jsonl(d / "queries.validated.jsonl"):
                seen.add(q["qkey"])
            st["rounds_val"].append(name)
            st["accepted"] += summ["accepted"]
            st["raw"] += summ["funnel"]["raw"]
            save_state(out, st)
            heartbeat(out, phase="validated", round=name, accepted=st["accepted"], round_accepted=summ["accepted"])
            print(f"[pipeline] rodada {name} validada: aceitas={summ['accepted']} ({100 * summ['acceptance_rate']:.1f}%) total_aceitas={st['accepted']}", flush=True)
        except BaseException as e:  # noqa: BLE001
            val_err.append(e)

    try:
        srv.start()
        r = len(st["rounds_gen"])
        # rodadas geradas e ainda nao validadas (retomada)
        pending = [x for x in st["rounds_gen"] if x not in st["rounds_val"]]
        for name in pending:
            validate_round(name)
            if val_err:
                raise val_err[0]
        t_run = time.time()
        while st["accepted"] < a.accepted and not val_err:
            name = f"r{r:04d}"
            d = out / "rounds" / name
            ns = argparse.Namespace(config=a.config, role=a.role, target=a.round_size, out=rel(d), seed=seed0 + r * 7919, schemas=None, limit_schemas=None,
                                    tasks=None, per_call=None, server_url=None, resume=True, call_prefix=f"{name}.",
                                    on_progress=lambda stt, _r=r: heartbeat(out, round=_r, phase="gen", accepted=st["accepted"], round_queries=stt.get("queries"),
                                                                           completion_tokens=stt.get("completion_tokens")))
            rc = G.run(ns, srv=srv)
            if rc:
                return rc
            st["rounds_gen"].append(name)
            save_state(out, st)
            if val_thread is not None:
                val_thread.join()
            val_thread = threading.Thread(target=validate_round, args=(name,), daemon=True)
            val_thread.start()
            heartbeat(out, round=r, accepted=st["accepted"], raw=st["raw"], phase="gen")
            r += 1
            if a.max_rounds and r >= a.max_rounds:
                break
            # se a validacao ficou muito atras, espera (evita fila de rodadas sem validar)
            while val_thread.is_alive() and len(st["rounds_gen"]) - len(st["rounds_val"]) > 1:
                heartbeat(out, round=r, accepted=st["accepted"], raw=st["raw"], phase="wait_validate")
                time.sleep(5)
        if val_thread is not None:
            val_thread.join()
        if val_err:
            raise val_err[0]
    finally:
        srv.close()
    # consolida
    acc = [q for name in st["rounds_val"] for q in read_jsonl(out / "rounds" / name / "queries.validated.jsonl")]
    write_jsonl(out / "queries.validated.jsonl", acc)
    save_state(out, st)
    heartbeat(out, phase="done", accepted=len(acc))
    summ = {"role": a.role, "accepted": len(acc), "raw": st["raw"], "rounds": len(st["rounds_val"]), "acceptance_rate": round(len(acc) / max(1, st["raw"]), 4),
            "wall_hours": round((time.time() - st["started"]) / 3600, 2)}
    (out / "pipeline_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ))
    if a.role == "hom" and not a.no_fim:
        import build_homolog as B
        ns = argparse.Namespace(config=a.config, queries=[rel(out / "queries.validated.jsonl")], grammar=a.grammar, n=a.n, cuts_per_snippet=30,
                                max_per_query=14, seed=a.seed, dry_run=False, train_schemas=False)
        return B.run(ns)
    if a.role == "train" and not a.no_fim:
        import llm_to_fim as L
        stf = L.build(acc, "train", out, cfg)
        print(json.dumps({k: stf[k] for k in ("snippets", "cuts_generated", "cuts_final", "none_or_irrelevant_share")}))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config")
    ap.add_argument("--role", choices=["train", "hom"], required=True)
    ap.add_argument("--accepted", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--round-size", type=int, default=4000)
    ap.add_argument("--max-rounds", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--server-url")
    ap.add_argument("--no-fim", action="store_true")
    ap.add_argument("--grammar", type=int, default=0, help="hom: snippets de gramatica extras no conjunto de homologacao")
    ap.add_argument("--n", type=int, help="hom: no. de cortes do conjunto (padrao homolog.min_cuts)")
    ap.add_argument("--resume", action="store_true")
    sys.exit(run(ap.parse_args(argv)))


if __name__ == "__main__":
    main()
