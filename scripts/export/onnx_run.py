#!/usr/bin/env python3
"""Executa um pacote ONNX (onnxruntime-genai 0.15.2) sobre os prompts de reports/export/work/<tier>_prompts.json. Modos:
  quality   : 500 prompts, greedy ate 96 tokens com as mesmas paradas do Generator do eval_teacher (tokens de parada do contrato, sufixo) -> tokens
  latency   : PROXY do harness da IDE: aquecimento (3 descartados) + N prompts (max 32 tokens, paradas do contrato), TTFT/tok/s, RAM de pico,
              + decodificacao forcada de 64 tokens (min_length) para tok/s sustentado; DML: confere a GPU (LUID 7800 XT) com contadores
  verify    : DML: geracao em thread enquanto a sonda le memoria dedicada / utilizacao por LUID; aborta (exit 3) se nao for a 7800 XT
  tokenizer : tokenizador nativo do GenAI x HF nos prompts (ids identicos)
venv: CPU = venv ROCm (-I); DML = .venv-winml (SEM -I; sitecustomize carrega o onnxruntime.dll do pacote WinML; nao importe `onnxruntime`).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RTX = "0x00000000_0x00012951"   # LUID da RX 7800 XT nesta sessao (confirmado com alocacao torch-ROCm; muda por boot -> --luid)
IGPU = "0x00000000_0x00014bfc"
PAD_FROM = 151665                # ids >= len(tokenizer) sao padding do vocab (HF os mascara)


def gpu_probe(pid: int, samples: int = 1) -> dict:
    r = subprocess.run(["powershell", "-NoProfile", "-File", str(ROOT / "scripts/export/gpu_probe.ps1"), "-ProcId", str(pid), "-Samples", str(samples)],
                       capture_output=True, text=True, timeout=120)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {"error": (r.stdout + r.stderr)[-400:]}


def open_model(pkg: str, provider: str, device_id: int | None, threads: int | None):
    import onnxruntime_genai as og
    cfg = og.Config(pkg)
    cfg.clear_providers()
    if provider == "cpu":
        if threads:
            cfg.overlay(json.dumps({"model": {"decoder": {"session_options": {"intra_op_num_threads": threads}}}}))
    else:
        cfg.append_provider(provider)
        if device_id is not None:
            cfg.set_provider_option(provider, "device_id", str(device_id))
    return og, og.Model(cfg)


class Runner:
    def __init__(self, og, model, tok_hf, stops):
        self.og, self.model, self.tok, self.stops = og, model, tok_hf, set(stops)

    def gen(self, ids, limit, suffix="", force=False):
        import numpy as np
        og = self.og
        p = og.GeneratorParams(self.model)
        opts = dict(do_sample=False, max_length=len(ids) + limit, batch_size=1)
        if force:
            opts["min_length"] = len(ids) + limit
        p.set_search_options(**opts)
        g = og.Generator(self.model, p)
        t = time.perf_counter()
        g.append_tokens(np.array(ids, dtype=np.int32))
        g.generate_next_token()
        ttft = time.perf_counter() - t
        out = [int(g.get_next_tokens()[0])]
        reason = "eos" if (not force and out[-1] in self.stops) else "max"
        t1 = time.perf_counter()
        while reason == "max" and len(out) < limit and not g.is_done():
            if not force and len(suffix) >= 2 and suffix in self.tok.decode(out, skip_special_tokens=False, clean_up_tokenization_spaces=False):
                reason = "suffix"
                break
            g.generate_next_token()
            out.append(int(g.get_next_tokens()[0]))
            if not force and out[-1] in self.stops:
                reason = "eos"
        dec = time.perf_counter() - t1
        del g
        return {"tokens": out, "stop": reason, "ttft": ttft, "dec_s": dec}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--provider", choices=["cpu", "dml"], default="cpu")
    ap.add_argument("--tier", choices=["t0", "t1"], required=True)
    ap.add_argument("--mode", choices=["quality", "latency", "verify", "tokenizer"], required=True)
    ap.add_argument("--device-id", type=int, help="DML: indice do adaptador (DXGI); a verificacao por contadores decide se serve")
    ap.add_argument("--threads", type=int, default=0, help="CPU: intra_op_num_threads (0 = padrao da IDE)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--luid", default=RTX)
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    prompts = json.loads((ROOT / "reports/export/work" / f"{a.tier}_prompts.json").read_text(encoding="utf-8"))
    import psutil
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(prompts["model"])
    proc = psutil.Process()
    res: dict = {"pkg": a.pkg, "provider": a.provider, "mode": a.mode, "device_id": a.device_id, "threads": a.threads, "pid": os.getpid()}
    res["machine_load_pct_before"] = psutil.cpu_percent(interval=2.0)
    t0 = time.perf_counter()
    og, model = open_model(a.pkg, a.provider, a.device_id, a.threads)
    res["load_s"] = round(time.perf_counter() - t0, 2)
    res["genai"] = getattr(og, "__version__", "")
    res["rss_after_load_gb"] = round(proc.memory_info().rss / 2**30, 3)
    runner = Runner(og, model, tok, prompts["stops"])
    items = prompts["items"]

    if a.mode == "tokenizer":
        t = og.Tokenizer(model)
        bad = []
        for it in items:
            enc = [int(x) for x in t.encode(it["prompt_text"])]
            if enc != it["prompt_ids"]:
                bad.append(it["id"])
        res.update(n=len(items), mismatches=len(bad), mismatch_ids=bad[:10], identical_pct=round(100 * (1 - len(bad) / len(items)), 2))

    elif a.mode == "quality":
        out, t = [], time.perf_counter()
        for k, it in enumerate(items):
            r = runner.gen(it["prompt_ids"], prompts["max_new"], it["suffix"])
            out.append({"id": it["id"], "tokens": r["tokens"], "stop": r["stop"], "ttft_ms": round(r["ttft"] * 1000, 1)})
            if (k + 1) % 100 == 0:
                print(k + 1, f"{time.perf_counter() - t:.0f}s", flush=True)
        res["items"] = out
        res["seconds"] = round(time.perf_counter() - t, 1)
        res["padding_token_outputs"] = sum(any(x >= PAD_FROM and x not in prompts["stops"] for x in o["tokens"]) for o in out)

    elif a.mode == "verify":
        stop = threading.Event()
        n_done = [0]

        def work():
            while not stop.is_set():
                for it in items[:40]:
                    if stop.is_set():
                        break
                    runner.gen(it["prompt_ids"], 32, "")
                    n_done[0] += 1
        th = threading.Thread(target=work)
        th.start()
        time.sleep(4)
        probe = gpu_probe(os.getpid(), samples=3)
        stop.set()
        th.join()
        res["probe"] = probe
        pd, pu = probe.get("proc_dedicated_mb", {}), probe.get("proc_engine_util_pct", {})
        res["on_7800xt"] = bool(pd.get(a.luid, 0) > 100 and pu.get(a.luid, 0) > 0 and pd.get(IGPU, 0) < 100 and pu.get(IGPU, 0) == 0)
        res["generations_during_probe"] = n_done[0]

    else:  # latency
        warm = 3
        sel = items[: a.n + warm]
        rows = []
        for k, it in enumerate(sel):
            r = runner.gen(it["prompt_ids"], 32, "")
            if k >= warm:
                n = len(r["tokens"])
                rows.append({"id": it["id"], "prompt_tokens": len(it["prompt_ids"]), "ttft_ms": r["ttft"] * 1000, "new_tokens": n,
                             "dec_tps": ((n - 1) / r["dec_s"]) if n > 2 and r["dec_s"] > 0 else None})
        dec = []
        for it in sel[warm:warm + 20]:   # decodificacao sustentada: 64 tokens forcados (min_length)
            r = runner.gen(it["prompt_ids"], 64, "", force=True)
            if len(r["tokens"]) >= 32 and r["dec_s"] > 0:
                dec.append((len(r["tokens"]) - 1) / r["dec_s"])
        mi = proc.memory_info()
        res.update(rows=rows, decode_forced_tps=dec, rss_end_gb=round(mi.rss / 2**30, 3),
                   peak_wset_gb=round(getattr(mi, "peak_wset", mi.rss) / 2**30, 3), private_gb=round(getattr(mi, "private", mi.rss) / 2**30, 3),
                   machine_load_pct_after=psutil.cpu_percent(interval=1.0))
        if a.provider == "dml":
            res["probe_end"] = gpu_probe(os.getpid(), samples=1)
            pd = res["probe_end"].get("proc_dedicated_mb", {})
            res["vram_dedicated_mb_7800xt"] = pd.get(a.luid)
            res["vram_dedicated_mb_igpu"] = pd.get(IGPU)
            res["on_7800xt"] = bool((pd.get(a.luid) or 0) > 100 and (pd.get(IGPU) or 0) < 100)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k not in ("items", "rows", "decode_forced_tps")}))
    if a.mode in ("verify", "latency") and a.provider == "dml" and not res.get("on_7800xt"):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
