#!/usr/bin/env python3
"""Fase 2 do mapa de GPU: medidas de latencia/paridade/VRAM por adaptador, UM processo por (pacote, provider, adaptador, threads).

Diferente de scripts/export/onnx_run.py (que recusa tudo que nao for a RX 7800 XT), aqui o adaptador DML e escolhido DE PROPOSITO por vendor/device id
(`--adapter 1002:747E` = 7800 XT; `1002:164E` = iGPU) e confirmado por contadores do Windows (gpu_probe.ps1 / vram_sampler.ps1) no LUID certo.
Escolher uma iGPU exige `--allow-igpu` explicito (registrado no JSON). Se o adaptador medido nao for o pedido, sai com codigo 3 e a medida e invalida.
O caminho normal de scripts/export (travas da 7800 XT) NAO e alterado.

venv: CPU = venv ROCm com -I; DML = .venv-winml SEM -I (ver scripts/export/README.md).

  modos: latency (n prompts + decodificacao forcada), parity (100 prompts greedy x HF bf16), verify (contadores durante geracao + pico de VRAM),
         ctx (prefill ~1.900 tokens, decode ate 2.048: pico de VRAM com contexto cheio)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "hwmap"))
WORK_V3 = ROOT / "reports" / "export" / "work_v3"
KNOWN = {"1002:747E": "7800xt", "1002:164E": "igpu"}


def _i(x):
    return x if isinstance(x, int) else int(str(x), 16)


def resolve_adapter(spec: str) -> dict:
    """spec 'VVVV:DDDD' (hex) -> adaptador DXGI desta sessao (indice, LUID, vram). Nunca por indice fixo."""
    from detect_gpus import enumerate_dxgi
    ven, dev = (int(x, 16) for x in spec.split(":"))
    hits = [a for a in enumerate_dxgi() if _i(a["vendor_id"]) == ven and _i(a["device_id"]) == dev]
    if len(hits) != 1:
        raise SystemExit(f"adaptador {spec}: {len(hits)} candidatos (esperado 1)")
    return hits[0]


def probe(pid: int, samples: int = 2) -> dict:
    r = subprocess.run(["powershell", "-NoProfile", "-File", str(ROOT / "scripts/export/gpu_probe.ps1"), "-ProcId", str(pid), "-Samples", str(samples)],
                       capture_output=True, text=True, timeout=180)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        return {"error": (r.stdout + r.stderr)[-400:]}


def start_sampler(pid: int, seconds: int) -> subprocess.Popen:
    return subprocess.Popen(["powershell", "-NoProfile", "-File", str(ROOT / "scripts/hwmap/vram_sampler.ps1"), "-ProcId", str(pid), "-Seconds", str(seconds)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def summarize_sampler(p: subprocess.Popen, timeout: int = 120) -> dict:
    out, err = p.communicate(timeout=timeout)
    rows = []
    for ln in out.splitlines():
        try:
            rows.append(json.loads(ln))
        except Exception:  # noqa: BLE001
            pass
    peak: dict = {"proc_dedicated_mb_max": {}, "proc_shared_mb_max": {}, "proc_engine_util_pct_max": {}, "samples": len(rows), "err": err[-200:] if not rows else ""}
    for r in rows:
        for key, tgt in (("ded", "proc_dedicated_mb_max"), ("sh", "proc_shared_mb_max"), ("util", "proc_engine_util_pct_max")):
            for luid, v in (r.get(key) or {}).items():
                peak[tgt][luid] = max(peak[tgt].get(luid, 0), v)
    return peak


class Runner:
    def __init__(self, og, model, tok, stops):
        self.og, self.model, self.tok, self.stops = og, model, tok, set(stops)

    def gen(self, ids, limit, suffix="", force=False, max_len=None):
        import numpy as np
        og = self.og
        p = og.GeneratorParams(self.model)
        opts = dict(do_sample=False, max_length=max_len or (len(ids) + limit), batch_size=1)
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
            if not force and len(suffix) >= 2 and self.tok is not None and suffix in self.tok.decode(out, skip_special_tokens=False, clean_up_tokenization_spaces=False):
                reason = "suffix"
                break
            g.generate_next_token()
            out.append(int(g.get_next_tokens()[0]))
            if not force and out[-1] in self.stops:
                reason = "eos"
        dec = time.perf_counter() - t1
        del g
        return {"tokens": out, "stop": reason, "ttft": ttft, "dec_s": dec}


def pct(v, q):
    v = sorted(v)
    if not v:
        return None
    k = (len(v) - 1) * q
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--provider", choices=["cpu", "dml"], required=True)
    ap.add_argument("--tier", choices=["t0", "t1"], required=True)
    ap.add_argument("--mode", choices=["latency", "parity", "verify", "ctx"], required=True)
    ap.add_argument("--adapter", help="DML: vendor:device hex (1002:747E = 7800 XT; 1002:164E = iGPU)")
    ap.add_argument("--allow-igpu", action="store_true", help="medir uma iGPU DE PROPOSITO (necessario quando o adaptador e integrado)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--dml-opt", action="append", default=[], help="DML: opcao extra do provider chave=valor (exploracao do seletor de adaptador)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--forced", type=int, default=20, help="prompts da decodificacao forcada de 64 tokens")
    ap.add_argument("--max-seconds", type=float, default=0, help="orcamento de tempo da fase de latencia (0 = sem limite); registra n efetivo")
    ap.add_argument("--evidence", action="store_true", help="liga o amostrador de contadores (GPU Process Memory / GPU Engine por pid) durante latency/parity: adapter_ok medido")
    ap.add_argument("--evidence-seconds", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if os.environ.get("KAPI_WATCHDOG"):
        import faulthandler
        faulthandler.dump_traceback_later(int(os.environ["KAPI_WATCHDOG"]), exit=True)
    import psutil
    prompts = json.loads((WORK_V3 / f"{a.tier}_prompts.json").read_text(encoding="utf-8"))
    items = prompts["items"]
    proc = psutil.Process()
    res: dict = {"pkg": Path(a.pkg).name, "provider": a.provider, "mode": a.mode, "threads": a.threads, "pid": os.getpid(), "allow_igpu": a.allow_igpu,
                 "adapter_requested": a.adapter, "n_req": a.n, "warm": a.warm}
    adapter = None
    if a.provider == "dml":
        if not a.adapter:
            raise SystemExit("--adapter obrigatorio em DML")
        adapter = resolve_adapter(a.adapter)
        if (adapter["dedicated_vram_mb"] < 1024 or KNOWN.get(a.adapter.upper().replace("0X", "")) == "igpu") and not a.allow_igpu:   # iGPU (VRAM dedicada < 1 GB / id conhecido)
            raise SystemExit("adaptador integrado: passe --allow-igpu (medida de proposito)")
        res["adapter"] = {k: adapter.get(k) for k in ("dxgi_index", "name", "luid", "vendor_id", "device_id", "dedicated_vram_mb")}
        other = [x for x in __import__("detect_gpus").enumerate_dxgi() if x["luid"] != adapter["luid"]]
        res["other_luids"] = {x["luid"]: x["name"] for x in other}
        res["luid_map"] = {x["luid"]: f"{_i(x['vendor_id']):04X}:{_i(x['device_id']):04X} {x['name']}" for x in __import__("detect_gpus").enumerate_dxgi()}
    import onnxruntime_genai as og
    res["genai"] = og.__version__
    res["machine_load_pct_before"] = psutil.cpu_percent(interval=2.0)
    t0 = time.perf_counter()
    cfg = og.Config(a.pkg)
    pkg_cfg = json.loads((Path(a.pkg) / "genai_config.json").read_text(encoding="utf-8"))
    no_share = not pkg_cfg.get("search", {}).get("past_present_share_buffer", True)   # MHA: sem graph capture; usa o provider_options do proprio pacote
    res["past_present_share_buffer"] = not no_share
    if not (a.provider == "dml" and no_share):
        cfg.clear_providers()
    if a.provider == "cpu":
        if a.threads:
            cfg.overlay(json.dumps({"model": {"decoder": {"session_options": {"intra_op_num_threads": a.threads}}}}))
    else:
        if not no_share:
            cfg.append_provider("dml")
        res["dml_opts"] = {}
        import dxgi_select
        ven, dev = (int(x, 16) for x in a.adapter.split(":"))
        res["dxgi_hook"] = dxgi_select.install(ven, dev)   # remapeia a enumeracao DXGI so neste processo (o GenAI 0.15.2 ignora device_id)
        for o in a.dml_opt:
            k, v = o.split("=", 1)
            cfg.set_provider_option("dml", k, v)
            res["dml_opts"][k] = v
    model = og.Model(cfg)
    res["load_s"] = round(time.perf_counter() - t0, 2)
    if a.provider == "dml":
        res["dxgi_hook_calls"] = dict(dxgi_select.calls)
    res["rss_after_load_gb"] = round(proc.memory_info().rss / 2**30, 3)
    tok = None
    if a.mode == "parity":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(prompts["model"])
    runner = Runner(og, model, tok, prompts["stops"])
    my_luid = adapter["luid"] if adapter else None

    def finish(code=0):
        mi = proc.memory_info()
        res.update(rss_end_gb=round(mi.rss / 2**30, 3), peak_wset_gb=round(getattr(mi, "peak_wset", mi.rss) / 2**30, 3),
                   private_gb=round(getattr(mi, "private", mi.rss) / 2**30, 3), machine_load_pct_after=psutil.cpu_percent(interval=1.0))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res), encoding="utf-8")
        print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "decode_forced_tps", "items")}))
        if code:
            raise SystemExit(code)

    def adapter_ok(pm: dict, pu: dict) -> bool:
        """processo tem memoria dedicada > 100 MB e engine ativo no LUID pedido e (se houver outros GPUs) ~0 nos demais."""
        mine_mem = pm.get(my_luid, 0) or 0
        others_mem = max([pm.get(l, 0) or 0 for l in res["other_luids"]] + [0])
        others_util = max([pu.get(l, 0) or 0 for l in res["other_luids"]] + [0])
        res["confirm"] = {"mine_mb": mine_mem, "others_mb_max": others_mem, "mine_util": pu.get(my_luid, 0), "others_util_max": others_util}
        return bool(mine_mem > 100 and others_mem <= 50 and others_util <= 1.0)

    ev = {"sam": None}
    if a.evidence and a.provider == "dml" and a.mode in ("latency", "parity"):
        ev["sam"] = start_sampler(os.getpid(), a.evidence_seconds)

    def collect_evidence():
        if ev["sam"] is None:
            return
        pk = summarize_sampler(ev["sam"], timeout=a.evidence_seconds + 90)
        res["sampler"] = pk
        pm, pu = pk["proc_dedicated_mb_max"], pk["proc_engine_util_pct_max"]
        res["vram_dedicated_mb_peak"] = pm.get(my_luid)
        res["vram_shared_mb_peak"] = pk["proc_shared_mb_max"].get(my_luid)
        res["adapter_ok"] = adapter_ok(pm, pu)
        res["evidence"] = "counters_during_run"

    if a.mode == "latency":
        # fase cronometrada SEM sonda (nao perturba o tempo); a confirmacao do adaptador vem no modo verify do mesmo pacote
        sel = items[: a.n + a.warm]
        rows = []
        t_start = time.perf_counter()
        for k, it in enumerate(sel):
            r = runner.gen(it["prompt_ids"], 32, "")
            if k >= a.warm:
                n = len(r["tokens"])
                rows.append({"id": it["id"], "prompt_tokens": len(it["prompt_ids"]), "ttft_ms": r["ttft"] * 1000, "new_tokens": n,
                             "dec_tps": ((n - 1) / r["dec_s"]) if n > 2 and r["dec_s"] > 0 else None})
            if a.max_seconds and time.perf_counter() - t_start > a.max_seconds and len(rows) >= 10:
                break
        res["rows_n"] = len(rows)
        res["timed_phase_s"] = round(time.perf_counter() - t_start, 1)
        dec = []
        t_f = time.perf_counter()
        for it in sel[a.warm:a.warm + a.forced]:
            r = runner.gen(it["prompt_ids"], 64, "", force=True)
            if len(r["tokens"]) >= 32 and r["dec_s"] > 0:
                dec.append((len(r["tokens"]) - 1) / r["dec_s"])
            if a.max_seconds and time.perf_counter() - t_f > a.max_seconds and len(dec) >= 3:
                break
        tt = [r["ttft_ms"] for r in rows]
        dd = [r["dec_tps"] for r in rows if r["dec_tps"]]
        res.update(rows=rows, decode_forced_tps=dec, ttft_mean=st.mean(tt), ttft_p50=st.median(tt), ttft_p95=pct(tt, 0.95), ttft_sd=st.pstdev(tt),
                   dec_tps_mean=st.mean(dd) if dd else None, dec_forced_mean=st.mean(dec) if dec else None, dec_forced_sd=st.pstdev(dec) if len(dec) > 1 else 0.0,
                   prompt_tokens_median=st.median(r["prompt_tokens"] for r in rows))
        collect_evidence()
        if a.provider == "dml":   # sonda NAO cronometrada, depois das medidas: memoria do processo por LUID (a confirmacao com engine esta no modo verify)
            res["probe_end"] = probe(os.getpid(), samples=1)
            pm = res["probe_end"].get("proc_dedicated_mb", {})
            res["vram_dedicated_mb_end"] = pm.get(my_luid)
            res["vram_shared_mb_end"] = res["probe_end"].get("proc_shared_mb", {}).get(my_luid)
            res["mem_on_requested_adapter"] = bool((pm.get(my_luid) or 0) + (res["probe_end"].get("proc_shared_mb", {}).get(my_luid) or 0) > 100
                                                   and max([pm.get(l, 0) or 0 for l in res["other_luids"]] + [0]) < 100)
        finish()

    elif a.mode == "parity":
        n = min(a.n, prompts["n_parity"])
        exact = first = 0
        fracs = []
        t = time.perf_counter()
        for it in items[:n]:
            r = runner.gen(it["prompt_ids"], prompts["max_new"], it["suffix"])
            ref = it["hf_tokens"]
            o = r["tokens"]
            exact += o == ref
            first += o[:1] == ref[:1]
            k = 0
            for x, y in zip(o, ref):
                if x != y:
                    break
                k += 1
            fracs.append(k / len(ref) if ref else 1.0)
        res.update(n=n, greedy_tokens_identical_pct=round(100 * exact / n, 1), first_token_identical_pct=round(100 * first / n, 1),
                   accepted_prefix_frac_mean_pct=round(100 * st.mean(fracs), 1), seconds=round(time.perf_counter() - t, 1))
        collect_evidence()
        finish()

    else:  # verify | ctx
        stop = threading.Event()
        cnt = [0]
        ids_long = []
        for it in items:
            ids_long += it["prompt_ids"]
            if len(ids_long) >= 1900:
                break
        ids_long = ids_long[:1900]

        def work():
            while not stop.is_set():
                if a.mode == "verify":
                    for it in items[:40]:
                        if stop.is_set():
                            break
                        runner.gen(it["prompt_ids"], 32, "")
                        cnt[0] += 1
                else:  # contexto cheio: prefill 1.900 tokens + decode forcado ate 2.048 (max_length)
                    r = runner.gen(ids_long, 2048 - len(ids_long), "", force=True, max_len=2048)
                    res.setdefault("ctx_runs", []).append({"ttft_ms": round(r["ttft"] * 1000, 1), "tokens": len(r["tokens"]), "tok_s": round((len(r["tokens"]) - 1) / r["dec_s"], 1)})
                    cnt[0] += 1
                    if cnt[0] >= 3:
                        return
        seconds = 14 if a.mode == "verify" else 40
        sam = start_sampler(os.getpid(), seconds) if a.provider == "dml" else None
        th = threading.Thread(target=work)
        th.start()
        if a.mode == "ctx":
            th.join(timeout=seconds + 60)
        else:
            time.sleep(seconds - 2)
        stop.set()
        th.join()
        res["generations_during_probe"] = cnt[0]
        if a.provider == "dml":
            pk = summarize_sampler(sam)
            res["sampler"] = pk
            res["probe_end"] = probe(os.getpid(), samples=1)
            pm = pk["proc_dedicated_mb_max"]
            pu = pk["proc_engine_util_pct_max"]
            res["vram_dedicated_mb_peak"] = pm.get(my_luid)
            res["vram_shared_mb_peak"] = pk["proc_shared_mb_max"].get(my_luid)
            res["on_requested_adapter"] = adapter_ok(pm, pu)
            res["adapter_ok"] = res["on_requested_adapter"]
            res["evidence"] = "counters_during_run"
            finish(0 if res["on_requested_adapter"] else 3)
        finish()


if __name__ == "__main__":
    main()
