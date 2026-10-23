#!/usr/bin/env python3
"""Servidor llama-server (Qwen3-14B GGUF) para a geracao: SELECAO DO DISPOSITIVO POR NOME + guarda de velocidade (licao do J12b: um job
ja rodou horas na iGPU por confiar em `Vulkan0`). Reusa parse_devices/select_device/list_devices de scripts/distill/generate_teacher.py.

Confirmacao independente (contadores do Windows): `gpu_confirm(pid)` le `\\GPU Adapter Memory(*)\\Dedicated Usage` e
`\\GPU Engine(*)` do pid; a 7800 XT e o adaptador com GBs de memoria dedicada.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import ROOT, load_cfg

sys.path.insert(0, str(ROOT / "scripts" / "distill"))
from generate_teacher import list_devices, parse_devices, select_device  # noqa: E402,F401


class LlmServer:
    def __init__(self, cfg: dict | None = None, url: str | None = None):
        self.cfg = cfg or load_cfg()
        sv = self.cfg["server"]
        self.sv = sv
        self.proc = None
        self.log_f = None
        self.gpu: dict = {}
        self.url = url or f"http://127.0.0.1:{sv['port']}"
        self.external = url is not None
        self.slots = int(sv["slots"])

    # ---------------------------------------------------------------- ciclo de vida
    def start(self) -> dict:
        if self.external:
            self.gpu = {"sanity": self.sanity_check(), "external": True}
            return self.gpu
        sv = self.sv
        exe = ROOT / sv["exe"]
        devices = list_devices(exe)
        chosen = select_device(devices, sv.get("device_match", "RX 7800 XT"))
        dev = chosen["id"]
        self.gpu = {"device_id": dev, "device_name": chosen["name"], "total_mib": chosen.get("total_mib"), "all_devices": devices}
        print(f"[device] dispositivos: {devices}\n[device] escolhido por NOME: -dev {dev} ({chosen['name']})", flush=True)
        if "graphics" in chosen["name"].lower() and "rx" not in chosen["name"].lower():
            raise SystemExit(f"dispositivo escolhido parece a iGPU: {chosen['name']}")
        cmd = [str(exe), "-m", str(ROOT / sv["gguf"]), "--host", "127.0.0.1", "--port", str(sv["port"]), "-ngl", "99", "-dev", dev,
               "--parallel", str(self.slots), "-c", str(self.slots * int(sv["ctx_per_slot"])), "-fa", "on", "--no-webui", "--jinja",
               "--reasoning-budget", "0"] + list(sv.get("extra") or [])
        try:        # porta ja servindo = servidor orfao/antigo: nunca anexar a ele (dispositivo desconhecido)
            urllib.request.urlopen(self.url + "/health", timeout=1)
            raise SystemExit(f"a porta {sv['port']} ja tem um servidor ativo; encerre-o ou use --server-url")
        except SystemExit:
            raise
        except Exception:  # noqa: BLE001
            pass
        log = ROOT / sv.get("log", "logs/llm_gen/llama-server.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        self.log_f = open(log, "wb")
        t0 = time.time()
        self.proc = subprocess.Popen(cmd, stdout=self.log_f, stderr=subprocess.STDOUT)
        import atexit
        atexit.register(self.close)
        try:
            for _ in range(900):
                try:
                    if urllib.request.urlopen(self.url + "/health", timeout=2).status == 200:
                        break
                except Exception:  # noqa: BLE001
                    if self.proc.poll() is not None:
                        raise SystemExit(f"llama-server morreu; veja {log}")
                    time.sleep(1)
            else:
                raise SystemExit("llama-server nao ficou pronto em 900 s")
            self.gpu["load_seconds"] = round(time.time() - t0, 1)
            self.gpu["sanity"] = self.sanity_check()
        except BaseException:
            self.close()
            raise
        self.gpu["pid"] = self.proc.pid
        return self.gpu

    def sanity_check(self) -> dict:
        """prompt ~300 tokens sem cache; aborta se prompt_per_second < min_prompt_tps (iGPU/CPU)."""
        text = " ".join(f"db.collection{i}.find({{ field{i}: {{ $gt: {i * 7} }} }})" for i in range(40))
        best = {"prompt_per_second": 0.0, "predicted_per_second": 0.0}
        for attempt in range(3):
            body = {"prompt": text + " " * attempt, "n_predict": 24, "temperature": 0.0, "cache_prompt": False, "stream": False}
            req = urllib.request.Request(self.url + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                tm = json.loads(r.read()).get("timings") or {}
            if (tm.get("prompt_per_second") or 0) > best["prompt_per_second"]:
                best = {"prompt_per_second": round(float(tm.get("prompt_per_second") or 0), 1),
                        "predicted_per_second": round(float(tm.get("predicted_per_second") or 0), 1),
                        "prompt_tokens": tm.get("prompt_n")}
        best["threshold_prompt_per_second"] = float(self.sv.get("min_prompt_tps", 300))
        print(f"[device] guarda: {best}", flush=True)
        if best["prompt_per_second"] < best["threshold_prompt_per_second"]:
            raise SystemExit(f"dispositivo lento: provavel iGPU ({best}); veja o log do servidor")
        return best

    def close(self):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except Exception:  # noqa: BLE001
                self.proc.kill()
            self.proc = None
        if self.log_f is not None:
            self.log_f.close()
            self.log_f = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *a):
        self.close()

    # ---------------------------------------------------------------- inferencia
    def chat(self, messages: list[dict], *, seed: int, schema: dict | None = None, max_tokens: int | None = None,
             temperature: float | None = None, timeout: int = 900) -> dict:
        s = self.cfg["sampling"]
        body = {
            "messages": messages, "seed": seed, "stream": False, "cache_prompt": True,
            "temperature": s["temperature"] if temperature is None else temperature,
            "top_p": s["top_p"], "top_k": s.get("top_k", 40), "min_p": s.get("min_p", 0.05),
            "presence_penalty": s.get("presence_penalty", 0.0), "max_tokens": max_tokens or s["max_tokens"],
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "queries", "strict": True, "schema": schema}}
        req = urllib.request.Request(self.url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read())
        dt = time.perf_counter() - t0
        choice = out["choices"][0]
        usage = out.get("usage") or {}
        return {"text": choice["message"].get("content") or "", "finish": choice.get("finish_reason"),
                "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"), "seconds": dt,
                "timings": out.get("timings") or {}}

    def map_chat(self, jobs: list[dict], workers: int | None = None, on_done=None):
        """jobs: [{messages, seed, schema, ...}] -> resultados na mesma ordem (threads = slots)."""
        def one(j):
            try:
                r = self.chat(j["messages"], seed=j["seed"], schema=j.get("schema"), max_tokens=j.get("max_tokens"))
                r["error"] = None
            except Exception as e:  # noqa: BLE001
                r = {"text": "", "error": f"{type(e).__name__}: {e}"[:300], "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}
            if on_done:
                on_done(j, r)
            return r
        with ThreadPoolExecutor(workers or self.slots) as ex:
            return list(ex.map(one, jobs))


# ------------------------------------------------------------------------------------------------------------------------------
def gpu_confirm(pid: int | None, seconds: float = 3.0) -> dict:
    """Contadores do Windows: memoria dedicada por adaptador e engine 3D/compute do `pid` (PowerShell Get-Counter). Retorna um dict
    {adapters: [{luid, dedicated_gb}], engines: [{luid, util}], verdict}. Veredito 'ok' quando o adaptador com mais memoria dedicada
    concentra o uso do processo (a 7800 XT tem GBs; a iGPU quase nada)."""
    ps = (
        "$a=(Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage').CounterSamples|"
        "%{[pscustomobject]@{n=$_.InstanceName;gb=[math]::Round($_.CookedValue/1GB,2)}};"
        f"Start-Sleep -Seconds {int(seconds)};"
        "$e=(Get-Counter '\\GPU Engine(*)\\Utilization Percentage').CounterSamples|"
        f"?{{$_.InstanceName -match 'pid_{pid}_' -and $_.CookedValue -gt 0}}|"
        "%{[pscustomobject]@{n=$_.InstanceName;u=[math]::Round($_.CookedValue,1)}};"
        "@{adapters=@($a);engines=@($e)}|ConvertTo-Json -Depth 4 -Compress"
    )
    res = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60)
    try:
        d = json.loads(res.stdout)
    except ValueError:
        return {"verdict": "indeterminado", "raw": (res.stdout + res.stderr)[:400]}
    ad = d.get("adapters") or []
    if isinstance(ad, dict):
        ad = [ad]
    eng = d.get("engines") or []
    if isinstance(eng, dict):
        eng = [eng]
    import re
    big = max(ad, key=lambda x: x["gb"], default=None)
    luid_big = re.search(r"luid_(0x[0-9a-f]+_0x[0-9a-f]+)", big["n"]).group(1) if big else None
    on_big = sum(e["u"] for e in eng if luid_big and luid_big in e["n"])
    other = sum(e["u"] for e in eng if not (luid_big and luid_big in e["n"]))
    verdict = "ok" if big and big["gb"] >= 4 and on_big > 0 and on_big >= other else "SUSPEITO"
    return {"verdict": verdict, "adapters": ad, "engine_util_on_big_adapter": on_big, "engine_util_elsewhere": other,
            "big_adapter": big}
