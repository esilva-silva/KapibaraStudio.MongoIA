"""Sonda de atencao T4 (Fase 05 §6.4c): SDPA no gfx1101/ROCm com head_dim/heads do Qwen3-8B, fwd+bwd, bf16, batch 1, causal.

Uso (venv ROCm do bake-off):
  C:/SlopStudio.MongoAI-artifacts/.venv/Scripts/python.exe -I scripts/probe/probe_attention.py --out reports/probe/attention_t4.json
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def run(backend, L, heads, kv, hd, gqa_native, reps, torch, F, SDPBackend, sdpa_kernel):
    dev = "cuda"
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    q = torch.randn(1, heads, L, hd, device=dev, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, kv, L, hd, device=dev, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, kv, L, hd, device=dev, dtype=torch.bfloat16, requires_grad=True)
    rep = heads // kv

    def step():
        if gqa_native:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        else:
            o = F.scaled_dot_product_attention(q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1), is_causal=True)
        o.backward(torch.ones_like(o))
    with sdpa_kernel(backend):
        step(); torch.cuda.synchronize()          # warmup
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        for _ in range(reps):
            step()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / reps
    peak = (torch.cuda.max_memory_allocated() - base) / 2**30
    del q, k, v
    return {"ms_fwd_bwd": round(dt * 1000, 2), "peak_gb_over_inputs": round(peak, 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "reports" / "probe" / "attention_t4.json"))
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192, 12288, 16384])
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    cfg = json.loads((ROOT / "BaseRepos" / "Qwen3-8B" / "config.json").read_text())
    heads, kv, hd = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    props = torch.cuda.get_device_properties(0)
    res = {"torch": torch.__version__, "hip": torch.version.hip, "device": torch.cuda.get_device_name(0),
           "arch": getattr(props, "gcnArchName", None), "vram_gb": round(props.total_memory / 2**30, 2),
           "aotriton_experimental_env": os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"),
           "heads": heads, "kv_heads": kv, "head_dim": hd, "layers": cfg["num_hidden_layers"],
           "flags": {"flash": torch.backends.cuda.flash_sdp_enabled(), "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                     "math": torch.backends.cuda.math_sdp_enabled()}, "results": []}
    backends = {"FLASH": SDPBackend.FLASH_ATTENTION, "EFFICIENT": SDPBackend.EFFICIENT_ATTENTION, "MATH": SDPBackend.MATH}
    for name, be in backends.items():
        for L in a.lengths:
            for gqa in (False, True):
                if name == "MATH" and gqa:
                    continue
                row = {"backend": name, "L": L, "gqa_native": gqa}
                try:
                    row.update(run(be, L, heads, kv, hd, gqa, a.reps, torch, F, SDPBackend, sdpa_kernel)); row["status"] = "ok"
                except torch.OutOfMemoryError as e:
                    row.update(status="oom", error=str(e).splitlines()[0][:160])
                except Exception as e:  # noqa: BLE001 - backend indisponivel
                    row.update(status="error", error=f"{type(e).__name__}: {str(e).splitlines()[0][:200]}")
                print(json.dumps(row), flush=True)
                res["results"].append(row)
                torch.cuda.empty_cache()
    # torch.profiler nao expoe atividade GPU no ROCm/Windows (AssertionError); o despacho e inferido por escala de memoria:
    # kernel fused => pico O(L) (~0,4 GB a 4k, ~1,5 GB a 16k); math => O(L^2) (8,4 GB a 4k, 32,7 GB a 8k, OOM >= 12k).
    ok = {(r["backend"], r["L"]): r for r in res["results"] if r["status"] == "ok" and not r["gqa_native"]}
    fl, ma = ok.get(("FLASH", 4096)), ok.get(("MATH", 4096))
    res["inferred_fused_kernel"] = bool(fl and ma and fl["peak_gb_over_inputs"] * 5 < ma["peak_gb_over_inputs"])
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
