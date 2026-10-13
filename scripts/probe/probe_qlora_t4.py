"""QLoRA curto do Qwen3-8B NF4 a 4k/8k/12k/16k reaproveitando bakeoff/scripts/probe_qlora.py (Fase 05 §6.4c / T4).
Uso: C:/SlopStudio.MongoAI-artifacts/.venv/Scripts/python.exe -I scripts/probe/probe_qlora_t4.py --micro-tokens 4096 8192 12288 16384
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, str(ROOT / "bakeoff" / "scripts"))
sys.path.insert(0, str(ROOT / "bakeoff" / "v1" / "scripts"))
import probe_qlora as pq  # noqa: E402
from compact.env import configure_caches, limit_vram, require_gpu  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "BaseRepos" / "Qwen3-8B"))
    ap.add_argument("--micro-tokens", type=int, nargs="+", default=[4096, 8192, 12288, 16384])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--vram-fraction", type=float, default=0.94)
    ap.add_argument("--reserve-gb", type=float, default=1.6)
    ap.add_argument("--out", default=str(ROOT / "reports" / "probe" / "qlora_t4.json"))
    a = ap.parse_args()
    configure_caches("C:/SlopStudio.MongoAI-artifacts")  # caches/tmp fora do repo
    rows = [r for r in pq.read_jsonl(ROOT / "bakeoff" / "data" / "v1_validation.jsonl")
            if r["type"] in ("fim", "completion", "chat", "correction")]
    gpu = require_gpu("7800 XT")
    gpu["vram_limit"] = limit_vram(gpu, a.vram_fraction, reserve_gb=a.reserve_gb)
    print("VRAM limit:", gpu["vram_limit"], flush=True)
    results = []
    for micro in a.micro_tokens:
        print(f"=== Qwen3-8B micro={micro}", flush=True)
        try:
            r = pq.probe(Path(a.model), micro, a.steps, a.warmup, a.rank, rows, gpu)
        except Exception as e:  # noqa: BLE001
            r = {"micro_tokens": micro, "status": "error", "error": f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"}
        r["vram_limit"] = gpu["vram_limit"]
        print(json.dumps(r, ensure_ascii=False), flush=True)
        results.append(r)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        import torch, gc; gc.collect(); torch.cuda.empty_cache()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
