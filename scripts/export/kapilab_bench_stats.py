#!/usr/bin/env python3
"""Resume os JSONL do `kapilab bench autocomplete` (kapilab-autocomplete-benchmark-sample-v3) -> reports/export/kapilab_bench_summary.json.
TTFT = runtimeTtftMilliseconds (do runtime GenAI dentro do caminho real da IDE); latencia observada = observedLatencyMilliseconds (inclui geracao ate 32 tokens)."""
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WS = ROOT / "tmp" / "kapilab_ws" / "tmp"


def p95(v):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(0.95 * (len(v) - 1))))]


out = {}
for f in sorted(WS.glob("bench_t*_cpu.jsonl")):
    rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if r.get("schema", "").startswith("kapilab-autocomplete-benchmark-sample")]
    t = [r["runtimeTtftMilliseconds"] for r in rows if r.get("runtimeTtftMilliseconds") is not None]
    ob = [r["observedLatencyMilliseconds"] for r in rows if r.get("observedLatencyMilliseconds") is not None]
    tps = [r["runtimeTokensPerSecond"] for r in rows if r.get("runtimeTokensPerSecond")]
    key = f.stem.replace("bench_", "").replace("_cpu", "")
    out[key] = {"n": len(rows), "outcomes": {o: sum(r.get("outcome") == o for r in rows) for o in sorted({r.get("outcome") for r in rows})},
                "used_fallback": any(r.get("usedFallback") for r in rows), "providers": sorted({str(r.get("provider")) for r in rows}),
                "ttft_mean_ms": round(st.mean(t), 1), "ttft_median_ms": round(st.median(t), 1), "ttft_p95_ms": round(p95(t), 1),
                "observed_mean_ms": round(st.mean(ob), 1), "observed_p95_ms": round(p95(ob), 1), "tok_s_mean": round(st.mean(tps), 1) if tps else None}
(ROOT / "reports" / "export" / "kapilab_bench_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
for k, v in out.items():
    print(k, v["n"], v["outcomes"], f"TTFT {v['ttft_mean_ms']}/{v['ttft_p95_ms']} ms obs {v['observed_mean_ms']}/{v['observed_p95_ms']} tok/s {v['tok_s_mean']} fb={v['used_fallback']} {v['providers']}")
