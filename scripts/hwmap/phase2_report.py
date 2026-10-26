#!/usr/bin/env python3
"""Agrega reports/hwmap/phase2_raw/*.json em reports/hwmap/phase2_measurements.json (+ tabelas markdown em stdout / --md)."""
from __future__ import annotations

import glob
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "reports" / "hwmap" / "phase2_raw"


def load(pattern: str) -> dict[str, dict]:
    return {Path(p).stem: json.loads(Path(p).read_text(encoding="utf-8")) for p in sorted(glob.glob(str(RAW / pattern)))}


def pct(v, q):
    v = sorted(v)
    if not v:
        return None
    k = (len(v) - 1) * q
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


def lat_group(prefix: str) -> dict:
    """prefix 'lat_t0-dml-fp16_7800xt' -> agrega r1..r3."""
    runs = load(f"{prefix}_r*.json")
    if not runs:
        return {}
    ttft_all, dec_all, forced_all = [], [], []
    per = []
    for name, d in runs.items():
        t = [r["ttft_ms"] for r in d["rows"]]
        ttft_all += t
        dec_all += [r["dec_tps"] for r in d["rows"] if r["dec_tps"]]
        forced_all += d["decode_forced_tps"]
        per.append({"run": name.rsplit("_", 1)[1], "n": len(t), "ttft_mean": round(st.mean(t), 1), "ttft_p95": round(pct(t, 0.95), 1),
                    "dec_tps_mean": round(st.mean([r["dec_tps"] for r in d["rows"] if r["dec_tps"]]), 1),
                    "dec_forced_mean": round(st.mean(d["decode_forced_tps"]), 1) if d["decode_forced_tps"] else None,
                    "load_before_10s": d.get("_gate", {}).get("idle_obs"), "noisy": d.get("_gate", {}).get("noisy"),
                    "psutil_load_before_2s": d.get("machine_load_pct_before"), "psutil_load_after_1s": d.get("machine_load_pct_after")})
    means = [p["ttft_mean"] for p in per]
    d0 = next(iter(runs.values()))
    var = (max(means) - min(means)) / st.mean(means) * 100 if len(means) > 1 else None
    tps = [p["dec_forced_mean"] for p in per if p["dec_forced_mean"]]
    var_tps = (max(tps) - min(tps)) / st.mean(tps) * 100 if len(tps) > 1 else None
    out = {"runs": per, "n_prompts_per_run": [p["n"] for p in per], "ttft_ms_mean": round(st.mean(ttft_all), 1), "ttft_ms_p50": round(st.median(ttft_all), 1),
           "ttft_ms_p95": round(pct(ttft_all, 0.95), 1), "dec_tps_mean": round(st.mean(dec_all), 1), "dec_forced_tps_mean": round(st.mean(forced_all), 1) if forced_all else None,
           "run_to_run_var_ttft_pct": round(var, 1) if var is not None else None, "run_to_run_var_tps_pct": round(var_tps, 1) if var_tps is not None else None,
           "peak_wset_gb": max(d["peak_wset_gb"] for d in runs.values()), "private_gb_max": max(d["private_gb"] for d in runs.values()),
           "rss_after_load_gb": round(st.mean(d["rss_after_load_gb"] for d in runs.values()), 3), "load_s": round(st.mean(d["load_s"] for d in runs.values()), 2),
           "prompt_tokens_median": d0.get("prompt_tokens_median"), "pkg": d0["pkg"], "provider": d0["provider"], "threads": d0.get("threads"),
           "adapter": d0.get("adapter"), "allow_igpu": d0.get("allow_igpu"), "dxgi_hook_calls": d0.get("dxgi_hook_calls")}
    ded = [d.get("vram_dedicated_mb_end") for d in runs.values() if d.get("vram_dedicated_mb_end") is not None]
    sh = [d.get("vram_shared_mb_end") for d in runs.values() if d.get("vram_shared_mb_end") is not None]
    if ded or sh:
        out.update(vram_dedicated_mb_end=max(ded) if ded else None, vram_shared_mb_end=max(sh) if sh else None,
                   mem_on_requested_adapter=all(d.get("mem_on_requested_adapter") for d in runs.values()))
    return out


def observed(d: dict) -> dict:
    """adaptador observado por contadores: LUID com maior pico (sampler) ou, sem sampler, o LUID do probe_end (memoria dedicada+compartilhada do pid)."""
    lm = d.get("luid_map") or {}
    sam = d.get("sampler")
    if sam:
        ded = sam.get("proc_dedicated_mb_max", {})
        sh = sam.get("proc_shared_mb_max", {})
        tot = {l: (ded.get(l) or 0) + (sh.get(l) or 0) for l in set(ded) | set(sh)}
        util = sam.get("proc_engine_util_pct_max", {})
        src = "sampler"
    else:
        pe = d.get("probe_end") or {}
        ded, sh = pe.get("proc_dedicated_mb", {}), pe.get("proc_shared_mb", {})
        tot = {l: (ded.get(l) or 0) + (sh.get(l) or 0) for l in set(ded) | set(sh)}
        util = pe.get("proc_engine_util_pct", {})
        src = "probe_end"
    if not tot:
        return {"observed": None, "source": src}
    top = max(tot, key=tot.get)
    return {"observed": lm.get(top, top), "luid": top, "mem_mb_by_luid": tot, "util_by_luid": util, "source": src}


def adapter_table() -> list[dict]:
    rows = []
    for p in sorted(glob.glob(str(RAW / "*.json"))):
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        if d.get("provider") != "dml" or "adapter_requested" not in d:
            continue
        name = Path(p).stem
        if name.startswith("micro_"):
            continue
        ob = observed(d)
        ev = d.get("evidence")
        rows.append({"run": name, "requested": d.get("adapter_requested"), "requested_name": (d.get("adapter") or {}).get("name"), "observed": ob.get("observed"),
                     "evidence": ev or "pinned_by_arg_only (+probe_end pos-run)" if ob.get("source") == "probe_end" else (ev or "counters"),
                     "adapter_ok": d.get("adapter_ok", d.get("mem_on_requested_adapter")), "mem_mb_by_luid": ob.get("mem_mb_by_luid"), "util_pct_by_luid": ob.get("util_by_luid")})
    return rows


def noisy_summary() -> dict:
    recs = [json.loads(l) for l in (ROOT / "reports/hwmap/phase2_runs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = [r for r in recs if r.get("rc") == 0]
    return {"records": len(recs), "rc0": len(ok), "rc_nonzero": [r["name"] for r in recs if r.get("rc")], "noisy": [r["name"] for r in ok if r.get("noisy")],
            "idle_last_obs_max": max((r["idle_obs"][-1] for r in ok if r.get("idle_obs")), default=None)}


def main() -> None:
    res: dict = {"latency": {}, "verify": load("verify_*.json"), "ctx": load("ctx_*.json"), "parity": load("parity_*.json"), "micro": {}}
    names = sorted({re.sub(r"_r\d$", "", Path(p).stem) for p in glob.glob(str(RAW / "lat_*_r*.json"))})
    for n in names:
        res["latency"][n] = lat_group(n)
    micro = load("micro_*.json")
    groups: dict = defaultdict(list)
    for name, d in micro.items():
        groups[re.sub(r"_\d\d$", "", name)].append(d)
    for g, ds in groups.items():
        ok = [d for d in ds if d.get("ok")]
        t = [d["ttft_ms_p50"] for d in ok]
        r = [d["decode_tok_s"] for d in ok]
        tot = [d["total_s"] for d in ok]
        wall = [d["_wall_s"] for d in ok]
        loads = [d["load_s"] for d in ok]
        res["micro"][g] = {"n_runs": len(ds), "n_ok": len(ok), "ttft_p50_ms": t, "decode_tok_s": r, "total_s": tot, "wall_s": wall, "load_s": loads,
                           "idle_obs": [d.get("_gate", {}).get("idle_obs") for d in ds], "noisy": [d.get("_gate", {}).get("noisy") for d in ds],
                           **({"ttft_mean": st.mean(t), "ttft_sd": st.pstdev(t), "ttft_cv_pct": st.pstdev(t) / st.mean(t) * 100, "ttft_min": min(t), "ttft_max": max(t),
                               "tps_mean": st.mean(r), "tps_sd": st.pstdev(r), "tps_cv_pct": st.pstdev(r) / st.mean(r) * 100, "tps_min": min(r), "tps_max": max(r),
                               "total_s_mean": st.mean(tot), "total_s_max": max(tot), "wall_s_mean": st.mean(wall), "wall_s_max": max(wall), "load_s_mean": st.mean(loads)} if ok else {}),
                           "errors": [d.get("error") for d in ds if not d.get("ok")]}
    res["adapter_table"] = adapter_table()
    res["evid"] = load("evid_*.json")
    res["noisy_summary"] = noisy_summary()
    out = ROOT / "reports" / "hwmap" / "phase2_measurements.json"
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    prev.update(res)
    out.write_text(json.dumps(prev, indent=1), encoding="utf-8")
    print(json.dumps({k: len(v) for k, v in res.items()}))


if __name__ == "__main__":
    sys.exit(main())
