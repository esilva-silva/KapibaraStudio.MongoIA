#!/usr/bin/env python3
"""Detecta os adaptadores graficos (Windows, sem instalar nada), aplica configs/gpu_compat.yaml e imprime em JSON
o adaptador escolhido e a variante recomendada (+ fallback, motivo, confianca).

Enumeracao: DXGI via ctypes (IDXGIFactory1::EnumAdapters1) - LUID, vendor/device id, VRAM dedicada/compartilhada, flag de software;
driver: IDXGIAdapter::CheckInterfaceSupport (versao do UMD) + Win32_VideoController (PowerShell/CIM) como complemento/fallback.
O adaptador e escolhido por vendor id / device id / VRAM dedicada, NUNCA pelo indice DXGI (a ordem varia por sessao).
NAO usa a GPU (so enumera) e NAO escreve em lugar nenhum.

Uso:
  python scripts/hwmap/detect_gpus.py                       # JSON no stdout
  python scripts/hwmap/detect_gpus.py --simulate caso.json  # {"adapters":[...], "cpu":{"physical_cores":12,"ram_gb":32}}
  python scripts/hwmap/detect_gpus.py --list-only           # so a enumeracao bruta
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "gpu_compat.yaml"
SCHEMA = 1

# ------------------------------------------------------------------ enumeracao (DXGI via ctypes)

class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class _DESC1(ctypes.Structure):
    _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", wintypes.UINT), ("DeviceId", wintypes.UINT),
                ("SubSysId", wintypes.UINT), ("Revision", wintypes.UINT), ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t), ("SharedSystemMemory", ctypes.c_size_t),
                ("AdapterLuid", _LUID), ("Flags", wintypes.UINT)]


class _GUID(ctypes.Structure):
    _fields_ = [("D1", wintypes.DWORD), ("D2", wintypes.WORD), ("D3", wintypes.WORD), ("D4", ctypes.c_ubyte * 8)]

    @classmethod
    def parse(cls, s: str) -> "_GUID":
        import uuid
        u = uuid.UUID(s)
        g = cls()
        g.D1, g.D2, g.D3 = u.time_low, u.time_mid, u.time_hi_version
        for i, b in enumerate(bytes([u.clock_seq_hi_variant, u.clock_seq_low]) + u.node.to_bytes(6, "big")):
            g.D4[i] = b
        return g


def _vcall(obj: ctypes.c_void_p, index: int, restype, *args):
    """Chama o metodo `index` da vtable COM de `obj`; args = [(ctype, valor), ...]."""
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    proto = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *[t for t, _ in args])
    return proto(vtbl[index])(obj, *[v for _, v in args])


def _ver_str(v: int) -> str:
    return f"{(v >> 48) & 0xFFFF}.{(v >> 32) & 0xFFFF}.{(v >> 16) & 0xFFFF}.{v & 0xFFFF}"


def enumerate_dxgi() -> list[dict]:
    if os.name != "nt":
        return []
    dxgi = ctypes.WinDLL("dxgi")
    fac = ctypes.c_void_p()
    iid = _GUID.parse("770aae78-f26f-4dba-a829-253c83d1b387")  # IDXGIFactory1
    dxgi.CreateDXGIFactory1.argtypes = [ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)]
    if dxgi.CreateDXGIFactory1(ctypes.byref(iid), ctypes.byref(fac)) != 0:
        return []
    out: list[dict] = []
    iid_dev = _GUID.parse("54ec77fa-1377-44e6-8c32-88fd5f44c84c")  # IDXGIDevice (para a versao do UMD)
    try:
        i = 0
        while True:
            ad = ctypes.c_void_p()
            hr = _vcall(fac, 12, ctypes.c_long, (wintypes.UINT, i), (ctypes.POINTER(ctypes.c_void_p), ctypes.byref(ad)))  # EnumAdapters1
            if hr != 0 or not ad:
                break
            d = _DESC1()
            _vcall(ad, 10, ctypes.c_long, (ctypes.POINTER(_DESC1), ctypes.byref(d)))  # GetDesc1
            umd = ctypes.c_int64()
            hr2 = _vcall(ad, 9, ctypes.c_long, (ctypes.POINTER(_GUID), ctypes.byref(iid_dev)), (ctypes.POINTER(ctypes.c_int64), ctypes.byref(umd)))
            luid = f"0x{d.AdapterLuid.HighPart & 0xFFFFFFFF:08x}_0x{d.AdapterLuid.LowPart:08x}"
            out.append({
                "dxgi_index": i,  # apenas informativo: NUNCA usar para escolher
                "name": d.Description, "vendor_id": int(d.VendorId), "device_id": int(d.DeviceId),
                "subsys_id": int(d.SubSysId), "revision": int(d.Revision), "luid": luid,
                "dedicated_vram_mb": int(d.DedicatedVideoMemory // (1024 * 1024)),
                "dedicated_system_mb": int(d.DedicatedSystemMemory // (1024 * 1024)),
                "shared_mb": int(d.SharedSystemMemory // (1024 * 1024)),
                "is_software": bool(d.Flags & 2) or d.VendorId == 0x1414,
                "driver_version_umd": _ver_str(umd.value) if hr2 == 0 else None,
            })
            _vcall(ad, 2, ctypes.c_ulong)  # Release
            i += 1
    finally:
        _vcall(fac, 2, ctypes.c_ulong)
    return out


def _powershell_json(script: str, timeout: int = 40):
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                           capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
        return json.loads(r.stdout) if r.stdout.strip() else None
    except Exception:
        return None


def _as_list(x):
    return [] if x is None else (x if isinstance(x, list) else [x])


_PS = ("$v=Get-CimInstance Win32_VideoController|Select Name,PNPDeviceID,DriverVersion,AdapterRAM;"
       "$c=Get-CimInstance Win32_Processor|Select Name,NumberOfCores,NumberOfLogicalProcessors;"
       "@{video=@($v);cpu=@($c)}|ConvertTo-Json -Depth 4 -Compress")


def _wmi_snapshot() -> dict:
    return _powershell_json(_PS) or {"video": [], "cpu": []}


def _parse_pnp(pnp: str):
    m = re.search(r"VEN_([0-9A-F]{4})&DEV_([0-9A-F]{4})", pnp or "", re.I)
    return (int(m.group(1), 16), int(m.group(2), 16)) if m else (None, None)


def adapters_from_wmi(wmi: dict) -> list[dict]:
    """Fallback quando o DXGI falha: sem LUID e sem VRAM correta (AdapterRAM e de 32 bits)."""
    out = []
    for v in _as_list(wmi.get("video")):
        ven, dev = _parse_pnp(v.get("PNPDeviceID", ""))
        if ven is None:
            ven = 0x1414 if "Basic Render" in (v.get("Name") or "") else 0
            dev = dev or 0
        out.append({"dxgi_index": None, "name": v.get("Name") or "", "vendor_id": ven, "device_id": dev, "subsys_id": None,
                    "revision": None, "luid": None, "dedicated_vram_mb": int((v.get("AdapterRAM") or 0) // (1024 * 1024)),
                    "dedicated_system_mb": 0, "shared_mb": 0, "is_software": ven == 0x1414,
                    "driver_version_umd": None, "source": "wmi"})
    return out


def detect_cpu(wmi: dict | None) -> dict:
    ram_gb = None
    try:
        class MS(ctypes.Structure):
            _fields_ = [("l", wintypes.DWORD), ("load", wintypes.DWORD), ("tp", ctypes.c_uint64), ("ap", ctypes.c_uint64),
                        ("tpf", ctypes.c_uint64), ("apf", ctypes.c_uint64), ("tv", ctypes.c_uint64), ("av", ctypes.c_uint64), ("ae", ctypes.c_uint64)]
        m = MS(); m.l = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        ram_gb = round(m.tp / 1024 ** 3, 1)
    except Exception:
        pass
    cores = None
    name = None
    if wmi:
        cs = _as_list(wmi.get("cpu"))
        if cs:
            cores = sum(int(c.get("NumberOfCores") or 0) for c in cs) or None
            name = cs[0].get("Name")
    logical = os.cpu_count() or 0
    return {"name": name, "physical_cores": cores or max(1, logical // 2), "logical_cores": logical, "ram_gb": ram_gb,
            "physical_cores_source": "wmi" if cores else "estimado(logico/2)"}


def merge_driver(adapters: list[dict], wmi: dict | None) -> None:
    if not wmi:
        return
    by = {}
    for v in _as_list(wmi.get("video")):
        by.setdefault(_parse_pnp(v.get("PNPDeviceID", "")), v.get("DriverVersion"))
    for a in adapters:
        a["driver_version"] = by.get((a["vendor_id"], a["device_id"])) or a.get("driver_version_umd")
        a.pop("driver_version_umd", None)


# ------------------------------------------------------------------ regras

def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    import yaml
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg.get("schema_version") != SCHEMA:
        raise SystemExit(f"schema_version {cfg.get('schema_version')} != {SCHEMA}")
    return cfg


def match_rule(a: dict, cfg: dict) -> dict | None:
    """Primeira regra do mesmo vendor cujo device_id esta numa faixa OU cujo nome casa o regex. Software e tratado primeiro."""
    for r in cfg["rules"]:
        if r["vendor_id"] != a["vendor_id"]:
            continue
        m = r.get("match", {})
        for lo, hi in m.get("device_id_ranges", []):
            if lo <= a["device_id"] <= hi:
                return r
        rx = m.get("name_regex")
        if rx and re.search(rx, a["name"], re.I):
            return r
    return None


def resolve_variant(rule: dict, a: dict, th: dict) -> tuple[str, str]:
    """-> (variante GPU ou 'cpu', motivo)."""
    gv = rule["gpu_variant"]
    if rule.get("supported") is False:
        return "cpu", f"familia fora da politica de suporte ({rule['family']})"
    if gv == "cpu":
        return "cpu", f"regra {rule['id']}: CPU e melhor/igual (ver evidence)"
    if gv == "auto_vram":
        vram = a["dedicated_vram_mb"]
        if vram >= th["t1_min_vram_mb"]:
            return "t1-dml-fp16", f"VRAM dedicada {vram} MB >= {th['t1_min_vram_mb']} MB (T1 dml-fp16 mediu 5.133 MB)"
        if vram >= th["t0_min_vram_mb"]:
            return "t0-dml-fp16", f"VRAM dedicada {vram} MB < {th['t1_min_vram_mb']} (T1) e >= {th['t0_min_vram_mb']} (T0 dml-fp16 mediu 2.933 MB)"
        return "cpu", f"VRAM dedicada {vram} MB < {th['t0_min_vram_mb']} MB"
    return gv, f"regra {rule['id']}: variante fixa {gv}"


def decide(adapters: list[dict], cpu: dict, cfg: dict) -> dict:
    th = cfg["thresholds"]
    cands, ignored, evaluated = [], [], []
    for a in adapters:
        rule = match_rule(a, cfg)
        if a.get("is_software"):
            rule = next(r for r in cfg["rules"] if r["id"] == "software-basic-render")
        if rule is None:
            rule = dict(cfg["unknown_vendor"], id="unknown_vendor", family="vendor desconhecido", kind="dgpu")
        variant, why = resolve_variant(rule, a, th)
        is_igpu = bool(rule.get("integrated")) or rule.get("kind") == "igpu"
        ev = {"luid": a.get("luid"), "name": a["name"], "vendor_id": f"0x{a['vendor_id']:04X}", "device_id": f"0x{a['device_id']:04X}",
              "dedicated_vram_mb": a["dedicated_vram_mb"], "shared_mb": a.get("shared_mb"), "driver_version": a.get("driver_version"),
              "is_software": bool(a.get("is_software")), "is_integrated": is_igpu, "rule": rule["id"], "family": rule["family"],
              "supported": rule.get("supported"), "variant_if_chosen": variant, "needs_microbench": bool(rule.get("needs_microbench")),
              "confidence": rule.get("confidence"), "why": why}
        evaluated.append(ev)
        if a.get("is_software"):
            ignored.append({"name": a["name"], "luid": a.get("luid"), "reason": "adaptador de software/WARP"})
        elif variant == "cpu":
            ignored.append({"name": a["name"], "luid": a.get("luid"), "reason": why})
        else:
            cands.append((ev, rule))

    cpu_variant = th["cpu_default_variant"]
    t1_cpu_ok = (cpu.get("physical_cores") or 0) >= th["cpu_t1_int8_min_physical_cores"] and (cpu.get("ram_gb") or 0) >= th["cpu_t1_int8_min_ram_gb"]

    def score(c):  # dGPU > iGPU; T1 > T0; mais VRAM; (nunca o indice)
        ev, _ = c
        return (not ev["is_integrated"], ev["variant_if_chosen"] == "t1-dml-fp16", ev["dedicated_vram_mb"])

    res = {"schema_version": SCHEMA, "config": str(DEFAULT_CONFIG.name), "cpu": cpu, "adapters": evaluated, "ignored": ignored,
           "cpu_options": {"default": cpu_variant, "t1_cpu_int8_allowed": t1_cpu_ok,
                           "t1_cpu_int8_rule": f">= {th['cpu_t1_int8_min_physical_cores']} nucleos fisicos e >= {th['cpu_t1_int8_min_ram_gb']} GB RAM (opcional; autocomplete usa T0 int4)"}}
    def mb_cands(exclude):
        out = []
        for e in evaluated:
            r = next((x for x in cfg["rules"] if x["id"] == e["rule"]), None)
            if r and r.get("candidate_variant") and e["variant_if_chosen"] == "cpu" and e["luid"] != exclude and not e["is_software"]:
                out.append({"name": e["name"], "luid": e["luid"], "variant": r["candidate_variant"]})
        return out

    if not cands:
        res["microbench_candidates"] = mb_cands(None)
        res.update(chosen_adapter=None, recommended_variant=cpu_variant, fallback_variant=cpu_variant, fallback_chain=[cpu_variant],
                   supported=True, needs_microbench=False, confidence="alta" if adapters else "media",
                   reason="nenhum adaptador elegivel (todos ignorados ou sem GPU): CPU T0 int4" + ("" if adapters else " [nenhum adaptador enumerado]"))
        return res
    ev, rule = max(cands, key=score)
    variant = ev["variant_if_chosen"]
    chain = [variant]
    if variant == "t1-dml-fp16" and ev["dedicated_vram_mb"] >= th["t0_min_vram_mb"]:
        chain.append("t0-dml-fp16")
    chain.append(rule.get("fallback", cpu_variant))
    needs = bool(rule.get("needs_microbench")) or rule.get("supported") == "unknown"
    res.update(chosen_adapter={k: ev[k] for k in ("luid", "name", "vendor_id", "device_id", "dedicated_vram_mb", "shared_mb", "driver_version", "is_integrated", "rule", "family")},
               recommended_variant=variant, fallback_variant=rule.get("fallback", cpu_variant), fallback_chain=chain,
               supported=rule.get("supported"), needs_microbench=needs, confidence=rule.get("confidence"),
               reason=ev["why"] + ("; micro-benchmark de primeira execucao obrigatorio antes de aceitar a GPU" if needs else ""),
               evidence=rule.get("evidence"))
    if ev["is_integrated"]:
        res["note"] = "iGPU escolhida porque nao ha dGPU elegivel e a regra [E] indica que pode ganhar da CPU; confirmar no micro-benchmark"
    # candidatas a micro-benchmark que ficaram em CPU (ex.: Iris Xe)
    res["microbench_candidates"] = mb_cands(ev["luid"])
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--simulate", help="JSON {adapters:[...], cpu:{...}} em vez de enumerar o hardware")
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--out", help="grava tambem o JSON neste arquivo")
    args = ap.parse_args(argv)
    cfg = load_config(Path(args.config))
    if args.simulate:
        sim = json.loads(Path(args.simulate).read_text(encoding="utf-8"))
        adapters, cpu = sim["adapters"], sim.get("cpu", {"physical_cores": 8, "ram_gb": 16})
        enum_src = "simulado"
    else:
        wmi = _wmi_snapshot()
        adapters = enumerate_dxgi()
        enum_src = "dxgi"
        if not adapters:
            adapters, enum_src = adapters_from_wmi(wmi), "wmi-fallback"
        merge_driver(adapters, wmi)
        cpu = detect_cpu(wmi)
    if args.list_only:
        print(json.dumps({"enumeration": enum_src, "adapters": adapters}, indent=2, ensure_ascii=False))
        return 0
    res = decide(adapters, cpu, cfg)
    res["enumeration"] = enum_src
    txt = json.dumps(res, indent=2, ensure_ascii=False)
    print(txt)
    if args.out:
        Path(args.out).write_text(txt, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
