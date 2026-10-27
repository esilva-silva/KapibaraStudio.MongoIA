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

Regras vigentes (decisoes D1-D5 de 2026-10-26, configs/gpu_compat.yaml schema 2): VRAM minima e TETO DE CONTEXTO por VRAM (`context_cap_tokens`),
iGPU sempre na CPU (`igpu_policy: cpu_always_until_measured`), Pascal/Polaris `unknown` decididos pelo micro-benchmark (sem variante fp32),
T1 int8 em CPU so como opcao lenta escolhida pelo usuario, nenhuma variante nova. A selecao do adaptador e por vendor/device/VRAM; como o GenAI-WinML
0.15.2 ignora device_id, a IDE precisa validar o adaptador EFETIVO (docs/prs/pr-5-selecao-de-adaptador-dml.md): ver `adapter_selection`.
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
SCHEMA = 2

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
    return {"name": name, "physical_cores": cores, "logical_cores": logical, "ram_gb": ram_gb,
            "physical_cores_source": "wmi" if cores else "unknown (nao inferir SMT)"}


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
        raise SystemExit(f"schema_version {cfg.get('schema_version')} != {SCHEMA} (ver `migration` no gpu_compat.yaml)")
    return cfg


def match_rule(a: dict, cfg: dict) -> dict | None:
    """Primeira regra do mesmo vendor cujo device_id esta numa faixa OU cujo nome casa o regex. Software e tratado primeiro."""
    for r in cfg["rules"]:
        if r["vendor_id"] != a["vendor_id"]:
            continue
        m = r.get("match", {})
        for lo, hi in m.get("device_id_ranges", []):
            if isinstance(a.get("device_id"), int) and lo <= a["device_id"] <= hi:
                return r
        rx = m.get("name_regex")
        if rx and re.search(rx, a["name"], re.I):
            return r
    return None


def cap_for(cfg: dict, variant: str, vram_mb: int) -> int | None:
    """Teto de contexto (tokens totais = prompt + saida) que a VRAM comporta para a variante, pela curva medida (context_cap_by_vram); None = nao cabe."""
    for min_vram, cap in sorted(cfg["context_cap_by_vram"].get(variant, []), key=lambda r: -r[0]):
        if vram_mb >= min_vram:
            return int(cap)
    return None


def resolve_variant(rule: dict, a: dict, cfg: dict, is_igpu: bool = False) -> dict:
    """-> {variant, why, context_cap_tokens, capped, force_microbench, t0_cap}. `variant` = variante GPU ou 'cpu'."""
    th = cfg["thresholds"]
    cpu_cap = th["cpu_context_cap_tokens"]

    def out(variant, why, cap=None, capped=False, force_mb=False, t0_cap=None):
        return {"variant": variant, "why": why, "context_cap_tokens": cap if variant != "cpu" else cpu_cap, "capped": capped,
                "force_microbench": force_mb, "t0_cap": t0_cap}

    gv = rule["gpu_variant"]
    if rule.get("supported") is False:
        return out("cpu", f"familia fora da politica de suporte ({rule['family']})")
    if is_igpu and cfg.get("igpu_policy") == "cpu_always_until_measured":
        return out("cpu", f"igpu_policy cpu_always_until_measured (D3): iGPU sempre na CPU nesta versao (regra {rule['id']})")
    if gv == "cpu":
        return out("cpu", f"regra {rule['id']}: CPU e melhor/igual (ver evidence)")
    if gv == "auto_vram":
        vram = a["dedicated_vram_mb"]
        floor = cfg["context_cap_min_useful_tokens"]
        c1, c0 = cap_for(cfg, "t1-dml-fp16", vram), cap_for(cfg, "t0-dml-fp16", vram)
        if c1 is not None and c1 >= floor:
            if vram >= th["t1_full_min_vram_mb"]:
                return out("t1-dml-fp16", f"VRAM dedicada {vram} MB >= {th['t1_full_min_vram_mb']} MB (D1: T1 dml-fp16 com contexto de ate {c1} tokens)", c1, False, False, c0)
            return out("t1-dml-fp16", f"VRAM dedicada {vram} MB entre {th['t1_capped_min_vram_mb']} e {th['t1_full_min_vram_mb']} MB (D1: T1 so com teto de contexto {c1} tokens e se o micro-benchmark aprovar)", c1, True, True, c0)
        if c0 is not None and c0 >= floor:
            return out("t0-dml-fp16", f"VRAM dedicada {vram} MB < {th['t1_capped_min_vram_mb']} MB (T1) e >= {th['t0_min_vram_mb']} MB (D1: T0 dml-fp16 com contexto de ate {c0} tokens)", c0)
        return out("cpu", f"VRAM dedicada {vram} MB < {th['t0_min_vram_mb']} MB (D1): sem VRAM para o T0 dml-fp16")
    return out(gv, f"regra {rule['id']}: variante fixa {gv}", cap_for(cfg, gv, a["dedicated_vram_mb"]) or th["cpu_context_cap_tokens"])


def decide(adapters: list[dict], cpu: dict, cfg: dict) -> dict:
    th = cfg["thresholds"]
    cands, ignored, evaluated = [], [], []
    for a in adapters:
        rule = match_rule(a, cfg)
        if a.get("is_software"):
            rule = next(r for r in cfg["rules"] if r["id"] == "software-basic-render")
        if rule is None:
            rule = dict(cfg["unknown_vendor"], id="unknown_vendor", family="vendor desconhecido", kind="dgpu")
        is_igpu = bool(rule.get("integrated")) or rule.get("kind") == "igpu"
        rv = resolve_variant(rule, a, cfg, is_igpu)
        variant, why = rv["variant"], rv["why"]
        ev = {"luid": a.get("luid"), "name": a["name"], "vendor_id": f"0x{a['vendor_id']:04X}", "device_id": (f"0x{a['device_id']:04X}" if isinstance(a.get("device_id"), int) else None),
              "dxgi_index": a.get("dxgi_index"), "dedicated_vram_mb": a["dedicated_vram_mb"], "shared_mb": a.get("shared_mb"), "driver_version": a.get("driver_version"),
              "is_software": bool(a.get("is_software")), "is_integrated": is_igpu, "rule": rule["id"], "family": rule["family"],
              "supported": rule.get("supported"), "variant_if_chosen": variant, "context_cap_tokens": rv["context_cap_tokens"],
              "context_capped": rv["capped"], "needs_microbench": bool(rule.get("needs_microbench")) or rv["force_microbench"],
              "confidence": rule.get("confidence"), "decision_ref": rule.get("decision_ref"), "why": why}
        evaluated.append(ev)
        if a.get("is_software"):
            ignored.append({"name": a["name"], "luid": a.get("luid"), "reason": "adaptador de software/WARP"})
        elif variant == "cpu":
            ignored.append({"name": a["name"], "luid": a.get("luid"), "reason": why})
        else:
            cands.append((ev, rule, rv))

    cpu_variant = th["cpu_default_variant"]
    t1_cpu_ok = (cpu.get("physical_cores") or 0) >= th["cpu_t1_int8_min_physical_cores"] and (cpu.get("ram_gb") or 0) >= th["cpu_t1_int8_min_ram_gb"]

    def score(c):  # dGPU > iGPU; T1 > T0; mais VRAM; (nunca o indice)
        ev = c[0]
        return (not ev["is_integrated"], ev["variant_if_chosen"] == "t1-dml-fp16", ev["dedicated_vram_mb"])

    pending = [{"name": e["name"], "luid": e["luid"], "rule": e["rule"],
                "candidate_after_measurement": next((r.get("candidate_after_measurement") for r in cfg["rules"] if r["id"] == e["rule"]), None)}
               for e in evaluated if e["is_integrated"] and not e["is_software"]]
    res = {"schema_version": SCHEMA, "config": str(DEFAULT_CONFIG.name), "cpu": cpu, "adapters": evaluated, "ignored": ignored,
           "igpu_policy": cfg.get("igpu_policy"),
           "cpu_options": {"default": cpu_variant, "context_cap_tokens": th["cpu_context_cap_tokens"],
                           "t1_cpu_int8_allowed": t1_cpu_ok, "t1_cpu_int8_is_default": False, "t1_cpu_int8_user_choice_only": True,
                           "t1_cpu_int8_declared_slow": True, "t1_cpu_int8_meets_ttft_gate": False,
                           "t1_cpu_int8_rule": (f"D4: opcional e declaradamente lento (escolha do usuario, nunca padrao), >= {th['cpu_t1_int8_min_physical_cores']} nucleos fisicos "
                                                f"e >= {th['cpu_t1_int8_min_ram_gb']} GB RAM; autocomplete usa T0 int4")},
           "microbench_candidates": [], "igpus_pending_measurement": pending}

    if not cands:
        res.update(chosen_adapter=None, recommended_variant=cpu_variant, fallback_variant=cpu_variant, fallback_chain=[cpu_variant],
                   context_cap_tokens=th["cpu_context_cap_tokens"], supported=True, needs_microbench=False,
                   confidence="alta" if adapters else "media", adapter_selection={"effective_adapter_check_required": False},
                   reason="nenhum adaptador elegivel (todos ignorados ou sem GPU): CPU T0 int4" + ("" if adapters else " [nenhum adaptador enumerado]"))
        return res
    ev, rule, rv = max(cands, key=score)
    variant = ev["variant_if_chosen"]
    chain = [variant]
    if (variant == "t1-dml-fp16" and rv["t0_cap"] is not None and rv["t0_cap"] >= cfg["context_cap_min_useful_tokens"]
            and not rule.get("microbench_fail_goes_to_cpu")):
        chain.append("t0-dml-fp16")
    chain.append(rule.get("fallback", cpu_variant))
    needs = ev["needs_microbench"] or rule.get("supported") == "unknown"
    # WMI fallback has no LUIDs: distinct entries still represent other controllers.
    others = [e for e in evaluated if e is not ev and not e["is_software"]
              and (e["luid"] is None or ev["luid"] is None or e["luid"] != ev["luid"])]
    res.update(chosen_adapter={k: ev[k] for k in ("luid", "name", "vendor_id", "device_id", "dxgi_index", "dedicated_vram_mb", "shared_mb", "driver_version",
                                                  "is_integrated", "rule", "family")},
               recommended_variant=variant, context_cap_tokens=rv["context_cap_tokens"], context_capped=rv["capped"],
               context_cap_by_variant={"t1-dml-fp16": cap_for(cfg, "t1-dml-fp16", ev["dedicated_vram_mb"]), "t0-dml-fp16": rv["t0_cap"]},
               fallback_variant=rule.get("fallback", cpu_variant), fallback_chain=chain,
               supported=rule.get("supported"), needs_microbench=needs, confidence=rule.get("confidence"), decision_ref=rule.get("decision_ref") or "D1",
               reason=ev["why"] + ("; micro-benchmark de primeira execucao obrigatorio antes de aceitar a GPU" if needs else ""),
               adapter_selection={"basis": "vendor/device/VRAM dedicada (nunca o indice DXGI)",
                                  "effective_adapter_check_required": True,
                                  "multi_adapter_machine": bool(others),
                                  "chosen_is_dxgi_default": (ev["dxgi_index"] == 0) if ev["dxgi_index"] is not None else None,
                                  "note": ("GenAI-WinML 0.15.2 ignora device_id/device_filter: validar o adaptador efetivo (DXGI + contadores) e cair para a CPU "
                                           "se for iGPU/WARP/lento (docs/prs/pr-5-selecao-de-adaptador-dml.md)")},
               evidence=rule.get("evidence"))
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
