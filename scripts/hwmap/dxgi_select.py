#!/usr/bin/env python3
"""Seleciona, SO NO PROCESSO ATUAL, qual adaptador DXGI o GenAI-WinML 0.15.2 usa no DirectML (sem mexer em configuracao do Windows).

Por que: o GenAI 0.15.2 cria o dispositivo D3D12/DML sozinho no adaptador "padrao" (o 0 da enumeracao DXGI de alto desempenho = RX 7800 XT) e IGNORA
`device_id`, `device_filter`, `performance_preference` e `device_filtering_options` (testado em reports/hwmap/phase2_measurements.md §0). A unica
forma de medir a iGPU sem alterar as "Configuracoes graficas" do Windows (preferencia por app, que e configuracao do sistema) e remapear a ordem de
enumeracao dentro do proprio processo: este modulo troca, na vtable do IDXGIFactory (memoria do processo), EnumAdapters / EnumAdapters1 /
EnumAdapterByGpuPreference por ganchos que colocam o adaptador pedido (vendor:device) na posicao 0. Efeito restrito ao processo; nada e gravado no sistema.

A escolha e sempre confirmada por contadores do Windows (phase2_bench.py: memoria dedicada e engine por LUID); este modulo so conta as chamadas.
"""
from __future__ import annotations

import ctypes
from ctypes import POINTER, WINFUNCTYPE, c_int32, c_uint32, c_void_p

_keep: list = []
calls = {"EnumAdapters": 0, "EnumAdapters1": 0, "EnumAdapterByGpuPreference": 0, "FactoryCreateDevice": 0, "FactoryCreateDevice_null_adapter": 0}


class _GUID(ctypes.Structure):
    _fields_ = [("a", ctypes.c_uint32), ("b", ctypes.c_uint16), ("c", ctypes.c_uint16), ("d", ctypes.c_ubyte * 8)]

    @classmethod
    def parse(cls, s: str):
        import uuid
        u = uuid.UUID(s)
        g = cls()
        g.a, g.b, g.c = u.time_low, u.time_mid, u.time_hi_version
        for i, x in enumerate(bytes([u.clock_seq_hi_variant, u.clock_seq_low]) + u.node.to_bytes(6, "big")):
            g.d[i] = x
        return g


class _DESC(ctypes.Structure):
    _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", c_uint32), ("DeviceId", c_uint32), ("SubSysId", c_uint32), ("Revision", c_uint32),
                ("DedicatedVideoMemory", ctypes.c_size_t), ("DedicatedSystemMemory", ctypes.c_size_t), ("SharedSystemMemory", ctypes.c_size_t),
                ("LuidLow", c_uint32), ("LuidHigh", ctypes.c_int32)]


def _vt(obj, idx, restype, *argtypes):
    vtbl = ctypes.cast(obj, POINTER(POINTER(c_void_p))).contents
    return WINFUNCTYPE(restype, c_void_p, *argtypes)(vtbl[idx])


def _release(p):
    if p:
        _vt(p, 2, c_uint32)(p)


def _adapters(fac):
    out = []
    i = 0
    while True:
        ad = c_void_p()
        if _vt(fac, 12, c_int32, c_uint32, POINTER(c_void_p))(fac, i, ctypes.byref(ad)) != 0:
            break
        d = _DESC()
        _vt(ad, 8, c_int32, POINTER(_DESC))(ad, ctypes.byref(d))   # IDXGIAdapter::GetDesc (indice 8)
        out.append((i, d.VendorId, d.DeviceId, d.Description))
        _release(ad)
        i += 1
    return out


def install(vendor_id: int, device_id: int) -> dict:
    """Coloca o adaptador vendor:device na posicao 0 das enumeracoes DXGI deste processo. Retorna {'perm': [...], 'target_index': n}."""
    dxgi = ctypes.WinDLL("dxgi")
    dxgi.CreateDXGIFactory1.argtypes = [POINTER(_GUID), POINTER(c_void_p)]
    fac = c_void_p()
    if dxgi.CreateDXGIFactory1(ctypes.byref(_GUID.parse("770aae78-f26f-4dba-a829-253c83d1b387")), ctypes.byref(fac)) != 0:
        raise RuntimeError("CreateDXGIFactory1 falhou")
    ads = _adapters(fac)
    hit = [i for i, v, d, _ in ads if v == vendor_id and d == device_id]
    if len(hit) != 1:
        raise RuntimeError(f"adaptador {vendor_id:04X}:{device_id:04X}: {len(hit)} candidatos em {ads}")
    t = hit[0]
    perm = [t] + [i for i, *_ in ads if i != t]
    vtbl = ctypes.cast(fac, POINTER(POINTER(c_void_p))).contents
    ENUM = WINFUNCTYPE(c_int32, c_void_p, c_uint32, POINTER(c_void_p))
    ENUM_PREF = WINFUNCTYPE(c_int32, c_void_p, c_uint32, c_int32, POINTER(_GUID), POINTER(c_void_p))
    orig = {7: ENUM(vtbl[7]), 12: ENUM(vtbl[12]), 29: ENUM_PREF(vtbl[29])}

    def remap(i):
        return perm[i] if i < len(perm) else i

    def h7(this, i, pp):
        calls["EnumAdapters"] += 1
        return orig[7](this, remap(i), pp)

    def h12(this, i, pp):
        calls["EnumAdapters1"] += 1
        return orig[12](this, remap(i), pp)

    def h29(this, i, pref, riid, pp):
        calls["EnumAdapterByGpuPreference"] += 1
        return orig[29](this, remap(i), 0, riid, pp)   # preferencia 0 = ordem de EnumAdapters1 (a remapeada acima)

    hooks = {7: ENUM(h7), 12: ENUM(h12), 29: ENUM_PREF(h29)}
    _keep.extend([hooks, orig])
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.VirtualProtect.argtypes = [c_void_p, ctypes.c_size_t, c_uint32, POINTER(c_uint32)]
    for idx, fn in hooks.items():
        slot = ctypes.cast(vtbl, c_void_p).value + idx * ctypes.sizeof(c_void_p)
        old = c_uint32()
        if not k32.VirtualProtect(slot, 8, 0x04, ctypes.byref(old)):   # PAGE_READWRITE
            raise RuntimeError("VirtualProtect falhou")
        ctypes.c_void_p.from_address(slot).value = ctypes.cast(fn, c_void_p).value
        k32.VirtualProtect(slot, 8, old.value, ctypes.byref(old))
    _keep.append(fac)   # mantem a fabrica viva
    # D3D12: o GenAI cria o dispositivo por ID3D12DeviceFactory::CreateDevice(adapter=NULL) => adaptador padrao do Windows. Substitui NULL pelo adaptador pedido.
    try:
        d3d = ctypes.WinDLL("d3d12")
        d3d.D3D12GetInterface.argtypes = [POINTER(_GUID), POINTER(_GUID), POINTER(c_void_p)]
        dfac = c_void_p()
        hr = d3d.D3D12GetInterface(ctypes.byref(_GUID.parse("114863bf-c386-4aee-b39d-8f0bbb062955")), ctypes.byref(_GUID.parse("61f307d3-d34e-4e7c-8374-3ba4de23cccb")), ctypes.byref(dfac))
        if hr == 0:
            target = c_void_p()
            _vt(fac, 12, c_int32, c_uint32, POINTER(c_void_p))(fac, t, ctypes.byref(target))
            _keep.append(target)
            dvt = ctypes.cast(dfac, POINTER(POINTER(c_void_p))).contents
            CD = WINFUNCTYPE(c_int32, c_void_p, c_void_p, c_int32, POINTER(_GUID), POINTER(c_void_p))
            orig_cd = CD(dvt[7])

            def hcd(this, adapter, fl, riid, pp):
                calls["FactoryCreateDevice"] += 1
                if not adapter:
                    calls["FactoryCreateDevice_null_adapter"] += 1
                    adapter = target.value
                return orig_cd(this, adapter, fl, riid, pp)
            hcd_c = CD(hcd)
            _keep.extend([hcd_c, orig_cd, dfac])
            slot = ctypes.cast(dvt, c_void_p).value + 7 * ctypes.sizeof(c_void_p)
            old = c_uint32()
            k32.VirtualProtect(slot, 8, 0x04, ctypes.byref(old))
            ctypes.c_void_p.from_address(slot).value = ctypes.cast(hcd_c, c_void_p).value
            k32.VirtualProtect(slot, 8, old.value, ctypes.byref(old))
        else:
            calls["d3d12_factory_hr"] = hr
    except OSError as e:
        calls["d3d12_factory_err"] = str(e)
    return {"target_dxgi_index": t, "perm": perm, "adapters": [(i, f"{v:04X}:{d:04X}", n) for i, v, d, n in ads]}
