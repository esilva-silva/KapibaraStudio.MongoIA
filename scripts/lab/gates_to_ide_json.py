"""configs/gates.yaml -> configs/gates.ide.json (so owner:kapilab; evita YAML no console do KapiLab).

Saida: {"schema","source","tiers":{T:{"scenarios":{<cenario>:[{id,metric,op,threshold}]}}},"common":[...],
        "excluded":{"inactive":[ids],"no_threshold":[ids]}}
Cenario = id sem o ultimo segmento (ex.: T1.dml_fp16). Gates inativos ou sem limiar numerico (todo/null)
nao entram (o KapiLab nao os avalia); ficam listados em "excluded".

Uso: python scripts/lab/gates_to_ide_json.py [--gates configs/gates.yaml] [--out configs/gates.ide.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml


def build(cfg: dict, raw_bytes: bytes) -> dict:
    out = {"schema": "kapi-gates-ide-v1", "source": "configs/gates.yaml",
           "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
           "gates_status": cfg.get("status"), "tiers": {}, "common": [],
           "excluded": {"inactive": [], "no_threshold": []}}

    def keep(g):
        if g.get("owner") != "kapilab":
            return None
        if g.get("inactive"):
            out["excluded"]["inactive"].append(g["id"])
            return None
        if g.get("todo") or g.get("threshold") is None:
            out["excluded"]["no_threshold"].append(g["id"])
            return None
        return {"id": g["id"], "metric": g["metric"], "op": g["op"], "threshold": g["threshold"]}

    for tier, t in (cfg.get("tiers") or {}).items():
        scen: dict[str, list] = {}
        for g in t.get("gates") or []:
            e = keep(g)
            if e:
                scen.setdefault(g["id"].rsplit(".", 1)[0], []).append(e)
        out["tiers"][tier] = {"scenarios": scen}
    common = cfg.get("common") or {}
    for g in (common.get("gates") or []) + (common.get("parity") or []):
        e = keep(g)
        if e:
            out["common"].append(e)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gates", default="configs/gates.yaml")
    ap.add_argument("--out", default="configs/gates.ide.json")
    args = ap.parse_args(argv)
    raw = Path(args.gates).read_bytes()
    doc = build(yaml.safe_load(raw), raw)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    n = sum(len(v) for t in doc["tiers"].values() for v in t["scenarios"].values()) + len(doc["common"])
    print(f"{args.out}: {n} gates kapilab")
    return 0


if __name__ == "__main__":
    sys.exit(main())
