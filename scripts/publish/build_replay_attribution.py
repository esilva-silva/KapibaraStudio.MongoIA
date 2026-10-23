#!/usr/bin/env python3
"""Gera `REPLAY_ATTRIBUTION.md` (sobe com o modelo) a partir de `data/replay/v1/LICENSES.md` (somente leitura, sem rede).

  python scripts/publish/build_replay_attribution.py            # escreve nos dois hf/
  python scripts/publish/build_replay_attribution.py --check    # exit 1 se os arquivos estiverem desatualizados

Mantem so os pacotes usados no treino (`uso == replay_train`); os de `eval_holdout` ficam de fora. Nao copia codigo de terceiros:
so nome, versao, licenca e a linha de copyright quando ela e uma linha de copyright de verdade.
"""
from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "replay" / "v1" / "LICENSES.md"
OUTS = [ROOT / "Repos" / "KapiCoder-Mongo-Nano" / "hf" / "REPLAY_ATTRIBUTION.md",
        ROOT / "Repos" / "KapiCoder-Mongo-Small" / "hf" / "REPLAY_ATTRIBUTION.md"]


def parse(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        if not line.startswith("| ") or line.startswith("| pacote") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
        if len(cells) != 8:
            continue
        name, version, lic, _origin, copyright_, use, files, tokens = cells
        rows.append({"name": name, "version": version, "license": lic, "copyright": copyright_, "use": use,
                     "files": int(files), "tokens": int(tokens)})
    return rows


def render(rows: list[dict]) -> str:
    train = [r for r in rows if r["use"] == "replay_train"]
    by_lic = collections.Counter(r["license"] for r in train)
    out = ["# Replay attribution / Atribuição do replay", "",
           "A small share of the fine-tuning tokens (about 12%) is generic JavaScript/TypeScript taken from npm packages already installed on the",
           "training machine, used only as training text (\"replay\") to limit forgetting of generic code. The code is **not redistributed** with this model.",
           "All packages below are under permissive licenses (MIT, Apache-2.0, BSD-2/3-Clause, ISC, some dual-licensed with another permissive option); original headers were kept in the training text.",
           "Packages used only for evaluation are not listed. Source table: the project's `data/replay/v1/LICENSES.md`.", "",
           "Uma parcela pequena dos tokens de ajuste fino (cerca de 12%) é código JavaScript/TypeScript genérico de pacotes npm, usado só como texto de treino",
           "(replay) para limitar o esquecimento de código genérico. O código **não é redistribuído** com este modelo.", "",
           f"Packages used in training: **{len(train)}**.", "",
           "| License | Packages |", "|---|---:|"]
    out += [f"| {k} | {v} |" for k, v in sorted(by_lic.items(), key=lambda kv: (-kv[1], kv[0]))]
    out += ["", "| Package | Version | License | Copyright line |", "|---|---|---|---|"]
    for r in sorted(train, key=lambda r: (r["name"].lower(), r["version"])):
        cr = r["copyright"] if r["copyright"].lower().startswith("copyright") else "-"
        cr = re.sub(r"\s*<?[\w.+-]+@[\w-]+\.[\w.]+>?", "", cr)  # sem e-mails de terceiros no card
        cr = re.sub(r"\s*\(\s*\)", "", cr)
        out.append(f"| {r['name']} | {r['version']} | {r['license']} | {cr.replace('|', '/')} |")
    out.append("")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    rows = parse(SRC.read_text(encoding="utf-8"))
    if not rows:
        print("nenhuma linha lida de", SRC)
        return 2
    text = render(rows)
    bad = 0
    for out in OUTS:
        if args.check:
            if not out.exists() or out.read_text(encoding="utf-8") != text:
                print("desatualizado:", out)
                bad += 1
        else:
            out.write_text(text, encoding="utf-8", newline="\n")
            print("escrito:", out, f"({len(text)} bytes)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
