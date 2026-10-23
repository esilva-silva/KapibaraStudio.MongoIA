#!/usr/bin/env python3
"""Executa um script (gerador/pipeline de treino) com o guard do conjunto de HOMOLOGACAO ativo, sem editar o script:

  python scripts/data/llm/run_guarded.py scripts/data/gen_fim.py --seed 1 ...

Qualquer open()/listdir()/scandir() sobre data/homolog levanta HomologAccessError (ver homolog_guard.py)."""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import homolog_guard  # noqa: E402

if len(sys.argv) < 2:
    raise SystemExit(__doc__)
homolog_guard.enforce()
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, str(Path(script).resolve().parent))
runpy.run_path(script, run_name="__main__")
