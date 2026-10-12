"""Verificação de sintaxe JavaScript em lote via Node (compila sem executar)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

SCRIPT = Path(__file__).with_name("node") / "syntax_check.mjs"


def node_available() -> bool:
    return shutil.which("node") is not None


def kind_for(dialect: str) -> str:
    """Tipo de verificação por dialeto: JSON (modo Agregação), mongosh (await no nível superior) ou script do Console."""
    return {"json": "json", "mongosh": "mongosh"}.get(dialect, "script")


def check_many(items: Iterable[tuple[str, str, str]], timeout: float = 900) -> dict[str, tuple[bool, str | None]]:
    """items: (id, code, kind) com kind em {"script", "mongosh", "json"}. Retorna id -> (ok, erro)."""
    payload = "".join(json.dumps({"id": i, "code": c, "kind": k}, ensure_ascii=False) + "\n" for i, c, k in items)
    if not payload:
        return {}
    if not node_available():
        raise RuntimeError("Node.js não encontrado no PATH; necessário para Syntax Valid.")
    process = subprocess.run(["node", str(SCRIPT)], input=payload.encode("utf-8"), capture_output=True, timeout=timeout, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"syntax_check.mjs falhou: {process.stderr.decode('utf-8', 'replace')[:500]}")
    results: dict[str, tuple[bool, str | None]] = {}
    for line in process.stdout.decode("utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            results[str(row["id"])] = (bool(row["ok"]), row.get("error"))
    return results
