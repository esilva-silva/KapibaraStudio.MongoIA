"""Guarda do papel `hom` (homologacao): schemas `hom` NUNCA entram em train/val/test/bench dos geradores/corpora de treino.

Regra unica (ver reports/datasets/hom-guard-report.md):
  * geradores: `load_schemas(..., include_hom=False)` (padrao) ignora schemas `hom`; `--include-hom` so para scripts de
    homologacao (gravam em outro diretorio, split `homolog`);
  * validadores/corpora: `assert_no_hom(registros, onde)` -> SystemExit(2) com mensagem clara.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import split_roles as _sr  # noqa: E402

EXIT_CODE = 2


class HomLeak(SystemExit):
    def __init__(self, msg: str):
        super().__init__(msg)
        self.code = msg


def _bare(x) -> str:
    return str(x).split("::", 1)[0].replace("\\", "/").rsplit("/", 1)[-1]


def is_hom_schema(schema_id=None, domain=None, split_role=None) -> bool:
    """True se o schema (id com/sem dominio) tem papel hom (JSON/manifesto) ou pertence a dominio HOM."""
    if split_role == "hom":
        return True
    if domain in _sr.HOM_DOMAINS:
        return True
    if schema_id is None:
        return False
    s = str(schema_id).replace("\\", "/")
    if "/" in s and s.split("/", 1)[0] in _sr.HOM_DOMAINS:
        return True
    return _sr.role_for(_bare(s)) == "hom"


def record_is_hom(rec: dict) -> bool:
    sid = rec.get("schema_id") or rec.get("schema")
    return is_hom_schema(sid, rec.get("domain"), rec.get("split_role") or rec.get("schema_split_role"))


def find_hom(records, limit: int = 5) -> tuple[int, list[str]]:
    n, ex = 0, []
    for r in records:
        if record_is_hom(r):
            n += 1
            if len(ex) < limit:
                ex.append(f"{r.get('id', '?')} ({r.get('schema_id') or r.get('schema') or r.get('domain')})")
    return n, ex


def assert_no_hom(records, where: str) -> None:
    """Falha (SystemExit != 0) se algum registro de schema `hom` estiver em `records` (entrada de treino)."""
    n, ex = find_hom(records)
    if n:
        raise HomLeak(f"[hom-guard] ERRO: {n} registro(s) de schema `hom` (homologacao) em {where}: {'; '.join(ex)}. "
                      f"Schemas hom nao podem aparecer em train/val/test/bench; remova-os da entrada.")


def assert_no_hom_ids(schema_ids, where: str) -> None:
    bad = sorted({str(s) for s in schema_ids if s is not None and is_hom_schema(s)})
    if bad:
        raise HomLeak(f"[hom-guard] ERRO: schemas `hom` em {where}: {', '.join(bad[:8])} ({len(bad)}). "
                      f"Schemas hom nao podem aparecer em train/val/test/bench.")


def filter_schemas(schemas, include_hom: bool = False):
    if include_hom:
        return list(schemas)
    return [s for s in schemas if not is_hom_schema(s.schema_id, s.domain, getattr(s, "raw_role", None))]


def drop_hom(records, include_hom: bool = False):
    """Filtro em streaming: descarta registros de schema `hom` (padrao). `include_hom=True` so p/ homologacao."""
    if include_hom:
        yield from records
        return
    for r in records:
        if not record_is_hom(r):
            yield r
