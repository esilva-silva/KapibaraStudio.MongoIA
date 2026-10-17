"""Utilitarios compartilhados da destilacao sequence-level (Fase 02 §3.5 / Fase 05 §6.2): caminhos, hashes, seeds por item,
leitura do pool de prompts e guarda do conjunto cego com o manifesto carregado uma unica vez.

Nenhum modulo daqui toca em scripts/data (apenas importa); nenhum le data/eval/blind (so via blind_guard).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Iterator

ROOT = Path(__file__).resolve().parents[2]
DATA_SCRIPTS = ROOT / "scripts" / "data"
TEACHER_SCRIPTS = ROOT / "scripts" / "teacher"
BAKEOFF_SCRIPTS = ROOT / "bakeoff" / "v1" / "scripts"
CONFIG_PATH = ROOT / "configs" / "distill_v2c.yaml"
ROCM_SITE = r"C:\SlopStudio.MongoAI-artifacts\.venv\Lib\site-packages"
os.environ.setdefault("KAPI_TOKENIZERS_SITE", ROCM_SITE)

for _p in (DATA_SCRIPTS, TEACHER_SCRIPTS, BAKEOFF_SCRIPTS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def load_config(path: str | Path | None = None) -> dict:
    import yaml

    with open(path or CONFIG_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def git_hash() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


# -- hashes / seeds ---------------------------------------------------------------------------------------------------------------
def unit_hash(salt: str, ident: str) -> float:
    """Hash uniforme em [0, 1) de `salt:ident` (blake2b 64 bits): amostragem deterministica independente da ordem."""
    return int.from_bytes(hashlib.blake2b(f"{salt}:{ident}".encode("utf-8"), digest_size=8).digest(), "big") / 2.0 ** 64


def item_seed(base_seed: int, ident: str, sample: int = 0) -> int:
    """Seed por item derivada do id (nao da posicao nem do lote): a mesma amostra sai igual em qualquer retomada."""
    return int.from_bytes(hashlib.blake2b(f"{base_seed}:{ident}:{sample}".encode("utf-8"), digest_size=8).digest(), "big") & 0x7FFFFFFFFFFFFFFF


def norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\r\n", "\n")).strip()


# -- pool -------------------------------------------------------------------------------------------------------------------------
def pool_shards(pool_dir: Path) -> list[Path]:
    return sorted(Path(pool_dir).glob("pool-[0-9][0-9][0-9][0-9][0-9].jsonl"))


def iter_pool(pool_dir: Path, start: int = 0, stop: int | None = None) -> Iterator[tuple[int, dict]]:
    """(idx, registro) do pool em ordem; `idx` = posicao global (a mesma usada pelo gerador e pelo filtro)."""
    idx = 0
    for shard in pool_shards(pool_dir):
        with open(shard, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                if idx >= start:
                    if stop is not None and idx >= stop:
                        return
                    yield idx, json.loads(line)
                idx += 1


def pool_size(pool_dir: Path) -> int:
    manifest = Path(pool_dir) / "pool.manifest.json"
    if manifest.exists():
        return int(json.loads(manifest.read_text(encoding="utf-8"))["records"])
    return sum(1 for _ in iter_pool(pool_dir))


def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


# -- conjunto cego ----------------------------------------------------------------------------------------------------------------
class BlindChecker:
    """blind_guard com o manifesto selado carregado UMA vez (check_text/check_record releem o arquivo a cada chamada).

    Usa apenas as primitivas publicas do blind_guard (hashes e `_manifest`); nunca le data/eval/blind diretamente.
    """

    def __init__(self, blind=None) -> None:
        import blind_guard as bg

        self.bg = bg
        manifest = bg._manifest(blind)  # noqa: SLF001
        self.prompts = set(manifest.get("prompt_sha256") or [])
        self.items = set((manifest.get("item_sha256") or {}).values())
        self.sealed = bool(self.prompts or self.items)

    def hit_text(self, prefix: str, suffix: str) -> bool:
        if not self.prompts:
            return False
        return self.bg.prompt_sha256(prefix + "\x00" + suffix) in self.prompts

    def hit_record(self, record: dict) -> bool:
        return bool(self.items) and self.bg.item_sha256(record) in self.items


def guard_inputs(paths: Iterable[Path]) -> None:
    import blind_guard as bg

    for p in paths:
        bg.check_path(p)


# -- registros "lite" (decontaminacao / dedup): mesmos campos de validate_pipeline.lite_record, sem importar o pipeline ------------
def lite_fim(r: dict, middle: str | None = None) -> dict:
    return {"id": r["id"], "split": r.get("split"), "snippet_id": r.get("snippet_id"), "collection": r.get("collection"),
            "prefix": r["prefix"], "middle": r["middle"] if middle is None else middle, "suffix": r["suffix"], "type": "fim",
            "domain": r.get("domain"), "holdout": r.get("holdout"), "cut_kind": r.get("cut_kind"), "kind": r.get("kind"),
            "group_id": r.get("group_id"), "schema_id": r.get("schema_id"), "cursor_role": r.get("cursor_role"),
            "category": r.get("category"), "dialect": r.get("dialect")}


def lite_chat(r: dict) -> dict:
    c = r.get("context") or {}
    return {"id": r["id"], "split": r.get("split"), "snippet_id": r.get("snippet_id"), "collection": r.get("collection"),
            "instruction": r.get("instruction", ""), "output": r.get("output", ""), "context": {"editor": c.get("editor")},
            "type": "chat", "kind": r.get("kind"), "domain": r.get("domain"), "holdout": r.get("holdout"),
            "group_id": r.get("group_id"), "language": r.get("language"), "schema_id": r.get("schema_id")}
