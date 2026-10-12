"""Configuração central, caminhos, JSONL e metadados de execução do pipeline compacto.

Derivado de SlopStudio.MongoAI (v1) scripts/mongoai/common.py; os caminhos agora vêm de configs/pipeline.yaml.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

WORKSPACE = Path(__file__).resolve().parents[2]
_ENV = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def expand(value: Any) -> Any:
    """Expande ${VAR} e ${VAR:-padrão} recursivamente em strings de configuração."""
    if isinstance(value, str):
        return _ENV.sub(lambda m: os.environ.get(m[1]) or (m[2] or ""), value)
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v) for v in value]
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with resolve(path).open("r", encoding="utf-8") as handle:
        return expand(yaml.safe_load(handle) or {})


def resolve(path: str | Path, base: Path = WORKSPACE) -> Path:
    path = Path(os.path.expandvars(str(path)))
    return path if path.is_absolute() else (base / path)


class Pipeline:
    """Acesso a configs/pipeline.yaml (placeholders {workspace} e variáveis de ambiente)."""

    def __init__(self, config_path: str | Path = "configs/pipeline.yaml") -> None:
        self.raw = load_yaml(config_path)
        self.workspace = Path(self.raw["workspace"])

    def fill(self, value: str | Path) -> Path:
        return resolve(str(value).replace("{workspace}", self.workspace.as_posix()), self.workspace)

    def path(self, name: str) -> Path:
        return self.fill(self.raw["paths"][name])

    def config(self, name: str) -> dict[str, Any]:
        return load_yaml(self.config_path(name))

    def config_path(self, name: str) -> Path:
        return self.fill(self.raw["configs"][name])

    @property
    def teacher_artifacts(self) -> Path:
        return Path(self.raw["teacher_artifacts"])

    @property
    def teacher_roots(self) -> list[Path]:
        return [self.teacher_artifacts] + [Path(p) for p in self.raw.get("teacher_extra_roots", []) if p]

    @property
    def ide_source(self) -> Path:
        return Path(self.raw["ide_source"])

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def gpu_expect(self) -> str:
        return self.raw["environment"].get("gpu_expect_device", "")

    def students(self) -> list[dict[str, Any]]:
        return [dict(s, config=self.fill(s["config"])) for s in self.raw["students"]]

    def student(self, student_id: str) -> dict[str, Any]:
        for item in self.students():
            if student_id in (item["id"], item["name"]):
                return item
        raise SystemExit(f"Student desconhecido: {student_id} (disponíveis: {[s['id'] for s in self.students()]})")

    def student_base_dir(self, student_id: str) -> Path:
        return self.path("student") / "base" / self.student(student_id)["hf_repo"].split("/")[-1]

    def teacher_inventory(self) -> dict[str, Any]:
        path = self.path("reports") / "teacher" / "teacher_inventory.json"
        if not path.exists():
            raise SystemExit(f"Inventário do professor ausente ({path}). Execute scripts/scan_teacher_artifacts.py primeiro.")
        return json.loads(path.read_text(encoding="utf-8"))

    def datasets(self, *parts: str) -> Path:
        return self.path("datasets").joinpath(*parts)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with resolve(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: JSON inválido: {exc}") from exc


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Escrita atômica (arquivo temporário + rename)."""
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    count = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    os.replace(tmp, path)
    return count


def write_json(path: str | Path, data: Any) -> None:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with resolve(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def stable_unit(key: str, seed: int | str = 0) -> float:
    """Número determinístico em [0, 1) independente de PYTHONHASHSEED."""
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def seeded_random(seed: int | str, *parts: Any) -> random.Random:
    return random.Random(int(sha256_text(":".join(map(str, (seed, *parts))))[:16], 16))


def pick(rng: random.Random, weights: dict) -> Any:
    items = [(k, w) for k, w in weights.items() if w > 0]
    return rng.choices([k for k, _ in items], weights=[w for _, w in items])[0]


def run_metadata(config: Any = None) -> dict[str, Any]:
    return {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config_sha256": sha256_text(json.dumps(config, sort_keys=True, default=str)) if config is not None else None,
    }


def human_bytes(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def dir_size(path: str | Path) -> int:
    return sum(p.stat().st_size for p in resolve(path).rglob("*") if p.is_file())


def log(message: str) -> None:
    import time

    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)
