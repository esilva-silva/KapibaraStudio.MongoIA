"""Download base/teacher models from Hugging Face into BaseRepos\\<name> with a pinned revision.

Only safetensors/GGUF weights, JSON configs, tokenizer files, LICENSE and README are fetched (never .bin/pickle).
Each download is recorded in BaseRepos\\base-repos.lock.json: repo id, commit SHA, license, files with size and SHA-256.

Usage:
    python -I scripts/download_base.py Qwen/Qwen2.5-Coder-0.5B [more repo ids...]
    python -I scripts/download_base.py Qwen/Qwen3-14B-GGUF --gguf-pattern "*Q4_K_M.gguf" --name Qwen3-14B-GGUF
    python -I scripts/download_base.py --plan            # downloads the default plan, in order
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
BASE_REPOS = WORKSPACE / "BaseRepos"
LOCK_FILE = BASE_REPOS / "base-repos.lock.json"

os.environ.setdefault("HF_HOME", str(WORKSPACE / "cache" / "hf"))

from huggingface_hub import HfApi, snapshot_download  # noqa: E402

WEIGHT_PATTERNS = ["*.safetensors", "*.safetensors.index.json"]
COMMON_PATTERNS = ["*.json", "*.txt", "*.model", "*.tiktoken", "*.jinja", "LICENSE*", "README*", "NOTICE*"]
DENY_PATTERNS = ["*.bin", "*.pt", "*.pth", "*.pkl", "*.ckpt", "*.h5", "*.msgpack", "*.onnx", "original/*"]

# Default plan (Fase 01/02/04): order = what unblocks work first.
PLAN: list[dict] = [
    {"repo": "Qwen/Qwen2.5-Coder-0.5B", "role": "base T0 Nano / T3"},
    {"repo": "Qwen/Qwen2.5-Coder-1.5B", "role": "base T1 Small / T3"},
    {"repo": "Qwen/Qwen2.5-Coder-7B", "role": "professor alternativo / T2 alternativo"},
    {"repo": "Qwen/Qwen2.5-Coder-14B", "role": "professor primario (KapiTeacher) / T2"},
    {"repo": "Qwen/Qwen3-0.6B", "role": "base Agent-Lite (NPU)"},
    {"repo": "Qwen/Qwen3-1.7B", "role": "base Agent-Lite (CPU)"},
    {"repo": "Qwen/Qwen3-8B", "role": "base T4 Agent"},
    {"repo": "Qwen/Qwen3-14B-GGUF", "role": "professor auxiliar instrucao/raciocinio (llama.cpp)",
     "gguf": "*Q4_K_M.gguf"},
    {"repo": "Qwen/Qwen3-4B-Thinking-2507", "role": "alternativa T4 (bake-off)"},
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_lock() -> dict:
    if LOCK_FILE.exists():
        return json.loads(LOCK_FILE.read_text(encoding="utf-8"))
    return {"version": 1, "repos": {}}


def save_lock(lock: dict) -> None:
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = LOCK_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(LOCK_FILE)


def download(repo: str, name: str | None, revision: str | None, gguf_pattern: str | None, role: str) -> None:
    api = HfApi()
    info = api.model_info(repo, revision=revision, files_metadata=True)
    sha = info.sha
    folder_name = name or repo.split("/")[-1]
    target = BASE_REPOS / folder_name

    allow = COMMON_PATTERNS + ([gguf_pattern] if gguf_pattern else WEIGHT_PATTERNS)
    remote_sizes = {s.rfilename: s.size for s in (info.siblings or []) if s.size is not None}
    print(f"[{repo}] revision {sha} -> {target}", flush=True)

    snapshot_download(
        repo_id=repo,
        revision=sha,
        local_dir=target,
        allow_patterns=allow,
        ignore_patterns=DENY_PATTERNS,
        max_workers=8,
    )

    files = []
    for path in sorted(p for p in target.rglob("*") if p.is_file() and ".cache" not in p.parts):
        rel = path.relative_to(target).as_posix()
        size = path.stat().st_size
        expected = remote_sizes.get(rel)
        if expected is not None and expected != size:
            raise RuntimeError(f"{repo}: size mismatch for {rel}: local {size} != remote {expected}")
        files.append({"path": rel, "size": size, "sha256": sha256_of(path)})
        print(f"  {rel}  {size / 2**30:.2f} GiB", flush=True)

    card = info.card_data.to_dict() if info.card_data else {}
    lock = load_lock()
    lock["repos"][folder_name] = {
        "repo_id": repo,
        "revision": sha,
        "role": role,
        "license": card.get("license"),
        "downloaded_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "total_bytes": sum(f["size"] for f in files),
        "files": files,
    }
    save_lock(lock)
    print(f"[{repo}] ok: {len(files)} files, {lock['repos'][folder_name]['total_bytes'] / 2**30:.2f} GiB", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("repos", nargs="*")
    parser.add_argument("--plan", action="store_true", help="download the default plan")
    parser.add_argument("--name", help="target folder name under BaseRepos (single repo only)")
    parser.add_argument("--revision", help="commit SHA to pin (single repo only); default = current main")
    parser.add_argument("--gguf-pattern", help="download only GGUF files matching this pattern")
    parser.add_argument("--role", default="")
    args = parser.parse_args()

    jobs = [dict(item) for item in PLAN] if args.plan else [
        {"repo": r, "role": args.role, "gguf": args.gguf_pattern, "name": args.name, "revision": args.revision}
        for r in args.repos
    ]
    if not jobs:
        parser.error("informe repos ou --plan")

    failures = 0
    for job in jobs:
        try:
            download(job["repo"], job.get("name"), job.get("revision"), job.get("gguf"), job.get("role", ""))
        except Exception as error:  # keep going with the remaining repos
            failures += 1
            print(f"[{job['repo']}] FAILED: {error}", file=sys.stderr, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
