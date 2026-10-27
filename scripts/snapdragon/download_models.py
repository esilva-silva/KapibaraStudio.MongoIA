"""Download immutable published snapshots, then hash the complete local package."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(config: dict, selected: list[str], root: Path) -> None:
    from huggingface_hub import snapshot_download
    for name in selected:
        spec = config["models"][name]
        revision = spec["revision"]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("revision must be an immutable 40-character commit SHA")
        destination = root / spec["repo_id"].split("/")[-1] / revision
        snapshot_download(repo_id=spec["repo_id"], revision=revision,
                          local_dir=destination, max_workers=2)
        files = []
        for path in sorted(destination.rglob("*")):
            rel = path.relative_to(destination)
            if not path.is_file() or ".cache" in rel.parts or rel.as_posix() == "source_manifest.json":
                continue
            files.append({"path": rel.as_posix(), "size": path.stat().st_size,
                          "sha256": file_hash(path)})
        required = {"model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"}
        if not required.issubset({entry["path"] for entry in files}):
            raise RuntimeError(f"Incomplete snapshot: {destination}")
        manifest = {"schema_version": 1, "repo_id": spec["repo_id"], "revision": revision,
                    "files": files, "total_bytes": sum(entry["size"] for entry in files)}
        (destination / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"model": name, "path": str(destination), "bytes": manifest["total_bytes"],
                          "files": len(files), "verified": True}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=["nano", "small"], default=["nano", "small"])
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    download(cfg, args.models, args.root or Path(cfg["models_root"]))


if __name__ == "__main__":
    main()
