"""Check all file hashes before using a downloaded/transferred package."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def verify(directory: Path, manifest_name: str | None = None) -> dict:
    directory = directory.resolve()
    if manifest_name is None:
        manifest_name = next((n for n in ("export_manifest.json", "source_manifest.json") if (directory / n).is_file()), "source_manifest.json")
    manifest = json.loads((directory / manifest_name).read_text(encoding="utf-8-sig"))
    files = manifest["files"]
    if not files:
        raise ValueError("Empty manifest")
    failures = []
    seen = set()
    for entry in files:
        relative = entry["path"]
        target = (directory / relative).resolve()
        if not target.is_relative_to(directory) or relative in seen:
            raise ValueError(f"Invalid or duplicate manifest path: {relative}")
        seen.add(relative)
        if not target.is_file():
            failures.append({"path": relative, "reason": "missing"})
            continue
        expected_size = entry.get("size", entry.get("bytes"))
        if expected_size is not None and target.stat().st_size != expected_size:
            failures.append({"path": relative, "reason": "size"})
            continue
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != entry["sha256"]:
            failures.append({"path": relative, "reason": "sha256"})
    return {"directory": str(directory), "manifest": manifest_name,
            "checked_files": len(files), "passed": not failures, "failures": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--manifest")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = verify(args.directory, args.manifest)
    data = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(data, encoding="utf-8")
    print(data)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
