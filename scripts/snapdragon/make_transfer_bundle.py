"""Bundle only the reproducible scripts and synthetic sample for the remote host."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--samples", default=ROOT / "data/calib/snapdragon-20261026", type=Path)
    args = parser.parse_args()
    samples = args.samples.resolve()
    if not samples.is_relative_to(ROOT):
        raise ValueError("Sample directory must be inside the project")
    paths = sorted((ROOT / "scripts/snapdragon").glob("*.py"))
    paths += [ROOT / "configs/snapdragon.json"]
    paths += sorted((ROOT / "configs/snapdragon").glob("*.json"))
    paths += sorted((ROOT / "configs/snapdragon").glob("LICENSE*.txt"))
    paths += [samples / name for name in ("calibration.jsonl", "evaluation.jsonl", "manifest.json")]
    entries = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            data = path.read_bytes()
            relative = path.relative_to(ROOT).as_posix()
            archive.writestr(relative, data)
            entries.append({"path": relative, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        archive.writestr("bundle_manifest.json", json.dumps({"schema_version": 1, "files": entries}, indent=2))
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    args.out.with_suffix(args.out.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    print(json.dumps({"path": str(args.out.resolve()), "size": args.out.stat().st_size, "sha256": digest, "files": len(entries)}))


if __name__ == "__main__":
    main()
