"""Download published FP16 exports and verify both package and source identity."""
import argparse
import json
from pathlib import Path
from download_models import file_hash
from verify_manifest import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--models", nargs="+", choices=("nano", "small"), default=["nano", "small"])
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    root = args.root or Path(cfg["models_root"])
    from huggingface_hub import snapshot_download
    for name in args.models:
        spec = cfg["published_exports"][name]
        source_spec = cfg["models"][name]
        source = root / source_spec["repo_id"].split("/")[-1] / source_spec["revision"]
        destination = root / spec["repo_id"].split("/")[-1] / spec["revision"]
        snapshot_download(repo_id=spec["repo_id"], revision=spec["revision"], local_dir=destination,
                          allow_patterns=["dml-fp16/*", "README.md"], max_workers=2)
        package = destination / "dml-fp16"
        result = verify(package)
        if not result["passed"]:
            raise RuntimeError(f"Export integrity mismatch: {result}")
        manifest = json.loads((package / "export_manifest.json").read_text(encoding="utf-8"))
        source_hashes = manifest["source"]["files_sha256"]
        if not {"model.safetensors", "config.json", "tokenizer.json"} <= source_hashes.keys():
            raise ValueError("Export has incomplete source provenance")
        for path, expected in source_hashes.items():
            local = (source / path).resolve()
            if not local.is_relative_to(source.resolve()) or not local.is_file() or file_hash(local) != expected:
                raise RuntimeError(f"Export does not match the pinned source: {name}/{path}")
        result.update(repo_id=spec["repo_id"], revision=spec["revision"], source_repo=source_spec["repo_id"],
                      source_revision=source_spec["revision"], source_identity_verified=True)
        (destination / "download_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
