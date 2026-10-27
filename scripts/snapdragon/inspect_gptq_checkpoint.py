"""Manually audit a completed GPTQ checkpoint without model autoload or inference.

All tensor reads use safetensors.safe_open(framework="pt", device="cpu"), one
at a time. No trust_remote_code, pickle loader, Transformers model or GPU job.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("File changed while hashing: " + str(path))
    return digest.hexdigest()


def validate_paths(artifact, source, out):
    artifact, source, out = Path(artifact).resolve(), Path(source).resolve(), Path(out).resolve()
    if artifact == source or artifact.is_relative_to(source) or source.is_relative_to(artifact):
        raise ValueError("Artifact and source must be separate, non-overlapping directories")
    if out.is_relative_to(artifact) or out.is_relative_to(source):
        raise ValueError("Report must be outside artifact and source directories")
    if not artifact.is_dir() or not source.is_dir():
        raise ValueError("Artifact and pinned source directories must exist")
    return artifact, source, out


def safe_file(root, relative):
    posix, windows = PurePosixPath(relative), PureWindowsPath(relative)
    if not relative or posix.is_absolute() or windows.drive or ".." in posix.parts or "\\" in relative:
        raise ValueError("Unsafe checkpoint-relative path: " + relative)
    path = root.joinpath(*posix.parts).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Missing or escaping checkpoint file: " + relative)
    return path


def checkpoint_files(artifact):
    paths, signatures, inventory = [], {}, []
    for path in sorted(artifact.rglob("*")):
        relative = path.relative_to(artifact)
        if any(part in (".cache", ".git", "__pycache__") for part in relative.parts) or not path.is_file():
            continue
        path = safe_file(artifact, relative.as_posix())
        stat = path.stat()
        digest = sha256(path)
        paths.append(path)
        signatures[str(path)] = (stat.st_size, stat.st_mtime_ns)
        inventory.append({"path": relative.as_posix(), "size": stat.st_size, "sha256": digest})
    if not paths:
        raise ValueError("Checkpoint directory is empty")
    return paths, signatures, inventory


def non_finite_count(torch, tensor):
    count = 0
    for chunk in tensor.reshape(-1).split(1 << 20):
        try:
            finite = torch.isfinite(chunk)
        except RuntimeError:
            finite = torch.isfinite(chunk.to(torch.float64))
        count += int((~finite).sum().item())
    return count


def tensor_audit(artifact, paths):
    # Mask devices before the first Torch import. An already initialized GPU
    # context cannot be made CPU-only by changing environment variables.
    existing = sys.modules.get("torch")
    if existing is not None and existing.cuda.is_initialized():
        raise RuntimeError("Use a dedicated CPU process; Torch CUDA/HIP is already initialized")
    os.environ["HIP_VISIBLE_DEVICES"] = "-1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import torch
    from safetensors import safe_open

    indices = [p for p in paths if p.name.endswith(".safetensors.index.json")]
    weights = {p.relative_to(artifact).as_posix(): p for p in paths if p.suffix == ".safetensors"}
    if len(indices) > 1 or not weights:
        raise ValueError("Expected a single safetensors checkpoint or one sharded index")
    weight_map = None
    if indices:
        weight_map = json.loads(indices[0].read_text(encoding="utf-8"))["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map or not all(isinstance(k, str) and isinstance(v, str) for k, v in weight_map.items()):
            raise ValueError("Invalid safetensors weight_map")
        for relative in set(weight_map.values()):
            safe_file(artifact, relative)
        if set(weight_map.values()) != set(weights):
            raise ValueError("Index shards differ from checkpoint safetensors files")
    elif len(weights) != 1:
        raise ValueError("Multiple safetensors files require an explicit shard index")
    seen, dtypes, tensor_counts, element_counts, bad = set(), Counter(), Counter(), Counter(), []
    per_file = []
    for relative, path in sorted(weights.items()):
        count = 0
        with safe_open(str(path), framework="pt", device="cpu") as tensors:
            for name in tensors.keys():
                if name in seen:
                    raise ValueError("Tensor appears in more than one shard: " + name)
                if weight_map is not None and weight_map.get(name) != relative:
                    raise ValueError("Tensor is missing or mapped to a different shard in the index: " + name)
                seen.add(name); count += 1
                tensor = tensors.get_tensor(name)
                if tensor.device.type != "cpu":
                    raise RuntimeError("Tensor was not read on CPU")
                dtype = str(tensor.dtype).removeprefix("torch.")
                dtypes[dtype] += 1
                kind = "floating" if tensor.is_floating_point() else "complex" if tensor.is_complex() else "boolean" if tensor.dtype == torch.bool else "integer"
                tensor_counts[kind] += 1
                element_counts[kind] += tensor.numel()
                if kind in ("floating", "complex"):
                    non_finite = non_finite_count(torch, tensor)
                    if non_finite:
                        bad.append({"tensor": name, "file": relative, "dtype": dtype, "shape": list(tensor.shape), "non_finite_elements": non_finite})
                del tensor
        per_file.append({"path": relative, "tensor_count": count})
    if not seen or weight_map is not None and seen != set(weight_map):
        raise ValueError("Checkpoint is empty or some indexed tensors are missing")
    return {"tensor_count": len(seen), "tensor_counts": dict(tensor_counts), "element_counts": dict(element_counts),
            "dtype_tensor_counts": dict(dtypes), "files": per_file, "non_finite_tensors": bad,
            "framework": "pt", "device": "cpu", "hip_visible_devices": os.environ["HIP_VISIBLE_DEVICES"],
            "one_tensor_at_a_time": True, "passed": not bad}


def config_audit(artifact):
    config = json.loads(safe_file(artifact, "config.json").read_text(encoding="utf-8"))
    quant = config.get("quantization_config") or {}
    head = []
    for pattern, options in (quant.get("dynamic") or {}).items():
        regex = pattern[2:] if pattern.startswith(("+:", "-:")) else pattern
        if not pattern.startswith("-:") and re.search(regex, "lm_head"):
            head.append({"pattern": pattern, "bits": options.get("bits")})
    failures = []
    if type(quant.get("bits")) is not int or quant["bits"] != 4:
        failures.append("Declared global quantization bits must be 4")
    if quant.get("quant_method") != "gptq":
        failures.append("Expected declared GPTQ quant_method")
    if not head or any(type(item["bits"]) is not int or item["bits"] != 8 for item in head):
        failures.append("Declared dynamic lm_head bits must be 8")
    return {"bits": quant.get("bits"), "quant_method": quant.get("quant_method"), "checkpoint_format": quant.get("checkpoint_format"),
            "lm_head_enabled": quant.get("lm_head"), "dynamic_lm_head": head, "group_size": quant.get("group_size"),
            "architecture": {key: config.get(key) for key in ("model_type", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "vocab_size", "max_position_embeddings")},
            "declarations_only": True, "failures": failures, "passed": not failures}


def tokenizer_audit(artifact, source, samples):
    from tokenizers import Tokenizer
    original_path, artifact_path = safe_file(source, "tokenizer.json"), safe_file(artifact, "tokenizer.json")
    original, candidate = Tokenizer.from_file(str(original_path)), Tokenizer.from_file(str(artifact_path))
    mismatch, stored_mismatch, counts, hashes = [], [], {}, {}
    for split, expected in (("calibration", 256), ("evaluation", 200)):
        path = safe_file(samples, split + ".jsonl")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != expected or len({row["id"] for row in rows}) != expected:
            raise ValueError("Expected complete sample count and unique IDs: " + split)
        counts[split], hashes[split] = len(rows), sha256(path)
        for row in rows:
            prompt = row.get("prompt") or row.get("prompt_text")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("Missing sample prompt")
            old = original.encode(prompt, add_special_tokens=False).ids
            new = candidate.encode(prompt, add_special_tokens=False).ids
            if old != new:
                mismatch.append({"split": split, "id": row["id"], "original_tokens": len(old), "artifact_tokens": len(new)})
            stored = row.get("prompt_ids")
            if not isinstance(stored, list) or any(type(value) is not int for value in stored) or stored != old:
                stored_mismatch.append({"split": split, "id": row["id"]})
    return {"counts": counts, "prompts_checked": sum(counts.values()), "sample_sha256": hashes,
            "source_tokenizer_sha256": sha256(original_path), "artifact_tokenizer_sha256": sha256(artifact_path),
            "id_mismatches": mismatch, "stored_id_mismatches": stored_mismatch, "passed": not mismatch and not stored_mismatch}


def inspect(artifact, source, samples):
    artifact, source, samples = Path(artifact).resolve(), Path(source).resolve(), Path(samples).resolve()
    if artifact == source or artifact.is_relative_to(source) or source.is_relative_to(artifact):
        raise ValueError("Artifact and source must be separate, non-overlapping directories")
    started = time.monotonic()
    report = {"schema": "gptq-checkpoint-inspection-v1", "artifact": str(artifact), "source": str(source),
              "inference_executed": False, "gpu_used": False, "qualified": False, "failures": [],
              "limitations": ["No model autoload, generation, perplexity, speed or quality measurement.",
                              "Quantization bits are configuration declarations, not independently decoded packing proof."],
              "versions": {name: importlib.metadata.version(name) for name in ("safetensors", "tokenizers", "torch")}}
    try:
        source_manifest_path = safe_file(source, "source_manifest.json")
        manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest.get("repo_id"), str) or not re.fullmatch("[0-9a-f]{40}", manifest.get("revision", "")):
            raise ValueError("Source must identify a pinned public repository revision")
        report["source_pin"] = {"repo_id": manifest["repo_id"], "revision": manifest["revision"], "manifest_sha256": sha256(source_manifest_path)}
        paths, signatures, report["files"] = checkpoint_files(artifact)
        report["config"] = config_audit(artifact)
        report["failures"].extend(report["config"]["failures"])
        report["tensors"] = tensor_audit(artifact, paths)
        if not report["tensors"]["passed"]:
            report["failures"].append("Checkpoint contains non-finite floating/complex values")
        report["tokenizer"] = tokenizer_audit(artifact, source, samples)
        if not report["tokenizer"]["passed"]:
            report["failures"].append("Tokenizer or stored prompt IDs differ")
        changed = [str(path) for path in paths if (path.stat().st_size, path.stat().st_mtime_ns) != signatures[str(path)]]
        report["stable_artifact_files"] = not changed
        if changed:
            report["failures"].append("Checkpoint changed during inspection: " + ", ".join(changed))
    except Exception as exc:
        report["failures"].append(type(exc).__name__ + ": " + str(exc))
    report["passed"] = not report["failures"]
    report["duration_seconds"] = round(time.monotonic() - started, 3)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        artifact, source, out = validate_paths(args.artifact, args.source, args.out)
    except ValueError as exc:
        parser.error(str(exc))
    report = inspect(artifact, source, args.samples)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"passed": report["passed"], "tensor_count": report.get("tensors", {}).get("tensor_count"),
                      "tokenizer_prompts": report.get("tokenizer", {}).get("prompts_checked"), "failures": report["failures"], "report": str(out)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
