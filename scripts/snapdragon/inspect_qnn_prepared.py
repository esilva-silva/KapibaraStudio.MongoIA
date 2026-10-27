"""Read-only structural/numeric/tokenizer audit of a prepared four-part QNN LLM.

This manual tool does not create sessions, run inference, repair resource paths,
load a model through Transformers, or write inside the artifact directory.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import time

import numpy as np
import onnx
from onnx import numpy_helper
from tokenizers import Tokenizer


COMPONENTS = ("embeddings", "context", "iterator", "lm_head")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe(value):
    tensor = value.type.tensor_type
    return {"name": value.name, "dtype": onnx.TensorProto.DataType.Name(tensor.elem_type),
            "shape": [dim.dim_value if dim.HasField("dim_value") else dim.dim_param for dim in tensor.shape.dim]}


def safe_file(root, relative):
    path = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    if not relative or path.is_absolute() or windows.is_absolute() or windows.drive or "\\" in relative or ".." in path.parts:
        raise ValueError("Unsafe artifact-relative path: " + relative)
    target = root.joinpath(*path.parts).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError("Missing or escaping artifact file: " + relative)
    return target


def all_tensors(graph):
    yield from graph.initializer
    for sparse in graph.sparse_initializer:
        yield sparse.values
        yield sparse.indices
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                yield attribute.t
            elif attribute.type == onnx.AttributeProto.TENSORS:
                yield from attribute.tensors
            elif attribute.type == onnx.AttributeProto.GRAPH:
                yield from all_tensors(attribute.g)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for nested in attribute.graphs:
                    yield from all_tensors(nested)


def numeric_audit(model, directory):
    # Q/DQ scales are a strict positive gate. Native block-quantized weight
    # scales may be signed; count/report their signs without treating them as Q/DQ.
    qdq_scales, weight_scales = set(), set()
    for node in model.graph.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear") and len(node.input) > 1:
            qdq_scales.add(node.input[1])
        elif node.op_type in ("MatMulNBits", "GatherBlockQuantized") and len(node.input) > 2:
            weight_scales.add(node.input[2])
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    absent = sorted((qdq_scales | weight_scales) - initializers.keys())
    floating_count = elements = scale_count = scale_elements = 0
    bad_finite, bad_positive, signed_weights = [], [], []
    scale_minimum = None
    for tensor in all_tensors(model.graph):
        typename = onnx.TensorProto.DataType.Name(tensor.data_type)
        floating = typename.startswith("FLOAT") or typename == "DOUBLE" or typename == "BFLOAT16"
        if not floating:
            if tensor.name in qdq_scales:
                bad_positive.append({"name": tensor.name, "reason": "Q/DQ scale is not floating point", "dtype": typename})
            continue
        array = numpy_helper.to_array(tensor, base_dir=str(directory))
        try:
            finite = bool(np.isfinite(array).all())
        except TypeError:
            finite = bool(np.isfinite(array.astype(np.float64)).all())
        floating_count += 1
        elements += int(array.size)
        if not finite:
            bad_finite.append(tensor.name)
        if tensor.name in qdq_scales:
            scale_count += 1
            scale_elements += int(array.size)
            if not finite or not array.size or not bool((array > 0).all()):
                bad_positive.append({"name": tensor.name, "reason": "non-finite, empty or non-positive Q/DQ scale"})
            elif array.size:
                minimum = float(array.min())
                scale_minimum = minimum if scale_minimum is None else min(scale_minimum, minimum)
        if tensor.name in weight_scales:
            signed_weights.append({"name": tensor.name, "elements": int(array.size), "finite": finite,
                "negative": int(np.count_nonzero(array < 0)), "zero": int(np.count_nonzero(array == 0)),
                "positive": int(np.count_nonzero(array > 0)),
                "minimum": float(array.min()) if array.size and finite else None,
                "maximum": float(array.max()) if array.size and finite else None})
    return {"floating_tensors": floating_count, "floating_elements": elements, "non_finite_tensors": bad_finite,
            "positive_scale_gate": "QuantizeLinear/DequantizeLinear only; signed block-weight scales reported separately",
            "qdq_scale_tensors": scale_count, "qdq_scale_elements": scale_elements,
            "minimum_qdq_scale": scale_minimum, "invalid_qdq_scales": bad_positive,
            "block_weight_scale_signs": signed_weights, "unresolved_scale_inputs": absent,
            "passed": not (bad_finite or bad_positive or absent)}


def inspect_component(directory, name, pipeline):
    path = safe_file(directory, pipeline["filename"])
    model = onnx.load(path, load_external_data=False)
    external = []
    for tensor in all_tensors(model.graph):
        if tensor.data_location != onnx.TensorProto.EXTERNAL:
            continue
        info = {item.key: item.value for item in tensor.external_data}
        target = safe_file(directory, info.get("location", ""))
        offset, length = int(info.get("offset", 0)), info.get("length")
        if offset < 0 or (length is not None and int(length) < 0):
            raise ValueError("Negative external-data extent: " + tensor.name)
        if offset > target.stat().st_size or length is not None and offset + int(length) > target.stat().st_size:
            raise ValueError("External-data extent exceeds file: " + tensor.name)
        external.append({"tensor": tensor.name, "location": info["location"], "offset": offset,
                         "length": int(length) if length is not None else None})
    # Full ONNX checker only; custom Microsoft kernels still need runtime validation.
    checker_error = None
    try:
        onnx.checker.check_model(str(path), full_check=True)
    except Exception as exc:
        # Preserve checker/security failures; do not bypass guards or mutate/copy artifacts here.
        checker_error = type(exc).__name__ + ": " + str(exc)
    inputs, outputs = [describe(v) for v in model.graph.input], [describe(v) for v in model.graph.output]
    failures = ["ONNX checker: " + checker_error] if checker_error else []
    if [v["name"] for v in inputs] != pipeline["inputs"]:
        failures.append("Pipeline input names/order differ from graph")
    if [v["name"] for v in outputs] != pipeline["outputs"]:
        failures.append("Pipeline output names/order differ from graph")
    try:
        numeric = numeric_audit(model, directory)
    except Exception as exc:
        numeric = {"passed": False, "error": type(exc).__name__ + ": " + str(exc)}
    if not numeric["passed"]:
        failures.append("Initializer/scales numeric audit failed")
    result = {"component": name, "file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path),
              "onnx_checker_full_check": "failed" if checker_error else "passed", "onnx_checker_error": checker_error, "inputs": inputs, "outputs": outputs,
              "node_counts": dict(Counter(node.op_type for node in model.graph.node)),
              "external_tensor_count": len(external), "external_data_locations": sorted({v["location"] for v in external}),
              "external_extents_safe": True, "external_file_hard_link_counts": {location: safe_file(directory, location).stat().st_nlink for location in {v["location"] for v in external}},
              "numeric": numeric, "failures": failures}
    return result


def resolve_resource(directory, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (directory / path).resolve()


def resource_diagnostics(config, directory):
    outside = []
    def visit(value, key_path="$"):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "model_path" and isinstance(item, str):
                    path = resolve_resource(directory, item)
                    if not path.is_relative_to(directory):
                        outside.append({"field": key_path + "." + key, "path": item, "exists": path.exists()})
                elif key == "additional_files" and isinstance(item, list):
                    for index, raw in enumerate(item):
                        path = resolve_resource(directory, raw)
                        if not path.is_relative_to(directory):
                            outside.append({"field": key_path + "." + key + f"[{index}]", "path": raw, "exists": path.exists()})
                else:
                    visit(item, key_path + "." + key)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, key_path + f"[{index}]")
    visit(config)
    return {"references_outside_artifact_directory": outside,
            "relocation_required": bool(outside), "paths_modified": False}


def validate_expected_window(genai, expected_window=64):
    """Require the explicitly requested supported window; never infer acceptance from the artifact."""
    if type(expected_window) is not int or expected_window not in (32, 64):
        raise ValueError("Expected window must be 32 or 64")
    actual = genai["model"]["decoder"]["sliding_window"]["window_size"]
    if type(actual) is not int or actual != expected_window:
        raise ValueError(f"GenAI window mismatch: requested {expected_window}, artifact {actual}")
    return expected_window


def validate_expected_head_size(genai, expected_head_size=64):
    """Keep the Nano default explicit; Small artifacts must be requested with head size128."""
    if type(expected_head_size) is not int or expected_head_size not in (64, 128):
        raise ValueError("Expected head size must be 64 or 128")
    actual = genai["model"]["decoder"]["head_size"]
    if type(actual) is not int or actual != expected_head_size:
        raise ValueError(f"GenAI head size mismatch: requested {expected_head_size}, artifact {actual}")
    return expected_head_size


def validate_interfaces(components, genai, expected_window=64, expected_head_size=64):
    window = validate_expected_window(genai, expected_window)
    head_size = validate_expected_head_size(genai, expected_head_size)
    failures = []
    decoder = genai["model"]["decoder"]
    layers, heads = decoder["num_hidden_layers"], decoder["num_key_value_heads"]
    hidden, vocab = decoder["hidden_size"], genai["model"]["vocab_size"]
    embedding_out = components["embeddings"]["outputs"]
    head_in = components["lm_head"]["inputs"]
    if len(components["embeddings"]["inputs"]) != 1 or components["embeddings"]["inputs"][0] != {"name": "input_ids", "dtype": "INT64", "shape": ["batch_size", "sequence_length"]}:
        failures.append("Embeddings input must be INT64 input_ids with symbolic batch/sequence")
    if len(embedding_out) != 1 or embedding_out[0]["dtype"] != "UINT16" or embedding_out[0]["shape"] != ["batch_size", "sequence_length", hidden]:
        failures.append("Embeddings output must be UINT16 [B,S,hidden]")
    if len(head_in) != 1 or head_in[0]["dtype"] != "UINT16" or head_in[0]["shape"] != ["batch_size", "sequence_length", hidden]:
        failures.append("Head input must be UINT16 [B,S,hidden]")
    if components["lm_head"]["outputs"] != [{"name": "logits", "dtype": "FLOAT", "shape": ["batch_size", "sequence_length", vocab]}]:
        failures.append("Head logits must be FLOAT [B,S,vocab]")
    for name, sequence in (("context", window), ("iterator", 1)):
        ins = {v["name"]: v for v in components[name]["inputs"]}
        outs = {v["name"]: v for v in components[name]["outputs"]}
        if len(embedding_out) == 1 and ins.get(embedding_out[0]["name"]) != {"name": embedding_out[0]["name"], "dtype": "UINT16", "shape": [1, sequence, hidden]}:
            failures.append(name + " embeddings boundary mismatch")
        if len(head_in) == 1 and outs.get(head_in[0]["name"]) != {"name": head_in[0]["name"], "dtype": "UINT16", "shape": [1, sequence, hidden]}:
            failures.append(name + " head boundary mismatch")
        for logical, shape in (("past_sequence_length", [1, 1]), ("total_sequence_length", [])):
            actual = decoder["inputs"][logical]
            if ins.get(actual) != {"name": actual, "dtype": "INT32", "shape": shape}:
                failures.append(name + " sequence-length input mismatch: " + actual)
        expected_inputs, expected_outputs = set(), set()
        for layer in range(layers):
            for kind in ("key", "value"):
                before = decoder["inputs"][f"past_{kind}_names"] % layer
                after = decoder["outputs"][f"present_{kind}_names"] % layer
                expected_inputs.add(before); expected_outputs.add(after)
                for label, value in ((before, ins.get(before)), (after, outs.get(after))):
                    if value is None or value["dtype"] != "FLOAT" or len(value["shape"]) != 4 or value["shape"][:2] != [1, heads] or value["shape"][-1] != head_size:
                        failures.append(name + " invalid KV: " + label)
                if before in ins and after in outs and ins[before]["shape"] != outs[after]["shape"]:
                    failures.append(name + " past/present shapes differ with shared buffer: " + before)
        if {key for key in ins if key.startswith("past_key_values.")} != expected_inputs or {key for key in outs if key.startswith("present.")} != expected_outputs:
            failures.append(name + " incomplete or unexpected KV names")
    return failures


def tokenizer_byte_parity(directory, source, required=False):
    """Optionally require the original tokenizer files byte-for-byte, independently of ID parity.

    config.json describes the derived model and is not compared against the source.
    generation_config.json and tokenizer auxiliaries follow export_gpu.TOKEN_FILES:
    require byte parity when they are present in the original snapshot.
    """
    report = {"required": required, "checked": False, "passed": True, "files": [],
              "optional_not_present_in_source": []}
    if not required:
        return report
    directory, source = Path(directory).resolve(), Path(source).resolve()
    mandatory = ("tokenizer.json", "tokenizer_config.json")
    optional = ("vocab.json", "merges.txt", "generation_config.json", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja")
    selected = list(mandatory)
    for name in optional:
        if (source / name).is_file():
            selected.append(name)
        else:
            report["optional_not_present_in_source"].append(name)
    for name in selected:
        entry = {"path": name, "source_sha256": None, "artifact_sha256": None, "matches": False}
        for root, label in ((source, "source"), (directory, "artifact")):
            try:
                path = safe_file(root, name)
                entry[label + "_sha256"] = sha256(path)
                entry[label + "_bytes"] = path.stat().st_size
            except (OSError, ValueError) as exc:
                entry[label + "_error"] = str(exc)
        entry["matches"] = entry["source_sha256"] is not None and entry["source_sha256"] == entry["artifact_sha256"]
        report["files"].append(entry)
    report["checked"] = True
    report["passed"] = all(entry["matches"] for entry in report["files"])
    return report


def tokenizer_parity(directory, source, sample_directory):
    original = Tokenizer.from_file(str(source / "tokenizer.json"))
    exported = Tokenizer.from_file(str(directory / "tokenizer.json"))
    mismatch, corpus_mismatch, counts, hashes = [], [], {}, {}
    for name, count in (("calibration", 256), ("evaluation", 200)):
        path = sample_directory / (name + ".jsonl")
        hashes[name] = sha256(path)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        counts[name] = len(rows)
        if len(rows) != count:
            raise ValueError("Unexpected sample count: " + name)
        for row in rows:
            prompt = row.get("prompt") or row["prompt_text"]
            old = original.encode(prompt, add_special_tokens=False).ids
            new = exported.encode(prompt, add_special_tokens=False).ids
            if old != new:
                mismatch.append({"split": name, "id": row["id"], "original_tokens": len(old), "exported_tokens": len(new)})
            if row.get("prompt_ids") != old:
                corpus_mismatch.append({"split": name, "id": row["id"]})
    return {"counts": counts, "prompts_checked": sum(counts.values()), "sample_sha256": hashes,
            "original_tokenizer_sha256": sha256(source / "tokenizer.json"), "exported_tokenizer_sha256": sha256(directory / "tokenizer.json"),
            "mismatches": mismatch, "corpus_prompt_id_mismatches": corpus_mismatch,
            "passed": not mismatch and not corpus_mismatch}


def inspect(directory, source, samples, expected_window=64, require_original_tokenizer=False, expected_head_size=64):
    started = time.monotonic()
    directory, source, samples = Path(directory).resolve(), Path(source).resolve(), Path(samples).resolve()
    report = {"schema": "qnn-prepared-artifact-validation-v1", "artifact_directory": str(directory),
              "scope": "structural, numeric and tokenizer audit only", "inference_executed": False, "qualified": False,
              "expected_window": expected_window, "expected_head_size": expected_head_size,
              "versions": {name: importlib.metadata.version(name) for name in ("onnx", "numpy", "tokenizers")},
              "components": {}, "failures": [], "limitations": ["No generation, HTP compilation, device profiling or quality measurement.",
                  "Symbolic KV sequence dimensions do not independently prove runtime capacity2048."]}
    try:
        report["byte_parity"] = tokenizer_byte_parity(directory, source, require_original_tokenizer)
        if not report["byte_parity"]["passed"]:
            report["failures"].append("Original tokenizer byte parity failed")
        composite = json.loads((directory / "model_config.json").read_text(encoding="utf-8"))
        genai = json.loads((directory / "genai_config.json").read_text(encoding="utf-8"))
        cfg = composite["config"]
        report["resource_diagnostics"] = resource_diagnostics(composite, directory)
        if composite["type"].lower() != "compositemodel" or cfg["model_component_names"] != list(COMPONENTS):
            raise ValueError("Expected four ordered CompositeModel components")
        if len(cfg["model_components"]) != 4:
            raise ValueError("Expected four component resource descriptions")
        decoder = genai["model"]["decoder"]
        expected = {"type": genai["model"]["type"], "context_length": genai["model"]["context_length"],
                    "window_size": decoder["sliding_window"]["window_size"], "kv_heads": decoder["num_key_value_heads"], "head_size": decoder["head_size"]}
        report["genai"] = expected
        validate_expected_window(genai, expected_window)
        validate_expected_head_size(genai, expected_head_size)
        if expected != {"type": "decoder-pipeline", "context_length": 2048, "window_size": expected_window, "kv_heads": 2, "head_size": expected_head_size}:
            raise ValueError("GenAI pipeline/context/window/KV metadata mismatch")
        if genai["search"]["max_length"] != 2048 or genai["search"]["past_present_share_buffer"] is not True:
            raise ValueError("Expected max_length2048 and shared past/present buffer")
        if decoder["sliding_window"].get("slide_key_value_cache") is not False:
            raise ValueError("Expected slide_key_value_cache=false")
        if len(decoder["pipeline"]) != 1 or set(decoder["pipeline"][0]) != set(COMPONENTS):
            raise ValueError("Expected one four-component GenAI pipeline")
        pipeline = decoder["pipeline"][0]
        if pipeline["context"].get("run_on_token_gen") is not False or pipeline["iterator"].get("run_on_prompt") is not False or pipeline["lm_head"].get("is_lm_head") is not True:
            raise ValueError("Context/iterator/head scheduling mismatch")
        for name, component in zip(COMPONENTS, cfg["model_components"]):
            if resolve_resource(directory, component["config"]["model_path"]) != directory or component["config"]["onnx_file_name"] != pipeline[name]["filename"]:
                raise ValueError("Component executable resource points outside output or differs from pipeline: " + name)
            try:
                result = inspect_component(directory, name, pipeline[name])
                report["components"][name] = result
                report["failures"].extend(name + ": " + error for error in result["failures"])
            except Exception as exc:
                report["failures"].append(name + ": " + type(exc).__name__ + ": " + str(exc))
        if set(report["components"]) == set(COMPONENTS):
            report["failures"].extend(validate_interfaces(report["components"], genai, expected_window, expected_head_size))
        report["tokenizer"] = tokenizer_parity(directory, source, samples)
        if not report["tokenizer"]["passed"]:
            report["failures"].append("Tokenizer or stored corpus IDs differ")
        external_names = {location for component in report["components"].values() for location in component["external_data_locations"]}
        required = {"model_config.json", "genai_config.json", "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", *external_names}
        required.update(component["file"] for component in report["components"].values())
        required.update(entry["path"] for entry in report["byte_parity"]["files"] if entry["artifact_sha256"] is not None)
        report["artifact_files"] = [{"path": name, "size": safe_file(directory, name).stat().st_size, "sha256": sha256(safe_file(directory, name))} for name in sorted(required)]
    except Exception as exc:
        report["failures"].append(type(exc).__name__ + ": " + str(exc))
    report["passed"] = not report["failures"]
    report["transfer_ready"] = report["passed"] and not report.get("resource_diagnostics", {}).get("relocation_required", True)
    report["duration_seconds"] = round(time.monotonic() - started, 3)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expected-window", type=int, choices=(32, 64), default=64,
                        help="Explicit expected prefill window; metadata and context boundaries must match")
    parser.add_argument("--expected-head-size", type=int, choices=(64, 128), default=64,
                        help="Expected KV head dimension: Nano64 (default), Small128")
    parser.add_argument("--require-original-tokenizer", action="store_true",
                        help="Require source tokenizer bytes and all tokenizer auxiliaries present in the source snapshot")
    args = parser.parse_args(argv)
    if args.out.resolve().is_relative_to(args.artifact_dir.resolve()):
        parser.error("Report must be outside the read-only artifact directory")
    report = inspect(args.artifact_dir, args.source, args.samples, args.expected_window, args.require_original_tokenizer, args.expected_head_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"passed": report["passed"], "transfer_ready": report["transfer_ready"], "components": list(report["components"]),
                      "tokenizer_prompts": report.get("tokenizer", {}).get("prompts_checked"), "failures": report["failures"], "report": str(args.out.resolve())}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
