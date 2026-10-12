"""Benchmark reproduzível: students (base e especializados), professor e exportações ONNX no contrato real da IDE.

Autocomplete (FIM e completion): Exact Match, Prefix Match, Edit Distance/Similarity, Mean Accepted Prefix Tokens (métrica principal),
aceitação por TAB (porta do IncrementalCompletion), FIM Accuracy (igualdade com espaços normalizados), Syntax Valid e MongoDB Valid
do documento reconstruído, métricas @32 tokens (padrão da IDE).
Chat (criação/alteração/explicação/correção): Chat Intent Accuracy (sintaxe válida + operação esperada + elementos obrigatórios +
coleção), recall de elementos, PT-BR Understanding e English Understanding (intenção por idioma), Correction Accuracy, respostas
completas em 256 tokens, taxa conversacional/Markdown.
Desempenho e recursos: TTFT, tokens/s, pico de RAM do processo, VRAM máxima, tamanho do modelo em disco e parâmetros.
Resultados por tipo, categoria, grupo do benchmark (CRUD, Aggregation, Atlas Search, JSON, Extended JSON, DSL, PT chat, EN chat...),
idioma, dialeto e domínio holdout.

Uso:
  python scripts/benchmark.py run --backend hf --model student/base/Qwen2.5-Coder-0.5B --run-name student-0.5b-base
  python scripts/benchmark.py run --backend hf --model <base> --adapter adapters/student-0.5b --run-name student-0.5b-adapter
  python scripts/benchmark.py run --backend ortgenai --model exports/int4/SlopCoder-Mongo-0.5B-ONNX-INT4 --run-name onnx-0.5b-int4
  python scripts/benchmark.py compare --runs student-0.5b-base student-0.5b-lora teacher-v1 --out reports/benchmark/comparison.md
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compact import PIPELINE_VERSION, metrics, syntax  # noqa: E402
from compact.common import Pipeline, dir_size, human_bytes, load_yaml, read_jsonl, resolve, run_metadata, write_json  # noqa: E402
from compact.contracts import (CHAT_TYPES, contract_for, ide_rejects, is_chat, max_new_tokens, postprocess,  # noqa: E402
                               prompt_ids)
from compact.env import configure_caches, limit_vram, require_gpu  # noqa: E402
from compact.mongo_validate import validate  # noqa: E402

OPERATION = re.compile(r"\.(find|findOne|countDocuments|estimatedDocumentCount|distinct|aggregate|insertOne|insertMany|updateOne|updateMany|"
                       r"replaceOne|deleteOne|deleteMany|createIndex|dropIndex|bulkWrite)\s*\(")


class PeakRss:
    def __init__(self, interval: float = 0.25) -> None:
        import psutil

        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self.process.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self) -> "PeakRss":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join()


def model_footprint(path: Path) -> dict:
    weights = [p for p in path.iterdir() if p.is_file() and (p.suffix in (".safetensors", ".bin", ".onnx", ".data") or p.name.endswith(".onnx.data"))]
    size = sum(p.stat().st_size for p in weights)
    params = None
    config = path / "config.json"
    if any(p.suffix == ".safetensors" for p in weights):
        import struct

        params = 0
        for file in (p for p in weights if p.suffix == ".safetensors"):
            with file.open("rb") as handle:
                header = json.loads(handle.read(struct.unpack("<Q", handle.read(8))[0]))
            for name, meta in header.items():
                if name != "__metadata__":
                    count = 1
                    for dim in meta["shape"]:
                        count *= dim
                    params += count
    return {"weights_bytes": size, "weights_size": human_bytes(size), "parameters": params, "has_config": config.exists()}


# -- backends -----------------------------------------------------------------------------------------------------------------------------
class HfBackend:
    name = "hf"

    def __init__(self, model_path: Path, adapter: Path | None, cfg: dict, gpu: dict) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = torch.device(f"cuda:{gpu['selected_index']}")
        self.tokenizer = AutoTokenizer.from_pretrained(adapter if adapter and (adapter / "tokenizer.json").exists() else model_path)
        started = time.time()
        load = cfg["hf"].get("load", "bf16")
        extra = {}
        if load == "nf4":  # bake-off Kapibara: 7B/14B não cabem em bf16 nos 16 GB; mesmo formato do QLoRA
            from transformers import BitsAndBytesConfig

            extra["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                                              bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map={"": self.device.index}, low_cpu_mem_usage=True,
                                                     attn_implementation=cfg["hf"].get("attn_implementation", "sdpa"), **extra)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter, torch_device=str(self.device))
        model.eval()
        self.model = model
        self.load_seconds = time.time() - started
        self.description = {"backend": "hf", "load": load, "model": str(model_path), "adapter": str(adapter) if adapter else None,
                            "device": torch.cuda.get_device_name(self.device), "load_seconds": round(self.load_seconds, 1)}
        torch.cuda.reset_peak_memory_stats(self.device)

    def generate(self, ids: list[int], limit: int, stops: frozenset[int], suffix: str) -> dict:
        torch = self.torch
        with torch.inference_mode():
            torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            output = self.model(input_ids=torch.tensor([ids], device=self.device), use_cache=True, logits_to_keep=1)
            token = int(output.logits[0, -1].argmax())
            torch.cuda.synchronize(self.device)
            ttft = time.perf_counter() - start
            generated, past = [token], output.past_key_values
            reason = "eos" if token in stops else "max"
            decode_start = time.perf_counter()
            while token not in stops and len(generated) < limit:
                if len(suffix) >= 2 and suffix in self.decode(generated):
                    reason = "suffix"
                    break
                output = self.model(input_ids=torch.tensor([[token]], device=self.device), past_key_values=past, use_cache=True)
                past = output.past_key_values
                token = int(output.logits[0, -1].argmax())
                generated.append(token)
                if token in stops:
                    reason = "eos"
            torch.cuda.synchronize(self.device)
        return {"tokens": generated, "ttft": ttft, "decode_seconds": time.perf_counter() - decode_start, "stop": reason}

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def memory(self) -> dict:
        return {"vram_max_allocated_gb": round(self.torch.cuda.max_memory_allocated(self.device) / 2**30, 2),
                "vram_max_reserved_gb": round(self.torch.cuda.max_memory_reserved(self.device) / 2**30, 2)}


class OrtGenAiBackend:
    name = "ortgenai"

    def __init__(self, model_path: Path, cfg: dict) -> None:
        import numpy as np
        import onnxruntime_genai as og
        from transformers import AutoTokenizer

        self.np, self.og = np, og
        started = time.time()
        config = og.Config(str(model_path))
        provider = cfg["ortgenai"].get("provider", "cpu")
        config.clear_providers()
        if provider != "cpu":
            config.append_provider(provider)
        threads = cfg["ortgenai"].get("intra_op_num_threads")
        if threads is not None:
            config.overlay(json.dumps({"model": {"decoder": {"session_options": {"intra_op_num_threads": int(threads)}}}}))
        self.model = og.Model(config)
        self.load_seconds = time.time() - started
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.description = {"backend": "ortgenai", "provider": provider, "model": str(model_path), "intra_op_num_threads": threads,
                            "load_seconds": round(self.load_seconds, 1), "onnxruntime_genai": og.__version__}

    def generate(self, ids: list[int], limit: int, stops: frozenset[int], suffix: str) -> dict:
        og = self.og
        params = og.GeneratorParams(self.model)
        params.set_search_options(max_length=len(ids) + limit, do_sample=False)
        generator = og.Generator(self.model, params)
        start = time.perf_counter()
        generator.append_tokens(self.np.array(ids, dtype=self.np.int32))
        generator.generate_next_token()
        token = int(generator.get_next_tokens()[0])
        ttft = time.perf_counter() - start
        generated = [token]
        reason = "eos" if token in stops else "max"
        decode_start = time.perf_counter()
        while token not in stops and len(generated) < limit and not generator.is_done():
            if len(suffix) >= 2 and suffix in self.decode(generated):
                reason = "suffix"
                break
            generator.generate_next_token()
            token = int(generator.get_next_tokens()[0])
            generated.append(token)
            if token in stops:
                reason = "eos"
        decode_seconds = time.perf_counter() - decode_start
        del generator
        return {"tokens": generated, "ttft": ttft, "decode_seconds": decode_seconds, "stop": reason}

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def memory(self) -> dict:
        return {}


# -- métricas por exemplo --------------------------------------------------------------------------------------------------------------------
def norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\r\n", "\n")).strip()


def compact_code(text: str) -> str:
    return re.sub(r"\s+", "", text).replace("'", '"')


def score_text(pred: str, ref: str, tokenizer: Any) -> dict:
    ends = [end for _, end in tokenizer(ref, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]]
    distance, similarity = metrics.edit_similarity(pred, ref)
    chunks, chars = metrics.tab_acceptance(pred, ref)
    accepted = metrics.accepted_prefix_tokens(pred, ref, ends)
    return {"exact_match": metrics.exact_match(pred, ref), "normalized_match": norm_ws(pred) == norm_ws(ref), "prefix_match": metrics.prefix_match(pred, ref),
            "edit_distance": distance, "edit_similarity": similarity, "accepted_prefix_tokens": accepted, "reference_tokens": len(ends),
            "accepted_prefix_ratio": accepted / len(ends) if ends else 0.0, "tab_chunks": chunks, "tab_chars": chars,
            "conversational": metrics.is_conversational(pred) or "```" in pred, "empty": not pred.strip()}


def intent_score(pred: str, example: dict, syntax_ok: bool | None) -> dict:
    intent = example["intent"]
    code = compact_code(pred)
    groups = intent["must_include"]
    satisfied = sum(1 for group in groups if any(compact_code(option) in code for option in group))
    detected = set(OPERATION.findall(pred))
    if example["dialect"] == "json" and pred.strip().startswith("["):
        detected.add("aggregate")
    operation_ok = any(op in detected for op in intent["operations"])
    collection = intent.get("collection")
    collection_ok = example["dialect"] == "json" or not collection or collection in pred
    return {"intent_elements_recall": satisfied / len(groups) if groups else 1.0, "intent_operation_ok": operation_ok, "intent_collection_ok": collection_ok,
            "intent_match": bool(syntax_ok) and operation_ok and collection_ok and satisfied == len(groups), "operations_detected": sorted(detected)}


def attach_validity(rows: list[dict], examples: dict[str, dict]) -> None:
    jobs = []
    for row in rows:
        example = examples[row["id"]]
        kind = syntax.kind_for(example["dialect"])
        for key in ("pred", "pred32"):
            text = row["prediction"] if key == "pred" else row["prediction_ide32"]
            if is_chat(example):
                jobs.append((f"{row['id']}|{key}", text.replace("\r\n", "\n"), kind))
            elif example["complete"]:
                jobs.append((f"{row['id']}|{key}", (example["prefix"] + text + example["suffix"]).replace("\r\n", "\n"), kind))
    results = syntax.check_many(jobs)
    texts = {job[0]: job[1] for job in jobs}
    for row in rows:
        example = examples[row["id"]]
        for key, target in (("pred", "metrics"), ("pred32", "metrics_ide32")):
            job = f"{row['id']}|{key}"
            if job not in results:
                row[target].update({"syntax_valid": None, "mongo_valid": None})
            else:
                ok = results[job][0]
                check = validate(texts[job], example["dialect"]) if ok else None
                row[target].update({"syntax_valid": ok, "mongo_valid": (check.valid if check and check.validatable else (False if not ok else None))})
            if is_chat(example):
                prediction = row["prediction"] if key == "pred" else row["prediction_ide32"]
                row[target].update(intent_score(prediction, example, row[target]["syntax_valid"]))


def evaluate_run(args: argparse.Namespace) -> int:
    pipe = Pipeline(args.pipeline)
    cfg = load_yaml(args.config) if args.config else pipe.config("benchmark")
    configure_caches(pipe.workspace)
    if args.load:
        cfg["hf"]["load"] = args.load
    if args.provider:
        cfg["ortgenai"]["provider"] = args.provider
    if args.threads is not None:
        cfg["ortgenai"]["intra_op_num_threads"] = args.threads
    run_dir = (resolve(args.report_dir) if args.report_dir else resolve(cfg["report_dir"])) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    dataset_path = resolve(args.dataset or cfg["dataset"])
    examples = list(read_jsonl(dataset_path))
    if args.types:
        examples = [e for e in examples if e["type"] in args.types]
    limit = args.limit or (cfg.get("run_limits") or {}).get(args.run_name)  # CLI > limite por execução na config
    if limit:
        examples = examples[:limit]  # o arquivo é embaralhado: os primeiros N formam um subconjunto representativo e comparável
    done: dict[str, dict] = {}
    if args.resume and predictions_path.exists():
        done = {row["id"]: row for row in read_jsonl(predictions_path)}
    elif predictions_path.exists():
        predictions_path.unlink()

    model_path = resolve(args.model)
    gpu = None
    if args.backend == "hf":
        gpu = require_gpu(pipe.gpu_expect)
        gpu["vram_limit"] = limit_vram(gpu, cfg["hf"].get("max_vram_fraction", 0.96))
        backend: HfBackend | OrtGenAiBackend = HfBackend(model_path, resolve(args.adapter) if args.adapter else None, cfg, gpu)
    else:
        backend = OrtGenAiBackend(model_path, cfg)
    tokenizer = backend.tokenizer
    contract = contract_for(tokenizer)
    footprint = model_footprint(model_path)
    if args.adapter:
        footprint["adapter"] = model_footprint(resolve(args.adapter))

    for example in examples[: cfg.get("warmup_examples", 0)]:
        backend.generate(prompt_ids(contract, example, cfg["context_tokens"]), 4, contract.stops, "")

    started = time.time()
    with PeakRss() as rss, predictions_path.open("a", encoding="utf-8", newline="\n") as handle:
        for index, example in enumerate(examples):
            if example["id"] in done:
                continue
            ids = prompt_ids(contract, example, cfg["context_tokens"])
            chat = is_chat(example)
            limit = cfg["chat_max_new_tokens"] if chat else cfg["max_new_tokens"]
            suffix = "" if chat else example["suffix"][:1024]
            result = backend.generate(ids, limit, contract.stops, suffix)
            tokens = [t for t in result["tokens"] if t not in contract.stops]
            raw = backend.decode(tokens)
            prediction = postprocess(raw, example)
            prediction32 = prediction if chat else postprocess(backend.decode(tokens[: cfg["ide_max_new_tokens"]]), example)
            reference = example["output"] if chat else example["completion"]
            complete = result["stop"] in ("eos", "suffix")
            row = {
                "id": example["id"], "type": example["type"], "category": example["category"], "bench_group": example.get("bench_group"),
                "language": example.get("language"), "dialect": example["dialect"], "domain": example.get("domain"), "holdout": example.get("holdout", False),
                "complete": example.get("complete"), "prompt_tokens": len(ids), "generated_tokens": len(result["tokens"]), "stop": result["stop"],
                "ttft_ms": round(result["ttft"] * 1000, 2),
                "tokens_per_second": round((len(result["tokens"]) - 1) / result["decode_seconds"], 2) if len(result["tokens"]) > 1 and result["decode_seconds"] > 0 else None,
                "raw": raw, "prediction": prediction, "prediction_ide32": prediction32, "reference": reference,
                "ide_rejected": ide_rejects(prediction, complete, require_complete=chat),
                "metrics": score_text(prediction, reference, tokenizer), "metrics_ide32": score_text(prediction32, reference, tokenizer),
            }
            row["metrics"]["ide_rejected"] = row["ide_rejected"]
            row["metrics_ide32"]["ide_rejected"] = ide_rejects(prediction32, complete, require_complete=chat)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            done[example["id"]] = row
            if (index + 1) % 50 == 0:
                auto = [r for r in done.values() if r["type"] not in CHAT_TYPES]
                print(f"[{index + 1}/{len(examples)}] {time.time() - started:.0f}s APT={metrics.mean([r['metrics']['accepted_prefix_tokens'] for r in auto]):.2f} "
                      f"EM={metrics.mean([r['metrics']['exact_match'] for r in done.values()]):.3f}", flush=True)
    elapsed = time.time() - started
    rows = [done[e["id"]] for e in examples if e["id"] in done]
    attach_validity(rows, {e["id"]: e for e in examples})
    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(rows)
    summary.update({"run_name": args.run_name, "pipeline_version": PIPELINE_VERSION, "backend": backend.description, "contract": contract.name,
                    "dataset": str(dataset_path), "examples": len(rows), "config": cfg, "wall_seconds": round(elapsed, 1),
                    "peak_rss_gb": round(rss.peak / 2**30, 2), "memory": backend.memory(), "footprint": footprint, "gpu": gpu, "run": run_metadata(cfg)})
    write_json(run_dir / "summary.json", summary)
    (run_dir / "report.md").write_text(render_report(summary), encoding="utf-8", newline="\n")
    print(json.dumps(summary["headline"], indent=1))
    return 0


# -- agregação ------------------------------------------------------------------------------------------------------------------------------
def aggregate(rows: list[dict], key: str = "metrics") -> dict:
    def avg(name: str, subset: Iterable[dict] | None = None) -> float | None:
        values = [r[key][name] for r in (rows if subset is None else subset) if r[key].get(name) is not None]
        return round(metrics.mean([float(v) for v in values]), 4) if values else None

    auto = [r for r in rows if r["type"] not in CHAT_TYPES]
    chat = [r for r in rows if r["type"] in CHAT_TYPES]
    fim = [r for r in rows if r["type"] == "fim"]
    ttft = [r["ttft_ms"] for r in rows]
    tps = [r["tokens_per_second"] for r in rows if r["tokens_per_second"]]
    return {
        "n": len(rows), "n_autocomplete": len(auto), "n_chat": len(chat),
        "mean_accepted_prefix_tokens": avg("accepted_prefix_tokens", auto), "accepted_prefix_ratio": avg("accepted_prefix_ratio", auto),
        "exact_match": avg("exact_match"), "exact_match_autocomplete": avg("exact_match", auto), "prefix_match": avg("prefix_match", auto),
        "edit_distance": avg("edit_distance"), "edit_similarity": avg("edit_similarity"), "mean_tab_chunks": avg("tab_chunks", auto),
        "fim_accuracy": avg("normalized_match", fim), "syntax_valid": avg("syntax_valid"), "syntax_evaluated": sum(1 for r in rows if r[key].get("syntax_valid") is not None),
        "mongo_valid": avg("mongo_valid"), "chat_intent_accuracy": avg("intent_match", chat), "chat_elements_recall": avg("intent_elements_recall", chat),
        "chat_exact_match": avg("normalized_match", chat), "conversational_rate": avg("conversational"), "empty_rate": avg("empty"),
        "ide_rejected_rate": avg("ide_rejected"), "eos_stop_rate": round(metrics.mean([r["stop"] != "max" for r in rows]), 4) if rows else None,
        "mean_generated_tokens": round(metrics.mean([r["generated_tokens"] for r in rows]), 2) if rows else None,
        "ttft_ms_mean": round(metrics.mean(ttft), 1) if ttft else None, "ttft_ms_p50": round(metrics.percentile(ttft, .5), 1) if ttft else None,
        "ttft_ms_p95": round(metrics.percentile(ttft, .95), 1) if ttft else None, "tokens_per_second_mean": round(metrics.mean(tps), 2) if tps else None,
    }


def group(rows: list[dict], key: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row.get(key))].append(row)
    return dict(sorted(grouped.items()))


def summarize(rows: list[dict]) -> dict:
    overall = aggregate(rows)
    chat = [r for r in rows if r["type"] in CHAT_TYPES]
    by_language = {k: aggregate(v) for k, v in group(chat, "language").items()}
    correction = [r for r in rows if r["type"] == "correction"]
    headline = {
        "mean_accepted_prefix_tokens": overall["mean_accepted_prefix_tokens"], "exact_match": overall["exact_match"],
        "exact_match_autocomplete": overall["exact_match_autocomplete"], "prefix_match": overall["prefix_match"], "edit_distance": overall["edit_distance"],
        "fim_accuracy": overall["fim_accuracy"], "syntax_valid": overall["syntax_valid"], "mongo_valid": overall["mongo_valid"],
        "chat_intent_accuracy": overall["chat_intent_accuracy"], "pt_br_understanding": (by_language.get("pt-BR") or {}).get("chat_intent_accuracy"),
        "english_understanding": (by_language.get("en") or {}).get("chat_intent_accuracy"),
        "correction_accuracy": aggregate(correction)["chat_intent_accuracy"] if correction else None,
        "conversational_rate": overall["conversational_rate"], "ttft_ms_mean": overall["ttft_ms_mean"], "ttft_ms_p95": overall["ttft_ms_p95"],
        "tokens_per_second_mean": overall["tokens_per_second_mean"],
    }
    return {"headline": headline, "overall": overall, "overall_ide32": aggregate(rows, "metrics_ide32"),
            "autocomplete_latency": aggregate([r for r in rows if r["type"] not in CHAT_TYPES]),
            "by_type": {k: aggregate(v) for k, v in group(rows, "type").items()},
            "by_bench_group": {k: aggregate(v) for k, v in group(rows, "bench_group").items()},
            "by_category": {k: aggregate(v) for k, v in group(rows, "category").items()},
            "by_language": by_language, "by_dialect": {k: aggregate(v) for k, v in group(rows, "dialect").items()},
            "by_holdout": {("holdout" if k == "True" else "in_domain"): aggregate(v) for k, v in group(rows, "holdout").items()}}


def fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 10 else f"{value:.1f}"
    return str(value)


HEADLINE = [("mean_accepted_prefix_tokens", "Mean Accepted Prefix Tokens"), ("exact_match", "Exact Match"), ("exact_match_autocomplete", "Exact Match (autocomplete)"),
            ("prefix_match", "Prefix Match"), ("edit_distance", "Edit Distance"), ("fim_accuracy", "FIM Accuracy"), ("syntax_valid", "Syntax Valid"),
            ("mongo_valid", "MongoDB Valid"), ("chat_intent_accuracy", "Chat Intent Accuracy"), ("pt_br_understanding", "PT-BR Understanding"),
            ("english_understanding", "English Understanding"), ("correction_accuracy", "Correction Accuracy"), ("conversational_rate", "Conversacional/Markdown"),
            ("ttft_ms_mean", "TTFT médio (ms)"), ("ttft_ms_p95", "TTFT p95 (ms)"), ("tokens_per_second_mean", "Tokens/s")]


def render_report(summary: dict) -> str:
    fp = summary["footprint"]
    lines = [f"# Benchmark `{summary['run_name']}`", "", f"- Backend: `{json.dumps(summary['backend'], ensure_ascii=False)}` · contrato `{summary['contract']}`",
             f"- Exemplos: {summary['examples']} · {summary['wall_seconds']} s · pico RAM {summary['peak_rss_gb']} GB · memória {summary['memory']}",
             f"- Modelo: {fp['weights_size']} em disco · parâmetros {fp['parameters'] or '—'}", "", "## Principais", "", "| métrica | valor | @32 tokens (IDE) |", "|---|---:|---:|"]
    ide32 = summary["overall_ide32"]
    for key, label in HEADLINE:
        lines.append(f"| {label} | {fmt(summary['headline'].get(key))} | {fmt(ide32.get(key) if key in ide32 else None)} |")
    for section, title in (("by_bench_group", "Por grupo do benchmark"), ("by_type", "Por tipo"), ("by_category", "Por categoria"),
                           ("by_language", "Chat por idioma"), ("by_dialect", "Por dialeto"), ("by_holdout", "Domínio")):
        lines += ["", f"## {title}", "", "| grupo | n | APT | EM | FIM acc | Syntax | Mongo | Intent | Conv. | TTFT ms |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, agg in summary[section].items():
            lines.append(f"| {name} | {agg['n']} | {fmt(agg['mean_accepted_prefix_tokens'])} | {fmt(agg['exact_match'])} | {fmt(agg['fim_accuracy'])} | "
                         f"{fmt(agg['syntax_valid'])} | {fmt(agg['mongo_valid'])} | {fmt(agg['chat_intent_accuracy'])} | {fmt(agg['conversational_rate'])} | {fmt(agg['ttft_ms_mean'])} |")
    return "\n".join(lines) + "\n"


def compare(args: argparse.Namespace) -> int:
    pipe = Pipeline(args.pipeline)
    cfg = pipe.config("benchmark")
    base = resolve(args.report_dir or cfg["report_dir"])
    summaries = [json.loads((base / name / "summary.json").read_text(encoding="utf-8")) for name in args.runs]
    reference = summaries[0]
    lines = ["# Comparação de modelos", "", f"Referência dos deltas: `{reference['run_name']}`.", "",
             "| métrica | " + " | ".join(f"`{s['run_name']}`" for s in summaries) + " |", "|---|" + "---:|" * len(summaries)]
    for key, label in HEADLINE:
        cells = []
        for s in summaries:
            value, ref = s["headline"].get(key), reference["headline"].get(key)
            delta = f" ({value - ref:+.3f})" if s is not reference and isinstance(value, (int, float)) and isinstance(ref, (int, float)) else ""
            cells.append(fmt(value) + delta)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("| Pico RAM (GB) | " + " | ".join(fmt(s["peak_rss_gb"]) for s in summaries) + " |")
    lines.append("| VRAM máx. (GB) | " + " | ".join(fmt(s["memory"].get("vram_max_allocated_gb")) for s in summaries) + " |")
    lines.append("| Tamanho do modelo | " + " | ".join(s["footprint"]["weights_size"] for s in summaries) + " |")
    groups = sorted({g for s in summaries for g in s["by_bench_group"]})
    lines += ["", "## APT (autocomplete) e Intent (chat) por grupo", "", "| grupo | " + " | ".join(f"`{s['run_name']}`" for s in summaries) + " |", "|---|" + "---:|" * len(summaries)]
    for name in groups:
        cells = []
        for s in summaries:
            agg = s["by_bench_group"].get(name)
            cells.append("—" if not agg else fmt(agg["chat_intent_accuracy"] if agg["n_chat"] else agg["mean_accepted_prefix_tokens"]))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    out = resolve(args.out) if args.out else base / "comparison.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--pipeline", default="configs/pipeline.yaml")
    run.add_argument("--config", help="padrão: configs/benchmark.yaml")
    run.add_argument("--backend", choices=["hf", "ortgenai"], required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--adapter")
    run.add_argument("--provider")
    run.add_argument("--load", choices=["bf16", "nf4"], help="carga HF (padrão: hf.load da config)")
    run.add_argument("--threads", type=int)
    run.add_argument("--dataset")
    run.add_argument("--types", nargs="*", choices=["fim", "completion", "chat", "correction"])
    run.add_argument("--limit", type=int)
    run.add_argument("--run-name", required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--report-dir")
    cmp_parser = sub.add_parser("compare")
    cmp_parser.add_argument("--pipeline", default="configs/pipeline.yaml")
    cmp_parser.add_argument("--runs", nargs="+", required=True)
    cmp_parser.add_argument("--report-dir")
    cmp_parser.add_argument("--out")
    args = parser.parse_args()
    return evaluate_run(args) if args.command == "run" else compare(args)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
