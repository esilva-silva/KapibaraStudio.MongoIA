#!/usr/bin/env python3
"""SFT QLoRA do KapiTeacher-Mongo-7B (Fase 02 §3.3 B / Fase 05 §6.2) sobre o corpus de scripts/teacher/build_teacher_corpus.py.

  C:/SlopStudio.MongoAI-artifacts/.venv/Scripts/python.exe -I scripts/teacher/train_teacher.py \
      --config Repos/KapiTeacher-Mongo-7B/configs/sft_v2c.yaml --resume auto

Receita: base NF4 double-quant (embed/lm_head bf16), LoRA r=64/alpha=128 em todos os lineares, gradient checkpointing, CE em blocos so
nas posicoes do alvo, PagedAdamW8bit, micro-batch = linha do corpus (<= 4096 tokens padded, bucketing por comprimento, padding a direita
e atencao causal sem mascara -> kernel flash do AOTriton), 12 linhas por passo (~49k tokens). Checkpoint atomico a cada 30 min / 5% dos
passos, retomada `--resume auto` (so com fingerprint identico), metrics.jsonl, run_manifest.json, eval periodico no val e parada por
orcamento de horas. Codigos de saida: 0 ok/orcamento, 75 pausa cooperativa (tmp/gpu.pause), 2 erro de uso/fingerprint, 130 Ctrl-C.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (BAKEOFF_SCRIPTS, GPU_LOCK, gpu_lock, library_versions, model_fingerprint, sha256_file,  # noqa: E402
                     sha256_text, write_json)

PAUSE_FILE = ROOT / "tmp" / "gpu.pause"
EOS_ID = 151643
EXIT_PAUSED = 75


# -- configuracao -------------------------------------------------------------------------------------------------------------------
def load_config(path: Path, overrides: list[str]) -> dict:
    import yaml

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for item in overrides:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def git_state() -> dict:
    def run(*a):
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--", "scripts", "configs", "Repos"))}


# -- corpus -------------------------------------------------------------------------------------------------------------------------
class Corpus:
    """Shards .npz do corpus do professor; um plano e uma lista de (shard, linha) em ordem determinista."""

    def __init__(self, directory: Path, max_train_tokens: int | None, accum: int) -> None:
        import numpy as np

        self.np = np
        self.dir = directory
        self.manifest_path = directory / "manifest.json"
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.manifest_sha = sha256_file(self.manifest_path)
        self.micro_tokens = int(self.manifest["micro_tokens"])
        self._cache: dict[int, dict] = {}
        self.plan: list[tuple[int, int, int]] = []  # (shard, row, real_tokens)
        for shard_index, shard in enumerate(self.manifest["shards"]):
            path = directory / shard["file"]
            with np.load(path) as data:
                offsets, row_ptr = data["ex_offsets"], data["row_ptr"]
            for row in range(len(row_ptr) - 1):
                tokens = int(offsets[row_ptr[row + 1]] - offsets[row_ptr[row]])
                self.plan.append((shard_index, row, tokens))
        total = 0
        cut = len(self.plan)
        if max_train_tokens:
            for index, (_, _, tokens) in enumerate(self.plan):
                total += tokens
                if total >= max_train_tokens:
                    cut = index + 1
                    break
        cut -= cut % accum
        self.plan = self.plan[:cut]
        self.rows = len(self.plan)
        self.steps = self.rows // accum
        self.real_tokens = sum(t for _, _, t in self.plan)

    def verify_hashes(self) -> None:
        for shard in self.manifest["shards"]:
            for key, name in (("sha256", "file"), ("meta_sha256", "meta_file")):
                if sha256_file(self.dir / shard[name]) != shard[key]:
                    raise SystemExit(f"corpus corrompido: {shard[name]} sha256 difere do manifest")

    def _shard(self, index: int) -> dict:
        if index not in self._cache:
            if len(self._cache) > 2:
                self._cache.pop(next(iter(self._cache)))
            with self.np.load(self.dir / self.manifest["shards"][index]["file"]) as data:
                self._cache[index] = {k: data[k] for k in ("ids", "ex_offsets", "loss_start", "row_ptr")}
        return self._cache[index]

    def row(self, plan_index: int) -> tuple:
        """-> (input_ids [n,L] int64, loss_mask [n,L] bool, real_tokens)"""
        shard_index, row, _ = self.plan[plan_index]
        return rows_to_arrays(self._shard(shard_index), row, self.np)


def rows_to_arrays(shard: dict, row: int, np) -> tuple:
    first, last = int(shard["row_ptr"][row]), int(shard["row_ptr"][row + 1])
    spans = [(int(shard["ex_offsets"][i]), int(shard["ex_offsets"][i + 1]), int(shard["loss_start"][i])) for i in range(first, last)]
    length = max(e - s for s, e, _ in spans)
    ids = np.full((len(spans), length), EOS_ID, dtype=np.int64)
    mask = np.zeros((len(spans), length), dtype=bool)
    for n, (s, e, ls) in enumerate(spans):
        ids[n, : e - s] = shard["ids"][s:e]
        mask[n, ls: e - s] = True
    return ids, mask, int(sum(e - s for s, e, _ in spans))


# -- LR ----------------------------------------------------------------------------------------------------------------------------
def lr_at(step: int, total: int, opt: dict) -> float:
    warm = max(1, math.ceil(total * opt["warmup_ratio"]))
    if step < warm:
        return opt["lr"] * (step + 1) / warm
    progress = min(1.0, (step - warm) / max(1, total - warm))
    floor = opt["lr"] * opt["min_lr_ratio"]
    return floor + 0.5 * (opt["lr"] - floor) * (1 + math.cos(math.pi * progress))


# -- checkpoints --------------------------------------------------------------------------------------------------------------------
def list_checkpoints(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted((p for p in directory.iterdir() if p.is_dir() and p.name.startswith("step-") and (p / "COMPLETE").exists()),
                  key=lambda p: int(p.name.split("-")[1]))


def fingerprint(cfg: dict, corpus: Corpus, model_sha: str, accum: int) -> dict:
    critical = {"lora": cfg["lora"], "optim": cfg["optim"], "loss": cfg["loss"], "micro_tokens": cfg["data"]["micro_tokens"],
                "accum": accum, "seed": cfg["run"]["seed"], "model_quant": cfg["model"]["quant"], "attn": cfg["model"]["attn"]}
    parts = {"model_sha256": model_sha, "corpus_manifest_sha256": corpus.manifest_sha, "train_config_sha256": sha256_text(json.dumps(critical, sort_keys=True))}
    parts["combined"] = sha256_text(json.dumps(parts, sort_keys=True))
    return parts


# -- perda --------------------------------------------------------------------------------------------------------------------------
def target_loss(model, input_ids, loss_mask, chunk: int):
    """Soma da CE nas posicoes do alvo (float32) e numero de posicoes. Logits so onde ha perda, em blocos com recomputacao."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    def piece(hidden, targets, head):
        return F.cross_entropy(F.linear(hidden.to(head.dtype), head).float(), targets, reduction="sum")

    inner = model.get_base_model()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = inner.model(input_ids=input_ids, use_cache=False).last_hidden_state
    mask = loss_mask[:, 1:]
    selected, targets = hidden[:, :-1][mask], input_ids[:, 1:][mask]
    head = inner.lm_head.weight
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, selected.shape[0], chunk):
        part = slice(start, start + chunk)
        total = total + checkpoint(piece, selected[part], targets[part], head, use_reentrant=False)
    return total, int(mask.sum())


# -- avaliacao rapida no val ------------------------------------------------------------------------------------------------------
def eval_val_loss(model, corpus_dir: Path, manifest: dict, device, chunk: int, heartbeat) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F

    meta = [json.loads(line) for line in (corpus_dir / "val" / "val_loss.meta.jsonl").read_text(encoding="utf-8").splitlines() if line]
    with np.load(corpus_dir / "val" / "val_loss.npz") as data:
        shard = {k: data[k] for k in ("ids", "ex_offsets", "loss_start", "row_ptr")}
    sums, counts = {}, {}
    inner = model.get_base_model()
    head = inner.lm_head.weight
    first = 0
    model.eval()
    with torch.no_grad():
        for row in range(len(shard["row_ptr"]) - 1):
            ids_np, mask_np, _ = rows_to_arrays(shard, row, np)
            n = ids_np.shape[0]
            ids = torch.from_numpy(ids_np).to(device)
            mask = torch.from_numpy(mask_np).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = inner.model(input_ids=ids, use_cache=False).last_hidden_state
            m = mask[:, 1:]
            owner = torch.arange(n, device=device)[:, None].expand_as(m)[m]
            h, t = hidden[:, :-1][m], ids[:, 1:][m]
            per = []
            for s in range(0, h.shape[0], chunk):
                per.append(F.cross_entropy(F.linear(h[s: s + chunk].to(head.dtype), head).float(), t[s: s + chunk], reduction="none"))
            per = torch.cat(per) if per else h.new_zeros(0)
            ex_sum = torch.zeros(n, device=device).index_add_(0, owner, per).cpu().tolist()
            ex_cnt = m.sum(1).cpu().tolist()
            for k in range(n):
                comp = meta[first + k]["component"]
                group = ("generic" if comp.startswith("replay") else "chat" if comp not in ("fim_ctx", "fim_raw", "completion") else "fim")
                sums[group] = sums.get(group, 0.0) + ex_sum[k]
                counts[group] = counts.get(group, 0) + ex_cnt[k]
            first += n
            heartbeat()
    model.train()
    out = {f"val_loss_{g}": round(sums[g] / counts[g], 5) for g in sums}
    # `val_loss` (selecao do best) so Mongo (fim+chat); `val_loss_generic` = monitor generic_js (G2.5), nunca entra na selecao
    mongo = [g for g in sums if g != "generic"]
    out["val_loss"] = round(sum(sums[g] for g in mongo) / max(1, sum(counts[g] for g in mongo)), 5)
    out["val_loss_tokens"] = sum(counts[g] for g in mongo)
    return out


def eval_val_gen(model, tokenizer, corpus_dir: Path, limit: int, max_new: int, device, heartbeat,
                 filename: str = "val_gen.jsonl", prefix: str = "gen") -> dict:
    """FIM greedy no val (prompt v2c editor-context-v1): EM, normalized match, syntax (node) e mongo_valid estatico.
    `filename=val_gen_generic.jsonl, prefix=gjs`: monitor Generic-JS (FIM cru, sem Mongo; syntax = node vm.Script do documento)."""
    import torch
    from _common import Generator
    from compact import syntax
    from compact.contracts import postprocess
    from compact.mongo_validate import validate

    rows = [json.loads(line) for line in (corpus_dir / "val" / filename).read_text(encoding="utf-8").splitlines() if line][:limit]
    if not rows:
        return {}
    gen = Generator(model, tokenizer, device)
    model.eval()
    model.config.use_cache = True
    stops = frozenset({EOS_ID})
    preds = []
    started = time.time()
    for n, row in enumerate(rows):
        ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
        result = gen.generate(ids, max_new, stops, row["suffix"][:1024])
        tokens = [t for t in result["tokens"] if t not in stops]
        raw = gen.decode(tokens)
        preds.append(postprocess(raw, {"type": "fim", "suffix": row["suffix"]}))
        if n % 10 == 0:
            heartbeat()
    model.config.use_cache = False
    model.train()

    def norm(text: str) -> str:
        import re
        return re.sub(r"\s+", " ", text.replace("\r\n", "\n")).strip()

    jobs = [(str(i), (r["prefix"] + p + r["suffix"]).replace("\r\n", "\n"), syntax.kind_for(r["dialect"])) for i, (r, p) in enumerate(zip(rows, preds))]
    syn = syntax.check_many(jobs)
    em = nm = sv = mv = nonempty = 0
    for i, (r, p) in enumerate(zip(rows, preds)):
        em += p.strip() == r["target"].strip()
        nm += norm(p) == norm(r["target"])
        ok = syn[str(i)][0]
        sv += ok
        check = validate(jobs[i][1], r["dialect"]) if ok and prefix == "gen" else None
        mv += bool(check.valid) if check is not None and check.validatable else 0
        nonempty += bool(p.strip())
    n = len(rows)
    out = {f"{prefix}_n": n, f"{prefix}_exact_match": round(em / n, 4), f"{prefix}_normalized_match": round(nm / n, 4),
           f"{prefix}_syntax_valid": round(sv / n, 4), f"{prefix}_nonempty": round(nonempty / n, 4),
           f"{prefix}_seconds": round(time.time() - started, 1)}
    if prefix == "gen":
        out["gen_mongo_valid"] = round(mv / n, 4)
    return out


# -- principal ----------------------------------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "Repos" / "KapiTeacher-Mongo-7B" / "configs" / "sft_v2c.yaml"))
    ap.add_argument("--resume", default="auto", help="auto | none | caminho do checkpoint (step-XXXXXX)")
    ap.add_argument("--set", dest="overrides", action="append", default=[], help="chave.pontilhada=valor (ex.: budget.hours=2)")
    ap.add_argument("--run-name", help="substitui run.name (smoke: pastas separadas)")
    ap.add_argument("--max-steps", type=int, help="para apos N passos totais (smoke); nao altera o cosseno")
    ap.add_argument("--total-steps", type=int, help="substitui o total de passos do cosseno (corte do job; registrado no manifest)")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="so o plano: passos, tokens e horas estimadas (sem GPU)")
    ap.add_argument("--tok-s", type=float, default=330.0, help="tok/s assumido no --dry-run")
    ap.add_argument("--verify-corpus", action="store_true", help="recalcula sha256 de todos os shards no inicio")
    args = ap.parse_args()

    cfg = load_config(Path(args.config), args.overrides)
    if args.run_name:
        cfg["run"]["name"] = args.run_name
    run = cfg["run"]["name"]
    accum = max(1, round(cfg["optim"]["tokens_per_step"] / cfg["data"]["micro_tokens"]))
    corpus = Corpus(ROOT / cfg["data"]["corpus_dir"], cfg["data"].get("max_train_tokens"), accum)
    if corpus.micro_tokens != cfg["data"]["micro_tokens"]:
        raise SystemExit(f"micro_tokens do corpus ({corpus.micro_tokens}) != config ({cfg['data']['micro_tokens']})")
    total_steps = args.total_steps or corpus.steps
    padded_per_step = accum * corpus.micro_tokens
    if args.dry_run:
        hours = corpus.real_tokens / args.tok_s / 3600
        print(json.dumps({"run": run, "rows": corpus.rows, "accum": accum, "steps": corpus.steps, "total_steps_schedule": total_steps,
                          "real_tokens": corpus.real_tokens, "tokens_per_step_padded": padded_per_step,
                          f"hours_at_{args.tok_s:.0f}_tok_s": round(hours, 2)}, indent=1))
        return 0
    if args.verify_corpus:
        corpus.verify_hashes()

    # ambiente ANTES de importar torch (caches, TMP, AOTriton experimental)
    from compact.env import configure_caches, environment_report, limit_vram, require_gpu
    configure_caches(ROOT)
    import torch

    ckpt_dir = ROOT / cfg["checkpoint"]["dir"] / run
    rep_dir = ROOT / cfg["paths"]["reports_dir"] / run
    ad_dir = ROOT / cfg["paths"]["adapters_dir"] / run
    for d in (ckpt_dir, rep_dir, ad_dir):
        d.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT / cfg["model"]["path"]
    model_fp = model_fingerprint(model_dir)
    fp = fingerprint(cfg, corpus, model_fp["model_sha256"], accum)

    # retomada
    resume_path = None
    if args.resume == "auto":
        found = list_checkpoints(ckpt_dir)
        resume_path = found[-1] if found else None
    elif args.resume not in ("none", ""):
        resume_path = Path(args.resume)
    state = {"step": 0, "elapsed_train_s": 0.0, "real_tokens": 0, "best_val_loss": None, "schedule_changes": [], "restarts": 0}
    if resume_path:
        saved = json.loads((resume_path / "state.json").read_text(encoding="utf-8"))
        if saved["fingerprint"]["combined"] != fp["combined"]:
            diff = {k: (saved["fingerprint"].get(k), fp[k]) for k in fp if saved["fingerprint"].get(k) != fp[k]}
            print(f"RETOMADA RECUSADA: fingerprint de {resume_path} difere ({diff}). Mude de --run-name ou use --resume none.", file=sys.stderr)
            return 2
        state.update(saved["state"])
        state["restarts"] = state.get("restarts", 0) + 1
        if args.total_steps is None:
            total_steps = saved["state"].get("total_steps", total_steps)
    elif args.resume == "auto" and (rep_dir / "metrics.jsonl").exists():
        (rep_dir / "metrics.jsonl").rename(rep_dir / f"metrics.{time.strftime('%Y%m%dT%H%M%S')}.jsonl")
    if args.total_steps and state.get("total_steps") not in (None, args.total_steps):
        state["schedule_changes"].append({"at_step": state["step"], "from": state.get("total_steps"), "to": args.total_steps})
    state["total_steps"] = total_steps
    if args.max_steps:
        total_steps_stop = min(args.max_steps, total_steps)
    else:
        total_steps_stop = total_steps

    with gpu_lock(f"train_teacher:{run}"):
        gpu = require_gpu("7800 XT")
        gpu["vram_limit"] = limit_vram(gpu, cfg["memory"]["max_vram_fraction"], reserve_gb=cfg["memory"]["vram_reserve_gb"])
        index = gpu["selected_index"]
        device = torch.device(f"cuda:{index}")
        print(f"[{run}] VRAM limit {gpu['vram_limit']} | passos {state['step']}/{total_steps} | "
              f"{corpus.rows} linhas, {corpus.real_tokens:,} tokens reais", flush=True)

        import bitsandbytes as bnb
        import numpy as np
        from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        load_started = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, device_map={"": index}, dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation=cfg["model"]["attn"],
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                                   bnb_4bit_compute_dtype=torch.bfloat16))
        off = [n for n, p in model.named_parameters() if p.device.type != "cuda"]
        if off:
            raise SystemExit(f"{len(off)} parametros fora da GPU: offload nao permitido")
        for p in model.parameters():
            p.requires_grad_(False)
        weights_gb = torch.cuda.memory_allocated(index) / 2**30
        model.config.use_cache = False
        if cfg["memory"].get("gradient_checkpointing", True):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        lora = cfg["lora"]
        model = get_peft_model(model, LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"], task_type="CAUSAL_LM",
                                                 target_modules=lora["targets"]))
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        o = cfg["optim"]
        optimizer = bnb.optim.PagedAdamW8bit([p for p in model.parameters() if p.requires_grad], lr=o["lr"], betas=tuple(o["betas"]),
                                             weight_decay=o["weight_decay"])
        if resume_path:
            set_peft_model_state_dict(model, load_file(str(resume_path / "adapter" / "adapter_model.safetensors"), device="cpu"))
            optimizer.load_state_dict(torch.load(resume_path / "optimizer.pt", map_location="cpu", weights_only=False))
            rng = torch.load(resume_path / "rng.pt", map_location="cpu", weights_only=False)
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            torch.cuda.set_rng_state(rng["cuda"], index)
            print(f"[{run}] retomado de {resume_path.name}: passo {state['step']}", flush=True)
        else:
            random.seed(cfg["run"]["seed"])
            np.random.seed(cfg["run"]["seed"] % 2**32)
            torch.manual_seed(cfg["run"]["seed"])
        model.train()
        load_seconds = time.time() - load_started
        torch.cuda.reset_peak_memory_stats(index)

        # manifest do run
        manifest_path = rep_dir / "run_manifest.json"
        first_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
        lock_entry = json.loads((ROOT / "BaseRepos" / "base-repos.lock.json").read_text(encoding="utf-8"))["repos"].get(model_dir.name, {})
        libs = library_versions()
        with contextlib.suppress(Exception):
            import peft
            libs["peft"] = peft.__version__
        run_manifest = {
            "schema": "teacher-run/1", "run": run, "stage": cfg["run"]["stage"], "started_utc": (first_manifest or {}).get("started_utc") or time.strftime("%FT%TZ", time.gmtime()),
            "last_start_utc": time.strftime("%FT%TZ", time.gmtime()), "restarts": state["restarts"], "git": git_state(),
            "config": {"path": rel(Path(args.config)), "sha256": sha256_file(Path(args.config)), "overrides": args.overrides, "effective": cfg},
            "dataset": {"corpus_manifest": rel(corpus.manifest_path), "corpus_manifest_sha256": corpus.manifest_sha,
                        "source_dataset": corpus.manifest.get("source", {}).get("manifest_sha256"),
                        "examples": corpus.manifest["counts"]["examples"], "tokens_in_corpus": corpus.manifest["counts"]["tokens"],
                        "tokens_this_run": corpus.real_tokens, "rows": corpus.rows, "replay": corpus.manifest.get("replay")},
            "contract": {"editor_context_v1_golden_sha256": sha256_file(ROOT / cfg["paths"]["contract_golden"]),
                         "render_contract_py_sha256": sha256_file(ROOT / "scripts" / "data" / "render_contract.py"),
                         "chat_template": corpus.manifest.get("chat_template"), "corpus_builder_sha256": corpus.manifest.get("builder_sha256")},
            "gates_yaml_sha256": sha256_file(ROOT / cfg["paths"]["gates"]),
            "base_model": {"path": rel(model_dir), "repo_id": lock_entry.get("repo_id"), "revision": lock_entry.get("revision"),
                           "model_sha256": model_fp["model_sha256"], "weights_gb_on_gpu": round(weights_gb, 2)},
            "fingerprint": fp, "libraries": libs, "environment": environment_report({k: gpu.get(k) for k in
                                                                                     ("device_name", "gcn_arch", "total_vram_gb", "torch", "hip", "vram_limit")}),
            "aotriton_experimental": os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"),
            "plan": {"accum_rows": accum, "micro_tokens": corpus.micro_tokens, "steps": corpus.steps, "total_steps_schedule": total_steps,
                     "trainable_params": trainable, "load_seconds": round(load_seconds, 1), "budget_hours": cfg["budget"]["hours"]},
            "schedule_changes": state["schedule_changes"], "status": "running",
        }
        write_json(manifest_path, run_manifest)

        heartbeat_path = rep_dir / "heartbeat.json"

        def heartbeat(extra: dict | None = None) -> None:
            with contextlib.suppress(OSError):
                heartbeat_path.write_text(json.dumps({"t": time.time(), "step": state["step"], **(extra or {})}), encoding="utf-8")

        def vram() -> dict:
            return {"vram_alloc_gb": round(torch.cuda.memory_allocated(index) / 2**30, 2),
                    "vram_peak_gb": round(torch.cuda.max_memory_allocated(index) / 2**30, 2),
                    "vram_reserved_peak_gb": round(torch.cuda.max_memory_reserved(index) / 2**30, 2)}

        def save_checkpoint(tag: str) -> Path:
            started = time.time()
            final = ckpt_dir / f"step-{state['step']:06d}"
            if final.exists():
                shutil.rmtree(final)
            tmp = ckpt_dir / f"{final.name}.tmp"
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
            model.save_pretrained(tmp / "adapter")
            torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
            torch.save({"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state(index)}, tmp / "rng.pt")
            (tmp / "state.json").write_text(json.dumps({"fingerprint": fp, "state": state, "tag": tag, "saved": time.strftime("%FT%T")}),
                                            encoding="utf-8")
            (tmp / "COMPLETE").write_text("ok", encoding="utf-8")
            tmp.rename(final)
            keep = cfg["checkpoint"]["keep"]
            for old in list_checkpoints(ckpt_dir)[:-keep]:
                shutil.rmtree(old, ignore_errors=True)
            print(f"[{run}] checkpoint {final.name} ({tag}) em {time.time() - started:.1f}s", flush=True)
            return final

        def run_eval(final: bool = False) -> dict:
            if args.no_eval:
                return {}
            ev = cfg["eval"]
            started = time.time()
            out = {"step": state["step"], "tokens": state["real_tokens"], "t_utc": time.strftime("%FT%TZ", time.gmtime())}
            if ev.get("val_loss", True):
                out.update(eval_val_loss(model, corpus.dir, corpus.manifest, device, cfg["loss"]["ce_chunk"], heartbeat))
            limit = ev["final_gen_examples"] if final else ev["gen_examples"]
            if limit:
                out.update(eval_val_gen(model, tokenizer, corpus.dir, limit, ev["max_new_tokens"], device, heartbeat))
            glimit = ev.get("final_generic_gen_examples", ev.get("generic_gen_examples", 0)) if final else ev.get("generic_gen_examples", 0)
            if glimit and (corpus.dir / "val" / "val_gen_generic.jsonl").exists():
                out.update(eval_val_gen(model, tokenizer, corpus.dir, glimit, ev["max_new_tokens"], device, heartbeat,
                                        filename="val_gen_generic.jsonl", prefix="gjs"))
            out["eval_seconds"] = round(time.time() - started, 1)
            with (rep_dir / "eval_history.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"[{run}] EVAL {json.dumps(out, ensure_ascii=False)}", flush=True)
            if out.get("val_loss") is not None and (state["best_val_loss"] is None or out["val_loss"] < state["best_val_loss"]):
                state["best_val_loss"] = out["val_loss"]
                best = ad_dir / "best"
                if best.exists():
                    shutil.rmtree(best)
                model.save_pretrained(best)
                (best / "best.json").write_text(json.dumps(out), encoding="utf-8")
            return out

        # agenda de checkpoints/avaliacoes
        ck, ev = cfg["checkpoint"], cfg["eval"]
        every_ck = ck.get("every_steps") or max(1, round(total_steps * ck["every_pct_steps"] / 100))
        every_ev = ev.get("every_steps") or max(1, round(total_steps * ev["every_pct_steps"] / 100))
        last_ck_time = time.time()
        metrics_file = rep_dir / "metrics.jsonl"
        status, exit_code = "completed", 0
        session_start = time.time()
        base_elapsed = state["elapsed_train_s"]
        chunk = cfg["loss"]["ce_chunk"]
        params = [p for p in model.parameters() if p.requires_grad]
        heartbeat({"phase": "start"})

        try:
            while state["step"] < total_steps_stop:
                step = state["step"]
                lr = lr_at(step, total_steps, o)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                rows = [corpus.row(step * accum + k) for k in range(accum)]
                n_target = sum(int(m[:, 1:].sum()) for _, m, _ in rows)
                real_tokens = sum(t for _, _, t in rows)
                padded = sum(i.shape[0] * i.shape[1] for i, _, _ in rows)
                torch.cuda.synchronize(index)
                begin = time.perf_counter()
                loss_sum = 0.0
                for ids_np, mask_np, _ in rows:
                    ids = torch.from_numpy(ids_np).to(device)
                    mask = torch.from_numpy(mask_np).to(device)
                    total, n = target_loss(model, ids, mask, chunk)
                    (total / n_target).backward()
                    loss_sum += float(total.detach())
                    heartbeat()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, o["grad_clip"]))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(index)
                dt = time.perf_counter() - begin
                loss = loss_sum / n_target
                if not math.isfinite(loss) or not math.isfinite(grad_norm):
                    raise RuntimeError(f"loss/grad nao finita no passo {step}: loss={loss} grad_norm={grad_norm}")
                state["step"] = step + 1
                state["real_tokens"] += real_tokens
                state["elapsed_train_s"] = base_elapsed + (time.time() - session_start)
                record = {"step": state["step"], "loss": round(loss, 5), "lr": lr, "grad_norm": round(grad_norm, 4), "tokens_real": real_tokens,
                          "tokens_padded": padded, "target_tokens": n_target, "step_s": round(dt, 2), "tok_s_real": round(real_tokens / dt, 1),
                          "tok_s_padded": round(padded / dt, 1), "tokens_seen": state["real_tokens"],
                          "elapsed_h": round(state["elapsed_train_s"] / 3600, 3), **vram(), "t": time.strftime("%FT%T")}
                with metrics_file.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(f"[{run}] step {state['step']}/{total_steps} loss {loss:.4f} lr {lr:.2e} gn {grad_norm:.2f} {dt:.1f}s "
                      f"{real_tokens / dt:.0f} tok/s peak {record['vram_peak_gb']} GB", flush=True)
                heartbeat({"phase": "step"})
                if state["elapsed_train_s"] >= cfg["budget"]["hours"] * 3600:
                    status = "stopped_budget"
                    break
                if PAUSE_FILE.exists():
                    status, exit_code = "paused", EXIT_PAUSED
                    break
                due_time = time.time() - last_ck_time >= ck["every_minutes"] * 60
                if state["step"] % every_ck == 0 or due_time:
                    save_checkpoint("time" if due_time else "steps")
                    last_ck_time = time.time()
                if state["step"] % every_ev == 0 and state["step"] < total_steps_stop:
                    run_eval()
                    heartbeat({"phase": "post-eval"})
            else:
                status = "completed" if state["step"] >= total_steps else "stopped_max_steps"
        except KeyboardInterrupt:
            status, exit_code = "interrupted", 130
        except torch.OutOfMemoryError as error:
            status, exit_code = "oom", 3
            print(f"[{run}] OOM: {str(error).splitlines()[0][:200]}", file=sys.stderr)

        # fim: checkpoint final, adapter final, avaliacao final
        summary = {}
        if status in ("completed", "stopped_budget", "stopped_max_steps", "paused", "interrupted"):
            save_checkpoint(status)
        if status in ("completed", "stopped_budget", "stopped_max_steps"):
            final_dir = ad_dir / "final"
            if final_dir.exists():
                shutil.rmtree(final_dir)
            model.save_pretrained(final_dir)
            summary = run_eval(final=True)
            (final_dir / "train_state.json").write_text(json.dumps({"state": state, "status": status, "fingerprint": fp, "final_eval": summary}),
                                                        encoding="utf-8")
        run_manifest.update({"status": status, "finished_utc": time.strftime("%FT%TZ", time.gmtime()), "final_step": state["step"],
                             "total_steps": total_steps, "tokens_seen": state["real_tokens"], "elapsed_train_h": round(state["elapsed_train_s"] / 3600, 3),
                             "best_val_loss": state["best_val_loss"], "final_eval": summary, "schedule_changes": state["schedule_changes"],
                             "peak_vram_gb": vram(), "restarts": state["restarts"]})
        write_json(manifest_path, run_manifest)
        print(f"[{run}] fim: {status} passo {state['step']}/{total_steps} tokens {state['real_tokens']:,}", flush=True)
        del optimizer, model
        torch.cuda.empty_cache()
        return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
