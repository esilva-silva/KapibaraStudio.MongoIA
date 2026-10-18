#!/usr/bin/env python3
"""SFT dos alunos T0 Nano (Qwen2.5-Coder-0.5B) e T1 Small (Qwen2.5-Coder-1.5B) sobre o corpus de scripts/student/build_student_corpus.py.

  C:/SlopStudio.MongoAI-artifacts/.venv/Scripts/python.exe -I scripts/student/train_student.py \
      --config Repos/KapiCoder-Mongo-Nano/configs/sft_distill_v1.yaml --resume auto

Estende scripts/teacher/train_teacher.py (reusa Corpus, rows_to_arrays, lr_at, checkpoints, eval no val): mesmo formato de shards, mesma perda
so no alvo (CE em blocos), mesmo `--resume auto` com fingerprint, `metrics.jsonl`, `run_manifest.json`, parada por orcamento de horas, pausa
cooperativa (tmp/gpu.pause -> codigo 75) e eval periodico no val v2c (val_loss FIM, EM, syntax, mongo_valid estatico, monitor Generic-JS).
O que muda: (1) `model.method`: `lora` (base bf16 + LoRA, adapter e merge bf16 no fim) ou `full` (todos os pesos; mestre fp32 + autocast bf16,
AdamW fp32 ou 8-bit); (2) varias epocas (permutacao determinista das linhas por epoca); (3) saida = modelo COMPLETO bf16 em
Repos/<modelo>/merged/<run> (o que o eval_teacher.py/eval_student.py carregam com --load bf16) alem do adapter (lora).
Os dois bracos (distill x gold) usam o MESMO yaml salvo `run.name`/`run.arm`/`data.corpus_dir`. Codigos de saida: 0 ok/orcamento, 75 pausa,
2 erro de uso/fingerprint, 3 OOM, 130 Ctrl-C.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "teacher"))
from _common import gpu_lock, library_versions, model_fingerprint, sha256_file, sha256_text, write_json  # noqa: E402
from train_teacher import (EOS_ID, EXIT_PAUSED, PAUSE_FILE, Corpus, base_model, eval_val_gen, eval_val_loss, git_state,  # noqa: E402
                           list_checkpoints, load_config, lr_at, rel)

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json")


# -- plano de epocas ----------------------------------------------------------------------------------------------------------------
class EpochCorpus(Corpus):
    """Corpus do professor com varias epocas: a epoca 0 segue a ordem do manifesto; as demais, uma permutacao determinista das linhas."""

    def __init__(self, directory: Path, max_train_tokens: int | None, accum: int, epochs: int, seed: int) -> None:
        super().__init__(directory, max_train_tokens, accum)
        self.epochs = epochs
        self.seed = seed
        self.accum = accum
        self.steps_per_epoch = self.steps
        self.total_steps = self.steps * epochs
        self._orders: dict[int, list[int]] = {}
        self._all: dict[int, dict] = {}

    def _shard(self, index: int) -> dict:   # todos os shards ficam em RAM (T1: ~250 MB): a permutacao da epoca 2 vira acesso aleatorio
        if index not in self._all:
            with self.np.load(self.dir / self.manifest["shards"][index]["file"]) as data:
                self._all[index] = {k: data[k] for k in ("ids", "ex_offsets", "loss_start", "row_ptr")}
        return self._all[index]

    def order(self, epoch: int) -> list[int]:
        if epoch not in self._orders:
            idx = list(range(self.steps_per_epoch * self.accum))
            if epoch > 0:
                random.Random(f"{self.seed}-plan-{epoch}").shuffle(idx)
            self._orders[epoch] = idx
        return self._orders[epoch]

    def plan_index(self, step: int, k: int) -> int:
        epoch, within = divmod(step, self.steps_per_epoch)
        return self.order(epoch)[within * self.accum + k]

    def epoch_tokens(self, step: int) -> int:
        epoch, within = divmod(step, self.steps_per_epoch)
        order = self.order(epoch)
        return sum(self.plan[order[within * self.accum + k]][2] for k in range(self.accum))


def fingerprint(cfg: dict, corpus: EpochCorpus, model_sha: str, accum: int) -> dict:
    critical = {"method": cfg["model"]["method"], "master": cfg["model"].get("master_dtype"), "lora": cfg.get("lora") if cfg["model"]["method"] == "lora" else None,
                "optim": cfg["optim"], "loss": cfg["loss"], "micro_tokens": cfg["data"]["micro_tokens"], "accum": accum, "epochs": cfg["data"]["epochs"],
                "seed": cfg["run"]["seed"], "attn": cfg["model"]["attn"], "ckpt": cfg["memory"].get("gradient_checkpointing", True)}
    parts = {"model_sha256": model_sha, "corpus_manifest_sha256": corpus.manifest_sha, "train_config_sha256": sha256_text(json.dumps(critical, sort_keys=True))}
    parts["combined"] = sha256_text(json.dumps(parts, sort_keys=True))
    return parts


# -- perda --------------------------------------------------------------------------------------------------------------------------
def target_loss(model, input_ids, loss_mask, chunk: int):
    """Soma da CE nas posicoes do alvo (float32) e numero de posicoes. Logits so onde ha perda, em blocos com recomputacao; a projecao do
    lm_head roda em bf16 (autocast) mesmo com o mestre fp32."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    def piece(hidden, targets, head):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = F.linear(hidden, head)
        return F.cross_entropy(logits.float(), targets, reduction="sum")

    inner = base_model(model)
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


# -- modelo / otimizador ------------------------------------------------------------------------------------------------------------
def build_optimizer(model, o: dict, method: str):
    import torch

    o = {**o, "lr": float(o["lr"]), "weight_decay": float(o["weight_decay"])}   # `2e-4` sem ponto decimal vira str no PyYAML
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    groups = [{"params": decay, "weight_decay": o["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}]
    name = o["name"]
    if name in ("adamw8bit", "paged_adamw8bit"):
        import bitsandbytes as bnb
        cls = bnb.optim.PagedAdamW8bit if name.startswith("paged") else bnb.optim.AdamW8bit
        return cls(groups, lr=o["lr"], betas=tuple(o["betas"]))
    try:
        return torch.optim.AdamW(groups, lr=o["lr"], betas=tuple(o["betas"]), fused=True)
    except (RuntimeError, ValueError):
        return torch.optim.AdamW(groups, lr=o["lr"], betas=tuple(o["betas"]))


def export_merged(model, method: str, cfg: dict, base_dir: Path, out_dir: Path) -> dict:
    """Modelo completo bf16 + arquivos do tokenizer (diretorio carregavel por AutoModelForCausalLM/eval_teacher.py --load bf16)."""
    import torch

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    if method == "lora":
        full = model.merge_and_unload()
    else:
        full = model
    full = full.to(torch.bfloat16)
    full.save_pretrained(out_dir, safe_serialization=True)
    for name in TOKENIZER_FILES:
        if (base_dir / name).exists():
            shutil.copy2(base_dir / name, out_dir / name)
    files = {p.name: sha256_file(p) for p in sorted(out_dir.iterdir()) if p.is_file()}
    info = {"path": rel(out_dir), "dtype": "bfloat16", "files_sha256": files,
            "combined_sha256": sha256_text("\n".join(f"{k}:{v}" for k, v in sorted(files.items())))}
    write_json(out_dir / "export_info.json", info)
    return info


# -- principal ----------------------------------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default="auto", help="auto | none | caminho do checkpoint (step-XXXXXX)")
    ap.add_argument("--set", dest="overrides", action="append", default=[], help="chave.pontilhada=valor (ex.: budget.hours=2)")
    ap.add_argument("--run-name", help="substitui run.name (smoke: pastas separadas)")
    ap.add_argument("--max-steps", type=int, help="para apos N passos totais (smoke); nao altera o cosseno")
    ap.add_argument("--total-steps", type=int, help="substitui o total de passos do cosseno (corte do job; registrado no manifest)")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--no-export", action="store_true", help="nao grava o modelo completo bf16 em merged/ ao final (smoke de memoria)")
    ap.add_argument("--dry-run", action="store_true", help="so o plano: passos, tokens e horas estimadas (sem GPU)")
    ap.add_argument("--tok-s", type=float, default=2400.0, help="tok/s assumido no --dry-run")
    ap.add_argument("--verify-corpus", action="store_true", help="recalcula sha256 de todos os shards no inicio")
    args = ap.parse_args()

    cfg = load_config(Path(args.config), args.overrides)
    if args.run_name:
        cfg["run"]["name"] = args.run_name
    run = cfg["run"]["name"]
    method = cfg["model"]["method"]
    if method not in ("lora", "full"):
        raise SystemExit(f"model.method invalido: {method}")
    accum = max(1, round(cfg["optim"]["tokens_per_step"] / cfg["data"]["micro_tokens"]))
    corpus = EpochCorpus(ROOT / cfg["data"]["corpus_dir"], cfg["data"].get("max_train_tokens"), accum, int(cfg["data"]["epochs"]), int(cfg["run"]["seed"]))
    if corpus.micro_tokens != cfg["data"]["micro_tokens"]:
        raise SystemExit(f"micro_tokens do corpus ({corpus.micro_tokens}) != config ({cfg['data']['micro_tokens']})")
    total_steps = args.total_steps or corpus.total_steps
    if args.dry_run:
        tokens_total = corpus.real_tokens * corpus.epochs
        hours = tokens_total / args.tok_s / 3600
        print(json.dumps({"run": run, "arm": cfg["run"].get("arm"), "method": method, "rows": corpus.rows, "accum": accum, "steps_per_epoch": corpus.steps_per_epoch,
                          "epochs": corpus.epochs, "total_steps_schedule": total_steps, "real_tokens_epoch": corpus.real_tokens, "real_tokens_total": tokens_total,
                          f"hours_at_{args.tok_s:.0f}_tok_s": round(hours, 2)}, indent=1))
        return 0
    if args.verify_corpus:
        corpus.verify_hashes()

    from compact.env import configure_caches, environment_report, limit_vram, require_gpu
    configure_caches(ROOT)
    import torch

    ckpt_dir = ROOT / cfg["checkpoint"]["dir"] / run
    rep_dir = ROOT / cfg["paths"]["reports_dir"] / run
    ad_dir = ROOT / cfg["paths"]["adapters_dir"] / run
    merged_dir = ROOT / cfg["paths"]["merged_dir"] / run
    for d in (ckpt_dir, rep_dir):
        d.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT / cfg["model"]["path"]
    model_fp = model_fingerprint(model_dir)
    fp = fingerprint(cfg, corpus, model_fp["model_sha256"], accum)

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
    total_steps_stop = min(args.max_steps, total_steps) if args.max_steps else total_steps

    with gpu_lock(f"train_student:{run}"):
        gpu = require_gpu("7800 XT")
        gpu["vram_limit"] = limit_vram(gpu, cfg["memory"]["max_vram_fraction"], reserve_gb=cfg["memory"]["vram_reserve_gb"])
        index = gpu["selected_index"]
        device = torch.device(f"cuda:{index}")
        print(f"[{run}] {method} | VRAM limit {gpu['vram_limit']} | passos {state['step']}/{total_steps} | {corpus.rows} linhas/epoca, "
              f"{corpus.real_tokens:,} tokens reais/epoca x {corpus.epochs}", flush=True)

        import numpy as np
        from transformers import AutoModelForCausalLM, AutoTokenizer

        load_started = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        master = cfg["model"].get("master_dtype", "fp32") if method == "full" else "bf16"
        dtype = torch.float32 if master == "fp32" else torch.bfloat16
        load_from = (resume_path / "model") if (resume_path and method == "full") else model_dir
        model = AutoModelForCausalLM.from_pretrained(load_from, device_map={"": index}, dtype=dtype, low_cpu_mem_usage=True,
                                                     attn_implementation=cfg["model"]["attn"])
        off = [n for n, p in model.named_parameters() if p.device.type != "cuda"]
        if off:
            raise SystemExit(f"{len(off)} parametros fora da GPU: offload nao permitido")
        model.config.use_cache = False
        if method == "lora":
            from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
            from safetensors.torch import load_file

            for p in model.parameters():
                p.requires_grad_(False)
            if cfg["memory"].get("gradient_checkpointing", True):
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                model.enable_input_require_grads()
            lora = cfg["lora"]
            model = get_peft_model(model, LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"], task_type="CAUSAL_LM",
                                                     target_modules=lora["targets"]))
            if resume_path:
                set_peft_model_state_dict(model, load_file(str(resume_path / "adapter" / "adapter_model.safetensors"), device="cpu"))
        else:
            for p in model.parameters():
                p.requires_grad_(True)
            if cfg["memory"].get("gradient_checkpointing", True):
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        weights_gb = torch.cuda.memory_allocated(index) / 2**30
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        o = cfg["optim"]
        optimizer = build_optimizer(model, o, method)
        if resume_path:
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

        manifest_path = rep_dir / "run_manifest.json"
        first_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
        lock_entry = json.loads((ROOT / "BaseRepos" / "base-repos.lock.json").read_text(encoding="utf-8"))["repos"].get(model_dir.name, {})
        libs = library_versions()
        with contextlib.suppress(Exception):
            import peft
            libs["peft"] = peft.__version__
        cm = corpus.manifest
        run_manifest = {
            "schema": "student-run/1", "run": run, "arm": cfg["run"].get("arm"), "tier": cfg["run"].get("tier"), "stage": cfg["run"]["stage"],
            "started_utc": (first_manifest or {}).get("started_utc") or time.strftime("%FT%TZ", time.gmtime()),
            "last_start_utc": time.strftime("%FT%TZ", time.gmtime()), "restarts": state["restarts"], "git": git_state(),
            "config": {"path": rel(Path(args.config)), "sha256": sha256_file(Path(args.config)), "overrides": args.overrides, "effective": cfg},
            "dataset": {"corpus_manifest": rel(corpus.manifest_path), "corpus_manifest_sha256": corpus.manifest_sha, "label_source": cm.get("label_source"),
                        "examples": cm["counts"]["examples"], "tokens_in_corpus": cm["counts"]["tokens"], "tokens_this_run": corpus.real_tokens * corpus.epochs,
                        "rows": corpus.rows, "epochs": corpus.epochs, "replay": {k: cm["replay"].get(k) for k in ("examples", "tokens", "token_share_achieved")},
                        "pairing": {"ids_sha256": (cm.get("verification") or {}).get("ids_sha256"), "rows_sha256": (cm.get("verification") or {}).get("rows_sha256")},
                        "student_corpus_config_sha256": cm["config"]["sha256"], "accepted_sha256": cm["source"].get("accepted_sha256")},
            "gates_yaml_sha256": sha256_file(ROOT / cfg["paths"]["gates"]),
            "rounds_yaml_sha256": sha256_file(ROOT / "configs" / "rounds.yaml") if (ROOT / "configs" / "rounds.yaml").exists() else None,
            "base_model": {"path": rel(model_dir), "repo_id": lock_entry.get("repo_id"), "revision": lock_entry.get("revision"),
                           "model_sha256": model_fp["model_sha256"], "weights_gb_on_gpu": round(weights_gb, 2)},
            "fingerprint": fp, "libraries": libs,
            "environment": environment_report({k: gpu.get(k) for k in ("device_name", "gcn_arch", "total_vram_gb", "torch", "hip", "vram_limit")}),
            "aotriton_experimental": os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"),
            "plan": {"method": method, "master_dtype": master, "optimizer": o["name"], "accum_rows": accum, "micro_tokens": corpus.micro_tokens,
                     "steps_per_epoch": corpus.steps_per_epoch, "total_steps_schedule": total_steps, "trainable_params": trainable,
                     "load_seconds": round(load_seconds, 1), "budget_hours": cfg["budget"]["hours"]},
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
            model.save_pretrained(tmp / ("adapter" if method == "lora" else "model"))
            torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
            torch.save({"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state(index)}, tmp / "rng.pt")
            (tmp / "state.json").write_text(json.dumps({"fingerprint": fp, "state": state, "tag": tag, "saved": time.strftime("%FT%T")}), encoding="utf-8")
            (tmp / "COMPLETE").write_text("ok", encoding="utf-8")
            tmp.rename(final)
            for old in list_checkpoints(ckpt_dir)[:-cfg["checkpoint"]["keep"]]:
                shutil.rmtree(old, ignore_errors=True)
            print(f"[{run}] checkpoint {final.name} ({tag}) em {time.time() - started:.1f}s", flush=True)
            return final

        def run_eval(final: bool = False) -> dict:
            if args.no_eval:
                return {}
            ev = cfg["eval"]
            started = time.time()
            out = {"step": state["step"], "epoch": round(state["step"] / corpus.steps_per_epoch, 3), "tokens": state["real_tokens"],
                   "t_utc": time.strftime("%FT%TZ", time.gmtime())}
            with torch.autocast("cuda", dtype=torch.bfloat16):
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
                if method == "lora":
                    best = ad_dir / "best"
                    if best.exists():
                        shutil.rmtree(best)
                    model.save_pretrained(best)
                    (best / "best.json").write_text(json.dumps(out), encoding="utf-8")
                else:   # full FT: so registra o passo do melhor val_loss (o checkpoint mais proximo fica em checkpoints/; sem copia de 1-3 GB por avaliacao)
                    (rep_dir / "best.json").write_text(json.dumps(out), encoding="utf-8")
            return out

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
                lr = lr_at(step, total_steps, {**o, "lr": float(o["lr"])})
                for group in optimizer.param_groups:
                    group["lr"] = lr
                rows = [corpus.row(corpus.plan_index(step, k)) for k in range(accum)]
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
                record = {"step": state["step"], "epoch": round(state["step"] / corpus.steps_per_epoch, 3), "loss": round(loss, 5), "lr": lr,
                          "grad_norm": round(grad_norm, 4), "tokens_real": real_tokens, "tokens_padded": padded, "target_tokens": n_target,
                          "step_s": round(dt, 2), "tok_s_real": round(real_tokens / dt, 1), "tok_s_padded": round(padded / dt, 1),
                          "tokens_seen": state["real_tokens"], "elapsed_h": round(state["elapsed_train_s"] / 3600, 3), **vram(), "t": time.strftime("%FT%T")}
                with metrics_file.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(f"[{run}] step {state['step']}/{total_steps} ep {record['epoch']} loss {loss:.4f} lr {lr:.2e} gn {grad_norm:.2f} {dt:.1f}s "
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

        summary, export = {}, None
        if status in ("completed", "stopped_budget", "stopped_max_steps", "paused", "interrupted"):
            save_checkpoint(status)
        if status in ("completed", "stopped_budget", "stopped_max_steps"):
            summary = run_eval(final=True)
            if method == "lora":
                final_dir = ad_dir / "final"
                if final_dir.exists():
                    shutil.rmtree(final_dir)
                model.save_pretrained(final_dir)
                (final_dir / "train_state.json").write_text(json.dumps({"state": state, "status": status, "fingerprint": fp, "final_eval": summary}), encoding="utf-8")
            if not args.no_export:
                export = export_merged(model, method, cfg, model_dir, merged_dir)
                (merged_dir / "train_state.json").write_text(json.dumps({"state": state, "status": status, "fingerprint": fp, "final_eval": summary}), encoding="utf-8")
        run_manifest.update({"status": status, "finished_utc": time.strftime("%FT%TZ", time.gmtime()), "final_step": state["step"], "total_steps": total_steps,
                             "tokens_seen": state["real_tokens"], "elapsed_train_h": round(state["elapsed_train_s"] / 3600, 3),
                             "best_val_loss": state["best_val_loss"], "final_eval": summary, "schedule_changes": state["schedule_changes"],
                             "peak_vram_gb": vram(), "restarts": state["restarts"], "export": export})
        write_json(manifest_path, run_manifest)
        print(f"[{run}] fim: {status} passo {state['step']}/{total_steps} tokens {state['real_tokens']:,}", flush=True)
        del optimizer, model
        torch.cuda.empty_cache()
        return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
