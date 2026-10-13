"""Utilitarios compartilhados dos scripts do professor (Fase 02): caminhos, trava de GPU, hashes, carga HF e geracao greedy.

Reusa o ambiente/venv ROCm do bake-off de 2026-10-09 (C:/SlopStudio.MongoAI-artifacts/.venv, torch 2.10.0+rocm7.14, transformers
4.57.6, bitsandbytes 0.50.2) e os modulos `compact` de bakeoff/v1/scripts (contratos de prompt, metricas, validacao estatica).
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BAKEOFF_SCRIPTS = ROOT / "bakeoff" / "v1" / "scripts"
BAKEOFF_WORKSPACE = ROOT / "bakeoff" / "v1"
BASE_REPOS = ROOT / "BaseRepos"
LOCK_FILE = BASE_REPOS / "base-repos.lock.json"
TEACHER_REPO = ROOT / "Repos" / "KapiTeacher-Mongo-7B"
GPU_LOCK = ROOT / "tmp" / "gpu.lock"
ENV_VENV_PY = ROOT / "env" / ".venv" / "Scripts" / "python.exe"  # tem pymongo (sem torch): usado so na fase de execucao

if str(BAKEOFF_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(BAKEOFF_SCRIPTS))


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def lock_entry(model_dir: Path) -> dict | None:
    if not LOCK_FILE.exists():
        return None
    return json.loads(LOCK_FILE.read_text(encoding="utf-8"))["repos"].get(model_dir.name)


def model_fingerprint(model_dir: Path) -> dict:
    """Hash do modelo: SHA-256 sobre 'arquivo:sha256' de todos os pesos/tokenizer/config, vindo do lock (ou calculado)."""
    entry = lock_entry(model_dir)
    files: dict[str, str] = {}
    source = "base-repos.lock.json"
    for name in sorted(p.name for p in model_dir.iterdir() if p.is_file()):
        if name.startswith("."):
            continue
        known = next((f["sha256"] for f in (entry or {}).get("files", []) if f["path"] == name), None)
        if known is None:
            known = sha256_file(model_dir / name)
            source = "lock+calculado"
        files[name] = known
    combined = sha256_text("\n".join(f"{k}:{v}" for k, v in sorted(files.items())))
    return {"model_sha256": combined, "files": files, "source": source, "repo_id": (entry or {}).get("repo_id"),
            "revision": (entry or {}).get("revision")}


def library_versions() -> dict[str, str]:
    import importlib
    import platform

    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("torch", "transformers", "bitsandbytes", "tokenizers", "safetensors", "accelerate", "pymongo"):
        try:
            out[name] = getattr(importlib.import_module(name), "__version__", "?")
        except Exception:  # noqa: BLE001
            out[name] = "indisponivel"
    try:
        import torch

        out["hip"] = str(getattr(torch.version, "hip", None))
    except Exception:  # noqa: BLE001
        pass
    return out


@contextlib.contextmanager
def gpu_lock(job: str):
    """tmp/gpu.lock (Fase 05 §4.2): uma GPU, um job por vez. Trava obsoleta (pid morto) e tomada."""
    import psutil

    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    if GPU_LOCK.exists():
        try:
            holder = json.loads(GPU_LOCK.read_text(encoding="utf-8"))
            if psutil.pid_exists(int(holder.get("pid", -1))):
                raise SystemExit(f"GPU ocupada: {GPU_LOCK} detido por {holder}")
        except (ValueError, OSError):
            pass
    GPU_LOCK.write_text(json.dumps({"pid": os.getpid(), "job": job, "started": time.strftime("%FT%T")}), encoding="utf-8")
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            GPU_LOCK.unlink()


def load_hf(model_dir: Path, load: str):
    """Mesmo carregamento do bake-off (HfBackend): bf16 ou NF4 double-quant com compute bf16, sdpa, GPU unica."""
    import torch
    from compact.env import configure_caches, limit_vram, require_gpu
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    configure_caches(BAKEOFF_WORKSPACE)
    gpu = require_gpu("7800 XT")
    gpu["vram_limit"] = limit_vram(gpu, 0.96)
    device = torch.device(f"cuda:{gpu['selected_index']}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    extra = {}
    if load == "nf4":
        extra["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                                          bnb_4bit_compute_dtype=torch.bfloat16)
    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map={"": device.index}, low_cpu_mem_usage=True,
                                                 attn_implementation="sdpa", **extra)
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    return model, tokenizer, device, gpu, round(time.time() - started, 1)


def auto_load(model_dir: Path) -> str:
    """bf16 para <= ~3B parametros (0.5B/1.5B), NF4 para 6.7B/7B (como no bake-off)."""
    import struct

    total = 0
    for file in model_dir.glob("*.safetensors"):
        with file.open("rb") as handle:
            header = json.loads(handle.read(struct.unpack("<Q", handle.read(8))[0]))
        total += sum(v_prod(meta["shape"]) for name, meta in header.items() if name != "__metadata__")
    return "bf16" if total < 4e9 else "nf4"


def v_prod(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


class Generator:
    """Geracao greedy com KV cache (igual ao HfBackend do bake-off) + mascara do padding do vocab (linhas > len(tokenizer) = -inf)."""

    def __init__(self, model, tokenizer, device) -> None:
        import torch

        self.torch, self.model, self.tokenizer, self.device = torch, model, tokenizer, device
        self.real_vocab = len(tokenizer)

    def generate(self, ids: list[int], limit: int, stops: frozenset[int], suffix: str = "") -> dict:
        torch = self.torch
        real = self.real_vocab
        with torch.inference_mode():
            torch.cuda.synchronize(self.device)
            start = time.perf_counter()
            out = self.model(input_ids=torch.tensor([ids], device=self.device), use_cache=True, logits_to_keep=1)
            token = int(out.logits[0, -1, :real].argmax())
            torch.cuda.synchronize(self.device)
            ttft = time.perf_counter() - start
            generated, past = [token], out.past_key_values
            reason = "eos" if token in stops else "max"
            while token not in stops and len(generated) < limit:
                if len(suffix) >= 2 and suffix in self.decode(generated):
                    reason = "suffix"
                    break
                out = self.model(input_ids=torch.tensor([[token]], device=self.device), past_key_values=past, use_cache=True)
                past = out.past_key_values
                token = int(out.logits[0, -1, :real].argmax())
                generated.append(token)
                if token in stops:
                    reason = "eos"
            torch.cuda.synchronize(self.device)
        return {"tokens": generated, "ttft": ttft, "stop": reason}

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
