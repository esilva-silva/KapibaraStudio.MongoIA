"""Sonda de QLoRA (job J03b, Fase 01 §6.6): tokens/s e VRAM de pico do professor candidato na RX 7800 XT.

Receita igual à do treino (Fase 02 §3.3 / v1): base NF4 double-quant (embedding e lm_head em bf16), LoRA em todos os lineares,
gradient checkpointing, cross-entropy em blocos só nas posições de resposta, PagedAdamW 8-bit. O alocador é limitado à VRAM
dedicada (limit_vram da v1): transbordo para a memória compartilhada do Windows vira OutOfMemoryError (= reprova).

Dados: exemplos de validação da v1 renderizados no contrato FIM da IDE (QwenFim/DeepSeekFim) e empacotados em sequências de
exatamente --micro-tokens tokens. Os pesos treinados são descartados; só as medições são gravadas.

Uso (venv ROCm da v1):
  python -I bakeoff/scripts/probe_qlora.py --model BaseRepos/Qwen2.5-Coder-14B --micro-tokens 4096 8192 --out reports/teacher_probe
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bakeoff" / "v1" / "scripts"))

from compact.contracts import contract_for, is_chat, prompt_ids  # noqa: E402
from compact.env import configure_caches, limit_vram, require_gpu  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def packed_batches(contract, rows: list[dict], micro_tokens: int, count: int) -> list[tuple[list[int], list[int]]]:
    """Sequências de micro_tokens tokens: (input_ids, mask de perda). Concatena exemplos prompt+resposta+EOS."""
    out, ids, loss = [], [], []
    for row in rows * 50:
        prompt = prompt_ids(contract, row, 2048)
        target = row["output"] if is_chat(row) else row["completion"]
        answer = contract.tokenizer.encode(target, add_special_tokens=False)[:256] + [contract.eos]
        ids += prompt + answer
        loss += [0] * len(prompt) + [1] * len(answer)
        while len(ids) >= micro_tokens:
            out.append((ids[:micro_tokens], loss[:micro_tokens]))
            ids, loss = ids[micro_tokens:], loss[micro_tokens:]
            if len(out) == count:
                return out
    raise SystemExit("dados insuficientes para o número de passos pedido")


def chunked_loss(model, input_ids, loss_mask, chunk: int = 1024):
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    def piece_ce(hidden, targets, head):
        logits = F.linear(hidden.to(head.dtype), head).float()
        return F.cross_entropy(logits, targets, reduction="sum")

    inner = model.get_base_model()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = inner.model(input_ids=input_ids, use_cache=False).last_hidden_state
    mask = loss_mask[:, 1:].bool()
    selected, targets = hidden[:, :-1][mask], input_ids[:, 1:][mask]
    head = inner.lm_head.weight
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, selected.shape[0], chunk):
        part = slice(start, start + chunk)
        total = total + checkpoint(piece_ce, selected[part], targets[part], head, use_reentrant=False)
    return total / max(1, int(mask.sum()))


def probe(model_path: Path, micro_tokens: int, steps: int, warmup: int, rank: int, rows: list[dict], gpu: dict) -> dict:
    import bitsandbytes as bnb
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    index = gpu["selected_index"]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(index)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    contract = contract_for(tokenizer)
    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map={"": index}, dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                               bnb_4bit_compute_dtype=torch.bfloat16))
    load_seconds = time.time() - started
    off_gpu = [n for n, p in model.named_parameters() if p.device.type != "cuda"]
    if off_gpu:
        raise SystemExit(f"{len(off_gpu)} parâmetros fora da GPU: offload não permitido")
    for param in model.parameters():
        param.requires_grad_(False)
    weights_gb = torch.cuda.memory_allocated(index) / 2**30
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = bnb.optim.PagedAdamW8bit([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    batches = packed_batches(contract, rows, micro_tokens, warmup + steps)
    model.train()

    result = {"model": str(model_path), "micro_tokens": micro_tokens, "lora_rank": rank, "load_seconds": round(load_seconds, 1),
              "weights_gb": round(weights_gb, 2), "trainable_params": trainable, "steps": steps, "warmup": warmup}
    timed_tokens, timed_seconds, losses = 0, 0.0, []
    try:
        for step, (ids, mask) in enumerate(batches):
            input_ids = torch.tensor([ids], device=f"cuda:{index}")
            loss_mask = torch.tensor([mask], device=f"cuda:{index}")
            torch.cuda.synchronize(index)
            begin = time.perf_counter()
            loss = chunked_loss(model, input_ids, loss_mask)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(index)
            elapsed = time.perf_counter() - begin
            losses.append(float(loss))
            if step >= warmup:
                timed_tokens += micro_tokens
                timed_seconds += elapsed
            if step % 10 == 0:
                print(f"  step {step:3d} loss {float(loss):.3f} {micro_tokens / elapsed:7.0f} tok/s "
                      f"peak {torch.cuda.max_memory_allocated(index) / 2**30:.2f} GB", flush=True)
        result["status"] = "ok"
    except torch.OutOfMemoryError as error:
        result["status"] = "oom"
        result["error"] = str(error).splitlines()[0][:300]
    result.update({
        "tokens_per_second": round(timed_tokens / timed_seconds, 1) if timed_seconds else None,
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated(index) / 2**30, 2),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved(index) / 2**30, 2),
        "loss_first": round(losses[0], 4) if losses else None, "loss_last": round(losses[-1], 4) if losses else None,
        "loss_finite": all(math.isfinite(v) for v in losses),
    })
    del optimizer, model
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--micro-tokens", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--vram-fraction", type=float, default=0.94)
    parser.add_argument("--data", default=str(ROOT / "bakeoff" / "data" / "v1_validation.jsonl"))
    parser.add_argument("--out", default=str(ROOT / "reports" / "teacher_probe"))
    args = parser.parse_args()

    configure_caches(ROOT)
    model_path = (ROOT / args.model) if not Path(args.model).is_absolute() else Path(args.model)
    rows = [r for r in read_jsonl(Path(args.data)) if r["type"] in ("fim", "completion", "chat", "correction")]
    gpu = require_gpu("7800 XT")
    gpu["vram_limit"] = limit_vram(gpu, args.vram_fraction, reserve_gb=1.6)
    print(f"VRAM limit: {gpu['vram_limit']}", flush=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for micro in args.micro_tokens:
        print(f"=== {model_path.name} micro_tokens={micro} rank={args.rank}", flush=True)
        result = probe(model_path, micro, args.steps, args.warmup, args.rank, rows, gpu)
        result["vram_limit"] = gpu["vram_limit"]
        print(json.dumps(result, ensure_ascii=False), flush=True)
        results.append(result)
    target = out_dir / f"{model_path.name}-r{args.rank}.json"
    target.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
