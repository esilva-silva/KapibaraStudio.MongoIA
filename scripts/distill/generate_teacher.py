#!/usr/bin/env python3
"""Geracao em lote do KapiTeacher-Mongo-7B (HF NF4 + adapter) sobre o pool de prompts FIM -> candidatas para a destilacao.

  C:/SlopStudio.MongoAI-artifacts/.venv/Scripts/python.exe -I scripts/distill/generate_teacher.py \\
      --pool data/distilled/v2c/pool --out data/distilled/v2c/gen --resume [--save-topk] [--samples 1] [--limit N]

Saida (append-only, retomavel):
  <out>/gen-NNNNN.jsonl   candidatas: 1 greedy (k=0, argmax com o padding do vocab mascarado) + `samples` amostras (k>=1, T=0,2),
                          UMA LINHA POR CANDIDATA com idx do pool, id, seed, raw, pred (pos-processamento da IDE), stop, logprobs.
  <out>/done.idx          log de commit por BLOCO: `block first_idx last_idx n_lines shard shard_bytes`. Na retomada cada shard e
                          TRUNCADO ao ultimo tamanho confirmado (blocos nao confirmados sao descartados e regerados); nada a deduplicar.
  <out>/topk/t-NNNNNN.npz (opcional, --save-topk) k=32: topk_ids uint32 [P,k], topk_logprobs fp16 [P,k], tail_mass fp16 [P] +
                          cand_idx/cand_k/n_pos para localizar cada candidata (formato `topk-v1` da Fase 02 §3.5, so posicoes da resposta).
  <out>/metrics.jsonl, heartbeat.json (watchdog da fila), gen_summary.json (vazao real, GB de top-k por 1k exemplos).

Determinismo: prompts e lotes dependem so do CONTEUDO do bloco (lotes por comprimento, ordenados por (tamanho, idx)); a seed de cada
amostra e blake2b(seed:id:k), nao a posicao; greedy nao usa aleatoriedade. Retomar regenera o bloco inteiro com os mesmos lotes.
O professor NUNCA ve o `gold`: o prompt e construido so de prefix/suffix/editor.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _distill_common import (ROOT, git_hash, item_seed, load_config, pool_shards, rel, resolve, sha256_file, write_json)  # noqa: E402

TOPK_BYTES_PER_POS = lambda k: k * 4 + k * 2 + 2   # noqa: E731  (uint32 ids + fp16 logprobs + fp16 tail_mass)


# ----------------------------------------------------------------------------------------------------------------------------------
# Motor (independente de torch: testavel com um backend falso)
# ----------------------------------------------------------------------------------------------------------------------------------
class Engine:
    def __init__(self, backend, pool_dir: Path, out_dir: Path, *, block_size: int, shard_size: int, max_new: int, samples: int,
                 temperature: float, top_p: float, topk_k: int, save_topk: bool, base_seed: int, batch_tokens: int, max_rows: int,
                 log=print) -> None:
        if shard_size % block_size:
            raise SystemExit("shard_size deve ser multiplo de block_size")
        self.b, self.pool_dir, self.out = backend, Path(pool_dir), Path(out_dir)
        self.block, self.shard_size = block_size, shard_size
        self.max_new, self.samples, self.temp, self.top_p = max_new, samples, temperature, top_p
        self.k, self.save_topk, self.seed = topk_k, save_topk, base_seed
        self.batch_tokens, self.max_rows, self.log = batch_tokens, max_rows, log
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "topk").mkdir(exist_ok=True)
        self.done_path = self.out / "done.idx"
        self.block_log: list[tuple[int, float]] = []
        self.stats = {"blocks": 0, "items": 0, "cands": 0, "prompt_tokens": 0, "padded_prompt_tokens": 0, "gen_tokens": 0, "seconds": 0.0,
                      "prefill_seconds": 0.0, "decode_seconds": 0.0, "topk_positions": 0, "topk_bytes": 0, "batches": 0}

    # -- retomada -----------------------------------------------------------------------------------------------------------------
    def committed(self) -> dict[int, dict]:
        """{bloco: {...}} do done.idx e TRUNCA os shards ao tamanho confirmado (descarta blocos nao confirmados / linhas rasgadas)."""
        done: dict[int, dict] = {}
        sizes: dict[str, int] = {}
        if self.done_path.exists():
            for line in self.done_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 6:
                    continue                                          # linha rasgada no fim do done.idx: ignora
                block, first, last, n, shard, size = parts
                done[int(block)] = {"first": int(first), "last": int(last), "lines": int(n), "shard": shard, "bytes": int(size)}
                sizes[shard] = max(sizes.get(shard, 0), int(size))
        for shard in sorted(self.out.glob("gen-[0-9][0-9][0-9][0-9][0-9].jsonl")):
            keep = sizes.get(shard.name, 0)
            if shard.stat().st_size != keep:
                with open(shard, "r+b") as handle:
                    handle.truncate(keep)
        return done

    # -- uma execucao ---------------------------------------------------------------------------------------------------------------
    def read_blocks(self, start_block: int, stop_block: int | None, only_idx=None):
        """Gera (bloco, [(idx, linha_bruta)]) em streaming sequencial do pool (so le/parsa o necessario)."""
        buf: list[tuple[int, str]] = []
        idx = 0
        for shard in pool_shards(self.pool_dir):
            with open(shard, encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    blk = idx // self.block
                    if stop_block is not None and blk >= stop_block:
                        if buf:
                            yield buf[0][0] // self.block, buf
                        return
                    if blk >= start_block and (only_idx is None or idx in only_idx):
                        buf.append((idx, line))
                    if buf and (idx + 1) % self.block == 0:       # fecha o bloco ao cruzar a fronteira
                        yield buf[0][0] // self.block, buf
                        buf = []
                    idx += 1
        if buf:
            yield buf[0][0] // self.block, buf

    def make_batches(self, items: list[dict]) -> list[list[dict]]:
        """Lotes por comprimento: ordena por (tamanho do prompt, idx) e enche ate max_rows / batch_tokens (tokens PADDED)."""
        order = sorted(items, key=lambda it: (len(it["ids"]), it["idx"]))
        batches, cur, cur_max = [], [], 0
        for it in order:
            n = len(it["ids"])
            new_max = max(cur_max, n)
            if cur and (len(cur) + 1 > self.max_rows or new_max * (len(cur) + 1) > self.batch_tokens):
                batches.append(cur)
                cur, cur_max, new_max = [], 0, n
            cur.append(it)
            cur_max = new_max
        if cur:
            batches.append(cur)
        return batches

    def run(self, *, resume: bool = True, start_block: int = 0, stop_block: int | None = None, max_blocks: int | None = None,
            hook=None, only_idx=None) -> dict:
        done = self.committed() if resume else {}
        if not resume:
            for p in list(self.out.glob("gen-*.jsonl")) + [self.done_path]:
                p.unlink(missing_ok=True)
        t_run = time.time()
        processed = 0
        for blk, raw in self.read_blocks(start_block, stop_block, only_idx):
            if blk in done:
                continue
            if max_blocks is not None and processed >= max_blocks:
                break
            t0 = time.perf_counter()
            items = []
            for idx, line in raw:
                rec = json.loads(line)
                items.append({"idx": idx, "id": rec["id"], "rec": rec, "ids": self.b.encode(rec)})
            lines: list[str] = []
            topk_parts: list[dict] = []
            for batch in self.make_batches(items):
                seeds = [[item_seed(self.seed, it["id"], k) for k in range(1, self.samples + 1)] for it in batch]
                res = self.b.generate(batch, seeds=seeds, max_new=self.max_new, temperature=self.temp, top_p=self.top_p,
                                      topk_k=self.k if self.save_topk else 0)
                st = res.get("stats", {})
                self.stats["batches"] += 1
                self.stats["prefill_seconds"] += st.get("prefill_seconds", 0.0)
                self.stats["decode_seconds"] += st.get("decode_seconds", 0.0)
                self.stats["padded_prompt_tokens"] += st.get("padded_tokens", 0)
                for it, cands in zip(batch, res["cands"]):
                    for c in cands:
                        rec = self.finalize(it, c)
                        if c.get("topk") is not None:
                            topk_parts.append({"idx": it["idx"], "k": c["k"], **c["topk"]})
                            rec["topk_pos"] = int(c["topk"]["ids"].shape[0])
                        lines.append(rec)
                        self.stats["gen_tokens"] += rec["gen_tokens"]
                self.stats["prompt_tokens"] += sum(len(it["ids"]) for it in batch)
                self.stats["cands"] += sum(len(c) for c in res["cands"])
                if hook:
                    hook(blk, len(lines))
            lines.sort(key=lambda r: (r["idx"], r["k"]))
            self.commit_block(blk, items, lines, topk_parts)
            dt = time.perf_counter() - t0
            self.stats["blocks"] += 1
            self.stats["items"] += len(items)
            self.stats["seconds"] += dt
            processed += 1
            self.block_log.append((len(items), dt))
            self.metric(blk, len(items), dt)
        self.stats["wall_seconds"] = round(time.time() - t_run, 1)
        return self.summary()

    # -- pos-processamento da candidata -----------------------------------------------------------------------------------------------
    def finalize(self, it: dict, c: dict) -> dict:
        rec = it["rec"]
        stops = self.b.stops
        toks = list(c["tokens"])
        cut = next((i for i, t in enumerate(toks) if t in stops), None)
        body = toks if cut is None else toks[:cut]
        raw = self.b.decode(body)
        pred = self.b.postprocess(raw, rec)
        suffix = rec["suffix"][:1024]
        if cut is not None:
            reason = "eos"
        elif len(suffix) >= 2 and suffix in raw:
            reason = "suffix"
        else:
            reason = "max"
        # numero de posicoes da resposta que o aluno veria: tokens cujo inicio esta antes do fim de `pred` (+ EOS se a parada foi natural)
        keep = len(body)
        if pred != raw:
            keep = next((n for n in range(1, len(body) + 1) if len(self.b.postprocess(self.b.decode(body[:n]), rec)) >= len(pred)), len(body))
        elif cut is not None:
            keep = len(body) + 1                                    # inclui a posicao do <|endoftext|>
        if c.get("topk") is not None:
            for key in ("ids", "lps", "tail"):
                c["topk"][key] = c["topk"][key][:keep]
        lps = list(c["logprobs"][:keep]) or [0.0]
        out = {"idx": it["idx"], "id": it["id"], "origin": rec.get("origin"), "k": c["k"], "mode": c["mode"], "seed": c.get("seed"),
               "raw": raw, "pred": pred, "stop": reason, "gen_tokens": len(toks), "kept_tokens": keep, "prompt_tokens": len(it["ids"]),
               "logprob_mean": round(sum(lps) / len(lps), 5), "logprob_min": round(min(lps), 5)}
        return out

    # -- commit ----------------------------------------------------------------------------------------------------------------------
    def commit_block(self, blk: int, items: list[dict], lines: list[dict], topk_parts: list[dict]) -> None:
        first, last = items[0]["idx"], items[-1]["idx"]
        shard = f"gen-{first // self.shard_size:05d}.jsonl"
        if self.save_topk and topk_parts:
            self.write_topk(blk, topk_parts)
        payload = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in lines)
        path = self.out / shard
        with open(path, "ab") as handle:
            handle.write(payload.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
            size = handle.tell()
        with open(self.done_path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(f"{blk} {first} {last} {len(lines)} {shard} {size}\n")
            handle.flush()
            os.fsync(handle.fileno())

    def write_topk(self, blk: int, parts: list[dict]) -> None:
        import numpy as np

        parts.sort(key=lambda p: (p["idx"], p["k"]))
        n_pos = np.array([p["ids"].shape[0] for p in parts], dtype=np.uint16)
        k = self.k
        ids = np.concatenate([p["ids"].reshape(-1, k) for p in parts]).astype(np.uint32)
        lps = np.concatenate([p["lps"].reshape(-1, k) for p in parts]).astype(np.float16)
        tail = np.concatenate([p["tail"].reshape(-1) for p in parts]).astype(np.float16)
        path = self.out / "topk" / f"t-{blk:06d}.npz"
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, cand_idx=np.array([p["idx"] for p in parts], dtype=np.uint32), cand_k=np.array([p["k"] for p in parts], dtype=np.uint8),
                 n_pos=n_pos, topk_ids=ids, topk_logprobs=lps, tail_mass=tail)
        os.replace(tmp, path)
        self.stats["topk_positions"] += int(ids.shape[0])
        self.stats["topk_bytes"] += int(path.stat().st_size)

    def metric(self, blk: int, n: int, dt: float) -> None:
        row = {"t": time.strftime("%FT%T"), "block": blk, "items": n, "seconds": round(dt, 1), "ex_per_s": round(n / dt, 3),
               "gen_tok_s": round(self.stats["gen_tokens"] / max(1e-9, self.stats["seconds"]), 1),
               "cum_items": self.stats["items"]}
        with open(self.out / "metrics.jsonl", "a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row) + "\n")
        self.heartbeat(row)
        self.log(f"[bloco {blk}] {n} itens em {dt:.0f}s ({row['ex_per_s']} ex/s)")

    def heartbeat(self, row: dict | None = None) -> None:
        write_json(self.out / "heartbeat.json", {"t": time.strftime("%FT%T"), "pid": os.getpid(), **(row or {}), "stats": self.stats})

    def summary(self) -> dict:
        s = self.stats
        sec = max(1e-9, s["seconds"])
        ex = max(1, s["items"])
        out = {**s, "ex_per_s": round(s["items"] / sec, 3), "ex_per_h": round(s["items"] / sec * 3600, 1),
               "gen_tok_s": round(s["gen_tokens"] / sec, 1), "prompt_tok_s": round(s["prompt_tokens"] / sec, 1),
               "mean_prompt_tokens": round(s["prompt_tokens"] / ex, 1), "mean_gen_tokens_per_cand": round(s["gen_tokens"] / max(1, s["cands"]), 1),
               "topk_gb_per_1k_examples": round(s["topk_bytes"] / 2 ** 30 / ex * 1000, 4) if s["topk_bytes"] else 0.0,
               "topk_bytes_per_position_theoretical": TOPK_BYTES_PER_POS(self.k)}
        if len(self.block_log) >= 2:       # regime permanente: descarta o 1o bloco (carga/aquecimento de kernels)
            n = sum(x for x, _ in self.block_log[1:])
            d = sum(t for _, t in self.block_log[1:])
            out["steady_ex_per_s"], out["steady_ex_per_h"] = round(n / d, 3), round(n / d * 3600, 1)
        return out


# ----------------------------------------------------------------------------------------------------------------------------------
# Backend HF (NF4 + adapter)
# ----------------------------------------------------------------------------------------------------------------------------------
class HFBackend:
    def __init__(self, model_dir: Path, adapter: Path | None, load: str, contract_name: str = "editor-context-v1") -> None:
        import torch
        from compact.contracts import contract_for, postprocess
        from _common import load_hf, sha256_file as _sha

        self.torch = torch
        self.model, self.tokenizer, self.device, self.gpu, self.load_seconds = load_hf(model_dir, load)
        self.adapter_info = None
        if adapter is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter, torch_device=str(self.device))
            self.adapter_info = {"path": str(adapter), "sha256": _sha(adapter / "adapter_model.safetensors")}
        self.model.eval()
        self.contract = contract_for(self.tokenizer)
        self.stops = frozenset(self.contract.stops)
        self.real = len(self.tokenizer)
        self.pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else min(self.stops)
        self.contract_name = contract_name
        self._post = postprocess
        self.stop_tensor = torch.tensor(sorted(self.stops), device=self.device)

    # -- prompt = o MESMO do eval_teacher (editor-context-v1): cabecalho do editor + janelas 4096/1024 + orcamento de 2048 tokens ------
    def encode(self, rec: dict) -> list[int]:
        import render_contract as rc

        prefix, suffix = rc.bound_window(rec["prefix"], rec["suffix"])
        before = rc.render_model_prefix(prefix, rec.get("editor"), rec.get("language") or rc.editor_language(rec))
        return self.contract.build(before, suffix, 2048)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def postprocess(self, text: str, rec: dict) -> str:
        return self._post(text, {"type": "fim", **rec})

    def generate(self, batch: list[dict], *, seeds: list[list[int]], max_new: int, temperature: float, top_p: float, topk_k: int) -> dict:
        torch = self.torch
        dev, real = self.device, self.real
        B = len(batch)
        R = 1 + (len(seeds[0]) if seeds else 0)
        L = max(len(it["ids"]) for it in batch)
        ids = torch.full((B, L), self.pad, dtype=torch.long)
        mask = torch.zeros((B, L), dtype=torch.long)
        for i, it in enumerate(batch):
            n = len(it["ids"])
            ids[i, L - n:] = torch.tensor(it["ids"], dtype=torch.long)
            mask[i, L - n:] = 1
        ids, mask = ids.to(dev), mask.to(dev)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        with torch.inference_mode():
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            out = self.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True, logits_to_keep=1)
            past = out.past_key_values
            logits = out.logits[:, -1, :real].float()
            last_pos = pos[:, -1:]
            if R > 1:                                      # a mesma pre-carga (prefill) serve a greedy e as amostras
                past.batch_repeat_interleave(R)
                logits = logits.repeat_interleave(R, 0)
                mask = mask.repeat_interleave(R, 0)
                last_pos = last_pos.repeat_interleave(R, 0)
            torch.cuda.synchronize(dev)
            t_prefill = time.perf_counter() - t0
            rows = B * R
            gens = {}
            for i in range(B):
                for j in range(1, R):
                    g = torch.Generator(device=dev)
                    g.manual_seed(seeds[i][j - 1])
                    gens[i * R + j] = g
            finished = torch.zeros(rows, dtype=torch.bool, device=dev)
            toks, chosen, tk_ids, tk_lps, tk_tail = [], [], [], [], []
            t1 = time.perf_counter()
            for step in range(max_new):
                lp = torch.log_softmax(logits, dim=-1)
                tok = logits.argmax(-1)
                for r, g in gens.items():
                    probs = torch.softmax(logits[r] / max(temperature, 1e-6), dim=-1)
                    if top_p < 1.0:
                        sp, si = probs.sort(descending=True)
                        cum = sp.cumsum(0)
                        sp = torch.where(cum - sp > top_p, torch.zeros_like(sp), sp)
                        tok[r] = si[torch.multinomial(sp / sp.sum(), 1, generator=g)]
                    else:
                        tok[r] = torch.multinomial(probs, 1, generator=g)[0]
                toks.append(tok)
                chosen.append(lp.gather(1, tok[:, None])[:, 0])
                if topk_k:
                    v, i_ = lp.topk(topk_k, dim=-1)
                    tk_ids.append(i_.to(torch.int32))
                    tk_lps.append(v.to(torch.float16))
                    tk_tail.append((1.0 - v.exp().sum(-1)).clamp(min=0).to(torch.float16))
                finished |= torch.isin(tok, self.stop_tensor)
                if bool(finished.all()):
                    break
                if step % 8 == 7 and step + 1 < max_new:   # parada por eco do sufixo (como a IDE); o pos-processamento corta igual
                    cur = torch.stack(toks, 1).cpu().tolist()
                    for r in torch.nonzero(~finished).flatten().tolist():
                        suf = batch[r // R]["rec"]["suffix"][:1024]
                        if len(suf) >= 2 and suf in self.decode([t for t in cur[r] if t not in self.stops]):
                            finished[r] = True
                    if bool(finished.all()):
                        break
                mask = torch.cat([mask, torch.ones((rows, 1), dtype=mask.dtype, device=dev)], dim=1)
                last_pos = last_pos + 1
                out = self.model(input_ids=tok[:, None], attention_mask=mask, position_ids=last_pos, past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[:, -1, :real].float()
            torch.cuda.synchronize(dev)
            t_decode = time.perf_counter() - t1
            tok_m = torch.stack(toks, 1).cpu().tolist()
            lp_m = torch.stack(chosen, 1).cpu().tolist()
            if topk_k:
                ids_m = torch.stack(tk_ids, 1).cpu().numpy()
                lps_m = torch.stack(tk_lps, 1).cpu().numpy()
                tail_m = torch.stack(tk_tail, 1).cpu().numpy()
        cands = []
        for i in range(B):
            row = []
            for j in range(R):
                r = i * R + j
                seq = tok_m[r]
                cut = next((n for n, t in enumerate(seq) if t in self.stops), None)
                n = len(seq) if cut is None else cut + 1
                c = {"k": j, "mode": "greedy" if j == 0 else "sample", "seed": None if j == 0 else seeds[i][j - 1],
                     "tokens": seq[:n], "logprobs": lp_m[r][:n]}
                if topk_k:
                    c["topk"] = {"ids": ids_m[r, :n], "lps": lps_m[r, :n], "tail": tail_m[r, :n]}
                row.append(c)
            cands.append(row)
        return {"cands": cands, "stats": {"prefill_seconds": t_prefill, "decode_seconds": t_decode, "padded_tokens": B * L}}


class LlamaServerBackend:
    """GGUF via llama-server (HTTP em 127.0.0.1): so sequence-level (greedy + amostras); sem top-k de logits (KD so via HF).

    O prompt e enviado como LISTA DE IDS (mesmos ids do HF/contrato Qwen FIM), portanto nao ha retokenizacao no servidor.
    Seed por item e repassada em cada requisicao; `--parallel N` slots servem os lotes do Engine por threads."""

    def __init__(self, gguf: Path | None, tokenizer_dir: Path, *, url: str | None, slots: int, ctx_per_slot: int, port: int = 8791,
                 device: str = "Vulkan0", exe: Path | None = None, extra: list[str] | None = None) -> None:
        import subprocess
        import urllib.request
        from compact.contracts import contract_for, postprocess
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self.contract = contract_for(self.tokenizer)
        self.stops = frozenset(self.contract.stops)
        self._post = postprocess
        self.slots = slots
        self.proc = None
        self.url = url or f"http://127.0.0.1:{port}"
        self.gguf = gguf
        self.load_seconds = 0.0
        self.adapter_info = None
        self.gpu = {"device_name": device}
        self.skip_greedy = False
        if url is None:
            exe = exe or ROOT / "cache" / "llama.cpp" / "b11693" / "bin" / "llama-server.exe"
            cmd = [str(exe), "-m", str(gguf), "--host", "127.0.0.1", "--port", str(port), "-ngl", "99", "-dev", device,
                   "--parallel", str(slots), "-c", str(slots * ctx_per_slot), "-fa", "on", "--no-webui"] + (extra or [])
            t0 = time.time()
            (ROOT / "logs" / "distill").mkdir(parents=True, exist_ok=True)
            self.log_f = open(ROOT / "logs" / "distill" / "llama-server.log", "wb")
            self.proc = subprocess.Popen(cmd, stdout=self.log_f, stderr=subprocess.STDOUT)
            for _ in range(600):
                try:
                    if urllib.request.urlopen(self.url + "/health", timeout=2).status == 200:
                        break
                except Exception:  # noqa: BLE001
                    if self.proc.poll() is not None:
                        raise SystemExit("llama-server morreu; veja logs/distill/llama-server.log")
                    time.sleep(1)
            else:
                raise SystemExit("llama-server nao ficou pronto em 600 s")
            self.load_seconds = round(time.time() - t0, 1)

    def close(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except Exception:  # noqa: BLE001
                self.proc.kill()
            self.log_f.close()

    def encode(self, rec: dict) -> list[int]:
        import render_contract as rc

        prefix, suffix = rc.bound_window(rec["prefix"], rec["suffix"])
        before = rc.render_model_prefix(prefix, rec.get("editor"), rec.get("language") or rc.editor_language(rec))
        return self.contract.build(before, suffix, 2048)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    def postprocess(self, text: str, rec: dict) -> str:
        return self._post(text, {"type": "fim", **rec})

    def _complete(self, ids: list[int], max_new: int, temperature: float, seed: int, top_p: float) -> dict:
        import urllib.request

        body = {"prompt": ids, "n_predict": max_new, "temperature": temperature, "seed": seed, "top_p": top_p,
                "top_k": 0 if temperature > 0 else 1, "cache_prompt": True, "return_tokens": True, "n_probs": 1, "stream": False}
        req = urllib.request.Request(self.url + "/completion", json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as resp:
            return json.loads(resp.read())

    def generate(self, batch: list[dict], *, seeds: list[list[int]], max_new: int, temperature: float, top_p: float, topk_k: int) -> dict:
        from concurrent.futures import ThreadPoolExecutor

        if topk_k:
            raise SystemExit("--save-topk so existe no backend HF (llama.cpp nao exporta o top-k completo)")
        jobs = []
        for i, it in enumerate(batch):
            if not self.skip_greedy:
                jobs.append((i, 0, it["ids"], 0.0, 0))
            for j, sd in enumerate(seeds[i], start=1):
                jobs.append((i, j, it["ids"], temperature, sd))
        t0 = time.perf_counter()
        with ThreadPoolExecutor(self.slots) as pool:
            outs = list(pool.map(lambda jb: self._complete(jb[2], max_new, jb[3], jb[4], top_p), jobs))
        dt = time.perf_counter() - t0
        cands = [[] for _ in batch]
        for (i, j, _, _, sd), o in zip(jobs, outs):
            toks = list(o.get("tokens") or [])
            lps = [float(cp.get("logprob", 0.0)) for cp in (o.get("completion_probabilities") or [])]
            lps = (lps + [0.0] * len(toks))[: len(toks)]
            if o.get("stop_type") == "eos" and not (toks and toks[-1] in self.stops):
                toks.append(min(self.stops))          # o servidor omite o EOS: reanexa para marcar a parada natural
                lps.append(0.0)
            cands[i].append({"k": j, "mode": "greedy" if j == 0 else "sample", "seed": None if j == 0 else sd, "tokens": toks, "logprobs": lps})
        return {"cands": cands, "stats": {"prefill_seconds": 0.0, "decode_seconds": dt, "padded_tokens": sum(len(it["ids"]) for it in batch)}}


# ----------------------------------------------------------------------------------------------------------------------------------
def retry_indices(filter_dir: Path) -> set[int]:
    """idx do pool cuja greedy (k=0) foi rejeitada em etapa de VALIDACAO (nao dedup/decontam/select) e que nao tem candidata aceita."""
    stages = {"semantic", "sanitize", "budget", "syntax", "api", "fields", "exec"}
    accepted = {json.loads(line)["teacher"]["pool_idx"] for line in open(filter_dir / "accepted.jsonl", encoding="utf-8")}
    out = set()
    for line in open(filter_dir / "rejected.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["k"] == 0 and r["stage"] in stages and r["pool_idx"] not in accepted:
            out.add(r["pool_idx"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--pool", help="pasta com pool-NNNNN.jsonl (padrao: <out_dir>/pool)")
    ap.add_argument("--out", help="pasta de saida (padrao: <out_dir>/gen)")
    ap.add_argument("--model", default="adapter", help="adapter (base+adapter sft_v2d, NF4) | merged (merge bf16 do outro agente, NF4) | caminho")
    ap.add_argument("--adapter")
    ap.add_argument("--load", choices=["nf4", "bf16"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--samples", type=int, help="amostras extras (T=0,2) alem da greedy; 0 = so greedy")
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--save-topk", action="store_true", help="grava top-k=32 logprobs por posicao da resposta (KD por logits)")
    ap.add_argument("--topk-k", type=int)
    ap.add_argument("--limit", type=int, help="processa so os N primeiros itens do pool (arredonda para blocos)")
    ap.add_argument("--block-size", type=int)
    ap.add_argument("--shard-size", type=int)
    ap.add_argument("--batch-tokens", type=int)
    ap.add_argument("--max-rows", type=int)
    ap.add_argument("--max-new", type=int)
    ap.add_argument("--backend", choices=["hf", "llama-server"], default="hf")
    ap.add_argument("--gguf", help="llama-server: caminho ou q8_0 | q5_k_m | f16 (Repos/KapiTeacher-Mongo-7B/gguf)")
    ap.add_argument("--server-url", help="llama-server ja em execucao (nao inicia outro)")
    ap.add_argument("--retry-from", help="2a passada: pasta de saida do filtro; gera so amostras (T=0,2) dos prompts cuja greedy foi rejeitada na validacao")
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--ctx-per-slot", type=int, default=2304)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    gc, tc = cfg["generate"], cfg["teacher"]
    out_root = resolve(cfg["out_dir"])
    pool_dir = resolve(a.pool) if a.pool else out_root / "pool"
    out = resolve(a.out) if a.out else out_root / "gen"
    block = a.block_size or gc["block_size"]
    base_model = resolve(tc["base"])
    adapter = None
    if a.model == "adapter":
        adapter = resolve(a.adapter or tc["adapter"])
    elif a.model == "merged":
        base_model = resolve(tc["merged"])
        if not (base_model / "config.json").exists():
            raise SystemExit(f"merge bf16 ainda nao existe: {rel(base_model)} (use --model adapter)")
    else:
        base_model = resolve(a.model)
        adapter = resolve(a.adapter) if a.adapter else None
    load = a.load or tc["load"]
    limit_blocks = None if not a.limit else -(-a.limit // block)
    only_idx = None
    if a.retry_from:
        only_idx = retry_indices(resolve(a.retry_from))
        if a.samples is None:
            a.samples = 1
        if a.backend != "llama-server":
            raise SystemExit("--retry-from usa o backend llama-server (so amostras)")
        if not a.out:
            out = out_root / "gen-retry"
        print(f"[retry] {len(only_idx)} prompts com greedy rejeitada na validacao")

    from _common import gpu_lock, library_versions, model_fingerprint

    engine_kwargs = dict(block_size=block, shard_size=a.shard_size or gc["shard_size"], max_new=a.max_new or gc["max_new_tokens"],
                         samples=gc["samples"] if a.samples is None else a.samples, temperature=a.temperature or gc["temperature"],
                         top_p=gc["top_p"], topk_k=a.topk_k or gc["topk"]["k"], save_topk=a.save_topk or gc["topk"]["enabled"],
                         base_seed=cfg["seed"], batch_tokens=a.batch_tokens or gc["batch_tokens"], max_rows=a.max_rows or gc["max_batch_rows"])
    with gpu_lock(f"distill_generate:{out.name}"):
        t0 = time.time()
        if a.backend == "llama-server":
            g = (a.gguf or "q8_0")
            names = {"q8_0": "Q8_0", "q5_k_m": "Q5_K_M", "f16": "f16"}
            gguf = resolve(g) if g.endswith(".gguf") else resolve(f"Repos/KapiTeacher-Mongo-7B/gguf/kapiteacher-mongo-7b-sft_v2d-{names[g.lower()]}.gguf")
            backend = LlamaServerBackend(gguf, resolve(tc["base"]), url=a.server_url, slots=a.slots, ctx_per_slot=a.ctx_per_slot)
            backend.skip_greedy = bool(a.retry_from)
            engine_kwargs.update(save_topk=False, max_rows=max(a.slots * 2, 8), batch_tokens=10 ** 9)
            base_model, adapter, load = gguf, None, a.gguf or "q8_0"
        else:
            backend = HFBackend(base_model, adapter, load)
        eng = Engine(backend, pool_dir, out, **engine_kwargs)
        meta = {"started": time.strftime("%FT%T"), "git": git_hash(), "config": rel(resolve(a.config) if a.config else ROOT / "configs" / "distill_v2c.yaml"),
                "pool": rel(pool_dir), "pool_manifest_sha256": sha256_file(pool_dir / "pool.manifest.json") if (pool_dir / "pool.manifest.json").exists() else None,
                "model": {"path": rel(base_model), "fingerprint": (model_fingerprint(base_model)["model_sha256"] if base_model.is_dir() else sha256_file(base_model)), "load": load, "backend": a.backend},
                "adapter": backend.adapter_info, "engine": {k: v for k, v in engine_kwargs.items()}, "libraries": library_versions(),
                "gpu": {k: backend.gpu.get(k) for k in ("device_name", "gcn_arch", "total_vram_gb", "torch", "hip")},
                "load_seconds": backend.load_seconds, "decoding": "greedy argmax (padding do vocab mascarado) + amostras T/seed por item"}
        write_json(out / "gen_meta.json", meta)
        summary = eng.run(resume=a.resume, max_blocks=limit_blocks, stop_block=limit_blocks, only_idx=only_idx)
        if a.backend == "hf":
            import torch

            summary["vram_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated(backend.device) / 2 ** 30, 2)
        else:
            backend.close()
        summary["wall_total_seconds"] = round(time.time() - t0, 1)
        write_json(out / "gen_summary.json", {**summary, "meta": meta})
        print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
