"""Paridade/latencia ONNX (Fase 01 exportabilidade). Subcomandos:
  ref   (venv ROCm)   : gera referencia HF bf16 greedy em N prompts FIM do val de data/generated/v1/fim_ctx.jsonl (somente leitura)
  run   (venv genai)  : roda um pacote ONNX (cpu|dml) com os token ids da referencia; grava ids, TTFT, tok/s, RSS
  cmp   (qualquer)    : compara run vs ref
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STOPS = ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|endoftext|>", "<|im_end|>", "<|fim_pad|>")


def cmd_ref(a):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if '"split":"val"' in l]
    step = max(1, len(rows) // a.n)
    rows = rows[::step][: a.n]
    tok = AutoTokenizer.from_pretrained(a.model)
    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype)).to(dev).eval()
    out = []
    for r in rows:
        ids = tok(r["prompt"], add_special_tokens=False)["input_ids"]
        t = time.perf_counter()
        with torch.no_grad():
            g = model.generate(torch.tensor([ids], device=dev), max_new_tokens=a.max_new, do_sample=False,
                               eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
        new = g[0, len(ids):].tolist()
        out.append({"id": r["id"], "prompt_ids": ids, "ref_ids": new, "middle": r["middle"], "sec": time.perf_counter() - t})
    Path(a.out).write_text(json.dumps({"model": a.model, "device": dev, "max_new": a.max_new, "items": out}), encoding="utf-8")
    print("saved", a.out, len(out))


def cmd_run(a):
    import numpy as np, psutil  # noqa: F401
    import onnxruntime_genai as og
    ref = json.loads(Path(a.ref).read_text(encoding="utf-8"))
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    t0 = time.perf_counter()
    cfg = og.Config(a.pkg)
    cfg.clear_providers()
    if a.provider == "cpu" and a.threads:
        cfg.overlay(json.dumps({"model": {"decoder": {"session_options": {"intra_op_num_threads": a.threads}}}}))
    if a.provider != "cpu":
        cfg.append_provider(a.provider)
    model = og.Model(cfg)
    load = time.perf_counter() - t0
    res, peak = [], proc.memory_info().rss
    for k, it in enumerate(ref["items"]):
        p = og.GeneratorParams(model)
        p.set_search_options(do_sample=False, max_length=len(it["prompt_ids"]) + ref["max_new"], batch_size=1)
        g = og.Generator(model, p)
        t = time.perf_counter()
        g.append_tokens(np.array(it["prompt_ids"], dtype=np.int32))
        g.generate_next_token()
        ttft = time.perf_counter() - t
        new = [int(g.get_next_tokens()[0])]
        t1 = time.perf_counter()
        while not g.is_done() and new[-1] != 151643:
            g.generate_next_token()
            new.append(int(g.get_next_tokens()[0]))
        dec = time.perf_counter() - t1
        peak = max(peak, proc.memory_info().rss)
        res.append({"id": it["id"], "ids": new, "ttft_ms": ttft * 1000, "dec_tps": (len(new) - 1) / dec if dec > 0 and len(new) > 1 else None,
                    "prompt_tokens": len(it["prompt_ids"])})
        del g
    out = {"pkg": a.pkg, "provider": a.provider, "genai": og.__version__ if hasattr(og, "__version__") else "", "load_s": load,
           "rss_idle_gb": rss0 / 2**30, "rss_peak_gb": peak / 2**30, "items": res}
    Path(a.out).write_text(json.dumps(out), encoding="utf-8")
    print("saved", a.out)


def cmd_cmp(a):
    import statistics as st
    ref = {i["id"]: i for i in json.loads(Path(a.ref).read_text(encoding="utf-8"))["items"]}
    run = json.loads(Path(a.run).read_text(encoding="utf-8"))
    exact = text = 0
    pref = []
    def cut(ids):  # remove EOS/pad and trailing
        return [t for t in ids if t not in (151643, 151645, 151662)][:]
    for it in run["items"]:
        r = ref[it["id"]]["ref_ids"]; o = it["ids"]
        exact += cut(r) == cut(o)
        n = 0
        for x, y in zip(cut(r), cut(o)):
            if x != y: break
            n += 1
        pref.append(n / max(1, len(cut(r))))
    n = len(run["items"])
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(a.tokenizer)
    def txt(ids):
        t = tk.decode(ids, skip_special_tokens=False)
        for st_ in STOPS:
            i = t.find(st_)
            if i >= 0: t = t[:i]
        return t
    mid = {i["id"]: i["middle"] for i in json.loads(Path(a.ref).read_text(encoding="utf-8"))["items"]}
    text_eq = sum(txt(ref[i["id"]]["ref_ids"]) == txt(i["ids"]) for i in run["items"])
    em_mid = sum(txt(i["ids"]).strip() == mid[i["id"]].strip() for i in run["items"])
    em_ref = sum(txt(ref[i["id"]]["ref_ids"]).strip() == mid[i["id"]].strip() for i in run["items"])
    ttft = sorted(i["ttft_ms"] for i in run["items"]); tps = [i["dec_tps"] for i in run["items"] if i["dec_tps"]]
    s = {"pkg": Path(run["pkg"]).name, "provider": run["provider"], "n": n, "exact_greedy_pct": round(100 * exact / n, 1),
         "text_match_pct": round(100 * text_eq / n, 1), "em_vs_middle_pct": round(100 * em_mid / n, 1), "em_vs_middle_ref_pct": round(100 * em_ref / n, 1), "mean_prefix_match_pct": round(100 * st.mean(pref), 1), "ttft_ms_mean": round(st.mean(ttft), 1),
         "ttft_ms_median": round(st.median(ttft), 1), "ttft_ms_p95": round(ttft[int(0.95 * (n - 1))], 1),
         "decode_tok_s": round(st.mean(tps), 1), "load_s": round(run["load_s"], 1), "rss_peak_gb": round(run["rss_peak_gb"], 2),
         "mean_prompt_tokens": round(st.mean(i["prompt_tokens"] for i in run["items"]))}
    print(json.dumps(s)); 
    if a.out: Path(a.out).write_text(json.dumps(s), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("ref"); r.add_argument("--model", required=True); r.add_argument("--data", default=str(ROOT / "data/generated/v1/fim_ctx.jsonl"))
    r.add_argument("--n", type=int, default=50); r.add_argument("--max-new", type=int, default=32); r.add_argument("--cpu", action="store_true"); r.add_argument("--dtype", default="bfloat16"); r.add_argument("--out", required=True)
    u = sub.add_parser("run"); u.add_argument("--pkg", required=True); u.add_argument("--provider", default="cpu"); u.add_argument("--threads", type=int, default=4); u.add_argument("--ref", required=True); u.add_argument("--out", required=True)
    c = sub.add_parser("cmp"); c.add_argument("--ref", required=True); c.add_argument("--run", required=True); c.add_argument("--tokenizer", default=str(ROOT / "BaseRepos/Qwen2.5-Coder-0.5B")); c.add_argument("--out")
    a = ap.parse_args(); {"ref": cmd_ref, "run": cmd_run, "cmp": cmd_cmp}[a.cmd](a)

if __name__ == "__main__":
    main()
