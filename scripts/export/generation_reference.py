#!/usr/bin/env python3
"""Referencia independente dos tokens de saida (GenAI Python). CPU, prompts sinteticos, sem rede ou corpus hom.

Consumida por RealModelPrefixCacheTests na IDE; nao mede qualidade nem latencia.
"""
import argparse
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ort-library", help="DLL ORT da IDE para comparar o MESMO runtime; nao altera o ambiente instalado")
    ap.add_argument("--prepared-tier", choices=["t0", "t1"], help="500 prompts val ja preparados em reports/export/work; nao le corpus hom")
    args = ap.parse_args()
    import ctypes
    import hashlib
    ort_handle = ctypes.WinDLL(str(Path(args.ort_library).resolve())) if args.ort_library else None
    import numpy as np
    import onnxruntime_genai as og
    cfg = og.Config(args.package)
    cfg.clear_providers()
    cfg.overlay(json.dumps({"model":{"decoder":{"session_options":{"intra_op_num_threads":0,"inter_op_num_threads":1,"log_severity_level":4}}}}))
    model = og.Model(cfg)
    tok = og.Tokenizer(model)
    markers = ["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|endoftext|>", "<|im_end|>", "<|fim_pad|>"]
    stops = {int(tok.encode(s)[0]) for s in markers if len(tok.encode(s)) == 1}
    rows = []
    cases = [("db.orders.find({", 96), ("db.orders.find({ status: ", 96), ("const total = ", 96),
                          ("// Consulta de clientes em Sao Paulo\ndb.clientes.find({", 96), ("db.orders.find({", 1), ("", 96),
                          ("/* " + "Collection orders: status string, customerId int, amount decimal. " * 20 + "*/\n" + "db.orders.find({", 96),
                          ("/* " + "Collection orders: status string, customerId int, amount decimal. " * 20 + "*/\n" + "db.orders.find({ status: ", 96)]
    source = None
    if args.prepared_tier:
        source = Path(__file__).resolve().parents[2] / "reports" / "export" / "work" / f"{args.prepared_tier}_prompts.json"
        prepared = json.loads(source.read_text(encoding="utf-8"))
        cases = [(row["prompt_ids"], int(prepared["max_new"])) for row in prepared["items"]]
    for text, maximum in cases:
        prompt = list(map(int, text if isinstance(text, list) else tok.encode("<|fim_prefix|>" + text + "<|fim_suffix|><|fim_middle|>")))
        params = og.GeneratorParams(model)
        params.set_search_options(do_sample=False, max_length=len(prompt)+maximum)
        g = og.Generator(model, params)
        g.append_tokens(np.array(prompt, dtype=np.int32))
        output = []
        eos_without_growth = False
        for _ in range(maximum):
            if g.is_done(): break
            before = len(g.get_sequence(0))
            g.generate_next_token()
            token = int(g.get_next_tokens()[0])
            if len(g.get_sequence(0)) == before:
                eos_without_growth = True
            if token in stops: break
            output.append(token)
        rows.append({"promptIds":prompt,"outputIds":output,"maximumTokens":maximum,"eosWithoutGrowth":eos_without_growth})
        del g
    result = {"schema":"kapilab-generation-reference-v1", "genaiVersion":og.__version__, "provider":"cpu", "preparedSourceSha256":hashlib.sha256(source.read_bytes()).hexdigest() if source else None, "items":rows, "ortDllSha256":hashlib.sha256(Path(args.ort_library).read_bytes()).hexdigest() if args.ort_library else None}
    Path(args.out).write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"n":len(rows), "eosWithoutGrowth":sum(r["eosWithoutGrowth"] for r in rows)}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
