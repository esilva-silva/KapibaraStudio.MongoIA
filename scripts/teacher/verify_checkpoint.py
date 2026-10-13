#!/usr/bin/env python3
"""G2.1 — verificacao do checkpoint do professor (fase-02-professor.md §3.1, §3.6).

  python -I scripts/teacher/verify_checkpoint.py [--model BaseRepos/Qwen2.5-Coder-7B] [--no-hf] [--no-gpu]

Verifica: SHA-256 dos arquivos x base-repos.lock.json x API do Hugging Face (revisao fixada; oid LFS sha256 e blob sha1 git),
config.json x tabela do §2, IDs FIM/eos e tokenizer compartilhado com os students, mascara do padding do vocab, FIM sanity
(20 prompts fixos, sem marcadores vazando) com o checkpoint em NF4 (como no bake-off). A paridade de IDs Python x C#
(QwenFimPromptBuilder + tokenizer .NET da IDE) NAO e possivel aqui: fica `pending_ide`, e os IDs esperados do lado Python (400 prompts)
sao gravados para o ContractProbe comparar depois.
Saida: Repos/KapiTeacher-Mongo-7B/reports/checkpoint_verification.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (BASE_REPOS, ROOT, TEACHER_REPO, Generator, gpu_lock, library_versions, load_hf, lock_entry,  # noqa: E402
                     read_jsonl, sha256_file, write_json)

EXPECTED_CONFIG = {  # tabela do §2 (7B alternativo -> escolhido)
    "architectures": ["Qwen2ForCausalLM"], "num_hidden_layers": 28, "hidden_size": 3584, "num_attention_heads": 28,
    "num_key_value_heads": 4, "vocab_size": 152064, "max_position_embeddings": 32768, "rope_theta": 1000000.0,
    "tie_word_embeddings": False,
}
EXPECTED_IDS = {"<|fim_prefix|>": 151659, "<|fim_middle|>": 151660, "<|fim_suffix|>": 151661, "<|fim_pad|>": 151662, "<|endoftext|>": 151643}
REAL_VOCAB_EXPECTED = 151665  # ~151.665 tokens reais

SANITY_PROMPTS = [  # (prefix, suffix) — chave dentro de $group, valor de $match, find({ ... }), etc.
    ("db.Users.find({", "})"),
    ("db.Orders.aggregate([ { $group: { _id: \"$status\", ", " } } ])"),
    ("db.Orders.aggregate([ { $match: { status: ", " } } ])"),
    ("db.Products.find({ price: { $gt: ", " } }).sort({ price: 1 })"),
    ("db.Customers.aggregate([\n  { $match: { active: true } },\n  { $group: { _id: \"$country\", total: { $", " } } }\n])"),
    ("db.Invoices.find({ createdAt: { $gte: ", " } })"),
    ("db.Sessions.countDocuments({ userId: ObjectId(\"", "\") })"),
    ("db.Logs.aggregate([ { $sort: { ", " } }, { $limit: 10 } ])"),
    ("db.Employees.find({ department: \"hr\" }, { ", " })"),
    ("db.Devices.aggregate([ { $unwind: \"$", "\" } ])"),
    ("db.Accounts.updateOne({ _id: ", " }, { $set: { balance: 0 } })"),
    ("db.Tickets.find({ $or: [ { status: \"open\" }, { ", " } ] })"),
    ("db.Orders.aggregate([ { $lookup: { from: \"Customers\", localField: \"", "\", foreignField: \"_id\", as: \"customer\" } } ])"),
    ("db.Metrics.distinct(\"", "\")"),
    ("const rows = db.Orders.find({ total: { $", " } }).toArray();"),
    ("db.Reviews.aggregate([\r\n  { $group: { _id: \"$productId\", avg: { $avg: \"$", "\" } } }\r\n])"),
    ("db.Posts.find({ tags: { $in: [", "] } })"),
    ("db.Shipments.aggregate([ { $project: { _id: 0, ", ": 1 } } ])"),
    ("db.Patients.find({ ", ": { $exists: true } })"),
    ("db.Readings.aggregate([ { $match: { ts: { $gte: ISODate(\"", "\") } } } ])"),
]


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()  # noqa: S324 (identificador git, nao seguranca)


def fetch_hf_tree(repo_id: str, revision: str, timeout: float = 30) -> list[dict]:
    url = f"https://huggingface.co/api/models/{repo_id}/tree/{revision}"
    with urllib.request.urlopen(url, timeout=timeout) as response:  # leitura de metadados publicos
        return json.loads(response.read().decode("utf-8"))


def check_integrity(model_dir: Path, use_hf: bool) -> dict:
    entry = lock_entry(model_dir)
    result: dict = {"lock_present": entry is not None, "files": {}, "status": "pass"}
    if entry is None:
        result["status"] = "fail"
        result["error"] = "modelo ausente de base-repos.lock.json"
        return result
    result["repo_id"], result["revision"] = entry["repo_id"], entry["revision"]
    hf: dict[str, dict] = {}
    if use_hf:
        try:
            hf = {item["path"]: item for item in fetch_hf_tree(entry["repo_id"], entry["revision"])}
            result["hf_api"] = {"available": True, "revision_queried": entry["revision"], "files_listed": len(hf)}
        except Exception as exc:  # noqa: BLE001
            result["hf_api"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    for item in entry["files"]:
        path = model_dir / item["path"]
        row: dict = {"size": path.stat().st_size if path.exists() else None, "lock_sha256": item["sha256"]}
        if not path.exists():
            row.update(status="fail", error="arquivo ausente")
        else:
            started = time.time()
            row["sha256"] = sha256_file(path)
            row["hash_seconds"] = round(time.time() - started, 1)
            row["matches_lock"] = row["sha256"] == item["sha256"] and row["size"] == item["size"]
            remote = hf.get(item["path"])
            if remote is None:
                row["matches_hf"] = None
            elif remote.get("lfs"):
                row["hf_lfs_sha256"] = remote["lfs"]["oid"]
                row["matches_hf"] = remote["lfs"]["oid"] == row["sha256"] and remote["lfs"]["size"] == row["size"]
            else:
                row["hf_git_blob_sha1"] = remote["oid"]
                row["matches_hf"] = remote["oid"] == git_blob_sha1(path)
            row["status"] = "pass" if row["matches_lock"] and row["matches_hf"] is not False else "fail"
        result["files"][item["path"]] = row
        if row["status"] != "pass":
            result["status"] = "fail"
    on_disk = {p.name for p in model_dir.iterdir() if p.is_file()}
    result["unlisted_files_on_disk"] = sorted(on_disk - {f["path"] for f in entry["files"]})
    result["pickle_or_bin_files"] = sorted(n for n in on_disk if n.endswith((".bin", ".pt", ".pth", ".pkl", ".ckpt")))
    if result["pickle_or_bin_files"]:
        result["status"] = "fail"
    result["hf_compared"] = sum(1 for r in result["files"].values() if r.get("matches_hf") is not None)
    result["revision_matches_teacher_choice"] = None
    choice = ROOT / "reports" / "teacher_choice.json"
    if choice.exists():
        pinned = json.loads(choice.read_text(encoding="utf-8"))["teacher"]["revision"]
        result["revision_matches_teacher_choice"] = pinned == entry["revision"]
    return result


def check_config(model_dir: Path) -> dict:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    rows = {k: {"expected": v, "actual": config.get(k), "ok": config.get(k) == v} for k, v in EXPECTED_CONFIG.items()}
    rows["eos_token_id"] = {"expected": 151643, "actual": config.get("eos_token_id"), "ok": config.get("eos_token_id") == 151643}
    return {"status": "pass" if all(r["ok"] for r in rows.values()) else "fail", "fields": rows,
            "extras": {k: config.get(k) for k in ("torch_dtype", "intermediate_size", "use_sliding_window", "sliding_window", "bos_token_id")}}


def check_tokenizer(model_dir: Path) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    ids = {}
    ok = True
    for text, expected in EXPECTED_IDS.items():
        encoded = tok.encode(text, add_special_tokens=False)
        actual = tok.convert_tokens_to_ids(text)
        single = encoded == [actual]
        ids[text] = {"expected": expected, "actual": actual, "encodes_to_single_token": single, "ok": actual == expected and single}
        ok &= ids[text]["ok"]
    tokenizer_hash = sha256_file(model_dir / "tokenizer.json")
    shared = {}
    for name in ("Qwen2.5-Coder-0.5B", "Qwen2.5-Coder-1.5B"):
        other = lock_entry(BASE_REPOS / name)
        value = next((f["sha256"] for f in other["files"] if f["path"] == "tokenizer.json"), None) if other else None
        shared[name] = {"tokenizer_json_sha256": value, "same_as_teacher": value == tokenizer_hash}
    ok &= all(v["same_as_teacher"] for v in shared.values())
    crlf = "db.a.find({\r\n  x: 1\r\n})\n"
    roundtrip = tok.decode(tok.encode(crlf, add_special_tokens=False), skip_special_tokens=False) == crlf
    ok &= roundtrip
    return {"status": "pass" if ok else "fail", "special_ids": ids, "len_tokenizer": len(tok), "real_vocab_expected": REAL_VOCAB_EXPECTED,
            "real_vocab_matches": len(tok) == REAL_VOCAB_EXPECTED, "tokenizer_json_sha256": tokenizer_hash, "shared_with_students": shared,
            "crlf_roundtrip_ok": roundtrip, "eos_token": tok.eos_token, "ide_tokenizer_note": "tokenizer .NET da IDE nao verificado aqui (pending_ide)"}, tok


def python_parity_ids(tokenizer, count: int = 400) -> dict:
    """IDs esperados do lado Python (QwenFim: PSM sem BOS, contexto 2048, orcamento -3, sufixo <= 25%) para o ContractProbe."""
    from compact.contracts import PREFIX_WINDOW, SUFFIX_WINDOW, QwenFim

    contract = QwenFim(tokenizer)
    rows = [r for r in read_jsonl(ROOT / "data" / "generated" / "v1" / "fim.jsonl") if r["split"] in ("val", "test")]
    rows = sorted(rows, key=lambda r: r["split_key"])[:count]
    out = ROOT / "Repos" / "KapiTeacher-Mongo-7B" / "reports" / "parity_python_ids.jsonl"
    digest = hashlib.sha256()
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            prefix, suffix = row["prefix"][-PREFIX_WINDOW:], row["suffix"][:SUFFIX_WINDOW]
            ids = contract.build(prefix, suffix, 2048)
            line = json.dumps({"id": row["id"], "prefix": prefix, "suffix": suffix, "context_tokens": 2048, "ids": ids}, ensure_ascii=False)
            handle.write(line + "\n")
            digest.update(line.encode("utf-8"))
    return {"file": str(out.relative_to(ROOT)), "prompts": len(rows), "sha256": digest.hexdigest()}


def gpu_checks(model_dir: Path, tokenizer, real_vocab: int) -> dict:
    import torch
    from compact.contracts import QwenFim

    contract = QwenFim(tokenizer)
    model, _, device, gpu, load_seconds = load_hf(model_dir, "nf4")
    gen = Generator(model, tokenizer, device)
    fim_ids = {contract.prefix_id, contract.suffix_id, contract.middle_id, EXPECTED_IDS["<|fim_pad|>"]}

    # -- padding do vocab
    head = model.lm_head.weight  # nao quantizado (bf16)
    rows = head.shape[0]
    real_rows = head[:real_vocab].float()
    pad_rows = head[real_vocab:].float()
    padding = {"lm_head_rows": rows, "real_vocab": real_vocab, "padding_rows": rows - real_vocab,
               "real_row_norm_mean": round(float(real_rows.norm(dim=1).mean()), 4),
               "padding_row_norm_mean": round(float(pad_rows.norm(dim=1).mean()), 4) if rows > real_vocab else None,
               "lm_head_dtype": str(head.dtype), "mask_applied_in_generation": f"logits[..., :{real_vocab}] antes do argmax"}

    # -- FIM sanity
    results, leaks, padded_argmax = [], 0, 0
    for index, (prefix, suffix) in enumerate(SANITY_PROMPTS):
        ids = contract.build(prefix, suffix, 2048)
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([ids], device=device), logits_to_keep=1).logits[0, -1]
        unmasked = int(logits.argmax())
        padded_argmax += int(unmasked >= real_vocab)
        out = gen.generate(ids, 64, contract.stops, suffix)
        tokens = out["tokens"]
        text = gen.decode([t for t in tokens if t not in contract.stops])
        leaked = any(t in fim_ids for t in tokens) or "<|fim_" in text
        leaks += int(leaked)
        results.append({"index": index, "prefix": prefix, "suffix": suffix, "generated": text, "n_tokens": len(tokens), "stop": out["stop"],
                        "last_token_id": tokens[-1], "marker_leak": leaked, "padding_id_generated": any(t >= real_vocab for t in tokens)})
    sanity = {"status": "pass" if leaks == 0 else "fail", "prompts": len(SANITY_PROMPTS), "without_markers": len(SANITY_PROMPTS) - leaks,
              "stopped_at_eos_or_suffix_echo": sum(1 for r in results if r["stop"] in ("eos", "suffix")),
              "unmasked_argmax_in_padding_at_prefill": padded_argmax, "results": results}
    padding["status"] = "pass" if rows == 152064 and real_vocab == REAL_VOCAB_EXPECTED and not any(r["padding_id_generated"] for r in results) else "fail"
    memory = {"vram_max_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2), "load_seconds": load_seconds,
              "load": "nf4", "gpu": gpu.get("device_name")}
    del model
    torch.cuda.empty_cache()
    return {"fim_sanity": sanity, "padding_mask": padding, "gpu": memory}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=str(BASE_REPOS / "Qwen2.5-Coder-7B"))
    parser.add_argument("--out", default=str(TEACHER_REPO / "reports" / "checkpoint_verification.json"))
    parser.add_argument("--no-hf", action="store_true", help="nao consultar a API do Hugging Face")
    parser.add_argument("--no-gpu", action="store_true", help="pula FIM sanity e padding (gate fica incompleto)")
    args = parser.parse_args()
    model_dir = Path(args.model).resolve()
    started = time.time()
    report: dict = {"gate": "G2.1", "spec": "docs/fase-02-professor.md §3.1/§3.6", "model_dir": str(model_dir),
                    "date": time.strftime("%F"), "libraries": library_versions()}
    report["integrity"] = check_integrity(model_dir, not args.no_hf)
    report["config"] = check_config(model_dir)
    report["tokenizer"], tokenizer = check_tokenizer(model_dir)
    report["python_ids_for_parity"] = python_parity_ids(tokenizer)
    report["python_csharp_parity"] = {
        "status": "pending_ide",
        "reason": ("Paridade de IDs Python x C# (QwenFimPromptBuilder + OnnxModelTokenizer da IDE) exige executar o codigo .NET da IDE "
                   "(tools/ContractProbe com referencia as DLLs da IDE em F:/source/esilva-silva/EsilvaSoft.KapibaraStudio), que nao esta "
                   "disponivel/foi executado nesta etapa. Nada foi comparado: nenhuma aprovacao e afirmada."),
        "prepared": "IDs esperados do builder Python (400 prompts de val/test) em Repos/KapiTeacher-Mongo-7B/reports/parity_python_ids.jsonl; "
                    "o ContractProbe deve produzir os mesmos IDs e exigir 100% de igualdade (G2.1).",
        "prompts": 400}
    if not args.no_gpu:
        with gpu_lock("verify_checkpoint"):
            report.update(gpu_checks(model_dir, tokenizer, len(tokenizer)))
    else:
        for key in ("fim_sanity", "padding_mask"):
            report[key] = {"status": "not_run", "reason": "--no-gpu"}
    report["window_perplexity"] = {"status": "not_run", "reason": "item opcional do §3.1 (2k/8k/32k); fora do escopo desta etapa e do criterio G2.1"}
    parts = {
        "integrity": report["integrity"]["status"], "config": report["config"]["status"], "tokenizer_ids": report["tokenizer"]["status"],
        "fim_sanity": report["fim_sanity"]["status"], "padding_mask": report["padding_mask"]["status"],
        "python_csharp_parity": report["python_csharp_parity"]["status"],
    }
    failed = [k for k, v in parts.items() if v == "fail"]
    pending = [k for k, v in parts.items() if v in ("pending_ide", "not_run")]
    report["g2_1"] = {"components": parts, "status": "fail" if failed else ("partial_pending_ide" if pending else "pass"),
                      "failed": failed, "pending": pending, "elapsed_seconds": round(time.time() - started, 1)}
    write_json(Path(args.out), report)
    print(json.dumps(report["g2_1"], indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
