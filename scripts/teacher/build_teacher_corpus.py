#!/usr/bin/env python3
"""Corpus de SFT do KapiTeacher-Mongo-7B a partir de data/validated/v2c (docs/fase-02-professor.md §3.3 B; Fase 05 §6.2).

  C:/SlopStudio.MongoAI-artifacts/.venv/Scripts/python.exe -I scripts/teacher/build_teacher_corpus.py \
      --config Repos/KapiTeacher-Mongo-7B/configs/corpus_v2c.yaml [--target-examples N] [--workers 12]

Entrada : somente registros `split == train` (e `val` para o mini-conjunto de validacao do treino). test/bench/blind nunca sao
          usados; as guardas de G2.7 estao em `check_isolation` e `verify_output` (e o relatorio as repete).
Saida   : data/teacher/v2c/{train-0000.npz, train-0000.meta.jsonl, ..., val/val_loss.npz, val/val_gen.jsonl, manifest.json}
          e Repos/KapiTeacher-Mongo-7B/reports/corpus_manifest.md.
Formato : cada shard .npz traz `ids` (uint32, exemplos concatenados), `ex_offsets` (int64, N+1), `loss_start` (int32, indice
          dentro do exemplo onde a perda comeca = primeiro token do alvo), `row_ptr` (int32, R+1: exemplos [row_ptr[r],row_ptr[r+1])
          formam a linha r, ja ordenados por comprimento). Uma LINHA e um micro-batch (n x Lmax <= micro_tokens, padding a direita,
          atencao causal sem mascara -> kernel flash). Nenhum exemplo e concatenado a outro: nao ha contaminacao entre exemplos.
Perda   : so no alvo (middle / middle+sufixo / output) + <|endoftext|>; prompt mascarado.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import multiprocessing as mp
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "data"))
import blind_guard  # noqa: E402
import render_contract as rc  # noqa: E402
import replay_cuts  # noqa: E402

FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE = rc.QWEN_FIM
EOS_TEXT = "<|endoftext|>"
EXPECTED_IDS = {"<|fim_prefix|>": 151659, "<|fim_middle|>": 151660, "<|fim_suffix|>": 151661, "<|endoftext|>": 151643}
CHAT_HEAD = "/* Answer the instruction with code. The JSON below is data, not executable code.\n"
CHAT_TAIL = "\nReturn only code, without Markdown or explanation. */\n"
CHAT_KINDS = ("chat", "chat_multiturn", "explain", "correction", "negative")
SHARD_RE = re.compile(r"^(fim-\d{5}|chat)\.jsonl$")


# -- utilidades -------------------------------------------------------------------------------------------------------------------
def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def unit(seed: int, salt: str, ident: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{salt}:{ident}".encode()).digest()[:8], "big")


def content_hash(rec: dict) -> int:
    if rec["type"] == "fim":
        text = f"{rec['prefix']}\x00{rec['suffix']}\x00{rec['middle']}"
    else:
        ctx = rec.get("context") or {}
        text = f"{rec['instruction']}\x00{ctx.get('editor') or ''}\x00{rec['output']}"
    return int.from_bytes(hashlib.sha1(text.encode("utf-8", "surrogatepass")).digest()[:8], "big")


_ESC = re.compile(r"[^ -~]|[\"&'+<>`\\]")
_SHORT = {"\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f", "\\": "\\\\"}


def dotnet_json_string(text: str) -> str:
    """System.Text.Json (JavaScriptEncoder.Default): nao-ASCII e `\"&'+<>`` viram \\uXXXX (hex maiusculo)."""
    def esc(match: re.Match) -> str:
        ch = match.group(0)
        if ch in _SHORT:
            return _SHORT[ch]
        o = ord(ch)
        if o > 0xFFFF:
            o -= 0x10000
            return "\\u%04X\\u%04X" % (0xD800 + (o >> 10), 0xDC00 + (o & 0x3FF))
        return "\\u%04X" % o
    return '"' + _ESC.sub(esc, text) + '"'


def plain_json_string(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def render_chat_prompt(rec: dict, cfg: dict) -> str:
    """teacher-chat-v1: FIM Qwen com sufixo vazio; prefixo `local-agent-fim-v0` (instruction + context em JSON compacto)."""
    ctx = rec.get("context") or {}
    lines = [f"Conexão (ID lógico): {ctx.get('connection') or ''}", f"Banco: {ctx.get('database') or ''}",
             f"Coleção: {ctx.get('collection') or ''}"]
    if cfg["chat"].get("include_known_collections") and ctx.get("known_collections"):
        lines.append("Coleções disponíveis: " + ", ".join(ctx["known_collections"]))
    blocks = []
    if ctx.get("selection"):
        blocks.append("Seleção do editor:\n" + ctx["selection"])
    if ctx.get("editor"):
        blocks.append("Conteúdo do editor:\n" + ctx["editor"])
    context = "\n".join(lines) + ("\n\n" + "\n".join(blocks) if blocks else "")
    context = context[: cfg["chat"].get("context_max_chars", 8192)]
    enc = dotnet_json_string if cfg["chat"].get("json_escaping", "dotnet") == "dotnet" else plain_json_string
    body = '{"instruction":%s,"context":%s}' % (enc(rec["instruction"]), enc(context))
    return f"{FIM_PREFIX}{CHAT_HEAD}{body}{CHAT_TAIL}{FIM_SUFFIX}{FIM_MIDDLE}"


def render_fim(rec: dict, component: str) -> tuple[str, str]:
    """(prompt, alvo) de um registro FIM v2c. O alvo NAO inclui <|endoftext|> (acrescentado na tokenizacao)."""
    prompt = rec["prompt"]
    if component == "fim_ctx":
        return prompt, rec["middle"]
    if component == "completion":  # mesmo contexto/prefixo; sufixo vazio; alvo = resto do documento a partir do cursor
        head, sep, _tail = prompt.partition(FIM_SUFFIX)
        assert sep, "prompt sem <|fim_suffix|>"
        return head + FIM_SUFFIX + FIM_MIDDLE, rec["middle"] + rec["suffix"]
    if component == "fim_raw":
        before, after = rc.bound_window(rec["prefix"], rec["suffix"])
        return rc.qwen_fim_text(before, after), rec["middle"]
    raise ValueError(component)


class _TokCounter:
    """Interface `.count` (replay_cuts.make_cut) sobre um `tokenizers.Tokenizer` ja carregado."""

    def __init__(self, tok) -> None:
        self._tok = tok

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False).ids) if text else 0


def chunk_replay_doc(text: str, window_chars: int, min_chars: int = 300) -> list[str]:
    """Janelas consecutivas alinhadas por linha (<= window_chars; uma linha maior que a janela fica sozinha)."""
    chunks, cur, size = [], [], 0
    for line in text.splitlines(keepends=True):
        if cur and size + len(line) > window_chars:
            chunks.append("".join(cur))
            cur, size = [], 0
        cur.append(line)
        size += len(line)
    if cur:
        tail = "".join(cur)
        if len(tail) >= min_chars or not chunks:
            chunks.append(tail)
        else:
            chunks[-1] += tail
    return chunks


def render_replay_chunk(text: str, ident: str, cfg: dict, seed: int, tc) -> tuple[str, str, str]:
    """-> (kind, prompt, target). `fim`: PSM cru com corte sintatico (middle <= 96 tokens; gen_fim.candidates/pick_end);
    `causal`: trecho em sequencia, perda em todos os tokens."""
    r = cfg["replay"]
    if unit(seed, "replay-kind", ident) / 2**64 < float(r.get("fim_fraction", 0.5)):
        rng = random.Random(unit(seed, "replay-cut", ident))
        cut = replay_cuts.make_cut(text, rng, tc, max_tokens=int(r.get("fim_max_middle_tokens", 96)), min_prefix_chars=60)
        if cut is None:  # lexer nao entendeu o trecho (fragmento/TS): corte por linha, do meio da linha ate o fim dela
            lines = text.split(chr(10))
            idx = [i for i, ln in enumerate(lines) if 12 <= len(ln.strip()) <= 240 and 0 < i < len(lines) - 1]
            if idx:
                i = idx[rng.randrange(len(idx))]
                ln = lines[i]
                col = len(ln) - len(ln.lstrip()) + rng.randrange(0, max(1, len(ln.strip()) // 2))
                start = sum(len(x) + 1 for x in lines[:i]) + col
                end = start + (len(ln) - col)
                if tc.count(text[start:end]) <= int(r.get("fim_max_middle_tokens", 96)):
                    cut = {"prefix": text[:start], "middle": text[start:end], "suffix": text[end:]}
        if cut is not None and cut["middle"].strip():
            return "fim", f"{FIM_PREFIX}{cut['prefix']}{FIM_SUFFIX}{cut['suffix']}{FIM_MIDDLE}", cut["middle"]
    return "causal", "", text


# -- fase 1: indice (somente leitura) ---------------------------------------------------------------------------------------------
def _scan_shard(path: str) -> dict:
    out = {"path": path, "train": [], "eval_ids": {}, "eval_content": {}, "groups": defaultdict(set), "schemas": defaultdict(set),
           "counts": Counter(), "violations": []}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            split = rec["split"]
            out["counts"][f"{split}:{rec['type']}"] += 1
            out["groups"][split].add(rec.get("group_id") or rec.get("group"))
            out["schemas"][split].add(rec.get("schema_id"))
            if split == "train":
                role = rec.get("split_role") or rec.get("schema_split_role")
                if role != "train" or rec.get("holdout"):
                    out["violations"].append((rec["id"], f"train com role={role} holdout={rec.get('holdout')}"))
                eligible = False
                if rec["type"] == "fim":
                    eligible = len(rec["middle"]) + len(rec["suffix"]) <= CFG["completion"]["max_chars"]
                out["train"].append((rec["id"], rec["type"], rec.get("kind") if rec["type"] == "chat" else rec["cut_kind"],
                                     eligible, rec.get("context_kind"), rec.get("category") or ""))
            else:
                out["eval_ids"][rec["id"]] = split
                out["eval_content"][content_hash(rec)] = split
                if split == "val":
                    out["val"] = out.get("val", [])
                    out["val"].append((rec["id"], rec["type"], rec.get("kind") if rec["type"] == "chat" else rec["cut_kind"]))
    out["groups"] = {k: sorted(v) for k, v in out["groups"].items()}
    out["schemas"] = {k: sorted(v) for k, v in out["schemas"].items()}
    return out


# -- fase 2: renderizacao + tokenizacao -------------------------------------------------------------------------------------------
_TOK = None


def _init_worker(cfg: dict, tok_path: str, blind_prompts: frozenset = frozenset()) -> None:
    global CFG, _TOK, BLIND_PROMPTS
    CFG = cfg
    BLIND_PROMPTS = blind_prompts
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from tokenizers import Tokenizer
    _TOK = Tokenizer.from_file(tok_path)


def _render_shard(args: tuple) -> dict:
    path, selected, eval_content, seed = args
    cfg, tok = CFG, _TOK
    items, dropped = [], Counter()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            # filtro barato antes do parse completo: o id esta no inicio da linha
            ident = line[7: line.index('"', 7)]
            if ident not in selected:
                continue
            rec = json.loads(line)
            comp, is_val = selected[ident]
            if rec["type"] == "fim":
                prompt, target = render_fim(rec, comp)
            else:
                prompt, target = render_chat_prompt(rec, cfg), rec["output"]
            if not is_val:
                if content_hash(rec) in eval_content:  # defesa em profundidade (G2.7): conteudo identico a val/test/bench
                    dropped["content_in_eval"] += 1
                    continue
                if blind_guard.prompt_sha256(prompt + "\x00" + target) in BLIND_PROMPTS:
                    dropped["blind"] += 1
                    continue
            items.append({"id": ident, "component": comp, "val": is_val, "prompt": prompt, "target": target, "rec": rec})
    if not items:
        return {"path": path, "items": [], "dropped": dropped}
    enc_p = tok.encode_batch([i["prompt"] for i in items], add_special_tokens=False) if any(i["prompt"] for i in items) else None
    enc_t = tok.encode_batch([i["target"] for i in items], add_special_tokens=False)
    eos = EXPECTED_IDS["<|endoftext|>"]
    out = []
    for n, item in enumerate(items):
        p_ids = enc_p[n].ids if item["prompt"] else []
        t_ids = enc_t[n].ids + [eos]
        if len(p_ids) + len(t_ids) > cfg["seq_len"]:
            dropped["too_long"] += 1
            continue
        if item["component"] == "completion" and len(t_ids) - 1 > 96:
            dropped["completion_gt_96"] += 1
            continue
        rec = item["rec"]
        meta = {"id": item["id"], "component": item["component"], "kind": rec.get("cut_kind") or rec.get("kind"),
                "domain": rec.get("domain"), "language": rec.get("language"), "category": rec.get("category"),
                "n_prompt": len(p_ids), "n_target": len(t_ids), "src": Path(path).name}
        row = {"meta": meta, "ids": p_ids + t_ids, "loss_start": len(p_ids)}
        if item["val"] and rec["type"] == "fim":
            row["gen"] = {"id": item["id"], "prompt": item["prompt"], "target": rec["middle"], "prefix": rec["prefix"],
                          "suffix": rec["suffix"], "dialect": rec["dialect"], "cut_kind": rec["cut_kind"], "domain": rec["domain"],
                          "schema_id": rec["schema_id"], "collection": rec["collection"]}
        out.append(row)
    return {"path": path, "items": out, "dropped": dropped}


# -- empacotamento em linhas (micro-batches) --------------------------------------------------------------------------------------
def make_rows(lengths: list[int], micro_tokens: int, max_per_row: int, rng: random.Random) -> list[list[int]]:
    """Agrupa indices por comprimento: ordena desc.; a linha comeca no mais longo e recebe vizinhos enquanto n x Lmax <= micro_tokens."""
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    rows, cur, cur_len = [], [], 0
    for i in order:
        if cur and ((len(cur) + 1) * cur_len > micro_tokens or len(cur) >= max_per_row):
            rows.append(cur)
            cur, cur_len = [], 0
        if not cur:
            cur_len = lengths[i]
        cur.append(i)
    if cur:
        rows.append(cur)
    rng.shuffle(rows)
    return rows


def write_shard(path: Path, examples: list[dict], micro_tokens: int, max_per_row: int, seed: int) -> dict:
    rng = random.Random(f"{seed}-rows-{path.stem}")
    lengths = [len(e["ids"]) for e in examples]
    rows = make_rows(lengths, micro_tokens, max_per_row, rng)
    ordered = [examples[i] for row in rows for i in row]
    row_ptr = np.zeros(len(rows) + 1, dtype=np.int32)
    row_ptr[1:] = np.cumsum([len(r) for r in rows])
    ex_offsets = np.zeros(len(ordered) + 1, dtype=np.int64)
    ex_offsets[1:] = np.cumsum([len(e["ids"]) for e in ordered])
    ids = np.fromiter((t for e in ordered for t in e["ids"]), dtype=np.uint32, count=int(ex_offsets[-1]))
    loss_start = np.array([e["loss_start"] for e in ordered], dtype=np.int32)
    np.savez(path, ids=ids, ex_offsets=ex_offsets, loss_start=loss_start, row_ptr=row_ptr)
    meta_path = path.with_suffix(".meta.jsonl")
    with meta_path.open("w", encoding="utf-8", newline="\n") as handle:
        for e in ordered:
            handle.write(json.dumps(e["meta"], ensure_ascii=False) + "\n")
    padded = sum(max(lengths[i] for i in row) * len(row) for row in rows)
    return {"file": path.name, "sha256": sha256_file(path), "meta_file": meta_path.name, "meta_sha256": sha256_file(meta_path),
            "examples": len(ordered), "rows": len(rows), "tokens": int(ex_offsets[-1]),
            "target_tokens": int(sum(e["meta"]["n_target"] for e in ordered)), "padded_tokens": int(padded),
            "pad_fraction": round(1 - ex_offsets[-1] / padded, 4) if padded else 0.0}


# -- verificacao independente (G2.7) ------------------------------------------------------------------------------------------------
def read_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def verify_output(out_dir: Path, splits_dir: Path) -> dict:
    train, val, test, bench = (read_ids(splits_dir / f"{n}.ids.txt") for n in ("train", "val", "test", "bench"))
    seen, comp, issues = set(), Counter(), []
    replay_n = 0
    for meta_path in sorted(out_dir.glob("train-*.meta.jsonl")):
        with meta_path.open(encoding="utf-8") as handle:
            for line in handle:
                m = json.loads(line)
                i = m["id"]
                comp[m["component"]] += 1
                if m["component"].startswith("replay"):  # replay: ids `rpl-...` (fora dos splits do Mongo); isolamento no bloco `replay`
                    replay_n += 1
                    if not i.startswith("rpl-") or m.get("src") != "replay":
                        issues.append((i, "replay com id/origem inesperados"))
                    if i in train or i in val or i in test or i in bench:
                        issues.append((i, "id de replay colide com id de split do Mongo"))
                    if i in seen:
                        issues.append((i, "id duplicado no corpus"))
                    seen.add(i)
                    continue
                if i not in train:
                    issues.append((i, "nao esta em train.ids.txt"))
                for name, ids in (("val", val), ("test", test), ("bench", bench)):
                    if i in ids:
                        issues.append((i, f"presente em {name}.ids.txt"))
                if i in seen:
                    issues.append((i, "id duplicado no corpus"))
                seen.add(i)
    return {"examples": len(seen), "replay_examples": replay_n, "by_component": dict(comp), "issues": issues[:20], "n_issues": len(issues),
            "in_train_ids": len(seen & train), "in_val_ids": len(seen & val), "in_test_ids": len(seen & test), "in_bench_ids": len(seen & bench)}


# -- principal ----------------------------------------------------------------------------------------------------------------------
def pick_lowest(pool: list[tuple], n: int, seed: int, salt: str) -> list[tuple]:
    return heapq.nsmallest(n, pool, key=lambda t: unit(seed, salt, t[0]))


def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def git_hash() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "Repos" / "KapiTeacher-Mongo-7B" / "configs" / "corpus_v2c.yaml"))
    ap.add_argument("--target-examples", type=int)
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--only-shards", type=int, help="usa so os N primeiros fim-*.jsonl (teste rapido)")
    ap.add_argument("--report", default=str(ROOT / "Repos" / "KapiTeacher-Mongo-7B" / "reports" / "corpus_manifest.md"))
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    global CFG, BLIND_PROMPTS
    CFG = cfg
    seed = int(cfg["seed"])
    src = ROOT / cfg["source_dir"]
    splits_dir = ROOT / cfg["splits_dir"]
    out_dir = Path(args.out) if args.out else ROOT / cfg["out_dir"]
    blind_guard.check_path(src)
    blind_guard.check_path(out_dir)
    if args.verify_only:
        print(json.dumps(verify_output(out_dir, splits_dir), indent=1))
        return 0
    n_target = args.target_examples or int(cfg["target_examples"])
    tok_path = ROOT / cfg["tokenizer"]
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tok_path))
    for text, want in EXPECTED_IDS.items():  # IDs FIM e EOS do Qwen2.5-Coder
        got = tok.encode(text, add_special_tokens=False).ids
        if got != [want]:
            raise SystemExit(f"tokenizer: {text} -> {got}, esperado [{want}]")
    manifest_blind = blind_guard._manifest()  # noqa: SLF001  (so hashes; o conteudo do blind nunca e lido)
    BLIND_PROMPTS = set(manifest_blind.get("prompt_sha256") or [])

    files = sorted(p for p in src.iterdir() if SHARD_RE.match(p.name))
    if args.only_shards:
        files = [p for p in files if p.name == "chat.jsonl"] + [p for p in files if p.name.startswith("fim-")][: args.only_shards]
    blind_guard.check_paths(files)
    print(f"[1/6] indexando {len(files)} arquivos de {src} com {args.workers} workers", flush=True)
    with mp.Pool(args.workers, initializer=_init_worker, initargs=(cfg, str(tok_path), frozenset())) as pool:
        scans = pool.map(_scan_shard, [str(p) for p in files], chunksize=1)
    train_rows: list[tuple] = []
    shard_of: dict[str, str] = {}
    eval_ids: dict[str, str] = {}
    eval_content: dict[int, str] = {}
    groups: dict[str, set] = defaultdict(set)
    schemas: dict[str, set] = defaultdict(set)
    counts, violations, val_rows = Counter(), [], []
    for s in scans:
        name = Path(s["path"]).name
        for row in s["train"]:
            shard_of[row[0]] = s["path"]
        train_rows += s["train"]
        eval_ids.update(s["eval_ids"])
        eval_content.update(s["eval_content"])
        for k, v in s["groups"].items():
            groups[k].update(v)
        for k, v in s["schemas"].items():
            schemas[k].update(v)
        counts.update(s["counts"])
        violations += s["violations"]
        val_rows += s.get("val", [])
        for r in s["val"] if "val" in s else []:
            shard_of[r[0]] = s["path"]
    print(f"      train={len(train_rows)} eval={len(eval_ids)} contagens={dict(counts)}", flush=True)

    # ---- G2.7 / isolamento: ids, grupos, schemas, conteudo --------------------------------------------------------------------------
    print("[2/6] verificando isolamento (G2.7)", flush=True)
    train_ids_file = read_ids(splits_dir / "train.ids.txt")
    split_files = {n: read_ids(splits_dir / f"{n}.ids.txt") for n in ("val", "test", "bench")}
    train_ids_scan = {r[0] for r in train_rows}
    iso = {
        "train_ids_scan_eq_splits_file": (train_ids_scan <= train_ids_file) if args.only_shards else (train_ids_scan == train_ids_file),
        "train_ids_only_in_scan": len(train_ids_scan - train_ids_file),
        "train_ids_only_in_file": len(train_ids_file - train_ids_scan),
        "eval_ids_by_split_scan": dict(Counter(eval_ids.values())),
        "train_x_val_ids": len(train_ids_scan & split_files["val"]),
        "train_x_test_ids": len(train_ids_scan & split_files["test"]),
        "train_x_bench_ids": len(train_ids_scan & split_files["bench"]),
        "eval_scan_eq_splits_files": {n: ({i for i, s in eval_ids.items() if s == n} <= split_files[n]) if args.only_shards
                                      else ({i for i, s in eval_ids.items() if s == n} == split_files[n]) for n in split_files},
        "group_overlap_train_x_eval": {n: len(groups["train"] & groups[n]) for n in ("val", "test", "bench")},
        "schema_overlap_train_x_eval": {n: len(schemas["train"] & schemas[n]) for n in ("val", "test", "bench")},
        "train_records_with_wrong_role_or_holdout": len(violations),
        "blind": {"sealed_manifest": bool(manifest_blind), "prompt_hashes": len(BLIND_PROMPTS),
                  "note": "conteudo do blind nunca lido; so hashes do manifesto (blind_guard)"},
    }
    hard_fail = (not iso["train_ids_scan_eq_splits_file"] or iso["train_x_val_ids"] or iso["train_x_test_ids"] or iso["train_x_bench_ids"]
                 or any(iso["group_overlap_train_x_eval"].values()) or any(iso["schema_overlap_train_x_eval"].values()) or violations
                 or not all(iso["eval_scan_eq_splits_files"].values()))
    if hard_fail:
        print(json.dumps(iso, indent=1, default=str), file=sys.stderr)
        raise SystemExit("G2.7: isolamento train x val/test/bench violado; corpus NAO gerado")

    # ---- selecao deterministica (menor hash estavel do id por componente) --------------------------------------------------------------
    print("[3/6] selecionando exemplos", flush=True)
    mix = cfg["mix"]
    want = {k: int(round(n_target * float(v))) for k, v in mix.items() if float(v) > 0 and k != "replay"}
    fim_all = [r for r in train_rows if r[1] == "fim"]
    chosen: dict[str, str] = {}
    avail = Counter()
    pop = {"completion": [r for r in fim_all if r[3]],
           "fim_raw": [r for r in fim_all if r[4] in cfg["fim_raw"]["contexts"]], "fim_ctx": fim_all}
    for comp in ("completion", "fim_raw", "fim_ctx"):
        pool = [r for r in pop[comp] if r[0] not in chosen]
        avail[comp] = len(pool)
        for r in pick_lowest(pool, want.get(comp, 0), seed, comp):
            chosen[r[0]] = comp
    for comp in CHAT_KINDS:
        pool = [r for r in train_rows if r[1] == "chat" and r[2] == comp]
        avail[comp] = len(pool)
        for r in pick_lowest(pool, want.get(comp, 0), seed, comp):
            chosen[r[0]] = comp
    sel_counts = Counter(chosen.values())
    short = {k: (want[k], sel_counts[k]) for k in want if sel_counts[k] < want[k]}

    vs = cfg["validation_subset"]
    val_fim = pick_lowest([r for r in val_rows if r[1] == "fim"], vs["fim"], seed, "val-fim")
    val_chat = pick_lowest([r for r in val_rows if r[1] == "chat"], vs["chat"], seed, "val-chat")
    val_sel = {r[0]: ("fim_ctx" if r[1] == "fim" else "chat", True) for r in val_fim + val_chat}
    selected = {i: (c, False) for i, c in chosen.items()}
    selected.update(val_sel)

    # ---- renderizacao + tokenizacao por arquivo ---------------------------------------------------------------------------------------
    print(f"[4/6] renderizando/tokenizando {len(selected)} exemplos (train {len(chosen)} + val {len(val_sel)})", flush=True)
    by_file: dict[str, dict] = defaultdict(dict)
    for ident, val in selected.items():
        by_file[shard_of[ident]][ident] = val
    tasks = [(p, sel, eval_content, seed) for p, sel in by_file.items()]
    results, dropped = [], Counter()
    with mp.Pool(args.workers, initializer=_init_worker, initargs=(cfg, str(tok_path), frozenset(BLIND_PROMPTS))) as pool:
        for r in pool.imap_unordered(_render_shard, tasks):
            results += r["items"]
            dropped.update(r["dropped"])

    # ---- replay: codigo generico licenciado (data/replay/v1), por orcamento de TOKENS sobre o total do corpus -------------------------------
    eos = EXPECTED_IDS["<|endoftext|>"]
    replay_status = {"status": "none", "examples": 0, "reason": cfg["replay"].get("reason", "desligado na config")}
    replay_val: list[dict] = []
    gen_generic: list[dict] = []
    rcfg = cfg["replay"]
    if rcfg.get("status") == "enabled" and float(mix.get("replay", 0)) > 0:
        print("[4b/6] replay (fonte local licenciada)", flush=True)
        rdir = ROOT / rcfg["source_dir"]
        blind_guard.check_path(rdir)
        docs = [json.loads(line) for line in (rdir / rcfg["train_file"]).open(encoding="utf-8")]
        eval_rows = [json.loads(line) for line in (rdir / rcfg["eval_file"]).open(encoding="utf-8")]
        monitor_rows = [json.loads(line) for line in (rdir / rcfg["monitor_file"]).open(encoding="utf-8")]
        eval_pkgs = {r["package"] for r in eval_rows} | {r["package"] for r in monitor_rows}
        eval_shas = {r["file_sha256"] for r in eval_rows} | {r["file_sha256"] for r in monitor_rows}
        rep_manifest = json.loads((rdir / "manifest.json").read_text(encoding="utf-8"))
        holdout_pkgs = {p["package"] for p in rep_manifest["packages"] if p["role"] == "eval_holdout"}
        bad_docs = [d["id"] for d in docs if d["package"] in eval_pkgs or d["package"] in holdout_pkgs or d["sha256"] in eval_shas]
        if bad_docs:
            raise SystemExit(f"G2.7/replay: {len(bad_docs)} documentos de replay em pacote/arquivo do holdout generic_js ({bad_docs[:3]})")
        tcount = _TokCounter(tok)
        chunks = []
        for d in docs:
            parts = chunk_replay_doc(d["text"], int(rcfg.get("window_chars", 4000)))
            for n, text in enumerate(parts):
                chunks.append({"id": f"{d['id']}-{n:03d}", "text": text, "last": n == len(parts) - 1, "doc": d})
        for c in chunks:
            c["kind"], c["prompt"], c["target"] = render_replay_chunk(c["text"], c["id"], cfg, seed, tcount)
        enc_p = tok.encode_batch([c["prompt"] for c in chunks], add_special_tokens=False)
        enc_t = tok.encode_batch([c["target"] for c in chunks], add_special_tokens=False)
        for c, ep, et in zip(chunks, enc_p, enc_t):
            c["p_ids"] = ep.ids if c["prompt"] else []
            c["t_ids"] = et.ids + ([eos] if (c["kind"] == "fim" or c["last"]) else [])
            c["tokens"] = len(c["p_ids"]) + len(c["t_ids"])
        # trechos que contem contexto+middle de um corte de avaliacao (boilerplate/helpers vendorizados entre pacotes): removidos do treino
        needles = [r["prefix"][-80:] + r["middle"] + r["suffix"][:80] for r in eval_rows + monitor_rows]
        contaminated = {c["id"] for c in chunks if any(n in c["text"] for n in needles)}
        dropped["replay_chunk_contains_eval_cut"] = len(contaminated)
        chunks = [c for c in chunks if c["id"] not in contaminated]
        train_nr = [r for r in results if r["meta"]["id"] in chosen]
        t_nr = sum(len(r["ids"]) for r in train_nr)
        share = float(mix["replay"])
        budget = int(round(t_nr * share / (1 - share)))
        picked, used = [], 0
        for c in sorted(chunks, key=lambda c: unit(seed, "replay-pick", c["id"])):
            if c["tokens"] > cfg["seq_len"]:
                dropped["replay_too_long"] += 1
                continue
            if used + c["tokens"] > budget:
                continue
            picked.append(c)
            used += c["tokens"]
        # isolamento do replay (G2.7 estendido): nenhum trecho contem contexto+middle de um corte de avaliacao; hash de conteudo vs Mongo
        sub_hits = sum(1 for c in picked if any(n in c["text"] for n in needles))
        mongo_hits = sum(1 for c in picked
                         if int.from_bytes(hashlib.sha1(c["text"].encode("utf-8", "surrogatepass")).digest()[:8], "big") in eval_content)
        pk_overlap = sorted({c["doc"]["package"] for c in picked} & (eval_pkgs | holdout_pkgs))
        rep_iso = {"package_overlap_with_generic_eval": pk_overlap,
                   "file_sha256_overlap_with_generic_eval": len({c["doc"]["sha256"] for c in picked} & eval_shas),
                   "chunks_containing_eval_cut": sub_hits, "chunks_matching_mongo_eval_content": mongo_hits,
                   "source_manifest_isolation": rep_manifest.get("isolation")}
        if pk_overlap or rep_iso["file_sha256_overlap_with_generic_eval"] or sub_hits or mongo_hits:
            raise SystemExit(f"G2.7/replay: vazamento replay x generic_js_fim_eval/val/test/bench: {rep_iso}")
        for c in picked:
            results.append({"meta": {"id": c["id"], "component": f"replay_{c['kind']}", "kind": c["kind"], "domain": c["doc"]["package"],
                                     "language": "ts" if c["doc"]["ext"] in (".ts", ".tsx") else "js", "category": c["doc"]["license"],
                                     "n_prompt": len(c["p_ids"]), "n_target": len(c["t_ids"]), "src": "replay"},
                            "ids": c["p_ids"] + c["t_ids"], "loss_start": len(c["p_ids"])})
        # val: 100 cortes do monitor (outros arquivos dos pacotes holdout) em FIM cru, so para val_loss_generic / val_gen_generic
        for r in monitor_rows:
            before, after = rc.bound_window(r["prefix"], r["suffix"])
            prompt = rc.qwen_fim_text(before, after)
            p_ids = tok.encode(prompt, add_special_tokens=False).ids
            t_ids = tok.encode(r["middle"], add_special_tokens=False).ids + [eos]
            replay_val.append({"meta": {"id": r["id"], "component": "replay_fim", "kind": "fim", "domain": r["package"], "language": "js",
                                        "category": r["license"], "n_prompt": len(p_ids), "n_target": len(t_ids), "src": "monitor"},
                               "ids": p_ids + t_ids, "loss_start": len(p_ids)})
            gen_generic.append({"id": r["id"], "prompt": prompt, "target": r["middle"], "prefix": r["prefix"], "suffix": r["suffix"],
                                "dialect": "js", "cut_kind": r["cut_kind"], "domain": r["package"]})
        kinds = Counter(c["kind"] for c in picked)
        tok_by_kind: Counter = Counter()
        lic_tokens: Counter = Counter()
        for c in picked:
            tok_by_kind[c["kind"]] += c["tokens"]
            lic_tokens[c["doc"]["license"]] += c["tokens"]
        replay_status = {
            "status": "enabled", "source_dir": rcfg["source_dir"],
            "files": {n: sha256_file(rdir / n) for n in (rcfg["train_file"], rcfg["eval_file"], rcfg["monitor_file"], "manifest.json",
                                                         "LICENSES.md", "files.jsonl")},
            "token_share_target": share, "token_budget": budget, "tokens": used, "non_replay_tokens": t_nr,
            "examples": len(picked), "by_kind": dict(kinds), "tokens_by_kind": dict(tok_by_kind),
            "available_chunks": len(chunks), "available_tokens": sum(c["tokens"] for c in chunks),
            "docs_used": len({c["doc"]["id"] for c in picked}), "packages_used": len({c["doc"]["package"] for c in picked}),
            "tokens_by_license": dict(lic_tokens), "window_chars": rcfg.get("window_chars", 4000),
            "fim_fraction": rcfg.get("fim_fraction", 0.5), "isolation": rep_iso, "monitor_examples": len(replay_val),
        }
        print(f"      replay: {len(picked)} trechos / {used:,} tokens (alvo {budget:,}; disponivel {replay_status['available_tokens']:,}) "
              f"sobre {t_nr:,} tokens nao-replay", flush=True)
    results.sort(key=lambda r: r["meta"]["id"])
    train_ex = [r for r in results if r["meta"]["id"] in chosen or r["meta"]["component"].startswith("replay")]
    val_ex = [r for r in results if r["meta"]["id"] in val_sel] + replay_val

    # ---- shards ------------------------------------------------------------------------------------------------------------------------
    print(f"[5/6] gravando {len(train_ex)} exemplos de treino em shards de {cfg['shard_examples']}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in list(out_dir.glob("train-*")) + list((out_dir / "val").glob("*")):
        old.unlink()
    random.Random(f"{seed}-global-shuffle").shuffle(train_ex)
    shards = []
    size = int(cfg["shard_examples"])
    for n, start in enumerate(range(0, len(train_ex), size)):
        shards.append(write_shard(out_dir / f"train-{n:04d}.npz", train_ex[start: start + size], cfg["micro_tokens"],
                                  cfg["max_examples_per_row"], seed))
    (out_dir / "val").mkdir(exist_ok=True)
    val_loss = [r for r in val_ex]
    val_shard = write_shard(out_dir / "val" / "val_loss.npz", val_loss, cfg["micro_tokens"], cfg["max_examples_per_row"], seed)
    gen = [r["gen"] for r in val_ex if "gen" in r][: vs["gen"]]
    with (out_dir / "val" / "val_gen.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for g in gen:
            handle.write(json.dumps(g, ensure_ascii=False) + "\n")
    with (out_dir / "val" / "val_gen_generic.jsonl").open("w", encoding="utf-8", newline="\n") as handle:  # monitor generic_js (replay)
        for g in gen_generic:
            handle.write(json.dumps(g, ensure_ascii=False) + "\n")

    # ---- manifesto + verificacao + relatorio -------------------------------------------------------------------------------------------
    print("[6/6] manifesto e verificacao", flush=True)
    src_manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    src_files = {}
    for p in files:
        declared = (src_manifest.get("files", {}).get(p.name) or {}).get("sha256")
        actual = sha256_file(p)
        src_files[p.name] = {"sha256": actual, "declared": declared, "match": declared == actual if declared else None}
    ver = verify_output(out_dir, splits_dir)
    tokens = sum(s["tokens"] for s in shards)
    by_comp: dict[str, dict] = defaultdict(lambda: {"examples": 0, "tokens": 0, "target_tokens": 0})
    for r in train_ex:
        c = by_comp[r["meta"]["component"]]
        c["examples"] += 1
        c["tokens"] += len(r["ids"])
        c["target_tokens"] += r["meta"]["n_target"]
    manifest = {
        "schema": "teacher-corpus/1", "built_utc": time.strftime("%FT%TZ", time.gmtime()), "git_hash": git_hash(),
        "config": {"path": rel(cfg_path),
                   "sha256": sha256_file(cfg_path)},
        "builder_sha256": sha256_file(Path(__file__)), "seed": seed, "dataset_version": cfg["dataset_version"],
        "source": {"dir": cfg["source_dir"], "manifest_sha256": sha256_file(src / "manifest.json"), "files": src_files,
                   "config_sha256": src_manifest.get("config", {}).get("sha256"),
                   "train_ids_sha256": sha256_file(splits_dir / "train.ids.txt"),
                   "val_ids_sha256": sha256_file(splits_dir / "val.ids.txt")},
        "tokenizer": {"path": cfg["tokenizer"], "sha256": sha256_file(tok_path), "special_ids": EXPECTED_IDS},
        "mix": {"target": {k: float(v) for k, v in mix.items()},
                "achieved": {k: round(v["examples"] / max(1, len(train_ex)), 4) for k, v in sorted(by_comp.items())},
                "available_in_train": dict(avail), "short_of_target": short},
        "replay": replay_status,
        "counts": {"examples": len(train_ex), "tokens": tokens, "target_tokens": sum(s["target_tokens"] for s in shards),
                   "padded_tokens": sum(s["padded_tokens"] for s in shards), "rows": sum(s["rows"] for s in shards),
                   "mean_tokens_per_example": round(tokens / max(1, len(train_ex)), 1), "dropped": dict(dropped),
                   "by_component": dict(by_comp)},
        "micro_tokens": cfg["micro_tokens"], "seq_len": cfg["seq_len"],
        "val": {"loss_file": "val/val_loss.npz", "loss_sha256": val_shard["sha256"], "examples": val_shard["examples"],
                "tokens": val_shard["tokens"], "gen_file": "val/val_gen.jsonl",
                "gen_sha256": sha256_file(out_dir / "val" / "val_gen.jsonl"), "gen_examples": len(gen),
                "gen_generic_file": "val/val_gen_generic.jsonl", "gen_generic_examples": len(gen_generic),
                "gen_generic_sha256": sha256_file(out_dir / "val" / "val_gen_generic.jsonl")},
        "shards": shards, "isolation": iso, "verification": ver,
        "chat_template": {"name": cfg["chat"]["template"], "head": CHAT_HEAD, "tail": CHAT_TAIL,
                          "include_known_collections": cfg["chat"]["include_known_collections"],
                          "escaping": "System.Text.Json JavaScriptEncoder.Default (\\uXXXX maiusculo)"},
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    manifest["manifest_sha256"] = sha256_file(out_dir / "manifest.json")
    write_report(Path(args.report), manifest, out_dir)
    ok = ver["n_issues"] == 0
    print(f"OK={ok} exemplos={len(train_ex)} tokens={tokens} shards={len(shards)} em {time.time() - t0:.0f}s")
    return 0 if ok else 3


def replay_report_lines(m: dict) -> list[str]:
    r = m["replay"]
    if r["status"] != "enabled":
        return [f"**Replay: `{r['status']}`**. {r.get('reason', '')}",
                "Impacto em G2.5 (queda <= 3 p.p. no Generic-JS FIM): sem replay o unico freio ao esquecimento e o LoRA (r=64, lr 1e-4, 1 epoca, "
                "base congelada em NF4) e os 10% de FIM cru; o risco de G2.5 nao e mitigado."]
    iso = r["isolation"]
    ok = (not iso["package_overlap_with_generic_eval"] and not iso["file_sha256_overlap_with_generic_eval"]
          and not iso["chunks_containing_eval_cut"] and not iso["chunks_matching_mongo_eval_content"])
    return [
        f"**Replay: `enabled`** - codigo JS/TS generico de terceiros, licencas permissivas (MIT/Apache-2.0/BSD/ISC), fonte local `{r['source_dir']}` "
        f"(sha256 do replay_train `{r['files']['replay_train.jsonl'][:16]}`, do eval `{r['files']['generic_js_fim_eval.jsonl'][:16]}`; "
        "inventario por arquivo e atribuicoes em `manifest.json`/`LICENSES.md` dessa pasta).",
        "",
        f"- {r['examples']} trechos de {r['docs_used']} arquivos / {r['packages_used']} pacotes: **{r['tokens']:,} tokens = "
        f"{r['tokens'] / max(1, r['tokens'] + r['non_replay_tokens']):.1%} do corpus** (alvo {r['token_share_target']:.0%}; orcamento {r['token_budget']:,}; "
        f"disponivel {r['available_tokens']:,}). Por tipo: {r['by_kind']} (tokens {r['tokens_by_kind']}). Janela {r['window_chars']} caracteres, alinhada por linha.",
        f"- `replay_fim`: FIM PSM cru (`<|fim_prefix|>..<|fim_suffix|>..<|fim_middle|>` -> middle <= 96 tokens, corte por limite sintatico com a "
        "logica de `gen_fim.candidates/pick_end` sobre spans de lexer JS); `replay_causal`: trecho em sequencia, perda em todos os tokens "
        "(EOS so no ultimo trecho do arquivo).",
        f"- Licencas (tokens): {r['tokens_by_license']}.",
        f"- Isolamento do replay x `generic_js_fim_eval` (500 cortes) / monitor / val-test-bench do Mongo: pacotes em comum "
        f"{len(iso['package_overlap_with_generic_eval'])}, arquivos (sha256) em comum {iso['file_sha256_overlap_with_generic_eval']}, trechos que contem "
        f"um corte de avaliacao (contexto+middle) {iso['chunks_containing_eval_cut']}, trechos com hash igual ao conteudo val/test/bench do Mongo "
        f"{iso['chunks_matching_mongo_eval_content']} -> **{'OK' if ok else 'VIOLADO'}**.",
        "- Impacto em G2.5: o replay de FIM cru e de codigo generico alimenta o freio ao esquecimento; a medicao e `generic_js_fim_eval` "
        "(adapter - base >= -3 p.p.). Ver README_job.md.",
    ]


def write_report(path: Path, m: dict, out_dir: Path) -> None:
    iso, ver, c = m["isolation"], m["verification"], m["counts"]
    L = ["# Corpus de SFT do KapiTeacher-Mongo-7B (dataset v2c)", "",
         f"Gerado por `scripts/teacher/build_teacher_corpus.py` em {m['built_utc']} (git `{m['git_hash'][:12]}`, seed {m['seed']}), "
         f"{m['elapsed_s']} s. Config `{m['config']['path']}` sha256 `{m['config']['sha256'][:16]}`. Saida: `{rel(out_dir)}` "
         f"(gitignored; manifest.json sha256 `{m.get('manifest_sha256', '')[:16]}`).", "",
         "## 1. Origem e hashes", "",
         f"- Dataset de origem: `{m['source']['dir']}` (manifest sha256 `{m['source']['manifest_sha256'][:16]}`; config do dataset "
         f"`{(m['source']['config_sha256'] or '')[:16]}`). `train.ids.txt` sha256 `{m['source']['train_ids_sha256'][:16]}`; "
         f"`val.ids.txt` `{m['source']['val_ids_sha256'][:16]}`.",
         f"- Tokenizer: `{m['tokenizer']['path']}` sha256 `{m['tokenizer']['sha256'][:16]}` (mesmo do base, dos students 0.5B/1.5B e da IDE); "
         f"IDs {m['tokenizer']['special_ids']}.", "",
         "| arquivo de origem | sha256 (calculado) | confere com o manifest do dataset |", "|---|---|---|"]
    for name, f in m["source"]["files"].items():
        L.append(f"| {name} | `{f['sha256'][:16]}` | {f['match']} |")
    L += ["", "## 2. Mistura", "",
          "Fracoes `alvo` dos componentes do Mongo sao relativas ao conjunto NAO-replay (por exemplos); `replay` e fracao do TOTAL de tokens "
          "(`mix.replay`); a coluna `% tokens` e a participacao real no total do corpus.", "",
          "| componente | alvo | obtido (exemplos) | exemplos | tokens | % tokens | tokens do alvo | disponivel em train |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    total_tokens = max(1, sum(v["tokens"] for v in c["by_component"].values()))
    for k, v in sorted(c["by_component"].items()):
        tgt = m["mix"]["target"].get("replay", 0) if k.startswith("replay") else m["mix"]["target"].get(k, 0)
        L.append(f"| {k} | {tgt:.3f} | {m['mix']['achieved'].get(k, 0):.3f} | {v['examples']} | {v['tokens']} | {v['tokens'] / total_tokens:.1%} | "
                 f"{v['target_tokens']} | {m['mix']['available_in_train'].get(k, '-')} |")
    L += ["", f"Alvos nao atingidos: {m['mix']['short_of_target'] or 'nenhum'}. Descartes na tokenizacao/guardas: {c['dropped'] or 'nenhum'}.", "",
          *replay_report_lines(m), "",
          "## 3. Tamanho", "",
          f"- Exemplos: **{c['examples']}** · tokens reais: **{c['tokens']:,}** (media {c['mean_tokens_per_example']}/ex.) · "
          f"tokens de alvo (com perda): {c['target_tokens']:,} ({c['target_tokens'] / max(1, c['tokens']):.1%}) · linhas (micro-batches de "
          f"<= {m['micro_tokens']} tokens padded): {c['rows']:,} · tokens padded {c['padded_tokens']:,} "
          f"(padding {1 - c['tokens'] / max(1, c['padded_tokens']):.1%}).",
          f"- Shards: {len(m['shards'])} x ate {m['shards'][0]['examples'] if m['shards'] else 0} exemplos; sha256 de cada um em `manifest.json`.",
          f"- Validacao do treino (split `val`): {m['val']['examples']} exemplos / {m['val']['tokens']:,} tokens para val_loss; "
          f"{m['val']['gen_examples']} FIM para geracao rapida (EM / mongo_valid).", "",
          "## 4. G2.7 - isolamento (0 exemplos de test/bench/blind/val no treino)", "",
          "Verificado em dois momentos: na leitura (indice completo de todos os shards v2c) antes de selecionar, e depois de gravar "
          "(`verify_output`, relendo os `.meta.jsonl` contra `data/splits/v2c/*.ids.txt`). Qualquer violacao aborta a geracao.", "",
          "| verificacao | resultado |", "|---|---|",
          f"| ids `split==train` lidos dos shards == `train.ids.txt` | {iso['train_ids_scan_eq_splits_file']} (so no scan: {iso['train_ids_only_in_scan']}; so no arquivo: {iso['train_ids_only_in_file']}) |",
          f"| ids val/test/bench lidos == arquivos de ids | {iso['eval_scan_eq_splits_files']} |",
          f"| train x val / test / bench (ids) | {iso['train_x_val_ids']} / {iso['train_x_test_ids']} / {iso['train_x_bench_ids']} |",
          f"| grupos (`group_id`) em comum train x val/test/bench | {iso['group_overlap_train_x_eval']} |",
          f"| schemas em comum train x val/test/bench | {iso['schema_overlap_train_x_eval']} |",
          f"| registros train com `split_role`!=train ou dominio holdout | {iso['train_records_with_wrong_role_or_holdout']} |",
          f"| conteudo identico a val/test/bench (hash de prefix+suffix+middle / instruction+editor+output) | descartados: {c['dropped'].get('content_in_eval', 0)} |",
          f"| prompt identico a item selado do blind (hash do manifesto) | descartados: {c['dropped'].get('blind', 0)}; blind selado: {iso['blind']['sealed_manifest']} ({iso['blind']['prompt_hashes']} hashes). Conteudo do blind nunca lido |",
          f"| **corpus final**: ids no corpus x train/val/test/bench ids | {ver['in_train_ids']} de {ver['examples'] - ver.get('replay_examples', 0)} do Mongo em train (+ {ver.get('replay_examples', 0)} de replay, ids `rpl-`); val {ver['in_val_ids']}; test {ver['in_test_ids']}; bench {ver['in_bench_ids']}; duplicados/problemas {ver['n_issues']} |",
          "", f"**G2.7: {'APROVADO' if ver['n_issues'] == 0 and ver['in_val_ids'] + ver['in_test_ids'] + ver['in_bench_ids'] == 0 else 'REPROVADO'}**", "",
          "## 5. Formatos renderizados", "",
          "- `fim_ctx`: `prompt` v2c (contrato `editor-context-v1`, CRLF fixo, ModelPrefix, janelas 4096/1024 UTF-16) + `<|fim_middle|>`; alvo = `middle` + `<|endoftext|>`.",
          "- `fim_raw`: `<|fim_prefix|>{prefixo<=4096}<|fim_suffix|>{sufixo<=1024}<|fim_middle|>` (como o QwenFimPromptBuilder sem cabecalho).",
          "- `completion`: mesmo prompt de `fim_ctx` com `<|fim_suffix|><|fim_middle|>` (sufixo vazio); alvo = `middle`+`suffix` (restos <= 240 caracteres).",
          f"- chat (`{m['chat_template']['name']}`): `<|fim_prefix|>` + `{m['chat_template']['head'].strip()}` + JSON `{{instruction, context}}` "
          f"(System.Text.Json) + `{m['chat_template']['tail'].strip()}` + `<|fim_suffix|><|fim_middle|>`; alvo = `output` + `<|endoftext|>`. "
          "Sem Qwen chat template (modelo base). `context` = `Conexão (ID lógico)`/`Banco`/`Coleção`" +
          (" + `Coleções disponíveis`" if m['chat_template']['include_known_collections'] else "") + " + blocos `Seleção/Conteúdo do editor`.",
          "- Perda somente nos tokens do alvo + `<|endoftext|>`; prompt mascarado. Sem concatenacao entre exemplos.", "",
          "## 6. Observacoes", "",
          "- O pool e uma amostra uniforme (menor hash estavel do id) de cada componente; nao ha sobre-amostragem `hard` (exige scoring do professor).",
          "- `chat_multiturn`: so o ultimo turno (o codigo do turno anterior ja esta em `context.editor`); `history` nao e usado.",
          "- Linhas (micro-batches) ja agrupadas por comprimento; o treino consome shards na ordem do manifesto e pode parar em qualquer ponto "
          "(prefixo aleatorio representativo)."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
