#!/usr/bin/env python3
"""Avaliacao no conjunto de HOMOLOGACAO v1 (data/homolog; 6.030 cortes FIM, schemas `hom`, author llm|grammar -- NAO humano).

Mesmo contrato e mesmo orcamento de scripts/teacher/eval_teacher.py / scripts/student/eval_student.py: `editor-context-v1`, greedy, ate 96 tokens,
contexto 2048, padding do vocab mascarado. Diferenca: os itens tem prefixo+predicao+sufixo como TEXTO (nao plano estruturado), entao o exec-valid
vem do validador por EXECUCAO DO TEXTO real (scripts/data/llm/validate_text.py: mongosh no WSL atras do proxy somente-leitura, allowlist, maxTimeMS),
e os campos inexistentes vem do plano capturado pelo proxy (plan_fields). O TETO e a mesma validacao sobre o `middle` de referencia.

  # 1) modelo completo (venv ROCm, GPU 7800 XT) + exec (env/.venv: pymongo + bson; WSL/Mongo) -- retomavel (--resume e cache de execucao)
  python -I scripts/eval/eval_homolog.py run --model Repos/KapiCoder-Mongo-Nano/merged/sft_distill_v1 --name t0_distill
  python -I scripts/eval/eval_homolog.py run --model BaseRepos/Qwen2.5-Coder-7B --adapter Repos/KapiTeacher-Mongo-7B/adapters/sft_v2d/final --load nf4 --name teacher_v2d
  # fases separadas
  python -I scripts/eval/eval_homolog.py gen   --model ... --name X [--adapter ...] [--load auto|bf16|nf4] [--limit N] [--resume]   (venv ROCm)
  env/.venv/Scripts/python.exe -I scripts/eval/eval_homolog.py score --name X                                                       (execucao)
  env/.venv/Scripts/python.exe -I scripts/eval/eval_homolog.py score --name _reference                                              (teto)
  env/.venv/Scripts/python.exe -I scripts/eval/eval_homolog.py report [--names a b ...]       (tabelas, lacuna vs val v2c, comparacao pareada)

Guardas: o diretorio data/homolog so e lido dentro de `with homolog_guard.authorized(motivo, quem)` (deixa linha em access.log.jsonl);
`homolog_guard.enforce()` fica ativo em todo o processo. Escritas so em data/homolog/results/ (predicoes por id; gitignored) e reports/ (SO agregados,
sem texto de item). Nenhum dado congelado e alterado.
"""
from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "scripts" / "data" / "llm", ROOT / "scripts" / "eval", ROOT / "scripts" / "teacher", ROOT / "scripts" / "data",
           ROOT / "bakeoff" / "v1" / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import homolog_guard as hg  # noqa: E402

RESULTS = ROOT / "data" / "homolog" / "results"
REPORTS = ROOT / "reports"
REPORT_DIR = REPORTS / "datasets"
ENV_VENV_PY = ROOT / "env" / ".venv" / "Scripts" / "python.exe"
MAX_NEW_TOKENS = 96
CONTEXT_TOKENS = 2048
CONTRACT = "editor-context-v1"
REFERENCE = "_reference"
EXEC_CACHE = "_exec_cache.jsonl"
GROUPS = ("context_kind", "cut_kind", "domain", "author")
CTX_ORDER = ("full", "irrelevant", "none")
# ordem e rotulos do relatorio (nome da execucao -> rotulo, val v2c summary.json)
MODELS = [
    ("base_0.5b", "Qwen2.5-Coder-0.5B base", "Repos/KapiCoder-Mongo-Nano/reports/eval/base-qwen2.5-coder-0.5b/summary.json"),
    ("t0_distill", "T0 Nano distilado", "Repos/KapiCoder-Mongo-Nano/reports/eval/sft_distill_v1/summary.json"),
    ("t0_gold", "T0 controle gold", "Repos/KapiCoder-Mongo-Nano/reports/eval/sft_gold_v1/summary.json"),
    ("base_1.5b", "Qwen2.5-Coder-1.5B base", "Repos/KapiCoder-Mongo-Small/reports/eval/base-qwen2.5-coder-1.5b/summary.json"),
    ("t1_distill", "T1 Small distilado", "Repos/KapiCoder-Mongo-Small/reports/eval/sft_distill_v1/summary.json"),
    ("teacher_v2d", "Professor v2d (7B NF4+LoRA)", "Repos/KapiTeacher-Mongo-7B/reports/eval/sft_v2d-final/summary.json"),
]
PAIRS = [  # (A, B): delta = B - A
    ("t0_gold", "t0_distill"), ("base_0.5b", "t0_distill"), ("base_0.5b", "t0_gold"),
    ("t0_distill", "t1_distill"), ("t0_gold", "t1_distill"), ("base_1.5b", "t1_distill"),
    ("teacher_v2d", "t1_distill"), ("teacher_v2d", "t0_distill"), ("teacher_v2d", "t0_gold"),
]


# -- E/S segura ---------------------------------------------------------------------------------------------------------------------
def _under(path: Path, root: Path) -> bool:
    p, r = os.path.normcase(os.path.realpath(path)), os.path.normcase(os.path.realpath(root))
    return p == r or p.startswith(r + os.sep)


def check_write_path(path) -> Path:
    """So se escreve em data/homolog/results/ e reports/."""
    path = Path(path)
    if not (_under(path, RESULTS) or _under(path, REPORTS)):
        raise PermissionError(f"escrita fora de data/homolog/results e reports: {path}")
    return path


def authz(reason: str, who: str = "eval_homolog", **detail):
    return hg.authorized(reason, who, detail={"cmd": "eval", **detail})


def write_text(path, text: str) -> None:
    path = check_write_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if _under(path, RESULTS):
        with authz("gravar resultados da avaliacao (predicoes por id / agregados)", detail_path=path.name):
            tmp.write_text(text, encoding="utf-8", newline="\n")
            os.replace(tmp, path)
    else:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, path)


def write_json(path, obj) -> None:
    write_text(path, json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    ctx = authz("ler resultados da avaliacao", detail_path=path.name) if _under(path, RESULTS) else None
    if ctx:
        ctx.__enter__()
    try:
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    finally:
        if ctx:
            ctx.__exit__()


def write_jsonl(path, rows) -> None:
    write_text(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def read_json(path) -> dict:
    path = Path(path)
    if _under(path, RESULTS):
        with authz("ler resultados da avaliacao", detail_path=path.name):
            return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def exists(path) -> bool:
    return Path(path).exists()


def run_dir(name: str) -> Path:
    return RESULTS / name


# -- itens (SO pela API autorizada) -------------------------------------------------------------------------------------------------
def order_key(item_id: str) -> str:
    return hashlib.sha1(item_id.encode()).hexdigest()


def load_items(reason: str, who: str, limit: int | None = None) -> list[dict]:
    """Le items.jsonl dentro de `authorized` (log de acesso). Ordem deterministica: sha1(id); --limit pega os primeiros (amostra espalhada)."""
    path = hg.set_dir() / "items" / "items.jsonl"
    with hg.authorized(reason, who, detail={"cmd": "eval", "limit": limit}):
        with open(path, encoding="utf-8") as fh:
            items = [json.loads(line) for line in fh if line.strip()]
    items.sort(key=lambda r: order_key(r["id"]))
    return items[:limit] if limit else items


# -- contadores do Windows: confirma a GPU de verdade ---------------------------------------------------------------------------------
_PS = r"""
$mem = (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage').CounterSamples | ForEach-Object { [pscustomobject]@{i=$_.InstanceName; v=[double]$_.CookedValue} }
$eng = (Get-Counter '\GPU Engine(*)\Utilization Percentage' -SampleInterval 1 -MaxSamples 3)[-1].CounterSamples |
  Where-Object { $_.InstanceName -like 'pid_%PID%_*' } | ForEach-Object { [pscustomobject]@{i=$_.InstanceName; v=[double]$_.CookedValue} }
[pscustomobject]@{mem=@($mem); eng=@($eng)} | ConvertTo-Json -Depth 4 -Compress
"""


def parse_gpu_counters(doc: dict, min_gb: float = 0.8) -> dict:
    """Pura: {adaptador(luid): GB dedicados} e {adaptador: % de uso do pid}. ok = o adaptador com mais memoria dedicada recebe >= 50% do uso do pid."""
    mem = {}
    for m in doc.get("mem") or []:
        mem[m["i"]] = round(m["v"] / 2**30, 3)
    eng = collections.defaultdict(float)
    for e in doc.get("eng") or []:
        name = e["i"]
        k = name.split("_luid_", 1)[1].split("_eng_", 1)[0] if "_luid_" in name else name
        eng["luid_" + k] += e["v"]
    big = max(mem, key=mem.get) if mem else None
    total = sum(eng.values())
    share = (eng.get(big, 0.0) / total) if (big and total > 0) else None
    verdict = ("inconclusivo" if share is None else
               ("ok" if (share >= 0.5 and mem[big] >= min_gb) else "REPROVADO"))
    return {"dedicated_gb_by_adapter": mem, "pid_engine_pct_by_adapter": {k: round(v, 1) for k, v in eng.items()}, "largest_adapter": big,
            "share_on_largest_adapter": None if share is None else round(share, 3), "verdict": verdict}


def sample_gpu(pid: int, out: dict) -> None:
    try:
        res = subprocess.run(["powershell", "-NoProfile", "-Command", _PS.replace("%PID%", str(pid))], capture_output=True, text=True, timeout=60)
        out.update(parse_gpu_counters(json.loads(res.stdout)))
        out["pid"] = pid
    except Exception as exc:  # noqa: BLE001
        out.update({"verdict": "inconclusivo", "error": repr(exc)[:200]})


# -- fase gen (venv ROCm) -------------------------------------------------------------------------------------------------------------
def phase_gen(a) -> int:
    hg.enforce()
    import eval_teacher as et
    from _common import Generator, auto_load, gpu_lock, library_versions, load_hf, model_fingerprint, sha256_file
    from compact import syntax
    from compact.contracts import contract_for, ide_rejects, postprocess
    from compact.mongo_validate import validate

    who = f"eval_homolog:{a.name}"
    out_dir = run_dir(a.name)
    items = load_items(f"avaliacao do modelo {a.name} (geracao greedy)", who, a.limit)
    pred_path = out_dir / "predictions.jsonl"
    done: dict[str, dict] = {}
    if a.resume and exists(pred_path):
        done = {r["id"]: r for r in read_jsonl(pred_path)}
    model_dir = Path(a.model).resolve()
    load = a.load if a.load != "auto" else auto_load(model_dir)
    wall0 = time.time()
    gpu_ev: dict = {}
    gpu_thread = None
    sanity = {"checked": 0, "prompt_equal": 0}
    with gpu_lock(f"eval_homolog:{a.name}"):
        model, tokenizer, device, gpu, load_seconds = load_hf(model_dir, load)
        import torch
        adapter_info = None
        if a.adapter:
            from peft import PeftModel
            adapter_dir = Path(a.adapter).resolve()
            model = PeftModel.from_pretrained(model, adapter_dir, torch_device=str(device))
            model.eval()
            adapter_info = {"path": str(adapter_dir), "sha256": sha256_file(adapter_dir / "adapter_model.safetensors")}
        contract = contract_for(tokenizer)
        gen = Generator(model, tokenizer, device)
        for row in items[:3]:
            gen.generate(et.build_ids(contract, row, CONTRACT), 4, contract.stops, "")
        started = time.time()
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [r for r in items if r["id"] not in done]
        fh = None
        with authz("abrir arquivo de predicoes por id (data/homolog/results)", who):
            fh = pred_path.open("a", encoding="utf-8", newline="\n")
        try:
            for index, row in enumerate(todo):
                ids = et.build_ids(contract, row, CONTRACT)
                if sanity["checked"] < 100:
                    sanity["checked"] += 1
                    sanity["prompt_equal"] += int(tokenizer.decode(ids, skip_special_tokens=False) == row.get("prompt"))
                result = gen.generate(ids, MAX_NEW_TOKENS, contract.stops, row["suffix"][:1024])
                tokens = [t for t in result["tokens"] if t not in contract.stops]
                raw = gen.decode(tokens)
                pred = postprocess(raw, row)
                ref = row["middle"]
                rec = {"id": row["id"], "cut_kind": row["cut_kind"], "context_kind": row["context_kind"], "domain": row["domain"],
                       "author": row["author"], "source": row["source"], "dialect": row["dialect"], "kind": row.get("kind"),
                       "schema_id": row["schema_id"], "collection": row["collection"], "prompt_tokens": len(ids),
                       "generated_tokens": len(result["tokens"]), "stop": result["stop"], "ttft_ms": round(result["ttft"] * 1000, 1),
                       "ide_rejected": ide_rejects(pred, result["stop"] in ("eos", "suffix")), "prediction": pred, "reference": ref,
                       "metrics": et.score_text(pred, ref, tokenizer)}
                done[row["id"]] = rec
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                if index == min(60, len(todo) - 1) and gpu_thread is None:    # amostra os contadores do Windows com a geracao em andamento
                    gpu_thread = threading.Thread(target=sample_gpu, args=(os.getpid(), gpu_ev), daemon=True)
                    gpu_thread.start()
                if (index + 1) % 100 == 0:
                    sub = list(done.values())
                    print(f"[{index + 1}/{len(todo)}] {time.time() - started:.0f}s  FIM acc={sum(r['metrics']['normalized_match'] for r in sub) / len(sub):.3f}", flush=True)
        finally:
            fh.close()
        gen_seconds = time.time() - started
        memory = {"vram_max_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 2**30, 2)}
        gpu_report = {k: gpu.get(k) for k in ("device_name", "gcn_arch", "total_vram_gb", "torch", "hip", "vram_limit")}
        stops = sorted(contract.stops)
        del model
    if gpu_thread:
        gpu_thread.join(timeout=90)
    # sintaxe (node vm.Script) e mongo_valid estatico do documento reconstruido: predicao e referencia (como eval_teacher)
    by_id = {r["id"]: r for r in items}
    rows = [done[r["id"]] for r in items if r["id"] in done]
    jobs = []
    for r in rows:
        src = by_id[r["id"]]
        for key, text in (("pred", r["prediction"]), ("ref", r["reference"])):
            jobs.append((f"{r['id']}|{key}", (src["prefix"] + text + src["suffix"]).replace("\r\n", "\n"), syntax.kind_for(src["dialect"])))
    sres = syntax.check_many(jobs)
    texts = {j[0]: j[1] for j in jobs}
    for r in rows:
        for key, target in (("pred", "metrics"), ("ref", "reference_checks")):
            job = f"{r['id']}|{key}"
            ok = sres[job][0]
            check = validate(texts[job], r["dialect"]) if ok else None
            mv = check.valid if check and check.validatable else (False if not ok else None)
            if key == "pred":
                r["metrics"]["syntax_valid"], r["metrics"]["mongo_valid"] = ok, mv
            else:
                r["reference_checks"] = {"syntax_valid": ok, "mongo_valid": mv}
    write_jsonl(pred_path, rows)
    meta = {"name": a.name, "model_path": str(model_dir), "model": model_fingerprint(model_dir), "load": load, "load_seconds": load_seconds,
            "adapter": adapter_info, "contract": CONTRACT, "model_contract": contract.name, "stop_ids": stops, "max_new_tokens": MAX_NEW_TOKENS,
            "context_tokens": CONTEXT_TOKENS, "decoding": "greedy (argmax), padding do vocab mascarado", "n": len(rows),
            "prompt_equals_sealed_prompt": sanity, "libraries": library_versions(), "gpu": gpu_report, "memory": memory,
            "gpu_counters_windows": gpu_ev, "gpu_seconds": round(time.time() - wall0, 1), "generation_seconds": round(gen_seconds, 1),
            "set_sha256": hg.manifest().get("set_sha256"), "finished": time.strftime("%FT%T")}
    write_json(out_dir / "gen_meta.json", meta)
    print(json.dumps({"n": len(rows), "gpu": gpu_report.get("device_name"), "counters": gpu_ev.get("verdict"), "adapters_gb": gpu_ev.get("dedicated_gb_by_adapter"),
                      "pid_pct": gpu_ev.get("pid_engine_pct_by_adapter"), "prompt_equal": sanity}, ensure_ascii=False))
    if gpu_ev.get("verdict") == "REPROVADO":
        print("ERRO: contadores do Windows indicam que a geracao NAO rodou na GPU dedicada; descarte estas predicoes.", file=sys.stderr)
        return 3
    return 0


# -- fase score (env/.venv; Mongo no WSL) -----------------------------------------------------------------------------------------------
def build_exec_text(item: dict, middle: str) -> tuple[str, str]:
    """(doc, texto_executavel). `doc` = prefix + middle + suffix (CRLF->LF, como eval_teacher). EJSON (dialect json) executa como
    `db.getCollection(c).aggregate(EJSON.parse("<doc>"))` -- o JSON e o texto real, interpretado como extended JSON; JSON invalido falha na execucao."""
    doc = (item["prefix"] + middle + item["suffix"]).replace("\r\n", "\n")
    if item["dialect"] == "json":
        return doc, f"db.getCollection({json.dumps(item['collection'])}).aggregate(EJSON.parse({json.dumps(doc)}));"
    return doc, doc


def exec_key(db: str, text: str) -> str:
    return hashlib.sha1((db + "\x00" + text).encode("utf-8")).hexdigest()


def phase_score(a) -> int:
    hg.enforce()
    import validate_text as vt
    from common import load_cfg, load_schema_infos, static_check

    who = f"eval_homolog:{a.name}"
    cfg = load_cfg(None)
    infos = {i.full_id: i for i in load_schema_infos(roles=("hom",))}
    items = {r["id"]: r for r in load_items(f"avaliacao {a.name}: execucao do texto real (validate_text)", who)}
    out_dir = run_dir(a.name)
    if a.name == REFERENCE:
        subjects = [(i, r["middle"]) for i, r in items.items()]
    else:
        subjects = [(r["id"], r["prediction"]) for r in read_jsonl(out_dir / "predictions.jsonl")]
    cache_path = RESULTS / EXEC_CACHE
    cache: dict[str, dict] = {}
    if exists(cache_path):
        cache = {r["key"]: r for r in read_jsonl(cache_path)}
    jobs, pending = {}, {}
    per_id = {}
    for iid, middle in subjects:
        it = items[iid]
        info = infos.get(it["schema_id"])
        if info is None:
            per_id[iid] = {"status": "schema", "key": None}
            continue
        doc, text = build_exec_text(it, middle)
        if not middle.strip():
            per_id[iid] = {"status": "empty", "key": None}
            continue
        why = static_check(doc, {info.schema_id, info.db})
        if why:
            per_id[iid] = {"status": "static", "key": None, "error": ",".join(why)[:200]}
            continue
        text = vt.rewrite_db(text, info.schema_id, info.db)
        key = exec_key(info.db, text)
        per_id[iid] = {"key": key}
        if key not in cache and key not in pending:
            pending[key] = {"id": key, "db": info.db, "text": text, "schema": info.full_id}
    print(f"[score {a.name}] {len(subjects)} itens; {len(pending)} execucoes novas ({len(subjects) - len(pending)} em cache/estaticas)", flush=True)
    meta = {"name": a.name, "validator": "validate_text (mongosh no WSL + ro_proxy) sobre prefix+pred+suffix", "executions_new": len(pending)}
    if pending:
        new = [p for p in pending.values()]
        # sintaxe (node vm.Script) antes: um bloco sintaticamente quebrado no REPL contaminaria os seguintes
        sx = vt.syntax_check([{"id": p["id"], "text": p["text"]} for p in new])
        batch_items, fresh = [], []
        for p in new:
            ok, err = sx.get(p["id"], (False, "sem resposta do node"))
            if ok:
                batch_items.append(p)
            else:
                fresh.append({"key": p["id"], "status": "syntax", "error": str(err)[:200], "n_docs": 0, "unknown_fields": [], "unknown_collections": []})
        import pymongo
        ex = cfg["execution"]
        mongo = pymongo.MongoClient("mongodb://" + ex["mongo_host"], serverSelectionTimeoutMS=4000, appname="kapi-eval-homolog-fp")
        dbs = sorted({p["db"] for p in batch_items})
        fp_before, op_before = vt.fingerprint(mongo, dbs), vt.opcounters(mongo)
        work = out_dir / "_exec"
        bs = int(ex["batch_size"])
        batches = [(work / f"b{n:05d}", [{"id": p["id"], "db": p["db"], "text": p["text"]} for p in batch_items[i:i + bs]], cfg)
                   for n, i in enumerate(range(0, len(batch_items), bs))]
        proxy_total = {"allowed": collections.Counter(), "blocked": collections.Counter(), "server_errors": 0}
        info_by_key = {p["id"]: infos[p["schema"]] for p in batch_items}
        t0 = time.time()
        write_lock = threading.Lock()
        cache_fh = None
        with authz("abrir cache de execucao (data/homolog/results)", who):
            cache_fh = cache_path.open("a", encoding="utf-8", newline="\n")
        from concurrent.futures import ThreadPoolExecutor

        def run_batch(arg):      # a autorizacao e por thread: cada lote grava/le seus arquivos temporarios dentro de data/homolog/results
            with authz("execucao em lote (arquivos temporarios de lote em data/homolog/results)", who):
                return vt.run_batch(arg)
        if True:
            with ThreadPoolExecutor(int(a.workers or ex["workers"])) as pool:
                for k, out in enumerate(pool.map(run_batch, batches), 1):
                    for key in ("allowed", "blocked"):
                        for c, v in (out["proxy"].get(key) or {}).items():
                            proxy_total[key][c] += v
                    proxy_total["server_errors"] += out["proxy"].get("server_errors", 0)
                    ents = []
                    for kk, res in out["results"].items():
                        e = {"key": kk, "status": res["status"], "error": (res.get("error") or "")[:200], "n_docs": res.get("n_docs", 0),
                             "unknown_fields": [], "unknown_collections": []}
                        if res.get("commands"):
                            an = vt.analyze(None, res, info_by_key[kk])
                            e["unknown_fields"], e["unknown_collections"] = an["unknown_fields"][:12], an["unknown_collections"][:6]
                        ents.append(e)
                    got = {e["key"] for e in ents}
                    for it in batches[k - 1][1]:
                        if it["id"] not in got:
                            ents.append({"key": it["id"], "status": "error", "error": "sem resultado (falha do host WSL)", "n_docs": 0,
                                         "unknown_fields": [], "unknown_collections": []})
                    with write_lock:
                        for e in ents:
                            cache_fh.write(json.dumps(e, ensure_ascii=False) + "\n")
                            cache[e["key"]] = e
                        cache_fh.flush()
                    if out["rc"] != 0:
                        print(f"[exec] lote {k}: rc={out['rc']} {out['stderr'][-200:]}", flush=True)
                    if k % 5 == 0 or k == len(batches):
                        print(f"[exec] lote {k}/{len(batches)}  ({time.time() - t0:.0f}s)", flush=True)
        cache_fh.close()
        with authz("registrar sintaxe invalida no cache de execucao", who):
            with cache_path.open("a", encoding="utf-8", newline="\n") as fh2:
                for e in fresh:
                    fh2.write(json.dumps(e, ensure_ascii=False) + "\n")
                    cache[e["key"]] = e
        fp_after, op_after = vt.fingerprint(mongo, dbs), vt.opcounters(mongo)
        mongo.close()
        import shutil
        with authz("remover lotes temporarios", who):
            shutil.rmtree(work, ignore_errors=True)
        meta.update({"readonly_proof": {
            "dbHash_before_equals_after": fp_before == fp_after, "dbs_checked": len(dbs),
            "server_opcounters_delta": {k: op_after.get(k, 0) - op_before.get(k, 0) for k in ("insert", "update", "delete")},
            "proxy_allowed": dict(proxy_total["allowed"]), "proxy_blocked": dict(proxy_total["blocked"]),
            "proxy_write_commands_forwarded": {k: v for k, v in proxy_total["allowed"].items()
                                               if k in ("insert", "update", "delete", "findandmodify", "createindexes", "drop", "dropdatabase", "create")},
            "maxTimeMS_injected": ex["max_time_ms"]}, "exec_wall_seconds": round(time.time() - t0, 1)})
    rows = []
    for iid, _ in subjects:
        p = per_id[iid]
        c = cache.get(p["key"]) if p.get("key") else None
        if c is None:
            rows.append({"id": iid, "status": p["status"], "error": p.get("error", ""), "n_docs": 0, "unknown_fields": [], "unknown_collections": []})
        else:
            rows.append({"id": iid, **{k: c[k] for k in ("status", "error", "n_docs", "unknown_fields", "unknown_collections")}})
    write_jsonl(out_dir / "exec.jsonl", rows)
    meta["status_distribution"] = dict(collections.Counter(r["status"] for r in rows))
    write_json(out_dir / "score_meta.json", meta)
    print(json.dumps(meta, ensure_ascii=False, default=str)[:1500])
    return 0


# -- agregacao ----------------------------------------------------------------------------------------------------------------------------
def mean(vals) -> float | None:
    vals = [float(v) for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def exec_flags(ex: dict | None, empty: bool) -> tuple[bool, bool]:
    """(exec_valid estrito, campo inexistente). exec_valid: executou (status ok, comando de leitura enviado), sem campo/colecao inexistente e predicao nao vazia."""
    if ex is None:
        return False, False
    halluc = bool(ex["unknown_fields"] or ex["unknown_collections"])
    return (ex["status"] == "ok" and not halluc and not empty), halluc


def merge_rows(preds: list[dict], exe: list[dict], ref_exe: list[dict]) -> list[dict]:
    e = {r["id"]: r for r in exe}
    f = {r["id"]: r for r in ref_exe}
    out = []
    for p in preds:
        m = p["metrics"]
        ev, hal = exec_flags(e.get(p["id"]), bool(m["empty"]))
        rv, _ = exec_flags(f.get(p["id"]), False)
        out.append({"id": p["id"], "cut_kind": p["cut_kind"], "context_kind": p["context_kind"], "domain": p["domain"], "author": p["author"],
                    "normalized_match": bool(m["normalized_match"]), "exact_match": bool(m["exact_match"]), "edit_similarity": float(m["edit_similarity"]),
                    "accepted_prefix_tokens": float(m["accepted_prefix_tokens"]), "syntax_valid": bool(m.get("syntax_valid")),
                    "mongo_valid": m.get("mongo_valid"), "empty": bool(m["empty"]), "exec_valid": ev, "field_hallucination": hal, "ref_exec_valid": rv,
                    "ref_syntax_valid": bool(p.get("reference_checks", {}).get("syntax_valid"))})
    return out


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    ev, ref = mean(r["exec_valid"] for r in rows), mean(r["ref_exec_valid"] for r in rows)
    both = [r for r in rows if r["ref_exec_valid"]]
    return {"n": len(rows), "exact_match": mean(r["exact_match"] for r in rows), "normalized_match": mean(r["normalized_match"] for r in rows),
            "edit_similarity": mean(r["edit_similarity"] for r in rows), "accepted_prefix_tokens": mean(r["accepted_prefix_tokens"] for r in rows),
            "syntax_valid": mean(r["syntax_valid"] for r in rows), "mongo_valid": mean(r["mongo_valid"] for r in rows),
            "exec_valid": ev, "ceiling_exec_valid": ref, "exec_over_ceiling": (round(ev / ref, 4) if ev is not None and ref else None),
            "exec_given_ref_ok": mean(r["exec_valid"] for r in both), "n_ref_ok": len(both),
            "nonexistent_fields_rate": mean(r["field_hallucination"] for r in rows), "empty_rate": mean(r["empty"] for r in rows)}


def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0 or p is None:
        return (float("nan"), float("nan"))
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def ratio_ci(rows: list[dict], resamples: int = 2000, seed: int = 20261023) -> tuple[float, float]:
    """IC95% (bootstrap de itens) de exec ÷ teto."""
    rng = random.Random(seed)
    rows = sorted(rows, key=lambda r: r["id"])        # independe da ordem de entrada
    n = len(rows)
    ev = [1 if r["exec_valid"] else 0 for r in rows]
    rf = [1 if r["ref_exec_valid"] else 0 for r in rows]
    out = []
    for _ in range(resamples):
        idx = rng.choices(range(n), k=n)
        den = sum(rf[i] for i in idx)
        out.append(sum(ev[i] for i in idx) / den if den else float("nan"))
    out.sort()
    return round(out[int(0.025 * resamples)], 4), round(out[int(0.975 * resamples)], 4)


def summarize(rows: list[dict]) -> dict:
    s = {"overall": aggregate(rows)}
    for g in GROUPS:
        buckets = collections.defaultdict(list)
        for r in rows:
            buckets[str(r[g])].append(r)
        s["by_" + g] = {k: aggregate(v) for k, v in sorted(buckets.items())}
    cross = collections.defaultdict(list)
    for r in rows:
        cross[f"{r['author']}/{r['context_kind']}"].append(r)
    s["by_author_context"] = {k: aggregate(v) for k, v in sorted(cross.items())}
    lo, hi = wilson(s["overall"]["normalized_match"], len(rows))
    s["overall"]["normalized_match_ci95"] = [lo, hi]
    s["overall"]["exec_over_ceiling_ci95"] = list(ratio_ci(rows))
    return s


def to_cmp(rows: list[dict]) -> list[dict]:
    return [{"id": r["id"], "cut_kind": r["cut_kind"], "context_kind": r["context_kind"], "domain": r["domain"], "author": r["author"],
             "metrics": {"normalized_match": r["normalized_match"], "exact_match": r["exact_match"], "edit_similarity": r["edit_similarity"],
                         "accepted_prefix_tokens": r["accepted_prefix_tokens"], "syntax_valid": r["syntax_valid"], "mongo_valid": bool(r["mongo_valid"]),
                         "exec_valid": r["exec_valid"], "no_nonexistent_field": not r["field_hallucination"]}} for r in rows]


def load_model_rows(name: str) -> list[dict]:
    d = run_dir(name)
    return merge_rows(read_jsonl(d / "predictions.jsonl"), read_jsonl(d / "exec.jsonl"), read_jsonl(run_dir(REFERENCE) / "exec.jsonl"))


# -- relatorio ------------------------------------------------------------------------------------------------------------------------------
def pct(v, digits: int = 1) -> str:
    return "n/d" if v is None else f"{100 * v:.{digits}f}"


def val_headline(path: str) -> dict | None:
    p = ROOT / path
    if not p.exists():
        return None
    s = json.loads(p.read_text(encoding="utf-8"))
    h, ceil = s["headline"], (s.get("reference_ceiling") or {}).get("exec_valid")
    full = (s.get("by_context_kind") or {}).get("full") or {}
    return {"n": s["n"], "normalized_match": h["normalized_match"], "exact_match": h["exact_match"], "edit_similarity": h["edit_similarity"],
            "syntax_valid": h["syntax_valid"], "mongo_valid": h["mongo_valid"], "exec_valid": h["exec_valid"], "ceiling_exec_valid": ceil,
            "exec_over_ceiling": round(h["exec_valid"] / ceil, 4) if ceil and h["exec_valid"] is not None else None,
            "nonexistent_fields_rate": h["nonexistent_fields_rate"], "nonexistent_fields_full": full.get("nonexistent_fields_rate"),
            "by_context_kind": {k: {"normalized_match": v["normalized_match"], "exec_valid": v["exec_valid"]} for k, v in (s.get("by_context_kind") or {}).items()}}


def table_overall(res: dict, labels: dict) -> list[str]:
    L = ["| modelo | n | FIM acc (norm.) IC95 | exact | edit sim. | sintaxe | mongo-style | exec-valid | teto | exec ÷ teto IC95 | campos inex. | campos inex. (full) |",
         "|---|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    for name, s in res.items():
        o = s["overall"]
        full = s["by_context_kind"].get("full", {})
        L.append(f"| {labels[name]} | {o['n']} | {pct(o['normalized_match'])} [{pct(o['normalized_match_ci95'][0])}, {pct(o['normalized_match_ci95'][1])}] | "
                 f"{pct(o['exact_match'])} | {pct(o['edit_similarity'])} | {pct(o['syntax_valid'])} | {pct(o['mongo_valid'])} | {pct(o['exec_valid'])} | "
                 f"{pct(o['ceiling_exec_valid'])} | {pct(o['exec_over_ceiling'])} [{pct(o['exec_over_ceiling_ci95'][0])}, {pct(o['exec_over_ceiling_ci95'][1])}] | "
                 f"{pct(o['nonexistent_fields_rate'])} | {pct(full.get('nonexistent_fields_rate'))} |")
    return L


def table_group(res: dict, labels: dict, group: str, order=None) -> list[str]:
    keys = list(order) if order else sorted({k for s in res.values() for k in s["by_" + group]})
    L = [f"| {group} (n) | " + " | ".join(labels[n] for n in res) + " |", "|---|" + "---:|" * len(res)]
    first = next(iter(res.values()))
    for k in keys:
        cells = []
        for n, s in res.items():
            a = s["by_" + group].get(k)
            cells.append("n/d" if not a else f"{pct(a['normalized_match'])} / {pct(a['exec_over_ceiling'])} / {pct(a['nonexistent_fields_rate'])}")
        L.append(f"| {k} ({first['by_' + group].get(k, {}).get('n', 0)}) | " + " | ".join(cells) + " |")
    return L


def table_gap(res: dict, labels: dict, val: dict) -> list[str]:
    L = ["| modelo | FIM acc val → hom (Δ pp) | edit sim. val → hom (Δ) | sintaxe val → hom (Δ) | exec ÷ teto val → hom (Δ pp) | campos inex. full val → hom (Δ pp) |",
         "|---|---|---|---|---|---|"]
    for n, s in res.items():
        v, o = val.get(n), s["overall"]
        if not v:
            continue
        full = s["by_context_kind"].get("full", {})

        def cell(a, b):
            return "n/d" if a is None or b is None else f"{pct(a)} → {pct(b)} ({100 * (b - a):+.1f})"
        L.append(f"| {labels[n]} | {cell(v['normalized_match'], o['normalized_match'])} | {cell(v['edit_similarity'], o['edit_similarity'])} | "
                 f"{cell(v['syntax_valid'], o['syntax_valid'])} | {cell(v['exec_over_ceiling'], o['exec_over_ceiling'])} | "
                 f"{cell(v['nonexistent_fields_full'], full.get('nonexistent_fields_rate'))} |")
    return L


def compare_pairs(names: list[str], rows_by: dict, min_group_n: int = 50) -> dict:
    import compare_models
    tmp = RESULTS / "_cmp"
    margins = {"normalized_match": 0.02, "exec_valid": 0.02}
    cmp_files = {}
    for n in names:
        p = tmp / f"{n}.cmp.jsonl"
        write_jsonl(p, to_cmp(rows_by[n]))
        cmp_files[n] = p
    out = {}
    for a, b in PAIRS:
        if a in cmp_files and b in cmp_files:
            with authz("comparacao pareada (ids e booleanos por item em data/homolog/results)", "eval_homolog:report"):
                out[f"{b}_vs_{a}"] = {g: compare_models.run(cmp_files[a], cmp_files[b], group_by=g, min_group_n=min_group_n, margins=margins)
                                      for g in ("context_kind", "author")}
    return out


def table_pairs(pairs: dict, labels: dict) -> list[str]:
    keys = ("normalized_match", "exec_valid", "edit_similarity", "no_nonexistent_field")
    L = ["| B − A (A → B) | n | " + " | ".join(f"Δ {k} [IC95]" for k in keys) + " | McNemar p (FIM / exec) |", "|---|---:|" + "---|" * len(keys) + "---|"]
    for name, by in pairs.items():
        res = by["context_kind"]
        b, a = name.split("_vs_")
        ov = {r["metric"]: r for r in res["overall"]}
        cells = [f"{100 * ov[k]['delta']:+.1f} [{100 * ov[k]['ci_low']:+.1f}, {100 * ov[k]['ci_high']:+.1f}]" if k in ov and k != "edit_similarity"
                 else (f"{ov[k]['delta']:+.3f} [{ov[k]['ci_low']:+.3f}, {ov[k]['ci_high']:+.3f}]" if k in ov else "n/d") for k in keys]
        ps = " / ".join(f"{ov[k]['p_holm']:.3g}" if k in ov and "p_holm" in ov[k] else "n/d" for k in ("normalized_match", "exec_valid"))
        L.append(f"| {labels.get(b, b)} − {labels.get(a, a)} | {res['n_aligned']} | " + " | ".join(cells) + f" | {ps} (Holm) |")
    return L


def table_pairs_group(pairs: dict, labels: dict, group: str, metric: str) -> list[str]:
    keys = sorted({g for by in pairs.values() for g in by[group]["groups"]})
    L = [f"| B − A | " + " | ".join(f"{group}={k}" for k in keys) + " |", "|---|" + "---|" * len(keys)]
    for name, by in pairs.items():
        b, a = name.split("_vs_")
        cells = []
        for k in keys:
            blk = by[group]["groups"].get(k)
            r = next((x for x in blk["metrics"] if x["metric"] == metric), None) if blk else None
            cells.append("n/d" if not r else f"{100 * r['delta']:+.1f} [{100 * r['ci_low']:+.1f}, {100 * r['ci_high']:+.1f}]")
        L.append(f"| {labels.get(b, b)} − {labels.get(a, a)} | " + " | ".join(cells) + " |")
    return L


TABLES_BEGIN, TABLES_END = "<!-- TABLES:BEGIN (gerado por scripts/eval/eval_homolog.py report; nao edite) -->", "<!-- TABLES:END -->"


def phase_report(a) -> int:
    hg.enforce()
    names = a.names or [n for n, _, _ in MODELS if exists(run_dir(n) / "exec.jsonl") and exists(run_dir(n) / "predictions.jsonl")]
    if not exists(run_dir(REFERENCE) / "exec.jsonl"):
        raise SystemExit("falta o teto: rode `score --name _reference`")
    labels = {n: l for n, l, _ in MODELS}
    rows_by = {n: load_model_rows(n) for n in names}
    ids = {n: {r["id"] for r in rows_by[n]} for n in names}
    common = set.intersection(*ids.values())
    if any(len(v) != len(common) for v in ids.values()):
        print(f"AVISO: conjuntos de ids diferentes; reportando na interseccao ({len(common)})", file=sys.stderr)
        rows_by = {n: [r for r in v if r["id"] in common] for n, v in rows_by.items()}
    res = {n: summarize(rows_by[n]) for n in names}
    val = {n: val_headline(p) for n, _, p in MODELS if n in names}
    pairs = compare_pairs(names, rows_by)
    meta = {n: read_json(run_dir(n) / "gen_meta.json") for n in names if exists(run_dir(n) / "gen_meta.json")}
    score_meta = {n: read_json(run_dir(n) / "score_meta.json") for n in names + [REFERENCE] if exists(run_dir(n) / "score_meta.json")}
    agg = {"format": "homolog-v1-eval-1", "set_sha256": hg.manifest().get("set_sha256"), "n": len(common), "models": res, "val_v2c": val,
           "pairs": {k: {g: {"n_aligned": v["n_aligned"], "overall": v["overall"], "groups": v["groups"], "groups_skipped": v["groups_skipped"]}
                         for g, v in by.items()} for k, by in pairs.items()},
           "gpu_evidence": {n: {"device": m["gpu"].get("device_name"), "load": m["load"], "windows_counters": {k: m["gpu_counters_windows"].get(k) for k in
                                ("verdict", "dedicated_gb_by_adapter", "pid_engine_pct_by_adapter", "share_on_largest_adapter")},
                                "prompt_equals_sealed_prompt": m.get("prompt_equals_sealed_prompt")} for n, m in meta.items()},
           "readonly_proof": {n: m.get("readonly_proof") for n, m in score_meta.items() if m.get("readonly_proof")},
           "status_distribution": {n: m.get("status_distribution") for n, m in score_meta.items()},
           "definitions": {
               "normalized_match": "pred == middle com espacos normalizados", "exec_valid": "prefix+pred+suffix executa (mongosh via proxy RO, status ok, comando de leitura enviado) sem campo/colecao inexistente no plano capturado e pred nao vazia",
               "ceiling_exec_valid": "mesma validacao sobre prefix+middle+suffix", "exec_over_ceiling": "exec_valid / ceiling_exec_valid (razao de medias)",
               "nonexistent_fields_rate": "itens cujo plano capturado cita campo/colecao fora do schema, sobre todos os itens",
               "json_dialect": "dialect json executa como db.getCollection(c).aggregate(EJSON.parse(texto))"}}
    write_json(REPORT_DIR / "homolog-v1-eval.json", agg)
    L = ["### T1. Homologacao v1 (n=%d), geral" % len(common), ""] + table_overall(res, labels)
    L += ["", "Celulas dos grupos abaixo: FIM acc / exec ÷ teto / campos inexistentes (em %).", ""]
    for g, order in (("context_kind", CTX_ORDER), ("author", None), ("domain", None), ("cut_kind", None)):
        L += [f"### T2. Por {g}", ""] + table_group(res, labels, g, order) + [""]
    L += ["### T2b. author × context_kind", ""]
    keys = sorted({k for s in res.values() for k in s["by_author_context"]})
    L += ["| author/context (n) | " + " | ".join(labels[n] for n in res) + " |", "|---|" + "---:|" * len(res)]
    for k in keys:
        L.append(f"| {k} ({next(iter(res.values()))['by_author_context'].get(k, {}).get('n', 0)}) | " +
                 " | ".join(f"{pct(s['by_author_context'][k]['normalized_match'])} / {pct(s['by_author_context'][k]['exec_over_ceiling'])} / "
                            f"{pct(s['by_author_context'][k]['nonexistent_fields_rate'])}" if k in s["by_author_context"] else "n/d" for s in res.values()) + " |")
    L += ["", "### T3. Lacuna de generalizacao (val v2c sintetico → homologacao)", ""] + table_gap(res, labels, val)
    L += ["", "(O teto do val v2c vem do validador `chat_validate` e o da homologacao do `validate_text`: a razao exec ÷ teto cruza dois validadores; "
          "compare tendencias, nao a segunda casa.)", ""]
    if pairs:
        L += ["### T4. Comparacao pareada (mesmos 100% dos itens; Δ em pp, IC95 bootstrap pareado; p McNemar-Holm)", ""] + table_pairs(pairs, labels)
        for g in ("author", "context_kind"):
            for metric in ("normalized_match", "exec_valid"):
                L += ["", f"#### T4 por {g}: Δ {metric} (pp)", ""] + table_pairs_group(pairs, labels, g, metric)
    gpu_lines = ["", "### T5. Procedencia", "", "| execucao | dispositivo (torch) | contadores do Windows | GB dedicados por adaptador | uso do pid por adaptador (%) | prompt = prompt selado (100 amostras) |", "|---|---|---|---|---|---|"]
    for n, m in meta.items():
        w = m.get("gpu_counters_windows") or {}
        gpu_lines.append(f"| {labels.get(n, n)} | {m['gpu'].get('device_name')} | {w.get('verdict')} | {w.get('dedicated_gb_by_adapter')} | {w.get('pid_engine_pct_by_adapter')} | "
                         f"{(m.get('prompt_equals_sealed_prompt') or {}).get('prompt_equal')}/{(m.get('prompt_equals_sealed_prompt') or {}).get('checked')} |")
    L += gpu_lines
    block = "\n".join(L)
    md_path = REPORT_DIR / "homolog-v1-eval.md"
    text = md_path.read_text(encoding="utf-8") if md_path.exists() else f"# Avaliacao no conjunto de homologacao v1\n\n{TABLES_BEGIN}\n{TABLES_END}\n"
    if TABLES_BEGIN in text and TABLES_END in text:
        pre, rest = text.split(TABLES_BEGIN, 1)
        post = rest.split(TABLES_END, 1)[1]
        text = pre + TABLES_BEGIN + "\n" + block + "\n" + TABLES_END + post
    else:
        text += "\n" + TABLES_BEGIN + "\n" + block + "\n" + TABLES_END + "\n"
    write_text(md_path, text)
    sys.stdout.buffer.write((block + chr(10)).encode("utf-8"))
    return 0


def phase_run(a) -> int:
    cmd = [sys.executable, "-I", str(Path(__file__).resolve()), "gen", "--model", a.model, "--name", a.name, "--load", a.load]
    if a.adapter:
        cmd += ["--adapter", a.adapter]
    if a.limit:
        cmd += ["--limit", str(a.limit)]
    if a.resume:
        cmd += ["--resume"]
    print("$", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        return rc
    if not exists(run_dir(REFERENCE) / "exec.jsonl"):
        subprocess.run([str(ENV_VENV_PY), "-I", str(Path(__file__).resolve()), "score", "--name", REFERENCE], check=True)
    return subprocess.run([str(ENV_VENV_PY), "-I", str(Path(__file__).resolve()), "score", "--name", a.name], check=False).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("gen", "run"):
        p = sub.add_parser(cmd)
        p.add_argument("--model", required=True)
        p.add_argument("--name", required=True)
        p.add_argument("--adapter")
        p.add_argument("--load", choices=["auto", "bf16", "nf4"], default="auto")
        p.add_argument("--limit", type=int)
        p.add_argument("--resume", action="store_true")
    p = sub.add_parser("score")
    p.add_argument("--name", required=True, help=f"nome da execucao ou {REFERENCE} (teto: executa o middle de referencia)")
    p.add_argument("--workers", type=int)
    p = sub.add_parser("report")
    p.add_argument("--names", nargs="*")
    a = ap.parse_args(argv)
    return {"gen": phase_gen, "score": phase_score, "report": phase_report, "run": phase_run}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
