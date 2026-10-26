#!/usr/bin/env python3
"""Prepara (e so prepara) o pacote de publicacao em PREVIEW PRIVADO no Hugging Face. Sem rede, sem token, sem upload.

Repositorios (decisao do usuario): `esilva/KapiCoder-Mongo-Nano` e `esilva/KapiCoder-Mongo-Small` (safetensors, pesos v3 = merged/sft_v3);
DEPOIS (desligado por padrao) `esilva/KapiCoder-Mongo-Nano-ONNX` e `esilva/KapiCoder-Mongo-Small-ONNX`. Sem versao no nome: versoes por tag git,
primeira tag `v0.1.0-preview`. Nao entram: professor, dataset, GGUF.

  python scripts/publish/stage_hf_package.py --tier nano                   # dry-run (padrao): lista arquivos, tamanhos, sha256 e checagens
  python scripts/publish/stage_hf_package.py --tier both --no-hash         # rapido: so tamanhos e checagens de texto (sem sha256 dos pesos)
  python scripts/publish/stage_hf_package.py --tier both --write-manifest  # grava Repos/*/hf/upload_manifest.json (todos os sha256)
  python scripts/publish/stage_hf_package.py --tier nano --check-manifest  # compara o plano atual com hf/upload_manifest.json
  python scripts/publish/stage_hf_package.py --tier both --stage           # monta cache/hf_stage/<tier>/ (fora do Git; diretorio vazio) e re-verifica os hashes
  python scripts/publish/stage_hf_package.py --tier nano --stage --clean   # idem, apagando antes um staging anterior (so dentro de cache/hf_stage)
  python scripts/publish/stage_hf_package.py --tier nano --target onnx     # (DEPOIS) plano do repositorio ONNX separado; exige os pacotes v3 completos

Arvore do repositorio principal (raiz plana; qualquer divergencia = erro):
  README.md LICENSE NOTICE.md REPLAY_ATTRIBUTION.md config.json generation_config.json model.safetensors tokenizer.json tokenizer_config.json vocab.json merges.txt
  provenance/{model_lock.json,upload_manifest.json}   evaluation/{val_v2c_summary.json,homologation_v1_summary.json}   examples/example_fim.py
Arquivos gerados aqui (nao existem no disco do workspace): provenance/*, evaluation/*, examples/example_fim.py.

Regras (qualquer violacao = exit 2, nada e copiado):
  * so entram arquivos do workspace do proprio tier: merged/sft_v3/ (pesos e tokenizer), hf/ (card, NOTICE, atribuicao), LICENSE; no modo ONNX, os pacotes de exports/;
  * lista de recusa: train_state.json, export_info.json, run_manifest.json, model.lock.json, *.jsonl, *.npz, checkpoints, otimizador, GGUF, notas internas;
  * varredura de conteudo: caminhos locais, usuario/e-mail da maquina, tokens/segredos, URI MongoDB com senha (texto, binarios < 16 MB, cabecalho do safetensors);
  * pesos e tokenizer conferem com `export_info.json`; o LICENSE confere com o da base; o card tem front matter YAML valido.

Este script nao instala nada, nao le variaveis de token e nao abre sockets. O envio e feito (se um dia for aprovado) por `upload_to_hf.py`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

CHUNK = 8 * 1024 * 1024
SCAN_MAX_BYTES = 16 * 1024 * 1024
TEXT_EXT = {".json", ".md", ".txt"}
ALLOWED_EXT = TEXT_EXT | {".safetensors", ".onnx", ".data", ".py"}
ALLOWED_NO_EXT = {"LICENSE"}
ALLOWED_PY = {"examples/example_fim.py"}
VERSION_TAG = "v0.1.0-preview"
DEFAULT_STAGE_BASE = ROOT / "cache" / "hf_stage"
TEMPLATE_EXAMPLE = SCRIPTS_DIR / "templates" / "example_fim.py"
GOLDEN = ROOT / "data" / "contracts" / "editor-context-v1.golden"

CORE_FILES = ("config.json", "generation_config.json", "merges.txt", "model.safetensors",
              "tokenizer.json", "tokenizer_config.json", "vocab.json")
CARD_FILES = ("README.md", "NOTICE.md", "REPLAY_ATTRIBUTION.md")
GENERATED_FILES = ("provenance/model_lock.json", "provenance/upload_manifest.json",
                   "evaluation/val_v2c_summary.json", "evaluation/homologation_v1_summary.json", "examples/example_fim.py")
MAIN_TREE = frozenset(CORE_FILES + CARD_FILES + ("LICENSE",) + GENERATED_FILES)
INTERNAL_HF_FILES = ("MODEL_CARD_REVIEW.md", "SOURCES.md", "upload_manifest.json", "upload_manifest_onnx.json", "ONNX_README.md", "ONNX_NOTICE.md")

# ONNX (repositorio separado, DEPOIS): pasta de destino -> sufixo do pacote em exports/ (`<nome>-v3-ONNX-<sufixo>`)
ONNX_ROOT_FILES = ("README.md", "LICENSE", "NOTICE.md")
ONNX_REQUIRED_PER_VARIANT = ("model.onnx", "genai_config.json", "tokenizer.json", "kapibarastudio-model.json", "export_manifest.json", "kapicoder_manifest.json")

REFUSE_NAMES = {
    "train_state.json", "export_info.json", "run_manifest.json", "model.lock.json", "best.json",
    "heartbeat.json", "metrics.jsonl", "eval_history.jsonl", "predictions.jsonl", "optimizer.pt", "rng_state.pt",
    "SOURCES.md", "MODEL_CARD_REVIEW.md", "README_job.md", ".env", ".netrc", "ONNX_README.md", "ONNX_NOTICE.md",
}
REFUSE_SUFFIXES = (".jsonl", ".npz", ".npy", ".parquet", ".pt", ".pth", ".bin", ".ckpt", ".gguf", ".pkl", ".pickle", ".log", ".csv", ".zip", ".gz", ".ps1", ".yaml", ".yml")
REFUSE_PARTS = {"checkpoints", "adapters", "data", "tmp", "logs", "reports", "gguf", "__pycache__", ".git"}

TIER_SPECS = {
    "nano": {"name": "KapiCoder-Mongo-Nano", "ws": "Repos/KapiCoder-Mongo-Nano", "run": "sft_v3", "tier": "T0", "t": "t0",
             "repo": "esilva/KapiCoder-Mongo-Nano", "onnx_repo": "esilva/KapiCoder-Mongo-Nano-ONNX", "base": "Qwen2.5-Coder-0.5B",
             "onnx_variants": {"cpu-int4": "int4", "cpu-int8": "int8", "dml-fp16": "dml-fp16"}},
    "small": {"name": "KapiCoder-Mongo-Small", "ws": "Repos/KapiCoder-Mongo-Small", "run": "sft_v3", "tier": "T1", "t": "t1",
              "repo": "esilva/KapiCoder-Mongo-Small", "onnx_repo": "esilva/KapiCoder-Mongo-Small-ONNX", "base": "Qwen2.5-Coder-1.5B",
              "onnx_variants": {"cpu-int8": "int8", "dml-fp16": "dml-fp16"}},
}

# --- padroes de varredura -------------------------------------------------------------------------------------------------------
PATH_PATTERNS = {
    "caminho Windows": re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:(?:\\|/)(?!/)"),
    "caminho de usuario": re.compile(r"(/Users/|/home/[A-Za-z]|\\Users\\|\\\\Users\\\\)"),
    "caminho do workspace": re.compile(r"(Repos[\\/]+Kapi|BaseRepos|SlopStudio|KapibaraStudio\.MongoIA|MongoIA[\\/]|\bchuke\b)"),
    "identificador pessoal": re.compile(r"(?i)(chukeduardo|gmail\.com)"),
}
SECRET_PATTERNS = {
    "token Hugging Face": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "atribuicao HF_TOKEN": re.compile(r"HF_TOKEN\s*[=:]\s*[\"']?\S{8,}"),
    "token GitHub": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    "chave sk-": re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "chave AWS": re.compile(r"AKIA[0-9A-Z]{16}"),
    "chave privada": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "bearer token": re.compile(r"(?i)bearer\s+[a-z0-9._\-]{20,}"),
    "api_key literal": re.compile(r"(?i)api[_-]?key\s*[:=]\s*[\"']?[A-Za-z0-9_\-]{12,}"),
    "URI MongoDB com senha": re.compile(r"mongodb(?:\+srv)?://[^/\s:@]+:[^@\s]+@"),
}
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}")
LOCAL_PATH_TOKEN = re.compile(r"(?:[A-Za-z]:[\\/]|(?:Base)?Repos[\\/]|tmp[\\/])[^\s\"']*")


@dataclass
class Entry:
    remote: str                  # caminho no repositorio HF
    source: Path | None          # arquivo local (absoluto); None = gerado
    kind: str                    # core | card | license | generated | onnx
    variant: str | None = None
    bytes: int = 0
    sha256: str | None = None
    expected_sha256: str | None = None
    data: bytes | None = None    # conteudo de arquivo gerado
    notes: list[str] = field(default_factory=list)

    def rel(self, root: Path) -> str:
        if self.source is None:
            return f"generated:{self.remote}"
        try:
            return self.source.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return str(self.source)


@dataclass
class Plan:
    tier: str
    root: Path
    spec: dict
    target: str = "main"                                     # main | onnx
    entries: list[Entry] = field(default_factory=list)
    excluded: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def repo_id(self) -> str:
        return self.spec["repo"] if self.target == "main" else self.spec["onnx_repo"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def dump_json(obj) -> bytes:
    return (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


# --- montagem do plano (repositorio principal) ----------------------------------------------------------------------------------
def ws_paths(root: Path, spec: dict) -> tuple[Path, Path, Path]:
    ws = root / spec["ws"]
    return ws, ws / "merged" / spec["run"], ws / "hf"


def build_plan(tier: str, root: Path = ROOT, spec: dict | None = None, target: str = "main") -> Plan:
    spec = spec or TIER_SPECS[tier]
    plan = Plan(tier=tier, root=root, spec=spec, target=target)
    if target == "onnx":
        return build_onnx_plan(plan)
    ws, merged, hf = ws_paths(root, spec)

    export_info = {}
    ei = merged / "export_info.json"
    if ei.exists():
        export_info = read_json(ei).get("files_sha256", {})
    else:
        plan.errors.append(f"export_info.json ausente em {merged} (necessario para conferir os hashes dos pesos)")

    for name in CORE_FILES:
        src = merged / name
        if not src.exists():
            plan.errors.append(f"arquivo obrigatorio ausente: {src}")
            continue
        plan.entries.append(Entry(name, src, "core", expected_sha256=export_info.get(name)))
    for name in CARD_FILES:
        src = hf / name
        if not src.exists():
            plan.errors.append(f"arquivo do card ausente: {src}")
            continue
        plan.entries.append(Entry(name, src, "card"))
    lic = ws / "LICENSE"
    if lic.exists():
        plan.entries.append(Entry("LICENSE", lic, "license", expected_sha256=base_license_sha(root, spec)))
    else:
        plan.errors.append(f"LICENSE ausente: {lic}")

    for name in INTERNAL_HF_FILES:
        plan.excluded.append({"source": f"{spec['ws']}/hf/{name}", "reason": "nota interna / manifesto interno: nunca sobe"})
    for name in ("train_state.json", "export_info.json"):
        if (merged / name).exists():
            plan.excluded.append({"source": f"{spec['ws']}/merged/{spec['run']}/{name}", "reason": "metadados de treino com caminhos locais: recusado"})
    for other in sorted((ws / "merged").glob("*")) if (ws / "merged").exists() else []:
        if other.is_dir() and other.name != spec["run"]:
            plan.excluded.append({"source": f"{spec['ws']}/merged/{other.name}", "reason": "outro checkpoint (so sft_v3 e publicado)"})
    plan.excluded.append({"source": f"{spec['ws']}/exports", "reason": "ONNX so em repositorio separado, depois (--target onnx)"})
    plan.excluded.append({"source": f"{spec['ws']}/gguf", "reason": "GGUF nao e publicado"})
    return plan


def base_license_sha(root: Path, spec: dict) -> str | None:
    lock = root / "BaseRepos" / "base-repos.lock.json"
    if not lock.exists():
        return None
    try:
        repos = read_json(lock)["repos"].get(spec["base"], {})
        for f in repos.get("files", []):
            if f["path"] == "LICENSE":
                return f["sha256"]
    except (KeyError, ValueError):
        return None
    return None


# --- ONNX (repositorio separado; desligado por padrao) --------------------------------------------------------------------------
def redact_paths(text: str) -> str:
    """Troca caminhos locais por seu ultimo componente (para metadados de exportacao copiados ao repositorio ONNX)."""
    def one(m: re.Match) -> str:
        parts = [p for p in re.split(r"[\\/]+", m.group(0)) if p]
        return parts[-1] if parts else ""
    return LOCAL_PATH_TOKEN.sub(one, text)


def sanitize_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, str):
        return redact_paths(obj)
    return obj


def build_onnx_plan(plan: Plan) -> Plan:
    spec, root = plan.spec, plan.root
    ws, _merged, hf = ws_paths(root, spec)
    for name, src_name in (("README.md", "ONNX_README.md"), ("NOTICE.md", "ONNX_NOTICE.md")):
        src = hf / src_name
        if src.exists():
            plan.entries.append(Entry(name, src, "card"))
        else:
            plan.errors.append(f"card do repositorio ONNX ausente: {src}")
    lic = ws / "LICENSE"
    if lic.exists():
        plan.entries.append(Entry("LICENSE", lic, "license", expected_sha256=base_license_sha(root, spec)))
    else:
        plan.errors.append(f"LICENSE ausente: {lic}")
    for dest, suffix in spec["onnx_variants"].items():
        pkg = ws / "exports" / f"{spec['name']}-v3-ONNX-{suffix}"
        onnx_entries(pkg, dest, plan)
    plan.excluded.append({"source": f"{spec['ws']}/merged", "reason": "pesos safetensors ficam no repositorio principal"})
    return plan


def onnx_entries(pkg: Path, dest: str, plan: Plan) -> None:
    if not pkg.exists():
        plan.errors.append(f"pacote ONNX v3 ausente: {pkg}")
        return
    man_path = pkg / "kapicoder_manifest.json"
    if not man_path.exists():
        plan.errors.append(f"kapicoder_manifest.json ausente em {pkg}")
        return
    man = read_json(man_path)
    have = set()
    for f in man["files"]:
        src = pkg / f["path"]
        if not src.exists():
            plan.errors.append(f"arquivo do manifesto ausente: {src}")
            continue
        plan.entries.append(Entry(f"{dest}/{f['path']}", src, "onnx", variant=dest, expected_sha256=f["sha256"]))
        have.add(f["path"])
    plan.entries.append(Entry(f"{dest}/kapicoder_manifest.json", man_path, "onnx", variant=dest))
    have.add("kapicoder_manifest.json")
    em = pkg / "export_manifest.json"
    if em.exists():
        data = read_json(em)
        clean = sanitize_json(data)
        if clean != data:
            clean["sanitized"] = "caminhos locais trocados pelo ultimo componente ao montar o staging; o original fica so no workspace"
        plan.entries.append(Entry(f"{dest}/export_manifest.json", None, "onnx", variant=dest, data=dump_json(clean),
                                  notes=["gerado a partir de export_manifest.json, sem caminhos locais"]))
        have.add("export_manifest.json")
    for must in ONNX_REQUIRED_PER_VARIANT:
        if must not in have:
            plan.errors.append(f"{dest}: arquivo obrigatorio ausente do pacote ONNX: {must}")


# --- arquivos gerados -----------------------------------------------------------------------------------------------------------
def publish_tooling_sha(root: Path = ROOT) -> str:
    h = hashlib.sha256()
    files = sorted(p for p in (root / "scripts" / "publish").rglob("*") if p.is_file() and "__pycache__" not in p.parts and "tests" not in p.parts)
    for p in files:
        h.update(p.relative_to(root / "scripts" / "publish").as_posix().encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def render_example(spec: dict) -> bytes:
    return TEMPLATE_EXAMPLE.read_text(encoding="utf-8").replace("__REPO_ID__", spec["repo"]).encode("utf-8")


def build_model_lock(plan: Plan) -> dict:
    spec, root = plan.spec, plan.root
    ws, merged, _hf = ws_paths(root, spec)
    lock = read_json(ws / "model.lock.json")
    rm = read_json(ws / "reports" / spec["run"] / "run_manifest.json")
    ei = read_json(merged / "export_info.json")
    eff = rm["config"]["effective"]
    ds = rm["dataset"]
    ev = read_json(ws / "reports" / "eval" / spec["run"] / "summary.json")
    core = {e.remote: (e.sha256 or e.expected_sha256) for e in plan.entries if e.kind == "core"}
    lora = eff.get("lora") if rm["plan"]["method"] == "lora" else None
    libs = rm["libraries"]
    return {
        "schema": "kapi-hf-model-lock/1",
        "model": {"name": spec["name"], "repo_id": spec["repo"], "generation": "v3", "release_tag": VERSION_TAG, "run": spec["run"],
                  "tier": spec["tier"], "status": "PREVIEW (private)"},
        "base_model": {"repo_id": lock["base"]["repo_id"], "revision": lock["base"]["revision"], "model_sha256": rm["base_model"]["model_sha256"], "license": "Apache-2.0"},
        "weights": {"format": "safetensors", "dtype": ei.get("dtype", "bfloat16"), "combined_sha256": ei.get("combined_sha256"), "files_sha256": dict(sorted(core.items()))},
        "training": {
            "method": rm["plan"]["method"], "master_dtype": rm["plan"]["master_dtype"], "optimizer": rm["plan"]["optimizer"],
            "peak_lr": eff["optim"]["lr"], "weight_decay": eff["optim"]["weight_decay"], "seed": eff["run"]["seed"],
            "lora": lora, "trainable_params": rm["plan"]["trainable_params"],
            "total_steps": rm["total_steps"], "steps_per_epoch": rm["plan"]["steps_per_epoch"], "tokens_seen": rm["tokens_seen"],
            "elapsed_train_h": rm["elapsed_train_h"], "started_utc": rm["started_utc"], "finished_utc": rm["finished_utc"],
            "git_commit": rm["git"]["commit"], "git_dirty": rm["git"]["dirty"],
            "train_config_sha256": rm["fingerprint"]["train_config_sha256"], "config_file_sha256": rm["config"]["sha256"],
            "libraries": {k: libs[k] for k in ("python", "torch", "transformers", "peft", "tokenizers", "safetensors") if k in libs},
            "hardware": rm["environment"]["gpu"]["device_name"], "single_training_run": True,
        },
        "dataset": {
            "name": "student corpus v3 (not published)", "published": False, "label_source": ds["label_source"],
            "examples": ds["examples"], "tokens_per_epoch": ds["tokens_in_corpus"], "replay_examples": ds["replay"]["examples"],
            "replay_token_share": ds["replay"]["token_share_achieved"],
            "corpus_manifest_sha256": ds["corpus_manifest_sha256"], "ids_sha256": ds["pairing"]["ids_sha256"],
            "rows_sha256": ds["pairing"]["rows_sha256"], "student_corpus_config_sha256": ds["student_corpus_config_sha256"],
            "accepted_sha256": ds["accepted_sha256"],
        },
        "evaluation": {
            "val_v2c": {"split": ev["split"], "n": ev["n"], "seed": ev["seed"], "sample_ids_sha256": ev["dataset"]["sample_ids_sha256"],
                        "dataset_sha256": ev["dataset"]["sha256"], "manifest_sha256": ev["dataset"]["manifest_sha256"]},
            "homologation_v1": {"set_sha256": read_json(root / "reports/datasets/homolog-v1-manifest.json")["set_sha256"], "n": 6030, "items_published": False},
            "contract": {"name": "editor-context-v1", "golden_sha256": sha256_file(GOLDEN) if GOLDEN.exists() else None},
        },
        "publish_tooling": {"scripts_publish_sha256": publish_tooling_sha(root)},
    }


_VAL_KEEP = ("n", "normalized_match", "exact_match", "edit_similarity", "syntax_valid", "mongo_valid", "exec_valid", "ceiling_exec_valid",
             "exec_over_ceiling", "nonexistent_fields_rate", "nonexistent_fields_full", "by_context_kind")


def build_val_summary(plan: Plan) -> dict:
    spec, root = plan.spec, plan.root
    ws, _m, _h = ws_paths(root, spec)
    ev = read_json(ws / "reports" / "eval" / spec["run"] / "summary.json")
    metrics = {run: read_json(ws / "reports" / "eval" / run / "measures.json")["metrics"]
               for run in ("sft_v3", "sft_distill_v1", "base-qwen2.5-coder-0.5b" if spec["t"] == "t0" else "base-qwen2.5-coder-1.5b")}
    labels = {"sft_v3": "this_model_v3", "sft_distill_v1": "previous_internal_checkpoint_v1_not_released"}
    names = {}
    for run, m in metrics.items():
        names[labels.get(run, "zero_shot_base")] = m
    return {
        "schema": "kapi-hf-eval-summary/1", "kind": "val_v2c",
        "label": "Internal evaluation on synthetic data; no blind human-written set. Aggregates only; no item text.",
        "model": spec["repo"], "split": ev["split"], "n": ev["n"], "seed": ev["seed"], "contract": ev["contract"],
        "dataset": {"sample_ids_sha256": ev["dataset"]["sample_ids_sha256"], "sha256": ev["dataset"]["sha256"], "manifest_sha256": ev["dataset"]["manifest_sha256"],
                    "stratify_by": ev["dataset"]["stratify_by"], "cut_kind": ev["dataset"]["cut_kind"], "domain": ev["dataset"]["domain"]},
        "decoding": ev["decoding"], "max_new_tokens": ev["max_new_tokens"], "context_tokens": ev["context_tokens"], "load": ev["load"],
        "this_model": {k: ev[k] for k in ("headline", "headline_ide32", "reference_ceiling", "exec_stage_distribution", "by_cut_kind", "by_domain", "by_context_kind") if k in ev},
        "metrics_by_model": names,
        "metric_definitions": ev.get("metric_definitions"),
    }


def build_hom_summary(plan: Plan) -> dict:
    spec, root = plan.spec, plan.root
    hom = read_json(root / f"reports/datasets/homolog-v3-eval-{spec['t']}.json")
    ro = {k: {kk: vv for kk, vv in v.items() if kk in ("dbHash_before_equals_after", "dbs_checked", "proxy_write_commands_forwarded", "maxTimeMS_injected")}
          for k, v in hom["readonly_proof"].items()}
    return {
        "schema": "kapi-hf-eval-summary/1", "kind": "homologation_v1",
        "label": "Internal evaluation, synthetic out-of-distribution homologation; LLM- and grammar-written, not written by people; "
                 "not fully independent of the v3 training data. Aggregates only; no item text.",
        "model": spec["repo"], "format": hom["format"], "set_sha256": hom["set_sha256"], "n": hom["n"],
        "models": hom["models"], "val_v2c": hom["val_v2c"], "pairs": hom["pairs"], "status_distribution": hom["status_distribution"],
        "readonly_proof": ro, "definitions": hom["definitions"],
    }


def generate_files(plan: Plan) -> dict[str, bytes]:
    """Conteudo dos arquivos gerados (lidos de relatorios/manifestos do workspace; nenhum caminho local entra)."""
    return {
        "examples/example_fim.py": render_example(plan.spec),
        "evaluation/val_v2c_summary.json": dump_json(build_val_summary(plan)),
        "evaluation/homologation_v1_summary.json": dump_json(build_hom_summary(plan)),
        "provenance/model_lock.json": dump_json(build_model_lock(plan)),
    }


def finalize_generated(plan: Plan) -> None:
    """Gera os arquivos que nao existem no workspace; precisa dos hashes dos demais (chamado apos o hash)."""
    if plan.target != "main":
        return
    try:
        gen = generate_files(plan)
    except Exception as exc:  # noqa: BLE001
        plan.errors.append(f"falha ao gerar arquivos de proveniencia/avaliacao: {type(exc).__name__}: {exc}")
        return
    for remote, data in gen.items():
        plan.entries.append(Entry(remote, None, "generated", data=data))
    for e in plan.entries:
        if e.data is not None and e.sha256 is None:
            e.bytes, e.sha256 = len(e.data), sha256_bytes(e.data)
    files = [{"path": e.remote, "bytes": e.bytes, "sha256": e.sha256} for e in sorted(plan.entries, key=lambda x: x.remote)]
    pm = {"schema": "kapi-hf-upload-manifest/2", "repo_id": plan.repo_id, "release_tag": VERSION_TAG, "tier": plan.tier,
          "note": "sha256 de todos os arquivos do repositorio, exceto este (provenance/upload_manifest.json)", "files": files}
    data = dump_json(pm)
    plan.entries.append(Entry("provenance/upload_manifest.json", None, "generated", data=data, bytes=len(data), sha256=sha256_bytes(data)))


# --- checagens ------------------------------------------------------------------------------------------------------------------
def check_remote_name(e: Entry, plan: Plan) -> None:
    p = PurePosixPath(e.remote)
    if p.is_absolute() or ".." in p.parts or "\\" in e.remote or re.match(r"^[A-Za-z]:", e.remote):
        plan.errors.append(f"{e.remote}: caminho remoto invalido")
    name = p.name
    if name in REFUSE_NAMES or any(part in REFUSE_PARTS for part in p.parts[:-1]):
        plan.errors.append(f"{e.remote}: nome/diretorio na lista de recusa")
    if name == "upload_manifest.json" and e.remote != "provenance/upload_manifest.json":
        plan.errors.append(f"{e.remote}: so provenance/upload_manifest.json (sem caminhos locais) pode subir")
    if name.lower().endswith(REFUSE_SUFFIXES):
        plan.errors.append(f"{e.remote}: extensao recusada")
    ext = Path(name).suffix.lower()
    if ext == ".py" and e.remote not in ALLOWED_PY:
        plan.errors.append(f"{e.remote}: script nao autorizado (so {sorted(ALLOWED_PY)})")
    elif ext not in ALLOWED_EXT and name not in ALLOWED_NO_EXT:
        plan.errors.append(f"{e.remote}: extensao fora da lista permitida ({ext or 'sem extensao'})")


def check_source_location(e: Entry, plan: Plan) -> None:
    if e.source is None:
        return
    spec, root = plan.spec, plan.root
    allowed = [root / spec["ws"] / "hf", root / spec["ws"] / "LICENSE"]
    if plan.target == "main":
        allowed.append(root / spec["ws"] / "merged" / spec["run"])
    else:
        allowed += [root / spec["ws"] / "exports" / f"{spec['name']}-v3-ONNX-{s}" for s in spec["onnx_variants"].values()]
    src = e.source.resolve()
    ok = any(src == a.resolve() or a.resolve() in src.parents for a in allowed)
    if not ok:
        plan.errors.append(f"{e.remote}: origem fora do workspace/pastas permitidas do tier: {e.source}")
    for o in (t for t in TIER_SPECS.values() if t["ws"] != spec["ws"]):
        if (root / o["ws"]).resolve() in src.parents:
            plan.errors.append(f"{e.remote}: arquivo de outro workspace ({o['name']})")


def scan_bytes(label: str, data: bytes, plan: Plan, text_checks: bool = True) -> None:
    text = data.decode("utf-8", errors="ignore")
    for name, rx in {**PATH_PATTERNS, **SECRET_PATTERNS}.items():
        m = rx.search(text)
        if m:
            snippet = text[max(0, m.start() - 20):m.end() + 20].replace("\n", " ")
            plan.errors.append(f"{label}: {name} encontrado: ...{snippet}...")
    if text_checks:
        m = EMAIL.search(text)
        if m:
            plan.errors.append(f"{label}: endereco de e-mail no texto: {m.group(0)}")


def safetensors_header(path: Path) -> dict:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n <= 0 or n > 64 * 1024 * 1024:
            raise ValueError(f"cabecalho do safetensors com tamanho absurdo ({n})")
        raw = f.read(n)
    return {"raw": raw, "json": json.loads(raw.decode("utf-8"))}


def check_entry_content(e: Entry, plan: Plan, tokenizer_shas: set[str]) -> None:
    label = e.remote
    ext = Path(e.remote).suffix.lower()
    if e.data is not None:
        e.bytes = len(e.data)
        if ext == ".json":
            try:
                json.loads(e.data.decode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                plan.errors.append(f"{label}: JSON invalido: {exc}")
        scan_bytes(label, e.data, plan)
        return
    size = e.source.stat().st_size
    e.bytes = size
    if e.source.name == "model.safetensors":
        try:
            hdr = safetensors_header(e.source)
        except Exception as exc:  # noqa: BLE001
            plan.errors.append(f"{label}: safetensors invalido: {exc}")
            return
        scan_bytes(label + " (cabecalho)", hdr["raw"], plan, text_checks=False)
        dtypes = {v.get("dtype") for k, v in hdr["json"].items() if k != "__metadata__"}
        if dtypes != {"BF16"}:
            plan.warnings.append(f"{label}: dtypes no safetensors = {sorted(map(str, dtypes))} (esperado so BF16)")
        return
    if name_is_big_binary(e):
        e.notes.append("binario grande: nao varrido por conteudo")
        return
    if size > SCAN_MAX_BYTES:
        plan.warnings.append(f"{label}: {size} bytes > limite de varredura; nao varrido")
        return
    data = e.source.read_bytes()
    if ext == ".json":
        try:
            json.loads(data.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            plan.errors.append(f"{label}: JSON invalido: {exc}")
    # tokenizer identico ao da base (hash do lock): nao varre o vocabulario, que contem sequencias arbitrarias
    if e.expected_sha256 and e.expected_sha256 in tokenizer_shas and e.source.name in ("tokenizer.json", "vocab.json", "merges.txt"):
        e.notes.append("identico ao tokenizer da base (hash do lock): varredura de texto dispensada")
        return
    scan_bytes(label, data, plan, text_checks=e.source.name not in ("tokenizer.json", "vocab.json", "merges.txt"))


def name_is_big_binary(e: Entry) -> bool:
    return e.source.name.endswith(".onnx.data") or e.source.suffix.lower() == ".data"


def base_tokenizer_shas(root: Path, spec: dict) -> set[str]:
    lock = root / "BaseRepos" / "base-repos.lock.json"
    out: set[str] = set()
    if lock.exists():
        try:
            for f in read_json(lock)["repos"].get(spec["base"], {}).get("files", []):
                if f["path"] in ("tokenizer.json", "vocab.json", "merges.txt"):
                    out.add(f["sha256"])
        except (KeyError, ValueError):
            pass
    return out


def check_card(plan: Plan) -> None:
    ws = plan.root / plan.spec["ws"] / "hf"
    card = ws / ("README.md" if plan.target == "main" else "ONNX_README.md")
    if not card.exists():
        return
    text = card.read_text(encoding="utf-8")
    try:
        from verify_card_numbers import front_matter
        fm = front_matter(text)
    except Exception as exc:  # noqa: BLE001
        plan.errors.append(f"{card.name}: front matter YAML invalido: {exc}")
        return
    if plan.target == "main":
        for k, want in (("license", "apache-2.0"), ("library_name", "transformers"), ("pipeline_tag", "text-generation")):
            if fm.get(k) != want:
                plan.errors.append(f"README.md: front matter {k}={fm.get(k)!r}, esperado {want!r}")
        if not fm.get("base_model"):
            plan.errors.append("README.md: base_model ausente")
        if fm.get("language") != ["en", "pt"]:
            plan.errors.append("README.md: language deve ser [en, pt]")
        need = {"mongodb", "code", "fim", "autocomplete"}
        if need - set(fm.get("tags") or []):
            plan.errors.append(f"README.md: tags ausentes: {sorted(need - set(fm.get('tags') or []))}")
        if "onnx" in (fm.get("tags") or []):
            plan.errors.append("README.md: tag `onnx` no repositorio principal (so safetensors)")
        for must in ("PREVIEW", "public preview"):
            if must not in text:
                plan.errors.append(f"README.md: falta o status {must!r}")
        pkg = TEMPLATE_EXAMPLE.read_text(encoding="utf-8").replace("__REPO_ID__", plan.spec["repo"])
        if pkg not in text:
            plan.errors.append("README.md: o exemplo do card difere de scripts/publish/templates/example_fim.py")
    else:
        if fm.get("license") != "apache-2.0":
            plan.errors.append("ONNX_README.md: license deve ser apache-2.0")
        if "onnx" not in (fm.get("tags") or []):
            plan.errors.append("ONNX_README.md: tag onnx ausente")
        for must in ("kapi", "GetSequence", "0.15.2", "PREVIEW"):
            if must not in text:
                plan.errors.append(f"ONNX_README.md: falta a mencao a {must!r}")


def check_tree(plan: Plan) -> None:
    present = {e.remote for e in plan.entries}
    if plan.target == "main":
        for missing in sorted(MAIN_TREE - present):
            plan.errors.append(f"obrigatorio ausente do pacote: {missing}")
        for extra in sorted(present - MAIN_TREE):
            plan.errors.append(f"arquivo fora da arvore autorizada do repositorio principal: {extra}")
    else:
        for missing in sorted(set(ONNX_ROOT_FILES) - present):
            plan.errors.append(f"obrigatorio ausente do pacote ONNX: {missing}")
        allowed = set(plan.spec["onnx_variants"])
        for r in sorted(present - set(ONNX_ROOT_FILES)):
            if PurePosixPath(r).parts[0] not in allowed:
                plan.errors.append(f"arquivo ONNX fora das pastas de variante autorizadas {sorted(allowed)}: {r}")


def run_checks(plan: Plan, do_hash: bool = True) -> Plan:
    tok_shas = base_tokenizer_shas(plan.root, plan.spec)
    seen = set()
    for e in plan.entries:
        if e.remote in seen:
            plan.errors.append(f"{e.remote}: duplicado no plano")
        seen.add(e.remote)
        check_remote_name(e, plan)
        check_source_location(e, plan)
        if e.source is not None and not e.source.exists():
            continue
        check_entry_content(e, plan, tok_shas)
        if e.data is not None:
            e.sha256 = sha256_bytes(e.data)
        elif do_hash:
            e.sha256 = sha256_file(e.source)
            if e.expected_sha256 and e.sha256 != e.expected_sha256:
                plan.errors.append(f"{e.remote}: sha256 {e.sha256[:12]}... != esperado {e.expected_sha256[:12]}... (fonte: export_info/manifesto/lock)")
    if do_hash and not plan.errors:
        finalize_generated(plan)
        for e in plan.entries:
            if e.kind == "generated" and e.source is None:
                check_remote_name(e, plan)
                check_entry_content(e, plan, tok_shas)
    elif plan.target == "main":
        # sem hash (ou com erros): so confere que a arvore faria sentido, listando o que seria gerado
        for remote in GENERATED_FILES:
            if all(e.remote != remote for e in plan.entries):
                plan.entries.append(Entry(remote, None, "generated", notes=["nao gerado (sem hash ou com erros)"]))
    check_card(plan)
    check_tree(plan)
    return plan


# --- saidas ---------------------------------------------------------------------------------------------------------------------
def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n} B"


def manifest(plan: Plan) -> dict:
    def ent(e: Entry) -> dict:
        d = {"path": e.remote, "source": e.rel(plan.root), "kind": e.kind, "bytes": e.bytes, "sha256": e.sha256}
        if e.variant:
            d["variant"] = e.variant
        return d

    files = sorted(plan.entries, key=lambda x: x.remote)
    return {
        "schema": "kapi-hf-upload-manifest/2",
        "status": "PREVIEW - nada foi enviado; enviar exige --i-have-user-approval e HF_TOKEN (scripts/publish/upload_to_hf.py)",
        "tier": plan.tier,
        "target": plan.target,
        "suggested_repo_id": plan.repo_id,
        "suggested_visibility": "private",
        "release_tag": VERSION_TAG,
        "base_model": plan.spec["base"],
        "files": [ent(e) for e in files],
        "excluded": plan.excluded,
        "totals": {"files": len(files), "bytes": sum(e.bytes for e in files)},
        "checks": {"errors": plan.errors, "warnings": plan.warnings},
    }


def print_plan(plan: Plan) -> None:
    print(f"== {plan.spec['name']} (tier {plan.tier}, alvo {plan.target}) - DRY-RUN: nada e copiado nem enviado")
    print(f"   repo sugerido: {plan.repo_id} (privado; tag {VERSION_TAG})")
    for e in sorted(plan.entries, key=lambda x: x.remote):
        sha = (e.sha256 or "-")[:16]
        note = f"  [{'; '.join(e.notes)}]" if e.notes else ""
        print(f"   {e.kind:9s} {e.remote:55s} {human(e.bytes):>10s}  sha256={sha}{note}")
    total = sum(e.bytes for e in plan.entries)
    print(f"   total a subir: {len(plan.entries)} arquivos, {human(total)}")
    for x in plan.excluded:
        print(f"   recusado/omitido: {x['source']} - {x['reason']}")
    for w in plan.warnings:
        print(f"   AVISO: {w}")
    for er in plan.errors:
        print(f"   ERRO: {er}")
    print(f"   resultado: {'FALHOU (' + str(len(plan.errors)) + ' erros)' if plan.errors else 'OK'}")


def manifest_path(plan: Plan) -> Path:
    name = "upload_manifest.json" if plan.target == "main" else "upload_manifest_onnx.json"
    return plan.root / plan.spec["ws"] / "hf" / name


def check_manifest(plan: Plan, full: bool) -> list[str]:
    path = manifest_path(plan)
    if not path.exists():
        return [f"{path} nao existe (rode --write-manifest)"]
    m = read_json(path)
    errs = []
    cur = {e.remote: e for e in plan.entries}
    man = {f["path"]: f for f in m["files"]}
    if set(cur) != set(man):
        errs.append(f"conjunto de arquivos mudou: so no plano {sorted(set(cur) - set(man))}, so no manifesto {sorted(set(man) - set(cur))}")
    for p, f in man.items():
        e = cur.get(p)
        if not e:
            continue
        if e.bytes != f["bytes"]:
            errs.append(f"{p}: tamanho {e.bytes} != manifesto {f['bytes']}")
        if e.sha256 and e.sha256 != f["sha256"]:
            errs.append(f"{p}: sha256 diferente do manifesto")
    return errs


def stage_dir_for(plan: Plan, base: Path) -> Path:
    return base / (plan.tier if plan.target == "main" else f"{plan.tier}-onnx")


def stage(plan: Plan, dest: Path, clean: bool = False) -> None:
    protected = [plan.root / t["ws"] / "hf" for t in TIER_SPECS.values()] + [plan.root / "data", plan.root / "scripts"]
    d = dest.resolve()
    for p in protected:
        if d == p.resolve() or p.resolve() in d.parents:
            raise SystemExit(f"destino dentro de area protegida: {p}")
    if dest.exists() and any(dest.iterdir()):
        base = (plan.root / "cache" / "hf_stage").resolve()
        if clean and base in d.parents:
            shutil.rmtree(dest)
        else:
            raise SystemExit(f"destino nao vazio: {dest} (use --clean; so e permitido dentro de cache/hf_stage)")
    for e in plan.entries:
        out = dest / Path(*PurePosixPath(e.remote).parts)
        out.parent.mkdir(parents=True, exist_ok=True)
        if e.data is not None:
            out.write_bytes(e.data)
        else:
            shutil.copy2(e.source, out)
        if e.sha256 and sha256_file(out) != e.sha256:
            raise SystemExit(f"copia corrompida: {e.remote}")
    print(f"   copiado para {dest} ({len(plan.entries)} arquivos); sha256 re-verificados; nada foi enviado")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=["nano", "small", "both"], default="both")
    ap.add_argument("--target", choices=["main", "onnx"], default="main", help="main = safetensors (padrao); onnx = repositorio ONNX separado (depois)")
    ap.add_argument("--dry-run", action="store_true", default=True, help="padrao; so lista e verifica")
    ap.add_argument("--no-hash", action="store_true", help="nao calcula sha256 (rapido; nao gera provenance/*; nao use para o manifesto)")
    ap.add_argument("--write-manifest", action="store_true", help="grava hf/upload_manifest[_onnx].json (calcula todos os sha256)")
    ap.add_argument("--check-manifest", action="store_true", help="compara o plano atual com o manifesto gravado")
    ap.add_argument("--stage", nargs="?", const=str(DEFAULT_STAGE_BASE), metavar="DIR",
                    help=f"monta o staging em DIR/<tier> (padrao {DEFAULT_STAGE_BASE.relative_to(ROOT)}; fora do Git); ainda sem rede")
    ap.add_argument("--clean", action="store_true", help="com --stage, apaga um staging anterior (apenas dentro de cache/hf_stage)")
    args = ap.parse_args(argv)

    tiers = ["nano", "small"] if args.tier == "both" else [args.tier]
    rc = 0
    for t in tiers:
        plan = build_plan(t, target=args.target)
        run_checks(plan, do_hash=not args.no_hash or args.write_manifest or args.check_manifest or bool(args.stage))
        print_plan(plan)
        if plan.errors:
            rc = 2
            continue
        if args.write_manifest:
            out = manifest_path(plan)
            out.write_text(json.dumps(manifest(plan), indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
            print(f"[{t}] manifesto gravado: {out}")
        if args.check_manifest:
            errs = check_manifest(plan, full=True)
            for er in errs:
                print("   MANIFESTO:", er)
            if errs:
                rc = 2
            else:
                print("   manifesto confere com o plano atual")
        if args.stage:
            stage(plan, stage_dir_for(plan, Path(args.stage)), clean=args.clean)
    return rc


if __name__ == "__main__":
    sys.exit(main())
