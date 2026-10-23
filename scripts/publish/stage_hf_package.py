#!/usr/bin/env python3
"""Prepara (e so prepara) o pacote de publicacao em PREVIEW no Hugging Face. Sem rede, sem token, sem upload.

  python scripts/publish/stage_hf_package.py --tier nano                  # dry-run (padrao): lista arquivos, tamanhos, sha256 e checagens
  python scripts/publish/stage_hf_package.py --tier both --no-hash        # rapido: so tamanhos e checagens de texto (sem sha256)
  python scripts/publish/stage_hf_package.py --tier small --include-onnx  # inclui as variantes ONNX padrao (nano: int4,dml-fp16; small: dml-fp16,int8)
  python scripts/publish/stage_hf_package.py --tier both --write-manifest # grava hf/upload_manifest.json (calcula todos os sha256, ONNX inclusive)
  python scripts/publish/stage_hf_package.py --tier nano --check-manifest # compara os arquivos atuais com hf/upload_manifest.json
  python scripts/publish/stage_hf_package.py --tier nano --stage D:\\hf_stage   # COPIA os arquivos para um diretorio vazio e re-verifica os hashes

Regras (qualquer violacao = exit 2, nada e copiado):
  * so entram arquivos do workspace do proprio tier: merged/<run>/ (pesos e tokenizer), exports/<pacote ONNX>/, hf/ (card, NOTICE, atribuicao) e o LICENSE;
  * lista de recusa: train_state.json, export_info.json, export_manifest.json, run_manifest.json, model.lock.json, *.jsonl, *.npz, checkpoints, otimizador, GGUF, notas internas;
  * varredura de conteudo (texto e binarios < 16 MB; cabecalho do safetensors): caminhos locais, usuario da maquina, tokens/segredos, URI MongoDB com senha;
  * pesos e tokenizer conferem com `export_info.json` / `kapicoder_manifest.json`; o LICENSE confere com o da base;
  * o card tem front matter YAML valido (license, base_model, library_name, pipeline_tag, tags, language).

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
ALLOWED_EXT = TEXT_EXT | {".safetensors", ".onnx", ".data"}
ALLOWED_NO_EXT = {"LICENSE"}

CORE_FILES = ("config.json", "generation_config.json", "merges.txt", "model.safetensors",
              "tokenizer.json", "tokenizer_config.json", "vocab.json")
CARD_FILES = ("README.md", "NOTICE.md", "REPLAY_ATTRIBUTION.md")
INTERNAL_HF_FILES = ("MODEL_CARD_REVIEW.md", "SOURCES.md", "upload_manifest.json")
REFUSE_NAMES = {
    "train_state.json", "export_info.json", "export_manifest.json", "run_manifest.json", "model.lock.json", "best.json",
    "heartbeat.json", "metrics.jsonl", "eval_history.jsonl", "predictions.jsonl", "optimizer.pt", "rng_state.pt",
    "SOURCES.md", "MODEL_CARD_REVIEW.md", "upload_manifest.json", "README_job.md", ".env", ".netrc",
}
REFUSE_SUFFIXES = (".jsonl", ".npz", ".npy", ".parquet", ".pt", ".pth", ".bin", ".ckpt", ".gguf", ".pkl", ".pickle", ".log", ".csv", ".zip", ".gz", ".py", ".ps1", ".yaml", ".yml")
REFUSE_PARTS = {"checkpoints", "adapters", "data", "tmp", "logs", "reports", "gguf", "__pycache__", ".git"}

TIER_SPECS = {
    "nano": {"name": "KapiCoder-Mongo-Nano", "ws": "Repos/KapiCoder-Mongo-Nano", "run": "sft_distill_v1",
             "onnx_default": ("int4", "dml-fp16"), "repo": "esilva/KapiCoder-Mongo-Nano", "base": "Qwen2.5-Coder-0.5B"},
    "small": {"name": "KapiCoder-Mongo-Small", "ws": "Repos/KapiCoder-Mongo-Small", "run": "sft_distill_v1",
              "onnx_default": ("dml-fp16", "int8"), "repo": "esilva/KapiCoder-Mongo-Small", "base": "Qwen2.5-Coder-1.5B"},
}

# --- padroes de varredura -------------------------------------------------------------------------------------------------------
PATH_PATTERNS = {
    "caminho Windows": re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:(?:\\|/)(?!/)"),
    "caminho de usuario": re.compile(r"(/Users/|/home/[A-Za-z]|\\Users\\|\\\\Users\\\\)"),
    "caminho do workspace": re.compile(r"(Repos[\\/]+Kapi|BaseRepos|SlopStudio|KapibaraStudio\.MongoIA|MongoIA[\\/]|\bchuke\b)"),
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


@dataclass
class Entry:
    remote: str                  # caminho no repositorio HF
    source: Path                 # arquivo local (absoluto)
    kind: str                    # core | card | license | onnx
    variant: str | None = None
    bytes: int = 0
    sha256: str | None = None
    expected_sha256: str | None = None
    notes: list[str] = field(default_factory=list)

    def rel(self, root: Path) -> str:
        try:
            return self.source.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return str(self.source)


@dataclass
class Plan:
    tier: str
    root: Path
    spec: dict
    entries: list[Entry] = field(default_factory=list)       # o que sobe por padrao (+ ONNX se pedido)
    optional_onnx: dict[str, list[Entry]] = field(default_factory=dict)
    excluded: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


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


# --- montagem do plano ----------------------------------------------------------------------------------------------------------
def build_plan(tier: str, root: Path = ROOT, onnx: list[str] | None = None, spec: dict | None = None) -> Plan:
    """`onnx` = variantes a incluir em `entries` (None = nenhuma); todas as variantes padrao ficam em `optional_onnx`."""
    spec = spec or TIER_SPECS[tier]
    plan = Plan(tier=tier, root=root, spec=spec)
    ws = root / spec["ws"]
    merged = ws / "merged" / spec["run"]
    hf = ws / "hf"

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
        plan.excluded.append({"source": f"{spec['ws']}/hf/{name}", "reason": "nota interna / manifesto: nunca sobe"})
    for name in ("train_state.json", "export_info.json"):
        if (merged / name).exists():
            plan.excluded.append({"source": f"{spec['ws']}/merged/{spec['run']}/{name}", "reason": "metadados de treino com caminhos locais: recusado"})

    for variant in spec["onnx_default"]:
        pkg = ws / "exports" / f"{spec['name']}-ONNX-{variant}"
        ents = onnx_entries(pkg, variant, plan)
        plan.optional_onnx[variant] = ents
        if onnx and variant in onnx:
            plan.entries.extend(ents)
    if onnx:
        unknown = [v for v in onnx if v not in spec["onnx_default"]]
        for v in unknown:
            pkg = ws / "exports" / f"{spec['name']}-ONNX-{v}"
            ents = onnx_entries(pkg, v, plan)
            plan.optional_onnx[v] = ents
            plan.entries.extend(ents)
            plan.warnings.append(f"variante ONNX fora da lista recomendada: {v}")
    pkg_dir = ws / "exports"
    if pkg_dir.exists():
        for d in sorted(pkg_dir.iterdir()):
            if d.is_dir() and d.name.startswith(spec["name"] + "-ONNX-"):
                v = d.name[len(spec["name"]) + 6:]
                if v not in plan.optional_onnx and not (onnx and v in onnx):
                    plan.excluded.append({"source": f"{spec['ws']}/exports/{d.name}", "reason": "variante ONNX nao recomendada/nao selecionada"})
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


def onnx_entries(pkg: Path, variant: str, plan: Plan) -> list[Entry]:
    if not pkg.exists():
        plan.errors.append(f"pacote ONNX ausente: {pkg}")
        return []
    man_path = pkg / "kapicoder_manifest.json"
    if not man_path.exists():
        plan.errors.append(f"kapicoder_manifest.json ausente em {pkg}")
        return []
    man = read_json(man_path)
    out = []
    for f in man["files"]:
        src = pkg / f["path"]
        if not src.exists():
            plan.errors.append(f"arquivo do manifesto ausente: {src}")
            continue
        out.append(Entry(f"onnx/{variant}/{f['path']}", src, "onnx", variant=variant, expected_sha256=f["sha256"]))
    # o proprio manifesto (nao se lista a si mesmo) viaja junto, para `--verify-hashes` na IDE
    out.append(Entry(f"onnx/{variant}/kapicoder_manifest.json", man_path, "onnx", variant=variant))
    return out


# --- checagens ------------------------------------------------------------------------------------------------------------------
def check_remote_name(e: Entry, plan: Plan) -> None:
    p = PurePosixPath(e.remote)
    if p.is_absolute() or ".." in p.parts or "\\" in e.remote or re.match(r"^[A-Za-z]:", e.remote):
        plan.errors.append(f"{e.remote}: caminho remoto invalido")
    name = p.name
    if name in REFUSE_NAMES or any(part in REFUSE_PARTS for part in p.parts[:-1]):
        plan.errors.append(f"{e.remote}: nome/diretorio na lista de recusa")
    if name.lower().endswith(REFUSE_SUFFIXES):
        plan.errors.append(f"{e.remote}: extensao recusada")
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXT and name not in ALLOWED_NO_EXT:
        plan.errors.append(f"{e.remote}: extensao fora da lista permitida ({ext or 'sem extensao'})")


def check_source_location(e: Entry, plan: Plan) -> None:
    spec, root = plan.spec, plan.root
    allowed = [root / spec["ws"] / "merged" / spec["run"], root / spec["ws"] / "hf", root / spec["ws"] / "LICENSE"]
    allowed += [root / spec["ws"] / "exports" / f"{spec['name']}-ONNX-{v}" for v in set(spec["onnx_default"]) | set(plan.optional_onnx)]
    src = e.source.resolve()
    ok = False
    for a in allowed:
        a = a.resolve()
        if src == a or a in src.parents:
            ok = True
    if not ok:
        plan.errors.append(f"{e.remote}: origem fora do workspace/pastas permitidas do tier: {e.source}")
    other = [t for t in TIER_SPECS.values() if t["ws"] != spec["ws"]]
    for o in other:
        if (root / o["ws"]).resolve() in src.parents:
            plan.errors.append(f"{e.remote}: arquivo de outro workspace ({o['name']})")


def scan_bytes(label: str, data: bytes, plan: Plan, text_checks: bool = True) -> None:
    try:
        text = data.decode("utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return
    for name, rx in {**PATH_PATTERNS, **SECRET_PATTERNS}.items():
        m = rx.search(text)
        if m:
            snippet = text[max(0, m.start() - 20):m.end() + 20].replace("\n", " ")
            plan.errors.append(f"{label}: {name} encontrado: ...{snippet}...")
    if text_checks:
        m = EMAIL.search(text)
        if m:
            plan.warnings.append(f"{label}: endereco de e-mail no texto: {m.group(0)}")


def safetensors_header(path: Path) -> dict:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n <= 0 or n > 64 * 1024 * 1024:
            raise ValueError(f"cabecalho do safetensors com tamanho absurdo ({n})")
        raw = f.read(n)
    return {"raw": raw, "json": json.loads(raw.decode("utf-8"))}


def check_entry_content(e: Entry, plan: Plan, tokenizer_shas: set[str]) -> None:
    size = e.source.stat().st_size
    e.bytes = size
    ext = e.source.suffix.lower()
    label = e.remote
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
    card = plan.root / plan.spec["ws"] / "hf" / "README.md"
    if not card.exists():
        return
    text = card.read_text(encoding="utf-8")
    try:
        from verify_card_numbers import front_matter
        fm = front_matter(text)
    except Exception as exc:  # noqa: BLE001
        plan.errors.append(f"README.md: front matter YAML invalido: {exc}")
        return
    for k, want in (("license", "apache-2.0"), ("library_name", "transformers"), ("pipeline_tag", "text-generation")):
        if fm.get(k) != want:
            plan.errors.append(f"README.md: front matter {k}={fm.get(k)!r}, esperado {want!r}")
    if not fm.get("base_model"):
        plan.errors.append("README.md: base_model ausente")
    if fm.get("language") != ["en", "pt"]:
        plan.errors.append("README.md: language deve ser [en, pt]")
    need = {"mongodb", "code", "fim", "autocomplete", "onnx"}
    if need - set(fm.get("tags") or []):
        plan.errors.append(f"README.md: tags ausentes: {sorted(need - set(fm.get('tags') or []))}")
    if "PREVIEW" not in text:
        plan.errors.append("README.md: falta o status PREVIEW")


def run_checks(plan: Plan, do_hash: bool = True, hash_onnx: bool | None = None) -> Plan:
    tok_shas = base_tokenizer_shas(plan.root, plan.spec)
    seen = set()
    for e in plan.entries:
        if e.remote in seen:
            plan.errors.append(f"{e.remote}: duplicado no plano")
        seen.add(e.remote)
        check_remote_name(e, plan)
        check_source_location(e, plan)
        if not e.source.exists():
            continue
        check_entry_content(e, plan, tok_shas)
        if do_hash and (e.kind != "onnx" or hash_onnx is not False):
            e.sha256 = sha256_file(e.source)
            if e.expected_sha256 and e.sha256 != e.expected_sha256:
                plan.errors.append(f"{e.remote}: sha256 {e.sha256[:12]}... != esperado {e.expected_sha256[:12]}... (fonte: export_info/manifesto/lock)")
    check_card(plan)
    present = {e.remote for e in plan.entries}
    for must in ("model.safetensors", "config.json", "tokenizer.json", "README.md", "LICENSE", "NOTICE.md"):
        if must not in present:
            plan.errors.append(f"obrigatorio ausente do pacote: {must}")
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

    default_remote = [e for e in plan.entries if e.kind != "onnx"]
    return {
        "schema": "kapi-hf-upload-manifest/1",
        "status": "PREVIEW - nada foi enviado; enviar exige --i-have-user-approval e HF_TOKEN (scripts/publish/upload_to_hf.py)",
        "tier": plan.tier,
        "suggested_repo_id": plan.spec["repo"],
        "suggested_visibility": "private",
        "base_model": plan.spec["base"],
        "files": [ent(e) for e in default_remote],
        "optional_onnx": {v: [ent(e) for e in ents] for v, ents in plan.optional_onnx.items()},
        "excluded": plan.excluded,
        "totals": {
            "default_files": len(default_remote),
            "default_bytes": sum(e.bytes for e in default_remote),
            "onnx_bytes": {v: sum(e.bytes for e in ents) for v, ents in plan.optional_onnx.items()},
        },
        "checks": {"errors": plan.errors, "warnings": plan.warnings},
    }


def print_plan(plan: Plan) -> None:
    print(f"== {plan.spec['name']} (tier {plan.tier}) - DRY-RUN: nada e copiado nem enviado")
    print(f"   repo sugerido: {plan.spec['repo']} (privado primeiro)")
    for e in plan.entries:
        sha = (e.sha256 or "-")[:16]
        note = f"  [{'; '.join(e.notes)}]" if e.notes else ""
        print(f"   {e.kind:7s} {e.remote:55s} {human(e.bytes):>10s}  sha256={sha}{note}")
    total = sum(e.bytes for e in plan.entries)
    print(f"   total a subir: {len(plan.entries)} arquivos, {human(total)}")
    for v, ents in plan.optional_onnx.items():
        selected = any(e in plan.entries for e in ents)
        print(f"   ONNX opcional {v:10s}: {len(ents)} arquivos, {human(sum(e.source.stat().st_size for e in ents if e.source.exists()))}"
              f"{' (incluido)' if selected else ' (nao incluido; use --include-onnx)'}")
    for x in plan.excluded:
        print(f"   recusado/omitido: {x['source']} - {x['reason']}")
    for w in plan.warnings:
        print(f"   AVISO: {w}")
    for er in plan.errors:
        print(f"   ERRO: {er}")
    print(f"   resultado: {'FALHOU (' + str(len(plan.errors)) + ' erros)' if plan.errors else 'OK'}")


def check_manifest(plan: Plan, full: bool) -> list[str]:
    path = plan.root / plan.spec["ws"] / "hf" / "upload_manifest.json"
    if not path.exists():
        return [f"{path} nao existe (rode --write-manifest)"]
    m = read_json(path)
    errs = []
    cur = {e.remote: e for e in plan.entries if e.kind != "onnx"}
    for e in cur.values():
        e.bytes = e.source.stat().st_size
    man = {f["path"]: f for f in m["files"]}
    if set(cur) != set(man):
        errs.append(f"conjunto de arquivos mudou: so no disco {sorted(set(cur) - set(man))}, so no manifesto {sorted(set(man) - set(cur))}")
    for p, f in man.items():
        e = cur.get(p)
        if not e:
            continue
        if e.bytes != f["bytes"]:
            errs.append(f"{p}: tamanho {e.bytes} != manifesto {f['bytes']}")
        if (full or e.bytes < 32 * 1024 * 1024) and sha256_file(e.source) != f["sha256"]:
            errs.append(f"{p}: sha256 diferente do manifesto")
    return errs


def stage(plan: Plan, dest: Path) -> None:
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"destino nao vazio: {dest}")
    protected = [plan.root / t["ws"] / "hf" for t in TIER_SPECS.values()] + [plan.root / "data"]
    d = dest.resolve()
    for p in protected:
        if d == p.resolve() or p.resolve() in d.parents:
            raise SystemExit(f"destino dentro de area protegida: {p}")
    for e in plan.entries:
        out = dest / Path(*PurePosixPath(e.remote).parts)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(e.source, out)
        if e.sha256 and sha256_file(out) != e.sha256:
            raise SystemExit(f"copia corrompida: {e.remote}")
    print(f"   copiado para {dest} ({len(plan.entries)} arquivos); nada foi enviado")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=["nano", "small", "both"], default="both")
    ap.add_argument("--dry-run", action="store_true", default=True, help="padrao; so lista e verifica")
    ap.add_argument("--include-onnx", nargs="*", metavar="VARIANTE", help="inclui variantes ONNX (sem argumentos = as recomendadas do tier)")
    ap.add_argument("--no-hash", action="store_true", help="nao calcula sha256 (rapido; nao use para gerar o manifesto)")
    ap.add_argument("--write-manifest", action="store_true", help="grava hf/upload_manifest.json (calcula todos os sha256, ONNX inclusive)")
    ap.add_argument("--check-manifest", action="store_true", help="compara arquivos atuais com hf/upload_manifest.json")
    ap.add_argument("--full", action="store_true", help="com --check-manifest, re-hasheia tambem os arquivos grandes")
    ap.add_argument("--stage", metavar="DIR", help="copia os arquivos (apos as checagens) para DIR vazio; ainda sem rede")
    args = ap.parse_args(argv)

    tiers = ["nano", "small"] if args.tier == "both" else [args.tier]
    rc = 0
    for t in tiers:
        spec = TIER_SPECS[t]
        onnx = None
        if args.include_onnx is not None:
            onnx = list(args.include_onnx) or list(spec["onnx_default"])
        plan = build_plan(t, onnx=onnx)
        if args.write_manifest:
            # o manifesto descreve o pacote padrao + todas as variantes ONNX opcionais, todas com sha256
            plan_all = build_plan(t, onnx=list(spec["onnx_default"]))
            run_checks(plan_all, do_hash=True)
            if plan_all.errors:
                print_plan(plan_all)
                rc = 2
                continue
            out = ROOT / spec["ws"] / "hf" / "upload_manifest.json"
            out.write_text(json.dumps(manifest(plan_all), indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
            print(f"[{t}] manifesto gravado: {out}")
            print_plan(plan_all)
            continue
        run_checks(plan, do_hash=not args.no_hash)
        print_plan(plan)
        if plan.errors:
            rc = 2
            continue
        if args.check_manifest:
            errs = check_manifest(plan, args.full)
            for er in errs:
                print("   MANIFESTO:", er)
            if errs:
                rc = 2
            else:
                print("   manifesto confere com os arquivos atuais")
        if args.stage:
            stage(plan, Path(args.stage) / t)
    return rc


if __name__ == "__main__":
    sys.exit(main())
