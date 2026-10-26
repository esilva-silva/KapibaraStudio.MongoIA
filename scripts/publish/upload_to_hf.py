#!/usr/bin/env python3
"""Mostra (e, somente com aprovacao explicita do usuario, executa) o envio do pacote em PREVIEW PRIVADO para o Hugging Face.

  python scripts/publish/upload_to_hf.py --tier nano
      -> padrao: SO IMPRIME os comandos exatos (criar o repositorio PRIVADO, subir a pasta de staging, criar a tag `v0.1.0-preview`,
         verificar o Hub) e confere o staging contra hf/upload_manifest.json. Nao le token, nao abre rede.

  python scripts/publish/upload_to_hf.py --tier nano --i-have-user-approval [--user-is-logged-in]
      -> executa create -> upload -> tag -> verify, MAS apenas se o usuario aprovou nesta sessao E ele mesmo forneceu a credencial:
         HF_TOKEN no ambiente (nunca por argumento) ou `--user-is-logged-in` (o proprio usuario rodou `hf auth login`; este script nao le o arquivo de credencial).
         O repositorio e criado PRIVADO, salvo `--public` (decisao do usuario).

  python scripts/publish/upload_to_hf.py --tier nano --steps verify [--i-have-user-approval ...]
      -> so a verificacao pos-upload: compara o sha256 de cada arquivo do manifesto com o do Hub na revisao da tag.

Fluxo: 1) stage_hf_package.py --write-manifest   2) stage_hf_package.py --stage   3) este script, so depois de o usuario aprovar.
Repositorio ONNX (DEPOIS, desligado por padrao): --target onnx (staging cache/hf_stage/<tier>-onnx e hf/upload_manifest_onnx.json).
Nada deste arquivo e executado nos testes; `huggingface_hub` so e importado nos passos que tocam a rede.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from stage_hf_package import DEFAULT_STAGE_BASE, TIER_SPECS, VERSION_TAG, sha256_file  # noqa: E402

EXIT_BAD_STAGE = 2
EXIT_NOT_APPROVED = 3
EXIT_VERIFY_FAILED = 4
STEPS = ("create", "upload", "tag", "verify")
HUB_AUTO_FILES = frozenset({".gitattributes"})   # criado pelo proprio Hub ao criar o repositorio


def manifest_file(tier: str, target: str, root: Path = ROOT) -> Path:
    name = "upload_manifest.json" if target == "main" else "upload_manifest_onnx.json"
    return root / TIER_SPECS[tier]["ws"] / "hf" / name


def load_manifest(tier: str, root: Path = ROOT, target: str = "main") -> dict:
    path = manifest_file(tier, target, root)
    if not path.exists():
        raise SystemExit(f"{path} nao existe: rode stage_hf_package.py --write-manifest" + (" --target onnx" if target == "onnx" else ""))
    return json.loads(path.read_text(encoding="utf-8"))


def expected_files(manifest: dict) -> dict[str, str]:
    """caminho remoto -> sha256 esperado."""
    return {f["path"]: f["sha256"] for f in manifest["files"]}


def expected_sizes(manifest: dict) -> dict[str, int]:
    return {f["path"]: f["bytes"] for f in manifest["files"]}


def verify_stage_dir(stage_dir: Path, expected: dict[str, str]) -> list[str]:
    errs: list[str] = []
    if not stage_dir.is_dir():
        return [f"diretorio de staging inexistente: {stage_dir}"]
    present = {p.relative_to(stage_dir).as_posix(): p for p in stage_dir.rglob("*") if p.is_file()}
    for extra in sorted(set(present) - set(expected)):
        errs.append(f"arquivo FORA do manifesto no staging: {extra}")
    for rel, sha in expected.items():
        p = present.get(rel)
        if p is None:
            errs.append(f"arquivo do manifesto ausente no staging: {rel}")
        elif not sha:
            errs.append(f"{rel}: manifesto sem sha256")
        elif sha256_file(p) != sha:
            errs.append(f"{rel}: sha256 diferente do manifesto")
    return errs


def git_blob_sha1(path: Path) -> str:
    """Hash de objeto git (blob) de um arquivo: o `blob_id` que o Hub devolve para arquivos que nao sao LFS."""
    size = path.stat().st_size
    h = hashlib.sha1(f"blob {size}\0".encode())  # noqa: S324 - e o formato do git, nao uso criptografico
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_remote(expected: dict[str, str], sizes: dict[str, int], remote: dict[str, dict], stage_dir: Path | None) -> list[str]:
    """Compara o manifesto com a arvore do Hub.

    `remote[path]` = {"size": int, "lfs_sha256": str|None, "blob_id": str|None}. Arquivos LFS trazem o sha256 verdadeiro; os pequenos (texto)
    so trazem o `blob_id` (sha1 do git), entao sao conferidos pelo sha1 de blob do arquivo do staging, que por sua vez tem o sha256 do manifesto.
    """
    errs: list[str] = []
    for extra in sorted(set(remote) - set(expected) - HUB_AUTO_FILES):
        errs.append(f"arquivo no Hub que nao esta no manifesto: {extra}")
    for rel, sha in expected.items():
        r = remote.get(rel)
        if r is None:
            errs.append(f"arquivo do manifesto ausente no Hub: {rel}")
            continue
        if r.get("size") is not None and sizes.get(rel) is not None and r["size"] != sizes[rel]:
            errs.append(f"{rel}: tamanho no Hub {r['size']} != manifesto {sizes[rel]}")
        if r.get("lfs_sha256"):
            if r["lfs_sha256"] != sha:
                errs.append(f"{rel}: sha256 no Hub {r['lfs_sha256'][:12]}... != manifesto {sha[:12]}...")
        elif r.get("blob_id"):
            if stage_dir is None:
                errs.append(f"{rel}: arquivo sem LFS e sem staging local para conferir o blob_id")
                continue
            local = stage_dir / rel
            if not local.is_file() or sha256_file(local) != sha:
                errs.append(f"{rel}: staging local ausente/diferente do manifesto; nao da para conferir o blob_id")
            elif git_blob_sha1(local) != r["blob_id"]:
                errs.append(f"{rel}: blob_id no Hub difere do arquivo do staging")
        else:
            errs.append(f"{rel}: o Hub nao devolveu sha256 nem blob_id")
    return errs


def commands(repo_id: str, stage_dir: Path, private: bool, message: str, tag: str, tag_message: str, tier: str, target: str) -> dict[str, list[str]]:
    """Comandos exatos por passo (so texto)."""
    priv = " --private" if private else ""
    tgt = " --target onnx" if target == "onnx" else ""
    return {
        "create": [f"hf repo create {repo_id} --repo-type model{priv}",
                   f"# (CLI antiga) huggingface-cli repo create {repo_id.split('/')[1]} --type model --organization-or-user {repo_id.split('/')[0]}{priv}"],
        "upload": [f'hf upload {repo_id} "{stage_dir}" . --repo-type model{priv} --commit-message "{message}"',
                   f'# (CLI antiga) huggingface-cli upload {repo_id} "{stage_dir}" . --repo-type model{priv} --commit-message "{message}"'],
        "tag": [f'hf repo tag create {repo_id} {tag} -m "{tag_message}" --repo-type model',
                f'# (CLI antiga) huggingface-cli tag {repo_id} {tag} -m "{tag_message}" --repo-type model'],
        "verify": [f"python scripts/publish/upload_to_hf.py --tier {tier}{tgt} --steps verify --i-have-user-approval  # [--user-is-logged-in]; compara sha256 do manifesto com o Hub na revisao {tag}"],
    }


def api_snippet(repo_id: str, stage_dir: Path, private: bool, message: str, tag: str, tag_message: str) -> str:
    return (
        "from huggingface_hub import HfApi\n"
        "api = HfApi()  # le HF_TOKEN do ambiente (ou o login feito pelo proprio usuario)\n"
        f'api.create_repo("{repo_id}", repo_type="model", private={private}, exist_ok=True)\n'
        f'api.upload_folder(repo_id="{repo_id}", repo_type="model", folder_path=r"{stage_dir}", commit_message="{message}")\n'
        f'api.create_tag("{repo_id}", tag="{tag}", tag_message="{tag_message}", repo_type="model")\n'
    )


def _do_create(repo_id: str, private: bool) -> None:  # pragma: no cover - so roda com aprovacao
    from huggingface_hub import HfApi
    HfApi().create_repo(repo_id, repo_type="model", private=private, exist_ok=True)


def _do_upload(repo_id: str, stage_dir: Path, message: str) -> None:  # pragma: no cover
    from huggingface_hub import HfApi
    HfApi().upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(stage_dir), commit_message=message)


def _do_tag(repo_id: str, tag: str, tag_message: str) -> None:  # pragma: no cover
    from huggingface_hub import HfApi
    HfApi().create_tag(repo_id, tag=tag, tag_message=tag_message, repo_type="model")


def fetch_remote(repo_id: str, revision: str) -> dict[str, dict]:  # pragma: no cover
    from huggingface_hub import HfApi
    out: dict[str, dict] = {}
    for item in HfApi().list_repo_tree(repo_id, revision=revision, recursive=True, repo_type="model"):
        if not hasattr(item, "blob_id"):          # pasta
            continue
        lfs = getattr(item, "lfs", None)
        sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        out[item.path] = {"size": item.size, "lfs_sha256": sha, "blob_id": item.blob_id}
    return out


def main(argv: list[str] | None = None, root: Path = ROOT) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=sorted(TIER_SPECS), required=True)
    ap.add_argument("--target", choices=["main", "onnx"], default="main", help="main = safetensors (padrao); onnx = repositorio ONNX separado (depois)")
    ap.add_argument("--stage-dir", type=Path, help="saida de `stage_hf_package.py --stage` (padrao: cache/hf_stage/<tier>[-onnx])")
    ap.add_argument("--repo-id", help="padrao: o sugerido no manifesto (esilva/...); decisao do usuario")
    ap.add_argument("--tag", default=VERSION_TAG)
    ap.add_argument("--steps", default=",".join(STEPS), help=f"subconjunto de {','.join(STEPS)} (padrao: todos, nessa ordem)")
    ap.add_argument("--public", action="store_true", help="cria o repositorio PUBLICO (padrao: privado)")
    ap.add_argument("--message", default=None)
    ap.add_argument("--i-have-user-approval", action="store_true",
                    help="o usuario aprovou, nesta sessao, este envio especifico; sem esta flag nada e executado")
    ap.add_argument("--user-is-logged-in", action="store_true",
                    help="o proprio usuario fez `hf auth login`; alternativa a HF_TOKEN (este script nao le o arquivo de credencial)")
    args = ap.parse_args(argv)

    steps = [s for s in args.steps.split(",") if s]
    bad = [s for s in steps if s not in STEPS]
    if bad:
        raise SystemExit(f"passos desconhecidos: {bad}")
    manifest = load_manifest(args.tier, root, args.target)
    if manifest.get("target", "main") != args.target:
        raise SystemExit("o manifesto e de outro alvo (main/onnx)")
    repo_id = args.repo_id or manifest["suggested_repo_id"]
    private = not args.public
    stage_dir = args.stage_dir or (DEFAULT_STAGE_BASE if root == ROOT else root / "cache" / "hf_stage") / (args.tier if args.target == "main" else f"{args.tier}-onnx")
    message = args.message or f"Add v3 {'weights' if args.target == 'main' else 'ONNX builds'} ({args.tag})"
    tag_message = f"{args.tag}: public preview, internal synthetic evaluation only"
    expected = expected_files(manifest)

    print(f"tier: {args.tier}   alvo: {args.target}   repo: {repo_id}   visibilidade: {'privada' if private else 'PUBLICA'}   tag: {args.tag}")
    print(f"staging: {stage_dir}   arquivos esperados: {len(expected)}")
    errs = verify_stage_dir(stage_dir, expected)
    if errs:
        for e in errs:
            print("ERRO:", e)
        return EXIT_BAD_STAGE
    print("staging confere com o manifesto (arquivos e sha256)")
    cmds = commands(repo_id, stage_dir, private, message, args.tag, tag_message, args.tier, args.target)
    print("\nComandos exatos (NAO executados; a sintaxe da CLI varia por versao do huggingface_hub, a API abaixo e a forma estavel):")
    for i, s in enumerate(STEPS, 1):
        print(f"  {i}. {s}")
        for c in cmds[s]:
            print("     " + c)
    print("\nOu pela API (passos 1 a 3):\n" + "".join("  " + line + "\n" for line in api_snippet(repo_id, stage_dir, private, message, args.tag, tag_message).splitlines()))
    print("Verificacao pos-upload (passo 4): lista a arvore do Hub na revisao da tag e compara, arquivo a arquivo, tamanho e sha256 com o manifesto\n"
          "(LFS: sha256 do Hub; arquivos pequenos: blob_id git do Hub contra o do staging). Qualquer diferenca, arquivo extra ou ausente = exit 4.")
    print("O token e lido de HF_TOKEN (variavel de ambiente); nunca passe token por argumento.")

    if not args.i_have_user_approval:
        print("\nDRY-RUN: nada foi enviado. Para executar, o usuario precisa aprovar e voce passa --i-have-user-approval "
              "(com HF_TOKEN definido pelo usuario, ou --user-is-logged-in).")
        return 0
    if not (os.environ.get("HF_TOKEN") or args.user_is_logged_in):
        print("\nRECUSADO: --i-have-user-approval foi passado, mas nem HF_TOKEN esta definido nem --user-is-logged-in foi informado. Nada foi enviado.")
        return EXIT_NOT_APPROVED
    print("\nCredencial do usuario presente (nao exibida). Executando os passos:", ", ".join(s for s in STEPS if s in steps))
    if "create" in steps:
        _do_create(repo_id, private)
        print("  repositorio criado/confirmado (privado)" if private else "  repositorio criado/confirmado (PUBLICO)")
    if "upload" in steps:
        _do_upload(repo_id, stage_dir, message)
        print("  upload concluido")
    if "tag" in steps:
        _do_tag(repo_id, args.tag, tag_message)
        print(f"  tag {args.tag} criada")
    if "verify" in steps:
        remote = fetch_remote(repo_id, args.tag)
        verr = compare_remote(expected, expected_sizes(manifest), remote, stage_dir)
        if verr:
            for e in verr:
                print("  VERIFICACAO:", e)
            return EXIT_VERIFY_FAILED
        print(f"  verificacao OK: {len(expected)} arquivos do Hub (revisao {args.tag}) batem com o manifesto")
    return 0


if __name__ == "__main__":
    sys.exit(main())
