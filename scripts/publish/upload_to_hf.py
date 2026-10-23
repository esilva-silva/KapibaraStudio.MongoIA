#!/usr/bin/env python3
"""Mostra (e, somente com aprovacao explicita do usuario, executa) o envio do pacote para o Hugging Face.

  python scripts/publish/upload_to_hf.py --tier nano --stage-dir D:\\hf_stage\\nano
      -> padrao: SO IMPRIME o comando `huggingface-cli upload` / `hf upload` e o equivalente em API. Nao le o token, nao abre rede.

  python scripts/publish/upload_to_hf.py --tier nano --stage-dir D:\\hf_stage\\nano --i-have-user-approval
      -> executa, MAS apenas se a variavel de ambiente HF_TOKEN estiver definida (nunca por argumento) e o diretorio de staging
         conferir byte a byte com `hf/upload_manifest.json`. O repositorio e criado PRIVADO, salvo `--public` (decisao do usuario).

Fluxo: 1) stage_hf_package.py --write-manifest   2) stage_hf_package.py --stage DIR   3) este script, so depois de o usuario aprovar.
O token nunca e impresso, gravado ou aceito como argumento. `huggingface_hub` so e importado no passo de execucao (nao e instalado por aqui).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from stage_hf_package import TIER_SPECS, sha256_file  # noqa: E402

EXIT_BAD_STAGE = 2
EXIT_NOT_APPROVED = 3


def load_manifest(tier: str, root: Path = ROOT) -> dict:
    path = root / TIER_SPECS[tier]["ws"] / "hf" / "upload_manifest.json"
    if not path.exists():
        raise SystemExit(f"{path} nao existe: rode stage_hf_package.py --write-manifest")
    return json.loads(path.read_text(encoding="utf-8"))


def expected_files(manifest: dict, with_onnx: list[str]) -> dict[str, str]:
    """caminho remoto -> sha256 esperado (pacote padrao + variantes ONNX escolhidas)."""
    out = {f["path"]: f["sha256"] for f in manifest["files"]}
    for v in with_onnx:
        if v not in manifest["optional_onnx"]:
            raise SystemExit(f"variante ONNX desconhecida no manifesto: {v}")
        out.update({f["path"]: f["sha256"] for f in manifest["optional_onnx"][v]})
    return out


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


def commands(repo_id: str, stage_dir: Path, private: bool, message: str) -> list[str]:
    flag = " --private" if private else ""
    return [
        f'huggingface-cli upload {repo_id} "{stage_dir}" . --repo-type model{flag} --commit-message "{message}"',
        f'hf upload {repo_id} "{stage_dir}" . --repo-type model{flag} --commit-message "{message}"',
    ]


def api_snippet(repo_id: str, stage_dir: Path, private: bool, message: str) -> str:
    return (
        "from huggingface_hub import HfApi\n"
        "api = HfApi()  # le HF_TOKEN do ambiente\n"
        f'api.create_repo("{repo_id}", repo_type="model", private={private}, exist_ok=True)\n'
        f'api.upload_folder(repo_id="{repo_id}", repo_type="model", folder_path=r"{stage_dir}", commit_message="{message}")\n'
    )


def _do_upload(repo_id: str, stage_dir: Path, private: bool, message: str) -> None:  # pragma: no cover - so roda com aprovacao
    from huggingface_hub import HfApi  # importado so aqui; o token vem de HF_TOKEN
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(stage_dir), commit_message=message)


def main(argv: list[str] | None = None, root: Path = ROOT) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=sorted(TIER_SPECS), required=True)
    ap.add_argument("--stage-dir", required=True, type=Path, help="saida de `stage_hf_package.py --stage` (um tier)")
    ap.add_argument("--repo-id", help="padrao: o sugerido no manifesto (esilva/...); decisao do usuario")
    ap.add_argument("--include-onnx", nargs="*", default=[], metavar="VARIANTE", help="variantes ONNX que estao no staging")
    ap.add_argument("--public", action="store_true", help="cria o repositorio PUBLICO (padrao: privado)")
    ap.add_argument("--message", default="Add PREVIEW model card and weights")
    ap.add_argument("--i-have-user-approval", action="store_true",
                    help="o usuario aprovou, nesta sessao, este envio especifico; sem esta flag nada e executado")
    args = ap.parse_args(argv)

    manifest = load_manifest(args.tier, root)
    repo_id = args.repo_id or manifest["suggested_repo_id"]
    private = not args.public
    expected = expected_files(manifest, args.include_onnx)

    print(f"tier: {args.tier}   repo: {repo_id}   visibilidade: {'privada' if private else 'PUBLICA'}")
    print(f"staging: {args.stage_dir}   arquivos esperados: {len(expected)}")
    errs = verify_stage_dir(args.stage_dir, expected)
    if errs:
        for e in errs:
            print("ERRO:", e)
        return EXIT_BAD_STAGE
    print("staging confere com hf/upload_manifest.json (arquivos e sha256)")
    print("\nComando equivalente (nao executado):")
    for c in commands(repo_id, args.stage_dir, private, args.message):
        print("  " + c)
    print("\nOu pela API:\n" + "".join("  " + line + "\n" for line in api_snippet(repo_id, args.stage_dir, private, args.message).splitlines()))
    print("O token e lido de HF_TOKEN (variavel de ambiente); nunca passe token por argumento.")

    if not args.i_have_user_approval:
        print("\nDRY-RUN: nada foi enviado. Para executar, o usuario precisa aprovar e voce passa --i-have-user-approval (com HF_TOKEN definido).")
        return 0
    if not os.environ.get("HF_TOKEN"):
        print("\nRECUSADO: --i-have-user-approval foi passado, mas HF_TOKEN nao esta definido no ambiente. Nada foi enviado.")
        return EXIT_NOT_APPROVED
    print("\nHF_TOKEN presente (nao exibido). Enviando...")
    _do_upload(repo_id, args.stage_dir, private, args.message)
    print("Envio concluido.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
