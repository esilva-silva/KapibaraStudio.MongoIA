"""Testes simples dos pacotes ONNX gerados (hashes do manifesto, genai_config, metadata kapi, tokenizer byte a byte).

Pacotes ausentes (exports/ e gitignored) sao ignorados. Hash completo dos arquivos grandes so com KAPI_EXPORT_FULL_HASH=1 (padrao: tamanho +
hash dos arquivos < 64 MB).
  python -I -m pytest scripts/export/tests -q
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

import build_variants as bv

ROOT = Path(__file__).resolve().parents[3]
FULL = os.environ.get("KAPI_EXPORT_FULL_HASH") == "1"
# (tier, variante, pacote, geracao): as duas geracoes (legado v2c/sft_distill_v1 e v3/sft_v3) sao testadas independentemente de KAPI_EXPORT_GEN
PACKAGES = [(t, v, bv.package_dir(t, v, g), g) for g in ("", "v3") for t in bv.TIERS for v in bv.VARIANTS
            if (bv.package_dir(t, v, g) / "export_manifest.json").exists()]
IDS = [f"{g or 'v2'}-{t}-{v}" for t, v, _, g in PACKAGES]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


pytestmark = pytest.mark.skipif(not PACKAGES, reason="nenhum pacote exportado (Repos/*/exports)")


@pytest.mark.parametrize("tier,variant,pkg,gen", PACKAGES, ids=IDS)
def test_manifest_hashes(tier, variant, pkg, gen):
    for name in ("kapicoder_manifest.json", "export_manifest.json"):
        m = load(pkg / name)
        assert m["files"], name
        for e in m["files"]:
            f = pkg / e["path"]
            assert f.is_file(), e["path"]
            assert f.stat().st_size == e["bytes"], e["path"]
            if FULL or e["bytes"] < 64 * 2**20:
                assert sha(f) == e["sha256"], e["path"]
    km = load(pkg / "kapicoder_manifest.json")
    assert km["schema"] == "kapilab-kapicoder-manifest-v1"
    assert "kapicoder_manifest.json" not in {e["path"] for e in km["files"]}
    em = load(pkg / "export_manifest.json")
    assert em["build"]["tool_version"] == "0.15.2"
    assert em["build"]["command"] and "onnxruntime_genai.models.builder" in em["build"]["command"]
    assert {e["path"] for e in em["files"]} == {p.name for p in pkg.iterdir() if p.is_file() and p.name != "export_manifest.json"}


@pytest.mark.parametrize("tier,variant,pkg,gen", PACKAGES, ids=IDS)
def test_genai_config(tier, variant, pkg, gen):
    c = load(pkg / "genai_config.json")
    assert c["model"]["type"] == "qwen2"                       # adapter Qwen da IDE
    assert c["model"]["eos_token_id"] == 151643
    assert c["search"]["do_sample"] is False
    assert c["search"]["max_length"] == 2048
    assert c["model"]["decoder"]["filename"] == "model.onnx"
    assert (pkg / "model.onnx").is_file()


@pytest.mark.parametrize("tier,variant,pkg,gen", PACKAGES, ids=IDS)
def test_kapi_metadata(tier, variant, pkg, gen):
    m = load(pkg / "kapibarastudio-model.json")
    assert m["schema"] == "kapilab-kapi-model-v1"
    assert m["contextContract"] == "editor-context-v1"
    assert m["architecture"] == "qwen2"
    assert m["runtime"]["minimumGenAi"] == "0.15.2"
    assert m["recommendedContextTokens"] == 2048 and m["generation"]["autocomplete"]["maxTokens"] == 32
    assert m["hardware"] == (["cpu"] if bv.VARIANTS[variant][1] == "cpu" else ["gpu"])
    assert m["license"] == "Apache-2.0" and "MODIFICADO" in m["modification"]
    assert m["datasetVersion"] == ("v3" if gen == "v3" else "v2c") and m["base"]["revision"] and m["base"]["repo_id"].startswith("Qwen/Qwen2.5-Coder")
    assert "<|fim_middle|>" in m["stopSequences"] and "<|endoftext|>" in m["stopSequences"]
    assert "npu" not in m and "agent" not in m and "reasoning" not in m
    assert (pkg / "LICENSE").read_text(encoding="utf-8").lstrip().startswith("Apache License")
    assert "modificad" in (pkg / "NOTICE.md").read_text(encoding="utf-8").lower()


@pytest.mark.parametrize("tier,variant,pkg,gen", PACKAGES, ids=IDS)
def test_tokenizer_is_byte_copy_of_merged(tier, variant, pkg, gen):
    merged = bv.merged_dir(tier, gen)
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"):
        assert sha(pkg / name) == sha(merged / name), name
    assert not (pkg / "chat_template.jinja").exists()


def test_variant_table_is_consistent():
    for v, (prec, ep, extra) in bv.VARIANTS.items():
        assert prec in ("fp32", "fp16", "int8", "int4") and ep in ("cpu", "dml")
        if ep == "dml" and prec == "int4":
            assert "shared_embeddings=false" in extra     # GatherBlockQuantized nao roda no DML
        if prec == "int8":
            assert "op_types_to_quantize=MatMul" in extra  # contorno do `-p int8` invalido (builder 0.15.2)


@pytest.mark.parametrize("tier,variant,pkg,gen", [x for x in PACKAGES if x[3] == "v3"] or [pytest.param(None, None, None, None, marks=pytest.mark.skip(reason="sem pacotes v3"))],
                         ids=[i for i, x in zip(IDS, PACKAGES) if x[3] == "v3"] or ["sem-v3"])
def test_v3_provenance_and_manifest_sources(tier, variant, pkg, gen):
    """Pacote v3: nome `<familia>-v3-ONNX-<variante>`, hashes do modelo de origem = export_info do merged/sft_v3, metadata ligada ao run_manifest do treino v3."""
    fam = bv.TIERS[tier]["family"]
    assert pkg.name == f"{fam}-v3-ONNX-{variant}"
    merged = bv.merged_dir(tier, "v3")
    info = load(merged / "export_info.json")
    em = load(pkg / "export_manifest.json")
    assert em["source"]["files_sha256"] == info["files_sha256"] and em["source"]["combined_sha256"] == info["combined_sha256"]
    assert em["source"]["merged_dir"].endswith("sft_v3") and em["generation"] == "v3" and em["source"]["dataset_version"] == "v3"
    m = load(pkg / "kapibarastudio-model.json")
    rm = load(ROOT / "Repos" / fam / "reports" / "sft_v3" / "run_manifest.json")
    assert m["training"]["run"] == "sft_v3"
    assert m["training"]["mergedWeightsSha256"] == info["files_sha256"]["model.safetensors"]
    assert m["training"]["corpusManifestSha256"] == rm["dataset"]["corpus_manifest_sha256"]
    assert m["training"]["trainConfigSha256"] == rm["config"]["sha256"]
    assert m["version"].startswith("3.") and "v3" in m["name"] and "eosWorkaround" not in m
    assert load(pkg / "genai_config.json")["model"]["eos_token_id"] == 151643     # pacote padrao: SEM o contorno do EOS (decisao do usuario pendente)
    assert "Apache" in (pkg / "LICENSE").read_text(encoding="utf-8")[:200] and "MODIFICADO" in m["modification"]


def test_package_dir_generations():
    assert bv.package_dir("t0", "int4", "").name == "KapiCoder-Mongo-Nano-ONNX-int4"
    assert bv.package_dir("t1", "dml-fp16", "v3").name == "KapiCoder-Mongo-Small-v3-ONNX-dml-fp16"
    assert bv.merged_dir("t0", "v3").name == "sft_v3" and bv.merged_dir("t1", "").name == "sft_distill_v1"
