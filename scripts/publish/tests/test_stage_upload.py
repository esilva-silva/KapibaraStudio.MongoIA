"""Testes do staging (hashes, arvore, ausencia de caminhos/segredos/dados, front matter, ONNX desligado por padrao) e do upload (so imprime; recusa sem aprovacao/credencial).

Os testes sinteticos montam um workspace minusculo em tmp_path; os testes `real_*` conferem o workspace de verdade (sem rede, somente leitura).
`KAPI_PUBLISH_RUN_EXAMPLE=1` (+ `KAPI_PUBLISH_PYTHON`) executa de verdade o examples/example_fim.py do staging, em CPU.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

import stage_hf_package as shp
import upload_to_hf as up

TEMPLATE = shp.TEMPLATE_EXAMPLE.read_text(encoding="utf-8")


def make_card(extra_tags: str = "", status: str = "PREVIEW / experimental, public preview", license_: str = "apache-2.0") -> str:
    example = TEMPLATE.replace("__REPO_ID__", "esilva/KapiCoder-Mongo-Nano")
    return f"""---
license: {license_}
base_model: Qwen/Qwen2.5-Coder-0.5B
library_name: transformers
pipeline_tag: text-generation
language:
- en
- pt
tags:
- mongodb
- code
- fim
- autocomplete
{extra_tags}---
# Test card
{status}

```python
{example}```
"""


CARD = make_card()

ONNX_CARD = """---
license: apache-2.0
base_model: esilva/KapiCoder-Mongo-Nano
pipeline_tag: text-generation
tags:
- onnx
- mongodb
---
# ONNX
PREVIEW. The `kapi` metadata serves Kapibara Studio. EOS problem of onnxruntime-genai 0.15.2: GetSequence.
"""


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tiny_safetensors() -> bytes:
    header = json.dumps({"w": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}).encode()
    return struct.pack("<Q", len(header)) + header + b"\x00\x01\x02\x03"


def write(p: Path, data) -> bytes:
    p.parent.mkdir(parents=True, exist_ok=True)
    b = data.encode("utf-8") if isinstance(data, str) else data
    p.write_bytes(b)
    return b


FAKE_GENERATED = {
    "examples/example_fim.py": TEMPLATE.replace("__REPO_ID__", "esilva/KapiCoder-Mongo-Nano").encode(),
    "evaluation/val_v2c_summary.json": b'{"kind": "val_v2c"}\n',
    "evaluation/homologation_v1_summary.json": b'{"kind": "homologation_v1"}\n',
    "provenance/model_lock.json": b'{"schema": "kapi-hf-model-lock/1"}\n',
}


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    """Workspace sintetico espelhando o layout real (tier nano) com a geracao de arquivos trocada por conteudo fixo."""
    spec = dict(shp.TIER_SPECS["nano"])
    ws = tmp_path / spec["ws"]
    merged = ws / "merged" / spec["run"]
    files = {
        "config.json": '{"architectures": ["Qwen2ForCausalLM"]}',
        "generation_config.json": '{"do_sample": false}',
        "merges.txt": "a b\n",
        "model.safetensors": tiny_safetensors(),
        "tokenizer.json": '{"version": "1.0"}',
        "tokenizer_config.json": '{"model_max_length": 32768}',
        "vocab.json": '{"a": 1}',
    }
    hashes = {name: sha(write(merged / name, data)) for name, data in files.items()}
    write(merged / "export_info.json", json.dumps({"path": "Repos\\x", "files_sha256": hashes}))
    write(merged / "train_state.json", '{"state": {}}')
    write(ws / "merged" / "sft_distill_v1" / "config.json", "{}")
    hf = ws / "hf"
    write(hf / "README.md", CARD)
    write(hf / "NOTICE.md", "# NOTICE\nmodified from Qwen\n")
    write(hf / "REPLAY_ATTRIBUTION.md", "# attribution\n")
    write(hf / "SOURCES.md", "C:\\IA\\internal\n")
    write(hf / "MODEL_CARD_REVIEW.md", "internal\n")
    write(hf / "ONNX_README.md", ONNX_CARD)
    write(hf / "ONNX_NOTICE.md", "# NOTICE\nONNX modified from Qwen\n")
    lic = write(ws / "LICENSE", "Apache License\n")
    write(tmp_path / "BaseRepos" / "base-repos.lock.json", json.dumps({"repos": {spec["base"]: {"files": [{"path": "LICENSE", "sha256": sha(lic)}]}}}))
    other = tmp_path / shp.TIER_SPECS["small"]["ws"]
    write(other / "merged" / "x.json", "{}")
    monkeypatch.setattr(shp, "generate_files", lambda plan: dict(FAKE_GENERATED))
    return tmp_path, spec


def add_onnx_packages(tree, only=None):
    root, spec = tree
    ws = root / spec["ws"]
    for dest, suffix in spec["onnx_variants"].items():
        if only and dest not in only:
            continue
        pkg = ws / "exports" / f"{spec['name']}-v3-ONNX-{suffix}"
        pf = {"model.onnx": b"onnx-bytes", "model.onnx.data": b"\x00" * 64, "genai_config.json": b"{}", "tokenizer.json": b"{}",
              "kapibarastudio-model.json": b"{}", "LICENSE": b"Apache License\n", "README.md": b"# pkg\n"}
        man_files = []
        for n, b in pf.items():
            write(pkg / n, b)
            man_files.append({"path": n, "sha256": sha(b), "bytes": len(b)})
        write(pkg / "kapicoder_manifest.json", json.dumps({"files": man_files}))
        write(pkg / "export_manifest.json", json.dumps({
            "source": {"merged_dir": "Repos/KapiCoder-Mongo-Nano/merged/sft_v3", "base": {"path": "BaseRepos/Qwen2.5-Coder-0.5B"}},
            "build": {"command": "C:\\Tools\\.venv\\Scripts\\python.exe -I -m builder -i C:\\IA\\x\\Repos\\KapiCoder-Mongo-Nano\\merged\\sft_v3 -o out", "tool_version": "0.15.2"}}))


def plan_for(tree, target="main"):
    root, spec = tree
    return shp.build_plan("nano", root=root, spec=spec, target=target)


# ----------------------------------------------------------------------------- repositorio principal
def test_clean_tree_passes_and_matches_the_authorized_tree(tree):
    plan = shp.run_checks(plan_for(tree))
    assert plan.errors == []
    assert {e.remote for e in plan.entries} == set(shp.MAIN_TREE)
    for e in plan.entries:
        if e.data is None:
            assert e.sha256 == sha(e.source.read_bytes())
            assert e.bytes == e.source.stat().st_size
        else:
            assert e.sha256 == sha(e.data) and e.bytes == len(e.data)
    excluded = {x["source"].rsplit("/", 1)[-1] for x in plan.excluded}
    assert {"train_state.json", "export_info.json", "SOURCES.md", "MODEL_CARD_REVIEW.md", "upload_manifest.json", "sft_distill_v1", "exports", "gguf"} <= excluded


def test_published_upload_manifest_lists_every_other_file_without_local_paths(tree):
    plan = shp.run_checks(plan_for(tree))
    by = {e.remote: e for e in plan.entries}
    pm = json.loads(by["provenance/upload_manifest.json"].data)
    assert pm["repo_id"] == "esilva/KapiCoder-Mongo-Nano" and pm["release_tag"] == "v0.1.0-preview"
    assert {f["path"] for f in pm["files"]} == set(shp.MAIN_TREE) - {"provenance/upload_manifest.json"}
    for f in pm["files"]:
        assert f["sha256"] == by[f["path"]].sha256 and f["bytes"] == by[f["path"]].bytes
    assert set(f for f in pm["files"][0]) == {"path", "bytes", "sha256"}   # sem "source": nada local


def test_unauthorized_or_missing_files_in_the_tree_are_errors(tree):
    root, spec = tree
    plan = plan_for(tree)
    ws = root / spec["ws"]
    plan.entries.append(shp.Entry("extra.json", ws / "merged" / spec["run"] / "config.json", "core"))
    shp.run_checks(plan)
    assert any("fora da arvore autorizada" in e and "extra.json" in e for e in plan.errors)
    plan2 = plan_for(tree)
    plan2.entries = [e for e in plan2.entries if e.remote != "NOTICE.md"]
    shp.run_checks(plan2)
    assert any("obrigatorio ausente" in e and "NOTICE.md" in e for e in plan2.errors)


def test_hash_mismatch_with_export_info_is_an_error(tree):
    root, spec = tree
    (root / spec["ws"] / "merged" / spec["run"] / "model.safetensors").write_bytes(tiny_safetensors() + b"x")
    plan = shp.run_checks(plan_for(tree))
    assert any("sha256" in e and "model.safetensors" in e for e in plan.errors)


@pytest.mark.parametrize("payload,needle", [
    ('{"path": "C:\\\\IA\\\\models\\\\x"}', "caminho Windows"),
    ('{"path": "C:/IA/models/x"}', "caminho Windows"),
    ('{"p": "/Users/someone/x"}', "caminho de usuario"),
    ('{"p": "Repos/KapiCoder-Mongo-Small/merged"}', "caminho do workspace"),
    ('{"p": "someone@gmail.com"}', "identificador pessoal"),
    ('{"p": "hf_' + "a" * 34 + '"}', "token Hugging Face"),
    ('{"p": "HF_TOKEN=abcdefghijk123"}', "HF_TOKEN"),
    ('{"p": "mongodb://user:secret@host/db"}', "URI MongoDB"),
    ('{"p": "-----BEGIN RSA PRIVATE KEY-----"}', "chave privada"),
])
def test_paths_and_secrets_in_text_files_are_rejected(tree, payload, needle):
    root, spec = tree
    cfg = root / spec["ws"] / "merged" / spec["run"] / "tokenizer_config.json"
    cfg.write_text(payload, encoding="utf-8")
    export = root / spec["ws"] / "merged" / spec["run"] / "export_info.json"
    info = json.loads(export.read_text(encoding="utf-8"))
    info["files_sha256"]["tokenizer_config.json"] = sha(cfg.read_bytes())  # isola a causa: so a varredura deve falhar
    export.write_text(json.dumps(info), encoding="utf-8")
    plan = shp.run_checks(plan_for(tree))
    assert any(needle in e for e in plan.errors), plan.errors


def test_email_in_a_card_is_an_error_and_in_a_generated_file_too(tree, monkeypatch):
    root, spec = tree
    (root / spec["ws"] / "hf" / "NOTICE.md").write_text("contact: someone@example.org\n", encoding="utf-8")
    assert any("e-mail" in e for e in shp.run_checks(plan_for(tree)).errors)
    (root / spec["ws"] / "hf" / "NOTICE.md").write_text("ok\n", encoding="utf-8")
    bad = dict(FAKE_GENERATED)
    bad["provenance/model_lock.json"] = b'{"owner": "someone@example.org", "p": "C:\\\\IA\\\\x"}\n'
    monkeypatch.setattr(shp, "generate_files", lambda plan: bad)
    errs = shp.run_checks(plan_for(tree)).errors
    assert any("provenance/model_lock.json" in e and "e-mail" in e for e in errs)
    assert any("provenance/model_lock.json" in e and "caminho Windows" in e for e in errs)


def test_url_with_scheme_is_not_a_windows_path():
    assert not shp.PATH_PATTERNS["caminho Windows"].search("see https://huggingface.co/x and http://localhost:8080/y")


def test_safetensors_header_with_path_is_rejected(tree):
    root, spec = tree
    header = json.dumps({"__metadata__": {"src": "C:\\\\IA\\\\x"}, "w": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    blob = struct.pack("<Q", len(header)) + header + b"\x00\x00"
    f = root / spec["ws"] / "merged" / spec["run"] / "model.safetensors"
    f.write_bytes(blob)
    export = root / spec["ws"] / "merged" / spec["run"] / "export_info.json"
    info = json.loads(export.read_text(encoding="utf-8"))
    info["files_sha256"]["model.safetensors"] = sha(blob)
    export.write_text(json.dumps(info), encoding="utf-8")
    plan = shp.run_checks(plan_for(tree))
    assert any("cabecalho" in e and "caminho" in e for e in plan.errors), plan.errors


def test_refused_names_and_extensions_and_foreign_sources(tree):
    root, spec = tree
    plan = plan_for(tree)
    ws = root / spec["ws"]
    cfg = ws / "merged" / spec["run"] / "config.json"
    bad = [
        shp.Entry("train_state.json", ws / "merged" / spec["run"] / "train_state.json", "core"),
        shp.Entry("data/train.jsonl", cfg, "core"),
        shp.Entry("optimizer.pt", cfg, "core"),
        shp.Entry("../escape.json", cfg, "core"),
        shp.Entry("C:/abs.json", cfg, "core"),
        shp.Entry("model.gguf", cfg, "core"),
        shp.Entry("notes.py", cfg, "core"),
        shp.Entry("examples/other.py", cfg, "core"),
        shp.Entry("checkpoints/x.json", cfg, "core"),
        shp.Entry("SOURCES.md", ws / "hf" / "SOURCES.md", "card"),
        shp.Entry("upload_manifest.json", cfg, "core"),
        shp.Entry("other.json", root / shp.TIER_SPECS["small"]["ws"] / "merged" / "x.json", "core"),
        shp.Entry("elsewhere.json", root / "BaseRepos" / "base-repos.lock.json", "core"),
        shp.Entry("old.json", ws / "merged" / "sft_distill_v1" / "config.json", "core"),
    ]
    plan.entries.extend(bad)
    shp.run_checks(plan, do_hash=False)
    text = "\n".join(plan.errors)
    for e in bad:
        assert any(e.remote in err for err in plan.errors), f"nao rejeitou {e.remote}\n{text}"
    assert "outro workspace" in text and "fora do workspace" in text
    # o unico .py e o unico upload_manifest.json autorizados
    ok = shp.Plan("nano", root, spec)
    shp.check_remote_name(shp.Entry("examples/example_fim.py", None, "generated"), ok)
    shp.check_remote_name(shp.Entry("provenance/upload_manifest.json", None, "generated"), ok)
    assert ok.errors == []


def test_card_front_matter_problems_are_errors(tree):
    root, spec = tree
    readme = root / spec["ws"] / "hf" / "README.md"
    readme.write_text(make_card(license_="mit"), encoding="utf-8")
    assert any("license" in e for e in shp.run_checks(plan_for(tree), do_hash=False).errors)
    readme.write_text("# no front matter\nPREVIEW\n", encoding="utf-8")
    assert any("front matter" in e for e in shp.run_checks(plan_for(tree), do_hash=False).errors)
    readme.write_text(make_card(status="ready"), encoding="utf-8")
    assert any("PREVIEW" in e for e in shp.run_checks(plan_for(tree), do_hash=False).errors)
    readme.write_text(make_card(status="PREVIEW / experimental"), encoding="utf-8")
    assert any("public preview" in e for e in shp.run_checks(plan_for(tree), do_hash=False).errors)
    readme.write_text(make_card(extra_tags="- onnx\n"), encoding="utf-8")
    assert any("onnx" in e for e in shp.run_checks(plan_for(tree), do_hash=False).errors)
    readme.write_text(CARD.replace("- autocomplete\n", ""), encoding="utf-8")
    assert any("tags" in e for e in shp.run_checks(plan_for(tree), do_hash=False).errors)
    readme.write_text(CARD.replace("add_special_tokens=False", "add_special_tokens=True"), encoding="utf-8")
    assert any("exemplo do card" in e for e in shp.run_checks(plan_for(tree), do_hash=False).errors)


def test_license_must_match_base_license(tree):
    root, spec = tree
    (root / spec["ws"] / "LICENSE").write_text("not apache\n", encoding="utf-8")
    assert any("LICENSE" in e and "sha256" in e for e in shp.run_checks(plan_for(tree)).errors)


def test_stage_copies_verified_files_including_generated(tree, tmp_path):
    root, spec = tree
    plan = shp.run_checks(plan_for(tree))
    dest = tmp_path / "stage_out"
    assert not dest.exists()
    shp.stage(plan, dest)
    got = {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()}
    assert got == set(shp.MAIN_TREE)
    assert "train_state.json" not in got and "SOURCES.md" not in got
    assert (dest / "examples" / "example_fim.py").read_bytes() == FAKE_GENERATED["examples/example_fim.py"]
    with pytest.raises(SystemExit):
        shp.stage(plan, dest)  # destino nao vazio
    with pytest.raises(SystemExit):
        shp.stage(plan, dest, clean=True)  # --clean so vale dentro de cache/hf_stage


def test_stage_clean_only_inside_cache_hf_stage(tree):
    root, spec = tree
    plan = shp.run_checks(plan_for(tree))
    dest = root / "cache" / "hf_stage" / "nano"
    shp.stage(plan, dest)
    shp.stage(plan, dest, clean=True)
    assert (dest / "model.safetensors").exists()


def test_stage_refuses_protected_destination(tree):
    root, spec = tree
    plan = shp.run_checks(plan_for(tree), do_hash=False)
    with pytest.raises(SystemExit):
        shp.stage(plan, root / spec["ws"] / "hf" / "out")
    with pytest.raises(SystemExit):
        shp.stage(plan, root / "scripts" / "out")


def test_default_stage_dir_is_outside_git():
    assert shp.DEFAULT_STAGE_BASE == shp.ROOT / "cache" / "hf_stage"
    ignored = (shp.ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "cache/" in ignored


def test_manifest_and_check_manifest(tree):
    root, spec = tree
    plan = shp.run_checks(plan_for(tree))
    m = shp.manifest(plan)
    assert m["suggested_visibility"] == "private" and m["release_tag"] == "v0.1.0-preview" and m["target"] == "main"
    write(shp.manifest_path(plan), json.dumps(m))
    assert shp.check_manifest(shp.run_checks(plan_for(tree)), full=True) == []
    (root / spec["ws"] / "hf" / "NOTICE.md").write_text("# changed\n", encoding="utf-8")
    assert shp.check_manifest(shp.run_checks(plan_for(tree)), full=True)


# ----------------------------------------------------------------------------- ONNX (desligado por padrao)
def test_onnx_is_off_by_default(tree):
    add_onnx_packages(tree)
    plan = shp.run_checks(plan_for(tree))
    assert plan.errors == [] and not any(e.kind == "onnx" for e in plan.entries)
    assert not any("onnx" in e.remote.lower() for e in plan.entries)


def test_onnx_plan_builds_self_contained_variant_folders_and_sanitizes_export_manifest(tree):
    add_onnx_packages(tree)
    plan = shp.run_checks(plan_for(tree, target="onnx"))
    assert plan.errors == [], plan.errors
    remotes = {e.remote for e in plan.entries}
    assert {"README.md", "LICENSE", "NOTICE.md"} <= remotes
    for variant in ("cpu-int4", "cpu-int8", "dml-fp16"):
        for must in shp.ONNX_REQUIRED_PER_VARIANT:
            assert f"{variant}/{must}" in remotes
    assert plan.repo_id == "esilva/KapiCoder-Mongo-Nano-ONNX"
    em = next(e for e in plan.entries if e.remote == "cpu-int4/export_manifest.json")
    text = em.data.decode("utf-8")
    assert "Repos" not in text and "C:" not in text and "BaseRepos" not in text
    assert json.loads(text)["sanitized"]
    assert "sft_v3" in text
    m = shp.manifest(plan)
    assert m["target"] == "onnx" and m["suggested_repo_id"] == "esilva/KapiCoder-Mongo-Nano-ONNX"


def test_onnx_missing_variant_or_card_is_an_error(tree):
    add_onnx_packages(tree, only={"cpu-int4", "cpu-int8"})
    plan = shp.run_checks(plan_for(tree, target="onnx"))
    assert any("pacote ONNX v3 ausente" in e and "dml-fp16" in e for e in plan.errors)
    root, spec = tree
    (root / spec["ws"] / "hf" / "ONNX_README.md").unlink()
    assert any("card do repositorio ONNX ausente" in e for e in shp.run_checks(plan_for(tree, target="onnx")).errors)


def test_onnx_card_must_warn_about_kapi_metadata_and_eos(tree):
    add_onnx_packages(tree)
    root, spec = tree
    (root / spec["ws"] / "hf" / "ONNX_README.md").write_text(ONNX_CARD.replace("GetSequence", "x"), encoding="utf-8")
    assert any("GetSequence" in e for e in shp.run_checks(plan_for(tree, target="onnx")).errors)


def test_redact_paths_keeps_only_last_component():
    out = shp.redact_paths("C:\\Tools\\.venv\\Scripts\\python.exe -I -i C:\\IA\\x\\Repos\\KapiCoder-Mongo-Nano\\merged\\sft_v3 -o out -c tmp\\cache_t0")
    assert out == "python.exe -I -i sft_v3 -o out -c cache_t0"
    assert shp.sanitize_json({"a": ["BaseRepos/Qwen2.5-Coder-0.5B"], "n": 3}) == {"a": ["Qwen2.5-Coder-0.5B"], "n": 3}


# ----------------------------------------------------------------------------- upload_to_hf
def make_stage_and_manifest(tree, tmp_path, monkeypatch):
    root, spec = tree
    plan = shp.run_checks(plan_for(tree))
    assert plan.errors == []
    write(shp.manifest_path(plan), json.dumps(shp.manifest(plan)))
    dest = tmp_path / "out" / "nano"
    shp.stage(plan, dest)
    monkeypatch.setitem(up.TIER_SPECS, "nano", spec)
    return dest


def no_hub(monkeypatch):
    for name in ("_do_create", "_do_upload", "_do_tag", "fetch_remote"):
        monkeypatch.setattr(up, name, lambda *a, **k: pytest.fail("nao deveria tocar o Hub"))


def test_upload_default_only_prints_exact_commands_and_never_imports_hub(tree, tmp_path, monkeypatch, capsys):
    dest = make_stage_and_manifest(tree, tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # um import falharia
    no_hub(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf_" + "z" * 30)  # mesmo com token no ambiente, sem a flag nada roda
    rc = up.main(["--tier", "nano", "--stage-dir", str(dest)], root=tree[0])
    out = capsys.readouterr().out
    assert rc == 0
    assert "hf repo create esilva/KapiCoder-Mongo-Nano --repo-type model --private" in out
    assert f'hf upload esilva/KapiCoder-Mongo-Nano "{dest}" . --repo-type model --private' in out
    assert 'hf repo tag create esilva/KapiCoder-Mongo-Nano v0.1.0-preview' in out
    assert "--steps verify" in out and "create_tag" in out and "private=True" in out
    assert "DRY-RUN: nada foi enviado" in out
    assert "zzzz" not in out  # o token nunca e impresso


def test_upload_with_flag_but_without_credential_is_refused(tree, tmp_path, monkeypatch, capsys):
    dest = make_stage_and_manifest(tree, tmp_path, monkeypatch)
    no_hub(monkeypatch)
    rc = up.main(["--tier", "nano", "--stage-dir", str(dest), "--i-have-user-approval"], root=tree[0])
    assert rc == up.EXIT_NOT_APPROVED
    assert "HF_TOKEN esta definido" in capsys.readouterr().out


def test_upload_runs_steps_in_order_only_with_flag_and_credential_and_is_private_by_default(tree, tmp_path, monkeypatch, capsys):
    dest = make_stage_and_manifest(tree, tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(up, "_do_create", lambda repo, private: calls.append(("create", repo, private)))
    monkeypatch.setattr(up, "_do_upload", lambda repo, d, msg: calls.append(("upload", repo, Path(d))))
    monkeypatch.setattr(up, "_do_tag", lambda repo, tag, msg: calls.append(("tag", repo, tag)))
    monkeypatch.setattr(up, "fetch_remote", lambda repo, rev: calls.append(("fetch", repo, rev)) or ok_remote(dest))
    monkeypatch.setenv("HF_TOKEN", "hf_" + "z" * 30)
    rc = up.main(["--tier", "nano", "--stage-dir", str(dest), "--i-have-user-approval"], root=tree[0])
    assert rc == 0
    assert calls == [("create", "esilva/KapiCoder-Mongo-Nano", True), ("upload", "esilva/KapiCoder-Mongo-Nano", dest),
                     ("tag", "esilva/KapiCoder-Mongo-Nano", "v0.1.0-preview"), ("fetch", "esilva/KapiCoder-Mongo-Nano", "v0.1.0-preview")]
    out = capsys.readouterr().out
    assert "zzzz" not in out and "verificacao OK" in out
    calls.clear()
    monkeypatch.delenv("HF_TOKEN")
    up.main(["--tier", "nano", "--stage-dir", str(dest), "--i-have-user-approval", "--user-is-logged-in", "--public", "--steps", "create"], root=tree[0])
    assert calls == [("create", "esilva/KapiCoder-Mongo-Nano", False)]


def ok_remote(stage: Path) -> dict:
    manifest = {p.relative_to(stage).as_posix(): p for p in stage.rglob("*") if p.is_file()}
    out = {}
    for rel, p in manifest.items():
        big = p.stat().st_size > 100
        out[rel] = {"size": p.stat().st_size, "lfs_sha256": up.sha256_file(p) if big else None, "blob_id": up.git_blob_sha1(p)}
    out[".gitattributes"] = {"size": 10, "lfs_sha256": None, "blob_id": "0" * 40}
    return out


def test_upload_verify_failure_has_its_own_exit_code(tree, tmp_path, monkeypatch, capsys):
    dest = make_stage_and_manifest(tree, tmp_path, monkeypatch)
    remote = ok_remote(dest)
    remote["config.json"] = dict(remote["config.json"], blob_id="1" * 40)
    monkeypatch.setattr(up, "fetch_remote", lambda repo, rev: remote)
    monkeypatch.setenv("HF_TOKEN", "hf_" + "z" * 30)
    rc = up.main(["--tier", "nano", "--stage-dir", str(dest), "--i-have-user-approval", "--steps", "verify"], root=tree[0])
    assert rc == up.EXIT_VERIFY_FAILED
    assert "config.json" in capsys.readouterr().out


def test_upload_refuses_a_stage_dir_that_differs_from_manifest(tree, tmp_path, monkeypatch, capsys):
    dest = make_stage_and_manifest(tree, tmp_path, monkeypatch)
    no_hub(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "hf_" + "z" * 30)
    (dest / "config.json").write_text("{}", encoding="utf-8")
    assert up.main(["--tier", "nano", "--stage-dir", str(dest), "--i-have-user-approval"], root=tree[0]) == up.EXIT_BAD_STAGE
    (dest / "config.json").write_text('{"architectures": ["Qwen2ForCausalLM"]}', encoding="utf-8")
    (dest / "extra.json").write_text("{}", encoding="utf-8")
    assert up.main(["--tier", "nano", "--stage-dir", str(dest), "--i-have-user-approval"], root=tree[0]) == up.EXIT_BAD_STAGE
    assert "FORA do manifesto" in capsys.readouterr().out


def test_compare_remote_matches_lfs_sha256_and_git_blob_ids(tmp_path):
    stage = tmp_path / "s"
    big = write(stage / "model.safetensors", b"x" * 500)
    small = write(stage / "README.md", "hello\n")
    expected = {"model.safetensors": sha(big), "README.md": sha(small)}
    sizes = {"model.safetensors": 500, "README.md": len(small)}
    good = {"model.safetensors": {"size": 500, "lfs_sha256": sha(big), "blob_id": "a" * 40},
            "README.md": {"size": len(small), "lfs_sha256": None, "blob_id": up.git_blob_sha1(stage / "README.md")},
            ".gitattributes": {"size": 5, "lfs_sha256": None, "blob_id": "b" * 40}}
    assert up.compare_remote(expected, sizes, good, stage) == []
    # git: sha1("blob 6\0hello\n") conhecido
    assert up.git_blob_sha1(stage / "README.md") == hashlib.sha1(b"blob 6\0hello\n").hexdigest()  # noqa: S324
    bad = json.loads(json.dumps(good))
    bad["model.safetensors"]["lfs_sha256"] = "0" * 64
    bad["README.md"]["blob_id"] = "0" * 40
    bad["extra.bin"] = {"size": 1, "lfs_sha256": None, "blob_id": "c" * 40}
    errs = up.compare_remote(expected, sizes, bad, stage)
    assert len(errs) == 3 and any("extra.bin" in e for e in errs) and any("model.safetensors" in e for e in errs) and any("README.md" in e for e in errs)
    missing = {k: v for k, v in good.items() if k != "README.md"}
    assert any("ausente no Hub" in e for e in up.compare_remote(expected, sizes, missing, stage))
    wrong_size = json.loads(json.dumps(good))
    wrong_size["model.safetensors"]["size"] = 1
    assert any("tamanho" in e for e in up.compare_remote(expected, sizes, wrong_size, stage))


def test_upload_script_has_no_token_argument_and_no_network_libs():
    src = (Path(up.__file__)).read_text(encoding="utf-8")
    assert "--token" not in src
    for lib in ("import requests", "import urllib", "import http.client", "import socket"):
        assert lib not in src
    assert not [ln for ln in src.splitlines() if ln.startswith(("import huggingface_hub", "from huggingface_hub"))], "huggingface_hub so dentro das funcoes de rede"
    s2 = Path(shp.__file__).read_text(encoding="utf-8")
    for lib in ("import requests", "import urllib", "import http.client", "import socket", "huggingface_hub", "os.environ"):
        assert lib not in s2


# ----------------------------------------------------------------------------- workspace real
@pytest.mark.parametrize("tier", ["nano", "small"])
def test_real_workspace_plan_is_clean_and_manifest_is_current(tier):
    spec = shp.TIER_SPECS[tier]
    merged = shp.ROOT / spec["ws"] / "merged" / spec["run"] / "model.safetensors"
    manifest_path = shp.ROOT / spec["ws"] / "hf" / "upload_manifest.json"
    if not merged.exists() or not manifest_path.exists():
        pytest.skip("pesos ou manifesto ausentes neste checkout")
    plan = shp.run_checks(shp.build_plan(tier))
    assert plan.errors == []
    assert {e.remote for e in plan.entries} == set(shp.MAIN_TREE)
    assert shp.check_manifest(plan, full=True) == [], "regenere com `stage_hf_package.py --write-manifest`"
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    for f in m["files"]:
        p = f["path"]
        assert not shp.PATH_PATTERNS["caminho Windows"].search(p) and ".." not in p and not p.startswith("/")
        assert len(f["sha256"]) == 64
    assert m["suggested_visibility"] == "private" and m["suggested_repo_id"] == f"esilva/{spec['name']}"
    # sha do manifesto == sha registrado no export_info do proprio modelo
    info = json.loads((merged.parent / "export_info.json").read_text(encoding="utf-8"))["files_sha256"]
    for f in m["files"]:
        if f["path"] in info:
            assert f["sha256"] == info[f["path"]]
    # so pesos v3; nenhum arquivo do outro tier
    other = "Small" if tier == "nano" else "Nano"
    assert not any(other in f["source"] for f in m["files"])
    assert all("sft_v3" in f["source"] or not f["source"].startswith(f"{spec['ws']}/merged") for f in m["files"])
    assert not any("exports" in f["source"] or "gguf" in f["source"] for f in m["files"])


@pytest.mark.parametrize("tier", ["nano", "small"])
def test_real_generated_files_are_aggregates_without_local_paths(tier):
    spec = shp.TIER_SPECS[tier]
    if not (shp.ROOT / spec["ws"] / "merged" / spec["run"] / "model.safetensors").exists():
        pytest.skip("pesos ausentes")
    plan = shp.run_checks(shp.build_plan(tier))
    assert plan.errors == []
    by = {e.remote: e for e in plan.entries}
    lock = json.loads(by["provenance/model_lock.json"].data)
    base = json.loads((shp.ROOT / spec["ws"] / "model.lock.json").read_text(encoding="utf-8"))["base"]
    assert lock["base_model"]["revision"] == base["revision"] and lock["base_model"]["repo_id"] == base["repo_id"]
    info = json.loads((shp.ROOT / spec["ws"] / "merged" / spec["run"] / "export_info.json").read_text(encoding="utf-8"))
    assert lock["weights"]["files_sha256"] == dict(sorted(info["files_sha256"].items()))
    assert lock["dataset"]["published"] is False and lock["training"]["single_training_run"] is True
    assert lock["model"]["release_tag"] == "v0.1.0-preview" and len(lock["training"]["git_commit"]) == 40
    hom = json.loads(by["evaluation/homologation_v1_summary.json"].data)
    src = json.loads((shp.ROOT / "reports/datasets" / f"homolog-v3-eval-{spec['t']}.json").read_text(encoding="utf-8"))
    assert hom["set_sha256"] == src["set_sha256"] and hom["n"] == 6030
    assert hom["models"] == src["models"]
    assert "gpu_evidence" not in hom
    val = json.loads(by["evaluation/val_v2c_summary.json"].data)
    assert val["n"] == 2000 and "this_model_v3" in val["metrics_by_model"] and "path" not in val["dataset"]
    for name in ("evaluation/val_v2c_summary.json", "evaluation/homologation_v1_summary.json", "provenance/model_lock.json"):
        text = by[name].data.decode("utf-8")
        for rx_name, rx in {**shp.PATH_PATTERNS, **shp.SECRET_PATTERNS}.items():
            assert not rx.search(text), f"{name}: {rx_name}"
        assert not shp.EMAIL.search(text)
    # nenhum texto de item: so chaves agregadas conhecidas
    assert not any(k in json.dumps(hom) for k in ('"prefix"', '"suffix"', '"middle"', '"prompt"', '"prediction"'))
    ex = by["examples/example_fim.py"].data.decode("utf-8")
    assert f'"esilva/{spec["name"]}"' in ex and "__REPO_ID__" not in ex


@pytest.mark.parametrize("tier", ["nano", "small"])
def test_real_card_hash_in_manifest_matches_card_on_disk(tier):
    spec = shp.TIER_SPECS[tier]
    mp = shp.ROOT / spec["ws"] / "hf" / "upload_manifest.json"
    if not mp.exists():
        pytest.skip("manifesto ausente")
    m = json.loads(mp.read_text(encoding="utf-8"))
    for f in m["files"]:
        if f["kind"] in ("card", "license"):
            assert shp.sha256_file(shp.ROOT / f["source"]) == f["sha256"], f"{f['path']}: regenere com --write-manifest"


def test_replay_attribution_is_up_to_date():
    import build_replay_attribution as bra
    if not bra.SRC.exists():
        pytest.skip("LICENSES.md ausente")
    assert bra.main(["--check"]) == 0


@pytest.mark.skipif(os.environ.get("KAPI_PUBLISH_RUN_EXAMPLE") != "1", reason="define KAPI_PUBLISH_RUN_EXAMPLE=1 e KAPI_PUBLISH_PYTHON (python com torch+transformers) para executar o exemplo")
@pytest.mark.parametrize("tier", ["nano", "small"])
def test_real_example_runs_on_cpu_with_the_staged_weights(tier):
    stage = shp.DEFAULT_STAGE_BASE / tier
    if not (stage / "model.safetensors").exists():
        pytest.skip("rode stage_hf_package.py --stage antes")
    env = dict(os.environ, HIP_VISIBLE_DEVICES="-1", CUDA_VISIBLE_DEVICES="-1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    py = os.environ.get("KAPI_PUBLISH_PYTHON", sys.executable)
    res = subprocess.run([py, "-I", str(stage / "examples" / "example_fim.py")], capture_output=True, text=True, env=env, timeout=1200)
    assert res.returncode == 0, res.stderr[-2000:]
    assert res.stdout.strip().startswith("'") and len(res.stdout.strip()) > 2
