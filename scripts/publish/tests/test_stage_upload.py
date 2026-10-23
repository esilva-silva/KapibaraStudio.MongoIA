"""Testes do staging (hashes, ausencia de caminhos/segredos/dados, front matter) e do script de upload (recusa sem aprovacao/token).

Os testes sinteticos montam um workspace minusculo em tmp_path; os testes `real_*` conferem o workspace de verdade (sem rede,
somente leitura). `KAPI_PUBLISH_FULL_HASH=1` re-hasheia tambem os pesos grandes.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import pytest

import stage_hf_package as shp
import upload_to_hf as up

FULL = os.environ.get("KAPI_PUBLISH_FULL_HASH") == "1"

CARD = """---
license: apache-2.0
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
- onnx
---
# Test card
PREVIEW / experimental
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


@pytest.fixture()
def tree(tmp_path):
    """Workspace sintetico espelhando o layout real (tier nano)."""
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
    hf = ws / "hf"
    write(hf / "README.md", CARD)
    write(hf / "NOTICE.md", "# NOTICE\nmodified from Qwen\n")
    write(hf / "REPLAY_ATTRIBUTION.md", "# attribution\n")
    write(hf / "SOURCES.md", "C:\\IA\\internal\n")
    write(hf / "MODEL_CARD_REVIEW.md", "internal\n")
    lic = write(ws / "LICENSE", "Apache License\n")
    write(tmp_path / "BaseRepos" / "base-repos.lock.json", json.dumps({"repos": {spec["base"]: {"files": [{"path": "LICENSE", "sha256": sha(lic)}]}}}))
    for v in spec["onnx_default"]:
        pkg = ws / "exports" / f"{spec['name']}-ONNX-{v}"
        pf = {"model.onnx": b"onnx-bytes", "model.onnx.data": b"\x00" * 64, "genai_config.json": b"{}", "LICENSE": b"Apache License\n"}
        man_files = []
        for n, b in pf.items():
            write(pkg / n, b)
            man_files.append({"path": n, "sha256": sha(b), "bytes": len(b)})
        write(pkg / "kapicoder_manifest.json", json.dumps({"files": man_files}))
        write(pkg / "export_manifest.json", '{"source": {"merged_dir": "Repos/KapiCoder-Mongo-Nano/merged"}}')
    other = tmp_path / shp.TIER_SPECS["small"]["ws"]
    write(other / "merged" / "x.json", "{}")
    return tmp_path, spec


def plan_for(tree, onnx=None):
    root, spec = tree
    return shp.build_plan("nano", root=root, onnx=onnx, spec=spec)


def test_clean_tree_passes_and_hashes_match(tree):
    plan = shp.run_checks(plan_for(tree))
    assert plan.errors == []
    by = {e.remote: e for e in plan.entries}
    assert set(by) == {*shp.CORE_FILES, *shp.CARD_FILES, "LICENSE"}
    for e in plan.entries:
        assert e.sha256 == sha(e.source.read_bytes())
        assert e.bytes == e.source.stat().st_size
    excluded = {x["source"].rsplit("/", 1)[-1] for x in plan.excluded}
    assert {"train_state.json", "export_info.json", "SOURCES.md", "MODEL_CARD_REVIEW.md", "upload_manifest.json"} <= excluded


def test_onnx_optional_entries_follow_package_manifest_and_skip_export_manifest(tree):
    plan = shp.run_checks(plan_for(tree, onnx=["int4"]))
    assert plan.errors == []
    remotes = {e.remote for e in plan.entries}
    assert "onnx/int4/model.onnx.data" in remotes and "onnx/int4/kapicoder_manifest.json" in remotes
    assert not any("export_manifest.json" in r for r in remotes)
    assert not any(r.startswith("onnx/dml-fp16/") for r in remotes)


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
    bad = [
        shp.Entry("train_state.json", ws / "merged" / spec["run"] / "train_state.json", "core"),
        shp.Entry("data/train.jsonl", ws / "merged" / spec["run"] / "config.json", "core"),
        shp.Entry("optimizer.pt", ws / "merged" / spec["run"] / "config.json", "core"),
        shp.Entry("../escape.json", ws / "merged" / spec["run"] / "config.json", "core"),
        shp.Entry("C:/abs.json", ws / "merged" / spec["run"] / "config.json", "core"),
        shp.Entry("model.gguf", ws / "merged" / spec["run"] / "config.json", "core"),
        shp.Entry("notes.py", ws / "merged" / spec["run"] / "config.json", "core"),
        shp.Entry("checkpoints/x.json", ws / "merged" / spec["run"] / "config.json", "core"),
        shp.Entry("SOURCES.md", ws / "hf" / "SOURCES.md", "card"),
        shp.Entry("other.json", root / shp.TIER_SPECS["small"]["ws"] / "merged" / "x.json", "core"),
        shp.Entry("elsewhere.json", root / "BaseRepos" / "base-repos.lock.json", "core"),
    ]
    plan.entries.extend(bad)
    shp.run_checks(plan, do_hash=False)
    text = "\n".join(plan.errors)
    for e in bad:
        assert any(e.remote in err for err in plan.errors), f"nao rejeitou {e.remote}\n{text}"
    assert "outro workspace" in text and "fora do workspace" in text


def test_card_front_matter_problems_are_errors(tree):
    root, spec = tree
    readme = root / spec["ws"] / "hf" / "README.md"
    readme.write_text(CARD.replace("license: apache-2.0", "license: mit"), encoding="utf-8")
    assert any("license" in e for e in shp.run_checks(plan_for(tree), do_hash=False).errors)
    readme.write_text("# no front matter\nPREVIEW\n", encoding="utf-8")
    assert any("front matter" in e for e in shp.run_checks(plan_for(tree), do_hash=False).errors)
    readme.write_text(CARD.replace("PREVIEW / experimental", "ready"), encoding="utf-8")
    assert any("PREVIEW" in e for e in shp.run_checks(plan_for(tree), do_hash=False).errors)
    readme.write_text(CARD.replace("- onnx\n", ""), encoding="utf-8")
    assert any("tags" in e for e in shp.run_checks(plan_for(tree), do_hash=False).errors)


def test_license_must_match_base_license(tree):
    root, spec = tree
    (root / spec["ws"] / "LICENSE").write_text("not apache\n", encoding="utf-8")
    assert any("LICENSE" in e and "sha256" in e for e in shp.run_checks(plan_for(tree)).errors)


def test_dry_run_default_copies_nothing_and_stage_copies_verified_files(tree, tmp_path, monkeypatch):
    root, spec = tree
    monkeypatch.setattr(shp, "ROOT", root)
    monkeypatch.setitem(shp.TIER_SPECS, "nano", spec)
    plan = shp.run_checks(plan_for(tree, onnx=["int4"]))
    dest = tmp_path / "stage_out"
    assert not dest.exists()
    shp.stage(plan, dest)
    got = {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()}
    assert got == {e.remote for e in plan.entries}
    assert "train_state.json" not in got and "SOURCES.md" not in got
    with pytest.raises(SystemExit):
        shp.stage(plan, dest)  # destino nao vazio


def test_stage_refuses_protected_destination(tree):
    root, spec = tree
    plan = shp.run_checks(plan_for(tree), do_hash=False)
    with pytest.raises(SystemExit):
        shp.stage(plan, root / spec["ws"] / "hf" / "out")


# ----------------------------------------------------------------------------- upload_to_hf
def make_stage_and_manifest(tree, tmp_path, monkeypatch):
    root, spec = tree
    monkeypatch.setattr(shp, "ROOT", root)
    plan = shp.run_checks(plan_for(tree, onnx=list(spec["onnx_default"])))
    assert plan.errors == []
    manifest = shp.manifest(plan)
    write(root / spec["ws"] / "hf" / "upload_manifest.json", json.dumps(manifest))
    stage_plan = shp.run_checks(plan_for(tree))
    dest = tmp_path / "out" / "nano"
    shp.stage(stage_plan, dest)
    monkeypatch.setitem(up.TIER_SPECS, "nano", spec)
    return dest


def test_upload_default_only_prints_and_never_imports_hub(tree, tmp_path, monkeypatch, capsys):
    dest = make_stage_and_manifest(tree, tmp_path, monkeypatch)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # um import falharia
    monkeypatch.setattr(up, "_do_upload", lambda *a, **k: pytest.fail("upload nao deveria rodar"))
    monkeypatch.setenv("HF_TOKEN", "hf_" + "z" * 30)  # mesmo com token no ambiente, sem a flag nada roda
    rc = up.main(["--tier", "nano", "--stage-dir", str(dest)], root=tree[0])
    out = capsys.readouterr().out
    assert rc == 0
    assert "huggingface-cli upload esilva/KapiCoder-Mongo-Nano" in out and "--private" in out
    assert "DRY-RUN: nada foi enviado" in out
    assert "zzzz" not in out  # o token nunca e impresso


def test_upload_with_flag_but_without_token_is_refused(tree, tmp_path, monkeypatch, capsys):
    dest = make_stage_and_manifest(tree, tmp_path, monkeypatch)
    monkeypatch.setattr(up, "_do_upload", lambda *a, **k: pytest.fail("upload nao deveria rodar"))
    rc = up.main(["--tier", "nano", "--stage-dir", str(dest), "--i-have-user-approval"], root=tree[0])
    assert rc == up.EXIT_NOT_APPROVED
    assert "HF_TOKEN nao esta definido" in capsys.readouterr().out


def test_upload_runs_only_with_flag_and_token_and_is_private_by_default(tree, tmp_path, monkeypatch, capsys):
    dest = make_stage_and_manifest(tree, tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(up, "_do_upload", lambda repo, d, private, msg: calls.append((repo, Path(d), private)))
    monkeypatch.setenv("HF_TOKEN", "hf_" + "z" * 30)
    rc = up.main(["--tier", "nano", "--stage-dir", str(dest), "--i-have-user-approval"], root=tree[0])
    assert rc == 0 and calls == [("esilva/KapiCoder-Mongo-Nano", dest, True)]
    assert "zzzz" not in capsys.readouterr().out
    calls.clear()
    up.main(["--tier", "nano", "--stage-dir", str(dest), "--i-have-user-approval", "--public"], root=tree[0])
    assert calls and calls[0][2] is False


def test_upload_refuses_a_stage_dir_that_differs_from_manifest(tree, tmp_path, monkeypatch, capsys):
    dest = make_stage_and_manifest(tree, tmp_path, monkeypatch)
    monkeypatch.setattr(up, "_do_upload", lambda *a, **k: pytest.fail("upload nao deveria rodar"))
    monkeypatch.setenv("HF_TOKEN", "hf_" + "z" * 30)
    (dest / "config.json").write_text("{}", encoding="utf-8")
    assert up.main(["--tier", "nano", "--stage-dir", str(dest), "--i-have-user-approval"], root=tree[0]) == up.EXIT_BAD_STAGE
    (dest / "config.json").write_text('{"architectures": ["Qwen2ForCausalLM"]}', encoding="utf-8")
    (dest / "extra.json").write_text("{}", encoding="utf-8")
    assert up.main(["--tier", "nano", "--stage-dir", str(dest), "--i-have-user-approval"], root=tree[0]) == up.EXIT_BAD_STAGE
    assert "FORA do manifesto" in capsys.readouterr().out


def test_upload_script_has_no_token_argument_and_no_network_libs():
    src = (Path(up.__file__)).read_text(encoding="utf-8")
    assert "--token" not in src
    for lib in ("import requests", "import urllib", "import http.client", "import socket"):
        assert lib not in src
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
    plan = shp.run_checks(shp.build_plan(tier, onnx=list(spec["onnx_default"])), do_hash=FULL)
    assert plan.errors == []
    assert shp.check_manifest(plan, full=FULL) == []
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = [f["path"] for f in m["files"]] + [f["path"] for v in m["optional_onnx"].values() for f in v]
    for p in paths:
        assert not shp.PATH_PATTERNS["caminho Windows"].search(p) and ".." not in p and not p.startswith("/")
        assert Path(p).name not in shp.REFUSE_NAMES and not p.lower().endswith(shp.REFUSE_SUFFIXES)
    assert all(f["sha256"] and len(f["sha256"]) == 64 for f in m["files"])
    assert all(f["sha256"] for v in m["optional_onnx"].values() for f in v)
    names = {f["path"] for f in m["files"]}
    assert {"model.safetensors", "config.json", "tokenizer.json", "README.md", "LICENSE", "NOTICE.md", "REPLAY_ATTRIBUTION.md"} <= names
    assert m["suggested_visibility"] == "private"
    # sha do manifesto == sha registrado no export_info do proprio modelo
    info = json.loads((merged.parent / "export_info.json").read_text(encoding="utf-8"))["files_sha256"]
    for f in m["files"]:
        if f["path"] in info:
            assert f["sha256"] == info[f["path"]]
    # nenhum arquivo do outro tier
    other = "Small" if tier == "nano" else "Nano"
    assert not any(other in f["source"] for f in m["files"])


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
