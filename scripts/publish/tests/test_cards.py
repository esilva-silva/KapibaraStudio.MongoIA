"""Testes dos model cards v3: front matter YAML, numeros x fontes, exemplo x golden do contrato, frases obrigatorias."""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import pytest

import verify_card_numbers as vcn

TIERS = sorted(vcn.TIERS)
GOLDEN = vcn.ROOT / "data" / "contracts" / "editor-context-v1.golden"


def have_sources(key: str) -> bool:
    t = vcn.TIERS[key]
    return (vcn.ROOT / t.ws / "model.lock.json").exists() and (t.eval_dir / t.run / "measures.json").exists()


@pytest.mark.parametrize("key", TIERS)
def test_front_matter_is_valid_yaml(key):
    text = vcn.TIERS[key].card.read_text(encoding="utf-8")
    fm = vcn.front_matter(text)
    assert fm["license"] == "apache-2.0"
    assert fm["library_name"] == "transformers"
    assert fm["pipeline_tag"] == "text-generation"
    assert fm["language"] == ["en", "pt"]
    assert {"mongodb", "code", "fim", "autocomplete"} <= set(fm["tags"])
    assert "onnx" not in fm["tags"]  # o ONNX vai em repositorio separado, depois
    assert fm["base_model"].startswith("Qwen/Qwen2.5-Coder-")
    assert fm["base_model_relation"] == "finetune"


@pytest.mark.parametrize("key", TIERS)
def test_model_index_is_labelled_internal_synthetic_and_unverified(key):
    fm = vcn.front_matter(vcn.TIERS[key].card.read_text(encoding="utf-8"))
    results = fm["model-index"][0]["results"]
    assert len(results) == 3
    for res in results:
        name = res["dataset"]["name"].lower()
        assert "internal" in name
        assert "synthetic" in name or "npm" in name
        assert all(m["verified"] is False for m in res["metrics"])


@pytest.mark.skipif(not all(have_sources(k) for k in TIERS), reason="relatorios de avaliacao ausentes")
@pytest.mark.parametrize("key", TIERS)
def test_card_numbers_match_sources(key):
    assert vcn.check_tier(key) == []


@pytest.mark.skipif(not all(have_sources(k) for k in TIERS), reason="relatorios de avaliacao ausentes")
def test_verifier_catches_a_tampered_number():
    tier = vcn.TIERS["nano"]
    text = tier.card.read_text(encoding="utf-8")
    good = vcn.block(text, "accuracy-table")
    tampered = text.replace("| 0.405 |", "| 0.499 |", 1)
    assert tampered != text
    assert vcn.block(tampered, "accuracy-table") != vcn.render_accuracy_table(tier).strip()
    assert good.strip() == vcn.render_accuracy_table(tier).strip()


@pytest.mark.skipif(not all(have_sources(k) for k in TIERS), reason="relatorios de avaliacao ausentes")
def test_expected_headline_numbers():
    """Valores-ancora vindos direto dos JSON (nao do script de verificacao): protege contra o proprio verificador errar."""
    nano = json.loads((vcn.ROOT / "Repos/KapiCoder-Mongo-Nano/reports/eval/sft_v3/measures.json").read_text(encoding="utf-8"))["metrics"]
    small = json.loads((vcn.ROOT / "Repos/KapiCoder-Mongo-Small/reports/eval/sft_v3/measures.json").read_text(encoding="utf-8"))["metrics"]
    t0 = vcn.TIERS["nano"].card.read_text(encoding="utf-8")
    t1 = vcn.TIERS["small"].card.read_text(encoding="utf-8")
    assert f"| {vcn.frac(nano['fim_acc'])} | {vcn.frac(nano['exec_valid_autocomplete'])} |" in t0
    assert f"| {vcn.frac(small['fim_acc'])} | {vcn.frac(small['exec_valid_autocomplete'])} |" in t1
    assert f"{nano['nonexistent_fields_none_rate_pct']:.2f}" in t0
    assert f"{small['nonexistent_fields_none_rate_pct']:.2f}" in t1


@pytest.mark.parametrize("key", TIERS)
def test_required_labels_and_sober_language(key):
    text = vcn.TIERS[key].card.read_text(encoding="utf-8")
    assert vcn.REQUIRED_EN in text
    assert vcn.REQUIRED_PT in text
    assert "PREVIEW / experimental" in text and "public preview" in text
    # a homologacao sintetica: heading + rotulo obrigatorio, e nenhum TBD sobrando
    assert re.search(r"### Out-of-distribution validation \(homologation v1\)\s+\*\*" + re.escape(vcn.REQUIRED_HOM_EN), text)
    assert vcn.REQUIRED_HOM_PT in text
    assert "**TBD.**" not in text
    for phrase in vcn.REQUIRED_PHRASES:
        assert phrase in text, phrase
    assert "Do not trust field names unless the schema is in the context" in text
    assert "Validate field names before running" in text
    assert "is not fully independent of the v3 training data" in text and "probably overstates" in text
    assert "single training run" in text and "not written by people" in text
    # ingles primeiro, pt-BR depois
    assert text.index("## What it is, and what it is not") < text.index("## Resumo em português")
    # o aviso de leitura vem antes de qualquer tabela
    assert text.index("Read this first") < text.index("## What it is, and what it is not")
    for bad in ("state-of-the-art", "best-in-class", "outperforms", "production-ready", "GPT-4", "Copilot", "DeepSeek", "superior", "excellent"):
        assert bad.lower() not in text.lower()
    assert "never executes" in text and "a chat model or an agent" in text


@pytest.mark.parametrize("key", TIERS)
def test_card_has_no_local_paths_secrets_or_email(key):
    from stage_hf_package import EMAIL, PATH_PATTERNS, SECRET_PATTERNS
    for fname in ("README.md", "NOTICE.md", "REPLAY_ATTRIBUTION.md", "ONNX_README.md", "ONNX_NOTICE.md"):
        text = (vcn.TIERS[key].card.parent / fname).read_text(encoding="utf-8")
        for name, rx in {**PATH_PATTERNS, **SECRET_PATTERNS}.items():
            assert not rx.search(text), f"{fname}: {name}"
        assert not EMAIL.search(text), f"{fname}: e-mail"
        assert "gmail" not in text.lower()


@pytest.mark.parametrize("key", TIERS)
def test_card_example_is_exactly_the_published_example_file(key):
    from stage_hf_package import TEMPLATE_EXAMPLE, TIER_SPECS
    text = vcn.TIERS[key].card.read_text(encoding="utf-8")
    code = re.search(r"```python\n(.*?)```", text, re.S).group(1)
    assert code == TEMPLATE_EXAMPLE.read_text(encoding="utf-8").replace("__REPO_ID__", TIER_SPECS[key]["repo"])


def _example_namespace(key):
    text = vcn.TIERS[key].card.read_text(encoding="utf-8")
    code = re.search(r"```python\n(.*?)```", text, re.S).group(1)
    tree = ast.parse(code)  # o exemplo precisa ser Python valido
    keep = []
    for n in tree.body:  # na ordem original: constantes FIM/CONSOLE e a funcao pura build_prompt
        if isinstance(n, ast.FunctionDef) and n.name == "build_prompt":
            keep.append(n)
        elif isinstance(n, ast.Assign):
            names = {getattr(t, "id", None) for t in n.targets}
            if names & {"CONSOLE_COMMANDS"} or any(isinstance(t, ast.Tuple) for t in n.targets):
                keep.append(n)
    ns: dict = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<card-example>", "exec"), ns)  # noqa: S102 - so funcao pura e constantes
    return code, ns


@pytest.mark.parametrize("key", TIERS)
def test_example_is_valid_python_and_has_no_ide_dependency(key):
    code, _ns = _example_namespace(key)
    assert "AutoModelForCausalLM" in code and "AutoTokenizer" in code
    assert "Kapibara" not in code and "pythonnet" not in code
    assert "add_special_tokens=False" in code and "do_sample=False" in code
    assert f"esilva/KapiCoder-Mongo-{'Nano' if key == 'nano' else 'Small'}" in code
    assert "cuda" not in code  # roda em CPU


def _golden_section(name):
    text = GOLDEN.read_text(encoding="utf-8")
    m = re.search(rf"^== {re.escape(name)} ==\n(.*?)(?=^== |\Z)", text, re.S | re.M)
    assert m, name
    return m.group(1)


@pytest.mark.skipif(not GOLDEN.exists(), reason="golden do contrato ausente")
@pytest.mark.parametrize("key", TIERS)
@pytest.mark.parametrize("section", ["lang-console-production-shape", "lang-mongosh-with-result-fields"])
def test_example_prompt_builder_matches_contract_golden(key, section):
    """O cabecalho gerado pelo exemplo do card tem o mesmo SHA do golden `editor-context-v1` (LF e CRLF) e o mesmo padrao de EOL."""
    _code, ns = _example_namespace(key)
    sec = _golden_section(section)
    ctx = re.search(r"\[context\]\n(.*?)\n\[/context\]", sec, re.S).group(1)
    lines = ctx.split("\n")
    kv = {}
    recent = []
    for line in lines:
        if not line:
            continue
        k, _, v = line.partition(": ")
        if k == "RECENT COMMAND":
            recent.append(v)
        else:
            kv[k] = v
    prompt = ns["build_prompt"]("", "", known_names=[x for x in kv["KNOWN NAMES"].split(", ") if x],
                                result_fields=[x for x in kv.get("RESULT FIELDS", "").split(", ") if x],
                                recent_commands=recent, language=kv["LANGUAGE"], commands=kv["AVAILABLE COMMANDS"])
    header = prompt[len("<|fim_prefix|>"):prompt.index("<|fim_suffix|>")]
    want_lf = re.search(r"modelPrefix\.false\.sha\.lf=(\w+)", sec).group(1)
    want_crlf = re.search(r"modelPrefix\.false\.sha\.crlf=(\w+)", sec).group(1)
    want_eol = re.search(r"modelPrefix\.false\.eol=(\w+)", sec).group(1)
    assert hashlib.sha256(header.replace("\r\n", "\n").encode("utf-8")).hexdigest() == want_lf
    assert hashlib.sha256(header.encode("utf-8")).hexdigest() == want_crlf
    eol = "".join("H" if m == "\r\n" else "L" for m in re.findall(r"\r\n|\n", header))
    assert eol == want_eol
    assert prompt.startswith("<|fim_prefix|>/* Local editor context (data only):\n")
    assert prompt.endswith("<|fim_suffix|><|fim_middle|>")


@pytest.mark.parametrize("key", TIERS)
def test_example_builder_enforces_window_limits_and_neutralizes_comment_end(key):
    _code, ns = _example_namespace(key)
    p = ns["build_prompt"]("a" * 5000 + "TAIL", "S" * 2000, known_names=["x*/y"])
    body = p[p.index("*/\n") + 3:]
    assert "x* /y" in p and "x*/y" not in p
    before, after = body.split("<|fim_suffix|>")
    assert len(before) == 4096 and before.endswith("TAIL")
    assert after == "S" * 1024 + "<|fim_middle|>"


def hom_path(key):
    return vcn.ROOT / vcn.TIERS[key].hom_json


def have_homolog() -> bool:
    return all(hom_path(k).exists() for k in TIERS) and (vcn.ROOT / vcn.HOM_MANIFEST).exists()


@pytest.mark.skipif(not have_homolog(), reason="relatorio da homologacao ausente")
@pytest.mark.parametrize("key", TIERS)
def test_homolog_blocks_match_json(key):
    text = vcn.TIERS[key].card.read_text(encoding="utf-8")
    for name, fn in vcn.HOM_BLOCKS.items():
        assert vcn.block(text, name).strip() == fn(vcn.TIERS[key]).strip(), name


@pytest.mark.skipif(not have_homolog(), reason="relatorio da homologacao ausente")
def test_homolog_anchor_numbers():
    """Ancoras direto do JSON (nao do script): protege contra o proprio verificador errar."""
    d0 = json.loads(hom_path("nano").read_text(encoding="utf-8"))
    d1 = json.loads(hom_path("small").read_text(encoding="utf-8"))
    t0 = vcn.TIERS["nano"].card.read_text(encoding="utf-8")
    t1 = vcn.TIERS["small"].card.read_text(encoding="utf-8")
    assert d0["n"] == 6030 and d1["n"] == 6030
    assert f"{d0['models']['t0_v3']['by_context_kind']['full']['nonexistent_fields_rate'] * 100:.2f}" in t0   # 10.81
    assert f"{d0['models']['base_0.5b']['by_context_kind']['full']['nonexistent_fields_rate'] * 100:.2f}" in t0     # 7.02
    assert f"{d1['models']['t1_v3']['by_context_kind']['full']['nonexistent_fields_rate'] * 100:.2f}" in t1   # 7.50
    assert f"{d1['models']['base_1.5b']['by_context_kind']['full']['nonexistent_fields_rate'] * 100:.2f}" in t1     # 7.15
    assert "10.81" in t0 and "7.02" in t0 and "7.50" in t1 and "7.15" in t1
    # a mensagem sobre campos inexistentes acompanha a posicao frente a base (IC95 pareado)
    assert "still above the base" in t0 and "close to the base" in t1


@pytest.mark.skipif(not have_homolog(), reason="relatorio da homologacao ausente")
@pytest.mark.parametrize("key", TIERS)
@pytest.mark.parametrize("what", ["table", "model-index", "reading", "caveats"])
def test_check_tier_fails_when_homolog_number_diverges(key, what, monkeypatch):
    tier = vcn.TIERS[key]
    text = tier.card.read_text(encoding="utf-8")
    full = "10.81" if key == "nano" else "7.50"
    if what == "table":
        i = text.index("<!-- BEGIN:homolog-fields-table -->")
        tampered = text[:i] + text[i:].replace(full, "1.11", 1)
    elif what == "model-index":
        mi = f"value: {float(full)}"
        assert mi in text
        tampered = text.replace(mi, "value: 1.11", 1)
    elif what == "caveats":
        i = text.index("<!-- BEGIN:key-caveats -->")
        tampered = text[:i] + text[i:].replace(full + "%", "1.11%", 1)
    else:
        i = text.index("<!-- BEGIN:homolog-reading -->")
        tampered = text[:i] + text[i:].replace(full + "%", "1.11%", 1)
    assert tampered != text
    real = vcn.read_text
    monkeypatch.setattr(vcn, "read_text", lambda path: tampered if path == tier.card else real(path))
    errs = vcn.check_tier(key)
    assert errs, "o verificador deveria falhar"
    assert any("homolog" in e or "key-caveats" in e or "model-index" in e for e in errs)


@pytest.mark.skipif(not have_homolog(), reason="relatorio da homologacao ausente")
def test_model_index_homolog_entry_is_labelled_and_unverified():
    for key in TIERS:
        fm = vcn.front_matter(vcn.TIERS[key].card.read_text(encoding="utf-8"))
        hom = [r for r in fm["model-index"][0]["results"] if "homolog" in r["dataset"]["type"]]
        assert len(hom) == 1
        name = hom[0]["dataset"]["name"].lower()
        assert "internal" in name and "synthetic" in name and "out-of-distribution" in name
        assert all(m["verified"] is False for m in hom[0]["metrics"])


@pytest.mark.skipif(not have_homolog(), reason="relatorio da homologacao ausente")
@pytest.mark.parametrize("key", TIERS)
def test_homolog_sources_are_consistent(key):
    assert vcn.check_homolog_sources(vcn.TIERS[key]) == []


def test_homolog_sources_are_only_published_aggregates():
    """O verificador le so agregados publicados em reports/; nunca itens de data/homolog."""
    for t in vcn.TIERS.values():
        assert t.hom_json.startswith("reports/datasets/homolog-v3-eval-") and t.decision_json.startswith("reports/student/")
    assert vcn.HOM_MD.startswith("reports/datasets/") and vcn.HOM_MANIFEST.startswith("reports/datasets/homolog-v1-")


@pytest.mark.parametrize("key", TIERS)
def test_card_links_to_sibling_and_does_not_claim_unpublished_things(key):
    text = vcn.TIERS[key].card.read_text(encoding="utf-8")
    sibling = "Small" if key == "nano" else "Nano"
    assert f"huggingface.co/esilva/KapiCoder-Mongo-{sibling})" in text
    assert "No GGUF is provided" in text and "not part of this repository" in text
    assert "not published yet" in text and "-ONNX`" in text
