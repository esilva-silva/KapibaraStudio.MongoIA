#!/usr/bin/env python3
"""Paridade do renderer Python (`render_contract.py`) com (a) o golden congelado da IDE e (b) o renderer da v1.

  python scripts/data/check_contract_parity.py [--tokenizer] [--out data/contracts/parity_report.json]

(a) `data/contracts/editor-context-v1.golden` (copia de `tests/.../Ai/Golden/editor-context-v1.golden` da IDE): 20 casos x
    {context, modelPrefix.false, modelPrefix.true}; confere texto, origem dos terminadores (H = CRLF do contrato, L = `\\n`
    literal do ModelPrefix) e o SHA-256 da materializacao CRLF. Os casos replicam `EditorContextV1Corpus.cs`.
(b) `bakeoff/data/v1_benchmark.jsonl` + `v1_validation.jsonl` contra `bakeoff/v1/scripts/compact/contracts.py` (somente leitura):
    texto do `model_prefix` e, com --tokenizer, ids FIM Qwen (`QwenFim.build`) com o `tokenizer.json` do Qwen2.5-Coder.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_contract as rc  # noqa: E402

ROOT = rc.ROOT
GOLDEN = ROOT / "data" / "contracts" / "editor-context-v1.golden"
V1_CONTRACTS = ROOT / "bakeoff" / "v1" / "scripts" / "compact" / "contracts.py"
V1_DATA = [("v1_benchmark", ROOT / "bakeoff" / "data" / "v1_benchmark.jsonl"),
           ("v1_validation", ROOT / "bakeoff" / "data" / "v1_validation.jsonl")]


# -- golden ------------------------------------------------------------------------------------------------------------------------
def parse_golden(content: str) -> dict:
    cases: dict = {}
    lines = content.replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines):
        if not lines[i].startswith("== "):
            i += 1
            continue
        cid = lines[i][3:].rstrip(" =")
        i += 1
        art: dict = {}
        meta: dict = {}
        while i < len(lines) and not lines[i].startswith("== "):
            ln = lines[i]
            if ln.startswith("[") and not ln.startswith("[/"):
                name = ln[1:-1]
                i += 1
                body = []
                while i < len(lines) and lines[i] != f"[/{name}]":
                    body.append(lines[i])
                    i += 1
                i += 1
                eol = lines[i][len(name + ".eol="):]
                sha_lf = lines[i + 1][len(name + ".sha.lf="):]
                sha_crlf = lines[i + 2][len(name + ".sha.crlf="):]
                i += 3
                art[name] = {"text": "\n".join(body), "eol": eol, "sha_lf": sha_lf, "sha_crlf": sha_crlf}
            else:
                for key in ("prefix", "suffix", "dictionary"):
                    if ln.startswith(key + "="):
                        meta[key] = ln[len(key) + 1:]
                i += 1
        cases[cid] = {"artifacts": art, **meta}
    return cases


def escape_inline(raw: str) -> str:
    out = []
    for c in raw:
        if c == "\\":
            out.append("\\\\")
        elif c == "\r":
            out.append("\\r")
        elif c == "\n":
            out.append("\\n")
        elif 0xD800 <= ord(c) <= 0xDFFF:
            out.append("\\u%04x" % ord(c))
        else:
            out.append(c)
    return "".join(out)


def decompose(raw: str) -> tuple[str, str]:
    text, eol = [], []
    i = 0
    while i < len(raw):
        c = raw[i]
        if c == "\r" and i + 1 < len(raw) and raw[i + 1] == "\n":
            text.append("\n"); eol.append("H"); i += 2; continue
        if c == "\n":
            text.append("\n"); eol.append("L")
        elif c == "\\":
            text.append("\\\\")
        elif c == "\r":
            text.append("\\r")
        elif 0xD800 <= ord(c) <= 0xDFFF:
            text.append("\\u%04x" % ord(c))
        else:
            text.append(c)
        i += 1
    return "".join(text), "".join(eol)


def recompose(text: str, eol: str, host: str) -> str:
    raw, k, i = [], 0, 0
    while i < len(text):
        c = text[i]
        if c == "\\" and i + 1 < len(text):
            n = text[i + 1]
            if n == "\\":
                raw.append("\\"); i += 2; continue
            if n == "r":
                raw.append("\r"); i += 2; continue
            if n == "u" and i + 5 < len(text):
                raw.append(chr(int(text[i + 2:i + 6], 16))); i += 6; continue
        if c == "\n":
            raw.append(host if k < len(eol) and eol[k] == "H" else "\n"); k += 1
        else:
            raw.append(c)
        i += 1
    return "".join(raw)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


LONG_TAIL = "x" * 90000 + "db.Customers.find({"
CONSOLE = "Mongo Console JavaScript"
MONGOSH = "JavaScript (mongosh)"
# id -> kwargs de render_contract.build_request  (espelha EditorContextV1Corpus.cs)
CASES: dict[str, dict] = {
    "lang-json-baseline": dict(text="", caret=0, language="json"),
    "lang-mongosh-with-result-fields": dict(text="db.customers.find({na", caret=22, language=MONGOSH, result_fields=["Name", "AccountId"]),
    "lang-console-production-shape": dict(text="db.Customers.find({})", caret=22, language=CONSOLE, known_names=["Production", "Customers", "Dev"],
                                          recent_commands=["db.Customers.findOne()", "db.Customers.count()", "db.Customers.drop()"]),
    "lang-null-defaults-to-shared-commands": dict(text="", caret=0, language=None),
    "names-overflow-takes-first-128": dict(text="", caret=0, language="json", known_names=["Field%03d" % i for i in range(1, 130)]),
    "names-filters-sensitive-and-reserved-tokens": dict(text="", caret=0, language="json",
                                                        known_names=["Safe1", "password: hidden", "Weird<|Field", "Weird<｜Field", "Safe2"]),
    "result-panel-disabled-hides-fields": dict(text="", caret=0, language="json", result_fields=["Name", "AccountId"], use_result_panel=False),
    "input-panel-disabled-hides-safe-input": dict(text="", caret=0, language="json", input_panel='{"Name":"Eduardo"}', use_input_panel=False),
    "input-overlong-cuts-at-1024": dict(text="", caret=0, language="json", input_panel="i" * 2000),
    "input-sensitive-excluded": dict(text="db.", caret=3, language="javascript", input_panel='{"password":"hidden"}', result_fields=[],
                                     known_names=["Production"], recent_commands=["const api_key = 'hidden'"]),
    "recent-five-truncates-to-three-and-cuts-256": dict(text="", caret=0, language="json",
                                                        recent_commands=["r" * 300, "c2", "c3", "c4", "c5"]),
    "predictive-bounded-enabled": dict(text=LONG_TAIL, caret=len(LONG_TAIL), language="javascript", input_panel='{"Name":"Eduardo"}',
                                       result_fields=["Name", "AccountId", "Active"], known_names=["Production", "Customers"],
                                       recent_commands=["db.Customers.findOne()"]),
    "predictive-bounded-disabled": dict(text=LONG_TAIL, caret=len(LONG_TAIL), language="javascript", input_panel='{"Name":"Eduardo"}',
                                        result_fields=["Name", "AccountId", "Active"], known_names=["Production", "Customers"],
                                        recent_commands=["db.Customers.findOne()"], use_input_panel=False, use_result_panel=False,
                                        use_editor_context=False),
    "window-caret-zero": dict(text="return value;", caret=0, language="json"),
    "window-mixed-eol-preserves-raw-bytes": dict(text="line1\r\nline2\nline3\r\nCURSOR", caret=len("line1\r\nline2\nline3\r\n"), language="json"),
    "window-surrogate-pair-split-by-char-index": dict(text="abc😀def", caret=4, language="json"),
    "context-exceeds-8192-gets-bounded": dict(text="", caret=0, language="json",
                                              known_names=["Name_%04d" % i + "a" * 70 for i in range(128)]),
    "modelprefix-neutralizes-star-slash": dict(text="", caret=0, language=MONGOSH, input_panel="check */ neutralized but safe value"),
    # estes dois constroem o AutocompleteRequest direto (contexto cru)
    "modelprefix-raw-commands-last-line-has-no-terminator": dict(raw_context="AVAILABLE COMMANDS: " + rc.COMMANDS_CONSOLE, prefix="db."),
    "modelprefix-raw-blank-context-short-circuits": dict(raw_context="   ", prefix="db."),
}


def live_artifacts(case: dict) -> tuple[dict, str, str, list[str]]:
    if "raw_context" in case:
        req = {"prefix": case["prefix"], "suffix": "", "context": case["raw_context"], "dictionary": []}
    else:
        req = rc.build_request(**case)
    return ({"context": req["context"],
             "modelPrefix.false": rc.model_prefix(req["prefix"], req["context"]),
             "modelPrefix.true": rc.model_prefix(req["prefix"], req["context"], deepseek_training_contract=True)},
            req["prefix"], req["suffix"], req["dictionary"])


def check_golden(path: Path = GOLDEN) -> dict:
    golden = parse_golden(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    checks = 0
    if set(golden) != set(CASES):
        failures.append(f"casos divergem: {sorted(set(golden) ^ set(CASES))}")
    for cid, case in CASES.items():
        g = golden.get(cid)
        if g is None:
            continue
        live, prefix, suffix, dictionary = live_artifacts(case)
        for name, raw in live.items():
            exp = g["artifacts"][name]
            head = raw if name == "context" else raw[: len(raw) - len(prefix)]
            if name != "context" and raw[len(raw) - len(prefix):] != prefix:
                failures.append(f"{cid}/{name}: cauda != prefixo")
            text, eol = decompose(head)
            ok = (text == exp["text"] and eol == exp["eol"] and _sha(head) == exp["sha_crlf"]
                  and _sha(recompose(exp["text"], exp["eol"], "\n")) == exp["sha_lf"])
            checks += 1
            if not ok:
                failures.append(f"{cid}/{name}")
        if "raw_context" not in case:
            for key, val in (("prefix", prefix), ("suffix", suffix), ("dictionary", "".join(dictionary))):
                checks += 1
                if escape_inline(val) != g[key]:
                    failures.append(f"{cid}/{key}")
    return {"cases": len(golden), "artifact_checks": checks, "failures": failures}


# -- v1 ----------------------------------------------------------------------------------------------------------------------------
def load_v1_contracts():
    spec = importlib.util.spec_from_file_location("v1_compact_contracts", V1_CONTRACTS)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v1_compact_contracts"] = mod
    spec.loader.exec_module(mod)
    return mod


class _TokShim:
    """Interface minima de transformers usada por compact.contracts.QwenFim, sobre `tokenizers`."""

    def __init__(self, tok: rc.Tokenizer):
        self._t = tok._tok

    def get_vocab(self):
        return self._t.get_vocab()

    def convert_tokens_to_ids(self, t):
        return self._t.token_to_id(t)

    def encode(self, text, add_special_tokens=False):
        return self._t.encode(text, add_special_tokens=add_special_tokens).ids


def check_v1(with_tokens: bool, limit: int | None = None) -> dict:
    v1 = load_v1_contracts()
    tok = shim = qwen = None
    if with_tokens:
        tok = rc.Tokenizer()
        shim = _TokShim(tok)
        qwen = v1.QwenFim(shim)
    out: dict = {}
    for name, path in V1_DATA:
        rows = [x for x in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip()) if x.get("type") in ("fim", "completion")]
        if limit:
            rows = rows[:limit]
        r = {"records": len(rows), "with_context": 0, "same_fixed_crlf": 0, "same_honor_newline": 0, "newline_lf_records": 0,
             "no_context_same": 0}
        if with_tokens:
            r["same_ids"] = 0
        for ex in rows:
            ctx = ex.get("context")
            prefix, suffix = ex["prefix"][-v1.PREFIX_WINDOW:], ex["suffix"][: v1.SUFFIX_WINDOW]
            v1_text = v1.model_prefix(prefix, ctx)
            if ctx:
                r["with_context"] += 1
                language = ctx.get("language") or rc.LANGUAGE_CONSOLE
                header = rc.build_context(language, ctx.get("known_names", []), ctx.get("result_fields", []), ctx.get("input_panel", ""),
                                          ctx.get("recent_commands", []))
                mine = rc.model_prefix(prefix, header)
                legacy = rc.model_prefix(prefix, rc.build_context(language, ctx.get("known_names", []), ctx.get("result_fields", []),
                                                                 ctx.get("input_panel", ""), ctx.get("recent_commands", []),
                                                                 _legacy_newline=ctx.get("newline", "\r\n")))
                r["newline_lf_records"] += ctx.get("newline", "\r\n") != "\r\n"
                r["same_fixed_crlf"] += mine == v1_text
                r["same_honor_newline"] += legacy == v1_text
                if with_tokens:
                    ids_v1 = qwen.build(v1_text, suffix, 2048)
                    ids_mine = rc.build_ids(tok, legacy, suffix)[0]
                    r["same_ids"] += ids_v1 == ids_mine
            else:
                r["no_context_same"] += (v1_text == prefix and (not with_tokens or qwen.build(v1_text, suffix, 2048)
                                                                == rc.build_ids(tok, prefix, suffix)[0]))
        r["no_context_records"] = r["records"] - r["with_context"]
        out[name] = r
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokenizer", action="store_true", help="compara tambem ids de token (requer `tokenizers`)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default=str(ROOT / "data" / "contracts" / "parity_report.json"))
    a = ap.parse_args(argv)
    rep = {"golden": check_golden(), "v1": check_v1(a.tokenizer, a.limit)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0 if not rep["golden"]["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
