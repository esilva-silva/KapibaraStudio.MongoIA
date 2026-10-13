#!/usr/bin/env python3
"""Cortes FIM por limite sintatico para JavaScript generico (replay / Generic-JS FIM do professor, fase-02 §3.2/§3.6 G2.5).

Reusa a logica de corte de `gen_fim.py` (`candidates` + `pick_end`: spans key/value/member/chain/obj/arr/stmt -> cut_kind
key|value|member|partial|rest|multiline|stmt_start, fim de middle em fronteira natural, middle <= `max_tokens` com o tokenizer Qwen).
O que muda: os spans nao vem do renderizador de snippets Mongo e sim de um lexer JS (`compact.jsparse.tokenize`, o mesmo do projeto).
Nenhum conteudo do projeto e usado; o texto vem do arquivo de entrada.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bakeoff" / "v1" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compact.jsparse import KEYWORDS, JsSyntaxError, tokenize  # noqa: E402
import gen_fim  # noqa: E402

OPEN = {"(": ")", "[": "]", "{": "}"}
VALUE_PREV = {"=", ":", "(", ",", "return", "=>", "?", "||", "&&", "??", "+", "-", "*", "/", "==", "===", "!=", "!==", "<", ">", "[", "+=", "-="}
OBJ_PREV = {"=", ":", "(", ",", "[", "?", "||", "&&", "??", "return", "+", "-", ":", "...", "!", "==", "===", "!=", "!=="}
CONT_AFTER = {"(", ",", "[", "{", ".", "?.", "=>", "=", "+", "-", "*", "/", "%", "&&", "||", "??", "?", ":", "==", "===", "!=", "!==", "<", ">",
              "<=", ">=", "+=", "-=", "&", "|", "^", "in", "of", "instanceof", "typeof", "new", "return", "!", "...", "const", "let", "var",
              "function", "async", "await", "class", "extends", "export", "import", "default", "else", "do", "case"}
CONT_BEFORE = {")", "]", "}", ".", "?.", ",", ";", "else", "catch", "finally", "=", "+", "*", "/", "&&", "||", "??", "?", ":", "==", "===",
               "!=", "!==", "<", ">", "<=", ">=", "=>", "&", "|", "in", "of", "instanceof", "while", "(", "["}
# cut_kind -> peso (distribuicao de uso; "operator"/"stage" sao do Mongo e nao existem aqui)
KIND_WEIGHTS = {"member": 0.18, "partial": 0.10, "value": 0.18, "key": 0.10, "rest": 0.20, "multiline": 0.10, "stmt_start": 0.14}


class Counter96:
    """Contador de tokens Qwen (interface `.count`) para `pick_end`."""

    def __init__(self, tokenizer_json: str):
        try:
            from tokenizers import Tokenizer
        except ImportError:  # env/.venv nao tem `tokenizers`: usa o site-packages do venv ROCm (como gen_fim.MiddleTokens)
            site = gen_fim.DEFAULT_TOK_SITE
            if site not in sys.path:
                sys.path.append(site)
            from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(tokenizer_json)

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False).ids) if text else 0


def js_spans(text: str) -> list[tuple[int, int, str, int]]:
    """Spans (start, end, role, unit_end) com os papeis de gen_fim; levanta JsSyntaxError se o lexer nao entender o arquivo."""
    toks = tokenize(text)[:-1]
    n = len(toks)
    match: dict[int, int] = {}
    stack: list[int] = []
    for i, t in enumerate(toks):
        if t.kind == "punct" and t.value in OPEN:
            stack.append(i)
        elif t.kind == "punct" and t.value in (")", "]", "}") and stack:
            match[stack.pop()] = i
    spans: list[tuple[int, int, str, int]] = []

    def expr_end(i: int) -> int:
        """fim (char) da expressao que comeca no token i: ate `,`/`;`/fechamento/quebra de linha no mesmo nivel."""
        depth, j, last = 0, i, toks[i].end
        while j < n and j - i < 400:
            t = toks[j]
            if j > i and depth == 0 and t.nl_before and toks[j - 1].value not in CONT_AFTER and t.value not in CONT_BEFORE:
                break
            if t.kind == "punct":
                if t.value in OPEN:
                    if j in match:
                        last = toks[match[j]].end
                        j = match[j] + 1
                        continue
                if t.value in (",", ";", ")", "]", "}") and depth == 0:
                    break
            last = t.end
            j += 1
        return last

    def block_like(i: int) -> bool:  # `{` em i abre bloco (e nao literal de objeto)?
        if i == 0:
            return True
        p = toks[i - 1]
        return not (p.kind == "punct" and p.value in OBJ_PREV)

    def statements(lo: int, hi: int) -> None:
        i = lo
        while i < hi:
            if toks[i].kind == "punct" and toks[i].value in (";", "}"):
                i += 1
                continue
            start, j, end_char = i, i, toks[i].end
            while j < hi:
                t = toks[j]
                if j > start and t.nl_before and toks[j - 1].value not in CONT_AFTER and t.value not in CONT_BEFORE:
                    break
                if t.kind == "punct" and t.value in OPEN and j in match and match[j] < hi:
                    close = match[j]
                    is_block = t.value == "{" and block_like(j)
                    if is_block:
                        statements(j + 1, close)
                    end_char = toks[close].end
                    j = close + 1
                    if is_block and (j >= hi or (toks[j].nl_before and toks[j].value not in CONT_BEFORE)):
                        break
                    continue
                end_char = t.end
                j += 1
                if t.kind == "punct" and t.value == ";":
                    break
            if end_char > toks[start].start:
                spans.append((toks[start].start, end_char, "stmt", end_char))
            i = max(j, start + 1)

    statements(0, n)
    for i, t in enumerate(toks):
        prev = toks[i - 1] if i else None
        nxt = toks[i + 1] if i + 1 < n else None
        if t.kind == "punct" and t.value in ("{", "[") and i in match:
            close = toks[match[i]]
            if "\n" in text[t.start:close.end]:
                spans.append((t.start, close.end, "obj" if t.value == "{" else "arr", close.end))
        if t.kind == "ident" and t.value not in KEYWORDS:
            if prev is not None and prev.kind == "punct" and prev.value in (".", "?."):
                u = t.end
                if nxt is not None and nxt.kind == "punct" and nxt.value == "(" and (i + 1) in match:
                    u = toks[match[i + 1]].end
                spans.append((t.start, t.end, "member", u))
            elif nxt is not None and nxt.kind == "punct" and nxt.value == ":" and prev is not None and prev.value in ("{", ","):
                spans.append((t.start, t.end, "key", max(t.end, expr_end(i + 2) if i + 2 < n else t.end)))
        if t.kind == "ident" and t.value not in KEYWORDS and (prev is None or not (prev.kind == "punct" and prev.value in (".", "?."))):
            # cadeia `a.b.c(` : do primeiro identificador ate o ultimo antes de `(`
            j = i
            while j + 2 < n and toks[j + 1].value == "." and toks[j + 2].kind == "ident":
                j += 2
            if j > i and j + 1 < n and toks[j + 1].value == "(":
                spans.append((t.start, toks[j].end, "chain", toks[j].end))
        if (t.kind in ("ident", "num", "str", "template", "regex") and prev is not None and (prev.value in VALUE_PREV)
                and prev.kind in ("punct", "ident") and (t.kind != "ident" or t.value not in KEYWORDS or t.value in ("null", "true", "false", "this"))):
            spans.append((t.start, t.end, "value", max(t.end, expr_end(i))))
    return spans


def make_cut(text: str, rng: random.Random, tc, max_tokens: int = 96, tries: int = 24, min_prefix_chars: int = 40,
             kind_weights: dict[str, float] | None = None) -> dict | None:
    """Um corte FIM {prefix, middle, suffix, cut_kind, cursor_role} ou None. Deterministico dado `rng`."""
    try:
        spans = js_spans(text)
    except (JsSyntaxError, RecursionError, IndexError):
        return None
    rec = {"text": text, "spans": spans}
    cands = gen_fim.candidates(rec)
    weights = kind_weights or KIND_WEIGHTS
    cfg = {"fim": {"max_tokens": max_tokens}}
    kinds = [k for k in weights if cands.get(k)]
    for _ in range(tries):
        if not kinds:
            return None
        kind = rng.choices(kinds, [weights[k] for k in kinds])[0]
        c = rng.choice(cands[kind])
        if c["start"] < min_prefix_chars or len(text) - c["start"] < min_prefix_chars:
            continue
        e = gen_fim.pick_end(rec, kind, c, cfg, rng, tc)
        if e is None:
            continue
        middle = text[c["start"]:e]
        if not any(ch.isalnum() for ch in middle):
            continue
        return {"prefix": text[:c["start"]], "middle": middle, "suffix": text[e:], "cut_kind": kind,
                "cursor_role": gen_fim.ROLE_TO_CURSOR.get(c["role"], "Unknown"), "mid_token": bool(c.get("mid_token")),
                "multiline": "\n" in middle}
    return None
