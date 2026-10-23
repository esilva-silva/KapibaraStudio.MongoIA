#!/usr/bin/env python3
"""Scanner JS minimo para consultas Mongo escritas por LLM (sem AST): mascara strings/comentarios e extrai os *spans* sintaticos que o
`gen_fim.candidates` espera (key, value, op, fieldref, stage, member, method, strarg, chain, obj, arr, stmt).

`scan(text)` -> {"tokens": [...], "spans": [(s, e, role, unit_end)], "masked": str}
Spans seguem a convencao do gerador de gramatica: chave/operador = texto INTERNO (sem aspas); valor/estagio/obj/arr = expressao inteira.
Heuristico (nao e um parser de JS completo), mas robusto a: aspas simples/duplas, template string, regex literal, comentarios, CRLF,
chaves sem aspas, variaveis, funcoes e blocos (uma chave `{` apos `)`/`=>`/`else` e bloco, nao objeto).
"""
from __future__ import annotations

import re

STAGES = {"$match", "$group", "$project", "$sort", "$limit", "$skip", "$unwind", "$lookup", "$addFields", "$set", "$unset",
          "$count", "$sortByCount", "$facet", "$bucket", "$bucketAuto", "$replaceRoot", "$replaceWith", "$sample", "$redact",
          "$densify", "$fill", "$setWindowFields", "$unionWith", "$graphLookup", "$geoNear", "$collStats", "$indexStats"}
KEYWORDS_BEFORE_EXPR = {"return", "typeof", "in", "of", "case", "new", "await", "yield", "void", "delete", "throw"}
_ID_START = re.compile(r"[A-Za-z_$À-￿]")
_ID_PART = re.compile(r"[A-Za-z0-9_$À-￿]")
PUNCT3 = ("===", "!==", "...", "**=", "<<=", ">>=", "??=", "||=", "&&=")
PUNCT2 = ("=>", "==", "!=", "<=", ">=", "&&", "||", "??", "?.", "++", "--", "+=", "-=", "*=", "/=", "%=", "**", "<<", ">>")


def tokenize(text: str):
    """[(tipo, ini, fim)] com tipo em str|tpl|num|id|punct|regex; comentarios ficam em `comments` [(ini, fim)]."""
    toks, comments = [], []
    i, n = 0, len(text)

    def prev_sig():
        return toks[-1] if toks else None

    while i < n:
        c = text[i]
        if c in " \t\r\n\f\v ﻿":
            i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            comments.append((i, j))
            i = j
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            comments.append((i, j))
            i = j
            continue
        if c in "\"'":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
                if j < n and text[j] == "\n" and text[j - 1] != "\\":
                    break
            j = min(n, j + 1)
            toks.append(("str", i, j))
            i = j
            continue
        if c == "`":
            j, depth = i + 1, 0
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == "$" and j + 1 < n and text[j + 1] == "{":
                    depth += 1
                    j += 2
                    continue
                if text[j] == "}" and depth:
                    depth -= 1
                elif text[j] == "`" and not depth:
                    j += 1
                    break
                j += 1
            toks.append(("tpl", i, min(j, n)))
            i = min(j, n)
            continue
        if c.isdigit() or (c == "." and i + 1 < n and text[i + 1].isdigit()):
            m = re.compile(r"0[xX][0-9a-fA-F]+n?|\d[\d_]*\.?\d*(?:[eE][+-]?\d+)?n?|\.\d+(?:[eE][+-]?\d+)?").match(text, i)
            j = m.end() if m else i + 1
            toks.append(("num", i, j))
            i = j
            continue
        if _ID_START.match(c):
            j = i + 1
            while j < n and _ID_PART.match(text[j]):
                j += 1
            toks.append(("id", i, j))
            i = j
            continue
        if c == "/":
            p = prev_sig()
            is_re = p is None or (p[0] == "punct" and text[p[1]:p[2]] in ("(", ",", ":", "=", "[", "!", "&", "|", "?", "{", "}", ";", "=>", "&&", "||", "??", "==", "===", "!=", "!==")) \
                or (p[0] == "id" and text[p[1]:p[2]] in KEYWORDS_BEFORE_EXPR)
            if is_re:
                j, in_cls = i + 1, False
                while j < n:
                    ch = text[j]
                    if ch == "\\":
                        j += 2
                        continue
                    if ch == "[":
                        in_cls = True
                    elif ch == "]":
                        in_cls = False
                    elif ch == "/" and not in_cls:
                        break
                    elif ch == "\n":
                        break
                    j += 1
                j = min(n, j + 1)
                while j < n and _ID_PART.match(text[j]):
                    j += 1
                toks.append(("regex", i, j))
                i = j
                continue
        for w in (PUNCT3, PUNCT2):
            k = len(w[0])
            if text[i:i + k] in w:
                toks.append(("punct", i, i + k))
                i += k
                break
        else:
            toks.append(("punct", i, i + 1))
            i += 1
    return toks, comments


def mask(text: str, keep_strings: bool = False) -> str:
    """Substitui comentarios (e strings, salvo keep_strings) por espacos, preservando posicoes e quebras de linha."""
    toks, comments = tokenize(text)
    chars = list(text)

    def blank(a, b):
        for k in range(a, b):
            if chars[k] not in "\r\n":
                chars[k] = " "
    for a, b in comments:
        blank(a, b)
    for t, a, b in toks:
        if t == "regex":
            blank(a, b)
    if not keep_strings:
        for t, a, b in toks:
            if t in ("str", "tpl"):
                blank(a + 1, b - 1)
    return "".join(chars)


def string_value(text: str, a: int, b: int) -> str:
    return text[a + 1:b - 1] if b - a >= 2 else ""


def scan(text: str) -> dict:
    toks, comments = tokenize(text)
    T = lambda k: text[toks[k][1]:toks[k][2]]  # noqa: E731
    n = len(toks)
    spans: list[tuple[int, int, str, int]] = []
    match: dict[int, int] = {}
    stack: list[int] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for k, (t, a, b) in enumerate(toks):
        if t != "punct":
            continue
        s = text[a:b]
        if s in "([{":
            stack.append(k)
        elif s in pairs and stack:
            while stack and T(stack[-1]) != pairs[s]:
                stack.pop()
            if stack:
                match[stack.pop()] = k
    # tipo de cada abertura
    kind: dict[int, str] = {}
    for k in match:
        s = T(k)
        pv = toks[k - 1] if k else None
        ps = text[pv[1]:pv[2]] if pv else ""
        if s == "(":
            kind[k] = "call" if pv and (pv[0] == "id" or ps in (")", "]")) and ps not in KEYWORDS_BEFORE_EXPR else "paren"
        elif s == "[":
            kind[k] = "idx" if pv and (pv[0] in ("id", "str") or ps in (")", "]")) and ps not in KEYWORDS_BEFORE_EXPR else "arr"
        else:
            if pv is None or ps in (")", "=>", ";", "{", "}", "else", "try", "finally", "do") and ps != "":
                kind[k] = "block"
            elif pv is None:
                kind[k] = "block"
            else:
                kind[k] = "obj"
    # um `{` apos `)` ou `=>`: bloco; um `{` apos `}`: bloco (statement seguinte) -- mas `}` de objeto seguido de `{`? raro
    end_of = lambda k: toks[match[k]][2]  # noqa: E731

    def expr_end(k: int, close_k: int) -> int:
        """indice do ultimo token da expressao que comeca em k, parando em `,`/fecho no mesmo nivel (k..close_k)."""
        j = k
        last = k
        while j < close_k:
            tj = T(j)
            if toks[j][0] == "punct" and tj in "([{" and j in match:
                last = match[j]
                j = match[j] + 1
                continue
            if toks[j][0] == "punct" and tj == ",":
                break
            last = j
            j += 1
        return last

    def simple_value(k: int, last: int) -> bool:
        """valor escalar (literal, identificador, regex, chamada/new de construtor) -- nao objeto/array/bloco."""
        t0 = toks[k][0]
        first = T(k)
        if first in ("{", "["):
            return False
        for j in range(k, last + 1):
            if toks[j][0] == "punct" and T(j) in "{[" and kind.get(j) in ("obj", "arr", "block"):
                return False
        return True

    # ---- objetos: chaves e valores ---------------------------------------------------------------------------
    for ok_, ck in match.items():
        if T(ok_) == "{" and kind[ok_] == "obj":
            j = ok_ + 1
            while j < ck:
                # inicio de propriedade em j
                tj, a, b = toks[j]
                if T(j) == "[" and j in match:           # chave computada
                    j = match[j] + 1
                    nxt = j
                elif tj in ("id", "str", "num"):
                    nxt = j + 1
                else:
                    j += 1
                    continue
                if nxt < ck and T(nxt) == ":":
                    key = text[a + 1:b - 1] if tj == "str" else text[a:b]
                    ks, ke = (a + 1, b - 1) if tj == "str" else (a, b)
                    vstart = nxt + 1
                    vlast = expr_end(vstart, ck) if vstart < ck else vstart
                    unit = toks[vlast][2] if vstart < ck else toks[nxt][2]
                    role = "op" if key.startswith("$") else "key"
                    if ke > ks:
                        spans.append((ks, ke, role, max(unit, ke)))
                    if vstart < ck:
                        va = toks[vstart][1]
                        vb = toks[vlast][2]
                        v0 = toks[vstart]
                        if v0[0] == "str" and vstart == vlast:
                            sv = text[v0[1] + 1:v0[2] - 1]
                            if sv.startswith("$") and not sv.startswith("$$") and len(sv) > 1:
                                spans.append((v0[1] + 1, v0[2] - 1, "fieldref", vb))
                            elif sv:
                                spans.append((va, vb, "value", vb))
                        elif simple_value(vstart, vlast):
                            spans.append((va, vb, "value", vb))
                    j = vlast + 1 if vstart < ck else nxt + 1
                    if j < ck and T(j) == ",":
                        j += 1
                else:
                    # propriedade abreviada ou outra construcao: pula ate a proxima virgula do nivel
                    last = expr_end(j, ck)
                    j = last + 1
                    if j < ck and T(j) == ",":
                        j += 1
        # multiline obj / arr
        if T(ok_) in "{[" and kind[ok_] in ("obj", "arr"):
            a, b = toks[ok_][1], toks[ck][2]
            if "\n" in text[a:b]:
                spans.append((a, b, "obj" if T(ok_) == "{" else "arr", b))
    # ---- arrays: elementos de pipeline (estagios), strings soltas em arrays -------------------------------------
    for ok_, ck in match.items():
        if T(ok_) == "[" and kind[ok_] == "arr":
            j = ok_ + 1
            elems = []
            while j < ck:
                last = expr_end(j, ck)
                elems.append((j, last))
                j = last + 1
                if j < ck and T(j) == ",":
                    j += 1
            def first_key(e):
                k0, l0 = e
                if T(k0) == "{" and k0 in match and k0 + 1 < match[k0]:
                    tk = toks[k0 + 1]
                    return text[tk[1] + 1:tk[2] - 1] if tk[0] == "str" else text[tk[1]:tk[2]]
                return None
            is_pipe = any(first_key(e) in STAGES for e in elems)
            for e in elems:
                k0, l0 = e
                a, b = toks[k0][1], toks[l0][2]
                if is_pipe and T(k0) == "{" and first_key(e) in STAGES:
                    spans.append((a, b, "stage", b))
                elif toks[k0][0] == "str" and k0 == l0:
                    sv = text[a + 1:b - 1]
                    if sv.startswith("$") and not sv.startswith("$$") and len(sv) > 1:
                        spans.append((a + 1, b - 1, "fieldref", b))
                    elif sv:
                        spans.append((a, b, "value", b))
                elif simple_value(k0, l0) and toks[k0][0] != "punct":
                    spans.append((a, b, "value", b))
    # ---- cadeias de metodo, membros, argumentos de string --------------------------------------------------------
    for k in range(1, n):
        if T(k) == "." and toks[k][0] == "punct" and k + 1 < n and toks[k + 1][0] == "id":
            nm = toks[k + 1]
            name = text[nm[1]:nm[2]]
            prevk = toks[k - 1]
            is_call = k + 2 < n and T(k + 2) == "(" and (k + 2) in match
            if is_call:
                spans.append((nm[1], nm[2], "method", nm[2]))
                ce = toks[match[k + 2]][2]
                spans.append((toks[k][1], ce, "chain", ce))
                # primeiro argumento string (getCollection("x"), distinct("campo"), $-strings ja cobertas)
                if k + 3 < match[k + 2] and toks[k + 3][0] == "str" and (k + 4 == match[k + 2] or T(k + 4) == ","):
                    sa, sb = toks[k + 3][1], toks[k + 3][2]
                    if sb - sa > 2:
                        spans.append((sa + 1, sb - 1, "strarg", sb))
            else:
                # db.colecao / a.b.c: membro
                spans.append((nm[1], nm[2], "member", nm[2]))
    # getCollection / getSiblingDB com argumento string ja entram em strarg via a regra acima
    # ---- statements --------------------------------------------------------------------------------------------------
    CONT_END = {",", ".", "=", "(", "[", "{", "?", ":", "&&", "||", "+", "-", "*", "=>", "??", "?."}
    CONT_NEXT = {".", ",", "?", ":", "&&", "||", "+", "-", "*", "/", "=", ")", "]", "}", "?.", "??", "else", "catch", "finally"}
    ranges = [(-1, n)] + [(k, match[k]) for k in match if kind[k] == "block"]
    for (ok_, ck) in ranges:
        j = ok_ + 1
        while j < ck:
            sstart, last, semi = j, j, False
            while j < ck:
                tj = T(j)
                if toks[j][0] == "punct" and tj in "([{" and j in match:
                    opened = j
                    last = match[j]
                    j = last + 1
                    if (tj == "{" and kind.get(opened) == "block" and j < ck
                            and "\n" in text[toks[last][2]:toks[j][1]] and T(j) not in CONT_NEXT):
                        break
                    continue
                if tj == ";" and toks[j][0] == "punct":
                    semi = True
                    break
                last = j
                if j + 1 < ck and "\n" in text[toks[j][2]:toks[j + 1][1]]:
                    if tj not in CONT_END and T(j + 1) not in CONT_NEXT:
                        j += 1
                        break
                j += 1
            e, a = toks[last][2], toks[sstart][1]
            if e > a and not (last == sstart and T(sstart) == ";"):
                spans.append((a, e, "stmt", e))
            if semi:
                j += 1
    return {"tokens": toks, "spans": sorted(spans, key=lambda s: (s[0], s[1], s[2])), "masked": mask(text), "comments": comments}


def dedupe_spans(spans):
    seen, out = set(), []
    for s in spans:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out
