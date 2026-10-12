"""Porta fiel de EsilvaSoft.SlopStudio.Application.IncrementalCompletion.NextLength.

Mantida idêntica ao C# para que a métrica de aceitação por TAB represente a IDE real.
"""
from __future__ import annotations


def _word(c: str) -> bool:
    return c.isalnum() or c in "_$"


def next_length(text: str) -> int:
    if not text:
        return 0
    i = 0
    n = len(text)
    while i < n and text[i] in " \t":
        i += 1
    if i < n and _word(text[i]):
        while i < n and _word(text[i]):
            i += 1
        if i < n and text[i] == ".":
            nxt = i + 1
            while nxt < n and _word(text[nxt]):
                nxt += 1
            if nxt == n or text[nxt] != "(":
                i += 1
        return i
    quote = ""
    while i < n:
        c = text[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = ""
            i += 1
            continue
        if c in "'\"`":
            quote = c
            i += 1
            continue
        if c == "\r" and i + 1 < n and text[i + 1] == "\n":
            return i + 2
        if c in "\n;,{}" or i >= 79:
            return i + 1
        i += 1
    return n


def chunks(text: str) -> list[str]:
    """Divide uma sugestão nos blocos que TABs sucessivos aceitariam."""
    result: list[str] = []
    while text:
        size = next_length(text)
        if size <= 0:
            break
        result.append(text[:size])
        text = text[size:]
    return result
