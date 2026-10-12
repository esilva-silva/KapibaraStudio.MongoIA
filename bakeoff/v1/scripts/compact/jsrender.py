"""Renderização de valores JS/Extended JSON com variações de estilo (aspas, indentação, largura, espaços)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

IDENT = re.compile(r"^[A-Za-z_$][\w$]*$")


class Raw:
    """Expressão JS literal (variável, chamada, operação). Não representável em JSON."""

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text


class Oid:
    __slots__ = ("hex",)

    def __init__(self, hex_value: str) -> None:
        self.hex = hex_value


class DateV:
    __slots__ = ("iso", "form")

    def __init__(self, iso: str, form: str = "isodate") -> None:
        self.iso = iso
        self.form = form  # isodate | newdate


class UuidV:
    __slots__ = ("value", "ctor")

    def __init__(self, value: str, ctor: str = "UUID") -> None:
        self.value = value
        self.ctor = ctor


class DecimalV:
    """Decimal128: NumberDecimal("…") no JavaScript, {"$numberDecimal": "…"} em Extended JSON."""

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text


class LongV:
    """Int64: NumberLong("…") no JavaScript, {"$numberLong": "…"} em Extended JSON."""

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text


class Regex:
    __slots__ = ("pattern", "flags")

    def __init__(self, pattern: str, flags: str = "") -> None:
        self.pattern = pattern
        self.flags = flags


class Money(float):
    """Float renderizado com duas casas quando necessário."""


@dataclass
class Style:
    indent: int = 2
    width: int = 80
    key_quotes: str = "minimal"  # minimal | all
    string_quote: str = '"'
    brace_space: bool = True
    json: bool = False
    semicolons: bool = False
    chain_break: bool = False  # quebra cadeias longas em linhas iniciadas por "."


def quote(text: str, style: Style) -> str:
    if style.json or style.string_quote == '"':
        return json.dumps(text, ensure_ascii=False)
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n") + "'"


def key(name: str, style: Style) -> str:
    if style.json or style.key_quotes == "all" or not IDENT.match(name):
        return quote(name, style)
    return name


def number(value: float | int) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, Money) or abs(value - round(value, 2)) < 1e-9:
        text = f"{value:.2f}".rstrip("0").rstrip(".")
        return text if text not in ("-0", "") else "0"
    return repr(round(value, 4))


def scalar(value: Any, style: Style) -> str | None:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return number(value)
    if isinstance(value, str):
        return quote(value, style)
    if isinstance(value, Raw):
        if style.json:
            raise ValueError("Raw não pode ser renderizado em JSON")
        return value.text
    if isinstance(value, Oid):
        return render_inline({"$oid": value.hex}, style) if style.json else f"ObjectId({quote(value.hex, style)})"
    if isinstance(value, DateV):
        if style.json:
            return render_inline({"$date": value.iso}, style)
        return f"new Date({quote(value.iso, style)})" if value.form == "newdate" else f"ISODate({quote(value.iso, style)})"
    if isinstance(value, UuidV):
        if style.json:
            return render_inline({"$uuid": value.value}, style)
        return f"{value.ctor}({quote(value.value, style)})"
    if isinstance(value, DecimalV):
        return render_inline({"$numberDecimal": value.text}, style) if style.json else f"NumberDecimal({quote(value.text, style)})"
    if isinstance(value, LongV):
        return render_inline({"$numberLong": value.text}, style) if style.json else f"NumberLong({quote(value.text, style)})"
    if isinstance(value, Regex):
        if style.json:
            return render_inline({"$regex": value.pattern, "$options": value.flags}, style)
        return f"/{value.pattern}/{value.flags}"
    return None


def render_inline(value: Any, style: Style) -> str:
    text = scalar(value, style)
    if text is not None:
        return text
    if isinstance(value, dict):
        if not value:
            return "{}"
        body = ", ".join(f"{key(k, style)}: {render_inline(v, style)}" for k, v in value.items())
        return "{ " + body + " }" if style.brace_space else "{" + body + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(render_inline(v, style) for v in value) + "]"
    raise TypeError(f"valor não renderizável: {type(value).__name__}")


def render(value: Any, style: Style, level: int = 0, used: int = 0) -> str:
    """Inline quando cabe na largura; caso contrário quebra objetos/arrays em linhas."""
    inline = render_inline(value, style)
    if not isinstance(value, (dict, list, tuple)) or not value or used + len(inline) <= style.width:
        return inline
    pad = " " * (style.indent * (level + 1))
    close = " " * (style.indent * level)
    if isinstance(value, dict):
        lines = []
        for k, v in value.items():
            head = f"{key(k, style)}: "
            lines.append(pad + head + render(v, style, level + 1, len(pad) + len(head)))
        return "{\n" + ",\n".join(lines) + "\n" + close + "}"
    lines = [pad + render(v, style, level + 1, len(pad)) for v in value]
    return "[\n" + ",\n".join(lines) + "\n" + close + "]"


def call(name: str, args: list[Any], style: Style, level: int = 0, used: int = 0) -> str:
    """Renderiza ``name(arg1, arg2)`` quebrando só o último argumento quando necessário."""
    rendered = [render_inline(a, style) for a in args]
    text = f"{name}(" + ", ".join(rendered) + ")"
    if used + len(text) <= style.width or not args:
        return text
    head = f"{name}(" + "".join(r + ", " for r in rendered[:-1])
    return head + render(args[-1], style, level, used + len(head)) + ")"
