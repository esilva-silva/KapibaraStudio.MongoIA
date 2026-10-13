#!/usr/bin/env python3
"""Renderer Python do contrato `editor-context-v1` + FIM Qwen (spec fase-03 §4.1).

Porta de `AutocompleteContextBuilder.Build/ModelPrefix` e `QwenFimPromptBuilder` da IDE (somente leitura), conferida contra
o golden congelado da IDE (`data/contracts/editor-context-v1.golden`, copia de
`tests/.../Ai/Golden/editor-context-v1.golden`) e contra o renderer da v1 (`bakeoff/v1/scripts/compact/contracts.py`, nao alterado).

Detalhes do contrato (todos cobertos por testes):
  * cabecalho: linhas `LANGUAGE`/`AVAILABLE COMMANDS`/`KNOWN NAMES`/`RESULT FIELDS`/`INPUT PANEL`/`RECENT COMMAND` terminadas em
    **CRLF fixo** (`ContractNewLine`), inclusive no Linux; o `context` e cortado em 8192 caracteres (`Bounded`);
  * `ModelPrefix` embrulha com `\\n` LITERAL: `/* Local editor context (data only):\\n` + contexto (com `*/` -> `* /`) +
    `\\nContinue at the cursor; output only the continuation. */\\n` + prefixo; contexto em branco devolve so o prefixo;
  * `AVAILABLE COMMANDS` por linguagem (`json`, `JavaScript (mongosh)`, demais = lista do Console);
  * janelas: 4096 caracteres antes e 1024 depois do cursor, contados em UNIDADES UTF-16 (como o `string` do .NET);
  * nomes/campos/comandos recentes/painel de entrada com segredo (`CompletionPrivacy`) ou `<|`/`<｜` sao descartados;
    ate 128 nomes, 128 campos, 3 comandos recentes de 256 caracteres, painel de entrada 1024;
  * FIM Qwen: `<|fim_prefix|>` + cauda do prefixo + `<|fim_suffix|>` + inicio do sufixo + `<|fim_middle|>`, orcamento
    `contextTokens - 3`, sufixo <= 25% do orcamento; o excedente corta o INICIO do prefixo (o cabecalho cai primeiro).

Uso programatico: `render_prompt(record, tokenizer)` -> dict(prompt, prompt_tokens, truncated, ...).
Hash esperado do prompt: `prompt_sha256(prompt)` (UTF-8, sem normalizacao).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

CONTRACT_ID = "editor-context-v1"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOKENIZER = ROOT / "BaseRepos" / "Qwen2.5-Coder-0.5B" / "tokenizer.json"

LANGUAGE_CONSOLE = "Mongo Console JavaScript"
LANGUAGE_SCRIPT = "JavaScript (mongosh)"
LANGUAGE_AGGREGATION = "json"
DIALECT_LANGUAGE = {"console": LANGUAGE_CONSOLE, "mongosh": LANGUAGE_SCRIPT, "json": LANGUAGE_AGGREGATION}

COMMANDS_CONSOLE = ("db.getCollection(name).find({}); getConnection(name).getDatabase(name).getCollection(name); "
                    "ConnectionPool.Connection.Database.Collection; console.log(value); ENV.get(name); ObjectId(value); UUID(value)")
COMMANDS_BY_LANGUAGE = {
    LANGUAGE_AGGREGATION: ("MongoDB aggregation pipeline: $match, $project, $group, $sort, $limit, $lookup, $unwind; "
                           "operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $and, $or"),
    LANGUAGE_SCRIPT: ("db.getCollection(name).find({}); db.getSiblingDB(name); findOne, aggregate, sort, limit, countDocuments; "
                      "print(value); ObjectId(value); UUID(value)"),
}

CONTRACT_NEWLINE = "\r\n"          # AutocompleteContextBuilder.ContractNewLine
WRAP_HEAD = "/* Local editor context (data only):\n"
WRAP_TAIL = "\nContinue at the cursor; output only the continuation. */\n"
PREFIX_WINDOW = 4096               # AutocompleteContextBuilder.Build com UseEditorContext
SUFFIX_WINDOW = 1024
CONTEXT_MAX_CHARS = 8192           # AutocompleteRequest.Bounded
MAX_NAMES = 128
MAX_FIELDS = 128
MAX_RECENT = 3
RECENT_MAX_CHARS = 256
INPUT_MAX_CHARS = 1024
CONTEXT_TOKENS = 2048              # AutocompleteSettings.ContextTokens
QWEN_FIM = ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>")

_SENSITIVE = re.compile(
    r"mongodb(?:\+srv)?://|(?:password|passwd|pwd|secret|access[_-]?token|api[_-]?key|authorization|connection[_-]?string)"
    r"\s*[\"']?\s*[:=]|Bearer\s+|-----BEGIN .*PRIVATE KEY-----", re.IGNORECASE)


# -- UTF-16 (o `string` do .NET) --------------------------------------------------------------------------------------------------
def _is_bmp(text: str) -> bool:
    return all(ord(c) <= 0xFFFF for c in text) if not text.isascii() else True


def _u16(text: str) -> bytes:
    return text.encode("utf-16-le", "surrogatepass")


def _from_u16(data: bytes) -> str:
    return data.decode("utf-16-le", "surrogatepass")


def u16_len(text: str) -> int:
    return len(text) if _is_bmp(text) else len(_u16(text)) // 2


def u16_slice(text: str, start: int | None, stop: int | None) -> str:
    """text[start:stop] em unidades UTF-16 (indices como em C#; sem indices negativos relativos ao fim: use u16_len)."""
    if _is_bmp(text):
        return text[start:stop]
    units = _u16(text)
    n = len(units) // 2
    s = 0 if start is None else max(0, min(n, start))
    e = n if stop is None else max(0, min(n, stop))
    return _from_u16(units[2 * s:2 * max(s, e)])


def u16_tail(text: str, count: int) -> str:
    n = u16_len(text)
    return text if n <= count else u16_slice(text, n - count, n)


# -- privacidade / filtros -----------------------------------------------------------------------------------------------------
def contains_sensitive(text: str) -> bool:
    return bool(_SENSITIVE.search(text))


def is_safe(text: str) -> bool:
    """AutocompleteContextBuilder.IsSafe."""
    return not contains_sensitive(text) and "<|" not in text and "<｜" not in text


# -- AutocompleteContextBuilder.Build ------------------------------------------------------------------------------------------------
def commands_for(language: str | None) -> str:
    return COMMANDS_BY_LANGUAGE.get(language or "", COMMANDS_CONSOLE)


def build_context(language: str | None, known_names: Sequence[str] = (), result_fields: Sequence[str] = (), input_panel: str = "",
                  recent_commands: Sequence[str] = (), *, use_editor_context: bool = True, use_result_panel: bool = True,
                  use_input_panel: bool = True, _legacy_newline: str | None = None) -> str:
    """`AutocompleteRequest.Context` (ja `Bounded`): cabecalho com CRLF fixo.

    `_legacy_newline` existe so para medir a paridade com o renderer da v1 (que honrava `context.newline`); nunca usar em dados."""
    names = [n for n in known_names if is_safe(n)][:MAX_NAMES]
    fields = [f for f in result_fields if is_safe(f)][:MAX_FIELDS] if use_result_panel else []
    nl = _legacy_newline or CONTRACT_NEWLINE
    out = ["LANGUAGE: " + (language or "") + nl, "AVAILABLE COMMANDS: " + commands_for(language) + nl,
           "KNOWN NAMES: " + ", ".join(names) + nl]
    if fields:
        out.append("RESULT FIELDS: " + ", ".join(fields) + nl)
    if use_input_panel and len(input_panel) > 0 and is_safe(input_panel):
        out.append("INPUT PANEL: " + u16_slice(input_panel, 0, INPUT_MAX_CHARS) + nl)
    if use_editor_context:
        for command in [c for c in recent_commands if is_safe(c)][:MAX_RECENT]:
            out.append("RECENT COMMAND: " + u16_slice(command, 0, RECENT_MAX_CHARS) + nl)
    return u16_slice("".join(out), 0, CONTEXT_MAX_CHARS)


def build_request(text: str, caret: int, language: str | None, input_panel: str = "", result_fields: Sequence[str] = (),
                  known_names: Sequence[str] = (), recent_commands: Sequence[str] = (), *, use_editor_context: bool = True,
                  use_result_panel: bool = True, use_input_panel: bool = True) -> dict:
    """AutocompleteContextBuilder.Build(snapshot, settings) -> {prefix, suffix, context, dictionary} (como `AutocompleteRequest.Bounded`)."""
    prefix, suffix = window(text, caret, use_editor_context=use_editor_context)
    names = [n for n in known_names if is_safe(n)][:MAX_NAMES]
    fields = [f for f in result_fields if is_safe(f)][:MAX_FIELDS] if use_result_panel else []
    context = build_context(language, known_names, result_fields, input_panel, recent_commands, use_editor_context=use_editor_context,
                            use_result_panel=use_result_panel, use_input_panel=use_input_panel)
    dictionary = list(dict.fromkeys(names + fields))[:256]
    return {"prefix": prefix, "suffix": suffix, "context": context, "dictionary": [u16_slice(w, 0, 128) for w in dictionary]}


def window(text: str, caret: int, *, use_editor_context: bool = True) -> tuple[str, str]:
    """(prefixo, sufixo) do snapshot: 4096 antes / 1024 depois (128 / "" sem contexto de editor); caret em unidades UTF-16."""
    n = u16_len(text)
    caret = max(0, min(caret, n))
    w = PREFIX_WINDOW if use_editor_context else 128
    prefix = u16_slice(text, max(0, caret - w), caret)
    suffix = u16_slice(text, caret, min(n, caret + SUFFIX_WINDOW)) if use_editor_context else ""
    return prefix, suffix


def bound_window(prefix: str, suffix: str) -> tuple[str, str]:
    """Janela sobre prefixo/sufixo ja separados (registros do dataset)."""
    return u16_tail(prefix, PREFIX_WINDOW), u16_slice(suffix, 0, SUFFIX_WINDOW)


# -- AutocompleteContextBuilder.ModelPrefix ----------------------------------------------------------------------------------------
def model_prefix(prefix: str, context: str | None, *, deepseek_training_contract: bool = False) -> str:
    if context is None or not context.strip():
        return prefix
    if deepseek_training_contract:
        label = "AVAILABLE COMMANDS: "
        start = context.find(label)
        if start >= 0:
            end = len(context)
            for ch in ("\r", "\n"):
                i = context.find(ch, start)
                if i >= 0:
                    end = min(end, i)
            context = context[:start] + label + COMMANDS_CONSOLE + context[end:]
    return WRAP_HEAD + context.replace("*/", "* /") + WRAP_TAIL + prefix


def editor_language(record: dict) -> str:
    return record.get("language") or DIALECT_LANGUAGE.get(record.get("dialect", "console"), LANGUAGE_CONSOLE)


def editor_context(editor: dict | None, language: str) -> str | None:
    """Cabecalho de um registro do dataset (`editor` do §4.4); `None` = registro sem contexto (8%)."""
    if not editor:
        return None
    return build_context(language, editor.get("known_names", []), editor.get("result_fields", []), editor.get("input_panel", ""),
                         editor.get("recent_commands", []))


def render_model_prefix(prefix: str, editor: dict | None, language: str) -> str:
    return model_prefix(prefix, editor_context(editor, language))


# -- FIM Qwen ------------------------------------------------------------------------------------------------------------------
def qwen_fim_text(before: str, after: str) -> str:
    return f"{QWEN_FIM[0]}{before}{QWEN_FIM[1]}{after}{QWEN_FIM[2]}"


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8", "surrogatepass")).hexdigest()


class Tokenizer:
    """Tokenizer do Qwen2.5-Coder (`tokenizer.json`, biblioteca `tokenizers`): mesmo vocabulario do adapter qwen2 da IDE."""

    def __init__(self, path: str | Path = DEFAULT_TOKENIZER):
        from tokenizers import Tokenizer as _Tok

        self.path = Path(path)
        self._tok = _Tok.from_file(str(self.path))
        self.marker_ids = tuple(self._tok.token_to_id(t) for t in QWEN_FIM)
        if any(m is None for m in self.marker_ids):
            raise ValueError("Tokenizer sem tokens FIM do Qwen Coder")

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids if text else []

    def encode_batch(self, texts: Iterable[str]) -> list[list[int]]:
        return [e.ids for e in self._tok.encode_batch(list(texts), add_special_tokens=False)]

    def decode(self, ids: Sequence[int]) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=False)


def build_ids(tokenizer: Tokenizer, before: str, after: str, context_tokens: int = CONTEXT_TOKENS,
              enc: tuple[list[int], list[int]] | None = None) -> tuple[list[int], dict]:
    """QwenFimPromptBuilder.Build. Retorna (ids, info) com contagens pre-truncamento."""
    b, a = enc if enc is not None else (tokenizer.encode(before), tokenizer.encode(after))
    budget = context_tokens - 3
    if budget < 0:
        raise ValueError("context_tokens insuficiente")
    suffix_count = min(len(a), budget // 4)
    prefix_count = min(len(b), budget - suffix_count)
    suffix_count = min(len(a), budget - prefix_count)
    m = tokenizer.marker_ids
    ids = [m[0]] + b[len(b) - prefix_count:] + [m[1]] + a[:suffix_count] + [m[2]]
    return ids, {"prefix_tokens": len(b), "suffix_tokens": len(a), "kept_prefix": prefix_count, "kept_suffix": suffix_count,
                 "truncated": prefix_count < len(b) or suffix_count < len(a)}


def render_prompt(record: dict, tokenizer: Tokenizer | None = None, context_tokens: int = CONTEXT_TOKENS) -> dict:
    """Prompt FIM final de um registro com `prefix`, `suffix`, `editor` (ou None) e `language`/`dialect`.

    Sem tokenizer: so o texto (sem orcamento). Com tokenizer: aplica o orcamento; se truncar, o texto e o decode dos ids mantidos.
    """
    prefix, suffix = bound_window(record["prefix"], record["suffix"])
    before = render_model_prefix(prefix, record.get("editor"), editor_language(record))
    text = qwen_fim_text(before, suffix)
    out = {"prompt": text, "before": before, "after": suffix, "truncated": False, "prompt_tokens": None}
    if tokenizer is None:
        return out
    ids, info = build_ids(tokenizer, before, suffix, context_tokens)
    out["prompt_tokens"] = len(ids)
    out["truncated"] = info["truncated"]
    out["info"] = info
    if info["truncated"]:
        out["prompt"] = tokenizer.decode(ids)
    return out
