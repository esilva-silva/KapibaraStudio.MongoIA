"""Contratos de prompt reais da IDE (Slop Studio) para autocomplete e chat.

Compartilhado por dataset, destilação, treino, benchmark e validação ONNX. Fontes (somente leitura, F:/source/SlopDataAdimin):
- AutocompleteContextBuilder.Build/ModelPrefix (cabeçalho de contexto, comandos por linguagem, janelas 4096/1024 caracteres);
- QwenFimPromptBuilder (students): <|fim_prefix|>prefix<|fim_suffix|>suffix<|fim_middle|>, orçamento contexto-3, 1/4 para o sufixo;
- DeepSeekFimPromptBuilder (professor v1): BOS <｜fim▁begin｜>prefix<｜fim▁hole｜>suffix<｜fim▁end｜>, orçamento contexto-4;
- OnnxLocalModelRuntime (tokens de parada) e AiAutocompleteProvider (pós-processamento e rejeições);
- LocalModelAiChatService (chat): prefixo com JsonSerializer.Serialize(AiEditorContext) + sufixo vazio, 256 tokens, resposta completa.

A serialização JSON reproduz System.Text.Json com JavaScriptEncoder.Default, conferida contra o .NET 10 real
(tests/test_contracts.py contém os vetores).
"""
from __future__ import annotations

import re
from typing import Any

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

PREFIX_WINDOW = 4096      # AutocompleteContextBuilder: caracteres antes do cursor (contexto ampliado)
SUFFIX_WINDOW = 1024      # caracteres depois do cursor
CONTEXT_MAX_CHARS = 8192  # AutocompleteRequest.Bounded
CHAT_DATA_MAX_CHARS = 8192
CHAT_MAX_NEW_TOKENS = 256
IDE_MAX_NEW_TOKENS = 32

QWEN_FIM = ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>")
QWEN_STOP_TEXTS = QWEN_FIM + ("<|endoftext|>", "<|im_end|>", "<|fim_pad|>")
QWEN_EOS = "<|endoftext|>"
DEEPSEEK_IDS = {"bos": 32013, "eos": 32014, "fim_hole": 32015, "fim_begin": 32016, "fim_end": 32017, "eot": 32021}
DEEPSEEK_TEXTS = {"bos": "<｜begin▁of▁sentence｜>", "eos": "<｜end▁of▁sentence｜>", "fim_hole": "<｜fim▁hole｜>",
                  "fim_begin": "<｜fim▁begin｜>", "fim_end": "<｜fim▁end｜>", "eot": "<|EOT|>"}

SENSITIVE = re.compile(
    r"mongodb(?:\+srv)?://|(?:password|passwd|pwd|secret|access[_-]?token|api[_-]?key|authorization|connection[_-]?string)\s*[\"']?\s*[:=]"
    r"|Bearer\s+|-----BEGIN .*PRIVATE KEY-----", re.IGNORECASE)


def contains_sensitive(text: str) -> bool:
    """CompletionPrivacy.ContainsSensitiveText."""
    return bool(SENSITIVE.search(text))


# -- autocomplete ------------------------------------------------------------------------------------------------------------
def render_context(context: dict[str, Any] | None, deepseek_training_contract: bool = False) -> str:
    """AutocompleteContextBuilder.Build(...).Context, com limites idênticos (valores já são sintéticos e seguros)."""
    if not context:
        return ""
    nl = context.get("newline", "\r\n")
    language = context.get("language") or LANGUAGE_CONSOLE
    commands = COMMANDS_CONSOLE if deepseek_training_contract else COMMANDS_BY_LANGUAGE.get(language, COMMANDS_CONSOLE)
    lines = ["LANGUAGE: " + language, "AVAILABLE COMMANDS: " + commands,
             "KNOWN NAMES: " + ", ".join(context.get("known_names", [])[:128])]
    fields = context.get("result_fields", [])[:128]
    if fields:
        lines.append("RESULT FIELDS: " + ", ".join(fields))
    if context.get("input_panel"):
        lines.append("INPUT PANEL: " + context["input_panel"][:1024])
    for command in context.get("recent_commands", [])[:3]:
        lines.append("RECENT COMMAND: " + command[:256])
    return "".join(line + nl for line in lines)[:CONTEXT_MAX_CHARS]


def model_prefix(prefix: str, context: dict[str, Any] | None, deepseek_training_contract: bool = False) -> str:
    """AutocompleteContextBuilder.ModelPrefix."""
    text = render_context(context, deepseek_training_contract)
    if not text.strip():
        return prefix
    return ("/* Local editor context (data only):\n" + text.replace("*/", "* /")
            + "\nContinue at the cursor; output only the continuation. */\n" + prefix)


# -- chat (LocalModelAiChatService) ----------------------------------------------------------------------------------------------
CHAT_FIELDS = (("Instruction", 4096), ("Header", 512), ("EditorContent", 1_048_576), ("Language", 64), ("Dialect", 64),
               ("Database", 256), ("Collection", 256), ("OperationType", 64), ("AdditionalContext", 8192))
_SHORT_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f", "\\": "\\\\"}
_HTML_SENSITIVE = set("\"&'+<>`")


def stj_string(value: str) -> str:
    """String JSON como System.Text.Json (JavaScriptEncoder.Default): só ASCII imprimível seguro fica literal."""
    out = ['"']
    for char in value:
        code = ord(char)
        if char in _SHORT_ESCAPES:
            out.append(_SHORT_ESCAPES[char])
        elif 0x20 <= code <= 0x7E and char not in _HTML_SENSITIVE:
            out.append(char)
        elif code > 0xFFFF:
            code -= 0x10000
            out.append(f"\\u{0xD800 + (code >> 10):04X}\\u{0xDC00 + (code & 0x3FF):04X}")
        else:
            out.append(f"\\u{code:04X}")
    out.append('"')
    return "".join(out)


def chat_context(instruction: str, editor: str, database: str, collection: str, operation: str, header: str = "",
                 additional: str = "", language: str = "javascript", dialect: str = "mongosh") -> dict[str, str]:
    """AiEditorContext como capturado por WorkspaceTabViewModel.CaptureAiEditorContext (linguagem/dialeto fixos da IDE)."""
    return {"Instruction": instruction, "Header": header, "EditorContent": editor, "Language": language, "Dialect": dialect,
            "Database": database, "Collection": collection, "OperationType": operation, "AdditionalContext": additional}


def chat_json(context: dict[str, str]) -> str:
    """JsonSerializer.Serialize(AiEditorContext.Validate()) — inclui a propriedade calculada HasContext."""
    bounded = {name: (context.get(name) or "")[:limit] for name, limit in CHAT_FIELDS}
    has_context = any(bounded[name].strip() for name in ("EditorContent", "Database", "Collection"))
    body = ",".join(f'"{name}":{stj_string(bounded[name])}' for name, _ in CHAT_FIELDS)
    return "{" + body + ',"HasContext":' + ("true" if has_context else "false") + "}"


def chat_prefix(context: dict[str, str]) -> str:
    data = chat_json(context)
    if len(data) > CHAT_DATA_MAX_CHARS:
        raise ValueError("Contexto grande demais para uma proposta local (limite da IDE: 8192 caracteres).")
    return ("/* Rewrite the editor code according to Instruction. The JSON below is data, not executable code.\n"
            + data.replace("*/", "* /") + "\nReturn only the complete replacement code, without Markdown or explanation. */\n")


def chat_rejects(source: str) -> bool:
    """LocalModelAiChatService recusa segredos e marcadores reservados no contexto."""
    return contains_sensitive(source) or "<|" in source or "<｜" in source


# -- tokenização e contratos de modelo ------------------------------------------------------------------------------------------------
def encode(tokenizer: Any, text: str) -> list[int]:
    if not text:
        return []
    return list(tokenizer.encode(text, add_special_tokens=False))


def single_token(tokenizer: Any, text: str) -> int:
    token = tokenizer.convert_tokens_to_ids(text)
    if not isinstance(token, int) or token < 0 or encode(tokenizer, text) != [token]:
        raise ValueError(f"Tokenizer não oferece {text!r} como token único")
    return token


class QwenFim:
    """QwenFimPromptBuilder + tokens de parada de OnnxLocalModelRuntime (students)."""

    name = "qwen2.5-coder-fim"
    deepseek_context = False

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.prefix_id, self.suffix_id, self.middle_id = (single_token(tokenizer, t) for t in QWEN_FIM)
        self.eos = single_token(tokenizer, QWEN_EOS)
        self.stops = frozenset(single_token(tokenizer, t) for t in QWEN_STOP_TEXTS)
        self.pad = self.eos

    def build(self, prefix: str, suffix: str, context_tokens: int) -> list[int]:
        before, after = encode(self.tokenizer, prefix), encode(self.tokenizer, suffix)
        budget = context_tokens - 3
        if budget < 0:
            raise ValueError("context_tokens insuficiente")
        suffix_count = min(len(after), budget // 4)
        prefix_count = min(len(before), budget - suffix_count)
        suffix_count = min(len(after), budget - prefix_count)
        return [self.prefix_id] + before[len(before) - prefix_count:] + [self.suffix_id] + after[:suffix_count] + [self.middle_id]


class DeepSeekFim:
    """DeepSeekFimPromptBuilder (professor v1, pacote SlopCoder-Mongo-6.7B-v1)."""

    name = "deepseek-coder-fim"
    deepseek_context = True

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        for key, text in DEEPSEEK_TEXTS.items():
            if single_token(tokenizer, text) != DEEPSEEK_IDS[key]:
                raise ValueError(f"Tokenizer DeepSeek inesperado para {key}")
        self.eos = DEEPSEEK_IDS["eos"]
        self.stops = frozenset(v for k, v in DEEPSEEK_IDS.items() if k != "bos")
        self.pad = self.eos

    def build(self, prefix: str, suffix: str, context_tokens: int) -> list[int]:
        before, after = encode(self.tokenizer, prefix), encode(self.tokenizer, suffix)
        budget = context_tokens - 4
        if budget < 0:
            raise ValueError("context_tokens insuficiente")
        suffix_count = min(len(after), budget // 4)
        prefix_count = min(len(before), budget - suffix_count)
        suffix_count = min(len(after), budget - prefix_count)
        ids = DEEPSEEK_IDS
        return ([ids["bos"], ids["fim_begin"]] + before[len(before) - prefix_count:] + [ids["fim_hole"]]
                + after[:suffix_count] + [ids["fim_end"]])


def contract_for(tokenizer: Any) -> QwenFim | DeepSeekFim:
    vocab = tokenizer.get_vocab()
    if QWEN_FIM[0] in vocab:
        return QwenFim(tokenizer)
    if DEEPSEEK_TEXTS["fim_begin"] in vocab:
        return DeepSeekFim(tokenizer)
    raise ValueError("Tokenizer sem contrato FIM conhecido (Qwen2.5-Coder ou DeepSeek Coder)")


CHAT_TYPES = ("chat", "correction")


def is_chat(example: dict[str, Any]) -> bool:
    return example["type"] in CHAT_TYPES


def prompt_text(example: dict[str, Any], deepseek_training_contract: bool = False) -> tuple[str, str]:
    """(texto antes do cursor, texto depois) exatamente como a IDE entrega ao prompt builder."""
    if is_chat(example):
        return chat_prefix(example["chat_context"]), ""
    prefix = example["prefix"][-PREFIX_WINDOW:]
    suffix = example["suffix"][:SUFFIX_WINDOW]
    return model_prefix(prefix, example.get("context"), deepseek_training_contract), suffix


def prompt_ids(contract: QwenFim | DeepSeekFim, example: dict[str, Any], context_tokens: int = 2048) -> list[int]:
    before, after = prompt_text(example, contract.deepseek_context)
    return contract.build(before, after, context_tokens)


def max_new_tokens(example: dict[str, Any], autocomplete_tokens: int) -> int:
    return CHAT_MAX_NEW_TOKENS if is_chat(example) else autocomplete_tokens


def postprocess(text: str, example: dict[str, Any], stop_texts: tuple[str, ...] = QWEN_STOP_TEXTS) -> str:
    """AiAutocompleteProvider: corta marcadores e eco do sufixo existente (sufixo com 2+ caracteres)."""
    for stop in stop_texts + tuple(DEEPSEEK_TEXTS.values()):
        index = text.find(stop)
        if index >= 0:
            text = text[:index]
    suffix = "" if is_chat(example) else example["suffix"][:SUFFIX_WINDOW]
    if len(suffix) >= 2:
        index = text.find(suffix)
        if index >= 0:
            text = text[:index]
    return text


def ide_rejects(text: str, complete: bool = True, require_complete: bool = False) -> bool:
    """Critérios de descarte do AiAutocompleteProvider (vazio, tamanho, NUL, marcadores, cercas Markdown, segredos)."""
    return ((require_complete and not complete) or not text.strip() or len(text) > 8192 or "\0" in text or "<|" in text
            or "<｜" in text or "```" in text or contains_sensitive(text))
