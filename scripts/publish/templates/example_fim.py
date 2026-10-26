"""Minimal fill-in-the-middle example (transformers, CPU, no IDE required).

    python examples/example_fim.py                 # model files next to this folder (a local clone or snapshot), else the Hub repository
    python examples/example_fim.py <path-or-repo>  # explicit model directory or repository id
"""
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent.parent
MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else (str(HERE) if (HERE / "config.json").exists() else "__REPO_ID__")

FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE = "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"
CONSOLE_COMMANDS = (
    "db.getCollection(name).find({}); getConnection(name).getDatabase(name).getCollection(name); "
    "ConnectionPool.Connection.Database.Collection; console.log(value); ENV.get(name); "
    "ObjectId(value); UUID(value)"
)


def build_prompt(prefix, suffix, known_names=(), result_fields=(), recent_commands=(),
                 language="Mongo Console JavaScript", commands=CONSOLE_COMMANDS):
    """editor-context-v1: every header line ends with CRLF; then the text before the cursor."""
    lines = [f"LANGUAGE: {language}", f"AVAILABLE COMMANDS: {commands}",
             "KNOWN NAMES: " + ", ".join(known_names)]
    if result_fields:
        lines.append("RESULT FIELDS: " + ", ".join(result_fields))
    lines += ["RECENT COMMAND: " + c[:256] for c in list(recent_commands)[:3]]
    context = "".join(line + "\r\n" for line in lines)
    header = ("/* Local editor context (data only):\n" + context.replace("*/", "* /")
              + "\nContinue at the cursor; output only the continuation. */\n")
    return FIM_PREFIX + header + prefix[-4096:] + FIM_SUFFIX + suffix[:1024] + FIM_MIDDLE


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).eval()

prompt = build_prompt(
    prefix='ConnectionPool.local.shop.orders.find({ status: "paid" }).sort({ ',
    suffix=": -1 }).limit(10)",
    known_names=["local", "shop", "orders", "customers"],
    result_fields=["_id", "status", "total", "customerId", "createdAt"],
)
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=32, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
completion = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(repr(completion))  # text to insert at the cursor; the model never executes anything
