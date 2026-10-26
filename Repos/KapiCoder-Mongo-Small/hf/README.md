---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-1.5B
base_model_relation: finetune
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
- qwen2
- synthetic-data
model-index:
- name: KapiCoder-Mongo-Small
  results:
  - task:
      type: text-generation
      name: MongoDB fill-in-the-middle autocomplete
    dataset:
      name: KapiCoder v2c validation split, internal and synthetic (n=2000)
      type: kapicoder-v2c-val-synthetic
    metrics:
    - type: fim_accuracy
      name: FIM accuracy (normalized exact match)
      value: 0.41
      verified: false
    - type: exec_valid
      name: Exec-valid (completion parses and runs read-only)
      value: 0.6015
      verified: false
    - type: exec_valid_vs_ceiling
      name: Exec-valid divided by validator ceiling (0.6255)
      value: 0.9616
      verified: false
    - type: syntax_valid
      name: Syntax-valid
      value: 0.998
      verified: false
    - type: nonexistent_fields_full_pct
      name: Nonexistent fields, schema visible (context_kind=full), percent
      value: 1.21
      verified: false
  - task:
      type: text-generation
      name: Generic JavaScript fill-in-the-middle (regression check)
    dataset:
      name: Permissive npm JavaScript FIM cuts, internal (n=500)
      type: kapicoder-generic-js-fim-internal
    metrics:
    - type: fim_accuracy
      name: FIM accuracy (normalized exact match)
      value: 0.606
      verified: false
  - task:
      type: text-generation
      name: MongoDB fill-in-the-middle autocomplete on unseen schemas
    dataset:
      name: Internal evaluation, synthetic out-of-distribution homologation v1 (n=6030, LLM-written, 28 unseen telecom/adtech schemas; not fully independent of the v3 training data)
      type: kapicoder-homolog-v1-synthetic-ood
    metrics:
    - type: fim_accuracy
      name: FIM accuracy (normalized exact match), homologation
      value: 0.4537
      verified: false
    - type: exec_valid
      name: Exec-valid (completion parses and runs read-only), homologation
      value: 0.8307
      verified: false
    - type: exec_valid_vs_ceiling
      name: Exec-valid divided by validator ceiling (0.9307), homologation
      value: 0.8926
      verified: false
    - type: nonexistent_fields_full_pct
      name: Nonexistent fields, schema visible (context_kind=full), percent, homologation
      value: 7.5
      verified: false
---

# KapiCoder-Mongo-Small (1.5B)

> **Status: PREVIEW / experimental, public preview.** Published for inspection and feedback, not for production use.
> **Internal evaluation on synthetic data; no blind human-written set.** Read [Limitations and risks](#limitations-and-risks) before using it.

KapiCoder-Mongo-Small is a small code-completion model (about 1.5 B parameters) fine-tuned from
[`Qwen/Qwen2.5-Coder-1.5B`](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B) to **complete MongoDB queries at the cursor**
(fill-in-the-middle, FIM). It was built for the *Kapibara Studio* editor, which sends it the text before and after the cursor plus a short
context header (the `editor-context-v1` contract described below). This is version **v3** of the weights (repository tag `v0.1.0-preview`).

<!-- BEGIN:key-caveats -->
**Read this first**

- **Internal evaluation, synthetic data, no human-written queries.** Every number in this card comes from the authors' own evaluation on synthetic data (a validation split of the training generator and a homologation set written by an LLM and a grammar); nothing was measured on queries written by people.
- **Invented field names on unseen schemas.** With the schema visible (`full`), 7.50% of this model's suggestions on the homologation set use a field that does not exist, against 7.15% for the zero-shot base and 9.19% for the previous internal checkpoint (not released): close to the base. Validate field names before running a suggestion.
- **The homologation set is not fully independent of the v3 training data.** The v3 corpus contains queries from the same LLM generator (Qwen3-14B) and grammar templates over other schemas (training cuts that shared long n-grams with the set were removed by decontamination, and its 28 schemas were never in training), and the set is LLM- and grammar-written, not written by people: the gain on LLM-written queries probably overstates the gain on queries written by people.
- **A single training run per model** (one seed) and a single greedy evaluation run: there is no estimate of variance between runs.
<!-- END:key-caveats -->

## What it is, and what it is not

**It is** an autocomplete model. Given the text before and after the cursor, it proposes a short continuation (the IDE caps it at 32 tokens)
for MongoDB Console JavaScript (`ConnectionPool.<conn>.<db>.<collection>.find(...)`, aggregation pipelines, `$search` stages, index definitions)
and JSON pipelines.

**It is not**
- a chat model or an agent: it does not follow instructions, hold a conversation, call tools or explain code;
- a query executor: it **only suggests text and never executes anything**. Whatever executes a suggestion is the host application;
- a validator: a suggestion can be syntactically valid and still wrong, refer to fields that do not exist, or be destructive if executed;
- a general coding model: it keeps some generic JavaScript ability (see the Generic-JS column below) but was not tuned for it.

## Intended use and out of scope

Intended: inline autocomplete suggestions inside an editor where a person reviews what is inserted, with the collection's field names visible in the
context header whenever possible.

Out of scope: running suggestions without review; generating or modifying production data unattended; security, compliance or access-control
decisions; any dialect other than the ones listed under Limitations; chat, code review or explanation tasks.

## How to use

### Prompt format (`editor-context-v1`)

The training and evaluation prompt is a Qwen FIM prompt whose "prefix" part starts with a comment block describing the editor state:

```
<|fim_prefix|>/* Local editor context (data only):
LANGUAGE: Mongo Console JavaScript
AVAILABLE COMMANDS: db.getCollection(name).find({}); getConnection(name)...; UUID(value)
KNOWN NAMES: local, shop, orders, customers
RESULT FIELDS: _id, status, total, customerId, createdAt

Continue at the cursor; output only the continuation. */
<text before the cursor><|fim_suffix|><text after the cursor><|fim_middle|>
```

Rules of the contract that the example below implements:
- header lines are `LANGUAGE`, `AVAILABLE COMMANDS`, `KNOWN NAMES` (connection, database and collection names), optional `RESULT FIELDS` (field names shown in the result panel), optional `RECENT COMMAND` (up to 3, 256 characters each);
- every header line ends with CRLF (`\r\n`); the lines around the block use `\n`;
- `*/` inside the header is rewritten to `* /`;
- the text before the cursor is limited to 4096 characters and the text after it to 1024; the whole prompt is budgeted to 2048 tokens (when it has to be cut, a quarter of the budget is reserved for the suffix and the rest goes to the prefix);
- greedy decoding, stop at `<|endoftext|>`; the IDE asks for at most 32 new tokens (the evaluation below allowed 96).

`LANGUAGE` values seen in training are `Mongo Console JavaScript` and `json` (see Limitations). Prompts without a header also work, but the evaluation shows
more nonexistent fields when the header does not list the collection's fields.

### Minimal example (transformers, no IDE required)

The same file is in the repository as `examples/example_fim.py`.

```python
"""Minimal fill-in-the-middle example (transformers, CPU, no IDE required).

    python examples/example_fim.py                 # model files next to this folder (a local clone or snapshot), else the Hub repository
    python examples/example_fim.py <path-or-repo>  # explicit model directory or repository id
"""
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent.parent
MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else (str(HERE) if (HERE / "config.json").exists() else "esilva/KapiCoder-Mongo-Small")

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
```

It runs on CPU. Use `add_special_tokens=False`: the FIM markers are already in the prompt text. Output varies with the prompt; always review the suggestion.
For the prompt above, greedy decoding on CPU (transformers 4.57.6, bf16 weights of this repository) prints `'total'`. With that transformers version, loading the tokenizer prints a warning about a regex pattern; the tokenizer files are byte-identical to those of the base model (same SHA-256, see `provenance/model_lock.json`).

## Training

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-1.5B`, revision `df3ce67c0e24480f20468b6ef2894622d69eb73b` (Apache-2.0). Training started from the base model, not from the previous internal checkpoint |
| Method | LoRA (r=64, alpha=128, dropout 0.05 on the q, k, v, o, gate, up and down projections; 73,859,072 trainable parameters) on the bf16 base, merged into full bf16 weights afterwards. AdamW, peak LR 1.5e-4, 4,617 steps of 32,768 padded tokens = 1.5 epochs (a 2-epoch plan was cut to 1.5 epochs to save time; the learning-rate schedule covers the 4,617 steps) |
| Training data | 309,571 examples (292,812 MongoDB examples + 16,759 generic-code replay examples), 86,000,018 tokens per epoch; 129,085,790 tokens seen in total |
| Sources of the MongoDB examples | 58.0% distilled corpus of the previous version (169,831 examples; targets written by a 7B teacher, `KapiTeacher-Mongo-7B`, a QLoRA fine-tune of Qwen2.5-Coder-7B, and kept only when they pass an execution check against a real MongoDB with read-only commands; raw-FIM cuts keep the original reference); 26.0% queries written by Qwen3-14B (76,248 examples; reference label, validated by executing the real text); 16.0% queries from a grammar over 61 synthetic v3 training schemas (46,733 examples; reference label). LLM + grammar = 42.0% of the MongoDB cuts |
| Context kinds | 34.8% of the MongoDB cuts have no schema or an unrelated schema in the header (`none` + `irrelevant`), with conservative targets that use only names visible in the text |
| Replay | 12.0% of the tokens are permissive JavaScript/TypeScript from npm packages (MIT, Apache-2.0, BSD, ISC), used to limit forgetting of generic code |
| Data origin | Synthetic: fictitious schemas (110 schemas in 14 domains from the previous version plus 61 synthetic v3 training schemas) and queries generated from templates, a grammar and an LLM. No personal data by construction. No human review of the training samples was done |
| Decontamination | The v3 corpus is the first training run decontaminated against the sealed homologation set (n-gram check; the previous checkpoint was trained without it, so its homologation numbers may be slightly optimistic) |
| Hardware | One AMD Radeon RX 7800 XT (16 GB), PyTorch ROCm, transformers 4.57.6 |
| Training time | 15.63 hours of training time |
| Runs | A single training run (seed 20261018); no repetition with other seeds |
| Output | Full model in bf16, safetensors |

The hyper-parameters are those of the previous run; the v3 change is the corpus. The previous internal checkpoint (v1, not released) was trained on 229,834 examples
of the distilled corpus only and appears below as a comparison row. The training set is not published.

## Evaluation

**Internal evaluation on synthetic data; no blind human-written set.** The first two tables below come from one internal evaluation: 2,000 examples of the
validation split of the same synthetic dataset (stratified by cut kind and domain, 12 domains, seed 20261008, schemas disjoint from training), prompt in the
`editor-context-v1` format with a 2048-token context, greedy decoding, up to 96 new tokens, bf16 on the RX 7800 XT. 600 examples of this validation split were also used to monitor
the validation loss during training. They describe behaviour on synthetic, template-generated queries and are **not** an estimate of quality on real queries.

Metric definitions:
- *FIM accuracy*: the generated middle equals the reference after whitespace normalization.
- *Exec-valid*: the completion, put back in the query, parses and runs read-only against a local MongoDB (2 s limit). It means "runs", not "answers the intended question".
- *Ceiling*: only 0.6255 of the **reference** completions are accepted by this validator (its Console parser covers a subset of the dialect), so *Exec ÷ ceiling* shows how close the model is to what the validator can accept.
- *Nonexistent fields*: share of suggestions that use a field name absent from the collection schema. `full` = the header lists the collection's names and result fields (n=1647); `irrelevant` = the header describes another collection (n=198); `none` = no header (n=155).
- *Generic-JS*: FIM accuracy on 500 cuts of permissive npm JavaScript from packages disjoint from the replay training packages; a regression check, not a goal.

<!-- BEGIN:accuracy-table -->
| Model | FIM acc | Exec-valid | Exec ÷ ceiling | Syntax-valid | Generic-JS FIM acc (n=500) |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-Coder-1.5B, zero-shot base | 0.142 | 0.316 | 0.5052 | 0.667 | 0.406 |
| Previous internal checkpoint, v1 (not released) | 0.4105 | 0.595 | 0.9512 | 0.9955 | 0.632 |
| **KapiCoder-Mongo-Small** (this model, v3) | 0.410 | 0.6015 | 0.9616 | 0.998 | 0.606 |

Internal evaluation, synthetic data, n=2,000 (Generic-JS: n=500). Validator ceiling for exec-valid: 0.6255.
<!-- END:accuracy-table -->

<!-- BEGIN:fields-table -->
| Nonexistent fields (%) by `context_kind` | `full` (n=1647) | `irrelevant` (n=198) | `none` (n=155) |
|---|---:|---:|---:|
| Qwen2.5-Coder-1.5B, zero-shot base | 3.76 | 10.61 | 12.26 |
| Previous internal checkpoint, v1 (not released) | 1.58 | 7.58 | 10.97 |
| **KapiCoder-Mongo-Small** (this model, v3) | 1.21 | 5.56 | 7.10 |
<!-- END:fields-table -->

Reading the table:
- Relative to the zero-shot base model, FIM accuracy, exec-valid and syntax-valid rise strongly on this synthetic set. Between the previous internal checkpoint and this model the validation numbers are close (see the table); the v3 corpus was built to change the behaviour on unseen schemas, which the next section measures.
- The base model produces mostly unusable output on this task (exec-valid 0.316 overall), so rates of nonexistent fields are not directly comparable with it.
- On this validation set the nonexistent-field rate is higher without a visible schema (`irrelevant`, `none`) than with it; the suggestion quality depends on the host sending field names.
- The 7B teacher scored 0.41 FIM accuracy on the same validation set in the previous version's evaluation; this model is a student of that data, not a replacement for it.

### Out-of-distribution validation (homologation v1)

**Internal evaluation, synthetic out-of-distribution homologation. The queries were not written by people.** The set has 6,030 fill-in-the-middle cuts from 28 schemas (telecom and adtech) that were never used in training, sealed as one set (SHA-256 starting with `aca0aae5`). The queries were written by Qwen3-14B and by a grammar, and Qwen is the same model family as the base models, so the set measures unseen schemas and unseen phrasing, not real use. **The set is not fully independent of the v3 training data**: the v3 corpus includes queries from the same LLM generator and grammar templates over other schemas (training cuts that shared long n-grams with the set were removed from the corpus by decontamination, and the 28 schemas were never in training, but the style was not removed), so the gain on LLM-written queries probably overstates the gain on queries written by people. No evaluation on human-written queries has been run: quality on real queries is still unmeasured. Protocol: the same `editor-context-v1` prompt, greedy decoding, up to 96 new tokens, 2048-token context, bf16 on the RX 7800 XT; exec-valid runs the completed text in mongosh behind a read-only proxy (no write command was forwarded). All models were run on the same 6,030 items. Intervals: Wilson for FIM accuracy, an item bootstrap for exec ÷ ceiling; paired comparisons use McNemar and a paired bootstrap with Holm adjustment (95% CI).

<!-- BEGIN:homolog-table -->
| Model | FIM acc (%) [95% CI] | Exec-valid (%) | Exec ÷ ceiling (%) [95% CI] | Syntax-valid (%) |
|---|---|---:|---|---:|
| Qwen2.5-Coder-1.5B, zero-shot base | 18.74 [17.78, 19.74] | 49.15 | 52.81 [51.51, 54.09] | 67.74 |
| Previous internal checkpoint, v1 (not released) | 36.97 [35.76, 38.20] | 78.59 | 84.44 [83.42, 85.40] | 99.02 |
| **KapiCoder-Mongo-Small** (this model, v3) | 45.37 [44.12, 46.63] | 83.07 | 89.26 [88.39, 90.07] | 99.72 |

Internal evaluation, synthetic out-of-distribution homologation, n=6,030. Validator ceiling for exec-valid: 93.07%.
<!-- END:homolog-table -->

<!-- BEGIN:homolog-fields-table -->
| Nonexistent fields (%) by `context_kind`, homologation | `full` (n=3,720) | `irrelevant` (n=1,155) | `none` (n=1,155) |
|---|---:|---:|---:|
| Qwen2.5-Coder-1.5B, zero-shot base | 7.15 | 9.61 | 10.04 |
| Previous internal checkpoint, v1 (not released) | 9.19 | 11.34 | 11.60 |
| **KapiCoder-Mongo-Small** (this model, v3) | 7.50 | 8.23 | 8.74 |
<!-- END:homolog-fields-table -->

<!-- BEGIN:homolog-gap-table -->
| Model | FIM acc (%): validation → homologation (Δ pp) | Exec ÷ ceiling (%): validation → homologation (Δ pp) | Nonexistent fields `full` (%): validation → homologation (Δ pp) |
|---|---|---|---|
| Qwen2.5-Coder-1.5B, zero-shot base | 14.20 → 18.74 (+4.54) | 50.52 → 52.81 (+2.29) | 3.76 → 7.15 (+3.39) |
| Previous internal checkpoint, v1 (not released) | 41.05 → 36.97 (-4.08) | 95.12 → 84.44 (-10.68) | 1.58 → 9.19 (+7.61) |
| **KapiCoder-Mongo-Small** (this model, v3) | 41.00 → 45.37 (+4.37) | 96.16 → 89.26 (-6.90) | 1.21 → 7.50 (+6.29) |

The exec ÷ ceiling column compares two validators (validation: ceiling 62.55%, homologation: ceiling 93.07%); read trends, not decimals.
<!-- END:homolog-gap-table -->

<!-- BEGIN:homolog-reading -->
Reading the homologation tables:

- **Generalisation gap.** From the synthetic validation set to unseen schemas, FIM accuracy goes 41.00% → 45.37% (+4.37 pp) and exec ÷ ceiling 96.16% → 89.26% (-6.90 pp). For the previous internal checkpoint (v1) the same step was 41.05% → 36.97% in FIM accuracy and 95.12% → 84.44% in exec ÷ ceiling. FIM accuracy is higher here than on the validation set, which is not evidence of better generalisation: see the next sections: the homologation is not fully independent of the v3 training data. The zero-shot base model goes 14.20% → 18.74%, so the homologation queries are easier for a model that was not trained here (interpretation, not tested).
- **Field names are the main risk. Do not trust field names unless the schema is in the context, and validate a suggestion before running it.** On schemas never seen in training this model still invents field names at a rate close to its base (paired difference in items without a nonexistent field, `full`: -0.3 pp [-1.4, +0.7], Holm p=1.000; the interval includes zero, and no equivalence test was run): 7.50% of suggestions with the schema visible (`full`) against 7.15% for the zero-shot base and 9.19% for the previous internal checkpoint (on the synthetic validation set it was 1.21%). Without the schema the rates are 8.23% (`irrelevant`) and 8.74% (`none`), against 9.61% and 10.04% for the base. Having the schema in the header did not lower the rate here (7.50% with it, 8.74% without); why is a hypothesis that was not tested. The rate is counted only on suggestions whose command reached the server, and the base model more often fails to parse (syntax-valid 67.74% against 99.72%), so the base rates cover fewer suggestions and are not directly comparable.
- **Against the base and the previous checkpoint** (same 6,030 items, paired): over the zero-shot base, FIM accuracy +26.6 pp [+25.4, +27.8] and exec-valid +33.9 pp [+32.6, +35.3]; over the previous internal checkpoint, FIM accuracy +8.4 pp [+7.6, +9.3], exec-valid +4.5 pp [+3.7, +5.3] and items without a nonexistent field in `full` +1.7 pp [+0.8, +2.6] (95% CIs, Holm-adjusted McNemar tests). The previous checkpoint is not released.
- **`grammar` against `llm`.** The gain over the zero-shot base in FIM accuracy is +25.7 pp [+24.2, +27.3] on `grammar` and +27.7 pp [+25.9, +29.4] on `llm`; exec ÷ ceiling is 91.93% on `grammar` and 86.58% on `llm` (previous checkpoint: 90.76% and 78.15%). The gain of the v3 model over the previous checkpoint in FIM accuracy is +0.7 pp [-0.0, +1.6] on `grammar` and +17.2 pp [+15.7, +18.7] on `llm`. Both parts of the set resemble what the v3 corpus was built from (queries from the same LLM and grammar templates over other schemas), so these are not independent tests of style; the gain on LLM-written queries probably overstates the gain on queries written by people.
- **Long cuts.** FIM accuracy on `stage` and `multiline` cuts is 10.56% and 11.25% (exact match against a long target within 96 new tokens); do not read the overall number as evenly spread over cut kinds.
<!-- END:homolog-reading -->

Limits of this evaluation: the authors are in the same family as the models (see above); the homologation is not fully independent of the v3 training data (see above); exec-valid means "runs", not "does what was intended"; the validator ceiling is 93.07% because 418 items refer to helpers (`ConnectionPool`, `getConnection`) that mongosh does not define; the smallest `cut_kind` has n=269 and the confidence intervals of the group tables are not shown; there is no multiplicity correction between groups; it is a single greedy run, with no variance across training runs.

## Limitations and risks

- **Synthetic data only.** Schemas are fictitious; the evaluation split uses unseen schemas of the same generator and templates. Expect lower scores on real collections, naming styles and query shapes.
- **Invented field names.** It can propose fields, collections or operators that do not exist, and on unseen schemas the rate is not below the zero-shot base's in every setting (homologation section above, rates by `context_kind`). Validate field names before running a suggestion; do not rely on them unless the schema is in the context.
- **The homologation numbers are not an estimate for real use**: the set is LLM- and grammar-written, not human-written, and is not fully independent of the v3 training data.
- **A single training run per model** (one seed): differences between this model and the previous checkpoint are not backed by repeated runs.
- **Dialects.** Trained on Mongo Console JavaScript (`ConnectionPool...` style) and JSON pipelines only. `mongosh` style prompts, other languages and chat are untested.
- **No safety guarantee.** Behaviour around destructive operations (for example deletes or drops) was neither trained nor tested; during evaluation only read commands were executed and a guard blocked writes (0 write executions). The host application must keep a human in the loop and must not run suggestions blindly.
- **Exec-valid is not correctness.** A validator-accepted completion may still not do what the user wanted; the validator itself accepts only 0.6255 of the references on the validation set.
- **Autocomplete scope.** Short continuations (the IDE caps them at 32 tokens); long stage or multi-line targets are rarely matched exactly.
- Not evaluated: bias or toxicity, long-context behaviour beyond 2048 tokens, multi-turn use, adversarial prompts, latency of the safetensors weights.

## Formats

| Format | Where | Notes |
|---|---|---|
| safetensors, bf16 | repository root | Reference weights, loaded with `transformers`. |

This repository contains only the safetensors weights, the tokenizer and metadata (`provenance/`, `evaluation/`, `examples/`). ONNX builds for the editor are not part of it; they are planned as a separate repository (`esilva/KapiCoder-Mongo-Small-ONNX`) and are not published yet. No GGUF is provided. Neither the teacher model nor the training data are part of this repository.

## License, modification notice and attribution

- License: **Apache License 2.0** (see `LICENSE`).
- **Modification notice:** this model is a **modified** version of `Qwen/Qwen2.5-Coder-1.5B` (revision above). Changes: LoRA fine-tuning (r=64) merged into the weights for MongoDB fill-in-the-middle autocomplete, with part of the labels distilled from a 7B teacher; export to safetensors bf16. It is not a product of Alibaba Cloud or the Qwen team and is not endorsed by them. See `NOTICE.md`.
- Attribution: base model **Qwen2.5-Coder** by the Qwen team (Hui et al., *Qwen2.5-Coder Technical Report*, arXiv:2409.12186), Apache-2.0. The teacher was derived from Qwen2.5-Coder-7B (Apache-2.0); its weights are not part of this repository. Part of the training queries was written by Qwen3-14B (Apache-2.0).
- The generic-code replay used npm packages under MIT, Apache-2.0, BSD-2/3-Clause and ISC licenses. The code itself is not redistributed; the package names, versions and licenses are listed in `REPLAY_ATTRIBUTION.md`.
- The `editor-context-v1` prompt format is documented above so that others can build prompts for this model. Its use is not restricted by this card beyond the model license.

Related model: [KapiCoder-Mongo-Nano](https://huggingface.co/esilva/KapiCoder-Mongo-Nano) (0.5B), same task and contract.

---

## Resumo em português (pt-BR)

**Status: PREVIEW / experimental, preview público.** KapiCoder-Mongo-Small é um modelo pequeno (cerca de 1,5 B de parâmetros) ajustado a partir do `Qwen/Qwen2.5-Coder-1.5B`
para **completar consultas MongoDB no cursor** (preenchimento no meio, FIM) no editor Kapibara Studio. Esta é a versão **v3** dos pesos.

- **O que é:** autocomplete. Sugere texto; **nunca executa consultas**. Não é chat nem agente, e não valida o que sugere.
- **Uso:** prompt FIM do Qwen (`<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`) com o cabeçalho de contexto `editor-context-v1` (veja o exemplo em inglês acima, que roda sem a IDE, também em `examples/example_fim.py`).
- **Treino:** LoRA (r=64, alpha=128), mesclado nos pesos bf16, 1,5 época (o plano de 2 épocas foi cortado). 309.571 exemplos sintéticos (292.812 de MongoDB e 16.759 de replay de código JS/TS permissivo, 12,0% dos tokens), 86.000.018 tokens por época, 129.085.790 tokens vistos, 4.617 passos, 15,63 horas em uma RX 7800 XT. Os rótulos vêm em parte de um professor de 7B (mantidos só quando passam por validação de execução somente leitura), em parte de consultas escritas pelo Qwen3-14B e de uma gramática (rótulo de referência). Sem dados pessoais por construção; não houve revisão humana das amostras. **Uma única execução de treino (uma semente).**
- **Avaliação:** **avaliação interna sobre dados sintéticos; sem conjunto cego humano.** As tabelas acima trazem os números reais (FIM acc, exec-valid, exec ÷ teto, campos inexistentes por `context_kind`, Generic-JS, base × modelo × checkpoint interno anterior não publicado). A homologação sintética fora da distribuição está no item seguinte; nada foi medido com consultas escritas por humanos.
<!-- BEGIN:homolog-pt -->
- **Homologação (avaliação interna, homologação sintética fora da distribuição):** 6.030 cortes de 28 schemas de telecom e adtech nunca vistos no treino, escritos pelo Qwen3-14B e por gramática, **não por humanos** (mesma família Qwen dos modelos). **A homologação não é totalmente independente do treino v3** (o corpus inclui consultas do mesmo gerador LLM e templates de gramática de outros schemas), então o ganho em consultas escritas pelo LLM provavelmente superestima o ganho em consultas humanas. FIM acc 45,37% (validação sintética: 41,00%; base: 18,74%; checkpoint interno anterior, não publicado: 36,97%); exec ÷ teto 89,26% (validação: 96,16%, com validadores diferentes). **Alerta:** com o schema visível (`full`), 7,50% das sugestões usam campos inexistentes, contra 7,15% da base zero-shot (validação sintética: 1,21%); sem schema, 8,23% (`irrelevant`) e 8,74% (`none`). Valide os nomes de campo antes de executar e não confie neles sem o schema no contexto. Executar não é acertar a intenção.
<!-- END:homolog-pt -->
- **Limites:** só dado sintético; em schemas novos há campos inexistentes mesmo com o schema visível (veja a homologação), e no conjunto de validação a taxa é maior sem schema visível; só dialetos Console e JSON; sem garantia de segurança. Valide os nomes de campo antes de executar.
- **Formatos:** apenas safetensors bf16 neste repositório; o ONNX para a IDE será um repositório separado, ainda não publicado.
- **Licença:** Apache-2.0, com **aviso de modificação**: modelo derivado e modificado do Qwen2.5-Coder-1.5B (equipe Qwen, Alibaba Cloud), sem endosso deles. O replay usa pacotes npm MIT/Apache-2.0/BSD/ISC, listados em `REPLAY_ATTRIBUTION.md`.
