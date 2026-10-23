---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-0.5B
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
- onnx
- qwen2
- synthetic-data
model-index:
- name: KapiCoder-Mongo-Nano
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
      value: 0.403
      verified: false
    - type: exec_valid
      name: Exec-valid (completion parses and runs read-only)
      value: 0.582
      verified: false
    - type: exec_valid_vs_ceiling
      name: Exec-valid divided by validator ceiling (0.6255)
      value: 0.9305
      verified: false
    - type: syntax_valid
      name: Syntax-valid
      value: 0.995
      verified: false
    - type: nonexistent_fields_full_pct
      name: Nonexistent fields, schema visible (context_kind=full), percent
      value: 2.06
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
      value: 0.518
      verified: false
---

# KapiCoder-Mongo-Nano (0.5B)

> **Status: PREVIEW / experimental.** Published for inspection and feedback, not for production use.
> **Internal evaluation on synthetic data; no blind human-written set.** Read [Limitations and risks](#limitations-and-risks) before using it.

KapiCoder-Mongo-Nano is a small code-completion model (about 494 M parameters) fine-tuned from
[`Qwen/Qwen2.5-Coder-0.5B`](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B) to **complete MongoDB queries at the cursor**
(fill-in-the-middle, FIM). It was built for the *Kapibara Studio* editor, which sends it the text before and after the cursor plus a short
context header (the `editor-context-v1` contract described below).

## What it is, and what it is not

**It is** an autocomplete model. Given the text before and after the cursor, it proposes a short continuation (the IDE caps it at 32 tokens)
for MongoDB Console JavaScript (`ConnectionPool.<conn>.<db>.<collection>.find(...)`, aggregation pipelines, `$search` stages, index definitions)
and JSON pipelines.

**It is not**
- a chat model or an agent: it does not follow instructions, hold a conversation, call tools or explain code;
- a query executor: it **only suggests text and never runs anything**. Whatever executes a suggestion is the host application;
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

```python
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "esilva/KapiCoder-Mongo-Nano"

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

## Training

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-0.5B`, revision `8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301` (Apache-2.0) |
| Method | Full fine-tuning (fp32 master weights, bf16 autocast), AdamW, peak LR 2e-5, 2 epochs, 3,168 steps of about 32.8 k tokens |
| Training data | 159,865 examples (151,180 MongoDB examples + 8,685 generic-code replay examples), 44,164,423 tokens per epoch; 88,304,300 tokens seen in total |
| Replay | 12.12% of the tokens are permissive JavaScript/TypeScript from npm packages (MIT, Apache-2.0, BSD, ISC), used to limit forgetting of generic code |
| Labels | Targets written by a 7B teacher (`KapiTeacher-Mongo-7B`, a QLoRA fine-tune of Qwen2.5-Coder-7B) and kept only when they pass an execution check against a real MongoDB with read-only commands; part of the examples keep the original reference as target |
| Data origin | Synthetic: fictitious schemas and queries generated from templates and fixtures. No personal data by construction. No human review of the training samples was done |
| Hardware | One AMD Radeon RX 7800 XT (16 GB), PyTorch ROCm, transformers 4.57.6 |
| Training time | 3.33 hours of training time |
| Output | Full model in bf16, safetensors |

The teacher label differs from the original reference in 54.27% of the MongoDB examples. The training set is not published.

## Evaluation

**Internal evaluation on synthetic data; no blind human-written set.** All numbers below come from one internal evaluation: 2,000 examples of the
validation split of the same synthetic dataset (stratified by cut kind and domain, seed 20261008, schemas disjoint from training), prompt in the
`editor-context-v1` format with a 2048-token context, greedy decoding, up to 96 new tokens, bf16 on the RX 7800 XT. They describe behaviour on
synthetic, template-generated queries and are **not** an estimate of quality on real queries.

Metric definitions:
- *FIM accuracy*: the generated middle equals the reference after whitespace normalization.
- *Exec-valid*: the completion, put back in the query, parses and runs read-only against a local MongoDB (2 s limit). It means "runs", not "answers the intended question".
- *Ceiling*: only 0.6255 of the **reference** completions are accepted by this validator (its Console parser covers a subset of the dialect), so *Exec ÷ ceiling* shows how close the model is to what the validator can accept.
- *Nonexistent fields*: share of suggestions that use a field name absent from the collection schema. `full` = the header lists the collection's names and result fields (n=1647); `irrelevant` = the header describes another collection (n=198); `none` = no header (n=155).
- *Generic-JS*: FIM accuracy on 500 cuts of permissive npm JavaScript from packages disjoint from the replay training packages; a regression check, not a goal.

<!-- BEGIN:accuracy-table -->
| Model | FIM acc | Exec-valid | Exec ÷ ceiling | Syntax-valid | Generic-JS FIM acc (n=500) |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-Coder-0.5B, zero-shot base | 0.076 | 0.213 | 0.3405 | 0.498 | 0.340 |
| **KapiCoder-Mongo-Nano** (this model) | 0.403 | 0.582 | 0.9305 | 0.995 | 0.518 |
| 0.5B trained on original labels (control, not released) | 0.4115 | 0.565 | 0.9033 | 0.997 | 0.504 |

Internal evaluation, synthetic data, n=2,000 (Generic-JS: n=500). Validator ceiling for exec-valid: 0.6255.
<!-- END:accuracy-table -->

<!-- BEGIN:fields-table -->
| Nonexistent fields (%) by `context_kind` | `full` (n=1647) | `irrelevant` (n=198) | `none` (n=155) |
|---|---:|---:|---:|
| Qwen2.5-Coder-0.5B, zero-shot base | 4.13 | 8.59 | 7.74 |
| **KapiCoder-Mongo-Nano** (this model) | 2.06 | 11.11 | 12.90 |
| 0.5B trained on original labels (control, not released) | 2.79 | 13.64 | 13.55 |
<!-- END:fields-table -->

Reading the table:
- Relative to the zero-shot base model, FIM accuracy, exec-valid and syntax-valid rise strongly on this synthetic set; the paired comparison is in the project's internal reports.
- In the `irrelevant` and `none` rows this model's nonexistent-field rate (11.11% and 12.90%) is above the zero-shot base model's (8.59% and 7.74%). The base model produces mostly unusable output on this task (exec-valid 0.213 overall), so the rates are not directly comparable, but fine-tuning did not reduce the problem there.
- **Distillation gave a small gain.** The same 0.5B model trained only on the original reference labels (same examples, same hyper-parameters, not released) has FIM accuracy 0.4115 against 0.403 for this model (difference -0.85 percentage points, not significant), while exec-valid is 1.7 points higher for this model (Holm-adjusted p=0.019). The 7B teacher itself scored 0.41 FIM accuracy on the same set, so the 0.5B student matches it here; the gate that required the teacher to beat the best student by 5 points failed (-0.15 points). The teacher's role became filtering and validation of labels.
- Without a visible schema (`irrelevant`, `none`) the nonexistent-field rate stays at 11.11% and 12.90%; training did not fix this. The suggestion quality depends on the host sending field names.

### Out-of-distribution validation

**TBD.** No evaluation on real, human-written queries or on a held-out set outside the synthetic generator has been run. A human-written blind set was planned and then dropped, so generalisation to real queries is unmeasured. This section will be filled in only with results from such an evaluation.

## Limitations and risks

- **Synthetic data only.** Schemas are fictitious (110 schemas in 14 domains; the evaluation split uses 12 domains and unseen schemas, but the same generator and templates). Expect lower scores on real collections, naming styles and query shapes.
- **It behaves like autocomplete.** It can propose fields, collections or operators that do not exist. Rates above are for the synthetic set; without a visible schema they were 11.11% (`irrelevant`) and 12.90% (`none`).
- **Dialects.** Trained on Mongo Console JavaScript (`ConnectionPool...` style) and JSON pipelines only. `mongosh` style prompts, other languages and chat are untested.
- **No safety guarantee.** Behaviour around destructive operations (for example deletes or drops) was neither trained nor tested; during evaluation only read commands were executed and a guard blocked writes (0 write executions). The host application must keep a human in the loop and must not run suggestions blindly.
- **Exec-valid is not correctness.** A validator-accepted completion may still not do what the user wanted; the validator itself accepts only 0.6255 of the references.
- **Small gain from distillation** (see above): do not read the use of a 7B teacher as a quality guarantee.
- **Quantised ONNX variants** can lose quality, mainly in the 1.5B sibling; check the ONNX section before choosing one.
- Latency on CPU depends on the host; see the ONNX section for the proxy numbers that were measured.
- Not evaluated: bias or toxicity, long-context behaviour beyond 2048 tokens, multi-turn use, adversarial prompts.

## Formats

| Format | Where | Notes |
|---|---|---|
| safetensors, bf16 | repository root | Reference weights, loaded with `transformers`. |
| ONNX (optional) | `onnx/<variant>/` | Built with `onnxruntime-genai` 0.15.2, `model.type=qwen2`. Only the variants listed below are in the repository. |

ONNX variants (sizes from the export report; quality measured on 500 validation prompts against the bf16 weights):

| Variant | Size | Target | Note |
|---|---:|---|---|
| `onnx/int4` | 392 MB | CPU | FIM accuracy and exec-valid within 0.8 points of the bf16 weights, none significant (p >= 0.34); greedy output identical in 79.0% of prompts |
| `onnx/dml-fp16` | 959 MB | GPU (DirectML) | Greedy output identical in 97.5% of prompts |

- The `kapibarastudio-model.json` and `kapicoder_manifest.json` files in each ONNX folder are metadata for the Kapibara Studio application (schema `kapilab-kapi-model-v1`); you can ignore them.
- **EOS problem in onnxruntime-genai 0.15.2.** The step that generates `<|endoftext|>` does not append that token to the sequence, so code that reads the last element of `GetSequence(0)` repeats the previous token (for example `1` becomes `11`). Read `get_next_tokens()` or check `is_done()` first. The packages are published as exported, without a workaround.
- CPU latency was measured only as a proxy, not by the official harness: time to first token with the `int4` package was 285 ms on average and 480 ms at p95 for prompts of about 270 tokens, which is above the internal target of 150/250 ms. The `dml-fp16` package measured 109/124 ms.

## License, modification notice and attribution

- License: **Apache License 2.0** (see `LICENSE`).
- **Modification notice:** this model is a **modified** version of `Qwen/Qwen2.5-Coder-0.5B` (revision above). Changes: full fine-tuning for MongoDB fill-in-the-middle autocomplete, with labels distilled from a 7B teacher; export to safetensors bf16 and, optionally, to ONNX. It is not a product of Alibaba Cloud or the Qwen team and is not endorsed by them. See `NOTICE.md`.
- Attribution: base model **Qwen2.5-Coder** by the Qwen team (Hui et al., *Qwen2.5-Coder Technical Report*, arXiv:2409.12186), Apache-2.0. The teacher was derived from Qwen2.5-Coder-7B (Apache-2.0); its weights are not part of this repository.
- The generic-code replay used npm packages under MIT, Apache-2.0, BSD-2/3-Clause and ISC licenses. The code itself is not redistributed; the package names, versions and licenses are listed in `REPLAY_ATTRIBUTION.md`.
- The `editor-context-v1` prompt format is documented above so that others can build prompts for this model. Its use is not restricted by this card beyond the model license.

Related model: [KapiCoder-Mongo-Small](https://huggingface.co/esilva/KapiCoder-Mongo-Small) (1.5B), same task and contract.

---

## Resumo em português (pt-BR)

**Status: PREVIEW / experimental.** KapiCoder-Mongo-Nano é um modelo pequeno (cerca de 494 M de parâmetros) ajustado a partir do `Qwen/Qwen2.5-Coder-0.5B`
para **completar consultas MongoDB no cursor** (preenchimento no meio, FIM) no editor Kapibara Studio.

- **O que é:** autocomplete. Sugere texto; **nunca executa consultas**. Não é chat nem agente, e não valida o que sugere.
- **Uso:** prompt FIM do Qwen (`<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`) com o cabeçalho de contexto `editor-context-v1` (veja o exemplo em inglês acima, que roda sem a IDE).
- **Treino:** ajuste completo (full fine-tuning), 159.865 exemplos sintéticos (151.180 de MongoDB e 8.685 de replay de código JS/TS permissivo, 12,12% dos tokens), 44.164.423 tokens por época, 2 épocas (88.304.300 tokens vistos), 3,33 horas em uma RX 7800 XT. Os rótulos vêm de um professor de 7B, mantidos só quando passam por validação de execução somente leitura. Sem dados pessoais por construção; não houve revisão humana das amostras.
- **Avaliação:** **avaliação interna sobre dados sintéticos; sem conjunto cego humano.** As tabelas acima trazem os números reais (FIM acc, exec-valid, exec ÷ teto, campos inexistentes por `context_kind`, Generic-JS, base × modelo). A homologação fora da distribuição está como **TBD** e não foi feita.
- **Limites:** só dado sintético; sem schema visível, 11,11% (`irrelevant`) e 12,90% (`none`) das sugestões usam campos inexistentes; só dialetos Console e JSON; sem garantia de segurança. A destilação trouxe ganho pequeno: o mesmo 0,5B treinado só nos rótulos originais empata em FIM acc, e o professor 7B também.
- **Formatos:** safetensors bf16 e, opcionalmente, ONNX (`onnxruntime-genai` 0.15.2). O ORT GenAI 0.15.2 tem um problema de EOS que repete o último token se a aplicação ler `GetSequence(0)[^1]`; veja a seção em inglês.
- **Licença:** Apache-2.0, com **aviso de modificação**: modelo derivado e modificado do Qwen2.5-Coder-0.5B (equipe Qwen, Alibaba Cloud), sem endosso deles. O replay usa pacotes npm MIT/Apache-2.0/BSD/ISC, listados em `REPLAY_ATTRIBUTION.md`.
