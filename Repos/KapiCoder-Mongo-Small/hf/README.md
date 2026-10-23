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
- onnx
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
      value: 0.4105
      verified: false
    - type: exec_valid
      name: Exec-valid (completion parses and runs read-only)
      value: 0.595
      verified: false
    - type: exec_valid_vs_ceiling
      name: Exec-valid divided by validator ceiling (0.6255)
      value: 0.9512
      verified: false
    - type: syntax_valid
      name: Syntax-valid
      value: 0.9955
      verified: false
    - type: nonexistent_fields_full_pct
      name: Nonexistent fields, schema visible (context_kind=full), percent
      value: 1.58
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
      value: 0.632
      verified: false
---

# KapiCoder-Mongo-Small (1.5B)

> **Status: PREVIEW / experimental.** Published for inspection and feedback, not for production use.
> **Internal evaluation on synthetic data; no blind human-written set.** Read [Limitations and risks](#limitations-and-risks) before using it.

KapiCoder-Mongo-Small is a small code-completion model (about 1.5 B parameters) fine-tuned from
[`Qwen/Qwen2.5-Coder-1.5B`](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B) to **complete MongoDB queries at the cursor**
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

MODEL_ID = sys.argv[1] if len(sys.argv) > 1 else "esilva/KapiCoder-Mongo-Small"

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
| Base model | `Qwen/Qwen2.5-Coder-1.5B`, revision `df3ce67c0e24480f20468b6ef2894622d69eb73b` (Apache-2.0) |
| Method | LoRA (r=64, alpha=128, dropout 0.05 on the q, k, v, o, gate, up and down projections; 73,859,072 trainable parameters) on the bf16 base, merged into full bf16 weights afterwards. AdamW, peak LR 1.5e-4, 3,422 steps of about 32.8 k tokens = 1.5 epochs (a 2-epoch plan was cut to 1.5 epochs to save time; the learning-rate schedule was recomputed for the shorter run) |
| Training data | 229,834 examples (217,306 MongoDB examples + 12,528 generic-code replay examples), 63,702,244 tokens per epoch; 95,649,450 tokens seen in total |
| Replay | 12.12% of the tokens are permissive JavaScript/TypeScript from npm packages (MIT, Apache-2.0, BSD, ISC), used to limit forgetting of generic code |
| Labels | Targets written by a 7B teacher (`KapiTeacher-Mongo-7B`, a QLoRA fine-tune of Qwen2.5-Coder-7B) and kept only when they pass an execution check against a real MongoDB with read-only commands; part of the examples keep the original reference as target |
| Data origin | Synthetic: fictitious schemas and queries generated from templates and fixtures. No personal data by construction. No human review of the training samples was done |
| Hardware | One AMD Radeon RX 7800 XT (16 GB), PyTorch ROCm, transformers 4.57.6 |
| Training time | 11.74 hours of training time |
| Output | Full model in bf16, safetensors |

The teacher label differs from the original reference in 54.11% of the MongoDB examples. The training set is not published.

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
| Qwen2.5-Coder-1.5B, zero-shot base | 0.142 | 0.316 | 0.5052 | 0.667 | 0.406 |
| **KapiCoder-Mongo-Small** (this model) | 0.4105 | 0.595 | 0.9512 | 0.9955 | 0.632 |

Internal evaluation, synthetic data, n=2,000 (Generic-JS: n=500). Validator ceiling for exec-valid: 0.6255.
<!-- END:accuracy-table -->

<!-- BEGIN:fields-table -->
| Nonexistent fields (%) by `context_kind` | `full` (n=1647) | `irrelevant` (n=198) | `none` (n=155) |
|---|---:|---:|---:|
| Qwen2.5-Coder-1.5B, zero-shot base | 3.76 | 10.61 | 12.26 |
| **KapiCoder-Mongo-Small** (this model) | 1.58 | 7.58 | 10.97 |
<!-- END:fields-table -->

Reading the table:
- Relative to the zero-shot base model, FIM accuracy, exec-valid and syntax-valid rise strongly on this synthetic set; the paired comparison is in the project's internal reports.
- **The effect of distillation at this size was not measured.** No control trained on the original reference labels exists for this model. In the 0.5B sibling, where that control was trained, the distilled model gained only a little: FIM accuracy 0.403 against 0.4115 for the control (difference -0.85 percentage points, not significant) and exec-valid 1.7 points higher (Holm-adjusted p=0.019). The 7B teacher scored 0.41 FIM accuracy on the same set, so the students match it here; the gate that required the teacher to beat the best student by 5 points failed (-0.15 points). The teacher's role became filtering and validation of labels.
- **Compared with the 0.5B sibling** (same 2,000 examples): FIM accuracy is 0.75 points higher (not significant) and exec-valid 1.3 points higher (p=0.0016). The two are close on FIM accuracy; the 1.5B model has the higher exec-valid and fewer nonexistent fields with a visible schema (1.58% against 2.06%).
- Without a visible schema (`irrelevant`, `none`) the nonexistent-field rate stays at 7.58% and 10.97%; training did not fix this. The suggestion quality depends on the host sending field names.

### Out-of-distribution validation

**TBD.** No evaluation on real, human-written queries or on a held-out set outside the synthetic generator has been run. A human-written blind set was planned and then dropped, so generalisation to real queries is unmeasured. This section will be filled in only with results from such an evaluation.

## Limitations and risks

- **Synthetic data only.** Schemas are fictitious (110 schemas in 14 domains; the evaluation split uses 12 domains and unseen schemas, but the same generator and templates). Expect lower scores on real collections, naming styles and query shapes.
- **It behaves like autocomplete.** It can propose fields, collections or operators that do not exist. Rates above are for the synthetic set; without a visible schema they were 7.58% (`irrelevant`) and 10.97% (`none`).
- **Dialects.** Trained on Mongo Console JavaScript (`ConnectionPool...` style) and JSON pipelines only. `mongosh` style prompts, other languages and chat are untested.
- **No safety guarantee.** Behaviour around destructive operations (for example deletes or drops) was neither trained nor tested; during evaluation only read commands were executed and a guard blocked writes (0 write executions). The host application must keep a human in the loop and must not run suggestions blindly.
- **Exec-valid is not correctness.** A validator-accepted completion may still not do what the user wanted; the validator itself accepts only 0.6255 of the references.
- **Small gain from distillation** (see above): do not read the use of a 7B teacher as a quality guarantee.
- **Quantised ONNX variants** can lose quality at this size: the INT4 builds lost 1.4 to 1.8 FIM-accuracy points, so they are not offered here.
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
| `onnx/dml-fp16` | 2,964 MB | GPU (DirectML) | FIM accuracy 0.4 points below the bf16 weights (not significant, p=0.50), exec-valid identical; greedy output identical in 96.5% of prompts |
| `onnx/int8` | 2,578 MB | CPU | FIM accuracy 0.4 points below the bf16 weights (not significant, p=0.50), exec-valid identical; greedy output identical in 94.5% of prompts |

The INT4 builds (1,090 MB and 952 MB for CPU, 1,293 MB for DirectML) are not offered: FIM accuracy fell by 1.4 to 1.8 points (p between 0.004 and 0.04) against the bf16 weights.

- The `kapibarastudio-model.json` and `kapicoder_manifest.json` files in each ONNX folder are metadata for the Kapibara Studio application (schema `kapilab-kapi-model-v1`); you can ignore them.
- **EOS problem in onnxruntime-genai 0.15.2.** The step that generates `<|endoftext|>` does not append that token to the sequence, so code that reads the last element of `GetSequence(0)` repeats the previous token (for example `1` becomes `11`). Read `get_next_tokens()` or check `is_done()` first. The packages are published as exported, without a workaround.
- Latency was measured only as a proxy, not by the official harness, on prompts of about 270 tokens: time to first token was 148 ms on average and 170 ms at p95 with `dml-fp16` (104 tokens/s), and 697 ms on average and 1192 ms at p95 on CPU with `int8`. The CPU figure is above the internal target of 500/650 ms (that target was defined on a chat metric, so the comparison is imperfect).

## License, modification notice and attribution

- License: **Apache License 2.0** (see `LICENSE`).
- **Modification notice:** this model is a **modified** version of `Qwen/Qwen2.5-Coder-1.5B` (revision above). Changes: LoRA fine-tuning merged into the weights, for MongoDB fill-in-the-middle autocomplete, with labels distilled from a 7B teacher; export to safetensors bf16 and, optionally, to ONNX. It is not a product of Alibaba Cloud or the Qwen team and is not endorsed by them. See `NOTICE.md`.
- Attribution: base model **Qwen2.5-Coder** by the Qwen team (Hui et al., *Qwen2.5-Coder Technical Report*, arXiv:2409.12186), Apache-2.0. The teacher was derived from Qwen2.5-Coder-7B (Apache-2.0); its weights are not part of this repository.
- The generic-code replay used npm packages under MIT, Apache-2.0, BSD-2/3-Clause and ISC licenses. The code itself is not redistributed; the package names, versions and licenses are listed in `REPLAY_ATTRIBUTION.md`.
- The `editor-context-v1` prompt format is documented above so that others can build prompts for this model. Its use is not restricted by this card beyond the model license.

Related model: [KapiCoder-Mongo-Nano](https://huggingface.co/esilva/KapiCoder-Mongo-Nano) (0.5B), same task and contract.

---

## Resumo em português (pt-BR)

**Status: PREVIEW / experimental.** KapiCoder-Mongo-Small é um modelo pequeno (cerca de 1,5 B de parâmetros) ajustado a partir do `Qwen/Qwen2.5-Coder-1.5B`
para **completar consultas MongoDB no cursor** (preenchimento no meio, FIM) no editor Kapibara Studio.

- **O que é:** autocomplete. Sugere texto; **nunca executa consultas**. Não é chat nem agente, e não valida o que sugere.
- **Uso:** prompt FIM do Qwen (`<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>`) com o cabeçalho de contexto `editor-context-v1` (veja o exemplo em inglês acima, que roda sem a IDE).
- **Treino:** LoRA r=64 (alpha 128) mesclado em bf16, 229.834 exemplos sintéticos (217.306 de MongoDB e 12.528 de replay de código JS/TS permissivo, 12,12% dos tokens), 63.702.244 tokens por época, 1,5 época (3.422 passos, 95.649.450 tokens vistos), 11,74 horas em uma RX 7800 XT. Os rótulos vêm de um professor de 7B, mantidos só quando passam por validação de execução somente leitura. Sem dados pessoais por construção; não houve revisão humana das amostras.
- **Avaliação:** **avaliação interna sobre dados sintéticos; sem conjunto cego humano.** As tabelas acima trazem os números reais (FIM acc, exec-valid, exec ÷ teto, campos inexistentes por `context_kind`, Generic-JS, base × modelo). A homologação fora da distribuição está como **TBD** e não foi feita.
- **Limites:** só dado sintético; sem schema visível, 7,58% (`irrelevant`) e 10,97% (`none`) das sugestões usam campos inexistentes; só dialetos Console e JSON; sem garantia de segurança. O efeito da destilação neste tamanho não foi medido (não há controle com rótulos originais); no irmão de 0,5B foi pequeno. Frente ao 0,5B, o ganho é de 0,75 p.p. em FIM acc (não significativo) e 1,3 p.p. em exec-valid.
- **Formatos:** safetensors bf16 e, opcionalmente, ONNX (`onnxruntime-genai` 0.15.2; variantes `dml-fp16` e `int8`; as INT4 perdem 1,4 a 1,8 p.p. e não são oferecidas). O ORT GenAI 0.15.2 tem um problema de EOS que repete o último token se a aplicação ler `GetSequence(0)[^1]`; veja a seção em inglês.
- **Licença:** Apache-2.0, com **aviso de modificação**: modelo derivado e modificado do Qwen2.5-Coder-1.5B (equipe Qwen, Alibaba Cloud), sem endosso deles. O replay usa pacotes npm MIT/Apache-2.0/BSD/ISC, listados em `REPLAY_ATTRIBUTION.md`.
