#!/usr/bin/env python3
"""Gera Repos/KapiTeacher-Mongo-7B/reports/gates.md a partir de checkpoint_verification.json e baseline_base/*/summary.json."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import TEACHER_REPO  # noqa: E402

REPORTS = TEACHER_REPO / "reports"
ORDER = ["Qwen2.5-Coder-7B", "Qwen2.5-Coder-1.5B", "Qwen2.5-Coder-0.5B", "deepseek-coder-6.7b-base"]


def pct(v) -> str:
    return "—" if v is None else f"{v * 100:.1f}%"


def main() -> int:
    ver = json.loads((REPORTS / "checkpoint_verification.json").read_text(encoding="utf-8"))
    summaries = {}
    for name in ORDER:
        path = REPORTS / "baseline_base" / name / "summary.json"
        if path.exists():
            summaries[name] = json.loads(path.read_text(encoding="utf-8"))
    comp = ver["g2_1"]["components"]
    integ = ver["integrity"]
    cfgs = ver["config"]["fields"]
    tok = ver["tokenizer"]
    san = ver["fim_sanity"]
    pad = ver["padding_mask"]
    L = ["# Gates do professor — KapiTeacher-Mongo-7B (Fase 02 §3.6)", "",
         f"Gerado em {time.strftime('%F %T')} por `scripts/teacher/write_gates.py`. Professor: Qwen2.5-Coder-7B Base "
         f"(`{integ['repo_id']}@{integ['revision'][:12]}`), decisao em `reports/teacher_choice.json` (14B descartado).", "",
         "## G2.1 Checkpoint — PARCIAL (`pending_ide`)", "",
         "| item | status | evidencia |", "|---|---|---|",
         f"| Hashes = HF (revisao fixada) | **{comp['integrity']}** | {len(integ['files'])} arquivos x `base-repos.lock.json`; "
         f"{integ['hf_compared']} comparados com a API do HF na revisao fixada (LFS sha256 / git blob sha1); sem .bin/pickle |",
         f"| `config.json` = tabela do §2 | **{comp['config']}** | 28 camadas, hidden 3584, 28 heads, 4 KV, vocab_size 152.064, 32.768 pos., rope_theta 1e6, "
         f"embeddings nao atrelados ({'todos conferem' if ver['config']['status'] == 'pass' else 'DIVERGE'}) |",
         f"| IDs FIM/eos e tokenizer | **{comp['tokenizer_ids']}** | fim_prefix 151659, fim_middle 151660, fim_suffix 151661, fim_pad 151662, eos 151643; "
         f"cada marcador = 1 token; `tokenizer.json` identico ao dos students 0.5B/1.5B (sha `{tok['tokenizer_json_sha256'][:12]}`); "
         f"{tok['len_tokenizer']} tokens reais; CRLF round-trip ok |",
         f"| FIM sanity 20 prompts | **{comp['fim_sanity']}** | {san['without_markers']}/{san['prompts']} sem marcadores `<|fim_*|>` vazando "
         f"(NF4, greedy, 64 tokens); {san['stopped_at_eos_or_suffix_echo']}/{san['prompts']} pararam em `<|endoftext|>` ou eco do sufixo, "
         f"os demais atingiram o limite sem vazar marcador |",
         f"| Padding do vocab mascarado | **{comp['padding_mask']}** | lm_head {pad['lm_head_rows']} linhas x {pad['real_vocab']} reais "
         f"({pad['padding_rows']} de padding); logits[:{pad['real_vocab']}] antes do argmax; nenhum ID de padding gerado; "
         f"argmax sem mascara no padding: {san['unmasked_argmax_in_padding_at_prefill']}/20 |",
         "| Paridade de IDs Python x C# (400 prompts) | **pending_ide** | NAO verificada: exige o `QwenFimPromptBuilder` + `OnnxModelTokenizer` da IDE via "
         "`tools/ContractProbe` (.NET), nao executado aqui. IDs esperados do lado Python gravados em `reports/parity_python_ids.jsonl` "
         "(400 prompts, sha256 em `checkpoint_verification.json`) para a comparacao futura. Sem aprovacao fingida. |",
         "| Perplexidade 2k/8k/32k (opcional §3.1) | not_run | fora do escopo desta etapa |", "",
         f"**G2.1 = {ver['g2_1']['status']}**: todos os itens executaveis aqui passaram; falta apenas a paridade Python x C# (depende da IDE).", ""]

    L += ["## Baseline zero-shot FIM no `val` (Fase 02 §3.2)", ""]
    if summaries:
        first = next(iter(summaries.values()))
        L += [f"Amostra: **n={first['n']}** de {first['dataset']['split_size']} exemplos do `val` de `data/generated/v1/fim.jsonl` "
              f"(estratificado por cut_kind x dominio, seed {first['seed']}; sha256 da amostra `{first['dataset']['sample_ids_sha256'][:16]}`, "
              f"dataset `{first['dataset']['sha256'][:16]}`). Contrato `{first['contract']}` (QwenFimPromptBuilder sobre prefix/suffix, sem cabecalho de "
              f"contexto, pois o v1 nao traz `context`), greedy, max {first['max_new_tokens']} tokens, contexto {first['context_tokens']}. "
              "FIM puro: chat/blind/Generic-JS nao avaliados nesta etapa.", "",
              "| modelo | carga | n | EM | FIM acc. (norm.) | chars aceitos | syntax | mongo_valid (estatico) | **exec-valid** | campos inex. | exec-valid @32 | GPU (min) |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, s in summaries.items():
            h = s["headline"]
            L.append(f"| {name} | {s['load']} | {s['n']} | {pct(h['exact_match'])} | {pct(h['normalized_match'])} | {h['chars_accepted']:.2f} | "
                     f"{pct(h['syntax_valid'])} | {pct(h['mongo_valid'])} | {pct(h['exec_valid'])} | {pct(h['nonexistent_fields_rate'])} | "
                     f"{pct(s['headline_ide32']['exec_valid'])} | {s['gpu_seconds'] / 60:.1f} |")
        ref = first["reference_ceiling"]
        L += ["", f"Teto do validador (mesmas checagens no `middle` de referencia): syntax {pct(ref.get('syntax_valid'))}, "
                  f"mongo_valid {pct(ref.get('mongo_valid'))}, exec-valid {pct(ref.get('exec_valid'))}; o restante e limite do parser de "
                  "subconjunto do Console (stage `api`), nao erro do modelo.",
              "exec-valid = sintaxe + API do Console + campos do schema + tipos + execucao somente-leitura (maxTimeMS 2.000) em `kapi_fx_*` "
              "sobre `prefix + predicao + suffix`, com predicao nao vazia. Detalhes e por cut_kind/dominio: "
              "`baseline_base/<modelo>/{summary.json,report.md}`.", ""]
    else:
        L += ["Nenhum summary.json encontrado.", ""]

    L += ["## Gates que exigem SFT", "",
          "| gate | status |", "|---|---|",
          "| G2.2 Ganho sobre o base (FIM-acc +15 p.p.; exec-valid >= 0,95; campos inex. <= 2%) | nao avaliado (exige SFT) |",
          "| G2.3 Chat curto (codigo) | nao avaliado (exige SFT) |",
          "| G2.4 Supera os alunos treinados sem professor (+5 p.p.) | nao avaliado (exige SFT; o baseline acima so da o ponto de partida do base) |",
          "| G2.5 Sem esquecimento grave (Generic-JS FIM) | nao avaliado (exige SFT; base do Generic-JS nao medido nesta etapa) |",
          "| G2.6 Professor auxiliar | nao avaliado (fora de escopo) |",
          "| G2.7 Contaminacao (0 test/bench/blind no treino) | nao avaliado (sem treino ainda) |",
          "| G2.8 Papel de T2 (DML INT4 na IDE) | nao avaliado (exige SFT + export) |", ""]
    (REPORTS / "gates.md").write_text("\n".join(L), encoding="utf-8", newline="\n")
    print("gates.md escrito")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
