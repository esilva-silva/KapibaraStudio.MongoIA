"""Manual CPU-only reconstruction of installed Olive/GPTQ calibration token provenance.

Loads tokenizers and input JSONL only. Never imports/constructs a model or runs a
quantization pass. GPTQ's two data-only functions are compiled directly from their
installed AST to avoid package import side effects; their bodies are unmodified.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import sys
import types

for _name in ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'):
    os.environ[_name] = '-1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

WORKTREE = Path(__file__).resolve().parents[2]
ARTIFACTS = Path('C:/IA/models/KapiSnapdragonArtifacts')
EXPECTED_CALIBRATION_SHA = '179736eed969a8a961af6d3fc410a3eead24306234d65ccf245c20cb16978b60'
DEFAULT_RECIPES = [
    ARTIFACTS / 'nano-qnn-prepare-rx7800xt/gptq-only-v2.json',
    ARTIFACTS / 'small-qnn-prepare-rx7800xt-v1/gptq-only-v1.json',
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token_digest(ids):
    # Explicit fixed width and byte order, independent of machine native dtype.
    import struct
    return hashlib.sha256(b''.join(struct.pack('<q', int(i)) for i in ids)).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def installed_function(path, function_name, namespace, class_name=None):
    """Compile the installed data-only body verbatim, without importing GPTQ kernels."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name) if class_name else tree
    node = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == function_name)
    require(not node.decorator_list, f'Unexpected decorators on {function_name}')
    # Postponed annotations avoid importing any type-only GPTQ dependencies.
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[function_name], {'file': str(path), 'sha256': digest(path), 'line': node.lineno, 'end_line': node.end_lineno}


def overlap(start, end, keep_start, keep_end):
    return max(0, min(end, keep_end) - max(start, keep_start))


def audit(recipe_path, original, source_sha):
    import torch
    from datasets import Dataset
    from olive.common.hf.utils import get_tokenizer
    from olive.data.component.dataloader import default_dataloader
    from olive.data.component.text_generation import TextGenParams, text_gen_pre_process
    import olive

    recipe = json.loads(recipe_path.read_text(encoding='utf-8'))
    data_name = recipe['passes']['g']['data_config']
    config = next(c for c in recipe['data_configs'] if c['name'] == data_name)
    require('dataloader_config' not in config, 'Audit expects installed default_dataloader(batch_size=1)')
    params = dict(config['pre_process_data_config'])
    require(params.pop('type') == 'text_generation_huggingface_pre_process', 'Unexpected preprocessor')
    source = Path(params.pop('model_name'))
    require(source == Path(recipe['input_model']['model_path']), 'Tokenizer source differs from model source')
    normalized_path = Path(config['load_dataset_config']['data_files'])
    normalized = read_jsonl(normalized_path)
    require(len(original) == len(normalized) == 256, 'Expected original and normalized 256 rows')
    require(len({r['id'] for r in original}) == len(original), 'Duplicate original IDs')
    require(all(r['text'] == s['prompt'] == s['prompt_text'] for r, s in zip(normalized, original)), 'Normalized texts/order mismatch')
    args = TextGenParams(**params)
    require(args.strategy == 'join' and not args.add_special_tokens and args.text_cols == ['text'], 'Unsupported provenance strategy')
    require(args.processing_batch_size >= len(original), 'Audit ledger expects a single processing batch')
    require(args.max_seq_len == 2048 and args.max_samples == 128, 'Recipe contract differs')
    tokenizer = get_tokenizer(str(source), local_files_only=True, trust_remote_code=False)
    prompt_ids = tokenizer([r['text'] for r in normalized], add_special_tokens=False, truncation=False)['input_ids']
    require(all(ids == row['prompt_ids'] for ids, row in zip(prompt_ids, original)), 'Stored original prompt IDs differ')
    require(all(normalized_row['text'] for normalized_row in normalized), 'Empty input removed by Olive')
    joiner_ids = tokenizer.encode(args.joiner, add_special_tokens=False)
    ledger, stream = [], []
    for index, (row, ids) in enumerate(zip(original, prompt_ids)):
        start = len(stream)
        stream.extend(ids)
        end = len(stream)
        stream.extend(joiner_ids)
        ledger.append({'row_index': index, 'id': row['id'], 'prompt_tokens': len(ids),
                       'prompt_stream_range': [start, end], 'separator_stream_range': [end, len(stream)],
                       'prompt_token_ids_sha256': token_digest(ids),
                       **{key: row[key] for key in ('domain', 'cut_kind') if key in row}})

    # Exact installed Olive implementation, with an entirely in-memory HF Dataset.
    dataset = Dataset.from_list(normalized)
    require(not dataset.cache_files, 'Unexpected cache-backed audit dataset')
    processed = text_gen_pre_process(dataset, tokenizer, params)
    olive_batches = [data[0] for data in default_dataloader(processed)]
    require(all(tuple(b['input_ids'].shape) == (1, 2048) for b in olive_batches), 'Unexpected Olive batch shape')
    expected_starts = list(range(0, len(stream) - args.max_seq_len, args.max_seq_len))[:args.max_samples]
    require(len(olive_batches) == len(expected_starts), 'Independent ledger vs Olive count mismatch')
    for b, start in zip(olive_batches, expected_starts):
        require(b['input_ids'].reshape(-1).tolist() == stream[start:start + 2048], 'Independent ledger vs Olive token mismatch')
        require(bool(torch.all(b['attention_mask'] == 1)), 'Unexpected padding or masked input')

    site = Path(olive.__file__).resolve().parent.parent
    namespace = {'torch': torch, 'LongTensor': torch.LongTensor, 'Tensor': torch.Tensor,
                 'log': logging.getLogger('gptq_attribution')}
    _, collate_source = installed_function(site / 'gptqmodel/utils/data.py', 'collate_data', namespace)
    prepare, prepare_source = installed_function(site / 'gptqmodel/models/base.py', 'prepare_dataset', namespace, 'BaseGPTQModel')
    # Qwen2GPTQ -> LlamaGPTQ -> BaseGPTQModel; no subclass batching override.
    for definition in ('qwen2.py', 'llama.py'):
        text = (site / 'gptqmodel/models/definitions' / definition).read_text(encoding='utf-8')
        require('support_batch_quantize' not in text and 'def prepare_dataset' not in text, 'Subclass data preparation changed')
    base_tree = ast.parse((site / 'gptqmodel/models/base.py').read_text(encoding='utf-8'))
    base_class = next(n for n in base_tree.body if isinstance(n, ast.ClassDef) and n.name == 'BaseGPTQModel')
    supports = [n.value.value for n in base_class.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'support_batch_quantize' for t in n.targets)]
    require(supports and all(v is True for v in supports), 'Base batch support changed')
    prepared = prepare(types.SimpleNamespace(tokenizer=tokenizer, support_batch_quantize=True), olive_batches)
    require(len(prepared) == len(olive_batches), 'GPTQ removed/rebatched inputs')
    require(all(torch.equal(a['input_ids'], b['input_ids']) and torch.equal(a['attention_mask'], b['attention_mask'])
                for a, b in zip(olive_batches, prepared)), 'GPTQ changed tokens/masks/order')
    status_path = recipe_path.with_name(recipe_path.name.replace('gptq-only-', 'gptq-job-status-'))
    status = json.loads(status_path.read_text(encoding='utf-8'))
    require(status['config_sha256'] == digest(recipe_path), 'Historical recipe SHA mismatch')
    lengths = [b['input_ids'].numel() for b in prepared]
    effective = status['effective_gptq_calibration']
    require((effective['batches'], effective['min_tokens'], effective['max_tokens']) == (len(prepared), min(lengths), max(lengths)), 'Historical batch counts/length mismatch')
    require(all(status['versions'][n] == importlib.metadata.version(n) for n in ('olive-ai', 'gptqmodel', 'transformers', 'datasets')), 'Historical/current package version mismatch')
    require(len(prepared) == 29, 'Expected recorded 29 blocks')
    used = sum(lengths)
    # Data-only feasibility check for a future, separately versioned tail policy.
    # This is NOT the current job input and does not run any model/pass.
    tail = stream[used:]
    hypothetical = prepare(types.SimpleNamespace(tokenizer=tokenizer, support_batch_quantize=True),
        olive_batches + [{'input_ids': torch.tensor([tail], dtype=torch.long),
                          'attention_mask': torch.ones((1, len(tail)), dtype=torch.long)}])
    require(len(hypothetical) == len(prepared) + 1, 'GPTQ did not retain proposed tail')
    require([t for b in hypothetical for t in b['input_ids'].reshape(-1).tolist()] == stream,
            'Proposed data-only tail preparation loses/reorders tokens')
    counts = Counter()
    for item in ledger:
        start, end = item['prompt_stream_range']
        sepstart, sepend = item['separator_stream_range']
        count = overlap(start, end, 0, used)
        classification = 'complete' if count == item['prompt_tokens'] else 'partial' if count else 'omitted'
        counts[classification] += 1
        item.update({'contribution': classification, 'included_prompt_tokens': count,
                     'omitted_prompt_tokens': item['prompt_tokens'] - count,
                     'included_prompt_local_range': [0, count],
                     'included_separator_tokens': overlap(sepstart, sepend, 0, used),
                     'omitted_separator_tokens': sepend - sepstart - overlap(sepstart, sepend, 0, used),
                     'batch_indices': [j for j in range(len(prepared)) if overlap(start, end, j * 2048, (j + 1) * 2048)]})
        item['complete_within_one_batch'] = classification == 'complete' and len(item['batch_indices']) == 1
        item['complete_token_coverage_but_split_between_batches'] = classification == 'complete' and len(item['batch_indices']) > 1
    batches = []
    for j, b in enumerate(prepared):
        start, end = j * 2048, (j + 1) * 2048
        segments = []
        for item in ledger:
            for kind, key in (('prompt', 'prompt_stream_range'), ('separator', 'separator_stream_range')):
                a, z = item[key]
                lo, hi = max(start, a), min(end, z)
                if lo < hi:
                    segments.append({'id': item['id'], 'kind': kind, 'tokens': hi - lo,
                                     'batch_local_range': [lo - start, hi - start], 'source_local_range': [lo - a, hi - a]})
        require(sum(s['tokens'] for s in segments) == 2048, 'Unattributed batch token')
        batches.append({'index': j, 'stream_range': [start, end], 'token_ids_sha256': token_digest(b['input_ids'].reshape(-1).tolist()),
                        'olive_token_ids_sha256': token_digest(olive_batches[j]['input_ids'].reshape(-1).tolist()),
                        'independent_ledger_token_ids_sha256': token_digest(stream[start:end]),
                        'three_token_hashes_equal': True,
                        'attention_mask_sha256': token_digest(b['attention_mask'].reshape(-1).tolist()), 'segments': segments})
    source_files = [site / p for p in ('olive/data/component/text_generation.py', 'olive/data/component/dataloader.py',
        'olive/data/component/dataset.py', 'olive/common/hf/utils.py', 'olive/passes/pytorch/train_utils.py',
        'olive/passes/pytorch/gptqmodel.py', 'gptqmodel/looper/loop_processor.py', 'gptqmodel/looper/module_looper.py',
        'olive/data/config.py', 'olive/data/container/data_container.py',
        'gptqmodel/models/definitions/qwen2.py', 'gptqmodel/models/definitions/llama.py')]
    return {'recipe': str(recipe_path), 'recipe_sha256': digest(recipe_path), 'source_snapshot': str(source),
            'normalized_calibration': str(normalized_path), 'normalized_sha256': digest(normalized_path),
            'original_sha256': source_sha, 'normalized_texts_and_order_match_original': True,
            'stored_prompt_ids_match': True, 'tokenizer_files': {p.name: digest(p) for p in source.iterdir() if p.name in ('tokenizer.json', 'tokenizer_config.json')},
            'parameters': {'strategy': str(args.strategy), 'add_special_tokens': args.add_special_tokens,
                'text_cols': args.text_cols, 'joiner': args.joiner, 'joiner_token_ids': joiner_ids,
                'processing_batch_size': args.processing_batch_size, 'max_samples_blocks': args.max_samples,
                'max_samples_cap_reached': len(prepared) >= args.max_samples, 'max_seq_len': args.max_seq_len,
                'olive_dataloader_batch_size': 1, 'gptq_batch_size': 1, 'gptq_concat_size': None,
                'gptq_min_length_exclusive': 10, 'gptq_filtered_blocks': 0, 'randomization': False,
                'per_prompt_truncation': False, 'padding_tokens': 0, 'added_bos_eos_tokens': 0},
            'summary': {'selected_prompt_rows': len(original), 'prompt_classifications': dict(counts),
                'complete_within_one_batch': sum(i['complete_within_one_batch'] for i in ledger),
                'complete_token_coverage_but_split_between_batches': sum(i['complete_token_coverage_but_split_between_batches'] for i in ledger),
                'complete_definition': 'Every token from the prompt is present in the consumed joined stream; this does not imply the whole prompt lies within a single2048-token batch.',
                'total_prompt_tokens': sum(map(len, prompt_ids)), 'total_separator_tokens': len(original) * len(joiner_ids),
                'joined_tokens': len(stream), 'batches': len(prepared), 'consumed_tokens': used,
                'consumed_prompt_tokens': sum(i['included_prompt_tokens'] for i in ledger),
                'consumed_separator_tokens': sum(i['included_separator_tokens'] for i in ledger),
                'discarded_overflow_tokens': len(stream) - used,
                'discarded_prompt_tokens': sum(i['omitted_prompt_tokens'] for i in ledger),
                'discarded_separator_tokens': sum(i['omitted_separator_tokens'] for i in ledger)},
            'joined_token_ids_sha256': token_digest(stream), 'consumed_token_ids_sha256': token_digest(stream[:used]),
            'discarded_overflow_token_ids_sha256': token_digest(stream[used:]),
            'verification': {'independent_id_ledger_equals_installed_olive_tokens': True,
                'installed_gptq_preparation_preserves_all_tokens_masks_order': True,
                'historical_recipe_sha_and_package_versions_match': True,
                'historical_batch_count_and_lengths_match': True,
                'historical_runtime_token_hash_comparison': 'unavailable: job status records lengths/counts, not token IDs or hashes; this is a deterministic CPU reconstruction, not a retroactive runtime token capture'},
            'installed_data_function_execution': {'method': 'unmodified AST bodies with postponed annotations, CPU tensors only; no GPTQ package/model import',
                'prepare_dataset': prepare_source, 'collate_data': collate_source},
            'installed_source_files': {str(p): digest(p) for p in source_files},
            'historical_status': {'path': str(status_path), 'sha256': digest(status_path), 'effective_gptq_calibration': effective},
            'coverage_exceptions': [i for i in ledger if i['contribution'] != 'complete'],
            'join_tail_support': {'installed_keep_tail_parameter': False,
                'pad_to_max_len': args.pad_to_max_len,
                'pad_to_max_len_applies_to_join': False,
                'drop_short_sequences_false_retains_join_tail': False,
                'reason': 'JOIN emits only range(0, len(joined_input_ids)-max_seq_len, max_seq_len), stores overflow, and exits when text_list is exhausted. No final flush/padding branch exists. An exact final2048 block would also remain in overflow because range stop is exclusive.',
                'required_change_for_30_batches': 'Explicitly versioned custom preprocessor or JOIN implementation change to append the final1881-token block; no existing keep-tail flag. batch_size1 supports this variable final length without new data or padding.'},
            'future_tail_policy_data_only_check': {'applied_to_current_jobs': False,
                'model_or_quantization_run': False, 'same_source_prompt_ids': True,
                'hypothetical_batches': len(hypothetical), 'hypothetical_last_tokens': len(tail),
                'hypothetical_valid_tokens': sum(b['input_ids'].numel() for b in hypothetical),
                'hypothetical_padding_tokens': 0,
                'whole_prompt_context_within_single_batch_guaranteed': False,
                'installed_gptq_prepare_retains_entire_original_joined_stream': True,
                'hypothetical_last_token_ids_sha256': token_digest(tail)},
            'tokenizer_load_warning': 'Transformers4.57.6 emitted an incorrect-Mistral-regex warning for both local tokenizers. No fix_mistral_regex flag was added: the original job mechanism is preserved and all256 stored prompt_ids match exactly.',
            'prompts': ledger, 'batches': batches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, default=WORKTREE / 'data/calib/snapdragon-20261026/calibration.jsonl')
    parser.add_argument('--recipe', action='append', type=Path)
    parser.add_argument('--out', type=Path, default=WORKTREE / 'reports/snapdragon/gptq-calibration-token-attribution-rx7800xt.json')
    args = parser.parse_args()
    out = args.out.resolve()
    require(out.is_relative_to((WORKTREE / 'reports').resolve()), 'Output must be inside worktree reports/')
    require(not out.exists(), 'Refusing to overwrite an existing audit report')
    source_sha = digest(args.samples)
    require(source_sha == EXPECTED_CALIBRATION_SHA, 'Original calibration SHA mismatch')
    original = read_jsonl(args.samples)
    versions = {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'gptqmodel', 'olive-ai', 'datasets')}
    require(versions['transformers'] == '4.57.6' and versions['gptqmodel'] == '4.2.0' and versions['olive-ai'] == '0.13.0', 'Use the original probe environment')
    jobs = [audit(p.resolve(), original, source_sha) for p in (args.recipe or DEFAULT_RECIPES)]
    require(len({j['consumed_token_ids_sha256'] for j in jobs}) == 1, 'Nano/Small reconstructed input token streams differ')
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'CPU tokenization and data preparation only; no model/quantization/calibration/GPU/cache writes',
        'hash_encoding': 'SHA256 of concatenated signed int64 little-endian token IDs; ranges are zero-based half-open',
        'script_sha256': digest(__file__), 'versions': versions, 'original_samples': str(args.samples.resolve()),
        'jobs': jobs, 'contract_assessment': {'all_256_prompts_completely_consumed': all(j['summary']['prompt_classifications'].get('complete', 0) == 256 for j in jobs),
            'finding': 'The 256-row dataset is intact. This is JOIN tail loss and an effective coverage distinction: 248 complete token coverage + 1 partial + 7 omitted. Of the 248, 220 lie within a single batch and 28 cross batch boundaries. max_samples=128 counts blocks, not prompt rows; the cap is not reached. No per-prompt tokenizer truncation or GPTQ downstream removal/reordering occurs here.',
            'historical_evidence_limit': 'Historical status captured 29 batches of 2048 and recipe/version metadata, not token IDs or hashes. Attribution by ID and batch hashes in this report are a verifiable deterministic reconstruction using the pinned inputs and installed data code, not historical runtime token capture.',
            'future_correction': 'If a future contract requires every selected prompt in full, explicitly version a custom preprocessor/JOIN change retaining the 1881-token tail: 30 batches, same 256 IDs, no new data. The installed GPTQ data-only preparation accepts 29 x 2048 + 1881 with batch_size 1 and no padding. Such a policy changes the quantization input and requires new GPTQ weights plus repeated activation calibration and evaluation before qualification. Also capture runtime token hashes. Retaining the tail ensures full token coverage but still splits some prompts across batches; an indivisible-prompt context requirement needs a different explicit batching policy. No current weights/job/caches were changed or rerun.'}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(out), 'jobs': [{'recipe': j['recipe'], 'summary': j['summary']} for j in jobs]}, indent=2))


if __name__ == '__main__':
    main()
