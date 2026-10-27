"""Versioned Olive 0.13 GPTQ preprocessing: ordered JOIN with retained unpadded tail.

No model, Torch or accelerator import. Register through DataConfig.user_script.
This changes calibration input and requires new weights and downstream validation.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import zipfile

POLICY = 'kapi_gptq_join_tail_v1'
OLIVE_WHEEL_SHA256 = 'b4b1768de04659cb9ed3b806ebe9b02f93e959479ca753334bb6ba592a0d5de4'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token_sha(ids):
    return hashlib.sha256(b''.join(struct.pack('<q', int(value)) for value in ids)).hexdigest()


def project_rows(rows):
    """Discard heterogeneous metadata without changing any calibration input."""
    result = []
    seen = set()
    for row in rows:
        identity, prompt, text, ids = (row.get(key) for key in ('id', 'prompt', 'prompt_text', 'prompt_ids'))
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError('Expected unique nonempty string row IDs')
        if not isinstance(prompt, str) or not prompt or not isinstance(text, str) or prompt != text:
            raise ValueError('Missing or inconsistent original FIM prompt')
        if not isinstance(ids, list) or not ids or any(type(value) is not int for value in ids):
            raise ValueError('Prompt token IDs must be nonempty integers')
        seen.add(identity)
        result.append({'id': identity, 'prompt': prompt, 'prompt_text': text, 'prompt_ids': ids.copy()})
    return result


def build_blocks(rows, tokenizer, max_seq_len=2048, joiner='\n\n', expected_count=256):
    if max_seq_len != 2048 or len(rows) != expected_count or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Expected unique ordered calibration rows and context 2048')
    separator = tokenizer.encode(joiner, add_special_tokens=False).ids
    stream, ledger = [], []
    for index, row in enumerate(rows):
        text = row.get('prompt_text') or row.get('prompt')
        if not isinstance(text, str) or not text or (row.get('prompt') is not None and row['prompt'] != text):
            raise ValueError('Missing or inconsistent original FIM prompt')
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if not ids or ids != row.get('prompt_ids'):
            raise ValueError('Original prompt token IDs mismatch')
        start = len(stream); stream.extend(ids); end = len(stream); stream.extend(separator)
        ledger.append({'id': row['id'], 'row_index': index, 'prompt_tokens': len(ids),
                       'prompt_token_ids_sha256': token_sha(ids), 'prompt_stream_range': [start, end],
                       'separator_stream_range': [end, len(stream)]})
    blocks = [stream[start:start + max_seq_len] for start in range(0, len(stream), max_seq_len)]
    # GPTQModel 4.2.0 prepare_dataset filters sequences of <=10 tokens. Never silently lose one.
    if not blocks or min(map(len, blocks)) <= 10:
        raise ValueError('A block would be removed by GPTQ minimum-length filtering')
    if [token for block in blocks for token in block] != stream:
        raise ValueError('Calibration token stream lost or reordered')
    for row in ledger:
        start, end = row['prompt_stream_range']
        row.update(covered_prompt_tokens=end - start, omitted_prompt_tokens=0,
                   batch_indices=list(range(start // max_seq_len, (end - 1) // max_seq_len + 1)))
    batches = [{'index': index, 'tokens': len(block), 'stream_range': [index * max_seq_len, index * max_seq_len + len(block)],
                'token_ids_sha256': token_sha(block), 'padding_tokens': 0,
                'prompt_ids': [row['id'] for row in ledger if index in row['batch_indices']]}
               for index, block in enumerate(blocks)]
    return blocks, {'policy': POLICY, 'selected_rows': len(rows), 'all_prompt_tokens_covered': True,
                    'joined_tokens': len(stream), 'consumed_tokens': sum(map(len, blocks)),
                    'stream_token_ids_sha256': token_sha(stream), 'add_special_tokens': False,
                    'batch_size': 1, 'padding': 'none', 'shuffle': False, 'max_seq_len': max_seq_len,
                    'joiner': joiner, 'joiner_token_ids': separator, 'batches': batches, 'prompts': ledger,
                    'scope': 'Full token coverage; prompts may cross block boundaries; not indivisible prompt batching'}


def kapi_gptq_join_tail_v1(dataset, model_name, audit_path, expected_stream_sha256, expected_count=256, **olive_parameters):
    import numpy as np
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(model_name) / 'tokenizer.json'))
    tokenizer.no_truncation(); tokenizer.no_padding()
    blocks, report = build_blocks(list(dataset), tokenizer, expected_count=expected_count)
    if report['stream_token_ids_sha256'] != expected_stream_sha256:
        raise ValueError('Runtime calibration token stream differs from generated recipe')
    out = Path(audit_path)
    if out.exists():
        raise ValueError('Runtime calibration audit path must be new')
    report.update(tokenizer_sha256=sha(Path(model_name) / 'tokenizer.json'), helper_sha256=sha(__file__),
                  scope_runtime='Actual preprocessor output before Olive DataLoader/GPTQ; not downstream consumption capture',
                  forwarded_parameter_names=sorted(olive_parameters))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return [({'input_ids': np.array(block, dtype=np.int64),
              'attention_mask': np.ones(len(block), dtype=np.int64)}, 0) for block in blocks]


# DataConfig imports user_script before resolving Registry components. Importing
# this helper in the CPU preflight does not import Olive or Torch if unavailable.
if importlib.util.find_spec('olive') is not None:
    from olive.data.registry import Registry
    Registry.register_pre_process(POLICY)(kapi_gptq_join_tail_v1)


def wheel_contract(wheel):
    if sha(wheel) != OLIVE_WHEEL_SHA256:
        raise ValueError('Olive wheel differs from the inspected exact 0.13.0 wheel')
    names = ('olive/data/config.py', 'olive/data/registry.py', 'olive/data/component/dataloader.py')
    with zipfile.ZipFile(wheel) as archive:
        metadata = archive.read('olive_ai-0.13.0.dist-info/METADATA').decode()
        if '\nVersion: 0.13.0\n' not in metadata:
            raise ValueError('Expected Olive 0.13.0 wheel')
        sources = {name: archive.read(name) for name in names}
    if (b'import_user_module(values["user_script"]' not in sources[names[0]] or
            b'def register_pre_process(' not in sources[names[1]] or
            b'def default_dataloader(dataset, batch_size=1, **kwargs)' not in sources[names[2]]):
        raise ValueError('Unsupported Olive data registration contract')
    return {'version': '0.13.0', 'wheel_sha256': sha(wheel),
            'sources_sha256': {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}}


def generate(recipe, samples, source, olive_wheel, out_dir):
    from tokenizers import Tokenizer
    out = Path(out_dir).resolve()
    if out.exists():
        raise ValueError('Use a new isolated output directory')
    recipe = Path(recipe).resolve(); samples = Path(samples).resolve(); source = Path(source).resolve()
    config = json.loads(recipe.read_text(encoding='utf-8'))
    if Path(config['input_model']['model_path']).resolve() != source:
        raise ValueError('Tokenizer source differs from recipe model source')
    if any(item['name'] == POLICY for item in config['data_configs']):
        raise ValueError('Recipe already contains the versioned tail policy')
    contract = wheel_contract(olive_wheel)
    rows = [json.loads(line) for line in samples.read_text(encoding='utf-8').splitlines() if line.strip()]
    projected = project_rows(rows)
    tokenizer = Tokenizer.from_file(str(source / 'tokenizer.json')); tokenizer.no_padding(); tokenizer.no_truncation()
    blocks, report = build_blocks(rows, tokenizer)
    projected_blocks, projected_report = build_blocks(projected, tokenizer)
    if projected_blocks != blocks or projected_report['stream_token_ids_sha256'] != report['stream_token_ids_sha256']:
        raise ValueError('Projected calibration token stream differs from original')
    projection_bytes = ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n' for row in projected).encode('utf-8')
    data_name = config['passes']['g']['data_config']
    if not isinstance(data_name, str):
        raise ValueError('Expected GPTQ named data_config')
    data = next(item for item in config['data_configs'] if item['name'] == data_name)
    # Never alter the shared activation data config: GPTQ gets a new named copy.
    data = copy.deepcopy(data); data['name'] = POLICY
    data.update(user_script=str(Path(__file__).resolve()), script_dir=str(Path(__file__).resolve().parent),
                load_dataset_config={'type': 'huggingface_dataset', 'data_name': 'json', 'data_files': str(out / 'calibration-projection.jsonl'), 'split': 'train'},
                pre_process_data_config={'type': POLICY, 'model_name': str(source), 'audit_path': str(out / 'runtime-token-ledger.json'),
                    'expected_stream_sha256': report['stream_token_ids_sha256'], 'expected_count': 256},
                dataloader_config={'type': 'default_dataloader', 'batch_size': 1, 'shuffle': False, 'drop_last': False})
    config['data_configs'].append(data); config['passes']['g']['data_config'] = POLICY
    config['cache_dir'] = str(out / 'cache'); config['output_dir'] = str(out / 'output')
    report.update(samples_sha256=sha(samples), projection_sha256=hashlib.sha256(projection_bytes).hexdigest(),
                  projection_fields=['id', 'prompt', 'prompt_text', 'prompt_ids'], projection_rows=len(projected),
                  projection_source_sha256=sha(samples), tokenizer_sha256=sha(source / 'tokenizer.json'),
                  parent_recipe_sha256=sha(recipe), helper_sha256=sha(__file__), olive_contract=contract,
                  historical_jobs_modified=False, weights_requantization_required=True)
    out.mkdir(parents=True)
    (out / 'calibration-projection.jsonl').write_bytes(projection_bytes)
    (out / 'gptq-join-tail-v1.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    (out / 'preflight-token-ledger.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('recipe', 'samples', 'source', 'olive-wheel', 'out-dir'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    report = generate(args.recipe, args.samples, args.source, args.olive_wheel, args.out_dir)
    print(json.dumps({'policy': POLICY, 'rows': report['selected_rows'], 'tokens': report['joined_tokens'],
                      'batch_lengths': [batch['tokens'] for batch in report['batches']]}))


if __name__ == '__main__':
    main()
