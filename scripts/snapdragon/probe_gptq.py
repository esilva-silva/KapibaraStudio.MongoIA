"""Run a tiny, random Qwen2 GPTQ kernel probe; does not quantize user weights."""
import argparse
import importlib.metadata
import json
import time
import traceback
from pathlib import Path


def read_prompts(path, limit=4):
    prompts = []
    with Path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get('prompt') or row.get('prompt_text')
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'Missing prompt at line {number}')
            prompts.append(value)
            if len(prompts) == limit:
                break
    if not prompts:
        raise ValueError('Calibration file contains no prompts')
    return prompts


def tokenize_prompts(tokenizer, prompts):
    samples = []
    for prompt in prompts[:4]:
        item = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=64)
        if item['input_ids'].shape[1] > 64:
            raise ValueError('Tokenizer exceeded 64-token limit')
        samples.append({'input_ids': item['input_ids'], 'attention_mask': item['attention_mask']})
    return samples


def inspect_gpu(torch, expected_gpu, require_gpu):
    available = torch.cuda.is_available()
    devices = []
    if available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append({'index': index, 'name': torch.cuda.get_device_name(index), 'vram_bytes': properties.total_memory})
    facts = {'hip': torch.version.hip, 'cuda': torch.version.cuda, 'cuda_available': available,
             'devices': devices, 'current_name': devices[0]['name'] if devices else None}
    if require_gpu:
        normalize = lambda value: ''.join(c.lower() for c in value if c.isalnum())
        if not devices or normalize(expected_gpu) not in normalize(devices[0]['name']):
            raise ValueError('GPU 0 must match ' + expected_gpu + '; visible devices: ' + json.dumps(devices) +
                             '. Select the intended device using HIP_VISIBLE_DEVICES before launching the process.')
    return facts


def run(source, calibration, device, expected_gpu='RX 7800 XT'):
    result = {'status': 'failed', 'scope': 'random tiny model only; no quality or full-run duration claim', 'device': device}
    started = time.monotonic()
    try:
        import torch
        result['versions'] = {}
        for name in ('torch', 'transformers', 'gptqmodel'):
            try:
                result['versions'][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                result['versions'][name] = None
        result['torch'] = inspect_gpu(torch, expected_gpu, False)
        inspect_gpu(torch, expected_gpu, device == 'cuda')
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
        from gptqmodel import QuantizeConfig
        from gptqmodel.models.auto import MODEL_MAP, BaseGPTQModel
        tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True, trust_remote_code=False)
        samples = tokenize_prompts(tokenizer, read_prompts(calibration))
        config = Qwen2Config(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
        model = Qwen2ForCausalLM(config).to(dtype=torch.bfloat16)
        quant_config = QuantizeConfig(bits=4, group_size=-1, sym=True, lm_head=True,
            dynamic={'+:.*lm_head*': {'bits': 8, 'sym': True, 'group_size': 32, 'desc_act': False}},
            device=device, rotation='hadamard')
        quantized = MODEL_MAP.get('qwen2', BaseGPTQModel)(model, False, quant_config,
            trust_remote_code=False, model_local_path=str(source))
        quantized.quantize(samples, tokenizer=tokenizer)
        result.update(status='passed', sample_count=len(samples), sample_token_lengths=[int(x['input_ids'].shape[1]) for x in samples])
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        result['traceback'] = traceback.format_exc()
    finally:
        result['duration_seconds'] = round(time.monotonic() - started, 3)
        try:
            if torch.cuda.is_available():
                result['peak_gpu_bytes'] = torch.cuda.max_memory_allocated()
        except Exception:
            pass
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--calibration', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--expected-gpu', default='RX 7800 XT')
    args = parser.parse_args(argv)
    result = run(args.source, args.calibration, args.device, args.expected_gpu)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
