"""Measure local HF or ORT GenAI inference without executing generated code."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import time
import uuid
import gc
import hashlib
import re
import os

CONTEXT = 2048
LIMIT = 32
MARKERS = ('<|fim_prefix|>', '<|fim_suffix|>', '<|fim_middle|>', '<|endoftext|>', '<|im_end|>', '<|fim_pad|>')

def source_commit():
    override = os.environ.get('KAPI_RUNNER_COMMIT')
    if override:
        return override
    gitdir = Path(__file__).resolve().parents[2] / '.git'
    try:
        if gitdir.is_file():
            gitdir = (gitdir.parent / gitdir.read_text().strip().split(': ', 1)[1]).resolve()
        head = (gitdir / 'HEAD').read_text().strip()
        if not head.startswith('ref: '):
            return head
        ref = head[5:]
        if (gitdir / ref).exists():
            return (gitdir / ref).read_text().strip()
        for line in (gitdir / 'packed-refs').read_text().splitlines():
            if line.endswith(' ' + ref):
                return line.split()[0]
    except (OSError, IndexError):
        pass
    return None

def gpu_name_matches(actual, expected):
    normalize = lambda value: re.sub(r'[^a-z0-9]', '', value.casefold())
    return normalize(expected) in normalize(actual)

def verify_gpu(torch, device, expected_gpu=None):
    if not device.startswith('cuda'):
        if expected_gpu:
            raise ValueError('Expected GPU requires an explicit CUDA/HIP device')
        return None
    parsed = torch.device(device)
    index = parsed.index if parsed.index is not None else torch.cuda.current_device()
    actual = torch.cuda.get_device_name(index)
    if expected_gpu and not gpu_name_matches(actual, expected_gpu):
        raise ValueError(f'GPU mismatch: {device} is {actual!r}; required {expected_gpu!r}')
    return {'index': index, 'name': actual, 'vram_bytes': torch.cuda.get_device_properties(index).total_memory,
            'hip': torch.version.hip}

def stop_ids(encode):
    result = set()
    for marker in MARKERS:
        ids = list(encode(marker))
        if len(ids) == 1:
            result.add(int(ids[0]))
    return result

def trim_suffix(result, item):
    suffix = item.get('suffix', '')
    if suffix and suffix in result['generated_text']:
        result['generated_text'] = result['generated_text'].split(suffix, 1)[0]
        result['stop_reason'] = 'suffix'
    return result

def memory():
    try:
        import psutil
        m = psutil.Process().memory_info()
        return {'rss_bytes': m.rss, 'peak_rss_bytes': getattr(m, 'peak_wset', None)}
    except ImportError:
        return {'rss_bytes': None, 'peak_rss_bytes': None, 'unavailable': 'psutil not installed'}

def load_corpus(path):
    items = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            item = json.loads(line)
            if 'prompt_text' not in item and 'prompt' in item:
                item['prompt_text'] = item['prompt']
            if not isinstance(item.get('prompt_text'), str) or 'id' not in item:
                raise ValueError('Each corpus item requires id and prompt_text')
            items.append(item)
    if not items:
        raise ValueError('Corpus is empty')
    return items

def check_ids(ids):
    if not ids or any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in ids):
        raise ValueError('prompt_ids must be a nonempty list of nonnegative integers')
    if len(ids) + LIMIT > CONTEXT:
        raise ValueError(f'Prompt exceeds {CONTEXT - LIMIT} token input budget; no implicit truncation')

class HFRunner:
    def __init__(self, model, device, expected_gpu=None):
        import torch
        gpu = verify_gpu(torch, device, expected_gpu)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(model, local_files_only=True, trust_remote_code=False,
                                                        torch_dtype=torch.float32 if device == 'cpu' else torch.bfloat16).to(device).eval()
        self.stops = stop_ids(lambda text: self.tok.encode(text, add_special_tokens=False))
        self.evidence = {'requested': device, 'effective': str(next(self.model.parameters()).device),
                         'torch_hip': torch.version.hip, 'scope': 'model tensor device', 'gpu': gpu,
                         'expected_gpu': expected_gpu, 'dtype': str(next(self.model.parameters()).dtype)}

    def sync(self):
        if self.device.startswith('cuda'):
            self.torch.cuda.synchronize(self.device)

    def generate(self, item):
        ids = self.tok.encode(item['prompt_text'], add_special_tokens=False)
        if item.get('prompt_ids') is not None and ids != item['prompt_ids']:
            raise ValueError('Corpus prompt_ids differ from model tokenizer')
        check_ids(ids)
        torch = self.torch
        x = torch.tensor([ids], device=self.device)
        mask = torch.ones_like(x)
        cache, tokens, first = None, [], None
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        self.sync()
        start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(LIMIT):
                result = self.model(input_ids=x, attention_mask=mask, past_key_values=cache, use_cache=True)
                token = int(result.logits[0, -1].argmax().item())
                self.sync()
                first = first or time.perf_counter()
                tokens.append(token)
                if token in eos | self.stops:
                    break
                if item.get('suffix') and item['suffix'] in self.tok.decode(tokens, skip_special_tokens=False):
                    break
                cache = result.past_key_values
                x = torch.tensor([[token]], device=self.device)
                mask = torch.cat((mask, torch.ones((1, 1), dtype=mask.dtype, device=self.device)), dim=1)
        end = time.perf_counter()
        content = [t for t in tokens if t not in eos | self.stops]
        stop = 'eos' if tokens[-1] in eos else 'fim_stop' if tokens[-1] in self.stops else 'max_tokens'
        result = measurement(tokens, self.tok.decode(content, skip_special_tokens=False), start, first,
                             end, stop, len(ids))
        result['content_tokens_before_suffix_trim'] = len(content)
        result['generator_creation_ms'] = 0.0
        return trim_suffix(result, item)

def pipeline_stages(decoder):
    stages = []
    for index, group in enumerate(decoder.get('pipeline') or []):
        if not isinstance(group, dict):
            raise ValueError('Invalid pipeline stage')
        if 'filename' in group:
            stages.append((group.get('name', f'stage-{index}'), group))
        else:
            for name, stage in group.items():
                if not isinstance(stage, dict) or 'filename' not in stage:
                    raise ValueError('Invalid grouped pipeline stage: ' + name)
                stages.append((name, stage))
    return stages


def stage_providers(stage):
    options = stage.get('session_options') or {}
    entries = options.get('provider_options') or []
    return [name.lower() for entry in entries for name in entry] or ['cpu']


def genai_session_overlay(original, profile_prefix, backend):
    import copy
    decoder = original['model']['decoder']
    session = copy.deepcopy(decoder.get('session_options') or {})
    mixed = bool(decoder.get('pipeline'))
    if backend == 'qnn' and mixed and 'qnn' in stage_providers({'session_options': session}):
        raise ValueError('Mixed pipeline decoder default must not be QNN')
    if backend == 'qnn' and not mixed:
        session['disable_cpu_ep_fallback'] = True
    if profile_prefix:
        session['enable_profiling'] = str(profile_prefix)
    overlay = {'session_options': session}
    if mixed:
        pipeline = copy.deepcopy(decoder['pipeline'])
        stages = pipeline_stages({'pipeline': pipeline})
        qnn_count = 0
        for index, (name, stage) in enumerate(stages):
            providers = stage_providers(stage)
            is_qnn = 'qnn' in providers
            is_transformer = (stage.get('run_on_token_gen') is False or stage.get('run_on_prompt') is False
                              or name.lower().startswith(('context', 'iterator', 'ctx', 'iter')))
            if backend == 'qnn' and is_transformer and not is_qnn:
                raise ValueError('Transformer pipeline stage must use QNN: ' + name)
            options = stage.setdefault('session_options', {})
            if backend == 'qnn' and is_qnn:
                qnn_count += 1
                options['disable_cpu_ep_fallback'] = True
                if profile_prefix:
                    for entry in options.get('provider_options', []):
                        for provider, provider_options in entry.items():
                            if provider.lower() == 'qnn':
                                provider_options['profiling_level'] = 'basic'
                                provider_options['profiling_file_path'] = f'{profile_prefix}-{index}.qnn.csv'
            if profile_prefix:
                options['enable_profiling'] = f'{profile_prefix}-{index}'
        if backend == 'qnn' and not qnn_count:
            raise ValueError('QNN pipeline has no QNN stage')
        overlay['pipeline'] = pipeline
    return {'model': {'decoder': overlay}}


def validate_qnn_options(options, htp_path):
    for key, value in (options or {}).items():
        if key == 'backend_path' and str(value).lower() != str(htp_path).lower():
            raise ValueError('QNN backend_path must be the HTP library')
        if key == 'backend_type' and str(value).lower() != 'htp':
            raise ValueError('QNN backend_type must be htp')


class GenAIRunner:
    def __init__(self, model, backend, profile_prefix=None, qnn_options=None):
        try:
            import onnxruntime_genai as og
        except (ImportError, OSError) as exc:
            raise RuntimeError(f'ORT GenAI {backend} unavailable on {platform.machine()}; no CPU fallback: {exc}') from exc
        if backend == 'dml' and (not hasattr(og, 'is_dml_available') or not og.is_dml_available()):
            raise RuntimeError('ORT GenAI build cannot confirm DirectML availability; no fallback')
        original = json.loads((Path(model) / 'genai_config.json').read_text(encoding='utf-8'))
        decoder = original['model']['decoder']
        mixed = bool(decoder.get('pipeline'))
        stages = pipeline_stages(decoder) if mixed else []
        intended_stages = [{'name': name, 'providers': stage_providers(stage)} for name, stage in stages]
        cfg = og.Config(str(model))
        if not mixed:
            cfg.clear_providers()
        if backend == 'qnn':
            import onnxruntime_qnn as qnn
            og.register_execution_provider_library(qnn.get_ep_name(), qnn.get_library_path())
            htp = qnn.get_qnn_htp_path()
            validate_qnn_options(qnn_options, htp)
            def validate_embedded(value):
                if isinstance(value, dict):
                    for key, child in value.items():
                        if key.lower() == 'qnn' and isinstance(child, dict):
                            validate_qnn_options(child, htp)
                        validate_embedded(child)
                elif isinstance(value, list):
                    for child in value:
                        validate_embedded(child)
            validate_embedded(decoder)
            if not mixed:
                cfg.append_provider('QNN')
                cfg.set_provider_option('QNN', 'backend_path', htp)
                for key, value in (qnn_options or {}).items():
                    if key not in ('backend_path', 'backend_type'):
                        cfg.set_provider_option('QNN', key, str(value))
                if profile_prefix:
                    cfg.set_provider_option('QNN', 'profiling_level', 'basic')
                    cfg.set_provider_option('QNN', 'profiling_file_path', str(profile_prefix) + '.qnn.csv')
        elif backend == 'dml' and not mixed:
            cfg.append_provider('dml')
        self.profile_prefix = profile_prefix
        self.backend = backend
        if profile_prefix or backend == 'qnn':
            cfg.overlay(json.dumps(genai_session_overlay(original, profile_prefix, backend)))
        self.og, self.model = og, og.Model(cfg)
        self.tok = og.Tokenizer(self.model)
        self.stops = stop_ids(self.tok.encode)
        config = json.loads((Path(model) / 'genai_config.json').read_text(encoding='utf-8'))
        eos = config['model'].get('eos_token_id')
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.evidence = {'requested': {'dml': 'DmlExecutionProvider', 'qnn': 'QNNExecutionProvider', 'cpu': 'CPUExecutionProvider'}[backend], 'available': True, 'configured': backend,
                         'effective': None, 'verified': False, 'intended_stages': intended_stages,
                         'reason': 'ORT GenAI Python does not expose per-node execution provider profiling here; availability and explicit configuration do not prove GPU execution'}

    def close(self):
        self.tok = None
        self.model = None
        gc.collect()
        counts, errors, files = {}, [], []
        if self.profile_prefix:
            files = sorted(self.profile_prefix.parent.glob(self.profile_prefix.name + '*.json'))
            for path in files:
                try:
                    for event in json.loads(path.read_text(encoding='utf-8')):
                        provider = event.get('args', {}).get('provider')
                        if provider:
                            counts[provider] = counts.get(provider, 0) + 1
                except Exception as exc:
                    errors.append(str(exc))
        qnn_csv = sorted(self.profile_prefix.parent.glob(self.profile_prefix.name + '*.csv')) if self.profile_prefix and self.backend == 'qnn' else []
        self.evidence['qnn_profile_files'] = [str(p) for p in qnn_csv]
        requested = {'dml': 'DmlExecutionProvider', 'qnn': 'QNNExecutionProvider', 'cpu': 'CPUExecutionProvider'}[self.backend]
        self.evidence.update(effective=list(counts), verified=counts.get(requested, 0) > 0,
                             kernel_events_by_provider=counts, cpu_kernel_events=counts.get('CPUExecutionProvider', 0),
                             profile_files=[str(p) for p in files], profile_errors=errors,
                             reason='Observed ORT kernel events' if counts else 'No provider kernel events captured; execution unverified')

    def generate(self, item):
        import numpy as np
        ids = [int(i) for i in self.tok.encode(item['prompt_text'])]
        if item.get('prompt_ids') is not None and ids != item['prompt_ids']:
            raise ValueError('Corpus prompt_ids differ from ORT tokenizer')
        check_ids(ids)
        p = self.og.GeneratorParams(self.model)
        p.set_search_options(do_sample=False, max_length=len(ids) + LIMIT, batch_size=1)
        start = time.perf_counter()
        g = self.og.Generator(self.model, p)
        creation_ms = (time.perf_counter() - start) * 1000
        tokens, first = [], None
        try:
            g.append_tokens(np.array(ids, dtype=np.int32))
            while not g.is_done() and len(tokens) < LIMIT:
                g.generate_next_token()
                token = int(g.get_next_tokens()[0])
                first = first or time.perf_counter()
                tokens.append(token)
                if token in self.eos | self.stops:
                    break
                if item.get('suffix') and item['suffix'] in self.tok.decode(tokens):
                    break
            end = time.perf_counter()
        finally:
            del g
        stop = 'eos' if tokens and tokens[-1] in self.eos else 'fim_stop' if tokens and tokens[-1] in self.stops else ('max_tokens' if len(tokens) == LIMIT else 'runtime_done')
        content = [t for t in tokens if t not in self.eos | self.stops]
        result = measurement(tokens, self.tok.decode(content), start, first, end, stop, len(ids))
        result['content_tokens_before_suffix_trim'] = len(content)
        result['generator_creation_ms'] = creation_ms
        return trim_suffix(result, item)

def measurement(tokens, text, start, first, end, stop, prompt_count):
    decode_s = end - first if first else 0
    return {'generated_ids': tokens, 'generated_text': text, 'prompt_tokens': prompt_count,
            'generated_tokens': len(tokens), 'ttft_ms': (first - start) * 1000 if first else None,
            'elapsed_s': end - start, 'decode_tokens_per_second': (len(tokens) - 1) / decode_s if len(tokens) > 1 and decode_s > 0 else None,
            'stop_reason': stop, 'eos': stop == 'eos'}

def run(model, corpus, backend, out, device='cpu', config=None, expected_gpu=None):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wall_start = time.perf_counter()
    report = {'schema': 'kapi-snapdragon-sample-v1', 'backend': backend, 'model': str(model),
              'timing_convention': 'v2: request starts before Generator construction; ends before final decode; decode rate excludes first token and includes stop-check decoding',
              'runner_version': 2, 'source_runner_commit': source_commit(),
              'source_runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'context_tokens': CONTEXT, 'max_new_tokens': LIMIT, 'greedy': True,
              'machine': platform.machine(), 'items': [], 'memory_before': memory()}
    failed = False
    def checkpoint(phase, **fields):
        report.update(phase=phase, wall_elapsed_s=time.perf_counter() - wall_start, **fields)
        out.with_name(out.name + '.progress.json').write_text(json.dumps(report, ensure_ascii=False), encoding='utf-8')
    checkpoint('validation_start')
    try:
        items = load_corpus(corpus)
        report['corpus_sha256'] = hashlib.sha256(Path(corpus).read_bytes()).hexdigest()
        recipe = json.loads(Path(config).read_text(encoding='utf-8')) if config else {}
        qnn_options = recipe.get('qnn_probe', {}).get('provider_options', {})
        checkpoint('model_load_start', total=len(items))
        load_start = time.perf_counter()
        runner = HFRunner(str(model), device, expected_gpu) if backend == 'hf' else GenAIRunner(model, backend, out.resolve().with_name(out.name + '.profile-' + uuid.uuid4().hex), qnn_options)
        checkpoint('model_load_done', model_load_elapsed_s=time.perf_counter() - load_start)
        report['provider_evidence'] = runner.evidence
        for item in items:
            row = {'id': item['id'], 'reference': item.get('reference', item.get('completion', item.get('target'))), 'suffix': item.get('suffix', '')}
            try:
                checkpoint('generator_start', current_item_id=item['id'])
                row.update(runner.generate(item))
            except Exception as exc:
                row['error'] = f'{type(exc).__name__}: {exc}'
                failed = True
            row['memory'] = memory()
            report['items'].append(row)
            if len(report['items']) % 10 == 0 or len(report['items']) == len(items):
                progress = {**report, 'status': 'running', 'processed': len(report['items']), 'total': len(items)}
                progress_path = out.with_name(out.name + '.progress.json')
                progress_path.write_text(json.dumps(progress, ensure_ascii=False), encoding='utf-8')
                print(f"Processed {len(report['items'])}/{len(items)} requests", flush=True)
        if hasattr(runner, 'close'):
            runner.close()
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        failed = True
    report['memory_after'] = memory()
    report['status'] = 'failed' if failed else 'completed'
    checkpoint('failed' if failed else 'completed')
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return 1 if failed else 0

def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'corpus', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--backend', choices=('hf', 'dml', 'qnn', 'cpu'), required=True)
    p.add_argument('--device', default='cpu', help='HF device: cpu or cuda:0 (ROCm uses the CUDA API)')
    p.add_argument('--config', type=Path, help='Optional recipe; QNN provider options from qnn_probe.provider_options')
    p.add_argument('--expected-gpu', help='Require selected HF CUDA/HIP device name to contain this model name')
    a = p.parse_args()
    raise SystemExit(run(a.model, a.corpus, a.backend, a.out, a.device, a.config, a.expected_gpu))

if __name__ == '__main__':
    main()
