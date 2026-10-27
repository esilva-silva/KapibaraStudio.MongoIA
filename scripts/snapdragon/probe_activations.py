"""CPU prefill activation probe; writes a temporary instrumented graph and JSON report."""
import argparse
import json
import os
import time
from pathlib import Path, PurePosixPath


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--corpus', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--work-dir', required=True, type=Path)
    p.add_argument('--kv-heads', required=True, type=int)
    p.add_argument('--head-size', required=True, type=int)
    p.add_argument('--threads', type=int, default=6)
    a = p.parse_args()
    if a.kv_heads <= 0 or a.head_size <= 0 or a.threads <= 0:
        p.error('KV dimensions and threads must be positive')
    import numpy as np
    import onnx
    import onnxruntime as ort
    model_path = a.model.resolve(strict=True)
    work = a.work_dir.resolve()
    if work.exists() or work.is_relative_to(model_path.parent) or model_path.parent.is_relative_to(work):
        p.error('work-dir must be a new separate directory')
    model = onnx.load(str(model_path), load_external_data=False)
    work.mkdir(parents=True)
    for tensor in model.graph.initializer:
        meta = {v.key: v.value for v in tensor.external_data}
        if 'location' not in meta: continue
        rel = PurePosixPath(meta['location'].replace('\\', '/'))
        if rel.is_absolute() or '..' in rel.parts: raise ValueError('Unsafe external file path')
        src = (model_path.parent / Path(*rel.parts)).resolve(strict=True)
        if not src.is_relative_to(model_path.parent): raise ValueError('External file outside model directory')
        dest = work / Path(*rel.parts)
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.link(src, dest)  # read-only use; never write these hard-linked files
    existing = {v.name for v in model.graph.output}; nodes = {}
    operators = {'SimplifiedLayerNormalization', 'SkipSimplifiedLayerNormalization', 'MatMul', 'Add',
                 'GroupQueryAttention', 'RotaryEmbedding', 'Mul', 'Sigmoid'}
    for node in model.graph.node:
        if node.op_type not in operators: continue
        for index in ([0, 3] if node.op_type == 'SkipSimplifiedLayerNormalization' and len(node.output) > 3 else [0]):
            name = node.output[index]
            if name and name not in existing:
                model.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT16, None))
                existing.add(name); nodes[name] = node.op_type
    probe = work / 'probe.onnx'
    onnx.save(model, str(probe))
    options = ort.SessionOptions(); options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.intra_op_num_threads = a.threads
    session = ort.InferenceSession(str(probe), options, providers=['CPUExecutionProvider'])
    row = json.loads(a.corpus.read_text(encoding='utf-8').splitlines()[0]); ids = row['prompt_ids']; length = len(ids)
    if not length: raise ValueError('Prompt must contain token IDs')
    feeds = {}
    for item in session.get_inputs():
        if item.name == 'input_ids': value = np.array([ids], dtype=np.int64)
        elif item.name == 'attention_mask': value = np.ones((1, length), dtype=np.int64)
        elif item.name == 'position_ids': value = np.arange(length, dtype=np.int64)[None, :]
        elif item.name.startswith('past_key_values.'):
            value = np.empty((1, a.kv_heads, 0, a.head_size), dtype=np.float16)
        else: raise ValueError('Unsupported model input: ' + item.name)
        feeds[item.name] = value
    start = time.perf_counter(); values = session.run(None, feeds); outputs = []
    for info, value in zip(session.get_outputs(), values):
        finite = np.isfinite(value)
        record = {'name': info.name, 'op': nodes.get(info.name), 'shape': list(value.shape),
            'nonfinite': int(value.size - np.count_nonzero(finite)),
            'finite_abs_max': float(np.max(np.abs(value[finite]))) if np.any(finite) else None}
        if info.name == 'logits':
            record['last_argmax'] = int(np.argmax(value[0, -1])); record['last_nonfinite'] = int(np.count_nonzero(~finite[0, -1]))
        outputs.append(record)
    report = {'schema': 'kapi-onnx-prefill-probe-v1', 'model': str(model_path), 'id': row.get('id'),
              'provider': 'CPUExecutionProvider', 'elapsed_s': time.perf_counter() - start, 'outputs': outputs}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'first_nonfinite': next((x for x in outputs if x['op'] and x['nonfinite']), None),
                      'logits': next((x for x in outputs if x['name'] == 'logits'), None)}, indent=2))


if __name__ == '__main__': main()
