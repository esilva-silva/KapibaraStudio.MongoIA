"""Read-only ONNX external-data and FP16 weight audit; never creates a session."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import struct

def source_index(root):
    result = {}
    for path in sorted(Path(root).glob('*.safetensors')):
        with path.open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            header = json.loads(stream.read(length))
        for name, info in header.items():
            if name != '__metadata__':
                result[name] = (path, 8 + length, info)
    return result

def source_name(name):
    name = name.replace('.attn.', '.self_attn.')
    name = name.replace('.MatMul.weight', '.weight').replace('.Add.bias', '.bias')
    return name

def audit(model_path, source=None, chunk_elements=1048576):
    import numpy as np
    import onnx
    model_path = Path(model_path).resolve()
    model = onnx.load(str(model_path), load_external_data=False)
    original = source_index(source) if source else {}
    rows, spans, errors = [], {}, []
    for tensor in model.graph.initializer:
        row = {'name': tensor.name, 'shape': list(tensor.dims), 'dtype': tensor.data_type}
        ext = {entry.key: entry.value for entry in tensor.external_data}
        if not ext:
            continue
        target = (model_path.parent / ext['location']).resolve()
        offset, length = int(ext.get('offset', '0')), int(ext.get('length', '0'))
        itemsize = {1: 4, 10: 2, 16: 2, 7: 8, 6: 4}.get(tensor.data_type)
        expected = math.prod(tensor.dims) * itemsize if itemsize else None
        row.update(file=str(target), offset=offset, length=length, exceeds_signed_int32=offset >= 2**31)
        if not target.is_relative_to(model_path.parent) or not target.is_file() or offset < 0 or length < 0 or offset + length > target.stat().st_size or (expected is not None and expected != length):
            row['error'] = 'External range or tensor byte length invalid'
            errors.append(tensor.name)
            rows.append(row)
            continue
        spans.setdefault(str(target), []).append((offset, offset + length, tensor.name))
        if tensor.data_type not in (1, 10):
            rows.append(row)
            continue
        data = np.memmap(target, mode='r', dtype='<f2' if tensor.data_type == 10 else '<f4', offset=offset, shape=(math.prod(tensor.dims),))
        stats = {'nan': 0, 'positive_inf': 0, 'negative_inf': 0, 'zero': 0, 'abs_max': 0.0}
        for start in range(0, len(data), chunk_elements):
            block = np.asarray(data[start:start + chunk_elements], dtype=np.float32)
            stats['nan'] += int(np.isnan(block).sum())
            stats['positive_inf'] += int(np.isposinf(block).sum())
            stats['negative_inf'] += int(np.isneginf(block).sum())
            stats['zero'] += int((block == 0).sum())
            finite = block[np.isfinite(block)]
            if len(finite):
                stats['abs_max'] = max(stats['abs_max'], float(np.abs(finite).max()))
        row['stats'] = stats
        key = source_name(tensor.name)
        if key in original:
            path, base, info = original[key]
            source_dtype = {'BF16': '<u2', 'F16': '<f2', 'F32': '<f4'}.get(info['dtype'])
            shape = tuple(info['shape'])
            transpose = len(shape) == 2 and '.MatMul.weight' in tensor.name
            expected_shape = tuple(reversed(shape)) if transpose else shape
            row['source'] = {'name': key, 'dtype': info['dtype'], 'shape': shape, 'transposed': transpose}
            if expected_shape == tuple(tensor.dims) and source_dtype:
                src = np.memmap(path, mode='r', dtype=source_dtype, offset=base + info['data_offsets'][0], shape=shape)
                # Deterministic bounded sample spanning the whole tensor, including >2 GiB offsets.
                ix = np.unique(np.linspace(0, len(data) - 1, min(4096, len(data)), dtype=np.int64))
                if transpose:
                    sample = src[ix % shape[0], ix // shape[0]]
                else:
                    sample = src.reshape(-1)[ix]
                if info['dtype'] == 'BF16':
                    sample = (sample.astype(np.uint32) << 16).view(np.float32)
                with np.errstate(over='ignore', invalid='ignore'):
                    expected_sample = sample.astype(data.dtype)
                actual = data[ix]
                row['source_comparison'] = {'sample_elements': len(ix), 'exact_matches': int((actual == expected_sample).sum()),
                                            'expected_cast_nonfinite': int((~np.isfinite(expected_sample)).sum()),
                                            'max_abs_error': float(np.max(np.abs(actual.astype(np.float32) - expected_sample.astype(np.float32))))}
                del src
            else:
                row['source_comparison'] = {'error': 'shape or dtype mapping mismatch'}
        del data
        rows.append(row)
    overlaps = []
    for path, ranges in spans.items():
        ranges.sort()
        for left, right in zip(ranges, ranges[1:]):
            if right[0] < left[1] and right[:2] != left[:2]:
                overlaps.append({'file': path, 'left': left, 'right': right})
    return {'schema': 'kapi-fp16-weight-audit-v1', 'model': str(model_path), 'no_inference': True,
            'invalid_ranges': errors, 'partial_overlaps': overlaps, 'tensors': rows,
            'nonfinite_tensor_names': [r['name'] for r in rows if any(r.get('stats', {}).get(k, 0) for k in ('nan', 'positive_inf', 'negative_inf'))]}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path, help='ONNX model path')
    parser.add_argument('--source', type=Path, help='Original safetensors snapshot for sampled identity check')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.model, args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=True) + '\n', encoding='utf-8')
    print(json.dumps({'invalid_ranges': len(report['invalid_ranges']), 'partial_overlaps': len(report['partial_overlaps']), 'nonfinite_tensors': report['nonfinite_tensor_names']}))

if __name__ == '__main__':
    main()
