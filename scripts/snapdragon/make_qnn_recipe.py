import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

from export_gpu import source_provenance, sha256

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / 'configs/snapdragon/qnn-upstream.json'
PROVENANCE = ROOT / 'configs/snapdragon/qnn-upstream-source.json'


def rebase_composite(model, artifact_root, original_artifact_root=None):
    """Rebase Olive CompositeModel resource paths while preserving model metadata."""
    if not artifact_root:
        raise ValueError('compile requires --artifact-root')
    root = Path(artifact_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('artifact root must be a directory')
    original = str(original_artifact_root or root)
    windows = bool(PureWindowsPath(original).drive)
    pure = PureWindowsPath if windows else PurePosixPath
    old = pure(original)
    if not old.is_absolute() or '..' in old.parts:
        raise ValueError('original artifact root must be an absolute canonical path')

    def resource(value):
        if not isinstance(value, str) or not value:
            raise ValueError('Invalid model resource path')
        path = pure(value)
        if '..' in path.parts:
            raise ValueError('Resource path contains parent traversal')
        if path.is_absolute():
            try:
                relative = path.relative_to(old)
            except ValueError as exc:
                raise ValueError('Resource is outside original artifact root: ' + value) from exc
        else:
            if path.drive:
                raise ValueError('Invalid resource path: ' + value)
            relative = path
        dest = root.joinpath(*relative.parts).resolve()
        if not dest.is_relative_to(root) or not dest.exists():
            raise ValueError('Resource missing or outside artifact root: ' + value)
        return str(dest)

    model = copy.deepcopy(model)
    if str(model.get('type', '')).lower() != 'compositemodel':
        raise ValueError('compile requires Olive CompositeModel from StaticLLM')
    def visit(obj):
        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                if key == 'model_path' and value is not None:
                    obj[key] = resource(value)
                elif key == 'additional_files' and value is not None:
                    if not isinstance(value, list):
                        raise ValueError('additional_files must be a list')
                    obj[key] = [resource(item) for item in value]
                else:
                    visit(value)
            if 'onnx_file_name' in obj and obj['onnx_file_name']:
                filename = pure(obj['onnx_file_name'])
                if filename.is_absolute() or '..' in filename.parts or filename.drive:
                    raise ValueError('Invalid onnx_file_name')
                base = Path(obj.get('model_path', root))
                target = (base / Path(*filename.parts)).resolve()
                if not target.is_relative_to(root) or not target.is_file():
                    raise ValueError('ONNX component missing or outside artifact root')
        elif isinstance(obj, list):
            for value in obj:
                visit(value)
    visit(model)
    return model


def generate(source, calibration, qnn_python, out_dir, phase, input_model_config=None, soc_model=None, gptq_device='cuda', artifact_root=None, original_artifact_root=None, prepare_python=None):
    source = Path(source).resolve()
    provenance = source_provenance(source)
    expected = json.loads(PROVENANCE.read_text(encoding='utf-8'))
    if sha256(TEMPLATE) != expected['sha256']:
        raise ValueError('Vendored recipe hash mismatch')
    template = json.loads(TEMPLATE.read_text(encoding='utf-8'))
    calibration = Path(calibration).resolve()
    texts = []
    for number, line in enumerate(calibration.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        value = row.get('prompt') or row.get('prompt_text')
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'Missing prompt or prompt_text at line {number}')
        texts.append(value)
    if not texts:
        raise ValueError('Calibration data is empty')
    if phase == 'compile':
        if not input_model_config:
            raise ValueError('compile requires --input-model-config')
        model_config = json.loads(Path(input_model_config).read_text(encoding='utf-8'))
        if not isinstance(model_config, dict) or not model_config.get('type') or not isinstance(model_config.get('config'), dict):
            raise ValueError('Expected Olive serialized model config with type and config')
        model_config = rebase_composite(model_config, artifact_root, original_artifact_root)
    if soc_model is not None and (not str(soc_model).isdigit() or int(soc_model) < 1):
        raise ValueError('soc-model must be a positive numeric identifier')
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / 'calibration.jsonl'
    data_path.write_text(''.join(json.dumps({'text': text}, ensure_ascii=False) + '\n' for text in texts), encoding='utf-8')
    config = copy.deepcopy(template)
    config['systems']['qnn_system']['python_environment_path'] = str(Path(qnn_python).resolve())
    config['output_dir'] = str(out_dir / 'output')
    config['cache_dir'] = str(out_dir / 'cache')
    config['no_artifacts'] = False
    if phase == 'prepare':
        config['input_model']['model_path'] = str(source)
        for data in config['data_configs']:
            # Olive 0.13 DataConfig ignores a top-level task. Select the component explicitly.
            data.pop('task', None)
            data['pre_process_data_config']['type'] = 'text_generation_huggingface_pre_process'
            data['pre_process_data_config']['model_name'] = str(source)
            data['load_dataset_config'] = {'data_name': 'json', 'data_files': str(data_path), 'split': 'train'}
            data['pre_process_data_config']['max_seq_len'] = 2048
            data['pre_process_data_config']['text_cols'] = ['text']
        config['passes'] = {k: v for k, v in config['passes'].items() if k not in ('cb', 'cp')}
        config['passes']['g']['device'] = gptq_device
        config['passes']['sq']['calibration_providers'] = ['CPUExecutionProvider']
        model_metadata = json.loads((source / 'config.json').read_text(encoding='utf-8'))
        head_dim = model_metadata.get('head_dim')
        if head_dim is None:
            hidden = model_metadata.get('hidden_size')
            heads = model_metadata.get('num_attention_heads')
            if type(hidden) is not int or type(heads) is not int or hidden <= 0 or heads <= 0 or hidden % heads:
                raise ValueError('Cannot prove KV head dimension from source config')
            head_dim = hidden // heads
        if type(head_dim) is not int or head_dim <= 0:
            raise ValueError('KV head dimension must be a positive integer')
        # Olive's calibration loader allocates KV tensors from concrete axes 1/3.
        # GenAI may leave head size symbolic; fix it without freezing past length.
        passes = {}
        for name, settings in config['passes'].items():
            if name == 'sq':
                passes['kv_shape'] = {'type': 'DynamicToFixedShape', 'dim_param': ['kv_cache_dim'],
                                      'dim_value': [head_dim], 'save_as_external_data': True}
            passes[name] = settings
        config['passes'] = passes
        mb = config['passes']['mb']
        for old, new in [('block_size', 'int4_block_size'), ('accuracy_level', 'int4_accuracy_level'), ('op_types_to_quantize', 'int4_op_types_to_quantize')]:
            mb[new] = mb.pop(old)
        mb['search'] = {'max_length': 2048}
        # The target accelerator selects the QNN-specific QDQ config; calibration
        # itself still executes on CPU through calibration_providers above.
        if prepare_python:
            config['systems']['qnn_system']['python_environment_path'] = str(Path(prepare_python).resolve())
    else:
        config['input_model'] = model_config
        config['data_configs'] = []
        config['passes'] = {k: v for k, v in config['passes'].items() if k in ('cb', 'cp')}
        config['passes']['cb']['disable_cpu_fallback'] = True
        provider_options = config['passes']['cb']['provider_options']
        provider_options.pop('soc_model', None)
        if soc_model is not None:
            provider_options['soc_model'] = str(soc_model)
    config_path = out_dir / f'{phase}.json'
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    manifest = {'source': provenance, 'recipe': expected, 'calibration_sha256': sha256(calibration),
                'phase': phase, 'gptq_device': gptq_device, 'calibration_provider': 'CPUExecutionProvider',
                'target_execution_provider': 'QNNExecutionProvider',
                'calibration_limits': {'gptq': {'strategy': 'join', 'max_samples': 128, 'max_seq_len': 2048},
                                       'activation': {'strategy': 'line-by-line', 'max_samples': 256, 'max_seq_len': 2048}},
                'changes': ['local calibration', 'CPU activation calibration', 'KV head axis from verified model metadata; past sequence stays dynamic', 'ModelBuilder Olive 0.13 option names', 'search.max_length 2048 (not validated as model capacity)', 'split preparation and compilation'],
                'status': 'generated_not_executed', 'context_validation': 'pending: inspect KV shapes and genai_config; run generation', 'config': str(config_path)}
    (out_dir / f'{phase}-manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return config


def main():
    p = argparse.ArgumentParser(description='Generate an experimental two-stage Olive QNN recipe, without running it')
    p.add_argument('--source', required=True)
    p.add_argument('--calibration', required=True)
    p.add_argument('--qnn-python', required=True, help='Olive PythonEnvironment directory containing the interpreter')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--phase', required=True, choices=('prepare', 'compile'))
    p.add_argument('--input-model-config')
    p.add_argument('--prepare-python', help='QNN-capable PythonEnvironment directory for prepare (defaults to --qnn-python)')
    p.add_argument('--artifact-root', help='Transferred Olive output directory (required for compile)')
    p.add_argument('--original-artifact-root', help='Original absolute output root, for cross-host rebasing')
    p.add_argument('--soc-model')
    p.add_argument('--gptq-device', choices=('cuda', 'cpu'), default='cuda')
    a = p.parse_args()
    try:
        generate(a.source, a.calibration, a.qnn_python, a.out_dir, a.phase, a.input_model_config, a.soc_model, a.gptq_device, a.artifact_root, a.original_artifact_root, a.prepare_python)
    except (ValueError, OSError, KeyError) as exc:
        p.exit(1, f'Recipe generation failed: {exc}\n')


if __name__ == '__main__':
    main()
