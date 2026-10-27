import argparse
import copy
import hashlib
import json
from pathlib import Path

from export_gpu import source_provenance, sha256

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / 'configs/snapdragon/qnn-upstream.json'
PROVENANCE = ROOT / 'configs/snapdragon/qnn-upstream-source.json'


def generate(source, calibration, qnn_python, out_dir, phase, input_model_config=None, soc_model=None, gptq_device='cuda'):
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
            data['task'] = 'text-generation'
            data['pre_process_data_config']['model_name'] = str(source)
            data['load_dataset_config'] = {'data_name': 'json', 'data_files': str(data_path), 'split': 'train'}
            data['pre_process_data_config']['max_seq_len'] = 2048
            data['pre_process_data_config']['text_cols'] = ['text']
        config['passes'] = {k: v for k, v in config['passes'].items() if k not in ('cb', 'cp')}
        config['passes']['g']['device'] = gptq_device
        config['passes']['sq']['calibration_providers'] = ['CPUExecutionProvider']
        mb = config['passes']['mb']
        for old, new in [('block_size', 'int4_block_size'), ('accuracy_level', 'int4_accuracy_level'), ('op_types_to_quantize', 'int4_op_types_to_quantize')]:
            mb[new] = mb.pop(old)
        mb['search'] = {'max_length': 2048}
        config.pop('target', None)
        config['systems'] = {}
    else:
        config['input_model'] = model_config
        config['data_configs'] = []
        config['passes'] = {k: v for k, v in config['passes'].items() if k in ('cb', 'cp')}
        provider_options = config['passes']['cb']['provider_options']
        provider_options.pop('soc_model', None)
        if soc_model is not None:
            provider_options['soc_model'] = str(soc_model)
    config_path = out_dir / f'{phase}.json'
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    manifest = {'source': provenance, 'recipe': expected, 'calibration_sha256': sha256(calibration),
                'phase': phase, 'gptq_device': gptq_device, 'calibration_provider': 'CPUExecutionProvider',
                'changes': ['local calibration', 'CPU activation calibration', 'ModelBuilder Olive 0.13 option names', 'context 2048', 'split preparation and compilation'],
                'status': 'generated_not_executed', 'config': str(config_path)}
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
    p.add_argument('--soc-model')
    p.add_argument('--gptq-device', choices=('cuda', 'cpu'), default='cuda')
    a = p.parse_args()
    try:
        generate(a.source, a.calibration, a.qnn_python, a.out_dir, a.phase, a.input_model_config, a.soc_model, a.gptq_device)
    except (ValueError, OSError, KeyError) as exc:
        p.exit(1, f'Recipe generation failed: {exc}\n')


if __name__ == '__main__':
    main()
