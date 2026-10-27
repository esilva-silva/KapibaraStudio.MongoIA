"""Prepare a separate validated QNN release copy; never uploads or runs inference."""
import argparse
import json
from pathlib import Path
import shutil

from download_qnn_prepared import file_inventory
from export_gpu import source_provenance, sha256, TOKEN_FILES
from run_sample import pipeline_stages, stage_providers
from verify_manifest import verify
from summarize_pipeline_profiles import summarize

COMMON_CORPUS = '43b6eb37732209fb89747f309477c651d11fb6eb0d22b9d23a007a90b2793bc8'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def validate_run(report, package, count):
    require(Path(report['model']).resolve() == package, 'Report model mismatch')
    require(report.get('status') == 'completed' and not report.get('error'), 'Run not completed')
    require(report.get('backend') == 'qnn' and report.get('greedy') is True and
            report.get('context_tokens') == 2048 and report.get('max_new_tokens') == 32, 'Run generation contract mismatch')
    items = report['items']
    require(len(items) == count and report.get('total') == count, 'Run count mismatch')
    require(all(not item.get('error') and type(item.get('generated_tokens')) is int and
                0 < item['generated_tokens'] <= 32 and len(item['generated_ids']) == item['generated_tokens']
                for item in items), 'Run item failure/token count mismatch')
    evidence = report.get('provider_evidence') or {}
    require(evidence.get('verified') is True and evidence.get('requested') == 'QNNExecutionProvider' and
            evidence.get('kernel_events_by_provider', {}).get('QNNExecutionProvider', 0) > 0, 'QNN evidence missing')


def validate_stages(summary, run_path):
    require(summary.get('report_sha256') == sha256(run_path) and Path(summary['report']).resolve() == Path(run_path).resolve(), 'Stage report hash/path mismatch')
    require(summary.get('status') == 'verified' and not summary.get('errors'), 'Stages not verified')
    stages = {stage['name']: stage for stage in summary['stages']}
    require(set(stages) == {'embeddings', 'context', 'iterator', 'lm_head'} and len(summary['stages']) == 4, 'Stage inventory mismatch')
    for name in ('context', 'iterator'):
        stage = stages[name]; counts = stage['kernel_events_by_provider']
        require(stage.get('valid') is True and stage.get('intended_providers') == ['qnn'] and
                counts.get('QNNExecutionProvider', 0) > 0 and counts.get('CPUExecutionProvider', 0) == 0, 'Transformer stage fallback/evidence mismatch')
    for name in ('embeddings', 'lm_head'):
        require(stages[name].get('intended_providers') == ['cpu'] and stages[name].get('valid') is True, 'CPU stage evidence mismatch')


def validate_compile_report(compile_report, package, expected_sha256):
    require(sha256(compile_report) == expected_sha256, 'Compile report SHA mismatch')
    report = read(compile_report)
    require(report.get('status') == 'compiled_not_inference_validated' and
            report.get('cpu_fallback_disabled') is True and report.get('device_type') == 'NPU', 'Compile report contract mismatch')
    for entry in file_inventory(report['artifacts']):
        path = package / entry['path']
        require(path.resolve().is_relative_to(package) and path.is_file() and
                path.stat().st_size == entry['bytes'] and sha256(path) == entry['sha256'], 'Compiled runtime artifact mismatch')
    require(set(report.get('versions', {})) >= {'onnxruntime', 'onnxruntime-qnn', 'onnx'}, 'Compile versions missing')
    return report


def prepare(package, run, comparison, stage_summary, boundary, corpus, out_dir, compile_report, runtime_lock, boundary_corpus):
    package = Path(package).resolve(strict=True); output = Path(out_dir).resolve()
    evidence_paths = [Path(p).resolve(strict=True) for p in (run, comparison, stage_summary, boundary, corpus, boundary_corpus)]
    require(not output.exists() and not output.is_relative_to(package) and not package.is_relative_to(output), 'Output must be new and disjoint')
    manifest = read(package / 'export_manifest.json')
    entries = file_inventory(manifest['files'])
    require({p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file()} ==
            {entry['path'] for entry in entries} | {'export_manifest.json'}, 'Package inventory mismatch')
    require(verify(package, 'export_manifest.json')['passed'], 'Package integrity mismatch')
    compiled = validate_compile_report(compile_report, package, manifest['compile_report_sha256'])
    runtime_lock = Path(runtime_lock).resolve(strict=True)
    require(runtime_lock.is_file() and runtime_lock.stat().st_size > 0, 'Runtime lock missing/empty')
    runtime_metadata = {'architecture': 'win-arm64', 'compile_versions': compiled['versions'],
                        'runtime_lock_sha256': sha256(runtime_lock),
                        'scope': 'Compile report records compiler versions; lock records intended inference dependencies, not observed loaded runtime versions'}
    provenance = manifest['source_provenance']; raw = source_provenance(provenance['path'])
    require(provenance['repo_id'] == raw['repo_id'] and provenance['revision'] == raw['revision'] and
            file_inventory(provenance['files']) == file_inventory(raw['files']), 'Original source provenance mismatch')
    for name in TOKEN_FILES:
        original = Path(provenance['path']) / name
        if original.is_file():
            require((package / name).is_file() and sha256(original) == sha256(package / name), 'Original tokenizer mismatch')
    config = read(package / 'genai_config.json'); decoder = config['model']['decoder']
    require(config['model']['context_length'] == 2048 and decoder['sliding_window']['window_size'] == 32, 'Context/window mismatch')
    require(stage_providers({'session_options': decoder.get('session_options', {})}) == ['cpu'], 'Decoder default must be CPU')
    stages = dict(pipeline_stages(decoder))
    require(set(stages) == {'embeddings', 'context', 'iterator', 'lm_head'}, 'Pipeline inventory mismatch')
    for name, stage in stages.items():
        require(stage_providers(stage) == (['qnn'] if name in ('context', 'iterator') else ['cpu']), 'Pipeline provider mismatch')
        if name in ('context', 'iterator'):
            options = stage['session_options']
            require(options.get('disable_cpu_ep_fallback') == '1' and
                    any(entry.get('qnn', {}).get('backend_type') == 'htp' for entry in options['provider_options']), 'HTP/fallback configuration mismatch')
    result = read(run); validate_run(result, package, 200)
    corpus_hash = sha256(corpus)
    require(corpus_hash == COMMON_CORPUS and result['corpus_sha256'] == corpus_hash, 'Corpus hash mismatch')
    ids = [json.loads(line)['id'] for line in Path(corpus).read_text(encoding='utf-8').splitlines() if line.strip()]
    require(len(ids) == 200 and len(set(ids)) == 200 and [item['id'] for item in result['items']] == ids, 'Corpus IDs/order mismatch')
    compared = read(comparison)
    require(compared.get('proxy') is True and compared.get('qualified') is False and compared.get('corpus_sha256') == corpus_hash and
            compared['candidate']['n'] == 200 and compared['baseline']['n'] == 200, 'Comparison contract mismatch')
    summary = read(stage_summary); validate_stages(summary, run)
    require(summarize(run) == summary, 'Stage summary differs from actual profiles')
    edge = read(boundary); validate_run(edge, package, 1)
    boundary_rows = [json.loads(line) for line in Path(boundary_corpus).read_text(encoding='utf-8').splitlines() if line.strip()]
    require(edge.get('corpus_sha256') == sha256(boundary_corpus) and len(boundary_rows) == 1 and
            boundary_rows[0]['id'] == edge['items'][0]['id'] and len(boundary_rows[0]['prompt_ids']) == 2016, 'Boundary corpus hash/IDs/token length mismatch')
    item = edge['items'][0]
    require(all(item.get(key) == value for key, value in {'prompt_tokens': 2016, 'padded_prompt_tokens': 2016,
            'generated_tokens': 32, 'max_length_requested': 2048, 'minimum_length_for_padding': 2048}.items()), 'Boundary 2016+32 mismatch')
    boundary_summary = summarize(boundary); validate_stages(boundary_summary, boundary)
    require(not (package / 'validation').exists() and not (package / 'VALIDATION.md').exists(), 'Package already contains release evidence')
    # All input invariants are checked before creating the output.
    shutil.copytree(package, output)
    for entry in entries:
        require(sha256(output / entry['path']) == entry['sha256'].lower(), 'Copied runtime integrity mismatch')
    validation = output / 'validation'; validation.mkdir()
    shutil.copy2(compile_report, validation / 'compile-report.json')
    shutil.copy2(runtime_lock, validation / 'snapdragon-arm64.lock.txt')
    (validation / 'runtime-metadata.json').write_text(json.dumps(runtime_metadata, indent=2) + '\n', encoding='utf-8')
    for index, path in enumerate(evidence_paths):
        shutil.copy2(path, validation / f'{index:02d}-{path.name}')
    for label, profile_summary in (('sample', summary), ('boundary', boundary_summary)):
        for stage in profile_summary['stages']:
            shutil.copy2(stage['profile'], validation / f'{label}-{stage["index"]}-{Path(stage["profile"]).name}')
    (validation / 'boundary-stage-summary.json').write_text(json.dumps(boundary_summary, indent=2) + '\n', encoding='utf-8')
    model_name = provenance['repo_id'].split('/')[-1]
    (output / 'VALIDATION.md').write_text(f'# Experimental {model_name} QNN release\n\n200 common corpus cases and the 2016+32 boundary passed with window 32, greedy generation and context 2048.\n\nproxy: true; qualified: false. Generated JavaScript was not executed; no IDE qualification. Embeddings and lm_head run on CPU; context and iterator use QNN HTP with CPU fallback disabled. This is a mixed pipeline. ORT kernel profiles establish provider execution; full-model QNN CSV timings are not established.\n\nTarget architecture: win-arm64. validation/compile-report.json records the compiler versions of ONNX Runtime, QNN plugin and ONNX. validation/snapdragon-arm64.lock.txt records intended inference dependencies. These do not establish the versions actually loaded by the inference process; this packaging tool does not inspect or install the GenAI runtime. See validation/runtime-metadata.json for the lock hash and scope.\n\nPublication: not published.\n', encoding='utf-8')
    manifest.update(status='sample_and_boundary_validated', publication='not_published', proxy=True, qualified=False,
                    validation={'sample_count': 200, 'boundary_2016_plus_32': True, 'corpus_sha256': corpus_hash,
                                'boundary_corpus_sha256': sha256(boundary_corpus), 'runtime': runtime_metadata})
    manifest['files'] = [{'path': path.relative_to(output).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha256(path)}
                         for path in sorted(output.rglob('*')) if path.is_file() and path != output / 'export_manifest.json']
    (output / 'export_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    require(verify(output, 'export_manifest.json')['passed'], 'Release verification failed')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'run', 'comparison', 'stage-summary', 'boundary', 'corpus', 'out-dir', 'compile-report', 'runtime-lock', 'boundary-corpus'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    try:
        manifest = prepare(args.package, args.run, args.comparison, args.stage_summary, args.boundary, args.corpus, args.out_dir,
                           args.compile_report, args.runtime_lock, args.boundary_corpus)
    except Exception as exc:
        parser.exit(1, f'Release preparation failed: {type(exc).__name__}: {exc}\n')
    print(json.dumps({'status': manifest['status'], 'publication': manifest['publication']}))


if __name__ == '__main__':
    main()
