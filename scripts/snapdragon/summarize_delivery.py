"""Summarize local Snapdragon delivery evidence without changing its status."""
import argparse
import hashlib
import json
import re
from pathlib import Path, PureWindowsPath


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(root, name):
    path=root/'reports'/'snapdragon'/name
    return json.loads(path.read_text(encoding='utf-8-sig')), {'path':str(path),'sha256':sha(path)}


def validate_comparisons(comparisons, expected=200):
    hashes=set()
    for name, obj in comparisons.items():
        if obj.get('qualified') is not False or obj.get('proxy') is not True:
            raise ValueError(name+': expected unqualified proxy evidence')
        if obj['candidate'].get('n') != expected or obj['baseline'].get('n') != expected:
            raise ValueError(name+': incomplete comparison')
        corpus=obj.get('corpus_sha256')
        if not isinstance(corpus,str) or not re.fullmatch(r'[0-9a-f]{64}',corpus): raise ValueError(name+': invalid corpus hash')
        hashes.add(corpus)
    if len(hashes)!=1:raise ValueError('Comparison corpus hashes do not match')
    return hashes.pop()


def successful_run(run, corpus, provider, expected_ids=None):
    items=run.get('items') or []
    return (run.get('status')=='completed' and run.get('corpus_sha256')==corpus and len(items)==200
        and len({x.get('id') for x in items})==200 and not any(x.get('error') for x in items)
        and (run.get('provider_evidence') or {}).get('requested')==provider
        and (run.get('provider_evidence') or {}).get('verified') is True
        and (expected_ids is None or [x.get('id') for x in items] == expected_ids)
        and all(type(x.get('generated_tokens')) is int and 0 <= x['generated_tokens'] <= 32 for x in items))


def boundary_ok(report, boundary_hash=None):
    rows=report.get('items') or []
    return (report.get('status')=='completed' and report.get('context_tokens')==2048 and report.get('max_new_tokens')==32 and report.get('greedy') is True and (boundary_hash is None or report.get('corpus_sha256')==boundary_hash) and len(rows)==1 and rows[0].get('prompt_tokens')==2016
        and rows[0].get('generated_tokens')==32 and not rows[0].get('error')
        and (report.get('provider_evidence') or {}).get('verified') is True)


def generation_protocol_ok(run):
    if run.get('executor') == 'memoryboundedreloads':
        batches = run.get('batches') or []
        if not batches: return False
        for batch in batches:
            report_path=Path(batch['report_path']); corpus_path=Path(batch['corpus_path'])
            if sha(report_path)!=batch['report_sha256'] or sha(corpus_path)!=batch['corpus_sha256']: return False
            child=json.loads(report_path.read_text(encoding='utf-8-sig'))
            if not generation_protocol_ok(child): return False
        return True
    return run.get('context_tokens')==2048 and run.get('max_new_tokens')==32 and run.get('greedy') is True


def read_dataset(root, config):
    directory=root/'data/calib/snapdragon-20261026'; manifest_path=directory/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    rows={}; audit={'manifest':{'path':str(manifest_path),'sha256':sha(manifest_path)}}
    for name,count in [('calibration',config['samples']['calibration_count']),('evaluation',config['samples']['evaluation_count'])]:
        entry=manifest['files'][name]; path=directory/entry['file']; digest=sha(path)
        data=[json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        if digest!=entry['sha256'] or len(data)!=count or entry['count']!=count:
            raise ValueError(name+' dataset hash/count mismatch')
        rows[name]=data; audit[name]={'path':str(path),'sha256':digest,'count':len(data)}
    for key in ('group_id','split_key','schema_id','prompt_sha256'):
        if {x[key] for x in rows['calibration']} & {x[key] for x in rows['evaluation']}:
            raise ValueError('Calibration/evaluation overlap: '+key)
    ids=[x['id'] for x in rows['evaluation']]
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate evaluation IDs')
    audit['disjoint_keys']=['group_id','split_key','schema_id','prompt_sha256']
    return audit,ids


def window32_files_ready(root, model='nano'):
    directory = root / 'reports/snapdragon'
    names = (f'{model}-npu-compiled-window32-200.json', f'{model}-npu-window32-vs-bf16.json',
             f'{model}-npu-window32-200-stage-summary.json', f'{model}-npu-window32-boundary-2048.json')
    return all((directory / name).is_file() for name in names)


def assess_window32(comparison, run, stage, boundary, corpus, ids, run_sha, boundary_hash):
    try:
        if validate_comparisons({'window32': comparison}) != corpus:
            return False
        if not successful_run(run, corpus, 'QNNExecutionProvider', ids) or not generation_protocol_ok(run):
            return False
        if stage.get('status') != 'verified' or stage.get('report_sha256') != run_sha:
            return False
        if not boundary_ok(boundary, boundary_hash):
            return False
        if (boundary.get('provider_evidence') or {}).get('requested') != 'QNNExecutionProvider':
            return False
        return True
    except (KeyError, ValueError, TypeError):
        return False


def private_release_ok(config, transfer, source_verification, smoke, evaluated_run, model='nano'):
    spec = config.get('npu_exports', {}).get(model) or {}
    try:
        local_manifest = Path(spec['local_path']) / 'export_manifest.json'
        release_hash = sha(local_manifest)
        downloaded = Path(source_verification['directory']).resolve()
        if sha(downloaded / 'export_manifest.json') != release_hash:
            return False
        smoke_rows = smoke.get('items') or []
        if not (spec.get('private') is True and transfer.get('private') is True
                and transfer.get('status') == 'verified' and transfer.get('repo_id') == spec['repo_id']
                and transfer.get('commit') == spec['revision'] and transfer.get('subfolder') == spec['subfolder']
                and transfer.get('manifest_sha256') == spec['manifest_sha256'] == release_hash
                and source_verification.get('passed') is True and source_verification.get('source_identity_verified') is True
                and source_verification.get('original_tokenizer_required') is True
                and source_verification.get('manifest_sha256') == release_hash
                and smoke.get('status') == 'completed' and Path(smoke['model']).resolve() == downloaded
                and (smoke.get('provider_evidence') or {}).get('requested') == 'QNNExecutionProvider'
                and (smoke.get('provider_evidence') or {}).get('verified') is True
                and len(smoke_rows) == 1 and smoke_rows[0].get('generated_tokens') == 32 and not smoke_rows[0].get('error')):
            return False
        release = json.loads(local_manifest.read_text(encoding='utf-8-sig'))
        evaluated = json.loads((Path(evaluated_run['model']) / 'export_manifest.json').read_text(encoding='utf-8-sig'))
        original = {x['path']: (x['sha256'], x['bytes']) for x in evaluated['files']}
        published = {x['path']: (x['sha256'], x['bytes']) for x in release['files']}
        return bool(original) and all(published.get(k) == v for k, v in original.items())
    except (KeyError, TypeError, OSError, ValueError):
        return False


def delivery_complete(delivery):
    return all(delivery.get(name + '_gpu', {}).get('status') == 'validated_200' and
               delivery[name + '_gpu'].get('boundary_2016_plus_32') is True and
               delivery.get(name + '_npu', {}).get('status') == 'validated_200_boundary_passed' and
               delivery[name + '_npu'].get('package', {}).get('release_identity_verified') is True
               for name in ('nano', 'small'))


def calibration_policy_evidence(spec, required, root):
    """Validate explicit actual-calibration receipts; a policy label is insufficient."""
    if spec.get('calibration_policy') != required:
        return False, 'Published calibration policy differs from required policy', {}
    evidence = {}
    try:
        reports = {}
        for key in ('activation_calibration_trace', 'gptq_runtime_receipt', 'gptq_completion_report'):
            path = Path(spec[key])
            if not path.is_absolute():
                path = root / path
            reports[key] = json.loads(path.read_text(encoding='utf-8-sig'))
            evidence[key] = {'path': str(path), 'sha256': sha(path)}
        trace = reports['activation_calibration_trace']
        fim = trace.get('fim_preflight') or {}; policy = fim.get('policy') or {}
        if not (fim.get('all_match') is True and fim.get('fim_complete') is True and fim.get('matched') == 256
                and policy.get('add_special_tokens') is False and policy.get('pad_to_max_len') is False
                and policy.get('batch_size') == 1 and trace.get('calibration_status') == 'input_complete'):
            return False, 'Actual activation FIM coverage/policy is incomplete', evidence
        receipt = reports['gptq_runtime_receipt']; batches = receipt.get('batches') or []
        if not (receipt.get('policy') == 'kapi_gptq_join_tail_v1' and receipt.get('all_prompt_tokens_covered') is True
                and receipt.get('selected_rows') == 256 and receipt.get('joined_tokens') == receipt.get('consumed_tokens') == 61273
                and receipt.get('add_special_tokens') is False and receipt.get('padding') == 'none'
                and receipt.get('batch_size') == 1 and receipt.get('shuffle') is False
                and [batch.get('tokens') for batch in batches] == [2048] * 29 + [1881]
                and receipt.get('stream_token_ids_sha256') == '07f7dec2ffed64e72f7d015bfd86bbce5d16f0297538c6fffd079fe9183da362'):
            return False, 'Actual GPTQ preprocessor full-stream receipt is incomplete', evidence
        job = reports['gptq_completion_report']; effective = job.get('effective_gptq_calibration') or {}
        normalized_complete = job.get('status') == 'completed'
        producer_complete = (job.get('status') == 'workflow_returned_artifact_validation_pending'
            and job.get('workflow_result_nonempty') is True
            and any(stage.get('pass') == 'GptqModel' and stage.get('status') == 'completed' for stage in job.get('stages', [])))
        if not ((normalized_complete or producer_complete) and effective.get('batches') == 30
                and effective.get('min_tokens') == 1881 and effective.get('max_tokens') == 2048):
            return False, 'GPTQ completion/effective batch evidence is incomplete', evidence
        for scope in ('quantize_entry', 'after_prepare_dataset'):
            observed = (job.get('runtime_calibration_audit') or {}).get(scope) or {}
            if not (observed.get('matched') is True and observed.get('batch_count') == 30
                    and observed.get('token_count') == 61273
                    and observed.get('stream_token_ids_sha256') == receipt['stream_token_ids_sha256']
                    and [batch.get('tokens') for batch in observed.get('batches', [])] == [2048] * 29 + [1881]):
                return False, 'GPTQ downstream token audit missing or incomplete: ' + scope, evidence
        return True, 'Actual activation trace, preprocessor receipt and both downstream GPTQ token audits match', evidence
    except (KeyError, OSError, ValueError, TypeError) as exc:
        return False, 'Required calibration evidence missing or invalid: ' + str(exc), evidence


def apply_calibration_policy_gate(delivery, config, root):
    required = config.get('required_npu_calibration_policy')
    if not required:
        return
    for model in ('nano', 'small'):
        key = model + '_npu'; variant = delivery[key]
        passed, reason, evidence = calibration_policy_evidence(config.get('npu_exports', {}).get(model) or {}, required, root)
        variant['calibration_policy_gate'] = {'required': required, 'passed': passed, 'reason': reason,
            'evidence': evidence, 'gptq_receipt_scope': 'Preprocessor receipt plus actual quantize_entry and after_prepare_dataset hashes/counts in the GPTQ completion report'}
        if not passed:
            variant['historical_status'] = variant['status']
            variant['status'] = 'calibration_revision_pending'


def evaluated_model_path(config, model):
    model_root = Path(config['models_root']).resolve()
    spec = config.get('npu_exports', {}).get(model) or {}
    value = spec.get('evaluated_path', str(model_root / f'KapiCoder-Mongo-{model.title()}-QNN-Compiled/mixed-htp-window32-v1'))
    if not isinstance(value, str) or not value:
        raise ValueError('evaluated_path must be a nonempty path')
    expected = Path(value)
    if not expected.is_absolute():
        expected = model_root / expected
    expected = expected.resolve()
    if not expected.is_relative_to(model_root) or expected == model_root:
        raise ValueError('evaluated_path must remain inside models_root')
    return expected


def summarize_npu32(root, config, model, corpus, ids, boundary_hash):
    names = {'comparison': f'{model}-npu-window32-vs-bf16.json',
             'run': f'{model}-npu-compiled-window32-200.json',
             'stage': f'{model}-npu-window32-200-stage-summary.json',
             'boundary': f'{model}-npu-window32-boundary-2048.json',
             'compile': f'{model}-qnn-htp-compile-window32.json',
             'transfer': f'{model}-npu-private-transfer-result.json',
             'source': f'{model}-npu-private-source-verification.json',
             'smoke': f'{model}-npu-private-download-smoke.json'}
    spec = config.get('npu_exports', {}).get(model) or {}
    customized = 'evidence_files' in spec
    if 'evaluated_path' in spec:
        try:
            evaluated_model_path(config, model)
        except (KeyError, ValueError, TypeError) as exc:
            return {'status': 'not_validated', 'error': 'Invalid evaluated_path: ' + str(exc)}, {}
    if customized:
        try:
            configured = spec['evidence_files']
            if not isinstance(configured, dict) or set(configured) != set(names):
                raise ValueError('evidence_files requires exactly all eight proof keys')
            validated = {}
            report_root = (root / 'reports/snapdragon').resolve()
            for key, value in configured.items():
                if not isinstance(value, str) or not value or '\\' in value or ':' in value:
                    raise ValueError('Evidence paths must be relative POSIX paths under reports/snapdragon')
                relative = Path(value)
                path = (root / relative).resolve()
                if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(report_root) or path == report_root:
                    raise ValueError('Evidence path outside reports/snapdragon')
                validated[key] = path.relative_to(report_root).as_posix()
            names = validated
        except (ValueError, TypeError) as exc:
            return {'status': 'not_validated', 'error': 'Invalid evidence_files: ' + str(exc)}, {}
    available = {key for key, name in names.items() if (root / 'reports/snapdragon' / name).is_file()}
    if not available and not customized:
        return (None if model == 'nano' else {'status': 'pending_remote_preparation'}), {}
    if available != set(names):
        try:
            for key in available:
                read(root, names[key])
        except (OSError, ValueError, TypeError) as exc:
            return {'status': 'not_validated', 'error': 'Invalid partial window32 evidence: ' + str(exc)}, {}
        return {'status': 'pending_evidence', 'missing': [names[key] for key in names if key not in available]}, {}
    sources = {}
    try:
        data = {}
        for key, name in names.items():
            data[key], sources[f'{model}_npu32_{key}'] = read(root, name)
        run = data['run']; comparison = data['comparison']
        expected_model = evaluated_model_path(config, model)
        passed = assess_window32(comparison, run, data['stage'], data['boundary'], corpus, ids,
                                 sources[f'{model}_npu32_run']['sha256'], boundary_hash)
        passed = passed and Path(run['model']).resolve() == expected_model.resolve() and Path(data['boundary']['model']).resolve() == expected_model.resolve()
        stage_names = {stage['name']: stage for stage in data['stage'].get('stages', [])}
        passed = passed and all(stage_names.get(name, {}).get('valid') is True and
                    stage_names[name].get('kernel_events_by_provider', {}).get('QNNExecutionProvider', 0) > 0 and
                    stage_names[name].get('kernel_events_by_provider', {}).get('CPUExecutionProvider', 0) == 0
                    for name in ('context', 'iterator'))
        manifest_path = Path(run['model']) / 'export_manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
        compiled = data['compile']
        passed = passed and manifest.get('compile_report_sha256') == sources[f'{model}_npu32_compile']['sha256'] and compiled.get('device_type') == 'NPU' and compiled.get('cpu_fallback_disabled') is True
        release_ok = private_release_ok(config, data['transfer'], data['source'], data['smoke'], run, model)
        passed = passed and release_ok
        return {'status': 'validated_200_boundary_passed' if passed else 'not_validated',
                'baseline': comparison.get('baseline'), 'candidate': comparison.get('candidate'),
                'stage_profile_status': data['stage'].get('status'), 'prefill_window': 32, 'cache_tokens': 2048,
                'boundary_2016_plus_32': boundary_ok(data['boundary'], boundary_hash),
                'package': {'path': str(expected_model), 'evaluated_package': run.get('model'),
                            'publication': 'private_verified' if release_ok else 'not_validated',
                            'release': config.get('npu_exports', {}).get(model), 'release_identity_verified': release_ok}}, sources
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {'status': 'not_validated', 'prefill_window': 32, 'error': 'Invalid window32 evidence: ' + str(exc)}, sources


def activation_trace_ok(trace, expected_count, head_size):
    if (trace.get('calibration_status') != 'input_complete'
            or trace.get('calibration_inputs_seen') != expected_count
            or trace.get('activation_type') != 'QUInt16'
            or trace.get('calibration_providers') != ['CPUExecutionProvider']):
        return False
    shapes = trace.get('first_input_shapes') or {}
    for name, info in shapes.items():
        if name.startswith('past_key_values.'):
            shape = info.get('shape')
            if not isinstance(shape, list) or len(shape) != 4 or shape[-1] != head_size:
                return False
    return True


def calibration_phases(root, config):
    summary, source = read(root, 'gptq-calibration-token-summary-rx7800xt.json')
    if summary.get('historical_runtime_token_hashes_captured') is not False:
        raise ValueError('Unexpected GPTQ reconstruction provenance')
    jobs = summary.get('jobs') or []
    phases = {'gptq_reconstruction': {'source': source,
        'historical_runtime_token_hashes_captured': False, 'scope': summary.get('scope'), 'jobs': jobs},
        'activation': {}}
    for model, head_size in [('nano',64), ('small',128)]:
        revision = config['models'][model]['revision']
        matched = [j for j in jobs if PureWindowsPath(j.get('source_snapshot','')).name.lower() == revision.lower()]
        if len(matched) != 1:
            raise ValueError('GPTQ reconstruction source revision mismatch: ' + model)
        trace_path = root / 'reports/snapdragon' / f'{model}-qnn-cpu-calibration-trace-rx7800xt.json'
        if not trace_path.is_file():
            phases['activation'][model] = {'status': 'pending', 'source_revision': revision}
            continue
        trace, evidence = read(root, trace_path.name)
        recipe_name = 'nano-qnn-prepare-v6-recipe-rx7800xt.json' if model == 'nano' else 'small-qnn-prepare-recipe-rx7800xt.json'
        recipe, recipe_evidence = read(root, recipe_name)
        configs = [c for c in recipe.get('data_configs', []) if c.get('name') == 'wikitext2_train_act']
        source_ok = (len(configs) == 1 and PureWindowsPath(configs[0]['pre_process_data_config'].get('model_name','')).name.lower() == revision.lower())
        valid = source_ok and activation_trace_ok(trace,config['samples']['calibration_count'],head_size)
        phases['activation'][model] = {'status': 'input_complete' if valid else 'not_validated',
            'source_revision': revision, 'expected_kv_head_size':head_size, 'trace':evidence, 'recipe':recipe_evidence, 'recipe_source_revision_matches':source_ok,
            'inputs_seen':trace.get('calibration_inputs_seen'), 'provider':trace.get('calibration_providers'),
            'activation_type':trace.get('activation_type')}
    return phases


def summarize(root):
    root=Path(root).resolve(strict=True)
    config_path=root/'configs/snapdragon.json';config=json.loads(config_path.read_text(encoding='utf-8-sig'))
    comparison_names={'nano_gpu':'nano-rmsnorm-split-vs-bf16-detailed.json',
        'small_gpu':'small-rmsnorm-split-batched-vs-bf16.json','nano_npu':'nano-npu-window64-vs-bf16.json'}
    comparisons={}; sources={'config':{'path':str(config_path),'sha256':sha(config_path)}}
    for key,name in comparison_names.items(): comparisons[key],sources[key+'_comparison']=read(root,name)
    corpus=validate_comparisons(comparisons,config['samples']['evaluation_count'])
    dataset,eval_ids=read_dataset(root,config)
    if corpus != dataset['evaluation']['sha256']: raise ValueError('Comparison hash does not match evaluation file')
    boundary_path=root/'tmp/snapdragon-boundary-2016.jsonl'
    boundary_hash=sha(boundary_path) if boundary_path.is_file() else None
    dataset['boundary']={'path':str(boundary_path),'sha256':boundary_hash,'exists':boundary_path.is_file()}
    file_names={'nano_gpu_run':'nano-gpu-rmsnorm-split-200.json',
        'small_gpu_run':'small-gpu-rmsnorm-split-batched-exe-200.json',
        'nano_npu_run':'nano-npu-compiled-window64-200.json',
        'nano_gpu_boundary':'nano-gpu-boundary-2048.json','small_gpu_boundary':'small-gpu-boundary-2048.json',
        'nano_gpu_publication':'nano-split-publication-result.json','small_gpu_publication':'small-split-publication-result.json',
        'small_gpu_single_failure':'small-gpu-rmsnorm-split-200.json',
        'nano_npu_stage_profile':'nano-npu-window64-200-stage-summary.json'}
    data={}
    for key,name in file_names.items(): data[key],sources[key]=read(root,name)
    delivery={}
    for name in ('nano','small'):
        run=data[name+'_gpu_run']; compare=comparisons[name+'_gpu']; pub=data[name+'_gpu_publication']
        passed=successful_run(run,corpus,'DmlExecutionProvider',eval_ids) and generation_protocol_ok(run) and boundary_ok(data[name+'_gpu_boundary'],boundary_hash) and pub.get('status')=='verified' and pub.get('commit')==config['experimental_exports'][name]['revision']
        if name=='small':passed=passed and (run.get('provider_evidence') or {}).get('all_child_verified') is True
        delivery[name+'_gpu']={'status':'validated_200' if passed else 'not_validated',
            'baseline':compare['baseline'],'candidate':compare['candidate'],
            'boundary_2016_plus_32':boundary_ok(data[name+'_gpu_boundary'],boundary_hash),
            'publication':{'status':pub.get('status'),'repo_id':pub.get('repo_id'),'commit':pub.get('commit'),'path_in_repo':pub.get('path_in_repo')}}
    stage=data['nano_npu_stage_profile']
    npu_ok=successful_run(data['nano_npu_run'],corpus,'QNNExecutionProvider',eval_ids) and generation_protocol_ok(data['nano_npu_run']) and stage.get('status')=='verified' and stage.get('report_sha256')==sources['nano_npu_run']['sha256']
    delivery['nano_npu']={'status':'validated_200_boundary_pending' if npu_ok else 'not_validated',
        'baseline':comparisons['nano_npu']['baseline'],'candidate':comparisons['nano_npu']['candidate'],
        'stage_profile_status':stage.get('status'),'prefill_window':64,'cache_tokens':2048,'boundary_2016_plus_32':'pending'}
    for model in ('nano', 'small'):
        variant, evidence = summarize_npu32(root, config, model, corpus, eval_ids, boundary_hash)
        if variant is not None:
            delivery[model + '_npu'] = variant
        sources.update(evidence)
    phases = calibration_phases(root, config)
    for model in ('nano', 'small'):
        if phases['activation'][model]['status'] != 'input_complete':
            current = delivery[model + '_npu']
            if current.get('status') == 'validated_200_boundary_passed':
                current['status'] = 'not_validated'
                current['calibration_error'] = 'Activation calibration trace is incomplete or invalid'
    apply_calibration_policy_gate(delivery, config, root)
    completion = delivery_complete(delivery)
    failure=data['small_gpu_single_failure']
    failed_rows=[row for row in failure.get('items',[]) if row.get('error')]
    delivery['small_gpu']['single_session_attempt']={'status':failure.get('status'),'failed_items':len(failed_rows),
        'memory_after':failure.get('memory_after'),'note':'Separate failed single-session attempt; successful result used fresh batches.'}
    return {'schema':'kapi-snapdragon-delivery-summary-v1','completion':completion,
        'scope':'proxy evaluation; not IDE qualified', 'corpus_sha256':corpus, 'dataset_audit':dataset, 'calibration_phases':phases,
        'samples':{'calibration_expected':config['samples']['calibration_count'],'evaluation_expected':config['samples']['evaluation_count']},
        'raw_models':config['models'],'published_exports':config.get('published_exports',{}),
        'experimental_exports':config.get('experimental_exports',{}),'npu_exports':config.get('npu_exports',{}),'delivery':delivery,'sources':sources}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():p.error('Output must not exist')
    try:
        data=summarize(a.root);a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        print(json.dumps({k:v['status'] for k,v in data['delivery'].items()},indent=2))
    except Exception as exc:p.exit(1,f'Summary failed: {type(exc).__name__}: {exc}\n')

if __name__=='__main__':main()
