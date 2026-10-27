"""Read-only minimum receipt gates for a prepared FIM exact-tail QNN package.
Manual development tool. No model/session/inference, calibration, GPU, upload or file mutation except --out.
"""
import argparse,hashlib,json,struct
from pathlib import Path
from datetime import datetime,timezone
from summarize_delivery import calibration_policy_evidence
STREAM='07f7dec2ffed64e72f7d015bfd86bbce5d16f0297538c6fffd079fe9183da362'
def require(value,message):
    if not value:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def ones_hash(count):return hashlib.sha256(struct.pack('<q',1)*count).hexdigest()
def fim_hash(values):return hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest()
def receipts(config):
    spec={k:config[k] for k in ('activation_calibration_trace','gptq_runtime_receipt','gptq_completion_report')}
    spec['calibration_policy']='fim-exact-tail-v1'
    valid,reason,evidence=calibration_policy_evidence(spec,'fim-exact-tail-v1',Path.cwd())
    require(valid,reason)
    ledger=read(config['gptq_runtime_receipt']);job=read(config['gptq_completion_report'])
    pre=read(config['gptq_token_preflight'])
    require(ledger['helper_sha256']==pre['helper_sha256'],'Runtime helper differs from token preflight pin')
    require(job['runtime_calibration_audit']['preflight_sha256']==sha(config['gptq_token_preflight']),'Job used a different token preflight')
    for scope in ('quantize_entry','after_prepare_dataset'):
        actual=job['runtime_calibration_audit'][scope]['batches']
        for expected,observed in zip(ledger['batches'],actual):
            require(expected['token_ids_sha256']==observed['token_ids_sha256'],'Runtime batch ID hash differs')
            require(observed['attention_mask_sha256']==ones_hash(expected['tokens']),'Runtime attention mask is not all ones')
    trace=read(config['activation_calibration_trace']);fim=trace['fim_preflight'];activation=read(config['activation_token_preflight'])
    require(trace.get('status')=='completed' and trace.get('calibration_inputs_seen')==256,'Activation helper/reader not complete256')
    require(trace.get('calibration_providers')==['CPUExecutionProvider'],'Activation provider differs')
    require(trace.get('fim_stride') is None or (type(trace.get('fim_stride')) is int and trace['fim_stride']==0),'Activation stride must not skip rows')
    require(trace.get('activation_type')=='QUInt16','Activation type differs from unsigned16')
    require(fim['preflight_sha256']==sha(config['activation_token_preflight']),'Actual activation preflight pin differs')
    require(len(fim['items'])==len(activation['items'])==256,'Actual activation item coverage differs')
    for expected,observed in zip(activation['items'],fim['items']):
        require(observed['id']==expected['id'] and observed['tokens_sha256']==fim_hash(expected['input_ids']) and observed['mask_sha256']==fim_hash(expected['attention_mask']),'Actual activation IDs/masks differ from frozen preflight')
        require(expected['attention_mask']==[1]*len(expected['input_ids']),'Activation padding or masked tokens present')
    for name in ('gptq_real_preflight','activation_reader_preflight'):
        data=read(config[name]);require(data.get('status')=='passed',name+' did not pass')
        evidence[name]={'path':config[name],'sha256':sha(config[name])}
    checkpoint=read(config['checkpoint_inspection']);require(checkpoint.get('passed') is True,'Checkpoint inspection failed')
    qnn=read(config['qnn_completion_report'])
    require(qnn.get('status')=='workflow_returned_artifact_validation_pending','QNN workflow has not returned successfully')
    expected={'ModelBuilder','MatMulNBitsToQDQ','GraphSurgeries','DynamicToFixedShape','OnnxStaticQuantization','SplitModel','StaticLLM'}
    completed={s['pass'] for s in qnn.get('stages',[]) if s.get('status')=='completed'}
    require(expected<=completed,'QNN pass chain incomplete')
    recipe=read(config['qnn_recipe']);require(recipe['passes']['st']['context_length']==32,'StaticLLM window is not32')
    require(recipe['passes']['sq']['calibration_providers']==['CPUExecutionProvider'],'Recipe activation provider differs')
    for name in ('gptq_token_preflight','activation_token_preflight','checkpoint_inspection','qnn_completion_report','qnn_recipe'):
        evidence[name]={'path':config[name],'sha256':sha(config[name])}
    return {'passed':True,'runtime_stream_sha256':STREAM,'actual_gptq_batches':30,'actual_activation_matched':256,'actual_activation_input_complete':True,'evidence':evidence}
def providers_cpu_only(value):
    if isinstance(value,dict):
        if 'provider_options' in value:
            for entry in value['provider_options'] or []:
                require(all(name.lower()=='cpu' for name in entry),'Package contains non-CPU provider options; no silent provider rewrite')
        for child in value.values():providers_cpu_only(child)
    elif isinstance(value,list):
        for child in value:providers_cpu_only(child)
def package(config):
    report=read(config['static_inspection']);root=Path(config['package']).resolve()
    require(report.get('passed') is True and report.get('transfer_ready') is True,'Static inspection/relocation incomplete')
    require(Path(report['artifact_directory']).resolve()==root,'Inspection refers to a different package')
    require(report.get('expected_window')==32 and report.get('expected_head_size')==config['expected_head_size'],'Static expected dimensions differ')
    require(report.get('byte_parity',{}).get('required') is True and report['byte_parity'].get('passed') is True,'Original tokenizer byte parity not required/passed')
    require(report.get('tokenizer',{}).get('passed') is True and report['tokenizer'].get('prompts_checked')==456,'Tokenizer456 audit incomplete')
    metadata=read(root/'genai_config.json');decoder=metadata['model']['decoder']
    require(metadata['model']['context_length']==2048 and decoder['sliding_window']['window_size']==32,'Package context/window mismatch')
    providers_cpu_only(metadata)
    for entry in report['artifact_files']:
        path=(root/entry['path']).resolve();require(path.is_relative_to(root) and path.is_file(),'Unsafe or missing inspected file')
        require(path.stat().st_size==entry['size'] and sha(path)==entry['sha256'],'Inspected package has changed')
    return {'passed':True,'original_tokenizer_bytes':True,'tokenizer_prompts':456,'context':2048,'window':32,'CPU_provider_configuration_only':True,'inspection_sha256':sha(config['static_inspection'])}
def validate_control(report,prompt_tokens,padded_tokens,maximum,stops):
    require(report.get('status')=='control_completed' and report.get('qualified') is False,'CPU control did not complete')
    require(report.get('prompt_tokens')==prompt_tokens and report.get('sequence_length_after_append')==padded_tokens,'CPU prompt/padding mismatch')
    require(report.get('max_length_requested')==maximum,'CPU generation budget mismatch')
    ids=report.get('generated_ids') or []
    require(1<=len(ids)<=32 and all(type(i) is int and i>=0 for i in ids),'Invalid generated token count/IDs')
    require(len(ids)==32 or ids[-1] in stops,'Early runtime_done without EOS/FIM; padding budget may be wrong')
    require(report.get('final_sequence_length')==padded_tokens+len(ids),'Final sequence length differs from appended generated tokens')
    return {'passed':True,'prompt_tokens':prompt_tokens,'padded_tokens':padded_tokens,'generated_tokens':len(ids),'stop':'max_tokens' if len(ids)==32 else 'explicit_EOS_or_FIM','max_length_requested':maximum}
def smoke(config):
    source=read(Path(config['source_snapshot'])/'tokenizer.json')
    markers={'<|fim_prefix|>','<|fim_suffix|>','<|fim_middle|>','<|endoftext|>','<|im_end|>','<|fim_pad|>'}
    stops={item['id'] for item in source['added_tokens'] if item['content'] in markers}
    return {'short':validate_control(read(config['smoke_short']),361,384,416,stops),
            'boundary':validate_control(read(config['smoke_boundary']),2016,2016,2048,stops),
            'scope':'Two CPU generation controls only; not a quality, HTP compilation or device inference qualification'}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--phase',choices=('receipts','package','smoke'),required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    config=read(a.manifest)
    require(not a.out.exists(),'Output must be new')
    require(not a.out.resolve().is_relative_to(Path(config['package']).resolve()) and not a.out.resolve().is_relative_to(Path(config['artifact_source']).resolve()),'Report must be outside artifact/package')
    result={'schema':'kapi-prepared-qnn-receipt-gates-v1','created_utc':datetime.now(timezone.utc).isoformat(),'phase':a.phase,'qualified':False,'htp_validated':False,'manifest_sha256':sha(a.manifest),'script_sha256':sha(__file__)}
    try:
        result['receipts']=receipts(config)
        if a.phase in ('package','smoke'):result['package']=package(config)
        if a.phase=='smoke':result['smoke']=smoke(config)
        result['passed']=True
    except Exception as exc:result.update(passed=False,error=f'{type(exc).__name__}: {exc}')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'passed':result['passed'],'phase':a.phase,'report':str(a.out),'error':result.get('error')},ensure_ascii=False))
    return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
