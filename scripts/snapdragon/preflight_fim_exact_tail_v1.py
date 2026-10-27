"""Manual real Olive data/reader preflight, bounded to two CPU inferences.
No quantization, GPU, environment installation, or job/cache/model mutation.
"""
from __future__ import annotations
import argparse, ast, copy, gc, hashlib, importlib.metadata, inspect, json, os, sys, tempfile, traceback, types
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=Path('C:/IA/models/KapiSnapdragonArtifacts/nano-fim-exact-tail-v1-rx7800xt')
LEGACY=Path('C:/IA/models/KapiSnapdragonArtifacts/nano-qnn-prepare-rx7800xt/qnn-output-v6')
EXPECTED_STREAM='07f7dec2ffed64e72f7d015bfd86bbce5d16f0297538c6fffd079fe9183da362'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,text):
    if not value: raise ValueError(text)
def json_file(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def configure_temp(path):
    for name in ('HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES'): os.environ[name]='-1'
    for name in ('HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE','PYTHONDONTWRITEBYTECODE'): os.environ[name]='1'
    for name,child in (('HF_HOME','hf'),('HF_HUB_CACHE','hf/hub'),('HF_DATASETS_CACHE','datasets'),('HF_ASSETS_CACHE','assets'),('TORCH_HOME','torch')):
        os.environ[name]=str(path/child)
    os.environ['TOKENIZERS_PARALLELISM']='false'
    for name in ('KAPI_QNN_PREP_TRACE_DIR','KAPI_QNN_FIM_PREFLIGHT'): os.environ.pop(name,None)
    sys.dont_write_bytecode=True

def isolated_data_config(raw,temp,audit_path=None):
    from olive.data.config import DataConfig
    config=copy.deepcopy(raw)
    config['load_dataset_config'].update(cache_dir=str(temp/'datasets'),keep_in_memory=True)
    if audit_path: config['pre_process_data_config']['audit_path']=str(audit_path)
    return DataConfig(**config)

def gptq(recipe,temp,report):
    import torch
    from olive.passes.pytorch.train_utils import get_calibration_dataset
    import olive
    from audit_gptq_calibration_tokens import installed_function
    from run_olive_fwht import validate_calibration_batches
    require(importlib.metadata.version('transformers')=='4.57.6','GPTQ preflight requires probe Transformers4.57.6')
    config=next(d for d in recipe['data_configs'] if d['name']==recipe['passes']['g']['data_config'])
    ledger_path=BASE/'tail/preflight-token-ledger.json'; ledger=json_file(ledger_path)
    runtime_path=temp/'gptq-runtime-token-ledger.json'
    data_config=isolated_data_config(config,temp,runtime_path)
    # Exact installed train_utils helper: supplied DataConfig means model=None is unused.
    # This runs the JSON loader, registered custom preprocessor, and DataLoader.
    batches=get_calibration_dataset(None,data_config)
    before=validate_calibration_batches(batches,ledger)
    runtime=json_file(runtime_path)
    require(runtime['stream_token_ids_sha256']==EXPECTED_STREAM,'Runtime custom stream hash differs')
    require(runtime['batches']==ledger['batches'],'Runtime/custom preflight batch ledgers differ')
    site=Path(olive.__file__).resolve().parent.parent
    ns={'torch':torch,'LongTensor':torch.LongTensor,'Tensor':torch.Tensor}
    _,collate_source=installed_function(site/'gptqmodel/utils/data.py','collate_data',ns)
    prepare,prepare_source=installed_function(site/'gptqmodel/models/base.py','prepare_dataset',ns,'BaseGPTQModel')
    prepared=prepare(types.SimpleNamespace(tokenizer=types.SimpleNamespace(pad_token_id=151643),support_batch_quantize=True),batches)
    after=validate_calibration_batches(prepared,ledger)
    require(before==after,'Installed GPTQ prepare/collate changed batches')
    require(before['stream_token_ids_sha256']==EXPECTED_STREAM,'Unexpected expected stream hash')
    report.update(status='passed',actual_olive_get_calibration_dataset=True,
        resolved_dataloader={'type':data_config.dataloader_config.type,'params':data_config.dataloader_params},
        input_config_overrides={'audit_path':str(runtime_path),'cache_dir':str(temp/'datasets'),'keep_in_memory':True},
        validation_before_gptq_prepare=before,validation_after_gptq_prepare=after,
        source_files={str(p):sha(p) for p in (ledger_path,ROOT/'scripts/snapdragon/gptq_join_tail_v1.py',ROOT/'scripts/snapdragon/run_olive_fwht.py',Path(inspect.getsourcefile(get_calibration_dataset)))},
        installed_gptq_data_functions={'prepare_dataset':prepare_source,'collate_data':collate_source},
        runtime_preprocessor_ledger={'path':str(runtime_path),'sha256':sha(runtime_path)},
        model_loaded=False,quantization_run=False,inference_count=0)

def guard_helpers():
    path=ROOT/'scripts/snapdragon/qnn_cpu_sitecustomize.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    names={'_fim_digest','validate_fim_stride','FimGuard','GuardedBatches'}
    selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    require(len(selected)==len(names),'Guard template helpers missing')
    namespace={'hashlib':hashlib,'json':json,'Path':Path}
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[])),str(path),'exec'),namespace)
    return namespace,path

def describe_array(value): return {'shape':list(value.shape),'dtype':str(value.dtype)}

def activation(recipe,temp,report):
    import numpy as np
    import onnxruntime as ort
    from olive.common.onnx_io import get_io_config
    from olive.data.component.dataloader import LLMAugmentedDataLoader
    require(importlib.metadata.version('transformers')=='5.3.0','Activation preflight requires qnn-cpu Transformers5.3')
    footprint_path=LEGACY/'footprint.json'; footprint=json_file(footprint_path)
    entries=[e for e in footprint.values() if e.get('from_pass')=='dynamictofixedshape']
    require(len(entries)==1,'Expected one legacy pre-SQ DynamicToFixedShape footprint entry')
    model_config=entries[0]['model_config_data']['config']
    model=Path(model_config['model_path'])/model_config['onnx_file_name']
    require(model.is_file(),'Legacy pre-SQ graph missing')
    io=get_io_config(str(model))
    config=next(d for d in recipe['data_configs'] if d['name']==recipe['passes']['sq']['data_config'])
    data_config=isolated_data_config(config,temp)
    # Exact installed DataConfig -> container -> reader -> LLMAugmentedDataLoader.
    reader=data_config.to_data_container().create_calibration_dataloader(model_path=str(model),io_config=io)
    require(isinstance(reader.dataloader,LLMAugmentedDataLoader),'Official reader did not create LLM augmentation')
    require(reader.data_iter is not None,'Expected reader data_iter already created')
    helpers,guard_source=guard_helpers(); record={}; writes=[]
    guard=helpers['FimGuard'](BASE/'activation/preflight.json',record,lambda:writes.append(guard.index if 'guard' in locals() else None))
    original_data_iter=reader.data_iter
    # Identical installation point/order as the template's _audited_qnn.
    augmented=reader.dataloader
    augmented.dataloader=helpers['GuardedBatches'](augmented.dataloader,guard)
    preflight=json_file(BASE/'activation/preflight.json')
    originals=[json.loads(x) for x in (ROOT/'data/calib/snapdragon-20261026/calibration.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    lengths=[len(r['prompt_ids']) for r in originals]
    shortest=lengths.index(min(lengths)); longest=lengths.index(max(lengths))
    require((min(lengths),max(lengths))==(8,549),'Unexpected min/max source lengths')
    selected={}; items=[]; scalar_promotions=[]
    for index,row in enumerate(originals):
        feed=reader.get_next(); require(feed is not None,'Reader ended before256')
        require(set(feed)==set(io['input_names']),'Reader IO names differ from graph')
        require(feed['input_ids'].shape==(1,lengths[index]),'Reader introduced padding/truncation')
        require(feed['input_ids'].reshape(-1).tolist()==row['prompt_ids'],'Reader altered frozen IDs')
        for name,dtype,shape in zip(io['input_names'],io['input_types'],io['input_shapes']):
            value=feed[name]
            require(str(value.dtype)==dtype,f'Reader dtype mismatch: {name}')
            if len(value.shape)!=len(shape):
                require(name=='total_seq_len' and shape==[] and value.shape==(1,),f'Reader rank mismatch: {name}: graph={shape}, actual={value.shape}')
                scalar_promotions.append({'row_index':index,'name':name,'graph_shape':shape,'reader_shape':list(value.shape),
                    'cause':'Installed Olive format_data uses np.ascontiguousarray on a scalar; feed is preserved without reshape'})
            require(all(not isinstance(d,int) or actual==d for d,actual in zip(shape,value.shape)),f'Reader fixed shape mismatch: {name}')
        kv_names=[n for n in feed if n.startswith('past_key_values.')]
        require(len(kv_names)==48,'Expected24layers x key/value')
        require(all(feed[n].shape==(1,2,0,64) and feed[n].dtype==np.float32 for n in kv_names),'Unexpected empty KV cache')
        require(feed['past_seq_len'].reshape(-1).tolist()==[lengths[index]-1],'Unexpected GQA sequence lengths')
        require(int(feed['total_seq_len'].item())==lengths[index],'Unexpected total sequence length')
        if index in (shortest,longest): selected[index]={k:v.copy() for k,v in feed.items()}
        items.append({'row_index':index,'id':row['id'],'tokens':lengths[index],
            'reader_input_ids_sha256':hashlib.sha256(np.asarray(feed['input_ids'],dtype='<i8').tobytes()).hexdigest(),
            'past_seq_len':int(feed['past_seq_len'].reshape(-1)[0]),'total_seq_len':int(feed['total_seq_len'].item())})
    require(reader.get_next() is None,'Reader yielded more than256')
    guard.finish()
    require(guard.index==256 and record['fim_preflight']['matched']==256 and record['fim_preflight']['all_match'] is True,'Actual guard did not see all256 raw batches')
    require(reader.data_iter is original_data_iter,'Reader iterator was replaced; not the intended guard test')
    report.update(reader_validation={'rows':len(items),'min_tokens':min(lengths),'max_tokens':max(lengths),
        'valid_tokens':sum(lengths),'padding_tokens':0,'wrapper_tokens':0,'truncated_tokens':0,
        'all_frozen_ids_preserved':True,'all_raw_attention_masks_ones':True,
        'reader_names_and_dtypes_match_legacy_graph':True,'strict_reader_shape_metadata_match':not scalar_promotions,
        'shape_metadata_exceptions':scalar_promotions,'llm_has_gqa':augmented.has_gqa,
        'kv_cache_shape':[1,2,0,64],'kv_tensor_count':48,'items':items},
        guard_validation={'applied_after_reader_data_iter_created':True,'original_data_iter_preserved':True,
            'template_installed_in_venv':False,'global_tracing_enabled':False,'template_sha256':sha(guard_source),
            'record':record},
        legacy_graph={'role':'Existing pre-SQ graph only for actual reader IO and the two authorized CPU inference checks; no new weights/calibration',
            'path':str(model),'sha256':sha(model),'footprint':str(footprint_path),'footprint_sha256':sha(footprint_path),
            'from_pass':entries[0]['from_pass'],'io':io},inferences=[],status='reader_passed_inferences_pending')
    # Only this one session and two calls; no calibration engine/pass is invoked.
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    options.log_severity_level=3
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    require(session.get_providers()==['CPUExecutionProvider'],'Unexpected inference provider')
    require([x.name for x in session.get_inputs()]==io['input_names'],'ORT session IO differs')
    for index in (shortest,longest):
        feed=selected[index]
        outputs=session.run(None,feed)
        require(len(outputs)==len(io['output_names']),'Unexpected output count')
        details=[]
        for name,value,expected_dtype,expected_shape in zip(io['output_names'],outputs,io['output_types'],io['output_shapes']):
            require(str(value.dtype)==expected_dtype,f'Output dtype mismatch: {name}')
            require(len(value.shape)==len(expected_shape),f'Output rank mismatch: {name}')
            require(all(not isinstance(d,int) or actual==d for d,actual in zip(expected_shape,value.shape)),f'Output fixed shape mismatch: {name}')
            finite=bool(np.isfinite(value).all())
            require(finite,f'Nonfinite output: {name}')
            if name=='logits': require(value.shape==(1,lengths[index],151936),'Unexpected logits shape')
            else: require(value.shape==(1,2,lengths[index],64),'Unexpected present KV shape')
            details.append({'name':name,**describe_array(value),'all_finite':finite})
        report['inferences'].append({'row_index':index,'id':originals[index]['id'],'tokens':lengths[index],
            'provider':'CPUExecutionProvider','inputs':{k:describe_array(v) for k,v in feed.items()},'outputs':details,'all_finite':True})
        print(f'CPU inference passed: index={index}, tokens={lengths[index]}, outputs={len(outputs)}, all finite',flush=True)
        del outputs;gc.collect()
    report.update(status='passed',runtime_input_contract_accepted_for_both_lengths=True,inference_count=2,quantization_run=False,activation_calibration_run=False,
        resolved_dataloader={'type':data_config.dataloader_config.type,'params':data_config.dataloader_params},
        source_files={str(p):sha(p) for p in (guard_source,BASE/'activation/preflight.json',Path(inspect.getsourcefile(LLMAugmentedDataLoader)),Path(inspect.getsourcefile(get_io_config)))})

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('gptq','activation'),required=True)
    parser.add_argument('--recipe',type=Path,default=BASE/'activation/recipe-fim-activation-v1.json')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    require(args.out.resolve().is_relative_to((ROOT/'reports').resolve()),'Report must be under worktree reports')
    require(not args.out.exists(),'Refusing to overwrite existing report')
    temp=Path(tempfile.mkdtemp(prefix='kapi-fim-real-preflight-'));configure_temp(temp)
    recipe=json_file(args.recipe); recipe_hash=sha(args.recipe)
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'phase':args.phase,'status':'started','script_sha256':sha(__file__),
        'recipe':str(args.recipe),'recipe_sha256':recipe_hash,'isolated_temp':str(temp),'GPU_execution':False,
        'scope':'Real CPU-only data preflight; exactly two activation inference calls if activation phase; no quantization/calibration/model/cache mutation'}
    try:
        report['versions']={n:importlib.metadata.version(n) for n in ('transformers','olive-ai','datasets','torch','gptqmodel')}
        (gptq if args.phase=='gptq' else activation)(recipe,temp,report)
        require(sha(args.recipe)==recipe_hash,'Recipe changed during preflight')
        report['recipe_unchanged_after_preflight']=True
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        raise
    finally:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'report':str(args.out.resolve()),'status':report['status'],'error':report.get('error'),'temp':str(temp)},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
