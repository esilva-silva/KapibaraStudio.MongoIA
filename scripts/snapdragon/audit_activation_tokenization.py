"""CPU-only reconstruction of the exact Olive activation preprocessing recipe.
No model/session/activation calibration is run; job inputs and caches are read only.
"""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, inspect, json, os, struct, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
for name in ('HIP_VISIBLE_DEVICES','CUDA_VISIBLE_DEVICES'): os.environ[name]='-1'
for name in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','PYTHONDONTWRITEBYTECODE'): os.environ[name]='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
BASE=Path('C:/IA/models/KapiSnapdragonArtifacts/small-qnn-prepare-rx7800xt-v1')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def token_sha(ids): return hashlib.sha256(b''.join(struct.pack('<q',int(i)) for i in ids)).hexdigest()
def rows(path): return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def require(value,text):
    if not value: raise ValueError(text)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',type=Path,default=BASE/'prepare-after-gptq-v1.json')
    p.add_argument('--samples',type=Path,default=ROOT/'data/calib/snapdragon-20261026/calibration.jsonl')
    p.add_argument('--status',type=Path)
    p.add_argument('--out',type=Path,default=ROOT/'reports/snapdragon/small-activation-tokenization-coverage-rx7800xt.json')
    a=p.parse_args()
    require(a.out.resolve().is_relative_to((ROOT/'reports').resolve()),'Output must be inside worktree reports')
    require(not a.out.exists(),'Refusing to overwrite an existing report')
    from datasets import Dataset
    from olive.common.hf.utils import get_tokenizer
    from olive.data.component.dataloader import default_dataloader
    from olive.data.component.pre_process_data import text_generation_huggingface_pre_process
    from olive.data.component.text_generation import TextGenParams,get_text
    import olive.data.component.text_generation as tg
    require(importlib.metadata.version('transformers')=='5.3.0','Use original activation qnn-cpu environment')
    recipe=json.loads(a.recipe.read_text(encoding='utf-8'))
    dc=next(c for c in recipe['data_configs'] if c['name']==recipe['passes']['sq']['data_config'])
    params=dict(dc['pre_process_data_config'])
    require(params.pop('type')=='text_generation_huggingface_pre_process','Unexpected preprocessor')
    source=Path(params['model_name']); normalized_path=Path(dc['load_dataset_config']['data_files'])
    normalized=rows(normalized_path); original=rows(a.samples)
    require(sha(a.samples)=='179736eed969a8a961af6d3fc410a3eead24306234d65ccf245c20cb16978b60','Original SHA mismatch')
    require(len(normalized)==len(original)==256,'Expected256 source rows')
    require(all(n['text']==r['prompt']==r['prompt_text'] for n,r in zip(normalized,original)),'Normalized rows/order mismatch')
    tokenizer=get_tokenizer(str(source),local_files_only=True,trust_remote_code=False)
    cfg=TextGenParams(**{k:v for k,v in params.items() if k!='model_name'})
    require(cfg.strategy=='line-by-line','Expected line-by-line strategy')
    require(cfg.chat_template is None and cfg.text_formatting_func is None and cfg.text_template is None,'Unexpected custom formatting')
    formatted=[get_text(n,cols=cfg.text_cols,add_special_tokens=cfg.add_special_tokens,tokenizer=tokenizer) for n in normalized]
    raw_ids=tokenizer([n['text'] for n in normalized],add_special_tokens=False,truncation=False)['input_ids']
    formatted_ids=tokenizer(formatted,add_special_tokens=False,truncation=False)['input_ids']
    dataset=Dataset.from_list(normalized); require(not dataset.cache_files,'Unexpected cache-backed input')
    processed=text_generation_huggingface_pre_process(dataset,**params)
    batches=list(default_dataloader(processed)); require(len(batches)==256,'Unexpected sample count')
    records=[]
    for i,((inputs,labels),raw,fmt,row) in enumerate(zip(batches,raw_ids,formatted_ids,original)):
        ids=inputs['input_ids'].reshape(-1).tolist(); mask=inputs['attention_mask'].reshape(-1).tolist(); label=labels.reshape(-1).tolist()
        require(len(ids)==len(mask)==2048,'Unexpected sample shape')
        require(set(mask)<=set((0,1)),'Nonbinary mask')
        valid=[x for x,m in zip(ids,mask) if m]
        require(valid==fmt[:2048],'Installed preprocessing differs from independent formatted tokenization')
        require(mask==[1]*len(valid)+[0]*(2048-len(valid)),'Unexpected padding side/mask gaps')
        require(all(x==tokenizer.pad_token_id for x,m in zip(ids,mask) if not m),'Unexpected padding token')
        require(all(y==(x if m else -100) for x,m,y in zip(ids,mask,label)),'Unexpected label masking')
        stored=row['prompt_ids']
        offsets=[j for j in range(len(valid)-len(stored)+1) if valid[j:j+len(stored)]==stored]
        offset=offsets[0] if offsets else None
        records.append({'row_index':i,'id':row['id'],**{k:row[k] for k in ('domain','cut_kind') if k in row},
            'stored_prompt_tokens':len(stored),'raw_tokenization_matches_stored':raw==stored,
            'actual_valid_tokens_equal_stored':valid==stored,'stored_ids_present_as_exact_subsequence':offset is not None,
            'stored_subsequence_offset':offset,'added_prefix_token_ids':valid[:offset] if offset is not None else None,
            'added_suffix_token_ids':valid[offset+len(stored):] if offset is not None else None,
            'valid_tokens_before_padding':len(valid),'formatted_tokens_without_truncation':len(fmt),
            'truncated_tokens':max(0,len(fmt)-len(valid)),'padding_tokens':2048-len(valid),
            'stored_token_ids_sha256':token_sha(stored),'raw_token_ids_sha256':token_sha(raw),
            'valid_token_ids_sha256':token_sha(valid),'padded_token_ids_sha256':token_sha(ids),'attention_mask_sha256':token_sha(mask)})
    # Supported future activation-only policy, exercised in memory only.
    # This does not replace the actual recipe or run activation quantization.
    no_wrapper_params=dict(params,add_special_tokens=False)
    no_wrapper=text_generation_huggingface_pre_process(Dataset.from_list(normalized),**no_wrapper_params)
    no_wrapper_batches=list(default_dataloader(no_wrapper))
    require(len(no_wrapper_batches)==256,'Unexpected no-wrapper sample count')
    no_wrapper_records=[]
    for (inputs,labels),row in zip(no_wrapper_batches,original):
        ids=inputs['input_ids'].reshape(-1).tolist(); mask=inputs['attention_mask'].reshape(-1).tolist()
        valid=[x for x,m in zip(ids,mask) if m]
        require(len(ids)==2048 and mask==[1]*len(valid)+[0]*(2048-len(valid)),'Unexpected no-wrapper mask')
        require(valid==row['prompt_ids'],'Supported add_special_tokens=False failed frozen-ID parity')
        require(all(x==tokenizer.pad_token_id for x,m in zip(ids,mask) if not m),'Unexpected no-wrapper pad')
        no_wrapper_records.append({'id':row['id'],'valid_tokens':len(valid),'padding_tokens':2048-len(valid),
            'valid_token_ids_sha256':token_sha(valid),'padded_token_ids_sha256':token_sha(ids),'attention_mask_sha256':token_sha(mask)})
    status_path=a.status or a.recipe.with_name(a.recipe.name.replace('prepare-after-gptq-','qnn-job-status-'))
    status=json.loads(status_path.read_text(encoding='utf-8'))
    code_files={Path(inspect.getsourcefile(f)).resolve() for f in (get_tokenizer,get_text,text_generation_huggingface_pre_process,default_dataloader)}
    code_files.add(Path(tg.__file__).resolve())
    summary={'rows_selected':len(original),'rows_emitted':len(records),'raw_tokenization_matches_stored_rows':sum(r['raw_tokenization_matches_stored'] for r in records),
        'actual_valid_ids_equal_stored_rows':sum(r['actual_valid_tokens_equal_stored'] for r in records),
        'stored_ids_intact_as_subsequence_rows':sum(r['stored_ids_present_as_exact_subsequence'] for r in records),
        'rows_actually_truncated':sum(r['truncated_tokens']>0 for r in records),
        'stored_prompt_tokens':sum(r['stored_prompt_tokens'] for r in records),'actual_valid_tokens':sum(r['valid_tokens_before_padding'] for r in records),
        'additional_valid_tokens':sum(r['valid_tokens_before_padding']-r['stored_prompt_tokens'] for r in records),
        'padding_tokens':sum(r['padding_tokens'] for r in records),'total_padded_tokens':256*2048,
        'min_valid_length':min(r['valid_tokens_before_padding'] for r in records),'max_valid_length':max(r['valid_tokens_before_padding'] for r in records),
        'prefix_token_patterns':dict(Counter(str(r['added_prefix_token_ids']) for r in records)),
        'suffix_token_patterns':dict(Counter(str(r['added_suffix_token_ids']) for r in records)),
        'mask_policy':'valid tokens=1; right padding=0; padding labels=-100'}
    parameters={'strategy':str(cfg.strategy),'recipe_add_special_tokens':cfg.add_special_tokens,'tokenizer_call_add_special_tokens':False,
        'max_samples':cfg.max_samples,'max_seq_len':cfg.max_seq_len,'padding_side':cfg.padding_side,'pad_to_max_len':cfg.pad_to_max_len,
        'dataloader_batch_size':1,'use_attention_mask':cfg.use_attention_mask,'tokenizer_truncation_argument':True,
        'bos_token':tokenizer.bos_token,'bos_token_id':tokenizer.bos_token_id,'eos_token':tokenizer.eos_token,'eos_token_id':tokenizer.eos_token_id,'pad_token_id':tokenizer.pad_token_id}
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'CPU tokenizer/data-preprocessing reconstruction only; no model/session/quantization/activation calibration was executed',
        'script_sha256':sha(__file__),'recipe':str(a.recipe.resolve()),'recipe_sha256':sha(a.recipe),'original_samples':str(a.samples.resolve()),
        'original_sha256':sha(a.samples),'normalized_samples':str(normalized_path),'normalized_sha256':sha(normalized_path),'normalized_texts_and_order_equal_original':True,
        'versions':{n:importlib.metadata.version(n) for n in ('transformers','olive-ai','datasets','torch','tokenizers')},'source_snapshot':str(source),
        'tokenizer_files':{p.name:sha(p) for p in source.iterdir() if p.name in ('tokenizer.json','tokenizer_config.json','config.json')},
        'installed_code_files':{str(p):sha(p) for p in code_files},'parameters':parameters,'summary':summary,
        'hash_encoding':'SHA256 of concatenated signed int64 little-endian token IDs',
        'interpretation':{
            'strict_frozen_ids_before_padding':all(r['actual_valid_tokens_equal_stored'] for r in records),
            'raw_transformers53_ids_equal_all_frozen_ids':all(r['raw_tokenization_matches_stored'] for r in records),
            'all_original_tokens_retained_without_truncation':all(r['stored_ids_present_as_exact_subsequence'] and r['truncated_tokens']==0 for r in records),
            'formatting_cause':'Recipe add_special_tokens=true makes installed Olive get_text return f"{tokenizer.bos_token} {text} {tokenizer.eos_token}". tokenizer.bos_token is None, so this inserts literal None plus space before the original prompt, then space plus EOS after it. The later tokenizer call uses add_special_tokens=false, so these are already in the rendered text.',
            'per_row_added_prefix':{'text':'None ','token_ids':[4064,220]},
            'per_row_added_suffix':{'text':' <|endoftext|>','token_ids':[220,151643]},
            'bos_special_token_added':False,'literal_none_prefix_added':True,'eos_special_token_added':True,
            'additional_valid_tokens_are_not_masked_padding':True,
            'comparison_to_gptq':'GPTQ JOIN used raw prompts with add_special_tokens=false and separators. Activation line-by-line uses all256 original prompts but adds a textual wrapper. No raw-tokenizer drift from the frozen IDs was observed in Transformers5.3.',
            'qualification_limit':'This establishes preprocessing/input coverage, not quality or completion of the currently running activation calibration. Any future strict token-parity policy requires an explicit recipe change and revalidation, not alteration of tokenizer files or implicit regex flags.'},
        'supported_activation_only_policy_check':{
            'applied_to_current_jobs':False,'execution':'CPU tokenization/data preprocessing only',
            'parameter':'add_special_tokens=False','parameter_is_supported':True,
            'rows_emitted':len(no_wrapper_records),'strict_frozen_id_parity_rows':len(no_wrapper_records),
            'valid_tokens':sum(r['valid_tokens'] for r in no_wrapper_records),
            'padding_tokens':sum(r['padding_tokens'] for r in no_wrapper_records),
            'added_bos_eos_or_literal_wrapper_tokens':0,'truncated_tokens':0,
            'min_valid_length':min(r['valid_tokens'] for r in no_wrapper_records),
            'max_valid_length':max(r['valid_tokens'] for r in no_wrapper_records),
            'records':no_wrapper_records},
        'future_policy_consequences':{
            'activation_wrapper_only':'Changing the supported activation add_special_tokens parameter to False removes the entire wrapper and restores strict frozen-ID parity. It does not change the GPTQ input policy or require repeating GPTQ. It requires repeating activation quantization/calibration and evaluation before qualification.',
            'gptq_join_tail':'Retaining the final1881-token JOIN tail changes GPTQ inputs/weights and therefore requires repeating GPTQ, activation quantization/calibration and evaluation. This is separate from the activation wrapper policy.',
            'quality_causality':'No quality threshold, quality effect or causal claim is established by this tokenization audit.',
            'current_jobs':'No recipe, tokenizer, cache, model or job was changed/restarted.'},
        'historical_evidence':{'snapshot_path':str(status_path),'snapshot_sha256':sha(status_path),'status_at_read':status.get('status'),
            'stages_at_read':status.get('stages'),'versions_at_read':status.get('versions'),
            'scope_limit':'This is a deterministic recipe reconstruction, not evidence that running ONNX calibration consumed all samples or runtime IDs/hashes; historical progress/counts must be reported separately.'},'records':records}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(a.out.resolve()),'summary':summary,'parameters':parameters},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
