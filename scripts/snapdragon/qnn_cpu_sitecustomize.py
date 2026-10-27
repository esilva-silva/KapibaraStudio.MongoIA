"""Opt-in sitecustomize template for an isolated CPU QNN preparation venv.

Copy only into that venv site-packages; never install into the IDE/base env.
Registers the official plugin. Set KAPI_QNN_PREP_TRACE_DIR to audit QNN
configuration and actual calibration reader inputs without changing data.
Helper completion is not calibration completion.
"""
import hashlib
import json
from pathlib import Path


def _fim_digest(values):
    payload = json.dumps(values, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def validate_fim_stride(stride):
    if stride is not None and (type(stride) is not int or stride != 0):
        raise ValueError('FIM preflight requires calibration stride None or 0')
    return stride


class FimGuard:
    """Validate raw batches before the LLM reader removes attention_mask."""
    def __init__(self, preflight_path, record, write):
        path = Path(preflight_path)
        data = json.loads(path.read_text(encoding='utf-8'))
        policy = data.get('policy') or {}
        if (data.get('schema') != 'kapi-fim-activation-preflight-v1' or data.get('count') != 256
                or len(data.get('items') or []) != 256 or policy.get('add_special_tokens') is not False
                or policy.get('pad_to_max_len') is not False or policy.get('batch_size') != 1):
            raise ValueError('Invalid FIM preflight policy or count')
        self.items = data['items']; self.record = record; self.write = write; self.index = 0
        record['fim_preflight'] = {'preflight_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'dataset_sha256': data['dataset_source_sha256'], 'tokenizer_sha256': data['tokenizer_sha256'],
            'policy': policy, 'matched': 0, 'all_match': False, 'items': []}
        write()

    def accept(self, batch):
        try:
            data = batch[0] if isinstance(batch, (list, tuple)) else batch
            if self.index >= len(self.items): raise ValueError('More than 256 calibration inputs')
            expected = self.items[self.index]
            def values(key):
                raw = data[key]
                if hasattr(raw, 'detach'): raw = raw.detach().cpu()
                rows = raw.tolist()
                if len(rows) != 1 or not isinstance(rows[0], list): raise ValueError('Expected batch size one')
                return rows[0]
            ids, mask = values('input_ids'), values('attention_mask')
            if ids != expected['input_ids'] or mask != expected['attention_mask']:
                raise ValueError('FIM IDs or attention mask mismatch at index ' + str(self.index))
            info = self.record['fim_preflight']
            info['items'].append({'id':expected['id'],'tokens_sha256':_fim_digest(ids),'mask_sha256':_fim_digest(mask)})
            self.index += 1; info['matched'] = self.index
            self.write()
        except Exception as exc:
            self.record['fim_preflight']['error'] = f'{type(exc).__name__}: {exc}'
            self.record['fim_preflight']['all_match'] = False
            self.record['calibration_status'] = 'failed'
            self.write()
            raise

    def finish(self):
        if self.index != 256:
            self.record['fim_preflight']['error'] = f'Expected 256 inputs; got {self.index}'
            self.record['calibration_status'] = 'failed'; self.write()
            raise ValueError(self.record['fim_preflight']['error'])
        self.record['fim_preflight']['all_match'] = True
        self.record['fim_preflight']['fim_complete'] = True
        self.write()


class GuardedBatches:
    def __init__(self, dataloader, guard): self.dataloader, self.guard = dataloader, guard
    def __len__(self): return len(self.dataloader)
    def __iter__(self):
        for batch in self.dataloader:
            self.guard.accept(batch)
            yield batch


# Isolated QNN preparation environment; no changes to base environments.
import os
import onnxruntime as _ort
import onnxruntime_qnn as _qnn
if _qnn.get_ep_name() not in _ort.get_available_providers():
    _ort.register_execution_provider_library(_qnn.get_ep_name(),_qnn.get_library_path())
if os.environ.get('KAPI_QNN_FIM_PREFLIGHT') and not os.environ.get('KAPI_QNN_PREP_TRACE_DIR'):
    raise RuntimeError('KAPI_QNN_FIM_PREFLIGHT requires KAPI_QNN_PREP_TRACE_DIR')
if os.environ.get('KAPI_QNN_PREP_TRACE_DIR'):
    import functools,inspect,json,time,uuid
    from pathlib import Path
    from onnxruntime.quantization.execution_providers import qnn as _config
    _real=_config.get_qnn_qdq_config
    @functools.wraps(_real)
    def _audited_qnn(*args,**kwargs):
        bound=inspect.signature(_real).bind_partial(*args,**kwargs);bound.apply_defaults()
        providers=bound.arguments.get('calibration_providers')
        if providers!=['CPUExecutionProvider']:
            raise RuntimeError('This preparation requires explicit CPU activation calibration')
        start=time.monotonic();folder=Path(os.environ['KAPI_QNN_PREP_TRACE_DIR']);folder.mkdir(parents=True,exist_ok=True)
        path=folder/('qnn-qdq-'+str(os.getpid())+'-'+uuid.uuid4().hex+'.json')
        record={'helper':'onnxruntime.quantization.execution_providers.qnn.get_qnn_qdq_config','calibration_providers':providers,'activation_type':str(bound.arguments.get('activation_type')),'weight_type':str(bound.arguments.get('weight_type')),'status':'running'}
        if os.environ.get('KAPI_QNN_FIM_PREFLIGHT'):
            record['fim_stride'] = bound.arguments.get('stride')
        path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        try:
            if os.environ.get('KAPI_QNN_FIM_PREFLIGHT'):
                validate_fim_stride(record['fim_stride'])
            result=_real(*args,**kwargs)
            extra=getattr(result,'extra_options',{}) or {};overrides=extra.get('TensorQuantOverrides',{})
            record.update(status='completed',seconds=time.monotonic()-start,extra_options_keys=sorted(extra),tensor_override_count=len(overrides),tensor_override_quant_types=sorted({str(v.get('quant_type')) for items in overrides.values() for v in items if isinstance(v,dict) and 'quant_type' in v}))
            reader=bound.arguments['calibration_data_reader']
            original_next=reader.get_next
            guard = None
            if os.environ.get('KAPI_QNN_FIM_PREFLIGHT'):
                guard = FimGuard(os.environ['KAPI_QNN_FIM_PREFLIGHT'], record,
                    lambda: path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8'))
                augmented = getattr(reader, 'dataloader', None)
                if augmented is None or not hasattr(augmented, 'dataloader'):
                    raise RuntimeError('FIM guard requires the Olive LLMAugmentedDataLoader')
                augmented.dataloader = GuardedBatches(augmented.dataloader, guard)
            record['calibration_inputs_seen']=0
            calibration_start=[None]
            def next_audited():
                if calibration_start[0] is None:calibration_start[0]=time.monotonic()
                batch=original_next()
                if batch is None:
                    if guard: guard.finish()
                    record['calibration_status']='input_complete'
                else:
                    record['calibration_status']='collecting'
                    record['calibration_inputs_seen']+=1
                    if record['calibration_inputs_seen']==1:record['first_input_shapes']={key:{'shape':list(value.shape),'dtype':str(value.dtype)} for key,value in batch.items()}
                record['input_iteration_wall_seconds']=time.monotonic()-calibration_start[0]
                if batch is None or record['calibration_inputs_seen']==1 or record['calibration_inputs_seen']%8==0:
                    path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
                return batch
            reader.get_next=next_audited
            path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');return result
        except Exception as error:
            record.update(status='failed',seconds=time.monotonic()-start,error=f'{type(error).__name__}: {error}')
            path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');raise
    _config.get_qnn_qdq_config=_audited_qnn
