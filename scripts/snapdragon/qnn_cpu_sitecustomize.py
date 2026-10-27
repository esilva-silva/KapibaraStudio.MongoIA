"""Opt-in sitecustomize template for an isolated CPU QNN preparation venv.

Copy only into that venv site-packages; never install into the IDE/base env.
Registers the official plugin. Set KAPI_QNN_PREP_TRACE_DIR to audit QNN
configuration and actual calibration reader inputs without changing data.
Helper completion is not calibration completion.
"""
# Isolated QNN preparation environment; no changes to base environments.
import os
import onnxruntime as _ort
import onnxruntime_qnn as _qnn
if _qnn.get_ep_name() not in _ort.get_available_providers():
    _ort.register_execution_provider_library(_qnn.get_ep_name(),_qnn.get_library_path())
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
        path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        try:
            result=_real(*args,**kwargs)
            extra=getattr(result,'extra_options',{}) or {};overrides=extra.get('TensorQuantOverrides',{})
            record.update(status='completed',seconds=time.monotonic()-start,extra_options_keys=sorted(extra),tensor_override_count=len(overrides),tensor_override_quant_types=sorted({str(v.get('quant_type')) for items in overrides.values() for v in items if isinstance(v,dict) and 'quant_type' in v}))
            reader=bound.arguments['calibration_data_reader']
            original_next=reader.get_next
            record['calibration_inputs_seen']=0
            calibration_start=[None]
            def next_audited():
                if calibration_start[0] is None:calibration_start[0]=time.monotonic()
                batch=original_next()
                if batch is None:record['calibration_status']='input_complete'
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
