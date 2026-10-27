"""Explicit process-local bridge from Olive0.13 ModelBuilder calls to GenAI0.18."""
import functools

ALIASES={'int4_block_size':'block_size','int4_accuracy_level':'accuracy_level','int4_is_symmetric':'is_symmetric','int4_op_types_to_quantize':'op_types_to_quantize','int4_nodes_to_exclude':'nodes_to_exclude','int4_algo_config':'algo_config'}

def cli_options(options):
    output=[]
    for old,value in options.items():
        if value is None:continue
        key=ALIASES.get(old,old)
        if isinstance(value,bool):text=str(value).lower()
        elif isinstance(value,(str,int,float)):text=str(value)
        elif isinstance(value,(list,tuple)) and all(isinstance(x,str) for x in value):
            text=('/' if key=='op_types_to_quantize' else ',').join(value)
        else:raise TypeError('Unsupported legacy ModelBuilder option: '+key)
        output.append(key+'='+text)
    return output

def install():
    import onnxruntime_genai
    if onnxruntime_genai.__version__!='0.18.0':raise RuntimeError('This compatibility bridge requires exact GenAI0.18.0')
    from onnxruntime_genai.models import builder
    from onnxruntime_genai.models.loaders.quant_model import QuantModel
    from olive.passes.onnx.model_builder import ModelBuilder
    original=builder.create_model
    if not getattr(original,'_olive_genai18_bridge',False):
        @functools.wraps(original)
        def create_model(model_name,input_path,output_dir,precision,execution_provider,cache_dir,**extra):
            precision=getattr(precision,'value',precision)
            if 'hf_details' in extra:prepared=extra
            else:prepared=builder.parse_extra_options(model_name,input_path,output_dir,precision,execution_provider,cache_dir,cli_options(extra))
            return original(model_name,input_path,output_dir,precision,execution_provider,cache_dir,**prepared)
        create_model._olive_genai18_bridge=True
        builder.create_model=create_model
    # Native0.18 loaders support Olive/GPTQ and dynamic bits. Do not patch removed quantized_model.
    ModelBuilder.maybe_patch_quant=staticmethod(lambda:None)
    return {'version':'0.18.0','legacy_quant_patch':'not applied; native models/loaders','extra_options':'parsed through official parse_extra_options','aliases':ALIASES}
