#!/usr/bin/env python3
"""Pos-processo da variante `int8-emb4`: quantiza a embedding (Gather) de um pacote int8 para GatherBlockQuantized INT4 (block 32).
O ORT 1.30 so suporta 4 bits em Gather; os MatMul seguem em INT8 (MatMulNBits). Reduz ~545 MB (embedding fp32) para ~75 MB no 0.5B.
  python -I scripts/export/quantize_gather.py <pacote_dir>     (sobrescreve model.onnx / model.onnx.data)
"""
import sys
from pathlib import Path

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer

pkg = Path(sys.argv[1])
model = onnx.load(str(pkg / "model.onnx"))
cfg = DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=False, accuracy_level=4, bits=4, op_types_to_quantize=("Gather",))
q = MatMulNBitsQuantizer(model, bits=4, block_size=32, accuracy_level=4, op_types_to_quantize=("Gather",), algo_config=cfg)
q.process()
tmp = pkg / "_q"
tmp.mkdir(exist_ok=True)
q.model.save_model_to_file(str(tmp / "model.onnx"), use_external_data_format=True)
(pkg / "model.onnx").unlink()
(pkg / "model.onnx.data").unlink()
(tmp / "model.onnx").replace(pkg / "model.onnx")
(tmp / "model.onnx.data").replace(pkg / "model.onnx.data")
tmp.rmdir()
print("ok", (pkg / "model.onnx.data").stat().st_size)
