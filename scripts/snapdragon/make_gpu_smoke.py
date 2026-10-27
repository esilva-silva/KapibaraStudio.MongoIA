"""Write a tiny input-dependent graph for explicit DirectML profiling."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    graph = helper.make_graph([
        helper.make_node("MatMul", ["x", "weight"], ["product"]),
        helper.make_node("Relu", ["product"], ["y"]),
    ], "adreno_smoke", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
       [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
       [numpy_helper.from_array(np.eye(4, dtype=np.float32), "weight")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    onnx.checker.check_model(model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.out))


if __name__ == "__main__":
    main()
