"""Read-only environment inventory and optional small QNN HTP execution test."""
import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
import tempfile
from pathlib import Path


PACKAGES = ("onnxruntime", "onnxruntime-qnn", "onnxruntime-genai", "onnx", "numpy", "olive-ai", "torch", "gptqmodel")


def register_qnn(ort):
    """Modern QNN is a plugin, so get_available_providers alone is insufficient."""
    import onnxruntime_qnn as qnn
    name = qnn.get_ep_name()
    if not any(device.ep_name == name for device in ort.get_ep_devices()):
        ort.register_execution_provider_library(name, qnn.get_library_path())
    return qnn


def inventory():
    facts = {"platform": platform.platform(), "machine": platform.machine(),
             "python": sys.version, "python_executable": sys.executable,
             "packages": {}, "environment": {}}
    for package in PACKAGES:
        try:
            facts["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            facts["packages"][package] = None
    import os
    for key in ("QNN_SDK_ROOT", "QAIRT_SDK_ROOT", "QNN_SDK_PATH"):
        value = os.environ.get(key)
        facts["environment"][key] = {"value": value, "exists": Path(value).exists() if value else False}
    try:
        ort = importlib.import_module("onnxruntime")
        facts["ort_providers"] = ort.get_available_providers()
        qnn = register_qnn(ort)
        facts["qnn_library"] = qnn.get_library_path()
        facts["qnn_htp_library"] = qnn.get_qnn_htp_path()
        facts["ep_devices"] = [{"provider": d.ep_name, "type": str(d.device.type)} for d in ort.get_ep_devices()]
    except Exception as exc:
        facts["ort_error"] = f"{type(exc).__name__}: {exc}"
    try:
        torch = importlib.import_module("torch")
        facts["torch"] = {"cuda_available": torch.cuda.is_available(),
                          "cuda_version": torch.version.cuda, "hip_version": torch.version.hip}
        if torch.cuda.is_available():
            facts["torch"]["device"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        facts["torch_error"] = f"{type(exc).__name__}: {exc}"
    return facts


def classify(facts, smoke):
    blockers = []
    if not facts["packages"].get("onnxruntime-qnn"):
        blockers.append("onnxruntime-qnn package not detected in this interpreter")
    if smoke.get("status") == "failed":
        blockers.append("HTP smoke failed; see error")
    if smoke.get("status") == "inconclusive":
        blockers.append("HTP smoke could not be executed; see reason")
    return blockers


def smoke_htp(settings):
    """Run a fixed QDQ Relu, with CPU fallback disabled; no model download."""
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        from onnx import TensorProto, helper, numpy_helper
    except Exception as exc:
        return {"status": "inconclusive", "reason": f"Missing or unloadable smoke dependency: {type(exc).__name__}: {exc}"}
    try:
        # QDQ representation tests a genuine quantized graph and fixed input shape.
        scale = numpy_helper.from_array(np.array(0.1, dtype=np.float32), "scale")
        zero = numpy_helper.from_array(np.array(128, dtype=np.uint8), "zero")
        nodes = [helper.make_node("QuantizeLinear", ["x", "scale", "zero"], ["xq"]),
                 helper.make_node("DequantizeLinear", ["xq", "scale", "zero"], ["xd"]),
                 helper.make_node("Relu", ["xd"], ["yr"]),
                 helper.make_node("QuantizeLinear", ["yr", "scale", "zero"], ["yq"]),
                 helper.make_node("DequantizeLinear", ["yq", "scale", "zero"], ["y"])]
        graph = helper.make_graph(nodes, "qnn_smoke", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
                                  [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])], [scale, zero])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
        onnx.checker.check_model(model)
        options = ort.SessionOptions()
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        qnn = register_qnn(ort)
        provider_options = dict(settings.get("provider_options", {}))
        provider_options.setdefault("backend_path", settings.get("backend_path", qnn.get_qnn_htp_path()))
        devices = [d for d in ort.get_ep_devices() if d.ep_name == qnn.get_ep_name()
                   and d.device.type == ort.OrtHardwareDeviceType.NPU]
        if not devices:
            return {"status": "inconclusive", "reason": "QNN plugin has no NPU device"}
        options.add_provider_for_devices(devices, provider_options)
        with tempfile.TemporaryDirectory(prefix="qnn-smoke-") as temp:
            model_path = Path(temp) / "smoke.onnx"
            onnx.save(model, str(model_path))
            session = ort.InferenceSession(str(model_path), sess_options=options)
            if "QNNExecutionProvider" not in session.get_providers():
                raise RuntimeError("QNN provider missing from session")
            actual = session.run(None, {"x": np.array([[-1, 0, 1, 2]], dtype=np.float32)})[0]
            np.testing.assert_allclose(actual, [[0, 0, 1, 2]], atol=0.11)
        return {"status": "passed", "provider": "QNNExecutionProvider", "backend": provider_options.get("backend_path"), "cpu_fallback_disabled": True}
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON file with optional qnn_probe settings")
    parser.add_argument("--out", type=Path, required=True, help="JSON report path")
    parser.add_argument("--smoke", action="store_true", help="run small local HTP graph")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    settings = config.get("qnn_probe", {})
    facts = inventory()
    smoke = smoke_htp(settings) if args.smoke else {"status": "not_run"}
    report = {"facts": facts, "smoke": smoke, "blockers": classify(facts, smoke)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if smoke["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
