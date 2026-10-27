import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / "qnn_probe.py"
spec = importlib.util.spec_from_file_location("qnn_probe", path)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ProbeTests(unittest.TestCase):
    def test_classifies_missing_package_and_smoke_failure(self):
        blockers = probe.classify({"packages": {"onnxruntime-qnn": None}}, {"status": "failed"})
        self.assertEqual(len(blockers), 2)
        self.assertIn("HTP smoke failed", blockers[1])

    def test_skipped_smoke_does_not_claim_success(self):
        self.assertEqual(probe.classify({"packages": {"onnxruntime-qnn": "1"}}, {"status": "not_run"}), [])

    def test_missing_dependencies_is_inconclusive(self):
        with patch.dict(sys.modules, {"numpy": None}):
            self.assertEqual(probe.smoke_htp({})["status"], "inconclusive")

    def test_cpu_fallback_is_disabled_before_session(self):
        class SessionOptions:
            def add_session_config_entry(self, key, value):
                self.settings = {key: value}
            def add_provider_for_devices(self, devices, options):
                self.devices = devices
                self.provider_options = options

        class FakeOnnx:
            TensorProto = types.SimpleNamespace(FLOAT=1)
            helper = types.SimpleNamespace(make_node=lambda *a: None, make_tensor_value_info=lambda *a: None,
                make_graph=lambda *a: None, make_model=lambda *a, **k: None, make_opsetid=lambda *a: None)
            numpy_helper = types.SimpleNamespace(from_array=lambda *a: None)
            checker = types.SimpleNamespace(check_model=lambda *a: None)
            save = staticmethod(lambda *a: None)

        def session(*args, **kwargs):
            self.assertEqual(kwargs["sess_options"].settings, {"session.disable_cpu_ep_fallback": "1"})
            self.assertEqual(kwargs["sess_options"].devices, [device])
            self.assertEqual(kwargs["sess_options"].provider_options["backend_path"], "test/QnnHtp.dll")
            raise RuntimeError("test provider failure")

        fake_numpy = types.SimpleNamespace(array=lambda *a, **k: None, float32=1, uint8=2)
        device = types.SimpleNamespace(ep_name="QNNExecutionProvider", device=types.SimpleNamespace(type="NPU"))
        fake_qnn = types.SimpleNamespace(get_ep_name=lambda: "QNNExecutionProvider", get_qnn_htp_path=lambda: "test/QnnHtp.dll")
        fake_ort = types.SimpleNamespace(SessionOptions=SessionOptions, get_ep_devices=lambda: [device],
            OrtHardwareDeviceType=types.SimpleNamespace(NPU="NPU"), InferenceSession=session)
        with patch.dict(sys.modules, {"numpy": fake_numpy, "onnx": FakeOnnx, "onnxruntime": fake_ort}), patch.object(probe, "register_qnn", return_value=fake_qnn):
            result = probe.smoke_htp({})
        self.assertEqual(result["status"], "failed")
        self.assertIn("test provider failure", result["error"])


if __name__ == "__main__":
    unittest.main()
