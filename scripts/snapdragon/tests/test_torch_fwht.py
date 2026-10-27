"""CPU-only parity against an independently constructed dense Walsh matrix."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import unittest

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise unittest.SkipTest("Optional FWHT numerical tests require PyTorch; run in the isolated ROCm/probe environment") from error


_SPEC = importlib.util.spec_from_file_location("torch_fwht", Path(__file__).resolve().parents[1] / "torch_fwht.py")
fwht = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fwht)


def dense_hadamard(n):
    # Closed-form Walsh signs, independent of the helper's butterfly algorithm.
    return torch.tensor([[1 if (i & j).bit_count() % 2 == 0 else -1 for j in range(n)]
                         for i in range(n)], dtype=torch.float64, device="cpu")


class TorchFwhtTests(unittest.TestCase):
    def assert_dense_parity(self, x, scale):
        before = x.clone()
        expected = (x.to(torch.float64) @ dense_hadamard(x.shape[-1]).T) * scale
        actual = fwht.hadamard_transform(x, scale=scale)
        tolerances = {torch.float32: (2e-6, 2e-6), torch.float16: (1e-3, 1e-3), torch.bfloat16: (8e-3, 8e-3)}
        rtol, atol = tolerances[x.dtype]
        torch.testing.assert_close(actual, expected.to(dtype=x.dtype), rtol=rtol, atol=atol)
        torch.testing.assert_close(x, before, rtol=0, atol=0)
        self.assertEqual(actual.shape, x.shape)
        self.assertEqual(actual.dtype, x.dtype)
        self.assertEqual(actual.device, x.device)
        self.assertEqual(actual.device.type, "cpu")

    def test_dense_reference_sizes_dtypes_and_scales(self):
        generator = torch.Generator(device="cpu").manual_seed(20261026)
        for n in (32, 64, 128):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                # Binary fractions keep the input exactly representable in all tested dtypes.
                x = (torch.randint(-32, 33, (2, 3, n), generator=generator, device="cpu") / 16).to(dtype)
                for scale in (1 / math.sqrt(n), 1, 0.25):
                    with self.subTest(n=n, dtype=dtype, scale=scale):
                        self.assert_dense_parity(x, scale)

    def test_noncontiguous_and_single_row_inputs(self):
        base = torch.arange(3 * 64, dtype=torch.float32, device="cpu").reshape(3, 64) / 64
        self.assertFalse(base[:, ::2].is_contiguous())
        self.assert_dense_parity(base[:, ::2], 0.25)
        self.assert_dense_parity(base[0], 1 / math.sqrt(64))

    def test_low_precision_inputs_accumulate_in_float32(self):
        for dtype, large in ((torch.float16, 2048), (torch.bfloat16, 256)):
            with self.subTest(dtype=dtype):
                x = torch.tensor([large, 1, -large, 1] * 8, dtype=dtype, device="cpu")
                actual = fwht.hadamard_transform(x)
                self.assertEqual(actual[0].item(), 16)
                self.assert_dense_parity(x, 1)

    def test_scalar_tensor_scale_used_by_gptq(self):
        x = torch.arange(64, dtype=torch.float32, device="cpu").reshape(2, 32)
        scale = 1 / torch.tensor(32, device="cpu").sqrt()
        expected = fwht.hadamard_transform(x, scale=float(scale))
        torch.testing.assert_close(fwht.hadamard_transform(x, scale), expected, rtol=0, atol=0)

    def test_unit_dimension_and_empty_leading_batch(self):
        x = torch.tensor([[2], [-3]], dtype=torch.bfloat16, device="cpu")
        self.assert_dense_parity(x, 0.25)
        empty = torch.empty((0, 2, 32), dtype=torch.float16, device="cpu")
        actual = fwht.hadamard_transform(empty)
        self.assertEqual(actual.shape, empty.shape)
        self.assertEqual(actual.device, empty.device)
        self.assertEqual(actual.dtype, empty.dtype)

    def test_backward_matches_dense_transpose(self):
        x = torch.arange(32, dtype=torch.float32, device="cpu").requires_grad_()
        gradient = torch.arange(32, 64, dtype=torch.float32, device="cpu")
        fwht.hadamard_transform(x, 0.25).backward(gradient)
        expected = (gradient.to(torch.float64) @ dense_hadamard(32)) * 0.25
        torch.testing.assert_close(x.grad, expected.to(torch.float32), rtol=0, atol=0)

    def test_rejects_non_tensor_and_unsupported_dtypes(self):
        with self.assertRaisesRegex(TypeError, "torch.Tensor"):
            fwht.hadamard_transform([1, 2])
        for dtype in (torch.float64, torch.int64, torch.bool, torch.complex64):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(TypeError, "dtype"):
                fwht.hadamard_transform(torch.ones(32, dtype=dtype, device="cpu"))

    def test_rejects_invalid_dimensions_and_layout(self):
        for shape in ((), (0,), (3,), (2, 31), (2, 33)):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "dimension"):
                fwht.hadamard_transform(torch.empty(shape, dtype=torch.float32, device="cpu"))
        sparse = torch.sparse_coo_tensor(torch.tensor([[0]], device="cpu"), torch.tensor([1.0], device="cpu"), size=(32,))
        with self.assertRaisesRegex(ValueError, "layout"):
            fwht.hadamard_transform(sparse)
        with self.assertRaisesRegex(ValueError, "device"):
            fwht.hadamard_transform(torch.empty(32, dtype=torch.float32, device="meta"))

    def test_rejects_invalid_scales(self):
        x = torch.ones(32, dtype=torch.float32, device="cpu")
        for scale in (True, "1", complex(1, 1), torch.tensor(1, device="cpu")):
            with self.subTest(scale=scale), self.assertRaises(TypeError):
                fwht.hadamard_transform(x, scale)
        for scale in (float("nan"), float("inf"), float("-inf"), torch.tensor([1.0], device="cpu")):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                fwht.hadamard_transform(x, scale)


if __name__ == "__main__":
    unittest.main()
