"""Isolated PyTorch Walsh-Hadamard transform; no package or import substitution.

Semantics for supported inputs follow the public FHT API: each last-dimension
row is multiplied by the Sylvester Hadamard matrix, then by ``scale``.
https://github.com/Dao-AILab/fast-hadamard-transform
Unlike upstream implicit padding, this helper explicitly requires a power of
two. Butterfly arithmetic and scaling accumulate in float32 on the input device.
"""
from __future__ import annotations

import math
from numbers import Real

import torch


_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _real_scale(scale) -> float:
    # GPTQ 4.2 also passes a zero-dimensional CPU tensor as the scale argument.
    if isinstance(scale, torch.Tensor):
        if scale.ndim != 0:
            raise ValueError("scale must be a scalar, not a tensor with dimensions")
        if not scale.is_floating_point():
            raise TypeError("scale tensor must have a real floating-point dtype")
        if scale.device.type not in ("cpu", "cuda"):
            raise ValueError("scale tensor device must be CPU or CUDA/HIP")
        value = float(scale.detach().item())
    elif isinstance(scale, Real) and not isinstance(scale, bool):
        value = float(scale)
    else:
        raise TypeError("scale must be a real number or a floating-point scalar tensor")
    if not math.isfinite(value):
        raise ValueError("scale must be finite")
    return value


def hadamard_transform(x: torch.Tensor, scale=1) -> torch.Tensor:
    """Return ``x @ H.T * scale`` along the last dimension, preserving dtype/device.

    ``x`` must be a strided CPU or CUDA/HIP tensor with at least one dimension,
    dtype float32/float16/bfloat16 and a positive power-of-two last dimension.
    Leading dimensions, including empty batches, are retained. Inputs are never
    modified. This implementation provides no native-kernel performance claim.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("x dtype must be float32, float16 or bfloat16")
    if x.layout != torch.strided:
        raise ValueError("x must have strided tensor layout")
    if x.device.type not in ("cpu", "cuda"):
        raise ValueError("x device must be CPU or CUDA/HIP")
    if x.ndim == 0:
        raise ValueError("x must have at least one dimension")
    n = x.shape[-1]
    if n <= 0 or n & (n - 1):
        raise ValueError("x last dimension must be a positive power of two")
    factor = _real_scale(scale)
    values = x.to(dtype=torch.float32)
    width = 1
    while width < n:
        pairs = values.reshape(*x.shape[:-1], n // (2 * width), 2, width)
        left, right = pairs[..., 0, :], pairs[..., 1, :]
        values = torch.cat((left + right, left - right), dim=-1).reshape(x.shape)
        width *= 2
    return (values * factor).to(dtype=x.dtype)
