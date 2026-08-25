# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""THead / PPU W8A16 RMSNorm.

Activation is 16-bit (FP16/BF16). Weight is grouped INT8 plus per-group
scale (group_size=128).

INT8 weights are typically static, so they are dequantized once per unique
storage and reused. The hot path is then a Gems-like RMSNorm that does not
write ``inv_rms``. CUDA Graph capture after warmup therefore records only
the RMSNorm launch.
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

_DEQUANT_CACHE = {}
_DEQUANT_CACHE_MAX = 16


@triton.jit
def prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


@libentry()
@triton.jit
def dequant_grouped_kernel(
    out_ptr,
    w_ptr,
    scale_ptr,
    N,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = ext.program_id(0)
    cols = pid * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    q = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + cols // GROUP_SIZE, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + cols, q * scale, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_simple_kernel(
    out_ptr,
    in_ptr,
    w_ptr,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(in_ptr + pid * N + cols, mask=mask, other=0.0).to(tl.float32)
    rrms = 1 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    y = (x * rrms).to(in_ptr.dtype.element_ty) * w
    tl.store(out_ptr + pid * N + cols, y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_simple_loop_kernel(
    out_ptr,
    in_ptr,
    w_ptr,
    N,
    eps,
    TILE_N: tl.constexpr,
):
    if tl.constexpr(in_ptr.dtype.element_ty == tl.float16) or tl.constexpr(
        in_ptr.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = in_ptr.dtype.element_ty

    pid = ext.program_id(0)
    acc = tl.zeros((TILE_N,), dtype=tl.float32)
    num_steps = tl.cdiv(N, TILE_N)
    for step in range(0, num_steps - 1):
        n_offsets = step * TILE_N + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets).to(tl.float32)
        acc += x * x
    n_offsets = (num_steps - 1) * TILE_N + tl.arange(0, TILE_N)
    mask = n_offsets < N
    x = tl.load(in_ptr + pid * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
    acc += x * x
    rrms = 1 / tl.sqrt(tl.sum(acc) / N + eps)

    prev_multiple = prev_multiple_of(N, TILE_N)
    for start_n in range(0, TILE_N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(
            in_ptr + pid * N + n_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(cdtype)
        w = tl.load(w_ptr + n_offsets, mask=mask, other=0.0)
        y = (x * rrms).to(in_ptr.dtype.element_ty) * w
        tl.store(out_ptr + pid * N + n_offsets, y, mask=mask)
    for start_n in range(TILE_N, N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        x = tl.load(
            in_ptr + pid * N + n_offsets,
            eviction_policy="evict_first",
        ).to(cdtype)
        w = tl.load(w_ptr + n_offsets)
        y = (x * rrms).to(in_ptr.dtype.element_ty) * w
        tl.store(out_ptr + pid * N + n_offsets, y)


def _dequant_weight(weight_q, weight_scale, group_size, out_dtype):
    n = weight_q.numel()
    key = (weight_q.data_ptr(), weight_scale.data_ptr(), n, out_dtype, group_size)
    cached = _DEQUANT_CACHE.get(key)
    if (
        cached is not None
        and cached.numel() == n
        and cached.dtype == out_dtype
        and cached.device == weight_q.device
    ):
        return cached
    w = torch.empty(n, device=weight_q.device, dtype=out_dtype)
    block = 1024
    dequant_grouped_kernel[triton.cdiv(n, block),](
        w, weight_q, weight_scale, n, group_size, block, num_warps=4
    )
    if len(_DEQUANT_CACHE) >= _DEQUANT_CACHE_MAX:
        _DEQUANT_CACHE.pop(next(iter(_DEQUANT_CACHE)))
    _DEQUANT_CACHE[key] = w
    return w


def rms_norm_w8a16_thead(
    x, normalized_shape, weight_q, weight_scale, eps=1e-5, group_size=128
):
    logger.debug("GEMS_THEAD RMS_NORM W8A16 FORWARD")
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)
    if N % group_size != 0:
        raise ValueError(
            f"normalized_shape product {N} must be divisible by group_size={group_size}"
        )
    if weight_q.dtype != torch.int8:
        raise TypeError(f"PPU W8A16 RMSNorm expects INT8 weight, got {weight_q.dtype}")
    if weight_scale.numel() != N // group_size:
        raise ValueError(
            f"weight_scale numel {weight_scale.numel()} != {N // group_size} groups"
        )
    x = x.contiguous()
    weight_q = weight_q.contiguous()
    weight_scale = weight_scale.contiguous()
    y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    with torch_device_fn.device(x.device):
        w = _dequant_weight(weight_q, weight_scale, group_size, x.dtype)
        if N <= 4096:
            rms_norm_simple_kernel[M,](
                y, x, w, N, eps, triton.next_power_of_2(N), num_warps=4
            )
        else:
            rms_norm_simple_loop_kernel[M,](y, x, w, N, eps, TILE_N=4096, num_warps=4)
    return y


# Public name matches the W8A16 RMSNorm API used by other vendors.
rms_norm_fp8_w8a16 = rms_norm_w8a16_thead
