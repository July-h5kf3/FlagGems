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

"""Hygon RMSNorm with grouped INT8 weights and 16-bit activations."""

import logging
import math
from collections.abc import Sequence

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit(do_not_specialize=["M", "eps"])
def rms_norm_w8a16_int8_tile_kernel(
    output,
    inputs,
    weight_int8,
    weight_scale,
    M,
    N: tl.constexpr,
    eps,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_INT64: tl.constexpr,
):
    program_id = tl.program_id(0)
    if USE_INT64:
        row_start = program_id.to(tl.int64) * BLOCK_M
    else:
        row_start = program_id * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    columns = tl.arange(0, BLOCK_N)
    valid_columns = columns < N
    offsets = rows[:, None] * N + columns[None, :]
    valid = (rows[:, None] < M) & valid_columns[None, :]

    activations = tl.load(inputs + offsets, valid, other=0.0).to(tl.float32)
    mean_square = tl.sum(activations * activations, axis=1) / N
    inverse_rms = tl.rsqrt(mean_square + eps)

    quantized_weight = tl.load(weight_int8 + columns, valid_columns, other=0).to(
        tl.float32
    )
    scales = tl.load(weight_scale + columns // GROUP_SIZE, valid_columns, other=0.0).to(
        tl.float32
    )
    # Reuse FP32 dequantized weights across the rows in this program.
    weight = quantized_weight * scales
    normalized = activations * inverse_rms[:, None] * weight[None, :]
    tl.store(output + offsets, normalized, valid)


def rms_norm_w8a16_int8(
    x: torch.Tensor,
    normalized_shape: Sequence[int],
    weight_int8: torch.Tensor,
    weight_scale: torch.Tensor,
    eps: float = 1e-5,
    group_size: int = 128,
) -> torch.Tensor:
    """Normalize trailing dimensions using FP32 grouped weight dequantization.

    ``weight_int8`` contains one shared weight per normalized element. Scales
    contain ``ceil(N / group_size)`` elements, including the final partial group.
    Accumulation, inverse RMS and weight multiplication use FP32; the output
    has the input shape, device and dtype.
    """
    logger.debug("GEMS_HYGON RMS_NORM_W8A16_INT8_FORWARD")
    normalized_shape = tuple(normalized_shape)
    if not normalized_shape or any(
        not isinstance(size, int) or size <= 0 for size in normalized_shape
    ):
        raise ValueError("normalized_shape must contain positive integer dimensions")
    if (
        len(normalized_shape) > x.ndim
        or tuple(x.shape[-len(normalized_shape) :]) != normalized_shape
    ):
        raise ValueError("normalized_shape must match the trailing input dimensions")
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError("group_size must be a positive integer")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("Hygon W8A16 RMSNorm expects FP16 or BF16 input")
    if weight_int8.dtype != torch.int8:
        raise TypeError("Hygon W8A16 RMSNorm expects INT8 weight")
    if weight_scale.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("weight_scale must be FP16, BF16 or FP32")
    if x.device.type != "cuda":
        raise ValueError("Hygon W8A16 RMSNorm expects device tensors")
    if x.device != weight_int8.device or x.device != weight_scale.device:
        raise ValueError("input, weight and scale must be on the same device")
    if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative")

    N = math.prod(normalized_shape)
    M = x.numel() // N
    if weight_int8.numel() != N:
        raise ValueError(f"weight_int8 must contain {N} elements")
    num_groups = triton.cdiv(N, group_size)
    if weight_scale.numel() != num_groups:
        raise ValueError(f"weight_scale must contain {num_groups} group scales")

    x = x.contiguous()
    weight_int8 = weight_int8.contiguous()
    weight_scale = weight_scale.contiguous()
    output = torch.empty_like(x)
    if M == 0:
        return output

    block_n = triton.next_power_of_2(N)
    if N <= 256:
        if M <= 256:
            block_m, num_warps = 1, 1
        elif M <= 2048:
            block_m, num_warps = 4, 2
        else:
            block_m, num_warps = 8, 4
    elif N == 2048:
        if M <= 256:
            block_m, num_warps = 1, 4
        elif M <= 2048:
            block_m, num_warps = 2, 4
        else:
            block_m, num_warps = 1, 2
    else:
        block_m = 1
        num_warps = 4 if block_n <= 4096 else 8

    # Native JIT keeps the selected warp count when M changes for the same N.
    with torch_device_fn.device(x.device):
        rms_norm_w8a16_int8_tile_kernel[(triton.cdiv(M, block_m),)](
            output,
            x,
            weight_int8,
            weight_scale,
            M,
            N,
            eps,
            GROUP_SIZE=group_size,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            USE_INT64=x.numel() > 2**31 - 1,
            num_warps=num_warps,
        )
    return output
