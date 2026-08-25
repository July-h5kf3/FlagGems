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
scale (group_size=128). PPU Triton can load INT8; the scale is applied
in FP32 after a unique load + reshape broadcast.

Dispatch:
- Power-of-two N <= 8192: 1D row kernel (unique scale + reshape broadcast).
- Otherwise: tiled 1D kernel (GROUPS_PER_TILE=64).
  A BLOCK_M reuse path was slower than one-row programs on PPU for the
  large-M 4096-wide shapes, so it is not used.
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


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_fp8_w8a16_kernel(
    out_ptr,
    in_ptr,
    w_ptr,
    w_scale_ptr,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    pid = ext.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(in_ptr + pid * N + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Unique scale load + broadcast. Gathering scale[cols // GROUP_SIZE]
    # is slower than a reshape broadcast on this backend.
    w_scale = tl.load(w_scale_ptr + tl.arange(0, NUM_GROUPS)).to(tl.float32)
    y = tl.reshape(
        tl.reshape(x, (NUM_GROUPS, GROUP_SIZE))
        * rrms
        * tl.reshape(w, (NUM_GROUPS, GROUP_SIZE))
        * w_scale[:, None],
        (BLOCK_SIZE,),
    )
    tl.store(out_ptr + pid * N + cols, y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_fp8_w8a16_grouped_tiled_kernel(
    out_ptr,
    in_ptr,
    w_ptr,
    w_scale_ptr,
    N,
    eps,
    GROUP_SIZE: tl.constexpr,
    GROUPS_PER_TILE: tl.constexpr,
):
    pid = ext.program_id(0)
    TILE_N: tl.constexpr = GROUPS_PER_TILE * GROUP_SIZE
    num_groups = N // GROUP_SIZE

    acc = 0.0
    for g0 in range(0, num_groups, GROUPS_PER_TILE):
        start_n = g0 * GROUP_SIZE
        cols = start_n + tl.arange(0, TILE_N)
        mask = cols < N
        x = tl.load(in_ptr + pid * N + cols, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x * x)
    rrms = 1 / tl.sqrt(acc / N + eps)

    for g0 in range(0, num_groups, GROUPS_PER_TILE):
        start_n = g0 * GROUP_SIZE
        cols = start_n + tl.arange(0, TILE_N)
        mask = cols < N
        x = tl.load(in_ptr + pid * N + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        groups = g0 + tl.arange(0, GROUPS_PER_TILE)
        gmask = groups < num_groups
        w_scale = tl.load(w_scale_ptr + groups, mask=gmask, other=0.0).to(tl.float32)
        y = tl.reshape(
            tl.reshape(x, (GROUPS_PER_TILE, GROUP_SIZE))
            * rrms
            * tl.reshape(w, (GROUPS_PER_TILE, GROUP_SIZE))
            * w_scale[:, None],
            (TILE_N,),
        )
        tl.store(out_ptr + pid * N + cols, y, mask=mask)


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
    num_groups = N // group_size
    with torch_device_fn.device(x.device):
        if N <= 8192 and N == triton.next_power_of_2(N):
            num_warps = 8 if (N == 4096 and M >= 512) else 4
            rms_norm_fp8_w8a16_kernel[M,](
                y,
                x,
                weight_q,
                weight_scale,
                N,
                eps,
                N,
                group_size,
                num_groups,
                num_warps=num_warps,
            )
        else:
            rms_norm_fp8_w8a16_grouped_tiled_kernel[M,](
                y,
                x,
                weight_q,
                weight_scale,
                N,
                eps,
                GROUP_SIZE=group_size,
                GROUPS_PER_TILE=64,
                num_warps=4,
            )
    return y


# Public name matches the W8A16 RMSNorm API used by other vendors.
rms_norm_fp8_w8a16 = rms_norm_w8a16_thead
