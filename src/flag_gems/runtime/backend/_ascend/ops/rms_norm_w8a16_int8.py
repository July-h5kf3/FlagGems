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

"""Ascend W8A16 RMSNorm.

Activation is 16-bit (FP16/BF16). Weight is grouped INT8 plus per-group scale
because the current CANN/Triton-Ascend data-movement path cannot load FP8
weights from GM into UB. The default group size is 128. The group count is
``ceil(N / group_size)``; a final partial group is supported on every path.

Dispatch:
- Power-of-two N <= 4096 with ``N % group_size == 0``: tiled kernel bound to
  the physical Vector core count. Each core processes consecutive row tiles
  and reuses dequantized, broadcast weights across its tiles.
- Power-of-two N <= 8192 with ``N % group_size == 0``: 1D row kernel.
  Full-row BLOCK_M overflows Ascend UB by a few hundred bytes. Scale is loaded
  uniquely and broadcast via reshape (no ``cols // group_size`` gather, no
  ``(G, 128)`` FP32 2D tile — that layout has a 512B row stride and conflicts
  on UB).
- Power-of-two GROUP_SIZE (ragged or not): tiled 1D kernel
  (GROUPS_PER_TILE chosen so the tile stays <= 8192 elements) to stay in UB.
  The group loop uses ``ceil(N / GROUP_SIZE)`` so the final partial group is
  processed with column masks.
- Non-power-of-two GROUP_SIZE: gather kernel. The reshape broadcast cannot
  express odd group sizes, so the per-column scale is gathered with
  ``cols // GROUP_SIZE`` (scalar path, correctness first).
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


@triton.jit
def prev_multiple_of(a, b):
    return tl.cdiv(a, b) * b - b


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_int8_w8a16_kernel(
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
    # Unique scale load + broadcast. Gathering ``scale[cols // GROUP_SIZE]``
    # is scalar-bound on Ascend (~7x slower than the grouped kernel).
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
def rms_norm_int8_w8a16_blockm_kernel(
    out_ptr,
    in_ptr,
    w_ptr,
    w_scale_ptr,
    M,
    N: tl.constexpr,
    eps,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    columns = tl.arange(0, N)
    weight_q = tl.load(w_ptr + columns).to(tl.float32)
    scales = tl.load(w_scale_ptr + tl.arange(0, N // GROUP_SIZE)).to(tl.float32)
    weight = tl.reshape(
        tl.reshape(weight_q, (N // GROUP_SIZE, GROUP_SIZE)) * scales[:, None], (N,)
    )
    weight = tl.broadcast_to(weight[None, :], (BLOCK_M, N))
    row_offsets = tl.arange(0, BLOCK_M)
    tiles_per_core = tl.cdiv(tl.cdiv(M, BLOCK_M), tl.num_programs(0))
    tile_start = tl.program_id(0) * tiles_per_core
    tile_stop = tl.minimum(tile_start + tiles_per_core, tl.cdiv(M, BLOCK_M))
    for tile in range(tile_start, tile_stop):
        rows = tile * BLOCK_M + row_offsets
        offsets = rows[:, None] * N + columns[None, :]
        mask = rows[:, None] < M
        x = tl.load(in_ptr + offsets, mask=mask, other=0).to(tl.float32)
        variance = tl.sum(x * x, axis=1) / N
        inverse_rms = tl.rsqrt(variance + eps)
        y = x * inverse_rms[:, None] * weight
        tl.store(out_ptr + offsets, y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_int8_w8a16_grouped_kernel(
    out_ptr,
    in_ptr,
    w_ptr,
    w_scale_ptr,
    N,
    eps,
    GROUP_SIZE: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
):
    pid = ext.program_id(0)
    # Contiguous 1D load. A 2D ``(NUM_GROUPS, 128)`` FP32 tile has a 512B
    # row stride (16 UB banks * 32B) and bank-conflicts on Ascend.
    BLOCK: tl.constexpr = NUM_GROUPS * GROUP_SIZE
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(in_ptr + pid * N + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rrms = 1 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w_scale = tl.load(w_scale_ptr + tl.arange(0, NUM_GROUPS)).to(tl.float32)
    y = tl.reshape(
        tl.reshape(x, (NUM_GROUPS, GROUP_SIZE))
        * rrms
        * tl.reshape(w, (NUM_GROUPS, GROUP_SIZE))
        * w_scale[:, None],
        (BLOCK,),
    )
    tl.store(out_ptr + pid * N + cols, y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_int8_w8a16_loop_kernel_fixed(
    out_ptr,
    in_ptr,
    w_ptr,
    w_scale_ptr,
    N,
    eps,
    TILE_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)

    acc = tl.zeros((TILE_N,), dtype=tl.float32)
    num_steps = tl.cdiv(N, TILE_N)

    for step in range(0, num_steps - 1):
        start_n = step * TILE_N
        n_offsets = start_n + tl.arange(0, TILE_N)
        x = tl.load(in_ptr + pid * N + n_offsets).to(tl.float32)
        acc += x * x

    start_n = (num_steps - 1) * TILE_N
    n_offsets = start_n + tl.arange(0, TILE_N)
    mask = n_offsets < N
    x = tl.load(in_ptr + pid * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
    acc += x * x

    var = tl.sum(acc) / N
    rrms = 1 / tl.sqrt(var + eps)

    prev_multiple = prev_multiple_of(N, TILE_N)

    for start_n in range(0, TILE_N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(
            in_ptr + pid * N + n_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        group_ids = n_offsets // GROUP_SIZE
        w = tl.load(w_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
        w_scale = tl.load(w_scale_ptr + group_ids, mask=mask, other=0.0).to(tl.float32)
        y = x * rrms * w * w_scale
        tl.store(out_ptr + pid * N + n_offsets, y, mask=mask)

    for start_n in range(TILE_N, N, TILE_N):
        n_offsets = (prev_multiple - start_n) + tl.arange(0, TILE_N)
        x = tl.load(
            in_ptr + pid * N + n_offsets,
            eviction_policy="evict_first",
        ).to(tl.float32)
        group_ids = n_offsets // GROUP_SIZE
        w = tl.load(w_ptr + n_offsets).to(tl.float32)
        w_scale = tl.load(w_scale_ptr + group_ids).to(tl.float32)
        y = x * rrms * w * w_scale
        tl.store(out_ptr + pid * N + n_offsets, y)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_int8_w8a16_grouped_tiled_kernel(
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
    # ``ceil`` so a final partial group is still visited; its scale is loaded
    # and the out-of-range columns are covered by the existing masks.
    num_groups = tl.cdiv(N, GROUP_SIZE)

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


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_int8_w8a16_gather_kernel(
    out_ptr,
    in_ptr,
    w_ptr,
    w_scale_ptr,
    N,
    eps,
    TILE_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    # Non-power-of-two GROUP_SIZE cannot use the reshape broadcast. Gather the
    # per-column scale (scalar-bound on Ascend; only odd group sizes land here).
    acc = tl.zeros((TILE_N,), dtype=tl.float32)
    for start_n in range(0, N, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(in_ptr + pid * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
        acc += x * x
    rrms = 1 / tl.sqrt(tl.sum(acc) / N + eps)

    for start_n in range(0, N, TILE_N):
        n_offsets = start_n + tl.arange(0, TILE_N)
        mask = n_offsets < N
        x = tl.load(in_ptr + pid * N + n_offsets, mask=mask, other=0.0).to(tl.float32)
        group_ids = n_offsets // GROUP_SIZE
        w = tl.load(w_ptr + n_offsets, mask=mask, other=0.0).to(tl.float32)
        w_scale = tl.load(w_scale_ptr + group_ids, mask=mask, other=0.0).to(tl.float32)
        y = x * rrms * w * w_scale
        tl.store(out_ptr + pid * N + n_offsets, y, mask=mask)


def rms_norm_w8a16_int8(
    x, normalized_shape, weight_int8, weight_scale, eps=1e-5, group_size=128
):
    logger.debug("GEMS_ASCEND RMS_NORM_W8A16_INT8_FORWARD")
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)
    if N <= 0 or group_size <= 0:
        raise ValueError(
            f"normalized_shape product {N} and group_size={group_size} must be positive"
        )
    num_groups = (N + group_size - 1) // group_size
    if weight_int8.dtype != torch.int8:
        raise TypeError(
            f"Ascend W8A16 RMSNorm expects INT8 weight, got {weight_int8.dtype}"
        )
    if weight_scale.numel() != num_groups:
        raise ValueError(
            f"weight_scale numel {weight_scale.numel()} != {num_groups} groups "
            f"(ceil({N}/{group_size}))"
        )
    x = x.contiguous()
    weight_q = weight_int8.contiguous()
    weight_scale = weight_scale.contiguous()
    y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    with torch_device_fn.device(x.device):
        # Power-of-two N <= 8192 with an exact group division: contiguous 1D
        # load + unique scale broadcast. Do not gather on
        # ``cols // group_size`` (~97% scalar) and do not use a ``(G, 128)``
        # FP32 2D tile (512B row stride, UB bank conflict). An exact division
        # of a power-of-two N implies a power-of-two GROUP_SIZE/NUM_GROUPS.
        if N <= 4096 and N == triton.next_power_of_2(N) and N % group_size == 0:
            block_m = (
                1
                if N >= 2048 and M <= 16
                else min(triton.next_power_of_2(max(M, 1)), max(1, 8192 // N))
            )
            cores = torch_device_fn.get_device_limit(torch_device_fn.current_device())[
                "vector_core_num"
            ]
            grid = min(triton.cdiv(M, block_m), cores)
            rms_norm_int8_w8a16_blockm_kernel[grid,](
                y, x, weight_q, weight_scale, M, N, eps, group_size, block_m
            )
        elif N <= 8192 and N == triton.next_power_of_2(N) and N % group_size == 0:
            rms_norm_int8_w8a16_kernel[M,](
                y,
                x,
                weight_q,
                weight_scale,
                N,
                eps,
                N,
                group_size,
                num_groups,
                num_warps=4,
            )
        elif group_size & (group_size - 1) == 0 and group_size <= 8192:
            # Ragged last group is handled in-kernel; the tile stays <= 8192
            # elements (128*128 grouped tile overflows Ascend UB).
            groups_per_tile = max(1, 8192 // group_size)
            rms_norm_int8_w8a16_grouped_tiled_kernel[M,](
                y,
                x,
                weight_q,
                weight_scale,
                N,
                eps,
                GROUP_SIZE=group_size,
                GROUPS_PER_TILE=groups_per_tile,
                num_warps=4,
            )
        else:
            # Non-power-of-two (or oversized) GROUP_SIZE: gather the scale.
            rms_norm_int8_w8a16_gather_kernel[M,](
                y,
                x,
                weight_q,
                weight_scale,
                N,
                eps,
                # 8192-wide tiles in the gather path overflow Ascend UB.
                TILE_N=4096,
                GROUP_SIZE=group_size,
                num_warps=4,
            )
    return y
