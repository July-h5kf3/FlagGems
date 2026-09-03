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

"""THead / PPU W8A8 GEMM with the ``mm_w8a8_fp8`` host API from FlagGems#3821.

    mm_w8a8_fp8(a, b, *, out_dtype=None)
    mm_w8a8_fp8_out(a, b, *, out)

Hopper #3821 uses FP8 E4M3 tensor cores. PPU Triton has no ``fp8e4nv``, so this
backend quantizes BF16/FP16/FP32 inputs to INT8 (per-row A, per-column B) and
runs an INT8 GEMM. A/B tiles are moved with AIU via
``tle.load(block_ptr, is_async=True)``. Quantized A/B are cached by storage
identity.

Requires a FlagTree build that lowers INT8 AIU to ``ppu.cp.async.aiu...2d.b8``
(FlagTree #1026 / QCLDC INT8 AIU). Older compilers emit ``.b16`` and fault.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

try:
    from triton.experimental.tle import language as tle_async
except ImportError:  # pragma: no cover
    tle_async = None

logger = logging.getLogger(__name__)

_CACHE_MAX_ENTRIES = int(os.environ.get("FLAGGEMS_MM_W8A8_CACHE_MAX_ENTRIES", "64"))
_AUTO_CACHE_A = os.environ.get("FLAGGEMS_MM_W8A8_AUTO_CACHE_A", "1") != "0"
_A_CACHE: OrderedDict = OrderedDict()
_B_CACHE: OrderedDict = OrderedDict()
_SUPPORTED_FLOAT = {torch.bfloat16, torch.float16, torch.float32}
_INT8_QMAX = 127


@triton.jit
def _grouped_pids(pid, m, n, block_m, block_n, group_m):
    grid_m = tl.cdiv(m, block_m)
    grid_n = tl.cdiv(n, block_n)
    width = group_m * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * group_m, group_m)
    pid_m = group_id * group_m + (pid % group_size)
    pid_n = (pid % width) // group_size
    return pid_m, pid_n


@libentry()
@triton.jit
def _mm_w8a8_int8_kernel(
    A_Q,
    B_Q,
    A_SCALE,
    B_SCALE,
    OUT,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    NUM_WARPS: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m, pid_n = _grouped_pids(pid, M, N, BLOCK_M, BLOCK_N, GROUP_M)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A_Q + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B_Q + offs_n[:, None] * K + offs_k[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        k = k0 * BLOCK_K + offs_k
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_n[:, None] < N) & (k[None, :] < K),
            other=0,
        )
        acc += tl.dot(a, tl.trans(b), out_dtype=tl.int32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    a_scale = tl.load(A_SCALE + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
    b_scale = tl.load(B_SCALE + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    out = (acc.to(tl.float32) * a_scale[:, None] * b_scale[None, :]).to(
        OUT.dtype.element_ty
    )
    tl.store(
        OUT + offs_m[:, None] * N + offs_n[None, :],
        out,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


if tle_async is not None:

    @libentry()
    @triton.jit(do_not_specialize_on_alignment=["A_Q", "B_Q", "OUT"])
    def _mm_w8a8_aiu_kernel(
        A_Q,
        B_Q,
        A_SCALE,
        B_SCALE,
        OUT,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        NUM_WARPS: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        a_block_ptr = tl.make_block_ptr(
            A_Q,
            shape=(M, K),
            strides=(K, 1),
            offsets=(pid_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        # B is physically contiguous [N, K], logically column-major [K, N].
        b_block_ptr = tl.make_block_ptr(
            B_Q,
            shape=(K, N),
            strides=(1, K),
            offsets=(0, pid_n * BLOCK_N),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(0, 1),
        )
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for _ in range(0, tl.cdiv(K, BLOCK_K)):
            a = tle_async.load(
                a_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
                is_async=True,
            )
            b = tle_async.load(
                b_block_ptr,
                boundary_check=(0, 1),
                padding_option="zero",
                is_async=True,
            )
            acc += tl.dot(a, b)
            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        a_scale = tl.load(A_SCALE + offs_m, mask=offs_m < M, other=0.0).to(tl.float32)
        b_scale = tl.load(B_SCALE + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        out = (acc.to(tl.float32) * a_scale[:, None] * b_scale[None, :]).to(
            OUT.dtype.element_ty
        )
        tl.store(
            OUT + offs_m[:, None] * N + offs_n[None, :],
            out,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


def _pick_tiles(m: int, n: int, k: int) -> tuple[int, int, int, int, int]:
    """Return BLOCK_M, BLOCK_N, BLOCK_K, warps, stages.

    Tiles follow FlagTree INT8 AIU GEMM tests: K tile >= 32, small epilogue
    uses 1 warp, 64-wide tiles use 4 warps.
    """
    block_k = 64 if k >= 64 else 32
    if m <= 16:
        block_m = 16
        if n <= 16:
            return block_m, 16, block_k, 1, 2
        if n <= 32:
            return block_m, 32, block_k, 1, 2
        return block_m, 64, block_k, 4, 2
    if m <= 32:
        block_n = 32 if n <= 32 else 64
        return 32, block_n, block_k, 4, 2
    return 64, 64, block_k, 4, 2


def _cache_key(source: torch.Tensor) -> tuple:
    return (
        int(source.data_ptr()),
        int(source.storage_offset()),
        tuple(source.shape),
        tuple(source.stride()),
        source.dtype,
        source.device.type,
        int(source.device.index) if source.device.index is not None else -1,
    )


def _quantize_a_per_row(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    source = a.float()
    scale = source.abs().amax(dim=1).clamp_min(1e-8).div(float(_INT8_QMAX))
    quantized = (
        torch.round(source / scale[:, None])
        .clamp(-_INT8_QMAX, _INT8_QMAX)
        .to(torch.int8)
    )
    return quantized.contiguous(), scale.contiguous()


def _quantize_b_per_col(b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    source = b.float()
    scale = source.abs().amax(dim=0).clamp_min(1e-8).div(float(_INT8_QMAX))
    quantized = (
        torch.round(source / scale[None, :])
        .clamp(-_INT8_QMAX, _INT8_QMAX)
        .to(torch.int8)
        .t()
        .contiguous()
    )
    return quantized, scale.contiguous()


def _store_cache(cache: OrderedDict, key: tuple, value):
    cache[key] = value
    cache.move_to_end(key)
    if len(cache) > _CACHE_MAX_ENTRIES:
        cache.popitem(last=False)
    return value


def _get_cached_a(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not _AUTO_CACHE_A:
        return _quantize_a_per_row(a)
    key = _cache_key(a)
    cached = _A_CACHE.get(key)
    if cached is not None:
        _A_CACHE.move_to_end(key)
        return cached
    return _store_cache(_A_CACHE, key, _quantize_a_per_row(a))


def _get_cached_b(b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    key = _cache_key(b)
    cached = _B_CACHE.get(key)
    if cached is not None:
        _B_CACHE.move_to_end(key)
        return cached
    return _store_cache(_B_CACHE, key, _quantize_b_per_col(b))


def clear_mm_w8a8_fp8_caches() -> None:
    _A_CACHE.clear()
    _B_CACHE.clear()


def _validate_mm_inputs(a, b) -> tuple[int, int, int]:
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("mm_w8a8_fp8 expects torch.Tensor inputs")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("mm_w8a8_fp8 expects rank-2 matrices")
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"incompatible dimensions: {tuple(a.shape)} vs {tuple(b.shape)}"
        )
    if a.dtype not in _SUPPORTED_FLOAT or b.dtype not in _SUPPORTED_FLOAT:
        raise TypeError(
            f"mm_w8a8_fp8 expects floating inputs, got {a.dtype} and {b.dtype}"
        )
    return a.shape[0], b.shape[1], a.shape[1]


def _prepare_inputs(a: torch.Tensor, b: torch.Tensor):
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    a_q, a_scale = _get_cached_a(a)
    b_q, b_scale = _get_cached_b(b)
    return a_q, a_scale, b_q, b_scale


def _launch(
    a_q: torch.Tensor,
    a_scale: torch.Tensor,
    b_q: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    m: int,
    n: int,
    k: int,
) -> torch.Tensor:
    block_m, block_n, block_k, num_warps, num_stages = _pick_tiles(m, n, k)
    logger.debug(
        "GEMS_THEAD MM_W8A8_AIU m=%s n=%s k=%s tiles=(%s,%s,%s) warps=%s aiu=%s",
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        num_warps,
        tle_async is not None,
    )
    with torch_device_fn.device(a_q.device):
        if tle_async is not None:
            grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
            _mm_w8a8_aiu_kernel[grid](
                a_q,
                b_q,
                a_scale,
                b_scale,
                out,
                m,
                n,
                k,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                NUM_WARPS=num_warps,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        else:
            grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
            _mm_w8a8_int8_kernel[grid](
                a_q,
                b_q,
                a_scale,
                b_scale,
                out,
                m,
                n,
                k,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                GROUP_M=4,
                NUM_WARPS=num_warps,
                num_warps=num_warps,
                num_stages=num_stages,
            )
    return out


def mm_w8a8_fp8(a, b, *, out_dtype=None):
    m, n, k = _validate_mm_inputs(a, b)
    a_q, a_scale, b_q, b_scale = _prepare_inputs(a, b)
    dtype = out_dtype or a.dtype
    out = torch.empty((m, n), device=a.device, dtype=dtype)
    return _launch(a_q, a_scale, b_q, b_scale, out, m, n, k)


def mm_w8a8_fp8_out(a, b, *, out):
    m, n, k = _validate_mm_inputs(a, b)
    if out.shape != (m, n):
        raise ValueError(f"out shape must be {(m, n)}, got {tuple(out.shape)}")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    a_q, a_scale, b_q, b_scale = _prepare_inputs(a, b)
    return _launch(a_q, a_scale, b_q, b_scale, out, m, n, k)
