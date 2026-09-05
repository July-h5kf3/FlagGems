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

"""Ascend W8A8 mm (same symbol as Hopper ``mm_w8a8_fp8``).

Public API matches NVIDIA PR #3821: BF16/FP16 inputs are quantized, then
``C = (A_q @ B_q) * a_scale[:, None] * b_scale[None, :]``.

Ascend 910 UB / DataCopy cannot load FP8, so weights and activations stay
INT8 plus per-row / per-column scale. TLE does the INT8 ``tl.dot``. A CommonIR
AIC-only epilogue applies the column scale with ``VDEQF16``, applies the row
scale with a 16x16 diagonal Cube matmul, and writes the requested dtype with a
second FixPipe. No INT32 GM workspace or AIV output pass is used.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Optional

import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext


from .ascendc.compile_fixpipe import (
    ensure_bitcode,
    install_cann90_custom_op_compat,
    makefile_compile,
    source_path,
    symbol as _fixpipe_symbol,
)

logger = logging.getLogger(__name__)

_INT8_MAX = 127.0
_B_CACHE: OrderedDict = OrderedDict()
_B_PACKED_CACHE: OrderedDict = OrderedDict()
_B_CACHE_MAX = 64
_C_WORKSPACE: dict[tuple, torch.Tensor] = {}
_MM_W8A8_OUTPUT_DTYPE = os.environ.get("FLAGGEMS_MM_W8A8_OUTPUT_DTYPE", "bf16").lower()
_FIXPIPE_M_MAJOR = os.environ.get("FLAGGEMS_MM_W8A8_M_MAJOR", "1") == "1"
_FIXPIPE_READY = False


def _init_fixpipe() -> None:
    global _FIXPIPE_READY
    if _FIXPIPE_READY:
        return
    bc = ensure_bitcode()
    try:
        install_cann90_custom_op_compat()

        @al.register_custom_op
        class fixpipe_vdeqf16:  # noqa: N801
            name = "fixpipe_vdeqf16"
            core = al.CORE.CUBE
            pipe = al.PIPE.PIPE_FIX
            mode = al.MODE.SIMD
            symbol = _fixpipe_symbol()
            bitcode = str(bc)
            source = str(source_path())
            compile = makefile_compile()

            def __init__(
                self,
                acc,
                deq,
                row_diag,
                c,
                pid_m,
                pid_n,
                tile_m,
                tile_n,
                acc_stride,
                ldc,
                load_deq,
                load_diag,
                out_bf16,
                out=None,
            ):
                # HIVM CustomOp verifier requires a tensor/memref `outs`
                # segment. C is a GM pointer written in-place, so it stays in
                # `ins`; `out` is a dummy UB tile only to form the op.
                assert out is not None, "fixpipe_vdeqf16 requires a dummy out tensor"
                self.arg_type["pid_m"] = tl.int32
                self.arg_type["pid_n"] = tl.int32
                self.arg_type["tile_m"] = tl.int32
                self.arg_type["tile_n"] = tl.int32
                self.arg_type["acc_stride"] = tl.int32
                self.arg_type["ldc"] = tl.int32
                self.arg_type["load_deq"] = tl.int32
                self.arg_type["load_diag"] = tl.int32
                self.arg_type["out_bf16"] = tl.int32
                del acc, deq, row_diag, c
    except AssertionError as exc:
        if "already used" not in str(exc):
            raise
    _FIXPIPE_READY = True
    logger.info("mm_w8a8_fp8 FixPipe Common IR ready (%s)", bc)


_init_fixpipe()


@libentry()
@triton.jit
def mm_w8a8_fp8_fixpipe_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    deq_ptr,
    row_diag_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    OUT_M: tl.constexpr,
    OUT_N: tl.constexpr,
    N_CORES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DISALLOW_ACC: tl.constexpr,
    M_MAJOR: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    # M/N/K/N_CORES are constexpr so the K trip count, grid and deq path
    # fold. Host pads to full tiles and packs B as contiguous KxN tiles.
    pid = ext.program_id(0)
    grid_m: tl.constexpr = tl.cdiv(OUT_M, BLOCK_M)
    grid_n: tl.constexpr = tl.cdiv(OUT_N, BLOCK_N)
    n_tiles: tl.constexpr = grid_m * grid_n
    cache_deq: tl.constexpr = OUT_N <= 8192

    with al.scope(core_mode="cube"):
        dummy = tl.full([16], 0, tl.float16)
        q = n_tiles // N_CORES
        r = n_tiles % N_CORES
        start = tl.where(pid < r, pid * (q + 1), r * (q + 1) + (pid - r) * q)
        count = q + tl.where(pid < r, 1, 0)
        if M_MAJOR:
            pid_m = start // grid_n
            pid_n = start % grid_n
        else:
            pid_n = start // grid_m
            pid_m = start % grid_m
        for i in tl.range(0, count):
            load_deq = 2 if (cache_deq and i == 0) else (0 if cache_deq else 1)
            load_diag = 1 if (not M_MAJOR or i == 0 or pid_n == 0) else 0
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
            off_m = (pid_m * BLOCK_M).to(tl.int32)
            off_n = (pid_n * BLOCK_N).to(tl.int32)
            tile_m = tl.minimum(BLOCK_M, OUT_M - off_m)
            tile_n = tl.minimum(BLOCK_N, OUT_N - off_n)
            a_block_ptr = tl.make_block_ptr(
                base=a_ptr,
                shape=(M, K),
                strides=(K, 1),
                offsets=(off_m, 0),
                block_shape=(BLOCK_M, BLOCK_K),
                order=(1, 0),
            )
            b_block_ptr = tl.make_block_ptr(
                base=b_ptr + pid_n * K * BLOCK_N,
                shape=(K, BLOCK_N),
                strides=(BLOCK_N, 1),
                offsets=(0, 0),
                block_shape=(BLOCK_K, BLOCK_N),
                order=(1, 0),
            )
            if DISALLOW_ACC:
                for k0 in tl.range(0, K, BLOCK_K, num_stages=2, disallow_acc_multi_buffer=True):
                    a = tl.load(a_block_ptr)
                    b = tl.load(b_block_ptr)
                    acc = tl.dot(a, b, acc, out_dtype=tl.int32)
                    a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
                    b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
            else:
                # Two L0C banks: FixPipe drains one while the next tile MMA fills the other.
                for k0 in tl.range(0, K, BLOCK_K, num_stages=2, disallow_acc_multi_buffer=False):
                    a = tl.load(a_block_ptr)
                    b = tl.load(b_block_ptr)
                    acc = tl.dot(a, b, acc, out_dtype=tl.int32)
                    a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
                    b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
            al.custom(
                "fixpipe_vdeqf16",
                acc,
                deq_ptr,
                row_diag_ptr,
                c_ptr,
                off_m,
                off_n,
                tile_m.to(tl.int32),
                tile_n.to(tl.int32),
                BLOCK_M,
                OUT_N,
                tl.cast(load_deq, tl.int32),
                tl.cast(load_diag, tl.int32),
                tl.cast(OUT_BF16, tl.int32),
                out=dummy,
            )
            if M_MAJOR:
                pid_n = pid_n + 1
                wrap = pid_n == grid_n
                pid_m = pid_m + wrap
                pid_n = tl.where(wrap, 0, pid_n)
            else:
                pid_m = pid_m + 1
                wrap = pid_m == grid_m
                pid_n = pid_n + wrap
                pid_m = tl.where(wrap, 0, pid_m)


@libentry()
@triton.jit
def mm_w8a8_fp8_int32_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    N_CORES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    M_MAJOR: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """INT8 Cube with a plain INT32 FixPipe drain.

    Row and column scales are fused in the existing Vector output kernel.
    This avoids the fixed CommonIR custom-op cost without adding a launch.
    """
    pid = ext.program_id(0)
    grid_m: tl.constexpr = tl.cdiv(M, BLOCK_M)
    grid_n: tl.constexpr = tl.cdiv(N, BLOCK_N)
    n_tiles: tl.constexpr = grid_m * grid_n
    q = n_tiles // N_CORES
    r = n_tiles % N_CORES
    start = tl.where(pid < r, pid * (q + 1), r * (q + 1) + (pid - r) * q)
    count = q + tl.where(pid < r, 1, 0)
    if GROUP_M == 0:
        if M_MAJOR:
            # Keep one A tile adjacent across N tiles. This is faster once M
            # is large and the packed B working set fits in L2.
            pid_m = start // grid_n
            pid_n = start % grid_n
        else:
            # Keep one packed B tile adjacent across M tiles.
            pid_n = start // grid_m
            pid_m = start % grid_m
    for i in tl.range(0, count):
        if GROUP_M > 0:
            # Bound the live A/B working set for wide matrices so both sides
            # are reused before the traversal advances to the next M group.
            tile_id = start + i
            group_width: tl.constexpr = GROUP_M * grid_n
            group_id = tile_id // group_width
            first_m = group_id * GROUP_M
            group_size_m = tl.minimum(grid_m - first_m, GROUP_M)
            tile_in_group = tile_id % group_width
            pid_m = first_m + tile_in_group % group_size_m
            pid_n = tile_in_group // group_size_m
        off_m = (pid_m * BLOCK_M).to(tl.int32)
        off_n = (pid_n * BLOCK_N).to(tl.int32)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr,
            shape=(M, K),
            strides=(K, 1),
            offsets=(off_m, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr + pid_n * K * BLOCK_N,
            shape=(K, BLOCK_N),
            strides=(BLOCK_N, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0),
        )
        for _k0 in tl.range(0, K, BLOCK_K, num_stages=2):
            a = tl.load(a_block_ptr)
            b = tl.load(b_block_ptr)
            acc = tl.dot(a, b, acc, out_dtype=tl.int32)
            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
        c_block_ptr = tl.make_block_ptr(
            base=c_ptr,
            shape=(M, N),
            strides=(N, 1),
            offsets=(off_m, off_n),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        tl.store(c_block_ptr, acc)
        if GROUP_M == 0:
            if M_MAJOR:
                pid_n = pid_n + 1
                wrap = pid_n == grid_n
                pid_m = pid_m + wrap
                pid_n = tl.where(wrap, 0, pid_n)
            else:
                pid_m = pid_m + 1
                wrap = pid_m == grid_m
                pid_n = pid_n + wrap
                pid_m = tl.where(wrap, 0, pid_m)


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def _row_scale_cast_kernel(
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    o_ptr,
    M,
    N,
    stride_cm,
    stride_om,
    BLOCK_N: tl.constexpr,
):
    row = ext.program_id(0)
    nprog = ext.num_programs(0)
    offs = tl.arange(0, BLOCK_N)
    for r in range(row, M, nprog):
        a_scale = tl.load(a_scale_ptr + r)
        for n0 in range(0, N, BLOCK_N):
            n_idx = n0 + offs
            mask = n_idx < N
            c = tl.load(c_ptr + r * stride_cm + n_idx, mask=mask, other=0).to(tl.float32)
            b_scale = tl.load(b_scale_ptr + n_idx, mask=mask, other=0).to(tl.float32)
            tl.store(o_ptr + r * stride_om + n_idx, c * a_scale * b_scale, mask=mask)


def _vector_grid(n: int) -> int:
    return max(1, min(n, 40 * 8))


def _quantize_int8_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xf = x.float()
    scale = xf.abs().amax(dim=1).clamp_min(1e-10) / _INT8_MAX
    q = (xf / scale[:, None]).round().clamp(-128, 127).to(torch.int8)
    return q.contiguous(), scale.contiguous()


def _quantize_int8_cols(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xf = x.float()
    scale = xf.abs().amax(dim=0).clamp_min(1e-10) / _INT8_MAX
    q = (xf / scale[None, :]).round().clamp(-128, 127).to(torch.int8)
    return q.contiguous(), scale.contiguous()


def _pack_deq_u64(scale: torch.Tensor) -> torch.Tensor:
    """Pack fp32 IEEE bits into the low 32 bits of int64 (AscendC uint64 deq)."""
    bits = scale.contiguous().to(torch.float32).view(torch.int32)
    return (bits.to(torch.int64) & 0xFFFFFFFF).contiguous()


def _pack_col_deq(b_s: torch.Tensor) -> torch.Tensor:
    """Pack an N-channel VDEQF16 scale vector."""
    n_pad = _align_up(b_s.numel(), 32)
    packed = _pack_deq_u64(b_s)
    if n_pad != b_s.numel():
        out = packed.new_zeros((n_pad,))
        out[: b_s.numel()] = packed
        return out.contiguous()
    return packed.contiguous()


def _prepare_aic_epilogue(
    a_s: torch.Tensor, b_s: torch.Tensor, block_m: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare scale inputs for the single AIC kernel.

    ``VDEQF16`` applies one N-channel vector to every row.  Fold the largest
    row scale into that vector and provide the remaining per-row ratios as
    packed 16x16 diagonal FP16 matrices.  The CommonIR epilogue consumes both
    inputs inside AIC and writes the final output directly.
    """
    assert block_m % 16 == 0 and block_m <= 128
    assert a_s.numel() % block_m == 0
    row_ref = a_s.abs().amax().clamp_min(1e-10)
    deq = _pack_col_deq(b_s * row_ref)
    row_ratio = (a_s / row_ref).to(torch.float16)
    groups = a_s.numel() // block_m
    m1 = block_m // 16
    row_diag_nd = torch.diag_embed(row_ratio.reshape(groups, block_m))
    # A1 fractal layout consumed by L1->L0A: [group, K1, M1, M0, K0].
    row_diag = (
        row_diag_nd.reshape(groups, m1, 16, m1, 16)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
    )
    return deq, row_diag


def _b_cache_key(b: torch.Tensor) -> tuple:
    return (b.data_ptr(), tuple(b.shape), tuple(b.stride()), b.dtype)


def _get_cached_b(b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = _b_cache_key(b)
    cached = _B_CACHE.get(key)
    if cached is not None:
        _B_CACHE.move_to_end(key)
        return cached
    q, scale = _quantize_int8_cols(b)
    packed = _pack_col_deq(scale)
    item = (q, scale, packed)
    _B_CACHE[key] = item
    if len(_B_CACHE) > _B_CACHE_MAX:
        _B_CACHE.popitem(last=False)
    return item


def _get_cached_b_int8(b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q, scale, _packed = _get_cached_b(b)
    return q, scale


def _c_workspace(device, m: int, n: int, dtype=torch.float16) -> torch.Tensor:
    key = (str(device), m, n, dtype)
    buf = _C_WORKSPACE.get(key)
    if buf is None or buf.shape != (m, n) or buf.dtype != dtype:
        buf = torch.empty((m, n), device=device, dtype=dtype)
        _C_WORKSPACE[key] = buf
    return buf


def clear_mm_w8a8_fp8_caches() -> None:
    _B_CACHE.clear()
    _B_PACKED_CACHE.clear()
    _C_WORKSPACE.clear()


def get_mm_w8a8_fp8_cache_stats() -> dict:
    return {
        "b_int8": len(_B_CACHE),
        "b_packed": len(_B_PACKED_CACHE),
        "cache_max_entries": _B_CACHE_MAX,
        "fixpipe": _FIXPIPE_READY,
    }


def _cube_core_count() -> int:
    """910B has 20 AIC. Mix SIMD ``get_block_idx`` only covers one wave."""
    env = os.environ.get("FLAGGEMS_CUBE_CORES")
    if env:
        return max(1, int(env))
    return 20


def _l0c_bytes(block_m: int, block_n: int) -> int:
    return block_m * block_n * 4


def _can_acc_pingpong(block_m: int, block_n: int) -> bool:
    """910B L0C is 128KB. Two int32 banks must fit for MMA/FixPipe overlap."""
    return _l0c_bytes(block_m, block_n) * 2 <= 128 * 1024


def _pick_fixpipe_tiles(M: int, N: int, K: int) -> tuple[int, int, int]:
    env = os.environ.get("FLAGGEMS_FIXPIPE_TILES")
    if env:
        parts = tuple(int(x) for x in env.split(","))
        if len(parts) == 3:
            return parts
    if N <= 64:
        # Avoid 4x-256x N padding. For the PR shapes, K=512 reduces the fixed
        # loop cost, while smaller M tiles expose enough Cube work.
        if K == 2048:
            block_m = 64 if M <= 64 else (256 if M >= 8192 else 128)
            return block_m, 64, 512
        return (256 if M >= 2048 else 128), 64, 256
    if N >= 65536 and M <= 16:
        # PR #5972 contains M=1..8 with N=248320. A 128-row tile performs up
        # to 128x more work than required; 16 is the Cube alignment minimum.
        return 16, 256, 256
    if M >= 8192 and N in (9216, 12288) and K == 2048:
        return 128, 256, 512
    if N == 1024 and K == 2048:
        if M <= 16:
            return 16, 64, 256
        if M <= 32:
            return 32, 64, 256
        if M <= 64:
            return 64, 64, 512
        if M <= 128:
            return 128, 64, 512
        if M >= 512:
            return 256, 128, 512
    if N == 256 and K == 2048:
        return (64 if M <= 256 else 128), 256, 512
    if N == 2048 and K == 512:
        if M <= 128:
            return 128, 128, 512
        if M >= 1024:
            return 256, 128, 512
    if N == 2048 and K == 4096:
        if M <= 16:
            return 16, 128, 256
        if M <= 32:
            return 32, 128, 256
        if M <= 64:
            return 64, 128, 512
        if M <= 128:
            return 128, 128, 512
        if M <= 256:
            return 256, 128, 512
        if M <= 416:
            return 128, 128, 512
        if 1024 <= M < 8192:
            return 256, 128, 512
    # 128x256 int32 fills the 128KB L0C; smaller MN for ping-pong is slower.
    return 128, 256, 256


def _resolve_out_dtype(a: torch.Tensor, out_dtype: Optional[torch.dtype]) -> torch.dtype:
    if out_dtype is not None:
        return out_dtype
    if _MM_W8A8_OUTPUT_DTYPE == "fp16":
        return torch.float16
    if a.dtype in (torch.float16, torch.bfloat16):
        return a.dtype
    return torch.bfloat16


def _align_up(x: int, align: int) -> int:
    return (x + align - 1) // align * align


def _pad_fixpipe_inputs(a_q, b_q, a_s, b_s, M, N, K, block_m, block_n, block_k):
    """Pad A/B and pack B into contiguous (N-tile, K-tile, BK, BN) storage."""
    m_pad = _align_up(M, max(16, block_m))
    n_pad = _align_up(N, max(32, block_n))
    k_pad = _align_up(K, max(32, block_k))
    if (m_pad, k_pad) == (M, K):
        a_pad, a_s_pad = a_q, a_s
    else:
        a_pad = a_q.new_zeros((m_pad, k_pad))
        a_pad[:M, :K] = a_q
        a_s_pad = a_s.new_zeros((m_pad,))
        a_s_pad[:M] = a_s

    if n_pad == N:
        b_s_pad = b_s
    else:
        b_s_pad = b_s.new_zeros((n_pad,))
        b_s_pad[:N] = b_s

    packed_key = (
        b_q.data_ptr(),
        tuple(b_q.shape),
        tuple(b_q.stride()),
        b_q.dtype,
        k_pad,
        n_pad,
        block_k,
        block_n,
    )
    packed = _B_PACKED_CACHE.get(packed_key)
    if packed is None:
        if (n_pad, k_pad) == (N, K):
            b_pad = b_q
        else:
            b_pad = b_q.new_zeros((k_pad, n_pad))
            b_pad[:K, :N] = b_q
        b_tiles = (
            b_pad.reshape(k_pad // block_k, block_k, n_pad // block_n, block_n)
            .permute(2, 0, 1, 3)
            .contiguous()
        )
        # Keep the source tensor alive so a recycled data_ptr cannot hit this entry.
        _B_PACKED_CACHE[packed_key] = (b_q, b_tiles)
        if len(_B_PACKED_CACHE) > _B_CACHE_MAX:
            _B_PACKED_CACHE.popitem(last=False)
    else:
        _B_PACKED_CACHE.move_to_end(packed_key)
        b_tiles = packed[1]
    return a_pad, b_tiles, a_s_pad, b_s_pad, m_pad, n_pad, k_pad


def _launch_fixpipe(a_q, b_q, a_s, b_s, out, M, N, K, deq=None):
    orig_m, orig_n = M, N
    block_m, block_n, block_k = _pick_fixpipe_tiles(M, N, K)
    block_m = min(block_m, 128)
    a_q, b_q, a_s, b_s, M, N, K = _pad_fixpipe_inputs(
        a_q, b_q, a_s, b_s, M, N, K, block_m, block_n, block_k
    )
    del deq
    deq, row_diag = _prepare_aic_epilogue(a_s, b_s, block_m)
    n_tiles = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
    wave = _cube_core_count()
    grid = min(n_tiles, wave)
    mm_w8a8_fp8_fixpipe_kernel[grid,](
        a_q,
        b_q,
        out,
        deq,
        row_diag,
        M,
        N,
        K,
        orig_m,
        orig_n,
        grid,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        DISALLOW_ACC=not _can_acc_pingpong(block_m, block_n),
        M_MAJOR=_FIXPIPE_M_MAJOR,
        OUT_BF16=1 if out.dtype == torch.bfloat16 else 0,
        mix_mode="aic",
        num_warps=1,
        num_stages=2,
        optimize_dynamic_offset=True,
        unit_flag=False,
        limit_auto_multi_buffer_of_local_buffer="no-l0c",
    )
    assert orig_n % 16 == 0, "AIC VDEQF16 path requires N to be a multiple of 16"
    return out


def _launch(a_q, b_q, a_s, b_s, out, M, N, K, deq=None):
    return _launch_fixpipe(a_q, b_q, a_s, b_s, out, M, N, K, deq=deq)


def mm_w8a8_fp8(a, b, *, out_dtype: Optional[torch.dtype] = None):
    logger.debug("GEMS_ASCEND MM_W8A8_FP8")
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    a_q, a_s = _quantize_int8_rows(a)
    b_q, b_s, deq = _get_cached_b(b)
    out = torch.empty((M, N), device=a.device, dtype=_resolve_out_dtype(a, out_dtype))
    with torch_device_fn.device(a.device):
        return _launch(a_q, b_q, a_s, b_s, out, M, N, K, deq=deq)


def mm_w8a8_fp8_out(a, b, *, out):
    logger.debug("GEMS_ASCEND MM_W8A8_FP8_OUT")
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    assert out.shape == (M, N), "incompatible output shape"
    a_q, a_s = _quantize_int8_rows(a)
    b_q, b_s, deq = _get_cached_b(b)
    with torch_device_fn.device(a.device):
        return _launch(a_q, b_q, a_s, b_s, out, M, N, K, deq=deq)
