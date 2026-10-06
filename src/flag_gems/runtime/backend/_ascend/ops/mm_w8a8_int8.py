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

"""Ascend matrix multiplication of prequantized INT8 inputs and FP32 scales."""

from __future__ import annotations

import logging
import math
import os

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

_FIXPIPE_M_MAJOR = os.environ.get("FLAGGEMS_MM_W8A8_M_MAJOR", "1") == "1"


@libentry()
@triton.jit
def _mm_w8a8_tiny_kernel(
    A,
    BT,
    SA,
    SB,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    OM: tl.constexpr,
    ON: tl.constexpr,
):
    # For K <= 64, every integer product and partial sum is exact in FP32.
    tiles: tl.constexpr = tl.cdiv(N, BN)
    tile = ext.program_id(0)
    row = tile // tiles
    col = tile % tiles * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    a = tl.load(A + row * K + kk, kk < K, other=0).to(tl.float32)
    b = tl.load(
        BT + col[:, None] * K + kk[None, :],
        (col[:, None] < N) & (kk[None, :] < K),
        other=0,
    ).to(tl.float32)
    acc = tl.sum(b * a[None, :], 1)
    sa = tl.load(SA + row)
    sb = tl.load(SB + col, col < N, other=0)
    tl.store(OUT + row * OM + col * ON, acc * sa * sb, col < N)


@libentry()
@triton.jit
def _mm_w8a8_mixed_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    SA,
    SB,
    OUT_M: tl.constexpr,
    OUT_N: tl.constexpr,
    SCM: tl.constexpr,
    SCN: tl.constexpr,
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
    """INT8 Cube matmul with FP32 row/column scaling in the same mixed kernel."""
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
        rows = off_m + tl.arange(0, BLOCK_M)
        cols = off_n + tl.arange(0, BLOCK_N)
        sa = tl.load(SA + rows, rows < OUT_M, other=0)
        sb = tl.load(SB + cols, cols < OUT_N, other=0)
        val = acc.to(tl.float32) * sa[:, None] * sb[None, :]
        tl.store(
            c_ptr + rows[:, None] * SCM + cols[None, :] * SCN,
            val,
            (rows[:, None] < OUT_M) & (cols[None, :] < OUT_N),
        )
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
@triton.jit
def mm_w8a8_int8_int32_kernel(
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

    Row and column scales are applied in the Vector output kernel.
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
@triton.jit
def _scale_int32_static_kernel(
    C, SA, SB, OUT, N: tl.constexpr, NP: tl.constexpr, R: tl.constexpr, X: tl.constexpr
):
    tile = ext.program_id(0)
    cols: tl.constexpr = N // X
    rr = tile // cols * R + tl.arange(0, R)
    cc = tile % cols * X + tl.arange(0, X)
    v = tl.load(C + rr[:, None] * NP + cc[None, :]).to(tl.float32)
    a = tl.load(SA + rr)
    b = tl.load(SB + cc)
    tl.store(OUT + rr[:, None] * N + cc[None, :], v * a[:, None] * b[None, :])


@libentry()
@triton.jit
def _scale_int32_dense_kernel(
    C,
    SA,
    SB,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    NP: tl.constexpr,
    R: tl.constexpr,
    X: tl.constexpr,
):
    # Full, contiguous output tiles avoid generic masked gather/scatter code.
    pid = ext.program_id(0)
    columns: tl.constexpr = N // X
    tiles: tl.constexpr = (M // R) * columns
    for tile in range(pid, tiles, ext.num_programs(0)):
        rr = tile // columns * R + tl.arange(0, R)
        cc = tile % columns * X + tl.arange(0, X)
        value = tl.load(C + rr[:, None] * NP + cc[None, :]).to(tl.float32)
        row_scale = tl.load(SA + rr)
        col_scale = tl.load(SB + cc)
        tl.store(
            OUT + rr[:, None] * N + cc[None, :],
            value * row_scale[:, None] * col_scale[None, :],
        )


@libentry()
@triton.jit
def _scale_int32_tiles_kernel(
    C,
    A,
    B,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    NP: tl.constexpr,
    R: tl.constexpr,
    X: tl.constexpr,
    OM: tl.constexpr,
    ON: tl.constexpr,
):
    pid = ext.program_id(0)
    cols: tl.constexpr = tl.cdiv(N, X)
    tiles: tl.constexpr = tl.cdiv(M, R) * cols
    for tile in range(pid, tiles, ext.num_programs(0)):
        r = tile // cols * R + tl.arange(0, R)
        col = tile % cols * X + tl.arange(0, X)
        c = tl.load(
            C + r[:, None] * NP + col[None, :],
            (r[:, None] < M) & (col[None, :] < N),
            other=0,
        ).to(tl.float32)
        a = tl.load(A + r, r < M, other=0)
        b = tl.load(B + col, col < N, other=0)
        tl.store(
            OUT + r[:, None] * OM + col[None, :] * ON,
            c * a[:, None] * b[None, :],
            (r[:, None] < M) & (col[None, :] < N),
        )


def _cube_core_count() -> int:
    """910B has 20 AIC. Mix SIMD ``get_block_idx`` only covers one wave."""
    env = os.environ.get("FLAGGEMS_CUBE_CORES")
    if env:
        return max(1, int(env))
    return 20


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


def _align_up(x: int, align: int) -> int:
    return (x + align - 1) // align * align


def _pad_fixpipe_inputs(a_q, b_q, a_s, b_s, M, N, K, block_m, block_n, block_k):
    """Prepare current inputs; every copy remains visible to graph replay."""
    m_pad = _align_up(M, max(16, block_m))
    n_pad = _align_up(N, max(32, block_n))
    k_pad = _align_up(K, max(32, block_k))
    if (m_pad, k_pad) == (M, K):
        a_pad = a_q.contiguous()
    else:
        a_pad = a_q.new_zeros((m_pad, k_pad))
        a_pad[:M, :K] = a_q
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
    return a_pad, b_tiles, a_s, b_s, m_pad, n_pad, k_pad


def _pick_int32_tiles(M: int, N: int, K: int) -> tuple[int, int, int]:
    if os.environ.get("FLAGGEMS_FIXPIPE_TILES"):
        return _pick_fixpipe_tiles(M, N, K)
    if N == 1024 and K == 2048 and 256 < M <= 448:
        return _align_up(triton.cdiv(M, 2), 16), 128, 512
    # Keep the whole short M dimension without padding it to 256 rows.
    if N == 2048 and K == 4096 and 128 < M <= 256:
        return _align_up(M, 16), 128, 512
    # Short K fits one 512-wide panel. Compact M tiles reduce padding for
    # small matrices; two M tiles balance the longest worker for larger M.
    if N == 2048 and K == 512 and 128 < M <= 224:
        return _align_up(M, 16), 128, 512
    if N == 2048 and K == 512 and 256 < M <= 512:
        return _align_up(triton.cdiv(M, 2), 16), 128, 512
    # Match short-wide tiles to the 20 Cube workers without excessive M padding.
    if N == 12288 and K == 2048 and 0 < M <= 32:
        return _align_up(M, 16), 128, 512
    if N == 9216 and K == 2048 and 32 < M <= 64:
        return _align_up(M, 16), 512, 256
    # Ascend block pointers and INT8 dot support M multiples of 16, not
    # only powers of two. Keep the L0C tile within 128 KiB.
    if 8192 <= N < 65536 and K == 2048 and 0 < M <= 512:
        m_tiles = triton.cdiv(M, 256)
        block_m = _align_up(triton.cdiv(M, m_tiles), 16)
        if block_m & (block_m - 1):
            return block_m, (256 if block_m <= 128 else 128), 512
    if N == 1024 and K == 2048 and 128 < M <= 256:
        return _align_up(triton.cdiv(M, 2), 16), 128, 512
    # Avoid padding short matrices to 128 rows, and reduce K-loop overhead.
    if N == 2048 and K == 512 and M <= 64:
        return max(16, triton.next_power_of_2(M)), 128, 512
    if N == 1024 and K == 2048 and M <= 32:
        return max(16, triton.next_power_of_2(M)), 64, 512
    if 8192 <= N < 65536 and K == 2048 and M <= 64:
        return max(16, triton.next_power_of_2(M)), 256, 512
    # Avoid a second tile iteration on the busiest cores for this narrow N.
    if N == 256 and K == 2048 and 320 < M <= 512:
        return 128, 64, 512
    # Narrow outputs need enough independent Cube tiles. BN=64 also avoids
    # the expensive INT8 packing path seen with 32-column panels on 910B.
    if 64 <= N <= 256 and K >= 1024 and (M <= 512 or N == 64):
        block_m = (
            16
            if M <= 64 or (N == 64 and M <= 128)
            else (32 if M <= 256 else (64 if M <= 512 else 128))
        )
        return block_m, 64, 512
    if N == 1024 and K == 2048 and 256 < M < 512:
        return 128, 256, 512
    if N >= 8192 and K == 2048 and 128 < M <= 512:
        # Do not add a 256-row padding penalty for M in (256, 384].
        if _align_up(M, 256) == _align_up(M, 128):
            return 256, 128, 512
    if 32 <= M <= 64 and N <= 256 and K <= 128:
        return 64, 64, max(32, triton.next_power_of_2(K))
    if 64 <= M <= 192 and N <= 512 and K <= 512:
        return 64, 128, min(256, max(32, triton.next_power_of_2(K)))
    if 192 < M <= 256 and N <= 1024 and K <= 1024:
        return 128, 128, 256
    if 256 < M <= 512 and N <= 1024 and K <= 1024:
        return 128, 256, min(512, max(32, triton.next_power_of_2(K)))
    block_m, block_n, block_k = _pick_fixpipe_tiles(M, N, K)
    if M <= 128 and K <= 256:
        block_m = min(block_m, max(16, triton.next_power_of_2(M)))
        block_n = min(block_n, max(32, triton.next_power_of_2(N)))
        block_k = min(block_k, max(32, triton.next_power_of_2(K)))
    return block_m, block_n, block_k


def _prepare_mm_w8a8_kernel(a_q, b_q, a_s, b_s, out, M, N, K):
    """Prepare a callable that executes only matmul and output scaling.

    Padding and layout copies are part of the public call and graph capture.
    Captured tensors remain alive in the returned closure.
    """
    if M <= 8 and N in (16, 32, 64) and K in (16, 32, 64):
        a_q = a_q.contiguous()
        b_t = b_q.t().contiguous()
        bn, bk = triton.next_power_of_2(N), triton.next_power_of_2(K)
        grid = M * triton.cdiv(N, bn)

        def call():
            _mm_w8a8_tiny_kernel[(grid,)](
                a_q,
                b_t,
                a_s,
                b_s,
                out,
                M,
                N,
                K,
                bn,
                bk,
                *out.stride(),
                multibuffer=False,
            )

        return call, {
            "path": "tiny_vector",
            "tiles": [1, bn, bk],
            "kernel_count": 1,
            "workspace_bytes": 0,
        }

    mixed = (
        M >= 1024
        and N >= 1024
        and K >= 1024
        and M * N * K >= 2048**3
        and out.is_contiguous()
    )
    # Large short-K and narrow-N outputs are dominated by the full INT32
    # workspace round trip; mixed execution keeps only per-core tile scratch.
    mixed_extra = out.is_contiguous() and (
        (M >= 1024 and N >= 1024 and K == 512)
        or (M >= 8192 and N in (64, 256) and K >= 1024)
    )
    mixed = mixed or mixed_extra
    if mixed and not os.environ.get("FLAGGEMS_FIXPIPE_TILES"):
        if N == 64:
            block_m, block_n, block_k = 128, 64, 512
        elif K == 512:
            block_m, block_n, block_k = 128, 256, 512
        else:
            block_m, block_n, block_k = 256, 128, (256 if N >= 8192 else 512)
    else:
        block_m, block_n, block_k = _pick_int32_tiles(M, N, K)
    a_q, b_q, a_s, b_s, mp, np, kp = _pad_fixpipe_inputs(
        a_q, b_q, a_s, b_s, M, N, K, block_m, block_n, block_k
    )
    grid = min(triton.cdiv(mp, block_m) * triton.cdiv(np, block_n), _cube_core_count())
    opts = dict(
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        M_MAJOR=_FIXPIPE_M_MAJOR,
        GROUP_M=0,
        num_warps=1,
        num_stages=2,
        optimize_dynamic_offset=True,
        unit_flag=False,
        limit_auto_multi_buffer_of_local_buffer="no-l0c",
    )
    if mixed:
        # One CV workspace slot keeps persistent mixed tiles synchronized on CANN 9.1.
        opts["set_workspace_multibuffer"] = 1

        def call():
            _mm_w8a8_mixed_kernel[(grid,)](
                a_q,
                b_q,
                out,
                a_s,
                b_s,
                M,
                N,
                *out.stride(),
                mp,
                np,
                kp,
                grid,
                **opts,
            )

        # The compiler uses one INT32 tile per block, not a full MxN buffer.
        return call, {
            "path": "mixed",
            "tiles": [block_m, block_n, block_k],
            "kernel_count": 1,
            "workspace_bytes": block_m * block_n * 4 * grid,
        }

    # Per-call ownership is required for concurrent streams and graph captures.
    acc = torch.empty((mp, np), dtype=torch.int32, device=out.device)
    if M >= 64 and N <= 1024:
        rows = 8 if M <= 64 else 16
        cols = min(256, triton.next_power_of_2(N))
    else:
        rows, cols = 4, min(1024, triton.next_power_of_2(N))
    if not out.is_contiguous():
        # Strided stores need additional index/scatter buffers in UB.
        rows, cols = 4, min(256, triton.next_power_of_2(N))
    static_output = (
        out.is_contiguous()
        and 16 <= M <= 128
        and M % 16 == 0
        and N in (32, 64, 128, 256)
    )
    if static_output:
        rows, cols = 16, N
    dense_output = (
        not static_output
        and out.is_contiguous()
        and (M >= 64 or (N >= 8192 and M >= 8))
        and M % 8 == 0
        and N % 64 == 0
    )
    if dense_output:
        cols = min(1024, N & -N)
        rows = 8
        if N in (9216, 12288) and M <= 512:
            if N == 12288:
                rows, cols = 8, 2048
            elif M % 16 == 0:
                rows, cols = 16, 1024
        if N == 1024 and M >= 1024 and M % 16 == 0:
            rows, cols = 16, 256
        if N <= 256 and M > 128:
            rows = 32 if M >= 1024 and M % 32 == 0 else (16 if M % 16 == 0 else 8)
    grid_v = min(40, triton.cdiv(M, rows) * triton.cdiv(N, cols))

    def call():
        mm_w8a8_int8_int32_kernel[(grid,)](
            a_q,
            b_q,
            acc,
            mp,
            np,
            kp,
            grid,
            **opts,
        )
        if static_output:
            _scale_int32_static_kernel[(M // rows,)](
                acc,
                a_s,
                b_s,
                out,
                N,
                np,
                rows,
                cols,
            )
        elif dense_output:
            _scale_int32_dense_kernel[(grid_v,)](
                acc, a_s, b_s, out, M, N, np, rows, cols
            )
        else:
            _scale_int32_tiles_kernel[(grid_v,)](
                acc,
                a_s,
                b_s,
                out,
                M,
                N,
                np,
                rows,
                cols,
                *out.stride(),
            )

    return call, {
        "path": "int32_vector",
        "tiles": [block_m, block_n, block_k],
        "scale_tile": [rows, cols],
        "static_output": static_output,
        "dense_output": dense_output,
        "epilogue": "triton",
        "kernel_count": 2,
        "workspace_bytes": mp * np * 4,
    }


def _launch(a_q, b_q, a_s, b_s, out, M, N, K):
    call, _ = _prepare_mm_w8a8_kernel(a_q, b_q, a_s, b_s, out, M, N, K)
    call()
    return out


@libentry()
@triton.jit
def finish_mm_kernel(
    PARTIAL,
    SA,
    SB,
    BIAS,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    PM: tl.constexpr,
    PN: tl.constexpr,
    OM: tl.constexpr,
    ON: tl.constexpr,
    PARTS: tl.constexpr,
    APPLY_SCALES: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < M * N
    rows = offsets // N
    columns = offsets % N
    if PARTIAL.dtype.element_ty == tl.int32:
        # With K < 2**31, both radix-2**24 limbs convert exactly to FP32.
        high = tl.full((BLOCK,), 0, tl.int32)
        low = tl.full((BLOCK,), 0, tl.int32)
        for part in range(PARTS):
            partial = tl.load(
                PARTIAL + part * (PM * PN) + rows * PN + columns, valid, other=0
            )
            low += partial & 0xFFFFFF
            high += (partial >> 24) + (low >> 24)
            low = low & 0xFFFFFF
        value = high.to(tl.float32) * 16777216.0 + low.to(tl.float32)
    else:
        value = tl.full((BLOCK,), 0, tl.float32)
        for part in range(PARTS):
            value += tl.load(
                PARTIAL + part * (PM * PN) + rows * PN + columns, valid, other=0
            )
    if APPLY_SCALES:
        scale_a = tl.load(SA + rows, valid, other=0)
        scale_b = tl.load(SB + columns, valid, other=0)
        value = value * scale_a * scale_b
    if HAS_BIAS:
        value += tl.load(BIAS + columns, valid, other=0).to(tl.float32)
    tl.store(OUT + rows * OM + columns * ON, value, valid)


def scaled_mm_arguments(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, tuple[int, ...]
]:
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("A and B must be prequantized INT8 tensors")
    if a.dtype != torch.int8 or b.dtype != torch.int8:
        raise TypeError("A and B must be prequantized INT8 tensors")
    if a.ndim < 1 or b.ndim != 2 or a.shape[-1] != b.shape[0]:
        raise ValueError("expected A[...,K] and B[K,N]")
    if a.device.type != "npu" or a.device != b.device:
        raise ValueError("A and B must be on the same NPU")
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("out_dtype must be FP16, BF16 or FP32")
    k, n = b.shape
    if k >= 2**31:
        raise ValueError("K must be smaller than 2**31")
    m = math.prod(a.shape[:-1])

    def _normalize(scale, size, name):
        if not isinstance(scale, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if scale.dtype != torch.float32 or scale.device != a.device:
            raise ValueError(f"{name} must be FP32 on the input device")
        if scale.numel() not in (1, size):
            raise ValueError(f"{name} must be scalar or contain {size} values")
        flat = scale.reshape(-1)
        if flat.numel() == 1:
            return flat.expand(size).contiguous()
        return flat.contiguous()

    sa = _normalize(scale_a, m, "scale_a")
    sb = _normalize(scale_b, n, "scale_b")
    if bias is not None:
        if not isinstance(bias, torch.Tensor):
            raise TypeError("bias must be a tensor")
        if bias.device != a.device or bias.dtype != out_dtype or bias.numel() != n:
            raise ValueError(
                "bias must contain N values of the output dtype on the input device"
            )
        bias = bias.reshape(-1).contiguous()
    return a.reshape(m, k), sa, sb, bias, (*a.shape[:-1], n)


def scaled_mm_execute(
    a: torch.Tensor,
    b: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    if m == 0 or n == 0:
        return out
    flat_out = out if out.ndim == 2 else out.view(m, n)
    partial_m, partial_n = m, n
    with torch_device_fn.device(a.device):
        if k == 0:
            partials, parts, apply_scales = flat_out, 0, True
        elif k > 65536:
            # Each INT32 partial stays below 2**31 even for all -128 inputs.
            parts = triton.cdiv(k, 65536)
            bm, bn, bk = _pick_int32_tiles(m, n, 65536)
            partial_m = _align_up(m, max(16, bm))
            partial_n = _align_up(n, max(32, bn))
            cores = min(
                _cube_core_count(),
                triton.cdiv(partial_m, bm) * triton.cdiv(partial_n, bn),
            )
            partials = torch.empty(
                (parts, partial_m, partial_n), device=a.device, dtype=torch.int32
            )
            for part in range(parts):
                start = part * 65536
                stop = min(start + 65536, k)
                a_part, b_part, _, _, _, _, padded_k = _pad_fixpipe_inputs(
                    a[:, start:stop],
                    b[start:stop],
                    sa,
                    sb,
                    m,
                    n,
                    stop - start,
                    bm,
                    bn,
                    bk,
                )
                mm_w8a8_int8_int32_kernel[(cores,)](
                    a_part,
                    b_part,
                    partials[part],
                    partial_m,
                    partial_n,
                    padded_k,
                    cores,
                    BLOCK_M=bm,
                    BLOCK_N=bn,
                    BLOCK_K=bk,
                    M_MAJOR=_FIXPIPE_M_MAJOR,
                    GROUP_M=0,
                    num_warps=1,
                    num_stages=2,
                    optimize_dynamic_offset=True,
                    unit_flag=False,
                    limit_auto_multi_buffer_of_local_buffer="no-l0c",
                )
            apply_scales = True
        elif bias is None:
            _launch(a, b, sa, sb, flat_out, m, n, k)
            return out
        else:
            # Keep FP32 through bias addition; rounding before bias changes results.
            partials = torch.empty((m, n), device=a.device, dtype=torch.float32)
            _launch(a, b, sa, sb, partials, m, n, k)
            parts, apply_scales = 1, False
        if k > 65536 and bias is not None:
            # Materialize scaled FP32 before bias, as in the short-K path.
            scaled = torch.empty((m, n), device=a.device, dtype=torch.float32)
            finish_mm_kernel[(triton.cdiv(m * n, 512),)](
                partials,
                sa,
                sb,
                None,
                scaled,
                m,
                n,
                partial_m,
                partial_n,
                n,
                1,
                parts,
                True,
                False,
                512,
                enable_fp_fusion=False,
            )
            partials, parts, apply_scales = scaled, 1, False
            partial_m, partial_n = m, n
        finish_mm_kernel[(triton.cdiv(m * n, 512),)](
            partials,
            sa,
            sb,
            bias,
            flat_out,
            m,
            n,
            partial_m,
            partial_n,
            *flat_out.stride(),
            parts,
            apply_scales,
            bias is not None,
            512,
            enable_fp_fusion=False,
        )
    return out


def mm_w8a8_int8(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute INT8 A[...,K] @ B[K,N], FP32 scales and optional output bias."""
    logger.debug("GEMS_ASCEND MM_W8A8_INT8")
    a2d, sa, sb, bias, shape = scaled_mm_arguments(
        a, b, scale_a, scale_b, out_dtype, bias
    )
    out = torch.empty(shape, device=a.device, dtype=out_dtype)
    return scaled_mm_execute(a2d, b, sa, sb, bias, out)


def mm_w8a8_int8_out(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    *,
    out: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Write prequantized INT8 matmul into caller-owned output without aliasing."""
    logger.debug("GEMS_ASCEND MM_W8A8_INT8_OUT")
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a tensor")
    a2d, sa, sb, normalized_bias, shape = scaled_mm_arguments(
        a, b, scale_a, scale_b, out.dtype, bias
    )
    if out.shape != shape or out.device != a.device:
        raise ValueError("out must have the result shape on the input device")
    if out.ndim != 2 and not out.is_contiguous():
        raise ValueError("batched output must be contiguous")
    if out.ndim == 2 and out.numel():
        rows, columns = out.shape
        stride_m, stride_n = out.stride()
        divisor = math.gcd(stride_m, stride_n)
        if (
            (stride_m == 0 and rows > 1)
            or (stride_n == 0 and columns > 1)
            or (
                divisor and rows > stride_n // divisor and columns > stride_m // divisor
            )
        ):
            raise ValueError("out elements must not overlap")
    if out.numel() and any(
        tensor is not None
        and tensor.numel()
        and out.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
        for tensor in (a, b, scale_a, scale_b, bias)
    ):
        raise ValueError("out must not alias inputs, scales or bias")
    return scaled_mm_execute(a2d, b, sa, sb, normalized_bias, out)
