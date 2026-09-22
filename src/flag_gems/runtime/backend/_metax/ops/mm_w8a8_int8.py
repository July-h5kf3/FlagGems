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

"""Floating-input W8A8 INT8 matrix multiplication on MetaX.

A[M,K] and B[K,N] are quantized symmetrically per A row and B column.
The public interfaces refresh quantization on every call, including Graph
replay. Internal prequantized helpers allow benchmarks to time GEMM, scaling
and conversion separately from input preparation. The GEMM uses MetaX MMA
and TLE asynchronous load annotations.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl
from triton.experimental.tle import language as tle_async

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_SUPPORTED_FLOAT = {torch.bfloat16, torch.float16, torch.float32}


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
def _mm_w8a8_kernel(
    A,
    B,
    SA,
    SB,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    USE_TLE: tl.constexpr,
    SPLIT_K: tl.constexpr = 1,
):
    pm, pn = _grouped_pids(tl.program_id(0), M, N, BM, BN, 8)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    CHUNK: tl.constexpr = triton.cdiv(triton.cdiv(K, BK), SPLIT_K) * BK
    if SPLIT_K == 1:
        start = 0
    else:
        start = tl.program_id(1) * CHUNK
    tl.static_assert(SPLIT_K == 1 or CHUNK <= 131071, "split must fit INT32")
    # Quantized operands are contiguous along K; modulo only pads M/N tiles.
    am = rm % M
    bn = rn % N
    if M * K > 2147483647:
        am = am.to(tl.int64)
    if N * K > 2147483647:
        bn = bn.to(tl.int64)
    pa = A + am[:, None] * K + start + rk[None, :]
    pb = B + bn[None, :] * K + start + rk[:, None]
    acc = tl.full((BM, BN), 0, tl.int32)
    if CHUNK > 131071:
        wide = tl.full((BM, BN), 0, tl.int64)
    for i in range(tl.cdiv(CHUNK, BK)):
        if USE_TLE:
            a = tle_async.load(
                pa, start + i * BK + rk[None, :] < K, other=0, is_async=True
            )
            b = tle_async.load(
                pb, start + i * BK + rk[:, None] < K, other=0, is_async=True
            )
        else:
            a = tl.load(pa, start + i * BK + rk[None, :] < K, other=0)
            b = tl.load(pb, start + i * BK + rk[:, None] < K, other=0)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        if CHUNK > 131071:
            # Flush well before signed INT32 overflow, including -128 * -128.
            if (i + 1) % (65536 // BK) == 0:
                wide += acc.to(tl.int64)
                acc = tl.full((BM, BN), 0, tl.int32)
        pa += BK
        pb += BK
    if CHUNK > 131071:
        value = (wide + acc.to(tl.int64)).to(tl.float32)
    else:
        value = acc.to(tl.float32)
    if SPLIT_K > 1:
        tl.store(
            C
            + tl.program_id(1).to(tl.int64) * M * N
            + rm[:, None].to(tl.int64) * N
            + rn[None, :],
            acc,
            (rm[:, None] < M) & (rn[None, :] < N),
        )
    else:
        sa = tl.load(SA + rm, rm < M, other=0)
        sb = tl.load(SB + rn, rn < N, other=0)
        value = value * sa[:, None] * sb[None, :]
        tl.store(
            C + rm[:, None].to(tl.int64) * N + rn[None, :],
            value,
            (rm[:, None] < M) & (rn[None, :] < N),
        )


@libentry()
@triton.jit
def _reduce_split_kernel(
    P,
    SA,
    SB,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    x = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    sk = tl.arange(0, triton.next_power_of_2(SPLITS))
    v = tl.load(
        P + sk[:, None].to(tl.int64) * M * N + x[None, :],
        (sk[:, None] < SPLITS) & (x[None, :] < M * N),
        other=0,
    )
    if K > 131071:
        v = v.to(tl.int64)
    total = tl.sum(v, 0).to(tl.float32)
    sa = tl.load(SA + x // N, x < M * N, other=0)
    sb = tl.load(SB + x % N, x < M * N, other=0)
    tl.store(OUT + x, total * sa * sb, x < M * N)


def _pick_split(m, n, k):
    if 64 <= m <= 128 and k >= 1024 and (1024 <= n < 2048 or n > 8192 or k > 4096):
        return 8 if n <= 8192 else 1
    if 128 < m <= 256 and n >= 1024 and k >= 1024:
        return 1
    if m >= 1024 and n >= 1024 and k >= 65536:
        return max(4, triton.next_power_of_2(triton.cdiv(k, 65536)))
    if m <= 256 and n <= 8192 and k >= 1024 and min(m, n) > 1:
        bm, bn, *_ = _pick_tiles(m, n, k)
        blocks = triton.cdiv(m, bm) * triton.cdiv(n, bn)
        if blocks < 104:
            return min(
                16, triton.next_power_of_2(triton.cdiv(208, blocks)), max(1, k // 256)
            )
    return 1


def _pick_tiles(m, n, k):
    if 1024 <= k <= 4096 and ((m >= 4096 and n >= 1024) or (m == 256 and n >= 16384)):
        return (256, 128, 64, 8, 2, "basic", True)
    if m <= 16:
        return (16, 64, 128 if k >= 1024 else 64, 4, 2, "basic", True)
    if m >= 1024 and n >= 1024:
        if k >= 65536:
            return (128, 128, 256, 8, 1, "basic", True)
        return (128, 128, 128, 8, 2, "basic", True)
    if (
        ((64 <= m <= 128 and (1024 <= n < 2048 or n > 8192 or k > 4096)) or m >= 1024)
        and n >= 512
        and k >= 1024
    ):
        return (128, 128, 128, 8, 2, "basic", True)
    if 128 < m <= 256 and n >= 2048 and k >= 1024:
        return (128, 128, 128, 8, 2, "basic", True)
    if m >= 128 and k <= 512:
        return (32, 64, 64, 4, 2, "basic", True)
    return (32 if m < 64 else 64, 64, 128 if k >= 1024 else 64, 4, 2, "basic", True)


@libentry()
@triton.jit
def _mm_w8a8_vector_kernel(
    A,
    B,
    SCALE_A,
    SCALE_B,
    OUT,
    LENGTH: tl.constexpr,
    K: tl.constexpr,
    M_ONE: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # A GEMV has no second matrix dimension to amortize padded MMA work.
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    k = tl.arange(0, BLOCK_K)
    if M_ONE:
        a = tl.load(A + k[None, :], k[None, :] < K, other=0).to(tl.int32)
        b = tl.load(
            B + r[:, None] * K + k[None, :],
            (r[:, None] < LENGTH) & (k[None, :] < K),
            other=0,
        ).to(tl.int32)
        sa = tl.load(SCALE_A)
        sb = tl.load(SCALE_B + r, r < LENGTH, other=0)
    else:
        a = tl.load(
            A + r[:, None] * K + k[None, :],
            (r[:, None] < LENGTH) & (k[None, :] < K),
            other=0,
        ).to(tl.int32)
        b = tl.load(B + k[None, :], k[None, :] < K, other=0).to(tl.int32)
        sa = tl.load(SCALE_A + r, r < LENGTH, other=0)
        sb = tl.load(SCALE_B)
    value = tl.sum(a * b, 1).to(tl.float32) * sa * sb
    tl.store(OUT + r, value, r < LENGTH)


@libentry()
@triton.jit
def _mm_w8a8_small_rows_kernel(
    A,
    B,
    SCALE_A,
    SCALE_B,
    OUT,
    LENGTH: tl.constexpr,
    K: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(1)
    # Small-M reductions avoid a separate split-K reduction launch.
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    k = tl.arange(0, BLOCK_K)
    a = tl.load(A + row * K + k[None, :], k[None, :] < K, other=0).to(tl.int32)
    b = tl.load(
        B + r[:, None] * K + k[None, :],
        (r[:, None] < LENGTH) & (k[None, :] < K),
        other=0,
    ).to(tl.int32)
    sa = tl.load(SCALE_A + row)
    sb = tl.load(SCALE_B + r, r < LENGTH, other=0)
    value = tl.sum(a * b, 1).to(tl.float32) * sa * sb
    tl.store(OUT + row * LENGTH + r, value, r < LENGTH)


@libentry()
@triton.jit
def _vector_split_kernel(
    A,
    B,
    P,
    LENGTH: tl.constexpr,
    K: tl.constexpr,
    M_ONE: tl.constexpr,
    BR: tl.constexpr,
    BK: tl.constexpr,
):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    k = tl.program_id(1) * BK + tl.arange(0, BK)
    if M_ONE:
        a = tl.load(A + k[None, :], k[None, :] < K, other=0).to(tl.int32)
        b = tl.load(
            B + r[:, None].to(tl.int64) * K + k[None, :],
            (r[:, None] < LENGTH) & (k[None, :] < K),
            other=0,
        ).to(tl.int32)
    else:
        a = tl.load(
            A + r[:, None].to(tl.int64) * K + k[None, :],
            (r[:, None] < LENGTH) & (k[None, :] < K),
            other=0,
        ).to(tl.int32)
        b = tl.load(B + k[None, :], k[None, :] < K, other=0).to(tl.int32)
    total = tl.sum(a * b, 1)
    tl.store(P + tl.program_id(1).to(tl.int64) * LENGTH + r, total, r < LENGTH)


@libentry()
@triton.jit
def _zero_output_kernel(OUT, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    tl.store(OUT + offsets, 0, offsets < SIZE)


def _validate_mm_inputs(a, b, scale_a, scale_b):
    if not all(isinstance(x, torch.Tensor) for x in (a, b, scale_a, scale_b)):
        raise TypeError("mm_w8a8_int8 expects Tensor inputs and scales")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected A[M,K] and B[K,N]")
    if a.dtype != torch.int8 or b.dtype != torch.int8:
        raise TypeError("A and B must be prequantized torch.int8 tensors")
    if a.device.type != "cuda" or any(
        x.device != a.device for x in (b, scale_a, scale_b)
    ):
        raise ValueError("inputs and scales must be on the same MetaX device")
    m, k = a.shape
    n = b.shape[1]
    if scale_a.shape not in ((m,), (m, 1)) or scale_b.shape not in ((n,), (1, n)):
        raise ValueError("expected scale_a[M] or [M,1], scale_b[N] or [1,N]")
    if any(
        x.dtype != torch.float32 or not x.is_contiguous() for x in (scale_a, scale_b)
    ):
        raise ValueError("scales must be contiguous FP32 tensors")
    return m, n, k


def _launch(a_q, a_scale, b_q, b_scale, out, m, n, k):
    bm, bn, bk, warps, stages, pipeline, use_tle = _pick_tiles(m, n, k)
    splits = _pick_split(m, n, k)
    if splits > 1:
        # Every integer partial must stay below the signed INT32 limit.
        splits = max(splits, triton.next_power_of_2(triton.cdiv(k, 65536)))
    target = (
        out
        if splits == 1
        else torch.empty((splits, m, n), device=out.device, dtype=torch.int32)
    )
    with torch_device_fn.device(a_q.device):
        _mm_w8a8_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn), splits)](
            a_q,
            b_q,
            a_scale,
            b_scale,
            target,
            m,
            n,
            k,
            bm,
            bn,
            bk,
            use_tle,
            splits,
            num_warps=warps,
            num_stages=stages,
            pipeline=pipeline,
            enable_fp_fusion=False,
        )
        if splits > 1:
            reduce_block = 512 if m <= 16 and n >= 2048 else 256
            reduce_warps = (2 if n >= 2048 else 1) if m <= 16 else 4
            if splits > 32:
                reduce_block = min(256, max(1, 8192 // triton.next_power_of_2(splits)))
                reduce_warps = 4
            _reduce_split_kernel[(triton.cdiv(m * n, reduce_block),)](
                target,
                a_scale,
                b_scale,
                out,
                m,
                n,
                k,
                splits,
                reduce_block,
                num_warps=reduce_warps,
                enable_fp_fusion=False,
            )
    return out


def _run_mm(a, b, scale_a, scale_b, out, m, n, k):
    if m == 0 or n == 0:
        return out
    if k == 0:
        with torch_device_fn.device(a.device):
            _zero_output_kernel[(triton.cdiv(m * n, 1024),)](out, m * n, BLOCK=1024)
        return out
    if (
        min(m, n) == 1
        and (k > 4096 or n > 16384)
        and a.stride() == (k, 1)
        and b.stride() == (1, k)
    ):
        bk = min(4096, triton.next_power_of_2(k))
        splits = triton.cdiv(k, bk)
        length = max(m, n)
        partial = torch.empty((splits, m, n), device=a.device, dtype=torch.int32)
        with torch_device_fn.device(a.device):
            _vector_split_kernel[(triton.cdiv(length, 4), splits)](
                a,
                b,
                partial,
                length,
                k,
                m == 1,
                4,
                bk,
                num_warps=4,
            )
            _reduce_split_kernel[(triton.cdiv(m * n, 256),)](
                partial,
                scale_a,
                scale_b,
                out,
                m,
                n,
                k,
                splits,
                256,
                num_warps=4,
                enable_fp_fusion=False,
            )
        return out
    if (
        min(m, n) == 1
        and k <= 4096
        and n <= 16384
        and a.stride() == (k, 1)
        and b.stride() == (1, k)
    ):
        length = max(m, n)
        with torch_device_fn.device(a.device):
            _mm_w8a8_vector_kernel[(triton.cdiv(length, 2),)](
                a,
                b,
                scale_a,
                scale_b,
                out,
                length,
                k,
                m == 1,
                BLOCK_R=2,
                BLOCK_K=triton.next_power_of_2(k),
                num_warps=1 if k <= 512 else 4,
            )
        return out
    if (
        m == 2
        and n <= 512
        and k <= 4096
        and a.stride() == (k, 1)
        and b.stride() == (1, k)
    ):
        with torch_device_fn.device(a.device):
            _mm_w8a8_small_rows_kernel[(triton.cdiv(n, 2), m)](
                a,
                b,
                scale_a,
                scale_b,
                out,
                n,
                k,
                BLOCK_R=2,
                BLOCK_K=triton.next_power_of_2(k),
                num_warps=1 if k <= 512 else 4,
            )
        return out
    # Normalize private prequantized inputs to K-contiguous operands.
    # Public floating-input preparation already produces these strides.
    a = a.contiguous()
    b = b.t().contiguous().t()
    return _launch(a, scale_a, b, scale_b, out, m, n, k)


def _mm_w8a8_int8_prequantized(a, b, scale_a, scale_b, *, out_dtype=torch.bfloat16):
    m, n, k = _validate_mm_inputs(a, b, scale_a, scale_b)
    if out_dtype not in _SUPPORTED_FLOAT:
        raise TypeError("out_dtype must be BF16, FP16 or FP32")
    out = torch.empty((m, n), device=a.device, dtype=out_dtype)
    return _run_mm(a, b, scale_a, scale_b, out, m, n, k)


def _mm_w8a8_int8_prequantized_out(a, b, scale_a, scale_b, *, out):
    m, n, k = _validate_mm_inputs(a, b, scale_a, scale_b)
    if out.shape != (m, n) or out.device != a.device or not out.is_contiguous():
        raise ValueError("out must be contiguous [M,N] on the input device")
    if out.dtype not in _SUPPORTED_FLOAT:
        raise TypeError("out must be BF16, FP16 or FP32")
    return _run_mm(a, b, scale_a, scale_b, out, m, n, k)


@libentry()
@triton.jit
def _quantize_mm_input_kernel(
    X,
    PEAK,
    Q,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    STRIDE_R: tl.constexpr,
    STRIDE_C: tl.constexpr,
    PER_ROW: tl.constexpr,
    COLUMN_MAJOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    if COLUMN_MAJOR:
        row, col = offsets % ROWS, offsets // ROWS
    else:
        row, col = offsets // COLS, offsets % COLS
    mask = offsets < ROWS * COLS
    value = tl.load(X + row * STRIDE_R + col * STRIDE_C, mask, other=0).to(tl.float32)
    peak = tl.load(PEAK + (row if PER_ROW else col), mask, other=1)
    # Correctly-rounded division avoids platform-dependent reciprocal error
    # changing the integer code at half-integer quantization boundaries.
    normalized = tl.div_rn(value, peak) * 127.0
    lower = tl.floor(normalized)
    frac = normalized - lower
    odd = (lower.to(tl.int32) & 1) != 0
    rounded = lower + tl.where((frac > 0.5) | ((frac == 0.5) & odd), 1.0, 0.0)
    quantized = tl.minimum(tl.maximum(rounded, -127.0), 127.0).to(tl.int8)
    tl.store(Q + offsets, quantized, mask)


@libentry()
@triton.jit
def _quantize_rows_kernel(
    X,
    Q,
    SCALE,
    R: tl.constexpr,
    K: tl.constexpr,
    SR: tl.constexpr,
    SK: tl.constexpr,
    BR: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    ks = tl.arange(0, BK)
    x = tl.load(
        X + rows[:, None].to(tl.int64) * SR + ks[None, :].to(tl.int64) * SK,
        (rows[:, None] < R) & (ks[None, :] < K),
        other=0,
    ).to(tl.float32)
    peak = tl.maximum(tl.max(tl.abs(x), 1), 1.0e-10)
    value = tl.div_rn(x, peak[:, None]) * 127.0
    lower = tl.floor(value)
    frac = value - lower
    odd = (lower.to(tl.int32) & 1) != 0
    rounded = lower + tl.where((frac > 0.5) | ((frac == 0.5) & odd), 1.0, 0.0)
    code = tl.minimum(tl.maximum(rounded, -127.0), 127.0).to(tl.int8)
    tl.store(
        Q + rows[:, None].to(tl.int64) * K + ks[None, :],
        code,
        (rows[:, None] < R) & (ks[None, :] < K),
    )
    tl.store(SCALE + rows, peak * (1.0 / 127.0), rows < R)


def _prepare_mm_w8a8_int8_inputs(a, b):
    """Quantize current inputs directly into row-major A and column-major B."""
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("mm_w8a8_int8 expects Tensor inputs")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected A[M,K] and B[K,N]")
    if a.dtype not in _SUPPORTED_FLOAT or b.dtype not in _SUPPORTED_FLOAT:
        raise TypeError("A and B must be FP16, BF16 or FP32 tensors")
    if a.device.type != "cuda" or a.device != b.device:
        raise ValueError("A and B must be on the same MetaX device")
    m, k = a.shape
    n = b.shape[1]
    # Empty outputs and empty reductions must not call amax on an empty axis.
    if m == 0 or n == 0 or k == 0:
        return (
            torch.empty((m, k), device=a.device, dtype=torch.int8),
            torch.empty((k, n), device=a.device, dtype=torch.int8),
            torch.ones(m, device=a.device, dtype=torch.float32),
            torch.ones(n, device=a.device, dtype=torch.float32),
        )
    scale_a = torch.empty(m, device=a.device, dtype=torch.float32)
    scale_b = torch.empty(n, device=a.device, dtype=torch.float32)
    a_q = torch.empty((m, k), device=a.device, dtype=torch.int8)
    b_q = torch.empty_strided((k, n), (1, k), device=a.device, dtype=torch.int8)
    with torch_device_fn.device(a.device):
        for x, q, scale in ((a, a_q, scale_a), (b.t(), b_q.t(), scale_b)):
            if k >= 1024 and x.stride(1) != 1:
                peak = x.float().abs().amax(dim=1).clamp_min(1e-10)
                torch.mul(peak, 1.0 / 127.0, out=scale)
                physical_q = torch.empty_strided(
                    x.shape, (1, x.shape[0]), device=x.device, dtype=torch.int8
                )
                _quantize_mm_input_kernel[(triton.cdiv(x.numel(), 1024),)](
                    x,
                    peak,
                    physical_q,
                    *x.shape,
                    *x.stride(),
                    True,
                    True,
                    BLOCK=1024,
                    enable_fp_fusion=False,
                )
                q.copy_(physical_q)
            elif k <= 16384:
                br = 1 if x.stride(1) == 1 else 16
                bk = triton.next_power_of_2(k)
                br = min(br, max(1, 16384 // bk))
                _quantize_rows_kernel[(triton.cdiv(x.shape[0], br),)](
                    x,
                    q,
                    scale,
                    x.shape[0],
                    k,
                    *x.stride(),
                    br,
                    bk,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            else:
                # Bound per-program reduction storage for very large K.
                peak = x.float().abs().amax(dim=1).clamp_min(1e-10)
                torch.mul(peak, 1.0 / 127.0, out=scale)
                _quantize_mm_input_kernel[(triton.cdiv(x.numel(), 1024),)](
                    x,
                    peak,
                    q,
                    *x.shape,
                    *x.stride(),
                    True,
                    False,
                    BLOCK=1024,
                    enable_fp_fusion=False,
                )
    return a_q, b_q, scale_a, scale_b


def mm_w8a8_int8(a, b, *, out_dtype=None):
    """Compute W8A8 GEMM from floating inputs using symmetric INT8 quantization.

    The call signature follows mm_w8a8_fp8; the quantization format is INT8.
    Default output is BF16. Quantization is recomputed, never cached by pointer.
    """
    logger.debug("GEMS MM_W8A8_INT8")
    out_dtype = torch.bfloat16 if out_dtype is None else out_dtype
    if out_dtype not in _SUPPORTED_FLOAT:
        raise TypeError("out_dtype must be BF16, FP16 or FP32")
    return _mm_w8a8_int8_prequantized(
        *_prepare_mm_w8a8_int8_inputs(a, b), out_dtype=out_dtype
    )


def mm_w8a8_int8_out(a, b, *, out):
    """Write floating-input INT8 GEMM into a contiguous caller-owned output."""
    logger.debug("GEMS MM_W8A8_INT8_OUT")
    return _mm_w8a8_int8_prequantized_out(*_prepare_mm_w8a8_int8_inputs(a, b), out=out)


@triton.jit
def _activation_code(x, peak, LOW_PRECISION: tl.constexpr):
    if LOW_PRECISION:
        fast = (peak > 1.0e-10) & (peak < 1.0e30)
    else:
        fast = False
    if fast:
        normalized = x * tl.div_rn(127.0, peak)
        # In this range, adding 1.5 * 2**23 rounds FP32 to an even integer.
        rounded = (normalized + 12582912.0) - 12582912.0
        # BF16/FP16 products below are exact in FP32. Correct reciprocal
        # error at exact half-integer boundaries without per-element div.
        distance = x * 254.0 - peak * (rounded * 2.0)
        odd = (rounded.to(tl.int32) & 1) != 0
        rounded = tl.where(
            odd & (tl.abs(distance) == peak),
            rounded + tl.where(distance > 0, 1.0, -1.0),
            rounded,
        )
    else:
        # Preserve the original division and rounding for FP32 and extreme peaks.
        normalized = tl.div_rn(x, peak) * 127.0
        lower = tl.floor(normalized)
        frac = normalized - lower
        odd = (lower.to(tl.int32) & 1) != 0
        rounded = lower + tl.where((frac > 0.5) | ((frac == 0.5) & odd), 1.0, 0.0)
        rounded = tl.minimum(tl.maximum(rounded, -127.0), 127.0)

    return rounded.to(tl.int8)


@libentry()
@triton.jit
def _activation_rows_kernel(X, Q, SCALE, K: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0)
    ks = tl.arange(0, BK)
    x = tl.load(X + row.to(tl.int64) * K + ks, ks < K, other=0).to(tl.float32)
    peak = tl.maximum(tl.max(tl.abs(x), 0), 1.0e-10)
    code = _activation_code(
        x, peak, X.dtype.element_ty == tl.bfloat16 or X.dtype.element_ty == tl.float16
    )
    tl.store(Q + row.to(tl.int64) * K + ks, code, ks < K)
    tl.store(SCALE + row, peak * (1.0 / 127.0))


@libentry()
@triton.jit
def _activation_peaks_kernel(
    X, PEAKS, K: tl.constexpr, BK: tl.constexpr, SPLITS: tl.constexpr
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    ks = chunk * BK + tl.arange(0, BK)
    x = tl.load(X + row.to(tl.int64) * K + ks, ks < K, other=0).to(tl.float32)
    tl.store(PEAKS + row.to(tl.int64) * SPLITS + chunk, tl.max(tl.abs(x), 0))


@libentry()
@triton.jit
def _activation_chunks_kernel(
    X, PEAKS, Q, SCALE, K: tl.constexpr, BK: tl.constexpr, SPLITS: tl.constexpr
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    sp = tl.arange(0, triton.next_power_of_2(SPLITS))
    peak = tl.maximum(
        tl.max(
            tl.load(PEAKS + row.to(tl.int64) * SPLITS + sp, sp < SPLITS, other=0), 0
        ),
        1.0e-10,
    )
    ks = chunk * BK + tl.arange(0, BK)
    x = tl.load(X + row.to(tl.int64) * K + ks, ks < K, other=0).to(tl.float32)
    code = _activation_code(
        x, peak, X.dtype.element_ty == tl.bfloat16 or X.dtype.element_ty == tl.float16
    )
    tl.store(Q + row.to(tl.int64) * K + ks, code, ks < K)
    if chunk == 0:
        tl.store(SCALE + row, peak * (1.0 / 127.0))


def _prepare_mm_w8a8_int8_activation(a):
    """Quantize current activations when INT8 weights are already available."""
    if a.ndim != 2 or a.dtype not in _SUPPORTED_FLOAT or a.device.type != "cuda":
        raise ValueError("expected a floating MetaX activation matrix")
    a = a.contiguous()
    m, k = a.shape
    q = torch.empty_like(a, dtype=torch.int8)
    scale = torch.empty(m, device=a.device, dtype=torch.float32)
    if m == 0 or k == 0:
        scale.fill_(1)
        return q, scale
    with torch_device_fn.device(a.device):
        if k <= 16384:
            _activation_rows_kernel[(m,)](
                a,
                q,
                scale,
                k,
                triton.next_power_of_2(k),
                num_warps=4,
                enable_fp_fusion=False,
            )
        elif k <= 1048576:
            bk = 4096
            splits = triton.cdiv(k, bk)
            peaks = torch.empty((m, splits), device=a.device, dtype=torch.float32)
            _activation_peaks_kernel[(m, splits)](
                a,
                peaks,
                k,
                bk,
                splits,
                num_warps=4,
            )
            _activation_chunks_kernel[(m, splits)](
                a,
                peaks,
                q,
                scale,
                k,
                bk,
                splits,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            # Bound the partial-peak reduction size for unusually long rows.
            peak = a.float().abs().amax(dim=1).clamp_min(1e-10)
            torch.mul(peak, 1.0 / 127.0, out=scale)
            _quantize_mm_input_kernel[(triton.cdiv(a.numel(), 1024),)](
                a,
                peak,
                q,
                m,
                k,
                k,
                1,
                True,
                False,
                BLOCK=1024,
                enable_fp_fusion=False,
            )
    return q, scale


@libentry()
@triton.jit
def _activation_gemv_kernel(
    A, B, SB, OUT, N: tl.constexpr, K: tl.constexpr, BK: tl.constexpr
):
    cols = tl.program_id(0) * 4 + tl.arange(0, 4)
    ks = tl.arange(0, BK)
    a = tl.load(A + ks, ks < K, other=0).to(tl.float32)
    peak = tl.maximum(tl.max(tl.abs(a), 0), 1.0e-10)
    aq = _activation_code(a, peak, True).to(tl.int32)
    b = tl.load(
        B + cols[:, None] * K + ks[None, :],
        (cols[:, None] < N) & (ks[None, :] < K),
        other=0,
    ).to(tl.int32)
    value = tl.sum(aq[None, :] * b, 1).to(tl.float32)
    sb = tl.load(SB + cols, cols < N, other=0)
    tl.store(OUT + cols, value * (peak * (1.0 / 127.0)) * sb, cols < N)


def _mm_w8a8_int8_prepared_weight_out(a, bq, scale_b, *, out):
    """Dynamic activations and caller-owned, prequantized constant weights."""
    if a.ndim == 2 and bq.ndim == 2:
        m, k = a.shape
        n = bq.shape[1]
        fused = (
            m == 1
            and 0 < n <= 1024
            and 0 < k <= 4096
            and a.dtype in (torch.bfloat16, torch.float16)
            and a.stride() == (k, 1)
            and bq.stride() == (1, k)
        )
        if fused:
            if (
                bq.shape[0] != k
                or bq.dtype != torch.int8
                or a.device.type != "cuda"
                or any(x.device != a.device for x in (bq, scale_b, out))
                or scale_b.dtype != torch.float32
                or not scale_b.is_contiguous()
                or scale_b.shape not in ((n,), (1, n))
                or out.shape != (m, n)
                or out.dtype not in _SUPPORTED_FLOAT
                or not out.is_contiguous()
            ):
                raise ValueError("invalid prequantized weight, scale or output")
            with torch_device_fn.device(a.device):
                _activation_gemv_kernel[(triton.cdiv(n, 4),)](
                    a,
                    bq,
                    scale_b,
                    out,
                    n,
                    k,
                    triton.next_power_of_2(k),
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            return out
    aq, scale_a = _prepare_mm_w8a8_int8_activation(a)
    return _mm_w8a8_int8_prequantized_out(aq, bq, scale_a, scale_b, out=out)
