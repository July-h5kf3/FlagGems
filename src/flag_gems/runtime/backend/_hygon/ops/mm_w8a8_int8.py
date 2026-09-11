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

"""Hygon W8A8 GEMM with INT8 dot products and overflow-safe long-K reduction."""

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
from triton.language.extra import libdevice

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

_FLOATS = (torch.float16, torch.bfloat16, torch.float32)


@libentry()
@triton.jit
def _prequant_vector(
    A,
    B,
    SA,
    SB,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    for row in tl.static_range(M):
        acc = tl.zeros((BN, BK), tl.int32)
        for start in range(tl.cdiv(K, BK)):
            k = start * BK + rk
            a = tl.load(A + row * K + k, k < K, other=0).to(tl.int32)
            b = tl.load(
                B + cols[:, None].to(tl.int64) * K + k[None, :],
                (cols[:, None] < N) & (k[None, :] < K),
                other=0,
            ).to(tl.int32)
            acc += a[None, :] * b
        dot = tl.sum(acc, 1)
        sa = tl.load(SA + row)
        sb = tl.load(SB + cols, cols < N, other=0)
        tl.store(C + row * N + cols, dot.to(tl.float32) * sa * sb, cols < N)


@libentry()
@triton.jit
def _grouped_int8(
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
    ST: tl.constexpr,
    GROUP: tl.constexpr,
):
    pid = tl.program_id(0)
    pm = tl.cdiv(M, BM)
    pn = tl.cdiv(N, BN)
    if GROUP == 0:
        im = pid % pm
        jn = pid // pm
    else:
        group = pid // (GROUP * pn)
        first = group * GROUP
        size = tl.minimum(pm - first, GROUP)
        im = first + pid % size
        jn = (pid % (GROUP * pn)) // size
    rm = im * BM + tl.arange(0, BM)
    rn = jn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.int32)
    for start in tle.gpu.pipeline(0, tl.cdiv(K, BK), num_stages=ST):
        k = start * BK + rk
        a = tl.load(
            A + rm[:, None] * K + k[None, :],
            (rm[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        b = tl.load(
            B + rn[None, :] * K + k[:, None],
            (rn[None, :] < N) & (k[:, None] < K),
            other=0,
        )
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
    sa = tl.load(SA + rm, rm < M, other=0)
    sb = tl.load(SB + rn, rn < N, other=0)
    tl.store(
        C + rm[:, None] * N + rn[None, :],
        acc.to(tl.float32) * sa[:, None] * sb[None, :],
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@libentry()
@triton.jit
def _split_mm(
    A,
    B,
    P,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    CHUNK: tl.constexpr,
    ST: tl.constexpr,
):
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    c = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    split = tl.program_id(2)
    acc = tl.zeros((BM, BN), tl.int32)
    for start in tle.gpu.pipeline(0, tl.cdiv(CHUNK, BK), num_stages=ST):
        k = split * CHUNK + start * BK + rk
        a = tl.load(
            A + r[:, None].to(tl.int64) * K + k[None, :],
            (r[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        b = tl.load(
            B + c[None, :].to(tl.int64) * K + k[:, None],
            (c[None, :] < N) & (k[:, None] < K),
            other=0,
        )
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
    tl.store(
        P + split.to(tl.int64) * M * N + r[:, None].to(tl.int64) * N + c[None, :],
        acc,
        (r[:, None] < M) & (c[None, :] < N),
    )


@libentry()
@triton.jit
def _split_reduce(
    P,
    SA,
    SB,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.int64)
    for s in range(SPLITS):
        acc += tl.load(
            P + s.to(tl.int64) * M * N + i.to(tl.int64), i < M * N, other=0
        ).to(tl.int64)
    sa = tl.load(SA + i // N, i < M * N, other=0)
    sb = tl.load(SB + i % N, i < M * N, other=0)
    tl.store(C + i, acc.to(tl.float32) * sa * sb, i < M * N)


@libentry()
@triton.jit
def _quantize_rows(
    X, Q, S, K: tl.constexpr, SR: tl.constexpr, SK: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    x = tl.load(X + row * SR + k * SK, k < K, other=0).to(tl.float32)
    peak = tl.maximum(tl.max(tl.abs(x), 0), 1e-10)
    q = libdevice.rint(tl.div_rn(x, peak) * 127.0).to(tl.int8)
    tl.store(Q + row * K + k, q, k < K)
    tl.store(S + row, peak * (1.0 / 127.0))


@libentry()
@triton.jit
def _quantize_columns_full(B, Q, S, K: tl.constexpr, N: tl.constexpr, BN: tl.constexpr):
    k = tl.arange(0, triton.next_power_of_2(K))
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    x = tl.load(
        B + k[:, None] * N + n[None, :], (k[:, None] < K) & (n[None, :] < N), other=0
    ).to(tl.float32)
    peak = tl.maximum(tl.max(tl.abs(x), 0), 1e-10)
    q = libdevice.rint(tl.div_rn(x, peak[None, :]) * 127.0).to(tl.int8)
    tl.store(Q + n[None, :] * K + k[:, None], q, (k[:, None] < K) & (n[None, :] < N))
    tl.store(S + n, peak * (1.0 / 127.0), n < N)


@libentry()
@triton.jit
def _gemm_tle(
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
    ST: tl.constexpr,
):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.int32)
    for start in tle.gpu.pipeline(0, tl.cdiv(K, BK), num_stages=ST):
        k = start * BK + rk
        a = tle.load(
            A + rm[:, None] * K + k[None, :],
            (rm[:, None] < M) & (k[None, :] < K),
            other=0,
            is_async=False,
        )
        off = rn[None, :] * K + k[:, None]
        b = tle.load(
            B + off, (rn[None, :] < N) & (k[:, None] < K), other=0, is_async=False
        )
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
    sa = tl.load(SA + rm, rm < M, other=0)
    sb = tl.load(SB + rn, rn < N, other=0)
    tl.store(
        C + rm[:, None] * N + rn[None, :],
        acc.to(tl.float32) * sa[:, None] * sb[None, :],
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@libentry()
@triton.jit
def _gemv_quantized(
    A, SA, B, C, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr, BN: tl.constexpr
):
    k = tl.arange(0, triton.next_power_of_2(K))
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    b = tle.load(
        B + k[:, None] * N + n[None, :],
        (k[:, None] < K) & (n[None, :] < N),
        other=0,
        is_async=False,
    ).to(tl.float32)
    peak = tl.maximum(tl.max(tl.abs(b), 0), 1e-10)
    q = libdevice.rint(tl.div_rn(b, peak[None, :]) * 127.0).to(tl.int8).to(tl.int32)
    for row in tl.static_range(M):
        a = tl.load(A + row * K + k, k < K, other=0).to(tl.int32)
        sa = tl.load(SA + row)
        acc = tl.sum(a[:, None] * q, 0)
        tl.store(
            C + row * N + n, acc.to(tl.float32) * sa * (peak * (1.0 / 127.0)), n < N
        )


@libentry()
@triton.jit
def _quantize(
    X, Q, S, K: tl.constexpr, SR: tl.constexpr, SK: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLOCK)
    peak = tl.full((BLOCK,), 0, tl.float32)
    for start in range(tl.cdiv(K, BLOCK)):
        k = start * BLOCK + c
        x = tl.load(X + row * SR + k * SK, k < K, other=0).to(tl.float32)
        peak = tl.maximum(peak, tl.abs(x))
    maximum = tl.maximum(tl.max(peak, 0), 1.0e-10)
    tl.store(S + row, maximum * (1.0 / 127.0))
    for start in range(tl.cdiv(K, BLOCK)):
        k = start * BLOCK + c
        x = tl.load(X + row * SR + k * SK, k < K, other=0).to(tl.float32)
        normalized = tl.div_rn(x, maximum) * 127.0
        lower = tl.floor(normalized)
        fraction = normalized - lower
        odd = (lower.to(tl.int32) & 1) != 0
        rounded = lower + tl.where(
            (fraction > 0.5) | ((fraction == 0.5) & odd), 1.0, 0.0
        )
        q = tl.minimum(tl.maximum(rounded, -127.0), 127.0).to(tl.int8)
        tl.store(Q + row * K + k, q, k < K)


@libentry()
@triton.jit
def _quantize_columns(
    X,
    Q,
    S,
    K: tl.constexpr,
    N: tl.constexpr,
    SK: tl.constexpr,
    SN: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
):
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    peak = tl.full((BK, BN), 0.0, tl.float32)
    # Coalesce row-major B loads across columns rather than striding by N.
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + rk
        x = tl.load(
            X + k[:, None].to(tl.int64) * SK + cols[None, :] * SN,
            (k[:, None] < K) & (cols[None, :] < N),
            other=0,
        ).to(tl.float32)
        peak = tl.maximum(peak, tl.abs(x))
    maximum = tl.maximum(tl.max(peak, 0), 1e-10)
    tl.store(S + cols, maximum * (1.0 / 127.0), cols < N)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + rk
        mask = (k[:, None] < K) & (cols[None, :] < N)
        x = tl.load(
            X + k[:, None].to(tl.int64) * SK + cols[None, :] * SN, mask, other=0
        ).to(tl.float32)
        normalized = tl.div_rn(x, maximum[None, :]) * 127.0
        lower = tl.floor(normalized)
        fraction = normalized - lower
        odd = (lower.to(tl.int32) & 1) != 0
        rounded = lower + tl.where(
            (fraction > 0.5) | ((fraction == 0.5) & odd), 1.0, 0.0
        )
        q = tl.minimum(tl.maximum(rounded, -127.0), 127.0).to(tl.int8)
        tl.store(Q + cols[None, :].to(tl.int64) * K + k[:, None], q, mask)


@libentry()
@triton.jit
def _gemm(
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
):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.int32)
    for start in range(tl.cdiv(K, BK)):
        k = start * BK + rk
        a = tl.load(
            A + rm[:, None].to(tl.int64) * K + k[None, :],
            (rm[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        b = tl.load(
            B + rn[None, :].to(tl.int64) * K + k[:, None],
            (rn[None, :] < N) & (k[:, None] < K),
            other=0,
        )
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
    sa = tl.load(SA + rm, rm < M, other=0)
    sb = tl.load(SB + rn, rn < N, other=0)
    out = acc.to(tl.float32) * sa[:, None] * sb[None, :]
    tl.store(
        C + rm[:, None].to(tl.int64) * N + rn[None, :],
        out,
        (rm[:, None] < M) & (rn[None, :] < N),
    )


@libentry()
@triton.jit
def _zero(C, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(C + i, 0.0, i < SIZE)


def _validate(a, b):
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("A and B must be tensors")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected A[M,K] and B[K,N]")
    if a.dtype not in _FLOATS or b.dtype not in _FLOATS:
        raise TypeError("A and B must be FP16, BF16 or FP32")
    if a.device.type != "cuda" or a.device != b.device:
        raise ValueError("A and B must be on the same Hygon device")
    return a.shape[0], b.shape[1], a.shape[1]


def _run(a, b, out, m, n, k):
    if m == 0 or n == 0:
        return out
    with torch_device_fn.device(a.device):
        if k == 0:
            _zero[(triton.cdiv(m * n, 1024),)](out, m * n, 1024)
            return out
        aq = torch.empty((m, k), device=a.device, dtype=torch.int8)
        sa = torch.empty((m,), device=a.device, dtype=torch.float32)
        # Keep signed 32-bit address arithmetic only when all offsets fit.
        fast_a = (
            k <= 4096
            and m * k < 2**31
            and (m - 1) * a.stride(0) + (k - 1) * a.stride(1) < 2**31
        )
        if fast_a:
            _quantize_rows[(m,)](
                a,
                aq,
                sa,
                k,
                *a.stride(),
                triton.next_power_of_2(k),
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            _quantize[(m,)](
                a, aq, sa, k, *a.stride(), 1024, num_warps=4, enable_fp_fusion=False
            )
        # Fusing B quantization avoids an INT8 temporary for GEMV. A shifted
        # output alias into B could race another column tile, so retain GEMM
        # for shared B storage. The check only reads host-side metadata.
        if (
            m == 1
            and 128 <= k <= 4096
            and n >= 128
            and b.is_contiguous()
            and n * k < 2**31
            and out.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()
        ):
            _gemv_quantized[(triton.cdiv(n, 8),)](
                aq, sa, b, out, m, k, n, 8, num_warps=8, enable_fp_fusion=False
            )
            return out
        bq = torch.empty((n, k), device=a.device, dtype=torch.int8)
        sb = torch.empty((n,), device=a.device, dtype=torch.float32)
        if b.is_contiguous() and k <= 4096 and n * k < 2**31 and n >= 16:
            bn = 8 if k <= 512 or k > 2048 else 16
            nw = 8 if 512 < k <= 1024 or k > 2048 else 4
            _quantize_columns_full[(triton.cdiv(n, bn),)](
                b, bq, sb, k, n, bn, num_warps=nw, enable_fp_fusion=False
            )
        elif b.stride(1) == 1 and n >= 16:
            _quantize_columns[(triton.cdiv(n, 16),)](
                b,
                bq,
                sb,
                k,
                n,
                *b.stride(),
                256,
                16,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            _quantize[(n,)](
                b,
                bq,
                sb,
                k,
                b.stride(1),
                b.stride(0),
                1024,
                num_warps=4,
                enable_fp_fusion=False,
            )
        _launch_prequantized(aq, bq, sa, sb, out, m, n, k)
    return out


def mm_w8a8_int8(a, b, *, out_dtype=None):
    """Quantize floating A rows/B columns, multiply INT8 and dequantize.

    Scales are max(abs(x), 1e-10)/127; codes use round-to-nearest-even of
    (x/peak)*127, clamped to [-127,127]. Inputs must be finite FP16/BF16/FP32
    matrices on one Hygon device; arbitrary input strides are supported.
    Output defaults to BF16. Quantization is recomputed on each invocation
    and Graph replay. Long K uses bounded INT32 partial products and an
    INT64 final reduction to prevent overflow. Forward inference only;
    autograd is not implemented.
    """
    m, n, k = _validate(a, b)
    dtype = torch.bfloat16 if out_dtype is None else out_dtype
    if dtype not in _FLOATS:
        raise TypeError("out_dtype must be FP16, BF16 or FP32")
    out = torch.empty((m, n), device=a.device, dtype=dtype)
    return _run(a, b, out, m, n, k)


def mm_w8a8_int8_out(a, b, *, out):
    """Write W8A8 GEMM to contiguous caller-owned FP16/BF16/FP32 output."""
    m, n, k = _validate(a, b)
    if not isinstance(out, torch.Tensor) or out.dtype not in _FLOATS:
        raise TypeError("out must be an FP16, BF16 or FP32 tensor")
    if out.shape != (m, n) or out.device != a.device or not out.is_contiguous():
        raise ValueError("out must have shape [M,N], same device and be contiguous")
    # Quantization finishes before GEMM writes out, so aliasing inputs is safe.
    return _run(a, b, out, m, n, k)


def _launch_prequantized(aq, bq, sa, sb, out, m, n, k):
    if m == 0 or n == 0:
        return out
    if k == 0:
        _zero[(triton.cdiv(m * n, 1024),)](out, m * n, 1024)
        return out
    if 3 <= m <= 8 and 512 <= n <= 1024 and 1024 <= k <= 4096:
        _grouped_int8[(triton.cdiv(n, 16),)](
            aq,
            bq,
            sa,
            sb,
            out,
            m,
            n,
            k,
            16,
            16,
            256,
            2,
            0,
            num_warps=2,
            num_stages=2,
        )
    elif m <= 8 and n <= 1024 and 1024 <= k <= 4096:
        bn = 1 if m >= 3 else 2
        _prequant_vector[(triton.cdiv(n, bn),)](
            aq, bq, sa, sb, out, m, n, k, bn, 4096, num_warps=4
        )
    elif k > 131071 or (k >= 65536 and m >= 1024 and n >= 1024):
        # Even -128 * -128 cannot overflow a chunk's INT32 accumulator.
        # INT64 reduction preserves the integer result across long K. Splitting
        # also exposes more parallel work for large tiles with few output blocks.
        chunk = 32768
        parts = triton.cdiv(k, chunk)
        partial = torch.empty((parts, m, n), device=aq.device, dtype=torch.int32)
        bm, bn, nw = (256, 256, 8) if m >= 1024 and n >= 1024 else (64, 128, 4)
        _split_mm[(triton.cdiv(m, bm), triton.cdiv(n, bn), parts)](
            aq, bq, partial, m, n, k, bm, bn, 128, chunk, 2, num_warps=nw, num_stages=2
        )
        _split_reduce[(triton.cdiv(m * n, 256),)](
            partial, sa, sb, out, m, n, parts, 256, num_warps=4
        )
    elif max(m * k, n * k, m * n) >= 2**31:
        _gemm[(triton.cdiv(m, 32), triton.cdiv(n, 64))](
            aq, bq, sa, sb, out, m, n, k, 32, 64, 64, num_warps=4, num_stages=1
        )
    elif m >= 65 and n >= 1024 and k >= 1024 and (m >= 256 or n >= 2048):
        if m >= 8192:
            if n >= 8192:
                bm, bn, bk, nw, st, group = 256, 512, 64, 16, 2, 8
            else:
                bm, bn, bk, nw, st, group = 256, 256, 128, 8, 2, 8
        elif m >= 4096:
            bm, bn, bk, nw, st, group = 128, 128, 128, 8, 2, 8
        elif m == 256:
            bm, bn, bk, nw, st, group = 128, 128, 128, 4, 2, 0
        elif m < 256:
            if k > 4096:
                bm, bn, bk, nw, st, group = 64, 64, 128, 4, 2, 0
            else:
                bm, bn, bk, nw, st, group = 128, 128, 256, 4, 1, 0
        else:
            bm, bn, bk, nw, st, group = 64, 128, 128, 4, 1 if m >= 2048 else 2, 0
        _grouped_int8[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
            aq,
            bq,
            sa,
            sb,
            out,
            m,
            n,
            k,
            bm,
            bn,
            bk,
            st,
            group,
            num_warps=nw,
            num_stages=st,
        )
    else:
        bm, bn = (64, 128) if m >= 1024 else (32, 64)
        stages = 1 if m >= 2048 else 2
        _gemm_tle[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
            aq,
            bq,
            sa,
            sb,
            out,
            m,
            n,
            k,
            bm,
            bn,
            128,
            stages,
            num_warps=4,
            num_stages=stages,
        )
    return out


def _prepare_mm_w8a8_int8_inputs(a, b):
    """Prepare current input data; callers explicitly choose timing scope."""
    m, n, k = _validate(a, b)
    aq = torch.empty((m, k), device=a.device, dtype=torch.int8)
    bq = torch.empty_strided((k, n), (1, k), device=a.device, dtype=torch.int8)
    sa = torch.empty(m, device=a.device, dtype=torch.float32)
    sb = torch.empty(n, device=a.device, dtype=torch.float32)
    if m and n and k:
        with torch_device_fn.device(a.device):
            if (
                k <= 4096
                and m * k < 2**31
                and (m - 1) * a.stride(0) + (k - 1) * a.stride(1) < 2**31
            ):
                _quantize_rows[(m,)](
                    a,
                    aq,
                    sa,
                    k,
                    *a.stride(),
                    triton.next_power_of_2(k),
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            else:
                _quantize[(m,)](
                    a, aq, sa, k, *a.stride(), 1024, num_warps=4, enable_fp_fusion=False
                )
            if b.is_contiguous() and k <= 4096 and n * k < 2**31 and n >= 16:
                bn = 8 if k <= 512 or k > 2048 else 16
                nw = 8 if 512 < k <= 1024 or k > 2048 else 4
                _quantize_columns_full[(triton.cdiv(n, bn),)](
                    b, bq, sb, k, n, bn, num_warps=nw, enable_fp_fusion=False
                )
            elif b.stride(1) == 1 and n >= 16:
                _quantize_columns[(triton.cdiv(n, 16),)](
                    b,
                    bq,
                    sb,
                    k,
                    n,
                    *b.stride(),
                    256,
                    16,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
            else:
                _quantize[(n,)](
                    b,
                    bq,
                    sb,
                    k,
                    b.stride(1),
                    b.stride(0),
                    1024,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
    return aq, bq, sa, sb


def _mm_w8a8_int8_prequantized_out(a, b, scale_a, scale_b, *, out):
    """Private PR-compatible timing entry point; inputs are already quantized."""
    if not all(isinstance(x, torch.Tensor) for x in (a, b, scale_a, scale_b, out)):
        raise TypeError("expected tensor inputs, scales and output")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("expected A[M,K] and B[K,N]")
    m, k = a.shape
    n = b.shape[1]
    if a.dtype != torch.int8 or b.dtype != torch.int8:
        raise TypeError("prequantized inputs must be INT8")
    if a.device.type != "cuda" or any(
        x.device != a.device for x in (b, scale_a, scale_b, out)
    ):
        raise ValueError("all tensors must be on the same Hygon device")
    if scale_a.shape not in ((m,), (m, 1)) or scale_b.shape not in ((n,), (1, n)):
        raise ValueError("expected per-row A scales and per-column B scales")
    if any(
        x.dtype != torch.float32 or not x.is_contiguous() for x in (scale_a, scale_b)
    ):
        raise ValueError("scales must be contiguous FP32 tensors")
    if out.shape != (m, n) or out.dtype not in _FLOATS or not out.is_contiguous():
        raise ValueError("out must be contiguous FP16/BF16/FP32 [M,N]")
    if out.numel() and any(
        x.numel() and out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
        for x in (a, b, scale_a, scale_b)
    ):
        raise ValueError("prequantized output must not alias inputs or scales")
    with torch_device_fn.device(a.device):
        # PR input preparation already supplies these layouts. Direct private
        # callers with other strides pay for normalization inside this call.
        aq = a.contiguous()
        bq = b.t().contiguous()
        return _launch_prequantized(aq, bq, scale_a, scale_b, out, m, n, k)


def _mm_w8a8_int8_prequantized(a, b, scale_a, scale_b, *, out_dtype=torch.bfloat16):
    if out_dtype not in _FLOATS:
        raise TypeError("out_dtype must be FP16, BF16 or FP32")
    if (
        not isinstance(a, torch.Tensor)
        or not isinstance(b, torch.Tensor)
        or a.ndim != 2
        or b.ndim != 2
    ):
        raise ValueError("expected matrix inputs")
    out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=out_dtype)
    return _mm_w8a8_int8_prequantized_out(a, b, scale_a, scale_b, out=out)
