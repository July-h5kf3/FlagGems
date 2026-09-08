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

"""Prequantized INT8 matrix multiplication on THead/PPU.

A is [M, K], B is [K, N], both torch.int8. FP32 scales are per row
of A and per column of B. C = int32(A @ B) * scale_a * scale_b.
Row-major A and column-major B use AIU; other layouts use strided loads.
No quantization, data cache, packing or layout copy occurs inside the op.
AIU requires FlagTree #1026 with correct INT8 .b8 lowering.
"""

from __future__ import annotations

import logging

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
def _mm_w8a8_int8_kernel(
    A_Q,
    B_Q,
    A_SCALE,
    B_SCALE,
    OUT,
    M,
    N,
    K,
    STRIDE_AM: tl.constexpr,
    STRIDE_AK: tl.constexpr,
    STRIDE_BK: tl.constexpr,
    STRIDE_BN: tl.constexpr,
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
    a_ptrs = A_Q + offs_m[:, None] * STRIDE_AM + offs_k[None, :] * STRIDE_AK
    b_ptrs = B_Q + offs_n[:, None] * STRIDE_BN + offs_k[None, :] * STRIDE_BK

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
        a_ptrs += BLOCK_K * STRIDE_AK
        b_ptrs += BLOCK_K * STRIDE_BK

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
        OUT_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        BOUNDARY: tl.constexpr,
        STORE_MASK: tl.constexpr,
        NUM_WARPS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_m, pid_n = _grouped_pids(pid, M, N, BLOCK_M, BLOCK_N, GROUP_M)
        a_block_ptr = tl.make_block_ptr(
            A_Q,
            shape=(M, K),
            strides=(K, 1),
            offsets=(pid_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        # B is physically contiguous [N, K], logically column-major [K, N].
        # N may be padded to BLOCK_N so skinny GEMM can skip load boundary checks.
        b_block_ptr = tl.make_block_ptr(
            B_Q,
            shape=(K, N),
            strides=(1, K),
            offsets=(0, pid_n * BLOCK_N),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(0, 1),
        )
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        if BOUNDARY:
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
                acc = tl.dot(a, b, acc=acc, out_dtype=tl.int32)
                a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
                b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
        else:
            for _ in range(0, tl.cdiv(K, BLOCK_K)):
                a = tle_async.load(a_block_ptr, is_async=True)
                b = tle_async.load(b_block_ptr, is_async=True)
                acc = tl.dot(a, b, acc=acc, out_dtype=tl.int32)
                a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
                b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        if STORE_MASK:
            a_scale = tl.load(A_SCALE + offs_m, mask=offs_m < M, other=0.0).to(
                tl.float32
            )
            b_scale = tl.load(B_SCALE + offs_n, mask=offs_n < N, other=0.0).to(
                tl.float32
            )
            out = (acc.to(tl.float32) * a_scale[:, None] * b_scale[None, :]).to(
                OUT.dtype.element_ty
            )
            tl.store(
                OUT + offs_m[:, None] * OUT_N + offs_n[None, :],
                out,
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < OUT_N),
            )
        else:
            a_scale = tl.load(A_SCALE + offs_m).to(tl.float32)
            b_scale = tl.load(B_SCALE + offs_n).to(tl.float32)
            out = (acc.to(tl.float32) * a_scale[:, None] * b_scale[None, :]).to(
                OUT.dtype.element_ty
            )
            tl.store(OUT + offs_m[:, None] * OUT_N + offs_n[None, :], out)


def _pick_tiles(m: int, n: int, k: int) -> tuple[int, int, int, int, int, int]:
    """Return BLOCK_M, BLOCK_N, BLOCK_K, warps, stages, GROUP_M.

    INT8 AIU v1 wants channel bytes of 32/64/128. Prefer BLOCK_K=128 on
    long K so each CTA does fewer AIU/MMA rounds and can prefetch deeper.
    Never pad BLOCK_N far past N: a 64-wide MMA on N=1 is ~64x wasted work.
    """
    if k >= 256:
        block_k = 128
    elif k >= 64:
        block_k = 64
    else:
        block_k = 32
    if k >= 2048:
        stages = 4
    elif k >= 512:
        stages = 3
    else:
        stages = 2
    group_m = 8
    # Keep BLOCK_N close to N so skinny GEMM does not compute unused columns.
    if n <= 16:
        block_n = 16
    elif n <= 32:
        block_n = 32
    elif n < 512:
        block_n = 64
    else:
        block_n = 128
    if m <= 16:
        # Tiny wide GEMM is launch-bound; one warp + BLOCK_K=128 beats gems BF16.
        if n <= 256 and k <= 256 and k >= 128:
            return 16, min(block_n, 64), 128, 1, 2, group_m
        warps = 1 if block_n <= 32 else 4
        return 16, block_n, block_k, warps, stages, group_m
    if m <= 32:
        warps = 2 if block_n <= 16 else 4
        return 32, block_n, block_k, warps, stages, group_m
    return 64, block_n, block_k, 4, stages, group_m


@libentry()
@triton.jit
def _zero_output_kernel(OUT, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
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
        raise ValueError("inputs and scales must be on the same PPU device")
    m, k = a.shape
    n = b.shape[1]
    if scale_a.shape not in ((m,), (m, 1)) or scale_b.shape not in ((n,), (1, n)):
        raise ValueError("expected scale_a[M] or [M,1], scale_b[N] or [1,N]")
    if any(
        x.dtype != torch.float32 or not x.is_contiguous() for x in (scale_a, scale_b)
    ):
        raise ValueError("scales must be contiguous FP32 tensors")
    return m, n, k


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
    (
        block_m,
        block_n,
        block_k,
        num_warps,
        num_stages,
        group_m,
    ) = _pick_tiles(m, n, k)
    n_b = n
    use_aiu = (
        tle_async is not None and a_q.stride() == (k, 1) and b_q.stride() == (1, k)
    )
    boundary = (m % block_m) != 0 or (n_b % block_n) != 0 or (k % block_k) != 0
    store_mask = boundary or (n != n_b)
    logger.debug(
        "GEMS_THEAD MM_W8A8_AIU m=%s n=%s k=%s n_b=%s "
        "tiles=(%s,%s,%s) warps=%s stages=%s aiu=%s boundary=%s store_mask=%s",
        m,
        n,
        k,
        n_b,
        block_m,
        block_n,
        block_k,
        num_warps,
        num_stages,
        use_aiu,
        boundary,
        store_mask,
    )
    with torch_device_fn.device(a_q.device):
        if use_aiu:
            grid = (triton.cdiv(m, block_m) * triton.cdiv(n_b, block_n),)
            _mm_w8a8_aiu_kernel[grid](
                a_q,
                b_q,
                a_scale,
                b_scale,
                out,
                m,
                n_b,
                k,
                n,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                GROUP_M=group_m,
                BOUNDARY=boundary,
                STORE_MASK=store_mask,
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
                *a_q.stride(),
                *b_q.stride(),
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                GROUP_M=group_m,
                NUM_WARPS=num_warps,
                num_warps=num_warps,
                num_stages=num_stages,
            )
    return out


def _run_mm(a, b, scale_a, scale_b, out, m, n, k):
    if m == 0 or n == 0:
        return out
    if k == 0:
        with torch_device_fn.device(a.device):
            _zero_output_kernel[(triton.cdiv(m * n, 1024),)](out, m * n, BLOCK=1024)
        return out
    return _launch(a, scale_a, b, scale_b, out, m, n, k)


def mm_w8a8_int8(a, b, scale_a, scale_b, *, out_dtype=torch.bfloat16):
    m, n, k = _validate_mm_inputs(a, b, scale_a, scale_b)
    if out_dtype not in _SUPPORTED_FLOAT:
        raise TypeError("out_dtype must be BF16, FP16 or FP32")
    out = torch.empty((m, n), device=a.device, dtype=out_dtype)
    return _run_mm(a, b, scale_a, scale_b, out, m, n, k)


def mm_w8a8_int8_out(a, b, scale_a, scale_b, *, out):
    m, n, k = _validate_mm_inputs(a, b, scale_a, scale_b)
    if out.shape != (m, n) or out.device != a.device or not out.is_contiguous():
        raise ValueError("out must be contiguous [M,N] on the input device")
    if out.dtype not in _SUPPORTED_FLOAT:
        raise TypeError("out must be BF16, FP16 or FP32")
    return _run_mm(a, b, scale_a, scale_b, out, m, n, k)
