# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""INT8 block-scaled BMM for T-Head PPU, with BF16 dequantization/accumulation."""

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .bmm_block_ldc_config import AIU_OVERRIDES, PACK_CONFIGS, get_config
from .bmm_block_ldc_small import small_bmm_kernel


@libentry()
@triton.jit
def _bmm_block_ldc_kernel(
    A,
    W,
    AS,
    WS,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    SA: tl.constexpr,
    SW: tl.constexpr,
    SAS: tl.constexpr,
    SWS: tl.constexpr,
    SO: tl.constexpr,
    SCALE_M: tl.constexpr,
    SCALE_N: tl.constexpr,
    SCALE_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALIGNED: tl.constexpr,
    USE_AIU: tl.constexpr,
    USE_AIU_W: tl.constexpr,
    TILE_K: tl.constexpr,
):
    batch = tl.program_id(1).to(tl.int64)
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BLOCK_M)
    nn = tl.cdiv(N, BLOCK_N)
    group = pid // (GROUP_M * nn)
    first_m = group * GROUP_M
    size_m = tl.minimum(nm - first_m, GROUP_M)
    local = pid % (GROUP_M * nn)
    pm = first_m + local % size_m
    pn = local // size_m
    rm = pm * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, TILE_K)
    if USE_AIU:
        a_bp = tl.make_block_ptr(
            A + batch * SA[0],
            shape=(M, K),
            strides=(SA[1], SA[2]),
            offsets=(pm * BLOCK_M, 0),
            block_shape=(BLOCK_M, TILE_K),
            order=(1, 0),
        )
        w_bp = tl.make_block_ptr(
            W + batch * SW[0],
            shape=(K, N),
            strides=(SW[1], SW[2]),
            offsets=(0, pn * BLOCK_N),
            block_shape=(TILE_K, BLOCK_N),
            order=(0, 1) if SW[1] == 1 else (1, 0),
        )
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.bfloat16)
    for kb in tl.range(
        tl.cdiv(K, TILE_K),
        loop_unroll_factor=(
            (K + TILE_K - 1) // TILE_K if K > 0 and K <= 256 and not USE_AIU else 1
        ),
    ):
        kk = kb * TILE_K + rk
        scale_k = kb * TILE_K // SCALE_K
        ap = A + batch * SA[0] + rm[:, None] * SA[1] + kk[None, :] * SA[2]
        wp = W + batch * SW[0] + kk[:, None] * SW[1] + rn[None, :] * SW[2]
        if K <= 256 and SCALE_M >= BLOCK_M and SCALE_M % BLOCK_M == 0:
            asp = (
                AS
                + batch * SAS[0]
                + (pm * BLOCK_M // SCALE_M) * SAS[1]
                + scale_k * SAS[2]
            )
        else:
            asp = AS + batch * SAS[0] + (rm // SCALE_M) * SAS[1] + scale_k * SAS[2]
        # BLOCK_N divides SCALE_N: all columns of the tile share one weight scale.
        wsp = (
            WS + batch * SWS[0] + scale_k * SWS[1] + (pn * BLOCK_N // SCALE_N) * SWS[2]
        )
        if USE_AIU:
            a = tle.load(a_bp, is_async=True)
            if USE_AIU_W:
                w = tle.load(w_bp, is_async=True)
            else:
                w = tl.load(wp)
            a_bp = tl.advance(a_bp, (0, TILE_K))
            w_bp = tl.advance(w_bp, (TILE_K, 0))
            a_scale = tl.load(asp)
        elif ALIGNED:
            a = tl.load(ap)
            w = tl.load(wp)
            a_scale = tl.load(asp)
        else:
            a = tl.load(ap, (rm[:, None] < M) & (kk[None, :] < K), other=0)
            w = tl.load(wp, (kk[:, None] < K) & (rn[None, :] < N), other=0)
            if K <= 256 and SCALE_M >= BLOCK_M and SCALE_M % BLOCK_M == 0:
                a_scale = tl.load(asp)
            else:
                a_scale = tl.load(asp, rm < M, other=0)
        w_scale = tl.load(wsp)
        row_scale = (a_scale.to(tl.float32) * w_scale.to(tl.float32)).to(tl.bfloat16)
        partial = tl.dot(a, w, out_dtype=tl.int32).to(tl.bfloat16)
        if K <= 256 and SCALE_M >= BLOCK_M and SCALE_M % BLOCK_M == 0:
            dequant = (partial * row_scale).to(tl.bfloat16)
        else:
            dequant = (partial * row_scale[:, None]).to(tl.bfloat16)
        if K <= TILE_K:
            acc = dequant
        else:
            acc = (acc + dequant).to(tl.bfloat16)
    op = O + batch * SO[0] + rm[:, None] * SO[1] + rn[None, :] * SO[2]
    if ALIGNED:
        tl.store(op, acc)
    else:
        tl.store(op, acc, (rm[:, None] < M) & (rn[None, :] < N))


@libentry()
@triton.jit
def _pack_weight_kernel(
    W,
    P,
    K: tl.constexpr,
    N: tl.constexpr,
    SW: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
):
    batch = tl.program_id(2).to(tl.int64)
    kk = tl.program_id(0) * BK + tl.arange(0, BK)
    nn = tl.program_id(1) * BN + tl.arange(0, BN)
    w = tl.load(
        W + batch * SW[0] + kk[:, None] * SW[1] + nn[None, :] * SW[2],
        (kk[:, None] < K) & (nn[None, :] < N),
        other=0,
    )
    tl.store(
        P + batch * K * N + nn[None, :] * K + kk[:, None],
        w,
        (kk[:, None] < K) & (nn[None, :] < N),
    )


def bmm_block_ldc(
    A,
    B,
    A_scale,
    B_scale,
    block_size=(128, 128, 128),
    out_dtype=torch.bfloat16,
    out=None,
):
    """Compute A[B,M,K] @ B[B,K,N] with block scales and BF16 accumulation.

    A_scale is [B,ceil(M/block_m),ceil(K/block_k)]; B_scale is
    [B,ceil(K/block_k),ceil(N/block_n)]. Set block_m=1 for per-row scales.
    Unlike the NVIDIA FP8 entry point, input tensors must be signed INT8.
    Arbitrary nonnegative input strides are supported; output must be contiguous.
    BF16 accumulation is approximate, including when out_dtype is float32.
    """
    tensors = (A, B, A_scale, B_scale)
    if any(t.ndim != 3 for t in tensors):
        raise ValueError("inputs and scales must have three dimensions")
    if A.dtype != torch.int8 or B.dtype != torch.int8:
        raise TypeError("PPU block BMM requires INT8 A and B")
    if any(t.device != A.device for t in tensors) or A.device.type != "cuda":
        raise ValueError("all inputs must be on the same PPU device")
    if any(
        t.dtype not in (torch.float32, torch.bfloat16, torch.float16)
        for t in tensors[2:]
    ):
        raise TypeError("scales must be float32, bfloat16 or float16")
    if len(block_size) != 3 or any(type(v) is not int or v <= 0 for v in block_size):
        raise ValueError("block_size must contain three positive integers")
    sm, sn, sk = block_size
    if sn < 16 or sn & (sn - 1) or sk != 128:
        raise NotImplementedError("requires power-of-two block_n >= 16 and block_k=128")
    batch, m, k = A.shape
    if B.shape[:2] != (batch, k):
        raise ValueError("batch or K dimension mismatch")
    n = B.shape[2]
    if A_scale.shape != (batch, triton.cdiv(m, sm), triton.cdiv(k, sk)):
        raise ValueError("incorrect A_scale shape")
    if B_scale.shape != (batch, triton.cdiv(k, sk), triton.cdiv(n, sn)):
        raise ValueError("incorrect B_scale shape")
    if out_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("unsupported output dtype")
    if out is None:
        out = torch.empty((batch, m, n), dtype=out_dtype, device=A.device)
    elif (
        out.shape != (batch, m, n)
        or out.dtype != out_dtype
        or out.device != A.device
        or not out.is_contiguous()
    ):
        raise ValueError("out must be contiguous with matching shape, dtype and device")
    if batch == 0 or m == 0 or n == 0:
        return out
    with torch_device_fn.device(A.device):
        if m <= 16 and n <= 32 and 0 < k <= 64:
            small_bmm_kernel[(m * triton.cdiv(n, 2), batch)](
                A,
                B,
                A_scale,
                B_scale,
                out,
                m,
                n,
                k,
                A.stride(),
                B.stride(),
                A_scale.stride(),
                B_scale.stride(),
                out.stride(),
                sm,
                sn,
                triton.next_power_of_2(k),
                2,
                num_warps=1,
                num_stages=1,
            )
            return out
        cfg = get_config(batch, m, n, k, sn)
        bm, bn = cfg["BLOCK_M"], cfg["BLOCK_N"]
        tile_k = min(sk, max(32, triton.next_power_of_2(k)))
        aligned = m % bm == 0 and n % bn == 0 and k % tile_k == 0
        use_aiu = (
            aligned
            and A.is_contiguous()
            and A.stride(2) == 1
            and B.stride(2) == 1
            and A.data_ptr() % 128 == 0
            and A.stride(0) % 128 == 0
            and A.stride(1) % 128 == 0
            and AIU_OVERRIDES.get((batch, m, n, k), True)
        )
        if use_aiu and m >= 256 and n >= 256 and k >= 128:
            packed = torch.empty((batch, n, k), dtype=torch.int8, device=B.device)
            pk, pn, pw = PACK_CONFIGS.get((batch, n, k), (32, 128, 4))
            _pack_weight_kernel[(triton.cdiv(k, pk), triton.cdiv(n, pn), batch)](
                B,
                packed,
                k,
                n,
                B.stride(),
                pk,
                pn,
                num_warps=pw,
            )
            B = packed.transpose(1, 2)
        _bmm_block_ldc_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn), batch)](
            A,
            B,
            A_scale,
            B_scale,
            out,
            m,
            n,
            k,
            A.stride(),
            B.stride(),
            A_scale.stride(),
            B_scale.stride(),
            out.stride(),
            sm,
            sn,
            sk,
            ALIGNED=aligned,
            USE_AIU=use_aiu,
            USE_AIU_W=(use_aiu and B.stride(1) == 1),
            TILE_K=tile_k,
            **cfg,
        )
    return out


bmm_w8a8_int8 = bmm_block_ldc
