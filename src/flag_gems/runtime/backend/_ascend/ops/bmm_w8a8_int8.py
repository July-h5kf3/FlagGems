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

"""Ascend channel-scaled INT8 batched matrix multiplication."""

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .mm_w8a8_int8 import direct_mm_kernel
from .mm_w8a8_int8 import mm_int8_vector_full
from .mm_w8a8_int8 import mm_int8_vector_loop
from .mm_w8a8_int8 import mm_w8a8_int8_out


direct_scaled_mm = direct_mm_kernel.fn
vector_loop = mm_int8_vector_loop.fn
vector_full = mm_int8_vector_full.fn


@libentry()
@triton.jit
def bmm_int8_vector_loop(
    A,
    B,
    SA,
    SB,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
):
    batch = tl.program_id(1).to(tl.int64)
    vector_loop(
        A + batch * M * K,
        B + batch * N * K,
        SA + batch * M,
        SB + batch * N,
        OUT + batch * M * N,
        M,
        N,
        K,
        K,
        1,
        1,
        K,
        N,
        1,
        BN,
        1024,
        True,
    )


@libentry()
@triton.jit
def bmm_int8_vector_full(
    A,
    B,
    SA,
    SB,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
):
    batch = tl.program_id(1).to(tl.int64)
    vector_full(
        A + batch * M * K,
        B + batch * N * K,
        SA + batch * M,
        SB + batch * N,
        OUT + batch * M * N,
        M,
        N,
        K,
        K,
        1,
        1,
        K,
        N,
        1,
        1,
        BK,
    )


@libentry()
@triton.jit
def bmm_direct_scaled_mm(
    A,
    B,
    SA,
    SB,
    OUT,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    PADDED_M: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    CORES: tl.constexpr,
):
    batch = tl.program_id(1).to(tl.int64)
    direct_scaled_mm(
        A + batch * PADDED_M * K,
        B + batch * N * K,
        SA + batch * M,
        SB + batch * N,
        OUT + batch * M * N,
        M,
        N,
        K,
        PADDED_M,
        K,
        1,
        1,
        K,
        N,
        1,
        BM,
        BN,
        BK,
        CORES,
        True,
    )


def bmm_w8a8_int8(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute channel-scaled INT8 A[B,M,K] @ W[B,K,N] on Ascend."""
    if activation.ndim != 3 or weight.ndim != 3:
        raise ValueError("expected activation[B,M,K] and weight[B,K,N]")
    batch, rows, reduction = activation.shape
    columns = weight.shape[2]
    if weight.shape[:2] != (batch, reduction):
        raise ValueError("batch and reduction dimensions must match")
    if activation.dtype != torch.int8 or weight.dtype != torch.int8:
        raise TypeError("activation and weight must be signed INT8 tensors")
    if activation.device.type != "npu" or any(
        tensor.device != activation.device
        for tensor in (weight, activation_scale, weight_scale)
    ):
        raise ValueError("inputs and scales must be on the same Ascend NPU")
    if not activation.is_contiguous() or not weight.transpose(1, 2).is_contiguous():
        raise ValueError("expected contiguous activation and column-major weight")
    if activation_scale.shape not in ((batch, rows), (batch, rows, 1)):
        raise ValueError("activation_scale must have shape [B,M] or [B,M,1]")
    if weight_scale.shape not in ((batch, columns), (batch, 1, columns)):
        raise ValueError("weight_scale must have shape [B,N] or [B,1,N]")
    if any(
        scale.dtype != torch.float32 or not scale.is_contiguous()
        for scale in (activation_scale, weight_scale)
    ):
        raise TypeError("scales must be contiguous FP32 tensors")
    if any(scale.requires_grad for scale in (activation_scale, weight_scale)):
        raise NotImplementedError("bmm_w8a8_int8 is forward-only")
    if reduction >= 2**31:
        raise ValueError("reduction must be smaller than 2**31")
    if out_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("output dtype must be bfloat16, float16 or float32")
    if out is None:
        out = torch.empty(
            (batch, rows, columns), dtype=out_dtype, device=activation.device
        )
    elif (
        out.shape != (batch, rows, columns)
        or out.dtype != out_dtype
        or out.device != activation.device
        or not out.is_contiguous()
        or out.requires_grad
    ):
        raise ValueError(
            "out must have matching shape, dtype and device, be contiguous "
            "and not require gradients"
        )
    if out.numel() and any(
        out.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
        for tensor in (activation, weight, activation_scale, weight_scale)
    ):
        raise ValueError("out must not share storage with an input")
    if out.numel() == 0:
        return out
    if reduction == 0 or reduction > 65536:
        for index in range(batch):
            mm_w8a8_int8_out(
                activation[index],
                weight[index],
                activation_scale[index],
                weight_scale[index],
                out=out[index],
            )
        return out
    with torch_device_fn.device(activation.device):
        if rows in (1, 2) and columns == 64 and reduction == 4096:
            block_n = 4 if rows == 1 else 8
            bmm_int8_vector_loop[(rows * triton.cdiv(columns, block_n), batch)](
                activation,
                weight,
                activation_scale,
                weight_scale,
                out,
                rows,
                columns,
                reduction,
                block_n,
                num_warps=1,
                multibuffer=False,
                enable_fp_fusion=False,
            )
            return out
        if (
            rows in (1, 2)
            and columns in (1, 4)
            and 1024 <= reduction <= 16384
            and reduction % 1024 == 0
        ):
            bmm_int8_vector_full[(rows * columns, batch)](
                activation,
                weight,
                activation_scale,
                weight_scale,
                out,
                rows,
                columns,
                reduction,
                triton.next_power_of_2(reduction),
                num_warps=1,
                multibuffer=False,
                enable_fp_fusion=False,
            )
            return out
        if columns <= 256:
            block_m = 16 if rows <= 128 else 128
            block_n = 32 if rows <= 128 else (64 if columns < 256 else 256)
            block_k = min(
                1024 if rows <= 128 else 512,
                max(32, triton.next_power_of_2(reduction)),
            )
        else:
            block_m = 16 if rows <= 16 else (32 if rows <= 32 else (64 if rows <= 256 else 128))
            block_n = 64 if rows <= 64 else (128 if rows <= 256 else 256)
            if rows <= 16:
                max_block_k = 2048
            else:
                max_block_k = 1024 if rows <= 256 else 512
            block_k = min(max_block_k, max(32, triton.next_power_of_2(reduction)))
        cores = min(20, triton.cdiv(rows, block_m) * triton.cdiv(columns, block_n))
        bmm_direct_scaled_mm[(cores, batch)](
            activation,
            weight,
            activation_scale,
            weight_scale,
            out,
            rows,
            columns,
            reduction,
            rows,
            block_m,
            block_n,
            block_k,
            cores,
            num_warps=1,
            num_stages=2,
            optimize_dynamic_offset=True,
            unit_flag=False,
            limit_auto_multi_buffer_of_local_buffer="no-l0c",
            set_workspace_multibuffer=1,
        )
    return out
