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

import random

import numpy as np
import pytest
import torch

import flag_gems
from flag_gems.ops.bmm_w8a8_fp8 import bmm_w8a8_fp8 as nvidia_bmm_w8a8_fp8

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    MNK_SHAPES = [
        (1, 1, 32),
    ]
    FLOAT_DTYPES = [torch.float32]
else:
    MNK_SHAPES = [
        (1, 1, 32),
        (15, 160, 1024),
        (495, 5333, 71),
    ]
    FLOAT_DTYPES = utils.FLOAT_DTYPES


@pytest.mark.bmm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_bmm(monkeypatch, M, N, K, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("#2799: Skipping fp32 bmm test on tsingmicro platform.")

    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        random.seed(0)

    batch = 4
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((batch, K, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)

    ref_out = torch.bmm(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        res_out = torch.bmm(mat1, mat2)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.bmm
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_bmm_non_contiguous(M, N, K, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2799: Skipping fp32 bmm test on tsingmicro.")

    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        random.seed(0)

    batch = 4
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2_raw = torch.randn((batch, N, K), dtype=dtype, device=flag_gems.device)
    # make mat2 non-contiguous
    mat2 = mat2_raw.transpose(1, 2)

    if N > 1 and K > 1:
        assert not mat2.is_contiguous()
    else:
        # Skipping non-contiguous test for small N or K
        return

    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_out = torch.bmm(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        res_out = torch.bmm(mat1, mat2)
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.bmm_out
@pytest.mark.parametrize("M, N, K", MNK_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_bmm_out(M, N, K, dtype):
    if flag_gems.vendor_name == "tsingmicro" and dtype == torch.float32:
        pytest.skip("Issue #2799: Skipping fp32 bmm test on tsingmicro.")

    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        random.seed(0)

    batch = 4
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((batch, K, N), dtype=dtype, device=flag_gems.device)
    out = torch.empty((batch, M, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)

    ref_out = torch.bmm(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        torch.bmm(mat1, mat2, out=out)

    utils.gems_assert_close(out, ref_out, dtype, reduce_dim=K)


# Retain the upstream FP8 path on NVIDIA; PPU uses signed INT8 quantization.
IS_PPU = flag_gems.vendor_name == "thead"
FP8_DTYPE = torch.int8 if IS_PPU else getattr(torch, "float8_e4m3fn", None)
bmm_w8a8_fp8 = flag_gems.bmm_w8a8_int8 if IS_PPU else nvidia_bmm_w8a8_fp8
FP8_W8A8_BLOCK_SCALE_BMM_SHAPES = [
    (2, 16, 32, 64),
    (4, 64, 64, 128),
    (2, 128, 128, 256),
]


def _is_fp8e4nv_supported():
    if IS_PPU:
        return torch.cuda.is_available()
    if flag_gems.device != "cuda" or FP8_DTYPE is None:
        return False
    major, minor = torch.cuda.get_device_capability()
    return major + minor / 10 >= 8.9


def _round_for_ppu(x):
    return x.round() if IS_PPU else x


def _quantize_a_fp8_per_mk_block(A, block_m=128, block_k=128):
    fp8_info = torch.iinfo(FP8_DTYPE) if IS_PPU else torch.finfo(FP8_DTYPE)
    batch, M, K = A.shape
    num_m_blocks = (M + block_m - 1) // block_m
    num_k_blocks = (K + block_k - 1) // block_k
    padded_m = num_m_blocks * block_m
    padded_k = num_k_blocks * block_k
    A_for_scale = A
    if padded_m != M:
        A_for_scale = torch.cat(
            [
                A_for_scale,
                torch.zeros((batch, padded_m - M, K), dtype=A.dtype, device=A.device),
            ],
            dim=1,
        )
    if padded_k != K:
        A_for_scale = torch.cat(
            [
                A_for_scale,
                torch.zeros(
                    (batch, padded_m, padded_k - K),
                    dtype=A.dtype,
                    device=A.device,
                ),
            ],
            dim=2,
        )
    A_blocked = A_for_scale.reshape(
        batch, num_m_blocks, block_m, num_k_blocks, block_k
    ).float()
    scale = (A_blocked.abs().amax(dim=(2, 4)) / fp8_info.max).clamp(min=1e-8)
    A_fp8 = (
        (_round_for_ppu(A_blocked / scale[:, :, None, :, None]))
        .clamp(fp8_info.min, fp8_info.max)
        .to(FP8_DTYPE)
    )
    A_fp8 = A_fp8.reshape(batch, padded_m, padded_k)[:, :M, :K].contiguous()
    return A_fp8, scale.float().contiguous()


def _dequant_a_fp8_per_mk_block(A_fp8, A_scale, block_m=128, block_k=128):
    M = A_fp8.shape[1]
    K = A_fp8.shape[2]
    m_ids = torch.arange(M, device=A_fp8.device) // block_m
    k_ids = torch.arange(K, device=A_fp8.device) // block_k
    scale = A_scale.index_select(1, m_ids).index_select(2, k_ids)
    return A_fp8.float() * scale.float()


def _quantize_b_fp8_per_nk_block(B, block_n=128, block_k=128):
    fp8_info = torch.iinfo(FP8_DTYPE) if IS_PPU else torch.finfo(FP8_DTYPE)
    batch, K, N = B.shape
    num_k_blocks = (K + block_k - 1) // block_k
    num_n_blocks = (N + block_n - 1) // block_n
    padded_k = num_k_blocks * block_k
    padded_n = num_n_blocks * block_n
    B_for_scale = B
    if padded_k != K:
        B_for_scale = torch.cat(
            [
                B_for_scale,
                torch.zeros((batch, padded_k - K, N), dtype=B.dtype, device=B.device),
            ],
            dim=1,
        )
    if padded_n != N:
        B_for_scale = torch.cat(
            [
                B_for_scale,
                torch.zeros(
                    (batch, padded_k, padded_n - N),
                    dtype=B.dtype,
                    device=B.device,
                ),
            ],
            dim=2,
        )
    B_blocked = B_for_scale.reshape(
        batch, num_k_blocks, block_k, num_n_blocks, block_n
    ).float()
    scale = (B_blocked.abs().amax(dim=(2, 4)) / fp8_info.max).clamp(min=1e-8)
    B_fp8 = (
        (_round_for_ppu(B_blocked / scale[:, :, None, :, None]))
        .clamp(fp8_info.min, fp8_info.max)
        .to(FP8_DTYPE)
    )
    B_fp8 = B_fp8.reshape(batch, padded_k, padded_n)[:, :K, :N].contiguous()
    return B_fp8, scale.float().contiguous()


def _dequant_b_fp8_per_nk_block(B_fp8, B_scale, block_n=128, block_k=128):
    K = B_fp8.shape[1]
    N = B_fp8.shape[2]
    k_ids = torch.arange(K, device=B_fp8.device) // block_k
    n_ids = torch.arange(N, device=B_fp8.device) // block_n
    scale = B_scale.index_select(1, k_ids).index_select(2, n_ids)
    return B_fp8.float() * scale.float()


@pytest.mark.bmm_w8a8_fp8
@pytest.mark.skipif(
    not _is_fp8e4nv_supported(),
    reason="Block-scale BMM requires NVIDIA FP8 or T-Head INT8 support",
)
@pytest.mark.parametrize("batch, M, N, K", FP8_W8A8_BLOCK_SCALE_BMM_SHAPES)
def test_bmm_w8a8_fp8(batch, M, N, K):
    torch.manual_seed(0)
    A = torch.randn((batch, M, K), dtype=torch.bfloat16, device=flag_gems.device)
    B = torch.randn((batch, K, N), dtype=torch.bfloat16, device=flag_gems.device)
    A_fp8, A_scale = _quantize_a_fp8_per_mk_block(A)
    B_fp8, B_scale = _quantize_b_fp8_per_nk_block(B)
    A_dequant = _dequant_a_fp8_per_mk_block(A_fp8, A_scale)
    B_dequant = _dequant_b_fp8_per_nk_block(B_fp8, B_scale)

    ref_out = torch.bmm(A_dequant, B_dequant).to(torch.bfloat16)
    res_out = bmm_w8a8_fp8(A_fp8, B_fp8, A_scale, B_scale)

    if IS_PPU:
        nrms = (
            (res_out.float() - ref_out.float()).square().mean()
            / ref_out.float().square().mean().clamp_min(1e-20)
        ).sqrt()
        assert torch.isfinite(res_out).all()
        assert nrms.item() < 0.10
    else:
        utils.gems_assert_close(res_out, ref_out, torch.bfloat16, reduce_dim=K)


@pytest.mark.bmm_w8a8_fp8
@pytest.mark.skipif(not IS_PPU, reason="PPU INT8 block BMM")
@pytest.mark.parametrize(
    "shape,block_m,block_n,strided",
    [
        ((2, 17, 35, 129), 1, 128, False),
        ((2, 129, 257, 259), 128, 128, False),
        ((1, 65, 49, 257), 7, 16, True),
        ((2, 64, 128, 256), 1, 128, True),
        ((1, 8, 16, 0), 128, 128, False),
        ((0, 8, 16, 128), 128, 128, False),
        ((1, 0, 16, 128), 128, 128, False),
        ((1, 8, 0, 128), 128, 128, False),
    ],
)
def test_bmm_w8a8_int8_edges(shape, block_m, block_n, strided):
    batch, m, n, k = shape
    torch.manual_seed(42)
    aq = torch.randint(
        -128, 128, (batch, m, k), dtype=torch.int8, device=flag_gems.device
    )
    bq = torch.randint(
        -128, 128, (batch, k, n), dtype=torch.int8, device=flag_gems.device
    )
    asc = (
        torch.rand(
            (batch, (m + block_m - 1) // block_m, (k + 127) // 128), device=aq.device
        )
        * 0.01
    )
    bsc = (
        torch.rand(
            (batch, (k + 127) // 128, (n + block_n - 1) // block_n), device=aq.device
        )
        * 0.01
    )
    if strided:
        aq = aq.transpose(1, 2).contiguous().transpose(1, 2)
        bq = bq.transpose(1, 2).contiguous().transpose(1, 2)
        asc = asc.transpose(1, 2).contiguous().transpose(1, 2)
        bsc = bsc.transpose(1, 2).contiguous().transpose(1, 2)
    out = torch.full(
        (batch, m, n), float("nan"), dtype=torch.bfloat16, device=aq.device
    )
    result = flag_gems.bmm_w8a8_int8(
        aq, bq, asc, bsc, block_size=(block_m, block_n, 128), out=out
    )
    assert result is out
    assert torch.isfinite(result).all()
    if result.numel() == 0:
        return
    if k == 0:
        assert torch.count_nonzero(result) == 0
        return
    ad = _dequant_a_fp8_per_mk_block(aq, asc, block_m=block_m)
    bd = _dequant_b_fp8_per_nk_block(bq, bsc, block_n=block_n)
    ref = torch.bmm(ad, bd)
    nrms = ((result.float() - ref).square().mean() / ref.square().mean()).sqrt()
    assert nrms.item() < 0.10


@pytest.mark.bmm_w8a8_fp8
@pytest.mark.skipif(not IS_PPU, reason="PPU INT8 block BMM")
def test_bmm_w8a8_int8_validation():
    a = torch.empty((1, 16, 128), dtype=torch.int8, device=flag_gems.device)
    b = torch.empty((1, 128, 16), dtype=torch.int8, device=flag_gems.device)
    scale = torch.ones((1, 1, 1), device=flag_gems.device)
    with pytest.raises(TypeError, match="INT8"):
        flag_gems.bmm_w8a8_int8(a.float(), b, scale, scale)
    with pytest.raises(ValueError, match="A_scale"):
        flag_gems.bmm_w8a8_int8(a, b, scale.expand(1, 2, 1), scale)
    with pytest.raises(NotImplementedError):
        flag_gems.bmm_w8a8_int8(a, b, scale, scale, block_size=(128, 128, 64))
    out = torch.empty((1, 16, 32), dtype=torch.bfloat16, device=a.device)[:, :, ::2]
    with pytest.raises(ValueError, match="out"):
        flag_gems.bmm_w8a8_int8(a, b, scale, scale, out=out)


@pytest.mark.bmm_w8a8_fp8
@pytest.mark.skipif(not IS_PPU, reason="PPU INT8 block BMM")
def test_bmm_w8a8_int8_upstream_shapes_nrms():
    # Reuse the upstream benchmark's shape list and input quantization exactly.
    # Sample 32 rows/columns per batch to avoid a full FP32 large-K reference.
    from benchmark.test_bmm import BmmW8A8Fp8Benchmark, _gems_bmm_w8a8_fp8

    bench = BmmW8A8Fp8Benchmark(op_name="bmm_w8a8_int8", torch_op=torch.bmm)
    bench.set_shapes()
    torch.manual_seed(0)
    for args in bench.get_input_iter(torch.bfloat16):
        a, b, aq, bq, asc, bsc, out, sm, sn, sk = args
        batch, m, k = a.shape
        n = b.shape[2]
        rows = torch.linspace(0, m - 1, min(m, 32), device=a.device).long()
        cols = torch.linspace(0, n - 1, min(n, 32), device=a.device).long()
        kids = torch.arange(k, device=a.device) // sk
        ad = aq[:, rows, :].float() * asc[:, rows // sm, :][:, :, kids]
        bd = bq[:, :, cols].float() * bsc[:, kids, :][:, :, cols // sn]
        ref = torch.bmm(ad, bd)
        original_ref = torch.bmm(a[:, rows, :].float(), b[:, :, cols].float())
        result = _gems_bmm_w8a8_fp8(*args)
        assert torch.isfinite(result).all()
        sampled = result[:, rows, :][:, :, cols].float()
        nrms = ((sampled - ref).square().mean() / ref.square().mean()).sqrt().item()
        total_nrms = (
            ((sampled - original_ref).square().mean() / original_ref.square().mean())
            .sqrt()
            .item()
        )
        print(
            f"shape={(batch, m, n, k)} dequant_nrms={nrms:.6f} total_nrms={total_nrms:.6f}"
        )
        assert nrms < 0.10 and total_nrms < 0.10


@pytest.mark.bmm_w8a8_fp8
@pytest.mark.skipif(not IS_PPU, reason="PPU INT8 block BMM")
@pytest.mark.parametrize("layout", ["offset", "padded", "broadcast"])
def test_bmm_w8a8_int8_unaligned_storage(layout):
    batch, m, n, k = 2, 256, 256, 512
    if layout == "broadcast":
        a = torch.randint(
            -128, 128, (1, m, k), dtype=torch.int8, device=flag_gems.device
        ).expand(batch, -1, -1)
    elif layout == "padded":
        storage = torch.randint(
            -128, 128, (batch, m, k + 1), dtype=torch.int8, device=flag_gems.device
        )
        a = storage[:, :, :k]
    else:
        storage = torch.randint(
            -128, 128, (batch * m * k + 1,), dtype=torch.int8, device=flag_gems.device
        )
        a = storage[1:].view(batch, m, k)
    b = torch.randint(-128, 128, (batch, k, n), dtype=torch.int8, device=a.device)
    asc = torch.rand((batch, 2, 4), device=a.device) * 0.01
    bsc = torch.rand((batch, 4, 2), device=a.device) * 0.01
    result = flag_gems.bmm_w8a8_int8(a, b, asc, bsc)
    ref = torch.bmm(
        _dequant_a_fp8_per_mk_block(a, asc), _dequant_b_fp8_per_nk_block(b, bsc)
    )
    nrms = ((result.float() - ref).square().mean() / ref.square().mean()).sqrt()
    assert torch.isfinite(result).all() and nrms.item() < 0.10


@pytest.mark.bmm_w8a8_fp8
@pytest.mark.skipif(not IS_PPU, reason="PPU INT8 block BMM")
def test_bmm_w8a8_int8_small_signed_extremes():
    a = torch.full((2, 16, 64), -128, dtype=torch.int8, device=flag_gems.device)
    b = torch.full((2, 64, 32), 127, dtype=torch.int8, device=a.device)
    b[:, :, ::2] = -128
    asc = torch.full((2, 1, 1), 0.5, device=a.device)
    bsc = torch.full((2, 1, 1), 0.25, device=a.device)
    result = flag_gems.bmm_w8a8_int8(a, b, asc, bsc)
    ref = (torch.bmm(a.float(), b.float()) * 0.125).bfloat16()
    torch.testing.assert_close(result, ref, rtol=0, atol=0)
