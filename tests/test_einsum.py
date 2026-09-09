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

import pytest
import torch

import flag_gems
from benchmark.test_einsum import (
    EINSUM_LOW_PRECISION_DTYPE,
    _einsum_low_precision_available,
    _make_block_einsum_inputs,
)

from .accuracy_utils import FLOAT_DTYPES, gems_assert_close, to_reference
from .conftest import QUICK_MODE

if QUICK_MODE:
    EINSUM_SHAPES = {
        "matmul": [(16, 32, 64)],
        "bmm": [(2, 16, 32, 64)],
        "dot": [64],
        "outer": [(16, 32)],
        "trace": [32],
        "transpose": [(16, 32)],
        "sum": [(16, 32, 64)],
    }
else:
    EINSUM_SHAPES = {
        "matmul": [(32, 64, 128), (16, 256, 32)],
        "bmm": [(4, 32, 64, 128), (8, 16, 256, 32)],
        "dot": [64, 256, 1024],
        "outer": [(32, 64), (128, 256)],
        "trace": [32, 64, 128],
        "transpose": [(32, 64), (64, 128, 256)],
        "sum": [(32, 64, 128)],
    }


@pytest.mark.einsum
@pytest.mark.parametrize("M, K, N", EINSUM_SHAPES["matmul"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_einsum_matmul(M, K, N, dtype):
    inp1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)
    ref_out = torch.einsum("ij,jk->ik", ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.einsum("ij,jk->ik", inp1, inp2)
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.einsum
@pytest.mark.parametrize("B, M, K, N", EINSUM_SHAPES["bmm"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_einsum_bmm(B, M, K, N, dtype):
    inp1 = torch.randn((B, M, K), dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn((B, K, N), dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)
    ref_out = torch.einsum("bij,bjk->bik", ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.einsum("bij,bjk->bik", inp1, inp2)
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.einsum
@pytest.mark.parametrize("size", EINSUM_SHAPES["dot"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_einsum_dot(size, dtype):
    inp1 = torch.randn(size, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(size, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)
    ref_out = torch.einsum("i,i->", ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.einsum("i,i->", inp1, inp2)
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=size)


@pytest.mark.einsum
@pytest.mark.parametrize("M, N", EINSUM_SHAPES["outer"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_einsum_outer(M, N, dtype):
    inp1 = torch.randn(M, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(N, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)
    ref_out = torch.einsum("i,j->ij", ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.einsum("i,j->ij", inp1, inp2)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.einsum
@pytest.mark.parametrize("size", EINSUM_SHAPES["trace"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_einsum_trace(size, dtype):
    inp = torch.randn((size, size), dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)
    ref_out = torch.einsum("ii->", ref_inp)
    with flag_gems.use_gems():
        res_out = torch.einsum("ii->", inp)
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=size)


@pytest.mark.einsum
@pytest.mark.parametrize("size", EINSUM_SHAPES["trace"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_einsum_diagonal(size, dtype):
    inp = torch.randn((size, size), dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)
    ref_out = torch.einsum("ii->i", ref_inp)
    with flag_gems.use_gems():
        res_out = torch.einsum("ii->i", inp)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.einsum
@pytest.mark.parametrize("shape", EINSUM_SHAPES["transpose"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_einsum_transpose(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)
    if len(shape) == 2:
        ref_out = torch.einsum("ij->ji", ref_inp)
        with flag_gems.use_gems():
            res_out = torch.einsum("ij->ji", inp)
    else:
        ref_out = torch.einsum("ijk->kji", ref_inp)
        with flag_gems.use_gems():
            res_out = torch.einsum("ijk->kji", inp)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.einsum
@pytest.mark.parametrize("shape", EINSUM_SHAPES["sum"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_einsum_sum_all(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)
    ref_out = torch.einsum("ijk->", ref_inp)
    with flag_gems.use_gems():
        res_out = torch.einsum("ijk->", inp)
    reduce_dim = shape[0] * shape[1] * shape[2]
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)


@pytest.mark.einsum
@pytest.mark.parametrize("shape", EINSUM_SHAPES["sum"])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_einsum_sum_dim(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)
    ref_out = torch.einsum("ijk->j", ref_inp)
    with flag_gems.use_gems():
        res_out = torch.einsum("ijk->j", inp)
    reduce_dim = shape[0] * shape[2]
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)


@pytest.mark.einsum
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_einsum_ellipsis(dtype):
    shape1 = (2, 3, 32, 64)
    shape2 = (2, 3, 64, 128)
    inp1 = torch.randn(shape1, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape2, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)
    ref_out = torch.einsum("...ij,...jk->...ik", ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.einsum("...ij,...jk->...ik", inp1, inp2)
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=64)


# Keep the low-precision contraction tests beside the existing floating einsum
# tests; both call the precision-dispatched implementation in ops/bmm.py.


_EINSUM_BATCHES = (1, 4, 8, 16, 32, 64, 128)
if not QUICK_MODE:
    _EINSUM_BATCHES += (4096, 8192, 16384, 32768)
_EINSUM_BLOCK_SHAPES = [
    (b, h, r, 1024) for h, r in [(8, 4096), (16, 7168)] for b in _EINSUM_BATCHES
]


@pytest.mark.fp8_einsum
@pytest.mark.skipif(
    not _einsum_low_precision_available(), reason="requires PPU INT8 or Hopper FP8"
)
@pytest.mark.parametrize("shape", _EINSUM_BLOCK_SHAPES)
def test_fp8_einsum_block_scaled(shape):
    x, xs, y, ys, xf, yf = _make_block_einsum_inputs(
        *shape, (128, 128), flag_gems.device, EINSUM_LOW_PRECISION_DTYPE
    )
    out = flag_gems.fp8_einsum("bhr,hdr->bhd", x, xs, y, ys)
    b, h, r, d = shape
    assert out.shape == (b, h, d) and out.is_contiguous()
    assert torch.isfinite(out).all()
    rows = torch.linspace(0, b - 1, min(b, 32), device=x.device).long()
    cols = torch.linspace(0, d - 1, min(d, 32), device=x.device).long()
    kk = torch.arange(r, device=x.device) // 128
    xd = x[rows].float() * xs[rows][:, :, kk]
    yd = y[:, cols].float() * ys[:, cols // 128, :][:, :, kk]
    ref = torch.einsum("bhr,hdr->bhd", xd, yd)
    original = torch.einsum("bhr,hdr->bhd", xf[rows].float(), yf[:, cols].float())
    sampled = out[rows][:, :, cols].float()
    nrms = ((sampled - ref).square().mean() / ref.square().mean()).sqrt().item()
    total = (
        ((sampled - original).square().mean() / original.square().mean()).sqrt().item()
    )
    print(f"shape={shape} dequant_nrms={nrms:.6f} total_nrms={total:.6f}")
    limit = 0.10 if flag_gems.vendor_name == "thead" else 0.20
    assert nrms < limit and total < limit
    # Validate the floating precision route for the same layouts, including
    # the largest interleaved input whose element offsets exceed int32.
    floating = flag_gems.fp8_einsum("bhr,hdr->bhd", xf, None, yf, None)
    floating_sample = floating[rows][:, :, cols].float()
    floating_nrms = (
        ((floating_sample - original).square().mean() / original.square().mean())
        .sqrt()
        .item()
    )
    assert floating_nrms < 0.01


@pytest.mark.einsum
@pytest.mark.parametrize("shape", [(3, 2, 129, 33), (16, 4, 256, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_einsum_precision_route(shape, dtype):
    b, h, r, d = shape
    x = torch.randn((b, h, r), dtype=dtype, device=flag_gems.device)
    y = torch.randn((h, d, r), dtype=dtype, device=x.device)
    out = flag_gems.fp8_einsum("bhr,hdr->bhd", x, None, y, None, output_dtype=dtype)
    ref = torch.einsum("bhr,hdr->bhd", x.float(), y.float())
    error = ((out.float() - ref).square().mean() / ref.square().mean()).sqrt()
    assert error.item() < 0.01


@pytest.mark.fp8_einsum
@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="PPU INT8")
@pytest.mark.parametrize("layout", ["contiguous", "offset", "padded", "broadcast"])
@pytest.mark.parametrize("shape", [(16, 2, 64, 32), (32, 2, 128, 128), (3, 2, 129, 33)])
def test_fp8_einsum_layouts(shape, layout):
    b, h, r, d = shape
    if layout == "offset":
        x = torch.randint(
            -128, 128, (b * h * r + 1,), dtype=torch.int8, device=flag_gems.device
        )[1:].view(b, h, r)
    elif layout == "padded":
        x = torch.randint(
            -128, 128, (b, h, r + 1), dtype=torch.int8, device=flag_gems.device
        )[:, :, :r]
    elif layout == "broadcast":
        x = torch.randint(
            -128, 128, (1, h, r), dtype=torch.int8, device=flag_gems.device
        ).expand(b, -1, -1)
    else:
        x = torch.randint(
            -128, 128, (b, h, r), dtype=torch.int8, device=flag_gems.device
        )
    y = torch.randint(-128, 128, (h, d, r), dtype=torch.int8, device=x.device)
    xs = torch.rand((b, h, (r + 127) // 128), device=x.device) * 0.01
    ys = torch.rand((h, (d + 127) // 128, (r + 127) // 128), device=x.device) * 0.01
    out = flag_gems.fp8_einsum("bhr,hdr->bhd", x, xs, y, ys)
    kk = torch.arange(r, device=x.device) // 128
    nn = torch.arange(d, device=x.device) // 128
    ref = torch.einsum(
        "bhr,hdr->bhd", x.float() * xs[:, :, kk], y.float() * ys[:, nn, :][:, :, kk]
    )
    nrms = ((out.float() - ref).square().mean() / ref.square().mean()).sqrt()
    assert torch.isfinite(out).all() and nrms.item() < 0.10


@pytest.mark.fp8_einsum
@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="PPU INT8")
@pytest.mark.parametrize("shape", [(0, 2, 128, 32), (3, 2, 0, 32), (3, 2, 128, 0)])
def test_fp8_einsum_empty(shape):
    b, h, r, d = shape
    x = torch.empty((b, h, r), dtype=torch.int8, device=flag_gems.device)
    y = torch.empty((h, d, r), dtype=torch.int8, device=x.device)
    xs = torch.ones((b, h, (r + 127) // 128), device=x.device)
    ys = torch.ones((h, (d + 127) // 128, (r + 127) // 128), device=x.device)
    out = flag_gems.fp8_einsum("bhr,hdr->bhd", x, xs, y, ys)
    assert out.shape == (b, h, d)
    assert torch.count_nonzero(out) == 0


@pytest.mark.fp8_einsum
@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="PPU INT8")
def test_fp8_einsum_validation_and_extremes():
    x = torch.full((16, 2, 64), -128, dtype=torch.int8, device=flag_gems.device)
    y = torch.full((2, 32, 64), 127, dtype=torch.int8, device=x.device)
    y[:, ::2, :] = -128
    xs = torch.full((16, 2, 1), 0.5, device=x.device)
    ys = torch.full((2, 1, 1), 0.25, device=x.device)
    out = flag_gems.fp8_einsum("bhr,hdr->bhd", x, xs, y, ys)
    ref = (torch.einsum("bhr,hdr->bhd", x.float(), y.float()) * 0.125).bfloat16()
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    with pytest.raises(ValueError, match="equation|supports"):
        flag_gems.fp8_einsum("bij,bjk->bik", x, xs, y, ys)
    with pytest.raises(ValueError, match="scale"):
        flag_gems.fp8_einsum("bhr,hdr->bhd", x, None, y, ys)
    with pytest.raises(TypeError, match="matching"):
        flag_gems.fp8_einsum("bhr,hdr->bhd", x, xs, y.float(), ys)
