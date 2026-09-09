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

import sys

import pytest
import torch

import flag_gems

if flag_gems.vendor_name == "thead":
    _backend = sys.modules[flag_gems.mm_w8a8_int8.__module__]
    _mm_w8a8_int8_prequantized = _backend._mm_w8a8_int8_prequantized
    _mm_w8a8_int8_prequantized_out = _backend._mm_w8a8_int8_prequantized_out


pytestmark = [
    pytest.mark.mm_w8a8_int8,
    pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="THead INT8 backend"),
]

SHAPES = [
    (1, 16, 16),
    (16, 1, 128),
    (256, 1, 2048),
    (2, 32, 32),
    (8, 64, 64),
    (16, 128, 64),
    (32, 128, 128),
    (64, 256, 128),
    (128, 256, 256),
    (192, 512, 512),
    (256, 768, 1024),
    (512, 1024, 1024),
    (16, 1, 2048),
    (16, 64, 2048),
    (16, 256, 2048),
    (16, 1024, 2048),
    (16, 2048, 512),
    (16, 2048, 4096),
    (16, 9216, 2048),
    (16, 12288, 2048),
    (1, 248320, 2048),
    (3, 17, 33),
    (17, 65, 129),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.mm_w8a8_int8
def test_mm_w8a8_int8(shape, dtype):
    m, n, k = shape
    a = torch.randint(-128, 128, (m, k), device=flag_gems.device, dtype=torch.int8)
    b = torch.randint(-128, 128, (n, k), device=flag_gems.device, dtype=torch.int8).t()
    sa = torch.rand(m, device=a.device) * 0.01
    sb = torch.rand(n, device=a.device) * 0.01
    ref = ((a.float() @ b.float()) * sa[:, None] * sb[None, :]).to(dtype)
    y = _mm_w8a8_int8_prequantized(a, b, sa, sb, out_dtype=dtype)
    torch.testing.assert_close(
        y, ref, rtol=1.0e-5 if dtype == torch.float32 else 1.0e-2, atol=1.0e-5
    )
    out = torch.empty_like(y)
    assert _mm_w8a8_int8_prequantized_out(a, b, sa, sb, out=out) is out
    torch.testing.assert_close(out, y, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["row_major", "sliced", "broadcast"])
@pytest.mark.mm_w8a8_int8
def test_mm_w8a8_int8_strides(layout):
    m, n, k = 17, 35, 67
    a = torch.randint(-10, 11, (m, k), device=flag_gems.device, dtype=torch.int8)
    b = torch.randint(-10, 11, (k, n), device=a.device, dtype=torch.int8)
    if layout == "sliced":
        a = torch.randint(-10, 11, (m * 2, k * 2), device=a.device, dtype=torch.int8)[
            1::2, 1::2
        ]
        b = torch.randint(-10, 11, (k * 2, n * 2), device=a.device, dtype=torch.int8)[
            1::2, 1::2
        ]
    if layout == "broadcast":
        a = a[:1].expand(m, k)
        b = b[:, :1].expand(k, n)
    sa = torch.ones(m, device=a.device)
    sb = torch.ones(n, device=a.device)
    y = _mm_w8a8_int8_prequantized(a, b, sa, sb, out_dtype=torch.float32)
    torch.testing.assert_close(y, a.float() @ b.float(), rtol=0, atol=0)


@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("use_out", [False, True])
@pytest.mark.mm_w8a8_int8
@pytest.mark.parametrize("shape", [(16, 32, 128), (1, 64, 2048), (128, 1, 2048)])
def test_mm_w8a8_int8_updates(use_graph, use_out, shape):
    m, n, k = shape
    a = torch.ones((m, k), device=flag_gems.device, dtype=torch.int8)
    b = torch.ones((n, k), device=a.device, dtype=torch.int8).t()
    sa = torch.ones((m, 1), device=a.device)
    sb = torch.ones((1, n), device=a.device)
    out = torch.empty((m, n), device=a.device)

    def call():
        if use_out:
            return _mm_w8a8_int8_prequantized_out(a, b, sa, sb, out=out)
        return _mm_w8a8_int8_prequantized(a, b, sa, sb, out_dtype=torch.float32)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = call()
    for av, bv, sav, sbv in [
        (1, 1, 1, 1),
        (2, 1, 1, 1),
        (2, 3, 1, 1),
        (2, 3, 2, 1),
        (2, 3, 2, 0.5),
        (0, 3, 1, 1),
    ]:
        a.fill_(av)
        b.fill_(bv)
        sa.fill_(sav)
        sb.fill_(sbv)
        if use_graph:
            graph.replay()
        else:
            y = call()
        torch.testing.assert_close(y, torch.full_like(y, k * av * bv * sav * sbv))


@pytest.mark.parametrize("shape", [(2, 3, 0), (0, 3, 8), (2, 0, 8), (0, 0, 0)])
@pytest.mark.mm_w8a8_int8
def test_mm_w8a8_int8_empty(shape):
    m, n, k = shape
    a = torch.empty((m, k), device=flag_gems.device, dtype=torch.int8)
    b = torch.empty((k, n), device=a.device, dtype=torch.int8)
    sa = torch.ones(m, device=a.device)
    sb = torch.ones(n, device=a.device)
    y = _mm_w8a8_int8_prequantized(a, b, sa, sb)
    torch.testing.assert_close(y, torch.zeros_like(y))


@pytest.mark.mm_w8a8_int8
def test_mm_w8a8_int8_reject_float():
    a = torch.ones((2, 3), device=flag_gems.device)
    b = torch.ones((3, 4), device=a.device)
    with pytest.raises(TypeError, match="prequantized"):
        _mm_w8a8_int8_prequantized(
            a, b, torch.ones(2, device=a.device), torch.ones(4, device=a.device)
        )


def _floating_int8_reference(a, b, dtype):
    # CPU reference computes the integer product exactly, independently of the
    # PPU quantization preparation and GEMM implementation.
    a, b = a.detach().float().cpu(), b.detach().float().cpu()
    if not a.shape[0] or not b.shape[1] or not a.shape[1]:
        return torch.zeros((a.shape[0], b.shape[1]), dtype=dtype)
    peak_a = a.abs().amax(1).clamp_min(1e-10)
    peak_b = b.abs().amax(0).clamp_min(1e-10)
    sa, sb = peak_a * (1.0 / 127), peak_b * (1.0 / 127)
    aq = torch.round((a / peak_a[:, None]) * 127).clamp(-127, 127).to(torch.int64)
    bq = torch.round((b / peak_b[None, :]) * 127).clamp(-127, 127).to(torch.int64)
    return ((aq @ bq).float() * sa[:, None] * sb[None, :]).to(dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("column_major", [False, True])
@pytest.mark.parametrize("shape", [(1, 17, 32), (17, 35, 67), (64, 128, 128)])
def test_mm_w8a8_int8_floating(dtype, out_dtype, column_major, shape):
    m, n, k = shape
    torch.manual_seed(42)
    a = torch.randn((m, k), device=flag_gems.device, dtype=dtype)
    b = torch.randn((n, k) if column_major else (k, n), device=a.device, dtype=dtype)
    if column_major:
        b = b.t()
    # Include all-zero rows/columns in scale handling.
    a[0] = 0
    b[:, 0] = 0
    ref = _floating_int8_reference(a, b, out_dtype)
    y = flag_gems.mm_w8a8_int8(a, b, out_dtype=out_dtype)
    torch.testing.assert_close(y.cpu(), ref, rtol=1e-5, atol=1e-4)
    out = torch.empty_like(y)
    assert flag_gems.mm_w8a8_int8_out(a, b, out=out) is out
    torch.testing.assert_close(out, y, rtol=0, atol=0)


@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("use_out", [False, True])
@pytest.mark.parametrize("column_major", [False, True])
def test_mm_w8a8_int8_floating_updates(use_graph, use_out, column_major):
    a = torch.ones((16, 32), device=flag_gems.device)
    b = torch.ones((17, 32) if column_major else (32, 17), device=a.device)
    if column_major:
        b = b.t()
    out = torch.empty((16, 17), device=a.device)

    def call():
        if use_out:
            return flag_gems.mm_w8a8_int8_out(a, b, out=out)
        return flag_gems.mm_w8a8_int8(a, b, out_dtype=torch.float32)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = call()
    for av, bv in [(1, 1), (2, -3), (0, 2), (0.5, 4)]:
        a.fill_(av)
        b.fill_(bv)
        if use_graph:
            graph.replay()
        else:
            y = call()
        torch.testing.assert_close(y, torch.full_like(y, 32 * av * bv))


@pytest.mark.parametrize("shape", [(2, 3, 0), (0, 3, 8), (2, 0, 8)])
def test_mm_w8a8_int8_floating_empty(shape):
    m, n, k = shape
    a = torch.empty((m, k), device=flag_gems.device)
    b = torch.empty((k, n), device=a.device)
    y = flag_gems.mm_w8a8_int8(a, b)
    assert y.dtype == torch.bfloat16
    torch.testing.assert_close(y, torch.zeros_like(y))
    out = torch.empty((m, n), device=a.device)
    assert flag_gems.mm_w8a8_int8_out(a, b, out=out) is out
    torch.testing.assert_close(out, torch.zeros_like(out))


def test_mm_w8a8_int8_floating_strides_and_validation():
    a = torch.randn((34, 134), device=flag_gems.device)[::2, ::2]
    b = torch.randn((134, 70), device=a.device)[::2, ::2]
    y = flag_gems.mm_w8a8_int8(a, b, out_dtype=torch.float32)
    torch.testing.assert_close(
        y.cpu(), _floating_int8_reference(a, b, torch.float32), rtol=1e-5, atol=1e-4
    )
    with pytest.raises(TypeError, match="FP16, BF16 or FP32"):
        flag_gems.mm_w8a8_int8(a.to(torch.int8), b.to(torch.int8))
    with pytest.raises(TypeError, match="out_dtype"):
        flag_gems.mm_w8a8_int8(a, b, out_dtype=torch.int8)
    with pytest.raises(ValueError, match="out must be contiguous"):
        flag_gems.mm_w8a8_int8_out(a, b, out=torch.empty((1, 1), device=a.device))


@pytest.mark.parametrize("layout", ["row_major", "column_major", "sliced", "broadcast"])
def test_mm_w8a8_int8_floating_packs_weight(layout):
    # All caller layouts must produce K-contiguous quantized weights for AIU.
    a = torch.randn((17, 67), device=flag_gems.device, dtype=torch.bfloat16)
    b = torch.randn((67, 35), device=a.device, dtype=a.dtype)
    if layout == "column_major":
        b = b.t().contiguous().t()
    elif layout == "sliced":
        storage = torch.empty((134, 70), device=a.device, dtype=a.dtype)
        storage[::2, ::2] = b
        b = storage[::2, ::2]
    elif layout == "broadcast":
        b = b[:, :1].expand(67, 35)
    aq, bq, _, _ = _backend._prepare_mm_w8a8_int8_inputs(a, b)
    assert aq.stride() == (67, 1)
    assert bq.stride() == (1, 67)
    expected_bq = _backend._prepare_mm_w8a8_int8_inputs(a, b.contiguous())[1]
    torch.testing.assert_close(bq, expected_bq, rtol=0, atol=0)
    y = flag_gems.mm_w8a8_int8(a, b, out_dtype=torch.float32)
    torch.testing.assert_close(
        y.cpu(), _floating_int8_reference(a, b, torch.float32), rtol=1e-5, atol=1e-4
    )
