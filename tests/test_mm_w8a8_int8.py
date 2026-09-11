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

import itertools
import sys

import pytest
import torch

import flag_gems

if hasattr(flag_gems, "mm_w8a8_int8_out"):
    _backend = sys.modules[flag_gems.mm_w8a8_int8.__module__]
    _mm_w8a8_int8_prequantized = _backend._mm_w8a8_int8_prequantized
    _mm_w8a8_int8_prequantized_out = _backend._mm_w8a8_int8_prequantized_out


pytestmark = [
    pytest.mark.mm_w8a8_int8,
    pytest.mark.skipif(
        not hasattr(flag_gems, "mm_w8a8_int8_out"),
        reason="mm_w8a8_int8 is not implemented by the active backend",
    ),
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
    # backend quantization preparation and GEMM implementation.
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
    # Allow one output-dtype rounding step at a halfway value; retain the
    # strict FP32 tolerance and independent integer reference.
    rtol = {torch.float16: 1e-3, torch.bfloat16: 8e-3, torch.float32: 1e-5}
    torch.testing.assert_close(y.cpu(), ref, rtol=rtol[out_dtype], atol=1e-4)
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
    with pytest.raises(ValueError, match="out must"):
        flag_gems.mm_w8a8_int8_out(a, b, out=torch.empty((1, 1), device=a.device))


@pytest.mark.parametrize("layout", ["row_major", "column_major", "sliced", "broadcast"])
def test_mm_w8a8_int8_floating_packs_weight(layout):
    # All caller layouts must produce K-contiguous quantized weights for the prequantized GEMM.
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


DTYPES = [torch.float16, torch.bfloat16, torch.float32]


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1),
        (3, 17, 33),
        (17, 65, 129),
        (64, 128, 256),
        (128, 128, 1024),
        (2, 7, 2051),
    ],
)
@pytest.mark.parametrize("dtype,out_dtype", list(itertools.product(DTYPES, DTYPES)))
def test_accuracy(shape, dtype, out_dtype):
    m, n, k = shape
    torch.manual_seed(5972)
    a = torch.randn(m, k, device=flag_gems.device, dtype=dtype)
    b = torch.randn(k, n, device=flag_gems.device, dtype=dtype)
    expected = _floating_int8_reference(a, b, out_dtype)
    actual = flag_gems.mm_w8a8_int8(a, b, out_dtype=out_dtype)
    torch.testing.assert_close(
        actual.cpu(),
        expected,
        rtol={torch.float16: 0.001, torch.bfloat16: 0.008, torch.float32: 1e-5}[
            out_dtype
        ],
        atol=1e-4,
    )
    out = torch.empty_like(actual)
    assert flag_gems.mm_w8a8_int8_out(a, b, out=out) is out
    torch.testing.assert_close(out, actual, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["transpose", "slice", "broadcast"])
def test_strides(layout):
    a = torch.randn(17, 66, device=flag_gems.device)
    b = torch.randn(66, 35, device=flag_gems.device)
    if layout == "transpose":
        a = a.t().contiguous().t()
        b = b.t().contiguous().t()
    elif layout == "slice":
        a = a[:, ::2]
        b = b[::2, :]
    else:
        a = a[:1].expand(17, -1)
        b = b[:, :1].expand(-1, 35)
    y = flag_gems.mm_w8a8_int8(a, b)
    torch.testing.assert_close(
        y.cpu(), _floating_int8_reference(a, b, torch.bfloat16), rtol=1e-5, atol=1e-4
    )


@pytest.mark.parametrize("shape", [(0, 5, 3), (3, 0, 5), (3, 5, 0), (0, 0, 0)])
def test_empty(shape):
    m, n, k = shape
    a = torch.empty(m, k, device=flag_gems.device)
    b = torch.empty(k, n, device=flag_gems.device)
    y = flag_gems.mm_w8a8_int8(a, b)
    assert y.shape == (m, n)
    if k == 0:
        assert torch.count_nonzero(y).item() == 0


def test_boundaries_and_updates():
    # Exact half-integers test nearest-even, including negative ties.
    v = torch.tensor(
        [127.0, -127.0, 0.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5], device=flag_gems.device
    )
    a = v.repeat(4, 1)
    b = torch.eye(9, device=flag_gems.device)
    a[1].zero_()
    a[2] *= 1e-12
    for _ in range(2):
        y = flag_gems.mm_w8a8_int8(a, b, out_dtype=torch.float32)
        torch.testing.assert_close(
            y.cpu(), _floating_int8_reference(a, b, torch.float32), rtol=1e-5, atol=1e-6
        )
        a.mul_(-2)
        b.mul_(3)


def test_graph_replay_and_alias():
    a = torch.randn(32, 32, device=flag_gems.device)
    b = torch.randn(32, 32, device=flag_gems.device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            flag_gems.mm_w8a8_int8(a, b)
    torch.cuda.current_stream().wait_stream(stream)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = flag_gems.mm_w8a8_int8(a, b, out_dtype=torch.float32)
    for _ in range(2):
        a.normal_()
        b.normal_()
        g.replay()
        torch.testing.assert_close(
            y.cpu(), _floating_int8_reference(a, b, torch.float32), rtol=1e-5, atol=1e-4
        )
    expected = _floating_int8_reference(a, b, torch.float32)
    assert flag_gems.mm_w8a8_int8_out(a, b, out=a) is a
    torch.testing.assert_close(a.cpu(), expected, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize(
    "case",
    ["dtype", "ndim", "shape", "device", "outdtype", "outshape", "outstride"],
)
def test_invalid(case):
    a = torch.ones(3, 4, device=flag_gems.device)
    b = torch.ones(4, 5, device=flag_gems.device)
    with pytest.raises((TypeError, ValueError)):
        if case == "dtype":
            flag_gems.mm_w8a8_int8(a.int(), b)
        elif case == "ndim":
            flag_gems.mm_w8a8_int8(a[0], b)
        elif case == "shape":
            flag_gems.mm_w8a8_int8(a, b[:3])
        elif case == "device":
            flag_gems.mm_w8a8_int8(a, b.cpu())
        elif case == "outdtype":
            flag_gems.mm_w8a8_int8(a, b, out_dtype=torch.int8)
        elif case == "outshape":
            flag_gems.mm_w8a8_int8_out(
                a, b, out=torch.empty(5, 3, device=flag_gems.device)
            )
        elif case == "outstride":
            flag_gems.mm_w8a8_int8_out(
                a, b, out=torch.empty(5, 3, device=flag_gems.device).t()
            )


@pytest.mark.parametrize("dtype", DTYPES)
def test_quantized_codes(dtype):
    backend = _backend
    x = torch.randn(19, 2051, device=flag_gems.device, dtype=dtype)
    q, _, scale, _ = backend._prepare_mm_w8a8_int8_inputs(x, x.t())
    cpu = x.cpu().float()
    peak = cpu.abs().amax(1).clamp_min(1e-10)
    torch.testing.assert_close(
        q.cpu(), (cpu / peak[:, None] * 127).round().to(torch.int8), rtol=0, atol=0
    )
    torch.testing.assert_close(scale.cpu(), peak * (1 / 127), rtol=0, atol=0)


@pytest.mark.parametrize(
    "adtype,bdtype", [(torch.float16, torch.bfloat16), (torch.float32, torch.float16)]
)
def test_mixed_dtypes(adtype, bdtype):
    a = torch.randn(5, 37, device=flag_gems.device, dtype=adtype)
    b = torch.randn(37, 9, device=flag_gems.device, dtype=bdtype)
    y = flag_gems.mm_w8a8_int8(a, b, out_dtype=torch.float32)
    torch.testing.assert_close(
        y.cpu(), _floating_int8_reference(a, b, torch.float32), rtol=1e-5, atol=1e-4
    )


def test_int32_limit():
    k = 2147483647 // (127 * 127)
    a = torch.ones(1, k, device=flag_gems.device)
    b = torch.ones(k, 1, device=flag_gems.device)
    y = flag_gems.mm_w8a8_int8(a, b, out_dtype=torch.float32)
    torch.testing.assert_close(y, torch.full_like(y, float(k)), rtol=1e-6, atol=0)


@pytest.mark.parametrize(
    "shape",
    [(1, 128, 128), (1, 257, 381), (1, 4096, 4096), (4, 128, 2051), (1025, 129, 257)],
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_optimized_paths(shape, dtype):
    m, n, k = shape
    a = torch.randn(m, k, device=flag_gems.device, dtype=dtype)
    b = torch.randn(k, n, device=flag_gems.device, dtype=dtype)
    out = flag_gems.mm_w8a8_int8(a, b, out_dtype=torch.float32)
    torch.testing.assert_close(
        out.cpu(), _floating_int8_reference(a, b, torch.float32), rtol=1e-5, atol=1e-4
    )


def test_fused_graph_updates_and_shifted_alias():
    a = torch.randn(1, 257, device=flag_gems.device)
    b = torch.randn(257, 257, device=flag_gems.device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            flag_gems.mm_w8a8_int8(a, b, out_dtype=torch.float32)
    torch.cuda.current_stream().wait_stream(stream)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = flag_gems.mm_w8a8_int8(a, b, out_dtype=torch.float32)
    for _ in range(2):
        a.normal_()
        b.normal_()
        g.replay()
        torch.testing.assert_close(
            y.cpu(), _floating_int8_reference(a, b, torch.float32), rtol=1e-5, atol=1e-4
        )
    ref = _floating_int8_reference(a, b, torch.float32)
    out = b.flatten()[5:262].view(1, 257)
    flag_gems.mm_w8a8_int8_out(a, b, out=out)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("dtype", DTYPES)
def test_optimized_quantization_ties(dtype):
    backend = _backend
    values = torch.tensor(
        [127.0, -127.0, 0.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5],
        device=flag_gems.device,
        dtype=dtype,
    )
    x = values.repeat(19, 29)[:, :257].contiguous()
    q, _, scale, _ = backend._prepare_mm_w8a8_int8_inputs(x, x.t())
    cpu = x.cpu().float()
    peak = cpu.abs().amax(1).clamp_min(1e-10)
    ref = (cpu / peak[:, None] * 127).round().to(torch.int8)
    torch.testing.assert_close(q.cpu(), ref, rtol=0, atol=0)
    b = x.t().contiguous()
    _, bq, _, scale = backend._prepare_mm_w8a8_int8_inputs(x, b)
    q = bq.t()
    torch.testing.assert_close(q.cpu(), ref, rtol=0, atol=0)
    torch.testing.assert_close(scale.cpu(), peak * (1.0 / 127.0), rtol=0, atol=0)


@pytest.mark.parametrize("k", [133145, 151936, 152064])
@pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="requires overflow-safe long-K reduction"
)
def test_long_k_overflow(k):
    backend = _backend
    a = torch.ones(1, k, device=flag_gems.device)
    b = torch.ones(k, 1, device=flag_gems.device)
    actual = flag_gems.mm_w8a8_int8(a, b, out_dtype=torch.float32)
    torch.testing.assert_close(
        actual, torch.full_like(actual, float(k)), rtol=1e-6, atol=0
    )
    aq = torch.full((1, k), -128, device=flag_gems.device, dtype=torch.int8)
    bq = torch.full((k, 1), -128, device=flag_gems.device, dtype=torch.int8)
    scale = torch.ones(1, device=flag_gems.device)
    actual = backend._mm_w8a8_int8_prequantized(
        aq, bq, scale, scale, out_dtype=torch.float32
    )
    expected = torch.full_like(actual, float(k * 128 * 128))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(1, 128, 257), (17, 65, 129), (128, 256, 1024)])
def test_prequantized_entry(shape):
    backend = _backend
    m, n, k = shape
    a = torch.randn(m, k, device=flag_gems.device, dtype=torch.bfloat16)
    b = torch.randn(k, n, device=flag_gems.device, dtype=torch.bfloat16)
    prepared = backend._prepare_mm_w8a8_int8_inputs(a, b)
    y = backend._mm_w8a8_int8_prequantized(*prepared, out_dtype=torch.float32)
    torch.testing.assert_close(
        y.cpu(), _floating_int8_reference(a, b, torch.float32), rtol=1e-5, atol=1e-4
    )
    out = torch.empty_like(y)
    assert backend._mm_w8a8_int8_prequantized_out(*prepared, out=out) is out
    torch.testing.assert_close(out, y, rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape",
    [
        (0, 3, 5),
        (3, 0, 5),
        (3, 5, 0),
        (2, 1024, 4096),
        (4, 512, 3584),
        (8, 512, 3584),
        (8, 1024, 4096),
        (98, 2049, 1024),
        (256, 1024, 1024),
    ],
)
def test_prequantized_dispatch(shape):
    backend = _backend
    m, n, k = shape
    a = torch.randint(-128, 128, (m, k), device=flag_gems.device, dtype=torch.int8)
    b = torch.randint(-128, 128, (n, k), device=flag_gems.device, dtype=torch.int8).t()
    sa = torch.ones(m, device=flag_gems.device)
    sb = torch.ones(n, device=flag_gems.device)
    actual = backend._mm_w8a8_int8_prequantized(a, b, sa, sb, out_dtype=torch.float32)
    expected = (a.cpu().long() @ b.cpu().long()).float()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.skipif(
    flag_gems.vendor_name != "hygon", reason="requires overflow-safe long-K reduction"
)
def test_large_tile_long_k_overflow():
    backend = _backend
    m, n, k = 1025, 1025, 131073
    a = torch.full((m, k), -128, device=flag_gems.device, dtype=torch.int8)
    b = torch.full((n, k), -128, device=flag_gems.device, dtype=torch.int8).t()
    sa = torch.full((m,), 0.5, device=flag_gems.device)
    sb = torch.full((n,), -0.25, device=flag_gems.device)
    actual = backend._mm_w8a8_int8_prequantized(a, b, sa, sb, out_dtype=torch.float32)
    expected = torch.full_like(actual, float(k * 128 * 128) * -0.125)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
