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

_HAS_INT8 = (
    flag_gems.vendor_name in ("thead", "metax")
    and hasattr(flag_gems, "mm_w8a8_int8")
    and hasattr(flag_gems, "mm_w8a8_int8_out")
)

if _HAS_INT8:
    _backend = sys.modules[flag_gems.mm_w8a8_int8.__module__]
    _mm_w8a8_int8_prequantized = _backend._mm_w8a8_int8_prequantized
    _mm_w8a8_int8_prequantized_out = _backend._mm_w8a8_int8_prequantized_out


pytestmark = [
    pytest.mark.mm_w8a8_int8,
    pytest.mark.skipif(
        not _HAS_INT8,
        reason="THead/MetaX INT8 backend",
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
    torch.testing.assert_close(
        y.cpu(), (a.cpu().long() @ b.cpu().long()).float(), rtol=0, atol=0
    )


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
    # All caller layouts must produce K-contiguous quantized weights for GEMM.
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


@pytest.mark.parametrize("k", [131071, 131072, 152064, 4194305])
def test_mm_w8a8_int8_accumulation_overflow(k):
    a = torch.full((2, k), -128, device=flag_gems.device, dtype=torch.int8)
    b = torch.full((3, k), -128, device=a.device, dtype=torch.int8).t()
    sa = torch.ones(2, device=a.device)
    sb = torch.ones(3, device=a.device)
    y = _mm_w8a8_int8_prequantized(a, b, sa, sb, out_dtype=torch.float32)
    torch.testing.assert_close(y, torch.full_like(y, float(16384 * k)), rtol=0, atol=0)


def test_mm_w8a8_int8_quantization_ties():
    # Peak 127 makes every half-integer an exact round-to-even boundary.
    vals = torch.tensor(
        [127, -127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5], device=flag_gems.device
    )
    a = vals[None, :].repeat(3, 1)
    b = a.t()
    aq, bq, sa, sb = _backend._prepare_mm_w8a8_int8_inputs(a, b)
    torch.testing.assert_close(aq.cpu(), a.cpu().round().to(torch.int8), rtol=0, atol=0)
    torch.testing.assert_close(bq.cpu(), b.cpu().round().to(torch.int8), rtol=0, atol=0)
    torch.testing.assert_close(sa, torch.ones_like(sa), rtol=0, atol=0)
    torch.testing.assert_close(sb, torch.ones_like(sb), rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape",
    [(384, 384, 384), (1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)],
)
def test_mm_w8a8_int8_core_samples(shape):
    m, n, k = shape
    torch.manual_seed(234)
    a = torch.randn((m, k), device=flag_gems.device, dtype=torch.bfloat16)
    b = torch.randn((k, n), device=a.device, dtype=a.dtype)
    y = flag_gems.mm_w8a8_int8(a, b)
    rows = torch.tensor([0, m // 2, m - 1], device=a.device)
    cols = torch.tensor([0, n // 2, n - 1], device=a.device)
    ref = _floating_int8_reference(a[rows], b[:, cols], torch.bfloat16)
    torch.testing.assert_close(
        y[rows[:, None], cols[None, :]].cpu(), ref, rtol=0, atol=0
    )


def test_mm_w8a8_int8_floating_large_stride():
    if flag_gems.vendor_name != "metax":
        pytest.skip("MetaX 64-bit addressing regression")
    # Only 134 input elements are accessed, but the second row is above the
    # signed 32-bit element-offset boundary. C550 has enough memory for this.
    free, _ = torch.cuda.mem_get_info()
    if free < 6 * (1 << 30):
        pytest.skip("requires 6 GiB free memory for the large-stride allocation")
    stride = (1 << 31) + 16
    a = torch.empty_strided(
        (2, 67), (stride, 1), device=flag_gems.device, dtype=torch.bfloat16
    )
    a[0].fill_(1)
    a[1].fill_(2)
    b = torch.ones((67, 3), device=a.device, dtype=a.dtype)
    b[:, 1].fill_(2)
    b[:, 2].fill_(3)
    actual = flag_gems.mm_w8a8_int8(a, b)
    expected = torch.tensor([[67, 134, 201], [134, 268, 402]], dtype=a.dtype)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("m", [2, 3, 4])
def test_mm_w8a8_int8_small_rows_replay(m):
    a = torch.randn((m, 3584), device=flag_gems.device, dtype=torch.bfloat16)
    b = torch.randn((3584, 512), device=a.device, dtype=a.dtype)
    for _ in range(2):
        flag_gems.mm_w8a8_int8(a, b)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = flag_gems.mm_w8a8_int8(a, b)
    for _ in range(2):
        a.normal_()
        b.normal_()
        graph.replay()
        torch.testing.assert_close(
            y.cpu(), _floating_int8_reference(a, b, a.dtype), rtol=0, atol=0
        )


@pytest.mark.parametrize("k", [2048, 20000])
def test_mm_w8a8_int8_strided_quantization(k):
    a = torch.randn((4, k * 2), device=flag_gems.device, dtype=torch.bfloat16)[:, ::2]
    b = torch.randn((k * 2, 134), device=a.device, dtype=a.dtype)[::2, ::2]
    actual = flag_gems.mm_w8a8_int8(a, b)
    torch.testing.assert_close(
        actual.cpu(), _floating_int8_reference(a, b, a.dtype), rtol=0, atol=0
    )


@pytest.mark.parametrize("k", [128, 4096, 14336, 18944])
@pytest.mark.parametrize("m", [1, 4])
def test_mm_w8a8_int8_prepared_weight_replay(k, m):
    a = torch.randn((m, k), device=flag_gems.device, dtype=torch.bfloat16)
    b = torch.randn((k, 64), device=a.device, dtype=a.dtype)
    _, bq, _, sb = _backend._prepare_mm_w8a8_int8_inputs(a, b)
    out = torch.empty((m, 64), device=a.device, dtype=a.dtype)

    def call():
        return _backend._mm_w8a8_int8_prepared_weight_out(a, bq, sb, out=out)

    for _ in range(2):
        call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(2):
        a.normal_()
        graph.replay()
        torch.testing.assert_close(
            out.cpu(), _floating_int8_reference(a, b, a.dtype), rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "dtype,bits,exponents",
    [
        (torch.bfloat16, 7, [-20, 0, 20]),
        (torch.float16, 10, [-8, 0, 8]),
    ],
)
def test_mm_w8a8_activation_mantissa_boundaries(dtype, bits, exponents):
    mantissa = 1 + torch.arange(1 << bits, dtype=torch.float32) / (1 << bits)
    base = (mantissa[None, :] * (2.0 ** -torch.arange(9).float())[:, None]).flatten()
    base = torch.cat((base, -base))
    for exponent in exponents:
        peaks = mantissa * (2.0**exponent)
        a = (base[None, :] * (2.0**exponent)).expand(len(peaks), -1)
        a = torch.minimum(torch.maximum(a, -peaks[:, None]), peaks[:, None]).to(dtype)
        aq, scale = _backend._prepare_mm_w8a8_int8_activation(a.to(flag_gems.device))
        peak = a.float().abs().amax(1).clamp_min(1e-10)
        expected = (
            torch.round(a.float() / peak[:, None] * 127).clamp(-127, 127).to(torch.int8)
        )
        torch.testing.assert_close(aq.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(scale.cpu(), peak * (1.0 / 127), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("k", [4096, 8192, 8193, 14336, 16384, 18944])
def test_mm_w8a8_activation_extreme_peaks(dtype, k):
    # Include clamped peaks, signed values, and FP32/BF16 fallback ranges.
    factors = [0.0, 1e-20, 1e-10, 1.0, 1000.0]
    if dtype != torch.float16:
        factors += [1e30, 1e35]
    a = torch.linspace(-1, 1, k)[None, :] * torch.tensor(factors)[:, None]
    a = a.to(dtype)
    aq, scale = _backend._prepare_mm_w8a8_int8_activation(a.to(flag_gems.device))
    peak = a.float().abs().amax(1).clamp_min(1e-10)
    expected = (
        torch.round(a.float() / peak[:, None] * 127).clamp(-127, 127).to(torch.int8)
    )
    torch.testing.assert_close(aq.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(scale.cpu(), peak * (1.0 / 127), rtol=0, atol=0)


def test_mm_w8a8_activation_long_row_fallback():
    a = torch.tensor([-1.0, 0.0, 1.0], device=flag_gems.device, dtype=torch.bfloat16)[
        :, None
    ].repeat(1, 1048577)
    aq, scale = _backend._prepare_mm_w8a8_int8_activation(a)
    expected = torch.tensor([-127, 0, 127], device=a.device, dtype=torch.int8)[:, None]
    assert torch.all(aq == expected)
    peak = torch.tensor([1.0, 1e-10, 1.0], device=a.device)
    torch.testing.assert_close(scale, peak * (1.0 / 127), rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape",
    [(4101, 4103, 1025), (193, 512, 1024), (4, 6145, 4096), (256, 13569, 1025)],
)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_mm_w8a8_profiled_tiles_edges(shape, out_dtype):
    m, n, k = shape
    a = torch.randint(-127, 128, (m, k), device=flag_gems.device, dtype=torch.int8)
    b = torch.randint(-127, 128, (n, k), device=a.device, dtype=torch.int8).t()
    sa = torch.rand(m, device=a.device) * 0.001
    sb = torch.rand(n, device=a.device) * 0.001
    out = torch.empty((m, n), device=a.device, dtype=out_dtype)
    _backend._mm_w8a8_int8_prequantized_out(a, b, sa, sb, out=out)
    rows = sorted(set([0, min(127, m - 1), min(128, m - 1), min(255, m - 1), m - 1]))
    cols = sorted(set([0, min(127, n - 1), min(255, n - 1), min(256, n - 1), n - 1]))
    ref = (
        (a[rows].cpu().long() @ b[:, cols].cpu().long()).float()
        * sa[rows].cpu()[:, None]
        * sb[cols].cpu()[None, :]
    ).to(out_dtype)
    torch.testing.assert_close(out[rows][:, cols].cpu(), ref, rtol=0, atol=0)


@pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX weight layout")
@pytest.mark.parametrize("shape,dtype", [
    ((4096, 2048, 1024), torch.bfloat16),
    ((4101, 2048, 1024), torch.bfloat16),
    ((4096, 4608, 3584), torch.bfloat16),
    ((64, 2048, 1024), torch.bfloat16),
    ((64, 2048, 1024), torch.float16),
    ((64, 2048, 1024), torch.float32),
    ((37, 269, 77), torch.bfloat16),
])
@pytest.mark.parametrize("single_shared", [False, True])
def test_mm_w8a8_packed_weight(shape, dtype, single_shared):
    if single_shared:
        from triton.backends.metax.compiler import MACAOptions
        if "single_shared_async" not in MACAOptions.__dataclass_fields__:
            pytest.skip("requires the experimental FlagTree direct-copy pipeline")
    m, n, k = shape
    aq = torch.randint(-128, 128, (m, k), device=flag_gems.device, dtype=torch.int8)
    bq = torch.randint(-128, 128, (n, k), device=aq.device, dtype=torch.int8).t()
    sa = torch.rand(m, device=aq.device) * 0.001
    sb = torch.rand(n, device=aq.device) * 0.001
    weight = _backend._pack_mm_w8a8_int8_weight(bq, sb, single_shared=single_shared)
    if (n, k) == (4608, 3584):
        assert weight.tiled is None
    expected = torch.empty((m, n), device=aq.device, dtype=dtype)
    _backend._mm_w8a8_int8_prequantized_out(aq, bq, sa, sb, out=expected)
    # Preparation owns a snapshot: later source changes cannot corrupt it.
    bq.zero_()
    sb.zero_()
    out = torch.empty_like(expected)
    _backend._mm_w8a8_int8_packed_prequantized_out(aq, sa, weight, out=out)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    rr, cc = [0, m - 1], [0, n - 1]
    ref = ((aq[rr].cpu().long() @ weight.standard[:, cc].cpu().long()).float()
           * sa[rr].cpu()[:, None] * weight.scale[cc].cpu()[None, :]).to(dtype)
    torch.testing.assert_close(out[rr][:, cc].cpu(), ref, rtol=0, atol=0)


@pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX weight layout")
@pytest.mark.parametrize("single_shared", [False, True])
def test_mm_w8a8_packed_graph_updates(single_shared):
    if single_shared:
        from triton.backends.metax.compiler import MACAOptions
        if "single_shared_async" not in MACAOptions.__dataclass_fields__:
            pytest.skip("requires the experimental FlagTree direct-copy pipeline")
    a = torch.randn((4096, 1024), device=flag_gems.device, dtype=torch.bfloat16)
    bq = torch.randint(-127, 128, (2048, 1024), device=a.device, dtype=torch.int8).t()
    sb = torch.full((2048,), 0.001, device=a.device)
    weight = _backend._pack_mm_w8a8_int8_weight(bq, sb, single_shared=single_shared)
    out = torch.empty((4096, 2048), device=a.device, dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _backend._mm_w8a8_int8_packed_weight_out(a, weight, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _backend._mm_w8a8_int8_packed_weight_out(a, weight, out=out)
    expected = torch.empty_like(out)
    for value in (0.0, 0.5):
        a.fill_(value)
        graph.replay()
        _backend._mm_w8a8_int8_prepared_weight_out(a, bq, sb, out=expected)
        torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX weight layout")
def test_mm_w8a8_packed_weight_rejects_invalid():
    b = torch.zeros((128, 256), device=flag_gems.device, dtype=torch.int8)
    scale = torch.ones(256, device=b.device)
    with pytest.raises(ValueError):
        _backend._pack_mm_w8a8_int8_weight(b.float(), scale)
    with pytest.raises(ValueError):
        _backend._pack_mm_w8a8_int8_weight(b, scale[:128])


@pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX compiler option")
def test_mm_w8a8_packed_single_shared_requires_compiler(monkeypatch):
    from triton.backends.metax.compiler import MACAOptions
    fields = dict(MACAOptions.__dataclass_fields__)
    fields.pop("single_shared_async", None)
    monkeypatch.setattr(MACAOptions, "__dataclass_fields__", fields)
    b = torch.zeros((128, 256), device=flag_gems.device, dtype=torch.int8)
    scale = torch.ones(256, device=b.device)
    with pytest.raises(RuntimeError, match="single_shared_async"):
        _backend._pack_mm_w8a8_int8_weight(b, scale, single_shared=True)
    with pytest.raises(TypeError, match="must be bool"):
        _backend._pack_mm_w8a8_int8_weight(b, scale, single_shared=1)
