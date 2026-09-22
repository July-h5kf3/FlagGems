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

from .accuracy_utils import gems_assert_equal

pytestmark = pytest.mark.mm_w8a8_int8


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


@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="thead only")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.mm_w8a8_int8
def test_mm_w8a8_int8(shape, dtype):
    m, n, k = shape
    a = torch.randint(-128, 128, (m, k), device=flag_gems.device, dtype=torch.int8)
    b = torch.randint(-128, 128, (n, k), device=flag_gems.device, dtype=torch.int8).t()
    sa = torch.rand(m, device=a.device) * 0.01
    sb = torch.rand(n, device=a.device) * 0.01
    y = flag_gems.mm_w8a8_int8(a, b, sa, sb, out_dtype=dtype)
    _assert_reference(a, b, sa, sb, y)
    out = torch.empty_like(y)
    assert flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out) is out
    torch.testing.assert_close(out, y, rtol=0, atol=0)


@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="thead only")
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
    y = flag_gems.mm_w8a8_int8(a, b, sa, sb, out_dtype=torch.float32)
    torch.testing.assert_close(y, a.float() @ b.float(), rtol=0, atol=0)


@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="thead only")
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
            return flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out)
        return flag_gems.mm_w8a8_int8(a, b, sa, sb, out_dtype=torch.float32)

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


@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="thead only")
@pytest.mark.parametrize("shape", [(2, 3, 0), (0, 3, 8), (2, 0, 8), (0, 0, 0)])
@pytest.mark.mm_w8a8_int8
def test_mm_w8a8_int8_empty(shape):
    m, n, k = shape
    a = torch.empty((m, k), device=flag_gems.device, dtype=torch.int8)
    b = torch.empty((k, n), device=a.device, dtype=torch.int8)
    sa = torch.ones(m, device=a.device)
    sb = torch.ones(n, device=a.device)
    y = flag_gems.mm_w8a8_int8(a, b, sa, sb)
    torch.testing.assert_close(y, torch.zeros_like(y))


@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="thead only")
@pytest.mark.mm_w8a8_int8
def test_mm_w8a8_int8_reject_float():
    a = torch.ones((2, 3), device=flag_gems.device)
    b = torch.ones((3, 4), device=a.device)
    with pytest.raises(TypeError, match="prequantized"):
        flag_gems.mm_w8a8_int8(
            a, b, torch.ones(2, device=a.device), torch.ones(4, device=a.device)
        )


def _assert_reference(a, b, sa, sb, out, bias=None):
    # Check all entries of small cases and a deterministic subset of large
    # cases against an independent CPU INT64 product, not a floating GEMM.
    m, k = a.shape
    n = b.shape[1]
    rows = list(range(m)) if m * n * k <= 1000000 else sorted({0, m // 2, m - 1})
    cols = list(range(n)) if m * n * k <= 1000000 else sorted({0, n // 2, n - 1})
    aa = a[rows].cpu().to(torch.int64)
    bb = b[:, cols].cpu().to(torch.int64)
    sa = sa.cpu().reshape(-1)
    sb = sb.cpu().reshape(-1)
    sa = sa.expand(m) if sa.numel() == 1 else sa
    sb = sb.expand(n) if sb.numel() == 1 else sb
    ref = (aa @ bb).float() * sa[rows, None] * sb[None, cols]
    if bias is not None:
        ref += bias.cpu()[cols].float()[None, :]
    torch.testing.assert_close(
        out[rows][:, cols].cpu(),
        ref.to(out.dtype),
        rtol=1e-5 if out.dtype == torch.float32 else 1e-2,
        atol=1e-4,
    )


@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="thead only")
@pytest.mark.parametrize(
    "shape",
    [
        (1, 17, 32),
        (17, 1, 128),
        (3, 35, 128),
        (17, 35, 67),
        (64, 128, 128),
        (1, 32769, 2048),
        (256, 16384, 2048),
        (1025, 65, 2048),
    ],
)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("scales", ["tensor", "axis", "mixed_a", "mixed_b"])
def test_scaled_bias(shape, out_dtype, scales):
    m, n, k = shape
    a = torch.randint(-128, 128, (m, k), device=flag_gems.device, dtype=torch.int8)
    b = torch.randint(-128, 128, (n, k), device=a.device, dtype=torch.int8).t()
    sa = torch.rand(1 if scales in ("tensor", "mixed_a") else m, device=a.device) * 0.01
    sb = torch.rand(1 if scales in ("tensor", "mixed_b") else n, device=a.device) * 0.02
    bias = torch.randn(n, device=a.device, dtype=out_dtype)
    # Exercise the vLLM-style positional output dtype and bias arguments.
    y = flag_gems.mm_w8a8_int8(a, b, sa, sb, out_dtype, bias)
    _assert_reference(a, b, sa, sb, y, bias)
    out = torch.empty_like(y)
    assert flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out, bias=bias) is out
    torch.testing.assert_close(out, y, rtol=0, atol=0)


@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="thead only")
@pytest.mark.parametrize("shape", [(2, 3, 0), (0, 3, 8), (2, 0, 8), (0, 0, 0)])
def test_scaled_empty_bias(shape):
    m, n, k = shape
    a = torch.empty(m, k, device=flag_gems.device, dtype=torch.int8)
    b = torch.empty(k, n, device=a.device, dtype=torch.int8)
    scale = torch.ones(1, device=a.device)
    bias = torch.randn(n, device=a.device, dtype=torch.bfloat16)
    y = flag_gems.mm_w8a8_int8(a, b, scale, scale, bias=bias)
    assert y.dtype == torch.bfloat16
    torch.testing.assert_close(y, bias[None, :].expand(m, n))


@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="thead only")
@pytest.mark.parametrize(
    "shape", [(3, 17, 128), (1, 32, 128), (32, 1, 128), (17, 35, 67), (1025, 65, 2048)]
)
@pytest.mark.parametrize("scalar", [False, True])
def test_scaled_graph_bias_updates(shape, scalar):
    m, n, k = shape
    a = torch.ones(m, k, device=flag_gems.device, dtype=torch.int8)
    b = torch.ones(n, k, device=a.device, dtype=torch.int8).t()
    sa = torch.ones(1 if scalar else m, device=a.device)
    sb = torch.ones(1 if scalar else n, device=a.device)
    bias = torch.zeros(n, device=a.device)
    out = torch.empty(m, n, device=a.device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out, bias=bias)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out, bias=bias)
    for av, bv, sav, sbv, bv_bias in [
        (1, 1, 1, 1, 0),
        (2, -3, 0.5, 0.25, 2),
        (0, 2, 2, 1, -3),
    ]:
        a.fill_(av)
        b.fill_(bv)
        sa.fill_(sav)
        sb.fill_(sbv)
        bias.fill_(bv_bias)
        graph.replay()
        torch.testing.assert_close(
            out, torch.full_like(out, k * av * bv * sav * sbv + bv_bias)
        )


@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="thead only")
@pytest.mark.parametrize(
    "invalid",
    [
        "a_dtype",
        "b_dtype",
        "scale_dtype",
        "scale_shape",
        "scale_stride",
        "scale_device",
        "bias_shape",
        "bias_dtype",
        "bias_device",
        "bias_stride",
        "output_dtype",
        "output_shape",
        "output_stride",
    ],
)
def test_scaled_validation(invalid):
    a = torch.ones(2, 3, device=flag_gems.device, dtype=torch.int8)
    b = torch.ones(3, 4, device=a.device, dtype=torch.int8)
    sa = torch.ones(2, device=a.device)
    sb = torch.ones(4, device=a.device)
    bias = torch.ones(4, device=a.device, dtype=torch.bfloat16)
    out = torch.empty(2, 4, device=a.device, dtype=torch.bfloat16)
    if invalid == "a_dtype":
        a = a.float()
    elif invalid == "b_dtype":
        b = b.float()
    elif invalid == "scale_dtype":
        sa = sa.half()
    elif invalid == "scale_shape":
        sa = torch.ones(2, 2, device=a.device)
    elif invalid == "scale_stride":
        sb = torch.ones(8, device=a.device)[::2]
    elif invalid == "scale_device":
        sa = sa.cpu()
    elif invalid == "bias_shape":
        bias = bias[None, :]
    elif invalid == "bias_dtype":
        bias = bias.float()
    elif invalid == "bias_device":
        bias = bias.cpu()
    elif invalid == "bias_stride":
        bias = torch.ones(8, device=a.device, dtype=out.dtype)[::2]
    elif invalid == "output_dtype":
        out = out.to(torch.int8)
    elif invalid == "output_shape":
        out = out[:1]
    elif invalid == "output_stride":
        out = torch.empty(2, 8, device=a.device, dtype=out.dtype)[:, ::2]
    with pytest.raises((TypeError, ValueError)):
        flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out, bias=bias)


@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="thead only")
@pytest.mark.parametrize(
    "shape", [(1, 17, 32), (3, 35, 128), (17, 35, 67), (3, 17, 33001)]
)
def test_tensor_scales_without_bias(shape):
    m, n, k = shape
    a = torch.randint(-128, 128, (m, k), device=flag_gems.device, dtype=torch.int8)
    b = torch.randint(-128, 128, (n, k), device=a.device, dtype=torch.int8).t()
    sa = torch.tensor(0.0125, device=a.device)
    sb = torch.tensor([[0.025]], device=a.device)
    y = flag_gems.mm_w8a8_int8(a, b, sa, sb)
    assert y.dtype == torch.bfloat16
    _assert_reference(a, b, sa, sb, y)


@pytest.mark.skipif(flag_gems.vendor_name != "thead", reason="thead only")
def test_scaled_unaligned_weight():
    m, n, k = 2, 16, 128
    a = torch.randint(-128, 128, (m, k), device=flag_gems.device, dtype=torch.int8)
    storage = torch.randint(-128, 128, (n * k + 1,), device=a.device, dtype=torch.int8)
    b = storage[1:].view(n, k).t()
    sa = torch.rand(m, device=a.device)
    sb = torch.rand(n, device=a.device)
    y = flag_gems.mm_w8a8_int8(a, b, sa, sb, torch.float32)
    _assert_reference(a, b, sa, sb, y)


def inputs(m, n, k, scalar=False, layout=False):
    a = torch.randint(-128, 128, (m, k), device=flag_gems.device, dtype=torch.int8)
    b = torch.randint(-128, 128, (n, k), device=flag_gems.device, dtype=torch.int8).t()
    if layout:
        a = a.t().contiguous().t()
        b = b.contiguous()
    sa = torch.rand((1,) if scalar else (m, 1), device=flag_gems.device) * 0.01
    sb = torch.rand((1,) if scalar else (1, n), device=flag_gems.device) * 0.01
    return a, b, sa, sb


def reference(a, b, sa, sb, bias=None, dtype=torch.float32):
    value = (a.cpu().long() @ b.cpu().long()).float()
    value = value * sa.cpu().reshape(-1, 1) * sb.cpu().reshape(1, -1)
    if bias is not None:
        value += bias.cpu().float()
    return value.to(dtype)


@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="hygon only")
@pytest.mark.parametrize(
    "shape",
    [
        (0, 4, 8),
        (3, 0, 8),
        (3, 4, 0),
        (1, 1, 1),
        (7, 13, 19),
        (1, 17, 8192),
        (2, 31, 2048),
        (4, 513, 1024),
        (3, 1025, 1024),
        (4, 1024, 4097),
        (98, 2048, 1024),
        (129, 17, 513),
        (2, 3, 131073),
        (64, 1, 65536),
        (2, 1, 262145),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "scalar,bias_on,layout",
    [(False, False, False), (False, True, True), (True, True, False)],
)
@pytest.mark.mm_w8a8_int8
def test_prequantized(shape, dtype, scalar, bias_on, layout):
    a, b, sa, sb = inputs(*shape, scalar=scalar, layout=layout)
    bias = (
        torch.randn(shape[1], device=flag_gems.device, dtype=dtype) if bias_on else None
    )
    expected = reference(a, b, sa, sb, bias, dtype)
    actual = flag_gems.mm_w8a8_int8(a, b, sa, sb, dtype, bias)
    gems_assert_equal(actual.cpu(), expected)
    out = torch.empty_like(actual)
    assert flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out, bias=bias) is out
    gems_assert_equal(out.cpu(), expected)


@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="hygon only")
@pytest.mark.parametrize("code", [-128, 127])
def test_long_k_overflow(code):
    a, b, sa, sb = inputs(2, 3, 262145)
    a.fill_(code)
    b.fill_(code)
    sa.fill_(1)
    sb.fill_(1)
    y = flag_gems.mm_w8a8_int8(a, b, sa, sb, torch.float32)
    gems_assert_equal(y.cpu(), reference(a, b, sa, sb))


@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="hygon only")
@pytest.mark.parametrize("a_scalar,b_scalar", [(True, False), (False, True)])
def test_mixed_scales(a_scalar, b_scalar):
    a, b, sa, sb = inputs(17, 13, 31)
    sa = sa[:1] if a_scalar else sa.flatten()
    sb = sb[:, :1].contiguous() if b_scalar else sb.flatten()
    y = flag_gems.mm_w8a8_int8(a, b, sa, sb)
    assert y.dtype == torch.bfloat16
    gems_assert_equal(y.cpu(), reference(a, b, sa, sb, dtype=torch.bfloat16))


@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="hygon only")
def test_graph_updates():
    a, b, sa, sb = inputs(17, 13, 31)
    bias = torch.randn(13, device=flag_gems.device)
    out = torch.empty((17, 13), device=flag_gems.device)
    for _ in range(3):
        flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out, bias=bias)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out, bias=bias)
    a.fill_(-128)
    b.fill_(127)
    sa.mul_(2)
    sb.mul_(3)
    bias.add_(1)
    graph.replay()
    gems_assert_equal(out.cpu(), reference(a, b, sa, sb, bias))


@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="hygon only")
@pytest.mark.parametrize(
    "bad",
    [
        "a_dtype",
        "b_dtype",
        "shape",
        "sa_shape",
        "sb_shape",
        "scale_dtype",
        "scale_stride",
        "bias_shape",
        "bias_dtype",
        "bias_device",
        "out_dtype",
        "out_shape",
        "out_stride",
        "alias",
    ],
)
def test_invalid(bad):
    a, b, sa, sb = inputs(3, 5, 7)
    bias = None
    out = torch.empty((3, 5), device=flag_gems.device)
    if bad == "a_dtype":
        a = a.float()
    elif bad == "b_dtype":
        b = b.float()
    elif bad == "shape":
        b = b[:2]
    elif bad == "sa_shape":
        sa = sa[:2]
    elif bad == "sb_shape":
        sb = sb[:, :2]
    elif bad == "scale_dtype":
        sa = sa.half()
    elif bad == "scale_stride":
        sa = torch.ones(6, device=flag_gems.device)[::2]
    elif bad == "bias_shape":
        bias = torch.ones(4, device=flag_gems.device)
    elif bad == "bias_dtype":
        bias = torch.ones(5, device=flag_gems.device, dtype=torch.int8)
    elif bad == "bias_device":
        bias = torch.ones(5)
    elif bad == "out_dtype":
        out = out.to(torch.int8)
    elif bad == "out_shape":
        out = out[:2]
    elif bad == "out_stride":
        out = torch.empty((5, 3), device=flag_gems.device).t()
    elif bad == "alias":
        sb = out[0]
    with pytest.raises((ValueError, TypeError)):
        flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out, bias=bias)


if flag_gems.vendor_name == "metax" and hasattr(flag_gems, "mm_w8a8_int8"):
    _backend = sys.modules[flag_gems.mm_w8a8_int8.__module__]
    _mm_w8a8_int8_prequantized = _backend._mm_w8a8_int8_prequantized
    _mm_w8a8_int8_prequantized_out = _backend._mm_w8a8_int8_prequantized_out
    _floating_mm = getattr(_backend, "_mm_w8a8_int8_floating", flag_gems.mm_w8a8_int8)
    _floating_mm_out = getattr(
        _backend, "_mm_w8a8_int8_floating_out", flag_gems.mm_w8a8_int8_out
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
    @pytest.mark.parametrize(
        "out_dtype", [torch.float16, torch.bfloat16, torch.float32]
    )
    @pytest.mark.parametrize("column_major", [False, True])
    @pytest.mark.parametrize("shape", [(1, 17, 32), (17, 35, 67), (64, 128, 128)])
    def test_mm_w8a8_int8_floating(dtype, out_dtype, column_major, shape):
        m, n, k = shape
        torch.manual_seed(42)
        a = torch.randn((m, k), device=flag_gems.device, dtype=dtype)
        b = torch.randn(
            (n, k) if column_major else (k, n), device=a.device, dtype=dtype
        )
        if column_major:
            b = b.t()
        # Include all-zero rows/columns in scale handling.
        a[0] = 0
        b[:, 0] = 0
        ref = _floating_int8_reference(a, b, out_dtype)
        y = _floating_mm(a, b, out_dtype=out_dtype)
        torch.testing.assert_close(y.cpu(), ref, rtol=1e-5, atol=1e-4)
        out = torch.empty_like(y)
        assert _floating_mm_out(a, b, out=out) is out
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
                return _floating_mm_out(a, b, out=out)
            return _floating_mm(a, b, out_dtype=torch.float32)

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
        y = _floating_mm(a, b)
        assert y.dtype == torch.bfloat16
        torch.testing.assert_close(y, torch.zeros_like(y))
        out = torch.empty((m, n), device=a.device)
        assert _floating_mm_out(a, b, out=out) is out
        torch.testing.assert_close(out, torch.zeros_like(out))

    def test_mm_w8a8_int8_floating_strides_and_validation():
        a = torch.randn((34, 134), device=flag_gems.device)[::2, ::2]
        b = torch.randn((134, 70), device=a.device)[::2, ::2]
        y = _floating_mm(a, b, out_dtype=torch.float32)
        torch.testing.assert_close(
            y.cpu(), _floating_int8_reference(a, b, torch.float32), rtol=1e-5, atol=1e-4
        )
        with pytest.raises(TypeError, match="FP16, BF16 or FP32"):
            _floating_mm(a.to(torch.int8), b.to(torch.int8))
        with pytest.raises(TypeError, match="out_dtype"):
            _floating_mm(a, b, out_dtype=torch.int8)
        with pytest.raises(ValueError, match="out must be contiguous"):
            _floating_mm_out(a, b, out=torch.empty((1, 1), device=a.device))

    @pytest.mark.parametrize(
        "layout", ["row_major", "column_major", "sliced", "broadcast"]
    )
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
        y = _floating_mm(a, b, out_dtype=torch.float32)
        torch.testing.assert_close(
            y.cpu(), _floating_int8_reference(a, b, torch.float32), rtol=1e-5, atol=1e-4
        )

    @pytest.mark.parametrize("k", [131071, 131072, 152064, 4194305])
    @pytest.mark.parametrize("m,n", [(1, 1), (1, 3), (2, 3)])
    def test_mm_w8a8_int8_accumulation_overflow(k, m, n):
        a = torch.full((m, k), -128, device=flag_gems.device, dtype=torch.int8)
        b = torch.full((n, k), -128, device=a.device, dtype=torch.int8).t()
        sa = torch.ones(m, device=a.device)
        sb = torch.ones(n, device=a.device)
        y = _mm_w8a8_int8_prequantized(a, b, sa, sb, out_dtype=torch.float32)
        torch.testing.assert_close(
            y, torch.full_like(y, float(16384 * k)), rtol=0, atol=0
        )

    def test_mm_w8a8_int8_quantization_ties():
        # Peak 127 makes every half-integer an exact round-to-even boundary.
        vals = torch.tensor(
            [127, -127, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5], device=flag_gems.device
        )
        a = vals[None, :].repeat(3, 1)
        b = a.t()
        aq, bq, sa, sb = _backend._prepare_mm_w8a8_int8_inputs(a, b)
        torch.testing.assert_close(
            aq.cpu(), a.cpu().round().to(torch.int8), rtol=0, atol=0
        )
        torch.testing.assert_close(
            bq.cpu(), b.cpu().round().to(torch.int8), rtol=0, atol=0
        )
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
        y = _floating_mm(a, b)
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
        actual = _floating_mm(a, b)
        expected = torch.tensor([[67, 134, 201], [134, 268, 402]], dtype=a.dtype)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    @pytest.mark.parametrize("m", [2, 3, 4])
    def test_mm_w8a8_int8_small_rows_replay(m):
        a = torch.randn((m, 3584), device=flag_gems.device, dtype=torch.bfloat16)
        b = torch.randn((3584, 512), device=a.device, dtype=a.dtype)
        for _ in range(2):
            _floating_mm(a, b)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            y = _floating_mm(a, b)
        for _ in range(2):
            a.normal_()
            b.normal_()
            graph.replay()
            torch.testing.assert_close(
                y.cpu(), _floating_int8_reference(a, b, a.dtype), rtol=0, atol=0
            )

    @pytest.mark.parametrize("k", [2048, 20000])
    def test_mm_w8a8_int8_strided_quantization(k):
        a = torch.randn((4, k * 2), device=flag_gems.device, dtype=torch.bfloat16)[
            :, ::2
        ]
        b = torch.randn((k * 2, 134), device=a.device, dtype=a.dtype)[::2, ::2]
        actual = _floating_mm(a, b)
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
        base = (
            mantissa[None, :] * (2.0 ** -torch.arange(9).float())[:, None]
        ).flatten()
        base = torch.cat((base, -base))
        for exponent in exponents:
            peaks = mantissa * (2.0**exponent)
            a = (base[None, :] * (2.0**exponent)).expand(len(peaks), -1)
            a = torch.minimum(torch.maximum(a, -peaks[:, None]), peaks[:, None]).to(
                dtype
            )
            aq, scale = _backend._prepare_mm_w8a8_int8_activation(
                a.to(flag_gems.device)
            )
            peak = a.float().abs().amax(1).clamp_min(1e-10)
            expected = (
                torch.round(a.float() / peak[:, None] * 127)
                .clamp(-127, 127)
                .to(torch.int8)
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
        a = torch.tensor(
            [-1.0, 0.0, 1.0], device=flag_gems.device, dtype=torch.bfloat16
        )[:, None].repeat(1, 1048577)
        aq, scale = _backend._prepare_mm_w8a8_int8_activation(a)
        expected = torch.tensor([-127, 0, 127], device=a.device, dtype=torch.int8)[
            :, None
        ]
        assert torch.all(aq == expected)
        peak = torch.tensor([1.0, 1e-10, 1.0], device=a.device)
        torch.testing.assert_close(scale, peak * (1.0 / 127), rtol=0, atol=0)

    @pytest.mark.parametrize(
        "shape",
        [(4101, 4103, 1025), (193, 512, 1024), (4, 6145, 4096), (256, 13569, 1025)],
    )
    @pytest.mark.parametrize(
        "out_dtype", [torch.bfloat16, torch.float16, torch.float32]
    )
    def test_mm_w8a8_profiled_tiles_edges(shape, out_dtype):
        m, n, k = shape
        a = torch.randint(-127, 128, (m, k), device=flag_gems.device, dtype=torch.int8)
        b = torch.randint(-127, 128, (n, k), device=a.device, dtype=torch.int8).t()
        sa = torch.rand(m, device=a.device) * 0.001
        sb = torch.rand(n, device=a.device) * 0.001
        out = torch.empty((m, n), device=a.device, dtype=out_dtype)
        _backend._mm_w8a8_int8_prequantized_out(a, b, sa, sb, out=out)
        rows = sorted(
            set([0, min(127, m - 1), min(128, m - 1), min(255, m - 1), m - 1])
        )
        cols = sorted(
            set([0, min(127, n - 1), min(255, n - 1), min(256, n - 1), n - 1])
        )
        ref = (
            (a[rows].cpu().long() @ b[:, cols].cpu().long()).float()
            * sa[rows].cpu()[:, None]
            * sb[cols].cpu()[None, :]
        ).to(out_dtype)
        torch.testing.assert_close(out[rows][:, cols].cpu(), ref, rtol=0, atol=0)

    @pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX weight layout")
    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((4096, 2048, 1024), torch.bfloat16),
            ((4101, 2048, 1024), torch.bfloat16),
            ((4096, 4608, 3584), torch.bfloat16),
            ((64, 2048, 1024), torch.bfloat16),
            ((64, 2048, 1024), torch.float16),
            ((64, 2048, 1024), torch.float32),
            ((37, 269, 77), torch.bfloat16),
        ],
    )
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
        ref = (
            (aq[rr].cpu().long() @ weight.standard[:, cc].cpu().long()).float()
            * sa[rr].cpu()[:, None]
            * weight.scale[cc].cpu()[None, :]
        ).to(dtype)
        torch.testing.assert_close(out[rr][:, cc].cpu(), ref, rtol=0, atol=0)

    @pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX weight layout")
    @pytest.mark.parametrize("single_shared", [False, True])
    def test_mm_w8a8_packed_graph_updates(single_shared):
        if single_shared:
            from triton.backends.metax.compiler import MACAOptions

            if "single_shared_async" not in MACAOptions.__dataclass_fields__:
                pytest.skip("requires the experimental FlagTree direct-copy pipeline")
        a = torch.randn((4096, 1024), device=flag_gems.device, dtype=torch.bfloat16)
        bq = torch.randint(
            -127, 128, (2048, 1024), device=a.device, dtype=torch.int8
        ).t()
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

    @pytest.mark.skipif(
        flag_gems.vendor_name != "metax", reason="MetaX compiler option"
    )
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

    @pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX scaled-mm API")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "shape,n",
        [
            ((32,), 48),
            ((2, 3, 32), 48),
            ((17, 33), 48),
            ((0, 16), 48),
            ((3, 0), 48),
            ((2, 3584), 512),
            ((1, 2, 3584), 512),
        ],
    )
    @pytest.mark.parametrize("scalar", [False, True])
    @pytest.mark.parametrize("with_bias", [False, True])
    def test_vllm_scaled_mm_api(dtype, shape, n, scalar, with_bias):
        import math

        m, k = math.prod(shape[:-1]), shape[-1]
        a = torch.randint(-128, 128, shape, device=flag_gems.device, dtype=torch.int8)
        b = torch.randint(-128, 128, (n, k), device=a.device, dtype=torch.int8).t()
        if n == 512 and scalar:
            a = torch.randint(
                -128, 128, (*shape[:-1], 2 * k), device=a.device, dtype=torch.int8
            )[..., ::2]
            b = torch.randint(-128, 128, (n, 2 * k), device=a.device, dtype=torch.int8)[
                :, ::2
            ].t()
        sa = torch.full((1,) if scalar else (m, 1), 0.001, device=a.device)
        sb = torch.full((1,) if scalar else (n, 1), 0.002, device=a.device)
        bias = torch.randn(n, device=a.device, dtype=dtype) if with_bias else None
        ref = (a.cpu().reshape(m, k).long() @ b.cpu().long()).float()
        ref = ref * sa.cpu().reshape(-1, 1) * sb.cpu().reshape(1, -1)
        if bias is not None:
            ref += bias.cpu().float()
        ref = ref.to(dtype).reshape(*shape[:-1], n)
        y = flag_gems.mm_w8a8_int8(a, b, sa, sb, dtype, bias)
        torch.testing.assert_close(y.cpu(), ref, rtol=0, atol=0)
        out = torch.empty_like(y)
        assert flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out, bias=bias) is out
        torch.testing.assert_close(out, y, rtol=0, atol=0)

    @pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX scaled-mm API")
    @pytest.mark.parametrize("with_bias", [False, True])
    def test_vllm_scaled_mm_replay(with_bias):
        a = torch.ones((2, 3, 32), device=flag_gems.device, dtype=torch.int8)
        b = torch.ones((48, 32), device=a.device, dtype=torch.int8).t()
        sa = torch.ones(1, device=a.device)
        sb = torch.ones((48, 1), device=a.device)
        bias = (
            torch.zeros(48, device=a.device, dtype=torch.bfloat16)
            if with_bias
            else None
        )
        out = torch.empty((2, 3, 48), device=a.device, dtype=torch.bfloat16)

        def call():
            return flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out, bias=bias)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                call()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        for av, bv, sv in [(0, 1, 1), (-128, -128, 0.001), (127, -128, 0.002)]:
            a.fill_(av)
            b.fill_(bv)
            sa.fill_(sv)
            sb.fill_(0.5)
            if bias is not None:
                bias.fill_(0.25)
            graph.replay()
            ref = torch.tensor(32 * av * bv, dtype=torch.float32) * sa.cpu()[0] * 0.5
            if with_bias:
                ref += 0.25
            torch.testing.assert_close(
                out.cpu(), ref.bfloat16().expand_as(out.cpu()), rtol=0, atol=0
            )

    @pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX scaled-mm API")
    def test_vllm_scaled_mm_validation():
        a = torch.ones((3, 32), device=flag_gems.device, dtype=torch.int8)
        b = torch.ones((32, 48), device=a.device, dtype=torch.int8)
        sa, sb = torch.ones(3, device=a.device), torch.ones(48, device=a.device)
        with pytest.raises(TypeError):
            flag_gems.mm_w8a8_int8(a.float(), b, sa, sb, torch.bfloat16)
        with pytest.raises(TypeError):
            flag_gems.mm_w8a8_int8(a, b, sa, sb, torch.float32)
        with pytest.raises(ValueError):
            flag_gems.mm_w8a8_int8(a, b, sa[:2], sb, torch.bfloat16)
        with pytest.raises(ValueError):
            flag_gems.mm_w8a8_int8(
                a, b, sa, sb, torch.bfloat16, torch.zeros(48, device=a.device)
            )
        with pytest.raises(ValueError):
            flag_gems.mm_w8a8_int8_out(
                a,
                b,
                sa,
                sb,
                out=torch.empty((48, 3), device=a.device, dtype=torch.bfloat16),
            )

    @pytest.mark.skipif(
        flag_gems.vendor_name != "metax", reason="MetaX split reduction"
    )
    @pytest.mark.parametrize(
        "shape",
        [
            (95, 1024, 1025),
            (96, 1024, 1025),
            (97, 1024, 1025),
            (98, 4096, 14336),
            (256, 512, 3585),
            (1024, 1024, 131073),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_mm_w8a8_large_split_reduction(shape, dtype):
        m, n, k = shape
        a = torch.randint(-128, 128, (m, k), device=flag_gems.device, dtype=torch.int8)
        b = torch.randint(-128, 128, (n, k), device=a.device, dtype=torch.int8).t()
        if k > 131071:
            # The sum across INT32 partials exceeds signed INT32.
            a.fill_(-128)
            b.fill_(-128)
        sa = torch.full((m, 1), 0.001, device=a.device)
        sb = torch.full((n, 1), 0.001, device=a.device)
        out = torch.empty((m, n), dtype=dtype, device=a.device)
        if dtype == torch.float32:
            _backend._mm_w8a8_int8_prequantized_out(a, b, sa, sb.t(), out=out)
        else:
            flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out)
        rr, cc = [0, m // 2, m - 1], [0, n // 2, n - 1]
        ref = (
            (a[rr].cpu().long() @ b[:, cc].cpu().long()).float()
            * sa[rr].cpu()
            * sb[cc].cpu().t()
        ).to(dtype)
        torch.testing.assert_close(out[rr][:, cc].cpu(), ref, rtol=0, atol=0)

    @pytest.mark.skipif(
        flag_gems.vendor_name != "metax", reason="MetaX split reduction"
    )
    def test_mm_w8a8_large_split_replay():
        m, n, k = 98, 1024, 4096
        a = torch.ones((m, k), device=flag_gems.device, dtype=torch.int8)
        b = torch.ones((n, k), device=a.device, dtype=torch.int8).t()
        sa = torch.full((m, 1), 0.001, device=a.device)
        sb = torch.full((n, 1), 0.001, device=a.device)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)

        def call():
            return flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                call()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        for av, bv, scale in [(0, 1, 0.001), (-128, -128, 0.001), (127, -128, 0.002)]:
            a.fill_(av)
            b.fill_(bv)
            sa.fill_(scale)
            graph.replay()
            ref = (
                torch.tensor(k * av * bv).float()
                * torch.tensor(scale)
                * torch.tensor(0.001)
            ).bfloat16()
            torch.testing.assert_close(out.cpu(), ref.expand(m, n), rtol=0, atol=0)

    @pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX small-M tiles")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize(
        "shape",
        [
            (2, 8192, 1024),
            (2, 8193, 1023),
            (2, 8193, 1024),
            (16, 8193, 4096),
            (16, 8193, 4097),
        ],
    )
    def test_mm_w8a8_small_m_warp_boundary(shape, dtype):
        m, n, k = shape
        a = torch.randint(-128, 128, (m, k), device=flag_gems.device, dtype=torch.int8)
        b = torch.randint(-128, 128, (n, k), device=a.device, dtype=torch.int8).t()
        sa = torch.rand((m, 1), device=a.device) * 0.001
        sb = torch.rand((n, 1), device=a.device) * 0.001
        out = flag_gems.mm_w8a8_int8(a, b, sa, sb, dtype)
        rr, cc = [0, m // 2, m - 1], [0, n // 2, n - 1]
        ref = (
            (a[rr].cpu().long() @ b[:, cc].cpu().long()).float()
            * sa[rr].cpu()
            * sb[cc].cpu().t()
        ).to(dtype)
        torch.testing.assert_close(out[rr][:, cc].cpu(), ref, rtol=0, atol=0)

    @pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX TLE")
    @pytest.mark.parametrize(
        "shape",
        [
            (3, 2048, 1024),
            (98, 2048, 1024),
            (256, 2048, 1024),
            (1024, 1024, 1024),
            (8192, 512, 1024),
        ],
    )
    @pytest.mark.parametrize("extreme", [False, True])
    def test_metax_tle_scaled_mm(shape, extreme, monkeypatch):
        m, n, k = shape
        a = torch.randint(-128, 128, (m, k), device=flag_gems.device, dtype=torch.int8)
        b = torch.randint(-128, 128, (n, k), device=a.device, dtype=torch.int8).t()
        if extreme:
            a.fill_(-128)
            b.fill_(-128)
        sa = torch.randn(m, 1, device=a.device) * 0.017
        sb = torch.randn(n, 1, device=a.device) * 0.023
        sa[0] = 0
        sb[0] = 0
        storage = torch.full((m * n + 64,), 123, device=a.device, dtype=torch.bfloat16)
        out = storage[: m * n].view(m, n)
        flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out)
        rows = torch.tensor(sorted({0, m // 2, m - 1}), device=a.device)
        cols = torch.arange(0, n, max(1, n // 31), device=a.device)
        ref = (a[rows].cpu().long() @ b[:, cols].cpu().long()).float()
        ref = (ref * sa[rows].cpu() * sb[cols].cpu().t()).bfloat16()
        torch.testing.assert_close(out[rows][:, cols].cpu(), ref, rtol=0, atol=0)
        out.zero_()
        flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out)
        torch.testing.assert_close(out[rows][:, cols].cpu(), ref, rtol=0, atol=0)
        assert (storage[m * n :] == 123).all()

    @pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX TLE")
    @pytest.mark.parametrize("m", [3, 98])
    def test_metax_tle_graph_updates(m, monkeypatch):
        n, k = 2048, 1024
        a = torch.ones((m, k), device=flag_gems.device, dtype=torch.int8)
        b = torch.ones((n, k), device=a.device, dtype=torch.int8).t()
        sa = torch.ones(m, 1, device=a.device)
        sb = torch.ones(n, 1, device=a.device)
        out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out)
            with torch.cuda.graph(graph := torch.cuda.CUDAGraph(), stream=stream):
                flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out)
            for av, bv, sav, sbv in [
                (0, 1, 1.0, 1.0),
                (-128, -128, 0.001, -0.017),
                (127, -128, 0.003, 0.002),
            ]:
                a.fill_(av)
                b.fill_(bv)
                sa.fill_(sav)
                sb.fill_(sbv)
                graph.replay()
                stream.synchronize()
                ref = (
                    torch.tensor(float(k * av * bv))
                    * torch.tensor(sav)
                    * torch.tensor(sbv)
                ).bfloat16()
                torch.testing.assert_close(out.cpu(), ref.expand(m, n), rtol=0, atol=0)
        torch.cuda.current_stream().wait_stream(stream)

    @pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX split grid")
    @pytest.mark.parametrize("m", [3, 4, 8, 16])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("offset", [0, 1, 16, 128, 512])
    def test_metax_tle_split_grid(m, dtype, offset):
        n, k = 4096, 4096
        a = torch.randint(-128, 128, (m, k), device=flag_gems.device, dtype=torch.int8)
        weights = torch.randint(
            -128, 128, (n * k + offset,), device=a.device, dtype=torch.int8
        )
        b = weights[offset:].view(n, k).t()
        sa = torch.randn(m, 1, device=a.device) * 0.001
        sb = torch.randn(n, 1, device=a.device) * 0.002
        bias = torch.randn(n, device=a.device, dtype=dtype)
        storage = torch.full((m * n + 64,), 123, device=a.device, dtype=dtype)
        out = storage[: m * n].view(m, n)
        for current_bias in (None, bias):
            flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out, bias=current_bias)
            ref = (a.cpu().long() @ b.cpu().long()).float() * sa.cpu() * sb.cpu().t()
            if current_bias is not None:
                ref += bias.cpu().float()
            torch.testing.assert_close(out.cpu(), ref.to(dtype), rtol=0, atol=0)
            assert (storage[m * n :] == 123).all()
        with torch.cuda.graph(graph := torch.cuda.CUDAGraph()):
            flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out)
        for av, bv, sav, sbv in [(0, 1, 0.001, 0.002), (-128, -128, 0.001, -0.002)]:
            a.fill_(av)
            b.fill_(bv)
            sa.fill_(sav)
            sb.fill_(sbv)
            graph.replay()
            ref = (
                torch.tensor(float(k * av * bv)) * torch.tensor(sav) * torch.tensor(sbv)
            ).to(dtype)
            torch.testing.assert_close(out.cpu(), ref.expand(m, n), rtol=0, atol=0)
            assert (storage[m * n :] == 123).all()

    @pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX streaming TLE")
    @pytest.mark.parametrize(
        "shape",
        [
            (64, 14336, 4096),
            (65, 16384, 4096),
            (98, 18944, 3584),
            (98, 28672, 4096),
            (127, 14336, 4096),
            (128, 14336, 4096),
            (128, 18944, 3584),
            (128, 28672, 4096),
            (256, 14336, 4096),
            (256, 18944, 3584),
            (256, 28672, 4096),
            (256, 4096, 14336),
            (8192, 1024, 4096),
            (256, 512, 3584),
            pytest.param((2048, 2048, 2048), id="stream_cube_2048"),
            pytest.param((64, 3584, 18944), id="long_k_64_3584_18944"),
            pytest.param((64, 4096, 14336), id="long_k_64_4096_14336"),
            pytest.param((98, 3584, 18944), id="long_k_98_3584_18944"),
            pytest.param((98, 4096, 14336), id="long_k_98_4096_14336"),
            pytest.param((128, 3584, 18944), id="long_k_128_3584_18944"),
            pytest.param((128, 4096, 14336), id="long_k_128_4096_14336"),
            pytest.param((256, 3584, 18944), id="long_k_256_3584_18944"),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_metax_tle_stream_shared(shape, dtype):
        if not getattr(_backend, "_STREAM_SHARED_MMA_AVAILABLE", False):
            pytest.skip("FlagTree compiler lacks stream_shared_mma")
        _check_metax_tle_graph(shape, dtype, 128)

    def _check_metax_tle_graph(shape, dtype, b_offset, a_offset=0, check_small=None):
        m, n, k = shape
        # Use a CPU integer oracle; the vendor FP32 matmul is not bit-exact.
        storage_a = torch.randint(
            -128, 128, (m * k + a_offset,), device=flag_gems.device, dtype=torch.int8
        )
        a = storage_a[a_offset:].view(m, k)
        storage_b = torch.randint(
            -128, 128, (n * k + b_offset,), device=a.device, dtype=torch.int8
        )
        b = storage_b[b_offset:].view(n, k).t()
        if (m, n, k) == (256, 37888, 3584) and hasattr(
            _backend, "_pick_aligned_stream"
        ):
            selected = _backend._pick_aligned_stream(a, b, m, n, k, dtype)
            expected = (
                _backend._STREAM_SHARED_MMA_AVAILABLE
                and a.data_ptr() % 16 == 0
                and b.data_ptr() % 16 == 0
            )
            assert (selected is not None) == expected
        if check_small is not None:
            selected = _backend._pick_small_stream(a, b, m, n, k, dtype)
            assert (selected is not None) == check_small
        sa = torch.randn(m, 1, device=a.device) * 0.001
        sb = torch.randn(n, 1, device=a.device) * 0.002
        storage = torch.full((m * n + 64,), 123, device=a.device, dtype=dtype)
        out = storage[: m * n].view(m, n)
        flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out)
        # Every INT8 product and intermediate integer sum is exactly representable
        # in FP64 under this bound. The separate INT64 sample below independently
        # checks this full-matrix reference.
        assert k * 128**2 < 2**53
        ac, bc = a.cpu(), b.cpu()
        integer = (ac.double() @ bc.double()).to(torch.int64)
        ref = (integer.float() * sa.cpu() * sb.cpu().t()).to(dtype)
        torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)
        rows = torch.tensor([0, m // 2, m - 1], device=a.device)
        cols = torch.arange(0, n, max(1, n // 31), device=a.device)
        cpu_ref = (a[rows].cpu().long() @ b[:, cols].cpu().long()).float()
        cpu_ref = (cpu_ref * sa[rows].cpu() * sb[cols].cpu().t()).to(dtype)
        torch.testing.assert_close(out[rows][:, cols].cpu(), cpu_ref, rtol=0, atol=0)
        assert (storage[m * n :] == 123).all()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out)
            with torch.cuda.graph(graph := torch.cuda.CUDAGraph(), stream=stream):
                flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out)
            for av, bv, sav, sbv in [
                (0, 1, 0.001, 0.002),
                (-128, -128, 0.001, -0.002),
                (127, -128, -0.003, 0.001),
            ]:
                a.fill_(av)
                b.fill_(bv)
                sa.fill_(sav)
                sb.fill_(sbv)
                graph.replay()
                stream.synchronize()
                value = (
                    torch.tensor(float(k * av * bv))
                    * torch.tensor(sav)
                    * torch.tensor(sbv)
                ).to(dtype)
                torch.testing.assert_close(
                    out.cpu(), value.expand(m, n), rtol=0, atol=0
                )
                assert (storage[m * n :] == 123).all()
        torch.cuda.current_stream().wait_stream(stream)

    @pytest.mark.skipif(
        flag_gems.vendor_name != "metax", reason="MetaX narrow reduction"
    )
    @pytest.mark.parametrize("m", [64, 98, 128])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_metax_tle_narrow_reduce(m, dtype):
        n, k = 512, 3584
        a = torch.randint(-128, 128, (m, k), device=flag_gems.device, dtype=torch.int8)
        b = torch.randint(-128, 128, (n, k), device=a.device, dtype=torch.int8).t()
        sa = torch.randn(m, 1, device=a.device) * 0.001
        sb = torch.randn(n, 1, device=a.device) * 0.002
        storage = torch.full((m * n + 64,), 123, device=a.device, dtype=dtype)
        out = storage[: m * n].view(m, n)
        flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out)
        ref = ((a.cpu().long() @ b.cpu().long()).float() * sa.cpu() * sb.cpu().t()).to(
            dtype
        )
        torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)
        assert (storage[m * n :] == 123).all()
        with torch.cuda.graph(graph := torch.cuda.CUDAGraph()):
            flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out)
        for av, bv in [(0, 127), (-128, -128), (127, -128)]:
            a.fill_(av)
            b.fill_(bv)
            sa.fill_(-0.003)
            sb.fill_(0.002)
            graph.replay()
            torch.cuda.synchronize()
            ref = (
                torch.tensor(float(k * av * bv))
                * torch.tensor(-0.003)
                * torch.tensor(0.002)
            ).to(dtype)
            torch.testing.assert_close(out.cpu(), ref.expand(m, n), rtol=0, atol=0)
            assert (storage[m * n :] == 123).all()

    @pytest.mark.skipif(
        flag_gems.vendor_name != "metax", reason="MetaX wide streaming TLE"
    )
    @pytest.mark.parametrize("k", [128256, 151936, 152064])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_metax_tle_wide_stream(k, dtype):
        if not getattr(_backend, "_STREAM_SHARED_MMA_WIDE_AVAILABLE", False):
            pytest.skip("FlagTree compiler lacks 256-deep streaming")
        m, n = 1848, 1536
        a = torch.randint(-128, 128, (m, k), device=flag_gems.device, dtype=torch.int8)
        bs = torch.randint(-128, 128, (n * k + 128,), device=a.device, dtype=torch.int8)
        b = bs[128:].view(n, k).t()
        sa = torch.randn(m, 1, device=a.device) * 0.001
        sb = torch.randn(n, 1, device=a.device) * 0.002
        storage = torch.full((m * n + 64,), 123, device=a.device, dtype=dtype)
        out = storage[: m * n].view(m, n)
        flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out)
        # INT8 products and every partial sum are exact in CPU FP64 because
        # K*128**2 < 2**53. This gives a full integer oracle without slow INT64 GEMM.
        assert k * 128**2 < 2**53
        ac, bc = a.cpu(), b.cpu().double()
        integer = torch.empty((m, n), dtype=torch.int64)
        for row in range(0, m, 128):
            integer[row : row + 128] = (ac[row : row + 128].double() @ bc).to(
                torch.int64
            )
        ref = (integer.float() * sa.cpu() * sb.cpu().t()).to(dtype)
        torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)
        rows = torch.tensor([0, m // 2, m - 1])
        cols = torch.arange(0, n, 47)
        exact = ac[rows].long() @ b[:, cols.to(a.device)].cpu().long()
        torch.testing.assert_close(integer[rows][:, cols], exact, rtol=0, atol=0)
        assert (storage[m * n :] == 123).all()
        del bc, integer, ref
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph := torch.cuda.CUDAGraph(), stream=stream):
                flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out)
            for av, bv in [(0, 127), (-128, -128), (127, -128)]:
                a.fill_(av)
                b.fill_(bv)
                sa.fill_(-0.003)
                sb.fill_(0.002)
                graph.replay()
                stream.synchronize()
                ref = (
                    torch.tensor(float(k * av * bv))
                    * torch.tensor(-0.003)
                    * torch.tensor(0.002)
                ).to(dtype)
                torch.testing.assert_close(out.cpu(), ref.expand(m, n), rtol=0, atol=0)
                assert (storage[m * n :] == 123).all()
        torch.cuda.current_stream().wait_stream(stream)

    @pytest.mark.skipif(
        flag_gems.vendor_name != "metax", reason="MetaX TLE grid ordering"
    )
    @pytest.mark.parametrize(
        "shape",
        [
            (m, n, k)
            for m in (65, 98, 128)
            for n, k in ((3584, 3584), (4096, 4096), (4608, 3584), (6144, 4096))
        ],
    )
    @pytest.mark.parametrize("b_offset", [0, 128])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_metax_tle_grid_order(shape, b_offset, dtype):
        _check_metax_tle_graph(shape, dtype, b_offset)

    @pytest.mark.skipif(
        flag_gems.vendor_name != "metax", reason="MetaX small streaming TLE"
    )
    @pytest.mark.parametrize(
        "shape",
        [(m, n, k) for m in (64, 65, 98, 128) for n, k in ((3584, 3584), (4096, 4096))]
        + [(256, n, k) for n, k in ((3584, 3584), (4096, 4096), (4608, 3584))],
    )
    @pytest.mark.parametrize("offsets", [(0, 0), (0, 16), (0, 128), (16, 0), (128, 0)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_metax_tle_small_stream(shape, offsets, dtype):
        if not getattr(_backend, "_STREAM_SHARED_MMA_SMALL_AVAILABLE", False):
            pytest.skip("FlagTree compiler lacks stream_shared_mma_mn")
        a_offset, b_offset = offsets
        expected = a_offset % 128 == 0 and b_offset % 128 == 0
        _check_metax_tle_graph(shape, dtype, b_offset, a_offset, expected)

    @pytest.mark.skipif(
        flag_gems.vendor_name != "metax", reason="MetaX aligned streaming TLE"
    )
    @pytest.mark.parametrize(
        "offsets",
        [(0, 0), (0, 16), (0, 128), (0, 512), (16, 0), (128, 0), (0, 1), (1, 0)],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_metax_tle_aligned_stream(offsets, dtype):
        if not getattr(_backend, "_STREAM_SHARED_MMA_AVAILABLE", False):
            pytest.skip("FlagTree compiler lacks stream_shared_mma")
        a_offset, b_offset = offsets
        _check_metax_tle_graph((256, 37888, 3584), dtype, b_offset, a_offset)

    @pytest.mark.skipif(
        flag_gems.vendor_name != "metax", reason="MetaX streaming split order"
    )
    @pytest.mark.parametrize("shape", [(98, 14336, 4096), (98, 4096, 14336)])
    @pytest.mark.parametrize("offsets", [(0, 0), (0, 16), (0, 128), (16, 0)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_metax_tle_stream_split_group(shape, offsets, dtype):
        if not getattr(_backend, "_STREAM_SHARED_MMA_AVAILABLE", False):
            pytest.skip("FlagTree compiler lacks streaming")
        m, n, k = shape
        ao, bo = offsets
        a = torch.randint(
            -128, 128, (m * k + ao,), device=flag_gems.device, dtype=torch.int8
        )[ao:].view(m, k)
        b = (
            torch.randint(-128, 128, (n * k + bo,), device=a.device, dtype=torch.int8)[
                bo:
            ]
            .view(n, k)
            .t()
        )
        sa = torch.randn(m, 1, device=a.device) * 0.001
        sb = torch.randn(n, 1, device=a.device) * 0.002
        storage = torch.full((m * n + 64,), 123, device=a.device, dtype=dtype)
        out = storage[32:-32].view(m, n)
        assert _backend._pick_stream_split_group(*shape) == (56 if n == 14336 else 16)
        flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out)
        # INT8 products and their full sums are exact in CPU FP64 in this range.
        assert k * 128**2 < 2**53
        ac, bc = a.cpu(), b.cpu()
        integer = (ac.double() @ bc.double()).to(torch.int64)
        rows, cols = [0, m // 2, m - 1], [0, n // 2, n - 1]
        torch.testing.assert_close(
            integer[rows][:, cols], ac[rows].long() @ bc[:, cols].long(), rtol=0, atol=0
        )
        ref = (integer.float() * sa.cpu() * sb.cpu().t()).to(dtype)
        torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)
        assert (storage[:32] == 123).all() and (storage[-32:] == 123).all()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph := torch.cuda.CUDAGraph(), stream=stream):
                flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out)
            for av, bv, sav, sbv in [
                (0, 127, 0.001, 0.002),
                (-128, -128, -0.003, 0.002),
                (127, -128, 0.001, -0.002),
                (-128, 127, 0.0, 0.002),
            ]:
                a.fill_(av)
                b.fill_(bv)
                sa.fill_(sav)
                sb.fill_(sbv)
                graph.replay()
                stream.synchronize()
                ref = (
                    torch.tensor(float(k * av * bv))
                    * torch.tensor(sav)
                    * torch.tensor(sbv)
                ).to(dtype)
                torch.testing.assert_close(out.cpu(), ref.expand(m, n), rtol=0, atol=0)
                assert (storage[:32] == 123).all() and (storage[-32:] == 123).all()
        torch.cuda.current_stream().wait_stream(stream)

    def _check_metax_zero_pad(shape, offsets, dtype):
        if not getattr(_backend, "_STREAM_SHARED_MMA_AVAILABLE", False):
            pytest.skip("FlagTree compiler lacks streaming")
        m, n, k = shape
        ao, bo = offsets
        a = torch.randint(
            -128, 128, (m * k + ao,), device=flag_gems.device, dtype=torch.int8
        )[ao:].view(m, k)
        b = (
            torch.randint(-128, 128, (n * k + bo,), device=a.device, dtype=torch.int8)[
                bo:
            ]
            .view(n, k)
            .t()
        )
        sa = torch.randn(m, 1, device=a.device) * 0.001
        sb = torch.randn(n, 1, device=a.device) * 0.002
        storage = torch.full((m * n + 64,), 123, device=a.device, dtype=dtype)
        out = storage[32:-32].view(m, n)
        assert _backend._pick_zero_pad_m(a, b, m, n, k, dtype) == (
            64 <= m < 128 and ao % 16 == 0 and bo % 16 == 0
        )
        flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out)
        # INT8 products and their full sums are exact in CPU FP64 in this range.
        assert k * 128**2 < 2**53
        ac, bc = a.cpu(), b.cpu()
        integer = (ac.double() @ bc.double()).to(torch.int64)
        rows, cols = [0, m // 2, m - 1], [0, n // 2, n - 1]
        torch.testing.assert_close(
            integer[rows][:, cols], ac[rows].long() @ bc[:, cols].long(), rtol=0, atol=0
        )
        ref = (integer.float() * sa.cpu() * sb.cpu().t()).to(dtype)
        torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)
        assert (storage[:32] == 123).all() and (storage[-32:] == 123).all()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph := torch.cuda.CUDAGraph(), stream=stream):
                flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out)
            for av, bv, sav, sbv in [
                (0, 127, 0.001, 0.002),
                (-128, -128, -0.003, 0.002),
                (127, -128, 0.001, -0.002),
                (-128, 127, 0.0, 0.002),
            ]:
                a.fill_(av)
                b.fill_(bv)
                sa.fill_(sav)
                sb.fill_(sbv)
                graph.replay()
                stream.synchronize()
                ref = (
                    torch.tensor(float(k * av * bv))
                    * torch.tensor(sav)
                    * torch.tensor(sbv)
                ).to(dtype)
                torch.testing.assert_close(out.cpu(), ref.expand(m, n), rtol=0, atol=0)
                assert (storage[:32] == 123).all() and (storage[-32:] == 123).all()
        torch.cuda.current_stream().wait_stream(stream)

    @pytest.mark.skipif(
        flag_gems.vendor_name != "metax", reason="MetaX zero-row padding"
    )
    @pytest.mark.parametrize(
        "shape",
        [(m, 37888, 3584) for m in (64, 65, 98, 127, 128)]
        + [(98, 128256, 4096), (98, 152064, 3584)],
    )
    @pytest.mark.parametrize("offsets", [(0, 0), (16, 128)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_metax_tle_zero_pad(shape, offsets, dtype):
        _check_metax_zero_pad(shape, offsets, dtype)

    @pytest.mark.skipif(
        flag_gems.vendor_name != "metax", reason="MetaX zero-row padding alignment"
    )
    @pytest.mark.parametrize(
        "shape", [(98, 37888, 3584), (98, 128256, 4096), (98, 152064, 3584)]
    )
    @pytest.mark.parametrize("offsets", [(1, 0), (0, 1)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_metax_tle_zero_pad_unaligned(shape, offsets, dtype):
        _check_metax_zero_pad(shape, offsets, dtype)

    def _check_metax_reverse(shape, offsets, dtype):
        if not getattr(_backend, "_STREAM_SHARED_MMA_AVAILABLE", False):
            pytest.skip("FlagTree compiler lacks streaming")
        m, n, k = shape
        ao, bo = offsets
        a = torch.randint(
            -128, 128, (m * k + ao,), device=flag_gems.device, dtype=torch.int8
        )[ao:].view(m, k)
        b = (
            torch.randint(-128, 128, (n * k + bo,), device=a.device, dtype=torch.int8)[
                bo:
            ]
            .view(n, k)
            .t()
        )
        sa = torch.randn(m, 1, device=a.device) * 0.001
        sb = torch.randn(n, 1, device=a.device) * 0.002
        storage = torch.full((m * n + 64,), 123, device=a.device, dtype=dtype)
        out = storage[32:-32].view(m, n)
        flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out)
        # INT8 products and their full sums are exact in CPU FP64 in this range.
        assert k * 128**2 < 2**53
        ac, bc = a.cpu(), b.cpu()
        integer = (ac.double() @ bc.double()).to(torch.int64)
        rows, cols = [0, m // 2, m - 1], [0, n // 2, n - 1]
        torch.testing.assert_close(
            integer[rows][:, cols], ac[rows].long() @ bc[:, cols].long(), rtol=0, atol=0
        )
        ref = (integer.float() * sa.cpu() * sb.cpu().t()).to(dtype)
        torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)
        assert (storage[:32] == 123).all() and (storage[-32:] == 123).all()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph := torch.cuda.CUDAGraph(), stream=stream):
                flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out)
            for av, bv, sav, sbv in [
                (0, 127, 0.001, 0.002),
                (-128, -128, -0.003, 0.002),
                (127, -128, 0.001, -0.002),
                (-128, 127, 0.0, 0.002),
            ]:
                a.fill_(av)
                b.fill_(bv)
                sa.fill_(sav)
                sb.fill_(sbv)
                graph.replay()
                stream.synchronize()
                ref = (
                    torch.tensor(float(k * av * bv))
                    * torch.tensor(sav)
                    * torch.tensor(sbv)
                ).to(dtype)
                torch.testing.assert_close(out.cpu(), ref.expand(m, n), rtol=0, atol=0)
                assert (storage[:32] == 123).all() and (storage[-32:] == 123).all()
        torch.cuda.current_stream().wait_stream(stream)

    @pytest.mark.skipif(
        flag_gems.vendor_name != "metax", reason="MetaX reverse MMA chains"
    )
    @pytest.mark.parametrize(
        "shape",
        [(m, 14336, 4096) for m in (64, 65, 98, 127, 128, 256)]
        + [
            (98, 4096, 14336),
            (256, 3584, 18944),
            (2048, 2048, 2048),
            (8192, 1024, 4096),
        ],
    )
    @pytest.mark.parametrize("offsets", [(0, 0), (16, 128), (1, 0), (0, 1)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_metax_tle_reverse_chains(shape, offsets, dtype):
        if not getattr(_backend, "_STREAM_SHARED_MMA_REVERSE_AVAILABLE", False):
            pytest.skip("FlagTree compiler lacks reverse MMA chains")
        _check_metax_reverse(shape, offsets, dtype)

    @pytest.mark.skipif(
        flag_gems.vendor_name != "metax", reason="MetaX streaming tail peeling"
    )
    @pytest.mark.parametrize(
        "shape",
        [(m, 14336, 4096) for m in (64, 65, 98, 127, 128, 256)]
        + [
            (98, 16384, 4096),
            (98, 18944, 3584),
            (98, 28672, 4096),
            (98, 3584, 18944),
            (98, 4096, 14336),
            (256, 18944, 3584),
            (256, 28672, 4096),
            (256, 37888, 3584),
            (256, 3584, 18944),
            (256, 4096, 14336),
            (2048, 2048, 2048),
        ],
    )
    @pytest.mark.parametrize("offsets", [(0, 0), (16, 128)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_metax_tle_tail_peel(shape, offsets, dtype):
        if not getattr(_backend, "_STREAM_SHARED_MMA_PEEL_AVAILABLE", False):
            pytest.skip("FlagTree compiler lacks streaming tail peeling")
        _check_metax_reverse(shape, offsets, dtype)

    @pytest.mark.skipif(
        flag_gems.vendor_name != "metax", reason="MetaX small streaming tail peeling"
    )
    @pytest.mark.parametrize(
        "shape",
        [
            (m, n, k)
            for m in (64, 65, 98, 127, 128)
            for n, k in ((3584, 3584), (4096, 4096))
        ]
        + [(256, 3584, 3584), (256, 4096, 4096), (256, 4608, 3584)],
    )
    @pytest.mark.parametrize("offsets", [(0, 0), (128, 128), (16, 0), (0, 16)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_metax_tle_small_tail_peel(shape, offsets, dtype):
        if not getattr(_backend, "_STREAM_SHARED_MMA_SMALL_PEEL_AVAILABLE", False):
            pytest.skip("FlagTree compiler lacks small streaming tail peeling")
        _check_metax_reverse(shape, offsets, dtype)

    @pytest.mark.skipif(
        flag_gems.vendor_name != "metax", reason="MetaX wide streaming tail peeling"
    )
    @pytest.mark.parametrize(
        "shape", [(1848, 1536, k) for k in (128256, 151936, 152064)]
    )
    @pytest.mark.parametrize("offsets", [(0, 0), (16, 128)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_metax_tle_wide_tail_peel(shape, offsets, dtype, monkeypatch):
        if not getattr(_backend, "_STREAM_SHARED_MMA_WIDE_PEEL_AVAILABLE", False):
            pytest.skip("FlagTree compiler lacks wide streaming tail peeling")
        kernel = _backend._mm_w8a8_kernel
        observed = []

        class CheckLaunch:
            def __getitem__(self, grid):
                def run(*args, **kwargs):
                    result = kernel[grid](*args, **kwargs)
                    if kwargs.get("stream_shared_mma_tile") == 256:
                        expected = (
                            args[0].data_ptr() % 128 == 0
                            and args[1].data_ptr() % 128 == 0
                        )
                        compiled = result[0]
                        assert kwargs["PEEL_WIDE_TAIL"] == expected
                        assert compiled.metadata.stream_shared_mma_wide_peel == expected
                        observed.append(expected)
                    return result

                return run

        monkeypatch.setattr(_backend, "_mm_w8a8_kernel", CheckLaunch())
        _check_metax_reverse(shape, offsets, dtype)
        assert observed

    @pytest.mark.skipif(
        flag_gems.vendor_name != "metax", reason="MetaX small-M long-K scheduling"
    )
    @pytest.mark.parametrize("m", range(2, 17))
    @pytest.mark.parametrize("offsets", [(0, 0), (16, 128), (1, 0), (0, 1)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_metax_tle_small_long_unprefetch(m, offsets, dtype, monkeypatch):
        kernel = _backend._mm_w8a8_kernel
        observed = []

        class CheckLaunch:
            def __getitem__(self, grid):
                def run(*args, **kwargs):
                    result = kernel[grid](*args, **kwargs)
                    assert result[0].metadata.scenario == "unprefetch"
                    observed.append(True)
                    return result

                return run

        monkeypatch.setattr(_backend, "_mm_w8a8_kernel", CheckLaunch())
        _check_metax_reverse((m, 4096, 14336), offsets, dtype)
        assert observed

    @pytest.mark.skipif(
        flag_gems.vendor_name != "metax", reason="MetaX small-M masked loads"
    )
    @pytest.mark.parametrize("m", [2, 3, 4, 5, 6, 7, 8, 9, 16])
    @pytest.mark.parametrize("nk", [(3584, 3584), (4608, 3584), (6144, 4096)])
    @pytest.mark.parametrize("offsets", [(0, 0), (16, 128), (1, 0), (0, 1)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_metax_tle_small_masked_loads(m, nk, offsets, dtype, monkeypatch):
        kernel = _backend._mm_w8a8_kernel
        observed = []

        class CheckLaunch:
            def __getitem__(self, grid):
                def run(*args, **kwargs):
                    expected_mask = (2 <= m <= 6 and nk == (3584, 3584)) or (
                        3 <= m <= 8 and nk == (6144, 4096)
                    )
                    if (
                        nk == (3584, 3584)
                        and offsets[0] % 128 == 0
                        and offsets[1] % 128 == 0
                    ):
                        if m == 2:
                            expected_mask = False
                    assert kwargs["MASK_M"] == expected_mask
                    observed.append(True)
                    return kernel[grid](*args, **kwargs)

                return run

        monkeypatch.setattr(_backend, "_mm_w8a8_kernel", CheckLaunch())
        _check_metax_reverse((m, *nk), offsets, dtype)
        assert observed

    @pytest.mark.skipif(
        flag_gems.vendor_name != "metax", reason="MetaX small-N streaming"
    )
    @pytest.mark.parametrize(
        "shape",
        [(m, 1024, 4096) for m in (63, 64, 65, 98, 127, 128, 129, 255, 256, 257)]
        + [(m, 512, 3584) for m in (63, 64, 65, 98, 127, 128, 129)]
        + [(256, 512, 3584)],
    )
    @pytest.mark.parametrize("offsets", [(0, 0), (128, 128), (16, 128), (1, 0)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_metax_tle_small_n_stream(shape, offsets, dtype, monkeypatch):
        kernel = _backend._mm_w8a8_kernel
        seen = []
        m, n, k = shape
        eligible = (
            ((64 <= m <= 128 or m == 256) and (n, k) == (1024, 4096))
            or shape == (256, 512, 3584)
            or (64 <= m <= 128 and (n, k) == (512, 3584))
        )

        class CheckLaunch:
            def __getitem__(self, grid):
                def run(*args, **kwargs):
                    expected = (
                        eligible
                        and args[0].data_ptr() % 128 == 0
                        and args[1].data_ptr() % 128 == 0
                    )
                    result = kernel[grid](*args, **kwargs)
                    assert kwargs["SINGLE_SHARED"] == expected
                    if expected:
                        assert args[8:11] == (64, 64, 128)
                        assert args[12] == (
                            7 if 64 <= m <= 128 and (n, k) == (512, 3584) else 4
                        )
                        assert result[0].metadata.stream_shared_mma_small_peel
                    seen.append(True)
                    return result

                return run

        monkeypatch.setattr(_backend, "_mm_w8a8_kernel", CheckLaunch())
        _check_metax_reverse(shape, offsets, dtype)
        assert seen

    @pytest.mark.skipif(
        flag_gems.vendor_name != "metax", reason="MetaX transposed streaming"
    )
    @pytest.mark.parametrize(
        "shape,offsets",
        [
            ((8192, n, k), off)
            for n, k in ((512, 3584), (1024, 4096))
            for off in ((0, 0), (128, 128), (16, 128), (1, 0), (0, 1))
        ]
        + [
            ((m, n, k), (0, 0))
            for m in (8191, 8193)
            for n, k in ((512, 3584), (1024, 4096))
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_metax_tle_transposed_stream(shape, offsets, dtype, monkeypatch):
        if not getattr(_backend, "_STREAM_SHARED_MMA_PEEL_AVAILABLE", False):
            pytest.skip("FlagTree compiler lacks streaming tail peeling")
        kernel = _backend._mm_w8a8_kernel
        seen = []
        m, n, k = shape
        expected = m == 8192 and all(off % 128 == 0 for off in offsets)

        class CheckLaunch:
            def __getitem__(self, grid):
                def run(*args, **kwargs):
                    assert kwargs.get("TRANSPOSED_OUT", False) == expected
                    if expected:
                        assert args[5:8] == (n, m, k)
                        assert args[8:13] == (128, 128, 128, True, 1)
                    result = kernel[grid](*args, **kwargs)
                    seen.append(True)
                    return result

                return run

        monkeypatch.setattr(_backend, "_mm_w8a8_kernel", CheckLaunch())
        _check_metax_reverse(shape, offsets, dtype)
        assert seen

    @pytest.mark.skipif(
        flag_gems.vendor_name != "metax", reason="MetaX square streaming"
    )
    @pytest.mark.parametrize(
        "shape,offsets",
        [
            ((4096, 4096, 4096), off)
            for off in ((0, 0), (128, 128), (16, 128), (1, 0), (0, 1))
        ]
        + [((m, 4096, 4096), (0, 0)) for m in (4095, 4097)],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_metax_tle_square_stream(shape, offsets, dtype, monkeypatch):
        if not getattr(_backend, "_STREAM_SHARED_MMA_PEEL_AVAILABLE", False):
            pytest.skip("FlagTree compiler lacks streaming tail peeling")
        kernel = _backend._mm_w8a8_kernel
        seen = []
        expected = shape == (4096, 4096, 4096) and all(x % 128 == 0 for x in offsets)

        class CheckLaunch:
            def __getitem__(self, grid):
                def run(*args, **kwargs):
                    assert kwargs.get("SINGLE_SHARED", False) == expected
                    if expected:
                        assert args[8:13] == (128, 128, 128, True, 1)
                        assert kwargs["num_warps"] == 4
                        assert kwargs["stream_shared_mma_reverse"]
                        assert kwargs["stream_shared_mma_peel"]
                    result = kernel[grid](*args, **kwargs)
                    seen.append(True)
                    return result

                return run

        monkeypatch.setattr(_backend, "_mm_w8a8_kernel", CheckLaunch())
        _check_metax_reverse(shape, offsets, dtype)
        assert seen

    @pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX K16 MMA layout")
    @pytest.mark.parametrize(
        "shape",
        [(m, n, k) for m in (2, 3, 4) for n, k in ((3584, 3584), (4608, 3584))]
        + [(2, 4096, 4096), (4, 4096, 4096)]
        + [(8, 3584, 3584), (8, 4608, 3584), (8, 14336, 4096)],
    )
    @pytest.mark.parametrize("offsets", [(0, 0), (128, 128), (16, 16), (1, 3)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_metax_small_mma_k16(shape, offsets, dtype, monkeypatch):
        if not getattr(_backend, "_SMALL_MMA_K16_AVAILABLE", False):
            pytest.skip("FlagTree compiler lacks small_mma_k16")
        ao, bo = offsets
        original = _backend._mm_w8a8_kernel
        launches = []

        class CheckedKernel:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    selected = bool(kwargs.get("small_mma_k16", False))
                    tuned_default_layout = shape == (3, 3584, 3584)
                    assert selected == (
                        ao % 128 == 0 and bo % 128 == 0 and not tuned_default_layout
                    )
                    if selected:
                        assert kwargs["num_warps"] == 2
                    launches.append(selected)
                    result = original[grid](*args, **kwargs)
                    compiled, _ = result
                    assert compiled.metadata.small_mma_k16 == selected
                    assert (
                        "metax.small_mma_k16.applied" in compiled.asm["ttgir"]
                    ) == selected
                    return result

                return launch

        monkeypatch.setattr(_backend, "_mm_w8a8_kernel", CheckedKernel())
        _check_metax_tle_graph(shape, dtype, bo, ao)
        assert launches

    @pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX K16 fallback")
    @pytest.mark.parametrize("compiler_available", [False, True])
    def test_metax_small_mma_k16_fallback(monkeypatch, compiler_available):
        # Check actual fallback launches with the capability disabled and with
        # shapes outside the tuned range; all use the same integer oracle.
        available = getattr(_backend, "_SMALL_MMA_K16_AVAILABLE", False)
        monkeypatch.setattr(
            _backend, "_SMALL_MMA_K16_AVAILABLE", available and compiler_available
        )
        for shape, dtype in [
            ((3, 4096, 4096), torch.bfloat16),
            ((2, 3584, 3584), torch.bfloat16),
        ]:
            _check_metax_tle_graph(shape, dtype, 0)

    @pytest.mark.skipif(
        flag_gems.vendor_name != "metax", reason="MetaX K16 bias fallback"
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_metax_small_mma_k16_bias(dtype, monkeypatch):
        a = torch.randint(
            -128, 128, (1, 3, 3584), device=flag_gems.device, dtype=torch.int8
        )
        b = torch.randint(
            -128, 128, (3584, 3584), device=a.device, dtype=torch.int8
        ).t()
        sa = torch.tensor([0.001], device=a.device)
        sb = torch.tensor([0.002], device=a.device)
        bias = torch.randn(3584, device=a.device, dtype=dtype)
        original = _backend._mm_w8a8_kernel
        launches = []

        class CheckedKernel:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    assert not kwargs.get("small_mma_k16", False)
                    launches.append(True)
                    return original[grid](*args, **kwargs)

                return launch

        monkeypatch.setattr(_backend, "_mm_w8a8_kernel", CheckedKernel())
        ref = (a.cpu().view(3, 3584).long() @ b.cpu().long()).float()
        ref = (
            (ref * sa.cpu() * sb.cpu() + bias.cpu().float()).to(dtype).view(1, 3, 3584)
        )
        y = flag_gems.mm_w8a8_int8(a, b, sa, sb, dtype, bias)
        torch.testing.assert_close(y.cpu(), ref, rtol=0, atol=0)
        assert launches

    @pytest.mark.skipif(
        flag_gems.vendor_name != "metax", reason="MetaX medium-row transpose"
    )
    @pytest.mark.parametrize(
        "shape,offsets",
        [
            ((m, n, k), off)
            for m in (64, 65, 98, 127, 128)
            for n, k in ((28672, 4096), (37888, 3584), (128256, 4096), (152064, 3584))
            for off in ((0, 0), (16, 128))
        ]
        + [
            ((98, n, k), (128, 128))
            for n, k in ((28672, 4096), (37888, 3584), (128256, 4096), (152064, 3584))
        ]
        + [((m, 128256, 4096), (0, 0)) for m in (63, 129)]
        + [((98, 14336, 4096), (0, 0)), ((98, 18944, 3584), (0, 0))],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_metax_tle_mid_transposed_stream(shape, offsets, dtype, monkeypatch):
        if not getattr(_backend, "_STREAM_SHARED_MMA_PEEL_AVAILABLE", False):
            pytest.skip("FlagTree compiler lacks streaming tail peeling")
        kernel = _backend._mm_w8a8_kernel
        seen = []
        m, n, k = shape
        expected = (
            64 <= m <= 128
            and (n, k) in ((28672, 4096), (37888, 3584), (128256, 4096), (152064, 3584))
            and all(off % 128 == 0 for off in offsets)
        )

        class CheckLaunch:
            def __getitem__(self, grid):
                def run(*args, **kwargs):
                    assert kwargs.get("TRANSPOSED_OUT", False) == expected
                    if expected:
                        assert args[5:8] == (n, m, k)
                        assert args[8:13] == (128, 128, 128, True, 1)
                        assert kwargs["GROUP_M"] == 1
                    result = kernel[grid](*args, **kwargs)
                    if expected:
                        assert result[0].metadata.stream_shared_mma == 1
                        assert result[0].metadata.stream_shared_mma_reverse
                        assert result[0].metadata.stream_shared_mma_peel
                    seen.append(True)
                    return result

                return run

        monkeypatch.setattr(_backend, "_mm_w8a8_kernel", CheckLaunch())
        _check_metax_reverse(shape, offsets, dtype)
        assert seen

    @pytest.mark.skipif(
        flag_gems.vendor_name != "metax", reason="MetaX split transpose"
    )
    @pytest.mark.parametrize(
        "shape,offsets",
        [
            ((98, n, k), off)
            for n, k in ((4096, 4096), (4608, 3584))
            for off in ((0, 0), (128, 128), (16, 128), (1, 0))
        ]
        + [((m, 4096, 4096), (0, 0)) for m in (97, 99)]
        + [((98, 3584, 3584), (0, 0)), ((98, 6144, 4096), (0, 0))],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_metax_split_transposed_stream(shape, offsets, dtype, monkeypatch):
        if not getattr(_backend, "_STREAM_SHARED_MMA_SMALL_PEEL_AVAILABLE", False):
            pytest.skip("FlagTree compiler lacks small streaming tail peeling")
        m, n, k = shape
        expected = (
            m == 98
            and (n, k) in ((4096, 4096), (4608, 3584))
            and all(x % 128 == 0 for x in offsets)
        )
        kernel = _backend._mm_w8a8_kernel
        seen = []

        class CheckedKernel:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    selected = kwargs.get("TRANSPOSED_OUT", False)
                    assert selected == expected
                    if selected:
                        assert args[5:13] == (n, m, k, 64, 64, 128, True, 2)
                        assert kwargs["SPLIT_STORE"] and kwargs["SINGLE_SHARED"]
                        assert kwargs["GROUP_M"] == 8
                    result = kernel[grid](*args, **kwargs)
                    if selected:
                        c = result[0]
                        assert c.metadata.num_warps == 4
                        assert c.metadata.stream_shared_mma == 1
                        assert c.metadata.stream_shared_mma_mn == 64
                        assert c.metadata.stream_shared_mma_small_peel
                    seen.append(selected)
                    return result

                return launch

        monkeypatch.setattr(_backend, "_mm_w8a8_kernel", CheckedKernel())
        _check_metax_reverse(shape, offsets, dtype)
        assert seen

    @pytest.mark.skipif(
        flag_gems.vendor_name != "metax", reason="MetaX long-K grouping"
    )
    @pytest.mark.parametrize(
        "shape,offsets,dtype",
        [
            ((8192, 3584, 18944), off, dtype)
            for off in ((0, 0), (128, 128), (16, 128), (128, 16))
            for dtype in (torch.float16, torch.bfloat16)
        ]
        + [
            (shape, (0, 0), torch.bfloat16)
            for shape in (
                (8191, 3584, 18944),
                (8193, 3584, 18944),
                (8192, 3583, 18944),
                (8192, 3584, 18943),
                (8192, 4096, 14336),
            )
        ],
    )
    def test_metax_long_k_group(shape, offsets, dtype, monkeypatch):
        m, n, k = shape
        ao, bo = offsets
        a = torch.randint(
            -128, 128, (m * k + ao,), device=flag_gems.device, dtype=torch.int8
        )[ao:].view(m, k)
        b = (
            torch.randint(-128, 128, (n * k + bo,), device=a.device, dtype=torch.int8)[
                bo:
            ]
            .view(n, k)
            .t()
        )
        sa = torch.randn(m, 1, device=a.device) * 0.001
        sb = torch.randn(n, 1, device=a.device) * 0.002
        out = torch.empty((m, n), device=a.device, dtype=dtype)
        expected = (
            shape == (8192, 3584, 18944)
            and dtype == torch.bfloat16
            and a.data_ptr() % 128 == 0
            and b.data_ptr() % 128 == 0
        )
        kernel = _backend._mm_w8a8_kernel
        seen = []

        class CheckLaunch:
            def __getitem__(self, grid):
                def run(*args, **kwargs):
                    assert kwargs["GROUP_M"] == (4 if expected else 8)
                    result = kernel[grid](*args, **kwargs)
                    if expected:
                        assert args[5:13] == (m, n, k, 256, 256, 64, True, 1)
                        assert result[0].metadata.num_warps == 8
                    seen.append(True)
                    return result

                return run

        monkeypatch.setattr(_backend, "_mm_w8a8_kernel", CheckLaunch())
        flag_gems.mm_w8a8_int8_out(a, b, sa, sb, out=out)
        rows, cols = [0, m // 2, m - 1], [0, n // 2, n - 1]
        integer = a[rows].cpu().long() @ b[:, cols].cpu().long()
        ref = (integer.float() * sa[rows].cpu() * sb[cols].cpu().t()).to(dtype)
        torch.testing.assert_close(out[rows][:, cols].cpu(), ref, rtol=0, atol=0)
        assert seen

    @pytest.mark.skipif(
        flag_gems.vendor_name != "metax", reason="MetaX M8 K16 boundaries"
    )
    @pytest.mark.parametrize(
        "shape",
        [
            (m, n, k)
            for m in (7, 9)
            for n, k in ((3584, 3584), (4608, 3584), (14336, 4096))
        ]
        + [(8, 4096, 4096), (8, 6144, 4096), (8, 18944, 3584)],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_metax_small_mma_k16_m8_fallback(shape, dtype, monkeypatch):
        original = _backend._mm_w8a8_kernel
        seen = []

        class CheckedKernel:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    assert not kwargs.get("small_mma_k16", False)
                    compiled, hit = original[grid](*args, **kwargs)
                    assert not compiled.metadata.small_mma_k16
                    seen.append(True)
                    return compiled, hit

                return launch

        monkeypatch.setattr(_backend, "_mm_w8a8_kernel", CheckedKernel())
        _check_metax_tle_graph(shape, dtype, 0, 0)
        assert seen

    @pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX machine unroll")
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("compiler_available", [True, False])
    @pytest.mark.parametrize("shape", [(1024, 1024, 1024), (256, 6144, 4096)])
    def test_metax_machine_unroll_cache(shape, dtype, compiler_available, monkeypatch):
        available = getattr(_backend, "_MMA_UNROLL_AVAILABLE", False)
        monkeypatch.setattr(
            _backend, "_MMA_UNROLL_AVAILABLE", available and compiler_available
        )
        original = _backend._mm_w8a8_kernel
        seen = []

        class CheckedKernel:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    enabled = (
                        available
                        and compiler_available
                        and args[5:8] in ((1024, 1024, 1024), (256, 6144, 4096))
                        and args[4].dtype in (torch.float16, torch.bfloat16)
                        and args[0].data_ptr() % 128 == 0
                        and args[1].data_ptr() % 128 == 0
                    )
                    expected = 8 if enabled else None
                    assert kwargs.get("mma_unroll_count") == expected
                    assert kwargs["MACHINE_UNROLL"] == (expected or 0)
                    result = original[grid](*args, **kwargs)
                    compiled, _ = result
                    assert (
                        getattr(compiled.metadata, "mma_unroll_count", None) == expected
                    )
                    assert compiled.metadata.num_stages == 2
                    seen.append(enabled)
                    return result

                return launch

        monkeypatch.setattr(_backend, "_mm_w8a8_kernel", CheckedKernel())
        if dtype == torch.float32:
            with pytest.raises(TypeError, match="out_dtype must be FP16 or BF16"):
                _check_metax_tle_graph(shape, dtype, 0, 0)
            assert not seen
            return
        # Both offsets satisfy LibEntry's 16-byte pointer specialization. Alternate
        # them to verify the explicit constexpr prevents cross-option cache reuse.
        for offset in (0, 16, 128, 0):
            _check_metax_tle_graph(shape, dtype, offset, offset)
        _check_metax_tle_graph((512, 512, 512), dtype, 0, 0)
        assert seen

    @pytest.mark.skipif(
        flag_gems.vendor_name != "metax", reason="MetaX FlagTune configurations"
    )
    @pytest.mark.parametrize(
        "shape,expected",
        [
            ((1, 6144, 4096), (64, 4, 2, True, False)),
            ((2, 3584, 3584), (64, 2, 4, False, True)),
            ((3, 3584, 3584), (32, 2, 4, True, False)),
        ],
    )
    @pytest.mark.parametrize("offset", [0, 1, 16, 128])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_metax_flagtune_small_dispatch(shape, expected, offset, dtype, monkeypatch):
        original = _backend._mm_w8a8_kernel
        observed = []

        class CheckedKernel:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    result = original[grid](*args, **kwargs)
                    compiled, _ = result
                    if offset % 128 == 0:
                        bn, warps, splits, mask_m, small = expected
                        assert args[8:13] == (16, bn, 128, True, splits)
                        assert kwargs["MASK_M"] == mask_m
                        assert kwargs["SMALL_MMA_K16"] == small
                        assert compiled.metadata.num_warps == warps
                    observed.append(True)
                    return result

                return launch

        monkeypatch.setattr(_backend, "_mm_w8a8_kernel", CheckedKernel())
        _check_metax_tle_graph(shape, dtype, offset, offset)
        assert observed

    @pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="MetaX FlagTune grid")
    @pytest.mark.parametrize("shape", [(98, 4608, 3584), (98, 4096, 4096)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_metax_flagtune_mid_grid(shape, dtype, monkeypatch):
        original = _backend._mm_w8a8_kernel
        seen = []

        class CheckedKernel:
            def __getitem__(self, grid):
                def launch(*args, **kwargs):
                    if kwargs.get("TRANSPOSED_OUT") and args[8] == 64:
                        tuned = shape == (98, 4608, 3584)
                        assert kwargs["SPLIT_GROUP"] == (1 if tuned else 0)
                        assert kwargs["GROUP_M"] == (16 if tuned else 8)
                        assert len(grid) == (1 if tuned else 2)
                        seen.append(True)
                    return original[grid](*args, **kwargs)

                return launch

        monkeypatch.setattr(_backend, "_mm_w8a8_kernel", CheckedKernel())
        for offset in (0, 1, 16, 128, 0):
            _check_metax_tle_graph(shape, dtype, offset, offset)
        assert seen
