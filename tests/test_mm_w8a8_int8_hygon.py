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

import pytest
import torch

import flag_gems

pytestmark = [
    pytest.mark.mm_w8a8_int8,
    pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon INT8"),
]
DTYPES = [torch.float16, torch.bfloat16, torch.float32]


def reference(a, b, dtype):
    a, b = a.cpu().float(), b.cpu().float()
    if a.shape[1] == 0:
        return torch.zeros((a.shape[0], b.shape[1]), dtype=dtype)
    pa = a.abs().amax(1).clamp_min(1e-10)
    pb = b.abs().amax(0).clamp_min(1e-10)
    aq = (a / pa[:, None] * 127).round().clamp(-127, 127).long()
    bq = (b / pb[None, :] * 127).round().clamp(-127, 127).long()
    return (
        (aq @ bq).float() * (pa[:, None] * (1 / 127)) * (pb[None, :] * (1 / 127))
    ).to(dtype)


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
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)
    expected = reference(a, b, out_dtype)
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
    a = torch.randn(17, 66, device="cuda")
    b = torch.randn(66, 35, device="cuda")
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
        y.cpu(), reference(a, b, torch.bfloat16), rtol=1e-5, atol=1e-4
    )


@pytest.mark.parametrize("shape", [(0, 5, 3), (3, 0, 5), (3, 5, 0), (0, 0, 0)])
def test_empty(shape):
    m, n, k = shape
    a = torch.empty(m, k, device="cuda")
    b = torch.empty(k, n, device="cuda")
    y = flag_gems.mm_w8a8_int8(a, b)
    assert y.shape == (m, n)
    if k == 0:
        assert torch.count_nonzero(y).item() == 0


def test_boundaries_and_updates():
    # Exact half-integers test nearest-even, including negative ties.
    v = torch.tensor(
        [127.0, -127.0, 0.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5], device="cuda"
    )
    a = v.repeat(4, 1)
    b = torch.eye(9, device="cuda")
    a[1].zero_()
    a[2] *= 1e-12
    for _ in range(2):
        y = flag_gems.mm_w8a8_int8(a, b, out_dtype=torch.float32)
        torch.testing.assert_close(
            y.cpu(), reference(a, b, torch.float32), rtol=1e-5, atol=1e-6
        )
        a.mul_(-2)
        b.mul_(3)


def test_graph_replay_and_alias():
    a = torch.randn(32, 32, device="cuda")
    b = torch.randn(32, 32, device="cuda")
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
            y.cpu(), reference(a, b, torch.float32), rtol=1e-5, atol=1e-4
        )
    expected = reference(a, b, torch.float32)
    assert flag_gems.mm_w8a8_int8_out(a, b, out=a) is a
    torch.testing.assert_close(a.cpu(), expected, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize(
    "case",
    ["dtype", "ndim", "shape", "device", "outdtype", "outshape", "outstride"],
)
def test_invalid(case):
    a = torch.ones(3, 4, device="cuda")
    b = torch.ones(4, 5, device="cuda")
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
            flag_gems.mm_w8a8_int8_out(a, b, out=torch.empty(5, 3, device="cuda"))
        elif case == "outstride":
            flag_gems.mm_w8a8_int8_out(a, b, out=torch.empty(5, 3, device="cuda").t())


@pytest.mark.parametrize("dtype", DTYPES)
def test_quantized_codes(dtype):
    import sys

    backend = sys.modules[flag_gems.mm_w8a8_int8.__module__]
    x = torch.randn(19, 2051, device="cuda", dtype=dtype)
    q = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty(19, device="cuda", dtype=torch.float32)
    backend._quantize[(19,)](
        x, q, scale, 2051, *x.stride(), 1024, num_warps=4, enable_fp_fusion=False
    )
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
    a = torch.randn(5, 37, device="cuda", dtype=adtype)
    b = torch.randn(37, 9, device="cuda", dtype=bdtype)
    y = flag_gems.mm_w8a8_int8(a, b, out_dtype=torch.float32)
    torch.testing.assert_close(
        y.cpu(), reference(a, b, torch.float32), rtol=1e-5, atol=1e-4
    )


def test_int32_limit():
    k = 2147483647 // (127 * 127)
    a = torch.ones(1, k, device="cuda")
    b = torch.ones(k, 1, device="cuda")
    y = flag_gems.mm_w8a8_int8(a, b, out_dtype=torch.float32)
    torch.testing.assert_close(y, torch.full_like(y, float(k)), rtol=1e-6, atol=0)


@pytest.mark.parametrize(
    "shape",
    [(1, 128, 128), (1, 257, 381), (1, 4096, 4096), (4, 128, 2051), (1025, 129, 257)],
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_optimized_paths(shape, dtype):
    m, n, k = shape
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)
    out = flag_gems.mm_w8a8_int8(a, b, out_dtype=torch.float32)
    torch.testing.assert_close(
        out.cpu(), reference(a, b, torch.float32), rtol=1e-5, atol=1e-4
    )


def test_fused_graph_updates_and_shifted_alias():
    a = torch.randn(1, 257, device="cuda")
    b = torch.randn(257, 257, device="cuda")
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
            y.cpu(), reference(a, b, torch.float32), rtol=1e-5, atol=1e-4
        )
    ref = reference(a, b, torch.float32)
    out = b.flatten()[5:262].view(1, 257)
    flag_gems.mm_w8a8_int8_out(a, b, out=out)
    torch.testing.assert_close(out.cpu(), ref, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("dtype", DTYPES)
def test_optimized_quantization_ties(dtype):
    import sys

    backend = sys.modules[flag_gems.mm_w8a8_int8.__module__]
    values = torch.tensor(
        [127.0, -127.0, 0.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5],
        device="cuda",
        dtype=dtype,
    )
    x = values.repeat(19, 29)[:, :257].contiguous()
    q = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty(19, device="cuda")
    backend._quantize_rows[(19,)](
        x, q, scale, 257, *x.stride(), 512, num_warps=4, enable_fp_fusion=False
    )
    cpu = x.cpu().float()
    peak = cpu.abs().amax(1).clamp_min(1e-10)
    ref = (cpu / peak[:, None] * 127).round().to(torch.int8)
    torch.testing.assert_close(q.cpu(), ref, rtol=0, atol=0)
    b = x.t().contiguous()
    backend._quantize_columns_full[(3,)](
        b, q, scale, 257, 19, 8, num_warps=4, enable_fp_fusion=False
    )
    torch.testing.assert_close(q.cpu(), ref, rtol=0, atol=0)
    torch.testing.assert_close(scale.cpu(), peak * (1.0 / 127.0), rtol=0, atol=0)


@pytest.mark.parametrize("k", [133145, 151936, 152064])
def test_long_k_overflow(k):
    import sys

    backend = sys.modules[flag_gems.mm_w8a8_int8.__module__]
    a = torch.ones(1, k, device="cuda")
    b = torch.ones(k, 1, device="cuda")
    actual = flag_gems.mm_w8a8_int8(a, b, out_dtype=torch.float32)
    torch.testing.assert_close(
        actual, torch.full_like(actual, float(k)), rtol=1e-6, atol=0
    )
    aq = torch.full((1, k), -128, device="cuda", dtype=torch.int8)
    bq = torch.full((k, 1), -128, device="cuda", dtype=torch.int8)
    scale = torch.ones(1, device="cuda")
    actual = backend._mm_w8a8_int8_prequantized(
        aq, bq, scale, scale, out_dtype=torch.float32
    )
    expected = torch.full_like(actual, float(k * 128 * 128))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(1, 128, 257), (17, 65, 129), (128, 256, 1024)])
def test_prequantized_entry(shape):
    import sys

    backend = sys.modules[flag_gems.mm_w8a8_int8.__module__]
    m, n, k = shape
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    prepared = backend._prepare_mm_w8a8_int8_inputs(a, b)
    y = backend._mm_w8a8_int8_prequantized(*prepared, out_dtype=torch.float32)
    torch.testing.assert_close(
        y.cpu(), reference(a, b, torch.float32), rtol=1e-5, atol=1e-4
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
    import sys

    backend = sys.modules[flag_gems.mm_w8a8_int8.__module__]
    m, n, k = shape
    a = torch.randint(-128, 128, (m, k), device="cuda", dtype=torch.int8)
    b = torch.randint(-128, 128, (n, k), device="cuda", dtype=torch.int8).t()
    sa = torch.ones(m, device="cuda")
    sb = torch.ones(n, device="cuda")
    actual = backend._mm_w8a8_int8_prequantized(a, b, sa, sb, out_dtype=torch.float32)
    expected = (a.cpu().long() @ b.cpu().long()).float()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_large_tile_long_k_overflow():
    import sys

    backend = sys.modules[flag_gems.mm_w8a8_int8.__module__]
    m, n, k = 1025, 1025, 131073
    a = torch.full((m, k), -128, device="cuda", dtype=torch.int8)
    b = torch.full((n, k), -128, device="cuda", dtype=torch.int8).t()
    sa = torch.full((m,), 0.5, device="cuda")
    sb = torch.full((n,), -0.25, device="cuda")
    actual = backend._mm_w8a8_int8_prequantized(a, b, sa, sb, out_dtype=torch.float32)
    expected = torch.full_like(actual, float(k * 128 * 128) * -0.125)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
