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

"""Ascend FP8 TopK: exact selection on dequantized data (ties may reorder)."""

import pytest
import torch

import flag_gems


@pytest.fixture(autouse=True)
def _require_ascend_common_ir():
    if flag_gems.device != "npu":
        pytest.skip("Ascend required")
    try:
        import triton.language.extra.cann.extension as al
    except ImportError:
        pytest.skip("FlagTree Common IR al.custom support required")
    if not all(hasattr(al, name) for name in ("custom", "register_custom_op", "scope")):
        pytest.skip("FlagTree Common IR al.custom support required")


CASES = [
    ((4, 128), 8),
    ((8, 256), 16),
    ((64, 1024), 32),
    ((64, 4096), 64),
    ((64, 8192), 128),
    ((128, 32768), 256),
    ((2, 33, 128), 5),
    ((3, 257), 17),
]


@pytest.mark.parametrize("shape,k", CASES)
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("row_scale", [True, False])
@pytest.mark.topk_w8a16_fp8
def test_topk_fp8(shape, k, largest, row_scale):
    if flag_gems.device != "npu":
        pytest.skip("Ascend required")
    torch.manual_seed(127)
    n = shape[-1]
    g = n if row_scale else 128
    q = torch.randn(shape).to(torch.float8_e4m3fn)
    s = (torch.rand(shape[:-1] + ((n + g - 1) // g,)) * 1.5 + 0.1).bfloat16()
    ref = q.float() * s.float().repeat_interleave(g, -1)[..., :n]
    v, i = flag_gems.topk_w8a16_fp8(q.npu(), s.npu(), k, group_size=g, largest=largest)
    v = v.cpu()
    i = i.cpu()
    torch.testing.assert_close(
        v, torch.topk(ref, k, largest=largest).values.bfloat16(), rtol=0, atol=0
    )
    torch.testing.assert_close(v, torch.gather(ref, -1, i).bfloat16(), rtol=0, atol=0)
    assert i.dtype == torch.int64
    ids = torch.sort(i).values
    assert (ids[..., 1:] != ids[..., :-1]).all()


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize(
    "kind", ["ties", "subnormal", "negative_scale", "zero_k", "all_k"]
)
@pytest.mark.topk_w8a16_fp8
def test_topk_fp8_edges(dtype, largest, kind):
    if flag_gems.device != "npu":
        pytest.skip("Ascend required")
    torch.manual_seed(98)
    q = torch.randn((3, 128)).to(dtype)
    if kind == "ties":
        q = torch.ones((3, 128)).to(dtype)
    if kind == "subnormal":
        q = (torch.arange(128).float().reshape(1, -1).repeat(3, 1) / 65536).to(dtype)
    s = (
        torch.tensor([[-0.3], [0.0], [1.3]], dtype=torch.float32)
        if kind == "negative_scale"
        else torch.ones((3, 1), dtype=torch.bfloat16)
    )
    ref = q.float() * s.float()
    k = 0 if kind == "zero_k" else 128 if kind == "all_k" else 16
    v, i = flag_gems.topk_w8a16_fp8(
        q.npu(), s.npu(), k, group_size=128, largest=largest, sorted=False
    )
    v = v.cpu()
    i = i.cpu()
    torch.testing.assert_close(
        v, torch.topk(ref, k, largest=largest).values.bfloat16(), rtol=0, atol=0
    )
    torch.testing.assert_close(v, torch.gather(ref, -1, i).bfloat16(), rtol=0, atol=0)
    ids = torch.sort(i).values
    assert (ids[..., 1:] != ids[..., :-1]).all()


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.topk_w8a16_fp8
def test_topk_fp8_maximum_row(dtype, largest):
    if flag_gems.device != "npu":
        pytest.skip("Ascend required")
    torch.manual_seed(33)
    q = torch.randn((2, 32768)).to(dtype)
    s = torch.ones((2, 1), dtype=torch.bfloat16)
    ref = q.float()
    v, i = flag_gems.topk_w8a16_fp8(
        q.npu(), s.npu(), 512, group_size=32768, largest=largest
    )
    v, i = v.cpu(), i.cpu()
    torch.testing.assert_close(
        v, torch.topk(ref, 512, largest=largest).values.bfloat16(), rtol=0, atol=0
    )
    torch.testing.assert_close(v, torch.gather(ref, -1, i).bfloat16(), rtol=0, atol=0)
    ids = torch.sort(i).values
    assert (ids[:, 1:] != ids[:, :-1]).all()


@pytest.mark.parametrize(
    "dtype,max_code", [(torch.float8_e4m3fn, 126), (torch.float8_e5m2, 123)]
)
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.topk_w8a16_fp8
def test_topk_fp8_all_finite_encodings(dtype, max_code, largest):
    if flag_gems.device != "npu":
        pytest.skip("Ascend required")
    codes = torch.cat(
        [torch.arange(max_code + 1), torch.arange(max_code + 1) + 128]
    ).to(torch.uint8)
    q = codes.view(dtype).reshape(1, -1)
    s = torch.ones((1, 1), dtype=torch.bfloat16)
    k = q.shape[-1]
    v, i = flag_gems.topk_w8a16_fp8(
        q.npu(), s.npu(), k, group_size=q.shape[-1], largest=largest
    )
    v, i = v.cpu(), i.cpu()
    ref = q.float()
    torch.testing.assert_close(
        v, torch.topk(ref, k, largest=largest).values.bfloat16(), rtol=0, atol=0
    )
    torch.testing.assert_close(v, torch.gather(ref, -1, i).bfloat16(), rtol=0, atol=0)


@pytest.mark.skipif(flag_gems.device != "npu", reason="Ascend required")
@pytest.mark.topk_w8a16_fp8
def test_topk_fp8_graph_replay_changed_inputs():
    torch.manual_seed(811)
    shape, k = (64, 4096), 64
    q_cpu = torch.randn(shape).to(torch.float8_e4m3fn)
    s_cpu = torch.ones((64, 1), dtype=torch.bfloat16)
    q, s = q_cpu.npu(), s_cpu.npu()
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        flag_gems.topk_w8a16_fp8(q, s, k, group_size=4096)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        values, indices = flag_gems.topk_w8a16_fp8(q, s, k, group_size=4096)
    for changed in (False, True):
        if changed:
            q_cpu = (torch.randn(shape) * 2).to(torch.float8_e4m3fn)
            s_cpu = torch.linspace(-1, 1, 64).reshape(64, 1).bfloat16()
            q.view(torch.uint8).copy_(q_cpu.view(torch.uint8).npu())
            s.copy_(s_cpu.npu())
        graph.replay()
        torch.npu.synchronize()
        ref = q_cpu.float() * s_cpu.float()
        torch.testing.assert_close(
            values.cpu(), torch.topk(ref, k).values.bfloat16(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            values.cpu(),
            torch.gather(ref, -1, indices.cpu()).bfloat16(),
            rtol=0,
            atol=0,
        )


@pytest.mark.skipif(flag_gems.device != "npu", reason="Ascend required")
@pytest.mark.topk_w8a16_fp8
def test_topk_fp8_large_cutoff_ties_and_zero_scale():
    q_cpu = torch.ones((64, 4096)).to(torch.float8_e4m3fn)
    s_cpu = ((torch.arange(64) % 3) - 1).reshape(64, 1).bfloat16()
    q, s = q_cpu.npu(), s_cpu.npu()
    ref = q_cpu.float() * s_cpu.float()
    for largest in (True, False):
        v, i = flag_gems.topk_w8a16_fp8(q, s, 64, group_size=4096, largest=largest)
        v, i = v.cpu(), i.cpu()
        torch.testing.assert_close(
            v, torch.topk(ref, 64, largest=largest).values.bfloat16(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            v, torch.gather(ref, -1, i).bfloat16(), rtol=0, atol=0
        )
        ordered = torch.sort(i).values
        assert (ordered[:, 1:] != ordered[:, :-1]).all()


@pytest.mark.skipif(flag_gems.device != "npu", reason="Ascend required")
@pytest.mark.topk_w8a16_fp8
def test_topk_fp8_unaligned_storage_and_current_stream():
    torch.manual_seed(953)
    shape, k = (3, 4096), 64
    q_cpu = torch.randn(shape).to(torch.float8_e4m3fn)
    q_storage = torch.empty(q_cpu.numel() + 1, device="npu", dtype=torch.uint8)
    q_storage[1:].copy_(q_cpu.reshape(-1).view(torch.uint8).npu())
    q = q_storage[1:].view(torch.float8_e4m3fn).reshape(shape)
    scales = torch.empty(4, device="npu", dtype=torch.bfloat16)
    s = scales[1:].reshape(3, 1)
    s_cpu = torch.tensor([[0.5], [-1.5], [2.0]], dtype=torch.bfloat16)
    s.copy_(s_cpu.npu())
    assert q.is_contiguous() and q.data_ptr() % 32 != 0
    for _ in range(2):
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            v, i = flag_gems.topk_w8a16_fp8(q, s, k, group_size=4096)
        stream.synchronize()
        ref = q_cpu.float() * s_cpu.float()
        torch.testing.assert_close(
            v.cpu(), torch.topk(ref, k).values.bfloat16(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            v.cpu(), torch.gather(ref, -1, i.cpu()).bfloat16(), rtol=0, atol=0
        )
        q_cpu = (-torch.randn(shape)).to(torch.float8_e4m3fn)
        q_storage[1:].copy_(q_cpu.reshape(-1).view(torch.uint8).npu())


@pytest.mark.topk_w8a16_fp8
def test_topk_fp8_missing_common_ir(monkeypatch):
    from importlib import import_module

    module = import_module("flag_gems.runtime.backend._ascend.ops.topk_w8a16_fp8")
    monkeypatch.setattr(module, "al", None)
    monkeypatch.setattr(module, "_READY", False)
    with pytest.raises(RuntimeError, match="requires FlagTree Common IR"):
        module._ensure()
