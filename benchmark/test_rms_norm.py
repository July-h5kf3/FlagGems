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

import statistics

import pytest
import torch

import flag_gems

from . import base, consts
from .conftest import Config


@pytest.mark.rms_norm
def test_rms_norm():
    def rms_norm_input_fn(shape, dtype, device):
        _, N = shape
        inp = torch.randn(shape, dtype=dtype, device=device)
        weight = torch.randn(N, dtype=dtype, device=device)
        yield inp, (N,), weight

    bench = base.GenericBenchmark2DOnly(
        op_name="rms_norm",
        input_fn=rms_norm_input_fn,
        torch_op=torch.nn.functional.rms_norm,
    )
    bench.run()


GROUP_SIZE = 128
FP8_DTYPE = torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else None


def _cuda_fp8_e4m3fn_available():
    if FP8_DTYPE is None or not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def _w8a16_available():
    # Ascend uses grouped INT8 weight (UB cannot load FP8). NVIDIA uses FP8 e4m3.
    return flag_gems.vendor_name == "ascend" or _cuda_fp8_e4m3fn_available()


def _quantize_int8_grouped(w, group_size=GROUP_SIZE):
    if w.ndim == 1:
        n = w.shape[0]
        assert n % group_size == 0
        wg = w.reshape(n // group_size, group_size).float()
        scale = (wg.abs().amax(dim=-1, keepdim=True) / 127).clamp(min=1e-8)
        q = (wg / scale).round().clamp(-128, 127).to(torch.int8)
        return q.reshape(n).contiguous(), scale.squeeze(-1).to(w.dtype).contiguous()
    m, n = w.shape
    assert n % group_size == 0
    wg = w.reshape(m, n // group_size, group_size).float()
    scale = (wg.abs().amax(dim=-1, keepdim=True) / 127).clamp(min=1e-8)
    q = (wg / scale).round().clamp(-128, 127).to(torch.int8)
    return q.reshape(m, n).contiguous(), scale.squeeze(-1).to(w.dtype).contiguous()


def _quantize_fp8_grouped(w, group_size=GROUP_SIZE):
    fp8_info = torch.finfo(FP8_DTYPE)
    if w.ndim == 1:
        n = w.shape[0]
        assert n % group_size == 0
        wg = w.reshape(n // group_size, group_size).float()
        scale = (wg.abs().amax(dim=-1, keepdim=True) / fp8_info.max).clamp(min=1e-8)
        q = (wg / scale).clamp(fp8_info.min, fp8_info.max).to(FP8_DTYPE)
        return q.reshape(n).contiguous(), scale.squeeze(-1).to(w.dtype).contiguous()
    m, n = w.shape
    assert n % group_size == 0
    wg = w.reshape(m, n // group_size, group_size).float()
    scale = (wg.abs().amax(dim=-1, keepdim=True) / fp8_info.max).clamp(min=1e-8)
    q = (wg / scale).clamp(fp8_info.min, fp8_info.max).to(FP8_DTYPE)
    return q.reshape(m, n).contiguous(), scale.squeeze(-1).to(w.dtype).contiguous()


def _torch_bf16_rms_norm(x, normalized_shape, weight_fp8, weight_scale, weight_ref):
    return torch.nn.functional.rms_norm(x, normalized_shape, weight_ref)


def _gems_bf16_rms_norm(x, normalized_shape, weight_fp8, weight_scale, weight_ref):
    return flag_gems.rms_norm(x, normalized_shape, weight_ref)


def _quantize_w8a16_weight(w, group_size=GROUP_SIZE):
    if flag_gems.vendor_name == "ascend":
        return _quantize_int8_grouped(w, group_size)
    return _quantize_fp8_grouped(w, group_size)


def _gems_rms_norm_w8a16(x, normalized_shape, weight_fp8, weight_scale, weight_ref):
    return flag_gems.rms_norm_w8a16_fp8(x, normalized_shape, weight_fp8, weight_scale)


def _do_bench_graph(fn, rep=100):
    # Capture once and replay, so host launch overhead is amortized out.
    if flag_gems.vendor_name == "ascend":
        device = torch.npu
        Graph = torch.npu.NPUGraph
        graph_ctx = lambda g: torch.npu.graph(g, capture_error_mode="relaxed")
    else:
        device = torch.cuda
        Graph = torch.cuda.CUDAGraph
        graph_ctx = torch.cuda.graph

    with device.stream(device.Stream()):
        fn()
        start_event = device.Event(enable_timing=True)
        end_event = device.Event(enable_timing=True)
        start_event.record()
        for _ in range(5):
            fn()
        end_event.record()
        device.synchronize()
        estimate_ms = start_event.elapsed_time(end_event) / 5
        n_repeat = 1000 if estimate_ms == 0 else max(1, int(rep / estimate_ms))

        graph = Graph()
        with graph_ctx(graph):
            for _ in range(n_repeat):
                fn()
        device.synchronize()

        times = []
        for _ in range(10):
            start_event = device.Event(enable_timing=True)
            end_event = device.Event(enable_timing=True)
            start_event.record()
            graph.replay()
            end_event.record()
            device.synchronize()
            times.append(start_event.elapsed_time(end_event) / n_repeat)
        return statistics.median(times)


class RmsNormFp8Benchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "M, N"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1, 4096),
            (16, 4096),
            (64, 4096),
            (256, 4096),
            (1024, 4096),
            (1, 8192),
            (64, 8192),
            (256, 8192),
            (1, 16384),
            (64, 16384),
        ]


class RmsNormFp8W8A16Benchmark(RmsNormFp8Benchmark):
    def get_input_iter(self, dtype):
        for shape in self.shapes:
            _, n = shape
            x = torch.randn(shape, dtype=dtype, device=self.device)
            weight = torch.randn(n, dtype=dtype, device=self.device)
            weight_fp8, weight_scale = _quantize_w8a16_weight(weight)
            yield x, (n,), weight_fp8, weight_scale, weight

    def get_latency(self, op, *args, **kwargs):
        return _do_bench_graph(
            lambda: op(*args, **kwargs),
            rep=Config.repetition,
        )


@pytest.mark.rms_norm_w8a16_fp8
@pytest.mark.skipif(
    not _w8a16_available(),
    reason="RMSNorm W8A16 requires Ascend or CUDA sm90+ float8_e4m3fn",
)
def test_rms_norm_w8a16_fp8():
    Config.mode = consts.BenchMode.CUDAGRAPH
    for op_name, baseline in (
        ("rms_norm_w8a16_fp8_vs_torch", _torch_bf16_rms_norm),
        ("rms_norm_w8a16_fp8_vs_gems", _gems_bf16_rms_norm),
    ):
        bench = RmsNormFp8W8A16Benchmark(
            op_name=op_name,
            torch_op=baseline,
            dtypes=[torch.bfloat16],
        )
        bench.set_gems(_gems_rms_norm_w8a16)
        bench.run()
