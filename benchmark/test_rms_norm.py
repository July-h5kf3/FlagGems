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

from . import base, consts


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
W8A16_SHAPES = [
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


def _quantize_int8_grouped(weight, group_size=GROUP_SIZE):
    n = weight.numel()
    grouped = weight.reshape(n // group_size, group_size).to(torch.float32)
    scale = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    quant = (grouped / scale).round().clamp(-128, 127).to(torch.int8)
    return (
        quant.reshape(n).contiguous(),
        scale.squeeze(-1).to(weight.dtype).contiguous(),
    )


def _dequant_int8_grouped(weight_q, weight_scale, group_size=GROUP_SIZE):
    return (
        weight_q.to(torch.float32)
        * weight_scale.to(torch.float32).repeat_interleave(group_size)
    ).to(weight_scale.dtype)


def _ref_rms_norm(x, normalized_shape, weight_q, weight_scale, weight_ref):
    return torch.nn.functional.rms_norm(x, normalized_shape, weight_ref, eps=1e-5)


def _gems_rms_norm_w8a16(x, normalized_shape, weight_q, weight_scale, weight_ref):
    return flag_gems.rms_norm_fp8_w8a16(
        x, normalized_shape, weight_q, weight_scale, eps=1e-5
    )


class RmsNormW8A16TheadBenchmark(base.Benchmark):
    DEFAULT_DTYPES = [torch.bfloat16]
    DEFAULT_SHAPE_DESC = "tokens, hidden"
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["gbps"]

    def set_shapes(self, shape_file_path=None):
        self.shapes = list(W8A16_SHAPES)

    def get_gbps(self, args, latency=None):
        x, _, weight_q, weight_scale, _ = args
        nbytes = (
            2 * x.numel() * x.element_size()
            + weight_q.numel() * weight_q.element_size()
            + weight_scale.numel() * weight_scale.element_size()
        )
        return nbytes * 1e-9 / (latency * 1e-3)

    def get_input_iter(self, cur_dtype):
        for tokens, hidden in self.shapes:
            x = torch.randn(tokens, hidden, dtype=cur_dtype, device=self.device)
            weight = torch.randn(hidden, dtype=cur_dtype, device=self.device)
            weight_q, weight_scale = _quantize_int8_grouped(weight)
            weight_ref = _dequant_int8_grouped(weight_q, weight_scale)
            yield x, (hidden,), weight_q, weight_scale, weight_ref


@pytest.mark.rms_norm
@pytest.mark.skipif(
    flag_gems.vendor_name != "thead",
    reason="W8A16 RMSNorm is implemented on THead / PPU only",
)
def test_rms_norm_w8a16_thead():
    bench = RmsNormW8A16TheadBenchmark(
        op_name="rms_norm_fp8_w8a16",
        torch_op=_ref_rms_norm,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_rms_norm_w8a16)
    bench.run()
