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

from . import base


def _mm_w8a8_fp8_available():
    return (
        getattr(flag_gems, "vendor_name", None) == "thead"
        and hasattr(torch, "float8_e4m3fn")
        and hasattr(flag_gems, "mm_w8a8_fp8")
    )


def _torch_mm(a, b):
    return torch.mm(a, b)


def _gems_mm_w8a8_fp8(a, b):
    return flag_gems.mm_w8a8_fp8(a, b, out_dtype=a.dtype)


class MmW8A8Fp8Benchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "M, N, K"

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (16, 128, 128),
            (64, 256, 256),
            (128, 1024, 1024),
            (256, 256, 2048),
            (512, 1024, 2048),
            (16, 2048, 4096),
            (64, 2048, 4096),
            (1, 12288, 2048),
        ]

    def get_input_iter(self, dtype):
        for m, n, k in self.shapes:
            a = torch.randn((m, k), dtype=dtype, device=self.device)
            b = torch.randn((k, n), dtype=dtype, device=self.device)
            yield a, b


@pytest.mark.mm_w8a8_fp8
@pytest.mark.skipif(
    not _mm_w8a8_fp8_available(),
    reason="mm_w8a8_fp8 is a THead/PPU operator",
)
def test_mm_w8a8_fp8_vs_torch_bf16():
    bench = MmW8A8Fp8Benchmark(
        op_name="mm_w8a8_fp8_vs_torch_bf16",
        torch_op=_torch_mm,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(_gems_mm_w8a8_fp8)
    bench.run()
