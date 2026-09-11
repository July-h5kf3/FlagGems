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

from .consts import FLOAT_DTYPES
from .test_blas_perf_parallel import ParallelBlasBenchmark, mm_input_fn


class ParallelMmW8A8Int8Benchmark(ParallelBlasBenchmark):
    """Shared W8A8 INT8 workloads; preparation is outside the selected timer.

    Use --mode cudagraph --level comprehensive --dtypes bfloat16 --dtypes
    float16 to cover the reference PR's 121 configurations and two B layouts.
    Both baseline variants use BF16, independently of the original input dtype.
    """

    SHAPE_CONFIG_KEYS = ("mm_w8a8_int8", "BlasBenchmark")

    def prepare_call(self, op, a, b):
        if op is self.torch_op:
            a_bf16, b_bf16 = a.to(torch.bfloat16), b.to(torch.bfloat16)
            return lambda: op(a_bf16, b_bf16)
        backend = sys.modules[flag_gems.mm_w8a8_int8.__module__]
        prepared = backend._prepare_mm_w8a8_int8_inputs(a, b)
        out = torch.empty(
            (a.shape[0], b.shape[1]), device=a.device, dtype=torch.bfloat16
        )
        return lambda: backend._mm_w8a8_int8_prequantized_out(*prepared, out=out)

    def get_latency(self, op, *args, **kwargs):
        # BF16 conversion, quantization, packing, and output allocation happen
        # before the inherited warmup/timing, including CUDA Graph capture.
        return super().get_latency(self.prepare_call(op, *args), **kwargs)

    def get_tflops(self, op, *args, **kwargs):
        a, b = args
        return 2 * a.shape[0] * a.shape[1] * b.shape[1]


@pytest.mark.mm_w8a8_int8
@pytest.mark.parametrize("baseline", ["torch", "flaggems"])
def test_mm_w8a8_int8(baseline):
    if not hasattr(flag_gems, "mm_w8a8_int8_out"):
        pytest.skip("mm_w8a8_int8 is not implemented by the active backend")
    bench = ParallelMmW8A8Int8Benchmark(
        input_fn=mm_input_fn,
        op_name="mm_w8a8_int8",
        torch_op=torch.mm if baseline == "torch" else flag_gems.mm,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.mm_w8a8_int8)
    print(f"BF16 baseline: {baseline}; quantization/layout preparation excluded")
    bench.run()
