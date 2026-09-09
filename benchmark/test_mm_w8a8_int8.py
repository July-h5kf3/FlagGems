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
from .test_blas_perf_parallel import (
    ParallelMmW8A8Fp8Benchmark,
    _mm_w8a8_fp8_output_dtype,
    mm_input_fn,
)


class ParallelMmW8A8Int8Benchmark(ParallelMmW8A8Fp8Benchmark):
    """Adapt input preparation while retaining upstream workloads and timing."""

    def get_latency(self, op, *args, **kwargs):
        if op is not self.torch_op:
            a, b = args
            backend = sys.modules[flag_gems.mm_w8a8_int8.__module__]
            out_dtype = _mm_w8a8_fp8_output_dtype(a)
            if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError(
                    "INT8 W8A8 benchmark requires a floating non-FP8 output"
                )
            prepared = backend._prepare_mm_w8a8_int8_inputs(a, b)
            out = torch.empty(
                (a.shape[0], b.shape[1]), device=a.device, dtype=out_dtype
            )

            # Preparation is outside the inherited warmup and CUDA Graph timer.
            def op():
                return backend._mm_w8a8_int8_prequantized_out(*prepared, out=out)

            args, kwargs = (), {}
        return super().get_latency(op, *args, **kwargs)

    def get_tflops(self, op, *args, **kwargs):
        a, b = args
        return 2 * a.shape[0] * a.shape[1] * b.shape[1]


@pytest.mark.mm_w8a8_int8
def test_mm_w8a8_int8():
    if flag_gems.vendor_name != "thead" or not hasattr(flag_gems, "mm_w8a8_int8_out"):
        pytest.skip("mm_w8a8_int8 benchmark requires the THead INT8 backend")
    bench = ParallelMmW8A8Int8Benchmark(
        input_fn=mm_input_fn,
        op_name="mm_w8a8_int8",
        torch_op=torch.Tensor.mm,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.mm_w8a8_int8)
    bench.run()
