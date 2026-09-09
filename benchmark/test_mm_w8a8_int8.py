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

from .consts import FLOAT_DTYPES
from .test_blas_perf_parallel import ParallelMmW8A8Fp8Benchmark, mm_input_fn

# Use the NVIDIA W8A8 benchmark's shapes, dtypes, layouts, baseline, and timer.
# The shared class prepares INT8 tensors/scales outside the timed GEMM path.


@pytest.mark.mm_w8a8_int8
def test_mm_w8a8_int8():
    if flag_gems.vendor_name != "thead" or not hasattr(flag_gems, "mm_w8a8_int8_out"):
        pytest.skip("mm_w8a8_int8 benchmark requires the THead INT8 backend")
    bench = ParallelMmW8A8Fp8Benchmark(
        input_fn=mm_input_fn,
        op_name="mm_w8a8_int8",
        torch_op=torch.Tensor.mm,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.mm_w8a8_int8)
    bench.run()
