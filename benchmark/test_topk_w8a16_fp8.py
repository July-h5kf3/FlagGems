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
from .bench_topk_w8a16_fp8_ascend import (
    SHAPES,
    check_fp8_result,
    make_inputs,
    require_common_ir,
)
from .conftest import Config


def _torch_bf16_topk(x, q, scale, k):
    return torch.topk(x, k)


def _gems_fp8_topk(x, q, scale, k):
    return flag_gems.topk_w8a16_fp8(q, scale, k, group_size=q.shape[-1])


class TopKFp8W8A16Benchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "M, N, K"

    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)

    def get_input_iter(self, dtype):
        for m, n, k in self.shapes:
            x, q, scale, reference = make_inputs(
                m, n, k, dtype=dtype, device=self.device
            )
            check_fp8_result(q, scale, k, reference)
            yield x, q, scale, k

    def record_shapes(self, x, q, scale, k):
        return (*x.shape, k)


@pytest.mark.topk_w8a16_fp8
def test_topk_w8a16_fp8_npugraph(monkeypatch):
    if not Config.query:
        if flag_gems.device != "npu":
            pytest.skip("Ascend NPU Graph benchmark")
        # A missing capability on the target backend must fail, not look like a pass.
        require_common_ir()
    # This is explicitly a Graph-only test, including the default /test CI route.
    # The patch is scoped to this test and BenchmarkResult records mode=npugraph.
    monkeypatch.setattr(Config, "mode", consts.BenchMode.NPUGRAPH)
    print("TopK acceptance uses actual torch.npu.NPUGraph (mode=npugraph).")
    bench = TopKFp8W8A16Benchmark(
        op_name="topk_w8a16_fp8", torch_op=_torch_bf16_topk, dtypes=[torch.bfloat16]
    )
    bench.set_gems(_gems_fp8_topk)
    bench.run()
