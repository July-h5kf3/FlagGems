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

from pathlib import Path

import pytest
import torch
import yaml

import flag_gems

from . import base
from .conftest import Config
from .consts import BenchLevel
from .test_blas_perf_parallel import ParallelBenchmarkMixin


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
EPSILON = 1e-5


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


def _torch_rms_norm_w8a16(x, normalized_shape, weight_q, weight_scale, weight_ref):
    return torch.nn.functional.rms_norm(x, normalized_shape, weight_ref, eps=EPSILON)


def _gems_rms_norm_w8a16_int8(
    x, normalized_shape, weight_int8, weight_scale, weight_ref
):
    return flag_gems.rms_norm_w8a16_int8(
        x,
        normalized_shape,
        weight_int8,
        weight_scale,
        eps=EPSILON,
        group_size=GROUP_SIZE,
    )


class RmsNormW8A16Benchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "M, N"
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))
    DEFAULT_SHAPES = [
        (2, 256),
        (256, 256),
        (32768, 256),
        (1, 2048),
        (496, 2048),
        (512, 2048),
        (1035, 2048),
        (16384, 2048),
    ]
    quantize_weight = None

    def set_shapes(self, shape_file_path=None):
        shape_file_path = shape_file_path or self.DEFAULT_SHAPE_FILES
        if Path(shape_file_path).resolve() != Path(self.DEFAULT_SHAPE_FILES).resolve():
            super().set_shapes(shape_file_path)
            return
        # The shared core YAML also has a generic Benchmark entry with 1D shapes.
        self.shapes = list(self.DEFAULT_SHAPES)
        self.shape_desc = self.DEFAULT_SHAPE_DESC
        if Config.bench_level == BenchLevel.COMPREHENSIVE and not Config.query:
            self.shapes = list(dict.fromkeys(self.shapes + self.set_more_shapes()))

    def set_more_shapes(self):
        # An explicit shape file must remain authoritative for manual shards.
        if (
            Path(Config.shape_file).resolve()
            != Path(self.DEFAULT_SHAPE_FILES).resolve()
        ):
            return []
        shape_path = Path(__file__).parent / "models_shapes" / "rms_norm_w8a16.yaml"
        with shape_path.open(encoding="utf-8") as shape_file:
            shape_config = yaml.safe_load(shape_file)
        return [tuple(shape) for shape in shape_config[self.op_name]["shapes"]]

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            _, n = shape
            x = torch.randn(shape, dtype=dtype, device=self.device)
            weight = torch.randn(n, dtype=dtype, device=self.device)
            weight_q, weight_scale = self.quantize_weight(weight)
            # Prepare the BF16 baseline weight before either implementation is timed.
            weight_ref = (
                weight_q.float()
                * weight_scale.float().repeat_interleave(GROUP_SIZE)[:n]
            ).to(dtype)
            yield x, (n,), weight_q, weight_scale, weight_ref


class RmsNormInt8W8A16Benchmark(ParallelBenchmarkMixin, RmsNormW8A16Benchmark):
    SHAPE_CONFIG_KEYS = ("rms_norm_w8a16",)
    RECORD_PARALLEL_RESULTS = True
    quantize_weight = staticmethod(_quantize_int8_grouped)
