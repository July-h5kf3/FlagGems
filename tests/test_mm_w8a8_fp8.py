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

from . import accuracy_utils as utils


def _thead_w8a8_fp8_available():
    return (
        getattr(flag_gems, "vendor_name", None) == "thead"
        and hasattr(torch, "float8_e4m3fn")
        and hasattr(flag_gems, "mm_w8a8_fp8")
    )


def _mm_w8a8_int8_reference(a, b):
    a_fp32 = a.float()
    a_scale = a_fp32.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    a_q = torch.round(a_fp32 / a_scale[:, None]).clamp(-127, 127)

    b_fp32 = b.float()
    b_scale = b_fp32.abs().amax(dim=0).clamp_min(1e-8) / 127.0
    b_q = torch.round(b_fp32 / b_scale[None, :]).clamp(-127, 127)

    return (a_q @ b_q) * a_scale[:, None] * b_scale[None, :]


@pytest.mark.mm_w8a8_fp8
@pytest.mark.parametrize(
    "M, N, K",
    [
        (1, 16, 16),
        (16, 1, 128),
        (256, 1, 2048),
        (2, 32, 32),
        (8, 64, 64),
        (16, 128, 64),
        (32, 128, 128),
        (64, 256, 128),
        (128, 256, 256),
        (192, 512, 512),
        (256, 768, 1024),
        (512, 1024, 1024),
        # Qwen3.5-35B-A3B-p32768d1024 families from FlagGems#3821
        (16, 1, 2048),
        (16, 64, 2048),
        (16, 256, 2048),
        (16, 1024, 2048),
        (16, 2048, 512),
        (16, 2048, 4096),
        (16, 9216, 2048),
        (16, 12288, 2048),
        (1, 248320, 2048),
    ],
)
@pytest.mark.skipif(
    not _thead_w8a8_fp8_available(),
    reason="mm_w8a8_fp8 requires the THead/PPU backend",
)
def test_mm_w8a8_fp8(M, N, K):
    dtype = torch.bfloat16
    torch.manual_seed(0)

    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    ref_out = utils.to_reference(_mm_w8a8_int8_reference(mat1, mat2), True)

    res_out = flag_gems.mm_w8a8_fp8(mat1, mat2, out_dtype=dtype)
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)
    res_out_reused = flag_gems.mm_w8a8_fp8_out(mat1, mat2, out=out)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)
    utils.gems_assert_close(res_out_reused, ref_out, dtype, reduce_dim=K)
