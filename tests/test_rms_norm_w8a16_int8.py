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

import numpy as np
import pytest
import torch

import flag_gems
from flag_gems.runtime.backend import get_customized_ops

from . import accuracy_utils as utils

GROUP_SIZE = 128
REGISTERED_OPS = dict(get_customized_ops(flag_gems.vendor_name))
INT8_IS_REGISTERED = "rms_norm_w8a16_int8" in REGISTERED_OPS


def _quantize_int8_weight(weight, group_size=GROUP_SIZE):
    grouped_weight = weight.float().reshape(-1, group_size)
    scale = (grouped_weight.abs().amax(dim=-1, keepdim=True) / 127).clamp(min=1e-8)
    weight_q = (
        (grouped_weight / scale)
        .round()
        .clamp(-128, 127)
        .to(torch.int8)
        .reshape_as(weight)
        .contiguous()
    )
    return weight_q, scale.squeeze(-1).to(weight.dtype).contiguous()


W8A16_SHAPES = [
    (2, 256),
    (32768, 256),
    (1, 2048),
    (496, 2048),
    (512, 2048),
    (1035, 2048),
    (16384, 2048),
    (1, 4096),
    (128, 4096),
    (512, 4096),
    (64, 8192),
    (1, 16384),
    (1, 32768),
]


def _assert_rms_norm_w8a16(
    inp,
    normalized_shape,
    weight_q,
    weight_scale,
    op,
    group_size=GROUP_SIZE,
    captured_output=None,
):
    eps = 1e-5
    ref_inp = utils.to_reference(inp).float()
    ref_weight_q = utils.to_reference(weight_q).float().reshape(-1)
    ref_scales = utils.to_reference(weight_scale).float().reshape(-1)
    ref_weight = (
        ref_weight_q * ref_scales.repeat_interleave(group_size)[: weight_q.numel()]
    ).reshape(normalized_shape)
    ref_out = torch.nn.functional.rms_norm(
        ref_inp, normalized_shape, ref_weight, eps=eps
    )
    if captured_output is None:
        res_out = op(
            inp,
            normalized_shape,
            weight_q,
            weight_scale,
            eps=eps,
            group_size=group_size,
        )
    else:
        res_out = captured_output
    if inp.numel() == 0:
        utils.gems_assert_equal(res_out, ref_out.to(inp.dtype))
        return
    utils.gems_assert_close(res_out, ref_out, inp.dtype)


def _run_rms_norm_w8a16_test(shape, quantize_weight, op):
    dtype = torch.bfloat16
    m, n = shape
    np.random.seed(0)
    np_inp = np.random.uniform(-0.1, 0.1, (m, n)).astype(np.float32)
    np_weight = np.random.uniform(-0.1, 0.1, (n,)).astype(np.float32)

    inp = torch.tensor(np_inp, dtype=dtype, device=flag_gems.device)
    weight = torch.tensor(np_weight, dtype=dtype, device=flag_gems.device)
    weight_q, weight_scale = quantize_weight(weight)
    _assert_rms_norm_w8a16(inp, (n,), weight_q, weight_scale, op)


@pytest.mark.rms_norm_w8a16
@pytest.mark.parametrize("shape", W8A16_SHAPES)
@pytest.mark.skipif(
    not INT8_IS_REGISTERED,
    reason="The active backend does not register grouped INT8 RMSNorm",
)
def test_rms_norm_w8a16_int8(shape):
    _run_rms_norm_w8a16_test(
        shape, _quantize_int8_weight, flag_gems.rms_norm_w8a16_int8
    )


def _quantize_int8_weight_ragged(weight, group_size=GROUP_SIZE):
    # Per-group INT8 quantization that keeps a short final group: padding
    # columns contribute zero to the group max, so every group (full or not)
    # gets exactly one scale.
    n = weight.numel()
    num_groups = (n + group_size - 1) // group_size
    padded = torch.zeros(
        num_groups * group_size, dtype=torch.float32, device=weight.device
    )
    padded[:n] = weight.float().reshape(-1)
    grouped_weight = padded.reshape(num_groups, group_size)
    scale = (grouped_weight.abs().amax(dim=-1) / 127).clamp(min=1e-8)
    weight_q = (
        (grouped_weight / scale.unsqueeze(-1))
        .round()
        .clamp(-128, 127)
        .to(torch.int8)
        .reshape(-1)[:n]
        .reshape_as(weight)
        .contiguous()
    )
    return weight_q, scale.to(weight.dtype).contiguous()


RAGGED_SHAPES = [
    (3, 65, 64),  # the second group keeps a single column
    (5, 1000, 128),  # ragged final group
    (3, 1056, 96),  # non-power-of-two group size (gather path)
    (513, 4096, 128),  # exact division control
]


@pytest.mark.rms_norm_w8a16
@pytest.mark.parametrize("m,n,group_size", RAGGED_SHAPES)
@pytest.mark.skipif(
    not INT8_IS_REGISTERED,
    reason="The active backend does not register grouped INT8 RMSNorm",
)
def test_rms_norm_w8a16_int8_ragged_shapes(m, n, group_size):
    dtype = torch.bfloat16
    np.random.seed(0)
    np_inp = np.random.uniform(-0.1, 0.1, (m, n)).astype(np.float32)
    np_weight = np.random.uniform(-0.1, 0.1, (n,)).astype(np.float32)

    inp = torch.tensor(np_inp, dtype=dtype, device=flag_gems.device)
    weight = torch.tensor(np_weight, dtype=dtype, device=flag_gems.device)
    weight_q, weight_scale = _quantize_int8_weight_ragged(weight, group_size)
    _assert_rms_norm_w8a16(
        inp,
        (n,),
        weight_q,
        weight_scale,
        flag_gems.rms_norm_w8a16_int8,
        group_size,
    )


@pytest.mark.rms_norm_w8a16
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scale_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.skipif(
    not INT8_IS_REGISTERED,
    reason="The active backend does not register grouped INT8 RMSNorm",
)
def test_rms_norm_w8a16_int8_input_and_scale_dtypes(input_dtype, scale_dtype):
    m, n, group_size = 3, 65, 64
    inp = torch.tensor(
        np.linspace(-0.1, 0.1, m * n, dtype=np.float32).reshape(m, n),
        dtype=input_dtype,
        device=flag_gems.device,
    )
    weight_q = torch.tensor(
        np.linspace(-128, 127, n).round().astype(np.int8),
        dtype=torch.int8,
        device=flag_gems.device,
    )
    weight_scale = torch.tensor(
        [0.0, -0.03125], dtype=scale_dtype, device=flag_gems.device
    )
    _assert_rms_norm_w8a16(
        inp,
        (n,),
        weight_q,
        weight_scale,
        flag_gems.rms_norm_w8a16_int8,
        group_size,
    )


@pytest.mark.rms_norm_w8a16
@pytest.mark.parametrize(
    "shape,normalized_shape",
    [((3, 2, 128), (2, 128)), ((0, 256), (256,))],
)
@pytest.mark.skipif(
    not INT8_IS_REGISTERED,
    reason="The active backend does not register grouped INT8 RMSNorm",
)
def test_rms_norm_w8a16_int8_normalized_shape(shape, normalized_shape):
    np.random.seed(0)
    dtype = torch.bfloat16
    inp = torch.tensor(
        np.random.uniform(-0.1, 0.1, shape).astype(np.float32),
        dtype=dtype,
        device=flag_gems.device,
    )
    weight = torch.tensor(
        np.random.uniform(-0.1, 0.1, normalized_shape).astype(np.float32),
        dtype=dtype,
        device=flag_gems.device,
    )
    weight_q, weight_scale = _quantize_int8_weight(weight)
    _assert_rms_norm_w8a16(
        inp,
        normalized_shape,
        weight_q,
        weight_scale,
        flag_gems.rms_norm_w8a16_int8,
    )


@pytest.mark.rms_norm_w8a16
@pytest.mark.skipif(
    not INT8_IS_REGISTERED,
    reason="The active backend does not register grouped INT8 RMSNorm",
)
def test_rms_norm_w8a16_int8_noncontiguous_inputs():
    m, n, group_size = 3, 65, 64
    inp = torch.tensor(
        np.linspace(-0.1, 0.1, m * n * 2, dtype=np.float32).reshape(m, n * 2),
        dtype=torch.bfloat16,
        device=flag_gems.device,
    )[:, ::2]
    weight_q = torch.tensor(
        (np.arange(n * 2) - n).astype(np.int8),
        dtype=torch.int8,
        device=flag_gems.device,
    )[::2]
    weight_scale = torch.tensor(
        [0.015625, 0.0, -0.03125, 0.0],
        dtype=torch.bfloat16,
        device=flag_gems.device,
    )[::2]
    _assert_rms_norm_w8a16(
        inp,
        (n,),
        weight_q,
        weight_scale,
        flag_gems.rms_norm_w8a16_int8,
        group_size,
    )


@pytest.mark.rms_norm_w8a16
@pytest.mark.skipif(
    not INT8_IS_REGISTERED,
    reason="The active backend does not register grouped INT8 RMSNorm",
)
@pytest.mark.skipif(
    flag_gems.device != "cuda", reason="Requires CUDA/HIP graph capture"
)
def test_rms_norm_w8a16_int8_graph_input_updates():
    m, n, group_size = 3, 65, 64
    inp = torch.tensor(
        np.linspace(-0.1, 0.1, m * n, dtype=np.float32).reshape(m, n),
        dtype=torch.bfloat16,
        device=flag_gems.device,
    )
    weight_q = torch.tensor(
        np.arange(n, dtype=np.int16) - 32,
        dtype=torch.int8,
        device=flag_gems.device,
    )
    weight_scale = torch.tensor(
        [0.015625, -0.03125], dtype=torch.bfloat16, device=flag_gems.device
    )
    op = flag_gems.rms_norm_w8a16_int8
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            op(inp, (n,), weight_q, weight_scale, eps=1e-5, group_size=group_size)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = op(
            inp, (n,), weight_q, weight_scale, eps=1e-5, group_size=group_size
        )
    inp.add_(0.125)
    weight_q.neg_()
    weight_scale.mul_(2.0)
    graph.replay()
    _assert_rms_norm_w8a16(
        inp,
        (n,),
        weight_q,
        weight_scale,
        op,
        group_size,
        captured_output=captured_output,
    )
