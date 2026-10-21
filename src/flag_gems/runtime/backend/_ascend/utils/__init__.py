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
from collections.abc import Callable

import torch

CORE_NUM = 40

try:
    import triton.runtime.driver as driver

    CORE_NUM = driver.active.utils.get_device_properties(torch.npu.current_device())[
        "num_vectorcore"
    ]
except (ImportError, AttributeError, RuntimeError, KeyError):
    CORE_NUM = 40

__all__ = [
    "CORE_NUM",
]


def benchmark_graph(
    fn: Callable[[], object], rep: float, grad_to_none: list[torch.Tensor] | None = None
) -> float:
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        fn()
        if grad_to_none is not None:
            for tensor in grad_to_none:
                tensor.detach_().requires_grad_(True)
                tensor.grad = None
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(5):
            fn()
        end.record()
        torch.npu.synchronize()
        estimate = start.elapsed_time(end) / 5
        repeats = 1000 if estimate == 0 else max(1, int(rep / estimate))
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            for _ in range(repeats):
                if grad_to_none is not None:
                    for tensor in grad_to_none:
                        tensor.grad = None
                fn()
        torch.npu.synchronize()
        samples = []
        for _ in range(10):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            torch.npu.synchronize()
            samples.append(start.elapsed_time(end) / repeats)
        graph.reset()
    torch.npu.current_stream().wait_stream(stream)
    return statistics.median(samples)
