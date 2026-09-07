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

"""Explicit NPU Graph timing; results are milliseconds per operator call."""

import math
import statistics

import torch


def do_bench_npugraph(fn, warmup=1000, rep=100, calls_per_graph=100, min_samples=30):
    """Use Graph replay, approximate millisecond budgets, and >=min_samples events.

    There is no fallback to eager/event-only kernel execution. The timed region
    is always an actual torch.npu.NPUGraph replay.
    """
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("npugraph timing requires an available Ascend NPU")
    if not hasattr(torch.npu, "NPUGraph"):
        raise RuntimeError("torch.npu.NPUGraph is required for npugraph timing")
    if calls_per_graph < 1 or min_samples < 1 or warmup < 0 or rep < 0:
        raise ValueError("Invalid NPU Graph timing configuration")
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(5):
            fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        for _ in range(calls_per_graph):
            fn()
    for _ in range(5):
        graph.replay()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(5):
        graph.replay()
    end.record()
    torch.npu.synchronize()
    graph_ms = max(start.elapsed_time(end) / 5, 0.001)
    # Bound Python-side event/launch allocation for empty or extremely short graphs.
    warm_replays = min(10000, math.ceil(warmup / graph_ms))
    samples = max(min_samples, min(10000, math.ceil(rep / graph_ms)))
    for _ in range(warm_replays):
        graph.replay()
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(samples)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(samples)]
    for begin, finish in zip(starts, ends):
        begin.record()
        graph.replay()
        finish.record()
    torch.npu.synchronize()
    return (
        statistics.median(a.elapsed_time(b) for a, b in zip(starts, ends))
        / calls_per_graph
    )
