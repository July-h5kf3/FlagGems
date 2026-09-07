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

import argparse
import json
import statistics
from contextlib import nullcontext

import torch
import torch_npu  # noqa: F401

import flag_gems

topk_w8a16_fp8 = flag_gems.topk_w8a16_fp8
p = argparse.ArgumentParser()
p.add_argument("--output", default="topk_w8a16_fp8_ascend.json")
p.add_argument("--only", type=int, default=-1)
p.add_argument("--skip-gems", action="store_true")
a = p.parse_args()


def event_bench(fn, repeat=30):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    starts = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
    ends = [torch.npu.Event(enable_timing=True) for _ in range(repeat)]
    for st, en in zip(starts, ends):
        st.record()
        fn()
        en.record()
    torch.npu.synchronize()
    return statistics.median(st.elapsed_time(en) * 1000 for st, en in zip(starts, ends))


def graph_bench(fn, repeat=30):
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(5):
            fn()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        for _ in range(100):
            fn()
    for _ in range(5):
        graph.replay()
    torch.npu.synchronize()
    return event_bench(graph.replay, repeat) / 100


rows = []
for case, (m, n, k) in enumerate(
    [
        (4, 128, 8),
        (8, 256, 16),
        (64, 1024, 32),
        (64, 4096, 64),
        (64, 8192, 128),
        (128, 32768, 256),
    ]
):
    if a.only >= 0 and a.only != case:
        continue
    torch.manual_seed(42)
    x = torch.randn((m, n), dtype=torch.bfloat16)
    s = (x.float().abs().amax(-1, keepdim=True) / 448).clamp_min(1e-8).bfloat16()
    q = (x.float() / s.float()).clamp(-448, 448).to(torch.float8_e4m3fn)
    ref = q.float() * s.float()
    x = x.npu()
    q = q.npu()
    s = s.npu()
    funcs = {
        "fp8": lambda: topk_w8a16_fp8(q, s, k, group_size=n),
        "torch_bf16": lambda: torch.topk(x, k),
        "gems_bf16": lambda: torch.topk(x, k),
    }
    if a.skip_gems:
        funcs.pop("gems_bf16")
    row = dict(shape=[m, n], k=k)
    for name, fn in funcs.items():
        print("START", m, n, k, name, flush=True)
        with flag_gems.use_gems() if name == "gems_bf16" else nullcontext():
            if name == "gems_bf16":
                registered = "topk" in flag_gems.all_registered_ops()
                row["gems_dispatch"] = (
                    "gems_kernel" if registered else "torch_fallback_topk_disabled"
                )
                print("GEMS_DISPATCH", row["gems_dispatch"], flush=True)
            v, i = fn()
            torch.npu.synchronize()
            if name == "fp8":
                torch.testing.assert_close(
                    v.cpu(), torch.topk(ref, k).values.bfloat16(), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    v.cpu(), torch.gather(ref, 1, i.cpu()).bfloat16(), rtol=0, atol=0
                )
            if name != "fp8":
                torch.testing.assert_close(
                    v.cpu(), torch.topk(x.cpu(), k).values, rtol=0, atol=0
                )
            row[name] = {}
            row[name]["eager_event_us"] = event_bench(fn)
            row[name]["npugraph_us"] = graph_bench(fn)
            print(name, row[name], flush=True)
    for mode in ("eager_event_us", "npugraph_us"):
        row[mode + "_speedup"] = {
            b: row[b][mode] / row["fp8"][mode] for b in funcs if b != "fp8"
        }
    rows.append(row)
    open(a.output, "w").write(json.dumps(rows, indent=2))
    print("RESULT", json.dumps(row), flush=True)
