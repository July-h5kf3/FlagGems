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

import flag_gems

if __package__:
    from .npu_graph import do_bench_npugraph
else:
    from npu_graph import do_bench_npugraph


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
    # Keep the historical standalone protocol (30 samples, 100 calls/graph).
    return do_bench_npugraph(fn, warmup=0, rep=0, min_samples=repeat) * 1000


SHAPES = [
    (4, 128, 8),
    (8, 256, 16),
    (64, 1024, 32),
    (64, 4096, 64),
    (64, 8192, 128),
    (128, 32768, 256),
]


def require_common_ir():
    if flag_gems.device != "npu":
        raise RuntimeError("Ascend required")
    try:
        import triton.language.extra.cann.extension as al
    except ImportError as exc:
        raise RuntimeError("FlagTree Common IR al.custom support required") from exc
    if not all(hasattr(al, name) for name in ("custom", "register_custom_op", "scope")):
        raise RuntimeError("FlagTree Common IR al.custom support required")


def make_inputs(m, n, k, dtype=torch.bfloat16, device="npu"):
    torch.manual_seed(42)
    x = torch.randn((m, n), dtype=dtype)
    s = (x.float().abs().amax(-1, keepdim=True) / 448).clamp_min(1e-8).to(dtype)
    q = (x.float() / s.float()).clamp(-448, 448).to(torch.float8_e4m3fn)
    ref = q.float() * s.float()
    return x.to(device), q.to(device), s.to(device), ref


def check_fp8_result(q, s, k, reference):
    values, indices = flag_gems.topk_w8a16_fp8(q, s, k, group_size=q.shape[-1])
    values, indices = values.cpu(), indices.cpu()
    torch.testing.assert_close(
        values, torch.topk(reference, k).values.bfloat16(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        values, torch.gather(reference, -1, indices).bfloat16(), rtol=0, atol=0
    )
    ordered = torch.sort(indices).values
    assert (ordered[..., 1:] != ordered[..., :-1]).all()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="topk_w8a16_fp8_ascend.json")
    p.add_argument("--only", type=int, default=-1)
    p.add_argument("--skip-gems", action="store_true")
    a = p.parse_args(argv)

    require_common_ir()
    rows = []
    for case, (m, n, k) in enumerate(SHAPES):
        if a.only >= 0 and a.only != case:
            continue
        x, q, s, ref = make_inputs(m, n, k)
        funcs = {
            "fp8": lambda: flag_gems.topk_w8a16_fp8(q, s, k, group_size=n),
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
                        v.cpu(),
                        torch.gather(ref, 1, i.cpu()).bfloat16(),
                        rtol=0,
                        atol=0,
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


if __name__ == "__main__":
    main()
