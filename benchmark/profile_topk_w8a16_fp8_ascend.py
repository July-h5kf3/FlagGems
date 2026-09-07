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
import csv
import json
from pathlib import Path

import torch
import torch_npu

import flag_gems

p = argparse.ArgumentParser()
p.add_argument("--only", type=int, default=-1)
p.add_argument("--root", default="profile-topk")
a = p.parse_args()
results = []
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
    x, q, s = x.npu(), q.npu(), s.npu()
    row = dict(shape=[m, n], k=k)
    for name, fn in {
        "fp8": lambda: flag_gems.topk_w8a16_fp8(q, s, k, group_size=n),
        "torch_bf16": lambda: torch.topk(x, k),
    }.items():
        for _ in range(5):
            fn()
        torch.npu.synchronize()
        folder = Path(a.root) / f"case{case}_{name}"
        print("PROFILE", case, name, flush=True)
        previous_csvs = set(folder.rglob("kernel_details.csv"))
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(folder)),
            experimental_config=torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1
            ),
        ):
            for _ in range(20):
                fn()
            torch.npu.synchronize()
        paths = [
            path
            for path in folder.rglob("kernel_details.csv")
            if path not in previous_csvs
        ]
        assert len(paths) == 1, paths
        with paths[0].open() as f:
            records = list(csv.DictReader(f))
        print("COLUMNS", list(records[0]), "ROWS", len(records), flush=True)
        counts = {}
        totals = {}
        for rec in records:
            key = rec.get("Name", rec.get("Type", "unknown"))
            counts[key] = counts.get(key, 0) + 1
            totals[key] = totals.get(key, 0) + float(rec["Duration(us)"])
        assert all(v == 20 for v in counts.values()), counts
        row[name] = {
            "device_us": sum(totals.values()) / 20,
            "kernel_counts": counts,
            "kernel_us": {key: val / 20 for key, val in totals.items()},
            "csv": str(paths[0]),
        }
        print("PROFILE_RESULT", name, json.dumps(row[name]), flush=True)
    row["speedup"] = row["torch_bf16"]["device_us"] / row["fp8"]["device_us"]
    results.append(row)
    Path(a.root).mkdir(exist_ok=True)
    (Path(a.root) / "summary.json").write_text(json.dumps(results, indent=2))
    print("RESULT", json.dumps(row), flush=True)
