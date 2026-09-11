# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare the public INT8 API with PPU vLLM's dynamic W8A8 path.

Run from the repository root:
    DNN_VENDOR=thead USE_FLAGTUNE=0 PYTHONPATH=src python -m \
        benchmark.benchmark_mm_w8a8_int8_vllm --output result.json

Both paths include dynamic activation quantization and return BF16. Weights
and per-channel scales are shared and prepared outside the timer. This tests
model matrix shapes, not end-to-end inference. vLLM is a benchmark-only
dependency; its PPU default Triton quantizer and DeepGEMM must be available.
"""

import argparse
import importlib.metadata
import json
import math
import statistics
from pathlib import Path

import torch
import triton

import flag_gems
from benchmark.consts import model_shapes


def validate(a, b, sb, gems_out, vllm_out, quantize):
    m, k = a.shape
    n = b.shape[1]
    rows = torch.linspace(0, m - 1, min(m, 8), device=a.device).long()
    cols = torch.linspace(0, n - 1, min(n, 32), device=a.device).long()
    a_cpu = a[rows].float().cpu()
    peak = a_cpu.abs().amax(1, keepdim=True).clamp_min(1e-10)
    q_cpu = torch.round((a_cpu / peak) * 127).clamp(-127, 127).long()
    b_cpu, sb_cpu = b[:, cols].cpu().long(), sb[cols].cpu()
    reference = ((q_cpu @ b_cpu).float() * (peak / 127) * sb_cpu).bfloat16()
    torch.testing.assert_close(
        gems_out[rows][:, cols].cpu(), reference, rtol=0.02, atol=0.02
    )
    vq, vs = quantize(a, k, dtype=torch.int8, use_triton=True, use_rounding=True)
    vq_cpu = vq[rows].cpu().long()
    reference = ((vq_cpu @ b_cpu).float() * vs[rows].cpu() * sb_cpu).bfloat16()
    torch.testing.assert_close(
        vllm_out[rows][:, cols].cpu(), reference, rtol=0.02, atol=0.02
    )
    return int((vq_cpu != q_cpu).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rep-ms", type=float, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--min-speedup", type=float, default=1.0)
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=["float16", "bfloat16"],
        default=["float16", "bfloat16"],
    )
    parser.add_argument(
        "--shape", nargs=3, type=int, action="append", metavar=("M", "N", "K")
    )
    args = parser.parse_args()
    if args.rep_ms <= 0 or args.repeats <= 0:
        parser.error("rep-ms and repeats must be positive")
    if flag_gems.vendor_name != "thead":
        parser.error("this comparison requires the PPU backend")

    import vllm.envs as envs
    from vllm.model_executor.layers.quantization.utils.int8_utils import (
        per_token_group_quant_int8,
    )
    from vllm.utils.deep_gemm import (
        get_deep_gemm_config,
        int8_gemm_nt,
        is_deep_gemm_supported,
    )

    if not (
        is_deep_gemm_supported()
        and envs.VLLM_SAIL_DENSE_USE_DEEP_GEMM
        and envs.VLLM_SAIL_USE_TRITON_INT8_QUANT
    ):
        parser.error("the installed vLLM must enable its default PPU INT8 path")

    shapes = args.shape or [shape[1:] for shape in model_shapes()]
    results = []
    report = {
        "torch": torch.__version__,
        "triton": triton.__version__,
        "vllm": importlib.metadata.version("vllm"),
        "device": torch.cuda.get_device_name(),
        "baseline": "PPU vLLM Triton per-token INT8 quantization + DeepGEMM",
        "out_dtype": "bfloat16",
        "rep_ms": args.rep_ms,
        "repeats": args.repeats,
        "min_speedup": args.min_speedup,
        "results": results,
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    torch.manual_seed(5972)
    for dtype_name in args.dtypes:
        for m, n, k in shapes:
            a = torch.randn(
                m, k, device=flag_gems.device, dtype=getattr(torch, dtype_name)
            )
            b = torch.randint(-127, 128, (n, k), device=a.device, dtype=torch.int8).t()
            sb = torch.rand(n, device=a.device) * 0.01 + 0.001

            def vllm_call():
                aq, sa = per_token_group_quant_int8(
                    a, k, dtype=torch.int8, use_triton=True, use_rounding=True
                )
                out = torch.empty(m, n, device=a.device, dtype=torch.bfloat16)
                int8_gemm_nt(
                    (aq, sa),
                    (b.t(), sb[:, None]),
                    out,
                    configs=get_deep_gemm_config(m, n, k, num_groups=1),
                )
                return out

            def gems_call():
                return flag_gems.mm_w8a8_int8(a, b, sb, out_dtype=torch.bfloat16)

            row = {"dtype": dtype_name, "M": m, "N": n, "K": k}
            results.append(row)
            try:
                row["sampled_quant_code_differences"] = validate(
                    a, b, sb, gems_call(), vllm_call(), per_token_group_quant_int8
                )
                samples = {"gems": [], "vllm": []}
                for repeat in range(args.repeats):
                    calls = [("gems", gems_call), ("vllm", vllm_call)]
                    if (len(results) + repeat) % 2:
                        calls.reverse()
                    for name, call in calls:
                        samples[name].append(
                            triton.testing.do_bench_cudagraph(call, rep=args.rep_ms)
                        )
                row["samples_ms"] = samples
                row["gems_ms"] = statistics.median(samples["gems"])
                row["vllm_ms"] = statistics.median(samples["vllm"])
                row["speedup"] = row["vllm_ms"] / row["gems_ms"]
            except Exception as exc:
                row["error"] = repr(exc)
                save()
                raise
            save()
            print(json.dumps(row), flush=True)

    report["summary"] = {}
    for dtype in args.dtypes:
        rows = [row for row in results if row["dtype"] == dtype]
        report["summary"][dtype] = {
            "cases": len(rows),
            "not_slower": sum(row["speedup"] >= 1 for row in rows),
            "geomean_speedup": math.exp(
                statistics.mean(math.log(row["speedup"]) for row in rows)
            ),
            "sum_latency_speedup": sum(row["vllm_ms"] for row in rows)
            / sum(row["gems_ms"] for row in rows),
            "min_speedup": min(row["speedup"] for row in rows),
        }
    save()
    print(json.dumps(report["summary"], indent=2), flush=True)
    return int(any(row["speedup"] < args.min_speedup for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
