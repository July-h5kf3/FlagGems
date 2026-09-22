# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""W8A8 timings use upstream BLAS shapes, including duplicate core entries.

W8A8_BASELINE=flagtree compares the unchanged MetaX BF16 mm compiled by
FlagTree; the default is torch.mm BF16. W8A8_SCOPE=full includes fresh
quantization, while the default prequantized scope matches PR #5972.
"""
import hashlib
import importlib
import importlib.metadata
import json
import os
import statistics
from pathlib import Path

import pytest
import torch
import triton

import flag_gems

from .conftest import Config
from .test_blas_perf_parallel import BlasBenchmark, mm_input_fn


def selected_mm_input_fn(*args, **kwargs):
    layout = os.environ.get("W8A8_LAYOUT")
    if layout not in (None, "nn", "nt"):
        raise ValueError("W8A8_LAYOUT must be nn or nt")
    for a, b in mm_input_fn(*args, **kwargs):
        if layout == "nn":
            b = b.contiguous()
        elif layout == "nt":
            b = b.t().contiguous().t()
        yield a, b


class MmW8A8Int8Benchmark(BlasBenchmark):
    def get_latency(self, op, *args, **kwargs):
        a, b = args
        backend = importlib.import_module(flag_gems.mm_w8a8_int8.__module__)
        baseline = os.environ.get("W8A8_BASELINE", "torch")
        scope = os.environ.get("W8A8_SCOPE", "prequantized")
        if baseline not in ("torch", "flagtree", "both") or scope not in (
            "prequantized",
            "full",
        ):
            raise ValueError("unknown W8A8_BASELINE or W8A8_SCOPE")
        if op is self.torch_op:
            call = (
                (lambda: flag_gems.mm(a, b))
                if baseline == "flagtree"
                else (lambda: torch.mm(a, b))
            )
        elif scope == "full":
            call = lambda: flag_gems.mm_w8a8_int8(a, b)
        else:
            prepared = backend._prepare_mm_w8a8_int8_inputs(a, b)
            out = torch.empty(
                (a.shape[0], b.shape[1]), device=a.device, dtype=torch.bfloat16
            )
            call = lambda: backend._mm_w8a8_int8_prequantized_out(*prepared, out=out)
        if op is not self.torch_op:
            # Sample independent CPU INT64 references for every workload,
            # outside both warmup and timing; full API tests live in tests/.
            rows = [0, a.shape[0] // 2, a.shape[0] - 1]
            cols = [0, b.shape[1] // 2, b.shape[1] - 1]
            ar = a[rows].float().cpu()
            br = b[:, cols].float().cpu()
            pa = ar.abs().amax(1).clamp_min(1e-10)
            pb = br.abs().amax(0).clamp_min(1e-10)
            aq = torch.round(ar / pa[:, None] * 127).clamp(-127, 127).long()
            bq = torch.round(br / pb[None, :] * 127).clamp(-127, 127).long()
            ref = (
                (aq @ bq).float()
                * (pa * (1.0 / 127))[:, None]
                * (pb * (1.0 / 127))[None, :]
            ).bfloat16()
            actual = call()[rows][:, cols].cpu()
            torch.testing.assert_close(actual, ref, rtol=0, atol=0)
        for _ in range(2):
            call()
        torch.cuda.synchronize()
        rounds = [
            triton.testing.do_bench_cudagraph(
                call, rep=Config.repetition, return_mode="median"
            )
            for _ in range(3)
        ]
        record = dict(
            shape=[a.shape[0], b.shape[1], a.shape[1]],
            a_stride=list(a.stride()),
            b_stride=list(b.stride()),
            dtype=str(a.dtype),
            baseline=baseline,
            scope=scope,
            op="baseline" if op is self.torch_op else "int8",
            timing="torch.cuda.CUDAGraph + events",
            rounds_ms=rounds,
        )
        print("W8A8_RECORD " + json.dumps(record), flush=True)
        path = os.environ.get("W8A8_RECORD_FILE")
        if path:
            with open(path, "a") as f:
                f.write(json.dumps(record) + "\n")
        if baseline == "both" and op is self.torch_op:
            for _ in range(2):
                flag_gems.mm(a, b)
            extra = [
                triton.testing.do_bench_cudagraph(
                    lambda: flag_gems.mm(a, b),
                    rep=Config.repetition,
                    return_mode="median",
                )
                for _ in range(3)
            ]
            record = dict(record, baseline="flagtree", rounds_ms=extra)
            print("W8A8_RECORD " + json.dumps(record), flush=True)
            if path:
                with open(path, "a") as f:
                    f.write(json.dumps(record) + "\n")
        return statistics.median(rounds)

    def get_tflops(self, op, *args, **kwargs):
        a, b = args
        return 2 * a.shape[0] * a.shape[1] * b.shape[1]


@pytest.mark.mm_w8a8_int8
def test_mm_w8a8_int8():
    if flag_gems.vendor_name not in ("metax", "thead") or not hasattr(
        flag_gems, "mm_w8a8_int8"
    ):
        pytest.skip("requires a W8A8 INT8 backend")
    backend = importlib.import_module(flag_gems.mm_w8a8_int8.__module__)
    print(
        "W8A8_ENV "
        + json.dumps(
            {
                "torch": torch.__version__,
                "triton": triton.__version__,
                "flagtree": importlib.metadata.version("flagtree"),
                "backend_source_sha256": hashlib.sha256(
                    Path(backend.__file__).read_bytes()
                ).hexdigest(),
                "device": torch.cuda.get_device_name(0),
                "seed": 20260911,
                "timing": "torch.cuda.CUDAGraph + events; 3 medians; rep in milliseconds",
            }
        ),
        flush=True,
    )
    torch.manual_seed(20260911)
    bench = MmW8A8Int8Benchmark(
        input_fn=selected_mm_input_fn,
        op_name="mm_w8a8_int8",
        torch_op=torch.mm,
        dtypes=[torch.bfloat16],
    )
    bench.set_gems(flag_gems.mm_w8a8_int8)
    bench.run()
