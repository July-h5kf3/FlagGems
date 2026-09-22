# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""W8A8 timings use upstream BLAS shapes, including duplicate core entries.

Default scope is prequantized: same INT8 tensors/scales, native INT8 baseline,
preallocated outputs, no quantization or layout preparation timed. Three paired
rounds alternate implementation order. W8A8_SCOPE=full/activation are explicit
legacy diagnostic scopes and are not part of the default performance result.
W8A8_BASELINE selects torch/flagtree BF16 only in the full diagnostic scope.
W8A8_COMPARE_GEMM=1 adds same-input native/public GEMM timings in activation scope.
W8A8_PACK_WEIGHT=1 with activation scope also measures an explicit packed-weight
path and its one-time preparation cost. W8A8_SINGLE_SHARED=1 selects the
experimental 128x128 packing and requires the corresponding FlagTree compiler.
W8A8_SCOPE=activation shares prequantized weights and compares activation
quantization plus GEMM against the native vLLM-MetaX mctlassEx path. Set
W8A8_NATIVE_LIBRARY to the vendor _C library providing INT8 quantization.
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
from .consts import FLOAT_DTYPES
from .test_blas_perf_parallel import (
    BlasBenchmark,
    ParallelBlasBenchmark,
    ParallelMmW8A8Fp8Benchmark,
    _mm_w8a8_fp8_output_dtype,
    mm_input_fn,
)


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
    def prequantized_latencies(self, a, b):
        """Prepare once, then time both implementations on identical INT8 tensors."""
        import mctlassEx

        backend = importlib.import_module(flag_gems.mm_w8a8_int8.__module__)
        if not hasattr(self, "native_handle"):
            self.native_handle = mctlassEx.mctlassExHandleWrapper()
            files = list(Path(mctlassEx.__file__).parent.glob("*.so"))
            print(
                "W8A8_NATIVE_ENV "
                + json.dumps(
                    {
                        "entry": "mctlassExHandleWrapper.mctlass_w8a8_scaled_mm_azp",
                        "version": importlib.metadata.version("mctlassEx"),
                        "library_sha256": {
                            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in files
                        },
                    }
                ),
                flush=True,
            )
        aq, bq, sa, sb = backend._prepare_mm_w8a8_int8_inputs(a, b)
        m, k = aq.shape
        n = bq.shape[1]
        aq = aq.contiguous()
        bq = bq.t().contiguous().t()
        sa, sb = sa.reshape(m, 1).contiguous(), sb.reshape(n, 1).contiguous()
        native_sb = sb.t()
        out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
        native_out = torch.empty_like(out)

        def gems_call():
            return flag_gems.mm_w8a8_int8_out(aq, bq, sa, sb, out=out)

        def native_call():
            self.native_handle.mctlass_w8a8_scaled_mm_azp(
                aq,
                bq,
                native_out,
                sa,
                native_sb,
                None,
                None,
                None,
                torch.cuda.current_stream().cuda_stream,
            )
            return native_out

        rows, cols = [0, m // 2, m - 1], [0, n // 2, n - 1]
        ref = (
            (aq[rows].cpu().long() @ bq[:, cols].cpu().long()).float()
            * sa[rows].cpu()
            * sb[cols].cpu().t()
        ).bfloat16()

        def validate(call):
            result = call()
            torch.testing.assert_close(result[rows][:, cols].cpu(), ref, rtol=0, atol=0)

        validate(gems_call)
        native_error = None
        if n % 16 or k % 16:
            native_error = "Native requires 16-aligned K/N; no fallback timed"
        else:
            validate(native_call)
        calls = {"flaggems_int8_gemm": gems_call}
        if native_error is None:
            calls["vllm_metax_int8_gemm"] = native_call
        for call in calls.values():
            for _ in range(2):
                call()
        torch.cuda.synchronize()
        # Ensure native launches enter the capture stream, rather than timing
        # an empty graph while work runs on a previously saved stream.
        saved_aq = aq.clone()
        for call in calls.values():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured_out = call()
            aq.zero_()
            graph.replay()
            torch.testing.assert_close(
                captured_out[rows][:, cols].cpu(), torch.zeros_like(ref), rtol=0, atol=0
            )
            aq.copy_(saved_aq)
            graph.replay()
            torch.testing.assert_close(
                captured_out[rows][:, cols].cpu(), ref, rtol=0, atol=0
            )
        del saved_aq, graph, captured_out
        rounds = {name: [] for name in calls}
        for round_index in range(3):
            order = list(calls)
            if round_index % 2:
                order.reverse()
            for name in order:
                rounds[name].append(
                    triton.testing.do_bench_cudagraph(
                        calls[name], rep=Config.repetition, return_mode="median"
                    )
                )
        common = dict(
            shape=[m, n, k],
            a_stride=list(a.stride()),
            b_stride=list(b.stride()),
            quantized_a_stride=list(aq.stride()),
            quantized_b_stride=list(bq.stride()),
            scale_a_shape=list(sa.shape),
            scale_b_shape=list(sb.shape),
            dtype=str(out.dtype),
            scope="prequantized",
            bias=False,
            timing="torch.cuda.CUDAGraph + events",
            rep_ms=Config.repetition,
            order="3 rounds alternating FlagGems/native order",
            preparation_timed=False,
            output_preallocated=True,
        )
        for name in ("flaggems_int8_gemm", "vllm_metax_int8_gemm"):
            record = dict(common, op=name, status="ok", rounds_ms=rounds.get(name))
            if name not in rounds:
                record.update(status="unavailable", error=native_error)
            print("W8A8_RECORD " + json.dumps(record), flush=True)
            if os.environ.get("W8A8_RECORD_FILE"):
                with open(os.environ["W8A8_RECORD_FILE"], "a") as f:
                    f.write(json.dumps(record) + "\n")
        return (
            statistics.median(rounds["flaggems_int8_gemm"]),
            (
                statistics.median(rounds["vllm_metax_int8_gemm"])
                if native_error is None
                else float("nan")
            ),
        )

    def activation_latency(self, op, a, b):
        backend = importlib.import_module(flag_gems.mm_w8a8_int8.__module__)
        common = dict(
            shape=[a.shape[0], b.shape[1], a.shape[1]],
            a_stride=list(a.stride()),
            b_stride=list(b.stride()),
            dtype=str(a.dtype),
            scope="activation",
            timing="torch.cuda.CUDAGraph + events",
        )

        def record(name, call=None, error=None, **extra):
            entry = dict(common, op=name, **extra)
            if error is not None:
                entry.update(status="unavailable", error=error, rounds_ms=None)
                rounds = None
            else:
                for _ in range(2):
                    call()
                torch.cuda.synchronize()
                rounds = [
                    triton.testing.do_bench_cudagraph(
                        call, rep=Config.repetition, return_mode="median"
                    )
                    for _ in range(3)
                ]
                entry.update(status="ok", rounds_ms=rounds)
            print("W8A8_RECORD " + json.dumps(entry), flush=True)
            if os.environ.get("W8A8_RECORD_FILE"):
                with open(os.environ["W8A8_RECORD_FILE"], "a") as f:
                    f.write(json.dumps(entry) + "\n")
            return statistics.median(rounds) if rounds else None

        if op is self.torch_op:
            latency = record("torch_bf16", lambda: torch.mm(a, b))
            record("flagtree_bf16", lambda: flag_gems.mm(a, b))
            return latency

        # One immutable quantized weight tensor is shared by both INT8 paths.
        _, bq, _, sb = backend._prepare_mm_w8a8_int8_inputs(a, b)
        m, k = a.shape
        n = b.shape[1]
        out = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)

        def gems_call():
            aq, sa = backend._prepare_mm_w8a8_int8_activation(a)
            return flag_gems.mm_w8a8_int8_out(aq, bq, sa, sb, out=out)

        rows = [0, m // 2, m - 1]
        cols = [0, n // 2, n - 1]
        b_sample = bq[:, cols].cpu().long()
        sb_sample = sb[cols].cpu().reshape(1, -1)

        def validate(output, aq, sa):
            ref = (
                (aq[rows].cpu().long() @ b_sample).float()
                * sa.reshape(-1)[rows].cpu()[:, None]
                * sb_sample
            ).bfloat16()
            torch.testing.assert_close(output[rows][:, cols].cpu(), ref, rtol=0, atol=0)

        aq, sa = backend._prepare_mm_w8a8_int8_activation(a)
        # Independent floating-input quantization reference for selected rows.
        ar = a[rows].float().cpu()
        peak = ar.abs().amax(1).clamp_min(1e-10)
        expected_q = (
            torch.round(ar / peak[:, None] * 127).clamp(-127, 127).to(torch.int8)
        )
        torch.testing.assert_close(aq[rows].cpu(), expected_q, rtol=0, atol=0)
        validate(gems_call(), aq, sa)

        native_q = torch.empty_like(a, dtype=torch.int8)
        native_sa = torch.empty((m, 1), device=a.device, dtype=torch.float32)
        native_out = torch.empty_like(out)
        if not hasattr(self, "native_handle"):
            import mctlassEx

            library = os.environ.get("W8A8_NATIVE_LIBRARY")
            if library:
                torch.ops.load_library(library)
            if not torch._C._dispatch_has_kernel_for_dispatch_key(
                "_C::dynamic_scaled_int8_quant", "CUDA"
            ):
                raise RuntimeError("native INT8 quantization CUDA kernel is missing")
            self.native_handle = mctlassEx.mctlassExHandleWrapper()
            files = list(Path(mctlassEx.__file__).parent.glob("*.so"))
            if library:
                files.append(Path(library))
            print(
                "W8A8_NATIVE_ENV "
                + json.dumps(
                    {
                        "entry": "mctlassExHandleWrapper.mctlass_w8a8_scaled_mm_azp",
                        "quant": "_C.dynamic_scaled_int8_quant",
                        "library_sha256": {
                            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in files
                        },
                    }
                ),
                flush=True,
            )

        def native_call():
            torch.ops._C.dynamic_scaled_int8_quant(native_q, a, native_sa, None)
            self.native_handle.mctlass_w8a8_scaled_mm_azp(
                native_q,
                bq,
                native_out,
                native_sa,
                sb.reshape(1, -1),
                None,
                None,
                None,
                torch.cuda.current_stream().cuda_stream,
            )
            return native_out

        try:
            if n % 16 or k % 16:
                raise ValueError(
                    "Native comparison requires 16-aligned K and N; "
                    "the vendor legacy wrapper uses Triton otherwise. "
                    "N=1 caused an illegal access in an isolated run; "
                    "no fallback is timed."
                )
            native_call()
            torch.cuda.synchronize()
            validate(native_out, native_q, native_sa)
            torch.testing.assert_close(
                native_sa[rows].cpu().reshape(-1), peak / 127, rtol=1e-6, atol=1e-12
            )
            delta = (native_q[rows].cpu().short() - expected_q.short()).abs()
            # Different rounding/division implementations may change boundary codes.
            if int(delta.max()) > 1:
                raise AssertionError(
                    "native activation quantization differs by more than one code"
                )
            record(
                "vllm_metax_int8",
                native_call,
                quant_mismatch_count=int((delta != 0).sum()),
                quant_sample_count=delta.numel(),
            )
        except (RuntimeError, AssertionError, ValueError) as error:
            if "illegal memory access" in str(error):
                raise
            record("vllm_metax_int8", error=str(error))
        if os.environ.get("W8A8_COMPARE_GEMM") == "1":

            def gems_gemm():
                return flag_gems.mm_w8a8_int8_out(aq, bq, sa, sb, out=out)

            validate(gems_gemm(), aq, sa)
            record("flaggems_int8_gemm", gems_gemm, scope="prequantized")

            def same_input_native_gemm():
                self.native_handle.mctlass_w8a8_scaled_mm_azp(
                    aq,
                    bq,
                    native_out,
                    sa.reshape(-1, 1),
                    sb.reshape(1, -1),
                    None,
                    None,
                    None,
                    torch.cuda.current_stream().cuda_stream,
                )
                return native_out

            try:
                if n % 16 or k % 16:
                    raise ValueError(
                        "Native INT8 requires 16-aligned K and N; no fallback timed"
                    )
                validate(same_input_native_gemm(), aq, sa)
                record(
                    "vllm_metax_int8_gemm", same_input_native_gemm, scope="prequantized"
                )
            except (RuntimeError, AssertionError, ValueError) as error:
                if "illegal memory access" in str(error):
                    raise
                record("vllm_metax_int8_gemm", error=str(error), scope="prequantized")
        if os.environ.get("W8A8_PACK_WEIGHT") == "1":
            single_shared = os.environ.get("W8A8_SINGLE_SHARED") == "1"
            packed_name = (
                "flaggems_int8_single_shared_packed"
                if single_shared
                else "flaggems_int8_packed"
            )
            weight = backend._pack_mm_w8a8_int8_weight(
                bq, sb, single_shared=single_shared
            )

            def packed_call():
                return backend._mm_w8a8_int8_packed_weight_out(a, weight, out=out)

            validate(packed_call(), aq, sa)
            record("flaggems_int8", gems_call)

            def packed_gemm():
                return backend._mm_w8a8_int8_packed_prequantized_out(
                    aq, sa, weight, out=out
                )

            validate(packed_gemm(), aq, sa)
            record(
                "flaggems_int8_gemm",
                lambda: flag_gems.mm_w8a8_int8_out(aq, bq, sa, sb, out=out),
                scope="prequantized",
            )
            record(packed_name + "_gemm", packed_gemm, scope="prequantized")
            if n % 16 == 0 and k % 16 == 0 and n > 1:

                def native_gemm():
                    self.native_handle.mctlass_w8a8_scaled_mm_azp(
                        aq,
                        bq,
                        native_out,
                        sa.reshape(-1, 1),
                        sb.reshape(1, -1),
                        None,
                        None,
                        None,
                        torch.cuda.current_stream().cuda_stream,
                    )
                    return native_out

                validate(native_gemm(), aq, sa)
                record("vllm_metax_int8_gemm", native_gemm, scope="prequantized")

            record(
                (
                    packed_name + "_weight_prepare"
                    if single_shared
                    else "flaggems_int8_weight_prepare"
                ),
                lambda: backend._pack_mm_w8a8_int8_weight(
                    bq, sb, single_shared=single_shared
                ),
                scope="weight_prepare",
            )
            return record(
                packed_name,
                packed_call,
                extra_tiled_bytes=0 if weight.tiled is None else weight.tiled.numel(),
            )
        return record("flaggems_int8", gems_call)

    def get_latency(self, op, *args, **kwargs):
        a, b = args
        if os.environ.get("W8A8_SCOPE", "prequantized") == "prequantized":
            # The public framework requests baseline then implementation latency.
            # Measure the pair together, retain only its two timing values.
            if op is self.torch_op:
                ours, native = self.prequantized_latencies(a, b)
                self._paired_gemm_latency = ours
                return native
            latency = self._paired_gemm_latency
            del self._paired_gemm_latency
            return latency
        if os.environ.get("W8A8_SCOPE") == "activation":
            return self.activation_latency(op, a, b)
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
            call = lambda: backend._mm_w8a8_int8_floating(a, b)
        else:
            prepared = backend._prepare_mm_w8a8_int8_inputs(a, b)
            out = torch.empty(
                (a.shape[0], b.shape[1]), device=a.device, dtype=torch.bfloat16
            )
            call = lambda: flag_gems.mm_w8a8_int8_out(*prepared, out=out)
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
def test_mm_w8a8_int8_metax():
    if flag_gems.vendor_name != "metax" or not hasattr(flag_gems, "mm_w8a8_int8"):
        pytest.skip("requires the MetaX W8A8 INT8 backend")
    backend = importlib.import_module(flag_gems.mm_w8a8_int8.__module__)
    print(
        "W8A8_ENV "
        + json.dumps(
            {
                "torch": torch.__version__,
                "triton": triton.__version__,
                "flagtree": importlib.metadata.version("flagtree"),
                "benchmark_source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
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


class ParallelMmW8A8Int8Benchmark(ParallelBlasBenchmark):
    """W8A8 INT8 workloads from prequantized INT8 inputs.

    Use --mode cudagraph --level comprehensive --dtypes bfloat16 --dtypes
    float16 to cover the reference PR's 121 configurations and two B layouts.
    The Torch baseline uses BF16, independently of the original input dtype.
    """

    SHAPE_CONFIG_KEYS = ("mm_w8a8_int8", "BlasBenchmark")

    def prepare_call(self, op, a, b):
        if op is self.torch_op:
            a_bf16, b_bf16 = a.to(torch.bfloat16), b.to(torch.bfloat16)
            return lambda: op(a_bf16, b_bf16)
        # Quantize both inputs offline; time only the public scaled INT8 GEMM.
        activation = a.float()
        peak_a = activation.abs().amax(dim=1).clamp_min(1e-10)
        scale_a = peak_a * (1.0 / 127.0)
        aq = (
            (activation / peak_a[:, None] * 127.0)
            .round()
            .clamp(-127, 127)
            .to(torch.int8)
        )
        weight = b.float()
        peak = weight.abs().amax(dim=0).clamp_min(1e-10)
        scale_b = peak * (1.0 / 127.0)
        bq = (weight / peak[None, :] * 127.0).round().clamp(-127, 127).to(torch.int8)
        bq = bq.t().contiguous().t()
        out = torch.empty(
            (a.shape[0], b.shape[1]), device=a.device, dtype=torch.bfloat16
        )
        return lambda: flag_gems.mm_w8a8_int8_out(aq, bq, scale_a, scale_b, out=out)

    def get_latency(self, op, *args, **kwargs):
        # Input quantization, layouts, baseline casts and output allocation are offline.
        # Time INT8 GEMM, scaling and any necessary reduction kernels.
        return super().get_latency(self.prepare_call(op, *args), **kwargs)

    def get_tflops(self, op, *args, **kwargs):
        a, b = args
        return 2 * a.shape[0] * a.shape[1] * b.shape[1]


@pytest.mark.mm_w8a8_int8
def test_mm_w8a8_int8():
    if flag_gems.vendor_name in ("thead", "metax") or not hasattr(
        flag_gems, "mm_w8a8_int8_out"
    ):
        pytest.skip("mm_w8a8_int8 is not implemented by the active backend")
    bench = ParallelMmW8A8Int8Benchmark(
        input_fn=mm_input_fn,
        op_name="mm_w8a8_int8",
        torch_op=torch.mm,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.mm_w8a8_int8)
    print("BF16 baseline: torch; A/B quantization offline; INT8 GEMM and scaling timed")
    bench.run()


class ParallelMmW8A8Int8THeadBenchmark(ParallelMmW8A8Fp8Benchmark):
    """Reuse upstream workloads and timing with prequantized INT8 inputs."""

    def get_latency(self, op, *args, **kwargs):
        if op is not self.torch_op:
            a, b = args
            out_dtype = _mm_w8a8_fp8_output_dtype(a)
            if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise ValueError(
                    "INT8 W8A8 benchmark requires a floating non-FP8 output"
                )
            peak_a = a.float().abs().amax(dim=1).clamp_min(1e-10)
            scale_a = peak_a * (1.0 / 127.0)
            a_q = (
                torch.round(a.float() / peak_a[:, None] * 127)
                .clamp(-127, 127)
                .to(torch.int8)
            )
            peak_b = b.float().abs().amax(dim=0).clamp_min(1e-10)
            scale_b = peak_b * (1.0 / 127.0)
            b_q = (
                torch.round((b.float() / peak_b[None, :]) * 127)
                .clamp(-127, 127)
                .to(torch.int8)
                .t()
                .contiguous()
                .t()
            )
            out = torch.empty(
                (a.shape[0], b.shape[1]), device=a.device, dtype=out_dtype
            )

            # Both quantizations, scales and the output allocation are untimed.
            # Replay invokes the public scaled-input API, including any copies.
            def op():
                return flag_gems.mm_w8a8_int8_out(a_q, b_q, scale_a, scale_b, out=out)

            args, kwargs = (), {}
        return super().get_latency(op, *args, **kwargs)

    def get_tflops(self, op, *args, **kwargs):
        a, b = args
        return 2 * a.shape[0] * a.shape[1] * b.shape[1]


@pytest.mark.mm_w8a8_int8
def test_mm_w8a8_int8_thead():
    if flag_gems.vendor_name != "thead" or not hasattr(flag_gems, "mm_w8a8_int8_out"):
        pytest.skip("mm_w8a8_int8 benchmark requires the THead INT8 backend")
    bench = ParallelMmW8A8Int8THeadBenchmark(
        input_fn=mm_input_fn,
        op_name="mm_w8a8_int8",
        torch_op=torch.Tensor.mm,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.mm_w8a8_int8)
    bench.run()
