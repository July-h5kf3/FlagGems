"""Offline configuration search using the upstream BMM benchmark inputs and timer."""

import argparse
import importlib
import json
import statistics
from pathlib import Path

import torch

import flag_gems
from benchmark import base, conftest
from benchmark.test_bmm import (
    BmmW8A8Fp8Benchmark,
    _gems_bmm_w8a8_fp8,
    _torch_bmm_bf16_block_scale_baseline,
    _triton_bmm_bf16_block_scale_baseline,
)
from flag_gems.utils.libentry import LibEntry, clear_libentry_dispatch_cache


def clear_dispatch():
    module = importlib.import_module(flag_gems.bmm_block_ldc.__module__)
    for value in vars(module).values():
        if isinstance(value, LibEntry):
            clear_libentry_dispatch_cache(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("bmm_block_ldc_tuning.json")
    )
    parser.add_argument(
        "--shape-index",
        nargs="+",
        type=int,
        choices=range(9),
        help="Zero-based upstream shape indices",
    )
    parser.add_argument(
        "--current-only",
        action="store_true",
        help="Validate and time the saved configuration only",
    )
    options = parser.parse_args()
    _search(options.output, options.shape_index, options.current_only)


def _search(output, shape_indices=None, current_only=False):
    Config = conftest.BenchConfig()
    base.Config = conftest.Config = Config

    cfg = importlib.import_module(
        flag_gems.bmm_block_ldc.__module__.rsplit(".", 1)[0] + ".bmm_block_ldc_config"
    )
    Config.warm_up = 5
    Config.repetition = 20
    bench = BmmW8A8Fp8Benchmark(
        op_name="bmm_w8a8_int8_offline", torch_op=lambda: None, dtypes=[torch.bfloat16]
    )
    bench.set_shapes()
    if shape_indices is not None:
        bench.shapes = [bench.shapes[i] for i in shape_indices]
    torch.manual_seed(0)
    candidates = [
        (32, 32, 4, 1, 1),
        (32, 64, 4, 2, 1),
        (32, 128, 4, 2, 4),
        (64, 32, 4, 2, 4),
        (64, 64, 4, 2, 4),
        (64, 128, 4, 2, 8),
        (64, 128, 8, 2, 4),
        (128, 64, 4, 2, 4),
        (128, 128, 8, 2, 8),
        (64, 128, 4, 1, 1),
        (64, 128, 4, 3, 8),
        (128, 128, 4, 2, 8),
    ]
    large_candidates = [
        (64, 64, 4, 3, 4),
        (64, 128, 4, 4, 8),
        (64, 128, 4, 5, 8),
        (128, 64, 4, 3, 8),
        (128, 64, 4, 4, 8),
        (128, 64, 8, 3, 8),
        (128, 128, 8, 3, 8),
        (128, 128, 8, 4, 8),
        (256, 64, 8, 3, 8),
        (64, 128, 8, 3, 8),
        (64, 128, 4, 3, 1),
        (64, 128, 4, 3, 16),
    ]
    records = []
    for args in bench.get_input_iter(torch.bfloat16):
        a, b, aq, bq, asc, bsc, out, sm, sn, sk = args
        batch, m, k = a.shape
        n = b.shape[2]
        key = (batch, m, n, k)
        rows = torch.linspace(0, m - 1, min(m, 32), device=a.device).long()
        cols = torch.linspace(0, n - 1, min(n, 32), device=a.device).long()
        kids = torch.arange(k, device=a.device) // sk
        ad = aq[:, rows, :].float() * asc[:, rows // sm, :][:, :, kids]
        bd = bq[:, :, cols].float() * bsc[:, kids, :][:, :, cols // sn]
        ref = torch.bmm(ad, bd)
        original_ref = torch.bmm(a[:, rows, :].float(), b[:, :, cols].float())
        baselines = {
            "torch_bf16_ms": bench.get_latency(
                _torch_bmm_bf16_block_scale_baseline, *args
            ),
            "flaggems_bf16_ms": bench.get_latency(
                _triton_bmm_bf16_block_scale_baseline, *args
            ),
        }
        saved = cfg.EXACT_CONFIGS[key]
        route = "direct_reduction" if m <= 16 and n <= 32 and 0 < k <= 64 else "dot"
        search = (
            [saved]
            if current_only or route == "direct_reduction"
            else list(
                dict.fromkeys(
                    [saved] + candidates + (large_candidates if m >= 512 else [])
                )
            )
        )
        trials = []
        for candidate in search:
            clear_dispatch()
            cfg.EXACT_CONFIGS[key] = candidate
            try:
                result = _gems_bmm_w8a8_fp8(*args)
                sampled = result[:, rows, :][:, :, cols].float()
                nrms = (
                    ((sampled - ref).square().mean() / ref.square().mean())
                    .sqrt()
                    .item()
                )
                original_nrms = (
                    (
                        (sampled - original_ref).square().mean()
                        / original_ref.square().mean()
                    )
                    .sqrt()
                    .item()
                )
                if (
                    not torch.isfinite(result).all().item()
                    or nrms >= 0.10
                    or original_nrms >= 0.10
                ):
                    raise ValueError(f"NRMS={nrms}, original_NRMS={original_nrms}")
                ms = bench.get_latency(_gems_bmm_w8a8_fp8, *args)
                trials.append(
                    dict(
                        config=candidate, ms=ms, nrms=nrms, original_nrms=original_nrms
                    )
                )
                print(key, candidate, ms, nrms, flush=True)
            except Exception as exc:
                trials.append(dict(config=candidate, error=str(exc)))
                print(key, candidate, type(exc).__name__, str(exc)[:200], flush=True)
        valid = [t for t in trials if "ms" in t]
        if not valid:
            raise RuntimeError(f"No valid configuration for {key}")
        # Recheck the three fastest candidates to reduce selection noise.
        for t in sorted(valid, key=lambda t: t["ms"])[:3]:
            clear_dispatch()
            cfg.EXACT_CONFIGS[key] = t["config"]
            t["recheck_samples_ms"] = [
                bench.get_latency(_gems_bmm_w8a8_fp8, *args) for _ in range(3)
            ]
            t["recheck_ms"] = statistics.median(t["recheck_samples_ms"])
        winner = min(
            [t for t in valid if "recheck_ms" in t], key=lambda t: t["recheck_ms"]
        )
        speedups = {
            name.replace("_ms", "_speedup"): value / winner["recheck_ms"]
            for name, value in baselines.items()
        }
        records.append(
            dict(
                shape=key,
                route=route,
                aiu=cfg.AIU_OVERRIDES.get(key, True),
                pack_config=cfg.PACK_CONFIGS.get((batch, n, k)),
                baselines=baselines,
                winner=winner,
                speedups=speedups,
                meets_1_3=all(v >= 1.3 for v in speedups.values()),
                trials=trials,
            )
        )
        cfg.EXACT_CONFIGS[key] = saved
        output.write_text(json.dumps(records, indent=2))
        print("BEST", key, winner, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
