# PPU INT8 block BMM

Repository: `/data/ldc/ops_work/FlagGems`, branch `PPU/ldc-bmm-block`.
Public APIs: `flag_gems.bmm_block_ldc` and `flag_gems.bmm_w8a8_int8`.

```python
out = flag_gems.bmm_w8a8_int8(
    A, B, A_scale, B_scale,
    block_size=(128, 128, 128),
    out_dtype=torch.bfloat16,
)
```

A is signed INT8 `[batch,M,K]`; B is signed INT8 `[batch,K,N]`.
A_scale is `[batch,ceil(M/block_m),ceil(K/block_k)]`; B_scale is
`[batch,ceil(K/block_k),ceil(N/block_n)]`. Set block_m=1 for per-row scales.
Scales accept FP32, BF16 and FP16. Input and scale strides are supported.
A supplied output must be contiguous with the requested shape, device and dtype.
Only power-of-two block_n >=16 and block_k=128 are supported. Zero K writes
zeros; empty output returns without launching a kernel.

Each K block computes INT8 dot into INT32, converts the partial to BF16,
combines A/weight scales into a BF16 row vector, then broadcasts it to the
output tile. Products and cross-block accumulation round to BF16. This also
applies when out_dtype is FP32. A direct integer-reduction specialization
handles very small matrices. Sampled NRMS below 0.10 is an empirical check,
not a guarantee for arbitrary distributions or long sequences of cancellation.

Aligned shapes use unmasked loads/stores. Suitable aligned A inputs use AIU
asynchronous loading. For M,N >=256, a Triton kernel transposes weights into
a fresh temporary on every call, enabling dual AIU loading. Temporary allocation,
packing and compute are all inside the timed operator. No persistent packed-input
cache or torch compute fallback is used. The AIU route conservatively requires contiguous A and
128-byte alignment of A's base, row stride and batch stride; other layouts use
ordinary loads. The emitted LLVM contains
`ppu.cp.async.aiu.bulk.tensor.shared.global.padz.swzl.zfill.2d.b8`.

## Environment

SSH: `zhiyuan-ppu`; container: `codex-ldc-block-bmm`.
Host physical PPU 14 maps to container CUDA 0 (PPU-ZW810E).
SDK: `2.1.0-a5f865`. PyTorch: `2.10.0`.
FlagTree was built from upstream main commit
`d96f5339bf73752cf5e3b83e76bdeff055966de2`, installed as
`flagtree 0.6.0+ppu.gitd96f5339`, Triton `3.6.0`.
Source: `/data/ldc/ops_work/FlagTree`; log: `/data/ldc/ops_work/flagtree-build.log`.
The source tree was clean at build time. No unrelated checkout was changed.

## Benchmark and tuning

The benchmark is the existing `benchmark/test_bmm.py` at FlagGems base commit
`f0d154f146d2ba7ad569832cde97087e80cd09dc`, class `BmmW8A8Fp8Benchmark`.
Its nine shapes, BF16 source inputs, quantization structure and `base.Benchmark`
timer are retained. PPU uses signed INT8 range and rounding; NVIDIA retains FP8.
Input quantization is outside timing, as upstream. Two benchmark variants compare
against native `torch.bmm(...,out=...)` and FlagGems `triton_bmm_out`.
The inherited “Torch Latency” table heading means the selected baseline;
read the operator suffix to identify which baseline was measured.
The parametrized pytest operator marker is identical to Benchmark.op_name:
`bmm_w8a8_int8_vs_flaggems_bf16` or `bmm_w8a8_int8_vs_torch_bf16` on PPU.
Each JSON entry therefore contains its own test status and nine timing results;
there is no separate misleading FP8 status entry. NVIDIA uses
`bmm_w8a8_fp8_vs_triton_bf16`. Select the tests with `-k w8a8` or the exact marker.

The target is >=1.3x **both baselines for each of the nine shapes**.
This target has not been achieved; see the accompanying results and raw JSON.
No aggregate speedup is substituted for this requirement.

`bmm_block_ldc_config.py` provides exact shapes and size-based fallbacks without
runtime autotuning. Configurations are tuned for contiguous tensors and
128x128x128 scale blocks. Other supported layouts are correctness paths.

`tools/tune_bmm_block_ldc.py` reuses the upstream input generator and latency
method. It includes the saved configuration in the search, clears LibEntry
launch caches before switching candidates, and rechecks the fastest three
candidates three times, selecting by median. Launch options such as num_stages
are absent from the LibEntry dispatch key, so omitting this clearing can time
an earlier configuration. The direct-reduction route has fixed launch parameters
and is measured once instead of pretending GEMM tile parameters tune that route.
The tool writes results; it does not automatically replace the saved config.
Independent full benchmark runs are required to confirm a proposed change.

```bash
# Run inside the configured container, with cwd /data/ldc/ops_work/FlagGems.
PYTHONPATH=src pytest -q -s tests/test_bmm.py -k w8a8 --tb=short
PYTHONPATH=src pytest -q -s benchmark/test_bmm.py -k w8a8 \
  --warmup 20 --iter 100 --record json --output /data/ldc/ops_work/bmm-results.json
PYTHONPATH=src:. python tools/tune_bmm_block_ldc.py \
  --shape-index 1 2 --output /data/ldc/ops_work/bmm-tuning.json
# A short check of the saved configuration, without searching:
PYTHONPATH=src:. python tools/tune_bmm_block_ldc.py \
  --shape-index 1 --current-only --output /data/ldc/ops_work/bmm-current.json
```

Accuracy tests cover the upstream nine shapes (sampled against dequantized FP32
and original BF16 inputs), strided/per-row scales, odd sizes, zero K, empty output,
argument validation, misaligned base/row storage, broadcast batch strides, and signed INT8 extremes.
Performance-search experiments that failed to improve the operator are retained
outside the repository under `/data/ldc/ops_work/` for diagnosis, not dispatched
by the production API.

## Measured performance for review

PPU-ZW810E / SDK 2.1 / source-built FlagTree d96f5339, 2026-09-09.
Upstream kernel mode, warmup=20 ms, rep=100 ms, median latency.
The two baseline cases time INT8 independently.

| (B,M,N,K) | INT8 vs Gems run (us) | vs FlagGems BF16 | INT8 vs torch run (us) | vs torch BF16 |
|---|---:|---:|---:|---:|
| (2,16,32,64) | 2.32 | 1.207x | 2.32 | 1.707x |
| (4,64,64,128) | 2.80 | 1.243x | 2.80 | 1.471x |
| (2,128,128,256) | 3.76 | 1.181x | 3.96 | 1.384x |
| (4,256,256,512) | 8.36 | 1.292x | 8.56 | 1.224x |
| (8,512,512,1024) | 41.88 | 1.784x | 41.88 | 1.262x |
| (8,1024,1024,2048) | 224.64 | 1.516x | 224.92 | 1.531x |
| (8,2048,2048,4096) | 1549.64 | 1.561x | 1559.56 | 1.331x |
| (4,4096,4096,16384) | 11338.76 | 1.684x | 11342.78 | 1.774x |
| (8,4096,4096,16384) | 22725.84 | 1.655x | 22678.84 | 1.784x |

With one invocation per shape, sum(baseline latency) / sum(INT8 latency) is
**1.659x vs FlagGems BF16** and **1.758x vs torch BF16**.
Only **4/9 shapes** individually exceed both baselines by 1.3x.
The two largest shapes account for about 95% of total INT8 time, so this
aggregate is not a claim that each shape meets the target.
NVIDIA behavior is retained in code but was not hardware-tested on this PPU machine.
