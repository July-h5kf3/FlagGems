# Ascend W8A8 MM (draft)

The Ascend backend implements `mm_w8a8_fp8`/`mm_w8a8_fp8_out` with INT8 inputs
and per-row/per-column FP32 scales. This backend uses INT8 Cube instructions;
it does not imply native FP8 matrix multiplication on Ascend 910B.

## Implementation

- Shape-dependent tiling using the existing `tl.dot` kernels.
- Exact INT32 accumulation followed by FP32 row/column scaling and an output cast.
- No custom-op registration, local AscendC compilation, or compiler IR rewriting.
- Native mixed kernels use one Cube-to-Vector workspace slot for correct
  persistent tile reuse on CANN 9.1.

## Local validation

Before removal of the local AscendC fragments, the CI-preparation snapshot passed
`pre-commit run --all-files`, all rule-check
scripts, and the repository `tools/test-op.sh` on CANN 9.0 with
Torch 2.10.0+cpu, torch-npu 2.10.0, and FlagTree 0.6.1+ascend3.5:
225 tests passed in normal mode and 225 passed with `--ref=cpu --quick`.
Five installed-wheel smoke checks also passed on that earlier implementation.
These results do not validate the current implementation; current validation
is recorded in the PR.

### Known blockers before ready for review

1. Backend tests on the CANN 8.5 CI environment remain unverified.
2. Re-run all 433 performance shapes on the submitted CI-compatible source and
   pinned compiler. The reference measurements below belong to an earlier source.

## Historical performance evidence (not this commit)

`reference_results/mm_w8a8_ascend_pr5972.csv` contains all 433 shapes from
FlagGems PR #5972, head `9fe20332b9407eaa88b14c613b316551e49d9ba2`.
On Ascend910B4-1, the measured operator source SHA256 was
`e376d807218f0f3b02286afaca7e4216218852d16e56216c46d7d387a413a0e1`.
The environment used Torch 2.10.0+cpu, torch-npu 2.10.0.post2,
FlagTree 0.6.0+ascend.gitf56cd1bd and CANN 9.0, with additional CANN 9.1 compiler
paths available. These are different from the CI-preparation environment above.

All 433 shapes passed their respective numerical checks and were faster by
six-sample median than Torch and default Ascend FlagGems BF16 MM. Geometric mean
speedups were 1.3307x versus Torch and 3.7976x versus default FlagGems.
Those gains have not yet been revalidated on the submitted source.

Timing used NPUGraph replay. MM, both output scales, and the final cast were
included; input quantization, padding, input/scale packing, compilation,
autotuning, and explicit allocations were excluded. Torch and FlagGems used the
faster of equivalent row/column-major B layouts. This is not end-to-end latency.
The quantized operator and BF16 baselines have different input precision semantics.

## Reproduce on a configured Ascend environment

```bash
# Source your matching CANN environment and activate the configured Python venv.
export FLAGTREE_BACKEND=ascend
export ASCEND_RT_VISIBLE_DEVICES=2  # choose an available device on your machine
export PYTHONPATH="$PWD/src"
python -m pytest -q tests/test_mm_w8a8_fp8.py
CHANGED_FILES=tests/test_mm_w8a8_fp8.py bash tools/test-op.sh local-mm

# Prepared-input kernel benchmark; the built-in list is a smaller smoke suite.
DTYPE=bf16 GEMS=1 python benchmark/bench_mm_w8a8_ascend.py mm-result.json
```

`SHAPES` can override the benchmark list with comma-separated `MxNxK` values.
The full shape list is in `mm_w8a8_ascend_shapes.json`. The standalone smoke
benchmark is not an upstream CI benchmark test and is not identical to the
historical shared-buffer three-way comparison harness.
