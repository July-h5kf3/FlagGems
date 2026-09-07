# Ascend W8A16 FP8 TopK

`flag_gems.topk_w8a16_fp8(x, x_scale, k, dim=-1, largest=True, sorted=True,
group_size=128)` selects along the last dimension of contiguous finite FP8
E4M3FN/E5M2 storage and returns BF16 values plus int64 indices. Scales are
contiguous FP16/BF16/FP32 tensors on the same NPU. Ties may return any valid,
unique indices; `sorted=False` may still return sorted results.

Precision is compared with TopK of `float32(decode(x))*float32(scale)`, rounded
to BF16 at the output. This does not promise identical results to the original
unquantized BF16 input. Quantization/transfers are outside operator timing.

## Implementation

- Triton handles data access, scheduling, decoding/scaling and output. Inline
  AscendC fragments are linked into its kernels through `al.custom`.
- Small K uses `Sort32` and four-way `MrgSort`, retaining only
  `max(8,next_power_of_2(K))` entries of each run when padded K <=32.
- Large row-scale cases search the 8-bit ordered-code cutoff in eight steps
  using vector comparisons and `GatherMask`. Values strictly above the cutoff
  plus sufficient cutoff ties form exactly K candidates; a second kernel sorts
  and decodes them. No approximate cutoff or full BF16 temporary is used.
- E5M2 bits map to the high byte of FP16. E4M3FN bits are repositioned, converted
  to FP32, then scaled by 256, preserving subnormal intermediates.
- Compiled launch functions are cached by kernel/grid/device/constexpr values,
  dtype and alignment. Streams and tensor contents remain dynamic. The only
  tensor cache is an immutable index table keyed by row length/device.
- Grids use the backend physical Vector Core count and explicit row loops.
  Automatic multibuffering is disabled. At N=32768 the selector requests
  180224 UB bytes plus small wrapper state. Offset inputs use the general path
  when the 32-byte-aligned row selector cannot be used.

The large-row path supports N=4096/8192/16384/32768 and power-of-two K in [8,512].
The general path uses local blocks <=2048 and requires K <= that block size and
padded merged candidates <=4096. N and group size must be positive. Unsupported
sizes fail explicitly.

## Requirements

Validated on Ascend 910B4-1/aarch64, CANN 9.0.0, torch 2.10.0+cpu,
torch_npu 2.10.0.post2, and FlagTree 0.6.0+ascend.gitf56cd1bd (Triton 3.5.1).
FlagTree Common IR `al.custom` support is required; native FP8 arithmetic is not.
Unsupported compiler builds receive an explicit capability error when this
operator is called, while normal FlagGems import remains available.

`ascendc/compile_topk.py` provides CCEC bitcode compilation and the CANN 9.0
CustomOp ABI/lowering adapter. Generated compiler wrappers live in a temporary
cache; no MM operator or source-tree write access is needed. C++ sources and
the helper package are included in the wheel.

## Verification and Graph acceptance

The clean PR worktree passed 64 tests: scales, formats, selection directions,
full finite encodings, nondivisible/leading dimensions, zero/full K, cutoff ties,
zero/negative scales, maximum row/K, changing Graph inputs, storage offsets,
streams, and unavailable Common IR. Each benchmark input is checked for exact
BF16 value agreement and index consistency before timing.

Final acceptance uses actual NPU Graph capture/replay, with 100 calls per graph
and the median of 30 replay event samples normalized per operator. Targets are
(4,128,8), (8,256,16), (64,1024,32), (64,4096,64), (64,8192,128), and
(128,32768,256), expressed as (M,N,K). All exceed 1.3x in the reported Graph run.

Ascend disables `topk` replacement in `CUSTOMIZED_UNUSED_OPS`, so the standard
Gems BF16 API path falls back to torch. The harness verifies and records this;
its two BF16 columns are not independent implementations. Eager/profiler data
are diagnostics and must not be substituted for Graph acceptance.

## Reproduce

Activate the matching environment, source CANN, and run from the repository:

```bash
export FLAGTREE_BACKEND=ascend
export PYTHONPATH="$PWD/src"
pytest -q tests/test_topk_w8a16_fp8.py
python benchmark/bench_topk_w8a16_fp8_ascend.py --output topk-graph.json
```

Optional diagnostics:

```bash
python benchmark/profile_topk_w8a16_fp8_ascend.py --root profile-topk
```

The shell runner accepts optional `ASCEND_ENV_FILE`, `TOPK_VENV` and
`TOPK_DEVICE` overrides. It does not assume a machine-specific virtualenv.
