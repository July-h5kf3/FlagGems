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

import hashlib
import subprocess
from pathlib import Path

import torch
import triton
import triton.language as tl

try:
    import triton.language.extra.cann.extension as al
except ImportError:
    al = None

from flag_gems.runtime.backend._ascend.ops.ascendc import compile_topk as helper
from flag_gems.runtime.backend._ascend.utils import CORE_NUM
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

_READY = False
_REV = ""
_INDEX_TABLES = {}
_LAUNCHERS = {}


def _launch(kernel, grid, args, tensor_count):
    """Cache compiled launch functions; tensor data and streams stay dynamic."""
    device = args[0].device.index
    key = (
        kernel,
        grid,
        device,
        args[tensor_count:],
        tuple((arg.dtype, arg.data_ptr() % 32) for arg in args[:tensor_count]),
    )
    runner = _LAUNCHERS.get(key)
    if runner is None:
        compiled, _ = kernel[grid](*args, multibuffer=False)
        _LAUNCHERS[key] = compiled[(grid + (1, 1))[:3]]
        return
    stream = triton.runtime.driver.active.get_current_stream(device)
    # Keep CompiledKernel's profiler/debug hooks; do not cache a stream or input.
    runner(*args, stream=stream)


def _ensure():
    global _READY, _REV
    if _READY:
        return
    if al is None or not all(
        hasattr(al, name) for name in ("custom", "register_custom_op", "scope")
    ):
        raise RuntimeError(
            "Ascend FP8 TopK requires FlagTree Common IR with al.custom support"
        )
    helper.install_cann90_custom_op_compat()
    src = Path(__file__).with_name("ascendc") / "topk_sort.cpp"
    _REV = hashlib.sha256(
        src.read_bytes() + Path(helper.__file__).read_bytes()
    ).hexdigest()[:16]
    bc = Path("/tmp") / ("flaggems_topk_" + _REV + ".bc")
    if not bc.exists():
        subprocess.check_call(
            [
                s.replace("dav-c220-cube", "dav-c220-vec")
                for s in helper.compile_cmd(src, bc)
            ]
            + ["-O3"]
        )

    @al.register_custom_op
    class topk_sort_pairs:
        name = "topk_sort_pairs"
        core = al.CORE.VECTOR
        pipe = al.PIPE.PIPE_V
        mode = al.MODE.SIMD
        symbol = "topk_sort_pairs"
        bitcode = str(bc)
        source = str(src)
        extra_attr = "flaggems_pass_outputs=true"
        compile = helper.makefile_compile().replace("dav-c220-cube", "dav-c220-vec")

        def __init__(self, values, indices, scratch, n, out=None):
            self.arg_type["n"] = tl.int32

    @al.register_custom_op
    class topk_select_codes:
        name = "topk_select_codes"
        core = al.CORE.VECTOR
        pipe = al.PIPE.PIPE_V
        mode = al.MODE.SIMD
        symbol = "topk_select_codes"
        bitcode = str(bc)
        source = str(src)
        extra_attr = "flaggems_pass_outputs=true"
        compile = helper.makefile_compile().replace("dav-c220-cube", "dav-c220-vec")

        def __init__(self, q, index, v, i, row, n, k, flip, scratch, out=None):
            for arg in ("row", "n", "k", "flip"):
                self.arg_type[arg] = tl.int32

    @al.register_custom_op
    class topk_sort_prefix:
        name = "topk_sort_prefix"
        core = al.CORE.VECTOR
        pipe = al.PIPE.PIPE_V
        mode = al.MODE.SIMD
        symbol = "topk_sort_prefix"
        bitcode = str(bc)
        source = str(src)
        extra_attr = "flaggems_pass_outputs=true"
        compile = helper.makefile_compile().replace("dav-c220-cube", "dav-c220-vec")

        def __init__(self, values, indices, scratch, n, keep, out=None):
            self.arg_type["n"] = tl.int32
            self.arg_type["keep"] = tl.int32

    _READY = True


@triton.jit
def _decode(q, E5: tl.constexpr):
    q = q.to(tl.uint16)
    if E5:
        # E5M2 and binary16 have the same exponent bias.
        bits = q << 8
        return bits.to(tl.float16, bitcast=True).to(tl.float32)
    else:
        # Reposition the sign/mantissa, then correct the exponent bias in
        # FP32. Casting before scaling preserves binary16 subnormals.
        bits = (q + (q & 128)) << 7
        return bits.to(tl.float16, bitcast=True).to(tl.float32) * 256.0


@triton.jit
def _sort(v, ids, B: tl.constexpr, K: tl.constexpr):
    scratch = tl.full((2 * B,), 0, tl.float32)
    dst = tl.full((2 * B,), 0, tl.float32)
    if B >= 128 and triton.next_power_of_2(K) <= 32:
        pairs = al.custom(
            "topk_sort_prefix",
            v,
            ids,
            scratch,
            B,
            max(8, triton.next_power_of_2(K)),
            out=dst,
        )
    else:
        pairs = al.custom("topk_sort_pairs", v, ids, scratch, B, out=dst)
    return pairs


@libentry()
@triton.jit
def _stage1(
    Q,
    S,
    V,
    Indices,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    NG: tl.constexpr,
    B: tl.constexpr,
    P: tl.constexpr,
    DESC: tl.constexpr,
    E5: tl.constexpr,
    TOTAL: tl.constexpr,
    CORES: tl.constexpr,
    REV: tl.constexpr,
):
    with al.scope(core_mode="vector"):
        pid = ext.program_id(0)
        for pid in range(pid, TOTAL, CORES):
            row = pid // P
            part = pid % P
            col = part * B + tl.arange(0, B)
            q = tl.load(Q + row * N + col, col < N, other=0)
            if G >= N:
                s = tl.load(S + row).to(tl.float32)
                qi = q.to(tl.int32)
                v = tl.where((qi & 128) != 0, -(qi & 127), qi & 127).to(tl.float32)
                v = tl.where(s < 0, -v, v)
            else:
                s = tl.load(S + row * NG + col // G, col < N, other=0).to(tl.float32)
                v = _decode(q, E5) * s
            if not DESC:
                v = -v
            v = tl.where(col < N, v, float("-inf"))
            pairs = _sort(v, col.to(tl.uint32), B, K)
            kk = tl.arange(0, triton.next_power_of_2(K))
            vals = tl.gather(pairs, 2 * kk, 0)
            ids = tl.gather(pairs, 2 * kk + 1, 0).to(tl.uint32, bitcast=True)
            if not DESC:
                vals = -vals
            if G >= N:
                code = tl.where(s < 0, -vals, vals).to(tl.int32)
                raw = tl.where(code < 0, 128 - code, code)
                vals = _decode(raw, E5) * s
            tl.store(V + pid * K + kk, vals, kk < K)
            tl.store(Indices + pid * K + kk, ids, kk < K)


@libentry()
@triton.jit
def _merge(
    V,
    Indices,
    Output,
    J,
    K: tl.constexpr,
    P: tl.constexpr,
    B: tl.constexpr,
    DESC: tl.constexpr,
    M: tl.constexpr,
    CORES: tl.constexpr,
    REV: tl.constexpr,
):
    with al.scope(core_mode="vector"):
        for row in range(ext.program_id(0), M, CORES):
            col = tl.arange(0, B)
            v = tl.load(V + row * P * K + col, col < P * K, other=0)
            ids = tl.load(Indices + row * P * K + col, col < P * K, other=0).to(
                tl.uint32
            )
            if not DESC:
                v = -v
            v = tl.where(col < P * K, v, float("-inf"))
            pairs = _sort(v, ids, B, K)
            kk = tl.arange(0, triton.next_power_of_2(K))
            vals = tl.gather(pairs, 2 * kk, 0)
            ids = tl.gather(pairs, 2 * kk + 1, 0).to(tl.uint32, bitcast=True)
            if not DESC:
                vals = -vals
            tl.store(Output + row * K + kk, vals, kk < K)
            tl.store(J + row * K + kk, ids, kk < K)


def topk_w8a16_fp8(x, x_scale, k, dim=-1, largest=True, sorted=True, group_size=128):
    """Select finite FP8 values, apply scales, and return BF16 values/int64 indices.

    Selection uses exact FP32 dequantized values; equal values may reorder.
    Only contiguous NPU inputs, last-dimension TopK, and finite scales/FP8
    values are supported. Quantization is performed by the caller.
    A constant index table is cached per device/row length for the row path.
    """
    assert x.ndim >= 1 and group_size > 0
    assert x.device.type == "npu" and x_scale.device == x.device
    assert x_scale.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert dim in (-1, x.ndim - 1)
    assert x.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    assert x.is_contiguous() and x_scale.is_contiguous()
    n = x.shape[-1]
    assert n > 0, "The last dimension must be nonempty"
    m = x.numel() // n
    ng = (n + group_size - 1) // group_size
    assert x_scale.numel() == m * ng and 0 <= k <= n
    if x.device.index != torch.npu.current_device():
        with torch.npu.device(x.device):
            return topk_w8a16_fp8(x, x_scale, k, dim, largest, sorted, group_size)
    shape = x.shape[:-1] + (k,)
    out = torch.empty(shape, dtype=torch.bfloat16, device=x.device)
    idx = torch.empty(shape, dtype=torch.int64, device=x.device)
    if k == 0 or m == 0:
        return out, idx
    _ensure()
    if (
        group_size >= n
        and x.data_ptr() % 32 == 0
        and n in (4096, 8192, 16384, 32768)
        and 8 <= k <= 512
        and k & (k - 1) == 0
    ):
        v = torch.empty((m * k,), dtype=torch.float32, device=x.device)
        i = torch.empty((m * k,), dtype=torch.int32, device=x.device)
        c = min(CORE_NUM, m)
        key = (x.device, n)
        if key not in _INDEX_TABLES:
            _INDEX_TABLES[key] = torch.arange(n, dtype=torch.int16).to(x.device)
        _launch(
            _row_select,
            (c,),
            (
                x.view(torch.uint8),
                _INDEX_TABLES[key],
                x_scale,
                v,
                i,
                n,
                k,
                m,
                c,
                largest,
                (5 * n + n // 4 + 8192) // 4,
                _REV,
            ),
            5,
        )
        _launch(
            _row_finish,
            (c,),
            (
                v,
                i,
                x_scale,
                out,
                idx,
                k,
                max(32, 1 << (k - 1).bit_length()),
                m,
                c,
                largest,
                x.dtype == torch.float8_e5m2,
                _REV,
            ),
            5,
        )
        return out, idx
    b = min(2048, max(32, (1 << (n - 1).bit_length())))
    p = (n + b - 1) // b
    assert k <= b and (1 << (p * k - 1).bit_length()) <= 4096
    if p == 1:
        v = out
        i = idx
    else:
        v = torch.empty((m * p * k,), dtype=torch.float32, device=x.device)
        i = torch.empty((m * p * k,), dtype=torch.int32, device=x.device)
    _launch(
        _stage1,
        (min(CORE_NUM, m * p),),
        (
            x.view(torch.uint8),
            x_scale,
            v,
            i,
            n,
            k,
            group_size,
            ng,
            b,
            p,
            largest,
            x.dtype == torch.float8_e5m2,
            m * p,
            min(CORE_NUM, m * p),
            _REV,
        ),
        4,
    )
    if p > 1:
        _launch(
            _merge,
            (min(CORE_NUM, m),),
            (
                v,
                i,
                out,
                idx,
                k,
                p,
                max(32, 1 << (p * k - 1).bit_length()),
                largest,
                m,
                min(CORE_NUM, m),
                _REV,
            ),
            4,
        )
    return out, idx


@libentry()
@triton.jit
def _row_finish(
    V,
    Indices,
    S,
    Output,
    J,
    K: tl.constexpr,
    B: tl.constexpr,
    M: tl.constexpr,
    C: tl.constexpr,
    DESC: tl.constexpr,
    E5: tl.constexpr,
    REV: tl.constexpr,
):
    with al.scope(core_mode="vector"):
        for row in range(ext.program_id(0), M, C):
            col = tl.arange(0, B)
            v = tl.load(V + row * K + col, col < K, other=float("-inf"))
            ids = tl.load(Indices + row * K + col, col < K, other=0).to(tl.uint32)
            pairs = _sort(v, ids, B, K)
            key = tl.gather(pairs, 2 * col, 0).to(tl.int32)
            idx = tl.gather(pairs, 2 * col + 1, 0).to(tl.uint32, bitcast=True)
            s = tl.load(S + row).to(tl.float32)
            if not DESC:
                key = 255 - key
            key = tl.where(s < 0, 255 - key, key)
            raw = tl.where(key >= 128, key - 128, 255 - key)
            values = _decode(raw, E5) * s
            tl.store(Output + row * K + col, values, col < K)
            tl.store(J + row * K + col, idx, col < K)


@libentry()
@triton.jit
def _row_select(
    Q,
    Index,
    S,
    V,
    Indices,
    N: tl.constexpr,
    K: tl.constexpr,
    M: tl.constexpr,
    C: tl.constexpr,
    DESC: tl.constexpr,
    WORDS: tl.constexpr,
    REV: tl.constexpr,
):
    with al.scope(core_mode="vector"):
        for row in range(ext.program_id(0), M, C):
            scale = tl.load(S + row).to(tl.float32)
            flip = (scale < 0) ^ (not DESC)
            scratch = tl.full((WORDS,), 0, tl.int32)
            dummy = tl.full((16,), 0, tl.int32)
            al.custom(
                "topk_select_codes",
                Q,
                Index,
                V,
                Indices,
                row,
                N,
                K,
                flip.to(tl.int32),
                scratch,
                out=dummy,
            )
