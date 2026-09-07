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

"""CCEC build and CANN 9.0 Common IR adapter for inline vector TopK.

CANN 9.0 cannot directly lower FlagTree's hivm.hir.custom. The adapter
keeps its core inference and links the AscendC fragment through a typed call.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_WRAP_DIR = Path(tempfile.gettempdir()) / (
    "flaggems_topk_"
    + str(os.getuid())
    + "_"
    + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
)
_SEGMENT3_RE = re.compile(
    r"operandSegmentSizes\s*=\s*array<i32:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*>"
)
_EMPTY_TMPS_RE = re.compile(r"\s*tmps\(\s*\)")
_SYMBOL_RE = re.compile(r'symbol\s*=\s*"([^"]+)"')
_LAST_CUSTOM_LINALG: str | None = None
_PATCHED = False
_DUMP = _WRAP_DIR / "ttadapter.mlir"


def _find_ccec() -> str:
    env = os.environ.get("CCEC") or os.environ.get("BISHENG")
    if env and Path(env).is_file():
        return env
    found = shutil.which("ccec") or shutil.which("bisheng")
    if found:
        return found
    toolkit = os.environ.get("ASCEND_HOME_PATH") or os.environ.get(
        "ASCEND_TOOLKIT_HOME"
    )
    if toolkit:
        cand = Path(toolkit) / "compiler" / "ccec_compiler" / "bin" / "ccec"
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError("ccec/bisheng not found; source the CANN set_env script")


def _tikcpp_include() -> Path:
    toolkit = os.environ.get("ASCEND_HOME_PATH") or os.environ.get(
        "ASCEND_TOOLKIT_HOME"
    )
    if toolkit:
        tik = Path(toolkit) / "aarch64-linux" / "tikcpp" / "tikcfw"
        if (tik / "kernel_operator.h").is_file():
            return tik
    tik = Path("/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/tikcpp/tikcfw")
    if (tik / "kernel_operator.h").is_file():
        return tik
    raise FileNotFoundError("tikcpp/tikcfw/kernel_operator.h not found")


def _cxx_includes() -> list[str]:
    incs: list[str] = []
    for ver in ("12", "11", "13"):
        base = Path(f"/usr/include/c++/{ver}")
        if (base / "cstdint").is_file():
            incs.extend([f"-I{base}", f"-I/usr/include/aarch64-linux-gnu/c++/{ver}"])
            break
    return incs


def compile_cmd(src: Path, out: Path) -> list[str]:
    tik = _tikcpp_include()
    return [
        _find_ccec(),
        "-x",
        "cce",
        "--cce-aicore-arch=dav-c220-vec",
        "--cce-aicore-only",
        "-std=c++17",
        f"-I{tik}",
        f"-I{tik / 'interface'}",
        f"-I{tik / 'impl'}",
        *_cxx_includes(),
        "-DFLAGGEMS_COMMON_IR_IFACE",
        # CANN 9.0 rejects ``-emit-llvm`` unless ``-c`` is also present.
        # The object is real LLVM bitcode (magic BC\\xc0\\xde), which
        # ``--link-aicore-bitcode`` can consume. Without ``-emit-llvm``
        # ccec writes a cube ELF relocatable that hivmc cannot link.
        "-emit-llvm",
        "-c",
        str(src),
        "-o",
        str(out),
    ]


def makefile_compile() -> str:
    """Recipe for the CustomOp ``compile`` attribute (``$<`` / ``$@``)."""
    tik = _tikcpp_include()
    incs = " ".join(_cxx_includes())
    return (
        f"{_find_ccec()} -x cce --cce-aicore-arch=dav-c220-vec --cce-aicore-only "
        f"-std=c++17 -I{tik} -I{tik / 'interface'} -I{tik / 'impl'} {incs} "
        f"-DFLAGGEMS_COMMON_IR_IFACE -emit-llvm -c $< -o $@"
    )


def _split_ins_outs(fragment: str) -> tuple[str, str]:
    """Split ``vals : types`` from an ``ins(...)`` / ``outs(...)`` body."""
    vals, tys = fragment.split(":", 1)
    return vals.strip(), tys.strip()


def _extract_balanced(src: str, open_idx: int) -> tuple[str, int]:
    open_c = src[open_idx]
    close_c = {"(": ")", "{": "}", "[": "]"}[open_c]
    depth = 0
    for j in range(open_idx, len(src)):
        ch = src[j]
        if ch == open_c:
            depth += 1
        elif ch == close_c:
            depth -= 1
            if depth == 0:
                return src[open_idx + 1 : j], j + 1
    raise ValueError(f"unbalanced {open_c} in HIVM custom op")


def lower_custom_op_to_call(mlir: str) -> str:
    """Replace ``hivm.hir.custom`` with an i64 ``func.call`` of ``_mlir_ciface_*``.

    CANN 9.0 hivmc does not implement ``hivm.hir.custom``. Passing memref
    descriptors into ``func.call`` fails later: HIVM wraps the AIC body and
    ``llvm.call`` cannot use values defined outside that region. Extract GM
    addresses next to the call (new SSA, new type) so the call stays legal.
    Legacy dummy ``outs`` stay out of the C ABI. Operations declaring
    ``extra_attr="flaggems_pass_outputs=true"`` receive output buffer addresses
    after their inputs. This preserves existing Fixpipe call signatures.
    """
    decls: list[str] = []
    pieces: list[str] = []
    pos = 0
    sink_n = 0
    while True:
        hit = mlir.find("hivm.hir.custom", pos)
        if hit < 0:
            pieces.append(mlir[pos:])
            break
        line_start = mlir.rfind("\n", 0, hit) + 1
        indent_and_assign = mlir[line_start:hit]
        indent_m = re.match(r"^(\s*)", indent_and_assign)
        indent = indent_m.group(1) if indent_m else ""
        pieces.append(mlir[pos:line_start])

        ins_i = mlir.find("ins(", hit)
        outs_i = mlir.find("outs(", hit)
        if ins_i < 0 or outs_i < 0:
            raise ValueError("hivm.hir.custom is missing ins/outs")
        ins_body, after_ins = _extract_balanced(mlir, ins_i + 3)
        outs_i = mlir.find("outs(", after_ins - 1)
        if outs_i < 0:
            raise ValueError("hivm.hir.custom is missing outs")
        _outs_body, after_outs = _extract_balanced(mlir, outs_i + 4)
        cursor = after_outs
        rest_head = mlir[cursor : cursor + 16]
        if rest_head.lstrip().startswith("tmps("):
            tmps_i = mlir.find("tmps(", cursor)
            _tmps_body, cursor = _extract_balanced(mlir, tmps_i + 4)
        attrs_body = ""
        skip = 0
        while cursor + skip < len(mlir) and mlir[cursor + skip] in " \t\n":
            skip += 1
        if cursor + skip < len(mlir) and mlir[cursor + skip] == "{":
            attrs_body, cursor = _extract_balanced(mlir, cursor + skip)
        loc = ""
        loc_i = mlir.find("loc(", cursor)
        newline_i = mlir.find("\n", cursor)
        if loc_i >= 0 and (newline_i < 0 or loc_i < newline_i):
            _loc_body, after_loc = _extract_balanced(mlir, loc_i + 3)
            loc = " " + mlir[loc_i:after_loc]
            cursor = after_loc
        if cursor < len(mlir) and mlir[cursor] == "\n":
            cursor += 1

        symbol_m = _SYMBOL_RE.search(attrs_body) or _SYMBOL_RE.search(mlir[hit:cursor])
        if symbol_m is None:
            raise ValueError("hivm.hir.custom is missing symbol")
        iface = f"_mlir_ciface_{symbol_m.group(1)}"
        ins_vals, ins_tys = _split_ins_outs(ins_body)
        operands = _split_mlir_list(ins_vals)
        types = _split_mlir_list(ins_tys)
        fragment = mlir[hit:cursor]
        if "flaggems_pass_outputs=true" in fragment:
            out_vals, out_tys = _split_ins_outs(_outs_body)
            operands.extend(_split_mlir_list(out_vals))
            types.extend(_split_mlir_list(out_tys))
        if len(operands) != len(types):
            raise ValueError("hivm.hir.custom operand/type count mismatch")
        call_vals: list[str] = []
        call_tys: list[str] = []
        prefix: list[str] = []
        for name, ty in zip(operands, types):
            # L0C acc is still a tensor at HIVM input. Materialize a memref so
            # we can extract the on-chip address; hivmc 9.0 cannot pass tensors
            # or memref descriptors through llvm.call.
            if ty.startswith("tensor"):
                m_name = f"%fp_m{sink_n}"
                sink_n += 1
                memref_ty = "memref" + ty[len("tensor") :]
                # CANN 9.0 only accepts ``to_memref %t : memref<...>``.
                prefix.append(
                    f"{indent}{m_name} = bufferization.to_memref {name} : {memref_ty}\n"
                )
                name, ty = m_name, memref_ty
            if ty.startswith("memref"):
                p_name = f"%fp_p{sink_n}"
                i_name = f"%fp_i{sink_n}"
                sink_n += 1
                prefix.append(
                    f"{indent}{p_name} = memref.extract_aligned_pointer_as_index "
                    f"{name} : {ty} -> index\n"
                )
                prefix.append(
                    f"{indent}{i_name} = arith.index_cast {p_name} : index to i64\n"
                )
                call_vals.append(i_name)
                call_tys.append("i64")
            else:
                call_vals.append(name)
                call_tys.append(ty)
        pieces.extend(prefix)
        pieces.append(
            f"{indent}func.call @{iface}({', '.join(call_vals)}) "
            f": ({', '.join(call_tys)}) -> (){loc}\n"
        )
        # hivmc on CANN 9.0 rejects hivm.vf_mode / hivm.pipe on func.func.
        # i64 callees need an explicit matching core type for hivmc.
        vector_call = "#hivm.tcore_type<VECTOR>" in fragment
        func_core = "AIV" if vector_call else "AIC"
        tensor_core = "VECTOR" if vector_call else "CUBE"
        attrs = [
            f"hivm.func_core_type = #hivm.func_core_type<{func_core}>",
            "hivm.part_of_mix",
            f"hivm.tcore_type = #hivm.tcore_type<{tensor_core}>",
        ]
        decl = (
            f"  func.func private @{iface}({', '.join(call_tys)}) "
            f"attributes {{{', '.join(attrs)}}}"
        )
        if decl not in decls:
            decls.append(decl)
        pos = cursor

    lowered = "".join(pieces)
    if decls:
        end = lowered.rfind("}")
        if end < 0:
            raise ValueError("cannot insert CustomOp callee: no module end")
        lowered = lowered[:end] + "\n".join(decls) + "\n" + lowered[end:]
    return lowered


def rewrite_custom_op_segments(mlir: str) -> str:
    """Flatten 3-element CustomOp segment sizes to CANN 9.0's 2-element form."""

    def _repl(match: re.Match[str]) -> str:
        ins, outs, tmps = (
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)),
        )
        if tmps != 0:
            raise ValueError(
                f"hivm.hir.custom has non-empty tmps={tmps}; cannot lower to CANN 9.0"
            )
        return f"operandSegmentSizes = array<i32: {ins}, {outs}>"

    return _EMPTY_TMPS_RE.sub("", _SEGMENT3_RE.sub(_repl, mlir))


def _split_mlir_list(src: str) -> list[str]:
    items: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in src:
        if ch in "(<":
            depth += 1
            buf.append(ch)
        elif ch in ")>":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            items.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        items.append("".join(buf).strip())
    return [x for x in items if x]


def _skip_mlir_type(src: str, i: int) -> int:
    while i < len(src) and src[i] in " \t":
        i += 1
    while i < len(src) and (src[i].isalnum() or src[i] in "._!"):
        i += 1
    while i < len(src) and src[i] == "<":
        depth = 0
        while i < len(src):
            if src[i] == "<":
                depth += 1
            elif src[i] == ">":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1
    return i


def rewrite_cann90_bufferization(mlir: str) -> str:
    """Drop FlagTree's ``to_tensor %x : memref<T> to tensor<T>`` result type.

    CANN 9.0 hivmc only parses ``bufferization.to_tensor %x : memref<T>``.
    """
    key = "bufferization.to_tensor"
    out: list[str] = []
    pos = 0
    while True:
        hit = mlir.find(key, pos)
        if hit < 0:
            out.append(mlir[pos:])
            break
        colon = mlir.find(":", hit)
        newline = mlir.find("\n", hit)
        if colon < 0 or (newline >= 0 and colon > newline):
            out.append(mlir[pos : hit + len(key)])
            pos = hit + len(key)
            continue
        ty_end = _skip_mlir_type(mlir, colon + 1)
        rest = mlir[ty_end : ty_end + 16]
        if rest.lstrip().startswith("to "):
            to_i = mlir.find("to ", ty_end)
            ty_end = _skip_mlir_type(mlir, to_i + 2)
            out.append(mlir[pos:colon])
            out.append(mlir[colon : mlir.find("to ", colon)])
            pos = ty_end
        else:
            out.append(mlir[pos:ty_end])
            pos = ty_end
    return "".join(out)


def _prepare_custom_linalg(linalg: str) -> str:
    """Keep ``hivm.hir.custom`` until vector core inference completes."""
    rewritten = rewrite_cann90_bufferization(linalg)
    _DUMP.write_text(rewritten)
    rewritten = rewrite_custom_op_segments(rewritten)
    (_WRAP_DIR / "ttadapter.rewritten.mlir").write_text(rewritten)
    return rewritten


def _find_real_hivmc() -> str:
    wrap = (_WRAP_DIR / "hivmc").resolve()
    cached = os.environ.get("FLAGGEMS_REAL_HIVMC")
    if cached:
        cand = Path(cached)
        if cand.is_file() and cand.resolve() != wrap:
            return str(cand)
    toolkit = (
        os.environ.get("ASCEND_HOME_PATH")
        or os.environ.get("ASCEND_TOOLKIT_HOME")
        or ""
    )
    candidates = [
        "/usr/local/Ascend/cann-9.0.0/bin/hivmc",
        "/usr/local/Ascend/cann-9.0.0/tools/bishengir/bin/hivmc",
    ]
    if toolkit:
        candidates.extend(
            [
                str(Path(toolkit) / "bin" / "hivmc"),
                str(Path(toolkit) / "tools" / "bishengir" / "bin" / "hivmc"),
            ]
        )
    for c in candidates:
        p = Path(c)
        if p.is_file() and p.resolve() != wrap:
            return str(p)
    for d in os.environ.get("PATH", "").split(":"):
        p = Path(d) / "hivmc"
        if p.is_file() and p.resolve() != wrap:
            return str(p)
    raise FileNotFoundError("real hivmc not found")


def prepare_hivmc_mlir(mlir: str) -> str:
    """Lower vector CustomOp ABI after CANN has inferred core placement."""
    rewritten = rewrite_cann90_bufferization(mlir)
    rewritten = rewrite_custom_op_segments(rewritten)
    if "hivm.hir.custom" in rewritten:
        rewritten = lower_custom_op_to_call(rewritten)
    return rewritten


_HIVMC_SHIM = """#!/usr/bin/env python3
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, @HELPER_DIR@)
from compile_topk import prepare_hivmc_mlir  # noqa: E402

_HIVMC_IN = (_HERE / "input.mlir")
_HIVMC_OUT = (_HERE / "rewritten.mlir")


def _real_hivmc() -> str:
    real = os.environ.get("FLAGGEMS_REAL_HIVMC")
    if real and Path(real).is_file() and Path(real).resolve() != Path(__file__).resolve():
        return real
    from compile_topk import _find_real_hivmc
    return _find_real_hivmc()


def main() -> None:
    args = sys.argv[1:]
    new_args = []
    for arg in args:
        path = Path(arg)
        if path.suffix == ".mlir" and path.is_file():
            text = path.read_text()
            _HIVMC_IN.write_text(text)
            if "hivm.hir.custom" in text:
                (_HERE / "custom_in.mlir").write_text(text)
                text = prepare_hivmc_mlir(text)
                path.write_text(text)
                (_HERE / "custom_out.mlir").write_text(text)
            _HIVMC_OUT.write_text(text)
        new_args.append(arg)
    real = _real_hivmc()
    os.execv(real, [real, *new_args])


if __name__ == "__main__":
    main()
""".replace("@HELPER_DIR@", repr(str(_DIR)))

_BISHENGIR_SHIM = """#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PATH="$HERE:${PATH}"
REAL="${FLAGGEMS_REAL_BISHENGIR:-/usr/local/Ascend/cann-9.0.0/bin/bishengir-compile}"
# Keep argv[0] as this wrapper so sibling hivmc lookup hits our shim.
exec -a "$0" "$REAL" "$@"
"""


def _ensure_hivmc_wrapper() -> Path:
    _WRAP_DIR.mkdir(parents=True, exist_ok=True)
    shim = _WRAP_DIR / "hivmc"
    if not shim.exists() or shim.read_text() != _HIVMC_SHIM:
        shim.write_text(_HIVMC_SHIM)
        shim.chmod(0o755)
    compiler = _WRAP_DIR / "bishengir-compile"
    if not compiler.exists() or compiler.read_text() != _BISHENGIR_SHIM:
        compiler.write_text(_BISHENGIR_SHIM)
        compiler.chmod(0o755)
    return _WRAP_DIR


def install_cann90_custom_op_compat() -> None:
    """Keep CustomOp for InferCoreType; rewrite it only when hivmc starts."""
    global _PATCHED
    if _PATCHED:
        return
    from triton.backends.ascend import compiler as ascend_compiler
    from triton.backends.ascend import utils as ascend_utils

    if getattr(
        ascend_compiler._compile_linalg_to_npu_bin, "_flaggems_topk_custom", False
    ):
        _PATCHED = True
        return

    wrap_dir = str(_ensure_hivmc_wrapper())
    real_hivmc = _find_real_hivmc()
    orig_to_bc = ascend_compiler.linalg_to_bc_by_triton_mlir_opt
    orig_to_lin = ascend_compiler.bc_to_linalg_by_bishengir_opt
    orig_to_bin = ascend_compiler._compile_linalg_to_npu_bin
    orig_get_compiler = ascend_compiler._get_npucompiler_path

    def _to_bc(linalg, metadata, opt):
        global _LAST_CUSTOM_LINALG
        if "hivm.hir.custom" in linalg:
            _LAST_CUSTOM_LINALG = linalg
            _DUMP.write_text(linalg)
            return b""
        _LAST_CUSTOM_LINALG = None
        return orig_to_bc(linalg, metadata, opt)

    def _to_lin(bc_data, metadata, opt):
        if not bc_data and _LAST_CUSTOM_LINALG:
            return _prepare_custom_linalg(_LAST_CUSTOM_LINALG)
        return orig_to_lin(bc_data, metadata, opt)

    def _wrapped_get_compiler():
        path, env = orig_get_compiler()
        env = dict(env)
        env["PATH"] = wrap_dir + ":" + env.get("PATH", "")
        env["FLAGGEMS_REAL_HIVMC"] = real_hivmc
        env["FLAGGEMS_REAL_BISHENGIR"] = path
        return str(Path(wrap_dir) / "bishengir-compile"), env

    def _to_bin(linalg, metadata, opt):
        if "hivm.hir.custom" in linalg or _SEGMENT3_RE.search(linalg):
            linalg = _prepare_custom_linalg(linalg)
            # Keep func.call at function scope so hivmc can lower it.
            orig_blockify = ascend_compiler._is_auto_map_parallel_blocks_enabled
            ascend_compiler._is_auto_map_parallel_blocks_enabled = lambda: False
            try:
                return orig_to_bin(linalg, metadata, opt)
            finally:
                ascend_compiler._is_auto_map_parallel_blocks_enabled = orig_blockify
        return orig_to_bin(linalg, metadata, opt)

    _to_bin._flaggems_topk_custom = True
    ascend_compiler.linalg_to_bc_by_triton_mlir_opt = _to_bc
    ascend_compiler.bc_to_linalg_by_bishengir_opt = _to_lin
    ascend_compiler._compile_linalg_to_npu_bin = _to_bin
    ascend_compiler._get_npucompiler_path = _wrapped_get_compiler
    ascend_utils._get_npucompiler_path = _wrapped_get_compiler
    _PATCHED = True
