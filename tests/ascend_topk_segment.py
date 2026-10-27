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

"""Runtime gate for hivm.hir.custom operandSegmentSizes=[0, 0].

flagtree==0.7.0+ascend3.5 already contains sort32/compare_scalar
(FlagTree #1156/#1159). Acceptance fails because the HIVM builder emits
operandSegmentSizes=[0, 0] while the operands are attached, not because
those symbols are missing. A rebuilt custom_ops.bc can keep that version
string and lower correctly, so the version string is not a skip signal.
"""

from importlib import import_module

SEGMENT_REASON = (
    "sort32/compare_scalar lowered to hivm.hir.custom with "
    "operandSegmentSizes=[0, 0]"
)

_PROBE_SRC = """
@triton.jit
def _sort32_segment_probe(B: tl.constexpr):
    v = tl.zeros((B,), tl.float32)
    ids = tl.zeros((B,), tl.uint32)
    al.custom("sort32", v, ids, B // 32, out=tl.full((2 * B,), 0, tl.float32))
"""

_probe_kernel = None
_probe_broken = None


def ascend_segment_error(exc):
    """True when ``exc`` is the HIVM custom-op segment mismatch."""
    stack = [exc]
    seen = set()
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        text = f"{type(current).__name__}: {current}"
        if "operandSegmentSizes" in text and (
            "hivm.hir.custom" in text or "sort32" in text or "compare_scalar" in text
        ):
            return True
        stack.append(getattr(current, "__cause__", None))
        stack.append(getattr(current, "__context__", None))
    return False


def _zero_segment_ir(text):
    """True when lowered IR still carries the zero segment attribute.

    A successful lowering may mention operandSegmentSizes with a non-zero
    layout (for example ``array<i32: 3, 1, 0>``). Only the all-zero attribute
    that fails the HIVM verifier is this bug.
    """
    if not text or "operandSegmentSizes" not in text:
        return False
    if not any(tok in text for tok in ("hivm.hir.custom", "sort32", "compare_scalar")):
        return False
    compact = "".join(str(text).split())
    return "operandSegmentSizes=array<i32:0,0>" in compact


def lowering_reports_segment_bug(lower):
    """Run ``lower`` once. True only for the known segment mismatch.

    ``lower`` returns IR text or raises. Unrelated compile failures, missing
    packages, and the FlagTree version string are not this bug.
    """
    try:
        text = lower()
    except Exception as exc:
        return ascend_segment_error(exc)
    return _zero_segment_ir(text)


def _compiled_text(compiled):
    if compiled is None:
        return ""
    chunks = []
    asm = getattr(compiled, "asm", None)
    if isinstance(asm, dict):
        chunks.extend(str(value) for value in asm.values() if value)
    elif asm:
        chunks.append(str(asm))
    metadata = getattr(compiled, "metadata", None)
    if metadata:
        chunks.append(str(metadata))
    return "\n".join(chunks)


def _sort32_probe_kernel():
    global _probe_kernel
    if _probe_kernel is not None:
        return _probe_kernel
    import triton
    import triton.language as tl
    from triton.language.extra.cann import extension as al

    import_module("triton.experimental.tle.language.dsa.ascend.custom_ops")
    globs = globals()
    globs["triton"] = triton
    globs["tl"] = tl
    globs["al"] = al
    if "_sort32_segment_probe" not in globs:
        exec(_PROBE_SRC, globs, globs)
    _probe_kernel = globs["_sort32_segment_probe"]
    return _probe_kernel


def _lower_sort32_probe():
    """Compile one sort32 custom op. Compile-only; do not launch it."""
    kernel = _sort32_probe_kernel()
    compiled = kernel.warmup(32, grid=(1,))
    return _compiled_text(compiled)


def ascend_sort32_segment_bug():
    """True when this process's FlagTree lowers sort32 with zero segments.

    Cached for the process. The version string is never consulted.
    """
    global _probe_broken
    if _probe_broken is None:
        _probe_broken = lowering_reports_segment_bug(_lower_sort32_probe)
    return _probe_broken
