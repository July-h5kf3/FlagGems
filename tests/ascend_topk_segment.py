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

"""Pin check for FlagTree 0.7.0+ascend3.5 hivm.hir.custom operand segments."""

import importlib.metadata

SEGMENT_REASON = (
    "flagtree 0.7.0+ascend3.5 lowers sort32/compare_scalar "
    "(FlagTree #1156/#1159) to hivm.hir.custom with operandSegmentSizes=[0, 0]"
)


def ascend_custom_segment_bug(device, vendor, flagtree_version):
    """True when Ascend topk custom ops must not be launched.

    The pinned compiler emits ``hivm.hir.custom`` with
    ``operandSegmentSizes=[0, 0]`` while the operands are still attached.
    An empty or different version is not this pin: a newer FlagTree can fix
    the builder without that version string, and those runs should try the
    kernel (the caller still skips if the same MLIR error is raised).
    """
    if device != "npu" and vendor != "ascend":
        return False
    ver = (flagtree_version or "").lower()
    return "0.7.0" in ver and "ascend3.5" in ver


def flagtree_version():
    try:
        return importlib.metadata.version("flagtree")
    except Exception:
        return ""


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
