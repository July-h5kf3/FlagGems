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

"""CPU-only gates for the Ascend topk pin and the Hygon mm FP8 skip.

These tests are intentionally unmarked. Acceptance selects ``topk_w8a16`` and
``mm_w8a8``; this file must not change those collections.
"""

from .ascend_topk_segment import ascend_custom_segment_bug, ascend_segment_error
from .mm_fp8_vendor_gate import nvidia_hopper_fp8


def test_flagtree_ascend35_segment_pin():
    assert ascend_custom_segment_bug("npu", "ascend", "0.7.0+ascend3.5")
    assert ascend_custom_segment_bug("npu", "ascend", "flagtree==0.7.0+ascend3.5")
    assert ascend_custom_segment_bug("cpu", "ascend", "0.7.0+ascend3.5")
    assert not ascend_custom_segment_bug("npu", "ascend", "0.8.0+ascend3.5")
    assert not ascend_custom_segment_bug("npu", "ascend", "0.7.0+ascend3.6")
    assert not ascend_custom_segment_bug("cuda", "nvidia", "0.7.0+ascend3.5")
    assert not ascend_custom_segment_bug("cuda", "hygon", "0.7.0+hcu3.6")
    assert not ascend_custom_segment_bug("npu", "ascend", "")
    assert not ascend_custom_segment_bug("npu", "ascend", None)

    mlir = RuntimeError(
        "operandSegmentSizes = array<i32: 0, 0> hivm.hir.custom sort32"
    )
    wrapped = RuntimeError("compile failed")
    wrapped.__cause__ = mlir
    assert ascend_segment_error(mlir)
    assert ascend_segment_error(wrapped)
    sample = (
        "'hivm.hir.custom' op operand count (4) does not match with the total "
        "size (0) specified in attribute 'operandSegmentSizes' "
        'name = "sort32", operandSegmentSizes = array<i32: 0, 0>'
    )
    assert ascend_segment_error(RuntimeError(sample))
    assert ascend_segment_error(
        RuntimeError("compare_scalar operandSegmentSizes mismatch")
    )
    capacity = ValueError("K and N exceed the supported Ascend sort capacity")
    assert not ascend_segment_error(capacity)
    assert not ascend_segment_error(RuntimeError("operandSegmentSizes only"))


def test_nvidia_hopper_fp8_gate_skips_hygon():
    assert not nvidia_hopper_fp8("hygon", "cuda", True, (9, 0), True, True)
    assert not nvidia_hopper_fp8("hygon", "cuda", True, (10, 0), True, True)
    assert not nvidia_hopper_fp8("nvidia", "cuda", False, (9, 0), True, True)
    assert not nvidia_hopper_fp8("nvidia", "cuda", True, (8, 9), True, True)
    assert not nvidia_hopper_fp8("nvidia", "cuda", True, (9, 0), False, True)
    assert not nvidia_hopper_fp8("nvidia", "cuda", True, (9, 0), True, False)
    assert not nvidia_hopper_fp8("nvidia", "cuda", True, None, True, True)
    assert not nvidia_hopper_fp8("nvidia", "cuda", True, (), True, True)
    assert not nvidia_hopper_fp8("nvidia", "musa", True, (9, 0), True, True)
    assert not nvidia_hopper_fp8("mthreads", "musa", True, (9, 0), True, True)
    assert nvidia_hopper_fp8("nvidia", "cuda", True, (9, 0), True, True)
    assert nvidia_hopper_fp8("nvidia", "cuda", True, (10, 1), True, True)
