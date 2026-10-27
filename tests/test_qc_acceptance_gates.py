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

"""CPU-only gates for the Ascend topk lowering probe and the Hygon mm FP8 skip.

These tests are intentionally unmarked. Acceptance selects ``topk_w8a16`` and
``mm_w8a8``; this file must not change those collections.
"""

from .ascend_topk_segment import ascend_segment_error, lowering_reports_segment_bug
from .mm_fp8_vendor_gate import nvidia_hopper_fp8


def _raise(exc):
    def lower():
        raise exc

    return lower


def test_ascend_segment_gate_follows_lowering_not_version():
    sample = (
        "'hivm.hir.custom' op operand count (4) does not match with the total "
        "size (0) specified in attribute 'operandSegmentSizes' "
        'name = "sort32", operandSegmentSizes = array<i32: 0, 0>'
    )
    assert lowering_reports_segment_bug(_raise(RuntimeError(sample)))
    wrapped = RuntimeError("compile failed")
    wrapped.__cause__ = RuntimeError(sample)
    assert ascend_segment_error(wrapped)
    assert lowering_reports_segment_bug(_raise(wrapped))
    assert lowering_reports_segment_bug(
        _raise(RuntimeError("compare_scalar operandSegmentSizes mismatch"))
    )
    assert lowering_reports_segment_bug(
        lambda: (
            'hivm.hir.custom name = "sort32", '
            "operandSegmentSizes = array<i32: 0, 0>"
        )
    )

    good_ir = (
        "flagtree==0.7.0+ascend3.5 hivm.hir.custom sort32 "
        "operandSegmentSizes = array<i32: 3, 1, 0>"
    )
    assert not lowering_reports_segment_bug(lambda: good_ir)
    assert not lowering_reports_segment_bug(lambda: "flagtree==0.7.0+ascend3.5")
    assert not lowering_reports_segment_bug(lambda: "")
    assert not lowering_reports_segment_bug(
        _raise(ValueError("K and N exceed the supported Ascend sort capacity"))
    )
    assert not lowering_reports_segment_bug(
        _raise(RuntimeError("operandSegmentSizes only"))
    )
    assert not lowering_reports_segment_bug(_raise(ImportError("cann extension")))


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
