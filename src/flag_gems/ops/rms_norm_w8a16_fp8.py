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

"""Public RMSNorm W8A16 entry used by tests and benchmarks (PR #4437).

Weight is grouped FP8 E4M3 plus per-group scale. Vendor backends may
override ``flag_gems.rms_norm_w8a16_fp8``.
"""


def rms_norm_w8a16_fp8(
    x, normalized_shape, weight, weight_scale, eps=1e-5, group_size=128
):
    import flag_gems

    impl = flag_gems.__dict__.get("rms_norm_w8a16_fp8")
    if impl is None or impl is rms_norm_w8a16_fp8:
        raise RuntimeError(
            "rms_norm_w8a16_fp8 is only available when a vendor backend "
            "registers the operator (THead / PPU)."
        )
    return impl(
        x,
        normalized_shape,
        weight,
        weight_scale,
        eps=eps,
        group_size=group_size,
    )
