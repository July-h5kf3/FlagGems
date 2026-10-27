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

"""NVIDIA Hopper gate for mm_w8a8 FP8. Hygon acceptance is INT8."""


def nvidia_hopper_fp8(vendor_name, device, cuda_ok, capability, has_e4m3, has_td):
    """Hopper FP8 scaled MM is NVIDIA-only.

    Hygon (and other CUDA-compatible vendors) report ``device_name == "cuda"``.
    Their mm_w8a8 acceptance path is INT8 (``mm_w8a8_int8``). A compute
    capability of 9 or higher must not pull those machines into this suite.
    """
    if vendor_name != "nvidia" or device != "cuda" or not cuda_ok:
        return False
    if not capability:
        return False
    try:
        major = capability[0]
    except (TypeError, IndexError):
        return False
    return major >= 9 and has_e4m3 and has_td
