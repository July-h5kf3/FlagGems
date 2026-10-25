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

from flag_gems.runtime import torch_device_fn


def supports(feature: str) -> bool:
    return feature in (
        "strided_inputs",
        "default_float32",
        "long_k",
        "accepts_scale_stride",
        "accepts_out_stride",
    )


def new_graph():
    return torch_device_fn.NPUGraph()
