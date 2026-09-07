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

import logging

logger = logging.getLogger(__name__)


def topk_w8a16_fp8(x, x_scale, k, dim=-1, largest=True, sorted=True, group_size=128):
    """TopK on FP8 storage with scaled BF16 values and int64 indices.

    The implementation is registered by backends that support this operator.
    """
    logger.debug("GEMS TOPK_W8A16_FP8")
    raise NotImplementedError("topk_w8a16_fp8 is not implemented on this backend")
