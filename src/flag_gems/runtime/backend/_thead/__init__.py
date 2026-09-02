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

"""
T-Head Zhenwu (真武) PPU Backend Configuration

Product: Zhenwu PPU (真武处理器)
- Model: Zhenwu 810E (supports up to 16 cards with ICN interconnect)
- Architecture: Proprietary T-Head AI accelerator architecture
- SDK: PPU SDK v2.0.0+

Key Features:
- Full CUDA API compatibility (cuda runtime & driver APIs)
- Triton support: 2.3.x, 3.0.x - 3.4.x with AIU extensions
- Accelerated libraries: acdnn, acblas, acfft, acsolver, acrand, acsparse
- Multi-card support: ICN interconnect, MIG (up to 8 instances), SRIOV
- Device management: ppu-smi tool (similar to nvidia-smi)

Hardware Capabilities:
- Tensor Core support with extended PTX instructions
- Dynamic frequency scaling (200MHz ~ max frequency)
- Support for FP16/BF16/FP32/INT8 precision
- High-bandwidth memory with optimized access patterns

PyTorch Integration:
- Uses torch.cuda interface (CUDA-compatible API)
- Compatible with existing CUDA-based PyTorch code
- No special torch.ppu module required

Reference:
- Official Documentation: https://help.aliyun.com/zh/document_detail/3011255.html
"""

import os
import subprocess

from backend_utils import VendorDescriptor


def _tool_exists(path):
    return bool(path) and os.path.isfile(path)


def _is_llvm_ppu_llc(path):
    try:
        out = subprocess.run(
            [path, "--version"], check=False, capture_output=True, text=True
        )
        text = (out.stdout or "") + (out.stderr or "")
        return "LLVM" in text
    except OSError:
        return False


def _ppu_sdk_roots():
    env_sdk = os.environ.get("PPU_SDK") or ""
    roots = [
        os.environ.get("PPU_SDK_2_1", ""),
        "/usr/local/PPU_SDK-2.1",
        (
            os.path.join(os.path.dirname(env_sdk.rstrip(os.sep)), "PPU_SDK-2.1")
            if env_sdk
            else ""
        ),
        "/root/renyz/PPU_SDK-2.1",
        os.path.expanduser("~/renyz/PPU_SDK-2.1"),
        env_sdk,
        "/usr/local/PPU_SDK",
    ]
    seen = set()
    unique = []
    for root in roots:
        if not root or root in seen:
            continue
        seen.add(root)
        unique.append(root)
    return unique


def ensure_llvm_ppu_llc():
    """FlagTree 3.6 emits LLVM IR; PPU SDK 2.0 ppu-llc is a TIX assembler."""
    current = os.environ.get("TRITON_PPU_LLC_PATH", "")
    if _tool_exists(current) and _is_llvm_ppu_llc(current):
        return current

    candidates = []
    for root in _ppu_sdk_roots():
        candidates.append(os.path.join(root, "bin", "ppu-llc"))
        candidates.append(os.path.join(root, "CUDA_SDK", "bin", "ppu-llc"))

    for path in candidates:
        if _tool_exists(path) and _is_llvm_ppu_llc(path):
            os.environ["TRITON_PPU_LLC_PATH"] = path
            formatter = os.path.join(os.path.dirname(path), "llvm-irformatter")
            if _tool_exists(formatter) and not os.environ.get("TRITON_IR_FORMATTER_PATH"):
                os.environ["TRITON_IR_FORMATTER_PATH"] = formatter
            return path
    return current or None


ensure_llvm_ppu_llc()

vendor_info = VendorDescriptor(
    vendor_name="thead",
    # PPU uses CUDA-compatible API, accessed via torch.cuda
    device_name="cuda",
    # PPU device management tool (similar to nvidia-smi)
    device_query_cmd="ppu-smi",
    # Use standard CUDA dispatch key
    dispatch_key=None,
    # PPU has custom Triton backend with AIU extensions
    # The compiler supports Triton 2.3.x - 3.4.x
    triton_extra_name=None,  # Uses standard CUDA path with PPU-specific compiler
)

# Operators that should use PyTorch native implementation
# Based on PPU SDK capabilities and performance characteristics
CUSTOMIZED_UNUSED_OPS = (
    # PPU has strong acceleration library support (acdnn, acblas, etc.)
    # Most operators should benefit from FlagGems optimization
    # This list can be tuned based on benchmarking results
)

__all__ = ["*"]
