#!/bin/bash

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

# Run only inside the pinned PPU SDK 2.1 CI image. Reuse its vendor Torch
# instead of resolving a different runtime from the package index.
set -euo pipefail

sdk_version=$(/usr/local/PPU_SDK/bin/ppu-llc --version 2>&1 || true)
echo "$sdk_version"
if ! grep -q 'ppu version: 2\.1\.' <<< "$sdk_version"; then
  echo "THead source builds require the pinned PPU SDK 2.1 CI image" >&2
  exit 1
fi

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
export USE_TRITON=""
source tools/env.sh thead

mirror=$(python -c 'import yaml; print(yaml.safe_load(open("src/flag_gems/backends.yaml"))["mirror"])')
flagtree_source=$(python -c 'import yaml; print(yaml.safe_load(open("src/flag_gems/backends.yaml"))["backends"]["thead"]["flagtree"])')

python -m pip install --index-url "$mirror" \
  "setuptools>=64,<77" "setuptools-scm>=8,<10" "scikit-build-core==0.12.2" wheel \
  "cmake>=3.20,<4" ninja "pybind11>=2.13.1" pybind11-stubgen
python -m pip install --index-url "$mirror" --no-build-isolation "$flagtree_source"
python -m pip install --index-url "$mirror" --no-build-isolation ".[test]"

# Assert that the image's PPU runtime and the PPU compiler were retained.
python - <<'CHECK'
import importlib.metadata
import torch
import triton
from triton.experimental.tle import language

version = importlib.metadata.version("flagtree")
assert "+ppu" in version, version
assert torch.__version__.startswith("2.10."), torch.__version__
print(f"FlagTree={version}; Torch={torch.__version__}; Triton={triton.__version__}")
CHECK
