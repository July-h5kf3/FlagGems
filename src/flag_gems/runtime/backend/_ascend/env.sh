#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

# Load the full SDK so compiler IR and runtime libraries use the same CANN.
export CANN_ENV_SCRIPT="${CANN_ENV_SCRIPT:-${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/set_env.sh}"
if [[ ! -f "$CANN_ENV_SCRIPT" ]]; then
    echo "CANN SDK not found: $CANN_ENV_SCRIPT. Set CANN_ENV_SCRIPT to a CANN >= 9.1.0 set_env.sh." >&2
    return 1
fi

source "$CANN_ENV_SCRIPT" >/dev/null || return 1
export FLAGTREE_BACKEND=ascend

python3 - <<'PY' || return 1
import os
from pathlib import Path
import platform
import re
import shutil

sdk = Path(os.environ["ASCEND_HOME_PATH"])
info = sdk / f"{platform.machine()}-linux" / "ascend_toolkit_install.info"
match = re.search(r"(?m)^version=(\d+)\.(\d+)\.(\d+)", info.read_text())
if match is None:
    raise SystemExit(f"Cannot read the CANN version from {info}")
if tuple(map(int, match.groups())) < (9, 1, 0):
    raise SystemExit(
        f"ascend-cann910 requires CANN >= 9.1.0; selected {'.'.join(match.groups())} "
        f"at {sdk}. FlagTree custom-op IR is incompatible with CANN 9.0. "
        "Set CANN_ENV_SCRIPT to a compatible SDK environment script."
    )
for compiler in ("bishengir-opt", "bishengir-compile"):
    selected = shutil.which(compiler)
    if selected is None or not Path(selected).resolve().is_relative_to(sdk.resolve()):
        raise SystemExit(f"{compiler} must come from the selected CANN SDK: {selected}")
print(f"CANN {'.'.join(match.groups())}: {sdk}")
PY
