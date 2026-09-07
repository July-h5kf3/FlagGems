#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
set -eo pipefail
TOPK_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${ASCEND_ENV_FILE:-}" ]]; then source "$ASCEND_ENV_FILE"; fi
if [[ -n "${TOPK_VENV:-}" ]]; then source "$TOPK_VENV/bin/activate"; fi
export FLAGTREE_BACKEND=ascend
if [[ -n "${TOPK_DEVICE:-}" ]]; then export ASCEND_RT_VISIBLE_DEVICES="$TOPK_DEVICE"; fi
export PYTHONPATH="$TOPK_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$TOPK_REPO"
if [[ $# -eq 0 ]]; then set -- benchmark/bench_topk_w8a16_fp8_ascend.py; fi
exec python -u "$@"
