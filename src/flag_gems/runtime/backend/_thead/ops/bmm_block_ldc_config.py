# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Offline PPU configurations. No runtime autotuning."""

# PPU-ZW810E, torch 2.10.0 / Triton 3.6.0, FlagTree d96f5339 / SDK 2.1, 2026-09-09.
# Values: BLOCK_M, BLOCK_N, num_warps, num_stages, GROUP_M.
EXACT_CONFIGS = {
    (2, 16, 32, 64): (16, 32, 2, 1, 1),
    (4, 64, 64, 128): (64, 16, 4, 1, 1),
    (2, 128, 128, 256): (64, 16, 4, 1, 1),
    (4, 256, 256, 512): (64, 64, 4, 2, 16),
    (8, 512, 512, 1024): (64, 128, 4, 3, 16),
    (8, 1024, 1024, 2048): (64, 128, 4, 3, 8),
    (8, 2048, 2048, 4096): (64, 128, 4, 3, 8),
    (4, 4096, 4096, 16384): (128, 128, 8, 4, 8),
    (8, 4096, 4096, 16384): (128, 128, 8, 4, 8),
}


def get_config(batch, m, n, k, scale_n=128):
    key = (batch, m, n, k)
    if key in EXACT_CONFIGS:
        bm, bn, warps, stages, group = EXACT_CONFIGS[key]
    elif m <= 64:
        bm, bn, warps, stages, group = 32, 64, 4, 2, 1
    elif k >= 4096:
        bm, bn, warps, stages, group = 128, 64, 4, 3, 8
    else:
        bm, bn, warps, stages, group = 64, 64, 4, 2, 4
    result = dict(
        BLOCK_M=bm,
        BLOCK_N=min(bn, scale_n),
        num_warps=warps,
        num_stages=stages,
        GROUP_M=group,
    )
    return result


AIU_OVERRIDES = {(4, 64, 64, 128): False, (2, 128, 128, 256): False}
PACK_CONFIGS = {
    (4, 256, 512): (128, 32, 4),
    (8, 512, 1024): (128, 128, 8),
    (8, 1024, 2048): (128, 64, 4),
    (8, 2048, 4096): (128, 128, 8),
    (4, 4096, 16384): (128, 128, 8),
    (8, 4096, 16384): (128, 128, 4),
}
