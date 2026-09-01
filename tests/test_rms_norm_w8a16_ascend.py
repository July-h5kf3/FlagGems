"""Structural checks for the migrated Ascend operator."""

import ast
from pathlib import Path


_MODULE = Path(__file__).parents[1] / 'src/flag_gems/ops/rmsnorm_w8a16_ascend.py'


def test_public_symbols_exist():
    tree = ast.parse(_MODULE.read_text(encoding="utf-8"))
    functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assignments = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    available = functions | assignments
    missing = {"rms_norm_w8a16_ascend", "rms_norm_fp8_w8a16"} - available
    assert not missing, missing
