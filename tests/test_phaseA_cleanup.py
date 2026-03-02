"""Tests for cleanup phases - verify duplicate definition removal.

This test suite verifies that no function name is defined more than once
at module level in the cleaned-up modules (Phase C).
"""

import ast
from pathlib import Path


def test_no_duplicate_function_definitions():
    """No function name defined more than once at module level."""
    files_to_check = [
        "functions/quotes_for_list_adjClose.py",
        "functions/GetParams.py",
        "functions/TAfunctions.py",
    ]
    for filepath in files_to_check:
        tree = ast.parse(Path(filepath).read_text())
        names = [
            n.name
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
        ]
        duplicates = [n for n in set(names) if names.count(n) > 1]
        assert not duplicates, (
            f"{filepath}: duplicate function definitions: {duplicates}"
        )
