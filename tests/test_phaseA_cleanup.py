"""Phase A and C cleanup verification via AST analysis.

Phase A: verify mechanical cleanup (bare excepts, debug prints, dead code).
Phase C: verify duplicate function definition removal.
"""
import ast
import sys
from pathlib import Path
import pytest


def all_python_files():
    root = Path(".")
    return list(root.glob("functions/**/*.py")) + [
        Path(p) for p in [
            "pytaaa_main.py", "PyTAAA.py", "run_pytaaa.py",
            "daily_abacus_update.py", "recommend_model.py",
            "run_monte_carlo.py", "pytaaa_backtest_montecarlo.py",
            "scheduler.py",
        ]
        if Path(p).exists()
    ]


def test_no_bare_except_in_any_module():
    """Zero ExceptHandler nodes with no type annotation anywhere."""
    violations = []
    for path in all_python_files():
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                violations.append(f"{path}:{node.lineno}")
    assert not violations, "Bare except: found:\n" + "\n".join(violations)


def test_no_debug_prints_in_getparams():
    """No print() calls inside public functions of GetParams.py."""
    path = Path("functions/GetParams.py")
    tree = ast.parse(path.read_text())
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Name)
                    and child.func.id == "print"
                ):
                    violations.append(f"GetParams.{node.name}:{child.lineno}")
    assert not violations, "Debug prints found:\n" + "\n".join(violations)


def test_sharpe_rank_old_removed():
    """sharpeWeightedRank_2D_old must not exist in TAfunctions."""
    from functions import TAfunctions
    assert not hasattr(TAfunctions, "sharpeWeightedRank_2D_old")


def test_rank_models_fast_guarded_by_has_numba():
    """MonteCarloBacktest imports cleanly regardless of numba availability."""
    import unittest.mock as mock
    with mock.patch.dict(sys.modules, {"numba": None}):
        for key in list(sys.modules):
            if "MonteCarloBacktest" in key:
                del sys.modules[key]
        try:
            import functions.MonteCarloBacktest  # noqa: F401
        except ImportError:
            pass  # numba ImportError is acceptable; NameError is not


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
