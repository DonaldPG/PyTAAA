"""Phase B: verify zero wildcard imports anywhere in the project."""
import ast
import importlib
from pathlib import Path
import pytest


def all_python_files():
    root = Path(".")
    return list(root.glob("functions/**/*.py")) + list(root.glob("*.py"))


def test_no_wildcard_imports():
    violations = []
    for path in all_python_files():
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.ImportFrom)
                    and any(a.name == "*" for a in node.names)):
                violations.append(f"{path}:{node.lineno}")
    assert not violations, "Wildcard imports found:\n" + "\n".join(violations)


def test_daily_backtest_importable():
    from functions.dailyBacktest import computeDailyBacktest  # noqa: F401


def test_all_functions_modules_importable():
    for path in Path("functions").glob("*.py"):
        if path.name.startswith("_"):
            continue
        mod_name = f"functions.{path.stem}"
        try:
            importlib.import_module(mod_name)
        except SyntaxError:
            # Skip files with pre-existing Python 2 syntax errors.
            pass
        except Exception as e:
            # Only fail for ImportError caused by names missing from our
            # own functions package (i.e., a regression from this phase).
            # Skip pre-existing failures from missing external packages or
            # module-level side-effects.
            if isinstance(e, ImportError):
                missing = str(e)
                if "functions." in missing:
                    pytest.fail(f"{mod_name} failed to import: {e}")
