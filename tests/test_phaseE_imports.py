"""Phase E: verify MakeValuePlot <-> WriteWebPage_pi circular import is resolved.

Confirms that:
- Each module imports cleanly in isolation (no circular dependency at import time)
- makeMinimumSpanningTree now lives in graph_plots, not MakeValuePlot
- graph_plots itself has no circular dependencies
"""

import importlib
import sys

import pytest


def _fresh_import(module_name: str):
    """Import a module with a clean cache for the two modules under test."""
    for key in list(sys.modules):
        if "MakeValuePlot" in key or "WriteWebPage_pi" in key or "graph_plots" in key:
            del sys.modules[key]
    return importlib.import_module(module_name)


def test_import_makevalueplot_first():
    """MakeValuePlot imports cleanly when loaded before WriteWebPage_pi."""
    _fresh_import("functions.MakeValuePlot")


def test_import_writewebpage_first():
    """WriteWebPage_pi imports cleanly when loaded before MakeValuePlot."""
    _fresh_import("functions.WriteWebPage_pi")


def test_makevalueplot_does_not_import_writewebpage():
    """MakeValuePlot must not import WriteWebPage_pi at module level."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path("functions/MakeValuePlot.py").read_text())
    module_level_imports = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        # Only module-level nodes have col_offset == 0
        and node.col_offset == 0
    ]
    bad = [
        node for node in module_level_imports
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and "WriteWebPage" in node.module
    ]
    assert not bad, (
        "MakeValuePlot still imports from WriteWebPage_pi at module level"
    )


def test_makeminimumspanningtree_not_in_makevalueplot():
    """makeMinimumSpanningTree must not be defined in MakeValuePlot."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path("functions/MakeValuePlot.py").read_text())
    names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert "makeMinimumSpanningTree" not in names, (
        "makeMinimumSpanningTree is still defined in MakeValuePlot.py — "
        "it should have been moved to graph_plots.py"
    )


def test_make_minimum_spanning_tree_importable_from_graph_plots():
    """makeMinimumSpanningTree must be importable from graph_plots."""
    from functions.graph_plots import makeMinimumSpanningTree  # noqa: F401


def test_graph_plots_has_no_circular_deps():
    """graph_plots must not import from MakeValuePlot or WriteWebPage_pi."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path("functions/graph_plots.py").read_text())
    bad = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and ("MakeValuePlot" in node.module or "WriteWebPage" in node.module)
    ]
    assert not bad, (
        "graph_plots.py imports from MakeValuePlot or WriteWebPage_pi — "
        "this would re-introduce the circular dependency"
    )
