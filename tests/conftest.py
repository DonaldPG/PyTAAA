"""Pytest configuration and shared fixtures for PyTAAA test suite."""
import sys
import pytest


@pytest.fixture(autouse=True)
def _cleanup_none_sys_modules():
    """Purge None sentinels from sys.modules after every test.

    Some tests (e.g. test_phaseE_imports.py) delete and re-import modules
    to check import ordering. During those re-imports, C-extension packages
    like scipy/matplotlib can leave None placeholders in sys.modules that
    corrupt subsequent imports. This fixture removes those sentinels after
    each test so module state is clean for the next one.
    """
    yield
    for key in list(sys.modules):
        if sys.modules.get(key) is None:
            del sys.modules[key]
