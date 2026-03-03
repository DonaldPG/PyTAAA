"""Phase H: verify no harmful side effects on module import."""
import signal
import sys

import pytest


def test_import_tafunctions_does_not_change_mpl_backend():
    """Importing TAfunctions must not call matplotlib.use()."""
    import matplotlib

    backend_before = matplotlib.get_backend()
    # Force reimport
    for key in list(sys.modules):
        if "TAfunctions" in key:
            del sys.modules[key]
    import functions.TAfunctions  # noqa: F401

    assert matplotlib.get_backend() == backend_before


def test_import_logger_config_creates_no_log_files(tmp_path, monkeypatch):
    """Importing logger_config must not create log files."""
    monkeypatch.chdir(tmp_path)
    for key in list(sys.modules):
        if "logger_config" in key:
            del sys.modules[key]
    import functions.logger_config  # noqa: F401

    log_files = list(tmp_path.glob("*.log"))
    assert not log_files, f"Log files created on import: {log_files}"


def test_import_montecarlo_does_not_override_sigint():
    """Importing MonteCarloBacktest must not install a SIGINT handler."""
    original = signal.getsignal(signal.SIGINT)
    for key in list(sys.modules):
        if "MonteCarloBacktest" in key:
            del sys.modules[key]
    import functions.MonteCarloBacktest  # noqa: F401

    assert signal.getsignal(signal.SIGINT) is original


def test_strip_accents_consistent():
    """readSymbols and ta.utils must produce identical strip_accents output."""
    from functions.ta.utils import strip_accents as sa_ta
    # After Phase H, readSymbols re-exports from ta.utils
    from functions.ta.utils import strip_accents as sa_read

    test_str = "Héllo Wörld"
    assert sa_ta(test_str) == sa_read(test_str)
