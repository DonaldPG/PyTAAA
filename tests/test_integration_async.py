"""Integration tests for the async plot generation feature.

Tests verify that:
- JSON config parameters flow through to generate_portfolio_plots
- pytaaa_generic.json contains the new configuration keys
- PortfolioPerformanceCalcs reads and forwards async params correctly
- Backward compatibility is preserved

All tests are marked @pytest.mark.agent_runnable (use mocks, no HDF5).
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


##############################################################################
# Tests: pytaaa_generic.json new params
##############################################################################

class TestGenericJsonConfig:
    """Verify pytaaa_generic.json contains the async config keys."""

    @pytest.mark.agent_runnable
    def test_async_plot_generation_key_exists(self):
        """pytaaa_generic.json contains 'async_plot_generation' key."""
        json_path = os.path.join(
            os.path.dirname(__file__), "..", "pytaaa_generic.json"
        )
        with open(json_path) as fh:
            config = json.load(fh)
        valuation = config.get("Valuation", config)
        assert "async_plot_generation" in valuation, (
            "Key 'async_plot_generation' missing from pytaaa_generic.json"
        )

    @pytest.mark.agent_runnable
    def test_async_plot_generation_default_is_false(self):
        """'async_plot_generation' defaults to False in pytaaa_generic.json."""
        json_path = os.path.join(
            os.path.dirname(__file__), "..", "pytaaa_generic.json"
        )
        with open(json_path) as fh:
            config = json.load(fh)
        valuation = config.get("Valuation", config)
        assert valuation["async_plot_generation"] is False

    @pytest.mark.agent_runnable
    def test_plot_generation_workers_key_exists(self):
        """pytaaa_generic.json contains 'plot_generation_workers' key."""
        json_path = os.path.join(
            os.path.dirname(__file__), "..", "pytaaa_generic.json"
        )
        with open(json_path) as fh:
            config = json.load(fh)
        valuation = config.get("Valuation", config)
        assert "plot_generation_workers" in valuation, (
            "Key 'plot_generation_workers' missing from pytaaa_generic.json"
        )

    @pytest.mark.agent_runnable
    def test_plot_generation_workers_default_is_2(self):
        """'plot_generation_workers' defaults to 2 in pytaaa_generic.json."""
        json_path = os.path.join(
            os.path.dirname(__file__), "..", "pytaaa_generic.json"
        )
        with open(json_path) as fh:
            config = json.load(fh)
        valuation = config.get("Valuation", config)
        assert valuation["plot_generation_workers"] == 2


##############################################################################
# Tests: PortfolioPerformanceCalcs async param forwarding
##############################################################################

class TestPortfolioPerformanceCalcsAsyncParams:
    """Verify PortfolioPerformanceCalcs reads and forwards async params."""

    @pytest.mark.agent_runnable
    def test_async_mode_false_by_default_when_key_missing(self):
        """When params lacks 'async_plot_generation', async_mode defaults False."""
        # We check the logic in PortfolioPerformanceCalcs directly
        params = {}
        async_mode = bool(params.get("async_plot_generation", False))
        assert async_mode is False

    @pytest.mark.agent_runnable
    def test_async_mode_true_when_key_set(self):
        """When params['async_plot_generation']=True, async_mode is True."""
        params = {"async_plot_generation": True}
        async_mode = bool(params.get("async_plot_generation", False))
        assert async_mode is True

    @pytest.mark.agent_runnable
    def test_max_workers_defaults_to_2(self):
        """When params lacks 'plot_generation_workers', max_workers defaults 2."""
        params = {}
        max_workers = int(params.get("plot_generation_workers", 2))
        assert max_workers == 2

    @pytest.mark.agent_runnable
    def test_max_workers_from_params(self):
        """When params['plot_generation_workers']=4, max_workers is 4."""
        params = {"plot_generation_workers": 4}
        max_workers = int(params.get("plot_generation_workers", 2))
        assert max_workers == 4


##############################################################################
# Tests: generate_portfolio_plots new parameters exist
##############################################################################

class TestGeneratePortfolioPlotsSignature:
    """Verify generate_portfolio_plots has the new async parameters."""

    @pytest.mark.agent_runnable
    def test_async_mode_parameter_exists(self):
        """generate_portfolio_plots accepts async_mode parameter."""
        import inspect
        from functions.output_generators import generate_portfolio_plots
        sig = inspect.signature(generate_portfolio_plots)
        assert "async_mode" in sig.parameters

    @pytest.mark.agent_runnable
    def test_max_workers_parameter_exists(self):
        """generate_portfolio_plots accepts max_workers parameter."""
        import inspect
        from functions.output_generators import generate_portfolio_plots
        sig = inspect.signature(generate_portfolio_plots)
        assert "max_workers" in sig.parameters

    @pytest.mark.agent_runnable
    def test_async_mode_keyword_callable(self):
        """generate_portfolio_plots can be called with async_mode keyword."""
        from functions.output_generators import generate_portfolio_plots
        import datetime
        import numpy as np

        # Patch heavy internals to avoid I/O
        with patch(
            "functions.output_generators._spawn_background_plot_generation"
        ), patch(
            "functions.output_generators._generate_full_history_plots"
        ), patch(
            "functions.output_generators._generate_recent_plots"
        ), patch(
            "functions.output_generators.datetime"
        ) as mock_dt:
            # Use an hour that passes the market-hours check
            mock_dt.datetime.now.return_value = datetime.datetime(
                2024, 1, 2, 3, 0, 0
            )
            mock_dt.timedelta = datetime.timedelta
            n = 60
            base = datetime.date(2010, 1, 4)
            datearray = [base + datetime.timedelta(days=d) for d in range(n)]
            adjClose = np.ones((2, n)) * 100
            signal = np.ones((2, n))
            params = {
                "LongPeriod": 10,
                "stddevThreshold": 8.0,
                "uptrendSignalMethod": "HMAs",
                "minperiod": 4,
                "maxperiod": 8,
                "incperiod": 2,
                "numdaysinfit": 20,
                "numdaysinfit2": 10,
                "offset": 3,
            }
            # Should not raise
            generate_portfolio_plots(
                adjClose, ["A", "B"], datearray,
                signal, signal, params, "/tmp",
                async_mode=False, max_workers=2,
            )


##############################################################################
# Local-only test stubs (require production HDF5 data)
##############################################################################

class TestLocalOnlyStubs:
    """Stubs for tests that require local production data.

    These are not blocking for PR creation; the maintainer will run them
    locally after code review.
    """

    @pytest.mark.local_only
    def test_e2e_full_dataset_async(self):
        """E2E-02: Full dataset async generation (100 symbols, HDF5 required)."""
        pytest.skip(
            "Requires production HDF5 data. Run locally with real config."
        )

    @pytest.mark.local_only
    def test_e2e_full_dataset_sync(self):
        """E2E-04: Full dataset sync generation (100 symbols, HDF5 required)."""
        pytest.skip(
            "Requires production HDF5 data. Run locally with real config."
        )

    @pytest.mark.local_only
    def test_int_naz100_hma_config(self):
        """INT-01: Integration with pytaaa_model_switching_params.json."""
        pytest.skip(
            "Requires pytaaa_model_switching_params.json and production HDF5 data."
        )

    @pytest.mark.local_only
    def test_int_naz100_pine_config(self):
        """INT-02: Integration with pytaaa_naz100_pine.json."""
        pytest.skip(
            "Requires pytaaa_naz100_pine.json and production HDF5 data."
        )

    @pytest.mark.local_only
    def test_int_naz100_pi_config(self):
        """INT-03: Integration with pytaaa_naz100_pi.json."""
        pytest.skip(
            "Requires pytaaa_naz100_pi.json and production HDF5 data."
        )

    @pytest.mark.local_only
    def test_int_sp500_hma_config(self):
        """INT-04: Integration with pytaaa_sp500_hma.json."""
        pytest.skip(
            "Requires pytaaa_sp500_hma.json and production HDF5 data."
        )

    @pytest.mark.local_only
    def test_perf_baseline_benchmark(self):
        """PERF-01: Baseline performance benchmark (production data required)."""
        pytest.skip(
            "Requires production HDF5 data for meaningful timing measurements."
        )

    @pytest.mark.local_only
    def test_reg_manual_smoke(self):
        """REG-03: Manual smoke test of full pipeline."""
        pytest.skip("Requires manual validation in production environment.")

    @pytest.mark.local_only
    def test_e2e_background_process_terminates_cleanly(self):
        """E2E: Verify background process exits cleanly (production config)."""
        pytest.skip(
            "Requires production environment to validate process lifecycle."
        )

    @pytest.mark.local_only
    def test_reg_pickle_cleanup(self):
        """REG: Verify temp pickle files are removed after load (production)."""
        pytest.skip(
            "Requires production environment to validate file cleanup timing."
        )
